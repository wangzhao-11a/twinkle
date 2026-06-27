"""NPU smoke driver for the author's AsyncMultiLoraGRPOPipeline (835354b).

Adapts the GPU/Megatron/35B/real-TQ cookbook to what the Ascend container has:
  - Transformers backend (MultiLoraTransformersModel) instead of Megatron
  - in-memory TransferQueue (FakeTQ) instead of the real transfer_queue lib
  - tiny model Qwen3-0.6B + 1 tenant + 1 train GPU + 1 sampler GPU
  - tiny data / 2 train_k so it's a fast end-to-end smoke

Only build_model() and build_data_plane() are overridden; everything else
(rollout, reward, advantage, workers, prompt feeders, run loop) reuses the
author's validated pipeline.
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict

from omegaconf import OmegaConf
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle_agentic.async_rl import TransferQueueDataPlane
from twinkle_agentic.async_rl.grpo_pipeline import (
    AsyncMultiLoraGRPOPipeline,
    context_dataset_config,
    primary_training_context,
    training_context_configs,
)

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', '/data1/Qwen3-0.6B')
MAX_STEPS = int(os.environ.get('MAX_STEPS', 2))
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 4))
DATA_NUM = int(os.environ.get('DATA_NUM', 16))
LORA_RANK = int(os.environ.get('LORA_RANK', 8))

# Collected by the instrumented advantage fn (runs in-driver), asserted in main().
_VERIFY_RECORDS: list = []


class _FakeTQ:
    """In-memory TransferQueue KV client (mirrors tests/.../fakes.py)."""

    def __init__(self):
        self.fields: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.tags: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def kv_put(self, key, partition_id, fields=None, tag=None):
        if fields:
            cur = dict(self.fields[partition_id].get(key) or {})
            cur.update(dict(fields))
            self.fields[partition_id][key] = cur
        elif key not in self.fields[partition_id]:
            self.fields[partition_id][key] = {}
        if tag:
            cur_tag = dict(self.tags[partition_id].get(key) or {})
            cur_tag.update(dict(tag))
            self.tags[partition_id][key] = cur_tag

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        if isinstance(keys, str):
            keys = [keys]
        sel = [select_fields] if isinstance(select_fields, str) else select_fields
        rows = [dict(self.fields[partition_id].get(k) or {}) for k in keys]
        names = set()
        for row in rows:
            names.update(row)
        if sel is not None:
            names.intersection_update(sel)
        return {n: [row.get(n) for row in rows] for n in names}

    def kv_list(self, partition_id=None):
        if partition_id is not None:
            return {partition_id: dict(self.tags.get(partition_id) or {})}
        return {pid: dict(tags) for pid, tags in self.tags.items()}

    def kv_clear(self, keys, partition_id):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self.fields.get(partition_id, {}).pop(k, None)
            self.tags.get(partition_id, {}).pop(k, None)


def _npu_build_dataset(cfg, context_cfg):
    """Module-level (self-free) dataset builder so the DataLoader factory is
    Ray-picklable.

    The author's build_prompt_feeders builds the factory as
    ``lambda: self.build_dataset(...)`` which closes over ``self``; in ray mode
    ``self`` transitively holds a non-picklable ``_thread.RLock`` (model /
    rollouter handles), so Ray cannot ship the factory to the DataLoader actor.
    This replica closes over only ``cfg`` + ``context_cfg`` (both OmegaConf, and
    both picklable).
    """
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor.llm import GSM8KProcessor

    dataset_cfg = context_dataset_config(cfg, context_cfg)
    data_slice = range(int(dataset_cfg.data_num)) if dataset_cfg.get('data_num') else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            dataset_cfg.dataset_id,
            subset_name=dataset_cfg.get('subset_name'),
            split=dataset_cfg.get('split', 'train'),
            data_slice=data_slice,
        )
    )
    template_cfg = cfg.model.template
    dataset.set_template(
        template_cfg.cls,
        model_id=context_cfg.base_model_id,
        max_length=template_cfg.get('max_length', 4096),
        truncation_strategy=template_cfg.get('truncation_strategy', 'delete'),
        enable_thinking=template_cfg.get('enable_thinking', False),
    )
    dataset.map(GSM8KProcessor(system=dataset_cfg.system_prompt))
    dataset.encode(add_generation_prompt=True)
    return dataset


def _verify_sample_text(s):
    """Best-effort extraction of the assistant-generated text from a sample
    dict, to eyeball generation coherence."""
    if not isinstance(s, dict):
        return ''
    for k in ('response', 'completion', 'output', 'generated_text'):
        v = s.get(k)
        if isinstance(v, str) and v:
            return v
    for container in (s.get('trajectory'), s):
        msgs = container.get('messages') if isinstance(container, dict) else None
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get('role') == 'assistant':
                    c = m.get('content')
                    if isinstance(c, str):
                        return c
    return ''


class NpuTransformersGRPOPipeline(AsyncMultiLoraGRPOPipeline):

    def build_model(self):
        from twinkle.model import MultiLoraTransformersModel
        from twinkle.processor import InputProcessor

        primary = primary_training_context(self.cfg)
        lc = self.cfg.model.lora
        lora_config = LoraConfig(
            target_modules=lc.target_modules, r=int(lc.r), lora_alpha=int(lc.lora_alpha),
            lora_dropout=float(lc.lora_dropout))
        contexts = training_context_configs(self.cfg)
        model = MultiLoraTransformersModel(
            model_id=primary.base_model_id, device_mesh=self.model_mesh, remote_group='model',
            max_loras=len(contexts) + 1, max_r=max(32, int(lc.r)))
        for cc in contexts:
            model.add_adapter_to_model(
                cc.adapter_name, lora_config,
                gradient_accumulation_steps=int(self.cfg.model.gradient_accumulation_steps))
            model.set_optimizer('AdamW', lr=float(self.cfg.model.optimizer.lr), adapter_name=cc.adapter_name)
            model.set_lr_scheduler(
                'CosineAnnealingLR', T_max=int(self.cfg.pipeline.max_steps), eta_min=0, adapter_name=cc.adapter_name)
            model.set_loss(
                self.cfg.model.loss.cls, adapter_name=cc.adapter_name,
                epsilon=float(self.cfg.model.loss.get('epsilon', 0.2)))
            model.set_processor(InputProcessor, padding_free=True, adapter_name=cc.adapter_name)
            model.set_template(
                self.cfg.model.template.cls, model_id=primary.base_model_id, adapter_name=cc.adapter_name,
                enable_thinking=False)
        return model

    def build_data_plane(self):
        return TransferQueueDataPlane(tq_client=_FakeTQ())

    def build_prompt_feeders(self):
        """Override the author's feeder build to use a self-free, Ray-picklable
        dataset factory (see _npu_build_dataset). Otherwise the factory lambda
        closes over self -> _thread.RLock -> 'cannot pickle' on the DataLoader
        actor (grpo_pipeline.py:276)."""
        import functools

        from twinkle.dataloader import DataLoader
        from twinkle_agentic.async_rl.prompt_feeder import PromptFeeder

        feeders = []
        max_pending_groups = self.cfg.pipeline.get('prompt_max_pending_groups')
        for context_cfg, context in zip(training_context_configs(self.cfg), self.contexts):
            dataset_cfg = context_dataset_config(self.cfg, context_cfg)
            dataloader = DataLoader(
                dataset=functools.partial(_npu_build_dataset, self.cfg, context_cfg),
                batch_size=int(dataset_cfg.batch_size),
                min_batch_size=int(dataset_cfg.batch_size),
                device_mesh=self.model_mesh,
                remote_group='model',
                # Unique per-tenant Ray actor id; without this, both tenants' DataLoaders
                # are built from this same line -> identical auto-name (call-site derived)
                # -> "name ... is already taken" collision (infra/__init__.py:572).
                instance_id=f'{context_cfg.adapter_name}_',
            )
            feeders.append(
                PromptFeeder(
                    context=context,
                    dataloader=dataloader,
                    rollouter=self.rollouter,
                    max_pending_groups=max_pending_groups,
                )
            )
        return feeders

    def build_advantage_fn(self):
        """Wrap the author's grpo_advantage_fn to PRINT the learning signal
        (reward distribution + advantage magnitude + a generated sample) so the
        output can be verified sane, not just that the pipeline completed."""
        from twinkle_agentic.async_rl.grpo_pipeline import grpo_advantage_fn

        def _logging_advantage_fn(samples, context):
            advantages, rewards = grpo_advantage_fn(samples, context)
            try:
                r = [float(x) for x in rewards]
                a = [float(x) for x in advantages]
                nz = sum(1 for x in r if x != 0.0)
                mean = (sum(r) / len(r)) if r else 0.0
                adv_absmax = max((abs(x) for x in a), default=0.0)
                adp = getattr(context, 'adapter_name', '?')
                print(
                    f'VERIFY_SIGNAL adapter={adp} n={len(r)} reward_nonzero={nz}/{len(r)} '
                    f'reward_mean={mean:.4f} reward_set={sorted(set(round(x, 3) for x in r))} '
                    f'adv_absmax={adv_absmax:.4f} adv_allzero={all(x == 0.0 for x in a)}',
                    flush=True)
                txt = _verify_sample_text(samples[0]) if samples else ''
                if samples:
                    print(f'VERIFY_SAMPLE adapter={adp} keys={list(samples[0].keys())[:14]} '
                          f'text={txt[:300]!r}', flush=True)
                _VERIFY_RECORDS.append({
                    'adapter': adp, 'rewards': r, 'advantages': a,
                    'reward_nonzero': nz, 'adv_absmax': adv_absmax, 'sample_text': txt,
                })
            except Exception as e:  # noqa
                print(f'VERIFY_SIGNAL_ERR {e!r}', flush=True)
            return advantages, rewards

        return _logging_advantage_fn


def build_cfg():
    return OmegaConf.create({
        'runtime': {'mode': 'ray', 'model_gpus': 1, 'sampler_gpus': 1, 'sampler_tp': 1, 'lazy_collect': False},
        'training_contexts': [{
            'tenant_id': 'tenant_a', 'training_run_id': 'gsm8k_npu_smoke', 'base_model_id': MODEL_ID,
            'adapter_name': 'tenant_a_lora', 'reward_type': 'gsm8k', 'loss_type': 'grpo', 'tool_profile': 'default',
        }, {
            'tenant_id': 'tenant_b', 'training_run_id': 'gsm8k_npu_smoke_b', 'base_model_id': MODEL_ID,
            'adapter_name': 'tenant_b_lora', 'reward_type': 'gsm8k', 'loss_type': 'grpo', 'tool_profile': 'default',
        }],
        'model': {
            'mixed_precision': 'bf16',
            'lora': {'target_modules': 'all-linear', 'r': LORA_RANK, 'lora_alpha': LORA_RANK * 2, 'lora_dropout': 0.05},
            'gradient_accumulation_steps': 1,
            'loss': {'cls': 'GRPOLoss', 'epsilon': 0.2},
            'optimizer': {'cls': 'AdamW', 'lr': 1.0e-5},
            'lr_scheduler': {'cls': 'CosineAnnealingLR', 'lr_decay_steps': MAX_STEPS, 'max_lr': 1.0e-5},
            'processor': {'cls': 'InputProcessor'},
            'template': {'cls': 'Template', 'max_length': 2048, 'truncation_strategy': 'delete',
                         'enable_thinking': False},
            'adapter_checkpoint_dir': 'output/npu_smoke/lora_sync',
        },
        'sampler': {
            'engine_args': {'gpu_memory_utilization': 0.3, 'max_model_len': 2048, 'max_lora_rank': max(8, LORA_RANK),
                            'max_loras': 4, 'enable_lora': True, 'enforce_eager': True},
            'template': {'cls': 'Template', 'enable_thinking': False},
            'sampling_params': {'max_tokens': 512, 'num_samples': 1, 'logprobs': 1, 'temperature': 1.0, 'top_p': 0.95},
            'num_generations': NUM_GENERATIONS,
        },
        'dataset': {
            'dataset_id': 'ms://modelscope/gsm8k', 'subset_name': 'main', 'split': 'train',
            'data_num': DATA_NUM, 'batch_size': 2,
            'system_prompt': ('You are a helpful math assistant. Solve the problem with minimal but correct '
                              'reasoning and put your final answer within \\boxed{}.'),
        },
        'transfer_queue': {'init': False},
        'pipeline': {
            'max_steps': MAX_STEPS, 'max_staleness': 1, 'target_groups_per_partition': 1, 'max_concurrent_groups': 2,
            'reward_batch_size': 1024, 'advantage_batch_size': 1024, 'mini_batch_size': 4, 'micro_batch_size': 1,
            'save_name_prefix': 'npu-smoke-lora', 'save_optimizer': False, 'is_sampler_checkpoint': False,
            'max_grad_norm': 1.0, 'norm_type': 2, 'algorithm': 'grpo',
        },
    })


def _run_verify(trained):
    """Assert the e2e OUTPUT is sane (not merely that the run completed).
    Prints a per-check report and exits 0 (PASS) / 1 (FAIL)."""
    recs = _VERIFY_RECORDS
    expected_adapters = {c.adapter_name for c in training_context_configs(build_cfg())}
    adapters = {r['adapter'] for r in recs}
    all_rewards = [x for r in recs for x in r['rewards']]
    finite = all(isinstance(x, float) and math.isfinite(x) for x in all_rewards)
    any_signal = any(r['adv_absmax'] > 1e-6 for r in recs)
    coherent = any(r['sample_text'] and any(ch.isdigit() for ch in r['sample_text']) for r in recs)

    checks = [
        ('advantage_batches_ran', len(recs) > 0, f'{len(recs)} batches'),
        ('both_adapters_trained', adapters >= expected_adapters, f'{sorted(adapters)}'),
        ('rewards_finite_nonempty', bool(all_rewards) and finite, f'{len(all_rewards)} rewards, finite={finite}'),
        ('reward_in_sane_range', bool(all_rewards) and all(0.0 <= x <= 100.0 for x in all_rewards), ''),
        ('nonzero_advantage_signal', any_signal, 'a batch with reward variance -> real GRPO gradient'),
        ('generation_coherent', coherent, 'a non-empty generation containing digits'),
        ('trained_enough', trained >= max(2, len(expected_adapters)), f'{trained} train_k'),
    ]
    print('===== NPU_E2E_VERIFY =====', flush=True)
    for name, ok, detail in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}{(" — " + detail) if detail else ""}', flush=True)
    passed = all(ok for _, ok, _ in checks)
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(f'NPU_E2E_VERIFY {"PASS" if passed else "FAIL"} ({n_ok}/{len(checks)} checks)', flush=True)
    sys.exit(0 if passed else 1)


def main():
    cfg = build_cfg()
    device_groups = [
        DeviceGroup(name='model', ranks=[0], device_type='npu'),
        DeviceGroup(name='sampler', ranks=[1], device_type='npu'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=1, dp_size=1)
    sampler_mesh = DeviceMesh.from_sizes(world_size=1, dp_size=1)
    twinkle.initialize(mode='ray', nproc_per_node=2, groups=device_groups, lazy_collect=False)

    logger.info(get_device_placement())
    logger.info(f'NPU smoke: model={MODEL_ID} steps={MAX_STEPS} num_gen={NUM_GENERATIONS} data={DATA_NUM}')
    pipeline = NpuTransformersGRPOPipeline(cfg, model_mesh=model_mesh, sampler_mesh=sampler_mesh)
    history = pipeline.run(max_steps=MAX_STEPS)
    trained = sum(1 for step in history if step.get('train') is not None)
    print(f'SMOKE_DONE trained_train_k={trained} steps_run={len(history)}')
    _run_verify(trained)


if __name__ == '__main__':
    main()
