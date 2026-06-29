# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple


class PartitionStatus(StrEnum):
    OPEN = 'OPEN'
    ROLLOUT_DONE = 'ROLLOUT_DONE'
    REWARD_DONE = 'REWARD_DONE'
    TRAIN_READY = 'TRAIN_READY'
    TRAINING = 'TRAINING'
    TRAIN_DONE = 'TRAIN_DONE'
    CLEARED = 'CLEARED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class AdapterState(StrEnum):
    LOADING = 'LOADING'
    ACTIVE = 'ACTIVE'
    DRAINING = 'DRAINING'
    CANCELLED = 'CANCELLED'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class TrainingContext:
    tenant_id: str
    training_run_id: str
    base_model_id: str
    adapter_name: str
    adapter_revision: Optional[str] = None
    policy_version: int = 0
    env_type: str = 'tool_calling'
    tool_profile: str = 'default'
    reward_type: str = 'default'
    loss_type: str = 'default'
    algorithm: str = 'grpo'

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'

    def partition_id(self, train_id: int | str) -> str:
        suffix = train_id if isinstance(train_id, str) and train_id.startswith('train_') else f'train_{train_id}'
        return f'{self.key}/{suffix}'

    def with_policy_version(self, policy_version: int, adapter_revision: Optional[str] = None) -> 'TrainingContext':
        return TrainingContext(
            tenant_id=self.tenant_id,
            training_run_id=self.training_run_id,
            base_model_id=self.base_model_id,
            adapter_name=self.adapter_name,
            adapter_revision=self.adapter_revision if adapter_revision is None else adapter_revision,
            policy_version=policy_version,
            env_type=self.env_type,
            tool_profile=self.tool_profile,
            reward_type=self.reward_type,
            loss_type=self.loss_type,
            algorithm=self.algorithm,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            'tenant_id': self.tenant_id,
            'training_run_id': self.training_run_id,
            'base_model_id': self.base_model_id,
            'adapter_name': self.adapter_name,
            'adapter_revision': self.adapter_revision,
            'policy_version': self.policy_version,
            'env_type': self.env_type,
            'tool_profile': self.tool_profile,
            'reward_type': self.reward_type,
            'loss_type': self.loss_type,
            'algorithm': self.algorithm,
        }

    def validate_metadata(self, metadata: Dict[str, Any], *, strict_policy_version: bool = True) -> None:
        expected = self.metadata()
        for key, expected_value in expected.items():
            if key == 'adapter_revision':
                continue
            if key == 'policy_version' and not strict_policy_version:
                continue
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f'context metadata mismatch for {key}: expected {expected_value!r}, got {actual_value!r}')


@dataclass
class PartitionMetadata:
    context: TrainingContext
    partition_id: str
    policy_version: int
    target_groups: int = 0
    ready_groups: int = 0
    status: PartitionStatus = PartitionStatus.OPEN
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    owner_worker_id: Optional[str] = None
    lease_deadline: Optional[float] = None
    num_rows: int = 0

    @property
    def logical_train_id(self) -> str:
        return self.partition_id.rsplit('/', 1)[-1]

    def touch(self) -> None:
        self.updated_at = time.time()

    def tag(self) -> Dict[str, Any]:
        tag = self.context.metadata()
        # Sample-level policy_version / adapter_revision must remain attached
        # to each row. Partition tags carry lifecycle state and the version that
        # opened the partition, but must not overwrite row generation metadata.
        tag.pop('policy_version', None)
        tag.pop('adapter_revision', None)
        tag.update({
            'partition_id': self.partition_id,
            'partition_policy_version': self.policy_version,
            'target_groups': self.target_groups,
            'ready_groups': self.ready_groups,
            'status': self.status.value,
            'num_rows': self.num_rows,
        })
        return tag


@dataclass
class AdapterRecord:
    tenant_id: str
    training_run_id: str
    adapter_name: str
    base_model_id: str
    state: AdapterState = AdapterState.LOADING
    policy_version: int = 0
    adapter_revision: Optional[str] = None
    train_slot_name: Optional[str] = None
    rollout_slot_name: Optional[str] = None
    live_partitions: set[str] = field(default_factory=set)
    in_flight_rollouts: int = 0
    training_partition: Optional[str] = None
    sync_in_progress: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_error: Optional[str] = None
    weight: float = 1.0

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'

    def touch(self) -> None:
        self.updated_at = time.time()


@dataclass(frozen=True)
class RolloutCapacity:
    available_groups: int
    action: str = 'submit'
    reason: str = ''
    sleep_seconds: float = 0.0

    @property
    def can_submit(self) -> bool:
        return self.available_groups > 0 and self.action == 'submit'

    # --- spec FR-003 compatibility aliases (canonical fields are above) ---
    @property
    def available_slots(self) -> int:
        return self.available_groups

    @property
    def should_sleep(self) -> bool:
        return self.action == 'sleep'

    @property
    def should_throttle(self) -> bool:
        # staleness.py flags the near-watermark case with this reason (still submits).
        return self.reason == 'near_staleness_limit'


@dataclass
class RolloutGroupRequest:
    """FR-011 partial-rollout interface reservation (spec §3.11).

    Not consumed yet (run_one_group currently takes ``(context, sample)``); the
    fields are reserved so partial rollout can be added later without a breaking
    data-protocol change.
    """
    context: Any = None
    sample: Any = None
    partial_state: Optional[Any] = None
    abort_count: int = 0
    protected: bool = False


@dataclass
class RolloutGroupResult:
    """FR-011 partial-rollout interface reservation (spec §3.11)."""
    metadata: Any = None
    status: str = 'ok'  # ok / timeout / aborted / failed
    loss_mask: Optional[Any] = None


@dataclass
class RolloutContextState:
    context: TrainingContext
    pending_groups: int
    in_flight_rollouts: int
    live_partitions: int
    open_partitions: int
    train_ready_partitions: int
    rollout_capacity: int
    last_submit_time: float = 0.0
    submitted_groups: int = 0
    weight: float = 1.0

    @property
    def context_key(self) -> str:
        return self.context.key


@dataclass(frozen=True)
class ComponentResult:
    component: str
    kind: str
    metadata: Optional[PartitionMetadata] = None
    count: int = 0


SampleRecord = Dict[str, Any]
RewardFn = Any
AdvantageFn = Any
TrainResult = Dict[str, Any]
ContextKey = Tuple[str, str, str]
