# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import List, Optional

from pyxpg import (
    CommandBuffer,
    PipelineStageFlags,
    TimelineSemaphore,
)


@dataclass
class SemaphoreInfo:
    sem: TimelineSemaphore
    wait_stage: PipelineStageFlags
    wait_value: int
    signal_value: int


@dataclass
class RendererFrame:
    index: int
    total_index: int

    cmd: CommandBuffer
    additional_semaphores: List[SemaphoreInfo]

    copy_cmd: Optional[CommandBuffer]
    copy_semaphores: List[SemaphoreInfo]
