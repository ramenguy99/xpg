from pyxpg import CommandBuffer, Image, DescriptorSet, PipelineStageFlags, TimelineSemaphore

from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class SemaphoreInfo:
    sem: TimelineSemaphore
    wait_stage: PipelineStageFlags
    wait_value: int
    signal_value: int

@dataclass
class RendererFrame:
    cmd: CommandBuffer
    image: Image
    index: int
    total_index: int

    viewport: Tuple[float, float, float, float]
    rect: Tuple[int, int, int, int]

    descriptor_set: DescriptorSet
    additional_semaphores: List[SemaphoreInfo]

    copy_cmd: Optional[CommandBuffer]
    copy_semaphores: List[SemaphoreInfo]

