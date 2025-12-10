# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Dict, List, Optional

from pyxpg import (
    BufferBarrier,
    CommandBuffer,
    ImageBarrier,
    PipelineStageFlags,
    TimelineSemaphore,
)

from .ffx import MipGenerationRequest
from .utils.gpu import MipGenerationFilter


@dataclass
class SemaphoreInfo:
    sem: TimelineSemaphore
    wait_stage: PipelineStageFlags
    wait_value: int
    signal_stage: PipelineStageFlags
    signal_value: int


@dataclass
class RendererFrame:
    index: int
    total_index: int

    cmd: CommandBuffer

    # Before and after upload barriers
    upload_property_pipeline_stages: PipelineStageFlags
    upload_before_buffer_barriers: List[BufferBarrier]
    upload_before_image_barriers: List[ImageBarrier]
    upload_after_buffer_barriers: List[BufferBarrier]
    upload_after_image_barriers: List[ImageBarrier]

    # Mip generation requests
    mip_generation_requests: Dict[MipGenerationFilter, List[MipGenerationRequest]]

    before_render_src_pipeline_stages: PipelineStageFlags
    before_render_dst_pipeline_stages: PipelineStageFlags
    before_render_image_barriers: List[ImageBarrier]

    between_viewport_render_src_pipeline_stages: PipelineStageFlags
    between_viewport_render_dst_pipeline_stages: PipelineStageFlags

    additional_semaphores: List[SemaphoreInfo]

    transfer_cmd: Optional[CommandBuffer]
    transfer_semaphores: List[SemaphoreInfo]

    transfer_upload_before_buffer_barriers: List[BufferBarrier]
    transfer_upload_before_image_barriers: List[ImageBarrier]
    transfer_upload_after_buffer_barriers: List[BufferBarrier]
    transfer_upload_after_image_barriers: List[ImageBarrier]
