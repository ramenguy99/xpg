# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from pyxpg import (
    AllocType,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    Context,
    Fence,
    Format,
    Image,
    ImageLayout,
    ImageUsageFlags,
    MemoryUsage,
)

from .renderer import FrameInputs
from .utils.gpu import _format_to_channels_dtype_int_table, get_image_pitch_rows_and_texel_size
from .utils.ring_buffer import RingBuffer


@dataclass
class HeadlessSwapchainFrame:
    image: Optional[Image]
    readback_buffer: Optional[Buffer]
    command_buffer: CommandBuffer
    transfer_command_buffer: Optional[CommandBuffer]
    fence: Fence

    def issue_readback(self) -> None:
        assert self.readback_buffer is not None
        assert self.image is not None

        self.command_buffer.image_barrier(
            self.image, ImageLayout.TRANSFER_SRC_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_SRC
        )
        self.command_buffer.copy_image_to_buffer(self.image, self.readback_buffer)
        self.command_buffer.memory_barrier(MemoryUsage.TRANSFER_SRC, MemoryUsage.HOST_READ)

    def realize_readback(self) -> NDArray[Any]:
        assert self.readback_buffer is not None
        assert self.image is not None

        self.fence.wait()

        channels, dtype, _ = _format_to_channels_dtype_int_table[self.image.format]
        shape = (self.image.height, self.image.width, channels)
        return np.frombuffer(self.readback_buffer.data, dtype).copy().reshape(shape)


class HeadlessSwapchain:
    def __init__(self, ctx: Context, num_frames_in_flight: int, format: Format):
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.format = format
        self.width = 0
        self.height = 0

        self.frames = RingBuffer(
            [
                HeadlessSwapchainFrame(
                    image=None,
                    readback_buffer=None,
                    command_buffer=CommandBuffer(ctx, name=f"windowless-swapchain-commands-{i}"),
                    transfer_command_buffer=CommandBuffer(
                        ctx, ctx.transfer_queue_family_index, name=f"windowless-swapchain-transfer-commands-{i}"
                    )
                    if self.ctx.has_transfer_queue
                    else None,
                    fence=Fence(ctx, signaled=True, name=f"windowless-swapchain-fence-{i}"),
                )
                for i in range(num_frames_in_flight)
            ]
        )

    def ensure_size(self, width: int, height: int) -> None:
        if self.width == width and self.height == height:
            return

        pitch, rows, _ = get_image_pitch_rows_and_texel_size(width, height, self.format)
        size = pitch * rows

        for i, f in enumerate(self.frames):
            f.image = Image(
                self.ctx,
                width,
                height,
                self.format,
                ImageUsageFlags.TRANSFER_SRC | ImageUsageFlags.TRANSFER_DST | ImageUsageFlags.COLOR_ATTACHMENT,
                AllocType.DEVICE,
                name=f"windowless-swapchain-image-{i}",
            )
            f.readback_buffer = Buffer(
                self.ctx,
                size,
                BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.TRANSFER_SRC,
                AllocType.HOST,
                name=f"windowless-swapchain-buffer-{i}",
            )

    def begin_frame(self) -> FrameInputs:
        frame = self.frames.get_current()
        assert frame.readback_buffer is not None
        assert frame.image is not None

        frame.fence.wait_and_reset()
        return FrameInputs(
            frame.image,
            frame.command_buffer,
            frame.transfer_command_buffer,
            [],
            [],
        )

    def end_frame(self, frame_inputs: FrameInputs) -> None:
        frame = self.frames.get_current()
        self.ctx.queue.submit(
            frame.command_buffer,
            wait_semaphores=[(s.sem, s.wait_value, s.wait_stage) for s in frame_inputs.additional_semaphores],
            signal_semaphores=[(s.sem, s.signal_value, s.signal_stage) for s in frame_inputs.additional_semaphores],
            fence=frame.fence,
        )

    def advance(self) -> None:
        self.frames.advance()

    def get_current(self) -> HeadlessSwapchainFrame:
        return self.frames.get_current()

    def get_current_and_advance(self) -> HeadlessSwapchainFrame:
        return self.frames.get_current_and_advance()
