# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pyxpg import (
    AllocType,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    Context,
    CullMode,
    Fence,
    Format,
    Image,
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    MemoryUsage,
    get_format_info,
)

from .ring_buffer import RingBuffer


def align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


def div_round_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def view_bytes(a: NDArray[Any]) -> memoryview:
    return a.reshape((-1,)).view(np.uint8).data


@dataclass
class BufferUploadInfo:
    data: memoryview
    buffer: Buffer
    offset: int


class MipGenerationFilter(Enum):
    AVERAGE = auto()
    AVERAGE_SRGB = auto()
    MAX = auto()
    MIN = auto()


@dataclass
class ImageUploadInfo:
    data: memoryview
    image: Image
    layout: ImageLayout

    level_0_view: Optional[ImageView]  # If None, no mip generation will happen
    mip_views: List[ImageView]
    mip_generation_filter: MipGenerationFilter


@dataclass
class BulkUploadState:
    buffer: Buffer
    cmd: CommandBuffer
    fence: Fence
    submitted: bool


class BulkUploader:
    def __init__(self, ctx: Context, size: int, count: int):
        self.ctx = ctx

        self.upload_states: List[BulkUploadState] = []
        upload_alignment = max(
            ctx.device_properties.limits.optimal_buffer_copy_offset_alignment,
            ctx.device_properties.limits.optimal_buffer_copy_row_pitch_alignment,
        )

        if ctx.has_transfer_queue:
            self.queue = ctx.transfer_queue
            self.queue_family_index = ctx.transfer_queue_family_index
        else:
            self.queue = ctx.queue
            self.queue_family_index = ctx.graphics_queue_family_index

        for i in range(count):
            self.upload_states.append(
                BulkUploadState(
                    buffer=Buffer(
                        ctx,
                        align_up(size, upload_alignment),
                        BufferUsageFlags.TRANSFER_SRC,
                        AllocType.HOST,
                        name=f"bulk-upload-buffer-{i}",
                    ),
                    cmd=CommandBuffer(ctx, self.queue_family_index, name=f"bulk-upload-commands-{i}"),
                    fence=Fence(ctx, name=f"bulk-upload-fence-{i}"),
                    submitted=False,
                )
            )

    def bulk_upload(
        self, uploads: List[Union[BufferUploadInfo, ImageUploadInfo]], mip_generation_requests: List[ImageUploadInfo]
    ) -> None:
        # Compute alignment requirements
        offset_alignment = max(self.ctx.device_properties.limits.optimal_buffer_copy_offset_alignment, 16)
        pitch_alignment = max(self.ctx.device_properties.limits.optimal_buffer_copy_row_pitch_alignment, 16)

        # State of current upload
        start_image_row = 0
        start_buffer_offset = 0

        # Stats
        total_bytes_uploaded = 0
        # print(f"Bulk uploads: {len(uploads)}")
        # begin_timestamp = perf_counter_ns()

        i = 0
        upload_buffer_index = 0
        while i < len(uploads):
            # print(f"Upload index {upload_buffer_index:3}")
            state = self.upload_states[upload_buffer_index]

            if state.submitted:
                state.fence.wait_and_reset()
                state.submitted = False

            with state.cmd as cmd:
                offset = 0

                while i < len(uploads) and offset < state.buffer.size:
                    state = self.upload_states[upload_buffer_index]

                    # Space left in upload buffer
                    remaining_size = state.buffer.size - offset

                    info = uploads[i]

                    if isinstance(info, BufferUploadInfo):
                        total_size = len(info.data)
                        size = total_size - start_buffer_offset
                        fitting_size = min(remaining_size, size)

                        # print(f"    Buffer {i:3} - offset {offset:12} - size {size:12} - remaining_size {remaining_size:12} - fitting size {fitting_size:12} - bufoffset {start_buffer_offset:12}")
                        state.buffer.data[offset : offset + fitting_size] = info.data[
                            start_buffer_offset : start_buffer_offset + fitting_size
                        ]

                        cmd.copy_buffer_range(
                            state.buffer, info.buffer, fitting_size, offset, info.offset + start_buffer_offset
                        )

                        if start_buffer_offset + fitting_size == total_size:
                            start_buffer_offset = 0
                            i += 1
                            total_bytes_uploaded += total_size
                        else:
                            start_buffer_offset += fitting_size
                    elif isinstance(info, ImageUploadInfo):
                        input_pitch, total_rows, texel_size = get_image_pitch_rows_and_texel_size(
                            info.image.width, info.image.height, info.image.format
                        )
                        if input_pitch * total_rows != len(info.data):
                            # print(type(info.data), info.data.shape)
                            raise ValueError(
                                f"ImageUploadInfo data size ({len(info.data)}) does not match pitch and rows ({input_pitch} x {total_rows} = {input_pitch * total_rows})"
                            )

                        # Adjust rows by number of rows already uploaded and align pitch
                        rows = total_rows - start_image_row
                        pitch = align_up(input_pitch, pitch_alignment)

                        if pitch > state.buffer.size:
                            raise RuntimeError(
                                f"Image pitch ({pitch}) is larger than upload buffer size ({state.buffer.size})"
                            )

                        # Stop if we can't fit a single row
                        if pitch > remaining_size:
                            break

                        # Compute how many rows we are actually uploading in this command
                        fitting_rows = min(rows, remaining_size // pitch)
                        fitting_size = fitting_rows * pitch

                        # print(f"    Image  {i:3} - offset {offset:12} - pitch {pitch:12} - rows {rows:12} - remaining_size {remaining_size:12} - fitting_rows {fitting_rows:12} - fitting_size {fitting_size:12} - start_image_row {start_image_row:12} - pitch alignment {pitch_alignment}")

                        # Copy image data to staging buffer expanding to correct pitch
                        upload_buffer_range = state.buffer.data[offset : offset + fitting_size]
                        # print(upload_buffer_range.shape, info.data.shape)
                        upload_buffer_2d_view = np.frombuffer(upload_buffer_range, np.uint8).reshape(
                            (fitting_rows, pitch)
                        )
                        data_buffer_2d_view = np.frombuffer(info.data, np.uint8).reshape((total_rows, input_pitch))

                        # print(upload_buffer_2d_view.shape, data_buffer_2d_view.shape)

                        # TODO: correctly handle block formats here, need to think about data shape looks for those
                        upload_buffer_2d_view[:, :input_pitch] = data_buffer_2d_view[
                            start_image_row : start_image_row + fitting_rows, :input_pitch
                        ]

                        # Tranisition to TRANSFER_DST_OPTIMAL if first upload
                        if start_image_row == 0:
                            cmd.image_barrier(
                                info.image,
                                ImageLayout.TRANSFER_DST_OPTIMAL,
                                MemoryUsage.NONE,
                                MemoryUsage.TRANSFER_DST,
                                undefined=True,
                            )

                        # Upload to image range
                        cmd.copy_buffer_to_image_range(
                            state.buffer,
                            info.image,
                            info.image.width,
                            fitting_rows,
                            0,
                            start_image_row,
                            offset,
                            pitch // texel_size,
                        )

                        if start_image_row + fitting_rows == total_rows:
                            # Transition to final layout if last upload
                            final_layout = info.layout

                            # Register for mip generation if requested and transition to GENERAL layout instead
                            if info.level_0_view is not None:
                                mip_generation_requests.append(info)
                                final_layout = ImageLayout.GENERAL

                            cmd.image_barrier(info.image, final_layout, MemoryUsage.TRANSFER_DST, MemoryUsage.NONE)
                            i += 1
                            start_image_row = 0
                            total_bytes_uploaded += total_rows * pitch
                        else:
                            start_image_row += fitting_rows
                    else:
                        raise TypeError(f"Expected BufferUploadInfo or ImageUploadInfo. Got {type(info)}")

                    # Advance image and buffer offset
                    offset += align_up(fitting_size, offset_alignment)

            # Submit batch signaling fence
            self.queue.submit(state.cmd, fence=state.fence)
            state.submitted = True

            # Advance buffer
            upload_buffer_index = (upload_buffer_index + 1) % len(self.upload_states)

        for s in self.upload_states:
            if s.submitted:
                s.fence.wait_and_reset()
                s.submitted = False

        # elapsed = (perf_counter_ns() - begin_timestamp) * 1e-9
        # print(f"Bulk upload of {total_bytes_uploaded/(1024 * 1024):.3f}MB in {elapsed * 1e3:6.3f}ms ({total_bytes_uploaded / (1024 * 1024 * 1024) / elapsed:.3f}GB/s)")


class UploadableBuffer(Buffer):
    def __init__(
        self,
        ctx: Context,
        size: int,
        usage_flags: BufferUsageFlags,
        name: Optional[str] = None,
    ):
        # TODO: we likely want an option to ensure we do not use BAR even if available for cases
        # where we want the GPU to do the upload.
        #
        # Maybe we should also handle integrated and CPU differently here
        super().__init__(
            ctx,
            size,
            usage_flags | BufferUsageFlags.TRANSFER_DST,
            AllocType.DEVICE_MAPPED_WITH_FALLBACK,
            name,
        )
        if not self.is_mapped:
            if name is not None:
                name = f"{name} - staging"
            self._staging = Buffer(
                ctx,
                size,
                BufferUsageFlags.TRANSFER_SRC,
                AllocType.HOST_WRITE_COMBINING,
                name,
            )

    @classmethod
    def from_data(  # type: ignore
        cls,
        ctx: Context,
        data: memoryview,
        usage_flags: BufferUsageFlags,
        name: Optional[str] = None,
    ) -> "UploadableBuffer":
        buf = cls(ctx, len(data), usage_flags, name=name)
        buf.upload_sync(data)
        return buf

    def upload(
        self,
        cmd: CommandBuffer,
        usage: MemoryUsage,
        data: Union[memoryview, NDArray[np.uint8]],
        offset: int = 0,
    ) -> None:
        if self.is_mapped:
            # print(self.data.c_contiguous, data.c_contiguous )
            # print(self.data.shape       , data.shape        )
            # print(self.data.readonly    , data.readonly     )
            # print(self.data.format    , data.format     )
            self.data[offset : offset + len(data)] = data
        else:
            self._staging.data[offset : offset + len(data)] = data
            cmd.copy_buffer_range(
                self._staging,
                self,
                len(data),
                src_offset=offset,
                dst_offset=offset,
            )
            if usage != MemoryUsage.NONE:
                cmd.memory_barrier(MemoryUsage.TRANSFER_DST, usage)

    def upload_sync(self, data: Union[memoryview, NDArray[np.uint8]], offset: int = 0) -> None:
        if self.is_mapped:
            # print(self.data.c_contiguous, data.c_contiguous )
            # print(self.data.shape       , data.shape        )
            # print(self.data.readonly    , data.readonly     )
            # print(self.data.format    , data.format     )
            self.data[offset : offset + len(data)] = data
        else:
            with self.ctx.sync_commands() as cmd:
                self.upload(cmd, MemoryUsage.NONE, data, offset)


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def get_image_pitch_rows_and_texel_size(width: int, height: int, format: Format) -> Tuple[int, int, int]:
    info = get_format_info(format)
    if info.size_of_block_in_bytes > 0:
        size = info.size_of_block_in_bytes
        pitch = div_ceil(width, info.block_side_in_pixels) * size
        rows = div_ceil(height, info.block_side_in_pixels)
    else:
        size = info.size
        pitch = width * size
        rows = height
    return pitch, rows, size


# NOTE: in the future this could use VK_EXT_host_image_copy to do uploads / transitions if available
class UploadableImage(Image):
    def __init__(
        self,
        ctx: Context,
        width: int,
        height: int,
        format: Format,
        usage_flags: ImageUsageFlags,
        dedicated_alloc: bool = False,
        name: Optional[str] = None,
    ):
        pitch, rows, _ = get_image_pitch_rows_and_texel_size(width, height, format)
        self._staging = Buffer(
            ctx,
            pitch * rows,
            BufferUsageFlags.TRANSFER_SRC,
            AllocType.HOST_WRITE_COMBINING,
            f"{name} - staging",
        )
        super().__init__(
            ctx,
            width,
            height,
            format,
            usage_flags | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE_DEDICATED if dedicated_alloc else AllocType.DEVICE,
        )

    @classmethod
    def from_data(  # type: ignore
        cls,
        ctx: Context,
        data: memoryview,
        layout: ImageLayout,
        width: int,
        height: int,
        format: Format,
        usage_flags: ImageUsageFlags,
        dedicated_alloc: bool = False,
        name: Optional[str] = None,
    ) -> "UploadableImage":
        buf = cls(ctx, width, height, format, usage_flags, dedicated_alloc, name)
        buf.upload_sync(data, layout)
        return buf

    def upload(
        self,
        cmd: CommandBuffer,
        layout: ImageLayout,
        src_usage: MemoryUsage,
        dst_usage: MemoryUsage,
        data: Union[memoryview, NDArray[np.uint8]],
    ) -> None:
        # Upload to staging buffer
        if data.shape is None or len(data.shape) != 1:
            raise ValueError(f"data must be flat array. Got shape: {data.shape}")
        if len(data) != self._staging.size:
            raise ValueError(f"data must be of size {self._staging.size}. Got size: {len(data)}")
        self._staging.data[:] = data
        cmd.image_barrier(
            self,
            ImageLayout.TRANSFER_DST_OPTIMAL,
            src_usage,
            MemoryUsage.TRANSFER_DST,
            undefined=True,
        )
        cmd.copy_buffer_to_image(self._staging, self)
        if layout == ImageLayout.UNDEFINED or layout == ImageLayout.TRANSFER_DST_OPTIMAL:
            cmd.memory_barrier(MemoryUsage.TRANSFER_DST, dst_usage)
        else:
            cmd.image_barrier(self, layout, MemoryUsage.TRANSFER_DST, dst_usage)

    def upload_sync(self, data: Union[memoryview, NDArray[np.uint8]], layout: ImageLayout) -> None:
        with self.ctx.sync_commands() as cmd:
            self.upload(cmd, layout, MemoryUsage.NONE, MemoryUsage.NONE, data)


@dataclass
class UniformBlockAllocation:
    buffer: UploadableBuffer
    offset: int
    size: int

    def upload(self, cmd: CommandBuffer, data: Union[memoryview, NDArray[np.uint8]]) -> None:
        if len(data) > self.size:
            raise IndexError("data is larger than buffer allocation")
        self.buffer.upload(cmd, MemoryUsage.SHADER_UNIFORM, data, self.offset)


@dataclass
class UniformBlock:
    buffers: RingBuffer[UploadableBuffer]
    size: int
    used: int = 0

    def alloc(self, size: int, alignment: int) -> UniformBlockAllocation:
        assert self.used + size < self.size
        alloc = UniformBlockAllocation(
            self.buffers.get_current(),
            self.used,
            size,
        )
        self.used += (size + alignment - 1) & ~(alignment - 1)
        return alloc

    def advance(self) -> None:
        self.buffers.advance()
        self.used = 0


class UniformPool:
    def __init__(self, ctx: Context, num_frames: int, block_size: int):
        self.ctx = ctx
        self.num_frames = num_frames
        self.block_size = block_size
        self.blocks: List[UniformBlock] = []
        self.alignment = max(
            16,
            self.ctx.device_properties.limits.min_uniform_buffer_offset_alignment,
        )
        self.max_uniform_buffer_range = self.ctx.device_properties.limits.max_uniform_buffer_range

        # Warmup first block
        self._alloc_block(block_size)

    def _alloc_block(self, min_size: int) -> UniformBlock:
        size = max(min_size, self.block_size)
        block_idx = len(self.blocks)
        block = UniformBlock(
            buffers=RingBuffer(
                [
                    UploadableBuffer(
                        self.ctx,
                        size,
                        BufferUsageFlags.UNIFORM,
                        name=f"uniform-pool-block-buf{block_idx}",
                    )
                    for _ in range(self.num_frames)
                ]
            ),
            size=size,
        )

        # Sync ringbuffer index, not necessary but makes for easier debugging
        if self.blocks:
            index = self.blocks[0].buffers.index
            block.buffers.set(index)

        self.blocks.append(block)
        return block

    def alloc(self, size: int) -> UniformBlockAllocation:
        if size > self.max_uniform_buffer_range:
            raise RuntimeError(
                f"allocation size ({size}) is larger than device limit ({self.max_uniform_buffer_range})"
            )

        for b in self.blocks:
            if b.used + size < b.size:
                return b.alloc(size, self.alignment)

        # No space left, alloc a new block
        return self._alloc_block(size).alloc(size, self.alignment)

    def advance(self) -> None:
        for b in self.blocks:
            b.advance()


_channels_dtype_int_to_format_table: Dict[Tuple[int, np.dtype, bool], Format] = {
    # normalized formats
    (1, np.dtype(np.uint8), False): Format.R8_UNORM,
    (2, np.dtype(np.uint8), False): Format.R8G8_UNORM,
    (3, np.dtype(np.uint8), False): Format.R8G8B8_UNORM,
    (4, np.dtype(np.uint8), False): Format.R8G8B8A8_UNORM,
    (1, np.dtype(np.int8), False): Format.R8_SNORM,
    (2, np.dtype(np.int8), False): Format.R8G8_SNORM,
    (3, np.dtype(np.int8), False): Format.R8G8B8_SNORM,
    (4, np.dtype(np.int8), False): Format.R8G8B8A8_SNORM,
    (1, np.dtype(np.uint16), False): Format.R16_UNORM,
    (2, np.dtype(np.uint16), False): Format.R16G16_UNORM,
    (3, np.dtype(np.uint16), False): Format.R16G16B16_UNORM,
    (4, np.dtype(np.uint16), False): Format.R16G16B16A16_UNORM,
    (1, np.dtype(np.int16), False): Format.R16_SNORM,
    (2, np.dtype(np.int16), False): Format.R16G16_SNORM,
    (3, np.dtype(np.int16), False): Format.R16G16B16_SNORM,
    (4, np.dtype(np.int16), False): Format.R16G16B16A16_SNORM,
    # integer formats
    (1, np.dtype(np.uint8), True): Format.R8_UINT,
    (2, np.dtype(np.uint8), True): Format.R8G8_UINT,
    (3, np.dtype(np.uint8), True): Format.R8G8B8_UINT,
    (4, np.dtype(np.uint8), True): Format.R8G8B8A8_UINT,
    (1, np.dtype(np.int8), True): Format.R8_SINT,
    (2, np.dtype(np.int8), True): Format.R8G8_SINT,
    (3, np.dtype(np.int8), True): Format.R8G8B8_SINT,
    (4, np.dtype(np.int8), True): Format.R8G8B8A8_SINT,
    (1, np.dtype(np.uint16), True): Format.R16_UINT,
    (2, np.dtype(np.uint16), True): Format.R16G16_UINT,
    (3, np.dtype(np.uint16), True): Format.R16G16B16_UINT,
    (4, np.dtype(np.uint16), True): Format.R16G16B16A16_UINT,
    (1, np.dtype(np.int16), True): Format.R16_SINT,
    (2, np.dtype(np.int16), True): Format.R16G16_SINT,
    (3, np.dtype(np.int16), True): Format.R16G16B16_SINT,
    (4, np.dtype(np.int16), True): Format.R16G16B16A16_SINT,
    (1, np.dtype(np.uint32), True): Format.R32_UINT,
    (2, np.dtype(np.uint32), True): Format.R32G32_UINT,
    (3, np.dtype(np.uint32), True): Format.R32G32B32_UINT,
    (4, np.dtype(np.uint32), True): Format.R32G32B32A32_UINT,
    (1, np.dtype(np.int32), True): Format.R32_SINT,
    (2, np.dtype(np.int32), True): Format.R32G32_SINT,
    (3, np.dtype(np.int32), True): Format.R32G32B32_SINT,
    (4, np.dtype(np.int32), True): Format.R32G32B32A32_SINT,
    # float formats
    (1, np.dtype(np.float16), False): Format.R16_SFLOAT,
    (2, np.dtype(np.float16), False): Format.R16G16_SFLOAT,
    (3, np.dtype(np.float16), False): Format.R16G16B16_SFLOAT,
    (4, np.dtype(np.float16), False): Format.R16G16B16A16_SFLOAT,
    (1, np.dtype(np.float32), False): Format.R32_SFLOAT,
    (2, np.dtype(np.float32), False): Format.R32G32_SFLOAT,
    (3, np.dtype(np.float32), False): Format.R32G32B32_SFLOAT,
    (4, np.dtype(np.float32), False): Format.R32G32B32A32_SFLOAT,
    (1, np.dtype(np.float64), False): Format.R64_SFLOAT,
    (2, np.dtype(np.float64), False): Format.R64G64_SFLOAT,
    (3, np.dtype(np.float64), False): Format.R64G64B64_SFLOAT,
    (4, np.dtype(np.float64), False): Format.R64G64B64A64_SFLOAT,
}

_format_to_channels_dtype_int_table: Dict[Format, Tuple[int, np.dtype, bool]] = {
    v: k for k, v in _channels_dtype_int_to_format_table.items()
}

_format_to_srgb_format_table: Dict[Format, Format] = {
    Format.R8_UNORM: Format.R8_SRGB,
    Format.R8G8_UNORM: Format.R8G8_SRGB,
    Format.R8G8B8_UNORM: Format.R8G8B8_SRGB,
    Format.R8G8B8A8_UNORM: Format.R8G8B8A8_SRGB,
}


def format_from_channels_dtype(channels: int, dtype: np.dtype, integer: bool = False) -> Format:
    try:
        return _channels_dtype_int_to_format_table[(channels, dtype, integer)]
    except KeyError:
        raise RuntimeError(
            f"Channels ({channels}) and dtype ({dtype}) combination does not match any format"
        ) from KeyError


def cull_mode_opposite_face(mode: CullMode) -> CullMode:
    if mode == CullMode.FRONT:
        return CullMode.BACK
    elif mode == CullMode.BACK:
        return CullMode.FRONT
    else:
        return mode


def readback(ctx: Context, img: Image, new_layout: ImageLayout) -> NDArray[Any]:
    channels, dtype, _ = _format_to_channels_dtype_int_table[img.format]
    shape = (img.height, img.width, channels)

    buffer = Buffer(ctx, int(np.prod(shape)) * dtype.itemsize, BufferUsageFlags.TRANSFER_DST, AllocType.HOST)
    with ctx.sync_commands() as cmd:
        cmd.image_barrier(img, ImageLayout.TRANSFER_SRC_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_SRC)
        cmd.copy_image_to_buffer(img, buffer)
        cmd.image_barrier(img, new_layout, MemoryUsage.TRANSFER_SRC, MemoryUsage.SHADER_READ_ONLY)
        cmd.memory_barrier(MemoryUsage.TRANSFER_SRC, MemoryUsage.HOST_READ)

    return np.frombuffer(buffer, dtype).copy().reshape(shape)


def readback_mips(ctx: Context, img: Image, new_layout: ImageLayout) -> List[NDArray[Any]]:
    channels, dtype, _ = _format_to_channels_dtype_int_table[img.format]

    buffers = []
    for m in range(img.mip_levels):
        w, h = max(img.width >> m, 1), max(img.height >> m, 1)
        buffers.append(Buffer(ctx, w * h * channels * dtype.itemsize, BufferUsageFlags.TRANSFER_DST, AllocType.HOST))

    with ctx.sync_commands() as cmd:
        cmd.image_barrier(img, ImageLayout.TRANSFER_SRC_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_SRC)
        for m, buffer in zip(range(img.mip_levels), buffers):
            w, h = max(img.width >> m, 1), max(img.height >> m, 1)
            cmd.copy_image_to_buffer_range(img, buffer, w, h, image_mip=m)
        cmd.image_barrier(img, new_layout, MemoryUsage.TRANSFER_SRC, MemoryUsage.SHADER_READ_ONLY)
        cmd.memory_barrier(MemoryUsage.TRANSFER_SRC, MemoryUsage.HOST_READ)

    arrays: List[NDArray[Any]] = []
    for m, b in zip(range(img.mip_levels), buffers):
        w, h = max(img.width >> m, 1), max(img.height >> m, 1)
        shape = (h, w, channels)
        arrays.append(np.frombuffer(b, dtype).copy().reshape(shape))

    return arrays


def to_srgb_format(format: Format) -> Format:
    try:
        return _format_to_srgb_format_table[format]
    except KeyError:
        raise RuntimeError(f"{format} does not have a corresponding sRGB format") from KeyError
