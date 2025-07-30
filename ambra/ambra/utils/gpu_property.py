from pyxpg import BufferUsageFlags, ImageUsageFlags, Format, ImageUsage, TimelineSemaphore, PipelineStageFlags, AllocType, Context, Buffer, MemoryUsage, CommandBuffer

from .threadpool import Promise
from ..scene import Object, Property, view_bytes
from ..renderer import Renderer, RendererFrame, UploadMethod
from .gpu import UploadableBuffer, UploadableImage
from .lru_pool import LRUPool
from typing import Optional, List, Dict
from enum import Enum, auto
from dataclasses import dataclass

import numpy as np

class CpuBuffer:
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, alloc_type: AllocType, name: Optional[str] = None):
        self.buf = Buffer(ctx, size, usage_flags, alloc_type, name=name)
        self.used_size = 0
        self.promise = Promise()

    def __repr__(self):
        return self.buf.__repr__()

    def destroy(self):
        self.buf.destroy()


class GpuBufferState(Enum):
    EMPTY = auto()
    LOAD = auto()
    PREFETCH = auto()
    RENDER = auto()


@dataclass
class SemaphoreInfo:
    sem: TimelineSemaphore
    wait_stage: PipelineStageFlags
    wait_value: int
    signal_value: int


class GpuBuffer:
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, alloc_type: AllocType, use_transfer_queue: bool, name: Optional[str] = None):
        self.buf = Buffer(ctx, size, usage_flags, alloc_type, name=name)
        self.state = GpuBufferState.EMPTY

        if use_transfer_queue:
            self.semaphore = TimelineSemaphore(ctx, name=f"{name}-semaphore")
        else:
            self.semaphore = None
        self.semaphore_value = 0

    def use(self, stage: PipelineStageFlags) -> SemaphoreInfo:
        info = SemaphoreInfo(self.semaphore, stage, self.semaphore_value, self.semaphore_value + 1)
        self.semaphore_value += 1
        return info

    def __repr__(self):
        return f"(buf={self.buf.__repr__()}, state={self.state}, semaphore={self.semaphore_value})"

    def destroy(self):
        if self.semaphore is not None:
            self.semaphore.destroy()
        self.buf.destroy()


@dataclass
class PrefetchState:
    commands: CommandBuffer
    prefetch_done_value: int


class GpuBufferProperty:
    def __init__(self, o: Object, r: Renderer, property: Property[np.ndarray], usage_flags: BufferUsageFlags, name: str = None):
        self.property = property

        self.current = None

        # Upload
        if self.property.upload.preupload:
            self.buffers = [
                UploadableBuffer.from_data(r.ctx, view_bytes(property.get_frame_by_index(i)), usage_flags, name)
                for i in range(property.num_frames)
            ]
        else:
            size = self.property.max_size()

            cpu_prefetch_count = self.property.upload.cpu_prefetch_count
            gpu_prefetch_count = self.property.upload.gpu_prefetch_count

            cpu_buffers_count = r.window.num_frames + cpu_prefetch_count
            gpu_buffers_count = r.window.num_frames + gpu_prefetch_count

            self.cpu_buffers = [CpuBuffer(r.ctx, size, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST, name=f"cpubuf-{name}-{i}") for i in range(cpu_buffers_count)]
            self.cpu_pool = LRUPool(self.cpu_buffers, r.window.num_frames, cpu_prefetch_count)

            if r.upload_method != UploadMethod.CPU_BUF:
                gpu_alloc_type = AllocType.DEVICE_MAPPED if r.upload_method == UploadMethod.BAR else AllocType.HOST
                self.gpu_buffers = [GpuBuffer(r.ctx, size, usage_flags | BufferUsageFlags.TRANSFER_DST, gpu_alloc_type, r.upload_method == UploadMethod.TRANSFER_QUEUE, name=f"gpubuf-{name}-{i}") for i in range(gpu_buffers_count)]
                self.gpu_pool = LRUPool(self.gpu_buffers, r.window.num_frames, gpu_prefetch_count)

            if r.upload_method == UploadMethod.TRANSFER_QUEUE:
                self.prefetch_states = [
                    PrefetchState(
                        commands=CommandBuffer(r.ctx, queue_family_index=r.ctx.transfer_queue_family_index, name=f"gpu-prefetch-commands-{name}-{i}"),
                        prefetch_done_value=0,
                    ) for i in range(gpu_prefetch_count)
                ]
                self.prefetch_states_lookup: Dict[GpuBuffer, PrefetchState] = {}

        o.destroy_callbacks.append(lambda: self.destroy())

    def _load_async(self, i: int, buf: CpuBuffer, thread_index: int):
        buf.used_size = self.property.get_frame_by_index_into_async(i, buf.buf.data, thread_index)

    def preload(self, r: Renderer, frame: RendererFrame):
        # Issue CPU loads if async, otherwise just prepare buffer
        if not self.property.upload.preupload:
            self.cpu_pool.release_frame(frame.index)

            def cpu_load(k: int, buf: CpuBuffer):
                if self.property.upload.async_load:
                    r.thread_pool.submit(buf.promise, self._load_async, k, buf)
                else:
                    buf.used_size = self.property.get_frame_by_index_into(k, buf.buf.data)

            property_frame_index = self.property.current_frame_index
            if not self.gpu_pool.is_available_or_prefetching(property_frame_index):
                self.current_cpu_buf = self.cpu_pool.get(property_frame_index, cpu_load)

    def render(self, r: Renderer, frame: RendererFrame):
        # TODO: handle all upload modes
        # - CPU buf needs to handle
        # - BAR and gfx should be similar to before, but require correct buffer allocation type
        # - Transfer needs per-frame semaphores in RendererFrame
        #
        # TODO: I guess to handle images (until we have host_image_copy, if ever), we should just
        # force gfx or transfer upload methods depending on availability. Also are layout transitions
        # guaranteed to be supported on transfer queue? Or there is some limitation with it?
        # Not clear how we reuse all of this logic without becoming crazy.
        #
        # Good thing is that when this is done "I think" we support all of the common async upload
        # use cases, and can just write rendering code that does not know where the resources
        # come from or how they are uploaded.

        # Issue GPU loads
        property_frame_index = self.property.current_frame_index

        if self.property.upload.preupload:
            self.current = self.buffers[property_frame_index]
        else:
            # Wait for buffer to be ready
            if r.upload_method == UploadMethod.CPU_BUF:
                cpu_buf = self.current_cpu_buf
                self.current_cpu_buf = None

                if self.property.upload.async_load:
                    cpu_buf.promise.get()

                self.cpu_pool.use_frame(frame.index, property_frame_index)
                self.current = cpu_buf.buf
            else:
                def gpu_load(k: int, gpu_buf: GpuBuffer):
                    cpu_buf = self.current_cpu_buf
                    self.current_cpu_buf = None

                    # Wait for buffer to be ready
                    if self.property.upload.async_load:
                        cpu_buf.promise.get()

                    if r.upload_method == UploadMethod.BAR:
                        gpu_buf.buf.data[:] = cpu_buf.buf.data[:]

                        # Buffer is immediately not in use anymore. Add back to the LRU.
                        # This moves back the buffer to the front of the LRU queue.
                        self.cpu_pool.give_back(k, cpu_buf)
                        gpu_buf.state = GpuBufferState.RENDER
                    else:
                        self.cpu_pool.use_frame(frame.index, k)
                        if r.upload_method == UploadMethod.GFX:
                            # Upload on gfx queue
                            frame.cmd.copy_buffer(cpu_buf.buf, gpu_buf.buf)
                            frame.cmd.memory_barrier(MemoryUsage.TRANSFER_WRITE, MemoryUsage.VERTEX_INPUT)
                            gpu_buf.state = GpuBufferState.RENDER
                        else:
                            assert r.upload_method == UploadMethod.TRANSFER_QUEUE
                            assert gpu_buf.state == GpuBufferState.EMPTY or gpu_buf.state == GpuBufferState.RENDER or GpuBufferState.PREFETCH, gpu_buf.state

                            # Upload on copy queue
                            frame.copy_semaphores.append(gpu_buf.use(PipelineStageFlags.TRANSFER))

                            frame.copy_cmd.copy_buffer(cpu_buf.buf, gpu_buf.buf)
                            frame.copy_cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.TRANSFER_WRITE, MemoryUsage.NONE, r.ctx.transfer_queue_family_index, r.ctx.graphics_queue_family_index)

                            gpu_buf.state = GpuBufferState.LOAD

                def gpu_ensure(k: int, gpu_buf: GpuBuffer):
                    assert gpu_buf.state == GpuBufferState.PREFETCH, gpu_buf.state

                self.gpu_pool.release_frame(frame.index)
                gpu_buf = self.gpu_pool.get(property_frame_index, gpu_load, gpu_ensure)
                self.gpu_pool.use_frame(frame.index, property_frame_index)

                if gpu_buf.state == GpuBufferState.LOAD or gpu_buf.state == GpuBufferState.PREFETCH:
                    if gpu_buf.state == GpuBufferState.PREFETCH:
                        frame.copy_semaphores.append(gpu_buf.use(PipelineStageFlags.TOP_OF_PIPE))
                        frame.copy_cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.TRANSFER_WRITE, MemoryUsage.NONE, r.ctx.transfer_queue_family_index, r.ctx.graphics_queue_family_index)

                    frame.cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.NONE, MemoryUsage.VERTEX_INPUT, r.ctx.transfer_queue_family_index, r.ctx.graphics_queue_family_index)
                    frame.additional_semaphores.append(gpu_buf.use(PipelineStageFlags.VERTEX_INPUT))
                    gpu_buf.state = GpuBufferState.RENDER

                assert gpu_buf.state == GpuBufferState.RENDER, gpu_buf.state
                self.current = gpu_buf.buf

    def prefetch(self, r: Renderer, frame: RendererFrame):
        if not self.property.upload.preupload:
            if self.property.upload.async_load:
                # Issue prefetches
                def cpu_prefetch_cleanup(k: int, buf: CpuBuffer) -> bool:
                    if buf.promise.is_set():
                        return True
                    return False

                def cpu_prefetch(k: int, buf: CpuBuffer):
                    r.thread_pool.submit(buf.promise, self._load_async, k, buf)

                # TODO: can likely improve prefetch logic, and should probably allow
                # this to be hooked / configured somehow
                #
                # maybe we should just compute the state of next (maybe few) frames
                # (assuming constant dt, playback state) once globally and call prefetch with this.
                prefetch_start = self.property.current_frame_index + 1
                prefetch_end = prefetch_start + self.property.upload.cpu_prefetch_count
                prefetch_range = [self.property.get_frame_index(self.property.current_time, i) for i in range(prefetch_start, prefetch_end)]
                self.cpu_pool.prefetch(prefetch_range, cpu_prefetch_cleanup, cpu_prefetch)

            if r.ctx.has_transfer_queue:
                def gpu_prefetch_cleanup(k: int, gpu_buf: GpuBuffer):
                    state = self.prefetch_states_lookup[gpu_buf]
                    if gpu_buf.semaphore.get_value() >= state.prefetch_done_value:
                        # Release prefetch state
                        self.prefetch_states.append(state)
                        del self.prefetch_states_lookup[gpu_buf]

                        # Release buffer
                        self.cpu_pool.release_manual(k)

                        assert gpu_buf.state == GpuBufferState.RENDER or gpu_buf.state == GpuBufferState.PREFETCH
                        return True
                    return False

                def gpu_prefetch(k: int, gpu_buf: GpuBuffer):
                    # We know that the cpu buffer is available, so just get it
                    cpu_next: CpuBuffer = self.cpu_pool.get(k, None)
                    self.cpu_pool.use_manual(k)

                    # Get free prefetch state
                    state = self.prefetch_states.pop()
                    self.prefetch_states_lookup[gpu_buf] = state

                    with state.commands:
                        state.commands.copy_buffer(cpu_next.buf, gpu_buf.buf)

                    assert gpu_buf.state == GpuBufferState.EMPTY or GpuBufferState.PREFETCH or gpu_buf.state == GpuBufferState.RENDER, gpu_buf.state
                    info = gpu_buf.use(PipelineStageFlags.TRANSFER)
                    r.ctx.transfer_queue.submit(
                        state.commands,
                        wait_semaphores = [ (info.sem, info.wait_stage) ],
                        wait_timeline_values = [ info.wait_value ],
                        signal_semaphores = [ info.sem ],
                        signal_timeline_values = [ info.signal_value ],
                    )
                    state.prefetch_done_value = info.signal_value
                    gpu_buf.state = GpuBufferState.PREFETCH

                # TODO: fix, same as above
                prefetch_start = self.property.current_frame_index + 1
                prefetch_end = prefetch_start + self.property.upload.gpu_prefetch_count
                prefetch_range = [self.property.get_frame_index(self.property.current_time, i) for i in range(prefetch_start, prefetch_end)]
                self.gpu_pool.prefetch([i for i in prefetch_range if self.cpu_pool.is_available(i)], gpu_prefetch_cleanup, gpu_prefetch)

    def get_current(self):
        return self.current

    def destroy(self):
        self.current = None
        if self.property.upload.preupload:
            for buf in self.buffers:
                buf.destroy()
            self.buffers.clear()
            self.cpu_pool.clear()
        else:
            for buf in self.cpu_buffers:
                buf.destroy()
            self.cpu_buffers.clear()
            for buf in self.gpu_buffers:
                buf.destroy()
            self.gpu_buffers.clear()
            self.gpu_pool.clear()


_channels_dtype_int_to_format_table = {
    # normalized formats
    (1, np.dtype(np.uint8),  False): Format.R8_UNORM,
    (2, np.dtype(np.uint8),  False): Format.R8G8_UNORM,
    (3, np.dtype(np.uint8),  False): Format.R8G8B8_UNORM,
    (4, np.dtype(np.uint8),  False): Format.R8G8B8A8_UNORM,
    (1, np.dtype(np.int8),   False): Format.R8_SNORM,
    (2, np.dtype(np.int8),   False): Format.R8G8_SNORM,
    (3, np.dtype(np.int8),   False): Format.R8G8B8_SNORM,
    (4, np.dtype(np.int8),   False): Format.R8G8B8A8_SNORM,
    (1, np.dtype(np.uint16), False): Format.R16_UNORM,
    (2, np.dtype(np.uint16), False): Format.R16G16_UNORM,
    (3, np.dtype(np.uint16), False): Format.R16G16B16_UNORM,
    (4, np.dtype(np.uint16), False): Format.R16G16B16A16_UNORM,
    (1, np.dtype(np.int16),  False): Format.R16_SNORM,
    (2, np.dtype(np.int16),  False): Format.R16G16_SNORM,
    (3, np.dtype(np.int16),  False): Format.R16G16B16_SNORM,
    (4, np.dtype(np.int16),  False): Format.R16G16B16A16_SNORM,

    # integer formats
    (1, np.dtype(np.uint8),  True): Format.R8_UINT,
    (2, np.dtype(np.uint8),  True): Format.R8G8_UINT,
    (3, np.dtype(np.uint8),  True): Format.R8G8B8_UINT,
    (4, np.dtype(np.uint8),  True): Format.R8G8B8A8_UINT,
    (1, np.dtype(np.int8),   True): Format.R8_SINT,
    (2, np.dtype(np.int8),   True): Format.R8G8_SINT,
    (3, np.dtype(np.int8),   True): Format.R8G8B8_SINT,
    (4, np.dtype(np.int8),   True): Format.R8G8B8A8_SINT,
    (1, np.dtype(np.uint16), True): Format.R16_UINT,
    (2, np.dtype(np.uint16), True): Format.R16G16_UINT,
    (3, np.dtype(np.uint16), True): Format.R16G16B16_UINT,
    (4, np.dtype(np.uint16), True): Format.R16G16B16A16_UINT,
    (1, np.dtype(np.int16),  True): Format.R16_SINT,
    (2, np.dtype(np.int16),  True): Format.R16G16_SINT,
    (3, np.dtype(np.int16),  True): Format.R16G16B16_SINT,
    (4, np.dtype(np.int16),  True): Format.R16G16B16A16_SINT,
    (1, np.dtype(np.uint32), True): Format.R32_UINT,
    (2, np.dtype(np.uint32), True): Format.R32G32_UINT,
    (3, np.dtype(np.uint32), True): Format.R32G32B32_UINT,
    (4, np.dtype(np.uint32), True): Format.R32G32B32A32_UINT,
    (1, np.dtype(np.int32),  True): Format.R32_SINT,
    (2, np.dtype(np.int32),  True): Format.R32G32_SINT,
    (3, np.dtype(np.int32),  True): Format.R32G32B32_SINT,
    (4, np.dtype(np.int32),  True): Format.R32G32B32A32_SINT,

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

class GpuImageProperty:
    def __init__(self, o: Object, r: Renderer, property: Property[np.ndarray], usage_flags: ImageUsageFlags, usage: ImageUsage, name: str = None):
        self.property = property

        # Upload
        prefer_preupload = r.prefer_preupload if property.prefer_preupload is None else property.prefer_preupload
        if prefer_preupload:
            self.images: List[UploadableImage] = []
            for i in range(property.num_frames):
                frame = property.get_frame_by_index(i)
                if len(frame.shape) != 3:
                    raise ValueError(f"Expected shape of length 3. Got: {len(frame.shape)}")

                height, width, channels = frame.shape
                try:
                    format = _channels_dtype_int_to_format_table[(channels, frame.dtype, False)]
                except KeyError:
                    raise ValueError(f"Combination of channels ({channels}) and dtype ({frame.dtype}) does not match any supported image format")

                img = UploadableImage.from_data(r.ctx, view_bytes(frame), usage, width, height, format, usage_flags, name)
                self.images.append(img)
        else:
            raise NotImplemented()

        o.update_callbacks.append(lambda time, frame: self.update(time, frame))
        o.destroy_callbacks.append(lambda: self.destroy())

    def update(self, time: int, frame: float):
        pass

    def get_current(self):
        return self.images[self.property.current_frame]

    def destroy(self):
        for img in self.images:
            img.destroy()
        self.images.clear()

