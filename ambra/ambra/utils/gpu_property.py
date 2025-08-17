from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Generic, List, Optional, TypeVar, Union

import numpy as np
from pyxpg import (
    AllocType,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    Context,
    Format,
    Image,
    ImageLayout,
    ImageUsageFlags,
    MemoryUsage,
    PipelineStageFlags,
    TimelineSemaphore,
)

from ..config import UploadMethod
from ..renderer_frame import RendererFrame, SemaphoreInfo
from ..scene import Property, view_bytes
from .gpu import UploadableBuffer, UploadableImage, get_image_pitch_and_rows
from .lru_pool import LRUPool
from .threadpool import Promise, ThreadPool


class CpuBuffer:
    def __init__(self, buffer: Buffer):
        self.buf = buffer
        self.used_size = 0
        self.promise = Promise[None]()

    def __repr__(self) -> str:
        return self.buf.__repr__()

    def destroy(self) -> None:
        self.buf.destroy()


class GpuResourceState(Enum):
    EMPTY = auto()
    LOAD = auto()
    PREFETCH = auto()
    RENDER = auto()


R = TypeVar("R", bound=Union[Buffer, Image])


class GpuResource(Generic[R]):
    def __init__(self, resource: R, semaphore: Optional[TimelineSemaphore]):
        self.resource = resource
        self.state = GpuResourceState.EMPTY
        self.semaphore = semaphore
        self.semaphore_value = 0

    def use(self, stage: PipelineStageFlags) -> SemaphoreInfo:
        assert self.semaphore is not None
        info = SemaphoreInfo(
            self.semaphore,
            stage,
            self.semaphore_value,
            self.semaphore_value + 1,
        )
        self.semaphore_value += 1
        return info

    def __repr__(self) -> str:
        return f"(resource={self.resource.__repr__()}, state={self.state}, semaphore={self.semaphore_value})"

    def destroy(self) -> None:
        if self.semaphore is not None:
            self.semaphore.destroy()
        self.resource.destroy()


@dataclass
class PrefetchState:
    commands: CommandBuffer
    prefetch_done_value: int


class GpuResourceProperty(Generic[R]):
    def __init__(
        self,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool[None],
        property: Property,
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ):
        self.ctx = ctx
        self.upload_method = upload_method
        self.thread_pool = thread_pool
        self.property = property
        self.name = name
        self.pipeline_stage_flags = pipeline_stage_flags

        self.current: Optional[R] = None
        self.current_cpu_buf: Optional[CpuBuffer] = None

        self.cpu_buffers: List[CpuBuffer] = []
        self.cpu_pool: Optional[LRUPool[int, CpuBuffer]] = None
        self.gpu_resources: List[GpuResource[R]] = []
        self.gpu_pool: Optional[LRUPool[int, GpuResource[R]]] = None
        self.prefetch_states: List[PrefetchState] = []
        self.prefetch_states_lookup: Dict[GpuResource[R], PrefetchState] = {}

        self.resources: List[R] = []

        # Upload
        if self.property.upload.preupload:
            for i in range(property.num_frames):
                frame = property.get_frame_by_index(i)
                res = self._create_uploaded_resource(frame, f"{name}-{i}")
                self.resources.append(res)
        else:
            cpu_prefetch_count = self.property.upload.cpu_prefetch_count
            gpu_prefetch_count = (
                self.property.upload.gpu_prefetch_count if upload_method == UploadMethod.TRANSFER_QUEUE else 0
            )

            cpu_buffers_count = num_frames_in_flight + cpu_prefetch_count + gpu_prefetch_count
            gpu_resources_count = num_frames_in_flight + gpu_prefetch_count

            self.cpu_buffers = [
                CpuBuffer(self._create_cpu_buffer(f"cpubuf-{name}-{i}")) for i in range(cpu_buffers_count)
            ]
            self.cpu_pool = LRUPool(self.cpu_buffers, num_frames_in_flight, cpu_prefetch_count)

            if upload_method != UploadMethod.CPU_BUF:
                for i in range(gpu_resources_count):
                    res = self._create_gpu_resource(f"gpubuf-{name}-{i}")
                    if upload_method == UploadMethod.TRANSFER_QUEUE:
                        semaphore = TimelineSemaphore(ctx, name=f"gpubuf-{name}-{i}-semaphore")
                    else:
                        semaphore = None
                    self.gpu_resources.append(GpuResource(res, semaphore))
                self.gpu_pool = LRUPool(
                    self.gpu_resources,
                    num_frames_in_flight,
                    gpu_prefetch_count,
                )

            if upload_method == UploadMethod.TRANSFER_QUEUE:
                self.prefetch_states = [
                    PrefetchState(
                        commands=CommandBuffer(
                            ctx,
                            queue_family_index=ctx.transfer_queue_family_index,
                            name=f"gpu-prefetch-commands-{name}-{i}",
                        ),
                        prefetch_done_value=0,
                    )
                    for i in range(gpu_prefetch_count)
                ]

    def _load_async(self, i: int, buf: CpuBuffer, thread_index: int) -> None:
        buf.used_size = self.property.get_frame_by_index_into(i, buf.buf.data, thread_index)

    def load(self, frame: RendererFrame) -> None:
        # Issue CPU loads if async
        if not self.property.upload.preupload:
            assert self.cpu_pool is not None

            self.cpu_pool.release_frame(frame.index)

            def cpu_load(k: int, buf: CpuBuffer) -> None:
                if self.property.upload.async_load:
                    self.thread_pool.submit(buf.promise, self._load_async, k, buf)  # type: ignore
                else:
                    buf.used_size = self.property.get_frame_by_index_into(k, buf.buf.data)

            property_frame_index = self.property.current_frame_index

            if self.upload_method == UploadMethod.CPU_BUF or (
                self.gpu_pool is not None and not self.gpu_pool.is_available_or_prefetching(property_frame_index)
            ):
                self.current_cpu_buf = self.cpu_pool.get(property_frame_index, cpu_load)

    def upload(self, frame: RendererFrame) -> None:
        # NOTE: unless we are doing BAR uploads here, we could delay
        # waiting for buffers to be ready to right before submit.
        # Even in the case of CPU uploads you could technically schedule them
        # asynchronously (ideally on a thread pool that does memcpy with the
        # GIL released) or have the loader thread do them (with care for
        # write combining memory). This way we can do python stuff until
        # the last moment we need the data to be ready.

        # Issue GPU loads
        property_frame_index = self.property.current_frame_index

        if self.property.upload.preupload:
            self.current = self.resources[property_frame_index]
        else:
            assert self.cpu_pool is not None

            # Wait for buffer to be ready
            if self.upload_method == UploadMethod.CPU_BUF:
                assert self.current_cpu_buf is not None
                cpu_buf = self.current_cpu_buf
                self.current_cpu_buf = None

                if self.property.upload.async_load:
                    cpu_buf.promise.get()

                self.cpu_pool.use_frame(frame.index, property_frame_index)

                # NOTE: only works for buffers for now, which is why we get a type error here.
                self.current = cpu_buf.buf  # type: ignore
            else:
                assert self.gpu_pool is not None

                def gpu_load(k: int, gpu_res: GpuResource[R]) -> None:
                    assert self.cpu_pool is not None
                    assert self.current_cpu_buf is not None

                    cpu_buf = self.current_cpu_buf
                    self.current_cpu_buf = None

                    # Wait for buffer to be ready
                    if self.property.upload.async_load:
                        cpu_buf.promise.get()

                    if self.upload_method == UploadMethod.BAR:
                        # NOTE: only works for buffers for now, which is why we get a type error here.
                        gpu_res.resource.data[: cpu_buf.used_size] = cpu_buf.buf.data[: cpu_buf.used_size]  # type: ignore

                        # Buffer is immediately not in use anymore. Add back to the LRU.
                        # This moves back the buffer to the front of the LRU queue.
                        self.cpu_pool.give_back(k, cpu_buf)
                        gpu_res.state = GpuResourceState.RENDER
                    else:
                        self.cpu_pool.use_frame(frame.index, k)
                        if self.upload_method == UploadMethod.GFX:
                            # Upload on gfx queue
                            self._cmd_upload(frame.cmd, cpu_buf, gpu_res)
                            self._cmd_barrier(frame.cmd, gpu_res)

                            gpu_res.state = GpuResourceState.RENDER
                        else:
                            assert self.upload_method == UploadMethod.TRANSFER_QUEUE
                            assert (
                                gpu_res.state == GpuResourceState.EMPTY
                                or gpu_res.state == GpuResourceState.RENDER
                                or GpuResourceState.PREFETCH
                            ), gpu_res.state
                            assert frame.copy_cmd is not None

                            frame.copy_semaphores.append(gpu_res.use(PipelineStageFlags.TRANSFER))

                            # Upload on copy queue
                            self._cmd_upload(frame.copy_cmd, cpu_buf, gpu_res)
                            self._cmd_release_barrier(frame.copy_cmd, gpu_res)

                            gpu_res.state = GpuResourceState.LOAD

                def gpu_ensure(k: int, gpu_res: GpuResource[R]) -> None:
                    assert gpu_res.state == GpuResourceState.PREFETCH, gpu_res.state

                self.gpu_pool.release_frame(frame.index)
                gpu_res = self.gpu_pool.get(property_frame_index, gpu_load, gpu_ensure)
                self.gpu_pool.use_frame(frame.index, property_frame_index)

                if gpu_res.state == GpuResourceState.LOAD or gpu_res.state == GpuResourceState.PREFETCH:
                    if gpu_res.state == GpuResourceState.PREFETCH:
                        assert frame.copy_cmd is not None
                        frame.copy_semaphores.append(gpu_res.use(PipelineStageFlags.TOP_OF_PIPE))
                        self._cmd_release_barrier(frame.copy_cmd, gpu_res)

                    self._cmd_acquire_barrier(frame.cmd, gpu_res)
                    frame.additional_semaphores.append(gpu_res.use(self.pipeline_stage_flags))
                    gpu_res.state = GpuResourceState.RENDER

                assert gpu_res.state == GpuResourceState.RENDER, gpu_res.state
                self.current = gpu_res.resource

    def prefetch(self) -> None:
        if not self.property.upload.preupload:
            assert self.cpu_pool is not None

            if self.property.upload.async_load:
                # Issue prefetches
                def cpu_prefetch_cleanup(k: int, buf: CpuBuffer) -> bool:
                    return buf.promise.is_set()

                def cpu_prefetch(k: int, buf: CpuBuffer) -> None:
                    self.thread_pool.submit(buf.promise, self._load_async, k, buf)  # type: ignore

                # TODO: can likely improve prefetch logic, and should probably allow
                # this to be hooked / configured somehow
                #
                # maybe we should just compute the state of next (maybe few) frames
                # (assuming constant dt, playback state) once globally and call prefetch with this.
                prefetch_start = self.property.current_frame_index + 1
                prefetch_end = prefetch_start + self.property.upload.cpu_prefetch_count
                prefetch_range = [self.property.get_frame_index(0, i) for i in range(prefetch_start, prefetch_end)]
                self.cpu_pool.prefetch(prefetch_range, cpu_prefetch_cleanup, cpu_prefetch)

            if self.upload_method == UploadMethod.TRANSFER_QUEUE:
                assert self.gpu_pool is not None

                def gpu_prefetch_cleanup(k: int, gpu_res: GpuResource[R]) -> bool:
                    state = self.prefetch_states_lookup[gpu_res]

                    assert gpu_res.semaphore is not None
                    assert self.cpu_pool is not None

                    if gpu_res.semaphore.get_value() >= state.prefetch_done_value:
                        # Release prefetch state
                        self.prefetch_states.append(state)
                        del self.prefetch_states_lookup[gpu_res]

                        # Release buffer
                        self.cpu_pool.release_manual(k)

                        assert gpu_res.state == GpuResourceState.RENDER or gpu_res.state == GpuResourceState.PREFETCH
                        return True
                    return False

                def gpu_prefetch(k: int, gpu_res: GpuResource[R]) -> None:
                    assert self.cpu_pool is not None

                    # We know that the cpu buffer is available, so just get it
                    cpu_next: CpuBuffer = self.cpu_pool.get(k, lambda _x, _y: None)
                    self.cpu_pool.use_manual(k)

                    # Get free prefetch state
                    state = self.prefetch_states.pop()
                    self.prefetch_states_lookup[gpu_res] = state

                    with state.commands:
                        self._cmd_upload(state.commands, cpu_next, gpu_res)

                    assert (
                        gpu_res.state == GpuResourceState.EMPTY
                        or GpuResourceState.PREFETCH
                        or gpu_res.state == GpuResourceState.RENDER
                    ), gpu_res.state
                    info = gpu_res.use(PipelineStageFlags.TRANSFER)
                    self.ctx.transfer_queue.submit(
                        state.commands,
                        wait_semaphores=[(info.sem, info.wait_stage)],
                        wait_timeline_values=[info.wait_value],
                        signal_semaphores=[info.sem],
                        signal_timeline_values=[info.signal_value],
                    )
                    state.prefetch_done_value = info.signal_value
                    gpu_res.state = GpuResourceState.PREFETCH

                # TODO: fix, same as above
                prefetch_start = self.property.current_frame_index + 1
                prefetch_end = prefetch_start + self.property.upload.gpu_prefetch_count
                prefetch_range = [self.property.get_frame_index(0, i) for i in range(prefetch_start, prefetch_end)]
                self.gpu_pool.prefetch(
                    [i for i in prefetch_range if self.cpu_pool.is_available(i)],
                    gpu_prefetch_cleanup,
                    gpu_prefetch,
                )

    def get_current(self) -> R:
        assert self.current
        return self.current

    def destroy(self) -> None:
        self.current = None
        if self.property.upload.preupload:
            for res in self.resources:
                res.destroy()
            self.resources.clear()
        else:
            assert self.cpu_pool
            for cpu_buf in self.cpu_buffers:
                cpu_buf.destroy()
            self.cpu_buffers.clear()
            for gpu_res in self.gpu_resources:
                gpu_res.destroy()
            self.gpu_resources.clear()
            self.cpu_pool.clear()
            if self.gpu_pool is not None:
                self.gpu_pool.clear()

    def _create_uploaded_resource(self, frame: np.ndarray, name: str) -> R:
        raise NotImplementedError

    def _create_cpu_buffer(self, name: str) -> Buffer:
        raise NotImplementedError

    def _create_gpu_resource(self, name: str) -> R:
        raise NotImplementedError

    def _cmd_upload(self, cmd: CommandBuffer, cpu_buf: CpuBuffer, gpu_res: GpuResource[R]) -> None:
        raise NotImplementedError

    def _cmd_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[R]) -> None:
        raise NotImplementedError

    def _cmd_acquire_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[R]) -> None:
        raise NotImplementedError

    def _cmd_release_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[R]) -> None:
        raise NotImplementedError


class GpuBufferProperty(GpuResourceProperty[Buffer]):
    def __init__(
        self,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool[None],
        property: Property,
        usage_flags: BufferUsageFlags,
        memory_usage: MemoryUsage,
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ):
        self.usage_flags = usage_flags
        self.memory_usage = memory_usage
        self.size = property.max_size()

        if upload_method == UploadMethod.CPU_BUF:
            self.cpu_usage_flags = self.usage_flags
        else:
            self.cpu_usage_flags = BufferUsageFlags.TRANSFER_SRC

        if upload_method == UploadMethod.BAR:
            self.gpu_alloc_type = AllocType.DEVICE_MAPPED
            self.gpu_usage_flags = usage_flags
        else:
            self.gpu_alloc_type = AllocType.DEVICE
            self.gpu_usage_flags = usage_flags | BufferUsageFlags.TRANSFER_DST

        super().__init__(
            ctx,
            num_frames_in_flight,
            upload_method,
            thread_pool,
            property,
            pipeline_stage_flags,
            name,
        )

    def _create_uploaded_resource(self, frame: np.ndarray, name: str) -> Buffer:
        return UploadableBuffer.from_data(self.ctx, view_bytes(frame), self.usage_flags, name=name)

    def _create_cpu_buffer(self, name: str) -> Buffer:
        return Buffer(
            self.ctx,
            self.size,
            self.cpu_usage_flags,
            AllocType.HOST,
            name=name,
        )

    def _create_gpu_resource(self, name: str) -> Buffer:
        return Buffer(
            self.ctx,
            self.size,
            self.gpu_usage_flags,
            self.gpu_alloc_type,
            name=name,
        )

    def _cmd_upload(
        self,
        cmd: CommandBuffer,
        cpu_buf: CpuBuffer,
        gpu_res: GpuResource[Buffer],
    ) -> None:
        cmd.copy_buffer_range(cpu_buf.buf, gpu_res.resource, cpu_buf.used_size)

    def _cmd_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[Buffer]) -> None:
        cmd.memory_barrier(MemoryUsage.TRANSFER_DST, self.memory_usage)

    def _cmd_acquire_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[Buffer]) -> None:
        cmd.buffer_barrier(
            gpu_res.resource,
            MemoryUsage.NONE,
            self.memory_usage,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )

    def _cmd_release_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[Buffer]) -> None:
        cmd.buffer_barrier(
            gpu_res.resource,
            MemoryUsage.TRANSFER_DST,
            MemoryUsage.NONE,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )


class GpuImageProperty(GpuResourceProperty[Image]):
    def __init__(
        self,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool[None],
        property: Property,
        format: Format,
        usage_flags: ImageUsageFlags,
        layout: ImageLayout,
        memory_usage: MemoryUsage,
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ):
        if upload_method != UploadMethod.GFX and upload_method != UploadMethod.TRANSFER_QUEUE:
            raise ValueError(
                f"GpuImageProperty supports only {UploadMethod.GFX} and {UploadMethod.TRANSFER_QUEUE} upload methods. Got {upload_method}."
            )
        self.usage_flags = usage_flags
        self.layout = layout
        self.memory_usage = memory_usage

        self.channels = property.channels()
        self.height = property.height()
        self.width = property.width()
        self.format = format

        self.pitch, self.rows = get_image_pitch_and_rows(self.width, self.height, self.format)
        super().__init__(
            ctx,
            num_frames_in_flight,
            upload_method,
            thread_pool,
            property,
            pipeline_stage_flags,
            name,
        )

    def _create_uploaded_resource(self, frame: np.ndarray, name: str) -> Image:
        return UploadableImage.from_data(
            self.ctx,
            view_bytes(frame),
            self.layout,
            self.width,
            self.height,
            self.format,
            self.usage_flags,
            name=name,
        )

    def _create_cpu_buffer(self, name: str) -> Buffer:
        return Buffer(
            self.ctx,
            self.pitch * self.rows,
            BufferUsageFlags.TRANSFER_SRC,
            AllocType.HOST,
            name=name,
        )

    def _create_gpu_resource(self, name: str) -> Image:
        return Image(
            self.ctx,
            self.width,
            self.height,
            self.format,
            self.usage_flags | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name=name,
        )

    def _cmd_upload(
        self,
        cmd: CommandBuffer,
        cpu_buf: CpuBuffer,
        gpu_res: GpuResource[Image],
    ) -> None:
        cmd.image_barrier(
            gpu_res.resource,
            ImageLayout.TRANSFER_DST_OPTIMAL,
            MemoryUsage.NONE,
            MemoryUsage.TRANSFER_DST,
        )
        cmd.copy_buffer_to_image(cpu_buf.buf, gpu_res.resource)

    def _cmd_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[Image]) -> None:
        cmd.image_barrier(
            gpu_res.resource,
            self.layout,
            MemoryUsage.TRANSFER_DST,
            self.memory_usage,
        )

    def _cmd_acquire_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[Image]) -> None:
        cmd.image_barrier(
            gpu_res.resource,
            self.layout,
            MemoryUsage.NONE,
            self.memory_usage,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )

    def _cmd_release_barrier(self, cmd: CommandBuffer, gpu_res: GpuResource[Image]) -> None:
        cmd.image_barrier(
            gpu_res.resource,
            self.layout,
            MemoryUsage.TRANSFER_DST,
            MemoryUsage.NONE,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )
