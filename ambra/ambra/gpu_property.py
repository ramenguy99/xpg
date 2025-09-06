from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

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

from .config import UploadMethod
from .property import Property, view_bytes
from .renderer_frame import RendererFrame, SemaphoreInfo
from .utils.gpu import BufferUploadInfo, ImageUploadInfo, get_image_pitch_rows_and_texel_size
from .utils.lru_pool import LRUPool
from .utils.threadpool import Promise, ThreadPool


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
        thread_pool: ThreadPool,
        out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
        property: Property,
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ):
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.upload_method = upload_method
        self.thread_pool = thread_pool
        self.property = property
        self.name = name
        self.pipeline_stage_flags = pipeline_stage_flags

        # Current state
        self.current: Optional[R] = None
        self.current_cpu_buf: Optional[CpuBuffer] = None

        # If preupload:
        self.resources: List[Tuple[R, bool]] = []
        self.dynamic = False

        # Otherwise:
        self.cpu_buffers: List[CpuBuffer] = []
        self.cpu_pool: Optional[LRUPool[int, CpuBuffer]] = None
        self.gpu_resources: List[GpuResource[R]] = []
        self.gpu_pool: Optional[LRUPool[int, GpuResource[R]]] = None
        self.prefetch_states: List[PrefetchState] = []
        self.prefetch_states_lookup: Dict[GpuResource[R], PrefetchState] = {}

        self.has_owned_upload_buffers = not self.property.upload.preupload

        # Upload
        if self.property.upload.preupload:
            if upload_method == UploadMethod.CPU_BUF:
                alloc_type = AllocType.HOST
            elif upload_method == UploadMethod.BAR:
                alloc_type = AllocType.DEVICE_MAPPED
            else:
                alloc_type = AllocType.DEVICE

            # NOTE: In the async_load case we could also do mapped upload in
            # threads here, but we first need to know the size of the frame.
            # This makes things more complicated because we could run into
            # issues using the context from multiple threads. It's not
            # clear where is the best case to ensure this is thread safe so
            # for now we don't do it.
            if property.upload.async_load:
                promises: List[Promise[np.ndarray]] = []
                for i in range(property.num_frames):
                    promise: Promise[np.ndarray] = Promise()
                    self.thread_pool.submit(promise, self._load_async, i)  # type: ignore
                    promises.append(promise)

            for i in range(property.num_frames):
                if property.upload.async_load:
                    frame = promises[i].get()
                else:
                    frame = property.get_frame_by_index(i)
                res = self._create_resource_for_preupload(frame, alloc_type, f"{name}-{i}")
                if upload_method == UploadMethod.CPU_BUF or upload_method == UploadMethod.BAR:
                    self._upload_mapped_resource(res, frame)
                else:
                    out_upload_list.append(self._create_bulk_upload_descriptor(res, frame))
                self.resources.append((res, True))
        else:
            cpu_prefetch_count = self.property.upload.cpu_prefetch_count
            gpu_prefetch_count = (
                self.property.upload.gpu_prefetch_count if upload_method == UploadMethod.TRANSFER_QUEUE else 0
            )
            cpu_buffers_count = num_frames_in_flight + cpu_prefetch_count + gpu_prefetch_count
            gpu_resources_count = 1 + gpu_prefetch_count

            cpu_alloc_type = AllocType.DEVICE_MAPPED if self.upload_method == UploadMethod.BAR else AllocType.HOST
            self.cpu_buffers = [
                CpuBuffer(self._create_cpu_buffer(f"cpubuf-{name}-{i}", cpu_alloc_type))
                for i in range(cpu_buffers_count)
            ]
            self.cpu_pool = LRUPool(self.cpu_buffers, num_frames_in_flight, cpu_prefetch_count)

            for i in range(gpu_resources_count):
                res = self._create_gpu_resource(f"gpubuf-{name}-{i}")
                if upload_method == UploadMethod.TRANSFER_QUEUE:
                    semaphore = TimelineSemaphore(ctx, name=f"gpubuf-{name}-{i}-semaphore")
                else:
                    semaphore = None
                self.gpu_resources.append(GpuResource(res, semaphore))

            self.gpu_pool = LRUPool(
                self.gpu_resources,
                1,
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
        # TODO: after invalidation or if configured, allocate owned prealloc buffers
        # and switch to streaming operations

    def _load_async(self, i: int, thread_index: int) -> np.ndarray:
        return self.property.get_frame_by_index(i, thread_index)

    def _load_async_into(self, i: int, buf: CpuBuffer, thread_index: int) -> None:
        buf.used_size = self.property.get_frame_by_index_into(i, buf.buf.data, thread_index)

    def load(self, frame: RendererFrame) -> None:
        # Issue CPU loads if async
        if not self.property.upload.preupload or self.dynamic:
            assert self.cpu_pool is not None

            self.cpu_pool.release_frame(frame.index)

            def cpu_load(k: int, buf: CpuBuffer) -> None:
                if self.property.upload.async_load:
                    self.thread_pool.submit(buf.promise, self._load_async_into, k, buf)  # type: ignore
                else:
                    buf.used_size = self.property.get_frame_by_index_into(k, buf.buf.data)

            property_frame_index = self.property.current_frame_index

            if self.dynamic:
                do_load = not self.resources[property_frame_index][1]
            else:
                do_load = (
                    self.upload_method == UploadMethod.CPU_BUF
                    or self.upload_method == UploadMethod.BAR
                    or (
                        self.gpu_pool is not None
                        and not self.gpu_pool.is_available_or_prefetching(property_frame_index)
                    )
                )

            if do_load:
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
            res, up_to_date = self.resources[property_frame_index]
            if not up_to_date:
                assert self.current_cpu_buf is not None
                cpu_buf = self.current_cpu_buf
                self.current_cpu_buf = None

                if self.property.upload.async_load:
                    cpu_buf.promise.get()

                self.cpu_pool.use_frame(frame.index, property_frame_index)

                # Upload on gfx queue
                self._cmd_before_barrier(frame.cmd, res)
                self._cmd_upload(frame.cmd, cpu_buf, res)
                self._cmd_after_barrier(frame.cmd, res)

                self.resources[property_frame_index] = (res, True)
            self.current = res
        else:
            assert self.cpu_pool is not None

            # Wait for buffer to be ready
            if self.upload_method == UploadMethod.CPU_BUF or self.upload_method == UploadMethod.BAR:
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

                    self.cpu_pool.use_frame(frame.index, k)
                    if self.upload_method == UploadMethod.GFX:
                        # Upload on gfx queue
                        self._cmd_before_barrier(frame.cmd, gpu_res.resource)
                        self._cmd_upload(frame.cmd, cpu_buf, gpu_res.resource)
                        self._cmd_after_barrier(frame.cmd, gpu_res.resource)

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
                        self._cmd_before_barrier(frame.copy_cmd, gpu_res.resource)
                        self._cmd_upload(frame.copy_cmd, cpu_buf, gpu_res.resource)
                        self._cmd_release_barrier(frame.copy_cmd, gpu_res.resource)

                        gpu_res.state = GpuResourceState.LOAD

                def gpu_ensure(k: int, gpu_res: GpuResource[R]) -> None:
                    assert gpu_res.state == GpuResourceState.PREFETCH, gpu_res.state

                gpu_res = self.gpu_pool.get(property_frame_index, gpu_load, gpu_ensure)
                self.gpu_pool.give_back(property_frame_index, gpu_res)

                if gpu_res.state == GpuResourceState.LOAD or gpu_res.state == GpuResourceState.PREFETCH:
                    if gpu_res.state == GpuResourceState.PREFETCH:
                        assert frame.copy_cmd is not None
                        frame.copy_semaphores.append(gpu_res.use(PipelineStageFlags.TOP_OF_PIPE))
                        self._cmd_release_barrier(frame.copy_cmd, gpu_res.resource)

                    self._cmd_acquire_barrier(frame.cmd, gpu_res.resource)
                    gpu_res.state = GpuResourceState.RENDER

                # If we are using the transfer queue to upload we have to guard the gpu resource
                # with a semaphore because we might need to reuse this buffer for pre-fetching
                # while this frame is still in flight. Using the use_frame/release_frame mechanism
                # with a single frame is not enough because we might try to start pre-fetching on
                # the next frame while this frame is still in flight.
                if self.upload_method == UploadMethod.TRANSFER_QUEUE:
                    frame.additional_semaphores.append(gpu_res.use(self.pipeline_stage_flags))

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
                    self.thread_pool.submit(buf.promise, self._load_async_into, k, buf)  # type: ignore

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
                    assert (
                        gpu_res.state == GpuResourceState.EMPTY
                        or GpuResourceState.PREFETCH
                        or gpu_res.state == GpuResourceState.RENDER
                    ), gpu_res.state

                    # We know that the cpu buffer is available, so just get it
                    cpu_next: CpuBuffer = self.cpu_pool.get(k, lambda _x, _y: None)
                    self.cpu_pool.use_manual(k)

                    # Get free prefetch state
                    state = self.prefetch_states.pop()
                    self.prefetch_states_lookup[gpu_res] = state

                    with state.commands:
                        self._cmd_before_barrier(state.commands, gpu_res.resource)
                        self._cmd_upload(state.commands, cpu_next, gpu_res.resource)

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

    def invalidate_frame(self, frame_index: int) -> None:
        upload = self.property.upload
        if upload.preupload:
            # In terms of state we can differentiate between static and dynamic properties
            # and allow promotion to static here. We could also technically allow a property
            # to specify dynamic to avoid promotion at runtime.
            #
            # We already have all frames uploaded here. But we can't just overwrite them
            # because they might be in use by previous frames.
            #
            # We need to transition to a LRU pool model:
            #  1) We could add all frames to an LRU pool and allocate a few new ones for streaming uploads.
            #  2) Alternatively we could allocate a separate LRU pool just for streaming stuff.
            #
            # There is also 2 cases, mapped and gpu.
            #  1) In mapped case we operate always on CPU timeline so we need to always manage resources in use.
            #  2) In non-mapped case we need to allocate "frames" staging buffers, but can then issue copies
            #     on the GPU with proper barriers.
            #
            # We could reuse some of the infrastructure for streaming, especially cpu buffers in the
            # unmapped resource case.
            # It's harder to reuse the gpu_pool (or the cpu_pool in the mapped case) because the buffers
            # in the preupload case are actually sized by the frame, which is not guaranteed to be the
            # same for all buffers.
            #
            # It's also not obvious if and how to handle the case where after invalidation the frame grew
            # bigger, or if we want to support this at all. We could say that for that case we tank the
            # performance hit of reallocating the buffer, and just queue the current buffer for being freed.
            #
            # Currently the main issue is that the LRU pools are not really designed with invalidation in mind
            # because they cannot handle multiple versions of a frame with the same key. It's not yet clear
            # to me if they should handle that with something like generation numbers, or if we should have
            # a different system to handle this.
            #
            # TODO: this is now broken and we need to rethink all of this invalidation / per-frame streaming topic.
            if not self.dynamic:
                print("PROMOTING")
                self.cpu_buffers = [
                    CpuBuffer(self._create_cpu_buffer(f"cpubuf-{self.name}-{i}", AllocType.HOST))
                    for i in range(self.num_frames_in_flight)
                ]
                self.cpu_pool = LRUPool(self.cpu_buffers, self.num_frames_in_flight, 0)
                self.dynamic = True

            # Invalidate frame
            self.cpu_pool.invalidate(frame_index)
            res, _ = self.resources[frame_index]
            self.resources[frame_index] = (res, False)
        else:
            assert self.cpu_pool is not None
            self.cpu_pool.invalidate(frame_index)
            if self.gpu_pool is not None:
                self.gpu_pool.invalidate(frame_index)

    def destroy(self) -> None:
        self.current = None
        if self.property.upload.preupload:
            for res, _ in self.resources:
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

    def _create_resource_for_preupload(self, frame: np.ndarray, alloc_type: AllocType, name: str) -> R:
        raise NotImplementedError

    def _upload_mapped_resource(self, resource: R, frame: np.ndarray) -> None:
        raise NotImplementedError

    def _create_bulk_upload_descriptor(
        self, resource: R, frame: np.ndarray
    ) -> Union[BufferUploadInfo, ImageUploadInfo]:
        raise NotImplementedError

    def _create_cpu_buffer(self, name: str, alloc_type: AllocType) -> Buffer:
        raise NotImplementedError

    def _create_gpu_resource(self, name: str) -> R:
        raise NotImplementedError

    def _cmd_upload(self, cmd: CommandBuffer, cpu_buf: CpuBuffer, resource: R) -> None:
        raise NotImplementedError

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError

    def _cmd_acquire_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError

    def _cmd_release_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError


class GpuBufferProperty(GpuResourceProperty[Buffer]):
    def __init__(
        self,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool,
        out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
        property: Property,
        usage_flags: BufferUsageFlags,
        memory_usage: MemoryUsage,
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ):
        self.usage_flags = usage_flags | BufferUsageFlags.TRANSFER_DST
        self.memory_usage = memory_usage
        self.size = property.max_size()

        if upload_method == UploadMethod.CPU_BUF or upload_method == UploadMethod.BAR:
            self.cpu_usage_flags = self.usage_flags
        else:
            self.cpu_usage_flags = BufferUsageFlags.TRANSFER_SRC

        super().__init__(
            ctx,
            num_frames_in_flight,
            upload_method,
            thread_pool,
            out_upload_list,
            property,
            pipeline_stage_flags,
            name,
        )

    def _create_resource_for_preupload(self, frame: np.ndarray, alloc_type: AllocType, name: str) -> Buffer:
        return Buffer(self.ctx, len(view_bytes(frame)), self.usage_flags, alloc_type, name=name)

    def _upload_mapped_resource(self, resource: Buffer, frame: np.ndarray) -> None:
        # NOTE: here we can assume that the buffer is always the same size as frame data.
        resource.data[:] = view_bytes(frame)

    def _create_bulk_upload_descriptor(self, resource: Buffer, frame: np.ndarray) -> BufferUploadInfo:
        return BufferUploadInfo(view_bytes(frame), resource)

    def _create_cpu_buffer(self, name: str, alloc_type: AllocType) -> Buffer:
        return Buffer(
            self.ctx,
            self.size,
            self.cpu_usage_flags,
            alloc_type,
            name=name,
        )

    def _create_gpu_resource(self, name: str) -> Buffer:
        return Buffer(
            self.ctx,
            self.size,
            self.usage_flags,
            AllocType.DEVICE,
            name=name,
        )

    def _cmd_upload(
        self,
        cmd: CommandBuffer,
        cpu_buf: CpuBuffer,
        resource: Buffer,
    ) -> None:
        cmd.copy_buffer_range(cpu_buf.buf, resource, cpu_buf.used_size)

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: Buffer) -> None:
        cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.TRANSFER_DST)

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: Buffer) -> None:
        cmd.memory_barrier(MemoryUsage.TRANSFER_DST, self.memory_usage)

    def _cmd_acquire_barrier(self, cmd: CommandBuffer, resource: Buffer) -> None:
        cmd.buffer_barrier(
            resource,
            MemoryUsage.NONE,
            self.memory_usage,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )

    def _cmd_release_barrier(self, cmd: CommandBuffer, resource: Buffer) -> None:
        cmd.buffer_barrier(
            resource,
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
        thread_pool: ThreadPool,
        out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
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
        self.usage_flags = usage_flags | ImageUsageFlags.TRANSFER_DST
        self.layout = layout
        self.memory_usage = memory_usage

        self.channels = property.channels()
        self.height = property.height()
        self.width = property.width()
        self.format = format

        self.pitch, self.rows, _ = get_image_pitch_rows_and_texel_size(self.width, self.height, self.format)
        super().__init__(
            ctx,
            num_frames_in_flight,
            upload_method,
            thread_pool,
            out_upload_list,
            property,
            pipeline_stage_flags,
            name,
        )

    def _create_resource_for_preupload(self, frame: np.ndarray, alloc_type: AllocType, name: str) -> Image:
        return Image(self.ctx, self.width, self.height, self.format, self.usage_flags, alloc_type, name=name)

    def _create_bulk_upload_descriptor(self, resource: Image, frame: np.ndarray) -> ImageUploadInfo:
        return ImageUploadInfo(view_bytes(frame), resource, self.layout)

    def _create_cpu_buffer(self, name: str, alloc_type: AllocType) -> Buffer:
        return Buffer(
            self.ctx,
            self.pitch * self.rows,
            BufferUsageFlags.TRANSFER_SRC,
            alloc_type,
            name=name,
        )

    def _create_gpu_resource(self, name: str) -> Image:
        return Image(
            self.ctx,
            self.width,
            self.height,
            self.format,
            self.usage_flags,
            AllocType.DEVICE,
            name=name,
        )

    def _cmd_upload(
        self,
        cmd: CommandBuffer,
        cpu_buf: CpuBuffer,
        resource: Image,
    ) -> None:
        cmd.copy_buffer_to_image(cpu_buf.buf, resource)

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: Image) -> None:
        cmd.image_barrier(
            resource,
            ImageLayout.TRANSFER_DST_OPTIMAL,
            MemoryUsage.ALL,
            MemoryUsage.TRANSFER_DST,
            undefined=True,
        )

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: Image) -> None:
        cmd.image_barrier(
            resource,
            self.layout,
            MemoryUsage.TRANSFER_DST,
            self.memory_usage,
        )

    def _cmd_acquire_barrier(self, cmd: CommandBuffer, resource: Image) -> None:
        cmd.image_barrier(
            resource,
            self.layout,
            MemoryUsage.NONE,
            self.memory_usage,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )

    def _cmd_release_barrier(self, cmd: CommandBuffer, resource: Image) -> None:
        cmd.image_barrier(
            resource,
            self.layout,
            MemoryUsage.TRANSFER_DST,
            MemoryUsage.NONE,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )
