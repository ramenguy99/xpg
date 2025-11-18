# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from numpy.typing import NDArray
from pyxpg import (
    AllocType,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    Context,
    DescriptorSet,
    DescriptorType,
    Image,
    ImageView,
    MemoryUsage,
    PipelineStageFlags,
    TimelineSemaphore,
)

from .config import UploadMethod
from .renderer_frame import RendererFrame, SemaphoreInfo
from .utils.gpu import (
    BufferUploadInfo,
    ImageUploadInfo,
    view_bytes,
)
from .utils.lru_pool import LRUPool
from .utils.ring_buffer import RingBuffer
from .utils.threadpool import Promise, ThreadPool

if TYPE_CHECKING:
    from .property import BufferProperty, Property, PropertyItem


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


@dataclass
class GpuBufferView:
    buffer: Buffer
    offset: int
    size: int

    def destroy(self) -> None:
        self.buffer.destroy()

    def buffer_and_offset(self) -> Tuple[Buffer, int]:
        return (self.buffer, self.offset)

    def write_descriptor(
        self, descriptor_set: DescriptorSet, type: DescriptorType, binding: int, element: int = 0
    ) -> None:
        descriptor_set.write_buffer(self.buffer, type, binding, element, self.offset, self.size)


@dataclass
class GpuImageView:
    image: Image
    # resource_view: ImageView
    srgb_resource_view: Optional[ImageView]

    mip_level_0_view: Optional[ImageView]
    mip_views: List[ImageView]

    def destroy(self) -> None:
        self.image.destroy()

    def view(self) -> Union[Image, ImageView]:
        if self.srgb_resource_view is not None:
            return self.srgb_resource_view
        else:
            # TODO: return proper view to correct layer
            return self.image


R = TypeVar("R", bound=Union[Buffer, Image])
V = TypeVar("V", bound=Union[GpuBufferView, GpuImageView])


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


class GpuProperty(Generic[V]):
    # Public API
    def get_current() -> V:
        raise NotImplementedError

    def load(self, frame: RendererFrame) -> None:
        pass

    def upload(self, frame: RendererFrame) -> None:
        pass

    def prefetch(self) -> None:
        pass

    def invalidate_frame(self, invalidated_property_frame_index: int) -> None:
        pass


class GpuPreuploadedArrayProperty(Generic[R, V], GpuProperty[V]):
    def __init__(
        self,
        data: NDArray[Any],
        frame_size: int,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
        property: "Property",
        name: str,
    ):
        self.frame_size = frame_size
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.upload_method = upload_method
        self.property = property
        self.name = name

        self.resource = self._create_preuploaded_array_resource(data, out_upload_list)
        self.staging_buffers: Optional[RingBuffer[Buffer]] = None
        self.invalid_frames = set()
        self.current: Optional[V] = None

    def upload(self, frame: RendererFrame) -> None:
        index = self.property.current_frame_index

        view = self._get_view(index)

        if index in self.invalid_frames:
            self.invalid_frames.remove(index)

            assert self.staging_buffers is not None

            # Get a free staging buffer
            staging = self.staging_buffers.get_current_and_advance()

            # Copy data into staging buffer
            self.property.get_frame_by_index_into(index, staging.data)

            # Record command to upload staging buffer into view
            self._cmd_before_barrier(frame.cmd, view)
            self._cmd_upload(frame.cmd, staging, view)
            self._cmd_after_barrier(frame.cmd, view)

        self.current = view

    def get_current(self) -> V:
        assert self.current is not None
        return self.current

    def invalidate_frame(self, invalidated_property_frame_index: int) -> None:
        self.invalid_frames.add(invalidated_property_frame_index)

        # Allocate staging buffers for this property
        if self.staging_buffers is None:
            self.staging_buffers = RingBuffer(
                [
                    Buffer(
                        self.ctx,
                        self.frame_size,
                        BufferUsageFlags.TRANSFER_SRC,
                        AllocType.HOST,
                        f"{self.name}-staging-{i}",
                    )
                    for i in range(self.num_frames_in_flight)
                ]
            )

    # Private API
    def _get_view(self, index: int) -> V:
        raise NotImplementedError

    def _create_preuploaded_array_resource(
        self, data: NDArray[Any], out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]]
    ) -> R:
        raise NotImplementedError

    def _cmd_upload(self, stagin_buffer: Buffer, view: V) -> None:
        raise NotImplementedError

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError


class GpuBufferPreuploadedArrayProperty(GpuPreuploadedArrayProperty[Buffer, GpuBufferView]):
    def __init__(
        self,
        data: NDArray[Any],
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
        property: "BufferProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | BufferUsageFlags.TRANSFER_DST
        self.pipeline_stage_flags = property.gpu_stage
        self.memory_usage = MemoryUsage.ALL  # TODO: new sync API

        super().__init__(
            data,
            property.max_size,
            ctx,
            num_frames_in_flight,
            upload_method,
            out_upload_list,
            property,
            name,
        )

    # Private API
    def _get_view(self, index: int) -> GpuBufferView:
        # TODO: handle alignment here (if data is padded use the padded size)
        return GpuBufferView(self.resource, self.frame_size * index, self.frame_size)

    def _create_preuploaded_array_resource(
        self, ctx: Context, data: NDArray[Any], out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]]
    ) -> R:
        if self.upload_method == UploadMethod.MAPPED_PREFER_HOST:
            alloc_type = AllocType.HOST
        elif self.upload_method == UploadMethod.MAPPED_PREFER_DEVICE:
            alloc_type = AllocType.DEVICE_MAPPED
        else:
            alloc_type = AllocType.DEVICE

        # TODO: handle alignment here (need to pad the data)
        buffer = Buffer(ctx, data.nbytes, self.usage_flags, alloc_type, self.name)

        if buffer.is_mapped:
            buffer.data[:] = data.data
        else:
            out_upload_list.append(BufferUploadInfo(view_bytes(data), buffer))
        return buffer

    def _cmd_upload(self, cmd: CommandBuffer, staging_buffer: Buffer, view: GpuBufferView) -> None:
        cmd.copy_buffer_range(staging_buffer, view.buffer, self.frame_size, 0, view.offset)

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        # TODO: Currently needs to be all because it could be run on the transfer queue. I am
        # not even sure we need any barrier for that case since it's a separate submit.
        cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.TRANSFER_DST)

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        cmd.memory_barrier(MemoryUsage.TRANSFER_DST, self.memory_usage)


class GpuPreuploadedProperty(Generic[R, V], GpuProperty[V]):
    def __init__(
        self,
        max_frame_size: int,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool,
        out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
        property: "Property",
        name: str,
    ):
        self.max_frame_size = max_frame_size
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.upload_method = upload_method
        self.thread_pool = thread_pool
        self.property = property
        self.name = name

        self.async_load = self.property.upload.async_load
        self.resource_views: List[V] = []
        self.staging_buffers: Optional[RingBuffer[CpuBuffer]] = None
        self.current_staging_buf: Optional[CpuBuffer] = None
        self.invalid_frames = set()
        self.current: Optional[V] = None

        # NOTE: In the async_load case we could also do mapped upload in
        # threads here, but we first need to know the size of the frame.
        # This makes things more complicated because we could run into
        # issues using the context from multiple threads. It's not
        # clear where is the best way to ensure this is thread safe so
        # for now we don't do it.
        if self.async_load:
            promises: List[Promise[PropertyItem]] = []
            for i in range(property.num_frames):
                promise: Promise[PropertyItem] = Promise()
                self.thread_pool.submit(promise, self._load_async, i)  # type: ignore
                promises.append(promise)

        # Collect frames
        frames = []
        for i in range(property.num_frames):
            if self.async_load:
                frame = promises[i].get()
            else:
                frame = property.get_frame_by_index(i)
            frames.append(frame)

        # Create resource and views
        self.resource, self.resource_views = self._create_preuploaded_resource_and_views(
            frames, out_upload_list, f"{name}-{i}"
        )

    def load(self, frame: RendererFrame) -> None:
        index = self.property.current_frame_index
        if index in self.invalid_frames:
            self.invalid_frames.remove(index)

            assert self.staging_buffers is not None

            buf = self.staging_buffers.get_current_and_advance()
            if self.async_load:
                self.thread_pool.submit(buf.promise, self._load_async_into, index, buf)  # type: ignore
            else:
                buf.used_size = self.property.get_frame_by_index_into(index, buf.buf.data)
            self.current_staging_buf = buf

    def upload(self, frame: RendererFrame) -> None:
        view = self.resource_views[self.property.current_frame_index]
        if self.current_staging_buf is not None:
            self._cmd_before_barrier(frame.cmd, view)
            self._cmd_upload(frame.cmd, self.current_staging_buf, view)
            self._cmd_after_barrier(frame.cmd, view)
        self.current = view

    def get_current(self) -> V:
        assert self.current is not None
        return self.current

    def invalidate_frame(self, invalidated_property_frame_index: int) -> None:
        self.invalid_frames.add(invalidated_property_frame_index)

        # Allocate staging buffers for this property
        if self.staging_buffers is None:
            self.staging_buffers = RingBuffer(
                [
                    CpuBuffer(
                        Buffer(
                            self.ctx,
                            self.max_frame_size,
                            BufferUsageFlags.TRANSFER_SRC,
                            AllocType.HOST,
                            f"{self.name}-staging-{i}",
                        )
                    )
                    for i in range(self.num_frames_in_flight)
                ]
            )

    # Private API
    def _load_async(self, i: int, thread_index: int) -> "PropertyItem":
        return self.property.get_frame_by_index(i, thread_index)

    def _load_async_into(self, i: int, buf: CpuBuffer, thread_index: int) -> None:
        buf.used_size = self.property.get_frame_by_index_into(i, buf.buf.data, thread_index)

    def _create_preuploaded_resource_and_views(
        self, frames: List[NDArray[Any]], out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]], name: str
    ) -> Tuple[R, List[V]]:
        raise NotImplementedError

    def _cmd_upload(self, stagin_buffer: Buffer, view: V) -> None:
        raise NotImplementedError

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: R) -> None:
        raise NotImplementedError


class GpuBufferPreuploadedProperty(GpuPreuploadedProperty[Buffer, GpuBufferView]):
    def __init__(
        self,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool,
        out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
        property: "BufferProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | BufferUsageFlags.TRANSFER_DST
        self.pipeline_stage_flags = property.gpu_stage
        self.memory_usage = MemoryUsage.ALL  # TODO: new sync API

        super().__init__(
            property.max_size,
            ctx,
            num_frames_in_flight,
            upload_method,
            thread_pool,
            out_upload_list,
            property.gpu_stage,
            property,
            name,
        )

    # Private API
    def _create_preuploaded_resource_and_views(
        self, frames: List[NDArray[Any]], out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]], name: str
    ) -> Tuple[R, List[V]]:
        # TODO: optionally decide not to batch here to allow append/insert/remove and uploads of bigger size

        if self.upload_method == UploadMethod.MAPPED_PREFER_HOST:
            alloc_type = AllocType.HOST
        elif self.upload_method == UploadMethod.MAPPED_PREFER_DEVICE:
            alloc_type = AllocType.DEVICE_MAPPED
        else:
            alloc_type = AllocType.DEVICE

        total_size_in_bytes = 0
        for frame in frames:
            total_size_in_bytes += len(view_bytes(frame))

        buffer = Buffer(self.ctx, total_size_in_bytes, self.usage_flags, alloc_type, name)

        # Create views and upload
        views: List[GpuBufferView] = []
        offset = 0
        for frame in frames:
            frame_bytes = view_bytes(frame)
            size = len(frame_bytes)

            if buffer.is_mapped:
                buffer.data[offset : offset + size] = frame_bytes
            else:
                out_upload_list.append(BufferUploadInfo(frame_bytes, buffer, offset, size))
            views.append(GpuBufferView(buffer, offset, size))

            # TODO: handle alignment here depending on usage flags (vbo, ubo, ssbo alignement constraints)
            offset += size

        return buffer, views

    def _cmd_upload(self, cmd: CommandBuffer, staging_buffer: Buffer, view: GpuBufferView) -> None:
        cmd.copy_buffer_range(staging_buffer, view.buffer, view.size, 0, view.offset)

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        # TODO: Currently needs to be all because it could be run on the transfer queue. I am
        # not even sure we need any barrier for that case since it's a separate submit.
        cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.TRANSFER_DST)

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        cmd.memory_barrier(MemoryUsage.TRANSFER_DST, self.memory_usage)


class GpuStreamingProperty(Generic[R, V], GpuProperty[V]):
    def __init__(
        self,
        max_frame_size: int,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool,
        property: "Property",
        name: str,
    ):
        self.max_frame_size = max_frame_size
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.upload_method = upload_method
        self.thread_pool = thread_pool
        self.property = property
        self.name = name

        # Variants
        self.async_load = self.property.upload.async_load
        self.mapped = (
            upload_method == UploadMethod.MAPPED_PREFER_DEVICE or upload_method == UploadMethod.MAPPED_PREFER_HOST
        )

        # Common
        self.current: Optional[V] = None

        # If not mapped
        self.gpu_resources: List[GpuResource[R]] = []
        self.gpu_pool: Optional[LRUPool[int, GpuResource[R]]] = None

        # If prefetch and not mapped
        self.prefetch_states: List[PrefetchState] = []
        self.prefetch_states_lookup: Dict[GpuResource[R], PrefetchState] = {}

        # Compute buffer counts
        gpu_prefetch_count = (
            self.property.upload.gpu_prefetch_count if upload_method == UploadMethod.TRANSFER_QUEUE else 0
        )
        gpu_resources_count = 1 + gpu_prefetch_count
        cpu_prefetch_count = self.property.upload.cpu_prefetch_count
        cpu_buffers_count = num_frames_in_flight + cpu_prefetch_count + gpu_prefetch_count

        # Allocate CPU buffers
        self.current_cpu_buf: Optional[CpuBuffer] = None
        self.cpu_buffers = [
            CpuBuffer(self._create_cpu_buffer(f"{self.name}-cpu-{i}")) for i in range(cpu_buffers_count)
        ]

        self.cpu_pool = LRUPool(
            self.cpu_buffers,
            num_frames_in_flight,
            cpu_prefetch_count,
        )

        if not self.mapped:
            for i in range(gpu_resources_count):
                res = self._create_gpu_resource(f"{name}-gpu-{i}")
                if upload_method == UploadMethod.TRANSFER_QUEUE:
                    semaphore = TimelineSemaphore(ctx, name=f"{name}-sem-{i}")
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
                            name=f"{name}-prefetch-{i}",
                        ),
                        prefetch_done_value=0,
                    )
                    for i in range(gpu_prefetch_count)
                ]

    def load(self, frame: RendererFrame) -> None:
        self.cpu_pool.release_frame(frame.index)

        def cpu_load(k: int, buf: CpuBuffer) -> None:
            if self.async_load:
                self.thread_pool.submit(buf.promise, self._load_async_into, k, buf)  # type: ignore
            else:
                buf.used_size = self.property.get_frame_by_index_into(k, buf.buf.data)

        property_frame_index = self.property.current_frame_index
        if self.mapped or not self.gpu_pool.is_available_or_prefetching(property_frame_index):
            self.current_cpu_buf = self.cpu_pool.get(property_frame_index, cpu_load)

    def upload(self, frame: RendererFrame) -> None:
        property_frame_index = self.property.current_frame_index

        # Wait for buffer to be ready
        if self.mapped:
            assert self.current_cpu_buf is not None
            cpu_buf = self.current_cpu_buf
            self.current_cpu_buf = None

            if self.async_load:
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
                if self.async_load:
                    cpu_buf.promise.get()

                self.cpu_pool.use_frame(frame.index, k)
                if self.upload_method == UploadMethod.GRAPHICS_QUEUE:
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
                    assert frame.transfer_cmd is not None

                    frame.transfer_semaphores.append(gpu_res.use(PipelineStageFlags.TRANSFER))

                    # Upload on copy queue
                    self._cmd_before_barrier(frame.transfer_cmd, gpu_res.resource)
                    self._cmd_upload(frame.transfer_cmd, cpu_buf, gpu_res.resource)
                    self._cmd_release_barrier(frame.transfer_cmd, gpu_res.resource)

                    gpu_res.state = GpuResourceState.LOAD

            def gpu_ensure(k: int, gpu_res: GpuResource[R]) -> None:
                assert gpu_res.state == GpuResourceState.PREFETCH, gpu_res.state

            gpu_res = self.gpu_pool.get(property_frame_index, gpu_load, gpu_ensure)
            self.gpu_pool.give_back(property_frame_index, gpu_res)

            if gpu_res.state == GpuResourceState.LOAD or gpu_res.state == GpuResourceState.PREFETCH:
                if gpu_res.state == GpuResourceState.PREFETCH:
                    assert frame.transfer_cmd is not None
                    frame.transfer_semaphores.append(gpu_res.use(PipelineStageFlags.TOP_OF_PIPE))
                    self._cmd_release_barrier(frame.transfer_cmd, gpu_res.resource)

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
        if self.property.upload.cpu_prefetch_count > 0:
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

        if self.upload_method == UploadMethod.TRANSFER_QUEUE and self.property.upload.gpu_prefetch_count > 0:
            assert self.gpu_pool is not None

            def gpu_prefetch_cleanup(k: int, gpu_res: GpuResource[R]) -> bool:
                state = self.prefetch_states_lookup[gpu_res]

                assert gpu_res.semaphore is not None

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
            available_range = []
            for i in range(prefetch_start, prefetch_end):
                frame_index = self.property.get_frame_index(0, i)
                if self.cpu_pool.is_available(frame_index):
                    available_range.append(frame_index)

            self.gpu_pool.prefetch(
                available_range,
                gpu_prefetch_cleanup,
                gpu_prefetch,
            )

    def get_current(self) -> V:
        assert self.current is not None
        return self.current

    def invalidate_frame(self, invalidated_property_frame_index: int) -> None:
        self.cpu_pool.increment_generation(invalidated_property_frame_index)
        if self.gpu_pool is not None:
            self.gpu_pool.increment_generation(invalidated_property_frame_index)

    # Private API
    def _load_async(self, i: int, thread_index: int) -> "PropertyItem":
        return self.property.get_frame_by_index(i, thread_index)

    def _load_async_into(self, i: int, buf: CpuBuffer, thread_index: int) -> None:
        buf.used_size = self.property.get_frame_by_index_into(i, buf.buf.data, thread_index)

    def _create_cpu_buffer(self, name: str) -> Buffer:
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


class GpuBufferStreamingProperty(GpuStreamingProperty[Buffer, GpuBufferView]):
    def __init__(
        self,
        ctx: Context,
        num_frames_in_flight: int,
        upload_method: UploadMethod,
        thread_pool: ThreadPool,
        property: "BufferProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | BufferUsageFlags.TRANSFER_DST
        self.pipeline_stage_flags = property.gpu_stage
        self.memory_usage = MemoryUsage.ALL  # TODO: new sync API

        super().__init__(
            property.max_size,
            ctx,
            num_frames_in_flight,
            upload_method,
            thread_pool,
            property,
            name,
        )

    def _create_cpu_buffer(self, name: str) -> Buffer:
        cpu_alloc_type = (
            AllocType.DEVICE_MAPPED if self.upload_method == UploadMethod.MAPPED_PREFER_DEVICE else AllocType.HOST
        )
        if self.mapped:
            cpu_buffer_usage_flags = self.usage_flags
        else:
            cpu_buffer_usage_flags = BufferUsageFlags.TRANSFER_SRC
        return Buffer(self.ctx, self.max_frame_size, cpu_buffer_usage_flags, cpu_alloc_type, name)

    def _create_gpu_resource(self, name: str) -> GpuBufferView:
        return GpuBufferView(
            Buffer(
                self.ctx,
                self.max_frame_size,
                self.usage_flags,
                AllocType.DEVICE,
                name=name,
            ),
            0,
            self.max_frame_size,
        )

    def _cmd_upload(
        self,
        cmd: CommandBuffer,
        cpu_buf: CpuBuffer,
        resource: GpuBufferView,
    ) -> None:
        cmd.copy_buffer_range(cpu_buf.buf, resource.buffer, cpu_buf.used_size, resource.offset)

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        # TODO: Currently needs to be all because it could be run on the transfer queue. I am
        # not even sure we need any barrier for that case since it's a separate submit.
        cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.TRANSFER_DST)

    def _cmd_after_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        cmd.memory_barrier(MemoryUsage.TRANSFER_DST, self.memory_usage)

    def _cmd_acquire_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        cmd.buffer_barrier(
            resource.buffer,
            MemoryUsage.NONE,
            self.memory_usage,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )

    def _cmd_release_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        cmd.buffer_barrier(
            resource.buffer,
            MemoryUsage.TRANSFER_DST,
            MemoryUsage.NONE,
            self.ctx.transfer_queue_family_index,
            self.ctx.graphics_queue_family_index,
        )


# class GpuResourceProperty(Generic[R]):
#     def __init__(
#         self,
#         ctx: Context,
#         num_frames_in_flight: int,
#         upload_method: UploadMethod,
#         thread_pool: ThreadPool,
#         out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
#         property: "Property",
#         pipeline_stage_flags: PipelineStageFlags,
#         name: str,
#     ):
#         self.ctx = ctx
#         self.num_frames_in_flight = num_frames_in_flight
#         self.upload_method = upload_method
#         self.thread_pool = thread_pool
#         self.property = property
#         self.name = name
#         self.pipeline_stage_flags = pipeline_stage_flags

#         # Common
#         self.current: Optional[R] = None
#         self.current_cpu_buf: Optional[CpuBuffer] = None
#         self.frame_generation_indices = [0] * self.property.num_frames

#         # Variants
#         self.preupload = self.property.upload.preupload
#         self.async_load = self.property.upload.async_load
#         self.mapped = (
#             upload_method == UploadMethod.MAPPED_PREFER_DEVICE or upload_method == UploadMethod.MAPPED_PREFER_HOST
#         )

#         # Preupload variants
#         self.dynamic = False
#         self.jagged = False

#         # If preupload
#         self.resources: List[R] = []

#         # If preupload and dynamic
#         self.resources_frame_generation: List[int] = []
#         self.property_frame_indices_in_flight: List[Optional[int]] = [None] * self.num_frames_in_flight

#         # If (preupload and dynamic) or streaming
#         self.cpu_buffers: List[CpuBuffer] = []
#         self.cpu_pool: Optional[LRUPool[int, CpuBuffer]] = None

#         # If streaming and not mapped
#         self.gpu_resources: List[GpuResource[R]] = []
#         self.gpu_pool: Optional[LRUPool[int, GpuResource[R]]] = None

#         # If streaming and prefetch and not mapped
#         self.prefetch_states: List[PrefetchState] = []
#         self.prefetch_states_lookup: Dict[GpuResource[R], PrefetchState] = {}

#         # Upload
#         if self.preupload:
#             if upload_method == UploadMethod.MAPPED_PREFER_HOST:
#                 alloc_type = AllocType.HOST
#             elif upload_method == UploadMethod.MAPPED_PREFER_DEVICE:
#                 alloc_type = AllocType.DEVICE_MAPPED
#             else:
#                 alloc_type = AllocType.DEVICE

#             # NOTE: In the async_load case we could also do mapped upload in
#             # threads here, but we first need to know the size of the frame.
#             # This makes things more complicated because we could run into
#             # issues using the context from multiple threads. It's not
#             # clear where is the best way to ensure this is thread safe so
#             # for now we don't do it.
#             if self.async_load:
#                 promises: List[Promise[PropertyItem]] = []
#                 for i in range(property.num_frames):
#                     promise: Promise[PropertyItem] = Promise()
#                     self.thread_pool.submit(promise, self._load_async, i)  # type: ignore
#                     promises.append(promise)

#             nbytes = 0
#             for i in range(property.num_frames):
#                 if self.async_load:
#                     frame = promises[i].get()
#                 else:
#                     frame = property.get_frame_by_index(i)

#                 if i == 0:
#                     nbytes = frame.nbytes
#                 elif nbytes != frame.nbytes:
#                     self.jagged = True

#                 res = self._create_resource_for_preupload(frame, alloc_type, f"{name}-{i}")
#                 if self.mapped:
#                     self._upload_mapped_resource(res, frame)
#                 else:
#                     out_upload_list.append(self._create_bulk_upload_descriptor(res, frame))
#                 self.resources.append(res)
#         else:
#             gpu_prefetch_count = (
#                 self.property.upload.gpu_prefetch_count if upload_method == UploadMethod.TRANSFER_QUEUE else 0
#             )
#             gpu_resources_count = 1 + gpu_prefetch_count
#             cpu_prefetch_count = self.property.upload.cpu_prefetch_count
#             cpu_buffers_count = num_frames_in_flight + cpu_prefetch_count + gpu_prefetch_count

#             cpu_alloc_type = (
#                 AllocType.DEVICE_MAPPED if self.upload_method == UploadMethod.MAPPED_PREFER_DEVICE else AllocType.HOST
#             )
#             self.cpu_buffers = [
#                 CpuBuffer(self._create_cpu_buffer(f"cpubuf-{name}-{i}", cpu_alloc_type))
#                 for i in range(cpu_buffers_count)
#             ]
#             self.cpu_pool = LRUPool(
#                 self.cpu_buffers,
#                 num_frames_in_flight,
#                 self.frame_generation_indices,  # type: ignore
#                 cpu_prefetch_count,
#             )

#             if not self.mapped:
#                 for i in range(gpu_resources_count):
#                     res = self._create_gpu_resource(f"gpubuf-{name}-{i}")
#                     if upload_method == UploadMethod.TRANSFER_QUEUE:
#                         semaphore = TimelineSemaphore(ctx, name=f"gpubuf-{name}-{i}-semaphore")
#                     else:
#                         semaphore = None
#                     self.gpu_resources.append(GpuResource(res, semaphore))

#                 self.gpu_pool = LRUPool(
#                     self.gpu_resources,
#                     1,
#                     self.frame_generation_indices,  # type: ignore
#                     gpu_prefetch_count,
#                 )

#                 if upload_method == UploadMethod.TRANSFER_QUEUE:
#                     self.prefetch_states = [
#                         PrefetchState(
#                             commands=CommandBuffer(
#                                 ctx,
#                                 queue_family_index=ctx.transfer_queue_family_index,
#                                 name=f"gpu-prefetch-commands-{name}-{i}",
#                             ),
#                             prefetch_done_value=0,
#                         )
#                         for i in range(gpu_prefetch_count)
#                     ]

#     def _load_async(self, i: int, thread_index: int) -> "PropertyItem":
#         return self.property.get_frame_by_index(i, thread_index)

#     def _load_async_into(self, i: int, buf: CpuBuffer, thread_index: int) -> None:
#         buf.used_size = self.property.get_frame_by_index_into(i, buf.buf.data, thread_index)

#     def load(self, frame: RendererFrame) -> None:
#         # Issue CPU loads if async
#         if not self.preupload or self.dynamic:
#             assert self.cpu_pool is not None

#             self.cpu_pool.release_frame(frame.index)

#             def cpu_load(k: int, buf: CpuBuffer) -> None:
#                 if self.async_load:
#                     self.thread_pool.submit(buf.promise, self._load_async_into, k, buf)  # type: ignore
#                 else:
#                     buf.used_size = self.property.get_frame_by_index_into(k, buf.buf.data)

#             property_frame_index = self.property.current_frame_index

#             if self.dynamic:
#                 if not self.jagged and self.mapped:
#                     do_load = not self.cpu_pool.is_available(property_frame_index)
#                 else:
#                     do_load = (
#                         self.resources_frame_generation[property_frame_index]
#                         != self.frame_generation_indices[property_frame_index]
#                     )
#             else:
#                 do_load = (
#                     self.mapped or not self.gpu_pool.is_available_or_prefetching(property_frame_index)  # type: ignore
#                 )

#             if do_load:
#                 self.current_cpu_buf = self.cpu_pool.get(property_frame_index, cpu_load)

#     def upload(self, frame: RendererFrame) -> None:
#         # NOTE: unless we are doing BAR uploads here, we could delay
#         # waiting for buffers to be ready to right before submit.
#         # Even in the case of CPU uploads you could technically schedule them
#         # asynchronously (ideally on a thread pool that does memcpy with the
#         # GIL released) or have the loader thread do them (with care for
#         # write combining memory). This way we can do python stuff until
#         # the last moment we need the data to be ready.

#         # Issue GPU loads
#         property_frame_index = self.property.current_frame_index

#         if self.preupload:
#             res = self.resources[property_frame_index]
#             self.current = res
#             if (cpu_buf := self.current_cpu_buf) is not None:
#                 assert self.cpu_pool is not None
#                 assert self.current_cpu_buf is not None
#                 self.current_cpu_buf = None

#                 if self.async_load:
#                     cpu_buf.promise.get()

#                 self.cpu_pool.use_frame(frame.index, property_frame_index)

#                 # Preupload dynamic gpu managed, or jagged, issue upload on GFX queue
#                 if self.jagged or not self.mapped:
#                     # Upload on gfx queue
#                     self._cmd_before_barrier(frame.cmd, res)
#                     self._cmd_upload(frame.cmd, cpu_buf, res)
#                     self._cmd_after_barrier(frame.cmd, res)

#                     # Update generation index for this frame
#                     self.resources_frame_generation[property_frame_index] = self.frame_generation_indices[
#                         property_frame_index
#                     ]
#                 else:
#                     # NOTE: only works for buffers for now, which is why we get a type error here.
#                     self.resources[property_frame_index] = cpu_buf.buf  # type: ignore
#                     self.current = cpu_buf.buf  # type: ignore
#             self.property_frame_indices_in_flight[frame.index] = property_frame_index
#         else:
#             assert self.cpu_pool is not None

#             # Wait for buffer to be ready
#             if self.mapped:
#                 assert self.current_cpu_buf is not None
#                 cpu_buf = self.current_cpu_buf
#                 self.current_cpu_buf = None

#                 if self.async_load:
#                     cpu_buf.promise.get()

#                 self.cpu_pool.use_frame(frame.index, property_frame_index)

#                 # NOTE: only works for buffers for now, which is why we get a type error here.
#                 self.current = cpu_buf.buf  # type: ignore
#             else:
#                 assert self.gpu_pool is not None

#                 def gpu_load(k: int, gpu_res: GpuResource[R]) -> None:
#                     assert self.cpu_pool is not None
#                     assert self.current_cpu_buf is not None

#                     cpu_buf = self.current_cpu_buf
#                     self.current_cpu_buf = None

#                     # Wait for buffer to be ready
#                     if self.async_load:
#                         cpu_buf.promise.get()

#                     self.cpu_pool.use_frame(frame.index, k)
#                     if self.upload_method == UploadMethod.GRAPHICS_QUEUE:
#                         # Upload on gfx queue
#                         self._cmd_before_barrier(frame.cmd, gpu_res.resource)
#                         self._cmd_upload(frame.cmd, cpu_buf, gpu_res.resource)
#                         self._cmd_after_barrier(frame.cmd, gpu_res.resource)

#                         gpu_res.state = GpuResourceState.RENDER
#                     else:
#                         assert self.upload_method == UploadMethod.TRANSFER_QUEUE
#                         assert (
#                             gpu_res.state == GpuResourceState.EMPTY
#                             or gpu_res.state == GpuResourceState.RENDER
#                             or GpuResourceState.PREFETCH
#                         ), gpu_res.state
#                         assert frame.transfer_cmd is not None

#                         frame.transfer_semaphores.append(gpu_res.use(PipelineStageFlags.TRANSFER))

#                         # Upload on copy queue
#                         self._cmd_before_barrier(frame.transfer_cmd, gpu_res.resource)
#                         self._cmd_upload(frame.transfer_cmd, cpu_buf, gpu_res.resource)
#                         self._cmd_release_barrier(frame.transfer_cmd, gpu_res.resource)

#                         gpu_res.state = GpuResourceState.LOAD

#                 def gpu_ensure(k: int, gpu_res: GpuResource[R]) -> None:
#                     assert gpu_res.state == GpuResourceState.PREFETCH, gpu_res.state

#                 gpu_res = self.gpu_pool.get(property_frame_index, gpu_load, gpu_ensure)
#                 self.gpu_pool.give_back(property_frame_index, gpu_res)

#                 if gpu_res.state == GpuResourceState.LOAD or gpu_res.state == GpuResourceState.PREFETCH:
#                     if gpu_res.state == GpuResourceState.PREFETCH:
#                         assert frame.transfer_cmd is not None
#                         frame.transfer_semaphores.append(gpu_res.use(PipelineStageFlags.TOP_OF_PIPE))
#                         self._cmd_release_barrier(frame.transfer_cmd, gpu_res.resource)

#                     self._cmd_acquire_barrier(frame.cmd, gpu_res.resource)
#                     gpu_res.state = GpuResourceState.RENDER

#                 # If we are using the transfer queue to upload we have to guard the gpu resource
#                 # with a semaphore because we might need to reuse this buffer for pre-fetching
#                 # while this frame is still in flight. Using the use_frame/release_frame mechanism
#                 # with a single frame is not enough because we might try to start pre-fetching on
#                 # the next frame while this frame is still in flight.
#                 if self.upload_method == UploadMethod.TRANSFER_QUEUE:
#                     frame.additional_semaphores.append(gpu_res.use(self.pipeline_stage_flags))

#                 assert gpu_res.state == GpuResourceState.RENDER, gpu_res.state
#                 self.current = gpu_res.resource

#     def prefetch(self) -> None:
#         if not self.preupload:
#             assert self.cpu_pool is not None

#             if self.async_load:
#                 # Issue prefetches
#                 def cpu_prefetch_cleanup(k: int, buf: CpuBuffer) -> bool:
#                     return buf.promise.is_set()

#                 def cpu_prefetch(k: int, buf: CpuBuffer) -> None:
#                     self.thread_pool.submit(buf.promise, self._load_async_into, k, buf)  # type: ignore

#                 # TODO: can likely improve prefetch logic, and should probably allow
#                 # this to be hooked / configured somehow
#                 #
#                 # maybe we should just compute the state of next (maybe few) frames
#                 # (assuming constant dt, playback state) once globally and call prefetch with this.
#                 prefetch_start = self.property.current_frame_index + 1
#                 prefetch_end = prefetch_start + self.property.upload.cpu_prefetch_count
#                 prefetch_range = [self.property.get_frame_index(0, i) for i in range(prefetch_start, prefetch_end)]
#                 self.cpu_pool.prefetch(prefetch_range, cpu_prefetch_cleanup, cpu_prefetch)

#             if self.upload_method == UploadMethod.TRANSFER_QUEUE:
#                 assert self.gpu_pool is not None

#                 def gpu_prefetch_cleanup(k: int, gpu_res: GpuResource[R]) -> bool:
#                     state = self.prefetch_states_lookup[gpu_res]

#                     assert gpu_res.semaphore is not None
#                     assert self.cpu_pool is not None

#                     if gpu_res.semaphore.get_value() >= state.prefetch_done_value:
#                         # Release prefetch state
#                         self.prefetch_states.append(state)
#                         del self.prefetch_states_lookup[gpu_res]

#                         # Release buffer
#                         self.cpu_pool.release_manual(k)

#                         assert gpu_res.state == GpuResourceState.RENDER or gpu_res.state == GpuResourceState.PREFETCH
#                         return True
#                     return False

#                 def gpu_prefetch(k: int, gpu_res: GpuResource[R]) -> None:
#                     assert self.cpu_pool is not None
#                     assert (
#                         gpu_res.state == GpuResourceState.EMPTY
#                         or GpuResourceState.PREFETCH
#                         or gpu_res.state == GpuResourceState.RENDER
#                     ), gpu_res.state

#                     # We know that the cpu buffer is available, so just get it
#                     cpu_next: CpuBuffer = self.cpu_pool.get(k, lambda _x, _y: None)
#                     self.cpu_pool.use_manual(k)

#                     # Get free prefetch state
#                     state = self.prefetch_states.pop()
#                     self.prefetch_states_lookup[gpu_res] = state

#                     with state.commands:
#                         self._cmd_before_barrier(state.commands, gpu_res.resource)
#                         self._cmd_upload(state.commands, cpu_next, gpu_res.resource)

#                     info = gpu_res.use(PipelineStageFlags.TRANSFER)
#                     self.ctx.transfer_queue.submit(
#                         state.commands,
#                         wait_semaphores=[(info.sem, info.wait_stage)],
#                         wait_timeline_values=[info.wait_value],
#                         signal_semaphores=[info.sem],
#                         signal_timeline_values=[info.signal_value],
#                     )
#                     state.prefetch_done_value = info.signal_value
#                     gpu_res.state = GpuResourceState.PREFETCH

#                 # TODO: fix, same as above
#                 prefetch_start = self.property.current_frame_index + 1
#                 prefetch_end = prefetch_start + self.property.upload.gpu_prefetch_count
#                 available_range = []
#                 for i in range(prefetch_start, prefetch_end):
#                     frame_index = self.property.get_frame_index(0, i)
#                     if self.cpu_pool.is_available(frame_index):
#                         available_range.append(frame_index)

#                 self.gpu_pool.prefetch(
#                     available_range,
#                     gpu_prefetch_cleanup,
#                     gpu_prefetch,
#                 )

#     def get_current(self) -> R:
#         assert self.current
#         return self.current

#     def invalidate_frame(self, invalidated_property_frame_index: int) -> None:
#         if self.preupload and not self.dynamic:
#             self.dynamic = True

#             if not self.jagged and self.mapped:
#                 # NOTE: only works for buffers for now, which is why we get a type error here.

#                 # Promote buffers to CPU LRU pool buffers
#                 self.cpu_buffers = [CpuBuffer(r) for r in self.resources]  # type: ignore
#                 pre_initialized: List[Optional[int]] = list(range(len(self.resources)))

#                 # Allocate extra staging buffers for LRU pool
#                 cpu_alloc_type = (
#                     AllocType.DEVICE_MAPPED
#                     if self.upload_method == UploadMethod.MAPPED_PREFER_DEVICE
#                     else AllocType.HOST
#                 )
#                 for i in range(self.num_frames_in_flight):
#                     self.cpu_buffers.append(
#                         CpuBuffer(self._create_cpu_buffer(f"cpubuf-{self.name}-{i}", cpu_alloc_type))
#                     )
#                     pre_initialized.append(None)

#                 # Initialize LRU pool
#                 self.cpu_pool = LRUPool(
#                     self.cpu_buffers,
#                     self.num_frames_in_flight,
#                     self.frame_generation_indices,  # type: ignore
#                     0,
#                     pre_initialized,
#                 )
#                 for frame_index, property_frame_index in enumerate(self.property_frame_indices_in_flight):
#                     if property_frame_index is not None:
#                         self.cpu_pool.use_frame(frame_index, property_frame_index)
#             else:
#                 # Allocate new staging buffers
#                 self.cpu_buffers = [
#                     CpuBuffer(self._create_cpu_buffer(f"cpubuf-{self.name}-{i}", AllocType.HOST))
#                     for i in range(self.num_frames_in_flight)
#                 ]
#                 self.cpu_pool = LRUPool(
#                     self.cpu_buffers,
#                     self.num_frames_in_flight,
#                     self.frame_generation_indices,  # type: ignore
#                 )
#                 self.resources_frame_generation = [0] * len(self.resources)

#         assert self.cpu_pool is not None

#         self.cpu_pool.increment_generation(invalidated_property_frame_index)
#         if self.gpu_pool is not None:
#             self.gpu_pool.increment_generation(invalidated_property_frame_index)

#         # Bump generation
#         self.frame_generation_indices[invalidated_property_frame_index] += 1

#     def destroy(self) -> None:
#         self.current = None
#         if self.preupload:
#             for res in self.resources:
#                 res.destroy()
#             self.resources.clear()
#         else:
#             assert self.cpu_pool
#             for cpu_buf in self.cpu_buffers:
#                 cpu_buf.destroy()
#             self.cpu_buffers.clear()
#             for gpu_res in self.gpu_resources:
#                 gpu_res.destroy()
#             self.gpu_resources.clear()
#             self.cpu_pool.clear()
#             if self.gpu_pool is not None:
#                 self.gpu_pool.clear()

#     def _create_resource_for_preupload(self, frame: "PropertyItem", alloc_type: AllocType, name: str) -> R:
#         raise NotImplementedError

#     def _upload_mapped_resource(self, resource: R, frame: "PropertyItem") -> None:
#         raise NotImplementedError

#     def _create_bulk_upload_descriptor(
#         self, resource: R, frame: "PropertyItem"
#     ) -> Union[BufferUploadInfo, ImageUploadInfo]:
#         raise NotImplementedError

#     def _create_cpu_buffer(self, name: str, alloc_type: AllocType) -> Buffer:
#         raise NotImplementedError

#     def _create_gpu_resource(self, name: str) -> R:
#         raise NotImplementedError

#     def _cmd_upload(self, cmd: CommandBuffer, cpu_buf: CpuBuffer, resource: R) -> None:
#         raise NotImplementedError

#     def _cmd_before_barrier(self, cmd: CommandBuffer, resource: R) -> None:
#         raise NotImplementedError

#     def _cmd_after_barrier(self, cmd: CommandBuffer, resource: R) -> None:
#         raise NotImplementedError

#     def _cmd_acquire_barrier(self, cmd: CommandBuffer, resource: R) -> None:
#         raise NotImplementedError

#     def _cmd_release_barrier(self, cmd: CommandBuffer, resource: R) -> None:
#         raise NotImplementedError


# class GpuBufferProperty(GpuResourceProperty[GpuBufferView]):
#     def __init__(
#         self,
#         ctx: Context,
#         num_frames_in_flight: int,
#         upload_method: UploadMethod,
#         thread_pool: ThreadPool,
#         out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
#         property: "BufferProperty",
#         usage_flags: BufferUsageFlags,
#         memory_usage: MemoryUsage,
#         pipeline_stage_flags: PipelineStageFlags,
#         name: str,
#     ):
#         self.usage_flags = usage_flags | BufferUsageFlags.TRANSFER_DST
#         self.memory_usage = memory_usage
#         self.size = property.max_size

#         super().__init__(
#             ctx,
#             num_frames_in_flight,
#             upload_method,
#             thread_pool,
#             out_upload_list,
#             property,
#             pipeline_stage_flags,
#             name,
#         )

#     def _create_resource_for_preupload(self, frame: "PropertyItem", alloc_type: AllocType, name: str) -> GpuBufferView:
#         buf = Buffer(self.ctx, len(view_bytes(frame)), self.usage_flags, alloc_type, name=name)
#         return GpuBufferView(buf, 0, buf.size)

#     def _upload_mapped_resource(self, resource: GpuBufferView, frame: "PropertyItem") -> None:
#         # NOTE: here we can assume that the buffer is always the same size as frame data.
#         resource.buffer.data[resource.offset : resource.offset + resource.size] = view_bytes(frame)

#     def _create_bulk_upload_descriptor(self, resource: GpuBufferView, frame: "PropertyItem") -> BufferUploadInfo:
#         # TODO: add support for ranged upload
#         return BufferUploadInfo(view_bytes(frame), resource.buffer)

#     def _create_cpu_buffer(self, name: str, alloc_type: AllocType) -> Buffer:
#         if self.mapped:
#             # TRANSFER_SRC is needed if promoted to dynamic
#             usage_flags = self.usage_flags | BufferUsageFlags.TRANSFER_SRC
#         else:
#             usage_flags = BufferUsageFlags.TRANSFER_SRC

#         return Buffer(
#             self.ctx,
#             self.size,
#             usage_flags,
#             alloc_type,
#             name=name,
#         )

#     def _create_gpu_resource(self, name: str) -> GpuBufferView:
#         return GpuBufferView(
#             Buffer(
#                 self.ctx,
#                 self.size,
#                 self.usage_flags,
#                 AllocType.DEVICE,
#                 name=name,
#             ),
#             0,
#             self.size,
#         )

#     def _cmd_upload(
#         self,
#         cmd: CommandBuffer,
#         cpu_buf: CpuBuffer,
#         resource: GpuBufferView,
#     ) -> None:
#         cmd.copy_buffer_range(cpu_buf.buf, resource.buffer, cpu_buf.used_size, resource.offset)

#     def _cmd_before_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
#         cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.TRANSFER_DST)

#     def _cmd_after_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
#         cmd.memory_barrier(MemoryUsage.TRANSFER_DST, self.memory_usage)

#     def _cmd_acquire_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
#         cmd.buffer_barrier(
#             resource.buffer,
#             MemoryUsage.NONE,
#             self.memory_usage,
#             self.ctx.transfer_queue_family_index,
#             self.ctx.graphics_queue_family_index,
#         )

#     def _cmd_release_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
#         cmd.buffer_barrier(
#             resource.buffer,
#             MemoryUsage.TRANSFER_DST,
#             MemoryUsage.NONE,
#             self.ctx.transfer_queue_family_index,
#             self.ctx.graphics_queue_family_index,
#         )


# class GpuImageProperty(GpuResourceProperty[GpuImageView]):
#     def __init__(
#         self,
#         ctx: Context,
#         num_frames_in_flight: int,
#         upload_method: UploadMethod,
#         thread_pool: ThreadPool,
#         out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]],
#         property: "ImageProperty",
#         usage_flags: ImageUsageFlags,
#         layout: ImageLayout,
#         memory_usage: MemoryUsage,
#         pipeline_stage_flags: PipelineStageFlags,
#         srgb: bool,
#         mips: bool,
#         name: str,
#     ):
#         if upload_method != UploadMethod.GRAPHICS_QUEUE and upload_method != UploadMethod.TRANSFER_QUEUE:
#             raise ValueError(
#                 f"GpuImageProperty supports only {UploadMethod.GRAPHICS_QUEUE} and {UploadMethod.TRANSFER_QUEUE} upload methods. Got {upload_method}."
#             )
#         self.usage_flags = usage_flags | ImageUsageFlags.TRANSFER_DST
#         self.layout = layout
#         self.memory_usage = memory_usage

#         # NOTE: we copy this here but we don't really expect this stuff to change, so we could avoid it.
#         # We have some issues with typing of self.property but maybe we should solve those.
#         self.height = property.height
#         self.width = property.width
#         self.format = property.format
#         self.srgb = srgb
#         self.mips = mips

#         self.pitch, self.rows, _ = get_image_pitch_rows_and_texel_size(self.width, self.height, self.format)
#         super().__init__(
#             ctx,
#             num_frames_in_flight,
#             upload_method,
#             thread_pool,
#             out_upload_list,
#             property,
#             pipeline_stage_flags,
#             name,
#         )

#     def __create_image(self, alloc_type: AllocType, name: str) -> GpuImageView:
#         # TODO: if using texture arrays, need to select layer here.
#         image_create_flags = ImageCreateFlags.NONE
#         if self.srgb:
#             image_create_flags |= ImageCreateFlags.MUTABLE_FORMAT

#         mip_levels = 1
#         if self.mips:
#             mip_levels = max(self.width.bit_length(), self.height.bit_length())

#         img = Image(
#             self.ctx,
#             self.width,
#             self.height,
#             self.format,
#             self.usage_flags,
#             alloc_type,
#             mip_levels=mip_levels,
#             create_flags=image_create_flags,
#             name=name,
#         )
#         srgb_view = None
#         if self.srgb:
#             srgb_view = ImageView(
#                 self.ctx,
#                 img,
#                 ImageViewType.TYPE_2D,
#                 to_srgb_format(self.format),
#                 usage_flags=ImageUsageFlags.SAMPLED,
#                 name=name,
#             )

#         mip_level_0_view = None
#         mip_views = []
#         if self.mips:
#             mip_level_0_view = ImageView(
#                 self.ctx,
#                 img,
#                 ImageViewType.TYPE_2D_ARRAY,
#                 to_srgb_format(self.format) if self.srgb else self.format,
#                 usage_flags=ImageUsageFlags.SAMPLED,
#                 name=name,
#             )
#             mip_views = [
#                 ImageView(
#                     self.ctx,
#                     img,
#                     ImageViewType.TYPE_2D_ARRAY,
#                     base_mip_level=m if m < img.mip_levels else 0,
#                     mip_level_count=1,
#                     name=name,
#                 )
#                 for m in range(mip_levels)
#             ]

#         return GpuImageView(img, srgb_view, mip_level_0_view, mip_views)

#     def _create_resource_for_preupload(self, frame: "PropertyItem", alloc_type: AllocType, name: str) -> GpuImageView:
#         return self.__create_image(alloc_type, name)

#     def _create_bulk_upload_descriptor(self, resource: GpuImageView, frame: "PropertyItem") -> ImageUploadInfo:
#         # TODO: upload to texture array
#         return ImageUploadInfo(
#             view_bytes(frame),
#             resource.image,
#             self.layout,
#             resource.mip_level_0_view,
#             resource.mip_views,
#             MipGenerationFilter.AVERAGE_SRGB if self.srgb else MipGenerationFilter.AVERAGE,
#         )

#     def _create_cpu_buffer(self, name: str, alloc_type: AllocType) -> Buffer:
#         return Buffer(
#             self.ctx,
#             self.pitch * self.rows,
#             BufferUsageFlags.TRANSFER_SRC,
#             alloc_type,
#             name=name,
#         )

#     def _create_gpu_resource(self, name: str) -> GpuImageView:
#         return self.__create_image(AllocType.DEVICE, name)

#     def _cmd_upload(
#         self,
#         cmd: CommandBuffer,
#         cpu_buf: CpuBuffer,
#         resource: GpuImageView,
#     ) -> None:
#         # TODO: if using texture arrays, need to select layer here.
#         cmd.copy_buffer_to_image(cpu_buf.buf, resource.image)

#     def _cmd_before_barrier(self, cmd: CommandBuffer, resource: GpuImageView) -> None:
#         # TODO: if using texture arrays, need to select layer here.
#         cmd.image_barrier(
#             resource.image,
#             ImageLayout.TRANSFER_DST_OPTIMAL,
#             MemoryUsage.ALL,
#             MemoryUsage.TRANSFER_DST,
#             undefined=True,
#         )

#     def _cmd_after_barrier(self, cmd: CommandBuffer, resource: GpuImageView) -> None:
#         # TODO: if using texture arrays, need to select layer here.
#         cmd.image_barrier(
#             resource.image,
#             self.layout,
#             MemoryUsage.TRANSFER_DST,
#             self.memory_usage,
#         )

#     def _cmd_acquire_barrier(self, cmd: CommandBuffer, resource: GpuImageView) -> None:
#         # TODO: if using texture arrays, need to select layer here.
#         cmd.image_barrier(
#             resource.image,
#             self.layout,
#             MemoryUsage.NONE,
#             self.memory_usage,
#             self.ctx.transfer_queue_family_index,
#             self.ctx.graphics_queue_family_index,
#         )

#     def _cmd_release_barrier(self, cmd: CommandBuffer, resource: GpuImageView) -> None:
#         # TODO: if using texture arrays, need to select layer here.
#         cmd.image_barrier(
#             resource.image,
#             self.layout,
#             MemoryUsage.TRANSFER_DST,
#             MemoryUsage.NONE,
#             self.ctx.transfer_queue_family_index,
#             self.ctx.graphics_queue_family_index,
#         )
