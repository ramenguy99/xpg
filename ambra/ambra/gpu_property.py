# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

from numpy.typing import NDArray
from pyxpg import (
    AccessFlags,
    AllocType,
    Buffer,
    BufferBarrier,
    BufferUsageFlags,
    CommandBuffer,
    Context,
    DescriptorSet,
    DescriptorType,
    Format,
    Image,
    ImageBarrier,
    ImageCreateFlags,
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    ImageViewType,
    MemoryUsage,
    PipelineStageFlags,
    TimelineSemaphore,
)

from .config import UploadMethod
from .ffx import MipGenerationRequest, SPDPipelineInstance
from .renderer_frame import RendererFrame, SemaphoreInfo
from .utils.gpu import (
    BufferUploadInfo,
    ImageUploadInfo,
    MipGenerationFilter,
    get_image_pitch_rows_and_texel_size,
    to_srgb_format,
    view_bytes,
)
from .utils.lru_pool import LRUPool
from .utils.ring_buffer import RingBuffer
from .utils.threadpool import Promise

if TYPE_CHECKING:
    from .property import BufferProperty, ImageProperty, Property, PropertyItem
    from .renderer import Renderer


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

    def append_view_to_destroy_list(self, destroy_list: List[Union[Buffer, Image, ImageView]]) -> None:
        pass

    def resource(self) -> Buffer:
        return self.buffer


@dataclass
class GpuImageView:
    image: Image
    srgb_resource_view: Optional[ImageView]
    mip_level_0_view: Optional[ImageView]
    mip_views: List[ImageView]

    def destroy(self) -> None:
        self.image.destroy()

    def view(self) -> Union[Image, ImageView]:
        if self.srgb_resource_view is not None:
            return self.srgb_resource_view
        else:
            return self.image

    def append_view_to_destroy_list(self, destroy_list: List[Union[Buffer, Image, ImageView]]) -> None:
        if self.srgb_resource_view is not None:
            destroy_list.append(self.srgb_resource_view)
        if self.mip_level_0_view is not None:
            destroy_list.append(self.mip_level_0_view)
        destroy_list.extend(self.mip_views)

    def resource(self) -> Image:
        return self.image


def _create_image_view(
    ctx: Context,
    width: int,
    height: int,
    format: Format,
    usage_flags: ImageUsageFlags,
    srgb: bool,
    mips: bool,
    name: str,
) -> GpuImageView:
    image_create_flags = ImageCreateFlags.NONE
    if srgb:
        image_create_flags |= ImageCreateFlags.MUTABLE_FORMAT

    mip_levels = 1
    if mips:
        mip_levels = max(width.bit_length(), height.bit_length())
        usage_flags |= ImageUsageFlags.STORAGE

    img = Image(
        ctx,
        width,
        height,
        format,
        usage_flags,
        AllocType.DEVICE,
        mip_levels=mip_levels,
        create_flags=image_create_flags,
        name=name,
    )
    srgb_view = None
    if srgb:
        srgb_view = ImageView(
            ctx,
            img,
            ImageViewType.TYPE_2D,
            to_srgb_format(format),
            usage_flags=ImageUsageFlags.SAMPLED,
            name=f"{name}-srgb",
        )

    mip_level_0_view = None
    mip_views = []
    if mips:
        mip_level_0_view = ImageView(
            ctx,
            img,
            ImageViewType.TYPE_2D_ARRAY,
            to_srgb_format(format) if srgb else format,
            usage_flags=ImageUsageFlags.SAMPLED,
            name=f"{name}-mip-0",
        )
        mip_views = [
            ImageView(
                ctx,
                img,
                ImageViewType.TYPE_2D_ARRAY,
                base_mip_level=m if m < img.mip_levels else 0,
                mip_level_count=1,
                name=f"{name}-mip-{m + 1}",
            )
            for m in range(mip_levels)
        ]

    return GpuImageView(img, srgb_view, mip_level_0_view, mip_views)


R = TypeVar("R", bound=Union[Buffer, Image])
V = TypeVar("V", bound=Union[GpuBufferView, GpuImageView])


class GpuResource(Generic[V]):
    def __init__(self, resource: V, semaphore: Optional[TimelineSemaphore]):
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
            stage,
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


class GpuPropertySupportedOperations(Flag):
    UPDATE = auto()
    APPEND = auto()
    POP = auto()
    INSERT = auto()
    REMOVE = auto()


class GpuProperty(Generic[V]):
    supported_operations: GpuPropertySupportedOperations
    name: str

    # Public API
    def get_current(self) -> V:
        raise NotImplementedError

    def enqueue_for_destruction(self) -> None:
        pass

    # Renderer API
    def load(self, frame: RendererFrame) -> None:
        pass

    def upload(self, frame: RendererFrame) -> None:
        pass

    def prefetch(self) -> None:
        pass

    # Edit API
    def update_frame(self, index: int) -> None:
        raise NotImplementedError

    def update_frames(self, indices: List[int]) -> None:
        raise NotImplementedError

    def update_frame_range(self, start: int, stop: int) -> None:
        raise NotImplementedError

    def append_frame(self) -> None:
        raise NotImplementedError

    def append_frames(self, count: int) -> None:
        raise NotImplementedError

    def pop_frame(self) -> None:
        raise NotImplementedError

    def pop_frames(self, count: int) -> None:
        raise NotImplementedError

    def insert_frame(self, index: int) -> None:
        raise NotImplementedError

    def insert_frames(self, index: int, count: int) -> None:
        raise NotImplementedError

    def remove_frame(self, index: int) -> None:
        raise NotImplementedError

    def remove_frames(self, indices: List[int]) -> None:
        raise NotImplementedError

    def remove_frame_range(self, start: int, stop: int) -> None:
        raise NotImplementedError


class GpuPreuploadedArrayProperty(GpuProperty[V], Generic[R, V]):
    def __init__(
        self,
        data: NDArray[Any],
        frame_size: int,
        renderer: "Renderer",
        upload_method: UploadMethod,
        property: "Property",
        name: str,
    ):
        self.frame_size = frame_size
        self.renderer = renderer
        self.upload_method = upload_method
        self.property = property
        self.supported_operations = GpuPropertySupportedOperations.UPDATE
        self.name = name

        self.resource, self.resource_views = self._create_preuploaded_array_resource_and_views(
            data, renderer.bulk_upload_list
        )
        self.staging_buffers: Optional[RingBuffer[Buffer]] = None
        self.current_staging_buf: Optional[Buffer] = None
        self.invalid_frames: Set[int] = set()
        self.current: Optional[V] = None

    # Public API
    def get_current(self) -> V:
        assert self.current is not None
        return self.current

    def enqueue_for_destruction(self) -> None:
        resources: List[Union[Buffer, Image, ImageView]] = [self.resource]
        if self.staging_buffers is not None:
            resources.extend(self.staging_buffers.items)
            self.staging_buffers = None

    def load(self, frame: RendererFrame) -> None:
        index = self.property.current_frame_index
        if index in self.invalid_frames:
            self.invalid_frames.remove(index)

            assert self.staging_buffers is not None

            buf = self.staging_buffers.get_current_and_advance()
            self.property.get_frame_by_index_into(index, buf.data)
            self.current_staging_buf = buf

            self._append_barriers_and_mip_requests_for_upload(frame, self.resource_views[index])

    # Renderer API
    def upload(self, frame: RendererFrame) -> None:
        view = self.resource_views[self.property.current_frame_index]
        if self.current_staging_buf is not None:
            self._cmd_upload(frame.cmd, self.current_staging_buf, view)
            self.current_staging_buf = None
        self.current = view

    # Edit API
    def update_frame(self, index: int) -> None:
        self.invalid_frames.add(index)
        self._promote_to_dynamic()

    def update_frames(self, indices: List[int]) -> None:
        for i in indices:
            self.invalid_frames.add(i)
        self._promote_to_dynamic()

    def update_frame_range(self, start: int, stop: int) -> None:
        for i in range(start, stop):
            self.invalid_frames.add(i)
        self._promote_to_dynamic()

    # Private API
    def _promote_to_dynamic(self) -> None:
        if self.staging_buffers is None:
            self.staging_buffers = RingBuffer(
                [
                    Buffer(
                        self.renderer.ctx,
                        self.frame_size,
                        BufferUsageFlags.TRANSFER_SRC,
                        AllocType.HOST,
                        f"{self.name}-staging-{i}",
                    )
                    for i in range(self.renderer.num_frames_in_flight)
                ]
            )

    def _create_preuploaded_array_resource_and_views(
        self, data: NDArray[Any], out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]]
    ) -> Tuple[R, List[V]]:
        raise NotImplementedError

    def _cmd_upload(self, cmd: CommandBuffer, staging_buffer: Buffer, view: V) -> None:
        raise NotImplementedError

    def _append_barriers_and_mip_requests_for_upload(self, frame: RendererFrame, resource: V) -> None:
        raise NotImplementedError


class GpuBufferPreuploadedArrayProperty(GpuPreuploadedArrayProperty[Buffer, GpuBufferView]):
    def __init__(
        self,
        data: NDArray[Any],
        renderer: "Renderer",
        property: "BufferProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | BufferUsageFlags.TRANSFER_DST
        self.pipeline_stage_flags = property.gpu_stage

        super().__init__(
            data,
            property.max_size,
            renderer,
            renderer.buffer_upload_method,
            property,
            name,
        )

    # Private API
    def _create_preuploaded_array_resource_and_views(
        self, data: NDArray[Any], out_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]]
    ) -> Tuple[Buffer, List[GpuBufferView]]:
        if self.upload_method == UploadMethod.MAPPED_PREFER_HOST:
            alloc_type = AllocType.HOST
        elif self.upload_method == UploadMethod.MAPPED_PREFER_DEVICE:
            alloc_type = AllocType.DEVICE_MAPPED
        else:
            alloc_type = AllocType.DEVICE

        # TODO: handle alignment here (need to pad the data)
        buffer = Buffer(self.renderer.ctx, data.nbytes, self.usage_flags, alloc_type, self.name)

        if buffer.is_mapped:
            buffer.data[:] = view_bytes(data)
        else:
            out_upload_list.append(BufferUploadInfo(view_bytes(data), buffer, 0))

        views = [GpuBufferView(buffer, self.frame_size * i, self.frame_size) for i in range(self.property.num_frames)]

        return buffer, views

    def _cmd_upload(self, cmd: CommandBuffer, staging_buffer: Buffer, view: GpuBufferView) -> None:
        cmd.copy_buffer_range(staging_buffer, view.buffer, self.frame_size, 0, view.offset)

    def _append_barriers_and_mip_requests_for_upload(self, frame: RendererFrame, resource: V) -> None:
        frame.upload_property_pipeline_stages |= self.pipeline_stage_flags


class GpuPreuploadedProperty(GpuProperty[V], Generic[R, V]):
    def __init__(
        self,
        max_frame_size: int,
        batched: bool,
        renderer: "Renderer",
        upload_method: UploadMethod,
        property: "Property",
        name: str,
    ):
        self.max_frame_size = max_frame_size
        self.batched = batched
        self.renderer = renderer
        self.upload_method = upload_method
        self.property = property
        self.supported_operations = (
            GpuPropertySupportedOperations.UPDATE
            | GpuPropertySupportedOperations.APPEND
            | GpuPropertySupportedOperations.POP
            | GpuPropertySupportedOperations.INSERT
            | GpuPropertySupportedOperations.REMOVE
            if not self.batched
            else GpuPropertySupportedOperations.UPDATE
        )
        self.name = name

        self.async_load = self.property.upload.async_load
        self.staging_buffers: Optional[RingBuffer[CpuBuffer]] = None
        self.current_staging_buf: Optional[CpuBuffer] = None
        self.invalid_frames: Set[int] = set()
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
                self.renderer.thread_pool.submit(promise, self._load_async, i)  # type: ignore
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
        self.resources, self.resource_views = self._create_preuploaded_resources_and_views(frames)

    # Public API
    def get_current(self) -> V:
        assert self.current is not None
        return self.current

    def enqueue_for_destruction(self) -> None:
        destroy_list: List[Union[Buffer, Image, ImageView]] = self.resources.copy()  # type: ignore
        for v in self.resource_views:
            v.append_view_to_destroy_list(destroy_list)
        self.renderer.enqueue_for_destruction(destroy_list)

    # Renderer API
    def load(self, frame: RendererFrame) -> None:
        index = self.property.current_frame_index
        if index in self.invalid_frames:
            self.invalid_frames.remove(index)

            assert self.staging_buffers is not None

            buf = self.staging_buffers.get_current_and_advance()
            if self.async_load:
                self.renderer.thread_pool.submit(buf.promise, self._load_async_into, index, buf)  # type: ignore
            else:
                buf.used_size = self.property.get_frame_by_index_into(index, buf.buf.data)
            self.current_staging_buf = buf

            self._append_barriers_and_mip_requests_for_upload(frame, self.resource_views[index])

    def upload(self, frame: RendererFrame) -> None:
        view = self.resource_views[self.property.current_frame_index]
        if self.current_staging_buf is not None:
            # Wait for buffer to be ready
            if self.async_load:
                self.current_staging_buf.promise.get()

            self._cmd_upload(frame.cmd, self.current_staging_buf.buf, view)
            self.current_staging_buf = None
        self.current = view

    # Edit API
    def update_frame(self, index: int) -> None:
        self.invalid_frames.add(index)
        self._promote_to_dynamic()

    def update_frames(self, indices: List[int]) -> None:
        for i in indices:
            self.invalid_frames.add(i)
        self._promote_to_dynamic()

    def update_frame_range(self, start: int, stop: int) -> None:
        for i in range(start, stop):
            self.invalid_frames.add(i)
        self._promote_to_dynamic()

    def append_frame(self) -> None:
        if self.batched:
            raise RuntimeError("Appending to a preuploaded and batched GPU property is not supported.")
        self.append_frames(1)

    def append_frames(self, count: int) -> None:
        if self.batched:
            raise RuntimeError("Appending to a preuploaded and batched GPU property is not supported.")

        current_count = len(self.resource_views)
        new_max_frame_size = 0
        for i in range(count):
            new_frame = self.property.get_frame_by_index(current_count + i)
            new_max_frame_size = max(new_max_frame_size, len(view_bytes(new_frame)))

            # Create and append a new resource and views
            resource, view = self._create_preuploaded_resource_and_view_for_frame(new_frame)
            self.resources.append(resource)
            self.resource_views.append(view)

        self._update_max_frame_size(new_max_frame_size)

    def pop_frame(self) -> None:
        if self.batched:
            raise RuntimeError("Popping from a preuploaded and batched GPU property is not supported.")

        res = self.resources.pop()
        view = self.resource_views.pop()

        destroy_list: List[Union[Buffer, Image, ImageView]] = [res]
        view.append_view_to_destroy_list(destroy_list)

        self.renderer.enqueue_for_destruction(destroy_list)

    def pop_frames(self, count: int) -> None:
        if self.batched:
            raise RuntimeError("Popping from a preuploaded and batched GPU property is not supported.")

        if count > 0:
            destroy_list: List[Union[Buffer, Image, ImageView]] = self.resources[-count:]  # type: ignore
            for v in self.resource_views[-count:]:
                v.append_view_to_destroy_list(destroy_list)
            self.renderer.enqueue_for_destruction(destroy_list)

            del self.resources[-count:]
            del self.resource_views[-count:]

    def insert_frame(self, index: int) -> None:
        if self.batched:
            raise RuntimeError("Inserting into a preuploaded and batched GPU property is not supported.")

        new_frame = self.property.get_frame_by_index(index)
        self._update_max_frame_size(len(view_bytes(new_frame)))

        # Create and append a new resource and views
        resource, view = self._create_preuploaded_resource_and_view_for_frame(new_frame)
        self.resources.insert(index, resource)
        self.resource_views.insert(index, view)

    def insert_frames(self, index: int, count: int) -> None:
        if self.batched:
            raise RuntimeError("Inserting in preuploaded and batched GPU property is not supported.")

        current_count = len(self.resource_views)

        new_resources = []
        new_resource_views = []
        new_max_frame_size = 0
        for i in range(count):
            new_frame = self.property.get_frame_by_index(current_count + i)
            new_max_frame_size = max(new_max_frame_size, len(view_bytes(new_frame)))

            # Create and append a new resource and views
            resource, view = self._create_preuploaded_resource_and_view_for_frame(new_frame)
            new_resources.append(resource)
            new_resource_views.append(view)

        self.resources = self.resources[:index] + new_resources + self.resources[index:]
        self.resource_views = self.resource_views[:index] + new_resource_views + self.resource_views[index:]

        self._update_max_frame_size(new_max_frame_size)

    def remove_frame(self, index: int) -> None:
        if self.batched:
            raise RuntimeError("Removing from a preuploaded and batched GPU property is not supported.")

        res = self.resources.pop(index)
        view = self.resource_views.pop(index)

        destroy_list: List[Union[Buffer, Image, ImageView]] = [res]
        view.append_view_to_destroy_list(destroy_list)
        self.renderer.enqueue_for_destruction([res])

    def remove_frames(self, indices: List[int]) -> None:
        if self.batched:
            raise RuntimeError("Removing from a preuploaded and batched GPU property is not supported.")

        indices_set = set(indices)

        keep_resources: List[R] = []
        keep_resource_views: List[V] = []
        destroy_list: List[Union[Buffer, Image, ImageView]] = []
        for i, (r, v) in enumerate(zip(self.resources, self.resource_views)):
            if i in indices_set:
                destroy_list.append(r)
                v.append_view_to_destroy_list(destroy_list)
            else:
                keep_resources.append(r)
                keep_resource_views.append(v)

        self.resources = keep_resources
        self.resource_views = keep_resource_views
        self.renderer.enqueue_for_destruction(destroy_list)

    def remove_frame_range(self, start: int, stop: int) -> None:
        if self.batched:
            raise RuntimeError("Removing from a preuploaded and batched GPU property is not supported.")

        # Collect resources to destroy and enqueue for destruction
        destroy_list: List[Union[Buffer, Image, ImageView]] = self.resources[start:stop]  # type: ignore
        for v in self.resource_views[start:stop]:
            v.append_view_to_destroy_list(destroy_list)
        self.renderer.enqueue_for_destruction(destroy_list)

        # Remove resources from lists
        self.resources = self.resources[:start] + self.resources[stop:]
        self.resource_views = self.resource_views[:start] + self.resource_views[stop:]

    # Private API
    def _promote_to_dynamic(self) -> None:
        if self.staging_buffers is None:
            self.staging_buffers = RingBuffer(
                [
                    CpuBuffer(
                        Buffer(
                            self.renderer.ctx,
                            self.max_frame_size,
                            BufferUsageFlags.TRANSFER_SRC,
                            AllocType.HOST,
                            f"{self.name}-staging-{i}",
                        )
                    )
                    for i in range(self.renderer.num_frames_in_flight)
                ]
            )

    def _update_max_frame_size(self, new_max_frame_size: int) -> None:
        # Update maximum frame size and clear current staging buffers if they can't fit the new frame.
        if new_max_frame_size > self.max_frame_size:
            self.max_frame_size = new_max_frame_size
            if self.staging_buffers is not None:
                self.renderer.enqueue_for_destruction([b.buf for b in self.staging_buffers])
            self.staging_buffers = None

    def _load_async(self, i: int, thread_index: int) -> "PropertyItem":
        return self.property.get_frame_by_index(i, thread_index)

    def _load_async_into(self, i: int, buf: CpuBuffer, thread_index: int) -> None:
        buf.used_size = self.property.get_frame_by_index_into(i, buf.buf.data, thread_index)

    def _create_preuploaded_resources_and_views(self, frames: List[NDArray[Any]]) -> Tuple[List[R], List[V]]:
        raise NotImplementedError

    def _create_preuploaded_resource_and_view_for_frame(self, frame: NDArray[Any]) -> Tuple[R, V]:
        raise NotImplementedError

    def _cmd_upload(self, cmd: CommandBuffer, staging_buffer: Buffer, view: V) -> None:
        raise NotImplementedError

    def _append_barriers_and_mip_requests_for_upload(self, frame: RendererFrame, resource: V) -> None:
        raise NotImplementedError


class GpuBufferPreuploadedProperty(GpuPreuploadedProperty[Buffer, GpuBufferView]):
    def __init__(
        self,
        renderer: "Renderer",
        property: "BufferProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | BufferUsageFlags.TRANSFER_DST
        self.pipeline_stage_flags = property.gpu_stage

        upload_method = renderer.buffer_upload_method
        if upload_method == UploadMethod.MAPPED_PREFER_HOST:
            self.alloc_type = AllocType.HOST
        elif upload_method == UploadMethod.MAPPED_PREFER_DEVICE:
            self.alloc_type = AllocType.DEVICE_MAPPED
        else:
            self.alloc_type = AllocType.DEVICE

        super().__init__(
            property.max_size,
            property.upload.batched,
            renderer,
            upload_method,
            property,
            name,
        )

    # Private API
    def _create_preuploaded_resources_and_views(
        self, frames: List[NDArray[Any]]
    ) -> Tuple[List[Buffer], List[GpuBufferView]]:
        resources: List[Buffer] = []
        views: List[GpuBufferView] = []
        if self.batched:
            # If batched allocate a single buffer for all the frames
            total_size_in_bytes = 0
            for frame in frames:
                total_size_in_bytes += len(view_bytes(frame))

            buffer = Buffer(self.renderer.ctx, total_size_in_bytes, self.usage_flags, self.alloc_type, self.name)
            resources.append(buffer)

            # Create views and upload
            offset = 0
            for frame in frames:
                frame_bytes = view_bytes(frame)
                size = len(frame_bytes)

                if buffer.is_mapped:
                    buffer.data[offset : offset + size] = frame_bytes
                else:
                    self.renderer.bulk_upload_list.append(BufferUploadInfo(frame_bytes, buffer, offset))
                views.append(GpuBufferView(buffer, offset, size))

                # TODO: handle alignment here depending on usage flags (vbo, ubo, ssbo alignement constraints)
                offset += size
        else:
            # If not batched allocate a separate buffer each frame
            for frame in frames:
                resource, view = self._create_preuploaded_resource_and_view_for_frame(frame)
                resources.append(resource)
                views.append(view)

        return resources, views

    def _create_preuploaded_resource_and_view_for_frame(self, frame: NDArray[Any]) -> Tuple[Buffer, GpuBufferView]:
        frame_bytes = view_bytes(frame)
        buffer = Buffer(self.renderer.ctx, len(frame_bytes), self.usage_flags, self.alloc_type, self.name)
        if buffer.is_mapped:
            buffer.data[:] = frame_bytes
        else:
            self.renderer.bulk_upload_list.append(BufferUploadInfo(frame_bytes, buffer, 0))
        return (buffer, GpuBufferView(buffer, 0, len(frame_bytes)))

    def _cmd_upload(self, cmd: CommandBuffer, staging_buffer: Buffer, view: GpuBufferView) -> None:
        cmd.copy_buffer_range(staging_buffer, view.buffer, view.size, 0, view.offset)

    def _append_barriers_and_mip_requests_for_upload(self, frame: RendererFrame, resource: V) -> None:
        frame.upload_property_pipeline_stages |= self.pipeline_stage_flags


class GpuImagePreuploadedProperty(GpuPreuploadedProperty[Image, GpuImageView]):
    def __init__(
        self,
        renderer: "Renderer",
        property: "ImageProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | ImageUsageFlags.TRANSFER_DST

        self.width = property.width
        self.height = property.height
        self.format = property.format
        self.layout = property.gpu_layout
        self.pipeline_stage_flags = property.gpu_stage
        self.srgb = property.gpu_srgb
        self.mips = property.gpu_mips

        # Created on promotion to dynamic, if needed
        self.spd_pipeline_instance: Optional[SPDPipelineInstance] = None

        self.pitch, self.rows, _ = get_image_pitch_rows_and_texel_size(self.width, self.height, self.format)

        super().__init__(
            self.pitch * self.rows,
            False,  # batching is not supported for images
            renderer,
            renderer.image_upload_method,
            property,
            name,
        )

    # Private API
    def _promote_to_dynamic(self) -> None:
        if self.mips and self.spd_pipeline_instance is not None:
            self.spd_pipeline_instance = self.renderer.spd_pipeline.alloc_instance(self.renderer, False)

            mip_levels = max(self.width.bit_length(), self.height.bit_length())
            self.spd_pipeline_instance.set_image_extents(self.width, self.height, mip_levels)

        return super()._promote_to_dynamic()

    def _create_preuploaded_resources_and_views(
        self, frames: List[NDArray[Any]]
    ) -> Tuple[List[Image], List[GpuImageView]]:
        resources: List[Image] = []
        views: List[GpuImageView] = []

        for frame in frames:
            resource, view = self._create_preuploaded_resource_and_view_for_frame(frame)
            resources.append(resource)
            views.append(view)

        return resources, views

    def _create_preuploaded_resource_and_view_for_frame(self, frame: NDArray[Any]) -> Tuple[Image, GpuImageView]:
        view = _create_image_view(
            self.renderer.ctx, self.width, self.height, self.format, self.usage_flags, self.srgb, self.mips, self.name
        )
        img = view.image

        self.renderer.bulk_upload_list.append(
            ImageUploadInfo(
                view_bytes(frame),
                img,
                self.layout,
                view.mip_level_0_view,
                view.mip_views,
                MipGenerationFilter.AVERAGE_SRGB if self.srgb else MipGenerationFilter.AVERAGE,
            )
        )

        return img, view

    def _cmd_upload(self, cmd: CommandBuffer, staging_buffer: Buffer, view: GpuImageView) -> None:
        cmd.copy_buffer_to_image(staging_buffer, view.image)

    def _append_barriers_and_mip_requests_for_upload(self, frame: RendererFrame, resource: GpuImageView) -> None:
        frame.upload_before_image_barriers.append(
            ImageBarrier(
                resource.image,
                ImageLayout.UNDEFINED,
                ImageLayout.TRANSFER_DST_OPTIMAL,
                self.pipeline_stage_flags,
                AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                PipelineStageFlags.COPY,
                AccessFlags.TRANSFER_WRITE,
            )
        )

        if self.mips:
            frame.upload_after_image_barriers.append(
                ImageBarrier(
                    resource.image,
                    ImageLayout.TRANSFER_DST_OPTIMAL,
                    ImageLayout.GENERAL,
                    PipelineStageFlags.COPY,
                    AccessFlags.TRANSFER_WRITE,
                    PipelineStageFlags.COMPUTE_SHADER,
                    AccessFlags.SHADER_SAMPLED_READ
                    | AccessFlags.SHADER_STORAGE_READ
                    | AccessFlags.SHADER_STORAGE_WRITE,
                )
            )
            mip_generation_filter = MipGenerationFilter.AVERAGE_SRGB if self.srgb else MipGenerationFilter.AVERAGE

            assert self.spd_pipeline_instance is not None
            assert resource.mip_level_0_view is not None

            descriptor_set = self.spd_pipeline_instance.get_and_write_current_and_advance(
                resource.mip_level_0_view, resource.mip_views
            )
            frame.mip_generation_requests.setdefault(mip_generation_filter, []).append(
                MipGenerationRequest(
                    resource.image,
                    self.spd_pipeline_instance.constants.tobytes(),
                    descriptor_set,
                    self.spd_pipeline_instance.groups_x,
                    self.spd_pipeline_instance.groups_y,
                    1,
                    ImageBarrier(
                        resource.image,
                        ImageLayout.GENERAL,
                        self.layout,
                        PipelineStageFlags.COMPUTE_SHADER,
                        AccessFlags.SHADER_SAMPLED_READ
                        | AccessFlags.SHADER_STORAGE_READ
                        | AccessFlags.SHADER_STORAGE_WRITE,
                        self.pipeline_stage_flags,
                        AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                    ),
                )
            )
        else:
            frame.upload_after_image_barriers.append(
                ImageBarrier(
                    resource.image,
                    ImageLayout.TRANSFER_DST_OPTIMAL,
                    self.layout,
                    PipelineStageFlags.COPY,
                    AccessFlags.TRANSFER_WRITE,
                    self.pipeline_stage_flags,
                    AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                )
            )


class GpuStreamingProperty(GpuProperty[V], Generic[R, V]):
    def __init__(
        self,
        renderer: "Renderer",
        upload_method: UploadMethod,
        property: "Property",
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ):
        self.renderer = renderer
        self.upload_method = upload_method
        self.property = property
        self.pipeline_stage_flags = pipeline_stage_flags
        self.supported_operations = (
            GpuPropertySupportedOperations.UPDATE
            | GpuPropertySupportedOperations.APPEND
            | GpuPropertySupportedOperations.POP
            | GpuPropertySupportedOperations.INSERT
            | GpuPropertySupportedOperations.REMOVE
        )
        self.name = name

        # Variants
        self.async_load = self.property.upload.async_load
        self.mapped = (
            upload_method == UploadMethod.MAPPED_PREFER_DEVICE or upload_method == UploadMethod.MAPPED_PREFER_HOST
        )

        # Common
        self.current: Optional[V] = None

        # If not mapped
        self.current_gpu_res: Optional[GpuResource[V]] = None
        self.gpu_resources: List[GpuResource[V]] = []
        self.gpu_pool: Optional[LRUPool[int, GpuResource[V]]] = None

        # If prefetch and not mapped
        self.prefetch_states: List[PrefetchState] = []
        self.prefetch_states_lookup: Dict[GpuResource[V], PrefetchState] = {}

        # Compute buffer counts
        gpu_prefetch_count = (
            self.property.upload.gpu_prefetch_count if upload_method == UploadMethod.TRANSFER_QUEUE else 0
        )
        gpu_resources_count = 1 + gpu_prefetch_count
        cpu_prefetch_count = self.property.upload.cpu_prefetch_count
        cpu_buffers_count = self.renderer.num_frames_in_flight + cpu_prefetch_count + gpu_prefetch_count

        # Allocate CPU buffers
        self.current_cpu_buf: Optional[CpuBuffer] = None
        self.cpu_buffers = [
            CpuBuffer(self._create_cpu_buffer(f"{self.name}-cpu-{i}")) for i in range(cpu_buffers_count)
        ]

        self.cpu_pool: LRUPool[int, CpuBuffer] = LRUPool(
            self.cpu_buffers,
            self.renderer.num_frames_in_flight,
            cpu_prefetch_count,
        )

        if not self.mapped:
            for i in range(gpu_resources_count):
                res = self._create_gpu_resource(f"{name}-gpu-{i}")
                if upload_method == UploadMethod.TRANSFER_QUEUE:
                    semaphore = TimelineSemaphore(self.renderer.ctx, name=f"{name}-sem-{i}")
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
                            self.renderer.ctx,
                            queue_family_index=self.renderer.ctx.transfer_queue_family_index,
                            name=f"{name}-prefetch-{i}",
                        ),
                        prefetch_done_value=0,
                    )
                    for i in range(gpu_prefetch_count)
                ]

    # Public API
    def get_current(self) -> V:
        assert self.current is not None
        return self.current

    def enqueue_for_destruction(self) -> None:
        self.renderer.enqueue_for_destruction(
            [cpu_buf.buf for cpu_buf in self.cpu_buffers] + [view.resource.resource() for view in self.gpu_resources]
        )

    # Renderer API
    def load(self, frame: RendererFrame) -> None:
        self.cpu_pool.release_frame(frame.index)

        def cpu_load(k: int, buf: CpuBuffer) -> None:
            if self.async_load:
                self.renderer.thread_pool.submit(buf.promise, self._load_async_into, k, buf)  # type: ignore
            else:
                buf.used_size = self.property.get_frame_by_index_into(k, buf.buf.data)

        property_frame_index = self.property.current_frame_index
        if self.mapped:
            self.current_cpu_buf = self.cpu_pool.get(property_frame_index, cpu_load)
        else:
            assert self.gpu_pool is not None

            def gpu_load(k: int, gpu_res: GpuResource[V]) -> None:
                assert self.current_cpu_buf is None

                cpu_buf = self.cpu_pool.get(property_frame_index, cpu_load)
                self.cpu_pool.use_frame(frame.index, k)

                self.current_cpu_buf = cpu_buf

                if self.upload_method == UploadMethod.GRAPHICS_QUEUE:
                    # Prepare for upload on graphics queue
                    self._append_barriers_for_upload_on_graphics_queue(frame, gpu_res.resource)
                    self._append_mip_generation_request(frame, gpu_res.resource)
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

                    # Prepare for upload on transfer queue
                    self._append_barriers_for_upload_on_transfer_queue(frame, gpu_res.resource)
                    self._append_release_barrier_on_transfer_queue(frame, gpu_res.resource)

                    gpu_res.state = GpuResourceState.LOAD

            def gpu_ensure(k: int, gpu_res: GpuResource[V]) -> None:
                assert gpu_res.state == GpuResourceState.PREFETCH, gpu_res.state

            gpu_res = self.gpu_pool.get(property_frame_index, gpu_load, gpu_ensure)
            self.gpu_pool.give_back(property_frame_index, gpu_res)

            if gpu_res.state == GpuResourceState.LOAD or gpu_res.state == GpuResourceState.PREFETCH:
                if gpu_res.state == GpuResourceState.PREFETCH:
                    assert frame.transfer_cmd is not None
                    frame.transfer_semaphores.append(gpu_res.use(PipelineStageFlags.ALL_COMMANDS))

                    # Sync: the resource was loaded by a prefetch operation on the transfer queue.
                    # Submit a release barrier on this frame to transfer ownership to this resource.
                    # The reason why this barrier needs to happen here instead of at the time of submission
                    # to the transfer queue, is because at that time we cannot guarantee that this resource
                    # will be used next on the graphics queue, the prefetch might be discarded and the resource
                    # used again later on the transfer queue. Since release-acquire barriers must be matched
                    # we issue the release only once we are sure that the acquire will happen.
                    self._append_release_barrier_on_transfer_queue(frame, gpu_res.resource)

                self._append_acquire_barrier_on_graphics_queue(frame, gpu_res.resource)
                self._append_mip_generation_request(frame, gpu_res.resource)
                gpu_res.state = GpuResourceState.RENDER

            assert gpu_res.state == GpuResourceState.RENDER, gpu_res.state

            # Sync: If we are using the transfer queue to upload we have to guard the gpu resource
            # with a semaphore because we might need to reuse this buffer for pre-fetching
            # while this frame is still in flight. Using the use_frame/release_frame mechanism
            # with a single frame is not enough because we might try to start pre-fetching on
            # the next frame while this frame is still in flight.
            if self.upload_method == UploadMethod.TRANSFER_QUEUE:
                frame.additional_semaphores.append(gpu_res.use(self.pipeline_stage_flags))

            self.current_gpu_res = gpu_res

    def upload(self, frame: RendererFrame) -> None:
        property_frame_index = self.property.current_frame_index

        # Wait for buffer to be ready if needed
        if self.async_load and self.current_cpu_buf is not None:
            self.current_cpu_buf.promise.get()

        # Wait for buffer to be ready. If this was already waited .get() is a no-op.
        if self.mapped:
            assert self.current_cpu_buf is not None
            cpu_buf = self.current_cpu_buf
            self.current_cpu_buf = None

            self.cpu_pool.use_frame(frame.index, property_frame_index)
            # NOTE: only works for buffers for now, which is why we get a type error here.
            self.current = GpuBufferView(cpu_buf.buf, 0, cpu_buf.used_size)  # type: ignore
        else:
            assert self.current_gpu_res is not None
            gpu_res = self.current_gpu_res
            self.current_gpu_res = None

            if self.current_cpu_buf is not None:
                cpu_buf = self.current_cpu_buf
                self.current_cpu_buf = None

                if self.upload_method == UploadMethod.GRAPHICS_QUEUE:
                    self._cmd_upload(frame.cmd, cpu_buf, gpu_res.resource)
                else:
                    assert self.upload_method == UploadMethod.TRANSFER_QUEUE
                    assert frame.transfer_cmd is not None
                    self._cmd_upload(frame.transfer_cmd, cpu_buf, gpu_res.resource)

            self.current = gpu_res.resource

    def prefetch(self) -> None:
        if self.property.upload.cpu_prefetch_count > 0:
            # Issue prefetches
            def cpu_prefetch_cleanup(k: int, buf: CpuBuffer) -> bool:
                return buf.promise.is_set()

            def cpu_prefetch(k: int, buf: CpuBuffer) -> None:
                self.renderer.thread_pool.submit(buf.promise, self._load_async_into, k, buf)  # type: ignore

            # TODO: can likely improve prefetch logic, and should probably allow
            # this to be hooked / configured somehow
            #
            # maybe we should just compute the state of next (maybe few) frames
            # (assuming constant dt, playback state) once globally and call prefetch with this.
            prefetch_start = self.property.current_frame_index + 1
            prefetch_end = prefetch_start + self.property.upload.cpu_prefetch_count
            prefetch_range = [
                self.property.animation.map_frame_index(i, 0, self.property.num_frames)
                for i in range(prefetch_start, prefetch_end)
            ]
            self.cpu_pool.prefetch(prefetch_range, cpu_prefetch_cleanup, cpu_prefetch)

        if self.upload_method == UploadMethod.TRANSFER_QUEUE and self.property.upload.gpu_prefetch_count > 0:
            assert self.gpu_pool is not None

            def gpu_prefetch_cleanup(k: int, gpu_res: GpuResource[V]) -> bool:
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

            def gpu_prefetch(k: int, gpu_res: GpuResource[V]) -> None:
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
                    # Sync: Flush previous upload on gfx queue and transition images.
                    self._cmd_before_barrier(state.commands, gpu_res.resource)
                    self._cmd_upload(state.commands, cpu_next, gpu_res.resource)

                info = gpu_res.use(PipelineStageFlags.TRANSFER)

                # Prefetch commands are submitted one at a time because we have no
                # knowledge or how far in the future they will be used. If they were
                # batched you would need to wait for all of them to complete before
                # you can use any of the resources. These submissions are potentially
                # expensive, but we assume that applications will make very sparse
                # use of prefetching, only for really large resources of which hopeufully
                # there are only one or a few.
                self.renderer.ctx.transfer_queue.submit(
                    state.commands,
                    wait_semaphores=[(info.sem, info.wait_value, info.wait_stage)],
                    signal_semaphores=[(info.sem, info.signal_value, info.signal_stage)],
                )
                state.prefetch_done_value = info.signal_value
                gpu_res.state = GpuResourceState.PREFETCH

            # TODO: fix, same as above
            prefetch_start = self.property.current_frame_index + 1
            prefetch_end = prefetch_start + self.property.upload.gpu_prefetch_count
            available_range = []
            for i in range(prefetch_start, prefetch_end):
                frame_index = self.property.animation.map_frame_index(i, 0, self.property.num_frames)
                if self.cpu_pool.is_available(frame_index):
                    available_range.append(frame_index)

            self.gpu_pool.prefetch(
                available_range,
                gpu_prefetch_cleanup,
                gpu_prefetch,
            )

    # Edit API
    def update_frame(self, index: int) -> None:
        self._invalidate_frame(index)

    def update_frames(self, indices: List[int]) -> None:
        self._invalidate_all_frames()

    def update_frame_range(self, start: int, stop: int) -> None:
        self._invalidate_all_frames()

    def append_frame(self) -> None:
        # Nothing to do here
        pass

    def append_frames(self, count: int) -> None:
        # Nothing to do here
        pass

    def pop_frame(self) -> None:
        self._invalidate_frame(self.property.num_frames - 1)

    def pop_frames(self, count: int) -> None:
        self._invalidate_all_frames()

    def insert_frame(self, index: int) -> None:
        self._invalidate_all_frames()

    def insert_frames(self, index: int, count: int) -> None:
        self._invalidate_all_frames()

    def remove_frame(self, index: int) -> None:
        self._invalidate_all_frames()

    def remove_frames(self, indices: List[int]) -> None:
        self._invalidate_all_frames()

    def remove_frame_range(self, start: int, stop: int) -> None:
        self._invalidate_all_frames()

    # Private API
    def _load_async(self, i: int, thread_index: int) -> "PropertyItem":
        return self.property.get_frame_by_index(i, thread_index)

    def _load_async_into(self, i: int, buf: CpuBuffer, thread_index: int) -> None:
        buf.used_size = self.property.get_frame_by_index_into(i, buf.buf.data, thread_index)

    def _invalidate_frame(self, index: int) -> None:
        self.cpu_pool.increment_generation(index)
        if self.gpu_pool is not None:
            self.gpu_pool.increment_generation(index)

    def _invalidate_all_frames(self) -> None:
        self.cpu_pool.increment_all_generations()
        if self.gpu_pool is not None:
            self.gpu_pool.increment_all_generations()

    def _create_cpu_buffer(self, name: str) -> Buffer:
        raise NotImplementedError

    def _create_gpu_resource(self, name: str) -> V:
        raise NotImplementedError

    def _append_mip_generation_request(self, frame: RendererFrame, resource: V) -> None:
        pass

    def _append_barriers_for_upload_on_graphics_queue(self, frame: RendererFrame, resource: V) -> None:
        raise NotImplementedError

    def _append_barriers_for_upload_on_transfer_queue(self, frame: RendererFrame, resource: V) -> None:
        raise NotImplementedError

    def _append_acquire_barrier_on_graphics_queue(self, frame: RendererFrame, resource: V) -> None:
        raise NotImplementedError

    def _append_release_barrier_on_transfer_queue(self, frame: RendererFrame, resource: V) -> None:
        raise NotImplementedError

    def _cmd_upload(self, cmd: CommandBuffer, cpu_buf: CpuBuffer, resource: V) -> None:
        raise NotImplementedError

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: V) -> None:
        pass


class GpuBufferStreamingProperty(GpuStreamingProperty[Buffer, GpuBufferView]):
    def __init__(
        self,
        renderer: "Renderer",
        property: "BufferProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | BufferUsageFlags.TRANSFER_DST
        self.max_frame_size = property.max_size

        super().__init__(
            renderer,
            renderer.buffer_upload_method,
            property,
            property.gpu_stage,
            name,
        )

    # Private API
    def _create_cpu_buffer(self, name: str) -> Buffer:
        cpu_alloc_type = (
            AllocType.DEVICE_MAPPED if self.upload_method == UploadMethod.MAPPED_PREFER_DEVICE else AllocType.HOST
        )
        if self.mapped:
            cpu_buffer_usage_flags = self.usage_flags
        else:
            cpu_buffer_usage_flags = BufferUsageFlags.TRANSFER_SRC
        return Buffer(self.renderer.ctx, self.max_frame_size, cpu_buffer_usage_flags, cpu_alloc_type, name)

    def _create_gpu_resource(self, name: str) -> GpuBufferView:
        return GpuBufferView(
            Buffer(
                self.renderer.ctx,
                self.max_frame_size,
                self.usage_flags,
                AllocType.DEVICE,
                name=name,
            ),
            0,
            self.max_frame_size,
        )

    def _append_barriers_for_upload_on_graphics_queue(self, frame: RendererFrame, resource: GpuBufferView) -> None:
        frame.upload_property_pipeline_stages |= self.pipeline_stage_flags

    def _append_barriers_for_upload_on_transfer_queue(self, frame: RendererFrame, resource: GpuBufferView) -> None:
        pass

    def _append_acquire_barrier_on_graphics_queue(self, frame: RendererFrame, resource: GpuBufferView) -> None:
        frame.upload_after_buffer_barriers.append(
            BufferBarrier(
                resource.buffer,
                PipelineStageFlags.NONE,
                AccessFlags.NONE,
                self.pipeline_stage_flags,
                AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                self.renderer.ctx.transfer_queue_family_index,
                self.renderer.ctx.graphics_queue_family_index,
            )
        )

    def _append_release_barrier_on_transfer_queue(self, frame: RendererFrame, resource: GpuBufferView) -> None:
        frame.transfer_upload_after_buffer_barriers.append(
            BufferBarrier(
                resource.buffer,
                PipelineStageFlags.COPY,
                AccessFlags.TRANSFER_WRITE,
                PipelineStageFlags.NONE,
                AccessFlags.NONE,
                self.renderer.ctx.transfer_queue_family_index,
                self.renderer.ctx.graphics_queue_family_index,
            )
        )

    def _cmd_upload(
        self,
        cmd: CommandBuffer,
        cpu_buf: CpuBuffer,
        resource: GpuBufferView,
    ) -> None:
        cmd.copy_buffer_range(cpu_buf.buf, resource.buffer, cpu_buf.used_size, resource.offset)
        resource.size = cpu_buf.used_size

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: GpuBufferView) -> None:
        # Needs to be all when going from application usage on graphics queue to copy on transfer queue for prefetching
        cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.TRANSFER_DST)


class GpuImageStreamingProperty(GpuStreamingProperty[Image, GpuImageView]):
    def __init__(
        self,
        renderer: "Renderer",
        property: "ImageProperty",
        name: str,
    ):
        self.usage_flags = property.gpu_usage | ImageUsageFlags.TRANSFER_DST

        self.width = property.width
        self.height = property.height
        self.format = property.format
        self.layout = property.gpu_layout
        self.srgb = property.gpu_srgb
        self.mips = property.gpu_mips

        self.spd_pipeline_instance = None
        if self.mips:
            mip_levels = max(self.width.bit_length(), self.height.bit_length())
            self.spd_pipeline_instance = renderer.spd_pipeline.alloc_instance(renderer, False) if self.mips else None
            self.spd_pipeline_instance.set_image_extents(self.width, self.height, mip_levels)

        self.pitch, self.rows, _ = get_image_pitch_rows_and_texel_size(self.width, self.height, self.format)

        super().__init__(
            renderer,
            renderer.image_upload_method,
            property,
            property.gpu_stage,
            name,
        )

    # Private API
    def _create_cpu_buffer(self, name: str) -> Buffer:
        return Buffer(self.renderer.ctx, self.pitch * self.rows, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST, name)

    def _create_gpu_resource(self, name: str) -> GpuImageView:
        return _create_image_view(
            self.renderer.ctx, self.width, self.height, self.format, self.usage_flags, self.srgb, self.mips, name
        )

    def _append_mip_generation_request(self, frame: RendererFrame, resource: GpuImageView) -> None:
        if not self.mips:
            return

        mip_generation_filter = MipGenerationFilter.AVERAGE_SRGB if self.srgb else MipGenerationFilter.AVERAGE

        assert self.spd_pipeline_instance is not None
        assert resource.mip_level_0_view is not None

        descriptor_set = self.spd_pipeline_instance.get_and_write_current_and_advance(
            resource.mip_level_0_view, resource.mip_views
        )
        frame.mip_generation_requests.setdefault(mip_generation_filter, []).append(
            MipGenerationRequest(
                resource.image,
                self.spd_pipeline_instance.constants.tobytes(),
                descriptor_set,
                self.spd_pipeline_instance.groups_x,
                self.spd_pipeline_instance.groups_y,
                1,
                ImageBarrier(
                    resource.image,
                    ImageLayout.GENERAL,
                    self.layout,
                    PipelineStageFlags.COMPUTE_SHADER,
                    AccessFlags.SHADER_SAMPLED_READ
                    | AccessFlags.SHADER_STORAGE_READ
                    | AccessFlags.SHADER_STORAGE_WRITE,
                    self.pipeline_stage_flags,
                    AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                ),
            )
        )

    def _append_barriers_for_upload_on_graphics_queue(self, frame: RendererFrame, resource: GpuImageView) -> None:
        frame.upload_before_image_barriers.append(
            ImageBarrier(
                resource.image,
                ImageLayout.UNDEFINED,
                ImageLayout.TRANSFER_DST_OPTIMAL,
                self.pipeline_stage_flags,
                AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                PipelineStageFlags.COPY,
                AccessFlags.TRANSFER_WRITE,
            )
        )

        if self.mips:
            frame.upload_after_image_barriers.append(
                ImageBarrier(
                    resource.image,
                    ImageLayout.TRANSFER_DST_OPTIMAL,
                    ImageLayout.GENERAL,
                    PipelineStageFlags.COPY,
                    AccessFlags.TRANSFER_WRITE,
                    PipelineStageFlags.COMPUTE_SHADER,
                    AccessFlags.SHADER_SAMPLED_READ
                    | AccessFlags.SHADER_STORAGE_READ
                    | AccessFlags.SHADER_STORAGE_WRITE,
                )
            )
        else:
            frame.upload_after_image_barriers.append(
                ImageBarrier(
                    resource.image,
                    ImageLayout.TRANSFER_DST_OPTIMAL,
                    self.layout,
                    PipelineStageFlags.COPY,
                    AccessFlags.TRANSFER_WRITE,
                    self.pipeline_stage_flags,
                    AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                )
            )

    def _append_barriers_for_upload_on_transfer_queue(self, frame: RendererFrame, resource: GpuImageView) -> None:
        frame.transfer_upload_before_image_barriers.append(
            ImageBarrier(
                resource.image,
                ImageLayout.UNDEFINED,
                ImageLayout.TRANSFER_DST_OPTIMAL,
                # Needs to be all when going from application usage on graphics queue to copy on transfer queue.
                PipelineStageFlags.ALL_COMMANDS,
                AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                PipelineStageFlags.COPY,
                AccessFlags.TRANSFER_WRITE,
            )
        )

    def _append_acquire_barrier_on_graphics_queue(self, frame: RendererFrame, resource: GpuImageView) -> None:
        if self.mips:
            frame.upload_after_image_barriers.append(
                ImageBarrier(
                    resource.image,
                    ImageLayout.TRANSFER_DST_OPTIMAL,
                    ImageLayout.GENERAL,
                    PipelineStageFlags.COPY,
                    AccessFlags.TRANSFER_WRITE,
                    PipelineStageFlags.COMPUTE_SHADER,
                    AccessFlags.SHADER_SAMPLED_READ
                    | AccessFlags.SHADER_STORAGE_READ
                    | AccessFlags.SHADER_STORAGE_WRITE,
                    self.renderer.ctx.transfer_queue_family_index,
                    self.renderer.ctx.graphics_queue_family_index,
                )
            )
        else:
            frame.upload_after_image_barriers.append(
                ImageBarrier(
                    resource.image,
                    # Layout transition should be specified twice the same for acquire/release pairs
                    # according to the vulkan synchronization examples.
                    ImageLayout.TRANSFER_DST_OPTIMAL,
                    self.layout,
                    # The spec says that these should be ignored, but validation seems to
                    # look at these when a layout transition is involved.
                    PipelineStageFlags.COPY,
                    AccessFlags.TRANSFER_WRITE,
                    self.pipeline_stage_flags,
                    AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                    self.renderer.ctx.transfer_queue_family_index,
                    self.renderer.ctx.graphics_queue_family_index,
                )
            )

    def _append_release_barrier_on_transfer_queue(self, frame: RendererFrame, resource: GpuImageView) -> None:
        frame.transfer_upload_after_image_barriers.append(
            ImageBarrier(
                resource.image,
                # Layout transition should be specified twice the same for acquire/release pairs
                # according to the vulkan synchronization examples.
                ImageLayout.TRANSFER_DST_OPTIMAL,
                ImageLayout.GENERAL if self.mips else self.layout,
                PipelineStageFlags.TRANSFER,
                AccessFlags.TRANSFER_WRITE,
                # The spec says that these should be ignored, but validation seems to
                # look at these when a layout transition is involved.
                PipelineStageFlags.COPY,
                AccessFlags.TRANSFER_WRITE,
                self.renderer.ctx.transfer_queue_family_index,
                self.renderer.ctx.graphics_queue_family_index,
            )
        )

    def _cmd_upload(self, cmd: CommandBuffer, buffer: CpuBuffer, view: GpuImageView) -> None:
        cmd.copy_buffer_to_image(buffer.buf, view.image)

    def _cmd_before_barrier(self, cmd: CommandBuffer, resource: GpuImageView) -> None:
        cmd.image_barrier(
            resource.image,
            ImageLayout.TRANSFER_DST_OPTIMAL,
            # Needs to be all when going from application usage on graphics queue to copy on transfer queue for prefetching
            MemoryUsage.ALL,
            MemoryUsage.TRANSFER_DST,
            undefined=True,
        )
