import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from pyxpg import (
    AllocType,
    BufferUsageFlags,
    CompareOp,
    Context,
    DepthAttachment,
    DescriptorType,
    DescriptorSetBinding,
    DeviceFeatures,
    Format,
    Gui,
    Image,
    ImageAspectFlags,
    ImageLayout,
    ImageUsageFlags,
    LoadOp,
    MemoryUsage,
    PhysicalDeviceType,
    PipelineStageFlags,
    RenderingAttachment,
    StoreOp,
    SwapchainOutOfDateError,
    Window,
    slang,
)

from .config import RendererConfig, UploadMethod
from .gpu_property import GpuBufferProperty, GpuImageProperty
from .property import BufferProperty, ImageProperty
from .renderer_frame import RendererFrame
from .shaders import compile
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .utils.gpu import (
    BufferUploadInfo,
    BulkUploader,
    ImageUploadInfo,
    UniformPool,
    UploadableBuffer,
)
from .utils.ring_buffer import RingBuffer
from .utils.threadpool import ThreadPool
from .viewport import Viewport
from .scene import LIGHT_TYPES_INFO, LightTypes

SHADERS_PATH = Path(__file__).parent.joinpath("shaders")

MAX_WORKERS = 16
DEFAULT_WORKERS = 4
if sys.version_info >= (3, 13):

    def get_cpu_count() -> Optional[int]:
        return os.process_cpu_count()
else:

    def get_cpu_count() -> Optional[int]:
        return os.cpu_count()


class Renderer:
    def __init__(self, ctx: Context, window: Window, config: RendererConfig):
        self.ctx = ctx
        self.window = window

        self.output_format = window.swapchain_format
        self.shadowmap_format = Format.D32_SFLOAT

        # Config
        self.background_color = config.background_color

        # Scene descriptors
        self.scene_descriptor_set_layout, self.scene_descriptor_pool, self.scene_descriptor_sets = (
            create_descriptor_layout_pool_and_sets_ringbuffer(
                ctx,
                [
                    DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),
                    # (len(LIGHT_TYPES_INFO), DescriptorType.STORAGE_BUFFER),
                    # (config.max_shadowmaps, DescriptorType.COMBINED_IMAGE_SAMPLER),
                ],
                window.num_frames,
                name="scene-descriptors",
            )
        )

        constants_dtype = np.dtype(
            {
                "view": (np.dtype((np.float32, (4, 4))), 0),
                "projection": (np.dtype((np.float32, (4, 4))), 64),
                "camera_position": (np.dtype((np.float32, (3,))), 128),
                "num_lights": (np.dtype((np.uint32, len(LIGHT_TYPES_INFO))), 140),
            }
        )  # type: ignore

        self.constants = np.zeros((1,), constants_dtype)
        self.uniform_buffers = RingBuffer(
            [UploadableBuffer(ctx, constants_dtype.itemsize, BufferUsageFlags.UNIFORM) for _ in range(window.num_frames)]
        )

        self.num_lights = [0] * len(LIGHT_TYPES_INFO)
        self.light_buffers = RingBuffer(
            [
                [UploadableBuffer(ctx, info.size * config.max_lights, BufferUsageFlags.UNIFORM) for info in LIGHT_TYPES_INFO]
                for _ in range(window.num_frames)
            ]
        )

        for set, buf, light_bufs in zip(self.scene_descriptor_sets, self.uniform_buffers, self.light_buffers):
            set.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0, 0)
            # for i, light_buf in enumerate(light_bufs):
            #     set.write_buffer(light_buf, DescriptorType.STORAGE_BUFFER, 1, i)
            # TODO: write shadowmap descs (?)

        self.uniform_pool = UniformPool(ctx, window.num_frames, config.uniform_pool_block_size)

        if config.thread_pool_workers is not None:
            self.num_workers = config.thread_pool_workers
        else:
            num_cpus = get_cpu_count()
            if num_cpus is not None:
                self.num_workers = min(num_cpus, MAX_WORKERS)
            else:
                self.num_workers = DEFAULT_WORKERS

        self.thread_pool: ThreadPool = ThreadPool(self.num_workers)
        self.total_frame_index = 0

        # Upload method for buffers
        if config.force_buffer_upload_method is not None:
            self.buffer_upload_method = config.force_buffer_upload_method
            if self.buffer_upload_method == UploadMethod.TRANSFER_QUEUE:
                if not ctx.has_transfer_queue:
                    raise RuntimeError("Transfer queue not available on picked device")
                if not ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES:
                    raise RuntimeError(
                        "Transfer queue upload requires timeline semaphores which are not available on picked device"
                    )
        else:
            # TODO: we could also check if BAR is available and large enough
            # and add a DEVICE_PREFER_MAPPED option to take advantage of it.
            if (
                ctx.device_properties.device_type == PhysicalDeviceType.INTEGRATED_GPU
                or ctx.device_properties.device_type == PhysicalDeviceType.CPU
            ):
                self.buffer_upload_method = UploadMethod.MAPPED_PREFER_DEVICE
            elif (
                config.use_transfer_queue_if_available
                and ctx.has_transfer_queue
                and ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES
            ):
                self.buffer_upload_method = UploadMethod.TRANSFER_QUEUE
            else:
                self.buffer_upload_method = UploadMethod.GRAPHICS_QUEUE

        # Upload method for images
        if config.force_image_upload_method is not None:
            if (
                config.force_image_upload_method == UploadMethod.MAPPED_PREFER_DEVICE
                or config.force_image_upload_method == UploadMethod.MAPPED_PREFER_HOST
            ):
                raise RuntimeError(
                    f"Upload method for images must be {UploadMethod.GRAPHICS_QUEUE} or {UploadMethod.TRANSFER_QUEUE}. Got {config.force_image_upload_method} in config.force_image_upload_method."
                )
            self.image_upload_method = config.force_image_upload_method
            if self.image_upload_method == UploadMethod.TRANSFER_QUEUE:
                if not ctx.has_transfer_queue:
                    raise RuntimeError("Transfer queue not available on picked device")
                if not ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES:
                    raise RuntimeError(
                        "Transfer queue upload requires timeline semaphores which are not available on picked device"
                    )
        else:
            if (
                config.use_transfer_queue_if_available
                and ctx.has_transfer_queue
                and ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES
            ):
                self.image_upload_method = UploadMethod.TRANSFER_QUEUE
            else:
                self.image_upload_method = UploadMethod.GRAPHICS_QUEUE

        self.gpu_properties: List[Union[GpuBufferProperty, GpuImageProperty]] = []

        # Allocate buffers for bulk upload
        self.bulk_uploader = BulkUploader(self.ctx, config.upload_buffer_size, config.upload_buffer_count)
        self.bulk_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]] = []

        # Will be populated in self.resize(), and will never be None after that
        self.depth_buffer: Image = None  # type: ignore
        self.depth_format = Format.D32_SFLOAT
        self.depth_clear_value = 1.0
        self.depth_compare_op = CompareOp.LESS
        self.resize(window.fb_width, window.fb_height)

    def resize(self, width: int, height: int) -> None:
        if self.depth_buffer is not None:
            self.depth_buffer.destroy()
        self.depth_buffer = Image(
            self.ctx,
            width,
            height,
            self.depth_format,
            ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE_DEDICATED,
            name="depth",
        )

    def add_gpu_buffer_property(
        self,
        property: BufferProperty,
        usage_flags: BufferUsageFlags,
        memory_usage: MemoryUsage,
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ) -> GpuBufferProperty:
        prop = GpuBufferProperty(
            self.ctx,
            self.window.num_frames,
            self.buffer_upload_method,
            self.thread_pool,
            self.bulk_upload_list,
            property,
            usage_flags,
            memory_usage,
            pipeline_stage_flags,
            name,
        )
        self.gpu_properties.append(prop)
        return prop

    def add_gpu_image_property(
        self,
        property: ImageProperty,
        usage_flags: ImageUsageFlags,
        layout: ImageLayout,
        memory_usage: MemoryUsage,
        pipeline_stage_flags: PipelineStageFlags,
        name: str,
    ) -> GpuImageProperty:
        prop = GpuImageProperty(
            self.ctx,
            self.window.num_frames,
            self.image_upload_method,
            self.thread_pool,
            self.bulk_upload_list,
            property,
            usage_flags,
            layout,
            memory_usage,
            pipeline_stage_flags,
            name,
        )
        self.gpu_properties.append(prop)
        return prop

    def get_builtin_shader(self, name: str, entry: str) -> slang.Shader:
        path = SHADERS_PATH.joinpath(name)
        return compile(path, entry)

    def render(self, viewport: Viewport, gui: Gui) -> None:
        # Create new objects, if any
        viewport.scene.create_if_needed(self)

        # Flush synchronous upload buffers after creating new objects
        if len(self.bulk_upload_list) > 0:
            self.bulk_uploader.bulk_upload(self.bulk_upload_list)
            self.bulk_upload_list.clear()

        # Render frame
        try:
            frame = self.window.begin_frame()
        except SwapchainOutOfDateError:
            return

        with frame.command_buffer as cmd:
            viewport_rect = (
                viewport.rect.x,
                viewport.rect.y + viewport.rect.height,
                viewport.rect.width,
                viewport.rect.y - viewport.rect.height,
            )
            rect = (
                viewport.rect.x,
                viewport.rect.y,
                viewport.rect.width,
                viewport.rect.height,
            )

            set = self.scene_descriptor_sets.get_current_and_advance()
            buf = self.uniform_buffers.get_current_and_advance()

            self.constants["projection"] = viewport.camera.projection()
            self.constants["view"] = viewport.camera.view()
            self.constants["camera_position"] = -viewport.camera.camera_from_world.translation
            buf.upload(
                cmd,
                MemoryUsage.ANY_SHADER_UNIFORM,
                self.constants.view(np.uint8),
            )

            if frame.transfer_command_buffer:
                frame.transfer_command_buffer.begin()

            f = RendererFrame(
                cmd,
                frame.image,
                self.total_frame_index % self.window.num_frames,
                self.total_frame_index,
                viewport_rect,
                rect,
                [],
                frame.transfer_command_buffer,
                [],
            )

            # Update properties
            for p in self.gpu_properties:
                p.load(f)

            for p in self.gpu_properties:
                p.upload(f)

            # Upload per-object data
            viewport.scene.upload(self, f)

            # Render shadows
            viewport.scene.render_shadowmaps(self, f)

            # Render scene
            cmd.image_barrier(
                frame.image,
                ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                MemoryUsage.ALL,
                MemoryUsage.COLOR_ATTACHMENT,
                undefined=True,
            )
            cmd.image_barrier(
                self.depth_buffer,
                ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                MemoryUsage.DEPTH_STENCIL_ATTACHMENT,
                MemoryUsage.DEPTH_STENCIL_ATTACHMENT,
                aspect_mask=ImageAspectFlags.DEPTH,
                undefined=True,
            )
            cmd.set_viewport(viewport_rect)
            cmd.set_scissors(rect)
            with f.cmd.rendering(
                f.rect,
                color_attachments=[
                    RenderingAttachment(frame.image, LoadOp.CLEAR, StoreOp.STORE, self.background_color)
                ],
                depth=DepthAttachment(self.depth_buffer, LoadOp.CLEAR, StoreOp.STORE, self.depth_clear_value),
            ):
                viewport.scene.render(self, f, set)

            # Render GUI
            with cmd.rendering(
                rect,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.LOAD,
                        store_op=StoreOp.STORE,
                        clear=self.background_color,
                    ),
                ],
            ):
                gui.render(cmd)
            cmd.image_barrier(
                frame.image,
                ImageLayout.PRESENT_SRC,
                MemoryUsage.COLOR_ATTACHMENT,
                MemoryUsage.PRESENT,
            )

            if frame.transfer_command_buffer:
                frame.transfer_command_buffer.end()
                if f.copy_semaphores:
                    self.ctx.transfer_queue.submit(
                        frame.transfer_command_buffer,
                        wait_semaphores=[(s.sem, s.wait_stage) for s in f.copy_semaphores],
                        wait_timeline_values=[s.wait_value for s in f.copy_semaphores],
                        signal_semaphores=[s.sem for s in f.copy_semaphores],
                        signal_timeline_values=[s.signal_value for s in f.copy_semaphores],
                    )

        self.window.end_frame(
            frame,
            additional_wait_semaphores=[(s.sem, s.wait_stage) for s in f.additional_semaphores],
            additional_wait_timeline_values=[s.wait_value for s in f.additional_semaphores],
            additional_signal_semaphores=[s.sem for s in f.additional_semaphores],
            additional_signal_timeline_values=[s.signal_value for s in f.additional_semaphores],
        )

        for p in self.gpu_properties:
            p.prefetch()

        self.uniform_pool.advance()
        self.total_frame_index += 1
