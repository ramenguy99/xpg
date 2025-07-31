from pyxpg import *

from .config import RendererConfig, UploadMethod
from .scene import Property
from .shaders import compile
from .utils.gpu import UploadableBuffer, UniformPool
from .utils.gpu_property import GpuBufferProperty, GpuImageProperty
from .utils.ring_buffer import RingBuffer
from .viewport import Viewport
from .utils.threadpool import ThreadPool
from .renderer_frame import RendererFrame

from pathlib import Path
from typing import List, Union
from functools import cache

import numpy as np

SHADERS_PATH = Path(__file__).parent.joinpath("shaders")

class Renderer:
    def __init__(self, ctx: Context, window: Window, config: RendererConfig):
        self.ctx = ctx
        self.window = window

        self.output_format = window.swapchain_format

        # Config
        self.background_color = config.background_color

        # Scene descriptors
        self.descriptor_sets: RingBuffer[DescriptorSet] = RingBuffer(window.num_frames, DescriptorSet, ctx, [
            DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
        ])

        constants_dtype = np.dtype ({
            "view":            (np.dtype((np.float32, (4, 4))), 0),
            "projection":      (np.dtype((np.float32, (4, 4))), 64),
            "camera_position": (np.dtype((np.float32, (3,))), 128),
        })

        self.constants = np.zeros((1,), constants_dtype)
        self.uniform_buffers: RingBuffer[UploadableBuffer] = RingBuffer(window.num_frames, UploadableBuffer, ctx, 64 * 2 + 12, BufferUsageFlags.UNIFORM)

        for set, buf in zip(self.descriptor_sets.items, self.uniform_buffers.items):
            set.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        self.uniform_pool = UniformPool(ctx, window.num_frames, config.uniform_pool_block_size)
        self.depth_buffer: Image = None
        self.depth_format = Format.D32_SFLOAT
        self.depth_clear_value = 1.0
        self.depth_compare_op = CompareOp.LESS

        # TODO: make configurable
        self.thread_pool = ThreadPool(4)
        self.total_frame_index = 0

        if config.force_upload_method is not None:
            self.upload_method = config.force_upload_method
            if self.upload_method == UploadMethod.TRANSFER_QUEUE:
                if not ctx.has_transfer_queue:
                    raise RuntimeError("Transfer queue not available on picked device")
                if not ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES:
                    raise RuntimeError("Transfer queue upload requires timeline semaphores which are not available on picked device")
        else:
            if ctx.device_properties.device_type == PhysicalDeviceType.INTEGRATED_GPU or ctx.device_properties.device_type == PhysicalDeviceType.CPU:
                self.upload_method = UploadMethod.CPU_BUF
            elif False: # TODO: Check if has resizable bar and if is configured to use it
                self.upload_method = UploadMethod.BAR
            elif config.use_transfer_queue_if_available and ctx.has_transfer_queue and ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES:
                self.upload_method = UploadMethod.TRANSFER_QUEUE
            else:
                self.upload_method = UploadMethod.GFX

        self.gpu_properties: List[Union[GpuBufferProperty, GpuImageProperty]] = []

        self.resize(window.fb_width, window.fb_height)

    def resize(self, width: int, height: int):
        if self.depth_buffer is not None:
            self.depth_buffer.destroy()
        self.depth_buffer = Image(self.ctx, width, height, self.depth_format, ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags.TRANSFER_DST, AllocType.DEVICE_DEDICATED, name="depth")

    def add_gpu_buffer_property(self, property: Property[np.ndarray], usage_flags: BufferUsageFlags, name: str):
        prop = GpuBufferProperty(self.ctx, self.window.num_frames, self.upload_method, self.thread_pool, property, usage_flags, name)
        self.gpu_properties.append(prop)
        return prop

    def add_gpu_image_property(self, property: Property[np.ndarray], usage_flags: ImageUsageFlags, usage: ImageUsage, name: str):
        prop = GpuImageProperty(self.ctx, self.window.num_frames, self.upload_method, self.thread_pool, property, usage_flags, usage, name)
        self.gpu_properties.append(prop)
        return prop

    @cache
    def get_builtin_shader(self, name: str, entry: str) -> slang.Shader:
        path = SHADERS_PATH.joinpath(name)
        return compile(path, entry)

    def render(self, viewport: Viewport, gui: Gui):
        frame = self.window.begin_frame()
        with frame.command_buffer as cmd:
            viewport_rect = (viewport.rect.x, viewport.rect.y + viewport.rect.height, viewport.rect.width, viewport.rect.y - viewport.rect.height)
            rect = (viewport.rect.x, viewport.rect.y, viewport.rect.width, viewport.rect.height)

            set: DescriptorSet = self.descriptor_sets.get_current_and_advance()
            buf: UploadableBuffer = self.uniform_buffers.get_current_and_advance()

            self.constants["projection"] = viewport.camera.projection()
            self.constants["view"] = viewport.camera.view()
            self.constants["camera_position"] = -viewport.camera.camera_from_world.translation
            buf.upload(cmd, MemoryUsage.ANY_SHADER_UNIFORM_READ, self.constants.view(np.uint8))

            cmd.set_viewport(viewport_rect)
            cmd.set_scissors(rect)

            cmd.use_image(frame.image, ImageUsage.TRANSFER_DST)
            cmd.clear_color_image(frame.image, self.background_color)
            cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)

            cmd.use_image(self.depth_buffer, ImageUsage.TRANSFER_DST, aspect_mask=ImageAspectFlags.DEPTH)
            cmd.clear_depth_stencil_image(self.depth_buffer, depth=self.depth_clear_value)
            cmd.use_image(self.depth_buffer, ImageUsage.DEPTH_STENCIL_ATTACHMENT, aspect_mask=ImageAspectFlags.DEPTH)

            if frame.transfer_command_buffer:
                frame.transfer_command_buffer.begin()

            f = RendererFrame(
                cmd,
                frame.image,
                self.total_frame_index % self.window.num_frames,
                self.total_frame_index,
                viewport_rect,
                rect,
                set,
                [],
                frame.transfer_command_buffer,
                [],
            )


            # Create new objects, if any
            viewport.scene.create_if_needed(self)

            # Update properties
            for p in self.gpu_properties:
                p.load(f)

            for p in self.gpu_properties:
                p.upload(f)

            # Render scene
            viewport.scene.render(self, f)

            # Render GUI
            with cmd.rendering(rect,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.LOAD,
                        store_op=StoreOp.STORE,
                        clear=self.background_color,
                    ),
                ]):
                gui.render(cmd)
            cmd.use_image(frame.image, ImageUsage.PRESENT)

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

        self.uniform_pool.advance()
        self.total_frame_index += 1
