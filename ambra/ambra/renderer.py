from pyxpg import *

from .config import RendererConfig
from .scene import Object
from .shaders import compile
from .utils.profile import profile
from .utils.gpu import UploadableBuffer, UniformPool
from .utils.ring_buffer import RingBuffer
from .viewport import Viewport
from .utils.threadpool import ThreadPool

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from functools import cache
from enum import Enum, auto

import numpy as np

SHADERS_PATH = Path(__file__).parent.joinpath("shaders")

@dataclass
class RendererFrame:
    cmd: CommandBuffer
    image: Image
    index: int
    total_index: int
    viewport: Tuple[float, float, float, float]
    scissors: Tuple[float, float, float, float]
    descriptor_set: DescriptorSet


class UploadMethod(Enum):
    CPU_BUF = auto()
    BAR = auto()
    GFX = auto()
    TRANSFER_QUEUE = auto()


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
        # TODO: make configurable
        self.thread_pool = ThreadPool(4)
        self.total_frame_index = 0

        if ctx.device_properties.device_type == PhysicalDeviceType.INTEGRATED_GPU or ctx.device_properties.device_type == PhysicalDeviceType.CPU:
            self.upload_method = UploadMethod.CPU_BUF
        elif False: # TODO: Check if has resizable bar and if is configured to use it
            pass
        elif ctx.has_transfer_queue:
            self.upload_method = UploadMethod.TRANSFER_QUEUE
        else:
            self.upload_method = UploadMethod.GFX

    @cache
    def get_builtin_shader(self, name: str, entry: str) -> slang.Shader:
        path = SHADERS_PATH.joinpath(name)
        return compile(path, entry)

    def render(self, viewport: Viewport, gui: Gui):
        with self.window.frame() as frame:
            with frame.command_buffer as cmd:
                viewport_rect = (viewport.rect.x, viewport.rect.y + viewport.rect.height, viewport.rect.width, viewport.rect.y - viewport.rect.height)
                rect = (viewport.rect.x, viewport.rect.y, viewport.rect.width, viewport.rect.height)

                set: DescriptorSet = self.descriptor_sets.get_current_and_advance()
                buf: UploadableBuffer = self.uniform_buffers.get_current_and_advance()

                self.constants["projection"] = viewport.camera.projection()
                self.constants["view"] = viewport.camera.view()
                self.constants["camera_position"] = -viewport.camera.camera_from_world.translation
                buf.upload(cmd, MemoryUsage.ANY_SHADER_UNIFORM_READ, self.constants.view(np.uint8))

                cmd.use_image(frame.image, ImageUsage.TRANSFER_DST)
                cmd.clear_color_image(frame.image, self.background_color)
                cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)

                f = RendererFrame(cmd, frame.image, self.total_frame_index % self.window.num_frames, self.total_frame_index, viewport_rect, rect, set)
                viewport.scene.render(self, f)
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

        self.uniform_pool.advance()
        self.total_frame_index += 1
