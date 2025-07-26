from pyxpg import *

from .config import RendererConfig
from .scene import Object
from .utils.profile import profile
from .utils.gpu import UploadableBuffer, UniformPool
from .utils.ring_buffer import RingBuffer
from .viewport import Viewport

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from functools import cache

import numpy as np

SHADERS_PATH = Path(__file__).parent.joinpath("shaders")

@dataclass
class RendererFrame:
    cmd: CommandBuffer
    viewport: Tuple[float, float, float, float]
    scissors: Tuple[float, float, float, float]
    descriptor_set: DescriptorSet


class Renderer:
    def __init__(self, ctx: Context, window: Window, config: RendererConfig):
        self.ctx = ctx
        self.window = window

        self.output_format = window.swapchain_format

        # Config
        self.background_color = config.background_color
        self.prefer_preupload = config.prefer_preupload

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

    @cache
    def get_builtin_shader(self, name: str, entry: str) -> slang.Shader:
        path = SHADERS_PATH.joinpath(name)
        with profile(f"compiling shader: {path}"):
            return slang.compile(str(path), entry)

    def render(self, viewport: Viewport, gui: Gui):
        with self.window.frame() as frame:
            with frame.command_buffer as cmd:
                cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)

                viewport_rect = (viewport.rect.x, viewport.rect.y + viewport.rect.height, viewport.rect.width, viewport.rect.y - viewport.rect.height)
                rect = (viewport.rect.x, viewport.rect.y, viewport.rect.width, viewport.rect.height)
                with cmd.rendering(rect,
                    color_attachments=[
                        RenderingAttachment(
                            frame.image,
                            load_op=LoadOp.CLEAR,
                            store_op=StoreOp.STORE,
                            clear=self.background_color,
                        ),
                    ]):

                    set: DescriptorSet = self.descriptor_sets.get_current_and_advance()
                    buf: UploadableBuffer = self.uniform_buffers.get_current_and_advance()

                    self.constants["projection"] = viewport.camera.projection()
                    self.constants["view"] = viewport.camera.view()
                    self.constants["camera_position"] = -viewport.camera.camera_from_world.translation

                    buf.upload(cmd, MemoryUsage.ANY_SHADER_UNIFORM_READ, self.constants.view(np.uint8))

                    f = RendererFrame(cmd, viewport_rect, rect, set)
                    viewport.scene.render(self, f)

                    gui.render(cmd)

                cmd.use_image(frame.image, ImageUsage.PRESENT)
        self.uniform_pool.advance()
