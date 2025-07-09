from pyxpg import *
from .config import RendererConfig
from .scene import Scene, Object

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

SHADERS_PATH = Path(__file__).parent.joinpath("shaders")

@dataclass
class RendererFrame:
    cmd: CommandBuffer
    viewport: Tuple[float, float, float, float]
    scissors: Tuple[float, float, float, float]
    
class Renderer:
    def __init__(self, ctx: Context, window: Window, config: RendererConfig):
        self.ctx = ctx
        self.window = window

        self.output_format = window.swapchain_format

        # Config
        self.background_color = config.background_color

    def get_builtin_shader(self, name: str, entry: str) -> slang.Shader:
        path = SHADERS_PATH.joinpath(name)
        return slang.compile(str(path), entry)
    
    def render(self, scene: Scene, gui: Gui):
        with self.window.frame() as frame:
            with frame.command_buffer as cmd:
                cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)

                viewport = [0, 0, self.window.fb_width, self.window.fb_height]
                with cmd.rendering(viewport,
                    color_attachments=[
                        RenderingAttachment(
                            frame.image,
                            load_op=LoadOp.CLEAR,
                            store_op=StoreOp.STORE,
                            clear=self.background_color,
                        ),
                    ]):

                    f = RendererFrame(cmd, viewport, viewport)
                    def render_obj(obj: Object):
                        obj.render(self, f)
                        for c in obj.children:
                            render_obj(c)

                    render_obj(scene)

                    gui.render(cmd)

                cmd.use_image(frame.image, ImageUsage.PRESENT)
    