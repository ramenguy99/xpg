from pyxpg import *
from .scene import Property, Object
from .renderer import Renderer, RendererFrame
from .utils.gpu_property import GpuBufferProperty

from typing import Optional
import numpy as np

class Line(Object):
    def __init__(self, lines: Property[np.ndarray], colors: Property[np.ndarray], line_width: Property[float] = 1.0, is_strip = False, name: Optional[str] = None):
        super().__init__(name)
        self.is_strip = is_strip
        self.lines: Property[np.ndarray] = self.add_property(lines, np.float32, (-1, 3,), name="lines")
        self.colors: Property[np.ndarray] = self.add_property(colors, np.uint32, (-1,), name="colors")
        self.line_width: Property[float] = self.add_property(line_width, np.float32, name="line_width")


    def create(self, r: Renderer):
        self.lines_buffer = GpuBufferProperty(self, r, self.lines, BufferUsageFlags.VERTEX, name=f"{self.name}-lines-3d")
        self.colors_buffer = GpuBufferProperty(self, r, self.colors, BufferUsageFlags.VERTEX, name=f"{self.name}-lines-3d")

        vert = r.get_builtin_shader("3d/basic.slang", "vertex_main")
        frag = r.get_builtin_shader("3d/basic.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages = [
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings = [
                VertexBinding(0, 12, VertexInputRate.VERTEX),
                VertexBinding(1, 4, VertexInputRate.VERTEX),
            ],
            vertex_attributes = [
                VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
                VertexAttribute(1, 1, Format.R32_UINT),
            ],
            rasterization=Rasterization(dynamic_line_width=True),
            input_assembly = InputAssembly(PrimitiveTopology.LINE_STRIP if self.is_strip else PrimitiveTopology.LINE_LIST),
            attachments = [
                Attachment(format=r.output_format)
            ],
            descriptor_sets = [ r.descriptor_sets.get_current() ],
        )
    
    def render(self, r: Renderer, frame: RendererFrame):
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            # vertex_buffers=[self.positions_buf, self.colors_buf],
            vertex_buffers=[
                self.lines_buffer.get_current(),
                self.colors_buffer.get_current(),
            ],
            descriptor_sets=[frame.descriptor_set],
            viewport=frame.viewport,
            scissors=frame.scissors,
        )
        frame.cmd.set_line_width(self.line_width.get_current() if r.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0)
        frame.cmd.draw(self.lines.get_current().shape[0])