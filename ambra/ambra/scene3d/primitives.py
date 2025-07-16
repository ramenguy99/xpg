from pyxpg import *
from . import Object
from ..renderer import Renderer, RendererFrame
from ..utils.uploadable_buffer import UploadableBuffer


from typing import Optional
import numpy as np

_counter = 0

class Line(Object):
    def __init__(self, positions: np.ndarray, colors: np.ndarray, line_width: float = 1.0, name: Optional[str] = None):
        global _counter

        if name == None:
            name = f"Line {_counter}"
            _counter += 1
        super().__init__(name)

        self.positions = positions
        self.colors = colors
        self.line_width = line_width
    
    def create(self, r: Renderer):
        def view_bytes(a: np.ndarray, dtype: np.dtype) -> memoryview:
            return a.astype(dtype, order="C", copy=False).reshape((-1,), copy=False).view(np.uint8).data

        self.positions_buf = UploadableBuffer.from_data(r.ctx, view_bytes(self.positions, np.float32), BufferUsageFlags.VERTEX, name=f"{self.name}-positions") 
        self.colors_buf = UploadableBuffer.from_data(r.ctx, view_bytes(self.colors, np.uint32), BufferUsageFlags.VERTEX, f"{self.name}-colors")

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
            input_assembly = InputAssembly(PrimitiveTopology.LINE_LIST),
            attachments = [
                Attachment(format=r.output_format)
            ],
            descriptor_sets = [ r.descriptor_sets.get_current() ],
        )

    def render(self, r: Renderer, frame: RendererFrame):
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[self.positions_buf, self.colors_buf],
            descriptor_sets=[frame.descriptor_set],
            viewport=frame.viewport,
            scissors=frame.scissors,
        )
        frame.cmd.set_line_width(self.line_width if r.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0)
        frame.cmd.draw(self.positions.shape[0])

    def destroy(self, r: Renderer):
        self.positions_buf.destroy()
        self.colors_buf.destroy()