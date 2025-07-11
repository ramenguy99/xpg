from pyxpg import *
from . import Object
from ..renderer import Renderer, RendererFrame
from ..utils.uploadable_buffer import UploadableBuffer


from typing import Optional
import numpy as np

_counter = 0

class Line(Object):
    def __init__(self, positions: np.ndarray, colors: np.ndarray, name: Optional[str] = None):
        global _counter

        if name == None:
            name = f"Line {_counter}"
            _counter += 1
        super().__init__(name)

        self.positions = positions
        self.colors = colors

    def create(self, r: Renderer):
        def view_bytes(a: np.ndarray, dtype: np.dtype) -> memoryview:
            return a.astype(dtype, order="C", copy=False).reshape((-1,), copy=False).view(np.uint8).data

        self.positions_buf = UploadableBuffer.from_data(r.ctx, view_bytes(self.positions, np.float32), BufferUsageFlags.VERTEX, name=f"{self.name}-positions") 
        self.colors_buf = UploadableBuffer.from_data(r.ctx, view_bytes(self.colors, np.uint32), BufferUsageFlags.VERTEX, f"{self.name}-colors")

        vert = r.get_builtin_shader("basic.slang", "vertex_main")
        frag = r.get_builtin_shader("basic.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages = [
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings = [
                VertexBinding(0, 8, VertexInputRate.VERTEX),
                VertexBinding(1, 4, VertexInputRate.VERTEX),
            ],
            vertex_attributes = [
                VertexAttribute(0, 0, Format.R32G32_SFLOAT),
                VertexAttribute(1, 1, Format.R32_UINT),
            ],
            input_assembly = InputAssembly(PrimitiveTopology.LINE_LIST),
            attachments = [
                Attachment(format=r.output_format)
            ]
        )

    def render(self, r: Renderer, frame: RendererFrame):
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[self.positions_buf, self.colors_buf],
            viewport=frame.viewport,
            scissors=frame.scissors,
        )
        frame.cmd.draw(self.positions.size)

    def destroy(self, r: Renderer):
        self.positions_buf.destroy()
        self.colors_buf.destroy()