from pyxpg import *
from .scene import Property, Object2D
from .renderer import Renderer, RendererFrame
from .utils.gpu_property import GpuBufferProperty, GpuImageProperty
from .utils.ring_buffer import RingBuffer

from typing import Optional
import numpy as np

class Lines(Object2D):
    def __init__(self,
                 lines: Property[np.ndarray],
                 colors: Property[np.ndarray],
                 line_width: Property[float] = 1.0,
                 is_strip = False,
                 name: Optional[str] = None,
                 translation: Optional[Property[np.ndarray]] = None,
                 rotation: Optional[Property[float]] = None,
                 scale: Optional[Property[np.ndarray]] = None
                ):
        super().__init__(name, translation, rotation, scale)
        self.is_strip = is_strip
        self.lines: Property[np.ndarray] = self.add_property(lines, np.float32, (-1, 2), name="lines")
        self.colors: Property[np.ndarray] = self.add_property(colors, np.uint32, (-1,), name="colors")
        self.line_width: Property[float] = self.add_property(line_width, np.float32, name="line_width")

    def create(self, r: Renderer):
        self.lines_buffer = GpuBufferProperty(self, r, self.lines, BufferUsageFlags.VERTEX, name=f"{self.name}-lines-2d")
        self.colors_buffer = GpuBufferProperty(self, r, self.colors, BufferUsageFlags.VERTEX, name=f"{self.name}-lines-2d")

        constants_dtype = np.dtype ({
            "transform": (np.dtype((np.float32, (3, 4))), 0),
        })
        self.constants = np.zeros((1,), constants_dtype)

        vert = r.get_builtin_shader("2d/basic.slang", "vertex_main")
        frag = r.get_builtin_shader("2d/basic.slang", "pixel_main")

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
            rasterization = Rasterization(dynamic_line_width=True),
            input_assembly = InputAssembly(PrimitiveTopology.LINE_STRIP if self.is_strip else PrimitiveTopology.LINE_LIST),
            attachments = [
                Attachment(format=r.output_format)
            ],
            descriptor_sets = [ r.descriptor_sets.get_current(), r.uniform_pool.descriptor_set ],
        )

    def render(self, r: Renderer, frame: RendererFrame):
        self.constants["transform"][0, :3, :3] = self.current_transform_matrix
        constants_alloc = r.uniform_pool.alloc(self.constants.itemsize)
        constants_alloc.upload(frame.cmd, self.constants.view(np.uint8))

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers = [
                self.lines_buffer.get_current(),
                self.colors_buffer.get_current(),
            ],
            descriptor_sets = [ frame.descriptor_set, constants_alloc.descriptor_set ],
            dynamic_offsets = [ constants_alloc.offset ],
            viewport = frame.viewport,
            scissors = frame.scissors,
        )
        frame.cmd.set_line_width(self.line_width.get_current() if r.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0)
        frame.cmd.draw(self.lines.get_current().shape[0])

class Image(Object2D):
    def __init__(self,
                 image: Property[np.ndarray],
                 name: Optional[str] = None,
                 translation: Optional[Property[np.ndarray]] = None,
                 rotation: Optional[Property[float]] = None,
                 scale: Optional[Property[np.ndarray]] = None
                ):
        super().__init__(name, translation, rotation, scale)
        self.image: Property[np.ndarray] = self.add_property(image, shape=(-1, -1, -1), name="image")

    def create(self, r: Renderer):
        self.images = GpuImageProperty(self, r, self.image, ImageUsageFlags.SAMPLED, ImageUsage.SHADER_READ_ONLY, name=f"{self.name}-image")
        self.sampler = Sampler(r.ctx, min_filter=Filter.LINEAR, mag_filter=Filter.LINEAR, u=SamplerAddressMode.CLAMP_TO_EDGE, v=SamplerAddressMode.CLAMP_TO_EDGE)

        constants_dtype = np.dtype ({
            "transform": (np.dtype((np.float32, (3, 4))), 0),
        })
        self.constants = np.zeros((1,), constants_dtype)
        self.descriptor_sets: RingBuffer[DescriptorSet] = RingBuffer(r.window.num_frames, DescriptorSet, r.ctx, [
            DescriptorSetEntry(1, DescriptorType.SAMPLER),
            DescriptorSetEntry(1, DescriptorType.SAMPLED_IMAGE),
        ])
        for set in self.descriptor_sets.items:
            set.write_sampler(self.sampler, 0)

        vert = r.get_builtin_shader("2d/basic_texture.slang", "vertex_main")
        frag = r.get_builtin_shader("2d/basic_texture.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages = [
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            input_assembly = InputAssembly(PrimitiveTopology.TRIANGLE_STRIP),
            attachments = [
                Attachment(format=r.output_format)
            ],
            descriptor_sets = [ r.descriptor_sets.get_current(), r.uniform_pool.descriptor_set, self.descriptor_sets.get_current() ],
        )

    def render(self, r: Renderer, frame: RendererFrame):
        self.constants["transform"][0, :3, :3] = self.current_transform_matrix
        constants_alloc = r.uniform_pool.alloc(self.constants.itemsize)
        constants_alloc.upload(frame.cmd, self.constants.view(np.uint8))

        descriptor_set = self.descriptor_sets.get_current_and_advance()
        descriptor_set.write_image(self.images.get_current(), ImageUsage.SHADER_READ_ONLY, DescriptorType.SAMPLED_IMAGE, 1)

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            descriptor_sets = [ frame.descriptor_set, constants_alloc.descriptor_set, descriptor_set ],
            dynamic_offsets = [ constants_alloc.offset ],
            viewport = frame.viewport,
            scissors = frame.scissors,
        )
        frame.cmd.draw(4)
