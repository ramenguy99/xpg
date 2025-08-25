from typing import Optional, Union

import numpy as np
from pyxpg import (
    Attachment,
    BufferUsageFlags,
    CullMode,
    Depth,
    DepthAttachment,
    DescriptorSet,
    DescriptorSetEntry,
    DescriptorType,
    DeviceFeatures,
    Filter,
    Format,
    FrontFace,
    GraphicsPipeline,
    ImageLayout,
    ImageUsageFlags,
    InputAssembly,
    MemoryUsage,
    PipelineStage,
    PipelineStageFlags,
    PrimitiveTopology,
    Rasterization,
    RenderingAttachment,
    Sampler,
    SamplerAddressMode,
    Shader,
    Stage,
    VertexAttribute,
    VertexBinding,
    VertexInputRate,
)

from .renderer import Renderer
from .renderer_frame import RendererFrame
from .scene import Object3D, Property
from .utils.ring_buffer import RingBuffer


class Lines(Object3D):
    def __init__(
        self,
        lines: Union[Property, np.ndarray],
        colors: Union[Property, np.ndarray],
        line_width: Union[Property, float] = 1.0,
        is_strip: bool = False,
        name: Optional[str] = None,
        translation: Optional[Property] = None,
        rotation: Optional[Property] = None,
        scale: Optional[Property] = None,
    ):
        super().__init__(name, translation, rotation, scale)
        self.is_strip = is_strip
        self.lines = self.add_property(lines, np.float32, (-1, 3), name="lines")
        self.colors = self.add_property(colors, np.uint32, (-1,), name="colors")
        self.line_width = self.add_property(line_width, np.float32, name="line_width")

    def create(self, r: Renderer) -> None:
        self.lines_buffer = r.add_gpu_buffer_property(
            self.lines,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-lines-3d",
        )
        self.colors_buffer = r.add_gpu_buffer_property(
            self.colors,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-colors-3d",
        )

        constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (4, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), constants_dtype)

        vert = r.get_builtin_shader("3d/basic.slang", "vertex_main")
        frag = r.get_builtin_shader("3d/basic.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=[
                VertexBinding(0, 12, VertexInputRate.VERTEX),
                VertexBinding(1, 4, VertexInputRate.VERTEX),
            ],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
                VertexAttribute(1, 1, Format.R32_UINT),
            ],
            rasterization=Rasterization(dynamic_line_width=True),
            input_assembly=InputAssembly(
                PrimitiveTopology.LINE_STRIP if self.is_strip else PrimitiveTopology.LINE_LIST
            ),
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, True, True, r.depth_compare_op),
            descriptor_sets=[
                r.descriptor_sets.get_current(),
                r.uniform_pool.descriptor_set,
            ],
        )

    def render(self, r: Renderer, frame: RendererFrame) -> None:
        self.constants["transform"] = self.current_transform_matrix
        constants_alloc = r.uniform_pool.alloc(self.constants.itemsize)
        constants_alloc.upload(frame.cmd, self.constants.view(np.uint8))

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.lines_buffer.get_current(),
                self.colors_buffer.get_current(),
            ],
            descriptor_sets=[
                frame.descriptor_set,
                constants_alloc.descriptor_set,
            ],
            dynamic_offsets=[constants_alloc.offset],
        )
        frame.cmd.set_line_width(
            self.line_width.get_current().item() if r.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0
        )

        with frame.cmd.rendering(
            frame.rect,
            color_attachments=[RenderingAttachment(frame.image)],
            depth=DepthAttachment(r.depth_buffer),
        ):
            frame.cmd.draw(self.lines.get_current().shape[0])


class Image(Object3D):
    def __init__(
        self,
        image: Property,
        format: Format,
        name: Optional[str] = None,
        translation: Optional[Property] = None,
        rotation: Optional[Property] = None,
        scale: Optional[Property] = None,
    ):
        super().__init__(name, translation, rotation, scale)
        self.image = self.add_property(image, shape=(-1, -1, -1), name="image")
        self.format = format

    def create(self, r: Renderer) -> None:
        self.images = r.add_gpu_image_property(
            self.image,
            self.format,
            ImageUsageFlags.SAMPLED,
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            MemoryUsage.SHADER_READ_ONLY,
            PipelineStageFlags.FRAGMENT_SHADER,
            name=f"{self.name}-image",
        )
        self.sampler = Sampler(
            r.ctx,
            min_filter=Filter.LINEAR,
            mag_filter=Filter.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_EDGE,
            v=SamplerAddressMode.CLAMP_TO_EDGE,
        )

        constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (4, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), constants_dtype)
        self.descriptor_sets = RingBuffer(
            [
                DescriptorSet(
                    r.ctx,
                    [
                        DescriptorSetEntry(1, DescriptorType.SAMPLER),
                        DescriptorSetEntry(1, DescriptorType.SAMPLED_IMAGE),
                    ],
                )
                for _ in range(r.window.num_frames)
            ]
        )
        for set in self.descriptor_sets.items:
            set.write_sampler(self.sampler, 0)

        vert = r.get_builtin_shader("3d/basic_texture.slang", "vertex_main")
        frag = r.get_builtin_shader("3d/basic_texture.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            input_assembly=InputAssembly(PrimitiveTopology.TRIANGLE_STRIP),
            attachments=[Attachment(format=r.output_format)],
            descriptor_sets=[
                r.descriptor_sets.get_current(),
                r.uniform_pool.descriptor_set,
                self.descriptor_sets.get_current(),
            ],
        )

    def render(self, r: Renderer, frame: RendererFrame) -> None:
        self.constants["transform"] = self.current_transform_matrix
        constants_alloc = r.uniform_pool.alloc(self.constants.itemsize)
        constants_alloc.upload(frame.cmd, self.constants.view(np.uint8))

        descriptor_set = self.descriptor_sets.get_current_and_advance()
        descriptor_set.write_image(
            self.images.get_current(),
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            DescriptorType.SAMPLED_IMAGE,
            1,
        )

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            descriptor_sets=[
                frame.descriptor_set,
                constants_alloc.descriptor_set,
                descriptor_set,
            ],
            dynamic_offsets=[constants_alloc.offset],
        )

        with frame.cmd.rendering(frame.rect, color_attachments=[RenderingAttachment(frame.image)]):
            frame.cmd.draw(4)


class Mesh(Object3D):
    def __init__(
        self,
        positions: Property,
        #  normals: Property[np.ndarray],
        indices: Optional[Property] = None,
        primitive_topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST,
        cull_mode: CullMode = CullMode.NONE,
        front_face: FrontFace = FrontFace.COUNTER_CLOCKWISE,
        name: Optional[str] = None,
        translation: Optional[Property] = None,
        rotation: Optional[Property] = None,
        scale: Optional[Property] = None,
    ):
        super().__init__(name, translation, rotation, scale)
        self.positions = self.add_property(positions, np.float32, (-1, 3), name="positions")
        # self.normals: Property = self.add_property(normals, np.float32, (-1, 3), name="normals")
        self.indices = self.add_property(indices, np.uint32, (-1,), name="indices") if indices is not None else None
        self.primitive_topology = primitive_topology
        self.cull_mode = cull_mode
        self.front_face = front_face

    def create(self, r: Renderer) -> None:
        self.positions_buffer = r.add_gpu_buffer_property(
            self.positions,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-positions",
        )
        # self.normals_buffer = r.add_gpu_buffer_property(self.normals, BufferUsageFlags.VERTEX, MemoryUsage.VERTEX_INPUT, PipelineStageFlags.VERTEX_INPUT, name=f"{self.name}-normals")
        self.indices_buffer = (
            r.add_gpu_buffer_property(
                self.indices,
                BufferUsageFlags.INDEX,
                MemoryUsage.VERTEX_INPUT,
                PipelineStageFlags.VERTEX_INPUT,
                name=f"{self.name}-indices",
            )
            if self.indices is not None
            else None
        )

        constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (4, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), constants_dtype)

        vert = r.get_builtin_shader("3d/mesh.slang", "vertex_main")
        frag = r.get_builtin_shader("3d/mesh.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=[
                VertexBinding(0, 12, VertexInputRate.VERTEX),
                # VertexBinding(1, 12, VertexInputRate.VERTEX),
            ],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
                # VertexAttribute(1, 1, Format.R32G32B32_SFLOAT),
            ],
            rasterization=Rasterization(cull_mode=self.cull_mode, front_face=self.front_face),
            input_assembly=InputAssembly(self.primitive_topology),
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, True, True, r.depth_compare_op),
            descriptor_sets=[
                r.descriptor_sets.get_current(),
                r.uniform_pool.descriptor_set,
            ],
        )

    def render(self, r: Renderer, frame: RendererFrame) -> None:
        self.constants["transform"] = self.current_transform_matrix
        constants_alloc = r.uniform_pool.alloc(self.constants.itemsize)
        constants_alloc.upload(frame.cmd, self.constants.view(np.uint8))

        index_buffer = self.indices_buffer.get_current() if self.indices_buffer is not None else None
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.positions_buffer.get_current(),
                # self.normals_buffer.get_current(),
            ],
            index_buffer=index_buffer,
            descriptor_sets=[
                frame.descriptor_set,
                constants_alloc.descriptor_set,
            ],
            dynamic_offsets=[constants_alloc.offset],
        )

        with frame.cmd.rendering(
            frame.rect,
            color_attachments=[RenderingAttachment(frame.image)],
            depth=DepthAttachment(r.depth_buffer),
        ):
            if self.indices is not None:
                frame.cmd.draw_indexed(self.indices.get_current().shape[0])
            else:
                frame.cmd.draw(self.positions.get_current().shape[0])


class AnimatedMesh(Object3D):
    def __init__(
        self,
        positions: Property,
        normals: Property,
        tangents: Property,
        uvs: Property,
        joint_indices: Property,
        weights: Property,
        joints: Property,
        texture: Property,
        texture_format: Format,
        indices: Optional[Property] = None,
        primitive_topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST,
        cull_mode: CullMode = CullMode.NONE,
        front_face: FrontFace = FrontFace.COUNTER_CLOCKWISE,
        name: Optional[str] = None,
        translation: Optional[Property] = None,
        rotation: Optional[Property] = None,
        scale: Optional[Property] = None,
    ):
        super().__init__(name, translation, rotation, scale)

        self.positions = self.add_property(positions, np.float32, (-1, 3), name="positions")
        self.normals = self.add_property(normals, np.float32, (-1, 3), name="normals")
        self.tangents = self.add_property(tangents, np.float32, (-1, 3), name="tangents")
        self.uvs = self.add_property(uvs, np.float32, (-1, 2), name="uvs")
        self.joint_indices = self.add_property(joint_indices, np.int16, (-1, 4), name="joint_indices")
        self.weights = self.add_property(weights, np.float32, (-1, 4), name="weights")

        self.joints = self.add_property(joints, np.float32, (-1, 4, 4), name="joints")

        self.texture = self.add_property(texture, shape=(-1, -1, -1), name="base_color_texture")
        self.texture_format = texture_format

        self.indices = self.add_property(indices, np.uint32, (-1,), name="indices") if indices is not None else None

        self.primitive_topology = primitive_topology
        self.cull_mode = cull_mode
        self.front_face = front_face

    def create(self, r: Renderer) -> None:
        self.positions_buffer = r.add_gpu_buffer_property(
            self.positions,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-positions",
        )
        self.normals_buffer = r.add_gpu_buffer_property(
            self.normals,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-normals",
        )
        self.tangents_buffer = r.add_gpu_buffer_property(
            self.tangents,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-tangents",
        )
        self.uvs_buffer = r.add_gpu_buffer_property(
            self.uvs,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-uvs",
        )
        self.joint_indices_buffer = r.add_gpu_buffer_property(
            self.joint_indices,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-joint_indices",
        )
        self.weights_buffer = r.add_gpu_buffer_property(
            self.weights,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-weights",
        )

        self.joints_buffer = r.add_gpu_buffer_property(
            self.joints,
            BufferUsageFlags.STORAGE,
            MemoryUsage.SHADER_READ_ONLY,
            PipelineStageFlags.VERTEX_SHADER,
            name=f"{self.name}-joints",
        )

        self.indices_buffer = (
            r.add_gpu_buffer_property(
                self.indices,
                BufferUsageFlags.INDEX,
                MemoryUsage.VERTEX_INPUT,
                PipelineStageFlags.VERTEX_INPUT,
                name=f"{self.name}-indices",
            )
            if self.indices is not None
            else None
        )

        self.texture = r.add_gpu_image_property(
            self.texture,
            self.texture_format,
            ImageUsageFlags.SAMPLED,
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            MemoryUsage.SHADER_READ_ONLY,
            PipelineStageFlags.FRAGMENT_SHADER,
            name=f"{self.name}-image",
        )
        self.sampler = Sampler(
            r.ctx,
            min_filter=Filter.LINEAR,
            mag_filter=Filter.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_EDGE,
            v=SamplerAddressMode.CLAMP_TO_EDGE,
        )

        constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (4, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), constants_dtype)

        self.descriptor_sets = RingBuffer(
            [
                DescriptorSet(
                    r.ctx,
                    [
                        DescriptorSetEntry(1, DescriptorType.SAMPLER),
                        DescriptorSetEntry(1, DescriptorType.SAMPLED_IMAGE),
                        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
                    ],
                )
                for _ in range(r.window.num_frames)
            ]
        )
        for set in self.descriptor_sets.items:
            set.write_sampler(self.sampler, 0)

        vert = r.get_builtin_shader("3d/mesh_animated.slang", "vertex_main")
        frag = r.get_builtin_shader("3d/mesh_animated.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=[
                VertexBinding(0, 12, VertexInputRate.VERTEX),
                VertexBinding(1, 12, VertexInputRate.VERTEX),
                VertexBinding(2, 12, VertexInputRate.VERTEX),
                VertexBinding(3, 8, VertexInputRate.VERTEX),
                VertexBinding(4, 8, VertexInputRate.VERTEX),
                VertexBinding(5, 16, VertexInputRate.VERTEX),
            ],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
                VertexAttribute(1, 1, Format.R32G32B32_SFLOAT),
                VertexAttribute(2, 2, Format.R32G32B32_SFLOAT),
                VertexAttribute(3, 3, Format.R32G32_SFLOAT),
                VertexAttribute(4, 4, Format.R16G16B16A16_UINT),
                VertexAttribute(5, 5, Format.R32G32B32A32_SFLOAT),
            ],
            rasterization=Rasterization(cull_mode=self.cull_mode, front_face=self.front_face),
            input_assembly=InputAssembly(self.primitive_topology),
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, True, True, r.depth_compare_op),
            descriptor_sets=[
                r.descriptor_sets.get_current(),
                r.uniform_pool.descriptor_set,
                self.descriptor_sets.get_current(),
            ],
        )

    def render(self, r: Renderer, frame: RendererFrame) -> None:
        self.constants["transform"] = self.current_transform_matrix
        constants_alloc = r.uniform_pool.alloc(self.constants.itemsize)
        constants_alloc.upload(frame.cmd, self.constants.view(np.uint8))

        descriptor_set = self.descriptor_sets.get_current_and_advance()
        descriptor_set.write_image(
            self.texture.get_current(),
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            DescriptorType.SAMPLED_IMAGE,
            1,
        )
        descriptor_set.write_buffer(
            self.joints_buffer.get_current(),
            DescriptorType.STORAGE_BUFFER,
            2,
        )

        index_buffer = self.indices_buffer.get_current() if self.indices_buffer is not None else None
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.positions_buffer.get_current(),
                self.normals_buffer.get_current(),
                self.tangents_buffer.get_current(),
                self.uvs_buffer.get_current(),
                self.joint_indices_buffer.get_current(),
                self.weights_buffer.get_current(),
            ],
            index_buffer=index_buffer,
            descriptor_sets=[
                frame.descriptor_set,
                constants_alloc.descriptor_set,
                descriptor_set,
            ],
            dynamic_offsets=[constants_alloc.offset],
        )

        with frame.cmd.rendering(
            frame.rect,
            color_attachments=[RenderingAttachment(frame.image)],
            depth=DepthAttachment(r.depth_buffer),
        ):
            if self.indices is not None:
                frame.cmd.draw_indexed(self.indices.get_current().shape[0])
            else:
                frame.cmd.draw(self.positions.get_current().shape[0])
