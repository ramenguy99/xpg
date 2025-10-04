# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import List, Optional, Tuple, Union

import numpy as np
from pyglm.glm import inverse, mat3x3, mat4x3, transpose
from pyxpg import (
    Attachment,
    BufferUsageFlags,
    CullMode,
    Depth,
    DescriptorSet,
    DeviceFeatures,
    Format,
    FrontFace,
    GraphicsPipeline,
    InputAssembly,
    MemoryUsage,
    PipelineStage,
    PipelineStageFlags,
    PrimitiveTopology,
    PushConstantsRange,
    Rasterization,
    Shader,
    Stage,
    VertexAttribute,
    VertexBinding,
    VertexInputRate,
)

from .materials import ColorMaterial, DiffuseMaterial, Material
from .property import BufferProperty, ImageProperty, as_image_property
from .renderer import Renderer
from .renderer_frame import RendererFrame
from .scene import Object, Object3D
from .utils.gpu import cull_mode_opposite_face


class Lines(Object3D):
    def __init__(
        self,
        lines: Union[BufferProperty, np.ndarray],
        colors: Union[BufferProperty, np.ndarray],
        line_width: Union[BufferProperty, float] = 1.0,
        is_strip: bool = False,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(name, translation, rotation, scale)
        self.is_strip = is_strip
        self.lines = self.add_buffer_property(lines, np.float32, (-1, 3), name="lines")
        self.colors = self.add_buffer_property(colors, np.uint32, (-1,), name="colors")
        self.line_width = self.add_buffer_property(line_width, np.float32, name="line_width")

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
            samples=r.msaa_samples,
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, True, True, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def update_transform(self, parent: Optional[Object]) -> None:
        super().update_transform(parent)
        self.constants["transform"] = mat4x3(self.current_transform_matrix)

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.lines_buffer.get_current(),
                self.colors_buffer.get_current(),
            ],
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.set_line_width(
            self.line_width.get_current().item() if r.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0
        )

        frame.cmd.draw(self.lines.get_current().shape[0])


class Mesh(Object3D):
    def __init__(
        self,
        positions: BufferProperty,
        indices: Optional[BufferProperty] = None,
        normals: Optional[BufferProperty] = None,
        uvs: Optional[BufferProperty] = None,
        primitive_topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST,
        cull_mode: CullMode = CullMode.NONE,
        front_face: FrontFace = FrontFace.COUNTER_CLOCKWISE,
        material: Optional[Material] = None,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        self.primitive_topology = primitive_topology
        self.cull_mode = cull_mode
        self.front_face = front_face
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
                "normal_matrix": (np.dtype((np.float32, (3, 4))), 48),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(
            name, translation, rotation, scale, material if material is not None else DiffuseMaterial((0.5, 0.5, 0.5))
        )

        # Add properties
        self.positions = self.add_buffer_property(positions, np.float32, (-1, 3), name="positions")
        self.normals = (
            self.add_buffer_property(normals, np.float32, (-1, 3), name="normals") if normals is not None else None
        )
        self.uvs = self.add_buffer_property(uvs, np.float32, (-1, 2), name="uvs") if uvs is not None else None
        self.indices = (
            self.add_buffer_property(indices, np.uint32, (-1,), name="indices") if indices is not None else None
        )

    def create(self, r: Renderer) -> None:
        self.positions_buffer = r.add_gpu_buffer_property(
            self.positions,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-positions",
        )
        self.normals_buffer = (
            r.add_gpu_buffer_property(
                self.normals,
                BufferUsageFlags.VERTEX,
                MemoryUsage.VERTEX_INPUT,
                PipelineStageFlags.VERTEX_INPUT,
                name=f"{self.name}-normals",
            )
            if self.normals is not None
            else None
        )
        self.uvs_buffer = (
            r.add_gpu_buffer_property(
                self.uvs,
                BufferUsageFlags.VERTEX,
                MemoryUsage.VERTEX_INPUT,
                PipelineStageFlags.VERTEX_INPUT,
                name=f"{self.name}-uvs",
            )
            if self.uvs is not None
            else None
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

        assert self.material is not None
        defines: List[Tuple[str, str]] = []
        defines.extend(self.material.shader_defines)

        vertex_bindings = [
            VertexBinding(0, 12, VertexInputRate.VERTEX),
        ]
        vertex_attributes = [
            VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
        ]

        vertex_binding_index = 1
        if self.normals is not None:
            defines.append(("VERTEX_NORMALS", str(vertex_binding_index)))
            vertex_bindings.append(VertexBinding(vertex_binding_index, 12, VertexInputRate.VERTEX))
            vertex_attributes.append(
                VertexAttribute(vertex_binding_index, vertex_binding_index, Format.R32G32B32_SFLOAT)
            )
            vertex_binding_index += 1

        vertex_binding_index = 1
        if self.uvs is not None:
            defines.append(("VERTEX_UVS", str(vertex_binding_index)))
            vertex_bindings.append(VertexBinding(vertex_binding_index, 8, VertexInputRate.VERTEX))
            vertex_attributes.append(VertexAttribute(vertex_binding_index, vertex_binding_index, Format.R32G32_SFLOAT))
            vertex_binding_index += 1

        vert = r.get_builtin_shader("3d/mesh.slang", "vertex_main", defines=defines)
        frag = r.get_builtin_shader("3d/mesh.slang", "pixel_main", defines=defines)
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=vertex_bindings,
            vertex_attributes=vertex_attributes,
            rasterization=Rasterization(cull_mode=self.cull_mode, front_face=self.front_face),
            input_assembly=InputAssembly(self.primitive_topology),
            samples=r.msaa_samples,
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, True, True, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
                self.material.descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

        depth_vert = r.get_builtin_shader("3d/mesh_depth.slang", "vertex_main")
        self.depth_pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, depth_vert.code), Stage.VERTEX),
            ],
            vertex_bindings=[
                VertexBinding(0, 12, VertexInputRate.VERTEX),
            ],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
            ],
            rasterization=Rasterization(cull_mode=cull_mode_opposite_face(self.cull_mode), front_face=self.front_face),
            input_assembly=InputAssembly(self.primitive_topology),
            attachments=[],
            depth=Depth(r.shadowmap_format, True, True, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_depth_descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype["transform"].itemsize)],
        )

    def update_transform(self, parent: Optional[Object]) -> None:
        super().update_transform(parent)
        self.constants["transform"] = mat4x3(self.current_transform_matrix)
        self.constants["normal_matrix"][:, :, :3] = transpose(inverse(mat3x3(self.current_transform_matrix)))

    def render_depth(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        index_buffer = self.indices_buffer.get_current() if self.indices_buffer is not None else None

        frame.cmd.bind_graphics_pipeline(
            self.depth_pipeline,
            vertex_buffers=[
                self.positions_buffer.get_current(),
            ],
            index_buffer=index_buffer,
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants["transform"].tobytes(),
        )

        if self.indices is not None:
            frame.cmd.draw_indexed(self.indices.get_current().shape[0])
        else:
            frame.cmd.draw(self.positions.get_current().shape[0])

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        assert self.material is not None

        index_buffer = self.indices_buffer.get_current() if self.indices_buffer is not None else None
        vertex_buffers = [
            self.positions_buffer.get_current(),
        ]
        if self.normals_buffer is not None:
            vertex_buffers.append(self.normals_buffer.get_current())
        if self.uvs_buffer is not None:
            vertex_buffers.append(self.uvs_buffer.get_current())

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=vertex_buffers,
            index_buffer=index_buffer,
            descriptor_sets=[
                scene_descriptor_set,
                self.material.descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )

        if self.indices is not None:
            frame.cmd.draw_indexed(self.indices.get_current().shape[0])
        else:
            frame.cmd.draw(self.positions.get_current().shape[0])


class Image(Mesh):
    def __init__(
        self,
        image: ImageProperty,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        material = ColorMaterial(as_image_property(image))
        positions = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
            ]
        ).reshape((4, 3))
        uvs = np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
            ]
        ).reshape((4, 2))
        super().__init__(
            positions,  # type: ignore
            uvs=uvs,  # type: ignore
            primitive_topology=PrimitiveTopology.TRIANGLE_STRIP,
            material=material,
            name=name,
            translation=translation,
            rotation=rotation,
            scale=scale,
        )