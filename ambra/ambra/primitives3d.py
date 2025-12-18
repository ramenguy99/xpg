# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from pyglm.glm import inverse, mat3, mat4x3, rotate, transpose, vec3, vec4
from pyxpg import (
    AllocType,
    Attachment,
    BlendFactor,
    BlendOp,
    Buffer,
    BufferUsageFlags,
    ComputePipeline,
    CullMode,
    Depth,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorType,
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

from .geometry import concatenate_meshes, create_arrow, create_cone, create_sphere, transform_mesh
from .gpu_sorting import GpuSortingPipeline, SortDataType, SortOptions
from .materials import ColorMaterial, DiffuseMaterial, Material
from .property import BufferProperty, ImageProperty, as_image_property
from .renderer import Renderer
from .renderer_frame import RendererFrame
from .scene import Object, Object3D
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .utils.gpu import cull_mode_opposite_face, div_round_up, view_bytes
from .viewport import Viewport


class ColormapKind(Enum):
    AUTUMN = 0
    BONE = 1
    COOL = 2
    COPPER = 3
    HOT = 4
    HSV = 5
    JET = 6
    SPRING = 7
    SUMMER = 8
    WINTER = 9


@dataclass
class Colormap:
    color: ColormapKind = ColormapKind.JET
    range_min: float = 0.0
    range_max: float = 1.0


@dataclass
class ColormapDistanceToPoint(Colormap):
    point: vec3 = vec3(0, 0, 0)


@dataclass
class ColormapDistanceToPlane(Colormap):
    normal: vec3 = vec3(0, 0, 1)


class Points(Object3D):
    def __init__(
        self,
        points: BufferProperty,
        colors: Optional[BufferProperty] = None,
        uniform_color: Optional[int] = None,
        colormap: Optional[Colormap] = None,
        point_size: Union[BufferProperty, float] = 1.0,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        if uniform_color is None and colors is None and colormap is None:
            uniform_color = 0xFFCCCCCC

        if uniform_color is not None:
            if colormap is not None or colors is not None:
                raise ValueError("Only one of uniform_color, colormap and colors can be not None")
            self.constants_dtype = np.dtype(
                {
                    "transform": (np.dtype((np.float32, (3, 4))), 0),
                    "point_size": (np.float32, 48),
                    "color": (np.uint32, 52),
                }
            )  # type: ignore
        elif colors is not None:
            if colormap is not None:
                raise ValueError("Only one of uniform_color, colormap and colors can be not None")
            self.constants_dtype = np.dtype(
                {
                    "transform": (np.dtype((np.float32, (3, 4))), 0),
                    "point_size": (np.float32, 48),
                }
            )  # type: ignore
        elif colormap is not None:
            self.constants_dtype = np.dtype(
                {
                    "transform": (np.dtype((np.float32, (3, 4))), 0),
                    "point_size": (np.float32, 48),
                    "colormap_measure": (np.uint32, 52),
                    "range_min": (np.float32, 56),
                    "range_inv_delta": (np.float32, 60),
                    "point_or_normal": (np.dtype((np.float32, (3,))), 64),
                }
            )  # type: ignore

        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(name, translation, rotation, scale, enabled=enabled, viewport_mask=viewport_mask)
        self.point_size = self.add_buffer_property(point_size, np.float32, name="point_size")
        self.points = self.add_buffer_property(points, np.float32, (-1, 3), name="points").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.colors = (
            self.add_buffer_property(colors, np.uint32, (-1,), name="colors").use_gpu(
                BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
            )
            if colors is not None
            else None
        )
        self.uniform_color = uniform_color
        self.colormap = colormap

    def create(self, r: Renderer) -> None:
        vertex_bindings = [VertexBinding(0, 12, VertexInputRate.VERTEX)]
        vertex_attributes = [VertexAttribute(0, 0, Format.R32G32B32_SFLOAT)]

        defines = []
        if self.uniform_color is not None:
            defines.append(("UNIFORM_COLOR", ""))
        elif self.colors is not None:
            defines.append(("VERTEX_COLORS", ""))
            vertex_bindings.append(VertexBinding(1, 4, VertexInputRate.VERTEX))
            vertex_attributes.append(VertexAttribute(1, 1, Format.R32_UINT))
        elif self.colormap is not None:
            defines.append(("COLORMAP", ""))

        vert = r.compile_builtin_shader("3d/points.slang", "vertex_main", defines=defines)
        frag = r.compile_builtin_shader("3d/points.slang", "pixel_main", defines=defines)

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=vertex_bindings,
            vertex_attributes=vertex_attributes,
            input_assembly=InputAssembly(PrimitiveTopology.POINT_LIST),
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

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        self.constants["point_size"] = self.point_size.get_current()
        if self.uniform_color is not None:
            self.constants["color"] = self.uniform_color
        elif self.colormap is not None:
            if isinstance(self.colormap, ColormapDistanceToPoint):
                measure = 0
                self.constants["point_or_normal"][:3] = self.colormap.point
            elif isinstance(self.colormap, ColormapDistanceToPlane):
                measure = 1
                self.constants["point_or_normal"][:3] = self.colormap.normal
            else:
                raise ValueError("colormap must be of type ColormapDistanceToPlane or ColormapDistanceToPoint")
            self.constants["colormap_measure"] = (measure << 16) | self.colormap.color.value
            self.constants["range_min"] = self.colormap.range_min
            self.constants["range_inv_delta"] = 1.0 / (self.colormap.range_max - self.colormap.range_min)

        points = self.points.get_current_gpu()

        vertex_buffers = [points.buffer_and_offset()]
        if self.colors is not None:
            vertex_buffers.append(self.colors.get_current_gpu().buffer_and_offset())

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=vertex_buffers,
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.draw(points.size // 12)


class Lines(Object3D):
    def __init__(
        self,
        lines: BufferProperty,
        colors: BufferProperty,
        line_width: Union[BufferProperty, float] = 1.0,
        is_strip: bool = False,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(name, translation, rotation, scale, enabled=enabled, viewport_mask=viewport_mask)
        self.is_strip = is_strip
        self.lines = self.add_buffer_property(lines, np.float32, (-1, 3), name="lines").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.colors = self.add_buffer_property(colors, np.uint32, (-1,), name="colors").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.line_width = self.add_buffer_property(line_width, np.float32, name="line_width")

    def create(self, r: Renderer) -> None:
        vert = r.compile_builtin_shader("3d/basic.slang", "vertex_main")
        frag = r.compile_builtin_shader("3d/basic.slang", "pixel_main")

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

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        lines = self.lines.get_current_gpu()
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                lines.buffer_and_offset(),
                self.colors.get_current_gpu().buffer_and_offset(),
            ],
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.set_line_width(
            self.line_width.get_current().item() if renderer.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0
        )

        frame.cmd.draw(lines.size // 12)


class Mesh(Object3D):
    def __init__(
        self,
        positions: BufferProperty,
        indices: Optional[BufferProperty] = None,
        normals: Optional[BufferProperty] = None,
        tangents: Optional[BufferProperty] = None,
        uvs: Optional[BufferProperty] = None,
        vertex_colors: Optional[BufferProperty] = None,
        instance_positions: Optional[BufferProperty] = None,
        primitive_topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST,
        cull_mode: CullMode = CullMode.NONE,
        front_face: FrontFace = FrontFace.COUNTER_CLOCKWISE,
        material: Optional[Material] = None,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
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

        if material is None:
            default_color = (0.5, 0.5, 0.5) if vertex_colors is None else (1.0, 1.0, 1.0)
            material = DiffuseMaterial(default_color)

        super().__init__(
            name,
            translation,
            rotation,
            scale,
            material,
            enabled,
            viewport_mask,
        )

        # Add properties
        self.positions = self.add_buffer_property(positions, np.float32, (-1, 3), name="positions").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.normals = (
            self.add_buffer_property(normals, np.float32, (-1, 3), name="normals").use_gpu(
                BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
            )
            if normals is not None
            else None
        )
        self.tangents = (
            self.add_buffer_property(tangents, np.float32, (-1, 3), name="tangents").use_gpu(
                BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
            )
            if tangents is not None
            else None
        )
        self.uvs = (
            self.add_buffer_property(uvs, np.float32, (-1, 2), name="uvs").use_gpu(
                BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
            )
            if uvs is not None
            else None
        )
        self.indices = (
            self.add_buffer_property(indices, np.uint32, (-1,), name="indices").use_gpu(
                BufferUsageFlags.INDEX, PipelineStageFlags.VERTEX_INPUT
            )
            if indices is not None
            else None
        )
        self.vertex_colors = (
            self.add_buffer_property(vertex_colors, np.uint32, (-1,), name="vertex_colors").use_gpu(
                BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
            )
            if vertex_colors is not None
            else None
        )
        self.instance_positions = (
            self.add_buffer_property(instance_positions, np.float32, (-1, 3), name="instance_positions").use_gpu(
                BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
            )
            if instance_positions is not None
            else None
        )

    def create(self, r: Renderer) -> None:
        assert self.material is not None
        defines: List[Tuple[str, str]] = []
        defines.extend(self.material.shader_defines)

        depth_defines: List[Tuple[str, str]] = []
        vertex_binding_index = 1
        vertex_bindings = [
            VertexBinding(0, 12, VertexInputRate.VERTEX),
        ]
        vertex_attributes = [
            VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
        ]

        depth_vertex_binding_index = 1
        depth_vertex_bindings = [
            VertexBinding(0, 12, VertexInputRate.VERTEX),
        ]
        depth_vertex_attributes = [
            VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
        ]

        if self.normals is not None:
            defines.append(("VERTEX_NORMALS", str(vertex_binding_index)))
            vertex_bindings.append(VertexBinding(vertex_binding_index, 12, VertexInputRate.VERTEX))
            vertex_attributes.append(
                VertexAttribute(vertex_binding_index, vertex_binding_index, Format.R32G32B32_SFLOAT)
            )
            vertex_binding_index += 1

        if self.tangents is not None:
            defines.append(("VERTEX_TANGENTS", str(vertex_binding_index)))
            vertex_bindings.append(VertexBinding(vertex_binding_index, 12, VertexInputRate.VERTEX))
            vertex_attributes.append(
                VertexAttribute(vertex_binding_index, vertex_binding_index, Format.R32G32B32_SFLOAT)
            )
            vertex_binding_index += 1

        if self.uvs is not None:
            defines.append(("VERTEX_UVS", str(vertex_binding_index)))
            vertex_bindings.append(VertexBinding(vertex_binding_index, 8, VertexInputRate.VERTEX))
            vertex_attributes.append(VertexAttribute(vertex_binding_index, vertex_binding_index, Format.R32G32_SFLOAT))
            vertex_binding_index += 1

        if self.vertex_colors is not None:
            defines.append(("VERTEX_COLORS", str(vertex_binding_index)))
            vertex_bindings.append(VertexBinding(vertex_binding_index, 4, VertexInputRate.VERTEX))
            vertex_attributes.append(VertexAttribute(vertex_binding_index, vertex_binding_index, Format.R32_UINT))
            vertex_binding_index += 1

        if self.instance_positions is not None:
            defines.append(("INSTANCE_POSITIONS", str(vertex_binding_index)))
            vertex_bindings.append(VertexBinding(vertex_binding_index, 12, VertexInputRate.INSTANCE))
            vertex_attributes.append(
                VertexAttribute(vertex_binding_index, vertex_binding_index, Format.R32G32B32_SFLOAT)
            )
            vertex_binding_index += 1

            depth_defines.append(("INSTANCE_POSITIONS", str(depth_vertex_binding_index)))
            depth_vertex_bindings.append(VertexBinding(depth_vertex_binding_index, 12, VertexInputRate.INSTANCE))
            depth_vertex_attributes.append(
                VertexAttribute(depth_vertex_binding_index, depth_vertex_binding_index, Format.R32G32B32_SFLOAT)
            )
            depth_vertex_binding_index += 1

        vert = r.compile_builtin_shader("3d/mesh.slang", "vertex_main", defines=defines)
        frag = r.compile_builtin_shader("3d/mesh.slang", "pixel_main", defines=defines)

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

        depth_vert = r.compile_builtin_shader("3d/mesh_depth.slang", "vertex_main", defines=depth_defines)
        self.depth_pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, depth_vert.code), Stage.VERTEX),
            ],
            vertex_bindings=depth_vertex_bindings,
            vertex_attributes=depth_vertex_attributes,
            rasterization=Rasterization(cull_mode=cull_mode_opposite_face(self.cull_mode), front_face=self.front_face),
            input_assembly=InputAssembly(self.primitive_topology),
            attachments=[],
            depth=Depth(r.shadow_map_format, True, True, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_depth_descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype["transform"].itemsize)],
        )

    def update_transform(self, parent: Optional[Object]) -> None:
        super().update_transform(parent)
        self.constants["transform"] = mat4x3(self.current_transform_matrix)
        self.constants["normal_matrix"][:, :, :3] = transpose(inverse(mat3(self.current_transform_matrix)))

    def render_depth(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        positions = self.positions.get_current_gpu()
        vertex_buffers = [
            positions.buffer_and_offset(),
        ]
        num_instances = 1
        if self.instance_positions is not None:
            vertex_buffers.append(self.instance_positions.get_current_gpu().buffer_and_offset())
            instance_positions_buf = self.instance_positions.get_current_gpu()
            vertex_buffers.append(instance_positions_buf.buffer_and_offset())
            num_instances = instance_positions_buf.size // 12

        indices = self.indices.get_current_gpu() if self.indices is not None else None
        frame.cmd.bind_graphics_pipeline(
            self.depth_pipeline,
            vertex_buffers=vertex_buffers,
            index_buffer=indices.buffer_and_offset() if indices is not None else None,
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants["transform"].tobytes(),
        )

        if indices is not None:
            frame.cmd.draw_indexed(indices.size // 4, num_instances)
        else:
            frame.cmd.draw(positions.size // 12, num_instances)

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        assert self.material is not None

        positions = self.positions.get_current_gpu()
        vertex_buffers = [
            positions.buffer_and_offset(),
        ]
        num_instances = 1
        if self.normals is not None:
            vertex_buffers.append(self.normals.get_current_gpu().buffer_and_offset())
        if self.tangents is not None:
            vertex_buffers.append(self.tangents.get_current_gpu().buffer_and_offset())
        if self.uvs is not None:
            vertex_buffers.append(self.uvs.get_current_gpu().buffer_and_offset())
        if self.vertex_colors is not None:
            vertex_buffers.append(self.vertex_colors.get_current_gpu().buffer_and_offset())
        if self.instance_positions is not None:
            instance_positions_buf = self.instance_positions.get_current_gpu()
            vertex_buffers.append(instance_positions_buf.buffer_and_offset())
            num_instances = instance_positions_buf.size // 12

        indices = self.indices.get_current_gpu() if self.indices is not None else None
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=vertex_buffers,
            index_buffer=indices.buffer_and_offset() if indices is not None else None,
            descriptor_sets=[
                scene_descriptor_set,
                self.material.descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )

        if indices is not None:
            frame.cmd.draw_indexed(indices.size // 4, num_instances)
        else:
            frame.cmd.draw(positions.size // 12, num_instances)


class Sphere(Mesh):
    def __init__(
        self,
        radius: float = 0.1,
        color: int = 0xFFCCCCCC,
        rings: int = 16,
        sectors: int = 32,
        instance_positions: Optional[BufferProperty] = None,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        v, n, f = create_sphere(radius, rings, sectors)
        c = np.full(v.shape[0], color, np.uint32)

        super().__init__(
            v,  # type: ignore
            f,  # type: ignore
            n,  # type: ignore
            vertex_colors=c,  # type: ignore
            instance_positions=instance_positions,
            name=name,
            translation=translation,
            rotation=rotation,
            scale=scale,
            enabled=enabled,
            viewport_mask=viewport_mask,
        )


class AxisGizmo(Mesh):
    def __init__(
        self,
        sphere_radius: float = 0.01,
        axis_radius: float = 0.005,
        axis_length: float = 0.1,
        sphere_color: int = 0xFFCCCCCC,
        x_axis_color: int = 0xFF0000FF,
        y_axis_color: int = 0xFF00FF00,
        z_axis_color: int = 0xFFFF0000,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        sphere_v, sphere_n, sphere_f = create_sphere(sphere_radius, 8, 16)
        sphere_c = np.full(sphere_v.shape[0], sphere_color, np.uint32)

        z_cylinder_v, z_cylinder_n, cylinder_f = create_arrow(
            axis_radius, axis_length * 0.8, axis_radius * 2.0, axis_length * 0.2, 16
        )
        z_cylinder_c = np.full(z_cylinder_v.shape[0], z_axis_color, np.uint32)

        x_cylinder_v, x_cylinder_n = transform_mesh(rotate(np.pi * 0.5, vec3(0, 1, 0)), z_cylinder_v, z_cylinder_n)
        x_cylinder_c = np.full(x_cylinder_v.shape[0], x_axis_color, np.uint32)

        y_cylinder_v, y_cylinder_n = transform_mesh(rotate(np.pi * 0.5, vec3(0, 0, 1)), x_cylinder_v, x_cylinder_n)
        y_cylinder_c = np.full(y_cylinder_v.shape[0], y_axis_color, np.uint32)
        (mesh_v, mesh_n, mesh_c), mesh_f = concatenate_meshes(
            [
                (sphere_v, sphere_n, sphere_c),
                (x_cylinder_v, x_cylinder_n, x_cylinder_c),
                (y_cylinder_v, y_cylinder_n, y_cylinder_c),
                (z_cylinder_v, z_cylinder_n, z_cylinder_c),
            ],
            [sphere_f, cylinder_f, cylinder_f, cylinder_f],
        )

        super().__init__(
            mesh_v,  # type: ignore
            mesh_f,  # type: ignore
            mesh_n,  # type: ignore
            vertex_colors=mesh_c,  # type: ignore
            name=name,
            translation=translation,
            rotation=rotation,
            scale=scale,
            enabled=enabled,
            viewport_mask=viewport_mask,
        )


class Image(Mesh):
    def __init__(
        self,
        image: ImageProperty,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
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
            enabled=enabled,
            viewport_mask=viewport_mask,
        )


class GridType(Enum):
    XY_PLANE = 0
    YZ_PLANE = 1
    XZ_PLANE = 2


class Grid(Object3D):
    def __init__(
        self,
        size: Tuple[float, float],
        grid_type: GridType,
        major_line_color: Tuple[float, float, float, float],
        minor_line_color: Tuple[float, float, float, float],
        base_color: Tuple[float, float, float, float],
        grid_scale: float = 0.1,
        major_grid_div: float = 10.0,
        axis_line_width: float = 0.08,
        major_line_width: float = 0.04,
        minor_line_width: float = 0.01,
        pos_axis_color_scale: float = 1.0,
        neg_axis_color_scale: float = 0.5,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "major_line_color": (np.dtype((np.float32, 4)), 0),
                "minor_line_color": (np.dtype((np.float32, 4)), 16),
                "base_color": (np.dtype((np.float32, 4)), 32),
                "size": (np.dtype((np.float32, 2)), 48),
                "grid_type": (np.uint32, 56),
                "inv_grid_scale": (np.float32, 60),
                "major_grid_div": (np.float32, 64),
                "axis_line_width": (np.float32, 68),
                "major_line_width": (np.float32, 72),
                "minor_line_width": (np.float32, 76),
                "pos_axis_color_scale": (np.float32, 80),
                "neg_axis_color_scale": (np.float32, 84),
            }
        )  # type: ignore

        self.constants = np.zeros((1,), self.constants_dtype)
        self.constants["major_line_color"] = major_line_color
        self.constants["minor_line_color"] = minor_line_color
        self.constants["base_color"] = base_color
        self.constants["size"] = size
        self.constants["grid_type"] = grid_type.value
        self.constants["inv_grid_scale"] = 1.0 / grid_scale
        self.constants["major_grid_div"] = major_grid_div
        self.constants["axis_line_width"] = axis_line_width
        self.constants["major_line_width"] = major_line_width
        self.constants["minor_line_width"] = minor_line_width
        self.constants["pos_axis_color_scale"] = pos_axis_color_scale
        self.constants["neg_axis_color_scale"] = neg_axis_color_scale

        self.is_transparent = base_color[3] < 1.0

        super().__init__(name, translation, rotation, scale, enabled=enabled, viewport_mask=viewport_mask)

    @classmethod
    def white(cls, size: Tuple[float, float], grid_type: GridType, **kwargs: Any) -> "Grid":
        return cls(size, grid_type, (0, 0, 0, 1), (0, 0, 0, 1), (1, 1, 1, 1), **kwargs)

    @classmethod
    def black(cls, size: Tuple[float, float], grid_type: GridType, **kwargs: Any) -> "Grid":
        return cls(size, grid_type, (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 1), **kwargs)

    @classmethod
    def transparent_white_lines(cls, size: Tuple[float, float], grid_type: GridType, **kwargs: Any) -> "Grid":
        return cls(size, grid_type, (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 0), **kwargs)

    @classmethod
    def transparent_black_lines(cls, size: Tuple[float, float], grid_type: GridType, **kwargs: Any) -> "Grid":
        return cls(size, grid_type, (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0), **kwargs)

    @classmethod
    def dark_gray_white_lines(cls, size: Tuple[float, float], grid_type: GridType, **kwargs: Any) -> "Grid":
        return cls(size, grid_type, (1, 1, 1, 1), (1, 1, 1, 1), (0.1, 0.1, 0.1, 1), **kwargs)

    @classmethod
    def light_gray_black_lines(cls, size: Tuple[float, float], grid_type: GridType, **kwargs: Any) -> "Grid":
        return cls(size, grid_type, (0, 0, 0, 1), (0, 0, 0, 1), (0.9, 0.9, 0.9, 1), **kwargs)

    def create(self, r: Renderer) -> None:
        vert = r.compile_builtin_shader("3d/grid.slang", "vertex_main")
        frag = r.compile_builtin_shader("3d/grid.slang", "pixel_main")
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            rasterization=Rasterization(cull_mode=CullMode.NONE),
            input_assembly=InputAssembly(PrimitiveTopology.TRIANGLE_STRIP),
            samples=r.msaa_samples,
            attachments=[
                Attachment(
                    format=r.output_format,
                    blend_enable=True,
                    src_color_blend_factor=BlendFactor.SRC_ALPHA,
                    dst_color_blend_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
                    color_blend_op=BlendOp.ADD,
                    src_alpha_blend_factor=BlendFactor.ONE,
                    dst_alpha_blend_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
                    alpha_blend_op=BlendOp.ADD,
                )
            ],
            depth=Depth(r.depth_format, True, not self.is_transparent, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.draw(4)


class Voxels(Object3D):
    def __init__(
        self,
        positions: BufferProperty,
        size: float,
        colors: Optional[BufferProperty] = None,
        uniform_color: Optional[int] = None,
        colormap: Optional[Colormap] = None,
        border_size: float = 0.1,
        border_factor: float = 0.5,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        if uniform_color is None and colors is None and colormap is None:
            uniform_color = 0xFFCCCCCC

        if uniform_color is not None:
            if colormap is not None or colors is not None:
                raise ValueError("Only one of uniform_color, colormap and colors can be not None")
            self.constants_dtype = np.dtype(
                {
                    "transform": (np.dtype((np.float32, (3, 4))), 0),
                    "object_camera_position": (np.dtype((np.float32, (3,))), 48),
                    "size": (np.float32, 60),
                    "border_size": (np.float32, 64),
                    "border_factor": (np.float32, 68),
                    "color": (np.uint32, 72),
                }
            )  # type: ignore
        elif colors is not None:
            if colormap is not None:
                raise ValueError("Only one of uniform_color, colormap and colors can be not None")
            self.constants_dtype = np.dtype(
                {
                    "transform": (np.dtype((np.float32, (3, 4))), 0),
                    "object_camera_position": (np.dtype((np.float32, (3,))), 48),
                    "size": (np.float32, 60),
                    "border_size": (np.float32, 64),
                    "border_factor": (np.float32, 68),
                }
            )  # type: ignore
        elif colormap is not None:
            self.constants_dtype = np.dtype(
                {
                    "transform": (np.dtype((np.float32, (3, 4))), 0),
                    "object_camera_position": (np.dtype((np.float32, (3,))), 48),
                    "size": (np.float32, 60),
                    "border_size": (np.float32, 64),
                    "border_factor": (np.float32, 68),
                    "colormap_measure": (np.uint32, 72),
                    "range_min": (np.float32, 76),
                    "point_or_normal": (np.dtype((np.float32, (3,))), 80),
                    "range_inv_delta": (np.float32, 92),
                }
            )  # type: ignore

        self.constants = np.zeros((1,), self.constants_dtype)
        self.size = size
        self.border_size = border_size
        self.border_factor = border_factor

        super().__init__(name, translation, rotation, scale, enabled=enabled, viewport_mask=viewport_mask)
        self.positions = self.add_buffer_property(positions, np.float32, (-1, 3), name="positions").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.colors = (
            self.add_buffer_property(colors, np.uint32, (-1,), name="colors").use_gpu(
                BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
            )
            if colors is not None
            else None
        )
        self.colormap = colormap
        self.uniform_color = uniform_color

    def create(self, r: Renderer) -> None:
        vertex_bindings = [VertexBinding(0, 12, VertexInputRate.INSTANCE)]
        vertex_attributes = [VertexAttribute(0, 0, Format.R32G32B32_SFLOAT)]

        defines = []
        if self.uniform_color is not None:
            defines.append(("UNIFORM_COLOR", ""))
        elif self.colors is not None:
            defines.append(("VERTEX_COLORS", ""))
            vertex_bindings.append(VertexBinding(1, 4, VertexInputRate.INSTANCE))
            vertex_attributes.append(VertexAttribute(1, 1, Format.R32_UINT))
        elif self.colormap is not None:
            defines.append(("COLORMAP", ""))

        vert = r.compile_builtin_shader("3d/voxels.slang", "vertex_main", defines=defines)
        frag = r.compile_builtin_shader("3d/voxels.slang", "pixel_main", defines=defines)

        indices = np.array(
            [
                [0, 1, 3],
                [3, 0, 2],
                [0, 4, 5],
                [1, 0, 5],
                [0, 2, 4],
                [2, 4, 6],
            ],
            np.uint32,
        )
        self.indices = Buffer.from_data(
            r.ctx,
            view_bytes(indices),
            BufferUsageFlags.INDEX | BufferUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name="voxel-indices",
        )

        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=vertex_bindings,
            vertex_attributes=vertex_attributes,
            input_assembly=InputAssembly(PrimitiveTopology.TRIANGLE_LIST),
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

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        self.constants["object_camera_position"] = vec3(
            inverse(self.current_transform_matrix) * vec4(viewport.camera.position(), 1.0)
        )  # type: ignore
        self.constants["size"] = self.size
        self.constants["border_size"] = self.border_size
        self.constants["border_factor"] = self.border_factor
        if self.uniform_color is not None:
            self.constants["color"] = self.uniform_color
        elif self.colormap is not None:
            if isinstance(self.colormap, ColormapDistanceToPoint):
                measure = 0
                self.constants["point_or_normal"][:3] = self.colormap.point
            elif isinstance(self.colormap, ColormapDistanceToPlane):
                measure = 1
                self.constants["point_or_normal"][:3] = self.colormap.normal
            else:
                raise ValueError("colormap must be of type ColormapDistanceToPlane or ColormapDistanceToPoint")
            self.constants["colormap_measure"] = (measure << 16) | self.colormap.color.value
            self.constants["range_min"] = self.colormap.range_min
            self.constants["range_inv_delta"] = 1.0 / (self.colormap.range_max - self.colormap.range_min)

        positions = self.positions.get_current_gpu()

        vertex_buffers = [positions.buffer_and_offset()]
        if self.colors is not None:
            vertex_buffers.append(self.colors.get_current_gpu().buffer_and_offset())

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=vertex_buffers,
            index_buffer=self.indices,
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.draw_indexed(18, positions.size // 12)


# NOTE: must be kept in sync with RenderFlags in shaders/3d/gaussian_splatting/shaderio.h
class GaussianSplatsRenderFlags(IntFlag):
    NONE = (0,)
    DISABLE_OPACITY = (1 << 0,)
    SHOW_SPHERICAL_HARMONICS_ONLY = (1 << 1,)
    ORTHOGRAPHIC_MODE = (1 << 2,)
    POINT_CLOUD_MODE = (1 << 3,)


class GaussianSplats(Object3D):
    DISTANCE_COMPUTE_WORKGROUP_SIZE = 256

    FRUSTUM_CULLING_AT_NONE = 0
    FRUSTUM_CULLING_AT_DIST = 1
    FRUSTUM_CULLING_AT_RASTER = 2

    FORMAT_FLOAT32 = 0
    FORMAT_FLOAT16 = 1
    FORMAT_UINT8 = 2

    def __init__(
        self,
        positions: BufferProperty,
        colors: BufferProperty,
        spherical_harmonics: BufferProperty,
        covariances: BufferProperty,
        flags: GaussianSplatsRenderFlags = GaussianSplatsRenderFlags.NONE,
        cull_at_dist: bool = True,
        mip_splatting_antialiasing: bool = False,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
                "inverse_transform": (np.dtype((np.float32, (3, 4))), 48),
                "splat_count": (np.uint32, 96),
                "alpha_cull_threshold": (np.float32, 100),
                "frustum_dilation": (np.float32, 104),
                "splat_scale": (np.float32, 108),
                "flags": (np.uint32, 112),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(name, translation, rotation, scale, enabled=enabled, viewport_mask=viewport_mask)
        self.positions = self.add_buffer_property(positions, np.float32, (-1, 3), name="positions").use_gpu(
            BufferUsageFlags.STORAGE,
            PipelineStageFlags.COMPUTE_SHADER,
        )
        self.colors = self.add_buffer_property(colors, np.float32, (-1, 4), name="colors").use_gpu(
            BufferUsageFlags.STORAGE,
            PipelineStageFlags.COMPUTE_SHADER,
        )
        self.spherical_harmonics = self.add_buffer_property(
            spherical_harmonics, None, (-1, -1), name="spherical-harmonics"
        ).use_gpu(
            BufferUsageFlags.STORAGE,
            PipelineStageFlags.COMPUTE_SHADER,
        )
        if self.spherical_harmonics.dtype == np.uint8:
            self.sh_format = self.FORMAT_UINT8
        elif self.spherical_harmonics.dtype == np.float16:
            self.sh_format = self.FORMAT_FLOAT16
        elif self.spherical_harmonics.dtype == np.float32:
            self.sh_format = self.FORMAT_FLOAT32
        else:
            raise TypeError(
                f"Spherical harmonics must be of dtype uint8, float16 or float32. Got {self.spherical_harmonics.dtype}."
            )
        self.covariances = self.add_buffer_property(covariances, np.float32, (-1, 6), name="covariances").use_gpu(
            BufferUsageFlags.STORAGE,
            PipelineStageFlags.COMPUTE_SHADER,
        )
        self.mip_splatting_antialiasing = mip_splatting_antialiasing
        self.flags = flags
        self.cull_at_dist = cull_at_dist
        self.alpha_cull_threshold = 1.0 / 255.0
        self.frustum_dilation = 0.2
        self.splat_scale = 1.0

    def create(self, r: Renderer) -> None:
        if self.sh_format == self.FORMAT_UINT8 and (r.ctx.device_features & DeviceFeatures.STORAGE_8BIT) == 0:
            raise RuntimeError("Device does not support 8bit storage")
        if self.sh_format == self.FORMAT_FLOAT16 and (r.ctx.device_features & DeviceFeatures.STORAGE_16BIT) == 0:
            raise RuntimeError("Device does not support 16bit storage")

        self.use_barycentric = (r.ctx.device_features & DeviceFeatures.FRAGMENT_SHADER_BARYCENTRIC) != 0
        self.use_mesh_shader = (r.ctx.device_features & DeviceFeatures.MESH_SHADER) != 0

        max_splats = self.positions.max_size // 12
        self.dists_buf = Buffer(
            r.ctx, max_splats * 4, BufferUsageFlags.STORAGE, AllocType.DEVICE, name=f"{self.name}-dists"
        )
        self.dists_alt_buf = Buffer(
            r.ctx, max_splats * 4, BufferUsageFlags.STORAGE, AllocType.DEVICE, name=f"{self.name}-dists-alt"
        )
        self.sorted_indices_buf = Buffer(
            r.ctx,
            max_splats * 4,
            BufferUsageFlags.STORAGE | BufferUsageFlags.VERTEX,
            AllocType.DEVICE,
            name=f"{self.name}-sorted-indices",
        )
        self.sorted_indices_alt_buf = Buffer(
            r.ctx, max_splats * 4, BufferUsageFlags.STORAGE, AllocType.DEVICE, name=f"{self.name}-sorted-indices-alt"
        )

        if not self.use_mesh_shader:
            quad_vertices = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], np.float32)
            quad_indices = np.array([0, 2, 1, 2, 0, 3], np.uint32)
            self.quad_vertices = Buffer.from_data(
                r.ctx,
                quad_vertices,
                BufferUsageFlags.VERTEX | BufferUsageFlags.TRANSFER_DST,
                AllocType.DEVICE,
                name=f"{self.name}-quad-vertices",
            )
            self.quad_indices = Buffer.from_data(
                r.ctx,
                quad_indices,
                BufferUsageFlags.INDEX | BufferUsageFlags.TRANSFER_DST,
                AllocType.DEVICE,
                name=f"{self.name}-quad-indices",
            )

            draw_parameters = np.array([6, max_splats, 0, 0, 0], np.uint32)
            self.draw_parameters_buf = Buffer.from_data(
                r.ctx,
                draw_parameters,
                BufferUsageFlags.INDIRECT | BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
                AllocType.DEVICE,
                name=f"{self.name}-draw-param",
            )
            self.draw_parameters_init = np.array([6, 0, 0, 0, 0], np.uint32).tobytes()
        else:
            mesh_workgroup_size = min(
                r.ctx.device_properties.mesh_shader_properties.max_preferred_mesh_work_group_invocations,
                r.ctx.device_properties.mesh_shader_properties.max_mesh_work_group_count[0],
            )
            draw_mesh_tasks_parameters = np.array(
                [div_round_up(max_splats, mesh_workgroup_size), 1, 1, max_splats], np.uint32
            )
            self.draw_mesh_tasks_buf = Buffer.from_data(
                r.ctx,
                draw_mesh_tasks_parameters,
                BufferUsageFlags.INDIRECT | BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
                AllocType.DEVICE,
                name=f"{self.name}-draw-mesh-tasks-param",
            )
            self.draw_mesh_tasks_init = np.array([0, 1, 1, 0], np.uint32).tobytes()

        defines = [
            ("DISTANCE_COMPUTE_WORKGROUP_SIZE", str(self.DISTANCE_COMPUTE_WORKGROUP_SIZE)),
            (
                "FRUSTUM_CULLING_MODE",
                str(self.FRUSTUM_CULLING_AT_DIST) if self.cull_at_dist else str(self.FRUSTUM_CULLING_AT_RASTER),
            ),
            ("USE_BARYCENTRIC", "1" if self.use_barycentric else "0"),
            (
                "MS_ANTIALIASING",
                "1" if self.mip_splatting_antialiasing else "0",
            ),  # Only useful when model comes from mip splatting
            ("MAX_SH_DEGREE", "3"),
            ("SH_FORMAT", str(self.sh_format)),
            # Using this as a flag significantly lowers performance even if disabled.
            # Likely due to worse occupancy because of higher register pressure or due to some optimization
            # not kicking in because of to the use of shader derivatives.
            ("WIREFRAME", "0"),
        ]

        if self.use_mesh_shader:
            defines.append(("RASTER_MESH_WORKGROUP_SIZE", str(mesh_workgroup_size)))

        dist_shader = r.compile_builtin_shader("3d/gaussian_splatting/dist.comp.slang", defines=defines)
        self.descriptor_layout, self.descriptor_pool, self.descriptor_sets = (
            create_descriptor_layout_pool_and_sets_ringbuffer(
                r.ctx,
                [DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER) for _ in range(7)],
                r.num_frames_in_flight,
            )
        )

        for s in self.descriptor_sets:
            s.write_buffer(self.dists_buf, DescriptorType.STORAGE_BUFFER, 0)
            s.write_buffer(self.sorted_indices_buf, DescriptorType.STORAGE_BUFFER, 1)
            if not self.use_mesh_shader:
                s.write_buffer(self.draw_parameters_buf, DescriptorType.STORAGE_BUFFER, 2)
            else:
                s.write_buffer(self.draw_mesh_tasks_buf, DescriptorType.STORAGE_BUFFER, 2)

        self.dist_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, dist_shader.code),
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
                self.descriptor_layout,
            ],
        )

        # Keys are UINT32 because we already convert them to being ordered when
        # interpreted as integers in the dist computation pass.
        self.sort_pipeline = GpuSortingPipeline(
            r,
            SortOptions(
                key_type=SortDataType.UINT32,
                payload_type=SortDataType.UINT32,
                indirect=self.cull_at_dist,
                unsafe_has_forward_thread_progress_guarantee=False,
            ),
        )

        if not self.use_mesh_shader:
            vert = r.compile_builtin_shader("3d/gaussian_splatting/threedgs_raster.vert.slang", defines=defines)
            frag = r.compile_builtin_shader("3d/gaussian_splatting/threedgs_raster.frag.slang", defines=defines)
        else:
            mesh = r.compile_builtin_shader(
                "3d/gaussian_splatting/threedgs_raster.mesh.slang", target="spirv_1_4", defines=defines
            )
            frag = r.compile_builtin_shader(
                "3d/gaussian_splatting/threedgs_raster.frag.slang", entry="main_mesh", defines=defines
            )

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX)
                if not self.use_mesh_shader
                else PipelineStage(Shader(r.ctx, mesh.code), Stage.MESH),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=[
                VertexBinding(0, 8, VertexInputRate.VERTEX),
                VertexBinding(1, 4, VertexInputRate.INSTANCE),
            ]
            if not self.use_mesh_shader
            else [],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32_SFLOAT),
                VertexAttribute(1, 1, Format.R32_UINT),
            ]
            if not self.use_mesh_shader
            else [],
            input_assembly=InputAssembly(
                PrimitiveTopology.TRIANGLE_LIST,
            ),
            samples=1,
            attachments=[
                Attachment(
                    format=r.output_format,
                    blend_enable=True,
                    src_color_blend_factor=BlendFactor.SRC_ALPHA,
                    dst_color_blend_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
                    color_blend_op=BlendOp.ADD,
                    src_alpha_blend_factor=BlendFactor.ONE,
                    dst_alpha_blend_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
                    alpha_blend_op=BlendOp.ADD,
                )
            ],
            depth=Depth(r.depth_format, False, False, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
                self.descriptor_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def update_transform(self, parent: Optional[Object]) -> None:
        super().update_transform(parent)
        self.constants["transform"] = mat4x3(self.current_transform_matrix)
        self.constants["inverse_transform"] = mat4x3(inverse(self.current_transform_matrix))

    def upload(self, r: Renderer, frame: RendererFrame) -> None:
        # Constants
        positions = self.positions.get_current_gpu()
        num_splats = positions.size // 12
        self.constants["splat_count"] = num_splats
        self.constants["alpha_cull_threshold"] = self.alpha_cull_threshold
        self.constants["frustum_dilation"] = self.frustum_dilation
        self.constants["splat_scale"] = self.splat_scale
        self.constants["flags"] = self.flags

        self.descriptor_set = self.descriptor_sets.get_current_and_advance()

        self.positions.get_current_gpu().write_descriptor(self.descriptor_set, DescriptorType.STORAGE_BUFFER, 3)
        self.colors.get_current_gpu().write_descriptor(self.descriptor_set, DescriptorType.STORAGE_BUFFER, 4)
        self.covariances.get_current_gpu().write_descriptor(self.descriptor_set, DescriptorType.STORAGE_BUFFER, 5)
        self.spherical_harmonics.get_current_gpu().write_descriptor(
            self.descriptor_set, DescriptorType.STORAGE_BUFFER, 6
        )

        count_buf = None
        if self.cull_at_dist:
            if self.use_mesh_shader:
                count_buf = self.draw_mesh_tasks_buf
            else:
                count_buf = self.draw_parameters_buf

        self.sort_pipeline.upload(
            r,
            self.dists_buf,
            self.dists_alt_buf,
            self.sorted_indices_buf,
            self.sorted_indices_alt_buf,
            count_buf,
        )

        frame.between_viewport_render_src_pipeline_stages |= PipelineStageFlags.COMPUTE_SHADER
        frame.between_viewport_render_dst_pipeline_stages |= PipelineStageFlags.TRANSFER

    def pre_render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        positions = self.positions.get_current_gpu()
        num_splats = positions.size // 12

        # Reset counters
        if self.cull_at_dist:
            if self.use_mesh_shader:
                frame.cmd.update_buffer(self.draw_mesh_tasks_buf, self.draw_mesh_tasks_init)
            else:
                frame.cmd.fill_buffer(self.draw_parameters_buf, 0, 4, 4)
            frame.upload_property_pipeline_stages |= PipelineStageFlags.COMPUTE_SHADER

        frame.cmd.memory_barrier(MemoryUsage.TRANSFER_DST, MemoryUsage.COMPUTE_SHADER)

        # Distance
        frame.cmd.bind_compute_pipeline(
            self.dist_pipeline,
            descriptor_sets=[scene_descriptor_set, self.descriptor_set],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.dispatch(div_round_up(num_splats, self.DISTANCE_COMPUTE_WORKGROUP_SIZE), 1, 1)

        frame.cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

        # Sort
        if self.cull_at_dist:
            if self.use_mesh_shader:
                count_offset = 12
            else:
                count_offset = 4

            self.sort_pipeline.run_indirect(
                renderer,
                frame.cmd,
                count_offset,
            )
        else:
            self.sort_pipeline.run(
                renderer,
                frame.cmd,
                num_splats,
            )

        frame.before_render_src_pipeline_stages |= PipelineStageFlags.COMPUTE_SHADER
        frame.before_render_dst_pipeline_stages |= (
            PipelineStageFlags.MESH_SHADER if self.use_mesh_shader else PipelineStageFlags.VERTEX_SHADER
        )

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.quad_vertices,
                self.sorted_indices_buf,
            ]
            if not self.use_mesh_shader
            else [],
            index_buffer=self.quad_indices if not self.use_mesh_shader else None,
            descriptor_sets=[
                scene_descriptor_set,
                self.descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        if self.use_mesh_shader:
            frame.cmd.draw_mesh_tasks_indirect(self.draw_mesh_tasks_buf, 0, 1, 16)
        else:
            frame.cmd.draw_indexed_indirect(self.draw_parameters_buf, 0, 1, 20)
