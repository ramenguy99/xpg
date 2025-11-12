# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from enum import Enum, IntFlag
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from pyglm.glm import inverse, mat3, mat4x3, transpose
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

from .gpu_sorting import GpuSortingPipeline, SortDataType, SortOptions
from .materials import ColorMaterial, DiffuseMaterial, Material
from .property import BufferProperty, ImageProperty, as_image_property
from .renderer import Renderer
from .renderer_frame import RendererFrame
from .scene import Object, Object3D
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .utils.gpu import cull_mode_opposite_face, div_round_up


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
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(name, translation, rotation, scale, enabled=enabled)
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

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.lines.get_current_gpu().to_buffer_offset(),
                self.colors.get_current_gpu().to_buffer_offset(),
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
        tangents: Optional[BufferProperty] = None,
        uvs: Optional[BufferProperty] = None,
        primitive_topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST,
        cull_mode: CullMode = CullMode.NONE,
        front_face: FrontFace = FrontFace.COUNTER_CLOCKWISE,
        material: Optional[Material] = None,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
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
            name,
            translation,
            rotation,
            scale,
            material if material is not None else DiffuseMaterial((0.5, 0.5, 0.5)),
            enabled,
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

    def create(self, r: Renderer) -> None:
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

        depth_vert = r.compile_builtin_shader("3d/mesh_depth.slang", "vertex_main")
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
        self.constants["normal_matrix"][:, :, :3] = transpose(inverse(mat3(self.current_transform_matrix)))

    def render_depth(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        index_buffer = None
        index_buffer_offset = 0
        if self.indices is not None:
            index_buffer_view = self.indices.get_current_gpu()
            index_buffer = index_buffer_view.buffer
            index_buffer_offset = index_buffer_view.offset

        frame.cmd.bind_graphics_pipeline(
            self.depth_pipeline,
            vertex_buffers=[
                self.positions.get_current_gpu().to_buffer_offset(),
            ],
            index_buffer=index_buffer,
            descriptor_sets=[
                scene_descriptor_set,
            ],
            index_buffer_offset=index_buffer_offset,
            push_constants=self.constants["transform"].tobytes(),
        )

        if self.indices is not None:
            frame.cmd.draw_indexed(self.indices.get_current().shape[0])
        else:
            frame.cmd.draw(self.positions.get_current().shape[0])

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        assert self.material is not None

        index_buffer = None
        index_buffer_offset = 0
        if self.indices is not None:
            index_buffer_view = self.indices.get_current_gpu()
            index_buffer = index_buffer_view.buffer
            index_buffer_offset = index_buffer_view.offset

        vertex_buffers = [
            self.positions.get_current_gpu().to_buffer_offset(),
        ]
        if self.normals is not None:
            vertex_buffers.append(self.normals.get_current_gpu().to_buffer_offset())
        if self.tangents is not None:
            vertex_buffers.append(self.tangents.get_current_gpu().to_buffer_offset())
        if self.uvs is not None:
            vertex_buffers.append(self.uvs.get_current_gpu().to_buffer_offset())

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=vertex_buffers,
            index_buffer=index_buffer,
            index_buffer_offset=index_buffer_offset,
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
        enabled: Optional[BufferProperty] = None,
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

        super().__init__(name, translation, rotation, scale, enabled=enabled)

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
                    dst_alpha_blend_factor=BlendFactor.ZERO,
                    alpha_blend_op=BlendOp.ADD,
                )
            ],
            depth=Depth(r.depth_format, True, not self.is_transparent, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.draw(4)


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

        super().__init__(name, translation, rotation, scale, enabled=enabled)
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
                    dst_alpha_blend_factor=BlendFactor.ZERO,
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
        num_splats = self.positions.get_current().shape[0]
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

        # Distances
        if self.cull_at_dist:
            frame.cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.ALL)
            if self.use_mesh_shader:
                frame.cmd.update_buffer(self.draw_mesh_tasks_buf, self.draw_mesh_tasks_init)
            else:
                frame.cmd.fill_buffer(self.draw_parameters_buf, 0, 4, 4)
            frame.cmd.memory_barrier(MemoryUsage.TRANSFER_DST, MemoryUsage.COMPUTE_SHADER)

        frame.cmd.bind_compute_pipeline(
            self.dist_pipeline,
            descriptor_sets=[frame.scene_descriptor_set, self.descriptor_set],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.dispatch(div_round_up(num_splats, self.DISTANCE_COMPUTE_WORKGROUP_SIZE), 1, 1)

        frame.cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

        # Sort
        if self.cull_at_dist:
            if self.use_mesh_shader:
                count_buf = self.draw_mesh_tasks_buf
                count_offset = 12
            else:
                count_buf = self.draw_parameters_buf
                count_offset = 4

            self.sort_pipeline.run_indirect(
                r,
                frame.cmd,
                count_buf,
                count_offset,
                self.dists_buf,
                self.dists_alt_buf,
                self.sorted_indices_buf,
                self.sorted_indices_alt_buf,
            )
        else:
            self.sort_pipeline.run(
                r,
                frame.cmd,
                num_splats,
                self.dists_buf,
                self.dists_alt_buf,
                self.sorted_indices_buf,
                self.sorted_indices_alt_buf,
            )

        frame.cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.ALL)

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
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
