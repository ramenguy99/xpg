import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from pyglm.glm import inverse, mat4x3
from pyxpg import (
    AllocType,
    Attachment,
    BlendFactor,
    BlendOp,
    Buffer,
    BufferUsageFlags,
    ComputePipeline,
    Depth,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorType,
    DeviceFeatures,
    Format,
    GraphicsPipeline,
    InputAssembly,
    MemoryUsage,
    PipelineStage,
    PipelineStageFlags,
    PrimitiveTopology,
    PushConstantsRange,
    Shader,
    Stage,
    VertexAttribute,
    VertexBinding,
    VertexInputRate,
)
from scipy.spatial.transform import Rotation

from ambra.config import Config, GuiConfig, RendererConfig
from ambra.gpu_sorting import GpuSortingPipeline, SortDataType, SortOptions
from ambra.primitives3d import Lines
from ambra.property import BufferProperty
from ambra.renderer import Renderer
from ambra.renderer_frame import RendererFrame
from ambra.scene import Object, Object3D
from ambra.utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from ambra.utils.gpu import div_round_up
from ambra.viewer import Viewer


@dataclass
class Splats:
    position: np.ndarray
    sh: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray


DISTANCE_COMPUTE_WORKGROUP_SIZE = 256

FRUSTUM_CULLING_AT_NONE = 0
FRUSTUM_CULLING_AT_DIST = 1
FRUSTUM_CULLING_AT_RASTER = 2

FORMAT_FLOAT32 = 0
FORMAT_FLOAT16 = 1
FORMAT_UINT8 = 2


def load_ply(path: Path, sh_degree: int = 1):
    with open(path, "rb") as f:
        # Read header.
        head = f.readline().decode("utf-8").strip().lower()
        if head != "ply":
            raise ValueError(f"Not a ply file: {head}")

        encoding = f.readline().decode("utf-8").strip().lower()
        if "binary_little_endian" not in encoding:
            raise ValueError(f"Invalid encoding: {encoding}")

        elements = f.readline().decode("utf-8").strip().lower()
        count = int(elements.split()[2])

        # Read until end of header.
        # TODO: properly parse header
        while f.readline().decode("utf-8").strip().lower() != "end_header":
            pass

        # Number of 32 bit floats used to encode Spherical Harmonics coefficients.
        # The last multiplication by 3 is because we have 3 components (RGB) for each coefficient.
        sh_coeffs = (sh_degree + 1) * (sh_degree + 1) * 3

        # Position (vec3), normal (vec3), spherical harmonics (sh_coeffs), opacity (float),
        # scale (vec3) and rotation (quaternion). All values are float32 (4 bytes).
        size = count * (3 + 3 + sh_coeffs + 1 + 3 + 4) * 4

        data = f.read(size)
        arr = np.frombuffer(data, dtype=np.float32).reshape((count, -1))

        # Positions.
        position = arr[:, :3].copy()

        # Currently we don't need normals for rendering.
        # normal = arr[:, 3:6].copy()

        # Spherical harmonic coefficients.
        sh = arr[:, 6 : 6 + sh_coeffs].copy()

        # Activate alpha: sigmoid(alpha).
        opacity = 1.0 / (1.0 + np.exp(-arr[:, 6 + sh_coeffs]))

        # Exponentiate scale.
        scale = np.exp(arr[:, 7 + sh_coeffs : 10 + sh_coeffs])

        # Normalize quaternions.
        rotation = arr[:, 10 + sh_coeffs : 14 + sh_coeffs].copy()
        rotation /= np.linalg.norm(rotation, ord=2, axis=1)[..., np.newaxis]

        # Convert from wxyz to xyzw.
        rotation = np.roll(rotation, -1, axis=1)

        return Splats(position, sh, opacity, scale, rotation)


class GaussianSplats(Object3D):
    def __init__(
        self,
        positions: BufferProperty,
        colors: BufferProperty,
        spherical_harmonics: BufferProperty,
        covariances: BufferProperty,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
                "inverse_transform": (np.dtype((np.float32, (3, 4))), 48),
                "splat_count": (np.uint32, 96),
                "alpha_cull_threshold": (np.float32, 100),
                "frustum_dilation": (np.float32, 104),
                "splat_scale": (np.float32, 108),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(name, translation, rotation, scale)
        self.positions = self.add_buffer_property(positions, np.float32, (-1, 3), name="positions")
        self.colors = self.add_buffer_property(colors, np.float32, (-1, 4), name="colors")
        self.spherical_harmonics = self.add_buffer_property(
            spherical_harmonics, np.uint8, (-1, -1), name="spherical-harmonics"
        )
        self.covariances = self.add_buffer_property(covariances, np.float32, (-1, 6), name="covariances")

    def create(self, r: Renderer) -> None:
        self.positions_buf = r.add_gpu_buffer_property(
            self.positions,
            BufferUsageFlags.STORAGE,
            MemoryUsage.COMPUTE_SHADER,
            PipelineStageFlags.COMPUTE_SHADER,
            f"{self.name}-positions",
        )
        self.colors_buf = r.add_gpu_buffer_property(
            self.colors,
            BufferUsageFlags.STORAGE,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_SHADER,
            f"{self.name}-colors",
        )
        self.covariances_buf = r.add_gpu_buffer_property(
            self.covariances,
            BufferUsageFlags.STORAGE,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_SHADER,
            f"{self.name}-covariances",
        )
        self.spherical_harmonics_buf = r.add_gpu_buffer_property(
            self.spherical_harmonics,
            BufferUsageFlags.STORAGE,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_SHADER,
            f"{self.name}-spherical-harmonics",
        )

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

        draw_parameters = np.array([6, max_splats, 0, 0, 0], np.uint32)
        self.draw_parameters_buf = Buffer.from_data(
            r.ctx,
            draw_parameters,
            BufferUsageFlags.INDIRECT | BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name=f"{self.name}-draw-param",
        )

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

        # TODO: actually check the storage type we are using instead of all
        if (r.ctx.device_features & DeviceFeatures.STORAGE_8BIT) == 0:
            raise RuntimeError("Device does not support 8bit storage")

        if (r.ctx.device_features & DeviceFeatures.STORAGE_16BIT) == 0:
            raise RuntimeError("Device does not support 16bit storage")

        # TODO: optionally do mesh shader pipeline if available
        # TODO: optionally do culling in dist if we have indirect draw count
        # TODO: optionally do barycentrics
        # TODO: optionally allow wireframe (if barycentrics are available), disable opacity mode and MS_ANTIALIASING
        # Currently these are compile time, but they should likely be runtime (or we can precompute permutations)
        defines = [
            ("DISTANCE_COMPUTE_WORKGROUP_SIZE", str(DISTANCE_COMPUTE_WORKGROUP_SIZE)),
            ("FRUSTUM_CULLING_MODE", str(FRUSTUM_CULLING_AT_RASTER)),
            ("USE_BARYCENTRIC", "0"),
            ("WIREFRAME", "0"),
            ("DISABLE_OPACITY_GAUSSIAN", "0"),
            ("MS_ANTIALIASING", "0"), # Only useful when model comes from mip splatting
            ("MAX_SH_DEGREE", "3"),
            ("SH_FORMAT", str(FORMAT_UINT8)),
            ("ORTHOGRAPHIC_MODE", "0"),
            ("POINT_CLOUD_MODE", "0"),
            ("SHOW_SH_ONLY", "0"),
        ]

        dist_shader = r.compile_builtin_shader("vk_3d_gaussian_splatting/dist.comp.slang", defines=defines)
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
            s.write_buffer(self.draw_parameters_buf, DescriptorType.STORAGE_BUFFER, 2)

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
            r, SortOptions(key_type=SortDataType.UINT32, payload_type=SortDataType.UINT32, descending=False)
        )

        vert = r.compile_builtin_shader("vk_3d_gaussian_splatting/threedgs_raster.vert.slang", defines=defines)
        frag = r.compile_builtin_shader("vk_3d_gaussian_splatting/threedgs_raster.frag.slang", defines=defines)

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=[
                VertexBinding(0, 8, VertexInputRate.VERTEX),
                VertexBinding(1, 4, VertexInputRate.INSTANCE),
            ],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32_SFLOAT),
                VertexAttribute(1, 1, Format.R32_UINT),
            ],
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

    def upload(self, r: Renderer, frame: RendererFrame):
        num_splats = self.positions.get_current().shape[0]
        self.constants["splat_count"] = num_splats
        self.constants["alpha_cull_threshold"] = 1.0 / 255.0
        self.constants["frustum_dilation"] = 0.2
        self.constants["splat_scale"] = 1.0

        self.descriptor_set = self.descriptor_sets.get_current_and_advance()
        self.descriptor_set.write_buffer(self.positions_buf.get_current(), DescriptorType.STORAGE_BUFFER, 3)
        self.descriptor_set.write_buffer(self.colors_buf.get_current(), DescriptorType.STORAGE_BUFFER, 4)
        self.descriptor_set.write_buffer(self.covariances_buf.get_current(), DescriptorType.STORAGE_BUFFER, 5)
        self.descriptor_set.write_buffer(self.spherical_harmonics_buf.get_current(), DescriptorType.STORAGE_BUFFER, 6)

        # Distances
        frame.cmd.bind_compute_pipeline(
            self.dist_pipeline,
            descriptor_sets=[frame.scene_descriptor_set, self.descriptor_set],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.dispatch(div_round_up(num_splats, DISTANCE_COMPUTE_WORKGROUP_SIZE), 1, 1)

        frame.cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

        # Sort
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
        # Draw indirect
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                self.quad_vertices,
                self.sorted_indices_buf,
            ],
            index_buffer=self.quad_indices,
            descriptor_sets=[
                scene_descriptor_set,
                self.descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.draw_indexed_indirect(self.draw_parameters_buf, 0, 1, 20)


splats = load_ply(sys.argv[1], 3)

if False:
    positions = np.array([[1, 0, 0]], np.float32)
    colors = np.array([[0.5, 0, 0.0, 1]], np.float32)
    sh = np.zeros((1, 13), np.uint8)
    covariances = np.array([[1, 0, 0, 2, 0, 3]], np.float32) * 1
else:
    positions = splats.position

    SH_C0 = 0.28209479177387814
    colors = np.hstack((splats.sh[:, :3] * SH_C0 + 0.5, splats.opacity[..., np.newaxis]))
    sh = np.transpose(np.clip(np.rint((splats.sh[:, 3:] * 0.5 + 0.5) * 255.0), 0.0, 255.0).astype(np.uint8).reshape((-1, 3, 15)), (0, 2, 1)).reshape((-1, 45))

    rots = Rotation.from_quat(splats.rotation).as_matrix()
    scale = np.empty((splats.scale.shape[0], 3, 3), np.float32)
    scale[:, 0, 0] = splats.scale[:, 0]
    scale[:, 1, 1] = splats.scale[:, 1]
    scale[:, 2, 2] = splats.scale[:, 2]

    covariance_matrices: np.ndarray = np.matmul(rots, scale)
    transformed_covariance: np.ndarray = np.matmul(covariance_matrices, np.transpose(covariance_matrices, (0, 2, 1)))
    covariances = transformed_covariance.reshape((-1, 9))[:, (0, 1, 2, 4, 5, 8)]


gs = GaussianSplats(positions, colors, sh, covariances)

v = Viewer(
    config=Config(
        gui=GuiConfig(
            stats=True,
            inspector=True,
        ),
        world_up=(0, -1, 0),
        renderer=RendererConfig(
            background_color=(0, 0, 0, 1),
        ),
    )
)
v.viewport.scene.objects.append(gs)

positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    np.float32,
)

colors = np.array(
    [
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
        0xFFFF0000,
        0xFFFF0000,
    ],
    np.uint32,
)

line_width = 1

line = Lines(positions, colors, line_width)

v.viewport.scene.objects.extend([line])

v.run()
