# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from pyglm.glm import inverse, normalize, orthoRH_ZO, quatLookAtRH, vec3, vec4
from pyxpg import (
    AccessFlags,
    AllocType,
    Buffer,
    BufferUsageFlags,
    ComputePipeline,
    Context,
    DepthAttachment,
    DescriptorSetBinding,
    DescriptorType,
    Filter,
    Format,
    Image,
    ImageAspectFlags,
    ImageBarrier,
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    ImageViewType,
    LoadOp,
    MemoryUsage,
    PipelineStageFlags,
    PushConstantsRange,
    Shader,
    StoreOp,
)

from .property import BufferProperty
from .renderer_frame import RendererFrame
from .scene import Object, Object3D
from .utils.descriptors import (
    create_descriptor_layout_pool_and_set,
    create_descriptor_layout_pool_and_sets,
    create_descriptor_pool_and_sets_ringbuffer,
)
from .utils.gpu import UploadableBuffer, view_bytes
from .utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from .renderer import Renderer


class LightTypes(Enum):
    DIRECTIONAL = 0


@dataclass
class LightInfo:
    size: int


directional_light_dtype = np.dtype(
    {
        "orthographic_camera": (np.dtype((np.float32, (4, 4))), 0),
        "radiance": (np.dtype((np.float32, (3,))), 64),
        "shadow_map_index": (np.dtype((np.int32, (1,))), 76),
        "direction": (np.dtype((np.float32, (3,))), 80),
        "bias": (np.dtype((np.float32, (1,))), 92),
    }
)  # type: ignore

# When adding a new light type, this also has to be added with a matching type to "shaders/2d/scene.slang" and "shaders/3d/scene.slang"
LIGHT_TYPES_INFO = [
    LightInfo(directional_light_dtype.itemsize),
]


class Light(Object3D):
    def render_shadow_maps(self, renderer: "Renderer", frame: RendererFrame, objects: List[Object]) -> None:
        pass


class PointLight(Light):
    def __init__(
        self,
        intensity: BufferProperty,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale, enabled=enabled)
        self.intensity = self.add_buffer_property(intensity, np.float32, (-1, 3), name="intensity")


class SpotLight(Light):
    def __init__(
        self,
        intensity: BufferProperty,
        stop_cosine: BufferProperty,
        falloff_start_cosine: BufferProperty,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale, enabled=enabled)
        self.intensity = self.add_buffer_property(intensity, np.float32, (-1, 3), name="intensity")
        self.stop_cosine = self.add_buffer_property(stop_cosine, np.float32, (-1, 3), name="stop_cosine")
        self.falloff_start_cosine = self.add_buffer_property(
            falloff_start_cosine, np.float32, (-1, 3), name="falloff_start_cosine"
        )


@dataclass(frozen=True)
class DirectionalShadowSettings:
    casts_shadow: bool = True
    shadow_map_size: int = 2048
    half_extent: float = 100.0
    z_near: float = 0.0
    z_far: float = 1000.0
    bias: float = 0.01


class DirectionalLight(Light):
    def __init__(
        self,
        radiance: BufferProperty,
        shadow_settings: Optional[DirectionalShadowSettings] = None,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale, enabled=enabled)
        self.radiance = self.add_buffer_property(radiance, np.float32, (3,), name="radiance")
        self.shadow_settings = shadow_settings if shadow_settings is not None else DirectionalShadowSettings()
        self.shadow_map: Optional[Image] = None

    @classmethod
    def look_at(
        cls,
        position: vec3,
        target: vec3,
        world_up: vec3,
        radiance: BufferProperty,
        shadow_settings: Optional[DirectionalShadowSettings] = None,
        name: Optional[str] = None,
        scale: Optional[BufferProperty] = None,
    ) -> "DirectionalLight":
        rotation = quatLookAtRH(normalize(target - position), world_up)
        return cls(
            radiance,
            shadow_settings,
            name,
            translation=np.asarray(position),  # type: ignore
            rotation=np.asarray(rotation),  # type: ignore
            scale=scale,
        )

    def create(self, r: "Renderer") -> None:
        if self.shadow_settings.casts_shadow:
            self.shadow_map = Image(
                r.ctx,
                self.shadow_settings.shadow_map_size,
                self.shadow_settings.shadow_map_size,
                r.shadow_map_format,
                ImageUsageFlags.SAMPLED | ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT,
                AllocType.DEVICE,
                name=f"{self.name}-shadow_map",
            )
            self.shadow_map_viewport = [
                0,
                0,
                self.shadow_settings.shadow_map_size,
                self.shadow_settings.shadow_map_size,
            ]

            self.descriptor_pool, self.descriptor_sets = create_descriptor_pool_and_sets_ringbuffer(
                r.ctx, r.scene_depth_descriptor_set_layout, r.num_frames_in_flight, name="scene-descriptors"
            )

            constants_dtype = np.dtype(
                {
                    "camera_matrix": (np.dtype((np.float32, (4, 4))), 0),
                }
            )  # type: ignore

            self.constants = np.zeros((1,), constants_dtype)

            self.uniform_buffers = RingBuffer(
                [
                    UploadableBuffer(r.ctx, constants_dtype.itemsize, BufferUsageFlags.UNIFORM)
                    for _ in range(r.num_frames_in_flight)
                ]
            )
            for set, buf in zip(self.descriptor_sets, self.uniform_buffers):
                set.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        # TODO: this and other matrices should be using config to know what is the deafult data handedness
        # we should also have default front/back face winding andr require dynamic state for culling mode.
        self.projection = orthoRH_ZO(
            -self.shadow_settings.half_extent,
            self.shadow_settings.half_extent,
            self.shadow_settings.half_extent,
            -self.shadow_settings.half_extent,
            self.shadow_settings.z_near,
            self.shadow_settings.z_far,
        )

        self.light_info = np.zeros((1,), directional_light_dtype)

    def upload_light(
        self, renderer: "Renderer", frame: RendererFrame, shadow_map_index: int, offset: int, buffer: UploadableBuffer
    ) -> None:
        view = inverse(self.current_transform_matrix)
        direction = vec3(self.current_transform_matrix * vec4(0, 0, -1, 0))
        self.light_info["orthographic_camera"] = self.projection * view
        self.light_info["radiance"] = self.radiance.get_current()
        self.light_info["shadow_map_index"] = shadow_map_index
        self.light_info["direction"] = direction
        self.light_info["bias"] = self.shadow_settings.bias

        buffer.upload(frame.cmd, MemoryUsage.NONE, view_bytes(self.light_info), offset)

    def upload(self, renderer: "Renderer", frame: RendererFrame) -> None:
        if not self.shadow_settings.casts_shadow:
            return

        assert self.shadow_map is not None
        view = inverse(self.current_transform_matrix)
        self.constants["camera_matrix"] = self.projection * view
        buf = self.uniform_buffers.get_current_and_advance()
        buf.upload(
            frame.cmd,
            MemoryUsage.NONE,  # Synchronized by automatic after-upload barrier
            self.constants.view(np.uint8),
        )
        frame.upload_property_pipeline_stages |= PipelineStageFlags.VERTEX_SHADER

        frame.upload_after_image_barriers.append(
            ImageBarrier(
                self.shadow_map,
                ImageLayout.UNDEFINED,
                ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                PipelineStageFlags.FRAGMENT_SHADER,
                AccessFlags.SHADER_SAMPLED_READ,
                PipelineStageFlags.EARLY_FRAGMENT_TESTS,
                AccessFlags.DEPTH_STENCIL_ATTACHMENT_READ | AccessFlags.DEPTH_STENCIL_ATTACHMENT_WRITE,
                aspect_mask=ImageAspectFlags.DEPTH,
            )
        )

        frame.before_render_image_barriers.append(
            ImageBarrier(
                self.shadow_map,
                ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                PipelineStageFlags.LATE_FRAGMENT_TESTS,
                AccessFlags.DEPTH_STENCIL_ATTACHMENT_READ | AccessFlags.DEPTH_STENCIL_ATTACHMENT_WRITE,
                PipelineStageFlags.FRAGMENT_SHADER,
                AccessFlags.SHADER_SAMPLED_READ,
                aspect_mask=ImageAspectFlags.DEPTH,
            )
        )

    def render_shadow_maps(self, renderer: "Renderer", frame: RendererFrame, objects: List[Object]) -> None:
        if not self.shadow_settings.casts_shadow:
            return

        assert self.shadow_map is not None

        descriptor_set = self.descriptor_sets.get_current_and_advance()

        frame.cmd.set_viewport(self.shadow_map_viewport)
        frame.cmd.set_scissors(self.shadow_map_viewport)
        with frame.cmd.rendering(
            self.shadow_map_viewport, [], DepthAttachment(self.shadow_map, LoadOp.CLEAR, StoreOp.STORE, 1.0)
        ):
            for o in objects:
                o.render_depth(renderer, frame, descriptor_set)


class UniformEnvironmentLight(Light):
    def __init__(
        self,
        radiance: BufferProperty,
        name: Optional[str] = None,
        enabled: Optional[BufferProperty] = None,
    ):
        super().__init__(name, enabled=enabled)
        self.radiance = self.add_buffer_property(radiance, np.float32, (3,), name="radiance")


@dataclass(frozen=True)
class IBLParams:
    # Skybox
    skybox_size: int = 1024

    # Irradiance
    irradiance_size: int = 32
    irradiance_samples_phi: int = 256
    irradiance_samples_theta: int = 64

    # Specular
    specular_size: int = 1024
    specular_mips: int = 5
    specular_samples: int = 1024


@dataclass
class EnvironmentCubemaps:
    irradiance_cubemap: NDArray[np.float16]
    specular_cubemap: List[NDArray[np.float16]]

    @classmethod
    def load(cls, file: Any, **kwargs: Any) -> "EnvironmentCubemaps":
        arrays = np.load(file, allow_pickle=False, **kwargs)
        irradiance_cubemap = arrays["irradiance"]
        specular_cubemap = []
        for i in range(16):
            k = f"specular_{i}"
            if k not in arrays:
                break
            specular_cubemap.append(arrays[k])
        if len(specular_cubemap) == 0:
            raise KeyError('no specular cubemap ("specular_0") found in the archive')
        return cls(irradiance_cubemap, specular_cubemap)

    def save(self, file: Any, **kwargs: Any) -> None:
        np.savez(
            file,
            allow_pickle=False,
            irradiance=self.irradiance_cubemap,
            **{f"specular_{i}": s for i, s in enumerate(self.specular_cubemap)},
            **kwargs,
        )

    def gpu(self, ctx: Context) -> "GpuEnvironmentCubemaps":
        # Create cubemaps
        irradiance_size = self.irradiance_cubemap.shape[1]
        irradiance_cubemap = Image(
            ctx,
            irradiance_size,
            irradiance_size,
            Format.R16G16B16A16_SFLOAT,
            ImageUsageFlags.SAMPLED
            | ImageUsageFlags.STORAGE
            | ImageUsageFlags.TRANSFER_SRC
            | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            array_layers=6,
            is_cube=True,
        )
        specular_cubemap = Image(
            ctx,
            self.specular_cubemap[0].shape[2],
            self.specular_cubemap[0].shape[1],
            Format.R16G16B16A16_SFLOAT,
            ImageUsageFlags.SAMPLED
            | ImageUsageFlags.STORAGE
            | ImageUsageFlags.TRANSFER_SRC
            | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            array_layers=6,
            mip_levels=len(self.specular_cubemap),
            is_cube=True,
        )

        # Create upload buffers
        irradiance_buffer = Buffer.from_data(
            ctx, self.irradiance_cubemap, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST
        )
        specular_buffers = [
            Buffer.from_data(ctx, a, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST) for a in self.specular_cubemap
        ]

        # Upload
        with ctx.sync_commands() as cmd:
            cmd.image_barrier(
                irradiance_cubemap, ImageLayout.TRANSFER_DST_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_DST
            )
            cmd.copy_buffer_to_image_range(
                irradiance_buffer, irradiance_cubemap, irradiance_size, irradiance_size, image_layer_count=6
            )
            cmd.image_barrier(
                irradiance_cubemap,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                MemoryUsage.TRANSFER_DST,
                MemoryUsage.SHADER_READ_ONLY,
            )

            cmd.image_barrier(
                specular_cubemap, ImageLayout.TRANSFER_DST_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_DST
            )
            for m, (b, a) in enumerate(zip(specular_buffers, self.specular_cubemap)):
                cmd.copy_buffer_to_image_range(
                    b, specular_cubemap, a.shape[1], a.shape[1], image_mip=m, image_layer_count=6
                )
            cmd.image_barrier(
                specular_cubemap,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                MemoryUsage.TRANSFER_DST,
                MemoryUsage.SHADER_READ_ONLY,
            )

        return GpuEnvironmentCubemaps(irradiance_cubemap, specular_cubemap)


@dataclass
class GpuEnvironmentCubemaps:
    irradiance_cubemap: Image
    specular_cubemap: Image

    def cpu(self, ctx: Context) -> EnvironmentCubemaps:
        # Create buffers
        irradiance_size = self.irradiance_cubemap.width
        irradiance_shape = (6, irradiance_size, irradiance_size, 4)
        irradiance_buffer = Buffer(ctx, np.prod(irradiance_shape) * 2, BufferUsageFlags.TRANSFER_DST, AllocType.HOST)  # type: ignore

        specular_size = self.specular_cubemap.width
        specular_mips = self.specular_cubemap.mip_levels
        specular_buffers = []
        for m in range(specular_mips):
            specular_mip_size = specular_size >> m
            specular_mip_shape = (6, specular_mip_size, specular_mip_size, 4)
            specular_buffers.append(
                Buffer(ctx, np.prod(specular_mip_shape) * 2, BufferUsageFlags.TRANSFER_DST, AllocType.HOST)  # type: ignore
            )

        # Readback
        with ctx.sync_commands() as cmd:
            cmd.image_barrier(
                self.irradiance_cubemap, ImageLayout.TRANSFER_SRC_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_SRC
            )
            cmd.copy_image_to_buffer_range(
                self.irradiance_cubemap, irradiance_buffer, irradiance_size, irradiance_size, image_layer_count=6
            )
            cmd.image_barrier(
                self.irradiance_cubemap,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                MemoryUsage.TRANSFER_SRC,
                MemoryUsage.SHADER_READ_ONLY,
            )

            cmd.image_barrier(
                self.specular_cubemap, ImageLayout.TRANSFER_SRC_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_SRC
            )
            for m, b in enumerate(specular_buffers):
                specular_mip_size = specular_size >> m
                cmd.copy_image_to_buffer_range(
                    self.specular_cubemap, b, specular_mip_size, specular_mip_size, image_mip=m, image_layer_count=6
                )
            cmd.image_barrier(
                self.specular_cubemap,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                MemoryUsage.TRANSFER_SRC,
                MemoryUsage.SHADER_READ_ONLY,
            )
            cmd.memory_barrier(MemoryUsage.TRANSFER_SRC, MemoryUsage.HOST_READ)

        # Copy into numpy arrays
        irradiance_array = np.frombuffer(irradiance_buffer.data, np.float16).copy().reshape(irradiance_shape)
        specular_arrays: List[NDArray[np.float16]] = []
        for m, b in enumerate(specular_buffers):
            specular_mip_size = specular_size >> m
            specular_mip_shape = (6, specular_mip_size, specular_mip_size, 4)
            specular_arrays.append(np.frombuffer(b, np.float16).copy().reshape(specular_mip_shape))

        return EnvironmentCubemaps(irradiance_array, specular_arrays)


class GGXLUTPipeline:
    def __init__(self, r: "Renderer"):
        self.lut_constants_dtype = np.dtype(
            {
                "samples": (np.uint32, 0),
            }
        )  # type: ignore
        self.lut_constants = np.zeros((1,), self.lut_constants_dtype)
        self.lut_shader = r.compile_builtin_shader("3d/ibl.slang", "entry_lut")
        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_set = create_descriptor_layout_pool_and_set(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
            ],
        )
        self.lut_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.lut_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.lut_constants_dtype.itemsize)],
        )

    def run(self, r: "Renderer", width: int = 512, height: int = 512, samples: int = 1024) -> Image:
        lut = Image(
            r.ctx,
            width,
            height,
            Format.R16G16_SFLOAT,
            ImageUsageFlags.SAMPLED
            | ImageUsageFlags.STORAGE
            | ImageUsageFlags.TRANSFER_SRC
            | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
        )

        self.lut_constants["samples"] = samples

        descriptor_set = self.descriptor_set
        descriptor_set.write_image(lut, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)

        with r.ctx.sync_commands() as cmd:
            cmd.image_barrier(lut, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.COMPUTE_SHADER, undefined=True)
            cmd.bind_compute_pipeline(
                self.lut_pipeline, descriptor_sets=[descriptor_set], push_constants=self.lut_constants.tobytes()
            )
            cmd.dispatch((width + 7) // 8, (height + 7) // 8, 6)
            cmd.image_barrier(lut, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.COMPUTE_SHADER, MemoryUsage.ALL)

        return lut


class IBLPipeline:
    def __init__(self, r: "Renderer"):
        # Common
        self.max_specular_mips = 16
        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_sets = (
            create_descriptor_layout_pool_and_sets(
                r.ctx,
                [
                    DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
                    DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
                ],
                self.max_specular_mips,
            )
        )

        # Skybox
        self.skybox_constants_dtype = np.dtype(
            {
                "size": (np.uint32, 0),
            }
        )  # type: ignore
        self.skybox_constants = np.zeros((1,), self.skybox_constants_dtype)
        self.skybox_shader = r.compile_builtin_shader("3d/ibl.slang", "entry_skybox")
        self.skybox_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.skybox_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.skybox_constants.itemsize)],
        )

        # Irradiance
        self.irradiance_constants_dtype = np.dtype(
            {
                "size": (np.uint32, 0),
                "samples_phi": (np.uint32, 4),
                "samples_theta": (np.uint32, 8),
            }
        )  # type: ignore
        self.irradiance_constants = np.zeros((1,), self.irradiance_constants_dtype)
        self.irradiance_shader = r.compile_builtin_shader("3d/ibl.slang", "entry_irradiance")
        self.irradiance_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.irradiance_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.irradiance_constants.itemsize)],
        )

        # Specular
        self.specular_constants_dtype = np.dtype(
            {
                "size": (np.uint32, 0),
                "samples": (np.uint32, 4),
                "roughness": (np.float32, 8),
                "skybox_resolution": (np.float32, 12),
            }
        )  # type: ignore
        self.specular_constants = np.zeros((1,), self.specular_constants_dtype)
        self.specular_shader = r.compile_builtin_shader("3d/ibl.slang", "entry_specular")
        self.specular_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.specular_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.specular_constants_dtype.itemsize)],
        )

    def run(self, r: "Renderer", equirectangular: Image, params: IBLParams) -> GpuEnvironmentCubemaps:
        skybox_size = params.skybox_size
        skybox_mip_levels = skybox_size.bit_length()
        skybox_cubemap = Image(
            r.ctx,
            skybox_size,
            skybox_size,
            Format.R16G16B16A16_SFLOAT,
            ImageUsageFlags.SAMPLED
            | ImageUsageFlags.STORAGE
            | ImageUsageFlags.TRANSFER_SRC
            | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            array_layers=6,
            mip_levels=skybox_mip_levels,
            is_cube=True,
        )

        # Skybox
        descriptor_set = self.descriptor_sets[0]
        self.skybox_constants["size"] = params.skybox_size
        cubemap_array_view = ImageView(r.ctx, skybox_cubemap, ImageViewType.TYPE_2D_ARRAY, mip_level_count=1)
        descriptor_set.write_image(cubemap_array_view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)
        descriptor_set.write_combined_image_sampler(
            equirectangular, ImageLayout.SHADER_READ_ONLY_OPTIMAL, r.linear_sampler, 1
        )

        with r.ctx.sync_commands() as cmd:
            cmd.image_barrier(
                skybox_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.COMPUTE_SHADER, undefined=True
            )
            cmd.bind_compute_pipeline(
                self.skybox_pipeline, descriptor_sets=[descriptor_set], push_constants=self.skybox_constants.tobytes()
            )
            cmd.dispatch((skybox_size + 7) // 8, (skybox_size + 7) // 8, 6)
            cmd.image_barrier(skybox_cubemap, ImageLayout.GENERAL, MemoryUsage.COMPUTE_SHADER, MemoryUsage.ALL)
            for m in range(1, skybox_mip_levels):
                src_size = skybox_size >> m - 1
                dst_size = skybox_size >> m
                cmd.blit_image_range(
                    skybox_cubemap,
                    skybox_cubemap,
                    src_size,
                    src_size,
                    dst_size,
                    dst_size,
                    Filter.LINEAR,
                    src_mip_level=m - 1,
                    dst_mip_level=m,
                    src_array_layer_count=6,
                    dst_array_layer_count=6,
                )
                cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.ALL)
            cmd.image_barrier(
                skybox_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.COMPUTE_SHADER, MemoryUsage.ALL
            )

        # Irradiance
        irradiance_size = params.irradiance_size
        irradiance_cubemap = Image(
            r.ctx,
            irradiance_size,
            irradiance_size,
            Format.R16G16B16A16_SFLOAT,
            ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_SRC,
            AllocType.DEVICE,
            array_layers=6,
            is_cube=True,
        )

        self.irradiance_constants["size"] = irradiance_size
        self.irradiance_constants["samples_phi"] = params.irradiance_samples_phi
        self.irradiance_constants["samples_theta"] = params.irradiance_samples_theta

        cubemap_array_view = ImageView(r.ctx, irradiance_cubemap, ImageViewType.TYPE_2D_ARRAY, mip_level_count=1)
        descriptor_set.write_image(cubemap_array_view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)
        descriptor_set.write_combined_image_sampler(
            skybox_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, r.linear_sampler, 1
        )

        with r.ctx.sync_commands() as cmd:
            cmd.image_barrier(
                irradiance_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.COMPUTE_SHADER, undefined=True
            )
            cmd.bind_compute_pipeline(
                self.irradiance_pipeline,
                descriptor_sets=[descriptor_set],
                push_constants=self.irradiance_constants.tobytes(),
            )
            cmd.dispatch((irradiance_size + 7) // 8, (irradiance_size + 7) // 8, 6)
            cmd.image_barrier(
                irradiance_cubemap,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                MemoryUsage.COMPUTE_SHADER,
                MemoryUsage.ALL,
                undefined=False,
            )

        # Specular
        specular_mips = min(self.max_specular_mips, params.specular_mips, params.specular_size.bit_length())
        specular_size = params.specular_size
        specular_cubemap = Image(
            r.ctx,
            params.specular_size,
            params.specular_size,
            Format.R16G16B16A16_SFLOAT,
            ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_SRC,
            AllocType.DEVICE,
            array_layers=6,
            mip_levels=specular_mips,
            is_cube=True,
        )

        self.specular_constants["samples"] = params.specular_samples
        self.specular_constants["skybox_resolution"] = skybox_size
        views = []
        with r.ctx.sync_commands() as cmd:
            cmd.image_barrier(
                specular_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.COMPUTE_SHADER, undefined=True
            )

            for m in range(specular_mips):
                descriptor_set = self.descriptor_sets[m]
                specular_mip_view = ImageView(
                    r.ctx, specular_cubemap, ImageViewType.TYPE_2D_ARRAY, base_mip_level=m, mip_level_count=1
                )
                descriptor_set.write_image(specular_mip_view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)
                descriptor_set.write_combined_image_sampler(
                    skybox_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, r.linear_sampler, 1
                )
                views.append(specular_mip_view)

                mip_size = specular_size >> m
                self.specular_constants["size"] = mip_size
                self.specular_constants["roughness"] = m / (specular_mips - 1)
                cmd.bind_compute_pipeline(
                    self.specular_pipeline,
                    descriptor_sets=[descriptor_set],
                    push_constants=self.specular_constants.tobytes(),
                )
                cmd.dispatch((mip_size + 7) // 8, (mip_size + 7) // 8, 6)

            cmd.image_barrier(specular_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.ALL, MemoryUsage.ALL)

        return GpuEnvironmentCubemaps(irradiance_cubemap, specular_cubemap)


class EnvironmentLight(Light):
    def __init__(
        self,
        equirectangular: Optional[NDArray[np.float32]] = None,
        cubemaps: Optional[EnvironmentCubemaps] = None,
        ibl_params: Optional[IBLParams] = None,
        name: Optional[str] = None,
        enabled: Optional[BufferProperty] = None,
    ):
        if not ((equirectangular is None) ^ (cubemaps is None)):
            raise RuntimeError('Exactly one of "equirectangular" and "cubemaps" must not be None')

        if equirectangular is not None:
            height, width, channels = equirectangular.shape
            if not channels == 3 or not (equirectangular.dtype == np.float32 or equirectangular.dtype == np.float64):
                raise ValueError(
                    f"Equirectangular map must have 3 channels and be of float32 or float64 dtype. Got {channels} channels and {equirectangular.dtype} dtype."
                )
            # TODO: check if RGB_F32 is supported instead of defaulting to RGBA_F32
            rgb_data = equirectangular.astype(np.float32, copy=False)
            equirectangular = np.dstack((rgb_data, np.ones((height, width, 1), np.float32)))

        self.equirectangular = equirectangular
        self.cubemaps = cubemaps
        self.ibl_params = ibl_params
        super().__init__(name, enabled=enabled)

    def create(self, r: "Renderer") -> None:
        if self.equirectangular is not None:
            height, width, _channels = self.equirectangular.shape
            equirectangular_img = Image.from_data(
                r.ctx,
                self.equirectangular,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                width,
                height,
                Format.R32G32B32A32_SFLOAT,
                ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST,
                AllocType.DEVICE,
            )
            self.gpu_cubemaps = r.run_ibl_pipeline(equirectangular_img, self.ibl_params)
        else:
            assert self.cubemaps is not None
            self.gpu_cubemaps = self.cubemaps.gpu(r.ctx)

    @classmethod
    def from_equirectangular(
        cls,
        equirectangular: NDArray[np.float32],
        ibl_params: Optional[IBLParams] = None,
    ) -> "EnvironmentLight":
        return cls(equirectangular=equirectangular, ibl_params=ibl_params)

    @classmethod
    def from_cubemaps(cls, cubemaps: EnvironmentCubemaps) -> "EnvironmentLight":
        return cls(cubemaps=cubemaps)


# class AreaLight(Light):
#     pass
