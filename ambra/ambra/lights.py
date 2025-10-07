# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np
from pyglm.glm import inverse, orthoRH_ZO, vec3, vec4
from pyxpg import (
    AllocType,
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
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    ImageViewType,
    LoadOp,
    MemoryUsage,
    PushConstantsRange,
    Shader,
    StoreOp,
)

from . import renderer
from .property import BufferProperty, view_bytes
from .renderer_frame import RendererFrame
from .scene import Object3D, Scene
from .utils.descriptors import create_descriptor_layout_pool_and_sets, create_descriptor_pool_and_sets_ringbuffer
from .utils.gpu import UploadableBuffer
from .utils.ring_buffer import RingBuffer


class LightTypes(Enum):
    DIRECTIONAL = 0


@dataclass
class LightInfo:
    size: int


directional_light_dtype = np.dtype(
    {
        "orthographic_camera": (np.dtype((np.float32, (4, 4))), 0),
        "radiance": (np.dtype((np.float32, (3,))), 64),
        "shadowmap_index": (np.dtype((np.int32, (1,))), 76),
        "direction": (np.dtype((np.float32, (3,))), 80),
        "bias": (np.dtype((np.float32, (1,))), 92),
    }
)  # type: ignore

# When adding a new light type, this also has to be added with a matching type to "shaders/2d/scene.slang" and "shaders/3d/scene.slang"
LIGHT_TYPES_INFO = [
    LightInfo(directional_light_dtype.itemsize),
]


class Light(Object3D):
    def render_shadowmaps(self, renderer: "renderer.Renderer", frame: RendererFrame, scene: Scene) -> None:
        pass


class PointLight(Light):
    def __init__(
        self,
        intensity: Union[BufferProperty, np.ndarray],
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale)
        self.intensity = self.add_buffer_property(intensity, np.float32, (-1, 3), name="intensity")


class SpotLight(Light):
    def __init__(
        self,
        intensity: Union[BufferProperty, np.ndarray],
        stop_cosine: Union[BufferProperty, np.ndarray],
        falloff_start_cosine: Union[BufferProperty, np.ndarray],
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale)
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
        radiance: Union[BufferProperty, np.ndarray],
        shadow_settings: Optional[DirectionalShadowSettings] = None,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale)
        self.radiance = self.add_buffer_property(radiance, np.float32, (3,), name="radiance")
        self.shadow_settings = shadow_settings if shadow_settings is not None else DirectionalShadowSettings()
        self.shadow_map: Optional[Image] = None

    def create(self, r: "renderer.Renderer") -> None:
        if self.shadow_settings.casts_shadow:
            self.shadow_map = Image(
                r.ctx,
                self.shadow_settings.shadow_map_size,
                self.shadow_settings.shadow_map_size,
                r.shadowmap_format,
                ImageUsageFlags.SAMPLED | ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT,
                AllocType.DEVICE,
                name=f"{self.name}-shadowmap",
            )
            self.shadow_map_viewport = [
                0,
                0,
                self.shadow_settings.shadow_map_size,
                self.shadow_settings.shadow_map_size,
            ]

            self.descriptor_pool, self.descriptor_sets = create_descriptor_pool_and_sets_ringbuffer(
                r.ctx, r.scene_depth_descriptor_set_layout, r.window.num_frames, name="scene-descriptors"
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
                    for _ in range(r.window.num_frames)
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

        self.light_buffer_offset, self.shadowmap_index = r.add_light(LightTypes.DIRECTIONAL, self.shadow_map)
        self.light_info = np.zeros((1,), directional_light_dtype)

    def upload(self, renderer: "renderer.Renderer", frame: RendererFrame) -> None:
        view = inverse(self.current_transform_matrix)
        direction = vec3(self.current_transform_matrix * vec4(0, 0, -1, 0))
        self.light_info["orthographic_camera"] = self.projection * view
        self.light_info["radiance"] = self.radiance.get_current()
        self.light_info["shadowmap_index"] = self.shadowmap_index
        self.light_info["direction"] = direction
        self.light_info["bias"] = self.shadow_settings.bias
        renderer.upload_light(frame, LightTypes.DIRECTIONAL, view_bytes(self.light_info), self.light_buffer_offset)

    def render_shadowmaps(self, renderer: "renderer.Renderer", frame: RendererFrame, scene: Scene) -> None:
        if not self.shadow_settings.casts_shadow:
            return

        assert self.shadow_map is not None

        view = inverse(self.current_transform_matrix)
        self.constants["camera_matrix"] = self.projection * view

        set = self.descriptor_sets.get_current_and_advance()
        buf = self.uniform_buffers.get_current_and_advance()

        buf.upload(
            frame.cmd,
            MemoryUsage.ANY_SHADER_UNIFORM,
            self.constants.view(np.uint8),
        )

        frame.cmd.image_barrier(
            self.shadow_map,
            ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            MemoryUsage.SHADER_READ_ONLY,
            MemoryUsage.DEPTH_STENCIL_ATTACHMENT,
            aspect_mask=ImageAspectFlags.DEPTH,
            undefined=True,
        )

        frame.cmd.set_viewport(self.shadow_map_viewport)
        frame.cmd.set_scissors(self.shadow_map_viewport)
        with frame.cmd.rendering(
            self.shadow_map_viewport, [], DepthAttachment(self.shadow_map, LoadOp.CLEAR, StoreOp.STORE, 1.0)
        ):
            scene.render_depth(renderer, frame, set)

        frame.cmd.image_barrier(
            self.shadow_map,
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            MemoryUsage.DEPTH_STENCIL_ATTACHMENT,
            MemoryUsage.SHADER_READ_ONLY,
            aspect_mask=ImageAspectFlags.DEPTH,
        )


class UniformEnvironmentLight(Light):
    def __init__(
        self,
        radiance: Union[BufferProperty, np.ndarray],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.radiance = self.add_buffer_property(radiance, np.float32, (3,), name="radiance")

    def create(self, r: "renderer.Renderer") -> None:
        r.add_uniform_environment_light(self)


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
    skybox_cubemap: np.ndarray
    irradiance_cubemap: np.ndarray
    specular_cubemap: List[np.ndarray]

    @classmethod
    def load(cls, file: Any, **kwargs) -> "EnvironmentCubemaps":
        arrays = np.load(file, allow_pickle=False, **kwargs)
        return cls(arrays["skybox_cubemap"], arrays["irradiance_cubemap"], arrays["specular_cubemap"])

    def save(self, file: Any, **kwargs) -> None:
        np.savez(
            file,
            allow_pickle=False,
            skybox_cubemap=self.skybox_cubemap,
            irradiance=self.irradiance_cubemap,
            specular=self.specular_cubemap,
        )

    def gpu(self, ctx: Context) -> "GpuEnvironmentCubemaps":
        # TODO: upload
        pass


@dataclass
class GpuEnvironmentCubemaps:
    skybox_cubemap: Image
    irradiance_cubemap: Image
    specular_cubemap: Image
    specular_mips: int

    def cpu(self, ctx: Context) -> EnvironmentCubemaps:
        # TODO: readback
        pass


class IBLPipeline:
    def __init__(self, r: "renderer.Renderer"):
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
        self.skybox_shader = r.get_builtin_shader("3d/ibl.slang", "entry_skybox")
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
        self.irradiance_shader = r.get_builtin_shader("3d/ibl.slang", "entry_irradiance")
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
        self.specular_shader = r.get_builtin_shader("3d/ibl.slang", "entry_specular")
        self.specular_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.specular_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.specular_constants_dtype.itemsize)],
        )

    def run(self, r: "renderer.Renderer", equirectangular: Image, params: IBLParams) -> GpuEnvironmentCubemaps:
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
            cmd.image_barrier(skybox_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, undefined=True)
            cmd.bind_compute_pipeline(
                self.skybox_pipeline, descriptor_sets=[descriptor_set], push_constants=self.skybox_constants.tobytes()
            )
            cmd.dispatch((skybox_size + 7) // 8, (skybox_size + 7) // 8, 6)
            cmd.image_barrier(skybox_cubemap, ImageLayout.GENERAL, MemoryUsage.IMAGE, MemoryUsage.ALL)
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
            cmd.image_barrier(skybox_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.IMAGE, MemoryUsage.ALL)

        # Irradiance
        irradiance_size = params.irradiance_size
        irradiance_cubemap = Image(
            r.ctx,
            irradiance_size,
            irradiance_size,
            Format.R16G16B16A16_SFLOAT,
            ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE,
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
                irradiance_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, undefined=True
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
                MemoryUsage.IMAGE,
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
            ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE,
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
                specular_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, undefined=True
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

        return GpuEnvironmentCubemaps(skybox_cubemap, irradiance_cubemap, specular_cubemap, specular_mips)


class EnvironmentLight(Light):
    def __init__(
        self,
        equirectangular: Optional[np.ndarray] = None,
        cubemaps: Optional[EnvironmentCubemaps] = None,
        ibl_params: Optional[IBLParams] = None,
        name=None,
    ):
        if not ((equirectangular is None) ^ (cubemaps is None)):
            print(equirectangular)
            print(cubemaps)
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
        super().__init__(name)

    def create(self, r: "renderer.Renderer"):
        if self.equirectangular is not None:
            height, width, channels = self.equirectangular.shape
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
            self.gpu_cubemaps = self.cubemaps.gpu()
        r.add_environment_light(self, self.gpu_cubemaps)

    @classmethod
    def from_equirectangular(
        cls, equirectangular: np.ndarray, ibl_params: Optional[IBLParams] = None
    ) -> "EnvironmentLight":
        return cls(equirectangular=equirectangular, ibl_params=ibl_params)

    @classmethod
    def from_cubemaps(cls, cubemaps: EnvironmentCubemaps) -> "EnvironmentLight":
        return cls(cubemaps=cubemaps)


# class AreaLight(Light):
#     pass
