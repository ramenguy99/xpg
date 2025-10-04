# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
from pyglm.glm import inverse, orthoRH_ZO, vec3, vec4
from pyxpg import (
    AllocType,
    BufferUsageFlags,
    DepthAttachment,
    DescriptorType,
    Image,
    ImageAspectFlags,
    ImageLayout,
    ImageUsageFlags,
    LoadOp,
    MemoryUsage,
    StoreOp,
)

from . import renderer
from .property import BufferProperty, view_bytes
from .renderer_frame import RendererFrame
from .scene import Object3D, Scene
from .utils.descriptors import create_descriptor_pool_and_sets_ringbuffer
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
        "shadowmap_index": (np.dtype((np.uint32, (1,))), 76),
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


# class AreaLight(Light):
#     pass

# class EnvironmentLight(Light):
#     pass
