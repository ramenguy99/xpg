from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from pyglm.glm import inverse, orthoRH_ZO
from pyxpg import (
    ImageAspectFlags,
    AllocType,
    BufferUsageFlags,
    DepthAttachment,
    DescriptorSet,
    DescriptorSetEntry,
    DescriptorType,
    Image,
    ImageLayout,
    ImageUsageFlags,
    LoadOp,
    MemoryUsage,
    StoreOp,
)

from .property import BufferProperty
from .renderer import Renderer
from .renderer_frame import RendererFrame
from .scene import Light, Scene
from .utils.gpu import UploadableBuffer
from .utils.ring_buffer import RingBuffer


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

    def create(self, r: Renderer):
        if not self.shadow_settings.casts_shadow:
            return

        self.shadow_map = Image(
            r.ctx,
            self.shadow_settings.shadow_map_size,
            self.shadow_settings.shadow_map_size,
            r.shadowmap_format,
            ImageUsageFlags.SAMPLED | ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT,
            AllocType.DEVICE,
            name=f"{self.name}-shadowmap",
        )
        self.shadow_map_viewport = [0, 0, self.shadow_settings.shadow_map_size, self.shadow_settings.shadow_map_size]

        self.descriptor_sets = RingBuffer(
            [
                DescriptorSet(
                    r.ctx,
                    [
                        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
                    ],
                )
                for _ in range(r.window.num_frames)
            ]
        )
        constants_dtype = np.dtype(
            {
                "camera": (np.dtype((np.float32, (4, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), constants_dtype)
        self.uniform_buffers = RingBuffer(
            [
                UploadableBuffer(r.ctx, constants_dtype.itemsize, BufferUsageFlags.UNIFORM)
                for _ in range(r.window.num_frames)
            ]
        )
        for set, buf in zip(self.descriptor_sets.items, self.uniform_buffers.items):
            set.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        self.projection = orthoRH_ZO(
            -self.shadow_settings.half_extent,
            self.shadow_settings.half_extent,
            -self.shadow_settings.half_extent,
            self.shadow_settings.half_extent,
            self.shadow_settings.z_near,
            self.shadow_settings.z_far,
        )

    def render_shadowmaps(self, renderer: Renderer, frame: RendererFrame, scene: Scene):
        if not self.shadow_settings.casts_shadow:
            return

        set: DescriptorSet = self.descriptor_sets.get_current_and_advance()
        buf: UploadableBuffer = self.uniform_buffers.get_current_and_advance()

        self.constants["camera"] = self.projection * inverse(self.current_transform_matrix)  # type: ignore
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
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
    ):
        super().__init__(name, translation, rotation, scale)
        self.radiance = self.add_buffer_property(radiance, np.float32, (-1, 3), name="radiance")


# class AreaLight(Light):
#     pass

# class EnvironmentLight(Light):
#     pass
