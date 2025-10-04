# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union, TypeAlias

import numpy as np
from numpy.typing import DTypeLike

from .property import BufferProperty, ImageProperty, as_buffer_property, as_image_property
from .utils.gpu import get_format_info
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .renderer import Renderer
from .renderer_frame import RendererFrame
from .gpu_property import GpuImageProperty

from pyxpg import DescriptorSetBinding, DescriptorType, Filter, ImageUsageFlags, ImageLayout, MemoryUsage, PipelineStageFlags, Sampler, SamplerMipmapMode

MaterialData: TypeAlias = Union[np.ndarray, BufferProperty, ImageProperty]

def as_material_property(
    property: MaterialData, channels: int, name: str
) -> MaterialData:
    if isinstance(property, ImageProperty):
        img = as_image_property(property, name=name)
        info = get_format_info(img.format)
        if info.channels < channels:
            raise ValueError(f"Material property \"{name}\" image must have at least {channels}, but {img.format} only has {info.channels}")
    else:
        return as_buffer_property(property, np.float32, (channels,) if channels > 1 else None, name=name)

class MaterialProperty:
    pass

class Material:
    def __init__(self, properties: List[Union[BufferProperty, ImageProperty]], dtype: DTypeLike, shader_defines: str):
        self.properties = properties
        self.dtype = dtype
        self.created = False
        self.shader_defines = shader_defines

    def create(self, r: Renderer) -> None:
        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_sets = create_descriptor_layout_pool_and_sets_ringbuffer(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),
                DescriptorSetBinding(1, DescriptorType.SAMPLER),
                *([DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE)] * len(self.properties)),
            ],
            r.window.num_frames,
        )

        # TODO: maybe group in dataclass with defaults and allow user to override this
        self.sampler = Sampler(r.ctx, Filter.LINEAR, Filter.LINEAR, SamplerMipmapMode.LINEAR)
        for set in self.descriptor_sets:
            set.write_sampler(self.sampler, 1, 0)

        self.constants = np.zeros((1,), self.dtype)

        self.gpu_properties: List[GpuImageProperty] = []
        for p in self.properties:
            if isinstance(p, ImageProperty):
                self.gpu_properties.append(
                    r.add_gpu_image_property(p, ImageUsageFlags.SAMPLED, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.SHADER_READ_ONLY, PipelineStageFlags.FRAGMENT_SHADER, {p.name})
                )
            else:
                self.gpu_properties.append(None)

    def create_if_needed(self, r: Renderer) -> None:
        if self.created:
            return
        self.create(r)
        self.created = True

    def upload(self, r: Renderer, frame: RendererFrame) -> None:
        self.descriptor_set = self.descriptor_sets.get_current_and_advance()

        for i, p in enumerate(self.properties):
            if isinstance(p, BufferProperty):
                self.constants[p.name] = p.get_current()
                image = r.zero_image
            else:
                assert self.gpu_properties[i] is not None
                self.constants[p.name] = 0.0
                image = self.gpu_properties[i].get_current()
            self.descriptor_set.write_image(image, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 2 + i)

        constants_alloc = r.uniform_pool.alloc(self.constants.itemsize)
        constants_alloc.upload(frame.cmd, self.constants.view(np.uint8))
        self.descriptor_set.write_buffer(
            constants_alloc.buffer,
            DescriptorType.UNIFORM_BUFFER,
            0,
            0,
            constants_alloc.offset,
            constants_alloc.size,
        )


# Unlit flat color
class BaseColorMaterial(Material):
    def __init__(self, color: MaterialData):
        self.color = as_material_property(color, 3, "color")
        super().__init__([color])

# Diffuse only
class DiffuseMaterial(Material):
    def __init__(self, diffuse: MaterialData):
        self.diffuse = as_material_property(diffuse, 3, "diffuse")
        super().__init__([self.diffuse], {
            "diffuse": (np.dtype((np.float32, (3,))), 0),
            "has_diffuse_texture": (np.dtype((np.uint32, (1,))), 12),
        }, [("MATERIAL_DIFFUSE", "1")])

# Blinn-phong
class DiffuseSpecularMaterial(Material):
    def __init__(self,
                 diffuse: MaterialData,
                 specular_strength: MaterialData,
                 specular_exponent: MaterialData = 32.0,
                 ):
        self.diffuse = as_material_property(diffuse, 3, True, "diffuse")
        self.specular_strength = as_material_property(specular_strength, 1, True, "specular_strength")
        self.specular_exponent = as_material_property(specular_exponent, 1, False, "specular_exponent")
        self.specular_tint = as_material_property(specular_tint, 1, False, "specular_exponent")
        super().__init__([self.diffuse, self.specular_strength, self.specular_exponent], {
            "diffuse": (np.dtype((np.float32, (3,))), 0),
            "has_diffuse_texture": (np.dtype((np.uint32, (1,))), 12),
            "specular_strength": (np.dtype((np.float32, (1,))), 16),
            "has_specular_strength_texture": (np.dtype((np.uint32, (1,))), 20),
            "specular_exponent": (np.dtype((np.float32, (1,))), 24),
            "has_specular_exponent_texture": (np.dtype((np.uint32, (1,))), 28),
        }, [("MATERIAL_DIFFUSE_SPECULAR", "1")])

# PBR metallic roughness
class PBRMaterial(Material):
    def __init__(
        self,
        albedo: MaterialData,
        roughness: Optional[MaterialData],
        metallic: Optional[MaterialData],
    ):
        self.albedo = as_material_property(albedo, 3, "albedo")
        self.roughness = as_material_property(roughness, 1, "roughness")
        self.metallic = as_material_property(metallic, 1, "metallic")
        super().__init__([albedo, roughness, metallic])
