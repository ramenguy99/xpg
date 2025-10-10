# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pyxpg import (
    BufferUsageFlags,
    DescriptorSetBinding,
    DescriptorType,
    Filter,
    ImageLayout,
    ImageUsageFlags,
    MemoryUsage,
    PipelineStageFlags,
    Sampler,
    SamplerAddressMode,
    SamplerMipmapMode,
    get_format_info,
)

from . import renderer
from .gpu_property import GpuImageProperty
from .property import BufferProperty, ImageProperty, as_buffer_property, as_image_property
from .renderer_frame import RendererFrame
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .utils.gpu import UploadableBuffer, align_up
from .utils.ring_buffer import RingBuffer

MaterialData = Union[float, Tuple[float, ...], NDArray[np.float32], BufferProperty, ImageProperty]


@dataclass
class MaterialProperty:
    name: str
    property: Union[BufferProperty, ImageProperty]
    channels: int
    allow_image: bool


def as_material_property(property: MaterialData, channels: int, allow_image: bool, name: str) -> MaterialProperty:
    if isinstance(property, ImageProperty):
        if not allow_image:
            raise ValueError(f'Material property "{name}" cannot be an image')

        img = as_image_property(property, name=name)
        info = get_format_info(img.format)
        if info.channels < channels:
            raise ValueError(
                f'Material property "{name}" image must have at least {channels}, but {img.format} only has {info.channels}'
            )
        return MaterialProperty(name, img, channels, allow_image)
    else:
        return MaterialProperty(
            name,
            as_buffer_property(property, np.float32, (channels,) if channels > 1 else None, name=name),
            channels,
            allow_image,
        )


class Material:
    def __init__(self, properties: List[MaterialProperty], shader_defines: List[Tuple[str, str]]):
        self.properties = properties
        self.created = False
        self.shader_defines = shader_defines

        offset = 0
        fields = {}
        for p in properties:
            alignment = p.channels * 4 if p.channels != 3 else 16
            offset = align_up(offset, alignment)

            fields[p.name] = (np.dtype((np.float32, (p.channels,))), offset)
            offset += p.channels * 4
            if p.allow_image:
                fields[f"has_{p.name}_texture"] = (np.uint32, offset)  # type: ignore
                offset += 4
        self.dtype = np.dtype(fields)  # type: ignore

        for p in properties:
            p.property.update_callbacks.append(lambda _: self.reupload())
        self.need_upload = True

    def create(self, r: "renderer.Renderer") -> None:
        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_sets = (
            create_descriptor_layout_pool_and_sets_ringbuffer(
                r.ctx,
                [
                    DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),
                    DescriptorSetBinding(1, DescriptorType.SAMPLER),
                ]
                + [DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE) for p in self.properties if p.allow_image],
                r.window.num_frames,
            )
        )

        self.uniform_buffers = RingBuffer(
            [
                UploadableBuffer(r.ctx, self.dtype.itemsize, BufferUsageFlags.UNIFORM)
                for _ in range(r.window.num_frames)
            ]
        )

        # TODO: expose filter parameters from material constructor
        # TODO: expose mipmap creation (requires importing kernels and exposing mips in pyxpg)
        self.sampler = Sampler(
            r.ctx,
            Filter.LINEAR,
            Filter.LINEAR,
            SamplerMipmapMode.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_EDGE,
            v=SamplerAddressMode.CLAMP_TO_EDGE,
        )
        for set, buf in zip(self.descriptor_sets, self.uniform_buffers):
            set.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0, 0)
            set.write_sampler(self.sampler, 1, 0)

        self.constants = np.zeros((1,), self.dtype)

        self.images: List[GpuImageProperty] = []
        for p in self.properties:
            if isinstance(p.property, ImageProperty):
                self.images.append(
                    r.add_gpu_image_property(
                        p.property,
                        ImageUsageFlags.SAMPLED,
                        ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                        MemoryUsage.SHADER_READ_ONLY,
                        PipelineStageFlags.FRAGMENT_SHADER,
                        f"{type(self).__name__} - {p.property.name}",
                    )
                )

    def create_if_needed(self, r: "renderer.Renderer") -> None:
        if self.created:
            return
        self.create(r)
        self.created = True

    def reupload(self) -> None:
        self.need_upload = True

    def upload(self, r: "renderer.Renderer", frame: RendererFrame) -> None:
        if not self.need_upload:
            return

        self.descriptor_set = self.descriptor_sets.get_current_and_advance()
        image_index = 0
        for p in self.properties:
            if isinstance(p.property, BufferProperty):
                self.constants[p.name] = p.property.get_current()
                image = r.zero_image
            else:
                self.constants[p.property.name] = 0.0
                self.constants[f"has_{p.name}_texture"] = True
                image = self.images[image_index].get_current()

            if p.allow_image:
                self.descriptor_set.write_image(
                    image, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, image_index + 2
                )
                image_index += 1

        self.uniform_buffers.get_current_and_advance().upload(
            frame.cmd, MemoryUsage.FRAGMENT_SHADER_UNIFORM, self.constants.view(np.uint8)
        )

        self.need_upload = False


# Unlit flat color
class ColorMaterial(Material):
    def __init__(self, color: MaterialData):
        self.color = as_material_property(color, 3, True, "color")
        super().__init__([self.color], [("MATERIAL_COLOR", "1")])


# Diffuse only
class DiffuseMaterial(Material):
    def __init__(self, diffuse: MaterialData):
        self.diffuse = as_material_property(diffuse, 3, True, "diffuse")
        super().__init__([self.diffuse], [("MATERIAL_DIFFUSE", "1")])


# Blinn-phong
class DiffuseSpecularMaterial(Material):
    def __init__(
        self,
        diffuse: MaterialData,
        specular_strength: MaterialData,
        specular_exponent: Union[float, BufferProperty] = 32.0,
        specular_tint: Union[float, BufferProperty] = 0.0,
    ):
        self.diffuse = as_material_property(diffuse, 3, True, "diffuse")
        self.specular_strength = as_material_property(specular_strength, 1, True, "specular_strength")
        self.specular_exponent = as_material_property(specular_exponent, 1, False, "specular_exponent")
        self.specular_tint = as_material_property(specular_tint, 1, False, "specular_tint")
        super().__init__(
            [self.diffuse, self.specular_strength, self.specular_exponent, self.specular_tint],
            [("MATERIAL_DIFFUSE_SPECULAR", "1")],
        )


# PBR metallic roughness
class PBRMaterial(Material):
    def __init__(
        self,
        albedo: MaterialData,
        roughness: MaterialData,
        metallic: MaterialData,
    ):
        self.albedo = as_material_property(albedo, 3, True, "albedo")
        self.roughness = as_material_property(roughness, 1, True, "roughness")
        self.metallic = as_material_property(metallic, 1, True, "metallic")
        super().__init__([self.albedo, self.roughness, self.metallic], [("MATERIAL_PBR", "1")])
