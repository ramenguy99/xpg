# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Flag, auto
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pyxpg import (
    BufferUsageFlags,
    DescriptorSetBinding,
    DescriptorType,
    Filter,
    Image,
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    MemoryUsage,
    PipelineStageFlags,
    Sampler,
    SamplerAddressMode,
    SamplerMipmapMode,
    get_format_info,
)

from . import renderer
from .property import BufferProperty, ImageProperty, as_buffer_property, as_image_property
from .renderer_frame import RendererFrame
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .utils.gpu import UploadableBuffer, align_up, view_bytes
from .utils.ring_buffer import RingBuffer

MaterialData = Union[float, Tuple[float, ...], NDArray[np.float32], BufferProperty, ImageProperty]


class MaterialPropertyFlags(Flag):
    ALLOW_IMAGE = auto()
    HAS_VALUE = auto()
    SRGB = auto()


@dataclass
class MaterialProperty:
    name: str
    property: Union[BufferProperty, ImageProperty]
    channels: int
    flags: MaterialPropertyFlags


def as_material_property(
    property: MaterialData, channels: int, flags: MaterialPropertyFlags, name: str
) -> MaterialProperty:
    if isinstance(property, ImageProperty):
        if not flags & MaterialPropertyFlags.ALLOW_IMAGE:
            raise ValueError(f'Material property "{name}" cannot be an image')

        img = as_image_property(property, name=name)
        info = get_format_info(img.format)
        if info.channels < channels:
            raise ValueError(
                f'Material property "{name}" image must have at least {channels}, but {img.format} only has {info.channels}'
            )
        return MaterialProperty(name, img, channels, flags)
    else:
        prop = as_buffer_property(property, np.float32, (channels,) if channels > 1 else (), name=name)
        return MaterialProperty(name, prop, channels, flags)


class Material:
    def __init__(self, properties: List[MaterialProperty], shader_defines: List[Tuple[str, str]]):
        self.properties = properties
        self.created = False
        self.shader_defines = shader_defines

        offset = 0
        fields = {}
        for p in properties:
            if p.flags & MaterialPropertyFlags.HAS_VALUE:
                alignment = p.channels * 4 if p.channels != 3 else 16
                offset = align_up(offset, alignment)
                fields[p.name] = (np.dtype((np.float32, (p.channels,))), offset)
                offset += p.channels * 4

            if p.flags & MaterialPropertyFlags.ALLOW_IMAGE:
                fields[f"has_{p.name}_texture"] = (np.uint32, offset)  # type: ignore
                offset += 4
        self.dtype = np.dtype(fields)  # type: ignore

        for p in properties:
            if isinstance(p.property, ImageProperty):
                p.property.use_gpu(
                    ImageUsageFlags.TRANSFER_DST
                    | ImageUsageFlags.SAMPLED
                    | ImageUsageFlags.STORAGE
                    | ImageUsageFlags.TRANSFER_SRC,
                    ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                    # MemoryUsage.SHADER_READ_ONLY,
                    PipelineStageFlags.FRAGMENT_SHADER,
                    mips=True,
                    srgb=bool(p.flags & MaterialPropertyFlags.SRGB),
                )
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
                + [
                    DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE)
                    for p in self.properties
                    if p.flags & MaterialPropertyFlags.ALLOW_IMAGE
                ],
                r.num_frames_in_flight,
            )
        )

        self.uniform_buffers = RingBuffer(
            [
                UploadableBuffer(r.ctx, self.dtype.itemsize, BufferUsageFlags.UNIFORM)
                for _ in range(r.num_frames_in_flight)
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
            image: Union[Image, ImageView]
            if isinstance(p.property, BufferProperty):
                if p.flags & MaterialPropertyFlags.HAS_VALUE:
                    self.constants[p.name] = p.property.get_current()
                self.constants[f"has_{p.name}_texture"] = False
                image = r.zero_image
            else:
                if p.flags & MaterialPropertyFlags.HAS_VALUE:
                    self.constants[p.name] = 0.0
                self.constants[f"has_{p.name}_texture"] = True
                image = p.property.get_current_gpu().view()

            if p.flags & MaterialPropertyFlags.ALLOW_IMAGE:
                self.descriptor_set.write_image(
                    image, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, image_index + 2
                )
                image_index += 1

        self.uniform_buffers.get_current_and_advance().upload(
            frame.cmd, MemoryUsage.SHADER_UNIFORM, view_bytes(self.constants)
        )

        self.need_upload = False


# Unlit flat color
class ColorMaterial(Material):
    def __init__(self, color: MaterialData):
        self.color = as_material_property(
            color, 3, MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE, "color"
        )
        super().__init__([self.color], [("MATERIAL_COLOR", "1")])


# Diffuse only
class DiffuseMaterial(Material):
    def __init__(self, diffuse: MaterialData, normal: Optional[ImageProperty] = None):
        self.diffuse = as_material_property(
            diffuse,
            3,
            MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE | MaterialPropertyFlags.SRGB,
            "diffuse",
        )
        self.normal = as_material_property(normal or (0.0, 0.0, 0.0), 3, MaterialPropertyFlags.ALLOW_IMAGE, "normal")
        super().__init__([self.diffuse, self.normal], [("MATERIAL_DIFFUSE", "1")])


# Blinn-phong
class DiffuseSpecularMaterial(Material):
    def __init__(
        self,
        diffuse: MaterialData,
        specular_strength: MaterialData,
        specular_exponent: Union[float, BufferProperty] = 32.0,
        specular_tint: Union[float, BufferProperty] = 0.0,
        normal: Optional[ImageProperty] = None,
    ):
        self.diffuse = as_material_property(
            diffuse,
            3,
            MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE | MaterialPropertyFlags.SRGB,
            "diffuse",
        )
        self.specular_strength = as_material_property(
            specular_strength,
            1,
            MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE,
            "specular_strength",
        )
        self.specular_exponent = as_material_property(
            specular_exponent, 1, MaterialPropertyFlags.HAS_VALUE, "specular_exponent"
        )
        self.specular_tint = as_material_property(specular_tint, 1, MaterialPropertyFlags.HAS_VALUE, "specular_tint")
        self.normal = as_material_property(normal or (0.0, 0.0, 0.0), 3, MaterialPropertyFlags.ALLOW_IMAGE, "normal")
        super().__init__(
            [self.diffuse, self.specular_strength, self.specular_exponent, self.specular_tint, self.normal],
            [("MATERIAL_DIFFUSE_SPECULAR", "1")],
        )


# PBR metallic roughness
class PBRMaterial(Material):
    def __init__(
        self,
        albedo: MaterialData,
        roughness: MaterialData,
        metallic: MaterialData,
        ao: Union[float, BufferProperty] = 1.0,
        normal: Optional[ImageProperty] = None,
    ):
        self.albedo = as_material_property(
            albedo,
            3,
            MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE | MaterialPropertyFlags.SRGB,
            "albedo",
        )
        self.roughness = as_material_property(
            roughness, 1, MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE, "roughness"
        )
        self.metallic = as_material_property(
            metallic, 1, MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE, "metallic"
        )
        self.ao = as_material_property(
            ao, 1, MaterialPropertyFlags.ALLOW_IMAGE | MaterialPropertyFlags.HAS_VALUE, "ao"
        )
        self.normal = as_material_property(normal or (0.0, 0.0, 0.0), 3, MaterialPropertyFlags.ALLOW_IMAGE, "normal")
        super().__init__([self.albedo, self.roughness, self.metallic, self.ao, self.normal], [("MATERIAL_PBR", "1")])
