# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union, TypeAlias

import numpy as np

from .property import BufferProperty, ImageProperty, as_buffer_property, as_image_property
from .utils.gpu import get_format_info

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
        return as_buffer_property(property, np.float32, (channels,), name=name)

class Material:
    def __init__(self, properties: List[Union[BufferProperty, ImageProperty]]):
        self.properties = properties

# Unlit flat color
class BaseColorMaterial(Material):
    def __init__(self, color: MaterialData):
        self.color = as_material_property(color, 3, "color")
        super().__init__([color])

# Diffuse only
class DiffuseMaterial(Material):
    def __init__(self, diffuse: MaterialData):
        self.diffuse = as_material_property(diffuse, 3, "diffuse")
        super().__init__([self.diffuse])

# Blinn-phong
class DiffuseSpecularMaterial(Material):
    def __init__(self,
                 diffuse: MaterialData,
                 specular: MaterialData,
                 ):
        self.diffuse = as_material_property(diffuse, 3, "diffuse")
        self.specular = as_material_property(diffuse, 1, "specular")
        super().__init__([diffuse, specular])

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
