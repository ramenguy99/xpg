# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union

import numpy as np

from .property import BufferProperty, ImageProperty, as_buffer_property, as_image_property


class ValueMaterialProperty:
    def __init__(self, value: Union[BufferProperty, np.ndarray]):
        self.value = as_buffer_property(value, np.float32, (-1,))


class TextureMaterialProperty:
    def __init__(self, image: Union[ImageProperty, np.ndarray]):
        self.image = as_image_property(image)


class Material:
    def __init__(self, properties: List[Union[ValueMaterialProperty, TextureMaterialProperty]]):
        pass


def check_material_property(
    property: Union[ValueMaterialProperty, TextureMaterialProperty], channels: int
) -> Union[ValueMaterialProperty, TextureMaterialProperty]:
    if isinstance(property, ValueMaterialProperty):
        # if property.value.shape
        pass
    elif isinstance(property, TextureMaterialProperty):
        pass
    else:
        raise TypeError(f"Unhandled type: {type(property)}")
    return property


class BaseColorMaterial(Material):
    def __init__(self, color: Union[ValueMaterialProperty, TextureMaterialProperty]):
        self.color = check_material_property(color, 3)
        super().__init__([color])


class DiffuseMaterial(Material):
    def __init__(self, albedo: Union[ValueMaterialProperty, TextureMaterialProperty]):
        self.albedo = check_material_property(albedo, 3)
        super().__init__([albedo])


class RoughnessMetallicMaterial(Material):
    def __init__(
        self,
        albedo: Union[ValueMaterialProperty, TextureMaterialProperty],
        roughness: Optional[Union[ValueMaterialProperty, TextureMaterialProperty]],
        metallic: Optional[Union[ValueMaterialProperty, TextureMaterialProperty]],
        roughness_metallic: Optional[Union[ValueMaterialProperty, TextureMaterialProperty]],
    ):
        self.albedo = check_material_property(albedo, 3)
        self.metallic_roughness: Optional[Union[ValueMaterialProperty, TextureMaterialProperty]] = None
        self.roughness: Optional[Union[ValueMaterialProperty, TextureMaterialProperty]] = None
        self.metallic: Optional[Union[ValueMaterialProperty, TextureMaterialProperty]] = None

        properties = [albedo]
        if roughness is not None or metallic is not None:
            if roughness is None or metallic is None:
                raise ValueError("if roughness or metallic are not None, they both must be not None")
            if roughness_metallic is not None:
                raise ValueError("if roughness or metallic are not None, roughness_metallic must be None")
            self.roughness = check_material_property(roughness, 1)
            self.metallic = check_material_property(metallic, 1)
            properties.append(roughness)
            properties.append(metallic)
        else:
            if roughness_metallic is None:
                raise ValueError("roughness, metallic, or roughness_metallic must be not None")
            self.roughness_metallic = check_material_property(roughness_metallic, 2)
            properties.append(roughness_metallic)

        super().__init__(properties)
