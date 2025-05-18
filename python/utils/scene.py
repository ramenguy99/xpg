from enum import IntEnum
from dataclasses import dataclass
from pyglm.glm import vec2, vec3, vec4, mat4
from typing import Union, List
from pathlib import Path
import struct

import numpy as np

class MaterialParameterKind(IntEnum):
    NONE = 0
    TEXTURE = 1
    VEC2 = 2
    VEC3 = 3
    VEC4 = 4

@dataclass
class MaterialParameter:
    kind: MaterialParameterKind
    value: Union[int, vec2, vec3, vec4, None] = None

@dataclass
class Material:
    base_color: MaterialParameter
    normal: MaterialParameter
    specular: MaterialParameter
    emissive: MaterialParameter

@dataclass
class Mesh:
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray

    transform: mat4
    material: Material

class ImageFormat(IntEnum):
    RGBA8 = 0
    SRGBA8 = 1
    RGBA8_BC7 = 2
    SRGBA8_BC7 = 3

@dataclass
class Image:
    width: int
    height: int
    format: ImageFormat
    data: np.ndarray

@dataclass
class Scene:
    meshes: List[Mesh]
    images: List[Image]


def parse_scene(path: Path) -> Scene:
    data = path.read_bytes()
    it = memoryview(data)

    def consume(size: int):
        nonlocal it
        view = it[:size]
        it = it[size:]
        return view

    def consume_u32():
        return struct.unpack("<I", consume(4))[0]

    def consume_u64():
        return struct.unpack("<Q", consume(8))[0]
    
    def consume_vec(dtype: type, elems: int):
        size = consume_u64()
        if elems > 1:
            return np.frombuffer(consume(size), dtype=dtype).copy().reshape((-1, elems))
        else:
            return np.frombuffer(consume(size), dtype=dtype).copy()

    def consume_vec2():
        return vec2.from_bytes(bytes(consume(8)))

    def consume_vec3():
        return vec3.from_bytes(bytes(consume(12)))

    def consume_vec4():
        return vec4.from_bytes(bytes(consume(16)))

    def consume_mat4():
        return mat4.from_bytes(bytes(consume(64)))
    
    def consume_material_param():
        kind = MaterialParameterKind(consume_u32())
        if kind == MaterialParameterKind.NONE:
            return MaterialParameter(kind)
        elif kind == MaterialParameterKind.VEC2:
            return MaterialParameter(kind, consume_vec2())
        elif kind == MaterialParameterKind.VEC3:
            return MaterialParameter(kind, consume_vec3())
        elif kind == MaterialParameterKind.VEC4:
            return MaterialParameter(kind, consume_vec4())
        elif kind == MaterialParameterKind.TEXTURE:
            return MaterialParameter(kind, consume_u32())
        else:
            raise ValueError(f"Unknown material parameter kind: {kind}")

    # Parse meshes
    num_meshes = consume_u64()
    meshes = []
    for _ in range(num_meshes):
        meshes.append(Mesh(
            positions=consume_vec(np.float32, 3),
            normals=consume_vec(np.float32, 3),
            tangents=consume_vec(np.float32, 3),
            uvs=consume_vec(np.float32, 2),
            indices=consume_vec(np.uint32, 1),
            transform=consume_mat4(),
            material=Material(
                base_color=consume_material_param(),
                normal=consume_material_param(),
                specular=consume_material_param(),
                emissive=consume_material_param(),
            ),
        ))
    num_images = consume_u64()
    images = []
    for _ in range(num_images):
        images.append(Image(
            width=consume_u32(),
            height=consume_u32(),
            format=ImageFormat(consume_u32()),
            data=consume_vec(np.uint8, 1)
        ))

    assert len(it) == 0
    
    # Parse images
    print(f"meshes: {num_meshes}")
    print(f"images: {num_images}")
    return Scene(meshes, images)

