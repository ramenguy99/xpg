from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pygltflib
from numpy.typing import NDArray
from PIL import Image
from pyglm.glm import mat4, quat, vec3

from ambra.transform3d import Transform3D


@dataclass
class Joint:
    name: str
    transform: Transform3D
    parent_index: int
    inverse_bind_matrix: mat4
    val: float = 1.0


@dataclass
class MeshData:
    # Geometry
    indices: Optional[NDArray] = None
    positions: Optional[NDArray] = None
    normals: Optional[NDArray] = None
    tangents: Optional[NDArray] = None
    uvs: Optional[NDArray] = None
    joint_indices: Optional[NDArray] = None
    weights: Optional[NDArray] = None

    # Material
    base_color_texture: Optional[NDArray] = None
    metallic_roughness_texture: Optional[NDArray] = None
    normal_texture: Optional[NDArray] = None
    ao_texture: Optional[NDArray] = None

    # Joints
    joints: List[Joint] = None


@dataclass
class Scene:
    meshes: List[MeshData]


def load(path: Path) -> Scene:
    gltf = pygltflib.GLTF2().load(path)

    meshes: List[MeshData] = []

    buffers = []
    for b in gltf.buffers:
        buf = path.parent.joinpath(b.uri).read_bytes()
        assert b.byteLength == len(buf), f"Expected: {b.byteLength}, Got: {len(buf)}"
        buffers.append(buf)

    def load_attribute(ai: int) -> NDArray:
        a = gltf.accessors[ai]
        assert a.bufferView is not None
        view = gltf.bufferViews[a.bufferView]

        data = memoryview(buffers[view.buffer])[view.byteOffset :][: view.byteLength][a.byteOffset :]

        dtype = {
            pygltflib.FLOAT: np.float32,
            pygltflib.UNSIGNED_SHORT: np.int16,
            pygltflib.BYTE: np.int8,
            pygltflib.UNSIGNED_BYTE: np.uint8,
            pygltflib.SHORT: np.int16,
            pygltflib.UNSIGNED_SHORT: np.uint16,
            pygltflib.UNSIGNED_INT: np.uint32,
        }[a.componentType]

        shape = {
            "SCALAR": (1,),
            "VEC2": (2,),
            "VEC3": (3,),
            "VEC4": (4,),
            "MAT4": (4, 4),
        }[a.type]

        if not data.c_contiguous:
            data = bytes(data)

        data_bytestride = np.prod(shape) * np.dtype(dtype).itemsize
        if view.byteStride is not None:
            assert view.byteStride == data_bytestride, f"{view.byteStride} -> {data_bytestride}"

        arr = np.frombuffer(data, dtype)
        if a.type != "SCALAR":
            arr = arr.reshape((-1, *shape))
        return arr[: a.count]

    def load_rec(ni: int, depth: int):
        node = gltf.nodes[ni]
        print("    " * depth + f"{ni}: {node}")

        joints: List[Joint] = []
        if node.skin is not None:
            skin = gltf.skins[node.skin]
            print("    " * (depth + 1) + ">" + str(skin))

            # Assume there is a single root joint and it's skeleton, this is not necessarily true but simiplifies
            # parsing.
            assert skin.skeleton is not None

            node_id_to_joint_index: Dict[int, int] = {}

            def load_joint_rec(ni: int, parent_index: int):
                node = gltf.nodes[ni]

                t = Transform3D(
                    translation=vec3(node.translation) if node.translation is not None else vec3(0.0),
                    rotation=quat(node.rotation[3], *node.rotation[:3])
                    if node.rotation is not None
                    else quat(1, 0, 0, 0),
                    # rotation=inverse(quat(node.rotation[3], *node.rotation[:3])) if node.rotation is not None else quat(1, 0, 0, 0),
                    # rotation=quat(node.rotation) if node.rotation is not None else quat(1, 0, 0, 0),
                    scale=vec3(node.scale) if node.scale is not None else vec3(1.0),
                )
                # print(t.rotation, node.rotation)
                # exit(1)
                j = Joint(node.name, t, parent_index, None)
                print(node.name, t)
                joint_index = len(joints)
                node_id_to_joint_index[ni] = joint_index
                joints.append(j)

                for c in node.children:
                    load_joint_rec(c, joint_index)

            load_joint_rec(skin.skeleton, -1)

            inverse_bind_matrices = load_attribute(skin.inverseBindMatrices)
            for i, j in enumerate(skin.joints):
                # Assume joints are given in traversal order, we would actually need to reorder here
                assert i == node_id_to_joint_index[j]

                joints[node_id_to_joint_index[j]].inverse_bind_matrix = mat4(inverse_bind_matrices[i])

        if node.mesh is not None:
            # Load mesh
            mesh = gltf.meshes[node.mesh]
            print("    " * (depth + 1) + ">" + str(mesh))

            for p in mesh.primitives:
                m = MeshData(joints=joints)

                print("    " * (depth + 2) + ">" + str(p))

                if p.indices is not None:
                    indices = gltf.accessors[p.indices]
                    print("    " * (depth + 3) + "|-> INDICES", p.indices, indices)

                    m.indices = load_attribute(p.indices)

                for k, v in p.attributes.__dict__.items():
                    if k.startswith("__"):
                        continue
                    if v is None:
                        continue
                    print("    " * (depth + 3) + "|-> " + k, v, gltf.accessors[v], flush=True)

                    if k == "POSITION":
                        m.positions = load_attribute(v)
                    elif k == "NORMAL":
                        m.normals = load_attribute(v)
                    elif k == "TANGENT":
                        m.tangents = load_attribute(v)
                    elif k == "TEXCOORD_0":
                        m.uvs = load_attribute(v)
                    elif k == "JOINTS_0":
                        m.joint_indices_0 = load_attribute(v)
                    elif k == "WEIGHTS_0":
                        m.weights_0 = load_attribute(v)
                    elif k == "JOINTS_1":
                        m.joint_indices_1 = load_attribute(v)
                    elif k == "WEIGHTS_1":
                        m.weights_1 = load_attribute(v)

                if p.material is not None:
                    material = gltf.materials[p.material]
                    print("    " * (depth + 3) + ">" + str(material))

                    if material.pbrMetallicRoughness is not None:

                        def load_image(texture_attribute):
                            img = None
                            if texture_attribute:
                                texture = gltf.textures[texture_attribute.index]
                                if texture.source is not None:
                                    image = gltf.images[texture.source]
                                    print("    " * (depth + 5) + ">" + str(image))
                                    img = np.array(Image.open(path.parent.joinpath(image.uri)))
                                if texture.sampler is not None:
                                    sampler = gltf.samplers[texture.sampler]
                                    print("    " * (depth + 5) + ">" + str(sampler))
                            return img

                        m.base_color_texture = load_image(material.pbrMetallicRoughness.baseColorTexture)
                        m.metallic_roughness_texture = load_image(
                            material.pbrMetallicRoughness.metallicRoughnessTexture
                        )
                    m.ao_texture = load_image(material.occlusionTexture)
                    m.normal_texture = load_image(material.normalTexture)

                meshes.append(m)

        for ci in node.children:
            load_rec(ci, depth + 1)

    for s in gltf.scenes:
        for ni in s.nodes:
            load_rec(ni, 0)

    return Scene(meshes)
