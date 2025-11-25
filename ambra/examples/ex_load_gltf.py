import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pygltflib
from PIL import Image
from pyglm.glm import mat4, quat, rotate, vec3

from ambra.config import CameraConfig, Config, GuiConfig
from ambra.primitives3d import AnimatedMesh, Lines
from ambra.property import ArrayImageProperty, UploadSettings
from ambra.scene import UploadSettings, as_property
from ambra.transform3d import Transform3D
from ambra.utils.gpu import Format
from ambra.utils.hook import hook

filename = Path(sys.argv[1])
gltf = pygltflib.GLTF2().load(filename)

buffers = []
for b in gltf.buffers:
    buf = filename.parent.joinpath(b.uri).read_bytes()
    assert b.byteLength == len(buf), f"Expected: {b.byteLength}, Got: {len(buf)}"
    buffers.append(buf)


def load_attribute(ai: int) -> np.ndarray:
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

    return np.frombuffer(data, dtype).reshape((-1, *shape))[: a.count]


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
    indices: np.ndarray = None
    positions: np.ndarray = None
    normals: np.ndarray = None
    tangents: np.ndarray = None
    uvs: np.ndarray = None
    joint_indices_0: np.ndarray = None
    weights_0: np.ndarray = None
    joint_indices_1: np.ndarray = None
    weights_1: np.ndarray = None

    # Material
    base_color_texture: np.ndarray = None

    # Joints
    joints: List[Joint] = None


meshes: List[MeshData] = []


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
                rotation=quat(node.rotation[3], *node.rotation[:3]) if node.rotation is not None else quat(1, 0, 0, 0),
                scale=vec3(node.scale) if node.scale is not None else vec3(1.0),
            )
            j = Joint(node.name, t, parent_index, None)
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
                    if material.pbrMetallicRoughness.baseColorTexture:
                        base_color_texture = gltf.textures[material.pbrMetallicRoughness.baseColorTexture.index]
                        print("    " * (depth + 4) + ">" + str(base_color_texture))
                        m.base_color_texture = np.array(Image.open(filename.parent.joinpath(base_color_texture.name)))

            meshes.append(m)

    for ci in node.children:
        load_rec(ci, depth + 1)


for s in gltf.scenes:
    for ni in s.nodes:
        load_rec(ni, 0)

joints_by_name: Dict[str, Joint] = {}
for md in meshes:
    for j in md.joints:
        joints_by_name[j.name] = j
        print(j)

joints_array = np.empty((len(md.joints), 4, 4), np.float32)


def fk():
    # Forward kinematics
    for i, j in enumerate(md.joints):
        if j.parent_index < 0:
            p = np.eye(4, dtype=np.float32)
        else:
            p = joints_array[j.parent_index]
        c = np.array(j.transform.as_mat4())
        joints_array[i] = p @ c

    # Inverse bind matrix
    for i, j in enumerate(md.joints):
        joints_array[i] = joints_array[i] @ np.array(j.inverse_bind_matrix)


md = meshes[0]
fk()

joints_prop = ArrayImageProperty(
    joints_array,
    np.float32,
    upload=UploadSettings(
        preupload=False,
    ),
)

m = AnimatedMesh(
    positions=md.positions,
    normals=md.normals,
    tangents=md.tangents[:, :3],  # TODO: handle sign?
    uvs=md.uvs,
    joint_indices=md.joint_indices_0,
    weights=md.weights_0,
    indices=md.indices.reshape((-1,)),
    joints=joints_prop,
    texture=np.dstack((md.base_color_texture, np.ones(md.base_color_texture.shape[:2], np.uint8))),
    texture_format=Format.R8G8B8A8_UNORM,
)


class CustomViewer(Viewer):
    def __init__(self, title="ambra", config=None, key_map=None):
        super().__init__(title, config, key_map)

    @hook
    def on_gui(self):
        if imgui.begin("Joints")[0]:
            for k, v in joints_by_name.items():
                u, v.val = imgui.slider_float(k, v.val, 0, 1)
                if u:
                    v.transform.rotation = rotate(quat(1, 0, 0, 0), 0.5 * np.pi * v.val, vec3(0, 0, 1))
                    fk()
                    m.joints_buffer.invalidate_frame(0)
        imgui.end()


v = CustomViewer(
    config=Config(
        gui=GuiConfig(
            stats=True,
            inspector=True,
        ),
        camera=CameraConfig(
            position=vec3(0, 0, -10),
            target=vec3(0),
        ),
    )
)

v.scene.objects.append(m)
v.run()
