import os
import sys

import cv2
import gltf
import numpy as np
from pyglm.glm import angleAxis, vec3
from pyxpg import Format

from ambra.config import *
from ambra.geometry import create_sphere
from ambra.lights import EnvironmentLight
from ambra.materials import DiffuseMaterial, PBRMaterial
from ambra.primitives3d import Grid, GridType, Mesh
from ambra.property import DataImageProperty
from ambra.utils.gpu import readback
from ambra.viewer import Viewer

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

equirectangular = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR_RGB)
lights = [EnvironmentLight.from_equirectangular(equirectangular)]

model = gltf.load(sys.argv[1])

meshes = []
for mesh_data in model.meshes:
    print(mesh_data.ao_texture.shape)
    material = PBRMaterial(
        DataImageProperty(mesh_data.base_color_texture, Format.R8G8B8A8_SRGB),
        roughness=DataImageProperty(mesh_data.metallic_roughness_texture[:, :, 1:2], Format.R8_UNORM),
        metallic=DataImageProperty(mesh_data.metallic_roughness_texture[:, :, 2:3], Format.R8_UNORM),
        ao=DataImageProperty(mesh_data.ao_texture[..., :1]),
        normal=DataImageProperty(mesh_data.normal_texture),
    )

    assert np.all(mesh_data.tangents[:, 3] > 0.0)
    m = Mesh(
        mesh_data.positions,
        mesh_data.indices,
        mesh_data.normals,
        mesh_data.tangents[:, :3],
        mesh_data.uvs,
        material=material,
        translation=(0, 0, 2),
        rotation=angleAxis(np.pi * 0.5, vec3(1, 0, 0)),
    )
    meshes.append(m)

mirror = PBRMaterial(
    albedo=(1, 1, 1),
    roughness=(0.01),
    metallic=1,
)
white = PBRMaterial(
    albedo=(1, 1, 1),
    roughness=(1.0),
    metallic=0,
)
dark_gray = PBRMaterial(
    albedo=(0.1, 0.1, 0.1),
    roughness=(1.0),
    metallic=0,
)
light_gray = PBRMaterial(
    albedo=(0.4, 0.4, 0.4),
    roughness=(1.0),
    metallic=0,
)
v, f = create_sphere(0.5, rings=32, sectors=64)
n = v / np.linalg.norm(v, axis=1, keepdims=True)
spheres = [
    Mesh(v, f, normals=n, translation=(2, 1, 0), material=mirror),
    Mesh(v, f, normals=n, translation=(2, 1, 1.1), material=white),
    Mesh(v, f, normals=n, translation=(2, 1, 2.2), material=light_gray),
    Mesh(v, f, normals=n, translation=(2, 1, 3.3), material=dark_gray),
]

g = Grid.transparent_black_lines((100, 100), GridType.XY_PLANE)

v = Viewer(
    config=Config(
        gui=GuiConfig(stats=True, playback=True, inspector=True, renderer=True),
        world_up=(0, 0, 1),
        renderer=RendererConfig(msaa_samples=4),
        camera=CameraConfig(
            position=(4, -4, 3),
            target=(0, 0, 1.5),
        ),
    )
)

v.viewport.scene.objects.extend(lights)
v.viewport.scene.objects.extend(meshes)
v.viewport.scene.objects.extend(spheres)
v.viewport.scene.objects.append(g)

v.run()
