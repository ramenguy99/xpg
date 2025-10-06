import os
import sys

import cv2
import numpy as np
from pyglm.glm import normalize, quatLookAtRH, vec3
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.geometry import create_sphere
from ambra.lights import DirectionalLight, DirectionalShadowSettings, EnvironmentLight, UniformEnvironmentLight
from ambra.materials import DiffuseMaterial, DiffuseSpecularMaterial, PBRMaterial
from ambra.primitives3d import Lines, Mesh
from ambra.property import as_buffer_property
from ambra.viewer import Viewer

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def main():
    viewer = Viewer(
        "materials",
        config=Config(
            window_width=1600,
            window_height=900,
            playback=PlaybackConfig(
                enabled=True,
                playing=True,
            ),
            camera=CameraConfig(
                position=vec3(10, 4, 10),
                target=vec3(0),
            ),
            world_up=(0, 0, 1),
            gui=GuiConfig(
                stats=True,
                playback=True,
            ),
        ),
    )

    plane_positions = np.array(
        [
            [-10.0, -10.0, -2.0],
            [10.0, -10.0, -2.0],
            [10.0, 10.0, -2.0],
            [-10.0, -10.0, -2.0],
            [10.0, 10.0, -2.0],
            [-10.0, 10.0, -2.0],
        ],
        np.float32,
    )

    origin_positions = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            np.float32,
        )
        * 5.0
    )

    origin_colors = np.array(
        [
            0xFF0000FF,
            0xFF0000FF,
            0xFF00FF00,
            0xFF00FF00,
            0xFFFF0000,
            0xFFFF0000,
        ],
        np.uint32,
    )

    v, f = create_sphere(0.5)
    n = v / np.linalg.norm(v, axis=1, keepdims=True)

    p_v = as_buffer_property(v, np.float32, (-1, 3))
    p_n = as_buffer_property(n, np.float32, (-1, 3))
    p_f = as_buffer_property(f, np.uint32, (-1,))

    spheres = []
    for y in range(8):
        for x in range(8):
            t_x = ((x + 0.5) / 8 * 2.0 - 1.0) * 5
            t_y = ((y + 0.5) / 8 * 2.0 - 1.0) * 5
            # m = DiffuseSpecularMaterial((x / 8, 0.0, 0.0), y / 8.0, specular_exponent=128)
            m = PBRMaterial((1, 0, 0), (x + 0.5) / 8, (y + 0.5) / 8)
            sphere = Mesh(p_v, p_f, normals=p_n, translation=(t_x, t_y, 0), material=m)
            spheres.append(sphere)

    p = Mesh(plane_positions, material=DiffuseMaterial((0.2, 0.2, 0.2)))
    o = Lines(origin_positions, origin_colors, 4.0)

    light_position = vec3(5, 6, 7) * 4.0
    light_target = vec3(0, 0, 0)

    signs = [
        vec3(-1, -1, 1),
        vec3(1, -1, 1),
        vec3(-1, 1, 1),
        vec3(1, 1, 1),
    ]

    lights = []
    for s in signs:
        rotation = quatLookAtRH(normalize(light_target - light_position * s), vec3(0, 0, 1))
        lights.append(
            DirectionalLight(
                np.array([1.0, 1.0, 1.0]),
                shadow_settings=DirectionalShadowSettings(half_extent=10.0, z_near=1.0, z_far=100),
                translation=light_position * s,
                rotation=rotation,
            )
        )

    # lights.append(UniformEnvironmentLight((0.03, 0.03, 0.03)))

    equirectangular = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(equirectangular.shape, equirectangular.dtype)
    lights.append(EnvironmentLight.from_equirectangular(equirectangular))

    viewer.viewport.scene.objects.extend(spheres + [p, o] + lights)
    viewer.run()


if __name__ == "__main__":
    main()
