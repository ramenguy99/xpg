import os
import sys

import cv2
import numpy as np
from pyglm.glm import normalize, quatLookAtRH, vec3
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.geometry import create_sphere
from ambra.lights import DirectionalLight, DirectionalShadowSettings, EnvironmentLight, UniformEnvironmentLight
from ambra.materials import PBRMaterial
from ambra.primitives3d import Lines, Mesh
from ambra.property import ArrayBufferProperty
from ambra.viewer import Viewer

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def main():
    viewer = Viewer(
        config=Config(
            window_width=1600,
            window_height=900,
            playback=PlaybackConfig(
                playing=True,
            ),
            camera=CameraConfig(
                position=vec3(-7, 7, 1.7),
                target=vec3(0),
            ),
            world_up=(0, 0, 1),
            gui=GuiConfig(
                stats=True,
                playback=True,
            ),
            renderer=RendererConfig(msaa_samples=4, background_color=(0.1, 0.1, 0.1, 1.0)),
        ),
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

    v, n, f = create_sphere(0.5, rings=32, sectors=64)

    p_v = ArrayBufferProperty(v[np.newaxis], np.float32)
    p_n = ArrayBufferProperty(n[np.newaxis], np.float32)
    p_f = ArrayBufferProperty(f[np.newaxis], np.uint32)

    spheres = []
    for y in range(8):
        for x in range(8):
            t_x = ((x + 0.5) / 8 * 2.0 - 1.0) * 5
            t_y = ((y + 0.5) / 8 * 2.0 - 1.0) * 5
            if (y ^ x) & 1:
                m = PBRMaterial((1.0, 1.0, 1.0), (x + 0.5) / 8, (y + 0.5) / 8)
            else:
                m = PBRMaterial((1.0, 0.0, 0.0), (x + 0.5) / 8, (y + 0.5) / 8)
            sphere = Mesh(p_v, p_f, normals=p_n, translation=(t_x, t_y, 1), material=m)
            spheres.append(sphere)

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
        rotation = quatLookAtRH(normalize(light_target - light_position * s), vec3(0, 1, 0))
        lights.append(
            DirectionalLight(
                np.array([0.3, 0.3, 0.3]),
                shadow_settings=DirectionalShadowSettings(casts_shadow=False),
                translation=light_position * s,
                rotation=rotation,
            )
        )

    equirectangular = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR_RGB)
    lights.append(EnvironmentLight.from_equirectangular(equirectangular))
    # lights.append(UniformEnvironmentLight((0.1, 0.1, 0.1)))

    viewer.scene.objects.extend(spheres + [o] + lights)
    viewer.run()


if __name__ == "__main__":
    main()
