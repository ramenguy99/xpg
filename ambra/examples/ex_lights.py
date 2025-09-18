import numpy as np
from pyglm.glm import vec3

from ambra.config import CameraConfig, Config, PlaybackConfig
from ambra.primitives3d import Lines, Mesh
from ambra.property import (
    AnimationBoundary,
    FrameAnimation,
    as_buffer_property,
)
from ambra.viewer import Viewer

viewer = Viewer(
    "primitives",
    config=Config(
        playback=PlaybackConfig(
            enabled=True,
            playing=True,
        ),
        camera=CameraConfig(
            position=vec3(3),
            target=vec3(0),
        ),
        world_up=(0, 0, 1),
    ),
)

cube_positions = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    np.float32,
)

cube_indices = np.array([
    0, 1, 3, 3, 1, 2,
    1, 5, 2, 2, 5, 6,
    5, 4, 6, 6, 4, 7,
    4, 0, 7, 7, 0, 3,
    3, 2, 7, 7, 2, 6,
    4, 5, 0, 0, 5, 1
])

plane_positions = np.array(
    [
        [-5.0, -5.0, -2.0],
        [ 5.0, -5.0, -2.0],
        [ 5.0,  5.0, -2.0],
        [-5.0, -5.0, -2.0],
        [ 5.0,  5.0, -2.0],
        [-5.0,  5.0, -2.0],
    ],
    np.float32,
)

m = Mesh(cube_positions, cube_indices)
p = Mesh(plane_positions)
viewer.viewport.scene.objects.extend([m, p])

viewer.run()
