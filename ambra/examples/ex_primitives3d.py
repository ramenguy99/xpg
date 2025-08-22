import numpy as np
from pyglm.glm import vec3

from ambra.config import CameraConfig, Config, PlaybackConfig
from ambra.primitives3d import Lines
from ambra.scene import (
    AnimationBoundary,
    FrameAnimation,
    as_property,
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
    ),
)

positions = np.array(
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

positions = np.linspace(positions, positions + np.array([0, 0, 0]), 100)

colors = np.array(
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

line_width = np.linspace(1, 32, 100)

# scale = np.linspace(np.array([1, 1, 1]), np.array([1, 1, 3]), 100)
translation = as_property(
    np.linspace(np.array([0, 0, 0]), np.array([0, 1, 1]), 50),
    np.float32,
    (3,),
    FrameAnimation(AnimationBoundary.MIRROR),
)

line = Lines(positions, colors, line_width, scale=None, translation=translation)

translation = as_property(
    np.linspace(np.array([1, 0, 0]), np.array([2, 0, 0]), 50),
    np.float32,
    (3,),
    FrameAnimation(AnimationBoundary.MIRROR),
)
line2 = Lines(positions, colors, line_width, scale=None, translation=translation)
viewer.viewport.scene.objects.extend([line, line2])

viewer.run()
