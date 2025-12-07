import numpy as np
from pyglm.glm import vec3

from ambra.config import CameraConfig, CameraProjection, Config, PlaybackConfig
from ambra.primitives2d import Lines
from ambra.property import AnimationBoundary, ArrayBufferProperty, FrameAnimation
from ambra.transform3d import RigidTransform3D
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        camera=CameraConfig(
            projection=CameraProjection.ORTHOGRAPHIC,
            ortho_half_extents=(10, 10),
            position=(0, 0, 0),
            target=(0, 0, 1),
        ),
        world_up=(0, 1, 0),
    ),
)

positions = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ],
    np.float32,
)

positions = np.linspace(positions, positions * np.array([2, 1]), 100)

colors = np.array(
    [
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
    ],
    np.uint32,
)

line_width = np.linspace(1, 32, 100)

scale = np.linspace(np.array([1, 1]), np.array([1, 3]), 100)
rotation = ArrayBufferProperty(
    np.linspace(0, 4 * np.pi, 50),
    np.float32,
)
translation = ArrayBufferProperty(
    np.linspace(np.array([0, 0]), np.array([1, 1]), 50),
    np.float32,
    FrameAnimation(AnimationBoundary.MIRROR),
)

line = Lines(
    positions,
    colors,
    line_width,
    scale=scale,
    translation=translation,
    rotation=rotation,
)

translation = ArrayBufferProperty(
    np.linspace(np.array([1, 0]), np.array([2, 0]), 50),
    np.float32,
    FrameAnimation(AnimationBoundary.MIRROR),
)
line2 = Lines(positions, colors, line_width, scale=None, translation=translation)

viewer.scene.objects.extend([line, line2])

viewer.run()
