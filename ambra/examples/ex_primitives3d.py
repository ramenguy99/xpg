import numpy as np
from pyglm.glm import vec3

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.geometry import create_axis3d_lines_and_colors
from ambra.primitives3d import Lines
from ambra.property import (
    AnimationBoundary,
    ArrayBufferProperty,
    FrameAnimation,
)
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        camera=CameraConfig(
            position=vec3(3),
            target=vec3(0),
        ),
        gui=GuiConfig(
            playback=True,
            stats=True,
            inspector=True,
        ),
    ),
)

positions, colors = create_axis3d_lines_and_colors()
positions = np.linspace(positions, positions + np.array([0, 0, 0]), 100)
line_width = np.linspace(1, 32, 100)

translation = ArrayBufferProperty(
    np.linspace(np.array([0, 0, 0]), np.array([0, 1, 1]), 50),
    np.float32,
    FrameAnimation(boundary=AnimationBoundary.MIRROR),
)

line = Lines(positions, colors, line_width, scale=None, translation=translation)

translation = ArrayBufferProperty(
    np.linspace(np.array([1, 0, 0]), np.array([2, 0, 0]), 100),
    np.float32,
    FrameAnimation(start_frame=10, boundary=AnimationBoundary.MIRROR),
)
line2 = Lines(positions, colors, line_width, scale=None, translation=translation)
viewer.scene.objects.extend([line, line2])

viewer.run()
