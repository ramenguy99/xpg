import numpy as np

from ambra.config import CameraConfig, CameraProjection, Config, GuiConfig, PlaybackConfig
from ambra.primitives2d import Lines
from ambra.property import AnimationBoundary, ArrayBufferProperty, FrameAnimation
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
        gui=GuiConfig(
            multiviewport=True,
        ),
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
line = Lines(
    positions,
    colors,
    line_width,
)

viewer.scene.objects.extend([line])

viewer.run()
