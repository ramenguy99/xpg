import numpy as np

from ambra.config import CameraConfig, CameraProjection, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.primitives3d import Lines
from ambra.property import AnimationBoundary, ArrayBufferProperty, FrameAnimation
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        world_up=(0, 1, 0),
        gui=GuiConfig(
            multiviewport=True,
            initial_number_of_viewports=2,
            stats=True,
            inspector=True,
        ),
        renderer=RendererConfig(
            msaa_samples=4,
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

line_width = 4

line = Lines(positions, colors, line_width)

viewer.scene.objects.extend([line])

viewer.run()
