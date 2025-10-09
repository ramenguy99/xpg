import numpy as np

from ambra.config import CameraConfig, CameraProjection, Config, PlaybackConfig, RendererConfig
from ambra.primitives3d import Grid, Lines, GridType
from ambra.viewer import Viewer

viewer = Viewer(
    "primitives",
    config=Config(
        playback=PlaybackConfig(
            enabled=True,
            playing=True,
        ),
    ),
)

grid = Grid.transparent_black_lines((100, 100), GridType.XY_PLANE)

viewer.viewport.scene.objects.extend([grid])
viewer.run()
