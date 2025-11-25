from ambra.config import Config, PlaybackConfig
from ambra.primitives3d import Grid, GridType
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
    ),
)

grid = Grid.transparent_black_lines((100, 100), GridType.XY_PLANE)

viewer.scene.objects.extend([grid])
viewer.run()
