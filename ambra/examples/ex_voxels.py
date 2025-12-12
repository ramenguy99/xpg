from ambra.config import Config, GuiConfig, PlaybackConfig
from ambra.primitives3d import Grid, GridType, Voxels, ColormapDistanceToPlane, ColormapKind
from ambra.viewer import Viewer

import numpy as np

def grid3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.meshgrid(x, y, z)
    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

N = 25
R = 10
positions: np.ndarray = grid3d(np.linspace(-R, R, N, dtype=np.float32), np.linspace(-R, R, N, dtype=np.float32), np.linspace(-R, R, N, dtype=np.float32))

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        gui=GuiConfig(
            stats=True,
            inspector=True,
            renderer=True,
        ),
    ),
)

voxels = Voxels(positions, 0.1, ColormapDistanceToPlane(ColormapKind.JET, -5, 5))

# grid = Grid.transparent_black_lines((100, 100), GridType.XY_PLANE)

viewer.scene.objects.extend([
    voxels,
    # grid,
])
viewer.run()
