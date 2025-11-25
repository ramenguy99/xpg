import numpy as np

positions = np.linspace(
    np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        np.float32,
    ),
    np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        np.float32,
    )
    * 10,
    10,
)

# print(positions)
# print(positions.shape)

if False:
    N = positions.shape[0]
    positions = np.repeat(positions, 1000000, 0).reshape((N, -1, 2))
else:
    N = positions.shape[0]
    positions = np.repeat(positions, 1000, 0).reshape((-1, N, 2))

# print(positions)
print(positions.shape)
# exit(1)

colors = [
    np.array(
        [
            0xFF0000FF,
            0xFF0000FF,
            0xFF00FF00,
            0xFF00FF00,
            0xFF000000,
            0xFF000000,
        ],
        np.uint32,
    ),
]
colors = np.repeat(colors, 2, 1)
print(colors.shape)

from ambra.config import CameraConfig, CameraProjection, Config, GuiConfig, RendererConfig
from ambra.primitives2d import Lines
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        preferred_frames_in_flight=2,
        gui=GuiConfig(
            stats=True,
        ),
        renderer=RendererConfig(
            upload_buffer_size=1024 * 1024 * 32,
            upload_buffer_count=1,
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

line_width = 4

line = Lines(positions, colors, line_width)
viewer.scene.objects.extend([line])

viewer.run()
