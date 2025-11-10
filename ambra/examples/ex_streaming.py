import numpy as np
from pyglm.glm import vec3

from ambra.config import (
    CameraConfig,
    CameraProjection,
    Config,
    PlaybackConfig,
    RendererConfig,
    UploadMethod,
)
from ambra.primitives2d import Lines
from ambra.property import ArrayBufferProperty, UploadSettings
from ambra.transform3d import RigidTransform3D
from ambra.viewer import Viewer

viewer = Viewer(
    "primitives",
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
        renderer=RendererConfig(
            force_buffer_upload_method=UploadMethod.GRAPHICS_QUEUE,
        ),
    ),
)

# positions = [
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
    100,
)
# ]
# print(positions.shape)

colors = [
    np.array(
        [
            0xFF0000FF,
            0xFF0000FF,
            0xFF00FF00,
            0xFF00FF00,
        ],
        np.uint32,
    ),
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

line_width = 4

positions = ArrayBufferProperty(positions, np.float32, upload=UploadSettings(preupload=False))
# positions = ArrayBufferProperty(positions, np.float32, upload=UploadSettings(preupload=False, async_load=True))

line = Lines(positions, colors, line_width)
viewer.viewport.scene.objects.extend([line])

viewer.run()
