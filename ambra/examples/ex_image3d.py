import numpy as np

from ambra.config import CameraConfig, Config, PlaybackConfig
from ambra.primitives3d import Image
from ambra.property import AnimationBoundary, ArrayBufferProperty
from ambra.utils.gpu import Format
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        camera=CameraConfig(
            position=(0.5, 0.5, -2),
            target=(0, 0, 0),
        ),
        world_up=(0, 1, 0),
    ),
)

translation = ArrayBufferProperty(
    np.linspace(np.array([0, 0, 0]), np.array([5, 0, 0]), 100),
    np.float32,
)

H, W = 32, 64
start = np.zeros((H, W, 4))
end = np.zeros((H, W, 4))
for i in range(H):
    for j in range(W):
        start[i, j, 0] = (i + 0.5) / H
        start[i, j, 1] = (j + 0.5) / W
        start[i, j, 3] = 0

        end[i, j, 0] = (j + 0.5) / W
        end[i, j, 2] = (i + 0.5) / H
        end[i, j, 3] = 0
image = (np.linspace(start, end, 100) * 255).astype(np.uint8)

img = Image(image, translation=translation, scale=(1, H / W, 1))
viewer.scene.objects.extend([img])

viewer.run()
