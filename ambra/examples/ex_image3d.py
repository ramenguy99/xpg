from ambra.scene import FrameAnimation, as_property, AnimationBoundary
from ambra.viewer import Viewer
from ambra.primitives3d import Image
from ambra.config import Config, PlaybackConfig
from ambra.utils.gpu import Format

import numpy as np

viewer = Viewer("primitives", 1280, 720, config=Config(
    playback=PlaybackConfig(
        enabled=True,
        playing=True,
    ),
    camera_position=(0.5, 0.5, -2),
    camera_target=(0, 0, 0),
    camera_up=(0, 1, 0),
))

translation = as_property(np.linspace(np.array([0, 0, 0]), np.array([5, 0, 0]), 100), np.float32, (3,), FrameAnimation(AnimationBoundary.HOLD))

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

img = Image(image, Format.R8G8B8A8_UNORM, translation=translation, scale=(1, H / W, 1))
viewer.viewport.scene.objects.extend([img])

viewer.run()