import numpy as np

from ambra.config import CameraConfig, CameraProjection, Config, PlaybackConfig
from ambra.primitives2d import Image
from ambra.scene import AnimationBoundary, FrameAnimation, as_property
from ambra.utils.gpu import Format
from ambra.viewer import Viewer

viewer = Viewer(
    "primitives",
    config=Config(
        playback=PlaybackConfig(
            enabled=True,
            playing=True,
        ),
        camera=CameraConfig(
            projection=CameraProjection.ORTHOGRAPHIC,
            ortho_half_extents=(10, 10),
        ),
    ),
)

# scale = np.linspace(np.array([1, 1]), np.array([1, 3]), 100)
rotation = as_property(
    np.linspace(0, 4 * np.pi, 50),
    np.float32,
    animation=FrameAnimation(AnimationBoundary.MIRROR),
)
translation = as_property(
    np.linspace(np.array([0, 0]), np.array([5, 0]), 100),
    np.float32,
    (2,),
    FrameAnimation(AnimationBoundary.HOLD),
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
img = Image(
    image,
    Format.R8G8B8A8_UNORM,
    translation=translation,
    rotation=rotation,
    scale=(1, H / W),
)

viewer.viewport.scene.objects.extend([img])
viewer.run()
