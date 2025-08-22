import numpy as np

from ambra.config import (
    CameraConfig,
    CameraProjection,
    Config,
    GuiConfig,
    PlaybackConfig,
    RendererConfig,
)
from ambra.primitives2d import Image
from ambra.scene import StreamingProperty, UploadSettings
from ambra.utils.gpu import Format
from ambra.viewer import Viewer

viewer = Viewer(
    "primitives",
    config=Config(
        # vsync = False,
        # preferred_frames_in_flight=3,
        playback=PlaybackConfig(
            enabled=True,
            playing=True,
        ),
        renderer=RendererConfig(
            background_color=(0, 0, 0, 1),
        ),
        gui=GuiConfig(
            stats=True,
            playback=True,
            inspector=True,
            renderer=True,
        ),
        camera=CameraConfig(
            projection=CameraProjection.ORTHOGRAPHIC,
            ortho_half_extents=(2, 2),
        ),
    ),
)

N = 100
W = 512
H = 256
C = 4


class GeneratedStreamingProperty(StreamingProperty):
    def width(self):
        return W

    def height(self):
        return H

    def channels(self):
        return C

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1):
        arr = np.full((H, W, C), frame_index, np.uint8)
        return arr


image_gen = GeneratedStreamingProperty(
    N,
    np.uint8,
    (H, W, C),
    upload=UploadSettings(
        preupload=False,
        async_load=True,
        cpu_prefetch_count=2,
        gpu_prefetch_count=2,
    ),
)

image = Image(image_gen, Format.R8G8B8A8_UNORM)
viewer.viewport.scene.objects.append(image)

viewer.run()
