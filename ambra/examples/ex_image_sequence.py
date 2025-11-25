import numpy as np

from ambra.config import (
    CameraConfig,
    CameraProjection,
    Config,
    GuiConfig,
    PlaybackConfig,
    RendererConfig,
    UploadMethod,
)
from ambra.primitives2d import Image
from ambra.property import ImageProperty, UploadSettings
from ambra.utils.gpu import Format
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        # vsync = False,
        # preferred_frames_in_flight=3,
        playback=PlaybackConfig(
            playing=True,
        ),
        renderer=RendererConfig(
            background_color=(0, 0, 0, 1),
            # force_image_upload_method=UploadMethod.GRAPHICS_QUEUE,
            # force_image_upload_method=UploadMethod.TRANSFER_QUEUE,
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
            position=(0, 0, 0),
            target=(0, 0, 1),
        ),
        world_up=(0, 1, 0),
    ),
)

N = 100
W = 512
H = 256
C = 4


class GeneratedStreamingProperty(ImageProperty):
    def get_frame_by_index(self, frame_index: int, thread_index: int = -1):
        arr = np.full((H, W, C), frame_index, np.uint8)
        return arr


image_gen = GeneratedStreamingProperty(
    W,
    H,
    Format.R8G8B8A8_UNORM,
    N,
    upload=UploadSettings(
        preupload=False,
        async_load=True,
        cpu_prefetch_count=2,
        gpu_prefetch_count=2,
    ),
)

image = Image(image_gen)
viewer.scene.objects.append(image)

viewer.run()
