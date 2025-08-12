from ambra.scene import StreamingProperty, UploadSettings
from ambra.viewer import Viewer
from ambra.primitives3d import Image
from ambra.transform3d import RigidTransform3D
from ambra.config import Config, PlaybackConfig, RendererConfig, UploadMethod, GuiConfig, CameraType

from pyglm.glm import vec3
import numpy as np

viewer = Viewer("primitives", 1900, 1000, config=Config(
    # vsync = False,
    # preferred_frames_in_flight=3,
    playback=PlaybackConfig(
        enabled=True,
        playing=True,
    ),
    renderer=RendererConfig(
        background_color=(0,0,0,1),
        # force_upload_method=UploadMethod.TRANSFER_QUEUE,
    ),
    gui=GuiConfig(
        stats=True,
        playback=True,
        inspector=True,
        renderer=True,
    ),
    camera_type=CameraType.ORTHOGRAPHIC,
    ortho_half_extents=(10, 10),
))

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

image_gen = GeneratedStreamingProperty(N, np.uint8, (H, W, C), upload=UploadSettings(
    preupload=False,
    # async_load=True,
    # cpu_prefetch_count=2,
    # gpu_prefetch_count=2,
))

image = Image(image_gen)
viewer.viewport.camera.camera_from_world = RigidTransform3D.look_at(vec3(0.5, 0.5, -2), vec3(0, 0, 0), vec3(0, 1, 0))
viewer.viewport.scene.objects.append(image)

viewer.run()