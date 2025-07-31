from ambra.scene import FrameAnimation, AnimationBoundary, StreamingProperty, UploadSettings, as_property
from ambra.viewer import Viewer
from ambra.primitives3d import Mesh
from ambra.transform3d import RigidTransform3D
from ambra.config import Config, PlaybackConfig, CameraType, RendererConfig, UploadMethod
from ambra.utils.io import read_exact, read_exact_at_offset_into, read_exact_at_offset
from pyglm.glm import vec3

import numpy as np
import struct
from pathlib import Path
import io

viewer = Viewer("primitives", 1280, 720, config=Config(
    playback=PlaybackConfig(
        enabled=True,
        playing=True,
    ),
    renderer=RendererConfig(
        force_upload_method=UploadMethod.GFX,
    )
))

# self.files = [open(path, "rb", buffering=0) for _ in range(WORKERS)]

path = Path("N:\\scenes\\smpl\\all_frames_20.bin")
files = [open(path, "rb", buffering=0) for _ in range(viewer.config.renderer.thread_pool_workers)]
file = files[0]
header = read_exact(file, 12)
N = struct.unpack("<I", header[0: 4])[0]
V = struct.unpack("<I", header[4: 8])[0]
I = struct.unpack("<I", header[8:12])[0]

indices = np.frombuffer(read_exact_at_offset(file, N * V * 12 + len(header), I * 4), np.uint32)

class FileStreamingProperty(StreamingProperty):
    def _get_frame_by_index_into_impl(self, file: io.FileIO, frame_index: int, out: memoryview) -> int:
        offset = 12 + frame_index * V * 12
        size = V * 12
        read_exact_at_offset_into(file, offset, out[:size])
        return size

    def get_frame_by_index_into(self, frame_index, out: memoryview) -> int:
        return self._get_frame_by_index_into_impl(file, frame_index, out)

    def get_frame_by_index_into_async(self, frame_index: int, out: memoryview, thread_index: int) -> int:
        return self._get_frame_by_index_into_impl(files[thread_index], frame_index, out)

positions = FileStreamingProperty(N, np.float32, (-1, 3), upload=UploadSettings(
    preupload=False,
    async_load=True,
))

mesh = Mesh(positions, indices=indices)
viewer.viewport.camera.camera_from_world = RigidTransform3D.look_at(vec3(10), vec3(0), vec3(0, 0, 1))
viewer.viewport.scene.objects.extend([mesh])

viewer.run()