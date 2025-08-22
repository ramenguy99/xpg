import logging
import struct
from pathlib import Path

import numpy as np

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.primitives3d import Mesh
from ambra.scene import StreamingProperty, UploadSettings
from ambra.utils.io import (
    read_exact,
    read_exact_at_offset,
    read_exact_at_offset_into,
)
from ambra.viewer import Viewer

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s.%(msecs)03d] %(levelname)-6s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

viewer = Viewer(
    "primitives",
    config=Config(
        window_x=10,
        window_y=50,
        window_width=1900,
        window_height=1000,
        # vsync = False,
        preferred_frames_in_flight=3,
        playback=PlaybackConfig(
            enabled=True,
            playing=True,
            frames_per_second=25.0,
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
            position=(10, -10, 10),
            target=(0, 0, 0),
        ),
        world_up=(0, -1, 0),
    ),
)

path = Path("N:\\scenes\\smpl\\all_frames_20.bin")
files = [open(path, "rb", buffering=0) for _ in range(viewer.renderer.num_workers)]
file = files[0]
header = read_exact(file, 12)
N = struct.unpack("<I", header[0:4])[0]
V = struct.unpack("<I", header[4:8])[0]
I = struct.unpack("<I", header[8:12])[0]

indices = np.frombuffer(read_exact_at_offset(file, N * V * 12 + len(header), I * 4), np.uint32)


class FileStreamingProperty(StreamingProperty):
    def max_size(self):
        return V * 12

    def _get_size_offset(frame_index: int):
        return V * 12, 12 + frame_index * V * 12

    def get_frame_by_index_into(self, frame_index: int, out: memoryview, thread_index: int = -1) -> int:
        size, offset = FileStreamingProperty._get_size_offset(frame_index)
        read_exact_at_offset_into(files[thread_index], offset, out[:size])
        return size

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1):
        size, offset = FileStreamingProperty._get_size_offset(frame_index)
        buf: np.ndarray = np.empty(size, np.uint8)
        read_exact_at_offset_into(files[thread_index], offset, buf.data)
        return buf.view(np.float32).reshape((-1, 3), copy=False)


positions = FileStreamingProperty(
    N,
    np.float32,
    (-1, 3),
    upload=UploadSettings(
        preupload=False,
        async_load=True,
        cpu_prefetch_count=2,
        gpu_prefetch_count=2,
    ),
)

mesh = Mesh(positions, indices=indices)
viewer.viewport.scene.objects.append(mesh)

viewer.run()
