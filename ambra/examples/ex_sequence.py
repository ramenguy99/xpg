import logging
import struct
from pathlib import Path

import numpy as np

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig, UploadMethod
from ambra.primitives3d import Mesh
from ambra.property import BufferProperty, UploadSettings
from ambra.utils.hook import hook
from ambra.utils.io import (
    read_exact,
    read_exact_at_offset,
    read_exact_at_offset_into,
)
from ambra.viewer import Viewer, imgui

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s.%(msecs)03d] %(levelname)-6s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NUM_WORKERS = 4

path = Path("N:\\scenes\\smpl\\all_frames_10.bin")
files = [open(path, "rb", buffering=0) for _ in range(NUM_WORKERS)]
file = files[0]
header = read_exact(file, 12)
N = struct.unpack("<I", header[0:4])[0]
V = struct.unpack("<I", header[4:8])[0]
I = struct.unpack("<I", header[8:12])[0]

indices = np.frombuffer(read_exact_at_offset(file, N * V * 12 + len(header), I * 4), np.uint32)


scale = 1.0


class FileStreamingProperty(BufferProperty):
    def _get_size_offset(frame_index: int):
        return V * 12, 12 + frame_index * V * 12

    # def get_frame_by_index_into(self, frame_index: int, out: memoryview, thread_index: int = -1) -> int:
    #     size, offset = FileStreamingProperty._get_size_offset(frame_index)
    #     read_exact_at_offset_into(files[thread_index], offset, out[:size])
    #     return size

    def get_frame_by_index(self, frame_index: int, thread_index: int = -1):
        size, offset = FileStreamingProperty._get_size_offset(frame_index)
        buf: np.ndarray = np.empty(size, np.uint8)
        read_exact_at_offset_into(files[thread_index], offset, buf.data)
        return buf.view(np.float32).reshape((-1, 3), copy=False) * scale


positions = FileStreamingProperty(
    V * 12,
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


class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        global scale
        if imgui.begin("Control"):
            u, scale = imgui.slider_float("SCALE", scale, 0.1, 10.0)
            if u:
                mesh.positions_buffer.invalidate_frame(mesh.positions.current_frame_index)
        imgui.end()


viewer = CustomViewer(
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
            force_buffer_upload_method=UploadMethod.GFX,
            upload_buffer_count=2,
            thread_pool_workers=NUM_WORKERS,
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

viewer.viewport.scene.objects.append(mesh)

viewer.run()
