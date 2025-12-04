from __future__ import annotations

from threading import Thread
from queue import Queue
from dataclasses import dataclass
from typing import Callable, Optional
from time import sleep
import numpy as np
from pyglm.glm import vec3
import time
from pyxpg import imgui

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.primitives3d import Colormap, ColormapDistanceToPlane, ColormapKind, Points
from ambra.viewer import Viewer

@dataclass
class CameraReading:
    camera_index: int
    data: bytes

WIDTH = 320
HEIGHT = 240

class Camera:
    def __init__(
        self,
        camera_index: int,
        device_path: str,
        on_frame: Callable[[CameraReading], None],
    ):
        self.camera_index = camera_index
        self.device_path = device_path
        self.on_frame = on_frame

        self.should_stop = False
        self.thread = Thread(target=self._entry, daemon=True)
        self.thread.start()

    def _entry(self):
        # TODO: open camera and start streaming

        start_t = time.monotonic_ns()
        while True:
            if self.should_stop:
                break
            sleep(1.0 / 30.0)

            t = (time.monotonic_ns() - start_t) * 1e-9 * 2
            x, y = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
            x = x.flatten() / WIDTH * 2.0
            y = y.flatten() / HEIGHT * 2.0
            z = np.sin(x * 5 + t) * np.cos(y * 5 + t) * 0.5 + 0.5
            pts = np.vstack((x, y, z), dtype=np.float32).T

            self.on_frame(CameraReading(self.camera_index, pts))

    def stop(self):
        self.should_stop = True
        self.thread.join()

queue: Queue[CameraReading] = Queue()

def on_frame(reading: CameraReading):
    queue.put(reading)
camera = Camera(0, "", on_frame)

pc: Optional[Points] = None

class CustomViewer(Viewer):
    def on_draw(self):
        global pc
        if not queue.empty():
            reading = queue.get()
            if pc is None:
                pc = Points(reading.data, colormap=ColormapDistanceToPlane(ColormapKind.JET, 0, 1), point_size=3)
                self.scene.objects.append(pc)
            else:
                pc.points.update_frame(0, reading.data)
        return super().on_draw()

    def on_gui(self):
        global pc
        if imgui.begin("Editor")[0]:
            if pc is not None:
                u, ps = imgui.slider_float("point size", pc.point_size.get_current(), 1, 10)
                if u:
                    pc.point_size.update_frame(0, ps)

        imgui.end()
        return super().on_gui()


viewer = CustomViewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        camera=CameraConfig(
            position=vec3(3),
            target=vec3(0),
        ),
        gui=GuiConfig(stats=True),
    ),
)

viewer.run()

camera.stop()