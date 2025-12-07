from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from time import sleep
from typing import Callable, Optional

import numpy as np
from pyglm.glm import vec3
from pyxpg import imgui

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.primitives3d import Colormap, ColormapDistanceToPlane, ColormapKind, Points
from ambra.property import AnimationBoundary, ListBufferProperty, ListTimeSampledAnimation, UploadSettings
from ambra.viewer import Viewer


@dataclass
class CameraReading:
    camera_index: int
    data: np.ndarray


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
        self.thread = Thread(target=self._entry)
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


def main():
    queue: Queue[CameraReading] = Queue()

    def on_frame(reading: CameraReading):
        queue.put(reading)

    camera = Camera(0, "", on_frame)

    try:
        pc: Optional[Points] = None

        total_idx = 0.0
        points_animation = ListTimeSampledAnimation(AnimationBoundary.HOLD, [total_idx])

        class CustomViewer(Viewer):
            def on_draw(self):
                nonlocal pc, total_idx
                if not queue.empty():
                    reading = queue.get()
                    if pc is None:
                        if False:
                            points_property = ListBufferProperty(
                                [reading.data],
                                np.float32,
                                (-1, 3),
                                reading.data.nbytes,
                                upload=UploadSettings(batched=False),
                            )
                        else:
                            points_property = ListBufferProperty(
                                [reading.data],
                                np.float32,
                                (-1, 3),
                                reading.data.nbytes,
                                # animation=points_animation,
                                upload=UploadSettings(batched=False, preupload=False),
                            )
                        pc = Points(
                            points_property, colormap=ColormapDistanceToPlane(ColormapKind.JET, 0, 1), point_size=3
                        )
                        self.scene.objects.append(pc)
                    else:
                        total_idx += 1

                        # pc.points.append_frames([reading.data] * 10)

                        pc.points.append_frame(reading.data)
                        # points_animation.timestamps.append(float(total_idx))

                        self.playback.set_max_time(pc.points.end_animation_time(self.playback.frames_per_second))
                        if True:
                            if not self.gui_playback_slider_held:
                                self.playback.set_frame(self.playback.num_frames - 1)
                return super().on_draw()

            def on_gui(self):
                nonlocal pc

                super().on_gui()
                if imgui.begin("Editor")[0]:
                    if pc is not None:
                        u, ps = imgui.slider_float("point size", pc.point_size.get_current(), 1, 10)
                        if u:
                            pc.point_size.update_frame(0, ps)

                        if imgui.button("Remove frames"):
                            pc.points.remove_frame_range(1, pc.points.num_frames)
                            pc.points.update(self.playback.current_time, self.playback.current_frame)

                        if imgui.button("Destroy"):
                            self.scene.objects.clear()
                            pc.destroy()
                            pc = None

                imgui.end()

        viewer = CustomViewer(
            config=Config(
                playback=PlaybackConfig(
                    playing=False,
                ),
                camera=CameraConfig(
                    position=vec3(3),
                    target=vec3(0),
                ),
                gui=GuiConfig(stats=True, playback=True, renderer=True, inspector=True),
            ),
        )

        viewer.run()
    finally:
        camera.stop()


main()
