import math
from dataclasses import dataclass

from pyglm.glm import ivec2

from .camera import Camera
from .config import PlaybackConfig
from .scene import Scene


class Playback:
    def __init__(self, config: PlaybackConfig):
        self.max_time = config.max_time
        self.num_frames = int(self.max_time * config.frames_per_second) if self.max_time else 0

        self.playing = config.playing
        self.frames_per_second = config.frames_per_second
        if config.initial_frame is None:
            self.current_time = config.initial_time or 0.0
            self.current_frame = int(self.current_time * self.frames_per_second)
        else:
            self.set_frame(config.initial_frame)

    def set_max_time(self, max_time: float) -> None:
        self.max_time = max(max_time, 0.0)
        self.num_frames = int(self.max_time * self.frames_per_second)

    def step(self, dt: float) -> None:
        self.set_time(self.current_time + dt)

    def toggle_play_pause(self) -> None:
        self.playing = not self.playing

    def set_time(self, time: float) -> None:
        assert self.max_time is not None
        time = max(time, 0.0)
        self.current_time = math.fmod(time, self.max_time) if self.max_time != 0.0 else 0.0
        self.current_frame = int(self.current_time * self.frames_per_second)

    def set_frame(self, frame: int) -> None:
        self.current_frame = frame % self.num_frames
        self.current_time = self.current_frame / self.frames_per_second


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Viewport:
    rect: Rect
    camera: Camera
    # camera_control_mode: CameraControlMode
    scene: Scene
    playback: Playback

    def resize(self, width: int, height: int) -> None:
        self.rect.width = width
        self.rect.height = height
        self.camera.ar = width / height

    def update(self, dt: float) -> None:
        pass

    def on_rotate_press(self, position: ivec2) -> None:
        pass

    def on_rotate_release(self, position: ivec2) -> None:
        pass

    def on_pan_press(self, position: ivec2) -> None:
        pass

    def on_pan_release(self, position: ivec2) -> None:
        pass

    def on_move(self, position: ivec2) -> None:
        pass

    def on_scroll(sself, scroll: ivec2) -> None:
        pass
