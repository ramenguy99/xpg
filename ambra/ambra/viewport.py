import math
from dataclasses import dataclass

from .scene import Scene
from .camera import Camera
from .config import PlaybackConfig

# TODO: playback config from viewer config for things like deafult fps/playing/time
class Playback:
    def __init__(self, config: PlaybackConfig):
        self.max_time = config.max_time
        self.playing = config.playing
        self.frames_per_second = config.frames_per_second

        if config.initial_frame is None:
            self.current_time = config.initial_time or 0.0
            self.current_frame = int(self.current_time * self.frames_per_second)
        else:
            self.current_time = config.initial_frame / self.frames_per_second
            self.current_frame = config.initial_frame

    
    def step(self, dt: float):
        if self.playing:
            self.current_time = math.fmod(self.current_time + dt, self.max_time) if self.max_time != 0.0 else 0.0
            self.current_frame = int(self.current_time * self.frames_per_second)
    
    def toggle_play_pause(self):
        self.playing = not self.playing

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
    # camera_control: CameraControl
    scene: Scene
    playback: Playback
