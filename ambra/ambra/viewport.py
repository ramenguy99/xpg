import math
from dataclasses import dataclass

from .scene import Scene
from .camera import Camera

# TODO: playback config from viewer config for things like deafult fps/playing/time
class Playback:
    def __init__(self):
        self.max_time = 0.0
        self.playing = False # Paused if false, Playing if True
        self.frames_per_second = 30.0

        self.current_time = 0.0
        self.current_frame = 0
    
    def step(self, dt: float):
        if self.playing:
            self.current_time = math.fmod(self.current_time + dt, self.max_time)
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
