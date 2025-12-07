import numpy as np
from pyglm.glm import vec3

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.geometry import create_sphere
from ambra.materials import ColorMaterial
from ambra.primitives3d import Mesh
from ambra.property import (
    AnimationBoundary,
    ArrayBufferProperty,
    FrameSampledAnimation,
    TimeSampledAnimation,
)
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
            max_time=3.0,
        ),
        camera=CameraConfig(
            position=vec3(0, 5, 0),
            target=vec3(0),
        ),
        gui=GuiConfig(
            stats=True,
            playback=True,
            inspector=True,
        ),
    ),
)

v, _, f = create_sphere(0.5, rings=32, sectors=64)

enabled_always = True

# Disabled at frame 0, enabled at frame 20, disabled again at frame 40
enabled_frame_range = ArrayBufferProperty(
    [1, 0],
    bool,
    animation=FrameSampledAnimation(
        [20, 40],
        AnimationBoundary.HOLD,
    ),
)

# Disabled at time 0s, enabled at time 1s, disabled again at time 2s
enabled_timestamp_range = ArrayBufferProperty(
    [1, 0],
    bool,
    animation=TimeSampledAnimation([1.0, 2.0], AnimationBoundary.HOLD),
)

always = Mesh(v, f, translation=(1.5, 0, 0.7), material=ColorMaterial((0.8, 0, 0)), enabled=enabled_always)
frame_range = Mesh(v, f, translation=(0, 0, 0.7), material=ColorMaterial((0, 0.8, 0)), enabled=enabled_frame_range)
timestamp_range = Mesh(
    v, f, translation=(-1.5, 0, 0.7), material=ColorMaterial((0, 0, 0.8)), enabled=enabled_timestamp_range
)

v_enabled_frame_range = ArrayBufferProperty(
    [v], np.float32, animation=FrameSampledAnimation([20, 40], boundary=AnimationBoundary.DISABLE)
)
v_enabled_timestamp_range = ArrayBufferProperty(
    [v], np.float32, animation=TimeSampledAnimation([1.0, 2.0], boundary=AnimationBoundary.DISABLE)
)

new_always = Mesh(v, f, translation=(1.5, 0, -0.7), material=ColorMaterial((0.8, 0, 0)))
new_frame_range = Mesh(v_enabled_frame_range, f, translation=(0, 0, -0.7), material=ColorMaterial((0, 0.8, 0)))
new_timestamp_range = Mesh(
    v_enabled_timestamp_range, f, translation=(-1.5, 0, -0.7), material=ColorMaterial((0, 0, 0.8))
)


viewer.scene.objects.extend(
    [
        always,
        frame_range,
        timestamp_range,
        new_always,
        new_frame_range,
        new_timestamp_range,
    ]
)

viewer.run()
