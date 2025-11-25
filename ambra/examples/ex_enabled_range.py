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
        ),
    ),
)

v, f = create_sphere(0.5, rings=32, sectors=64)

enabled_always = True

# Disabled at frame 0, enabled at frame 20, disabled again at frame 40
enabled_frame_range = ArrayBufferProperty(
    [0, 1, 0],
    bool,
    animation=FrameSampledAnimation(AnimationBoundary.HOLD, [0, 20, 40]),
)

# Enabled at time 0s, enabled at time 1s, disabled again at time 2s
enabled_timestamp_range = ArrayBufferProperty(
    [0, 1, 0],
    bool,
    animation=TimeSampledAnimation(AnimationBoundary.HOLD, [0.0, 1.0, 2.0]),
)

always = Mesh(v, f, translation=(1.5, 0, 0), material=ColorMaterial((0.8, 0, 0)), enabled=enabled_always)
frame_range = Mesh(v, f, translation=(0, 0, 0), material=ColorMaterial((0, 0.8, 0)), enabled=enabled_frame_range)
timestamp_range = Mesh(
    v, f, translation=(-1.5, 0, 0), material=ColorMaterial((0, 0, 0.8)), enabled=enabled_timestamp_range
)

viewer.scene.objects.extend([always, frame_range, timestamp_range])

viewer.run()
