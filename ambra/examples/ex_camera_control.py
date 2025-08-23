import numpy as np
from pyglm.glm import vec3

from ambra.config import CameraConfig, CameraControlMode, Config, GuiConfig, PlaybackConfig
from ambra.keybindings import KeyMap, Modifiers, MouseButton, MouseButtonBinding
from ambra.primitives3d import Lines
from ambra.viewer import Viewer

viewer = Viewer(
    "primitives",
    config=Config(
        wait_events=True,
        playback=PlaybackConfig(
            enabled=True,
            playing=True,
        ),
        gui=GuiConfig(
            stats=True,
        ),
        world_up=(0, 1, 0),
        camera=CameraConfig(
            position=(0, 3, 3),
            target=(0, 0, 0),
        ),
    ),
    key_map=KeyMap(
        camera_rotate=MouseButtonBinding(MouseButton.LEFT, Modifiers.NONE),
        camera_pan=MouseButtonBinding(MouseButton.RIGHT, Modifiers.NONE),
    ),
)

positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    np.float32,
)

colors = np.array(
    [
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
        0xFFFF0000,
        0xFFFF0000,
    ],
    np.uint32,
)

line_width = 4

line = Lines(positions, colors, line_width)

viewer.viewport.scene.objects.extend([line])

viewer.run()
