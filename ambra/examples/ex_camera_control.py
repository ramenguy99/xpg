from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.geometry import create_axis3d_lines_and_colors
from ambra.keybindings import KeyMap, Modifiers, MouseButton, MouseButtonBinding
from ambra.primitives3d import Lines
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        wait_events=True,
        playback=PlaybackConfig(
            playing=True,
        ),
        gui=GuiConfig(
            stats=True,
        ),
        world_up=(0, 1, 0),
        camera=CameraConfig(
            position=(3, 3, 3),
            target=(0, 0, 0),
        ),
    ),
    key_map=KeyMap(
        camera_rotate=MouseButtonBinding(MouseButton.LEFT, Modifiers.NONE),
        camera_pan=MouseButtonBinding(MouseButton.RIGHT, Modifiers.NONE),
    ),
)

line = Lines(*create_axis3d_lines_and_colors(), 4)
viewer.scene.objects.extend([line])
viewer.run()
