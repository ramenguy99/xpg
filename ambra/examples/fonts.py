from pathlib import Path

from pyxpg import imgui

from ambra.config import Config, GuiConfig
from ambra.viewer import Viewer


class CustomViewer(Viewer):
    def on_gui(self):
        imgui.push_font_float(font, 15)
        super().on_gui()
        imgui.pop_font()


viewer = CustomViewer(
    "Fonts",
    config=Config(
        preferred_frames_in_flight=3,
        gui=GuiConfig(
            stats=True,
        ),
    ),
)

font = viewer.gui.add_font_ttf(
    "Roboto", Path(__file__).parent.parent.parent.joinpath("Roboto-Medium.ttf").read_bytes()
)
viewer.run()
