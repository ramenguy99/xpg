from pathlib import Path

from pyxpg import imgui

from ambra.config import Config, DefaultFontPreference, GuiConfig
from ambra.viewer import Viewer


class CustomViewer(Viewer):
    def on_gui(self):
        imgui.push_font(font, 15)
        super().on_gui()
        imgui.pop_font()


viewer = CustomViewer(
    "Fonts",
    config=Config(
        gui=GuiConfig(
            stats=True,
            default_font_preference=DefaultFontPreference.VECTOR,
        ),
    ),
)

path = Path(__file__).absolute().parent.parent.parent.joinpath("Roboto-Medium.ttf")
font = viewer.gui.add_font_ttf("Roboto", path.read_bytes())

viewer.run()
