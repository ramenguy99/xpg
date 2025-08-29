from ambra.config import Config, GuiConfig, RendererConfig
from ambra.viewer import Viewer

viewer = Viewer(
    "Hello World",
    config=Config(
        preferred_frames_in_flight=3,
        gui=GuiConfig(
            stats=True,
        ),
    ),
)
viewer.run()
