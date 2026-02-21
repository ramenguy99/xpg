import numpy as np

from ambra.config import Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.geometry import create_axis3d_lines_and_colors
from ambra.primitives3d import Lines
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        world_up=(0, 1, 0),
        gui=GuiConfig(
            multiviewport=True,
            initial_number_of_viewports=2,
            stats=True,
            inspector=True,
        ),
        renderer=RendererConfig(
            msaa_samples=4,
        ),
    ),
)

line = Lines(*create_axis3d_lines_and_colors(), 4)

viewer.scene.objects.extend([line])

viewer.run()
