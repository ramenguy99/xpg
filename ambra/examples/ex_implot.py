import numpy as np
from pyxpg import imgui
from pyxpg.imgui import implot

from ambra.config import Config, GuiConfig, RendererConfig
from ambra.utils.hook import hook
from ambra.viewer import Viewer

data = np.array(
    [
        [1, 5, 3, 2],
        [10, 50, 30, 20],
        [100, 500, 300, 200],
    ],
    np.int64,
)


class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        if imgui.begin("YO")[0]:
            if implot.begin_plot("Plot", flags=implot.PlotFlags.NO_FRAME):
                implot.plot_line("Line", data[:, 0], data[:, 1][::-1])
                implot.end_plot()
        imgui.end()


viewer = CustomViewer(
    config=Config(
        preferred_frames_in_flight=3,
        gui=GuiConfig(
            stats=True,
        ),
    ),
)

viewer.run()
