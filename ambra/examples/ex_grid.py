import numpy as np
from pyglm.glm import normalize, vec4
from pyxpg import imgui

from ambra.config import Config, GuiConfig, PlaybackConfig
from ambra.primitives3d import Grid, GridType
from ambra.viewer import Viewer


class CustomViewer(Viewer):
    def on_gui(self):
        global grid
        if imgui.begin("Hello")[0]:
            u, t = imgui.drag_float3("Translation", grid.translation.get_current(), 0.01)
            if u:
                grid.translation.update_frame(0, np.array(t))

            u, s = imgui.drag_float3("Scale", grid.scale.get_current(), 0.01)
            if u:
                grid.scale.update_frame(0, np.array(s))

            u, r = imgui.drag_float4("Rotation", grid.rotation.get_current(), 0.1)
            if u:
                grid.rotation.update_frame(0, np.array(normalize(vec4(r))))
        imgui.end()

        return super().on_gui()


viewer = CustomViewer(
    config=Config(
        gui=GuiConfig(
            stats=True,
        ),
    ),
)

grid = Grid.transparent_black_lines((100, 100), GridType.XY_PLANE)

viewer.scene.objects.extend([grid])
viewer.run()
