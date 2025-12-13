import numpy as np

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.primitives3d import ColormapDistanceToPlane, ColormapKind, Grid, GridType, Voxels
from ambra.viewer import Viewer, imgui


def grid3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.meshgrid(x, y, z)
    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))


positions: np.ndarray = grid3d(
    np.linspace(-5, 5, 11, endpoint=True, dtype=np.float32),
    np.linspace(-5, 5, 11, endpoint=True, dtype=np.float32),
    np.linspace(-5, 5, 11, endpoint=True, dtype=np.float32),
)


class CustomViewer(Viewer):
    def on_gui(self):
        if imgui.begin("Voxels")[0]:
            u, border_size = imgui.drag_float("Border size", colors_voxels.border_size, 0.01, 0, 1.0)
            if u:
                colors_voxels.border_size = border_size
                uniform_color_voxels.border_size = border_size
                colormap_voxels.border_size = border_size
            u, border_factor = imgui.drag_float("Border factor", colors_voxels.border_factor, 0.01, 0.0, 2.0)
            if u:
                colors_voxels.border_factor = border_factor
                uniform_color_voxels.border_factor = border_factor
                colormap_voxels.border_factor = border_factor

        imgui.end()
        return super().on_gui()


viewer = CustomViewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        gui=GuiConfig(
            stats=True,
            inspector=True,
            renderer=True,
            playback=True,
        ),
        renderer=RendererConfig(
            msaa_samples=4,
        ),
        camera=CameraConfig(
            position=(0, 25, 25),
            z_near=0.1,
        ),
    ),
)

angle = np.linspace(0, 2.0 * np.pi, 100, dtype=np.float32)
sin = np.sin(angle * 0.5)
cos = np.cos(angle * 0.5)
rotation = np.column_stack((cos, np.zeros_like(sin), np.zeros_like(cos), sin))

colors: np.ndarray = grid3d(
    np.linspace(0, 1.0, 11, endpoint=True, dtype=np.float32),
    np.linspace(0, 1.0, 11, endpoint=True, dtype=np.float32),
    np.linspace(0, 1.0, 11, endpoint=True, dtype=np.float32),
)
colors = (
    (colors[:, 0] * 255).astype(np.uint32)
    | (colors[:, 1] * 255).astype(np.uint32) << 8
    | (colors[:, 2] * 255).astype(np.uint32) << 16
    | 0xFF000000
)
colors_voxels = Voxels(positions, 1.0, colors=colors, rotation=rotation, translation=(-17, 0, 0))
colormap_voxels = Voxels(positions, 1.0, colormap=ColormapDistanceToPlane(ColormapKind.JET, -5, 5), rotation=rotation)
uniform_color_voxels = Voxels(positions, 1.0, uniform_color=0xFFCCCCCC, rotation=rotation, translation=(17, 0, 0))

viewer.scene.objects.extend(
    [
        colors_voxels,
        colormap_voxels,
        uniform_color_voxels,
    ]
)
viewer.run()
