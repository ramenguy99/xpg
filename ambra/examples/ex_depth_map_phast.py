import json
from pathlib import Path
from ambra.geometry import create_axis3d_lines_and_colors
from ambra.property import ArrayImageProperty
import numpy as np
from pyxpg import Format, imgui

from ambra.config import CameraConfig, Config, GuiConfig, RendererConfig
from ambra.primitives3d import GaussianSplatsDepthMap, GaussianSplatsRenderFlags, Lines
from ambra.utils.hook import hook
from ambra.viewer import Viewer

# Edge filtering
# - Bilateral filtering
# - Edge-aware robust filtering
# - Use color and depth together
# - Integrate confidence

# Covariance estimation / filtering
# - Find orhtonormal axis with a better method than graham-schmidt
# - Wider neighbour footprint (e.g. 5x5)
# - Detect single direction neighborhood (use only left or right instead of both)
# - Bounded anisotropy (relative difference between variances)


camera = "35554540"
# dataset_dir = Path(r"N:\scenes\balgrist")
dataset_dir = Path(r"/mnt/scenes/balgrist/single")
calib = json.load(dataset_dir.joinpath("cam_info.json").open("r"))

intrin = calib[camera]["calibration"]["intrinsic_matrix"]
fx, fy, cx, cy = float(intrin[0]), float(intrin[4]), float(intrin[2]), float(intrin[5])
depth = np.load(dataset_dir.joinpath(camera, "depth_0000000000000000000.npy"))
color = np.load(dataset_dir.joinpath(camera, "rgb_0000000000000000000.npy"))
color = np.dstack((color, np.ones((color.shape[0], color.shape[1], 1), np.uint8)))


depth_prop = ArrayImageProperty(depth[np.newaxis, ..., np.newaxis], Format.R16_UINT)

# gs = Points(positions, color_u32.flatten())
gs = GaussianSplatsDepthMap(
    depth_prop,
    color,
    (fx, fy, cx, cy),
    cull_at_dist=False,
)

class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        if imgui.begin("Gaussian splatting")[0]:
            _, v = imgui.checkbox("Disable opacity", (gs.flags & GaussianSplatsRenderFlags.DISABLE_OPACITY) != 0)
            if v:
                gs.flags |= GaussianSplatsRenderFlags.DISABLE_OPACITY
            else:
                gs.flags &= ~GaussianSplatsRenderFlags.DISABLE_OPACITY

            _, v = imgui.checkbox(
                "Show SH only", (gs.flags & GaussianSplatsRenderFlags.SHOW_SPHERICAL_HARMONICS_ONLY) != 0
            )
            if v:
                gs.flags |= GaussianSplatsRenderFlags.SHOW_SPHERICAL_HARMONICS_ONLY
            else:
                gs.flags &= ~GaussianSplatsRenderFlags.SHOW_SPHERICAL_HARMONICS_ONLY

            _, v = imgui.checkbox("Point cloud mode", (gs.flags & GaussianSplatsRenderFlags.POINT_CLOUD_MODE) != 0)
            if v:
                gs.flags |= GaussianSplatsRenderFlags.POINT_CLOUD_MODE
            else:
                gs.flags &= ~GaussianSplatsRenderFlags.POINT_CLOUD_MODE
            _, gs.alpha_cull_threshold = imgui.drag_float(
                "Alpha cull threshold", gs.alpha_cull_threshold, v_speed=0.01, v_min=0.0, v_max=1.0
            )
            _, gs.frustum_dilation = imgui.drag_float(
                "Frustum dilation", gs.frustum_dilation, v_speed=0.01, v_min=0.0, v_max=1.0
            )
            _, gs.splat_scale = imgui.drag_float("Splat scale", gs.splat_scale, v_speed=0.01, v_min=0.0, v_max=10.0)

            _, gs.depth_threshold = imgui.drag_float("Depth threshold", gs.depth_threshold, v_speed=0.001, v_min=0.001, v_max=1.0)
            _, gs.max_standard_deviation = imgui.drag_float("Max std", gs.max_standard_deviation, v_speed=0.00001, v_min=0.0001, v_max=0.05, format=r'%.5f')
        imgui.end()


CONSISTENCY_TEST = False

v = CustomViewer(
# v = Viewer(
    config=Config(
        window=not CONSISTENCY_TEST,
        window_width=1920,
        window_height=1080,
        # enable_synchronization_validation=False,
        # enable_gpu_based_validation=True,
        gui=GuiConfig(
            stats=True,
            inspector=True,
            renderer=True,
            # multiviewport=True,
            # initial_number_of_viewports=2,
            # initial_number_of_viewports=1,
        )
        if not CONSISTENCY_TEST
        else GuiConfig(),
        world_up=(0, -1, 0),
        renderer=RendererConfig(
            # background_color=(0, 0, 0, 0),
            background_color=(1, 1, 1, 1),
        ),
        camera=CameraConfig(
            # position=(-1, -1, -1)
            position=(3.252, 0.544, -1.437),
            target=(-0.5, 0.65, 2.82),
        ),
    )
)

ax3d_lines, ax3d_colors = create_axis3d_lines_and_colors()
line = Lines(ax3d_lines, ax3d_colors)

v.scene.objects.extend([
    gs,
    line
])

v.run()
