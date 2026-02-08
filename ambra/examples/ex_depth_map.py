import json
from pathlib import Path

import numpy as np
from numba import njit
from pyxpg import imgui

from ambra.config import CameraConfig, Config, GuiConfig, RendererConfig
from ambra.geometry import create_axis3d_lines_and_colors
from ambra.primitives3d import GaussianSplatsDepthMap, GaussianSplatsRenderFlags, Lines, Points
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

@njit(cache=True)
def unproject(x: float, y: float, depth: float, fx: float, fy: float, cx: float, cy: float):
    px = (x - cx) * depth / fx
    py = (y - cy) * depth / fy
    pz = depth
    return np.array([px, py, pz], np.float32)


@njit(cache=True)
def normalize(v):
    return v / np.linalg.norm(v)

@njit(cache=True)
def projected_sq_dist(p, c, axis):
    v = p - c
    return (np.dot(v, axis)) ** 2

@njit(cache=True)
def compute_rot_scales_from_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, positions: np.ndarray, covariances: np.ndarray, valid: np.ndarray):
    h, w = depth.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            d_c = depth[y, x]
            d_l = depth[y, x - 1]
            d_r = depth[y, x + 1]
            d_t = depth[y - 1, x]
            d_b = depth[y + 1, x]

            if (
                abs(d_l - d_c) > 0.05 or
                abs(d_r - d_c) > 0.05 or
                abs(d_t - d_c) > 0.05 or
                abs(d_b - d_c) > 0.05
            ):
                continue

            p_c = unproject(x, y, d_c, fx, fy, cx, cy)
            p_l = unproject(x - 1, y, d_l, fx, fy, cx, cy)
            p_r = unproject(x + 1, y, d_r, fx, fy, cx, cy)
            p_t = unproject(x, y - 1, d_t, fx, fy, cx, cy)
            p_b = unproject(x, y + 1, d_b, fx, fy, cx, cy)

            x_v = p_r - p_l
            y_v = p_t - p_b

            e1 = normalize(x_v)
            e2 = normalize(y_v)
            e2 = normalize(e2 - np.dot(e2, e1) * e1)
            e3 = normalize(np.cross(e1, e2))

            var_x = min(
                projected_sq_dist(p_l, p_c, e1),
                projected_sq_dist(p_r, p_c, e1)
            )

            var_y = min(
                projected_sq_dist(p_t, p_c, e2),
                projected_sq_dist(p_b, p_c, e2)
            )

            # var_x = np.dot(x_v, x_v) * 0.25
            # var_y = np.dot(y_v, y_v) * 0.25

            # var_x = min(var_x, 0.05)
            # var_y = min(var_y, 0.05)
            var_x = min(var_x, 0.01 * 0.01)
            var_y = min(var_y, 0.01 * 0.01)
            var_z = 0.0 #z_variance_scale * (var_x + var_y) * 0.5

            R = np.column_stack((e1.astype(np.float32), e2.astype(np.float32), e3.astype(np.float32)))
            D = np.array([
                [var_x, 0, 0],
                [0, var_y, 0],
                [0, 0, var_z],
            ], np.float32)
            cov = R @ D @ R.T

            covariances[y, x, :, :] = cov
            # covariances[y, x, :, :] = D
            positions[y, x, :] = p_c
            valid[y, x] = 1


camera = "35554540"
# dataset_dir = Path(r"N:\scenes\balgrist")
dataset_dir = Path(r"/mnt/scenes/balgrist/single")
calib = json.load(dataset_dir.joinpath("cam_info.json").open("r"))

intrin = calib[camera]["calibration"]["intrinsic_matrix"]
fx, fy, cx, cy = float(intrin[0]), float(intrin[4]), float(intrin[2]), float(intrin[5])
depth = np.load(dataset_dir.joinpath(camera, "depth_0000000000000000000.npy")).astype(np.float32) * 1e-3
color_u8 = np.load(dataset_dir.joinpath(camera, "rgb_0000000000000000000.npy")).astype(np.uint32)
color_u32 = np.full_like(color_u8[:, :, 0], 0xFF000000, np.uint32) | (color_u8[:, :, 2] << 16) |(color_u8[:, :, 1] << 8) | (color_u8[:, :, 0])
color = color_u8.astype(np.float32) / 255.0
color = np.dstack((color, np.ones((color.shape[0], color.shape[1], 1), np.float32)))

positions = np.zeros((*depth.shape, 3), np.float32)
covariances = np.zeros((*depth.shape, 3, 3), np.float32)
valid = np.zeros(depth.shape, np.uint8)

compute_rot_scales_from_depth(depth, fx, fy, cx, cy, positions, covariances, valid)
# print(valid.sum(), depth.size)
# exit(1)

colors = color.reshape((-1, 4))
positions = positions.reshape((-1, 3))
covariances = covariances.reshape((-1, 9))[:, (0, 1, 2, 4, 5, 8)]
sh = np.zeros((positions.shape[0], 45), np.uint8)

print(positions.shape, positions.dtype)
print(colors.shape, colors.dtype)
print(sh.shape, sh.dtype)
print(covariances.shape, covariances.dtype)
# exit(1)

# gs = Points(positions, color_u32.flatten())
gs = GaussianSplatsDepthMap(
    positions,
    colors,
    sh,
    covariances,
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
        imgui.end()


CONSISTENCY_TEST = False

v = CustomViewer(
# v = Viewer(
    config=Config(
        window=not CONSISTENCY_TEST,
        window_width=1920,
        window_height=1080,
        # enable_synchronization_validation=False,
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
