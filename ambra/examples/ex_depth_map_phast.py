import json
from pathlib import Path

import cv2
import numpy as np
from pyglm.glm import ivec2, mat4
from pyxpg import Action, Format, Key, Modifiers, imgui

from ambra.config import CameraConfig, Config, GuiConfig, RendererConfig
from ambra.geometry import create_axis3d_lines_and_colors
from ambra.primitives3d import GaussianSplatsDepthMap, GaussianSplatsRenderFlags, Lines
from ambra.property import ArrayImageProperty
from ambra.transform3d import RigidTransform3D
from ambra.utils.hook import hook
from ambra.viewer import Viewer
from ambra.widgets import ImageInspector

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

splat_scale = 0.8
depth_threshold = 0.03
max_std = 0.015
ratio_threshold = 0.05

SINGLE_IMAGE = False
if SINGLE_IMAGE:
    dataset_dir = Path(r"/mnt/scenes/balgrist/single")

    depth = np.load(dataset_dir.joinpath(camera, "depth_0000000000000000000.npy"))
    color = np.load(dataset_dir.joinpath(camera, "rgb_0000000000000000000.npy"))
    color = np.dstack((color, np.full((color.shape[0], color.shape[1], 1), 255, np.uint8)))

    depth_prop = ArrayImageProperty(depth[np.newaxis, ..., np.newaxis], Format.R16_UINT)
    color_prop = color
else:
    dataset_dir = Path(r"/mnt/scenes/balgrist/gs_small")
    cameras = sorted([f.name for f in dataset_dir.iterdir() if f.is_dir()])

    calib = json.load(dataset_dir.joinpath("cam_info.json").open("r"))

    timestamps = sorted([int(f.stem) for f in dataset_dir.joinpath(cameras[0], "images").iterdir()])[:150]
    
    w = calib[cameras[0]]["calibration"]["width"]
    h = calib[cameras[0]]["calibration"]["height"]

    color_imgs = np.empty((len(timestamps), len(cameras), h, w, 4), np.uint8)
    depth_imgs = np.empty((len(timestamps), len(cameras), h, w, 1), np.uint16)

    intrinsics = []
    extrinsics = []
    for camera_index, c in enumerate(cameras):
        intrin = calib[c]["calibration"]["intrinsic_matrix"]
        intrinsics.append((float(intrin[0]), float(intrin[4]), float(intrin[2]), float(intrin[5])))
        extrinsics.append(RigidTransform3D.from_mat4(mat4(np.array(calib[c]["c2w"]))))

        for frame_index, t in enumerate(timestamps):
            rgb = np.load(dataset_dir.joinpath(c, "images", f"{t:019}.npy"))
            depth = np.load(dataset_dir.joinpath(c, "depths", f"{t:019}.npy"))
            confidence = np.load(dataset_dir.joinpath(c, "confidence", f"{t:019}.npy"))

            depth[confidence >= 50] = 0

            alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, np.uint8)
            color_imgs[frame_index, camera_index] = np.dstack((rgb, alpha))
            depth_imgs[frame_index, camera_index] = depth[..., np.newaxis]
            # depthcmap_imgs.append(np.dstack((cv2.applyColorMap((depth >> 4).astype(np.uint8), cv2.COLORMAP_JET), alpha)))
        
    depth_prop = ArrayImageProperty(depth_imgs, Format.R16_UINT)
    color_prop = ArrayImageProperty(color_imgs, Format.R8G8B8A8_UNORM)

    gs = GaussianSplatsDepthMap(
        depth_prop,
        color_prop,
        intrinsics,
        extrinsics,
        max_standard_deviation=max_std,
        depth_threshold=depth_threshold,
        relative_ratio_threshold=ratio_threshold,
        cull_at_dist=False,
    )
    gs.splat_scale = splat_scale


class CustomViewer(Viewer):
    def on_key(self, key: Key, action: Action, modifiers: Modifiers):
        if imgui.get_io().want_capture_keyboard:
            return
        
        if action == Action.PRESS:
            if key == Key.N1:
                gs.camera_mask = gs.camera_mask ^ 0b0001
            elif key == Key.N2:
                gs.camera_mask = gs.camera_mask ^ 0b0010
            elif key == Key.N3:
                gs.camera_mask = gs.camera_mask ^ 0b0100
            elif key == Key.N4:
                gs.camera_mask = gs.camera_mask ^ 0b1000
            elif key == Key.N5:
                gs.camera_mask = 0b1111

        return super().on_key(key, action, modifiers)
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
            _, gs.max_standard_deviation = imgui.drag_float("Max std", gs.max_standard_deviation, v_speed=0.0001, v_min=0.0001, v_max=0.05, format="%.4f")
            _, gs.use_eigendec = imgui.checkbox("Use eigendec", gs.use_eigendec)
            _, gs.relative_ratio_threshold = imgui.drag_float("Ratio threshold", gs.relative_ratio_threshold, v_speed=0.001, v_min=0.0, v_max=1.0)
            _, gs.var_z = imgui.drag_float("Z variance", gs.var_z, v_speed=0.001, v_min=0.0, v_max=1.0)
            imgui.text(f"Camera mask: {gs.camera_mask:04b}")
        imgui.end()


CONSISTENCY_TEST = False

v = CustomViewer(
# v = Viewer(
    config=Config(
        window=not CONSISTENCY_TEST,
        window_width=1920,
        window_height=1080,
        # enable_synchronization_validation=False,
        enable_gpu_based_validation=True,
        gui=GuiConfig(
            stats=True,
            inspector=True,
            renderer=True,
            playback=True,
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


# class DepthImageInspector(ImageInspector):
#     def tooltip(self, pixel_image_coordinates: ivec2):
#         text = depth_prop.get_current()[pixel_image_coordinates.y, pixel_image_coordinates.x, 0] * 1e-3
#         imgui.text(f"({pixel_image_coordinates.x}, {pixel_image_coordinates.y}): {text:8.04}m")

# v.scene.widgets.extend([
#     ImageInspector("RGB", color_prop),
#     DepthImageInspector("Depth", depthcmap_prop),
# ])

v.run()
