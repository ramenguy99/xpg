import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyxpg import imgui
from scipy.spatial.transform import Rotation

from ambra.config import CameraConfig, Config, GuiConfig, RendererConfig
from ambra.primitives3d import GaussianSplats, GaussianSplatsRenderFlags, Lines
from ambra.property import ArrayBufferProperty, UploadSettings
from ambra.utils.hook import hook
from ambra.viewer import Viewer


@dataclass
class Splats:
    position: np.ndarray
    sh: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray


def load_ply(path: Path, sh_degree: int = 1):
    with open(path, "rb") as f:
        # Read header.
        head = f.readline().decode("utf-8").strip().lower()
        if head != "ply":
            raise ValueError(f"Not a ply file: {head}")

        encoding = f.readline().decode("utf-8").strip().lower()
        if "binary_little_endian" not in encoding:
            raise ValueError(f"Invalid encoding: {encoding}")

        elements = f.readline().decode("utf-8").strip().lower()
        count = int(elements.split()[2])

        # Read until end of header.
        # TODO: properly parse header
        while f.readline().decode("utf-8").strip().lower() != "end_header":
            pass

        # Number of 32 bit floats used to encode Spherical Harmonics coefficients.
        # The last multiplication by 3 is because we have 3 components (RGB) for each coefficient.
        sh_coeffs = (sh_degree + 1) * (sh_degree + 1) * 3

        # Position (vec3), normal (vec3), spherical harmonics (sh_coeffs), opacity (float),
        # scale (vec3) and rotation (quaternion). All values are float32 (4 bytes).
        size = count * (3 + 3 + sh_coeffs + 1 + 3 + 4) * 4

        data = f.read(size)
        arr = np.frombuffer(data, dtype=np.float32).reshape((count, -1))

        # Positions.
        position = arr[:, :3].copy()

        # Currently we don't need normals for rendering.
        # normal = arr[:, 3:6].copy()

        # Spherical harmonic coefficients.
        sh = arr[:, 6 : 6 + sh_coeffs].copy()

        # Activate alpha: sigmoid(alpha).
        opacity = 1.0 / (1.0 + np.exp(-arr[:, 6 + sh_coeffs]))

        # Exponentiate scale.
        scale = np.exp(arr[:, 7 + sh_coeffs : 10 + sh_coeffs])

        # Normalize quaternions.
        rotation = arr[:, 10 + sh_coeffs : 14 + sh_coeffs].copy()
        rotation /= np.linalg.norm(rotation, ord=2, axis=1)[..., np.newaxis]

        # Convert from wxyz to xyzw.
        rotation = np.roll(rotation, -1, axis=1)

        return Splats(position, sh, opacity, scale, rotation)


splats = load_ply(sys.argv[1], 3)

if False:
    positions = np.array([[1, 0, 0]], np.float32)
    colors = np.array([[0.5, 0, 0.0, 1]], np.float32)
    sh = np.zeros((1, 13), np.uint8)
    covariances = np.array([[1, 0, 0, 2, 0, 3]], np.float32) * 1
else:
    positions = splats.position
    # positions = np.vstack((np.linspace(positions * 0.5, positions, 30), np.linspace(positions, positions * 0.5, 30)))

    SH_C0 = 0.28209479177387814
    colors = np.hstack((splats.sh[:, :3] * SH_C0 + 0.5, splats.opacity[..., np.newaxis]))
    sh = np.transpose(
        np.clip(np.rint((splats.sh[:, 3:] * 0.5 + 0.5) * 255.0), 0.0, 255.0).astype(np.uint8).reshape((-1, 3, 15)),
        # splats.sh[:, 3:].astype(np.float16).reshape((-1, 3, 15)),
        # splats.sh[:, 3:].reshape((-1, 3, 15)),
        (0, 2, 1),
    ).reshape((-1, 45))

    rots = Rotation.from_quat(splats.rotation).as_matrix()
    scale = np.zeros((splats.scale.shape[0], 3, 3), np.float32)
    scale[:, 0, 0] = splats.scale[:, 0]
    scale[:, 1, 1] = splats.scale[:, 1]
    scale[:, 2, 2] = splats.scale[:, 2]

    covariance_matrices: np.ndarray = np.matmul(rots, scale)
    transformed_covariance: np.ndarray = np.matmul(covariance_matrices, np.transpose(covariance_matrices, (0, 2, 1)))
    covariances = transformed_covariance.reshape((-1, 9))[:, (0, 1, 2, 4, 5, 8)]

# positions = ArrayBufferProperty(
#     positions,
#     np.float32,
#     upload=UploadSettings(preupload=False, cpu_prefetch_count=2, gpu_prefetch_count=2, async_load=True),
# )
gs = GaussianSplats(positions, colors, sh, covariances)


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


v = CustomViewer(
    config=Config(
        gui=GuiConfig(
            stats=True,
            inspector=True,
            renderer=True,
        ),
        world_up=(0, -1, 0),
        renderer=RendererConfig(
            background_color=(0, 0, 0, 1),
        ),
        camera=CameraConfig(position=(-1, -1, -1)),
    )
)
v.scene.objects.append(gs)

positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    np.float32,
)

colors = np.array(
    [
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
        0xFFFF0000,
        0xFFFF0000,
    ],
    np.uint32,
)

line_width = 1

line = Lines(positions, colors, line_width)

v.scene.objects.extend([line])

v.run()
