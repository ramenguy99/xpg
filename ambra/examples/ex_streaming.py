from ambra.scene import FrameAnimation, AnimationBoundary, StreamingProperty, UploadSettings, as_property
from ambra.viewer import Viewer
from ambra.primitives2d import Lines
from ambra.transform3d import RigidTransform3D
from ambra.config import Config, PlaybackConfig, CameraType, RendererConfig, UploadMethod
from pyglm.glm import vec3

import numpy as np

viewer = Viewer("primitives", 1280, 720, config=Config(
    playback=PlaybackConfig(
        enabled=True,
        playing=True,
    ),
    camera_type=CameraType.ORTHOGRAPHIC,
    ortho_half_extents=(10, 10),
    renderer=RendererConfig(
        force_upload_method=UploadMethod.GFX,
    )
))

# positions = [
positions = np.linspace(
    np.array([
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 0.0,  0.0],
        [ 1.0,  1.0],
    ], np.float32),
    np.array([
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 0.0,  0.0],
        [ 1.0,  1.0],
    ], np.float32) * 10,
100)
# ]
# print(positions.shape)

colors = [
    np.array([
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
    ], np.uint32),
    np.array([
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
        0xFF000000,
        0xFF000000,
    ], np.uint32),
]

line_width = 4

# positions = StreamingProperty(np.linspace(np.array([0, 0]), np.array([1, 1]), 50), np.float32, (2,), FrameAnimation(AnimationBoundary.MIRROR))
positions = as_property(positions, np.float32, (-1, 2), upload=UploadSettings(preupload=False))
# positions = as_property(positions, np.float32, (-1, 2), upload=UploadSettings(preupload=False, async_load=True))

line = Lines(positions, colors, line_width)
viewer.viewport.camera.camera_from_world = RigidTransform3D.look_at(vec3(0, 0, -1), vec3(0, 0, 0), vec3(0, -1, 0))
viewer.viewport.scene.objects.extend([line])

viewer.run()