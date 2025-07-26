from ambra.scene import FrameAnimation, DataProperty, as_property, AnimationBoundary
from ambra.viewer import Viewer
from ambra.primitives2d import Lines
from ambra.transform3d import RigidTransform3D
from ambra.config import Config, PlaybackConfig, CameraType
from pyglm.glm import vec3

import numpy as np

viewer = Viewer("primitives", 1280, 720, config=Config(
    playback=PlaybackConfig(
        enabled=True,
        playing=True,
    ),
    camera_type=CameraType.ORTHOGRAPHIC,
    ortho_half_extents=(10, 10),
))

positions = np.array([
    [ 0.0,  0.0],
    [ 1.0,  0.0],
    [ 0.0,  0.0],
    [ 0.0,  1.0],
], np.float32)

positions = np.linspace(positions, positions * np.array([2, 1]), 100)

colors = np.array([
    0xFF0000FF,
    0xFF0000FF,
    0xFF00FF00,
    0xFF00FF00,
], np.uint32)

line_width = np.linspace(1, 32, 100)

scale = np.linspace(np.array([1, 1]), np.array([1, 3]), 100)
rotation = as_property(np.linspace(0, 4 * np.pi, 50), np.float32, animation=FrameAnimation(AnimationBoundary.HOLD))
translation = as_property(np.linspace(np.array([0, 0]), np.array([1, 1]), 50), np.float32, (2,), FrameAnimation(AnimationBoundary.MIRROR))

line = Lines(positions, colors, line_width, scale=scale, translation=translation, rotation=rotation)

translation = as_property(np.linspace(np.array([1, 0]), np.array([2, 0]), 50), np.float32, (2,), FrameAnimation(AnimationBoundary.MIRROR))
line2 = Lines(positions, colors, line_width, scale=None, translation=translation)

viewer.viewport.camera.camera_from_world = RigidTransform3D.look_at(vec3(0, 0, -1), vec3(0, 0, 0), vec3(0, -1, 0))
viewer.viewport.scene.objects.extend([line, line2])

viewer.run()