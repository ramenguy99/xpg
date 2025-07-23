from ambra.viewer import Viewer
from ambra.primitives3d import Line
from ambra.transform3d import RigidTransform
from ambra.config import Config, PlaybackConfig
from pyglm.glm import vec3

import numpy as np

viewer = Viewer("primitives", 1280, 720, config=Config(
    playback=PlaybackConfig(
        enabled=True,
        playing=True,
    )
))

positions = np.array([
    [ 0.0,  0.0, 0.0],
    [ 1.0,  0.0, 0.0],
    [ 0.0,  0.0, 0.0],
    [ 0.0,  1.0, 0.0],
    [ 0.0,  0.0, 0.0],
    [ 0.0,  0.0, 1.0],
], np.float32)

positions = np.linspace(positions, positions + np.array([3, 0, 0]), 100)

colors = np.array([
    0xFF0000FF,
    0xFF0000FF,
    0xFF00FF00,
    0xFF00FF00,
    0xFFFF0000,
    0xFFFF0000,
], np.uint32)

line_width = np.linspace(1, 32, 100)
line = Line(positions, colors, line_width)
line.create(viewer.renderer)

viewer.viewport.camera.camera_from_world = RigidTransform.look_at(vec3(3), vec3(0), vec3(0, 0, 1))
viewer.viewport.scene.objects.append(line)

viewer.run()