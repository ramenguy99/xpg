import ambra
from  ambra.scene.primitives import Line

import numpy as np

viewer = ambra.Viewer("primitives", 1280, 720)


positions = np.array([
    [0.5, 0.5],
    [-0.5, -0.5],
], np.float32)

colors = np.array([
    0x0000FFFF,
    0xFF0000FF,
], np.uint32)

line = Line(positions, colors)
line.create(viewer.renderer)

viewer.scene.children.append(line)

viewer.run()