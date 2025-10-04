from typing import Tuple

import numpy as np


# TODO: replace with array version
def create_sphere(radius: float = 1.0, rings: int = 16, sectors: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    inv_r = 1.0 / (rings - 1)
    inv_s = 1.0 / (sectors - 1)

    vertices = np.zeros((rings * sectors, 3))
    v, n = 0, 0
    for r in range(rings):
        for s in range(sectors):
            y = np.sin(-np.pi / 2 + np.pi * r * inv_r)
            x = np.cos(2 * np.pi * s * inv_s) * np.sin(np.pi * r * inv_r)
            z = np.sin(2 * np.pi * s * inv_s) * np.sin(np.pi * r * inv_r)

            vertices[v] = np.array([x, y, z]) * radius

            v += 1
            n += 1

    faces = np.zeros([(rings - 1) * (sectors - 1) * 2, 3], dtype=np.int32)
    i = 0
    for r in range(rings - 1):
        for s in range(sectors - 1):
            faces[i] = np.array([r * sectors + s, (r + 1) * sectors + (s + 1), r * sectors + (s + 1)])
            faces[i + 1] = np.array([r * sectors + s, (r + 1) * sectors + s, (r + 1) * sectors + (s + 1)])
            i += 2

    return vertices, faces.flatten()
