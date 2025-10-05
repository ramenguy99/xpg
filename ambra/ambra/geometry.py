from typing import Tuple

import numpy as np
from pyglm.glm import vec3


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


_cube_positions = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    np.float32,
)
_cube_indices = np.array(
    [1, 5, 2, 2, 5, 6, 5, 4, 6, 6, 4, 7, 3, 2, 7, 7, 2, 6, 0, 1, 3, 3, 1, 2, 4, 0, 7, 7, 0, 3, 4, 5, 0, 0, 5, 1],
    np.uint32,
)


def create_cube(position: vec3 = (0, 0, 0), half_extents: vec3 = (1, 1, 1)):
    return _cube_positions * np.asarray(half_extents, np.float32) + np.asarray(
        position, np.float32
    ), _cube_indices.copy()
