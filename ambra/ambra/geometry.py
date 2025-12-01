from typing import Any, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from pyglm.glm import mat3, mat4


# TODO: replace with array version
def create_sphere(
    radius: float = 1.0, rings: int = 16, sectors: int = 32
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    inv_r = 1.0 / (rings - 1)
    inv_s = 1.0 / (sectors - 1)

    vertices = np.zeros((rings * sectors, 3), np.float32)
    v, n = 0, 0
    for r in range(rings):
        for s in range(sectors):
            y = np.sin(-np.pi / 2 + np.pi * r * inv_r)
            x = np.cos(2 * np.pi * s * inv_s) * np.sin(np.pi * r * inv_r)
            z = np.sin(2 * np.pi * s * inv_s) * np.sin(np.pi * r * inv_r)

            vertices[v] = np.array([x, y, z], np.float32) * radius

            v += 1
            n += 1

    normals = vertices.copy()

    faces = np.zeros([(rings - 1) * (sectors - 1) * 2, 3], dtype=np.uint32)
    i = 0
    for r in range(rings - 1):
        for s in range(sectors - 1):
            faces[i] = np.array([r * sectors + s, (r + 1) * sectors + (s + 1), r * sectors + (s + 1)], np.uint32)
            faces[i + 1] = np.array([r * sectors + s, (r + 1) * sectors + s, (r + 1) * sectors + (s + 1)], np.uint32)
            i += 2

    return vertices, normals, faces.flatten()


def create_disk(
    radius: float = 1.0, sectors: int = 8
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    angle = 2 * np.pi / sectors

    vertices = np.zeros((sectors + 1, 3), np.float32)
    vertices[1:, 0] = radius * np.cos(np.arange(sectors, dtype=np.float32) * angle)
    vertices[1:, 1] = radius * np.sin(np.arange(sectors, dtype=np.float32) * angle)

    normals = np.zeros_like(vertices)
    normals[:, 2] = 1.0

    faces = np.zeros((sectors, 3), dtype=np.uint32)
    idxs = np.array(range(1, sectors + 1), dtype=np.uint32)
    faces[:, 2] = idxs
    faces[:-1, 1] = idxs[1:]
    faces[-1, 1] = 1

    return vertices, normals, faces.flatten()


def create_cylinder(
    radius: float = 1.0, height: float = 1.0, sectors: int = 8
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    angle = 2 * np.pi / sectors

    vertices = np.zeros((sectors * 2, 3), np.float32)
    vertices[:sectors, 0] = radius * np.cos(np.arange(sectors, dtype=np.float32) * angle)
    vertices[:sectors, 1] = radius * np.sin(np.arange(sectors, dtype=np.float32) * angle)
    vertices[sectors:, :2] = vertices[:sectors, :2]
    vertices[:sectors, 2] = 0
    vertices[sectors:, 2] = height

    normals = np.zeros_like(vertices)
    normals[:, :2] = vertices[:, :2]

    idxs_bot = np.arange(0, sectors, dtype=np.uint32)
    idxs_top = idxs_bot + sectors
    faces_top = np.zeros((sectors, 3), dtype=np.uint32)
    faces_top[:, 0] = idxs_top
    faces_top[:-1, 1] = idxs_top[1:]
    faces_top[-1, 1] = idxs_top[0]
    faces_top[:, 2] = idxs_bot
    faces_bot = np.zeros((sectors, 3), dtype=np.uint32)
    faces_bot[:, 0] = faces_top[:, 1]
    faces_bot[:-1, 1] = idxs_bot[1:]
    faces_bot[-1, 1] = idxs_bot[0]
    faces_bot[:, 2] = idxs_bot
    faces = np.concatenate([faces_top, faces_bot], axis=0)

    return vertices, normals, faces.flatten()


_cube_positions = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ],
    np.float32,
)

_cube_normals = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    np.float32,
)

_cube_indices = np.array(
    [
        #  front and back
        0,
        3,
        2,
        2,
        1,
        0,
        4,
        5,
        6,
        6,
        7,
        4,
        #  left and right
        11,
        8,
        9,
        9,
        10,
        11,
        12,
        13,
        14,
        14,
        15,
        12,
        #  bottom and top
        16,
        17,
        18,
        18,
        19,
        16,
        20,
        21,
        22,
        22,
        23,
        20,
    ],
    np.uint32,
)


def create_cube(
    position: Tuple[float, float, float] = (0, 0, 0), extents: Tuple[float, float, float] = (1, 1, 1)
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    return (
        _cube_positions * np.asarray(extents, np.float32) + np.asarray(position, np.float32),
        _cube_normals.copy(),
        _cube_indices.copy(),
    )


def concatenate_meshes(
    attributes: Sequence[Sequence[NDArray[Any]]], indices: Sequence[NDArray[np.uint32]]
) -> Tuple[List[NDArray[np.float32]], NDArray[np.uint32]]:
    total_offset = 0
    offsets = []
    for m in attributes:
        a_len = m[0].shape[0]
        for a in m[1:]:
            if a_len != a.shape[0]:
                raise ValueError("all attributes must have the same number of vertices")
        offsets.append(total_offset)
        total_offset += a_len

    cc_attributes = [
        np.concatenate([attributes[m][i] for m in range(len(attributes))]) for i in range(len(attributes[0]))
    ]
    cc_indices = np.concatenate([idx + o for idx, o in zip(indices, offsets)])
    return cc_attributes, cc_indices


def transform_positions(transform: mat4, vertices: NDArray[np.float32]) -> NDArray[np.float32]:
    t = np.array(transform)
    v_h = np.hstack((vertices, np.ones((vertices.shape[0], 1), np.float32)))
    return (v_h @ t.T)[:, :3]  # type: ignore


def transform_directions(transform: mat3, directions: NDArray[np.float32]) -> NDArray[np.float32]:
    t = np.array(transform)
    return directions @ t.T  # type: ignore


def transform_mesh(
    transform: mat4, vertices: NDArray[np.float32], normals: NDArray[np.float32]
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    return transform_positions(transform, vertices), transform_directions(mat3(transform), normals)
