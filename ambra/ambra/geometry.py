from typing import Any, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from pyglm.glm import mat3, mat4, normalize, vec3


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


def create_cone(
    radius: float = 1.0, height: float = 1.0, sectors: int = 8
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    angle = 2 * np.pi / sectors

    c = np.cos(np.arange(sectors, dtype=np.float32) * angle)
    s = np.sin(np.arange(sectors, dtype=np.float32) * angle)
    vertices = np.zeros((sectors + 1, 3), np.float32)
    vertices[:sectors, 0] = radius * c
    vertices[:sectors, 1] = radius * s
    vertices[sectors, 2] = height

    normal = np.array(normalize(vec3(height, 0.0, radius)))
    normals = np.zeros_like(vertices)
    normals[:sectors, 0] = c * normal[0]
    normals[:sectors, 1] = s * normal[0]

    faces = np.zeros((sectors, 3), dtype=np.uint32)
    faces[:, 0] = np.arange(sectors, dtype=np.uint32)
    faces[:-1, 1] = faces[1:, 0]
    faces[-1, 1] = 0
    faces[:, 2] = sectors
    return vertices, normals, faces.flatten()


def create_arrow(
    radius: float = 0.1, height: float = 0.8, tip_radius: float = 0.2, tip_height: float = 0.2, sectors: int = 8
):
    bottom_lid_v, bottom_lid_n, bottom_lid_f = create_disk(radius, sectors)
    top_lid_v, top_lid_n, top_lid_f = create_disk(tip_radius, sectors)
    top_lid_v[:, 2] = height
    top_lid_n[:, 2] = -1.0
    cylinder_v, cylinder_n, cylinder_f = create_cylinder(radius, height, sectors)
    tip_v, tip_n, tip_f = create_cone(tip_radius, tip_height, sectors)
    tip_v[:, 2] += height
    (v, n), f = concatenate_meshes(
        [
            (bottom_lid_v, bottom_lid_n),
            (top_lid_v, top_lid_n),
            (cylinder_v, cylinder_n),
            (tip_v, tip_n),
        ],
        [bottom_lid_f, top_lid_f, cylinder_f, tip_f],
    )
    return v, n, f


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


def create_cube_edges(min_p: Tuple[float, float, float], max_p: Tuple[float, float, float]) -> NDArray[np.float32]:
    vertices = np.zeros((8, 3), dtype=np.float32)
    vertices[0:4] = min_p
    vertices[1, 0] = max_p[0]
    vertices[2, 0:2] = max_p[0:2]
    vertices[3, 1] = max_p[1]
    vertices[4:] = max_p
    vertices[4, 0:2] = min_p[0:2]
    vertices[7, 0] = min_p[0]
    vertices[5, 1] = min_p[1]

    lines = np.zeros((24, 3), dtype=np.float32)
    lines[0:2] = vertices[0:2]
    lines[2:4] = vertices[1:3]
    lines[4:6] = vertices[2:4]
    lines[6:8] = vertices[[3, 0]]
    lines[8:10] = vertices[4:6]
    lines[10:12] = vertices[5:7]
    lines[12:14] = vertices[6:8]
    lines[14:16] = vertices[[7, 4]]
    lines[16:18] = vertices[[0, 4]]
    lines[18:20] = vertices[[1, 5]]
    lines[20:22] = vertices[[2, 6]]
    lines[22:24] = vertices[[3, 7]]

    return lines


def create_normal_lines(positions: NDArray[np.float32], normals: NDArray[np.float32], length: float = 0.01):
    n = positions.shape[0]
    lines = np.zeros([n * 2, 3])
    lines[::2] = positions
    lines[1::2] = positions + normals * length
    return lines


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
