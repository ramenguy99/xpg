from typing import Any, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from pyglm.glm import mat3, mat4, normalize, vec3


def create_sphere(
    radius: float = 1.0, rings: int = 16, sectors: int = 32
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    inv_r = 1.0 / (rings - 1)
    inv_s = 1.0 / (sectors - 1)

    r = np.arange(rings, dtype=np.float32)
    s = np.arange(sectors, dtype=np.float32)

    sin_r = np.sin(np.pi * r * inv_r)
    y_r = np.sin(-np.pi / 2 + np.pi * r * inv_r)
    cos_s = np.cos(2 * np.pi * s * inv_s)
    sin_s = np.sin(2 * np.pi * s * inv_s)

    # Broadcasting (rings, 1) * (1, sectors)
    x = (sin_r[:, None] * cos_s[None, :]).reshape(-1)
    y = np.repeat(y_r, sectors)
    z = (sin_r[:, None] * sin_s[None, :]).reshape(-1)

    normals = np.empty((rings * sectors, 3), dtype=np.float32)
    normals[:, 0] = x
    normals[:, 1] = y
    normals[:, 2] = z

    vertices = normals * radius

    r = np.arange(rings - 1, dtype=np.uint32)
    s = np.arange(sectors - 1, dtype=np.uint32)

    base = (r[:, None] * sectors + s[None, :]).reshape(-1)

    v0 = base
    v1 = base + sectors + 1
    v2 = base + 1
    v3 = base + sectors

    faces = np.empty(((rings - 1) * (sectors - 1) * 2, 3), dtype=np.uint32)
    faces[0::2, 0] = v0
    faces[0::2, 1] = v1
    faces[0::2, 2] = v2
    faces[1::2, 0] = v0
    faces[1::2, 1] = v3
    faces[1::2, 2] = v1

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
    faces[:, 1] = idxs
    faces[:-1, 2] = idxs[1:]
    faces[-1, 2] = 1

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
    normals = normals / np.linalg.norm(normals, axis=1)[..., np.newaxis]

    idxs_bot = np.arange(0, sectors, dtype=np.uint32)
    idxs_top = idxs_bot + sectors
    faces_top = np.zeros((sectors, 3), dtype=np.uint32)
    faces_top[:, 0] = idxs_top
    faces_top[:, 1] = idxs_bot
    faces_top[:-1, 2] = idxs_top[1:]
    faces_top[-1, 2] = idxs_top[0]
    faces_bot = np.zeros((sectors, 3), dtype=np.uint32)
    faces_bot[:, 0] = faces_top[:, 2]
    faces_bot[:, 1] = idxs_bot
    faces_bot[:-1, 2] = idxs_bot[1:]
    faces_bot[-1, 2] = idxs_bot[0]
    faces = np.concatenate([faces_top, faces_bot], axis=0)

    return vertices, normals, faces.flatten()


def create_capped_cylinder(
    radius: float = 1.0, height: float = 1.0, sectors: int = 8
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    cylinder_v, cylinder_n, cylinder_f = create_cylinder(radius, height, sectors)
    cap_v, cap_n, cap_f = create_disk(radius, sectors)

    # Translate top cap
    top_cap_v = cap_v + np.array([0.0, 0.0, height], np.float32)

    # Flip bot cap normals and windng order
    bot_cap_n = -cap_n
    bot_cap_f = cap_f.copy().reshape((-1, 3))
    bot_cap_f[:, (2, 1)] = bot_cap_f[:, (1, 2)]
    bot_cap_f = bot_cap_f.flatten()  # type: ignore

    (v, n), f = concatenate_meshes(
        [
            (cap_v, bot_cap_n),
            (top_cap_v, cap_n),
            (cylinder_v, cylinder_n),
        ],
        [bot_cap_f, cap_f, cylinder_f],
    )
    return v, n, f


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


def create_capped_cone(
    radius: float = 1.0, height: float = 1.0, sectors: int = 8
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    cap_v, cap_n, cap_f = create_disk(radius, sectors)
    cap_n = -cap_n
    cap_f = cap_f.copy().reshape((-1, 3))
    cap_f[:, (2, 1)] = cap_f[:, (1, 2)]
    cap_f = cap_f.flatten()

    cone_v, cone_n, cone_f = create_cone(radius, height, sectors)
    (v, n), f = concatenate_meshes(
        [
            (cap_v, cap_n),
            (cone_v, cone_n),
        ],
        [cap_f, cone_f],
    )
    return v, n, f


def create_arrow(
    radius: float = 0.1, height: float = 0.8, tip_radius: float = 0.2, tip_height: float = 0.2, sectors: int = 8
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    bot_cap_v, cap_n, cap_f = create_disk(radius, sectors)
    top_cap_v, _, _ = create_disk(tip_radius, sectors)

    cap_n = -cap_n
    cap_f = cap_f.copy().reshape((-1, 3))
    cap_f[:, (2, 1)] = cap_f[:, (1, 2)]
    cap_f = cap_f.flatten()
    top_cap_v = top_cap_v + np.array([0.0, 0.0, height], np.float32)

    cylinder_v, cylinder_n, cylinder_f = create_cylinder(radius, height, sectors)
    tip_v, tip_n, tip_f = create_cone(tip_radius, tip_height, sectors)
    tip_v[:, 2] += height
    (v, n), f = concatenate_meshes(
        [
            (bot_cap_v, cap_n),
            (top_cap_v, cap_n),
            (cylinder_v, cylinder_n),
            (tip_v, tip_n),
        ],
        [cap_f, cap_f, cylinder_f, tip_f],
    )
    return v, n, f


def create_plane(
    position: Tuple[float, float, float] = (0, 0, 0), extents: Tuple[float, float] = (1, 1)
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]]:
    v = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        np.float32,
    ) * np.array([extents[0], extents[1], 0.0], np.float32) + np.array(position, np.float32)

    n = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        np.float32,
    )

    f = np.array([0, 1, 2, 0, 2, 3], np.uint32)

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


def create_cube_edges(
    min_p: Tuple[float, float, float] = (-0.5, -0.5, -0.5),
    max_p: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    color: int = 0xFFFF0000,
) -> Tuple[NDArray[np.float32], NDArray[np.uint32]]:
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

    return lines, np.full(lines.shape[0], color, np.uint32)


def create_normal_lines(
    positions: NDArray[np.float32],
    normals: NDArray[np.float32],
    length: float = 0.01,
    color: int = 0xFFFF0000,
) -> Tuple[NDArray[np.float32], NDArray[np.uint32]]:
    n = positions.shape[0]
    lines = np.zeros((n * 2, 3), np.float32)
    lines[::2] = positions
    lines[1::2] = positions + normals * length
    return lines, np.full(lines.shape[0], color, np.uint32)


def create_axis3d_lines_and_colors(
    length: float = 1.0,
    x_color: int = 0xFF0000FF,
    y_color: int = 0xFF00FF00,
    z_color: int = 0xFFFF0000,
) -> Tuple[NDArray[np.float32], NDArray[np.uint32]]:
    lines = (
        np.array(
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
        * length
    )

    colors = np.array(
        [
            x_color,
            x_color,
            y_color,
            y_color,
            z_color,
            z_color,
        ],
        np.uint32,
    )

    return lines, colors


def create_axis2d_lines_and_colors(
    length: float = 1.0,
    x_color: int = 0xFF0000FF,
    y_color: int = 0xFF00FF00,
) -> Tuple[NDArray[np.float32], NDArray[np.uint32]]:
    lines = (
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            np.float32,
        )
        * length
    )

    colors = np.array(
        [
            x_color,
            x_color,
            y_color,
            y_color,
        ],
        np.uint32,
    )

    return lines, colors


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
