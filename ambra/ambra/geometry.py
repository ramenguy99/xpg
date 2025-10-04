import numpy as np

# TODO: replace with array version
def create_sphere(radius=1.0, rings=16, sectors=32):
    R = 1.0 / (rings - 1)
    S = 1.0 / (sectors - 1)

    vertices = np.zeros((rings * sectors, 3))
    v, n = 0, 0
    for r in range(rings):
        for s in range(sectors):
            y = np.sin(-np.pi / 2 + np.pi * r * R)
            x = np.cos(2 * np.pi * s * S) * np.sin(np.pi * r * R)
            z = np.sin(2 * np.pi * s * S) * np.sin(np.pi * r * R)

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