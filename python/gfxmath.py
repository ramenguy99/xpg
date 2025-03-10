import numpy as np

def grid3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.meshgrid(x, y, z)
    return np.vstack((x.flatten(), y.flatten(), z.flatten())).T

def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a)

def lookat(pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - pos)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = f
    view[0, 3] = -np.dot(s, pos)
    view[1, 3] = -np.dot(u, pos)
    view[2, 3] =  np.dot(f, pos)
    return view

def perspective(fov_degrees: float, aspect_ratio_width_over_height: float, near: float, far: float):
    t = np.tan(fov_degrees * 0.5)
    ar = aspect_ratio_width_over_height

    return np.array([
        [1 / (ar * t),            0,                               0,  0],
        [           0,        1 / t,                               0,  0],
        [           0,            0,     (near + far) / (near - far), -1],
        [           0,            0, (2 * near * far) / (near - far),  0],
    ], dtype=np.float32).T