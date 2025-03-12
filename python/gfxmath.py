import numpy as np
from dataclasses import dataclass

@dataclass
class Vec2:
    x: float
    y: float
    
    @classmethod
    def fromarray(cls, a: np.ndarray) -> 'Vec2':
        if a.shape != (2,):
            raise ValueError(f"Expected array with shape (2,) got {a.shape}")
        return cls(a[0], a[1])
    
    def __array__(self, dtype=np.float32, copy=True):
        if not copy:
            raise ValueError()

        return np.array((self.x, self.y), dtype)

@dataclass
class Vec3:
    x: float
    y: float
    z: float
    
    @classmethod
    def fromarray(cls, a: np.ndarray) -> 'Vec3':
        if a.shape != (3,):
            raise ValueError(f"Expected array with shape (3,) got {a.shape}")
        return cls(a[0], a[1], a[2])

    def __array__(self, dtype=np.float32, copy=True):
        if not copy:
            raise ValueError()

        return np.array((self.x, self.y, self.z), dtype)

@dataclass
class Vec4:
    x: float
    y: float
    z: float
    w: float
    
    @classmethod
    def fromarray(cls, a: np.ndarray) -> 'Vec4':
        if a.shape != (4,):
            raise ValueError(f"Expected array with shape (4,) got {a.shape}")
        return cls(a[0], a[1], a[2], a[3])

    def __array__(self, dtype=np.float32, copy=True):
        if not copy:
            raise ValueError()

        return np.array((self.x, self.y, self.z, self.w), dtype)

@dataclass
class Mat4:
    r0: Vec4
    r1: Vec4
    r2: Vec4
    r3: Vec4

    @classmethod
    def fromarray(cls, a: np.ndarray) -> 'Vec4':
        if a.shape != (4,4):
            raise ValueError(f"Expected array with shape (4,) got {a.shape}")
        return cls(
            Vec4.fromarray(a[0]),
            Vec4.fromarray(a[1]),
            Vec4.fromarray(a[2]),
            Vec4.fromarray(a[3])
        )

    def __array__(self, dtype=np.float32, copy=True):
        if not copy:
            raise ValueError()

        return np.array([self.r0, self.r1, self.r2, self.r3], dtype)

def normalize(a: np.ndarray) -> np.ndarray:
    a = np.array(a)
    return a / np.linalg.norm(a)

def lookat_rh(pos: Vec3, target: Vec3, up: Vec3) -> Mat4:
    pos = np.array(pos)
    target = np.array(target)
    up = np.array(up)

    f = normalize(target - pos)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[0, 3] = -np.dot(s, pos)
    view[1, 3] = -np.dot(u, pos)
    view[2, 3] =  np.dot(f, pos)
    return Mat4.fromarray(view)

def lookat_lh(pos: Vec3, target: Vec3, up: Vec3) -> Mat4:
    pos = np.array(pos)
    target = np.array(target)
    up = np.array(up)

    f = normalize(target - pos)
    s = normalize(np.cross(up, f))
    u = np.cross(f, s)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = f
    view[0, 3] = -np.dot(s, pos)
    view[1, 3] = -np.dot(u, pos)
    view[2, 3] = -np.dot(f, pos)
    return Mat4.fromarray(view)

# NOTE: y is flipped compared to GLM because Vulkan has opposite y coordinate in screen space.
def perspective_rh_no(fov_degrees: float, aspect_ratio_width_over_height: float, near: float, far: float) -> Mat4:
    t = np.tan(np.deg2rad(fov_degrees) * 0.5)
    ar = aspect_ratio_width_over_height

    return Mat4.fromarray(np.array([
        [1 / (ar * t),            0,                               0,  0],
        [           0,       -1 / t,                               0,  0],
        [           0,            0,     (near + far) / (near - far), -1],
        [           0,            0, (2 * near * far) / (near - far),  0],
    ], dtype=np.float32).T)

def perspective_rh_zo(fov_degrees: float, aspect_ratio_width_over_height: float, near: float, far: float):
    t = np.tan(np.deg2rad(fov_degrees) * 0.5)
    ar = aspect_ratio_width_over_height

    return Mat4.fromarray(np.array([
        [1 / (ar * t),            0,                               0,  0],
        [           0,       -1 / t,                               0,  0],
        [           0,            0,              far / (near - far), -1],
        [           0,            0,     (near * far) / (near - far),  0],
    ], dtype=np.float32).T)

def perspective_lh_zo(fov_degrees: float, aspect_ratio_width_over_height: float, near: float, far: float):
    t = np.tan(np.deg2rad(fov_degrees) * 0.5)
    ar = aspect_ratio_width_over_height

    return Mat4.fromarray(np.array([
        [1 / (ar * t),            0,                               0,  0],
        [           0,       -1 / t,                               0,  0],
        [           0,            0,              far / (far - near),  1],
        [           0,            0,     (near * far) / (near - far),  0],
    ], dtype=np.float32).T)

lookat = lookat_rh
perspective = perspective_rh_zo

def grid3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.meshgrid(x, y, z)
    return np.vstack((x.flatten(), y.flatten(), z.flatten())).T
