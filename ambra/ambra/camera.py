from dataclasses import dataclass
from pyglm.glm import vec2, vec3, mat4, perspectiveRH_ZO, orthoRH_ZO
from .transform3d import RigidTransform

# Users:
#   - Application code wants minimal effor to describe camera transformations
#       - Camera controls (FPV / Trackball / orbit)
#       - Camera definition helpers (lookat, ortho, perspective)
#   - Viewer just cares about view and projection transforms -> does not want to recompute or check if changed though
#      -> setter issues where we want to know if parameters change, but do not want to recompute after every change
#         -> V / P can be recomputed once per frame, seems reasonable

@dataclass
class CameraDepth:
    z_min: float
    z_max: float


@dataclass
class Camera:
    camera_from_world: RigidTransform
    depth: CameraDepth

    def view(self) -> mat4:
        return self.camera_from_world.as_mat4()
    
    def projection(self) -> mat4:
        return mat4(1.0)


@dataclass
class OrthographicCamera(Camera):
    center: vec2
    half_extents: vec2

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3, z_min: float, z_max: float, center: vec2, half_extents: vec2):
        return cls(
            camera_from_world = RigidTransform.look_at(position, target, up),
            depth = CameraDepth(z_min, z_max),
            center = center,
            half_extents = half_extents,
        )

    def projection(self):
        top_left = self.center - self.half_extents
        bottom_right = self.center + self.half_extents
        return orthoRH_ZO(top_left.x, bottom_right.x, bottom_right.y, top_left.y, self.depth.z_min, self.depth.z_max)


@dataclass
class PerspectiveCamera(Camera):
    fov: float
    """Vertical field of view in radians"""

    ar: float
    """Horizontal over vertical aspect ratio"""

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3, z_min: float, z_max: float, fov: float, ar: float):
        return cls(
            camera_from_world = RigidTransform.look_at(position, target, up),
            depth = CameraDepth(z_min, z_max),
            fov = fov,
            ar = ar,
        )
    
    def projection(self):
        return perspectiveRH_ZO(self.fov, self.ar, self.depth.z_min, self.depth.z_max)


class CameraControl:
    pass

@dataclass
class FpvControl(CameraControl):
    up: vec3
    linear_speed: float
    horizontal_angular_speed: float
    vertical_angular_speed: float


@dataclass
class OrbitControl(CameraControl):
    up: vec3
    linear_speed: float
    horizontal_angular_speed: float
    vertical_angular_speed: float
    distance: float


@dataclass
class TrackballControl(CameraControl):
    linear_speed: float
    angular_speed: float
    distance: float