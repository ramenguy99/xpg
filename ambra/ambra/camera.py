from dataclasses import dataclass
from pyglm.glm import vec2, vec3, mat4, perspectiveRH_ZO, orthoRH_ZO
from .transform import RigidTransform

# Users:
#   - Application code wants minimal effor to describe camera transformations
#       - Camera controls (FPV / Trackball / orbit)
#       - Camera definition helpers (lookat, ortho, perspective)
#   - Viewer just cares about view and projection transforms -> does not want to recompute or check if changed though
#      -> setter issues where we want to know if parameters change, but do not want to recompute after every change
#         -> V / P can be recomputed once per frame, seems reasonable

@dataclass
class Depth:
    zmin: float
    zmax: float

@dataclass
class Orthographic:
    center: vec2
    half_extents: vec2

@dataclass
class Camera:
    camera_from_world: RigidTransform
    depth: Depth

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3, zmin: float, zmax: float):
        return cls(
            camera_from_world = RigidTransform.look_at(position, target, up),
            depth = Depth(zmin, zmax),
        )
    
    def view(self) -> mat4:
        return self.camera_from_world.as_mat4()

@dataclass
class OrthographicCamera:
    camera: Camera
    orthographic: Orthographic

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3, zmin: float, zmax: float, center: vec2, half_extents: vec2):
        return cls(
            camera = Camera.look_at(position, target, up, zmin, zmax),
            orthographic = Orthographic(center, half_extents),
        )

    def view(self):
        return self.camera.view()

    def projection(self):
        top_left = self.orthographic.center - self.orthographic.half_extents
        bottom_right = self.orthographic.center + self.orthographic.half_extents
        return orthoRH_ZO(top_left.x, bottom_right.x, bottom_right.y, top_left.y, self.camera.depth.zmin, self.camera.depth.zmax)

@dataclass
class Perspective:
    fov: float
    """Vertical field of view in radians"""

    ar: float
    """Horizontal over vertical aspect ratio"""

@dataclass
class PerspectiveCamera:
    camera: Camera
    perspective: Perspective

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3, zmin: float, zmax: float, fov: float, ar: float):
        return cls(
            camera = Camera.look_at(position, target, up, zmin, zmax),
            perspective = Perspective(fov, ar),
        )
    
    def view(self):
        return self.camera.view()

    def projection(self):
        return perspectiveRH_ZO(self.perspective.fov, self.perspective.ar, self.camera.depth.zmin, self.camera.depth.zmax)

@dataclass
class FpvControl:
    up: vec3
    linear_speed: float
    horizontal_angular_speed: float
    vertical_angular_speed: float

@dataclass
class OrbitControl:
    up: vec3
    linear_speed: float
    horizontal_angular_speed: float
    vertical_angular_speed: float
    distance: float

@dataclass
class TrackballControl:
    linear_speed: float
    angular_speed: float
    distance: float