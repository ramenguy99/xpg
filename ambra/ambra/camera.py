# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Tuple

from pyglm.glm import mat3_cast, mat4, orthoRH_ZO, perspectiveRH_ZO, row, vec2, vec3

from .config import Handedness
from .transform3d import RigidTransform3D

# Users:
#   - Application code wants minimal effor to describe camera transformations
#       - Camera controls (FPV / Trackball / orbit)
#       - Camera definition helpers (lookat, ortho, perspective)
#   - Viewer just cares about view and projection transforms -> does not want to recompute or check if changed though
#      -> setter issues where we want to know if parameters change, but do not want to recompute after every change
#         -> V / P can be recomputed once per frame, seems reasonable


@dataclass
class CameraDepth:
    z_near: float
    z_far: float


@dataclass
class Camera:
    camera_from_world: RigidTransform3D
    depth: CameraDepth
    """Horizontal over vertical aspect ratio"""

    def view(self) -> mat4:
        return self.camera_from_world.as_mat4()

    def projection(self, ar: float) -> mat4:
        return mat4(1.0)

    def position(self) -> vec3:
        return self.camera_from_world.inverse().translation

    def right(self) -> vec3:
        r = mat3_cast(self.camera_from_world.rotation)
        return vec3(row(r, 0))

    def up(self) -> vec3:
        r = mat3_cast(self.camera_from_world.rotation)
        return vec3(row(r, 1))

    def front(self) -> vec3:
        r = mat3_cast(self.camera_from_world.rotation)
        return vec3(row(r, 2))

    def right_up_front(self) -> Tuple[vec3, vec3, vec3]:
        r = mat3_cast(self.camera_from_world.rotation)
        return vec3(row(r, 0)), vec3(row(r, 1)), vec3(row(r, 2))


@dataclass
class OrthographicCamera(Camera):
    center: vec2
    half_extents: vec2

    @classmethod
    def look_at(
        cls,
        position: vec3,
        target: vec3,
        up: vec3,
        z_near: float,
        z_far: float,
        center: vec2,
        half_extents: vec2,
        handedness: Handedness,
    ) -> "OrthographicCamera":
        return cls(
            camera_from_world=RigidTransform3D.look_at(position, target, up, handedness),
            depth=CameraDepth(z_near, z_far),
            center=center,
            half_extents=half_extents,
        )

    def projection(self, ar: float) -> mat4:
        half_extents = self.half_extents * vec2(ar, 1)
        top_left = self.center - half_extents
        bottom_right = self.center + half_extents
        return orthoRH_ZO(
            top_left.x,
            bottom_right.x,
            bottom_right.y,
            top_left.y,
            self.depth.z_near,
            self.depth.z_far,
        )


@dataclass
class PerspectiveCamera(Camera):
    fov: float
    """Vertical field of view in radians"""

    @classmethod
    def look_at(
        cls,
        position: vec3,
        target: vec3,
        up: vec3,
        z_near: float,
        z_far: float,
        fov: float,
        handedness: Handedness,
    ) -> "PerspectiveCamera":
        return cls(
            camera_from_world=RigidTransform3D.look_at(position, target, up, handedness),
            depth=CameraDepth(z_near, z_far),
            fov=fov,
        )

    def projection(self, ar: float) -> mat4:
        return perspectiveRH_ZO(self.fov, ar, self.depth.z_near, self.depth.z_far)
