# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from pyglm.glm import (
    decompose,
    inverse,
    mat4,
    mat4_cast,
    normalize,
    quat,
    quat_cast,
    quatLookAtLH,
    quatLookAtRH,
    vec3,
    vec4,
)

from .config import Handedness


@dataclass
class RigidTransform3D:
    translation: vec3
    rotation: quat

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3, handedness: Handedness) -> "RigidTransform3D":
        d = normalize(target - position)
        if handedness == Handedness.RIGHT_HANDED:
            inv_rot = quatLookAtRH(d, up)
        else:
            inv_rot = quatLookAtLH(d, up)
        rot = inverse(inv_rot)

        return cls(
            translation=rot * -position,  # type: ignore
            rotation=rot,  # type: ignore
        )

    @classmethod
    def from_mat4(cls, m: mat4) -> "RigidTransform3D":
        return cls(
            rotation=quat_cast(m),
            translation=vec3(m[3, 0], m[3, 1], m[3, 2]),
        )

    def as_mat4(self) -> mat4:
        m = mat4_cast(self.rotation)
        m[3, 0] = self.translation.x
        m[3, 1] = self.translation.y
        m[3, 2] = self.translation.z
        return m

    @classmethod
    def identity(cls) -> "RigidTransform3D":
        return cls(
            translation=vec3(0),
            rotation=quat(1, 0, 0, 0),
        )

    def inverse(self) -> "RigidTransform3D":
        inverse_rotation = inverse(self.rotation)
        return RigidTransform3D(translation=-(inverse_rotation * self.translation), rotation=inverse_rotation)  # type:ignore

    def __matmul__(self, other: "RigidTransform3D") -> "RigidTransform3D":
        return RigidTransform3D(
            rotation=self.rotation * other.rotation,
            translation=self.rotation * other.translation + self.translation,
        )


@dataclass
class Transform3D:
    translation: vec3
    rotation: quat
    scale: vec3

    @classmethod
    def identity(cls) -> "Transform3D":
        return cls(
            translation=vec3(0),
            rotation=quat(1, 0, 0, 0),
            scale=vec3(1),
        )

    @classmethod
    def from_mat4(cls, m: mat4) -> "Transform3D":
        t = Transform3D(vec3(), quat(), vec3())
        skew = vec3()
        perspective = vec4()
        decompose(mat4(m), t.scale, t.rotation, t.translation, skew, perspective)
        return t

    def as_mat4(self) -> mat4:
        m = mat4_cast(self.rotation)
        m[0] *= self.scale[0]
        m[1] *= self.scale[1]
        m[2] *= self.scale[2]
        m[3, 0] = self.translation.x
        m[3, 1] = self.translation.y
        m[3, 2] = self.translation.z
        return m
