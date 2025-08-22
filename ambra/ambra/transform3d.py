from dataclasses import dataclass

from pyglm.glm import (
    inverse,
    mat4,
    mat4_cast,
    normalize,
    quat,
    quatLookAtLH,
    quatLookAtRH,
    vec3,
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

    def as_mat4(self) -> mat4:
        m = mat4_cast(self.rotation)
        m[0] *= self.scale[0]
        m[1] *= self.scale[1]
        m[2] *= self.scale[2]
        m[3, 0] = self.translation.x
        m[3, 1] = self.translation.y
        m[3, 2] = self.translation.z
        return m

    def inverse(self) -> "Transform3D":
        inverse_rotation = inverse(self.rotation)
        return Transform3D(
            scale=1.0 / self.scale,
            rotation=inverse_rotation,  # type:ignore
            translation=-(inverse_rotation * self.translation),  # type:ignore
        )
