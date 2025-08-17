from dataclasses import dataclass

from pyglm.glm import (
    inverse,
    mat4,
    mat4_cast,
    normalize,
    quat,
    quatLookAtRH,
    vec3,
)


@dataclass
class RigidTransform3D:
    translation: vec3
    rotation: quat

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3) -> "RigidTransform3D":
        d = normalize(target - position)
        rot = inverse(quatLookAtRH(d, up))

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
