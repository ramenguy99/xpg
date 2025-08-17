from dataclasses import dataclass

from pyglm.glm import mat3, rotate, vec2


@dataclass
class RigidTransform2D:
    translation: vec2
    rotation: float

    @classmethod
    def identity(cls) -> "RigidTransform2D":
        return cls(
            translation=vec2(0),
            rotation=0,
        )


@dataclass
class Transform2D:
    translation: vec2
    rotation: float
    scale: vec2

    @classmethod
    def identity(cls) -> "Transform2D":
        return cls(
            translation=vec2(0),
            rotation=0,
            scale=vec2(1),
        )

    def as_mat3(self) -> mat3:
        m = rotate(self.rotation)
        m[0] *= self.scale[0]
        m[1] *= self.scale[1]
        m[2, 0] = self.translation.x
        m[2, 1] = self.translation.y
        return m
