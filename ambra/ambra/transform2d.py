from dataclasses import dataclass
from pyglm.glm import vec2

@dataclass
class RigidTransform:
    translation: vec2
    rotation: float

    @classmethod
    def identity(cls):
        return cls(
            translation=vec2(0),
            rotation=0,
        )

@dataclass
class Transform:
    translation: vec2
    rotation: float
    scale: vec2

    @classmethod
    def identity(cls):
        return cls(
            translation=vec2(0),
            rotation=0,
            scale=vec2(1),
        )
