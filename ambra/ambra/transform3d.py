from dataclasses import dataclass
from pyglm.glm import vec3, quat, mat4, quatLookAtRH, normalize, mat4_cast, inverse

from pyglm.glm import lookAtRH

@dataclass
class RigidTransform:
    translation: vec3
    rotation: quat

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3):
        d = normalize(target - position)
        rot = inverse(quatLookAtRH(d, up))

        return cls(
            translation = rot * -position,
            rotation = rot,
        )
    
    def as_mat4(self) -> mat4:
        m = mat4_cast(self.rotation)
        m[3, 0] = self.translation.x
        m[3, 1] = self.translation.y
        m[3, 2] = self.translation.z
        return m

    @classmethod
    def identity(cls):
        return cls(
            translation=vec3(0),
            rotation=quat(1, 0, 0, 0),
        )

@dataclass
class Transform:
    translation: vec3
    rotation: quat
    scale: vec3

    @classmethod
    def identity(cls):
        return cls(
            translation=vec3(0),
            rotation=quat(1, 0, 0, 0),
            scale=vec3(1),
        )
        