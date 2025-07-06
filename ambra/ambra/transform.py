from dataclasses import dataclass
from pyglm.glm import vec3, quat, mat4, quatLookAtRH, normalize, mat4_cast

@dataclass
class RigidTransform:
    translation: vec3
    orientation: quat

    @classmethod
    def look_at(cls, position: vec3, target: vec3, up: vec3):
        d = normalize(target - position)
        return cls(
            translation = -position,
            orientation = quatLookAtRH(d, up),
        )
    
    def as_mat4(self) -> mat4:
        m = mat4_cast(self.orientation)
        m[:3, 3] = self.translation
        return m

@dataclass
class Transform:
    translation: vec3
    rotation: quat
    scale: vec3

    @classmethod
    def identity(cls):
        Transform(
            translation=vec3(1),
            rotation=quat(1, 0, 0, 0),
            scale=vec3(1),
        )
        