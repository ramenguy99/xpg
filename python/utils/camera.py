from dataclasses import dataclass
from pyglm.glm import vec3, perspectiveRH_ZO, lookAtRH

@dataclass
class Camera:
    position: vec3
    target: vec3
    world_up: vec3
    fov: float
    ar: float
    zmin: float
    zmax: float

    def projection(self):
        return perspectiveRH_ZO(self.fov, self.ar, self.zmin, self.zmax)

    def view(self):
        return lookAtRH(self.position, self.target, self.world_up)