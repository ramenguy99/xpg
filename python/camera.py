from gfxmath import Mat4, Vec3, perspective, lookat
from dataclasses import dataclass

@dataclass
class Camera:
    position: Vec3
    target: Vec3
    world_up: Vec3
    fov: float
    ar: float
    zmin: float
    zmax: float

    def projection(self):
        return perspective(self.fov, self.ar, self.zmin, self.zmax)

    def view(self):
        return lookat(self.position, self.target, self.world_up)