from dataclasses import dataclass
from typing import Tuple
from enum import Enum

@dataclass
class ServerConfig:
    enabled: bool = False
    address: str = "localhost"
    port: int = 9168
    max_connections: int = 4

@dataclass
class RendererConfig:
    background_color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    prefer_preupload: bool = True

class CameraType(Enum):
    PERSPECTIVE = 0
    ORTHOGRAPHIC = 1

@dataclass
class Config:
    # Window
    wait_events: bool = True

    # Scene
    camera_type: CameraType = CameraType.PERSPECTIVE
    z_min: float = 0.001
    z_max: float = 1000.0
    perspective_vertical_fov: float = 45.0
    ortho_center: Tuple[float, float] = (0.0, 0.0)
    ortho_half_extents: Tuple[float, float] = (50, 50)

    renderer = RendererConfig()
    server = ServerConfig()