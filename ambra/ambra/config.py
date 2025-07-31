from dataclasses import dataclass, field
from typing import Tuple, Optional
from enum import Enum

@dataclass
class ServerConfig:
    enabled: bool = False
    address: str = "localhost"
    port: int = 9168
    max_connections: int = 4

class UploadMethod(Enum):
    GFX = 0
    TRANSFER_QUEUE = 1
    CPU_BUF = 2
    BAR = 3

@dataclass
class RendererConfig:
    background_color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    uniform_pool_block_size: int = 32 * 1024 * 1024
    thread_pool_workers: Optional[int] = None
    use_transfer_queue_if_available: bool = True
    force_upload_method: Optional[UploadMethod] = None

@dataclass
class PlaybackConfig:
    enabled: bool = True
    playing: bool = False
    frames_per_second: float = 30.0
    initial_time: float = 0.0
    initial_frame: Optional[int] = None
    max_time: Optional[float] = None

class CameraType(Enum):
    PERSPECTIVE = 0
    ORTHOGRAPHIC = 1

@dataclass
class Config:
    # Window
    wait_events: bool = False

    # Scene
    camera_type: CameraType = CameraType.PERSPECTIVE
    z_min: float = 0.001
    z_max: float = 1000.0
    perspective_vertical_fov: float = 45.0
    ortho_center: Tuple[float, float] = (0.0, 0.0)
    ortho_half_extents: Tuple[float, float] = (1, 1)

    renderer: RendererConfig = field(default_factory=RendererConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    server: ServerConfig = field(default_factory=ServerConfig)