from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class LogLevel(Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    DISABLED = 5


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
    force_buffer_upload_method: Optional[UploadMethod] = None
    force_image_upload_method: Optional[UploadMethod] = None


@dataclass
class PlaybackConfig:
    enabled: bool = True
    playing: bool = False
    frames_per_second: float = 30.0
    initial_time: float = 0.0
    initial_frame: Optional[int] = None
    max_time: Optional[float] = None


class CameraProjection(Enum):
    PERSPECTIVE = 0
    ORTHOGRAPHIC = 1


class CameraControlMode(Enum):
    NONE = 0
    ORBIT = 1
    TRACKBALL = 2
    FIRST_PERSON = 3
    # PAN_AND_ZOOM_ORTHO = 4


class Handedness(Enum):
    LEFT_HANDED = 0
    RIGHT_HANDED = 1


@dataclass
class GuiConfig:
    stats: bool = False
    playback: bool = False
    inspector: bool = False
    renderer: bool = False


@dataclass
class CameraConfig:
    # Inital state
    projection: CameraProjection = CameraProjection.PERSPECTIVE
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    z_min: float = 0.001
    z_max: float = 1000.0

    # If type is CameraProjection.PERSPECTIVE
    perspective_vertical_fov: float = 45.0

    # If type is CameraProjection.ORTHOGRAPHIC
    ortho_center: Tuple[float, float] = (0.0, 0.0)
    ortho_half_extents: Tuple[float, float] = (1.0, 1.0)

    # Viewport camera controls
    control_mode: CameraControlMode = CameraControlMode.ORBIT
    rotation_speed: Tuple[float, float] = (0.005, 0.005)
    pan_speed: Tuple[float, float] = (0.01, 0.01)
    pan_distance_speed_scale: float = 0.1
    pan_min_speed_scale: float = 0.1
    zoom_speed: float = 0.1
    zoom_distance_speed_scale: float = 1.0
    zoom_min_speed_scale: float = 2.0
    zoom_min_target_distance: float = 0.01


@dataclass
class Config:
    # Logging
    log_level: LogLevel = LogLevel.DISABLED

    # Window
    window_x: Optional[int] = None
    window_y: Optional[int] = None
    window_width: int = 1280
    window_height: int = 720
    wait_events: bool = False
    vsync: bool = True
    preferred_frames_in_flight: int = 2

    # Vulkan
    force_physical_device_index: Optional[int] = None
    prefer_discrete_gpu: bool = True
    enable_validation_layer: bool = True
    enable_synchronization_validation: bool = True
    enable_gpu_based_validation: bool = False

    # Scene
    world_up: Tuple[float, float, float] = (0, 1, 0)
    handedness: Handedness = Handedness.RIGHT_HANDED
    camera: CameraConfig = field(default_factory=CameraConfig)

    # Stats
    stats_frame_time_count: int = 32

    # Sub-configs
    renderer: RendererConfig = field(default_factory=RendererConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
