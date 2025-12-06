# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from pyxpg import LogLevel


@dataclass
class ServerConfig:
    enabled: bool = False
    address: str = "localhost"
    port: int = 9168


class UploadMethod(Enum):
    GRAPHICS_QUEUE = 0
    TRANSFER_QUEUE = 1
    MAPPED_PREFER_HOST = 2
    MAPPED_PREFER_DEVICE = 3


@dataclass
class RendererConfig:
    background_color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    msaa_samples: int = 1
    uniform_pool_block_size: int = 4 * 1024 * 1024
    upload_buffer_size: int = 32 * 1024 * 1024
    upload_buffer_count: int = 2
    thread_pool_workers: Optional[int] = None
    use_transfer_queue_if_available: bool = True
    force_buffer_upload_method: Optional[UploadMethod] = None
    force_image_upload_method: Optional[UploadMethod] = None
    max_lights_per_type: int = 32
    max_shadow_maps: int = 8
    mip_generation_batch_size: int = 32


@dataclass
class PlaybackConfig:
    playing: bool = False
    frames_per_second: float = 30.0
    playback_speed_multiplier: float = 1.0
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
    RIGHT_HANDED = 0
    LEFT_HANDED = 1


@dataclass
class GuiConfig:
    multiviewport: bool = False
    initial_number_of_viewports: int = 1
    max_viewport_count: int = 8

    ini_filename: Optional[str] = "imgui.ini"  # if set to None disables.

    stats: bool = False
    playback: bool = False
    inspector: bool = False
    renderer: bool = False


@dataclass
class CameraConfig:
    # Inital state
    projection: CameraProjection = CameraProjection.PERSPECTIVE
    position: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    z_near: float = 0.01
    z_far: float = 1000.0

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
    # Logging.
    #
    # If not None, the Viewer constructor sets the pyxpg log level to this value.
    # If None preserve the existing pyxpg log level.
    # The pyxpg log level is global for the module and defaults to DISABLED.
    #
    # Log messages from pyxpg are always printed to stdout.
    log_level: Optional[LogLevel] = None

    # Window
    window: bool = True
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
    world_up: Tuple[float, float, float] = (0, 0, 1)
    handedness: Handedness = Handedness.RIGHT_HANDED

    # Stats
    stats_frame_time_count: int = 32

    # Sub-configs
    camera: CameraConfig = field(default_factory=CameraConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
