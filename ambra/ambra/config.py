from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ServerConfig:
    enabled: bool = False
    address: str = "localhost"
    port: int = 9168
    max_connections: int = 4

@dataclass
class RendererConfig:
    background_color: Tuple[float, float, float, float] = (1, 1, 1, 1)

@dataclass
class Config:
    # Window
    wait_events: bool = True

    renderer = RendererConfig()
    server = ServerConfig()