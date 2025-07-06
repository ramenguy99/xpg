from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Config:
    # Window
    wait_events: bool = True
    background_color: Tuple[float, float, float, float] = (1, 1, 1, 1)

    # Server
    server_enabled: bool = False
    server_address: str = "localhost"
    server_port: int = 9168
    server_max_connections: int = 4