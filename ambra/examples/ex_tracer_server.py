import argparse
import sys

import numpy as np

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig, ServerConfig
from ambra.primitives3d import (
    ColormapDistanceToPlane,
    ColormapKind,
    Grid,
    GridType,
    Points,
)
from ambra.tracer_property import TracerBufferProperty, TracerLiveSource, TracerOfflineSource
from ambra.viewer import Viewer

CLOCK = "monotonic_ts"
WIDTH = 64
HEIGHT = 48


parser = argparse.ArgumentParser()
parser.add_argument("--db", required=False, default="", help="Path to db file")
args = parser.parse_args()

if args.db:
    src = TracerOfflineSource(CLOCK, args.db)
    src.db.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{CLOCK}_/points" ON "/points" ({CLOCK})')
    src.db.commit()
    first_timestamp = src.db.execute(f'SELECT {CLOCK} from "/points" ORDER BY {CLOCK} ASC').fetchone()[0]
else:
    src = TracerLiveSource(CLOCK, collect_history=True, collect_to_sqlite_db=True, persist_sqlite_db=False)
    first_timestamp = None

grid = Grid.transparent_black_lines((100, 100), GridType.XY_PLANE, grid_scale=0.1, name="Grid")


class PointsProperty(TracerBufferProperty):
    def __init__(self, name=""):
        super().__init__(np.float32, (-1, 3), WIDTH * HEIGHT * 12, name)

    def process(self, points, **kwargs):
        return points


points_property = src.register("/points", PointsProperty())
points = Points(points_property, colormap=ColormapDistanceToPlane(ColormapKind.JET, 0, 1), point_size=3)

# Create viewer
v = Viewer(
    config=Config(
        window_width=1920,
        window_height=1080,
        gui=GuiConfig(stats=True, renderer=True, playback=True, inspector=True),
        renderer=RendererConfig(
            msaa_samples=4,
        ),
        playback=PlaybackConfig(
            playback_speed_multiplier=1.0,
            playing=True,
            lock_to_last_frame=isinstance(src, TracerLiveSource),
        ),
        server=ServerConfig(
            enabled=True,
            address="0.0.0.0",
            port=9168,
        ),
        camera=CameraConfig(
            position=(6, 6, 6),
        ),
    ),
)

# Scene
v.scene.objects.extend(
    [
        grid,
        points,
    ]
)

with src.attach(v, first_timestamp):
    v.run()
