import av
import numpy as np
from numpy.typing import NDArray

from ambra.config import Config, GuiConfig
from ambra.primitives3d import Lines
from ambra.viewer import Viewer

width, height = 1920, 1080
viewer = Viewer(
    config=Config(
        window=False,
        window_width=width,
        window_height=height,
        gui=GuiConfig(stats=True),
    ),
)

positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    np.float32,
)

colors = np.array(
    [
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
        0xFFFF0000,
        0xFFFF0000,
    ],
    np.uint32,
)

line = Lines(positions, colors, 1.0, translation=np.linspace((0, 0, 0), (1, 0, 0), 50))
viewer.scene.objects.extend([line])


container = av.open("test.mp4", mode="w")

stream = container.add_stream("h264", rate=viewer.playback.frames_per_second, options={})
stream.width = width
stream.height = height
stream.pix_fmt = "yuv420p"


def on_frame(img: NDArray[np.uint8]):
    frame = av.VideoFrame.from_ndarray(img[:, :, :3], format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


viewer.render_video(on_frame)

# Flush stream
for packet in stream.encode():
    container.mux(packet)

# Close the file
container.close()
