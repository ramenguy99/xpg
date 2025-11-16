import sys

import numpy as np
import PIL.Image
from pyxpg import *

from ambra.config import Config
from ambra.utils.gpu import MipGenerationFilter, readback_mips
from ambra.viewer import Viewer

if False:
    data = np.array(
        [
            [
                [127, 127, 127, 255],
                [127, 127, 127, 255],
                [127, 127, 127, 255],
                [127, 127, 127, 255],
            ],
            [
                [127, 127, 127, 255],
                [127, 127, 127, 255],
                [127, 127, 127, 255],
                [127, 127, 127, 255],
            ],
        ],
        np.uint8,
    )
elif False:
    data = np.array(
        [
            [
                [255, 255, 255, 255],
                [0, 0, 0, 255],
            ],
            [
                [0, 0, 0, 255],
                [255, 255, 255, 255],
            ],
        ],
        np.uint8,
    )
else:
    data = np.array(
        [
            [
                [255, 255, 255, 255],
                [255, 255, 255, 255],
                [0, 0, 0, 255],
                [0, 0, 0, 255],
            ],
            [
                [255, 255, 255, 255],
                [255, 255, 255, 255],
                [0, 0, 0, 255],
                [0, 0, 0, 255],
            ],
            [
                [0, 0, 0, 255],
                [0, 0, 0, 255],
                [255, 255, 255, 255],
                [255, 255, 255, 255],
            ],
            [
                [0, 0, 0, 255],
                [0, 0, 0, 255],
                [255, 255, 255, 255],
                [255, 255, 255, 255],
            ],
        ],
        np.uint8,
    )

height, width = data.shape[:2]
num_mips = max(width.bit_length(), height.bit_length())

v = Viewer(config=Config(window=False))

img = Image.from_data(
    v.ctx,
    data,
    ImageLayout.GENERAL,
    width,
    height,
    Format.R8G8B8A8_UNORM,
    ImageUsageFlags.TRANSFER_DST | ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_SRC,
    AllocType.DEVICE,
    create_flags=ImageCreateFlags.MUTABLE_FORMAT,
    mip_levels=num_mips,
)

v.renderer.spd_pipeline.run_sync(v.renderer, img, ImageLayout.TRANSFER_SRC_OPTIMAL, MipGenerationFilter.AVERAGE)

mips = readback_mips(v.ctx, img, ImageLayout.SHADER_READ_ONLY_OPTIMAL)
for m in mips:
    print(m.shape)
    print(m)
