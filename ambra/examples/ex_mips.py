import sys

import numpy as np
import PIL.Image
from pyxpg import *

from ambra.utils.gpu import readback_mips
from ambra.viewer import Viewer

data = np.asarray(PIL.Image.open(sys.argv[1]))
if data.shape[2] == 3:
    data = np.dstack((data, np.full(data.shape[:2], 255, data.dtype)))

height, width = data.shape[:2]
num_mips = max(width.bit_length(), height.bit_length())

v = Viewer()

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

v.renderer.spd_pipeline.run(v.renderer, img, ImageLayout.TRANSFER_SRC_OPTIMAL)

mips = readback_mips(v.ctx, img, ImageLayout.SHADER_READ_ONLY_OPTIMAL)
for m in mips:
    h, w = m.shape[:2]
    PIL.Image.fromarray(m).save(f"mip_{w}x{h}.png")
