import sys

import numpy as np
import PIL.Image
from pyxpg import *

from ambra.viewer import Viewer
from ambra.utils.gpu import readback_mips
from ambra.utils.profile import profile

data = np.asarray(PIL.Image.open(sys.argv[1]))

# data = data[:64, :64].copy()

height, width = data.shape[:2]
num_mips = max(width.bit_length(), height.bit_length())

v = Viewer()

# TODO:
# - Can't really explain the SPD_MAX_LEVELS + 1 yet, maybe the max does not include level 0?
# - Figure out if MUTABLE_FORMAT is fine for sRGB, only other option is to copy after downsampling.
# - Fix barriers for mips, enhance or move away from from_data
with profile("Create Image"):
    img = Image.from_data(
        v.ctx,
        data,
        # ImageLayout.GENERAL,
        ImageLayout.TRANSFER_SRC_OPTIMAL,
        width,
        height,
        Format.R8G8B8A8_UNORM,
        ImageUsageFlags.TRANSFER_DST | ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_SRC,
        AllocType.DEVICE,
        create_flags=ImageCreateFlags.MUTABLE_FORMAT,
        mip_levels=num_mips,
    )

with profile("Mips"):
    v.renderer.spd_pipeline.run(v.renderer, img, ImageLayout.TRANSFER_SRC_OPTIMAL)

with profile("Readback"):
    mips = readback_mips(v.ctx, img, ImageLayout.SHADER_READ_ONLY_OPTIMAL)
    for m in mips:
        h, w = m.shape[:2]
        PIL.Image.fromarray(m).save(f"mip_{w}x{h}.png")
