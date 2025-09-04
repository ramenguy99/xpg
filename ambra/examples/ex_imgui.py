import numpy as np
from pyxpg import *

from ambra.config import Config, GuiConfig
from ambra.utils.hook import hook
from ambra.viewer import Viewer


class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        global texture
        if imgui.begin("Hello")[0]:
            imgui.image(texture, (100, 100))
        imgui.end()


viewer = CustomViewer(
    "primitives",
    config=Config(
        gui=GuiConfig(
            stats=True,
        ),
    ),
)


H, W = 32, 64
img_data = np.zeros((H, W, 4))
for i in range(H):
    for j in range(W):
        img_data[i, j, 0] = (i + 0.5) / H
        img_data[i, j, 1] = (j + 0.5) / W
        img_data[i, j, 3] = 0
img_data = (img_data * 255.0).astype(np.uint8)

img = Image.from_data(
    viewer.ctx,
    img_data,
    ImageLayout.SHADER_READ_ONLY_OPTIMAL,
    W,
    H,
    Format.R8G8B8A8_UNORM,
    ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST,
    AllocType.DEVICE,
)
sampler = Sampler(viewer.ctx)
set = DescriptorSet(
    viewer.ctx,
    [
        DescriptorSetEntry(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
    ],
)
set.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, sampler, 0)

texture = imgui.Texture(set)

viewer.run()
