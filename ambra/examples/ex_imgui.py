import numpy as np
from pyglm.glm import vec2, ivec2
from pyxpg import *

from ambra.config import Config, GuiConfig
from ambra.utils.hook import hook
from ambra.viewer import Viewer


def image_range(image: imgui.Texture, size: ivec2, top_left: ivec2, bottom_left: ivec2, image_size: ivec2):
    imgui.image(image, imgui.Vec2(*size), imgui.Vec2(*(vec2(top_left) / vec2(image_size))), imgui.Vec2(*(vec2(bottom_left) / vec2(image_size))))

class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        global texture
        if imgui.begin("Hello")[0]:
            image_range(texture, (100, 100), (-2, -2), (10,10), (8, 8))
        imgui.end()


viewer = CustomViewer(
    "primitives",
    config=Config(
        gui=GuiConfig(
            stats=True,
        ),
    ),
)


H, W = 8, 8
img_data = np.zeros((H, W, 4))
for i in range(H):
    for j in range(W):
        img_data[i, j, 0] = (i + 0.5) / H
        img_data[i, j, 1] = (j + 0.5) / W
        img_data[i, j, 3] = 1
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
sampler = Sampler(viewer.ctx, u = SamplerAddressMode.CLAMP_TO_BORDER, v = SamplerAddressMode.CLAMP_TO_BORDER, border_color=BorderColor.FLOAT_OPAQUE_BLACK)
set = DescriptorSet(
    viewer.ctx,
    [
        DescriptorSetEntry(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
    ],
)
set.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, sampler, 0)

texture = imgui.Texture(set)

viewer.run()
