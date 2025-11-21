import numpy as np
from pyglm.glm import ivec2, vec2
from pyxpg import *

from ambra.config import Config, GuiConfig
from ambra.utils.descriptors import create_descriptor_layout_pool_and_set
from ambra.utils.hook import hook
from ambra.viewer import Viewer


class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        global texture, img_data
        if imgui.begin("Image view")[0]:
            io = imgui.get_io()
            mouse_pos = ivec2(io.mouse_pos.x, io.mouse_pos.y)
            draw_list = imgui.get_window_draw_list()

            cursor_pos = imgui.get_cursor_screen_pos()
            pos = ivec2(cursor_pos.x, cursor_pos.y)

            image_size = ivec2(img_data.shape[1], img_data.shape[0])
            image_top_left = ivec2(-2, -2)
            image_bottom_right = ivec2(4, 4)
            image_visible_size = image_bottom_right - image_top_left

            ar = image_visible_size.x / image_visible_size.y

            avail = imgui.get_content_region_avail()
            available = ivec2(avail.x, avail.y)

            # height = available.x / ar
            height = available.y
            view_size = ivec2(ar * height, height)

            imgui.image(
                texture,
                imgui.Vec2(*view_size),
                imgui.Vec2(*(vec2(image_top_left) / vec2(image_size))),
                imgui.Vec2(*(vec2(image_bottom_right) / vec2(image_size))),
            )
            draw_list.add_rect(cursor_pos, imgui.Vec2(*(pos + view_size)), 0xFFFFFFFF, thickness=2)

            pixel_size = vec2(view_size) / vec2(image_visible_size)
            mouse_relative_pos = mouse_pos - pos
            if (
                mouse_relative_pos.x >= 0
                and mouse_relative_pos.y >= 0
                and mouse_relative_pos.x < view_size.x
                and mouse_relative_pos.x < view_size.x
            ):
                pixel_coordinates = ivec2(vec2(mouse_relative_pos) / pixel_size)
                pixel_pos = vec2(pos) + vec2(pixel_coordinates) * pixel_size
                draw_list.add_rect(
                    imgui.Vec2(*pixel_pos), imgui.Vec2(*(pixel_pos + pixel_size)), 0xFF00FFFF, thickness=2
                )

                pixel_image_coordinates = pixel_coordinates + image_top_left
                if (
                    pixel_image_coordinates.x >= 0
                    and pixel_image_coordinates.y >= 0
                    and pixel_image_coordinates.x < image_size.x
                    and pixel_image_coordinates.y < image_size.y
                ):
                    imgui.begin_tooltip()
                    imgui.text(
                        f"({pixel_image_coordinates.x}, {pixel_image_coordinates.y}): {img_data[pixel_image_coordinates.y, pixel_image_coordinates.x]}"
                    )
                    imgui.end_tooltip()
        imgui.end()


viewer = CustomViewer(
    config=Config(
        gui=GuiConfig(
            stats=True,
        ),
    ),
)


H, W = 8, 16
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
sampler = Sampler(
    viewer.ctx,
    u=SamplerAddressMode.CLAMP_TO_BORDER,
    v=SamplerAddressMode.CLAMP_TO_BORDER,
    border_color=BorderColor.FLOAT_OPAQUE_BLACK,
)

layout, pool, set = create_descriptor_layout_pool_and_set(
    viewer.ctx,
    [
        DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER, stage_flags=Stage.FRAGMENT),
    ],
)
set.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, sampler, 0)

texture = imgui.Texture(set)

viewer.run()
