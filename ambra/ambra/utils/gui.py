# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT
from pyxpg import (
    DescriptorPool,
    DescriptorPoolSize,
    DescriptorSetBinding,
    DescriptorSetLayout,
    DescriptorType,
    Device,
    Image,
    ImageLayout,
    Sampler,
    Stage,
    imgui,
)
from pyxpg.imgui import Vec2


class GuiImage:
    def __init__(
        self,
        device: Device,
        image: Image,
        sampler: Sampler,
        image_layout: ImageLayout = ImageLayout.SHADER_READ_ONLY_OPTIMAL,
    ) -> None:
        # Could be cached inside Gui object instead of re-created for each image.
        self.image_layout = DescriptorSetLayout(
            device, [DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE, stage_flags=Stage.FRAGMENT)]
        )
        self.sampler_layout = DescriptorSetLayout(
            device, [DescriptorSetBinding(1, DescriptorType.SAMPLER, stage_flags=Stage.FRAGMENT)]
        )

        self.pool = DescriptorPool(
            device,
            [
                DescriptorPoolSize(1, DescriptorType.SAMPLED_IMAGE),
                DescriptorPoolSize(1, DescriptorType.SAMPLER),
            ],
            2,
        )

        self.image_set = self.pool.allocate_descriptor_set(self.image_layout)
        self.sampler_set = self.pool.allocate_descriptor_set(self.sampler_layout)

        self.image_set.write_image(image, image_layout, DescriptorType.SAMPLED_IMAGE, 0)
        self.sampler_set.write_sampler(sampler, 0)

        self.texture = imgui.Texture(self.image_set)

    def draw(
        self,
        draw_list: imgui.DrawList,
        p_min: Vec2,
        p_max: Vec2,
        uv_min: Vec2 = (0, 0),
        uv_max: Vec2 = (1, 1),
        col: int = 4294967295,
    ) -> None:
        draw_list.set_sampler(self.sampler_set)
        draw_list.add_image(self.texture, p_min, p_max, uv_min, uv_max, col)
        draw_list.reset_sampler()

    def draw_square(
        self, draw_list: imgui.DrawList, uv_min: Vec2 = (0, 0), uv_max: Vec2 = (1, 1), col: int = 4294967295
    ) -> None:
        p1 = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        p2 = imgui.Vec2(p1.x + avail.y, p1.y + avail.y)
        self.draw(draw_list, p1, p2, uv_min, uv_max, col)
