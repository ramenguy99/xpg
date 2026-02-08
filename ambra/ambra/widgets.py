from typing import Optional

from pyglm.glm import clamp, floor, ivec2, vec2
from pyxpg import (
    BorderColor,
    DescriptorSetBinding,
    DescriptorType,
    ImageLayout,
    ImageUsageFlags,
    PipelineStageFlags,
    Sampler,
    SamplerAddressMode,
    Stage,
    imgui,
)

from ambra.renderer_frame import RendererFrame
from ambra.utils.ring_buffer import RingBuffer

from .property import ImageProperty
from .renderer import Renderer
from .scene import Widget
from .utils.descriptors import create_descriptor_layout_pool_and_sets


class ImageInspector(Widget):
    def __init__(self, title: str, image: ImageProperty):
        super().__init__(title)

        self.image = self.add_image_property(image, name="image").use_gpu(
            ImageUsageFlags.SAMPLED,
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            PipelineStageFlags.FRAGMENT_SHADER,
            mips=True,
        )

        self.zoom = 0.0
        self.zoom_speed = 0.25
        self.offset = vec2(0, 0)
        self.dragging = False
        self.drag_start_pos = ivec2(0, 0)
        self.drag_start_offset = vec2(0, 0)

        # Public API
        self.pixel_under_cursor: Optional[ivec2] = None

    def create(self, r: Renderer) -> None:
        self.sampler = Sampler(
            r.ctx,
            u=SamplerAddressMode.CLAMP_TO_BORDER,
            v=SamplerAddressMode.CLAMP_TO_BORDER,
            border_color=BorderColor.FLOAT_OPAQUE_BLACK,
        )

        self.descriptor_layout, self.descriptor_pool, self.descriptor_sets = create_descriptor_layout_pool_and_sets(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER, stage_flags=Stage.FRAGMENT),
            ],
            r.num_frames_in_flight,
        )
        self.texture_and_sets = RingBuffer([(s, imgui.Texture(s)) for s in self.descriptor_sets])

    def upload(self, r: Renderer, frame: RendererFrame) -> None:
        s, self.texture = self.texture_and_sets.get_current_and_advance()
        s.write_combined_image_sampler(
            self.image.get_current_gpu().image, ImageLayout.SHADER_READ_ONLY_OPTIMAL, self.sampler, 0
        )

    def _draw(self) -> None:
        self.pixel_under_cursor = None

        image_size = ivec2(self.image.width, self.image.height)
        image_visible_size = vec2(image_size) * float(2.0**self.zoom)

        ar = image_visible_size.x / image_visible_size.y
        avail = imgui.get_content_region_avail()
        if avail.x <= 0 or avail.y <= 0:
            return

        available = ivec2(avail.x, avail.y)
        height = min(available.x / ar, available.y)
        view_size = ivec2(ar * height, height)

        io = imgui.get_io()

        mouse_pos = ivec2(io.mouse_pos.x, io.mouse_pos.y)
        pixel_size = vec2(view_size) / image_visible_size
        if self.dragging:
            self.offset = self.drag_start_offset - vec2(mouse_pos - self.drag_start_pos) / pixel_size

        draw_list = imgui.get_window_draw_list()

        cursor_pos = imgui.get_cursor_screen_pos()
        pos = ivec2(cursor_pos.x, cursor_pos.y)
        mouse_relative_pos = mouse_pos - pos

        image_top_left = vec2(self.offset)
        image_bottom_right = image_top_left + image_visible_size

        imgui.image(
            self.texture,
            imgui.Vec2(*view_size),
            imgui.Vec2(*(image_top_left / vec2(image_size))),
            imgui.Vec2(*(image_bottom_right / vec2(image_size))),
        )

        if imgui.is_item_hovered() and io.mouse_wheel != 0:
            self.zoom -= io.mouse_wheel * self.zoom_speed

            before_mouse_pixel_coordinates = vec2(mouse_relative_pos) / pixel_size
            after_pixel_size = vec2(view_size) / (vec2(image_size) * (2.0**self.zoom))
            after_mouse_pixel_coordinates = vec2(mouse_relative_pos) / after_pixel_size

            self.offset -= after_mouse_pixel_coordinates - before_mouse_pixel_coordinates

        if imgui.is_item_clicked(imgui.MouseButton.LEFT):
            self.dragging = True
            self.drag_start_pos = mouse_pos
            self.drag_start_offset = vec2(self.offset)
        elif not imgui.is_mouse_down(imgui.MouseButton.LEFT):
            self.dragging = False

        if (
            mouse_relative_pos.x >= 0
            and mouse_relative_pos.y >= 0
            and mouse_relative_pos.x < view_size.x
            and mouse_relative_pos.y < view_size.y
        ):
            window_rect_pos = vec2(pos)
            window_rect_end_pos = vec2(pos) + vec2(view_size)

            pixel_coordinates = vec2(mouse_relative_pos) / pixel_size + image_top_left
            pixel_pos = window_rect_pos + (vec2(ivec2(pixel_coordinates)) - image_top_left) * pixel_size
            pixel_image_coordinates = ivec2(floor(pixel_coordinates))

            if (
                pixel_image_coordinates.x >= 0
                and pixel_image_coordinates.y >= 0
                and pixel_image_coordinates.x < image_size.x
                and pixel_image_coordinates.y < image_size.y
            ):
                pixel_rect_pos = pixel_pos
                pixel_rect_end_pos = pixel_pos + pixel_size

                pixel_rect_pos = clamp(pixel_rect_pos, window_rect_pos, window_rect_end_pos)
                pixel_rect_end_pos = clamp(pixel_rect_end_pos, window_rect_pos, window_rect_end_pos)

                draw_list.add_rect(
                    imgui.Vec2(*pixel_rect_pos), imgui.Vec2(*pixel_rect_end_pos), 0xFF00FFFF, thickness=2
                )

                imgui.set_next_window_pos(imgui.Vec2(pixel_rect_end_pos.x + 1.0, pixel_rect_end_pos.y + 1.0))
                imgui.begin_tooltip()
                self.tooltip(pixel_image_coordinates)
                imgui.end_tooltip()

                self.pixel_under_cursor = pixel_image_coordinates

        draw_list.add_rect(cursor_pos, imgui.Vec2(*(pos + view_size)), 0xFFFFFFFF, thickness=2)

        imgui.set_cursor_screen_pos(cursor_pos)
        imgui.invisible_button("###invisible", imgui.Vec2(*view_size))

    def tooltip(self, pixel_image_coordinates: ivec2) -> None:
        img_data = self.image.get_current()
        values = img_data[pixel_image_coordinates.y, pixel_image_coordinates.x]
        values_text = ", ".join([str(v) for v in values])
        imgui.text(f"({pixel_image_coordinates.x}, {pixel_image_coordinates.y}): [{values_text}]")

    def gui(self) -> None:
        if imgui.begin(f"{self.title}###widget-{self.uid}")[0]:  # noqa: SIM102
            # NOTE: widgets are not disabled if their properties are not enabled to avoid UI appearing and disappearing.
            # They are supposed to handle missing properties gracefully in their implementation.
            if self.image.current_animation_enabled:
                self._draw()
        imgui.end()
