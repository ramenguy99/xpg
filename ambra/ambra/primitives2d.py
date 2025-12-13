# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import Optional, Union

import numpy as np
from pyglm.glm import mat3
from pyxpg import (
    Attachment,
    BufferUsageFlags,
    Depth,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorType,
    DeviceFeatures,
    Filter,
    Format,
    GraphicsPipeline,
    ImageLayout,
    ImageUsageFlags,
    InputAssembly,
    PipelineStage,
    PipelineStageFlags,
    PrimitiveTopology,
    PushConstantsRange,
    Rasterization,
    Sampler,
    SamplerAddressMode,
    Shader,
    Stage,
    VertexAttribute,
    VertexBinding,
    VertexInputRate,
)

from .property import BufferProperty, ImageProperty
from .renderer import Renderer
from .renderer_frame import RendererFrame
from .scene import Object, Object2D
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .viewport import Viewport


class Lines(Object2D):
    def __init__(
        self,
        lines: BufferProperty,
        colors: BufferProperty,
        line_width: Union[BufferProperty, float] = 1.0,
        is_strip: bool = False,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        super().__init__(name, translation, rotation, scale, enabled=enabled, viewport_mask=viewport_mask)
        self.is_strip = is_strip
        self.lines = self.add_buffer_property(lines, np.float32, (-1, 2), name="lines").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.colors = self.add_buffer_property(colors, np.uint32, (-1,), name="colors").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.line_width = self.add_buffer_property(line_width, np.float32, name="line_width")

    def create(self, r: Renderer) -> None:
        vert = r.compile_builtin_shader("2d/basic.slang", "vertex_main")
        frag = r.compile_builtin_shader("2d/basic.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=[
                VertexBinding(0, 8, VertexInputRate.VERTEX),
                VertexBinding(1, 4, VertexInputRate.VERTEX),
            ],
            vertex_attributes=[
                VertexAttribute(0, 0, Format.R32G32_SFLOAT),
                VertexAttribute(1, 1, Format.R32_UINT),
            ],
            rasterization=Rasterization(dynamic_line_width=True),
            input_assembly=InputAssembly(
                PrimitiveTopology.LINE_STRIP if self.is_strip else PrimitiveTopology.LINE_LIST
            ),
            samples=r.msaa_samples,
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, False, False, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def update_transform(self, parent: Optional[Object]) -> None:
        super().update_transform(parent)
        self.constants["transform"][:, :3, :3] = mat3(self.current_transform_matrix)

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        lines = self.lines.get_current_gpu()
        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=[
                lines.buffer_and_offset(),
                self.colors.get_current_gpu().buffer_and_offset(),
            ],
            descriptor_sets=[
                scene_descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )
        frame.cmd.set_line_width(
            self.line_width.get_current().item() if renderer.ctx.device_features & DeviceFeatures.WIDE_LINES else 1.0
        )

        frame.cmd.draw(lines.size // 8)


class Image(Object2D):
    def __init__(
        self,
        image: ImageProperty,
        mips: bool = False,
        name: Optional[str] = None,
        translation: Optional[BufferProperty] = None,
        rotation: Optional[BufferProperty] = None,
        scale: Optional[BufferProperty] = None,
        enabled: Optional[BufferProperty] = None,
        viewport_mask: Optional[int] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)
        super().__init__(name, translation, rotation, scale, enabled=enabled, viewport_mask=viewport_mask)
        self.image = self.add_image_property(image, name="image").use_gpu(
            ImageUsageFlags.SAMPLED,
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            PipelineStageFlags.FRAGMENT_SHADER,
            mips=mips,
        )

    def create(self, r: Renderer) -> None:
        if not (r.ctx.device_features & DeviceFeatures.SHADER_DRAW_PARAMETERS):
            raise RuntimeError(
                f"Image primitive requires {DeviceFeatures.SHADER_DRAW_PARAMETERS} whis is not available on current device."
            )

        self.sampler = Sampler(
            r.ctx,
            min_filter=Filter.LINEAR,
            mag_filter=Filter.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_EDGE,
            v=SamplerAddressMode.CLAMP_TO_EDGE,
        )

        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_sets = (
            create_descriptor_layout_pool_and_sets_ringbuffer(
                r.ctx,
                [
                    DescriptorSetBinding(1, DescriptorType.SAMPLER),
                    DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),
                ],
                r.num_frames_in_flight,
                name=f"{self.name}-descriptors",
            )
        )
        for set in self.descriptor_sets:
            set.write_sampler(self.sampler, 0)

        vert = r.compile_builtin_shader("2d/basic_texture.slang", "vertex_main")
        frag = r.compile_builtin_shader("2d/basic_texture.slang", "pixel_main")

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            input_assembly=InputAssembly(PrimitiveTopology.TRIANGLE_STRIP),
            samples=r.msaa_samples,
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, False, False, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
                self.descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def update_transform(self, parent: Optional[Object]) -> None:
        super().update_transform(parent)
        self.constants["transform"][:, :3, :3] = mat3(self.current_transform_matrix)

    def render(
        self, renderer: Renderer, frame: RendererFrame, viewport: Viewport, scene_descriptor_set: DescriptorSet
    ) -> None:
        descriptor_set = self.descriptor_sets.get_current_and_advance()
        descriptor_set.write_image(
            self.image.get_current_gpu().view(),
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            DescriptorType.SAMPLED_IMAGE,
            1,
        )

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            descriptor_sets=[
                scene_descriptor_set,
                descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )

        frame.cmd.draw(4)
