from enum import Enum
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from pyxpg import (
    CommandBuffer,
    CullMode,
    Depth,
    DescriptorSet,
    GraphicsPipeline,
    InputAssembly,
    PipelineStage,
    PrimitiveTopology,
    PushConstantsRange,
    Rasterization,
    Shader,
    Stage,
)

if TYPE_CHECKING:
    from .renderer import Renderer


class GridType(Enum):
    XY_PLANE = 0
    YZ_PLANE = 1
    XZ_PLANE = 2


class DrawGrid:
    def __init__(
        self,
        size: Tuple[float, float],
        grid_type: GridType,
        major_line_color: int,
        minor_line_color: int,
        base_color: int,
        grid_scale: float = 0.1,
        major_grid_div: float = 10.0,
        axis_line_width: float = 0.08,
        major_line_width: float = 0.04,
        minor_line_width: float = 0.01,
        pos_axis_color_scale: float = 1.0,
        neg_axis_color_scale: float = 0.5,
        test_depth: bool = True,
        write_depth: bool = False,
        is_transparent: bool = True,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
                "size": (np.dtype((np.float32, 2)), 48),
                "major_line_color": (np.uint32, 56),
                "minor_line_color": (np.uint32, 60),
                "base_color": (np.uint32, 64),
                "grid_type": (np.uint32, 68),
                "inv_grid_scale": (np.float32, 72),
                "major_grid_div": (np.float32, 76),
                "axis_line_width": (np.float32, 80),
                "major_line_width": (np.float32, 84),
                "minor_line_width": (np.float32, 88),
                "pos_axis_color_scale": (np.float32, 92),
                "neg_axis_color_scale": (np.float32, 96),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)
        self.constants["major_line_color"] = major_line_color
        self.constants["minor_line_color"] = minor_line_color
        self.constants["base_color"] = base_color
        self.constants["size"] = size
        self.constants["grid_type"] = grid_type.value
        self.constants["inv_grid_scale"] = 1.0 / grid_scale
        self.constants["major_grid_div"] = major_grid_div
        self.constants["axis_line_width"] = axis_line_width
        self.constants["major_line_width"] = major_line_width
        self.constants["minor_line_width"] = minor_line_width
        self.constants["pos_axis_color_scale"] = pos_axis_color_scale
        self.constants["neg_axis_color_scale"] = neg_axis_color_scale

        self.test_depth = test_depth
        self.write_depth = write_depth
        self.is_transparent = is_transparent

    def create(self, r: "Renderer") -> None:
        defines = []
        if self.is_transparent:
            defines.extend(r.transparency_mode_defines)

        vert = r.compile_builtin_shader("3d/grid.slang", "vertex_main", defines=defines)
        frag = r.compile_builtin_shader("3d/grid.slang", "pixel_main", defines=defines)

        self.pipeline = GraphicsPipeline(
            r.device,
            stages=[
                PipelineStage(Shader(r.device, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.device, frag.code), Stage.FRAGMENT),
            ],
            rasterization=Rasterization(cull_mode=CullMode.NONE),
            input_assembly=InputAssembly(PrimitiveTopology.TRIANGLE_STRIP),
            samples=r.msaa_samples,
            attachments=r.transparent_attachments if self.is_transparent else r.opaque_attachments,
            depth=r.transparent_depth_mode
            if self.is_transparent
            else Depth(r.depth_format, self.test_depth, self.write_depth, r.depth_compare_op),
            descriptor_set_layouts=r.transparent_descriptor_set_layouts
            if self.is_transparent
            else r.opaque_descriptor_set_layouts,
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def render(self, cmd: CommandBuffer, scene_descriptor_sets: List[DescriptorSet]) -> None:
        cmd.bind_graphics_pipeline(
            self.pipeline,
            descriptor_sets=scene_descriptor_sets,
            push_constants=self.constants.tobytes(),
        )
        cmd.draw(4)
