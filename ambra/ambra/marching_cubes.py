# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

import numpy as np
from pyxpg import (
    ComputePipeline,
    DescriptorSetBinding,
    DescriptorSetLayout,
    DescriptorType,
    PushConstantsRange,
    Shader,
    Stage,
)

from .utils.gpu import get_min_max_and_required_subgroup_size

if TYPE_CHECKING:
    from .renderer import Renderer

BLOCK_SIZE_X = 8
BLOCK_SIZE_Y = 8
BLOCK_SIZE_Z = 8
BLOCK_TOTAL_SIZE = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z

class MarchingCubesPipeline:
    def __init__(self, r: "Renderer"):
        min_subgroup_size, max_subgroup_size, required_subgroup_size = get_min_max_and_required_subgroup_size(r.ctx, Stage.COMPUTE, 32, BLOCK_TOTAL_SIZE)

        defines = [
            ("BLOCK_SIZE_X", str(BLOCK_SIZE_X)),
            ("BLOCK_SIZE_Y", str(BLOCK_SIZE_Y)),
            ("BLOCK_SIZE_Z", str(BLOCK_SIZE_Z)),
            ("MIN_WAVE_SIZE", str(min_subgroup_size)),
            ("MAX_WAVE_SIZE", str(max_subgroup_size)),
        ]

        # Check step
        self.check_constants_dtype = np.dtype(
            {
                "level": (np.float32, 0),
                "num_blocks_x": (np.uint32, 4),
                "num_blocks_y": (np.uint32, 8),
            }
        )  # type: ignore

        self.check_descriptor_set_layout = DescriptorSetLayout(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE), # 0 - in volume
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 1 - out blocks
            ]
        )

        self.check_shader = r.compile_builtin_shader("3d/marching_cubes/check_surface.slang", defines=defines)

        self.check_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.check_shader.code),
            descriptor_set_layouts=[self.check_descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.check_constants_dtype.itemsize)],
            required_subgroup_size=required_subgroup_size,
        )

        # Compact step
        self.compact_constants_dtype = np.dtype(
            {
                "num_blocks": (np.uint32, 0),
            }
        )  # type: ignore

        self.compact_descriptor_set_layout = DescriptorSetLayout(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 0 - in blocks
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 1 - out valid block count
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 2 - out compacted blocks
            ]
        )

        self.compact_shader = r.compile_builtin_shader("3d/marching_cubes/compact.slang")

        self.compact_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.compact_shader.code),
            descriptor_set_layouts=[self.compact_descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.compact_constants_dtype.itemsize)],
            required_subgroup_size=required_subgroup_size,
        )

        # Marching cubes step
        self.march_constants_dtype = np.dtype(
            {
                "level": (np.float32, 0),
                "blocks": (np.dtype((np.uint32, 2)), 4),
                "invert_normals": (np.uint32, 12),
                "block_spacing": (np.dtype((np.float32, 3)), 16),
                "max_vertices": (np.uint32, 28),
                "max_indices": (np.uint32, 32),
            }
        )  # type: ignore

        self.march_descriptor_set_layout = DescriptorSetLayout(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),  # 0 - in volume
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 1 - in compacted blocks
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 2 - in compacted blocks
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 3 - in tri table
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 4 - out positions
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 5 - out normals
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 6 - out indices
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 7 - out num vertices
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER), # 8 - out num indices
            ]
        )

        self.march_shader = r.compile_builtin_shader("3d/marching_cubes/marching_cubes.slang", defines=defines)

        self.march_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.march_shader.code),
            descriptor_set_layouts=[self.march_descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.march_constants_dtype.itemsize)],
            required_subgroup_size=required_subgroup_size,
        )