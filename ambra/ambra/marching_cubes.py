# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pyxpg import (
    AllocType,
    Buffer,
    BufferUsageFlags,
    ComputePipeline,
    DescriptorPool,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorSetLayout,
    DescriptorType,
    Image,
    ImageLayout,
    MemoryBarrier,
    MemoryUsage,
    PipelineStageFlags,
    PushConstantsRange,
    Shader,
    Stage,
)

from .utils.descriptors import create_descriptor_pool_and_sets_ringbuffer
from .utils.gpu import div_ceil, get_min_max_and_required_subgroup_size, view_bytes
from .utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from .renderer import Renderer

BLOCK_SIZE_X = 8
BLOCK_SIZE_Y = 8
BLOCK_SIZE_Z = 8
BLOCK_TOTAL_SIZE = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z

COMPACT_GROUP_SIZE = 128


@dataclass
class MarchingCubesPipelineInstance:
    check_constants: NDArray[Any]
    compact_constants: NDArray[Any]
    march_constants: NDArray[Any]

    check_descriptor_pool: DescriptorPool
    check_descriptor_sets: RingBuffer[DescriptorSet]
    compact_descriptor_pool: DescriptorPool
    compact_descriptor_sets: RingBuffer[DescriptorSet]
    march_descriptor_pool: DescriptorPool
    march_descriptor_sets: RingBuffer[DescriptorSet]

    valid_blocks_counter_buf: Buffer
    vertices_counter_buf: Buffer
    indices_counter_buf: Buffer
    vertices_counter_readback_buf: Optional[Buffer]
    indices_counter_readback_buf: Optional[Buffer]


@dataclass
class MarchingCubesResult:
    positions: Buffer
    normals: Buffer
    indices: Buffer


class MarchingCubesPipeline:
    def __init__(self, r: "Renderer"):
        min_subgroup_size, max_subgroup_size, required_subgroup_size = get_min_max_and_required_subgroup_size(
            r.ctx, Stage.COMPUTE, 32, BLOCK_TOTAL_SIZE
        )

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
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),  # 0 - in volume
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 1 - out blocks
            ],
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
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 0 - in blocks
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 1 - out valid block count
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 2 - out compacted blocks
            ],
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
                "num_blocks_x": (np.uint32, 4),
                "num_blocks_y": (np.uint32, 8),
                "invert_normals": (np.uint32, 12),
                "volume_spacing": (np.dtype((np.float32, 3)), 16),
                "max_vertices": (np.uint32, 28),
                "max_indices": (np.uint32, 32),
            }
        )  # type: ignore

        self.march_descriptor_set_layout = DescriptorSetLayout(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),  # 0 - in volume
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 1 - in compacted blocks
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 2 - in tri table
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 3 - out positions
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 4 - out normals
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 5 - out indices
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 6 - out num vertices
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 7 - out num indices
            ],
        )

        self.march_shader = r.compile_builtin_shader("3d/marching_cubes/marching_cubes.slang", defines=defines)
        self.march_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.march_shader.code),
            descriptor_set_layouts=[self.march_descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.march_constants_dtype.itemsize)],
            required_subgroup_size=required_subgroup_size,
        )

        self.march_count_shader = r.compile_builtin_shader(
            "3d/marching_cubes/marching_cubes.slang", defines=[*defines, ("COUNT_ONLY", "")]
        )
        self.march_count_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.march_count_shader.code),
            descriptor_set_layouts=[self.march_descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.march_constants_dtype.itemsize)],
            required_subgroup_size=required_subgroup_size,
        )

        self.tri_table = Buffer.from_data(
            r.ctx,
            view_bytes(TRIS_TABLE),
            BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name="marching-cubes-tri-table",
        )

        self.sync_instance = self.alloc_instance(r, single_set=True)

    def alloc_instance(self, r: "Renderer", single_set: bool = False) -> MarchingCubesPipelineInstance:
        # Initialize constants
        check_constants = np.zeros((1,), self.check_constants_dtype)
        compact_constants = np.zeros((1,), self.compact_constants_dtype)
        march_constants = np.zeros((1,), self.march_constants_dtype)

        # Allocate descriptor pools and sets
        number_of_sets = r.num_frames_in_flight if not single_set else 1
        check_pool, check_descriptor_sets = create_descriptor_pool_and_sets_ringbuffer(
            r.ctx,
            self.check_descriptor_set_layout,
            number_of_sets,
            name="marching-cubes-check",
        )
        compact_pool, compact_descriptor_sets = create_descriptor_pool_and_sets_ringbuffer(
            r.ctx,
            self.compact_descriptor_set_layout,
            number_of_sets,
            name="marching-cubes-compact",
        )
        march_pool, march_descriptor_sets = create_descriptor_pool_and_sets_ringbuffer(
            r.ctx,
            self.march_descriptor_set_layout,
            number_of_sets,
            name="marching-cubes-march",
        )

        # Allocate counter buffers
        valid_blocks_counter_buf = Buffer.from_data(
            r.ctx,
            view_bytes(np.array([0, 1, 1], np.uint32)),  # Alloc 3 elements to allow using this for indirect dispatch
            BufferUsageFlags.STORAGE
            | BufferUsageFlags.TRANSFER_DST
            | BufferUsageFlags.INDIRECT
            | BufferUsageFlags.TRANSFER_SRC,
            AllocType.DEVICE,
            name="marching-cubes-valid-blocks-counter",
        )

        vertices_counter_buf = Buffer.from_data(
            r.ctx,
            view_bytes(np.zeros((1,), np.uint32)),
            BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.TRANSFER_SRC,
            AllocType.DEVICE_MAPPED_WITH_FALLBACK,
            name="marching-cubes-vertices-counter",
        )
        if not vertices_counter_buf.is_mapped:
            vertices_counter_readback_buf = Buffer.from_data(
                r.ctx,
                view_bytes(np.zeros((1,), np.uint32)),
                BufferUsageFlags.TRANSFER_DST,
                AllocType.HOST,
                name="marching-cubes-vertices-counter-readback",
            )
        else:
            vertices_counter_readback_buf = None

        indices_counter_buf = Buffer.from_data(
            r.ctx,
            view_bytes(np.zeros((1,), np.uint32)),
            BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST | BufferUsageFlags.TRANSFER_SRC,
            AllocType.DEVICE_MAPPED_WITH_FALLBACK,
            name="marching-cubes-indices-counter",
        )
        if not indices_counter_buf.is_mapped:
            indices_counter_readback_buf = Buffer.from_data(
                r.ctx,
                view_bytes(np.zeros((1,), np.uint32)),
                BufferUsageFlags.TRANSFER_DST,
                AllocType.HOST,
                name="marching-cubes-indices-counter-readback",
            )
        else:
            indices_counter_readback_buf = None

        # Write static descriptors
        for s in compact_descriptor_sets:
            s.write_buffer(valid_blocks_counter_buf, DescriptorType.STORAGE_BUFFER, 1)

        for s in march_descriptor_sets:
            s.write_buffer(self.tri_table, DescriptorType.STORAGE_BUFFER, 2)
            s.write_buffer(vertices_counter_buf, DescriptorType.STORAGE_BUFFER, 6)
            s.write_buffer(indices_counter_buf, DescriptorType.STORAGE_BUFFER, 7)

        return MarchingCubesPipelineInstance(
            check_constants,
            compact_constants,
            march_constants,
            check_pool,
            check_descriptor_sets,
            compact_pool,
            compact_descriptor_sets,
            march_pool,
            march_descriptor_sets,
            valid_blocks_counter_buf,
            vertices_counter_buf,
            indices_counter_buf,
            vertices_counter_readback_buf,
            indices_counter_readback_buf,
        )

    def run_sync(
        self,
        r: "Renderer",
        sdf: Image,
        size: Tuple[float, float, float],
        level: float = 0.0,
        invert_normals: bool = False,
    ) -> MarchingCubesResult:
        width, height, depth = sdf.width, sdf.height, sdf.depth
        blocks_x = div_ceil(width, BLOCK_SIZE_X - 1)
        blocks_y = div_ceil(height, BLOCK_SIZE_Y - 1)
        blocks_z = div_ceil(depth, BLOCK_SIZE_Z - 1)
        total_blocks = blocks_x * blocks_y * blocks_z

        blocks_buffer = Buffer(
            r.ctx,
            total_blocks * 4,
            (BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_SRC),
            AllocType.DEVICE,
            name="marching-cubes-sync-blocks",
        )
        valid_blocks_buffer = Buffer(
            r.ctx,
            total_blocks * 4,
            (BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_SRC),
            AllocType.DEVICE,
            name="marching-cubes-sync-valid-blocks",
        )

        instance = self.sync_instance
        check_set = instance.check_descriptor_sets.get_current()
        check_set.write_image(sdf, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)
        check_set.write_buffer(blocks_buffer, DescriptorType.STORAGE_BUFFER, 1)

        compact_set = instance.compact_descriptor_sets.get_current()
        compact_set.write_buffer(blocks_buffer, DescriptorType.STORAGE_BUFFER, 0)
        compact_set.write_buffer(valid_blocks_buffer, DescriptorType.STORAGE_BUFFER, 2)

        march_set = instance.march_descriptor_sets.get_current()
        march_set.write_image(sdf, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)
        march_set.write_buffer(valid_blocks_buffer, DescriptorType.STORAGE_BUFFER, 1)
        march_set.write_buffer(r.zero_buffer, DescriptorType.STORAGE_BUFFER, 3)
        march_set.write_buffer(r.zero_buffer, DescriptorType.STORAGE_BUFFER, 4)
        march_set.write_buffer(r.zero_buffer, DescriptorType.STORAGE_BUFFER, 5)

        instance.check_constants["level"] = level
        instance.check_constants["num_blocks_x"] = blocks_x
        instance.check_constants["num_blocks_y"] = blocks_y

        instance.compact_constants["num_blocks"] = total_blocks

        instance.march_constants["level"] = level
        instance.march_constants["num_blocks_x"] = blocks_x
        instance.march_constants["num_blocks_y"] = blocks_y
        instance.march_constants["invert_normals"] = invert_normals
        instance.march_constants["volume_spacing"] = (size[0] / width, size[1] / height, size[2] / depth)
        instance.march_constants["max_vertices"] = 0
        instance.march_constants["max_indices"] = 0

        with r.ctx.sync_commands() as cmd:
            cmd.bind_compute_pipeline(
                self.check_pipeline, descriptor_sets=[check_set], push_constants=instance.check_constants.tobytes()
            )
            cmd.dispatch(blocks_x, blocks_y, blocks_z)

            cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

            cmd.bind_compute_pipeline(
                self.compact_pipeline,
                descriptor_sets=[compact_set],
                push_constants=instance.compact_constants.tobytes(),
            )
            cmd.dispatch(div_ceil(total_blocks, COMPACT_GROUP_SIZE), 1, 1)

            cmd.memory_barrier_full(
                MemoryBarrier(
                    src_stage=PipelineStageFlags.COMPUTE_SHADER,
                    dst_stage=PipelineStageFlags.COMPUTE_SHADER | PipelineStageFlags.DRAW_INDIRECT,
                )
            )

            cmd.bind_compute_pipeline(
                self.march_count_pipeline,
                descriptor_sets=[march_set],
                push_constants=instance.march_constants.tobytes(),
            )
            cmd.dispatch_indirect(instance.valid_blocks_counter_buf, 0)

            readback_src_stage = PipelineStageFlags.NONE
            if instance.vertices_counter_readback_buf is None or instance.indices_counter_readback_buf is None:
                readback_src_stage |= PipelineStageFlags.COMPUTE_SHADER
            if instance.vertices_counter_readback_buf is not None or instance.indices_counter_readback_buf is not None:
                # Ensure compute is done before issuing copy for readback
                readback_src_stage |= PipelineStageFlags.TRANSFER
                cmd.memory_barrier_full(
                    MemoryBarrier(src_stage=PipelineStageFlags.COMPUTE_SHADER, dst_stage=PipelineStageFlags.TRANSFER)
                )

            # Copy to readback buffer
            if instance.vertices_counter_readback_buf is not None:
                cmd.copy_buffer(instance.vertices_counter_buf, instance.vertices_counter_readback_buf)
            if instance.indices_counter_readback_buf is not None:
                cmd.copy_buffer(instance.indices_counter_buf, instance.indices_counter_readback_buf)

            # Ensure copy is done before clearing counter
            if instance.vertices_counter_readback_buf is not None or instance.indices_counter_readback_buf is not None:
                cmd.memory_barrier_full(
                    MemoryBarrier(src_stage=PipelineStageFlags.TRANSFER, dst_stage=PipelineStageFlags.TRANSFER)
                )

            # Clear GPU counter buffer. If the buffer is mapped we do this from the CPU later instead
            if instance.vertices_counter_readback_buf is not None:
                cmd.fill_buffer(instance.vertices_counter_buf, 0, 4)
            if instance.indices_counter_readback_buf is not None:
                cmd.fill_buffer(instance.indices_counter_buf, 0, 4)

            # Ensure copy is visible to host (either written form compute or from copy)
            cmd.memory_barrier_full(MemoryBarrier(src_stage=readback_src_stage, dst_stage=PipelineStageFlags.HOST))

        vertices_counter_readback_buf = (
            instance.vertices_counter_readback_buf
            if instance.vertices_counter_readback_buf is not None
            else instance.vertices_counter_buf
        )
        indices_counter_readback_buf = (
            instance.indices_counter_readback_buf
            if instance.indices_counter_readback_buf is not None
            else instance.indices_counter_buf
        )

        vertices_arr = np.frombuffer(vertices_counter_readback_buf.data, np.uint32)
        vertices_count = vertices_arr[0]
        vertices_arr[0] = 0

        indices_arr = np.frombuffer(indices_counter_readback_buf.data, np.uint32)
        indices_count = indices_arr[0]
        indices_arr[0] = 0

        positions_buf = Buffer(
            r.ctx,
            vertices_count * 12,
            (BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_SRC | BufferUsageFlags.VERTEX),
            AllocType.DEVICE,
            name="marching-cubes-sync-positions",
        )
        normals_buf = Buffer(
            r.ctx,
            vertices_count * 12,
            (BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_SRC | BufferUsageFlags.VERTEX),
            AllocType.DEVICE,
            name="marching-cubes-sync-normals",
        )
        indices_buf = Buffer(
            r.ctx,
            indices_count * 4,
            (BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_SRC | BufferUsageFlags.INDEX),
            AllocType.DEVICE,
            name="marching-cubes-sync-indices",
        )
        march_set.write_buffer(positions_buf, DescriptorType.STORAGE_BUFFER, 3)
        march_set.write_buffer(normals_buf, DescriptorType.STORAGE_BUFFER, 4)
        march_set.write_buffer(indices_buf, DescriptorType.STORAGE_BUFFER, 5)

        instance.march_constants["max_vertices"] = vertices_count
        instance.march_constants["max_indices"] = indices_count
        with r.ctx.sync_commands() as cmd:
            cmd.bind_compute_pipeline(
                self.march_pipeline, descriptor_sets=[march_set], push_constants=instance.march_constants.tobytes()
            )
            cmd.dispatch_indirect(instance.valid_blocks_counter_buf, 0)
            cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.ALL)

        blocks_buffer.destroy()
        valid_blocks_buffer.destroy()

        return MarchingCubesResult(positions_buf, normals_buf, indices_buf)


X = 255
TRIS_TABLE = np.array(
    [
        [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [0, 8, 3, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [0, 1, 9, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [1, 8, 3, 9, 8, 1, X, X, X, X, X, X, X, X, X, X],
        [1, 2, 10, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [0, 8, 3, 1, 2, 10, X, X, X, X, X, X, X, X, X, X],
        [9, 2, 10, 0, 2, 9, X, X, X, X, X, X, X, X, X, X],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, X, X, X, X, X, X, X],
        [3, 11, 2, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [0, 11, 2, 8, 11, 0, X, X, X, X, X, X, X, X, X, X],
        [1, 9, 0, 2, 3, 11, X, X, X, X, X, X, X, X, X, X],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, X, X, X, X, X, X, X],
        [3, 10, 1, 11, 10, 3, X, X, X, X, X, X, X, X, X, X],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, X, X, X, X, X, X, X],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, X, X, X, X, X, X, X],
        [9, 8, 10, 10, 8, 11, X, X, X, X, X, X, X, X, X, X],
        [4, 7, 8, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [4, 3, 0, 7, 3, 4, X, X, X, X, X, X, X, X, X, X],
        [0, 1, 9, 8, 4, 7, X, X, X, X, X, X, X, X, X, X],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, X, X, X, X, X, X, X],
        [1, 2, 10, 8, 4, 7, X, X, X, X, X, X, X, X, X, X],
        [3, 4, 7, 3, 0, 4, 1, 2, 10, X, X, X, X, X, X, X],
        [9, 2, 10, 9, 0, 2, 8, 4, 7, X, X, X, X, X, X, X],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, X, X, X, X],
        [8, 4, 7, 3, 11, 2, X, X, X, X, X, X, X, X, X, X],
        [11, 4, 7, 11, 2, 4, 2, 0, 4, X, X, X, X, X, X, X],
        [9, 0, 1, 8, 4, 7, 2, 3, 11, X, X, X, X, X, X, X],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, X, X, X, X],
        [3, 10, 1, 3, 11, 10, 7, 8, 4, X, X, X, X, X, X, X],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, X, X, X, X],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, X, X, X, X],
        [4, 7, 11, 4, 11, 9, 9, 11, 10, X, X, X, X, X, X, X],
        [9, 5, 4, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [9, 5, 4, 0, 8, 3, X, X, X, X, X, X, X, X, X, X],
        [0, 5, 4, 1, 5, 0, X, X, X, X, X, X, X, X, X, X],
        [8, 5, 4, 8, 3, 5, 3, 1, 5, X, X, X, X, X, X, X],
        [1, 2, 10, 9, 5, 4, X, X, X, X, X, X, X, X, X, X],
        [3, 0, 8, 1, 2, 10, 4, 9, 5, X, X, X, X, X, X, X],
        [5, 2, 10, 5, 4, 2, 4, 0, 2, X, X, X, X, X, X, X],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, X, X, X, X],
        [9, 5, 4, 2, 3, 11, X, X, X, X, X, X, X, X, X, X],
        [0, 11, 2, 0, 8, 11, 4, 9, 5, X, X, X, X, X, X, X],
        [0, 5, 4, 0, 1, 5, 2, 3, 11, X, X, X, X, X, X, X],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, X, X, X, X],
        [10, 3, 11, 10, 1, 3, 9, 5, 4, X, X, X, X, X, X, X],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, X, X, X, X],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, X, X, X, X],
        [5, 4, 8, 5, 8, 10, 10, 8, 11, X, X, X, X, X, X, X],
        [9, 7, 8, 5, 7, 9, X, X, X, X, X, X, X, X, X, X],
        [9, 3, 0, 9, 5, 3, 5, 7, 3, X, X, X, X, X, X, X],
        [0, 7, 8, 0, 1, 7, 1, 5, 7, X, X, X, X, X, X, X],
        [1, 5, 3, 3, 5, 7, X, X, X, X, X, X, X, X, X, X],
        [9, 7, 8, 9, 5, 7, 10, 1, 2, X, X, X, X, X, X, X],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, X, X, X, X],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, X, X, X, X],
        [2, 10, 5, 2, 5, 3, 3, 5, 7, X, X, X, X, X, X, X],
        [7, 9, 5, 7, 8, 9, 3, 11, 2, X, X, X, X, X, X, X],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, X, X, X, X],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, X, X, X, X],
        [11, 2, 1, 11, 1, 7, 7, 1, 5, X, X, X, X, X, X, X],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, X, X, X, X],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, X],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, X],
        [11, 10, 5, 7, 11, 5, X, X, X, X, X, X, X, X, X, X],
        [10, 6, 5, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [0, 8, 3, 5, 10, 6, X, X, X, X, X, X, X, X, X, X],
        [9, 0, 1, 5, 10, 6, X, X, X, X, X, X, X, X, X, X],
        [1, 8, 3, 1, 9, 8, 5, 10, 6, X, X, X, X, X, X, X],
        [1, 6, 5, 2, 6, 1, X, X, X, X, X, X, X, X, X, X],
        [1, 6, 5, 1, 2, 6, 3, 0, 8, X, X, X, X, X, X, X],
        [9, 6, 5, 9, 0, 6, 0, 2, 6, X, X, X, X, X, X, X],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, X, X, X, X],
        [2, 3, 11, 10, 6, 5, X, X, X, X, X, X, X, X, X, X],
        [11, 0, 8, 11, 2, 0, 10, 6, 5, X, X, X, X, X, X, X],
        [0, 1, 9, 2, 3, 11, 5, 10, 6, X, X, X, X, X, X, X],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, X, X, X, X],
        [6, 3, 11, 6, 5, 3, 5, 1, 3, X, X, X, X, X, X, X],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, X, X, X, X],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, X, X, X, X],
        [6, 5, 9, 6, 9, 11, 11, 9, 8, X, X, X, X, X, X, X],
        [5, 10, 6, 4, 7, 8, X, X, X, X, X, X, X, X, X, X],
        [4, 3, 0, 4, 7, 3, 6, 5, 10, X, X, X, X, X, X, X],
        [1, 9, 0, 5, 10, 6, 8, 4, 7, X, X, X, X, X, X, X],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, X, X, X, X],
        [6, 1, 2, 6, 5, 1, 4, 7, 8, X, X, X, X, X, X, X],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, X, X, X, X],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, X, X, X, X],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, X],
        [3, 11, 2, 7, 8, 4, 10, 6, 5, X, X, X, X, X, X, X],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, X, X, X, X],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, X, X, X, X],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, X],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, X, X, X, X],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, X],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, X],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, X, X, X, X],
        [10, 4, 9, 6, 4, 10, X, X, X, X, X, X, X, X, X, X],
        [4, 10, 6, 4, 9, 10, 0, 8, 3, X, X, X, X, X, X, X],
        [10, 0, 1, 10, 6, 0, 6, 4, 0, X, X, X, X, X, X, X],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, X, X, X, X],
        [1, 4, 9, 1, 2, 4, 2, 6, 4, X, X, X, X, X, X, X],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, X, X, X, X],
        [0, 2, 4, 4, 2, 6, X, X, X, X, X, X, X, X, X, X],
        [8, 3, 2, 8, 2, 4, 4, 2, 6, X, X, X, X, X, X, X],
        [10, 4, 9, 10, 6, 4, 11, 2, 3, X, X, X, X, X, X, X],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, X, X, X, X],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, X, X, X, X],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, X],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, X, X, X, X],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, X],
        [3, 11, 6, 3, 6, 0, 0, 6, 4, X, X, X, X, X, X, X],
        [6, 4, 8, 11, 6, 8, X, X, X, X, X, X, X, X, X, X],
        [7, 10, 6, 7, 8, 10, 8, 9, 10, X, X, X, X, X, X, X],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, X, X, X, X],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, X, X, X, X],
        [10, 6, 7, 10, 7, 1, 1, 7, 3, X, X, X, X, X, X, X],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, X, X, X, X],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, X],
        [7, 8, 0, 7, 0, 6, 6, 0, 2, X, X, X, X, X, X, X],
        [7, 3, 2, 6, 7, 2, X, X, X, X, X, X, X, X, X, X],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, X, X, X, X],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, X],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, X],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, X, X, X, X],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, X],
        [0, 9, 1, 11, 6, 7, X, X, X, X, X, X, X, X, X, X],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, X, X, X, X],
        [7, 11, 6, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [7, 6, 11, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [3, 0, 8, 11, 7, 6, X, X, X, X, X, X, X, X, X, X],
        [0, 1, 9, 11, 7, 6, X, X, X, X, X, X, X, X, X, X],
        [8, 1, 9, 8, 3, 1, 11, 7, 6, X, X, X, X, X, X, X],
        [10, 1, 2, 6, 11, 7, X, X, X, X, X, X, X, X, X, X],
        [1, 2, 10, 3, 0, 8, 6, 11, 7, X, X, X, X, X, X, X],
        [2, 9, 0, 2, 10, 9, 6, 11, 7, X, X, X, X, X, X, X],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, X, X, X, X],
        [7, 2, 3, 6, 2, 7, X, X, X, X, X, X, X, X, X, X],
        [7, 0, 8, 7, 6, 0, 6, 2, 0, X, X, X, X, X, X, X],
        [2, 7, 6, 2, 3, 7, 0, 1, 9, X, X, X, X, X, X, X],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, X, X, X, X],
        [10, 7, 6, 10, 1, 7, 1, 3, 7, X, X, X, X, X, X, X],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, X, X, X, X],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, X, X, X, X],
        [7, 6, 10, 7, 10, 8, 8, 10, 9, X, X, X, X, X, X, X],
        [6, 8, 4, 11, 8, 6, X, X, X, X, X, X, X, X, X, X],
        [3, 6, 11, 3, 0, 6, 0, 4, 6, X, X, X, X, X, X, X],
        [8, 6, 11, 8, 4, 6, 9, 0, 1, X, X, X, X, X, X, X],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, X, X, X, X],
        [6, 8, 4, 6, 11, 8, 2, 10, 1, X, X, X, X, X, X, X],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, X, X, X, X],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, X, X, X, X],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, X],
        [8, 2, 3, 8, 4, 2, 4, 6, 2, X, X, X, X, X, X, X],
        [0, 4, 2, 4, 6, 2, X, X, X, X, X, X, X, X, X, X],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, X, X, X, X],
        [1, 9, 4, 1, 4, 2, 2, 4, 6, X, X, X, X, X, X, X],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, X, X, X, X],
        [10, 1, 0, 10, 0, 6, 6, 0, 4, X, X, X, X, X, X, X],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, X],
        [10, 9, 4, 6, 10, 4, X, X, X, X, X, X, X, X, X, X],
        [4, 9, 5, 7, 6, 11, X, X, X, X, X, X, X, X, X, X],
        [0, 8, 3, 4, 9, 5, 11, 7, 6, X, X, X, X, X, X, X],
        [5, 0, 1, 5, 4, 0, 7, 6, 11, X, X, X, X, X, X, X],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, X, X, X, X],
        [9, 5, 4, 10, 1, 2, 7, 6, 11, X, X, X, X, X, X, X],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, X, X, X, X],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, X, X, X, X],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, X],
        [7, 2, 3, 7, 6, 2, 5, 4, 9, X, X, X, X, X, X, X],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, X, X, X, X],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, X, X, X, X],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, X],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, X, X, X, X],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, X],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, X],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, X, X, X, X],
        [6, 9, 5, 6, 11, 9, 11, 8, 9, X, X, X, X, X, X, X],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, X, X, X, X],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, X, X, X, X],
        [6, 11, 3, 6, 3, 5, 5, 3, 1, X, X, X, X, X, X, X],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, X, X, X, X],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, X],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, X],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, X, X, X, X],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, X, X, X, X],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, X, X, X, X, X, X, X],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, X],
        [1, 5, 6, 2, 1, 6, X, X, X, X, X, X, X, X, X, X],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, X],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, X, X, X, X],
        [0, 3, 8, 5, 6, 10, X, X, X, X, X, X, X, X, X, X],
        [10, 5, 6, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [11, 5, 10, 7, 5, 11, X, X, X, X, X, X, X, X, X, X],
        [11, 5, 10, 11, 7, 5, 8, 3, 0, X, X, X, X, X, X, X],
        [5, 11, 7, 5, 10, 11, 1, 9, 0, X, X, X, X, X, X, X],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, X, X, X, X],
        [11, 1, 2, 11, 7, 1, 7, 5, 1, X, X, X, X, X, X, X],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, X, X, X, X],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, X, X, X, X],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, X],
        [2, 5, 10, 2, 3, 5, 3, 7, 5, X, X, X, X, X, X, X],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, X, X, X, X],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, X, X, X, X],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, X],
        [1, 3, 5, 3, 7, 5, X, X, X, X, X, X, X, X, X, X],
        [0, 8, 7, 0, 7, 1, 1, 7, 5, X, X, X, X, X, X, X],
        [9, 0, 3, 9, 3, 5, 5, 3, 7, X, X, X, X, X, X, X],
        [9, 8, 7, 5, 9, 7, X, X, X, X, X, X, X, X, X, X],
        [5, 8, 4, 5, 10, 8, 10, 11, 8, X, X, X, X, X, X, X],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, X, X, X, X],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, X, X, X, X],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, X],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, X, X, X, X],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, X],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, X],
        [9, 4, 5, 2, 11, 3, X, X, X, X, X, X, X, X, X, X],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, X, X, X, X],
        [5, 10, 2, 5, 2, 4, 4, 2, 0, X, X, X, X, X, X, X],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, X],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, X, X, X, X],
        [8, 4, 5, 8, 5, 3, 3, 5, 1, X, X, X, X, X, X, X],
        [0, 4, 5, 1, 0, 5, X, X, X, X, X, X, X, X, X, X],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, X, X, X, X],
        [9, 4, 5, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [4, 11, 7, 4, 9, 11, 9, 10, 11, X, X, X, X, X, X, X],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, X, X, X, X],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, X, X, X, X],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, X],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, X, X, X, X],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, X],
        [11, 7, 4, 11, 4, 2, 2, 4, 0, X, X, X, X, X, X, X],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, X, X, X, X],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, X, X, X, X],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, X],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, X],
        [1, 10, 2, 8, 7, 4, X, X, X, X, X, X, X, X, X, X],
        [4, 9, 1, 4, 1, 7, 7, 1, 3, X, X, X, X, X, X, X],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, X, X, X, X],
        [4, 0, 3, 7, 4, 3, X, X, X, X, X, X, X, X, X, X],
        [4, 8, 7, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [9, 10, 8, 10, 11, 8, X, X, X, X, X, X, X, X, X, X],
        [3, 0, 9, 3, 9, 11, 11, 9, 10, X, X, X, X, X, X, X],
        [0, 1, 10, 0, 10, 8, 8, 10, 11, X, X, X, X, X, X, X],
        [3, 1, 10, 11, 3, 10, X, X, X, X, X, X, X, X, X, X],
        [1, 2, 11, 1, 11, 9, 9, 11, 8, X, X, X, X, X, X, X],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, X, X, X, X],
        [0, 2, 11, 8, 0, 11, X, X, X, X, X, X, X, X, X, X],
        [3, 2, 11, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [2, 3, 8, 2, 8, 10, 10, 8, 9, X, X, X, X, X, X, X],
        [9, 10, 2, 0, 9, 2, X, X, X, X, X, X, X, X, X, X],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, X, X, X, X],
        [1, 10, 2, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [1, 3, 8, 9, 1, 8, X, X, X, X, X, X, X, X, X, X],
        [0, 9, 1, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [0, 3, 8, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
    ],
    dtype=np.uint32,
)
