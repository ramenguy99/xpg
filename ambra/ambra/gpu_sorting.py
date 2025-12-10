# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
from pyxpg import (
    AccessFlags,
    AllocType,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    ComputePipeline,
    DescriptorSetBinding,
    DescriptorType,
    DeviceFeatures,
    MemoryBarrier,
    MemoryUsage,
    PipelineStageFlags,
    PushConstantsRange,
    Shader,
)

from . import renderer
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .utils.gpu import div_round_up

RADIX_SORT_BITS = 8
RADIX_SORT_RADIX = 1 << RADIX_SORT_BITS
RADIX_SORT_PASSES = 4
RADIX_SORT_GLOBAL_HIST_PARTITION_SIZE = 32768


class SortDataType(Enum):
    FLOAT32 = auto()
    INT32 = auto()
    UINT32 = auto()


# TODO: tuning based on vendor ID with fallback to default
# and validation on pipeline creation.
#
# Found so far:
#
# Apple M4 Max:
# keys_per_thread = 7,
# threads_per_threadblock = 256,
# partition_size = 7 * 256,
# total_shared_memory = 4096,
#
# NVDIA RTX 3060 Mobile:
# keys_per_thread = 15
# threads_per_threadblock = 256
# partition_size = 3840
# total_shared_memory = 4096
@dataclass(frozen=True)
class SortTuningParameters:
    lock_wave32: bool = False
    keys_per_thread: int = 15
    threads_per_threadblock: int = 256
    partition_size: int = 3840
    total_shared_memory: int = 4096


@dataclass(frozen=True)
class SortOptions:
    key_type: SortDataType
    payload_type: Optional[SortDataType] = None
    max_count: int = 256 * 1024 * 1024  # Max number of elements that can be sorted
    descending: bool = False  # If True sort in descending order. Otherwise ascending.
    unsafe_has_forward_thread_progress_guarantee: bool = False
    indirect: bool = False  # If True sort will be dispatched indirectly (currently direct dispatch is not allowed if this is set, but could be extended to allow both)


@dataclass
class DeviceRadixSortPipeline:
    init: ComputePipeline
    upsweep: ComputePipeline
    scan: ComputePipeline
    downsweep: ComputePipeline


@dataclass
class OnesweepPipeline:
    init: ComputePipeline
    global_histogram: ComputePipeline
    scan: ComputePipeline
    onesweep: ComputePipeline


class GpuSortingPipeline:
    def __init__(self, r: "renderer.Renderer", options: SortOptions, tuning: Optional[SortTuningParameters] = None):
        self.options = options
        self.tuning = tuning or SortTuningParameters()

        self.constants_dtype = np.dtype(
            {
                "numKeys": (np.uint32, 0),
                "radixShift": (np.uint32, 4),
                "threadBlocks": (np.uint32, 8),
                "isPartial": (np.uint32, 12),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        # NOTE: could switch to a different descriptor set just for the init kernel to use less max slots
        descriptors = [
            DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 0 - Keys
            DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 1 - Alt-keys
            DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 2 - Payload
            DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 3 - Alt-payload
            DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 4 - Global hist
            DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 5 - Pass hist
            DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 6 - Index (sweep only)
        ]

        if options.indirect:
            descriptors.extend(
                [
                    DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),  #  7 - indirect constants
                    DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  #  8 - indirect constants write
                    DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  #  9 - indirect dispatch write
                    DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),  # 10 - indirect num keys
                ]
            )

        # Allocate 2 sets per frame to ping-pong radix passes
        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_sets = (
            create_descriptor_layout_pool_and_sets_ringbuffer(
                r.ctx,
                descriptors,
                2 * r.num_frames_in_flight,
            )
        )

        enable_16_bit = (r.ctx.device_features & DeviceFeatures.SHADER_INT16) != 0

        # Permutations are:
        # 3 key types
        # 4 payload types (3 dtypes + 1 if not sorting pairs)
        # 2 sorting directions
        # 2 modes with 16 bits or 32 bits storage
        # Total:
        #   48 permutations max
        #   12 permutations if never sorting pairs
        #    6 permutations if never sorting descending
        #    3 permutations if never using 16 bit storage
        defines = []
        if options.key_type == SortDataType.FLOAT32:
            defines.append(("KEY_FLOAT", ""))
        elif options.key_type == SortDataType.INT32:
            defines.append(("KEY_INT", ""))
        elif options.key_type == SortDataType.UINT32:
            defines.append(("KEY_UINT", ""))
        else:
            raise ValueError(f"Unhandled key data type: {options.key_type}")

        if options.payload_type is not None:
            defines.append(("SORT_PAIRS", ""))
            if options.payload_type == SortDataType.FLOAT32:
                defines.append(("PAYLOAD_FLOAT", ""))
            elif options.payload_type == SortDataType.INT32:
                defines.append(("PAYLOAD_INT", ""))
            elif options.payload_type == SortDataType.UINT32:
                defines.append(("PAYLOAD_UINT", ""))
            else:
                raise ValueError(f"Unhandled payload data type: {options.payload_type}")
        else:
            defines.append(("PAYLOAD_UINT", ""))

        if not options.descending:
            defines.append(("SHOULD_ASCEND", ""))

        if enable_16_bit:
            defines.append(("ENABLE_16_BIT", ""))

        defines.append(("MAX_DISPATCH_DIM", str(r.ctx.device_properties.limits.max_compute_work_group_count[0])))
        if options.indirect:
            defines.append(("INDIRECT_DISPATCH", ""))

        # Tuning
        defines.append((f"KEYS_PER_THREAD_{self.tuning.keys_per_thread}", ""))
        defines.append((f"PART_SIZE_{self.tuning.partition_size}", ""))
        defines.append((f"D_TOTAL_SMEM_{self.tuning.total_shared_memory}", ""))
        defines.append((f"D_DIM_{self.tuning.threads_per_threadblock}", ""))

        def create_pipeline(shader_name: str, entry: str) -> ComputePipeline:
            shader = r.compile_builtin_shader(
                Path("GPUSorting", shader_name),
                entry=entry,
                defines=defines,
                include_paths=[renderer.SHADERS_PATH.joinpath("GPUSorting")],
            )
            return ComputePipeline(
                r.ctx,
                Shader(r.ctx, shader.code),
                descriptor_set_layouts=[self.descriptor_set_layout],
                push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
            )

        if options.unsafe_has_forward_thread_progress_guarantee:
            self.onesweep_pipeline = OnesweepPipeline(
                init=create_pipeline("SweepCommon.slang", "InitSweep"),
                global_histogram=create_pipeline("SweepCommon.slang", "GlobalHistogram"),
                scan=create_pipeline("SweepCommon.slang", "Scan"),
                onesweep=create_pipeline("OneSweep.slang", "DigitBinningPass"),
            )
        else:
            self.device_radix_sort_pipeline = DeviceRadixSortPipeline(
                init=create_pipeline("DeviceRadixSort.slang", "InitDeviceRadixSort"),
                upsweep=create_pipeline("DeviceRadixSort.slang", "Upsweep"),
                scan=create_pipeline("DeviceRadixSort.slang", "Scan"),
                downsweep=create_pipeline("DeviceRadixSort.slang", "Downsweep"),
            )

        self.global_histogram_buf = Buffer(
            r.ctx,
            RADIX_SORT_RADIX * RADIX_SORT_PASSES * 4,
            BufferUsageFlags.STORAGE,
            AllocType.DEVICE,
            name="gpu-sorting-global-histogram",
        )

        pass_histogram_buf_size = div_round_up(options.max_count, self.tuning.partition_size) * RADIX_SORT_RADIX * 4
        if options.unsafe_has_forward_thread_progress_guarantee:
            pass_histogram_buf_size *= RADIX_SORT_PASSES

        self.pass_histogram_buf = Buffer(
            r.ctx,
            pass_histogram_buf_size,
            BufferUsageFlags.STORAGE,
            AllocType.DEVICE,
            name="gpu-sorting-pass-histogram",
        )
        if options.unsafe_has_forward_thread_progress_guarantee:
            self.index_buf = Buffer(
                r.ctx,
                RADIX_SORT_PASSES * 4,
                BufferUsageFlags.STORAGE,
                AllocType.DEVICE,
                name="gpu-sorting-index",
            )

        if options.indirect:
            self.indirect_constants_buf = Buffer(
                r.ctx,
                16,
                BufferUsageFlags.STORAGE | BufferUsageFlags.UNIFORM,
                AllocType.DEVICE,
                name="gpu-sorting-indirect-constants",
            )

            self.indirect_dispatch_buf = Buffer(
                r.ctx,
                48 if options.unsafe_has_forward_thread_progress_guarantee else 24,
                BufferUsageFlags.STORAGE | BufferUsageFlags.INDIRECT,
                AllocType.DEVICE,
                name="gpu-sorting-indirect-dispatch-buf",
            )

        for s in self.descriptor_sets:
            s.write_buffer(self.global_histogram_buf, DescriptorType.STORAGE_BUFFER, 4)
            s.write_buffer(self.pass_histogram_buf, DescriptorType.STORAGE_BUFFER, 5)
            if options.unsafe_has_forward_thread_progress_guarantee:
                s.write_buffer(self.index_buf, DescriptorType.STORAGE_BUFFER, 6)
            if options.indirect:
                s.write_buffer(
                    self.indirect_constants_buf, DescriptorType.UNIFORM_BUFFER, 7
                )  # Indirect constants for reading
                s.write_buffer(
                    self.indirect_constants_buf, DescriptorType.STORAGE_BUFFER, 8
                )  # Indirect constants for writing in init shader
                s.write_buffer(
                    self.indirect_dispatch_buf, DescriptorType.STORAGE_BUFFER, 9
                )  # Indirect dispatch counts for writing in init shader

    def upload(
        self,
        r: "renderer.Renderer",
        key: Buffer,
        key_alt: Buffer,
        payload: Buffer,
        payload_alt: Buffer,
        indirect_count: Optional[Buffer],
    ) -> None:
        set0 = self.descriptor_sets.get_current_and_advance()
        set1 = self.descriptor_sets.get_current_and_advance()

        set0.write_buffer(key, DescriptorType.STORAGE_BUFFER, 0)
        set0.write_buffer(key_alt, DescriptorType.STORAGE_BUFFER, 1)
        set0.write_buffer(payload, DescriptorType.STORAGE_BUFFER, 2)
        set0.write_buffer(payload_alt, DescriptorType.STORAGE_BUFFER, 3)

        if indirect_count is not None:
            set0.write_buffer(indirect_count, DescriptorType.STORAGE_BUFFER, 10)

        set1.write_buffer(key_alt, DescriptorType.STORAGE_BUFFER, 0)
        set1.write_buffer(key, DescriptorType.STORAGE_BUFFER, 1)
        set1.write_buffer(payload_alt, DescriptorType.STORAGE_BUFFER, 2)
        set1.write_buffer(payload, DescriptorType.STORAGE_BUFFER, 3)

        self.current_sets = [set0, set1]

    def _run(
        self,
        r: "renderer.Renderer",
        cmd: CommandBuffer,
        count: int,
        indirect_count_offset: Optional[int],
    ) -> None:
        # Count exceeds max
        if self.options.max_count < count:
            raise ValueError(
                f"count ({count}) must be smaller than max_count ({self.options.max_count}) specified on sorting pipeline creation"
            )

        set_index = 0
        sets = self.current_sets

        thread_blocks = div_round_up(count, self.tuning.partition_size)

        max_dispatch_size = r.ctx.device_properties.limits.max_compute_work_group_count[0]
        full_blocks = thread_blocks // max_dispatch_size
        partial_blocks = thread_blocks - full_blocks * max_dispatch_size

        self.constants["numKeys"] = count if indirect_count_offset is None else (indirect_count_offset >> 2)
        self.constants["threadBlocks"] = thread_blocks

        if self.options.unsafe_has_forward_thread_progress_guarantee:
            # Onesweep sort

            self.constants["radixShift"] = 0

            # Init
            cmd.bind_descriptor_sets(self.onesweep_pipeline.init, [sets[0]])
            cmd.bind_pipeline(self.onesweep_pipeline.init)
            cmd.push_constants(self.onesweep_pipeline.init, self.constants.tobytes())
            cmd.dispatch(256, 1, 1)

            dst_stage = PipelineStageFlags.COMPUTE_SHADER
            dst_access = AccessFlags.SHADER_READ | AccessFlags.SHADER_WRITE
            if indirect_count_offset is not None:
                dst_stage |= PipelineStageFlags.DRAW_INDIRECT
                dst_access |= AccessFlags.INDIRECT_COMMAND_READ
            cmd.memory_barrier_full(
                MemoryBarrier(
                    PipelineStageFlags.COMPUTE_SHADER,
                    AccessFlags.SHADER_READ | AccessFlags.SHADER_WRITE,
                    dst_stage,
                    dst_access,
                )
            )

            # Global hist
            cmd.bind_pipeline(self.onesweep_pipeline.global_histogram)

            if indirect_count_offset is not None:
                self.constants["isPartial"] = 0
                cmd.push_constants(self.onesweep_pipeline.global_histogram, self.constants.tobytes())
                cmd.dispatch_indirect(self.indirect_dispatch_buf, 0)

                self.constants["isPartial"] = 1
                cmd.push_constants(self.onesweep_pipeline.global_histogram, self.constants.tobytes())
                cmd.dispatch_indirect(self.indirect_dispatch_buf, 12)
            else:
                global_hist_thread_blocks = div_round_up(count, RADIX_SORT_GLOBAL_HIST_PARTITION_SIZE)
                global_hist_full_blocks = global_hist_thread_blocks // max_dispatch_size
                global_hist_partial_blocks = global_hist_thread_blocks - global_hist_full_blocks * max_dispatch_size
                self.constants["threadBlocks"] = global_hist_thread_blocks

                if full_blocks > 0:
                    self.constants["isPartial"] = 0
                    cmd.push_constants(self.onesweep_pipeline.global_histogram, self.constants.tobytes())
                    cmd.dispatch(max_dispatch_size, global_hist_full_blocks, 1)

                if partial_blocks > 0:
                    self.constants["isPartial"] = (full_blocks << 1) | 1
                    cmd.push_constants(self.onesweep_pipeline.global_histogram, self.constants.tobytes())
                    cmd.dispatch(global_hist_partial_blocks, 1, 1)

            cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

            # Scan
            cmd.bind_pipeline(self.onesweep_pipeline.scan)
            self.constants["threadBlocks"] = thread_blocks
            cmd.push_constants(self.onesweep_pipeline.scan, self.constants.tobytes())
            cmd.dispatch(RADIX_SORT_PASSES, 1, 1)
            cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

            # Sweep
            self.constants["isPartial"] = 0
            cmd.bind_pipeline(self.onesweep_pipeline.onesweep)
            for radix_shift in range(0, 32, 8):
                # Bind descriptor set (skip first because already bound)
                if radix_shift > 0:
                    cmd.bind_descriptor_sets(self.onesweep_pipeline.onesweep, [sets[set_index]])
                set_index = 1 - set_index

                self.constants["radixShift"] = radix_shift
                cmd.push_constants(self.onesweep_pipeline.onesweep, self.constants.tobytes())

                if indirect_count_offset is not None:
                    cmd.dispatch_indirect(self.indirect_dispatch_buf, 24)
                    cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)
                    cmd.dispatch_indirect(self.indirect_dispatch_buf, 36)
                else:
                    if full_blocks > 0:
                        cmd.dispatch(max_dispatch_size, full_blocks, 1)
                        # Need a memory barrier between passes because they are dependent
                        cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

                    if partial_blocks > 0:
                        cmd.dispatch(partial_blocks, 1, 1)

                # Skip barrier for last iteration
                if radix_shift < 24:
                    cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

        else:
            # Device radix sort

            # Init
            cmd.bind_descriptor_sets(self.device_radix_sort_pipeline.init, [sets[0]])
            cmd.push_constants(self.device_radix_sort_pipeline.init, self.constants.tobytes())
            cmd.bind_pipeline(self.device_radix_sort_pipeline.init)
            cmd.dispatch(1, 1, 1)

            dst_stage = PipelineStageFlags.COMPUTE_SHADER
            dst_access = AccessFlags.SHADER_READ | AccessFlags.SHADER_WRITE
            if indirect_count_offset is not None:
                dst_stage |= PipelineStageFlags.DRAW_INDIRECT
                dst_access |= AccessFlags.INDIRECT_COMMAND_READ
            cmd.memory_barrier_full(
                MemoryBarrier(
                    PipelineStageFlags.COMPUTE_SHADER,
                    AccessFlags.SHADER_READ | AccessFlags.SHADER_WRITE,
                    dst_stage,
                    dst_access,
                )
            )

            for radix_shift in range(0, 32, 8):
                # Bind descriptor set (skip first because already bound)
                if radix_shift > 0:
                    cmd.bind_descriptor_sets(self.device_radix_sort_pipeline.upsweep, [sets[set_index]])
                set_index = 1 - set_index

                # Upsweep
                self.constants["radixShift"] = radix_shift

                cmd.bind_pipeline(self.device_radix_sort_pipeline.upsweep)
                if indirect_count_offset is not None:
                    self.constants["isPartial"] = 0
                    cmd.push_constants(self.device_radix_sort_pipeline.upsweep, self.constants.tobytes())
                    cmd.dispatch_indirect(self.indirect_dispatch_buf, 0)

                    self.constants["isPartial"] = 1
                    cmd.push_constants(self.device_radix_sort_pipeline.upsweep, self.constants.tobytes())
                    cmd.dispatch_indirect(self.indirect_dispatch_buf, 12)
                else:
                    if full_blocks > 0:
                        self.constants["isPartial"] = 0
                        cmd.push_constants(self.device_radix_sort_pipeline.upsweep, self.constants.tobytes())
                        cmd.dispatch(max_dispatch_size, full_blocks, 1)

                    if partial_blocks > 0:
                        self.constants["isPartial"] = (full_blocks << 1) | 1
                        cmd.push_constants(self.device_radix_sort_pipeline.upsweep, self.constants.tobytes())
                        cmd.dispatch(partial_blocks, 1, 1)

                cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

                # Scan
                cmd.bind_pipeline(self.device_radix_sort_pipeline.scan)
                cmd.dispatch(256, 1, 1)

                cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

                # Downsweep
                cmd.bind_pipeline(self.device_radix_sort_pipeline.downsweep)
                if indirect_count_offset is not None:
                    self.constants["isPartial"] = 0
                    cmd.push_constants(self.device_radix_sort_pipeline.downsweep, self.constants.tobytes())
                    cmd.dispatch_indirect(self.indirect_dispatch_buf, 0)

                    self.constants["isPartial"] = 1
                    cmd.push_constants(self.device_radix_sort_pipeline.downsweep, self.constants.tobytes())
                    cmd.dispatch_indirect(self.indirect_dispatch_buf, 12)
                else:
                    if full_blocks > 0:
                        self.constants["isPartial"] = 0
                        cmd.push_constants(self.device_radix_sort_pipeline.downsweep, self.constants.tobytes())
                        cmd.dispatch(max_dispatch_size, full_blocks, 1)

                    if partial_blocks > 0:
                        self.constants["isPartial"] = (full_blocks << 1) | 1
                        cmd.push_constants(self.device_radix_sort_pipeline.downsweep, self.constants.tobytes())
                        cmd.dispatch(partial_blocks, 1, 1)

                # Skip barrier for last iteration
                if radix_shift < 24:
                    cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)

    def run(
        self,
        r: "renderer.Renderer",
        cmd: CommandBuffer,
        count: int,
    ) -> None:
        if self.options.indirect:
            raise RuntimeError("GpuSortingPipeline was created for indirect dispatch. Direct dispatch is not allowed.")

        # Nothing to do
        if count <= 1:
            return

        self._run(r, cmd, count, None)

    def run_indirect(
        self,
        r: "renderer.Renderer",
        cmd: CommandBuffer,
        count_offset: int,
    ) -> None:
        if not self.options.indirect:
            raise RuntimeError("GpuSortingPipeline was created for direct dispatch. Indirect dispatch is not allowed.")
        self._run(r, cmd, 0, count_offset)
