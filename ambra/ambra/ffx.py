from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
from numpy.typing import NDArray
from pyxpg import (
    AllocType,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    ComputePipeline,
    DescriptorPool,
    DescriptorPoolCreateFlags,
    DescriptorSet,
    DescriptorSetBinding,
    DescriptorSetLayout,
    DescriptorSetLayoutCreateFlags,
    DescriptorType,
    DeviceFeatures,
    Filter,
    Format,
    Image,
    ImageBarrier,
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    ImageViewType,
    MemoryUsage,
    PushConstantsRange,
    Sampler,
    SamplerAddressMode,
    Shader,
    SubgroupFeatureFlags,
)

from .utils.descriptors import create_descriptor_pool_and_sets_ringbuffer
from .utils.gpu import MipGenerationFilter, view_bytes
from .utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from .renderer import Renderer

SPD_MAX_LEVELS = 12


class SPDPipelineInstance:
    def __init__(
        self,
        atomic_counters: Buffer,
        descriptor_pool: DescriptorPool,
        descriptor_sets: RingBuffer[DescriptorSet],
        constants: NDArray[Any],
    ):
        self.atomic_counters = atomic_counters
        self.descriptor_pool = descriptor_pool
        self.descriptor_sets = descriptor_sets
        self.constants = constants

        self.groups_x = 0
        self.groups_y = 0

    def set_image_extents(self, width: int, height: int, mips: int, x: int = 0, y: int = 0) -> None:
        rect_width = width
        rect_height = height

        group_offset_x = x // 64
        group_offset_y = y // 64

        group_end_x = (x + rect_width - 1) // 64
        group_end_y = (y + rect_height - 1) // 64

        groups_x = group_end_x + 1 - group_offset_x
        groups_y = group_end_y + 1 - group_offset_y

        groups = groups_x * groups_y

        # image.mip_levels includes level 0. The kernel expects just the number of levels after 0.
        mips_after_level_0 = min(mips - 1, SPD_MAX_LEVELS)

        self.constants["mips"] = mips_after_level_0
        self.constants["numWorkGroups"] = groups
        self.constants["workGroupOffset"] = (group_offset_x, group_offset_y)
        self.constants["invInputSize"] = (1.0 / width, 1.0 / height)

        self.groups_x = groups_x
        self.groups_y = groups_y

    def get_and_write_current_and_advance(
        self, level_0_view: Union[Image, ImageView], mip_views: List[ImageView]
    ) -> DescriptorSet:
        s = self.descriptor_sets.get_current_and_advance()
        s.write_image(level_0_view, ImageLayout.GENERAL, DescriptorType.SAMPLED_IMAGE, 2)
        for m in range(SPD_MAX_LEVELS + 1):
            view = mip_views[m] if m < len(mip_views) else mip_views[0]
            if m == 6:
                s.write_image(view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 3, 0)
            s.write_image(view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 4, m)
        return s


@dataclass
class MipGenerationRequest:
    image: Image
    push_constants: bytes
    descriptor_set: DescriptorSet
    groups_x: int
    groups_y: int
    groups_z: int
    after_barrier: ImageBarrier


class SPDPipeline:
    def __init__(self, r: "Renderer"):
        from .renderer import SHADERS_PATH  # noqa: PLC0415

        self.constants_dtype = np.dtype(
            {
                "mips": (np.uint32, 0),
                "numWorkGroups": (np.uint32, 4),
                "workGroupOffset": (np.dtype((np.uint32, 2)), 8),
                "invInputSize": (np.dtype((np.float32, 2)), 16),
            }
        )  # type: ignore

        # NOTE: using UPDATE_AFTER_BIND because that allows for a larger limit on total
        # number of bound storage images in MoltenVK. This is because MoltenVK switches
        # to using argument buffers when this flag is set. Not sure why it does not
        # expose the higher limit anyways when argument buffers are available.
        # Newer versions of MoltenVK seem to have increased this limit anyway.
        #
        # See: https://github.com/KhronosGroup/MoltenVK/issues/1610
        self.descriptor_set_layout = DescriptorSetLayout(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.SAMPLER),
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),
                DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
                DescriptorSetBinding(SPD_MAX_LEVELS + 1, DescriptorType.STORAGE_IMAGE),
            ],
            DescriptorSetLayoutCreateFlags.UPDATE_AFTER_BIND_POOL,
        )

        # Check hardware support for float16 and quad operations. Otherwise fallback to LDS.
        use_float16 = (r.ctx.device_features & DeviceFeatures.SHADER_FLOAT16_INT8) != 0
        use_wave_ops = (
            r.ctx.device_properties.subgroup_properties.supported_operations & SubgroupFeatureFlags.QUAD
        ) != 0 and (r.ctx.device_features & DeviceFeatures.SHADER_SUBGROUP_EXTENDED_TYPES) != 0

        defines = [
            ("FFX_GPU", ""),
            ("FFX_SLANG", ""),
            ("FFX_HLSL_SM", "62"),
            ("FFX_SPD_OPTION_LINEAR_SAMPLE", "1"),
            ("FFX_HALF", "1" if use_float16 else "0"),
            ("FFX_NO_16_BIT_CAST", ""),
            ("FFX_SPD_OPTION_WAVE_INTEROP_LDS", "0" if use_wave_ops else "1"),
        ]
        self.avg_shader = r.compile_builtin_shader(
            "ffx/ffx_spd_downsample_pass.slang",
            entry="CS",
            defines=[*defines, ("FFX_SPD_OPTION_DOWNSAMPLE_FILTER", "0")],
            include_paths=[SHADERS_PATH.joinpath("ffx")],
        )
        self.avg_srgb_shader = r.compile_builtin_shader(
            "ffx/ffx_spd_downsample_pass.slang",
            entry="CS",
            defines=[*defines, ("FFX_SPD_OPTION_DOWNSAMPLE_FILTER", "0"), ("FFX_SPD_SRGB", "1")],
            include_paths=[SHADERS_PATH.joinpath("ffx")],
        )
        self.min_shader = r.compile_builtin_shader(
            "ffx/ffx_spd_downsample_pass.slang",
            entry="CS",
            defines=[*defines, ("FFX_SPD_OPTION_DOWNSAMPLE_FILTER", "1")],
            include_paths=[SHADERS_PATH.joinpath("ffx")],
        )
        self.max_shader = r.compile_builtin_shader(
            "ffx/ffx_spd_downsample_pass.slang",
            entry="CS",
            defines=[*defines, ("FFX_SPD_OPTION_DOWNSAMPLE_FILTER", "2")],
            include_paths=[SHADERS_PATH.joinpath("ffx")],
        )

        self.avg_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.avg_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )
        self.avg_srgb_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.avg_srgb_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )
        self.min_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.min_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )
        self.max_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.max_shader.code),
            descriptor_set_layouts=[self.descriptor_set_layout],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

        self.linear_clamp_sampler = Sampler(
            r.ctx,
            min_filter=Filter.LINEAR,
            mag_filter=Filter.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_EDGE,
            v=SamplerAddressMode.CLAMP_TO_EDGE,
            w=SamplerAddressMode.CLAMP_TO_EDGE,
        )

        self.sync_instance = self.alloc_instance(r, single_set=True)

    def alloc_instance(self, r: "Renderer", single_set: bool = False) -> SPDPipelineInstance:
        constants = np.zeros((1,), self.constants_dtype)

        atomic_counters = Buffer.from_data(
            r.ctx,
            view_bytes(np.zeros((6,), np.uint32)),
            BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name="ffx-spd-atomic-counters",
        )

        number_of_sets = r.num_frames_in_flight if not single_set else 1
        pool, descriptor_sets = create_descriptor_pool_and_sets_ringbuffer(
            r.ctx, self.descriptor_set_layout, number_of_sets, DescriptorPoolCreateFlags.UPDATE_AFTER_BIND, "hello"
        )

        for s in descriptor_sets:
            s.write_sampler(self.linear_clamp_sampler, 0)
            s.write_buffer(atomic_counters, DescriptorType.STORAGE_BUFFER, 1)

        return SPDPipelineInstance(atomic_counters, pool, descriptor_sets, constants)

    def _filter_to_pipeline(self, filter: MipGenerationFilter) -> ComputePipeline:
        if filter == MipGenerationFilter.AVERAGE:
            return self.avg_pipeline
        elif filter == MipGenerationFilter.AVERAGE_SRGB:
            return self.avg_srgb_pipeline
        elif filter == MipGenerationFilter.MIN:
            return self.min_pipeline
        elif filter == MipGenerationFilter.MAX:
            return self.max_pipeline
        else:
            raise RuntimeError(f"Unhandled mipmap generation filter: {filter}")

    def run(
        self,
        cmd: CommandBuffer,
        image: Image,
        new_layout: ImageLayout,
        view_level_0: Union[Image, ImageView],
        mip_views: List[ImageView],
        instance: SPDPipelineInstance,
        filter: MipGenerationFilter,
    ) -> None:
        s = instance.descriptor_sets.get_current_and_advance()

        s.write_image(view_level_0, ImageLayout.GENERAL, DescriptorType.SAMPLED_IMAGE, 2)
        for m in range(SPD_MAX_LEVELS + 1):
            view = mip_views[m] if m < len(mip_views) else mip_views[0]
            if m == 6:
                s.write_image(view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 3, 0)
            s.write_image(view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 4, m)

        pipeline = self._filter_to_pipeline(filter)
        cmd.bind_compute_pipeline(pipeline, descriptor_sets=[s], push_constants=instance.constants.tobytes())

        cmd.image_barrier(image, ImageLayout.GENERAL, MemoryUsage.ALL, MemoryUsage.COMPUTE_SHADER)
        cmd.dispatch(instance.groups_x, instance.groups_y, image.depth)
        cmd.image_barrier(image, new_layout, MemoryUsage.COMPUTE_SHADER, MemoryUsage.ALL)

    def run_sync_with_views(
        self,
        r: "Renderer",
        image: Image,
        new_layout: ImageLayout,
        view_level_0: Union[Image, ImageView],
        mip_views: List[ImageView],
        filter: MipGenerationFilter,
    ) -> None:
        self.sync_instance.set_image_extents(image.width, image.height, image.mip_levels)
        with r.ctx.sync_commands() as cmd:
            self.run(cmd, image, new_layout, view_level_0, mip_views, self.sync_instance, filter)

    def run_sync(self, r: "Renderer", image: Image, new_layout: ImageLayout, filter: MipGenerationFilter) -> None:
        level_0_view = ImageView(
            r.ctx,
            image,
            ImageViewType.TYPE_2D_ARRAY,
            format=Format.R8G8B8A8_SRGB if filter == MipGenerationFilter.AVERAGE_SRGB else Format.R8G8B8A8_UNORM,
            usage_flags=ImageUsageFlags.SAMPLED,
        )
        views = [
            ImageView(
                r.ctx,
                image,
                ImageViewType.TYPE_2D_ARRAY,
                base_mip_level=m,
                mip_level_count=1,
            )
            for m in range(min(image.mip_levels, SPD_MAX_LEVELS + 1))
        ]
        self.run_sync_with_views(r, image, new_layout, level_0_view, views, filter)

    def run_batched(
        self, cmd: CommandBuffer, mip_generation_requests: Dict[MipGenerationFilter, List[MipGenerationRequest]]
    ) -> List[ImageBarrier]:
        barriers = []
        for filter, requests in mip_generation_requests.items():
            pipeline = self._filter_to_pipeline(filter)
            cmd.bind_pipeline(pipeline)
            for req in requests:
                cmd.bind_descriptor_sets(pipeline, descriptor_sets=[req.descriptor_set])
                cmd.push_constants(pipeline, req.push_constants)
                cmd.dispatch(req.groups_x, req.groups_y, req.groups_z)
                barriers.append(req.after_barrier)
        return barriers
