import numpy as np
from pyxpg import (
    AllocType,
    Buffer,
    Format,
    BufferUsageFlags,
    ComputePipeline,
    DescriptorSetBinding,
    DescriptorType,
    Filter,
    Image,
    ImageLayout,
    ImageView,
    ImageViewType,
    MemoryUsage,
    PushConstantsRange,
    Sampler,
    SamplerAddressMode,
    Shader,
    ImageUsageFlags,
)

from . import renderer
from .property import view_bytes
from .utils.descriptors import create_descriptor_layout_pool_and_set

SPD_MAX_LEVELS = 12


class SPDPipeline:
    def __init__(self, r: "renderer.Renderer"):
        self.constants_dtype = np.dtype(
            {
                "mips": (np.uint32, 0),
                "numWorkGroups": (np.uint32, 4),
                "workGroupOffset": (np.dtype((np.uint32, 2)), 8),
                "invInputSize": (np.dtype((np.float32, 2)), 16),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)

        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_set = create_descriptor_layout_pool_and_set(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.SAMPLER),
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),
                DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
                DescriptorSetBinding(SPD_MAX_LEVELS + 1, DescriptorType.STORAGE_IMAGE),
            ],
        )

        defines = [
            ("FFX_GPU", ""),
            ("FFX_SLANG", ""),
            ("FFX_HLSL_SM", "62"),
            ("FFX_SPD_OPTION_LINEAR_SAMPLE", "1"),
            # TODO: Enable if float supported. We can check for VK_KHR_shader_float16_int8
            # and store somewhere (flot16 or uint8) what is supported if the feature is requested.
            # All hardware we target should have boths
            ("FFX_HALF", "0"),
            ("FFX_NO_16_BIT_CAST", ""),
            # TODO: Disable if wave quad ops are supported (set it to 0).
            # need to bubble up this info from xpg with VkPhysicalDeviceSubgroupProperties
            # This kernel requires Quad operations in compute, all hardware we target should have this.
            ("FFX_SPD_OPTION_WAVE_INTEROP_LDS", "1"),
        ]
        self.avg_shader = r.compile_builtin_shader(
            "ffx/ffx_spd_downsample_pass.slang",
            entry="CS",
            defines=[*defines, ("FFX_SPD_OPTION_DOWNSAMPLE_FILTER", "0")],
            include_paths=[renderer.SHADERS_PATH.joinpath("ffx")],
        )
        self.min_shader = r.compile_builtin_shader(
            "ffx/ffx_spd_downsample_pass.slang",
            entry="CS",
            defines=[*defines, ("FFX_SPD_OPTION_DOWNSAMPLE_FILTER", "1")],
            include_paths=[renderer.SHADERS_PATH.joinpath("ffx")],
        )
        self.max_shader = r.compile_builtin_shader(
            "ffx/ffx_spd_downsample_pass.slang",
            entry="CS",
            defines=[*defines, ("FFX_SPD_OPTION_DOWNSAMPLE_FILTER", "2")],
            include_paths=[renderer.SHADERS_PATH.joinpath("ffx")],
        )

        self.avg_pipeline = ComputePipeline(
            r.ctx,
            Shader(r.ctx, self.avg_shader.code),
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

        self.atomic_counters = Buffer.from_data(r.ctx, view_bytes(np.zeros((6,), np.uint32)), BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE, name="ffx-spd-atomic-counters")

        self.descriptor_set.write_sampler(self.linear_clamp_sampler, 0)
        self.descriptor_set.write_buffer(self.atomic_counters, DescriptorType.STORAGE_BUFFER, 1)

    def run(self, r: "renderer.Renderer", image: Image, new_layout: ImageLayout):
        full_srgb_view = ImageView(r.ctx, image, ImageViewType.TYPE_2D_ARRAY, format=Format.R8G8B8A8_SRGB, usage_flags=ImageUsageFlags.SAMPLED)
        self.descriptor_set.write_image(full_srgb_view, ImageLayout.GENERAL, DescriptorType.SAMPLED_IMAGE, 2)

        views = []

        for m in range(SPD_MAX_LEVELS + 1):
            # Default to mip 0 if out of bounds
            view = ImageView(r.ctx, image, ImageViewType.TYPE_2D_ARRAY, base_mip_level=m if m < image.mip_levels else 0, mip_level_count=1)
            if m == 6:
                self.descriptor_set.write_image(view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 3, 0)
            self.descriptor_set.write_image(view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 4, m)
            views.append(view)

        rect_x = 0
        rect_y = 0
        rect_width = image.width
        rect_height = image.height

        group_offset_x = rect_x // 64
        group_offset_y = rect_y // 64

        group_end_x = (rect_x + rect_width - 1) // 64
        group_end_y = (rect_y + rect_height - 1) // 64

        groups_x = group_end_x + 1 - group_offset_x
        groups_y = group_end_y + 1 - group_offset_y

        groups = groups_x * groups_y
        mips = min(image.mip_levels - 1, SPD_MAX_LEVELS)

        self.constants["mips"] = mips
        self.constants["numWorkGroups"] = groups
        self.constants["workGroupOffset"] = (group_offset_x, group_offset_y)
        self.constants["invInputSize"] = (1.0 / image.width, 1.0 / image.height)

        with r.ctx.sync_commands() as cmd:
            cmd.bind_compute_pipeline(self.avg_pipeline, descriptor_sets=[ self.descriptor_set ], push_constants=self.constants.tobytes())

            # TODO: we need two barriers here because the other mips were not transitioned to the right layout on creation.
            # We should rethink how this is supposed to work for .from_data, but especially for GpuImageProperty.
            cmd.image_barrier(image, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, mip_level_count=1)
            cmd.image_barrier(image, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, base_mip_level=0, undefined=True)

            cmd.dispatch(groups_x, groups_y, image.depth)
            cmd.image_barrier(image, new_layout, MemoryUsage.IMAGE, MemoryUsage.ALL)
