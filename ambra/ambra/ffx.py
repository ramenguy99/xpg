import numpy as np
from pyxpg import (
    AllocType,
    Buffer,
    BufferUsageFlags,
    ComputePipeline,
    DescriptorPoolCreateFlags,
    DescriptorSetBinding,
    DescriptorSetLayoutCreateFlags,
    DescriptorType,
    DeviceFeatures,
    Filter,
    Format,
    Image,
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

        # NOTE: using UPDATE_AFTER_BIND because that allows for a larger limit on total
        # number of bound storage images in MoltenVK. This is because MoltenVK switches
        # to using argument buffers when this flag is set. Not sure why it does not
        # expose the higher limit anyways when argument buffers are avialable.
        # Newer versions of MoltenVK seem to have increased this limit anyway.
        #
        # See: https://github.com/KhronosGroup/MoltenVK/issues/1610
        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_set = create_descriptor_layout_pool_and_set(
            r.ctx,
            [
                DescriptorSetBinding(1, DescriptorType.SAMPLER),
                DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),
                DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),
                DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
                DescriptorSetBinding(SPD_MAX_LEVELS + 1, DescriptorType.STORAGE_IMAGE),
            ],
            layout_flags=DescriptorSetLayoutCreateFlags.UPDATE_AFTER_BIND_POOL,
            pool_flags=DescriptorPoolCreateFlags.UPDATE_AFTER_BIND,
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

        self.atomic_counters = Buffer.from_data(
            r.ctx,
            view_bytes(np.zeros((6,), np.uint32)),
            BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name="ffx-spd-atomic-counters",
        )

        self.descriptor_set.write_sampler(self.linear_clamp_sampler, 0)
        self.descriptor_set.write_buffer(self.atomic_counters, DescriptorType.STORAGE_BUFFER, 1)

    def run(self, r: "renderer.Renderer", image: Image, new_layout: ImageLayout) -> None:
        # TODO: potentially good place to use descriptor update templates
        full_srgb_view = ImageView(
            r.ctx, image, ImageViewType.TYPE_2D_ARRAY, format=Format.R8G8B8A8_SRGB, usage_flags=ImageUsageFlags.SAMPLED
        )
        self.descriptor_set.write_image(full_srgb_view, ImageLayout.GENERAL, DescriptorType.SAMPLED_IMAGE, 2)

        views = []
        for m in range(SPD_MAX_LEVELS + 1):
            # Default to mip 0 if out of bounds
            view = ImageView(
                r.ctx,
                image,
                ImageViewType.TYPE_2D_ARRAY,
                base_mip_level=m if m < image.mip_levels else 0,
                mip_level_count=1,
            )
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

        # image.mip_levels includes level 0. The kernel expects just
        # the number of levels after 0.
        mips = min(image.mip_levels - 1, SPD_MAX_LEVELS)

        self.constants["mips"] = mips
        self.constants["numWorkGroups"] = groups
        self.constants["workGroupOffset"] = (group_offset_x, group_offset_y)
        self.constants["invInputSize"] = (1.0 / image.width, 1.0 / image.height)

        with r.ctx.sync_commands() as cmd:
            cmd.bind_compute_pipeline(
                self.avg_pipeline, descriptor_sets=[self.descriptor_set], push_constants=self.constants.tobytes()
            )
            cmd.image_barrier(image, ImageLayout.GENERAL, MemoryUsage.ALL, MemoryUsage.IMAGE)
            cmd.dispatch(groups_x, groups_y, image.depth)
            cmd.image_barrier(image, new_layout, MemoryUsage.IMAGE, MemoryUsage.ALL)
