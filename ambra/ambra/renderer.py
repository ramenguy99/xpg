# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from pyxpg import (
    AccessFlags,
    AllocType,
    BorderColor,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    CompareOp,
    Context,
    DepthAttachment,
    DescriptorSetBinding,
    DescriptorSetLayout,
    DescriptorType,
    DeviceFeatures,
    Filter,
    Format,
    Gui,
    Image,
    ImageAspectFlags,
    ImageBarrier,
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    LoadOp,
    MemoryBarrier,
    MemoryUsage,
    PhysicalDeviceType,
    PipelineStageFlags,
    RenderingAttachment,
    ResolveMode,
    Sampler,
    SamplerAddressMode,
    SamplerMipmapMode,
    StoreOp,
    slang,
)

from .config import RendererConfig, UploadMethod
from .ffx import SPDPipeline
from .lights import (
    LIGHT_TYPES_INFO,
    DirectionalLight,
    EnvironmentLight,
    GGXLUTPipeline,
    GpuEnvironmentCubemaps,
    IBLParams,
    IBLPipeline,
    Light,
    LightTypes,
    UniformEnvironmentLight,
)
from .renderer_frame import RendererFrame, SemaphoreInfo
from .scene import Object, Scene
from .shaders import compile
from .utils.gpu import (
    BufferUploadInfo,
    BulkUploader,
    ImageUploadInfo,
    UniformPool,
)
from .utils.threadpool import ThreadPool
from .viewport import Viewport

if TYPE_CHECKING:
    from .gpu_property import GpuProperty

SHADERS_PATH = Path(__file__).parent.joinpath("shaders")

MAX_WORKERS = 16
DEFAULT_WORKERS = 4
if sys.version_info >= (3, 13):

    def get_cpu_count() -> Optional[int]:
        return os.process_cpu_count()
else:

    def get_cpu_count() -> Optional[int]:
        return os.cpu_count()


@dataclass
class FrameInputs:
    image: Image
    command_buffer: CommandBuffer
    transfer_command_buffer: Optional[CommandBuffer]
    additional_semaphores: List[SemaphoreInfo]
    transfer_semaphores: List[SemaphoreInfo]


class Renderer:
    def __init__(
        self,
        ctx: Context,
        width: int,
        height: int,
        num_frames_in_flight: int,
        output_format: Format,
        multiviewport: bool,
        config: RendererConfig,
    ):
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.output_format = output_format
        self.multiviewport = multiviewport

        self.shadow_map_format = Format.D32_SFLOAT

        # Config
        self.background_color = config.background_color

        # Scene descriptors
        self.max_shadow_maps = config.max_shadow_maps

        self.linear_sampler = Sampler(
            ctx,
            Filter.LINEAR,
            Filter.LINEAR,
            SamplerMipmapMode.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_EDGE,
            v=SamplerAddressMode.CLAMP_TO_EDGE,
            name="linear-sampler",
        )
        self.shadow_sampler = Sampler(
            ctx,
            Filter.LINEAR,
            Filter.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_BORDER,
            v=SamplerAddressMode.CLAMP_TO_BORDER,
            compare_enable=True,
            compare_op=CompareOp.LESS,
            border_color=BorderColor.FLOAT_OPAQUE_WHITE,
            name="shadow-sampler",
        )

        self.zero_buffer = Buffer.from_data(
            ctx,
            np.zeros((4,), np.uint8),
            BufferUsageFlags.STORAGE | BufferUsageFlags.UNIFORM | BufferUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name="zero-buffer",
        )
        self.zero_image = Image.from_data(
            ctx,
            np.zeros((1, 1, 4), np.uint8),
            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
            1,
            1,
            Format.R8G8B8A8_UNORM,
            ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            name="zero-image",
        )
        self.zero_cubemap = Image(
            self.ctx,
            1,
            1,
            Format.R16G16B16A16_SFLOAT,
            ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE,
            array_layers=6,
            is_cube=True,
        )
        with ctx.sync_commands() as cmd:
            cmd.image_barrier(
                self.zero_cubemap,
                ImageLayout.TRANSFER_DST_OPTIMAL,
                MemoryUsage.NONE,
                MemoryUsage.TRANSFER_DST,
                undefined=True,
            )
            cmd.clear_color_image(self.zero_cubemap, (0, 0, 0, 1))
            cmd.image_barrier(
                self.zero_cubemap,
                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                MemoryUsage.TRANSFER_DST,
                MemoryUsage.SHADER_READ_ONLY,
            )

        # Maps from the index of the frame starting from which it's safe to destroy objects to the list of objets to destroy.
        self.destruction_queue: Dict[int, List[Union[Buffer, Image, ImageView]]] = {}
        self.shader_cache: Dict[Tuple[Union[str, Tuple[str, str], Path], ...], slang.Shader] = {}

        self.scene_descriptor_set_layout = DescriptorSetLayout(
            ctx,
            [
                DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),  # 0 - Constants
                DescriptorSetBinding(1, DescriptorType.SAMPLER),  # 1 - shadow_map sampler
                DescriptorSetBinding(1, DescriptorType.SAMPLER),  # 2 - Cubemap sampler
                DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),  # 3 - Environment irradiance cubemap
                DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),  # 4 - Environment specular cubemap
                DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),  # 5 - Environment GGX LUT
                *[DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER) for _ in LIGHT_TYPES_INFO],
                DescriptorSetBinding(self.max_shadow_maps, DescriptorType.SAMPLED_IMAGE),
            ],
            name="scene-descriptor-layout",
        )

        self.scene_depth_descriptor_set_layout = DescriptorSetLayout(
            ctx,
            [
                DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),
            ],
            name="scene-depth-descriptor-set-layout",
        )

        self.scene_constants_dtype = np.dtype(
            {
                "view": (np.dtype((np.float32, (4, 4))), 0),
                "projection": (np.dtype((np.float32, (4, 4))), 64),
                "world_camera_position": (np.dtype((np.float32, (3,))), 128),
                "ambient_light": (np.dtype((np.float32, (3,))), 144),
                "has_environment_light": (np.uint32, 156),
                "focal": (np.dtype((np.float32, (2,))), 160),
                "inverse_viewport_size": (np.dtype((np.float32, (2,))), 168),
                "max_specular_mip": (np.float32, 176),
                "num_lights": (np.dtype((np.uint32, (len(LIGHT_TYPES_INFO),))), 180),
            }
        )  # type: ignore

        self.max_lights_per_type = config.max_lights_per_type

        self.spd_pipeline = SPDPipeline(self)
        self.spd_pipeline_instances = [
            self.spd_pipeline.alloc_instance(self, True) for _ in range(config.mip_generation_batch_size)
        ]

        self.ibl_default_params = IBLParams()
        self.ibl_pipeline: Optional[IBLPipeline] = None
        self.ggx_lut_pipeline = GGXLUTPipeline(self)
        self.ggx_lut = self.ggx_lut_pipeline.run(self)

        self.uniform_pool = UniformPool(ctx, self.num_frames_in_flight, config.uniform_pool_block_size)

        if config.thread_pool_workers is not None:
            self.num_workers = config.thread_pool_workers
        else:
            num_cpus = get_cpu_count()
            if num_cpus is not None:
                self.num_workers = min(num_cpus, MAX_WORKERS)
            else:
                self.num_workers = DEFAULT_WORKERS

        self.thread_pool: ThreadPool = ThreadPool(self.num_workers)
        self.total_frame_index = 0

        # Upload method for buffers
        if config.force_buffer_upload_method is not None:
            self.buffer_upload_method = config.force_buffer_upload_method
            if self.buffer_upload_method == UploadMethod.TRANSFER_QUEUE:
                if not ctx.has_transfer_queue:
                    raise RuntimeError("Transfer queue not available on picked device")
                if not ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES:
                    raise RuntimeError(
                        "Transfer queue upload requires timeline semaphores which are not available on picked device"
                    )
        else:
            # TODO: we could also check if BAR is available and large enough
            # and add a DEVICE_PREFER_MAPPED option to take advantage of it.
            if (
                ctx.device_properties.device_type == PhysicalDeviceType.INTEGRATED_GPU
                or ctx.device_properties.device_type == PhysicalDeviceType.CPU
            ):
                self.buffer_upload_method = UploadMethod.MAPPED_PREFER_DEVICE
            elif (
                config.use_transfer_queue_if_available
                and ctx.has_transfer_queue
                and ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES
            ):
                self.buffer_upload_method = UploadMethod.TRANSFER_QUEUE
            else:
                self.buffer_upload_method = UploadMethod.GRAPHICS_QUEUE

        # Upload method for images
        if config.force_image_upload_method is not None:
            if (
                config.force_image_upload_method == UploadMethod.MAPPED_PREFER_DEVICE
                or config.force_image_upload_method == UploadMethod.MAPPED_PREFER_HOST
            ):
                raise RuntimeError(
                    f"Upload method for images must be {UploadMethod.GRAPHICS_QUEUE} or {UploadMethod.TRANSFER_QUEUE}. Got {config.force_image_upload_method} in config.force_image_upload_method."
                )
            self.image_upload_method = config.force_image_upload_method
            if self.image_upload_method == UploadMethod.TRANSFER_QUEUE:
                if not ctx.has_transfer_queue:
                    raise RuntimeError("Transfer queue not available on picked device")
                if not ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES:
                    raise RuntimeError(
                        "Transfer queue upload requires timeline semaphores which are not available on picked device"
                    )
        else:
            if (
                config.use_transfer_queue_if_available
                and ctx.has_transfer_queue
                and ctx.device_features & DeviceFeatures.TIMELINE_SEMAPHORES
            ):
                self.image_upload_method = UploadMethod.TRANSFER_QUEUE
            else:
                self.image_upload_method = UploadMethod.GRAPHICS_QUEUE

        # Allocate buffers for bulk upload
        self.bulk_uploader = BulkUploader(self.ctx, config.upload_buffer_size, config.upload_buffer_count)
        self.bulk_upload_list: List[Union[BufferUploadInfo, ImageUploadInfo]] = []

        # Enabled gpu properties cache for prefetch after frame submission
        self.enabled_gpu_properties: Set[GpuProperty[Any]] = set()

        # MSAA
        # TODO:
        # - proper thing to do would be to check max supported count for the color
        #   and depth format in use.
        # - additionally we could use StoreOP.DONT_CARE, TRANSIENT_ATTACHMENT and LAZILY_ALLOCATED memory
        #   for best performance on tilers.
        # - if at some point we want to expose multiple MSAA passes we can add a way to disable it.
        # - For retrieving color/depth we can do it by returning the resolved framebuffer (allocating one more for depth).
        self.msaa_samples = max(1, config.msaa_samples)
        self.msaa_target: Optional[Image] = None

        # Will be populated in self.resize(), and will never be None after that
        self.depth_buffer: Image = None  # type: ignore
        self.depth_format = Format.D32_SFLOAT
        self.depth_clear_value = 1.0
        self.depth_compare_op = CompareOp.LESS
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        if self.depth_buffer is not None:
            self.depth_buffer.destroy()
        self.depth_buffer = Image(
            self.ctx,
            width,
            height,
            self.depth_format,
            ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags.TRANSFER_DST,
            AllocType.DEVICE_DEDICATED,
            samples=self.msaa_samples,
            name="depth",
        )
        if self.msaa_samples > 1:
            if self.msaa_target is not None:
                self.msaa_target.destroy()
            self.msaa_target = Image(
                self.ctx,
                width,
                height,
                self.output_format,
                ImageUsageFlags.COLOR_ATTACHMENT,
                AllocType.DEVICE_DEDICATED,
                samples=self.msaa_samples,
            )

    def run_ibl_pipeline(
        self, equirectangular: Image, ibl_params: Optional["IBLParams"] = None
    ) -> "GpuEnvironmentCubemaps":
        if self.ibl_pipeline is None:
            self.ibl_pipeline = IBLPipeline(self)
        if ibl_params is None:
            ibl_params = self.ibl_default_params
        return self.ibl_pipeline.run(self, equirectangular, ibl_params)

    def compile_shader(
        self,
        path: Union[Path, str],
        entry: str = "main",
        target: str = "spirv_1_3",
        defines: Optional[List[Tuple[str, str]]] = None,
        include_paths: Optional[List[Union[Path, str]]] = None,
    ) -> slang.Shader:
        defines = defines or []
        include_paths = include_paths or []
        key = (path, entry, target, *defines, *include_paths)
        if s := self.shader_cache.get(key):
            return s
        shader = compile(Path(path), entry, target, defines=defines, include_paths=include_paths)
        self.shader_cache[key] = shader
        return shader

    def compile_builtin_shader(
        self,
        path: Union[Path, str],
        entry: str = "main",
        target: str = "spirv_1_3",
        defines: Optional[List[Tuple[str, str]]] = None,
        include_paths: Optional[List[Union[Path, str]]] = None,
    ) -> slang.Shader:
        return self.compile_shader(SHADERS_PATH.joinpath(SHADERS_PATH, path), entry, target, defines, include_paths)

    def render(self, scene: Scene, viewports: List[Viewport], frame: FrameInputs, gui: Gui) -> None:
        # Stage: destroy resources enqueued for destruction for this or an earlier frame
        if self.destruction_queue:
            for index, resources in list(self.destruction_queue.items()):
                if index <= self.total_frame_index:
                    for resource in resources:
                        resource.destroy()
                del self.destruction_queue[index]

        # Stage: create new objects
        enabled_objects: List[Object] = []
        enabled_lights: List[Light] = []
        enabled_gpu_properties: Set[GpuProperty[Any]] = set()

        def visit(o: Object) -> None:
            # Filter objects that are hidden in the GUI or that are not enabled this frame
            if o.gui_enabled and o.enabled.get_current():
                properties_enabled = True
                o.create_if_needed(self)

                if o.material is not None:
                    for mp, _ in o.material.properties:
                        if mp.property.current_animation_enabled:
                            mp.property.create(self)
                            if mp.property.gpu_property is not None:
                                enabled_gpu_properties.add(mp.property.gpu_property)
                        else:
                            properties_enabled = False
                for p in o.properties:
                    if p.current_animation_enabled:
                        p.create(self)
                        if p.gpu_property is not None:
                            enabled_gpu_properties.add(p.gpu_property)
                    else:
                        properties_enabled = False

                # Filter objects that have any property disabled for this frame
                if properties_enabled:
                    if isinstance(o, Light):
                        enabled_lights.append(o)
                    enabled_objects.append(o)

        scene.visit_objects(visit)

        # Stage: synchronous buffer upload after creating new objects
        if len(self.bulk_upload_list) > 0:
            mip_generation_requests: List[ImageUploadInfo] = []
            self.bulk_uploader.bulk_upload(self.bulk_upload_list, mip_generation_requests)
            self.bulk_upload_list.clear()

            # Process mip generation requests in batches.
            # The reason why this happens with sync command batches is that we
            # don't know upfront how many mips we have to create and we are
            # not allocating a pipeline instance (buffers and descriptors) to
            # run each of these generations independently.
            batch_size = len(self.spd_pipeline_instances)
            for i in range(0, len(mip_generation_requests), batch_size):
                with self.ctx.sync_commands() as cmd:
                    for j, instance in enumerate(self.spd_pipeline_instances):
                        if i + j >= len(mip_generation_requests):
                            break
                        req = mip_generation_requests[i + j]
                        assert req.level_0_view is not None

                        instance.set_image_extents(req.image.width, req.image.height, req.image.mip_levels)
                        self.spd_pipeline.run(
                            cmd,
                            req.image,
                            req.layout,
                            req.level_0_view,
                            req.mip_views,
                            instance,
                            req.mip_generation_filter,
                        )

        cmd = frame.command_buffer

        f = RendererFrame(
            # Frame counters
            self.total_frame_index % self.num_frames_in_flight,
            self.total_frame_index,
            # Graphics command list
            cmd,
            # Upload stages and barriers
            PipelineStageFlags(0),
            [],
            [],
            [],
            [],
            # Mip Generation requests
            {},
            # Before render
            PipelineStageFlags(0),
            PipelineStageFlags(0),
            [],
            # Between viewport render
            PipelineStageFlags(0),
            PipelineStageFlags(0),
            # Frame additional semaphores
            frame.additional_semaphores,
            # Transfer queue commands
            frame.transfer_command_buffer,
            # Transfer queue semaphores
            frame.transfer_semaphores,
            # Transfer queue barriers
            [],
            [],
            [],
            [],
        )

        # Stage: Load properties in CPU memory and prepare upload_before barriers
        for p in enabled_gpu_properties:
            p.load(f)

        # Sync: before-upload barriers
        cmd.barriers(
            memory_barriers=[
                MemoryBarrier(
                    PipelineStageFlags.ALL_COMMANDS,  # Would need to store the last stage as well
                    AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                    PipelineStageFlags.TRANSFER,
                    AccessFlags.TRANSFER_READ | AccessFlags.TRANSFER_WRITE,
                ),
            ],
            buffer_barriers=f.upload_before_buffer_barriers,
            image_barriers=f.upload_before_image_barriers,
        )

        # Sync: before-upload transfer queue barriers (layout transitions)
        if frame.transfer_semaphores:
            assert frame.transfer_command_buffer is not None
            frame.transfer_command_buffer.barriers(
                memory_barriers=[
                    MemoryBarrier(
                        PipelineStageFlags.ALL_COMMANDS,
                        AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                        PipelineStageFlags.TRANSFER,
                        AccessFlags.TRANSFER_READ | AccessFlags.TRANSFER_WRITE,
                    )
                ],
                buffer_barriers=f.transfer_upload_before_buffer_barriers,
                image_barriers=f.transfer_upload_before_image_barriers,
            )

        # Stage: upload all properties, objects and materials and colect upload_after barriers
        for p in enabled_gpu_properties:
            p.upload(f)

        # Upload scene constants for each viewport
        for viewport_index, viewport in enumerate(viewports):
            viewport_width, viewport_height = viewport.rect.width, viewport.rect.height

            if viewport_width <= 0 or viewport_height <= 0:
                continue

            environment_light: Optional[EnvironmentLight] = None
            uniform_environment_radiance = np.array((0, 0, 0), np.float32)
            light_buffer_offset = 0
            shadow_map_index = 0
            num_lights = 0
            light_buffers = viewport.scene_light_buffers.get_current_and_advance()
            for l in enabled_lights:
                if l.viewport_mask is None or l.viewport_mask & (1 << viewport_index):
                    if isinstance(l, DirectionalLight):
                        if num_lights >= self.max_lights_per_type:
                            raise RuntimeError(
                                f'Too many ligths of type: {LightTypes.DIRECTIONAL}. Increase "config.renderer.max_lights_per_type" (current value: {self.max_lights_per_type}) to allow more lights.'
                            )

                        map_index = -1
                        if l.shadow_map is not None:
                            map_index = shadow_map_index
                            shadow_map_index += 1

                            if map_index >= self.max_shadow_maps:
                                raise RuntimeError(
                                    f'Too many shadow_maps. Increase "config.renderer.max_shadow_maps" (current value: {self.max_shadow_maps}) to allow more shadow_maps.'
                                )

                        light_type_idx = LightTypes.DIRECTIONAL.value
                        offset = light_buffer_offset

                        l.upload_light(self, f, map_index, offset, light_buffers[light_type_idx])
                        light_buffer_offset += LIGHT_TYPES_INFO[light_type_idx].size
                        num_lights += 1
                    elif isinstance(l, EnvironmentLight):
                        if environment_light is not None:
                            raise RuntimeError(
                                f"More than one environment light enabled for viewport index {viewport_index}"
                            )
                        environment_light = l
                    elif isinstance(l, UniformEnvironmentLight):
                        uniform_environment_radiance += l.radiance.get_current()

            buf = viewport.scene_uniform_buffers.get_current_and_advance()
            proj = viewport.camera.projection(viewport_width / viewport_height)

            constants = viewport.scene_constants
            constants["projection"] = proj
            constants["view"] = viewport.camera.view()
            constants["world_camera_position"] = viewport.camera.position()
            constants["num_lights"] = num_lights
            constants["ambient_light"] = uniform_environment_radiance
            if environment_light is not None:
                constants["has_environment_light"] = 1
                constants["max_specular_mip"] = environment_light.gpu_cubemaps.specular_cubemap.mip_levels - 1.0
            else:
                constants["has_environment_light"] = 0
                constants["max_specular_mip"] = 0.0
            constants["inverse_viewport_size"] = (1.0 / viewport_width, 1.0 / viewport_height)
            constants["focal"] = (proj[0][0] * 0.5 * viewport_width, proj[1][1] * 0.5 * viewport_height)

            buf.upload(
                cmd,
                MemoryUsage.NONE,  # None because will be synced by after-upload barriers
                constants.view(np.uint8),
            )

        self.enabled_gpu_properties = enabled_gpu_properties

        for o in enabled_objects:
            if o.material is not None:
                o.material.upload(self, f)
            o.upload(self, f)

        # Sync: after-upload transfer queue barriers (queue release)
        if frame.transfer_semaphores:
            assert frame.transfer_command_buffer is not None
            frame.transfer_command_buffer.barriers(
                buffer_barriers=f.transfer_upload_after_buffer_barriers,
                image_barriers=f.transfer_upload_after_image_barriers,
            )

        # Stage: in flight mip generation
        if f.mip_generation_requests:
            # Sync: after-upload image barriers before mip generation
            cmd.barriers(image_barriers=f.upload_after_image_barriers)

            # Batched mip creation.
            #
            # Assumes that every streaming mip object has a spd pipeline instance
            # preallocated, so that we can run all of them in the same submit.
            # This is the path that streaming and dynamic properties will take when
            # they need on the fly mip generation that does not stall.
            # Streaming properties will preallocate an spd pipeline instance while
            # preuploaded properties will allocate one when promoted to dynamic.
            mip_after_image_barriers = self.spd_pipeline.run_batched(f.cmd, f.mip_generation_requests)

            # Sync: after-upload barriers and post mip generation barriers
            cmd.barriers(
                memory_barriers=[
                    MemoryBarrier(
                        PipelineStageFlags.TRANSFER,
                        AccessFlags.TRANSFER_READ | AccessFlags.TRANSFER_WRITE,
                        f.upload_property_pipeline_stages,
                        AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                    ),
                ],
                buffer_barriers=f.upload_after_buffer_barriers,
                image_barriers=mip_after_image_barriers,
            )
        else:
            # Sync: after-upload barriers
            cmd.barriers(
                memory_barriers=[
                    MemoryBarrier(
                        PipelineStageFlags.TRANSFER,
                        AccessFlags.TRANSFER_READ | AccessFlags.TRANSFER_WRITE,
                        f.upload_property_pipeline_stages,
                        AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                    ),
                ],
                buffer_barriers=f.upload_after_buffer_barriers,
                image_barriers=f.upload_after_image_barriers,
            )

        # Stage: Render shadows
        for l in enabled_lights:
            l.render_shadow_maps(self, f, enabled_objects)

        f.before_render_image_barriers.extend(
            [
                ImageBarrier(
                    frame.image,
                    ImageLayout.UNDEFINED,
                    ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                    PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                    AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                    PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                    AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                ),
                ImageBarrier(
                    self.depth_buffer,
                    ImageLayout.UNDEFINED,
                    ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    PipelineStageFlags.LATE_FRAGMENT_TESTS,
                    AccessFlags.DEPTH_STENCIL_ATTACHMENT_READ | AccessFlags.DEPTH_STENCIL_ATTACHMENT_WRITE,
                    PipelineStageFlags.EARLY_FRAGMENT_TESTS,
                    AccessFlags.DEPTH_STENCIL_ATTACHMENT_READ | AccessFlags.DEPTH_STENCIL_ATTACHMENT_WRITE,
                    aspect_mask=ImageAspectFlags.DEPTH,
                ),
            ]
        )

        before_gui_barriers: List[ImageBarrier] = []
        if self.multiviewport:
            for v in viewports:
                assert v.image is not None
                f.before_render_image_barriers.append(
                    ImageBarrier(
                        v.image,
                        ImageLayout.UNDEFINED,
                        ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                        PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                        AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                        PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                        AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                    )
                )
                before_gui_barriers.append(
                    ImageBarrier(
                        v.image,
                        ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                        ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                        PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                        AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                        PipelineStageFlags.FRAGMENT_SHADER,
                        AccessFlags.SHADER_SAMPLED_READ | AccessFlags.SHADER_SAMPLED_READ,
                    )
                )

        if self.msaa_target is not None:
            f.before_render_image_barriers.append(
                ImageBarrier(
                    self.msaa_target,
                    ImageLayout.UNDEFINED,
                    ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                    PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                    AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                    PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                    AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                )
            )

        # Sync: before-render barriers
        cmd.barriers(
            memory_barriers=[
                MemoryBarrier(
                    f.before_render_src_pipeline_stages,
                    AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                    f.before_render_dst_pipeline_stages,
                    AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                ),
            ],
            image_barriers=f.before_render_image_barriers,
        )

        for viewport_index, viewport in enumerate(viewports):
            if viewport.rect.width <= 0 or viewport.rect.height <= 0:
                continue
            viewport_bit = 1 << viewport_index

            descriptor_set = viewport.scene_descriptor_sets.get_current_and_advance()

            shadow_map_index = 0
            for l in enabled_lights:
                if l.viewport_mask is None or l.viewport_mask & (1 << viewport_index):
                    if isinstance(l, DirectionalLight) and l.shadow_map is not None:
                        descriptor_set.write_image(
                            l.shadow_map,
                            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                            DescriptorType.SAMPLED_IMAGE,
                            6 + len(LIGHT_TYPES_INFO),
                            shadow_map_index,
                        )
                        shadow_map_index += 1
                    elif isinstance(l, EnvironmentLight):
                        descriptor_set.write_image(
                            l.gpu_cubemaps.irradiance_cubemap,
                            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                            DescriptorType.SAMPLED_IMAGE,
                            3,
                        )
                        descriptor_set.write_image(
                            l.gpu_cubemaps.specular_cubemap,
                            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                            DescriptorType.SAMPLED_IMAGE,
                            4,
                        )

            if viewport_index > 0 and f.between_viewport_render_src_pipeline_stages:
                cmd.barriers(
                    memory_barriers=[
                        MemoryBarrier(
                            f.between_viewport_render_src_pipeline_stages,
                            AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                            f.between_viewport_render_dst_pipeline_stages,
                            AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                        ),
                    ],
                )

            # Stage: pre-render
            for o in enabled_objects:
                if o.viewport_mask is None or (o.viewport_mask & viewport_bit):
                    o.pre_render(self, f, viewport, descriptor_set)

            # Sync: before-render barriers
            if f.before_render_src_pipeline_stages:
                cmd.barriers(
                    memory_barriers=[
                        MemoryBarrier(
                            f.before_render_src_pipeline_stages,
                            AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                            f.before_render_dst_pipeline_stages,
                            AccessFlags.MEMORY_READ | AccessFlags.MEMORY_WRITE,
                        ),
                    ],
                )

            viewport_rect = (
                0,
                viewport.rect.height,
                viewport.rect.width,
                -viewport.rect.height,
            )
            rect = (
                0,
                0,
                viewport.rect.width,
                viewport.rect.height,
            )

            # Stage: render
            cmd.set_viewport(viewport_rect)
            cmd.set_scissors(rect)

            image = frame.image if viewport.image is None else viewport.image
            with cmd.rendering(
                rect,
                color_attachments=[
                    (
                        RenderingAttachment(
                            self.msaa_target,
                            load_op=LoadOp.CLEAR,
                            store_op=StoreOp.STORE,
                            clear=self.background_color,
                            resolve_mode=ResolveMode.AVERAGE,
                            resolve_image=image,
                        )
                        if self.msaa_target is not None
                        else RenderingAttachment(image, LoadOp.CLEAR, StoreOp.STORE, self.background_color)
                    )
                ],
                depth=DepthAttachment(self.depth_buffer, LoadOp.CLEAR, StoreOp.STORE, self.depth_clear_value),
            ):
                for o in enabled_objects:
                    if o.viewport_mask is None or (o.viewport_mask & viewport_bit):
                        o.render(self, f, viewport, descriptor_set)

        if before_gui_barriers:
            cmd.barriers(image_barriers=before_gui_barriers)

        # Stage: Render GUI
        with cmd.rendering(
            (0, 0, frame.image.width, frame.image.height),
            color_attachments=[
                RenderingAttachment(
                    frame.image,
                    load_op=LoadOp.CLEAR if self.multiviewport else LoadOp.LOAD,
                    store_op=StoreOp.STORE,
                    clear=self.background_color,
                ),
            ],
        ):
            gui.render(cmd)

        # Sync: after render barriers
        cmd.image_barrier_full(
            ImageBarrier(
                frame.image,
                ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                ImageLayout.PRESENT_SRC,
                PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                AccessFlags.NONE,
            )
        )

        self.uniform_pool.advance()
        self.total_frame_index += 1

    def prefetch(self) -> None:
        for p in self.enabled_gpu_properties:
            p.prefetch()

    def enqueue_for_destruction(self, resources: Iterable[Union[Buffer, Image, ImageView]]) -> None:
        self.destruction_queue.setdefault(self.total_frame_index + self.num_frames_in_flight, []).extend(resources)

    def wait_and_destroy(self) -> None:
        self.ctx.wait_idle()
        for resources in self.destruction_queue.values():
            for resource in resources:
                resource.destroy()
        self.destruction_queue.clear()
