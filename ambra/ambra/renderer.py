# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from pyxpg import (
    AccelerationStructure,
    AccelerationStructureMesh,
    AccessFlags,
    AllocType,
    BorderColor,
    Buffer,
    BufferUsageFlags,
    CommandBuffer,
    CompareOp,
    ComputePipeline,
    Context,
    DepthAttachment,
    DescriptorBindingFlags,
    DescriptorPool,
    DescriptorSet,
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
    IndexType,
    LoadOp,
    MemoryBarrier,
    MemoryUsage,
    PhysicalDeviceType,
    PipelineStageFlags,
    PushConstantsRange,
    RenderingAttachment,
    ResolveMode,
    Sampler,
    SamplerAddressMode,
    SamplerMipmapMode,
    Shader,
    StoreOp,
    slang,
)

from .config import RendererConfig, RenderMode, UploadMethod
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
from .materials import (
    ColorMaterial,
    DiffuseMaterial,
    DiffuseSpecularMaterial,
    PBRMaterial,
)
from .property import BufferProperty, ImageProperty, Property, PropertyItem
from .renderer_frame import RendererFrame, SemaphoreInfo
from .scene import Object, Scene
from .shaders import compile
from .utils.descriptors import create_descriptor_layout_pool_and_set, create_descriptor_pool_and_sets
from .utils.gpu import (
    AccelerationStructureInstanceInfo,
    BufferUploadInfo,
    BulkUploader,
    ImageUploadInfo,
    UniformPool,
    UploadableBuffer,
    div_round_up,
    view_bytes,
)
from .utils.ring_buffer import RingBuffer
from .utils.threadpool import ThreadPool
from .viewport import PathTracerViewport, Viewport

if TYPE_CHECKING:
    from .gpu_property import GpuProperty
    from .materials import Material

logger = logging.getLogger(__name__)

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


@dataclass
class PathTracer:
    # Static
    pipeline: ComputePipeline
    descriptor_set: DescriptorSet
    descriptor_pool: DescriptorPool
    descriptor_set_layout: DescriptorSetLayout

    viewport_descriptor_sets: List[DescriptorSet]
    viewport_descriptor_pool: DescriptorPool
    viewport_descriptor_set_layout: DescriptorSetLayout

    # Scene
    acceleration_structure: Optional[AccelerationStructure]
    instances_buf: Optional[Buffer]
    materials_buf: Optional[Buffer]


class Renderer:
    def __init__(
        self,
        ctx: Context,
        width: int,
        height: int,
        num_frames_in_flight: int,
        output_format: Format,
        multiviewport: bool,
        max_viewports: int,
        config: RendererConfig,
    ):
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.output_format = output_format
        self.multiviewport = multiviewport
        self.max_viewports = max_viewports
        self.shadow_map_format = Format.D32_SFLOAT

        # Config
        self.background_color = config.background_color
        self.supports_path_tracing = bool(ctx.device_features & DeviceFeatures.RAY_QUERY)
        if config.render_mode == RenderMode.PATH_TRACER and not self.supports_path_tracing:
            logger.warning(
                "Path tracing cannot be enabled because DeviceFeatures.RAY_QUERY is not supported. Falling back to raster mode."
            )
            self.render_mode = RenderMode.RASTER
        else:
            self.render_mode = config.render_mode

        # Path tracing
        self.path_tracer: Optional[PathTracer] = None
        self.path_tracer_max_bounces = config.path_tracer_max_bounces
        self.path_tracer_samples_per_frame = config.path_tracer_samples_per_frame
        self.path_tracer_max_samples_per_pixel = config.path_tracer_max_samples_per_pixel
        self.path_tracer_max_textures = config.path_tracer_max_textures
        self.path_tracer_clip_value = config.path_tracer_clip_value
        self.path_tracer_use_background_color = config.path_tracer_use_background_color
        self.path_tracer_instance_dtype = np.dtype(
            {
                "normal_matrix": (np.dtype((np.float32, (3, 4))), 0),
                "positions": (np.uint64, 48),
                "normals": (np.uint64, 56),
                "tangents": (np.uint64, 64),
                "uvs": (np.uint64, 72),
                "indices": (np.uint64, 80),
                "material_index": (np.uint32, 88),
                "_padding": (np.uint32, 92),
            }
        )  # type: ignore
        self.path_tracer_material_dtype = np.dtype(
            {
                "albedo": (np.dtype((np.float32, (3,))), 0),
                "albedo_texture": (np.uint32, 12),
                "roughness": (np.float32, 16),
                "roughness_texture": (np.uint32, 20),
                "metallic": (np.float32, 24),
                "metallic_texture": (np.uint32, 28),
                "normal_texture": (np.uint32, 32),
                "_padding": (np.uint32, 36),
                "_padding1": (np.uint32, 40),
                "_padding2": (np.uint32, 44),
            }
        )  # type: ignore
        self.path_tracer_constants_dtype = np.dtype(
            {
                "width": (np.uint32, 0),
                "height": (np.uint32, 4),
                "max_bounces": (np.uint32, 8),
                "num_directional_lights": (np.uint32, 12),
                "camera_position": (np.dtype((np.float32, (3,))), 16),
                "viewport_mask": (np.uint32, 28),
                "camera_forward": (np.dtype((np.float32, (3,))), 32),
                "film_dist": (np.float32, 44),
                "camera_up": (np.dtype((np.float32, (3,))), 48),
                "max_samples_per_pixel": (np.uint32, 60),
                "camera_right": (np.dtype((np.float32, (3,))), 64),
                "clip_value": (np.float32, 76),
                "ambient_light": (np.dtype((np.float32, (3,))), 80),
                "has_environment_light": (np.uint32, 92),
                "background_color": (np.dtype((np.float32, (3,))), 96),
                "use_background_color": (np.uint32, 108),
            }
        )  # type: ignore
        self.path_tracer_push_constants_dtype = np.dtype(
            {
                "sample_index": (np.uint32, 0),
            }
        )  # type: ignore
        self.path_tracer_push_constants = np.zeros(1, self.path_tracer_push_constants_dtype)

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
        self.zero_triangle: Optional[Buffer] = None
        self.zero_material: Optional[Material] = None

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
        # - additionally we could use StoreOp.DONT_CARE, TRANSIENT_ATTACHMENT and LAZILY_ALLOCATED memory
        #   for best performance on tilers.
        # - if at some point we want to expose multiple MSAA passes we can add a way to disable it.
        # - for retrieving color/depth we can do it by returning the resolved framebuffer (allocating one more for depth).
        self.msaa_samples = max(1, config.msaa_samples)
        self.msaa_target: Optional[Image] = None

        # Will be populated in self.resize(), and will never be None after that
        self.depth_buffer: Image = None  # type: ignore
        self.depth_format = Format.D32_SFLOAT
        self.depth_clear_value = 1.0
        self.depth_compare_op = CompareOp.LESS
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        self.render_width = width
        self.render_height = height

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
                    for j, spd_instance in enumerate(self.spd_pipeline_instances):
                        if i + j >= len(mip_generation_requests):
                            break
                        req = mip_generation_requests[i + j]
                        assert req.level_0_view is not None

                        spd_instance.set_image_extents(req.image.width, req.image.height, req.image.mip_levels)
                        self.spd_pipeline.run(
                            cmd,
                            req.image,
                            req.layout,
                            req.level_0_view,
                            req.mip_views,
                            spd_instance,
                            req.mip_generation_filter,
                        )

        path_tracing = self.render_mode == RenderMode.PATH_TRACER

        # Stage: path tracing pipeline creation
        if path_tracing and self.path_tracer is None:
            path_tracer_descriptor_set_layout, path_tracer_descriptor_pool, path_tracer_descriptor_set = (
                create_descriptor_layout_pool_and_set(
                    self.ctx,
                    [
                        DescriptorSetBinding(1, DescriptorType.ACCELERATION_STRUCTURE),
                        DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),
                        DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER),
                        DescriptorSetBinding(1, DescriptorType.SAMPLER),
                        DescriptorSetBinding(
                            self.path_tracer_max_textures,
                            DescriptorType.SAMPLED_IMAGE,
                            DescriptorBindingFlags.VARIABLE_DESCRIPTOR_COUNT | DescriptorBindingFlags.PARTIALLY_BOUND,
                        ),
                    ],
                    self.path_tracer_max_textures,
                    name="path-tracer",
                )
            )
            path_tracer_viewport_descriptor_set_layout = DescriptorSetLayout(
                self.ctx,
                [
                    DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),  # 0 - Constants
                    *[DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER) for _ in LIGHT_TYPES_INFO],  # 1 - Lights
                    DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),  # 2 - Output
                    DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),  # 3 - Environment light
                    DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),  # 4 - Accumulation
                ],
                name="path-tracer-viewport-descriptor-layout",
            )
            shader = self.compile_builtin_shader(Path("3d", "path_tracer", "path_tracer.slang"))
            path_tracer_pipeline = ComputePipeline(
                self.ctx,
                Shader(self.ctx, shader.code),
                descriptor_set_layouts=[
                    path_tracer_descriptor_set_layout,
                    path_tracer_viewport_descriptor_set_layout,
                ],
                push_constants_ranges=[PushConstantsRange(self.path_tracer_push_constants_dtype.itemsize)],
                name="pathtracer",
            )

            # TODO: dynamic viewport creation
            viewport_descriptor_pool, viewport_descriptor_sets = create_descriptor_pool_and_sets(
                self.ctx,
                path_tracer_viewport_descriptor_set_layout,
                self.max_viewports * self.num_frames_in_flight,
                name="pathtracer-viewport-scene-descriptor-sets",
            )
            self.viewports: List[Viewport] = []
            for viewport_index, v in enumerate(viewports):
                descriptor_sets = RingBuffer(
                    viewport_descriptor_sets[
                        viewport_index * self.num_frames_in_flight : (viewport_index + 1) * self.num_frames_in_flight
                    ]
                )
                uniform_buffers = RingBuffer(
                    [
                        UploadableBuffer(self.ctx, self.path_tracer_constants_dtype.itemsize, BufferUsageFlags.UNIFORM)
                        for _ in range(self.num_frames_in_flight)
                    ]
                )
                constants = np.zeros((1,), self.path_tracer_constants_dtype)
                for s, buf, light_bufs in zip(descriptor_sets, uniform_buffers, v.scene_light_buffers):
                    s.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0)
                    for i, light_buf in enumerate(light_bufs):
                        s.write_buffer(light_buf, DescriptorType.STORAGE_BUFFER, 1, i)
                v.path_tracer_viewport = PathTracerViewport(
                    descriptor_sets,
                    uniform_buffers,
                    constants,
                    0,
                    v.camera.camera_from_world.copy(),
                    None,
                )

            self.path_tracer = PathTracer(
                pipeline=path_tracer_pipeline,
                descriptor_set=path_tracer_descriptor_set,
                descriptor_pool=path_tracer_descriptor_pool,
                descriptor_set_layout=path_tracer_descriptor_set_layout,
                viewport_descriptor_set_layout=path_tracer_viewport_descriptor_set_layout,
                viewport_descriptor_pool=viewport_descriptor_pool,
                viewport_descriptor_sets=viewport_descriptor_sets,
                # Scene
                acceleration_structure=None,
                instances_buf=None,
                materials_buf=None,
            )

        cmd = frame.command_buffer
        f = RendererFrame(
            # Frame counters
            self.total_frame_index % self.num_frames_in_flight,
            self.total_frame_index,
            path_tracing,
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

            if path_tracing:
                assert viewport.path_tracer_viewport is not None
                assert self.path_tracer is not None
                buf = viewport.path_tracer_viewport.uniform_buffers.get_current_and_advance()
                constants = viewport.path_tracer_viewport.constants

                if viewport.camera.camera_from_world != viewport.path_tracer_viewport.last_camera_transform:
                    viewport.path_tracer_viewport.sample_index = 0
                    viewport.path_tracer_viewport.last_camera_transform = viewport.camera.camera_from_world.copy()

                camera_right, camera_up, camera_front = viewport.camera.right_up_front()
                constants["width"] = viewport_width
                constants["height"] = viewport_height
                constants["max_bounces"] = self.path_tracer_max_bounces
                constants["num_directional_lights"] = num_lights
                constants["camera_position"] = viewport.camera.position()
                constants["viewport_mask"] = 1 << viewport_index
                constants["camera_forward"] = camera_front
                constants["film_dist"] = viewport.camera.film_dist()
                constants["camera_up"] = camera_up
                constants["max_samples_per_pixel"] = self.path_tracer_max_samples_per_pixel
                constants["camera_right"] = camera_right
                constants["clip_value"] = self.path_tracer_clip_value
                constants["ambient_light"] = uniform_environment_radiance
                if environment_light is not None:
                    constants["has_environment_light"] = 1
                else:
                    constants["has_environment_light"] = 0
                constants["background_color"] = self.background_color[:3]
                constants["use_background_color"] = self.path_tracer_use_background_color
                buf.upload(
                    cmd,
                    MemoryUsage.NONE,
                    constants.view(np.uint8),
                )
            else:
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

        # Stage: post upload
        for o in enabled_objects:
            o.post_upload(self, f)

        # Create and upload path tracing acceleration structures
        if path_tracing:
            assert self.path_tracer is not None

            # Create or resize accumulation buffer
            for v in viewports:
                assert v.path_tracer_viewport is not None
                if (
                    v.path_tracer_viewport.accumulation_image is None
                    or self.render_width != v.path_tracer_viewport.accumulation_image.width
                    or self.render_height != v.path_tracer_viewport.accumulation_image.height
                ):
                    # Assume no rendering is in flight if resolution changed
                    if v.path_tracer_viewport.accumulation_image is not None:
                        v.path_tracer_viewport.accumulation_image.destroy()

                    v.path_tracer_viewport.accumulation_image = Image(
                        self.ctx,
                        self.render_width,
                        self.render_height,
                        Format.R32G32B32A32_SFLOAT,
                        ImageUsageFlags.STORAGE,
                        AllocType.DEVICE_DEDICATED,
                        name="accumulation",
                    )
                    cmd.image_barrier(
                        v.path_tracer_viewport.accumulation_image,
                        ImageLayout.GENERAL,
                        MemoryUsage.NONE,
                        MemoryUsage.COMPUTE_SHADER,
                    )
                    for s in v.path_tracer_viewport.descriptor_sets:
                        s.write_image(
                            v.path_tracer_viewport.accumulation_image,
                            ImageLayout.GENERAL,
                            DescriptorType.STORAGE_IMAGE,
                            4,
                        )

            # Stage: path tracing structures creation
            if self.path_tracer.acceleration_structure is None:
                # Flush commands and re-open per-frame command buffers.
                #
                # We need this because acceleration structure creation could use just uploaded positions and indices.
                # Since we also want to render a frame this frame we ensure that all uploads happened and then reset
                # command buffers.
                #
                # We could skip this if we knew that we don't have any positions or indices uploaded this frame
                # but there is no simple way to detect that right now.
                # A better system would have each primitive manage the creation and update of their acceleration
                # structures and avoid this kind of stall.
                # Since we are now anyways stalling the pipeline to create acceleration structures this overhead
                # should be negligible compared to the actual build time and synchronization.
                f.cmd.end()
                self.ctx.queue.submit(
                    f.cmd,
                    wait_semaphores=[(s.sem, s.wait_value, s.wait_stage) for s in f.additional_semaphores],
                    signal_semaphores=[(s.sem, s.signal_value, s.signal_stage) for s in f.additional_semaphores],
                )
                if f.transfer_semaphores:
                    assert f.transfer_cmd is not None
                    f.transfer_cmd.end()
                    self.ctx.transfer_queue.submit(
                        f.transfer_cmd,
                        wait_semaphores=[(s.sem, s.wait_value, s.wait_stage) for s in f.transfer_semaphores],
                        signal_semaphores=[(s.sem, s.signal_value, s.signal_stage) for s in f.transfer_semaphores],
                    )

                # Wait for commands to complete
                self.ctx.wait_idle()

                # Reset commands and semaphores
                f.cmd.begin()
                if f.transfer_semaphores:
                    assert f.transfer_cmd is not None
                    f.transfer_cmd.begin()
                f.additional_semaphores.clear()
                f.transfer_semaphores.clear()

                instances: List[AccelerationStructureInstanceInfo] = []
                materials: Dict[Material, int] = {}
                for o in enabled_objects:
                    if o.material is not None:
                        if o.material in materials:
                            material_index = materials[o.material]
                        else:
                            material_index = len(materials)
                            materials[o.material] = material_index
                        o.append_acceleration_structure_instances(instances, material_index)

                if not instances:
                    if self.zero_triangle is None:
                        self.zero_triangle = Buffer.from_data(
                            self.ctx,
                            np.array([0.0, 0.0, 0.0], np.float32),
                            BufferUsageFlags.SHADER_DEVICE_ADDRESS
                            | BufferUsageFlags.ACCELERATION_STRUCTURE_INPUT
                            | BufferUsageFlags.TRANSFER_DST,
                            AllocType.DEVICE_MAPPED_WITH_FALLBACK,
                            "zero-triangle",
                        )
                    instances.append(
                        AccelerationStructureInstanceInfo(
                            np.zeros((3, 4), np.float32),
                            np.zeros((3, 4), np.float32),
                            3,
                            self.zero_triangle.address,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                        )
                    )
                if not materials:
                    if self.zero_material is None:
                        self.zero_material = ColorMaterial((0, 0, 0))
                    materials[self.zero_material] = 0

                # Instances and acceleration structures
                acceleration_structure_meshes: List[AccelerationStructureMesh] = []
                instances_data = np.zeros(len(instances), self.path_tracer_instance_dtype)
                for i, instance in enumerate(instances):
                    instances_data[i]["normal_matrix"] = instance.normal_matrix
                    instances_data[i]["positions"] = instance.positions_address
                    instances_data[i]["normals"] = instance.normals_address
                    instances_data[i]["tangents"] = instance.tangents_address
                    instances_data[i]["uvs"] = instance.uvs_address
                    instances_data[i]["indices"] = instance.indices_address
                    instances_data[i]["material_index"] = instance.material_index

                    acceleration_structure_meshes.append(
                        AccelerationStructureMesh(
                            instance.positions_address,
                            12,
                            instance.positions_count,
                            Format.R32G32B32_SFLOAT,
                            instance.indices_address,
                            IndexType.UINT32 if instance.indices_address else IndexType.NONE,
                            instance.primitive_count,
                            tuple(instance.transform.T.flatten()),
                            instance.viewport_mask,
                        )
                    )
                self.path_tracer.acceleration_structure = AccelerationStructure(
                    self.ctx, acceleration_structure_meshes, name="acceleration-structure"
                )
                self.path_tracer.instances_buf = Buffer.from_data(
                    self.ctx,
                    view_bytes(instances_data),
                    BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
                    AllocType.DEVICE_MAPPED_WITH_FALLBACK,
                    name="path-tracer-instances",
                )

                # Upload materials and write bindless descriptors
                textures: Dict[Union[Image, ImageView], int] = {}
                materials_data = np.zeros(len(materials), self.path_tracer_material_dtype)

                def alloc_texture(p: Property) -> int:
                    if isinstance(p, ImageProperty):
                        view = p.get_current_gpu().view()
                        if view in textures:
                            texture_index = textures[view]
                        else:
                            texture_index = len(textures)
                            textures[view] = texture_index
                        return texture_index
                    else:
                        return 0xFFFFFFFF

                def get_value(p: Property) -> Union[float, PropertyItem]:
                    if isinstance(p, BufferProperty):
                        return p.get_current()
                    else:
                        return 0.0

                for i, material in enumerate(materials.keys()):
                    if isinstance(material, DiffuseMaterial):
                        materials_data[i]["albedo"] = get_value(material.diffuse[0].property)
                        materials_data[i]["albedo_texture"] = alloc_texture(material.diffuse[0].property)
                        materials_data[i]["roughness"] = 1.0
                        materials_data[i]["roughness_texture"] = 0xFFFFFFFF
                        materials_data[i]["metallic"] = 0.0
                        materials_data[i]["metallic_texture"] = 0xFFFFFFFF
                        materials_data[i]["normal_texture"] = alloc_texture(material.normal[0].property)
                    elif isinstance(material, DiffuseSpecularMaterial):
                        # TODO: better mapping or remove
                        materials_data[i]["albedo"] = get_value(material.diffuse[0].property)
                        materials_data[i]["albedo_texture"] = alloc_texture(material.diffuse[0].property)
                        materials_data[i]["roughness"] = (
                            1 - min(get_value(material.specular_exponent[0].property), 64.0) / 64.0
                        )
                        materials_data[i]["roughness_texture"] = 0xFFFFFFFF
                        materials_data[i]["metallic"] = 0.0
                        materials_data[i]["metallic_texture"] = 0xFFFFFFFF
                        materials_data[i]["normal_texture"] = 0xFFFFFFFF
                    elif isinstance(material, ColorMaterial):
                        # TODO: use emissive instead
                        materials_data[i]["albedo"] = get_value(material.color[0].property)
                        materials_data[i]["albedo_texture"] = alloc_texture(material.color[0].property)
                        materials_data[i]["roughness"] = 1.0
                        materials_data[i]["roughness_texture"] = 0xFFFFFFFF
                        materials_data[i]["metallic"] = 0.0
                        materials_data[i]["metallic_texture"] = 0xFFFFFFFF
                        materials_data[i]["normal_texture"] = 0xFFFFFFFF
                    elif isinstance(material, PBRMaterial):
                        materials_data[i]["albedo"] = get_value(material.albedo[0].property)
                        materials_data[i]["albedo_texture"] = alloc_texture(material.albedo[0].property)
                        materials_data[i]["roughness"] = get_value(material.roughness[0].property)
                        materials_data[i]["roughness_texture"] = alloc_texture(material.roughness[0].property)
                        materials_data[i]["metallic"] = get_value(material.metallic[0].property)
                        materials_data[i]["metallic_texture"] = alloc_texture(material.metallic[0].property)
                        materials_data[i]["normal_texture"] = alloc_texture(material.normal[0].property)
                    else:
                        materials_data[i]["albedo"] = 0.0
                        materials_data[i]["albedo_texture"] = 0xFFFFFFFF
                        materials_data[i]["roughness"] = 1.0
                        materials_data[i]["roughness_texture"] = 0xFFFFFFFF
                        materials_data[i]["metallic"] = 0.0
                        materials_data[i]["metallic_texture"] = 0xFFFFFFFF
                        materials_data[i]["normal_texture"] = 0xFFFFFFFF
                if self.path_tracer.materials_buf is not None:
                    self.path_tracer.materials_buf.destroy()
                self.path_tracer.materials_buf = Buffer.from_data(
                    self.ctx,
                    view_bytes(materials_data),
                    BufferUsageFlags.STORAGE | BufferUsageFlags.TRANSFER_DST,
                    AllocType.DEVICE_MAPPED_WITH_FALLBACK,
                    name="path-tracer-materials",
                )

                # Static descriptors
                self.path_tracer.descriptor_set.write_acceleration_structure(
                    self.path_tracer.acceleration_structure, 0
                )
                self.path_tracer.descriptor_set.write_buffer(
                    self.path_tracer.instances_buf, DescriptorType.STORAGE_BUFFER, 1
                )
                self.path_tracer.descriptor_set.write_buffer(
                    self.path_tracer.materials_buf, DescriptorType.STORAGE_BUFFER, 2
                )
                self.path_tracer.descriptor_set.write_sampler(self.linear_sampler, 3)
                for i, image in enumerate(textures):
                    self.path_tracer.descriptor_set.write_image(
                        image, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 4, i
                    )

        # Stage: Render shadows
        if not path_tracing:
            for l in enabled_lights:
                l.render_shadow_maps(self, f, enabled_objects)

        before_gui_barriers: List[ImageBarrier] = []
        if path_tracing:
            if self.multiviewport:
                f.before_render_image_barriers.append(
                    ImageBarrier(
                        frame.image,
                        ImageLayout.UNDEFINED,
                        ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                        PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                        AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                        PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                        AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                    )
                )
                for v in viewports:
                    assert v.image is not None
                    f.before_render_image_barriers.append(
                        ImageBarrier(
                            v.image,
                            ImageLayout.UNDEFINED,
                            ImageLayout.GENERAL,
                            PipelineStageFlags.FRAGMENT_SHADER,
                            AccessFlags.SHADER_SAMPLED_READ,
                            PipelineStageFlags.COMPUTE_SHADER,
                            AccessFlags.SHADER_STORAGE_READ | AccessFlags.SHADER_STORAGE_WRITE,
                        )
                    )
                    before_gui_barriers.append(
                        ImageBarrier(
                            v.image,
                            ImageLayout.GENERAL,
                            ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                            PipelineStageFlags.COMPUTE_SHADER,
                            AccessFlags.SHADER_STORAGE_READ | AccessFlags.SHADER_STORAGE_WRITE,
                            PipelineStageFlags.FRAGMENT_SHADER,
                            AccessFlags.SHADER_SAMPLED_READ,
                        )
                    )
            else:
                f.before_render_image_barriers.append(
                    ImageBarrier(
                        frame.image,
                        ImageLayout.UNDEFINED,
                        ImageLayout.GENERAL,
                        PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                        AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                        PipelineStageFlags.COMPUTE_SHADER,
                        AccessFlags.SHADER_STORAGE_READ | AccessFlags.SHADER_STORAGE_WRITE,
                    )
                )
                before_gui_barriers.append(
                    ImageBarrier(
                        frame.image,
                        ImageLayout.GENERAL,
                        ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                        PipelineStageFlags.COMPUTE_SHADER,
                        AccessFlags.SHADER_STORAGE_READ | AccessFlags.SHADER_STORAGE_WRITE,
                        PipelineStageFlags.COLOR_ATTACHMENT_OUTPUT,
                        AccessFlags.COLOR_ATTACHMENT_WRITE | AccessFlags.COLOR_ATTACHMENT_READ,
                    )
                )
        else:
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

            if self.multiviewport:
                for v in viewports:
                    assert v.image is not None
                    f.before_render_image_barriers.append(
                        ImageBarrier(
                            v.image,
                            ImageLayout.UNDEFINED,
                            ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                            PipelineStageFlags.FRAGMENT_SHADER,
                            AccessFlags.SHADER_SAMPLED_READ,
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

            if path_tracing:
                assert self.path_tracer is not None
                assert viewport.path_tracer_viewport is not None
                descriptor_set = viewport.path_tracer_viewport.descriptor_sets.get_current_and_advance()
                written_environment = False
                for l in enabled_lights:
                    if l.viewport_mask is None or l.viewport_mask & (1 << viewport_index):  # noqa: SIM102
                        if isinstance(l, EnvironmentLight):
                            descriptor_set.write_image(
                                l.gpu_cubemaps.specular_cubemap,
                                ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                                DescriptorType.SAMPLED_IMAGE,
                                3,
                            )
                            written_environment = True
                if not written_environment:
                    descriptor_set.write_image(
                        self.zero_cubemap,
                        ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                        DescriptorType.SAMPLED_IMAGE,
                        3,
                    )
                descriptor_set.write_image(
                    frame.image if viewport.image is None else viewport.image,
                    ImageLayout.GENERAL,
                    DescriptorType.STORAGE_IMAGE,
                    2,
                )
                cmd.bind_compute_pipeline(self.path_tracer.pipeline, [self.path_tracer.descriptor_set, descriptor_set])
                for _ in range(self.path_tracer_samples_per_frame):
                    self.path_tracer_push_constants["sample_index"] = viewport.path_tracer_viewport.sample_index
                    cmd.push_constants(self.path_tracer.pipeline, self.path_tracer_push_constants.tobytes())
                    cmd.dispatch(div_round_up(viewport.rect.width, 8), div_round_up(viewport.rect.height, 8), 1)
                    cmd.memory_barrier(MemoryUsage.COMPUTE_SHADER, MemoryUsage.COMPUTE_SHADER)
                    if viewport.path_tracer_viewport.sample_index < self.path_tracer_max_samples_per_pixel:
                        viewport.path_tracer_viewport.sample_index += 1
                    if viewport.path_tracer_viewport.sample_index >= self.path_tracer_max_samples_per_pixel:
                        break
            else:
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

    def toggle_path_tracer(self) -> None:
        if self.render_mode == RenderMode.RASTER:
            if self.supports_path_tracing:
                self.render_mode = RenderMode.PATH_TRACER
            else:
                logger.warning(
                    "Path tracing cannot be enabled because DeviceFeatures.RAY_QUERY is not supported. Falling back to raster mode."
                )
        else:
            self.render_mode = RenderMode.RASTER
