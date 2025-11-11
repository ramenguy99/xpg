# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pyxpg import (
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
    ImageLayout,
    ImageUsageFlags,
    LoadOp,
    MemoryUsage,
    PhysicalDeviceType,
    RenderingAttachment,
    ResolveMode,
    Sampler,
    SamplerAddressMode,
    SamplerMipmapMode,
    StoreOp,
    slang,
)

from . import ffx, lights, scene
from .config import RendererConfig, UploadMethod
from .gpu_property import GpuBufferProperty, GpuImageProperty
from .renderer_frame import RendererFrame, SemaphoreInfo
from .shaders import compile
from .utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from .utils.gpu import (
    BufferUploadInfo,
    BulkUploader,
    ImageUploadInfo,
    UniformPool,
    UploadableBuffer,
)
from .utils.ring_buffer import RingBuffer
from .utils.threadpool import ThreadPool
from .viewport import Viewport

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
        config: RendererConfig,
    ):
        self.ctx = ctx
        self.num_frames_in_flight = num_frames_in_flight
        self.output_format = output_format

        self.shadowmap_format = Format.D32_SFLOAT

        # Config
        self.background_color = config.background_color

        # Scene descriptors
        self.max_shadowmaps = config.max_shadowmaps
        self.num_shadowmaps = 0

        self.linear_sampler = Sampler(
            ctx,
            Filter.LINEAR,
            Filter.LINEAR,
            SamplerMipmapMode.LINEAR,
            u=SamplerAddressMode.CLAMP_TO_EDGE,
            v=SamplerAddressMode.CLAMP_TO_EDGE,
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

        self.shader_cache: Dict[Tuple[Union[str, Tuple[str, str], Path], ...], slang.Shader] = {}

        self.scene_descriptor_set_layout, self.scene_descriptor_pool, self.scene_descriptor_sets = (
            create_descriptor_layout_pool_and_sets_ringbuffer(
                ctx,
                [
                    DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),  # 0 - Constants
                    DescriptorSetBinding(1, DescriptorType.SAMPLER),  # 1 - Shadowmap sampler
                    DescriptorSetBinding(1, DescriptorType.SAMPLER),  # 2 - Cubemap sampler
                    DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),  # 3 - Environment irradiance cubemap
                    DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),  # 4 - Environment specular cubemap
                    DescriptorSetBinding(1, DescriptorType.SAMPLED_IMAGE),  # 5 - Environment GGX LUT
                    *[DescriptorSetBinding(1, DescriptorType.STORAGE_BUFFER) for _ in lights.LIGHT_TYPES_INFO],
                    DescriptorSetBinding(self.max_shadowmaps, DescriptorType.SAMPLED_IMAGE),
                ],
                self.num_frames_in_flight,
                name="scene-descriptors",
            )
        )

        self.scene_depth_descriptor_set_layout = DescriptorSetLayout(
            ctx,
            [
                DescriptorSetBinding(1, DescriptorType.UNIFORM_BUFFER),
            ],
            name="scene-depth-descriptor-set-layout",
        )

        constants_dtype = np.dtype(
            {
                "view": (np.dtype((np.float32, (4, 4))), 0),
                "projection": (np.dtype((np.float32, (4, 4))), 64),
                "world_camera_position": (np.dtype((np.float32, (3,))), 128),
                "ambient_light": (np.dtype((np.float32, (3,))), 144),
                "has_environment_light": (np.uint32, 156),
                "focal": (np.dtype((np.float32, (2,))), 160),
                "inverse_viewport_size": (np.dtype((np.float32, (2,))), 168),
                "max_specular_mip": (np.float32, 176),
                "num_lights": (np.dtype((np.uint32, (len(lights.LIGHT_TYPES_INFO),))), 180),
            }
        )  # type: ignore

        self.constants = np.zeros((1,), constants_dtype)
        self.uniform_buffers = RingBuffer(
            [
                UploadableBuffer(ctx, constants_dtype.itemsize, BufferUsageFlags.UNIFORM)
                for _ in range(self.num_frames_in_flight)
            ]
        )

        self.max_lights_per_type = config.max_lights_per_type
        self.num_lights = [0] * len(lights.LIGHT_TYPES_INFO)
        self.light_buffers = RingBuffer(
            [
                [
                    UploadableBuffer(ctx, info.size * self.max_lights_per_type, BufferUsageFlags.STORAGE)
                    for info in lights.LIGHT_TYPES_INFO
                ]
                for _ in range(self.num_frames_in_flight)
            ]
        )

        self.spd_pipeline = ffx.SPDPipeline(self)

        self.ibl_default_params = lights.IBLParams()
        self.ibl_pipeline: Optional[lights.IBLPipeline] = None
        self.ggx_lut_pipeline = lights.GGXLUTPipeline(self)
        self.ggx_lut = self.ggx_lut_pipeline.run(self)

        for s, buf, light_bufs in zip(self.scene_descriptor_sets, self.uniform_buffers, self.light_buffers):
            s.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0)
            s.write_sampler(self.shadow_sampler, 1)
            s.write_sampler(self.linear_sampler, 2)
            s.write_image(self.zero_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 3)
            s.write_image(self.zero_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 4)
            s.write_image(self.ggx_lut, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 5)

            for i, light_buf in enumerate(light_bufs):
                s.write_buffer(light_buf, DescriptorType.STORAGE_BUFFER, 6, i)
            for i in range(self.max_shadowmaps):
                s.write_image(
                    self.zero_image,
                    ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                    DescriptorType.SAMPLED_IMAGE,
                    6 + len(light_bufs),
                    i,
                )

        self.uniform_environment_lights: List[lights.UniformEnvironmentLight] = []
        self.environment_light: Optional[lights.EnvironmentLight] = None

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
        self.enabled_gpu_properties: Set[Union[GpuBufferProperty, GpuImageProperty]] = set()

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
        self, equirectangular: Image, ibl_params: Optional["lights.IBLParams"] = None
    ) -> "lights.GpuEnvironmentCubemaps":
        if self.ibl_pipeline is None:
            self.ibl_pipeline = lights.IBLPipeline(self)
        if ibl_params is None:
            ibl_params = self.ibl_default_params
        return self.ibl_pipeline.run(self, equirectangular, ibl_params)

    def add_uniform_environment_light(self, light: "lights.UniformEnvironmentLight") -> None:
        self.uniform_environment_lights.append(light)

    def add_environment_light(
        self, light: "lights.EnvironmentLight", cubemaps: "lights.GpuEnvironmentCubemaps"
    ) -> None:
        if self.environment_light is not None:
            raise RuntimeError(
                "Attempting to add a second environment light. Only a single light environment light is currently supported"
            )
        self.environment_light = light

        for set in self.scene_descriptor_sets:
            set.write_image(
                cubemaps.irradiance_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 3
            )
            set.write_image(
                cubemaps.specular_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 4
            )

    def add_light(self, light_type: "lights.LightTypes", shadowmap: Optional[Image]) -> Tuple[int, int]:
        if self.num_lights[light_type.value] >= self.max_lights_per_type:
            raise RuntimeError(
                f'Too many ligths of type: {light_type}. Increase "config.renderer.max_lights_per_type" (current value: {self.max_lights_per_type}) to allow more lights.'
            )

        # Allocate light
        light_idx = self.num_lights[light_type.value]
        self.num_lights[light_type.value] += 1

        shadowmap_idx = -1
        if shadowmap is not None:
            if self.num_shadowmaps >= self.max_shadowmaps:
                raise RuntimeError(
                    f'Too many shadowmaps. Increase "config.renderer.max_shadowmaps" (current value: {self.max_shadowmaps}) to allow more shadowmaps.'
                )

            # Allocate light
            shadowmap_idx = self.num_shadowmaps
            self.num_shadowmaps += 1

            for set in self.scene_descriptor_sets:
                set.write_image(
                    shadowmap,
                    ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                    DescriptorType.SAMPLED_IMAGE,
                    6 + len(lights.LIGHT_TYPES_INFO),
                    shadowmap_idx,
                )

        return light_idx * lights.LIGHT_TYPES_INFO[light_type.value].size, shadowmap_idx

    def upload_light(
        self, frame: RendererFrame, light_type: "lights.LightTypes", data: memoryview, offset: int
    ) -> None:
        self.light_buffers.get_current()[light_type.value].upload(
            frame.cmd, MemoryUsage.SHADER_READ_ONLY, data, offset
        )

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

    def render(self, viewport: Viewport, frame: FrameInputs, gui: Gui) -> None:
        enabled_objects: List[scene.Object] = []
        enabled_lights: List[lights.Light] = []
        enabled_gpu_properties: Set[Union[GpuBufferProperty, GpuImageProperty]] = set()

        def visit(o: scene.Object) -> None:
            if o.enabled:
                enabled_objects.append(o)
                if isinstance(o, lights.Light):
                    enabled_lights.append(o)
                o.create_if_needed(self)
                if o.material is not None:
                    for mp in o.material.properties:
                        if mp.property.gpu_property is not None:
                            enabled_gpu_properties.add(mp.property.gpu_property)
                for p in o.properties:
                    if p.gpu_property is not None:
                        enabled_gpu_properties.add(p.gpu_property)

        viewport.scene.visit_objects(visit)

        # Flush synchronous upload buffers after creating new objects
        if len(self.bulk_upload_list) > 0:
            self.bulk_uploader.bulk_upload(self.bulk_upload_list)
            self.bulk_upload_list.clear()

        cmd = frame.command_buffer
        viewport_rect = (
            viewport.rect.x,
            viewport.rect.y + viewport.rect.height,
            viewport.rect.width,
            viewport.rect.y - viewport.rect.height,
        )
        rect = (
            viewport.rect.x,
            viewport.rect.y,
            viewport.rect.width,
            viewport.rect.height,
        )

        descriptor_set = self.scene_descriptor_sets.get_current_and_advance()
        buf = self.uniform_buffers.get_current_and_advance()

        proj = viewport.camera.projection()
        self.constants["projection"] = proj
        self.constants["view"] = viewport.camera.view()
        self.constants["world_camera_position"] = viewport.camera.position()
        self.constants["num_lights"] = self.num_lights
        self.constants["ambient_light"] = np.sum([l.radiance.get_current() for l in self.uniform_environment_lights])
        if self.environment_light is not None:
            self.constants["has_environment_light"] = 1
            self.constants["max_specular_mip"] = self.environment_light.gpu_cubemaps.specular_cubemap.mip_levels - 1.0
        else:
            self.constants["has_environment_light"] = 0
            self.constants["max_specular_mip"] = 0.0
        self.constants["inverse_viewport_size"] = (1.0 / viewport.rect.width, 1.0 / viewport.rect.height)
        self.constants["focal"] = (proj[0][0] * 0.5 * viewport.rect.width, proj[1][1] * 0.5 * viewport.rect.height)

        buf.upload(
            cmd,
            MemoryUsage.SHADER_UNIFORM,
            self.constants.view(np.uint8),
        )

        f = RendererFrame(
            self.total_frame_index % self.num_frames_in_flight,
            self.total_frame_index,
            descriptor_set,
            cmd,
            frame.additional_semaphores,
            frame.transfer_command_buffer,
            frame.transfer_semaphores,
        )

        for p in enabled_gpu_properties:
            p.load(f)

        for p in enabled_gpu_properties:
            p.upload(f)

        self.enabled_gpu_properties = enabled_gpu_properties

        for o in enabled_objects:
            if o.material is not None:
                o.material.upload(self, f)
            o.upload(self, f)

        # Render shadows
        for l in enabled_lights:
            l.render_shadowmaps(self, f, enabled_objects)

        # Render scene
        cmd.image_barrier(
            frame.image,
            ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
            MemoryUsage.ALL,
            MemoryUsage.COLOR_ATTACHMENT,
            undefined=True,
        )
        cmd.image_barrier(
            self.depth_buffer,
            ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            MemoryUsage.DEPTH_STENCIL_ATTACHMENT,
            MemoryUsage.DEPTH_STENCIL_ATTACHMENT,
            aspect_mask=ImageAspectFlags.DEPTH,
            undefined=True,
        )
        if self.msaa_target is not None:
            cmd.image_barrier(
                self.msaa_target,
                ImageLayout.COLOR_ATTACHMENT_OPTIMAL,
                MemoryUsage.COLOR_ATTACHMENT,
                MemoryUsage.COLOR_ATTACHMENT,
                undefined=True,
            )

        cmd.set_viewport(viewport_rect)
        cmd.set_scissors(rect)
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
                        resolve_image=frame.image,
                    )
                    if self.msaa_target is not None
                    else RenderingAttachment(frame.image, LoadOp.CLEAR, StoreOp.STORE, self.background_color)
                )
            ],
            depth=DepthAttachment(self.depth_buffer, LoadOp.CLEAR, StoreOp.STORE, self.depth_clear_value),
        ):
            for o in enabled_objects:
                o.render(self, f, descriptor_set)

        # Render GUI
        with cmd.rendering(
            rect,
            color_attachments=[
                RenderingAttachment(
                    frame.image,
                    load_op=LoadOp.LOAD,
                    store_op=StoreOp.STORE,
                    clear=self.background_color,
                ),
            ],
        ):
            gui.render(cmd)
        cmd.image_barrier(
            frame.image,
            ImageLayout.PRESENT_SRC,
            MemoryUsage.COLOR_ATTACHMENT,
            MemoryUsage.PRESENT,
        )

        self.uniform_pool.advance()
        self.light_buffers.advance()
        self.total_frame_index += 1

    def prefetch(self) -> None:
        for p in self.enabled_gpu_properties:
            p.prefetch()
