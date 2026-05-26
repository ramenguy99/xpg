# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from pathlib import Path
from queue import Empty, Queue
from time import perf_counter_ns
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pyglm.glm import dvec2, ivec2, normalize, vec3
from pyxpg import (
    AccessFlags,
    Action,
    AllocType,
    BorderColor,
    BufferUsageFlags,
    DescriptorSetBinding,
    DescriptorType,
    Device,
    DeviceFeatures,
    Format,
    Gui,
    Image,
    ImageAspectFlags,
    ImageBarrier,
    ImageCreateFlags,
    ImageLayout,
    ImageUsageFlags,
    ImageView,
    ImageViewType,
    Instance,
    Key,
    LogLevel,
    MemoryHeapFlags,
    Modifiers,
    MouseButton,
    PipelineStageFlags,
    Sampler,
    SamplerAddressMode,
    Stage,
    SwapchainOutOfDateError,
    SwapchainStatus,
    Window,
    imgui,
    process_events,
    set_log_level,
)

from .config import Config
from .gpu_property import (
    GpuBufferView,
    GpuImageView,
    GpuPreuploadedArrayProperty,
    GpuPreuploadedProperty,
    GpuProperty,
    GpuStreamingProperty,
)
from .headless import HeadlessSwapchain, HeadlessSwapchainFrame
from .icons_fontawesome import IconsFontAwesome7
from .keybindings import KeyMap
from .lights import LIGHT_TYPES_INFO
from .renderer import FrameInputs, Renderer
from .scene import Object, Scene, Widget
from .server import Client, Message, RawMessage, Server, parse_builtin_messages
from .utils.descriptors import (
    create_descriptor_layout_pool_and_sets,
    create_descriptor_pool_and_sets,
)
from .utils.gpu import UploadableBuffer, float4_to_uint32, to_srgb_format
from .utils.lru_pool import LRUPool
from .utils.ring_buffer import RingBuffer
from .viewport import OrthographicViewType, Playback, Rect, Viewport, ViewportImage


class Viewer:
    def __init__(
        self,
        title: str = "ambra",
        config: Optional[Config] = None,
        key_map: Optional[KeyMap] = None,
    ):
        # Config
        config = config if config is not None else Config()

        # Key bindings
        self.key_map = key_map if key_map is not None else KeyMap()

        # Logging
        if config.log_level is not None:
            set_log_level(LogLevel(config.log_level.value))

        # Viewer
        self.running = False

        # Device
        self.instance = Instance(
            version=(1, 1),
            presentation=config.window,
            enable_validation_layer=config.enable_validation_layer,
            enable_synchronization_validation=config.enable_synchronization_validation,
            enable_gpu_based_validation=config.enable_gpu_based_validation,
            enable_shader_debug_printf=config.enable_shader_debug_printf,
            shader_debug_printf_buffer_size=config.shader_debug_printf_buffer_size,
        )

        self.device = Device(
            self.instance,
            version=(1, 1),
            required_features=DeviceFeatures.SYNCHRONIZATION_2
            | DeviceFeatures.DYNAMIC_RENDERING
            | DeviceFeatures.DESCRIPTOR_INDEXING
            | DeviceFeatures.STORAGE_IMAGE_READ_WRITE_WITHOUT_FORMAT
            | (DeviceFeatures.SWAPCHAIN_MUTABLE_FORMAT if config.window else 0),
            optional_features=DeviceFeatures.RAY_QUERY
            | DeviceFeatures.HOST_QUERY_RESET
            | DeviceFeatures.WIDE_LINES
            | DeviceFeatures.TIMELINE_SEMAPHORES
            | DeviceFeatures.SHADER_DRAW_PARAMETERS
            | DeviceFeatures.SHADER_FLOAT16_INT8
            | DeviceFeatures.SHADER_INT16
            | DeviceFeatures.SHADER_INT64
            | DeviceFeatures.SHADER_SUBGROUP_EXTENDED_TYPES
            | DeviceFeatures.STORAGE_8BIT
            | DeviceFeatures.STORAGE_16BIT
            | DeviceFeatures.DRAW_INDIRECT_COUNT
            | DeviceFeatures.MESH_SHADER
            | DeviceFeatures.FRAGMENT_SHADER_BARYCENTRIC
            | DeviceFeatures.SUBGROUP_SIZE_CONTROL
            | DeviceFeatures.IMAGE_FORMAT_LIST
            | DeviceFeatures.INDEPENDENT_BLEND
            | DeviceFeatures.SAMPLE_RATE_SHADING,
            presentation=config.window,
            force_physical_device_index=0xFFFFFFFF
            if config.force_physical_device_index is None
            else config.force_physical_device_index,
            prefer_discrete_gpu=config.prefer_discrete_gpu,
        )

        # Window
        self.window: Optional[Window] = None
        if config.window:
            self.window = Window(
                self.device,
                title,
                config.window_width,
                config.window_height,
                x=config.window_x,
                y=config.window_y,
                preferred_frames_in_flight=config.preferred_frames_in_flight,
                preferred_swapchain_usage_flags=ImageUsageFlags.COLOR_ATTACHMENT
                | ImageUsageFlags.TRANSFER_DST
                | ImageUsageFlags.TRANSFER_SRC
                | ImageUsageFlags.STORAGE,
                vsync=config.vsync,
                create_srgb_views=True,
            )
            if self.window.swapchain_srgb_format == Format.UNDEFINED:
                raise RuntimeError(
                    f"Window does not support sRGB format for output format: {self.window.swapchain_format}"
                )
            self.window.set_callbacks(
                draw=self.on_draw,
                key_event=self.on_key,
                mouse_button_event=lambda pos, button, action, modifiers: self.on_mouse_button(
                    ivec2(pos), button, action, modifiers
                ),
                mouse_move_event=lambda pos: self.on_mouse_move(ivec2(pos)),
                mouse_scroll_event=lambda pos, scroll: self.on_scroll(ivec2(pos), dvec2(scroll)),
            )

        # Headless swapchain for screenshots and videos
        output_format = Format.R8G8B8A8_UNORM if self.window is None else self.window.swapchain_format
        srgb_output_format = to_srgb_format(output_format)
        self.headless_swapchain = HeadlessSwapchain(self.device, 2, output_format, srgb_output_format)

        if self.window is not None:
            render_width, render_height = self.window.fb_width, self.window.fb_height
            num_frames_in_flight = self.window.num_frames

            # GUI
            self.gui = Gui(self.window, config.gui.default_font_size, config.gui.default_font_preference)
        else:
            render_width, render_height = config.window_width, config.window_height
            num_frames_in_flight = self.headless_swapchain.num_frames_in_flight

            # GUI
            self.gui = Gui(
                self.instance,
                self.device,
                num_frames_in_flight,
                output_format,
                config.gui.default_font_size,
                config.gui.default_font_preference,
            )

        # Add icon font
        icons_font = Path(__file__).parent.joinpath("fonts", IconsFontAwesome7.FONT_ICON_FILE_NAME_FAS_TTF)
        if icons_font.exists():
            self.gui.add_font_ttf(
                "icons-font-awesome-7", icons_font.read_bytes(), size_pixels=13.0, glyph_offset_y=2.0, merge_mode=True
            )

        self.multiviewport = config.gui.multiviewport
        self.gui.set_ini_filename(config.gui.ini_filename)

        self.gui_show_stats = config.gui.stats
        self.gui_show_inspector = config.gui.inspector
        self.gui_show_playback = config.gui.playback
        self.gui_show_renderer = config.gui.renderer
        self.gui_selected_obj: Optional[Object] = None
        self.gui_selected_gpu_property: Optional[GpuProperty[Any]] = None
        self.gui_playback_slider_held = False
        self.gui_playback_slider_current_num_frames: Optional[int] = None
        self.gui_playback_selected_property_index = -1
        self.gui_playback_zoom = 0.0
        self.gui_playback_offset_frames = 0.0

        # Disable ImGui asserts
        imgui.get_io().config_error_recovery_enable_assert = False

        # Make ImGui automatically scale fonts on DPI change events
        imgui.get_io().config_dpi_scale_fonts = True

        style = imgui.get_style()
        style.font_scale_main = config.gui.font_scale

        # Renderer
        self.renderer = Renderer(
            self.device,
            render_width,
            render_height,
            num_frames_in_flight,
            output_format,
            srgb_output_format,
            self.multiviewport,
            config.gui.max_viewport_count,
            config.renderer,
        )

        # Playback
        self.playback = Playback(config.playback)
        self.last_frame_timestamp = 0

        # Stats
        self.frame_time_index = 0
        self.frame_times = np.zeros(config.stats_frame_time_count, np.float32)

        # Scene
        self.scene = Scene("scene")

        # Widgets
        self.widgets: List[Widget] = []

        # Viewport
        if self.multiviewport:
            if config.gui.initial_number_of_viewports > config.gui.max_viewport_count:
                raise ValueError(
                    "config.gui.initial_number_of_viewports must be less than or equal to config.gui.max_viewport_count"
                )
            self.viewport_sampler = Sampler(
                self.device,
                u=SamplerAddressMode.CLAMP_TO_BORDER,
                v=SamplerAddressMode.CLAMP_TO_BORDER,
                border_color=BorderColor.FLOAT_OPAQUE_BLACK,
            )
            self.viewport_descriptor_layout, self.viewport_descriptor_pool, self.viewport_descriptor_sets = (
                create_descriptor_layout_pool_and_sets(
                    self.device,
                    [
                        DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER, stage_flags=Stage.FRAGMENT),
                    ],
                    config.gui.max_viewport_count,
                )
            )
            num_viewports = config.gui.initial_number_of_viewports
            max_viewports = config.gui.max_viewport_count
        else:
            num_viewports = 1
            max_viewports = 1

        self.viewport_scene_descriptor_pool, self.viewport_scene_descriptor_sets = create_descriptor_pool_and_sets(
            self.device,
            self.renderer.scene_descriptor_set_layout,
            max_viewports * self.renderer.num_frames_in_flight,
            name="viewport-scene-descriptor-sets",
        )
        self.viewports: List[Viewport] = []
        for viewport_index in range(min(num_viewports, max_viewports)):
            scene_descriptor_sets = RingBuffer(
                self.viewport_scene_descriptor_sets[
                    viewport_index * self.renderer.num_frames_in_flight : (viewport_index + 1)
                    * self.renderer.num_frames_in_flight
                ]
            )
            scene_uniform_buffers = RingBuffer(
                [
                    UploadableBuffer(
                        self.device, self.renderer.scene_constants_dtype.itemsize, BufferUsageFlags.UNIFORM
                    )
                    for _ in range(self.renderer.num_frames_in_flight)
                ]
            )
            scene_light_buffers = RingBuffer(
                [
                    [
                        UploadableBuffer(
                            self.device, info.size * self.renderer.max_lights_per_type, BufferUsageFlags.STORAGE
                        )
                        for info in LIGHT_TYPES_INFO
                    ]
                    for _ in range(self.renderer.num_frames_in_flight)
                ]
            )

            for s, buf, light_bufs in zip(scene_descriptor_sets, scene_uniform_buffers, scene_light_buffers):
                s.write_buffer(buf, DescriptorType.UNIFORM_BUFFER, 0)
                s.write_sampler(self.renderer.shadow_sampler, 1)
                s.write_sampler(self.renderer.linear_sampler, 2)
                s.write_image(
                    self.renderer.zero_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 3
                )
                s.write_image(
                    self.renderer.zero_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 4
                )
                s.write_image(
                    self.renderer.ggx_lut, ImageLayout.SHADER_READ_ONLY_OPTIMAL, DescriptorType.SAMPLED_IMAGE, 5
                )

                for i, light_buf in enumerate(light_bufs):
                    s.write_buffer(light_buf, DescriptorType.STORAGE_BUFFER, 6, i)
                for i in range(self.renderer.max_shadow_maps):
                    s.write_image(
                        self.renderer.zero_image,
                        ImageLayout.SHADER_READ_ONLY_OPTIMAL,
                        DescriptorType.SAMPLED_IMAGE,
                        6 + len(light_bufs),
                        i,
                    )

            scene_constants = np.zeros((1,), self.renderer.scene_constants_dtype)

            viewport_image: Optional[ViewportImage] = None
            if self.multiviewport:
                usage_flags = ImageUsageFlags.SAMPLED | ImageUsageFlags.COLOR_ATTACHMENT
                if self.renderer.supports_path_tracing:
                    usage_flags |= ImageUsageFlags.STORAGE
                img = Image(
                    self.device,
                    render_width,
                    render_height,
                    output_format,
                    usage_flags,
                    AllocType.DEVICE,
                    create_flags=ImageCreateFlags.MUTABLE_FORMAT,
                    format_list=[output_format, srgb_output_format],
                    name=f"viewport-{viewport_index}",
                )
                srgb_view = ImageView(
                    self.device,
                    img,
                    ImageViewType.TYPE_2D,
                    srgb_output_format,
                    ImageAspectFlags.COLOR,
                    usage_flags=ImageUsageFlags.COLOR_ATTACHMENT,
                )

                s = self.viewport_descriptor_sets[viewport_index]
                s.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, self.viewport_sampler, 0)
                texture = imgui.Texture(s)

                viewport_image = ViewportImage(img, srgb_view, texture)

            self.viewports.append(
                Viewport(
                    rect=Rect(0, 0, render_width, render_height),
                    playback=self.playback,
                    camera_config=config.camera,
                    handedness=config.handedness,
                    world_up=normalize(vec3(config.world_up)),
                    scene_descriptor_sets=scene_descriptor_sets,
                    scene_uniform_buffers=scene_uniform_buffers,
                    scene_light_buffers=scene_light_buffers,
                    scene_constants=scene_constants,
                    viewport_image=viewport_image,
                    name=f"viewport-{viewport_index}",
                )
            )
        self.active_viewport = self.viewports[0] if self.viewports else None

        # Server
        self.server = Server(self.on_raw_message_async, config.server)
        self.server_message_queue: Queue[Tuple[Client, RawMessage]] = Queue()
        self.raw_message_callbacks: List[Callable[[Viewer, Client, RawMessage], None]] = []

        # Config
        self.wait_events = config.wait_events

    def on_key(self, key: Key, action: Action, modifiers: Modifiers) -> None:
        if imgui.get_io().want_capture_keyboard:
            return

        # Press
        if action == Action.PRESS:
            if self.key_map.toggle_play_pause.is_active(key, modifiers):
                self.playback.toggle_play_pause()
                # If just restarted playing, reset last_frame_timestamp
                if self.playback.playing:
                    self.last_frame_timestamp = perf_counter_ns()
            if self.key_map.first_frame.is_active(key, modifiers):
                self.playback.set_frame(0)
            if self.key_map.last_frame.is_active(key, modifiers):
                self.playback.set_frame(self.playback.num_frames - 1)
            if self.key_map.exit.is_active(key, modifiers):
                self.running = False
            if self.key_map.toggle_path_tracer.is_active(key, modifiers):
                self.renderer.toggle_path_tracer()
            if self.active_viewport is not None:
                if self.key_map.ortho_view_positive_x.is_active(key, modifiers):
                    self.active_viewport.set_temporary_ortho_view(OrthographicViewType.POSITIVE_X)
                if self.key_map.ortho_view_positive_y.is_active(key, modifiers):
                    self.active_viewport.set_temporary_ortho_view(OrthographicViewType.POSITIVE_Y)
                if self.key_map.ortho_view_positive_z.is_active(key, modifiers):
                    self.active_viewport.set_temporary_ortho_view(OrthographicViewType.POSITIVE_Z)
                if self.key_map.ortho_view_negative_x.is_active(key, modifiers):
                    self.active_viewport.set_temporary_ortho_view(OrthographicViewType.NEGATIVE_X)
                if self.key_map.ortho_view_negative_y.is_active(key, modifiers):
                    self.active_viewport.set_temporary_ortho_view(OrthographicViewType.NEGATIVE_Y)
                if self.key_map.ortho_view_negative_z.is_active(key, modifiers):
                    self.active_viewport.set_temporary_ortho_view(OrthographicViewType.NEGATIVE_Z)

        # Press + repeat
        if action == Action.PRESS or action == Action.REPEAT:
            if self.key_map.next_frame.is_active(key, modifiers):
                self.playback.set_frame(self.playback.current_frame + 1)
            if self.key_map.previous_frame.is_active(key, modifiers):
                self.playback.set_frame(self.playback.current_frame - 1)

    def on_mouse_button(
        self,
        position: ivec2,
        button: MouseButton,
        action: Action,
        modifiers: Modifiers,
    ) -> None:
        if imgui.get_io().want_capture_mouse:
            return

        if self.active_viewport is not None:
            if action == Action.PRESS:
                if self.key_map.camera_rotate.is_active(button, modifiers):
                    self.active_viewport.on_rotate_press(position)
                if self.key_map.camera_pan.is_active(button, modifiers):
                    self.active_viewport.on_pan_press(position)
            if action == Action.RELEASE:
                if self.key_map.camera_rotate.button == button:
                    self.active_viewport.on_rotate_release()
                if self.key_map.camera_pan.button == button:
                    self.active_viewport.on_pan_release()

    def on_mouse_move(self, position: ivec2) -> None:
        if self.active_viewport is not None:
            self.active_viewport.on_move(position)

    def on_scroll(self, position: ivec2, scroll: dvec2) -> None:
        if imgui.get_io().want_capture_mouse:
            return

        if self.active_viewport is not None:
            # Callbacks are only called if a window is present
            assert self.window is not None
            modifiers = self.window.get_modifiers_state()

            if modifiers == self.key_map.camera_zoom_modifiers:
                self.active_viewport.on_zoom(scroll)
            if modifiers == self.key_map.camera_zoom_move_modifiers:
                self.active_viewport.on_zoom_with_movement(scroll)

    def on_resize(self, width: int, height: int) -> None:
        pass

    def _render(self, render_to_window: bool) -> None:
        # Begin frame
        if render_to_window:
            assert self.window is not None
            try:
                frame = self.window.begin_frame()
            except SwapchainOutOfDateError:
                return
            assert frame.srgb_image_view is not None
            frame_inputs = FrameInputs(
                frame.image, frame.srgb_image_view, frame.command_buffer, frame.transfer_command_buffer, [], []
            )
        else:
            frame_inputs = self.headless_swapchain.begin_frame()

        # Begin recording on command buffers
        frame_inputs.command_buffer.begin()
        if frame_inputs.transfer_command_buffer is not None:
            frame_inputs.transfer_command_buffer.begin()

        # Draw GUI
        with self.gui.frame():
            # Pre-render gui with things that can affect the rendering of this frame.
            # Especially important when running with wait_events set to True because
            # otherwise we may not see the effect of camera movement and other scene
            # edits due to events that cause a single frame of rendering
            self.on_gui()

            # Render scene
            self.renderer.render(self.scene, self.viewports, frame_inputs)

            # Other parts of the gui have to happen after rendering for multiple reasons:
            # - Widgets can rely on properties being already loaded and uploaded
            #   which happens in render, they also will be created here on first use.
            # - Renderer UI for this frame will be up to date with latest object and property state.
            #
            # TODO: User defined widgets will still have a frame of lag. Also some of the imgui interactions
            # require multiple frames after an event to take effect. In practice to fully support wait_events
            # we anyways need to introduce an "animation_playing" or "num_frames_to_render" state to render
            # more than one frame per event.
            self.on_gui_after_render()

        # HACK: If not rendering to the window, we draw the GUI twice so that
        # imgui renders all windows, including those for which the size of the contents hasn't
        # been computed yet.
        if not render_to_window:
            with self.gui.frame():
                self.on_gui()
                self.on_gui_after_render()

        # Render GUI
        self.renderer.render_gui(frame_inputs, self.gui)

        if not render_to_window:
            self.headless_swapchain.get_current().issue_readback()
        else:
            frame.command_buffer.image_barrier_full(
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

        # Submit transfer queue commands, if submitted
        if frame_inputs.transfer_command_buffer is not None:
            frame_inputs.transfer_command_buffer.end()

            # If there is no semaphore to signal, we assume there is no commands to submit.
            # Even if someone recorded commands, this would be a bug because
            # there is no way to know when those commands would complete.
            if frame_inputs.transfer_semaphores:
                self.device.transfer_queue.submit(
                    frame_inputs.transfer_command_buffer,
                    wait_semaphores=[(s.sem, s.wait_value, s.wait_stage) for s in frame_inputs.transfer_semaphores],
                    signal_semaphores=[
                        (s.sem, s.signal_value, s.signal_stage) for s in frame_inputs.transfer_semaphores
                    ],
                )

        # Close recording on the graphics commands
        frame_inputs.command_buffer.end()

        # End frame
        if render_to_window:
            assert self.window is not None
            self.window.end_frame(
                frame,
                additional_wait_semaphores=[
                    (s.sem, s.wait_value, s.wait_stage) for s in frame_inputs.additional_semaphores
                ],
                additional_signal_semaphores=[
                    (s.sem, s.signal_value, s.signal_stage) for s in frame_inputs.additional_semaphores
                ],
            )
        else:
            self.headless_swapchain.end_frame(frame_inputs)

        # Prefetch next frames
        self.renderer.prefetch()

    def _get_framebuffer_size(self) -> Tuple[int, int]:
        if self.window is not None:
            width = self.window.fb_width
            height = self.window.fb_height
        else:
            # In headless mode there is always a single viewport
            width = self.viewports[0].rect.width
            height = self.viewports[0].rect.height
        return width, height

    def render_image(self) -> NDArray[np.uint8]:
        # Set max time if not set
        if self.playback.max_time is None:
            self.playback.set_max_time(self.scene.end_animation_time(self.playback.frames_per_second))

        # Update scene
        self.scene.update(self.playback.current_time, self.playback.current_frame)

        # Get framebuffer size
        width, height = self._get_framebuffer_size()

        # Resize if needed
        self.headless_swapchain.ensure_size(width, height)

        # TODO: I think in the headless case we also need to resize the renderer
        # here, for things like depth buffer and msaa targets. Double check this
        # by making a test that renders in headless mode at different resolutions
        # on the same viewer.

        io = imgui.get_io()
        io.display_size = imgui.Vec2(width, height)
        io.delta_time = 1.0 / 60.0

        # Render to headless swapchain
        self._render(False)

        # Get current headless swapchain frame
        frame = self.headless_swapchain.get_current_and_advance()

        # Readback frame
        return frame.realize_readback()

    def render_video_playback_range(
        self, on_frame: Callable[[NDArray[np.uint8], int, float], bool], start_time: float, end_time: float
    ) -> None:
        # Set max time if not set
        if self.playback.max_time is None:
            self.playback.set_max_time(self.scene.end_animation_time(self.playback.frames_per_second))

        # Get framebuffer size
        width, height = self._get_framebuffer_size()

        # Resize if needed
        self.headless_swapchain.ensure_size(width, height)

        io = imgui.get_io()
        io.display_size = imgui.Vec2(width, height)
        io.delta_time = 1.0 / 60.0

        # Readback queue for pipelining rendering and readback.
        readback_queue: Queue[Tuple[HeadlessSwapchainFrame, int, float]] = Queue()

        def deque_frame() -> None:
            old_frame, frame_index, time = readback_queue.get()
            img = old_frame.realize_readback()
            on_frame(img, frame_index, time)

        self.playback.set_time(start_time)

        while self.playback.current_time < end_time:
            if readback_queue.qsize() >= self.headless_swapchain.num_frames_in_flight:
                deque_frame()

            # Update scene
            self.scene.update(self.playback.current_time, self.playback.current_frame)

            # Render to headless swapchain
            self._render(False)

            # Get current headless swapchain frame
            frame = self.headless_swapchain.get_current_and_advance()

            # Enqueue for future readback
            readback_queue.put((frame, self.playback.current_frame, self.playback.current_time))

            if self.playback.current_frame == self.playback.num_frames - 1:
                break
            self.playback.set_frame(self.playback.current_frame + 1)

        # Drain readback queue
        while not readback_queue.empty():
            deque_frame()

    def render_video(self, on_frame: Callable[[NDArray[np.uint8]], bool]) -> None:
        # Set max time if not set
        if self.playback.max_time is None:
            self.playback.set_max_time(self.scene.end_animation_time(self.playback.frames_per_second))

        begin_frame_index = 0
        end_frame_index = begin_frame_index + self.playback.num_frames

        # Get framebuffer size
        width, height = self._get_framebuffer_size()

        # Resize if needed
        self.headless_swapchain.ensure_size(width, height)

        io = imgui.get_io()
        io.display_size = imgui.Vec2(width, height)
        io.delta_time = 1.0 / 60.0

        # Readback queue for pipelining rendering and readback.
        readback_queue: Queue[HeadlessSwapchainFrame] = Queue()

        def deque_frame() -> None:
            old_frame = readback_queue.get()
            img = old_frame.realize_readback()
            on_frame(img)

        for frame_index in range(begin_frame_index, end_frame_index):
            if readback_queue.qsize() >= self.headless_swapchain.num_frames_in_flight:
                deque_frame()

            self.playback.set_frame(frame_index)

            # Update scene
            self.scene.update(self.playback.current_time, self.playback.current_frame)

            # Render to headless swapchain
            self._render(False)

            # Get current headless swapchain frame
            frame = self.headless_swapchain.get_current_and_advance()

            # Enqueue for future readback
            readback_queue.put(frame)

        # Drain readback queue
        while not readback_queue.empty():
            deque_frame()

    def on_draw(self) -> None:
        # Callbacks are only called if a window is present
        assert self.window is not None

        # Compute dt
        timestamp = perf_counter_ns()
        dt = (timestamp - self.last_frame_timestamp) * 1e-9
        self.last_frame_timestamp = timestamp

        self.frame_times[self.frame_time_index] = dt
        self.frame_time_index = (self.frame_time_index + 1) % self.frame_times.size

        # Set max time if not set
        if self.playback.max_time is None:
            self.playback.set_max_time(self.scene.end_animation_time(self.playback.frames_per_second))

        # Step dt
        if self.playback.playing and not self.gui_playback_slider_held:
            self.playback.step(dt)

        # Update scene
        self.scene.update(self.playback.current_time, self.playback.current_frame)

        # Resize
        swapchain_status = self.window.update_swapchain()
        if swapchain_status == SwapchainStatus.MINIMIZED:
            return
        if swapchain_status == SwapchainStatus.RESIZED:
            width, height = self.window.fb_width, self.window.fb_height

            self.renderer.resize(width, height)

            if not self.multiviewport:
                # In the multi-viewport case rect resizing is handled as part of
                # the GUI update.
                self.viewports[0].resize(width, height)
            else:
                for viewport_index, viewport in enumerate(self.viewports):
                    assert viewport.viewport_image is not None
                    viewport.viewport_image.image.destroy()
                    viewport.viewport_image.srgb_image_view.destroy()

                    usage_flags = ImageUsageFlags.SAMPLED | ImageUsageFlags.COLOR_ATTACHMENT
                    if self.renderer.supports_path_tracing:
                        usage_flags |= ImageUsageFlags.STORAGE
                    img = Image(
                        self.device,
                        width,
                        height,
                        self.renderer.output_format,
                        usage_flags,
                        AllocType.DEVICE,
                        create_flags=ImageCreateFlags.MUTABLE_FORMAT,
                        format_list=[self.renderer.output_format, self.renderer.srgb_output_format],
                        name=f"viewport-{viewport_index}",
                    )
                    srgb_view = ImageView(
                        self.device,
                        img,
                        ImageViewType.TYPE_2D,
                        self.renderer.srgb_output_format,
                        ImageAspectFlags.COLOR,
                        usage_flags=ImageUsageFlags.COLOR_ATTACHMENT,
                    )

                    s = self.viewport_descriptor_sets[viewport_index]
                    s.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, self.viewport_sampler, 0)

                    viewport.viewport_image.image = img
                    viewport.viewport_image.srgb_image_view = srgb_view

            self.on_resize(width, height)

        # Render to window
        self._render(True)

    def on_raw_message_async(self, client: Client, raw_message: RawMessage) -> None:
        self.server_message_queue.put((client, raw_message))
        if self.window is not None:
            self.window.post_empty_event()

    def on_raw_message(self, client: Client, raw_message: RawMessage) -> None:
        # Call into registered callbacks
        for callback in self.raw_message_callbacks:
            callback(self, client, raw_message)

        message = parse_builtin_messages(raw_message)
        if message is not None:
            self.on_message(client, message)

    def on_message(self, client: Client, message: Message) -> None:
        pass

    def gui_stats(self) -> None:
        imgui.set_next_window_pos(imgui.Vec2(10, 10))
        imgui.set_next_window_bg_alpha(0.3)
        if imgui.begin(
            "Stats",
            flags=imgui.WindowFlags.NO_TITLE_BAR
            | imgui.WindowFlags.ALWAYS_AUTO_RESIZE
            | imgui.WindowFlags.NO_RESIZE
            | imgui.WindowFlags.NO_SAVED_SETTINGS
            | imgui.WindowFlags.NO_MOVE
            | imgui.WindowFlags.NO_FOCUS_ON_APPEARING
            | imgui.WindowFlags.NO_NAV
            | imgui.WindowFlags.NO_MOUSE_INPUTS
            | imgui.WindowFlags.NO_SCROLLBAR,
        )[0]:
            avg_dt = self.frame_times.mean()
            avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            last_dt = self.frame_times[self.frame_time_index]
            last_fps = 1.0 / last_dt if last_dt > 0 else 0.0
            imgui.text(f"{self.device.device_properties.device_name}")
            if self.window is not None:
                imgui.text(f"Window size:     [{self.window.fb_width}x{self.window.fb_height}]")
            imgui.text(f"FPS:             {avg_fps:6.2f} ({last_fps:6.2f})")
            imgui.text(f"Frame time (ms): {avg_dt * 1000.0:6.2f} ({last_dt * 1000.0:6.2f})")
            for i, (heap, stats) in enumerate(
                zip(self.device.memory_properties.memory_heaps, self.device.heap_statistics)
            ):
                if heap.flags & MemoryHeapFlags.VK_MEMORY_HEAP_DEVICE_LOCAL:
                    kind = "GPU"
                else:
                    kind = "CPU"
                imgui.text(
                    f"Heap {i} - {kind}: {stats.block_bytes / (1024 * 1024 * 1024):4.02f} / {heap.size / (1024 * 1024 * 1024):4.02f} GB"
                )
        imgui.end()

    def gui_playback(self) -> None:
        if imgui.begin("Playback", flags=imgui.WindowFlags.NO_SCROLLBAR)[0]:
            target_ticks_spacing_px = 40
            min_ticks_spacing_frames = 1
            text_height = 15
            char_width = 6

            style = imgui.get_style()
            imgui.push_style_var(imgui.StyleVar.FRAME_ROUNDING, 2.0)
            w = imgui.get_content_region_avail().x
            button_size_x = 25
            button_size_y = 25
            button_w = button_size_x * 5
            imgui.set_cursor_pos_x(w * 0.3 - button_w * 0.5)
            if imgui.button(IconsFontAwesome7.ICON_BACKWARD_FAST, size=(button_size_x, button_size_y)):
                self.playback.set_frame(0)
            imgui.same_line(spacing=0)
            if imgui.button(IconsFontAwesome7.ICON_BACKWARD_STEP, size=(button_size_x, button_size_y)):
                self.playback.set_frame(self.playback.current_frame - 1)
            imgui.same_line(spacing=0)

            pushed = False
            if self.playback.playing:
                imgui.push_style_color_im_vec4(imgui.Col.BUTTON, style.colors(imgui.Col.BUTTON_ACTIVE))
                pushed = True
            if imgui.button(
                f"{IconsFontAwesome7.ICON_PAUSE if self.playback.playing else IconsFontAwesome7.ICON_PLAY}###play",
                size=(button_size_x, button_size_y),
            ):
                self.playback.playing = not self.playback.playing
            if pushed:
                imgui.pop_style_color()

            imgui.same_line(spacing=0)
            if imgui.button(IconsFontAwesome7.ICON_FORWARD_STEP, size=(button_size_x, button_size_y)):
                self.playback.set_frame(self.playback.current_frame + 1)
            imgui.same_line(spacing=0)
            if imgui.button(IconsFontAwesome7.ICON_FORWARD_FAST, size=(button_size_x, button_size_y)):
                self.playback.set_frame(self.playback.num_frames - 1)

            imgui.same_line(spacing=10)

            pushed = False
            if self.playback.lock_to_last_frame:
                imgui.push_style_color_im_vec4(imgui.Col.BUTTON, style.colors(imgui.Col.BUTTON_ACTIVE))
                pushed = True
            if imgui.button(
                f"{IconsFontAwesome7.ICON_LOCK if self.playback.lock_to_last_frame else IconsFontAwesome7.ICON_LOCK_OPEN}###lock",
                size=(button_size_x, button_size_y),
            ):
                self.playback.lock_to_last_frame = not self.playback.lock_to_last_frame
            if pushed:
                imgui.pop_style_color()

            imgui.same_line(spacing=20)
            fw = min((w - (button_size_x * 6 + 30)), char_width * 12)
            imgui.set_next_item_width(fw)
            _, self.playback.playback_speed_multiplier = imgui.drag_float(
                "###speed", self.playback.playback_speed_multiplier, 0.01, 0.01, 10.0, format="%.2fx"
            )

            imgui.same_line(spacing=30)
            imgui.set_next_item_width(fw)
            imgui.text(f"{self.playback.frames_per_second} fps")
            imgui.same_line(spacing=30)
            imgui.text(f"{self.playback.current_time:7.3f} / {self.playback.max_time: 7.3f} s")
            imgui.pop_style_var()

            imgui.begin_child("Left", (150, 0), imgui.ChildFlags.RESIZE_X, imgui.WindowFlags.NO_SCROLLBAR)
            imgui.separator_text("Properties")
            imgui.set_cursor_pos(imgui.Vec2(0, 20))
            child_size = imgui.get_content_region_avail()
            imgui.begin_child(
                "Properties", imgui.Vec2(child_size.x, child_size.y - text_height), 0, 0
            )  # , imgui.WindowFlags.NO_SCROLLBAR)
            scroll = imgui.get_scroll_y()

            # TODO: compute this only once per frame
            props = self.scene.collect_dynamic_properties()
            for i, p in enumerate(props):
                if imgui.selectable(
                    f"{p.name}###{i}",
                    i == self.gui_playback_selected_property_index,
                    imgui.SelectableFlags.ALLOW_DOUBLE_CLICK,
                ):
                    if self.gui_playback_selected_property_index == i:
                        self.gui_playback_selected_property_index = -1
                    else:
                        self.gui_playback_selected_property_index = i
            imgui.end_child()
            imgui.end_child()

            imgui.same_line()
            current_bg_color = style.colors(imgui.Col.TAB)
            current_bg_color = float4_to_uint32((current_bg_color.x, current_bg_color.y, current_bg_color.z, 1.0))
            current_separator_color = imgui.get_style().colors(imgui.Col.SEPARATOR)
            current_separator_color = float4_to_uint32(
                (current_separator_color.x, current_separator_color.y, current_separator_color.z, 1.0)
            )
            current_text_color = 0xFFFFFFFF
            current_line_color = 0xFFFFFFFF
            text_color = 0xFFBABABA
            bg_color = 0xFF303030
            tick_color = 0xFF000000
            ts_color = text_color
            ts_selected_color = current_text_color

            list = imgui.get_window_draw_list()

            pos = imgui.get_cursor_screen_pos()
            size = imgui.get_content_region_avail()

            cursor_rect_min = imgui.Vec2(pos.x - 5, pos.y)
            cursor_rect_max = imgui.Vec2(pos.x + size.x + 5, pos.y + size.y)

            pos.x += 10
            size.x -= 30

            imgui.set_cursor_screen_pos(pos)
            imgui.begin_child("Timeline", size, 0, imgui.WindowFlags.NO_SCROLLBAR)
            imgui.end_child()

            # Make space for bottom labels
            size.y -= text_height

            if size.x > 0 and size.y > 0:
                io = imgui.get_io()

                # Mouse button state management
                if imgui.is_item_clicked(imgui.MouseButton.LEFT):
                    self.gui_playback_slider_held = True
                    self.gui_playback_slider_current_num_frames = self.playback.num_frames
                    self.playback.lock_to_last_frame = False
                elif not imgui.is_mouse_down(imgui.MouseButton.LEFT):
                    self.gui_playback_slider_held = False
                    self.gui_playback_slider_current_num_frames = None

                if imgui.is_item_clicked(imgui.MouseButton.RIGHT):
                    self.gui_playback_dragging = True
                    self.gui_playback_drag_start_offset_frames = self.gui_playback_offset_frames
                    self.gui_playback_drag_start_mouse_position_px = io.mouse_pos.x
                elif not imgui.is_mouse_down(imgui.MouseButton.RIGHT):
                    self.gui_playback_dragging = False
                    self.gui_playback_drag_start_offset_frames = None

                # Init range and zoom
                mouse_x_px = io.mouse_pos.x - pos.x
                range_frames = max((self.playback.num_frames - 1), 1)
                scale_frames_from_px = 2**self.gui_playback_zoom / size.x * range_frames

                # Zooming
                if imgui.is_item_hovered() and io.mouse_wheel != 0:
                    mouse_before_frames = self.gui_playback_offset_frames + (scale_frames_from_px * mouse_x_px)

                    self.gui_playback_zoom -= io.mouse_wheel * 0.2
                    max_zoom = np.log2(range_frames)
                    self.gui_playback_zoom = np.clip(np.round(self.gui_playback_zoom * 1000) * 0.001, -max_zoom, 0)

                    scale_frames_from_px = 2**self.gui_playback_zoom / size.x * range_frames
                    mouse_after_frames = self.gui_playback_offset_frames + (scale_frames_from_px * mouse_x_px)

                    # Keep cursor position stable
                    self.gui_playback_offset_frames += mouse_after_frames - mouse_before_frames

                scale_px_from_frames = 1 / scale_frames_from_px

                # Dragging timeline
                if self.gui_playback_dragging:
                    delta_px = io.mouse_pos.x - self.gui_playback_drag_start_mouse_position_px
                    self.gui_playback_offset_frames = (
                        self.gui_playback_drag_start_offset_frames + scale_frames_from_px * delta_px
                    )

                frames_in_view = scale_frames_from_px * size.x
                frames_outside = range_frames - frames_in_view
                self.gui_playback_offset_frames = np.clip(self.gui_playback_offset_frames, -frames_outside, 0)

                # Dragging frame cursor
                if self.gui_playback_slider_held:
                    frame = np.clip(
                        int(np.round(scale_frames_from_px * mouse_x_px - self.gui_playback_offset_frames)),
                        0,
                        self.playback.num_frames - 1,
                    )
                    self.playback.set_frame(frame)

                # Ticks
                def next_power_of_ten(v: float):
                    return 10 ** (np.floor(np.log10(v)) + 1)

                def num_digits(n: int):
                    if n == 0:
                        return 1
                    return int(np.log10(abs(n))) + 1

                def next_multiple_down(v: float, base: int) -> int:
                    return (int(v) // base) * base

                def next_multiple_up(v: float, base: int) -> int:
                    return (int(np.ceil(v) + base - 1) // base) * base

                desired_ticks_count = scale_frames_from_px * target_ticks_spacing_px
                rounded_ticks_spacing_frames = next_power_of_ten(desired_ticks_count)
                ticks_spacing_frames = max(min_ticks_spacing_frames, rounded_ticks_spacing_frames)

                # Skip top ticks text line
                pos.y += text_height
                size.y -= text_height

                first_visible_frame = -self.gui_playback_offset_frames
                last_visible_frame = -self.gui_playback_offset_frames + scale_frames_from_px * size.x
                first_visible_tick = next_multiple_up(first_visible_frame, ticks_spacing_frames)
                last_visible_tick = next_multiple_down(last_visible_frame, ticks_spacing_frames) + 1

                ticks = np.arange(first_visible_tick, last_visible_tick, ticks_spacing_frames)
                ticks_min = np.empty((ticks.size, 2), np.float32)
                ticks_max = np.empty((ticks.size, 2), np.float32)
                ticks_min[:, 0] = pos.x + scale_px_from_frames * (ticks + self.gui_playback_offset_frames)
                ticks_min[:, 1] = pos.y
                ticks_max[:, 0] = ticks_min[:, 0] + 1
                ticks_max[:, 1] = ticks_min[:, 1] + size.y

                for t, t_min, t_max in zip(ticks, ticks_min, ticks_max):
                    # Top ticks text (frames)
                    list.add_text(
                        imgui.Vec2(t_min[0] - char_width * (num_digits(t) * 0.5), t_min[1] - text_height),
                        text_color,
                        f"{int(t)}",
                    )

                    # Bottom ticks text (timestamps)
                    ts = t / self.playback.frames_per_second
                    list.add_text(
                        imgui.Vec2(t_min[0] - char_width * ((num_digits(int(ts)) + 2) * 0.5), t_max[1]),
                        text_color,
                        f"{ts:.1f}",
                    )

                # Draw frame cursor text and text bg
                list.push_clip_rect(cursor_rect_min, cursor_rect_max)
                current_min = imgui.Vec2(
                    pos.x + scale_px_from_frames * (self.playback.current_frame + self.gui_playback_offset_frames),
                    pos.y,
                )
                current_max = imgui.Vec2(current_min.x + 1, pos.y + size.y)

                text_width = char_width * max(3, num_digits(self.playback.current_frame) + 1)
                text_pos = imgui.Vec2(current_min.x - text_width * 0.5, current_min.y - text_height)
                text_min = imgui.Vec2(text_pos.x - 5, text_pos.y)
                text_max = imgui.Vec2(current_min.x + text_width * 0.5 + 7, current_min.y - 2)
                list.add_rect_filled(text_min, text_max, current_bg_color, 3.0)

                time_text_width = char_width * (num_digits(int(self.playback.current_time)) + 4)
                time_text_pos = imgui.Vec2(current_min.x - time_text_width * 0.5, current_min.y)
                time_text_min = imgui.Vec2(time_text_pos.x - 5, time_text_pos.y + size.y)
                time_text_max = imgui.Vec2(
                    current_min.x + time_text_width * 0.5 + char_width * 0.5,
                    current_min.y - 2 + size.y + text_height,
                )
                list.add_rect_filled(time_text_min, time_text_max, current_bg_color, 3.0)

                text_width = char_width * num_digits(self.playback.current_frame)
                text_pos = imgui.Vec2(current_min.x - text_width * 0.5, current_min.y - text_height)
                list.add_text(text_pos, current_text_color, f"{int(self.playback.current_frame)}")

                time_text_pos = imgui.Vec2(time_text_min.x + char_width, time_text_min.y)
                list.add_text(time_text_pos, current_text_color, f"{self.playback.current_time:.2f}")
                list.pop_clip_rect()

                if size.x > 0 and size.y > 0:
                    # Draw body
                    rect_min = imgui.Vec2(pos.x - 1, pos.y)
                    rect_max = imgui.Vec2(pos.x + size.x + 1, pos.y + size.y)
                    list.push_clip_rect(rect_min, rect_max)
                    list.add_rect_filled(rect_min, rect_max, bg_color)

                    # Draw tick lines
                    list.add_rect_filled_batch(ticks_min, ticks_max, np.array([tick_color]), np.array([0]))

                    # Draw subticks (partial lines only on top bottom at 1 / 10 spacing)
                    if ticks_spacing_frames != 1:
                        subticks = np.arange(first_visible_tick, last_visible_tick, ticks_spacing_frames // 10)
                        subticks_min = np.empty((subticks.size, 2), np.float32)
                        subticks_max = np.empty((subticks.size, 2), np.float32)
                        subticks_min[:, 0] = pos.x + scale_px_from_frames * (
                            subticks + self.gui_playback_offset_frames
                        )
                        subticks_min[:, 1] = pos.y
                        subticks_max[:, 0] = subticks_min[:, 0] + 1
                        subticks_max[:, 1] = subticks_min[:, 1] + size.y * 0.1
                        list.add_rect_filled_batch(subticks_min, subticks_max, np.array([tick_color]), np.array([0]))

                        subticks_min[:, 1] = pos.y + size.y * 0.9
                        subticks_max[:, 1] = subticks_min[:, 1] + size.y
                        list.add_rect_filled_batch(subticks_min, subticks_max, np.array([tick_color]), np.array([0]))

                    # Draw frame cursor line
                    list.add_rect_filled(current_min, current_max, current_line_color)

                    # Draw property frame timestamps indicators
                    for y, p in enumerate(props):
                        ts = p.get_timestamps(self.playback.frames_per_second, first_visible_frame, last_visible_frame)
                        x = scale_px_from_frames * ts * self.playback.frames_per_second

                        center = np.empty((ts.size, 2), np.float32)
                        center[:, 0] = pos.x + x + 0.5 + scale_px_from_frames * self.gui_playback_offset_frames
                        center[:, 1] = pos.y + (y * (text_height + 2)) + text_height * 0.5 + 3 - scroll
                        list.add_circle_filled_batch(
                            center,
                            np.array([3]),
                            np.array(
                                [ts_color if y != self.gui_playback_selected_property_index else ts_selected_color]
                            ),
                            4,
                        )
                    list.pop_clip_rect()

        else:
            self.gui_playback_slider_current_num_frames = None
            self.gui_playback_slider_held = False

        imgui.end()

    def gui_inspector(self) -> None:
        if imgui.begin("Inspector")[0]:
            for i, v in enumerate(self.viewports):
                if imgui.tree_node_ex(f"Camera - {v.name}##{i}" if self.multiviewport else "Camera"):
                    imgui.begin_disabled()
                    imgui.drag_float3(f"Position##{i}", tuple(v.camera.position()))
                    right, up, front = v.camera.right_up_front()
                    imgui.drag_float3(f"Right##{i}", tuple(right))
                    imgui.drag_float3(f"Up##{i}", tuple(up))
                    imgui.drag_float3(f"Front##{i}", tuple(front))
                    imgui.end_disabled()
                    imgui.tree_pop()
            imgui.separator()

            def pre(o: Object) -> bool:
                flags = imgui.TreeNodeFlags.OPEN_ON_ARROW | imgui.TreeNodeFlags.FRAME_PADDING
                if o.gui_expanded:
                    flags |= imgui.TreeNodeFlags.DEFAULT_OPEN
                if self.gui_selected_obj == o:
                    flags |= imgui.TreeNodeFlags.SELECTED
                if not o.children:
                    flags |= imgui.TreeNodeFlags.LEAF
                    flags |= imgui.TreeNodeFlags.BULLET

                imgui.push_style_var_im_vec2(imgui.StyleVar.FRAME_PADDING, imgui.Vec2(0, 3))
                o.gui_expanded = imgui.tree_node_ex(f"{o.name}##tree_node_{o.uid}", flags)
                imgui.pop_style_var()

                if imgui.is_item_clicked():
                    if self.gui_selected_obj == o:
                        self.gui_selected_obj = None
                    else:
                        self.gui_selected_obj = o

                imgui.same_line(imgui.get_content_region_max().x - 25)
                _, o.gui_enabled = imgui.checkbox(f"##enabled_{o.uid}", o.gui_enabled)

                return o.gui_expanded

            def post(o: Object) -> None:
                imgui.tree_pop()

            self.scene.visit_objects_pre_post(pre, post)

            imgui.separator()

            if self.gui_selected_obj is not None:
                self.gui_selected_obj.gui()

        imgui.end()

    def gui_renderer(self) -> None:
        if imgui.begin("Renderer")[0]:
            imgui.text("GPU properties:")
            imgui.indent(5)
            for i, p in enumerate(self.renderer.enabled_gpu_properties):
                s = imgui.selectable(f"{p.name}##{i}", self.gui_selected_gpu_property == p)
                if s:
                    if self.gui_selected_gpu_property == p:
                        self.gui_selected_gpu_property = None
                    else:
                        self.gui_selected_gpu_property = p
            imgui.indent(-5)

            imgui.separator()
            if self.gui_selected_gpu_property is not None:
                imgui.separator_text(type(self.gui_selected_gpu_property).__name__)
                if isinstance(self.gui_selected_gpu_property, GpuPreuploadedArrayProperty):
                    imgui.text(f"Frame size: {self.gui_selected_gpu_property.frame_size}")
                    imgui.text(f"Upload method: {self.gui_selected_gpu_property.upload_method}")
                    inv = ", ".join([str(i) for i in self.gui_selected_gpu_property.invalid_frames])
                    imgui.text(f"Invalid frames: {{{inv}}}")
                    imgui.text("Resource:")
                    imgui.indent()
                    imgui.text(f"{self.gui_selected_gpu_property.resource}")
                    imgui.text(f"{self.gui_selected_gpu_property.resource.alloc}")
                    imgui.unindent()
                if isinstance(self.gui_selected_gpu_property, GpuPreuploadedProperty):
                    imgui.text(f"Frame size: {self.gui_selected_gpu_property.max_frame_size}")
                    imgui.text(f"Batched: {self.gui_selected_gpu_property.batched}")
                    imgui.text(f"Async load: {self.gui_selected_gpu_property.async_load}")
                    imgui.text(f"Upload method: {self.gui_selected_gpu_property.upload_method}")
                    inv = ", ".join([str(i) for i in self.gui_selected_gpu_property.invalid_frames])
                    imgui.text(f"Invalid frames: {{{inv}}}")
                    imgui.text(f"Resources: {len(self.gui_selected_gpu_property.resources)}")
                    if isinstance(self.gui_selected_gpu_property.current, GpuBufferView):
                        imgui.indent()
                        imgui.text(f"{self.gui_selected_gpu_property.current.buffer}")
                        imgui.unindent()
                    elif isinstance(self.gui_selected_gpu_property.current, GpuImageView):
                        imgui.indent()
                        imgui.text(f"{self.gui_selected_gpu_property.current.image}")
                        imgui.unindent()
                elif isinstance(self.gui_selected_gpu_property, GpuStreamingProperty):
                    imgui.text(f"Upload method: {self.gui_selected_gpu_property.upload_method}")
                    imgui.text("CPU Resource:")
                    imgui.indent()
                    imgui.text(f"{self.gui_selected_gpu_property.cpu_buffers[0].buf}")
                    imgui.text(f"{self.gui_selected_gpu_property.cpu_buffers[0].buf.alloc}")
                    imgui.unindent()
                    imgui.text("GPU Resource:")
                    if self.gui_selected_gpu_property.gpu_resources:
                        imgui.indent()
                        v = self.gui_selected_gpu_property.gpu_resources[0].resource
                        imgui.text(f"{v}")
                        if isinstance(v, GpuBufferView):
                            imgui.text(f"{v.buffer.alloc}")
                        elif isinstance(v, GpuImageView):
                            imgui.text(f"{v.image.alloc}")
                        imgui.unindent()

                    def drawpool(name: str, pool: Optional[LRUPool[int, Any]], count: int) -> None:
                        if pool is None:
                            return
                        imgui.separator_text(name)
                        imgui.text("Map")
                        imgui.indent()
                        for lu_k, lu_v in pool.lookup.items():
                            imgui.text(f"{lu_k} {lu_v}")
                        imgui.unindent()

                        imgui.text("LRU")
                        imgui.indent()
                        i = 0
                        for lru_k, lru_v in pool.lru.items():
                            imgui.text(f"{lru_k} {lru_v}")
                            i += 1
                        for _ in range(i, count):
                            imgui.text("<EMPTY>")
                        imgui.unindent()

                        imgui.text("In Flight")
                        imgui.indent()
                        for if_v in pool.in_flight:
                            imgui.text(f"{if_v}")
                        imgui.unindent()

                        imgui.text("Prefetching")
                        imgui.indent()
                        i = 0
                        for pre_k, pre_v in pool.prefetch_store.items():
                            imgui.text(f"{pre_k} {pre_v}")
                            i += 1
                        for _ in range(i, pool.max_prefetch):
                            imgui.text("<EMPTY>")
                        imgui.unindent()

                        imgui.text("Generation indices")
                        imgui.indent()
                        for gen_k, gen_index in pool.current_generation.items():
                            imgui.text(f"{gen_k}: {gen_index}")
                        imgui.unindent()

                    drawpool(
                        "CPU",
                        self.gui_selected_gpu_property.cpu_pool,
                        len(self.gui_selected_gpu_property.cpu_buffers),
                    )
                    drawpool(
                        "GPU",
                        self.gui_selected_gpu_property.gpu_pool,
                        len(self.gui_selected_gpu_property.gpu_resources),
                    )

                    if self.gui_selected_gpu_property.cpu_pool or self.gui_selected_gpu_property.gpu_pool:
                        imgui.separator_text("Frame states")
                        imgui.spacing()

                        start = imgui.get_cursor_screen_pos()
                        dl = imgui.get_window_draw_list()

                        num_frames = self.gui_selected_gpu_property.property.num_frames
                        current_frame = self.gui_selected_gpu_property.property.current_frame_index

                        p_min = np.empty((num_frames, 2), np.float32)
                        p_max = np.empty((num_frames, 2), np.float32)
                        delta_x = 5 * np.arange(num_frames, dtype=np.float32)
                        p_min[:, 0] = start.x + delta_x
                        p_min[:, 1] = start.y
                        p_max[:, 0] = (start.x + 6) + delta_x
                        p_max[:, 1] = start.y + 20
                        dl.add_rect_batch(
                            p_min,
                            p_max,
                            np.array((0xFFFFFFFF,), np.uint32),
                            np.array((0.0,), np.float32),
                            np.array((1.0,), np.float32),
                        )

                        for (
                            (c_k, _),
                            c_v,
                        ) in self.gui_selected_gpu_property.cpu_pool.lookup.items():
                            cursor = imgui.Vec2(start.x + 5 * c_k, start.y)
                            color = 0xFF00FF00 if c_k >= current_frame else 0xFF0000FF
                            color = color if not c_v.prefetching else 0xFF00FFFF
                            dl.add_rect_filled(
                                imgui.Vec2(cursor.x + 1, cursor.y + 1),
                                imgui.Vec2(cursor.x + 5, cursor.y + 19),
                                color,
                            )
                        imgui.spacing()
                        imgui.spacing()
                        imgui.spacing()
                        imgui.spacing()
                        imgui.spacing()

                        if self.gui_selected_gpu_property.gpu_pool:
                            p_min[:, 1] = start.y + 22
                            p_max[:, 1] = start.y + 42
                            dl.add_rect_batch(
                                p_min,
                                p_max,
                                np.array((0xFFFFFFFF,), np.uint32),
                                np.array((0.0,), np.float32),
                                np.array((1.0,), np.float32),
                            )

                            for (
                                (g_k, _),
                                g_v,
                            ) in self.gui_selected_gpu_property.gpu_pool.lookup.items():
                                cursor = imgui.Vec2(start.x + 5 * g_k, start.y)
                                color = 0xFF00FF00 if g_k >= current_frame else 0xFF0000FF
                                color = color if not g_v.prefetching else 0xFF00FFFF
                                dl.add_rect_filled(
                                    imgui.Vec2(cursor.x + 1, cursor.y + 23),
                                    imgui.Vec2(cursor.x + 5, cursor.y + 41),
                                    color,
                                )
                            imgui.spacing()
                            imgui.spacing()
                            imgui.spacing()
                            imgui.spacing()
                            imgui.spacing()
        imgui.end()

    def on_gui(self) -> None:
        imgui.dock_space_over_viewport(flags=imgui.DockNodeFlags.PASSTHRU_CENTRAL_NODE)

        if self.multiviewport:
            imgui.push_style_var_im_vec2(imgui.StyleVar.WINDOW_PADDING, imgui.Vec2(0.0, 0.0))
            fb_width, fb_height = self._get_framebuffer_size()

            io = imgui.get_io()
            imgui_modifiers = io.key_mods

            mapped_mods = (
                (Modifiers.SHIFT if imgui_modifiers & imgui.Key.MOD_SHIFT else 0)
                | (Modifiers.CTRL if imgui_modifiers & imgui.Key.MOD_CTRL else 0)
                | (Modifiers.ALT if imgui_modifiers & imgui.Key.MOD_ALT else 0)
                | (Modifiers.SUPER if imgui_modifiers & imgui.Key.MOD_SUPER else 0)
            )
            mouse_button_map = {
                MouseButton.LEFT: imgui.MouseButton.LEFT,
                MouseButton.RIGHT: imgui.MouseButton.RIGHT,
                MouseButton.MIDDLE: imgui.MouseButton.MIDDLE,
            }
            rotate_button = mouse_button_map[self.key_map.camera_rotate.button]
            pan_button = mouse_button_map[self.key_map.camera_pan.button]

            prevent_left_button_window_move = (
                self.key_map.camera_rotate.button == MouseButton.LEFT
                and self.key_map.camera_rotate.mods == Modifiers.NONE
            ) or (
                self.key_map.camera_pan.button == MouseButton.LEFT and self.key_map.camera_pan.mods == Modifiers.NONE
            )

            if prevent_left_button_window_move:
                old_move_from_title_bar_only_config = io.config_windows_move_from_title_bar_only
                io.config_windows_move_from_title_bar_only = True

            output_width, output_height = self._get_framebuffer_size()

            for v in self.viewports:
                assert v.viewport_image is not None

                imgui.set_next_window_size_constraints(
                    imgui.Vec2(0, 0), imgui.Vec2(v.viewport_image.image.width, v.viewport_image.image.height)
                )
                if imgui.begin(v.name)[0]:
                    cursor_pos = imgui.get_cursor_screen_pos()
                    pos = ivec2(cursor_pos.x, cursor_pos.y)
                    avail = imgui.get_content_region_avail()
                    size = ivec2(avail.x, avail.y)
                    v.rect.x = pos.x
                    v.rect.y = pos.y
                    v.resize(min(size.x, output_width), min(size.y, output_height))
                    imgui.image(
                        v.viewport_image.imgui_texture,
                        avail,
                        imgui.Vec2(0, 0),
                        imgui.Vec2(size.x / fb_width, size.y / fb_height),
                    )

                    # Rotate
                    if imgui.is_item_clicked(rotate_button) and mapped_mods == self.key_map.camera_rotate.mods:
                        self.active_viewport = v
                        mouse_pos = imgui.get_mouse_pos()
                        v.on_rotate_press(ivec2(mouse_pos.x, mouse_pos.y))
                    elif not imgui.is_mouse_down(rotate_button):
                        v.on_rotate_release()

                    # Pan
                    if imgui.is_item_clicked(pan_button) and mapped_mods == self.key_map.camera_pan.mods:
                        self.active_viewport = v
                        mouse_pos = imgui.get_mouse_pos()
                        v.on_pan_press(ivec2(mouse_pos.x, mouse_pos.y))
                    elif not imgui.is_mouse_down(pan_button):
                        v.on_pan_release()

                    # Zoom
                    if imgui.is_item_hovered():
                        if mapped_mods == self.key_map.camera_zoom_modifiers:
                            v.on_zoom(dvec2(0, io.mouse_wheel))
                        elif mapped_mods == self.key_map.camera_zoom_move_modifiers:
                            v.on_zoom_with_movement(dvec2(0, io.mouse_wheel))
                else:
                    v.rect.width = 0
                    v.rect.height = 0
                imgui.end()

            if prevent_left_button_window_move:
                io.config_windows_move_from_title_bar_only = old_move_from_title_bar_only_config

            imgui.pop_style_var()

        if self.gui_show_stats:
            self.gui_stats()
        if self.gui_show_playback:
            self.gui_playback()
        if self.gui_show_inspector:
            self.gui_inspector()

    def on_gui_after_render(self) -> None:
        # Renderer GUI is drawn after because render state is updated by rendering (read-only in GUI).
        if self.gui_show_renderer:
            self.gui_renderer()

        for w in self.scene.widgets:
            w.gui()

    def run(self) -> None:
        if self.window is None:
            raise RuntimeError("Viewer.run() can only be called if a window is created (Config.window must be True)")

        self.last_frame_timestamp = perf_counter_ns()

        self.running = True
        while self.running:
            # Draw first to display window on wayland
            self.on_draw()

            # Process events
            should_wait = self.wait_events and not (self.playback.playing and self.playback.num_frames > 1)
            if self.renderer.path_tracer:
                should_wait = should_wait and not any(
                    v.path_tracer_viewport.sample_index < self.renderer.path_tracer_max_samples_per_pixel
                    for v in self.viewports
                    if v.path_tracer_viewport
                )
            process_events(should_wait)

            # Check if window was closed
            if self.window.should_close():
                break

            # Process server messages
            if self.server is not None:
                while True:
                    try:
                        client, raw_message = self.server_message_queue.get_nowait()
                    except Empty:
                        break
                    self.on_raw_message(client, raw_message)
        self.device.wait_idle()

        if self.server is not None:
            self.server.shutdown()
