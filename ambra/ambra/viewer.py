# Copyright Dario Mylonopoulos
# SPDX-License-Identifier: MIT

from queue import Empty, Queue
from time import perf_counter_ns
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pyglm.glm import dvec2, ivec2, normalize, vec3
from pyxpg import (
    Action,
    AllocType,
    BorderColor,
    BufferUsageFlags,
    Context,
    DescriptorSetBinding,
    DescriptorType,
    DeviceFeatures,
    Format,
    Gui,
    Image,
    ImageLayout,
    ImageUsageFlags,
    Key,
    LogLevel,
    MemoryHeapFlags,
    Modifiers,
    MouseButton,
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
from .keybindings import KeyMap
from .lights import LIGHT_TYPES_INFO
from .renderer import FrameInputs, Renderer
from .scene import Object, Scene
from .server import Client, Message, RawMessage, Server, parse_builtin_messages
from .utils.descriptors import (
    create_descriptor_layout_pool_and_sets,
    create_descriptor_pool_and_sets,
)
from .utils.gpu import UploadableBuffer
from .utils.lru_pool import LRUPool
from .utils.ring_buffer import RingBuffer
from .viewport import Playback, Rect, Viewport


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

        # Context
        self.ctx = Context(
            version=(1, 1),
            required_features=DeviceFeatures.SYNCHRONIZATION_2
            | DeviceFeatures.DYNAMIC_RENDERING
            | DeviceFeatures.DESCRIPTOR_INDEXING
            | DeviceFeatures.STORAGE_IMAGE_READ_WRITE_WITHOUT_FORMAT,
            optional_features=DeviceFeatures.RAY_QUERY
            | DeviceFeatures.HOST_QUERY_RESET
            | DeviceFeatures.WIDE_LINES
            | DeviceFeatures.TIMELINE_SEMAPHORES
            | DeviceFeatures.SHADER_DRAW_PARAMETERS
            | DeviceFeatures.SHADER_FLOAT16_INT8
            | DeviceFeatures.SHADER_INT16
            | DeviceFeatures.SHADER_SUBGROUP_EXTENDED_TYPES
            | DeviceFeatures.STORAGE_8BIT
            | DeviceFeatures.STORAGE_16BIT
            | DeviceFeatures.DRAW_INDIRECT_COUNT
            | DeviceFeatures.MESH_SHADER
            | DeviceFeatures.FRAGMENT_SHADER_BARYCENTRIC,
            preferred_frames_in_flight=config.preferred_frames_in_flight,
            preferred_swapchain_usage_flags=ImageUsageFlags.COLOR_ATTACHMENT
            | ImageUsageFlags.TRANSFER_DST
            | ImageUsageFlags.TRANSFER_SRC,
            vsync=config.vsync,
            force_physical_device_index=0xFFFFFFFF
            if config.force_physical_device_index is None
            else config.force_physical_device_index,
            prefer_discrete_gpu=config.prefer_discrete_gpu,
            enable_validation_layer=config.enable_validation_layer,
            enable_synchronization_validation=config.enable_synchronization_validation,
            enable_gpu_based_validation=config.enable_gpu_based_validation,
        )

        # Headless swapchain for screenshots and videos
        self.headless_swapchain = HeadlessSwapchain(self.ctx, 2, Format.R8G8B8A8_UNORM)

        # Window
        self.window: Optional[Window] = None
        if config.window:
            self.window = Window(
                self.ctx,
                title,
                config.window_width,
                config.window_height,
                x=config.window_x,
                y=config.window_y,
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
            render_width, render_height = self.window.fb_width, self.window.fb_height
            output_format = self.window.swapchain_format
            num_frames_in_flight = self.window.num_frames

            # GUI
            self.gui = Gui(self.window)
        else:
            render_width, render_height = config.window_width, config.window_height
            output_format = self.headless_swapchain.format
            num_frames_in_flight = self.headless_swapchain.num_frames_in_flight

            # GUI
            self.gui = Gui(self.ctx, num_frames_in_flight, output_format)

        self.multiviewport = config.gui.multiviewport
        self.gui.set_ini_filename(config.gui.ini_filename)

        self.gui_show_stats = config.gui.stats
        self.gui_show_inspector = config.gui.inspector
        self.gui_show_playback = config.gui.playback
        self.gui_show_renderer = config.gui.renderer
        self.gui_selected_obj: Optional[Object] = None
        self.gui_selected_gpu_property: Optional[GpuProperty[Any]] = None
        self.gui_playback_slider_held = False

        # Disable ImGui asserts
        imgui.get_io().config_error_recovery_enable_assert = False

        # Make ImGui automatically scale fonts on DPI change events
        imgui.get_io().config_dpi_scale_fonts = True

        # Renderer
        self.renderer = Renderer(
            self.ctx,
            render_width,
            render_height,
            num_frames_in_flight,
            output_format,
            self.multiviewport,
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

        # Viewport
        if self.multiviewport:
            if config.gui.initial_number_of_viewports > config.gui.max_viewport_count:
                raise ValueError(
                    "config.gui.initial_number_of_viewports must be less than or equal to config.gui.max_viewport_count"
                )
            self.viewport_sampler = Sampler(
                self.ctx,
                u=SamplerAddressMode.CLAMP_TO_BORDER,
                v=SamplerAddressMode.CLAMP_TO_BORDER,
                border_color=BorderColor.FLOAT_OPAQUE_BLACK,
            )
            self.viewport_descriptor_layout, self.viewport_descriptor_pool, self.viewport_descriptor_sets = (
                create_descriptor_layout_pool_and_sets(
                    self.ctx,
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
            self.ctx,
            self.renderer.scene_descriptor_set_layout,
            max_viewports * self.renderer.num_frames_in_flight,
            name="viewport-scene-descriptor-sets",
        )
        self.viewports: List[Viewport] = []
        for viewport_index in range(min(num_viewports, config.gui.max_viewport_count)):
            scene_descriptor_sets = RingBuffer(
                self.viewport_scene_descriptor_sets[
                    viewport_index * self.renderer.num_frames_in_flight : (viewport_index + 1)
                    * self.renderer.num_frames_in_flight
                ]
            )
            scene_uniform_buffers = RingBuffer(
                [
                    UploadableBuffer(self.ctx, self.renderer.scene_constants_dtype.itemsize, BufferUsageFlags.UNIFORM)
                    for _ in range(self.renderer.num_frames_in_flight)
                ]
            )
            scene_light_buffers = RingBuffer(
                [
                    [
                        UploadableBuffer(
                            self.ctx, info.size * self.renderer.max_lights_per_type, BufferUsageFlags.STORAGE
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

            img = None
            texture = None
            if self.multiviewport:
                img = Image(
                    self.ctx,
                    render_width,
                    render_height,
                    output_format,
                    ImageUsageFlags.SAMPLED | ImageUsageFlags.COLOR_ATTACHMENT,
                    AllocType.DEVICE,
                    name=f"viewport-{viewport_index}",
                )

                s = self.viewport_descriptor_sets[viewport_index]
                s.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, self.viewport_sampler, 0)
                texture = imgui.Texture(s)

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
                    image=img,
                    imgui_texture=texture,
                    name=f"viewport-{viewport_index}",
                )
            )
        self.active_viewport = self.viewports[0] if self.viewports else None

        # Server
        self.server = Server(self.on_raw_message_async, config.server)
        self.server_message_queue: Queue[Tuple[Client, RawMessage]] = Queue()

        # Config
        self.wait_events = config.wait_events

    def on_key(self, key: Key, action: Action, modifiers: Modifiers) -> None:
        if imgui.get_io().want_capture_keyboard:
            return

        # Press
        if action == Action.PRESS:
            if self.key_map.toggle_play_pause.is_active(key, modifiers):
                self.playback.toggle_play_pause()
            if self.key_map.exit.is_active(key, modifiers):
                self.running = False

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
                self.active_viewport.zoom(scroll, False)
            if modifiers == self.key_map.camera_zoom_move_modifiers:
                self.active_viewport.zoom(scroll, True)

    def on_resize(self, width: int, height: int) -> None:
        pass

    def _render(self, render_to_window: bool) -> None:
        # GUI
        with self.gui.frame():
            self.on_gui()

        # HACK: If not rendering to the window, we draw the GUI twice so that
        # imgui renders all windows, including those for which the size of the contents hasn't
        # been computed yet.
        if not render_to_window:
            with self.gui.frame():
                self.on_gui()

        # Begin frame
        if render_to_window:
            assert self.window is not None
            try:
                frame = self.window.begin_frame()
            except SwapchainOutOfDateError:
                return
            frame_inputs = FrameInputs(frame.image, frame.command_buffer, frame.transfer_command_buffer, [], [])
        else:
            frame_inputs = self.headless_swapchain.begin_frame()

        # Begin recording on command buffers
        frame_inputs.command_buffer.begin()
        if frame_inputs.transfer_command_buffer:
            frame_inputs.transfer_command_buffer.begin()

        # Render
        self.renderer.render(self.scene, self.viewports, frame_inputs, self.gui)

        if not render_to_window:
            self.headless_swapchain.get_current().issue_readback()

        # Submit transfer queue commands, if submitted
        if frame_inputs.transfer_command_buffer:
            frame_inputs.transfer_command_buffer.end()

            # If there is no semaphore to signal, we assume there is no commands to submit.
            # Even if someone recorded commands, this would be a bug because
            # there is no way to know when those commands would complete.
            if frame_inputs.transfer_semaphores:
                self.ctx.transfer_queue.submit(
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
                    assert viewport.image is not None
                    viewport.image.destroy()
                    img = Image(
                        self.ctx,
                        width,
                        height,
                        self.renderer.output_format,
                        ImageUsageFlags.SAMPLED | ImageUsageFlags.COLOR_ATTACHMENT,
                        AllocType.DEVICE,
                        name=f"viewport-{viewport_index}",
                    )

                    s = self.viewport_descriptor_sets[viewport_index]
                    s.write_combined_image_sampler(img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, self.viewport_sampler, 0)
                    viewport.image = img

            self.on_resize(width, height)

        # Render to window
        self._render(True)

    def on_raw_message_async(self, client: Client, raw_message: RawMessage) -> None:
        self.server_message_queue.put((client, raw_message))
        if self.window is not None:
            self.window.post_empty_event()

    def on_raw_message(self, client: Client, raw_message: RawMessage) -> None:
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
            imgui.text(f"{self.ctx.device_properties.device_name}")
            if self.window is not None:
                imgui.text(f"Window size:     [{self.window.fb_width}x{self.window.fb_height}]")
            imgui.text(f"FPS:             {avg_fps:6.2f} ({last_fps:6.2f})")
            imgui.text(f"Frame time (ms): {avg_dt * 1000.0:6.2f} ({last_dt * 1000.0:6.2f})")
            for i, (heap, stats) in enumerate(zip(self.ctx.memory_properties.memory_heaps, self.ctx.heap_statistics)):
                if heap.flags & MemoryHeapFlags.VK_MEMORY_HEAP_DEVICE_LOCAL:
                    kind = "GPU"
                else:
                    kind = "CPU"
                imgui.text(
                    f"Heap {i} - {kind}: {stats.block_bytes / (1024 * 1024 * 1024):4.02f} / {heap.size / (1024 * 1024 * 1024):4.02f} GB"
                )
        imgui.end()

    def gui_playback(self) -> None:
        self.gui_playback_slider_held = False
        if imgui.begin("Playback")[0]:
            _, self.playback.playing = imgui.checkbox("Playing", self.playback.playing)
            _, self.playback.playback_speed_multiplier = imgui.drag_float(
                "Playback speed", self.playback.playback_speed_multiplier, 0.01, 0.01, 10.0, format="%.2fx"
            )
            _, frame = imgui.slider_int(
                "Frame",
                self.playback.current_frame,
                0,
                self.playback.num_frames - 1,
            )
            if imgui.is_item_active():
                self.gui_playback_slider_held = True
                self.playback.set_frame(frame)
            imgui.text(f"Time (s): {self.playback.current_time:7.3f} / {self.playback.max_time: 7.3f}")
            imgui.text(f"Playback FPS: {self.playback.frames_per_second:7.3f}")
        imgui.end()

    def gui_inspector(self) -> None:
        if imgui.begin("Inspector")[0]:

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

            output_width, output_height = self._get_framebuffer_size()

            for v in self.viewports:
                assert v.imgui_texture is not None
                assert v.image is not None

                imgui.set_next_window_size_constraints(imgui.Vec2(0, 0), imgui.Vec2(v.image.width, v.image.height))
                if imgui.begin(v.name)[0]:
                    cursor_pos = imgui.get_cursor_screen_pos()
                    pos = ivec2(cursor_pos.x, cursor_pos.y)
                    avail = imgui.get_content_region_avail()
                    size = ivec2(avail.x, avail.y)
                    v.rect.x = pos.x
                    v.rect.y = pos.y
                    v.rect.width = min(size.x, output_width)
                    v.rect.height = min(size.y, output_height)
                    imgui.image(
                        v.imgui_texture, avail, imgui.Vec2(0, 0), imgui.Vec2(size.x / fb_width, size.y / fb_height)
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
                            v.zoom(dvec2(0, io.mouse_wheel), False)
                        elif mapped_mods == self.key_map.camera_zoom_move_modifiers:
                            v.zoom(dvec2(0, io.mouse_wheel), True)

                    # If pan or rotate is happening on just left click, prevent imgui window move
                    # by creating an invisible button on the whole window content area.
                    # Moving is still possible by dragging the window title bar.
                    if prevent_left_button_window_move and avail.x > 0 and avail.y > 0:
                        imgui.set_cursor_screen_pos(cursor_pos)
                        imgui.invisible_button(v.name, avail)
                else:
                    v.rect.width = 0
                    v.rect.height = 0
                imgui.end()
            imgui.pop_style_var()

        if self.gui_show_stats:
            self.gui_stats()
        if self.gui_show_playback:
            self.gui_playback()
        if self.gui_show_inspector:
            self.gui_inspector()
        if self.gui_show_renderer:
            self.gui_renderer()

    def run(self) -> None:
        if self.window is None:
            raise RuntimeError("Viewer.run() can only be called if a window is created (Config.window must be True)")

        self.last_frame_timestamp = perf_counter_ns()

        self.running = True
        while self.running:
            process_events(self.wait_events)

            if self.server is not None:
                while True:
                    try:
                        client, raw_message = self.server_message_queue.get_nowait()
                    except Empty:
                        break
                    self.on_raw_message(client, raw_message)

            if self.window.should_close():
                break

            self.on_draw()
        self.ctx.wait_idle()

        if self.server is not None:
            self.server.shutdown()
