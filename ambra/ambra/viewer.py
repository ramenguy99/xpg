import logging
from queue import Empty, Queue
from time import perf_counter_ns
from typing import Any, Optional, Tuple, Union

import numpy as np
from pyglm.glm import ivec2, vec3
from pyxpg import (
    Action,
    Context,
    DeviceFeatures,
    Gui,
    Key,
    LogCapture,
    LogLevel,
    Modifiers,
    MouseButton,
    SwapchainStatus,
    Window,
    imgui,
    process_events,
    set_log_level,
)

from .config import Config
from .keybindings import KeyMap
from .renderer import Renderer
from .scene import Object, Scene
from .server import Client, Message, RawMessage, Server, parse_builtin_messages
from .utils.gpu_property import GpuBufferProperty, GpuImageProperty
from .utils.lru_pool import LRUPool
from .viewport import Playback, Rect, Viewport

_log_levels = {
    LogLevel.TRACE: logging.DEBUG,
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARN: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
}


_logger = logging.getLogger(__name__)


def _log(level: LogLevel, c: str, s: str) -> None:
    _logger.log(_log_levels[level], f"[{c}] {s}")  # noqa: G004


class Viewer:
    def __init__(
        self,
        title: str = "ambra",
        config: Optional[Config] = None,
        key_map: Optional[KeyMap] = None,
    ):
        config = config if config is not None else Config()

        # Key bindings
        self.key_map = key_map if key_map is not None else KeyMap()

        # Logging
        self.log_capture = LogCapture(_log)
        set_log_level(LogLevel(config.log_level.value))

        # Viewer
        self.running = False

        # Context
        self.ctx = Context(
            required_features=DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.DYNAMIC_RENDERING,
            optional_features=DeviceFeatures.RAY_QUERY
            | DeviceFeatures.HOST_QUERY_RESET
            | DeviceFeatures.WIDE_LINES
            | DeviceFeatures.TIMELINE_SEMAPHORES,
            preferred_frames_in_flight=config.preferred_frames_in_flight,
            vsync=config.vsync,
            force_physical_device_index=0xFFFFFFFF
            if config.force_physical_device_index is None
            else config.force_physical_device_index,
            prefer_discrete_gpu=config.prefer_discrete_gpu,
            enable_validation_layer=config.enable_validation_layer,
            enable_synchronization_validation=config.enable_synchronization_validation,
            enable_gpu_based_validation=config.enable_gpu_based_validation,
        )

        # Window
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
            mouse_scroll_event=lambda pos, scroll: self.on_scroll(ivec2(pos), ivec2(scroll)),
        )

        # GUI
        self.gui = Gui(self.window)
        self.gui_show_stats = config.gui.stats
        self.gui_show_inspector = config.gui.inspector
        self.gui_show_playback = config.gui.playback
        self.gui_show_renderer = config.gui.renderer
        self.gui_selected_obj: Optional[Object] = None
        self.gui_selected_gpu_property: Optional[Union[GpuBufferProperty, GpuImageProperty]] = None

        # Renderer
        self.renderer = Renderer(self.ctx, self.window, config.renderer)

        # Playback
        self.playback = Playback(config.playback)
        self.last_frame_timestamp = 0

        # Stats
        self.frame_time_index = 0
        self.frame_times = np.zeros(config.stats_frame_time_count, np.float32)

        # Viewport
        self.viewport = Viewport(
            rect=Rect(0, 0, self.window.fb_width, self.window.fb_height),
            scene=Scene("scene"),
            playback=self.playback,
            camera_config=config.camera,
            handedness=config.handedness,
            world_up=vec3(config.world_up),
        )

        # Server
        self.server = Server(self.on_raw_message_async, config.server)
        self.server_message_queue: Queue[Tuple[Client, RawMessage]] = Queue()

        # Config
        self.wait_events = config.wait_events

    def on_key(self, key: Key, action: Action, modifiers: Modifiers) -> None:
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

        if action == Action.PRESS:
            if self.key_map.camera_rotate.is_active(button, modifiers):
                self.viewport.on_rotate_press(position)
            if self.key_map.camera_pan.is_active(button, modifiers):
                self.viewport.on_pan_press(position)
        if action == Action.RELEASE:
            if self.key_map.camera_rotate.button == button:
                self.viewport.on_rotate_release()
            if self.key_map.camera_pan.button == button:
                self.viewport.on_pan_release()

    def on_mouse_move(self, position: ivec2) -> None:
        self.viewport.on_move(position)

    def on_scroll(self, position: ivec2, scroll: ivec2) -> None:
        if imgui.get_io().want_capture_mouse:
            return

        modifiers = self.window.get_modifiers_state()
        if modifiers == self.key_map.camera_zoom_modifiers:
            self.viewport.zoom(scroll, False)
        if modifiers == self.key_map.camera_zoom_move_modifiers:
            self.viewport.zoom(scroll, True)

    def on_resize(self, width: int, height: int) -> None:
        pass

    def on_draw(self) -> None:
        # Compute dt
        timestamp = perf_counter_ns()
        dt = (timestamp - self.last_frame_timestamp) * 1e-9
        self.last_frame_timestamp = timestamp

        self.frame_times[self.frame_time_index] = dt

        # Step dt
        if self.playback.max_time is None:
            self.playback.set_max_time(self.viewport.scene.max_animation_time(self.playback.frames_per_second))
        if self.playback.playing:
            self.playback.step(dt)

        # Resize
        swapchain_status = self.window.update_swapchain()
        if swapchain_status == SwapchainStatus.MINIMIZED:
            return
        if swapchain_status == SwapchainStatus.RESIZED:
            width, height = self.window.fb_width, self.window.fb_height

            self.renderer.resize(width, height)
            self.viewport.resize(width, height)

            self.on_resize(width, height)

        # GUI
        with self.gui.frame():
            self.on_gui()

        # Step scene
        self.viewport.scene.update(self.playback.current_time, self.playback.current_frame)

        # Render
        self.renderer.render(self.viewport, self.gui)

        self.frame_time_index = (self.frame_time_index + 1) % self.frame_times.size

    def on_raw_message_async(self, client: Client, raw_message: RawMessage) -> None:
        self.server_message_queue.put((client, raw_message))
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
            | imgui.WindowFlags.NO_NAV,
        )[0]:
            avg_dt = self.frame_times.mean()
            avg_fps = 1.0 / avg_dt
            last_dt = self.frame_times[self.frame_time_index]
            last_fps = 1.0 / last_dt
            imgui.text(f"{self.ctx.device_properties.device_name}")
            imgui.text(f"Window size:     [{self.window.fb_width}x{self.window.fb_height}]")
            imgui.text(f"FPS:             {avg_fps:6.2f} ({last_fps:6.2f})")
            imgui.text(f"Frame time (ms): {avg_dt * 1000.0:6.2f} ({last_dt * 1000.0:6.2f})")
        imgui.end()

    def gui_playback(self) -> None:
        if imgui.begin("Playback")[0]:
            _, self.playback.playing = imgui.checkbox("Playing", self.playback.playing)
            imgui.text(f"Time (s): {self.playback.current_time:7.3f} / {self.playback.max_time: 7.3f}")
            u, frame = imgui.slider_int(
                "Frame",
                self.playback.current_frame,
                0,
                self.playback.num_frames - 1,
            )
            if u:
                self.playback.set_frame(frame)
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

                imgui.push_style_var_im_vec2(imgui.StyleVar.FRAME_PADDING, imgui.Vec2(0, 1))
                o.gui_expanded = imgui.tree_node_ex(f"{o.name}##tree_node_{o.uid}", flags)
                imgui.pop_style_var()

                if imgui.is_item_clicked():
                    if self.gui_selected_obj == o:
                        self.gui_selected_obj = None
                    else:
                        self.gui_selected_obj = o

                return o.gui_expanded

            def post(o: Object) -> None:
                imgui.tree_pop()

            self.viewport.scene.visit_objects_pre_post(pre, post)

            imgui.separator()

            if self.gui_selected_obj is not None:
                self.gui_selected_obj.gui()

        imgui.end()

    def gui_renderer(self) -> None:
        if imgui.begin("Renderer")[0]:
            imgui.text("GPU properties:")
            imgui.indent(5)
            for p in self.renderer.gpu_properties:
                s = imgui.selectable(p.name, self.gui_selected_gpu_property == p)
                if s:
                    if self.gui_selected_gpu_property == p:
                        self.gui_selected_gpu_property = None
                    else:
                        self.gui_selected_gpu_property = p
            imgui.indent(-5)

            imgui.separator()
            if self.gui_selected_gpu_property is not None:

                def drawpool(name: str, pool: Optional[LRUPool[int, Any]], count: int) -> None:
                    if pool is None:
                        return
                    imgui.separator_text(name)
                    imgui.text("Map")
                    imgui.indent()
                    for lu_k, lu_v in pool.lookup.items():
                        imgui.text(f"{lu_k:03d} {lu_v}")
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
                    imgui.separator()

                    start = imgui.get_cursor_screen_pos()
                    dl = imgui.get_window_draw_list()

                    num_frames = self.gui_selected_gpu_property.property.num_frames
                    current_frame = self.gui_selected_gpu_property.property.current_frame_index

                    if self.gui_selected_gpu_property.cpu_pool:
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
                            c_k,
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
                            g_k,
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

        if self.gui_show_stats:
            self.gui_stats()
        if self.gui_show_playback:
            self.gui_playback()
        if self.gui_show_inspector:
            self.gui_inspector()
        if self.gui_show_renderer:
            self.gui_renderer()

    def run(self) -> None:
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

        if self.server is not None:
            self.server.shutdown()
