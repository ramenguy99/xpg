from typing import Optional
from queue import Queue, Empty
from time import perf_counter_ns

from pyglm.glm import ivec2, vec2
from pyxpg import Context, Window, Gui, DeviceFeatures, Key, MouseButton, Action, Modifiers, SwapchainStatus, process_events

from .config import Config, CameraType
from .server import Server, Client, RawMessage, Message, parse_builtin_messages
from .renderer import Renderer
from .scene import Scene
from .camera import PerspectiveCamera, OrthographicCamera, CameraDepth
from .transform3d import RigidTransform3D
from .viewport import Viewport, Playback, Rect

class Viewer:
    def __init__(self, title: str = "ambra", width: Optional[int] = None, height: Optional[int] = None, config: Optional[Config] = None):
        self.config = config if config is not None else Config()

        self.ctx = Context(
            required_features=DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.DYNAMIC_RENDERING,
            optional_features=DeviceFeatures.RAY_QUERY | DeviceFeatures.HOST_QUERY_RESET | DeviceFeatures.WIDE_LINES,
            enable_validation_layer=True,
            enable_synchronization_validation=True,
        )

        self.window = Window(self.ctx, title, width, height)
        self.window.set_callbacks(
            draw=self.on_draw,
            key_event=self.on_key,
            mouse_button_event=lambda pos, button, action, modifiers: self.on_mouse_button(ivec2(pos), button, action, modifiers),
            mouse_move_event=lambda pos: self.on_mouse_move(ivec2(pos)),
            mouse_scroll_event=lambda pos, scroll: self.on_scroll(ivec2(pos), ivec2(scroll)),
        )

        self.gui = Gui(self.window)

        self.renderer = Renderer(self.ctx, self.window, self.config.renderer)

        if self.config.camera_type == CameraType.PERSPECTIVE:
            camera = PerspectiveCamera(RigidTransform3D.identity(), CameraDepth(self.config.z_min, self.config.z_max), self.window.fb_width / self.window.fb_height, self.config.perspective_vertical_fov)
        elif self.config.camera_type == CameraType.ORTHOGRAPHIC:
            camera = OrthographicCamera(RigidTransform3D.identity(), CameraDepth(self.config.z_min, self.config.z_max), self.window.fb_width / self.window.fb_height, vec2(self.config.ortho_center), vec2(self.config.ortho_half_extents))
        else:
            raise RuntimeError(f"Unhandled camera type {self.config.camera_type}")

        self.playback = Playback(self.config.playback)
        self.last_frame_timestamp = 0

        self.viewport = Viewport(
            rect=Rect(0, 0, self.window.fb_width, self.window.fb_height),
            camera=camera,
            scene=Scene("scene"),
            playback=self.playback
        )

        self.server = Server(self.on_raw_message_async, self.config.server)
        self.server_message_queue = Queue()

        # Config
        self.wait_events = self.config.wait_events


    def on_key(self, key: Key, action: Action, modifiers: Modifiers):
        pass

    def on_mouse_button(self, position: ivec2, button: MouseButton, action: Action, modifiers: Modifiers):
        pass

    def on_mouse_move(self, position: ivec2):
        pass

    def on_scroll(self, position: ivec2, scroll: ivec2):
        pass

    def on_resize(self, width: int, height: int):
        pass

    def on_draw(self):
        # Compute dt
        timestamp = perf_counter_ns()
        dt = (timestamp - self.last_frame_timestamp) * 1e-9
        self.last_frame_timestamp = timestamp

        # Step dt
        if self.playback.max_time is None:
            self.playback.max_time = self.viewport.scene.max_animation_time(self.playback.frames_per_second)
        self.playback.step(dt)

        # Resize
        swapchain_status = self.window.update_swapchain()
        if swapchain_status == SwapchainStatus.MINIMIZED:
            return
        if swapchain_status == SwapchainStatus.RESIZED:
            width, height = self.window.fb_width, self.window.fb_height
            self.viewport.resize(width, height)
            # NOTE: at some point the renderer would also likely want to be notified
            # of resize events for resizing framebuffer-sized resources
            self.on_resize(width, height)

        # GUI
        with self.gui.frame():
            self.on_gui()

        # Step scene
        self.viewport.scene.update(self.playback.current_time, self.playback.current_frame)

        # Render
        self.renderer.render(self.viewport, self.gui)

    def on_raw_message_async(self, client: Client, raw_message: RawMessage):
        self.server_message_queue.put((client, raw_message))
        self.window.post_empty_event()

    def on_raw_message(self, client: Client, raw_message: RawMessage):
        message = parse_builtin_messages(raw_message)
        if message is not None:
            self.on_message(client, message)

    def on_message(self, client: Client, message: Message):
        pass

    def on_gui(self):
        pass

    def run(self):
        self.last_frame_timestamp = perf_counter_ns()

        while True:
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