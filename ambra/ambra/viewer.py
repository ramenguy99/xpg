from pyxpg import *
from pyglm.glm import ivec2

from typing import Optional
from .config import Config, CameraType
from .server import Server, Client, RawMessage, Message, parse_builtin_messages
from queue import Queue, Empty
from .renderer import Renderer, Viewport, Rect
from .scene import Scene
from .camera import PerspectiveCamera, OrthographicCamera, CameraDepth
from .transform3d import RigidTransform

from dataclasses import dataclass

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
            draw=lambda _: self.on_draw(),
            key_event=lambda key, action, modifiers: self.on_key(key, action, modifiers),
            mouse_button_event=lambda pos, button, action, modifiers: self.on_mouse_button(ivec2(pos), button, action, modifiers),
            mouse_move_event=lambda pos: self.on_mouse_move(ivec2(pos)),
            mouse_scroll_event=lambda pos, scroll: self.on_scroll(ivec2(pos), ivec2(scroll)),
        )

        self.gui = Gui(self.window)

        self.renderer = Renderer(self.ctx, self.window, self.config.renderer)

        if self.config.camera_type == CameraType.PERSPECTIVE:
            camera = PerspectiveCamera(RigidTransform.identity(), CameraDepth(self.config.z_min, self.config.z_max), self.config.perspective_vertical_fov, self.window.fb_width / self.window.fb_height)
        elif self.config.camera_type == CameraType.ORTHOGRAPHIC:
            camera = OrthographicCamera(RigidTransform.identity(), CameraDepth(self.config.z_min, self.config.z_max))
        else:
            raise RuntimeError(f"Unhandled camera type {self.config.camera_type}")

        self.viewport = Viewport(
            rect=Rect(0, 0, self.window.fb_width, self.window.fb_height),
            camera=camera,
            scene=Scene("scene"),
        )

        self.server = Server(lambda c, m: self.on_raw_message_async(c, m), self.config.server)
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

    def on_draw(self):
        # Resize
        swapchain_status = self.window.update_swapchain()
        if swapchain_status == SwapchainStatus.MINIMIZED:
            return
        if swapchain_status == SwapchainStatus.RESIZED:
            pass

        # GUI
        with self.gui.frame():
            self.on_gui()

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