from pyxpg import *
from typing import Tuple
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.NOTSET,
    format='[%(asctime)s.%(msecs)03d] %(levelname)-6s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log_levels = {
    LogLevel.TRACE: logging.DEBUG,
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARN: logging.WARN,
    LogLevel.ERROR: logging.ERROR,
}

def log_callback(level, ctx, s):
    logger.log(log_levels[level], f"[{ctx}] {s}")

log_ovverride = LogCapture(log_callback)

ctx = Context(
    device_features=DeviceFeatures.DYNAMIC_RENDERING | DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.PRESENTATION, 
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

window = Window(ctx, "Hello", 1280, 720)
gui = Gui(window)

def draw():
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    with gui.frame():
        if imgui.begin("wow"):
            imgui.text("Hello")
        imgui.end()

    with window.frame() as frame:
        with frame.command_buffer as cmd:
            cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)

            viewport = [0, 0, window.fb_width, window.fb_height]
            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.CLEAR,
                        store_op=StoreOp.STORE,
                        clear=[0.1, 0.2, 0.4, 1],
                    ),
                ]):
                gui.render(cmd)

            cmd.use_image(frame.image, ImageUsage.PRESENT)

def mouse_button_event(p: Tuple[int, int], button: MouseButton, action: Action, mods: Modifiers):
    print("Mouse button event:", p, button, action, mods)

window.set_callbacks(
    draw,
    mouse_button_event=mouse_button_event,
)


while True:
    process_events(False)

    if window.should_close():
        break

    draw()