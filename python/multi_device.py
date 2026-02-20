from typing import Tuple
from pyxpg import *

instance = Instance(
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

device_0 = Device(
    instance,
    force_physical_device_index=0
)
device_1 = Device(
    instance,
    force_physical_device_index=1
)

window_0 = Window(device_0, "Minimal 0", 1280, 720)
window_1 = Window(device_1, "Minimal 1", 1280, 720)

# Draw function
def draw(window: Window, clear: Tuple[float, float, float, float]):
    # Update swapchain, this handles resizing the window buffers
    swapchain_status = window.update_swapchain()
    if swapchain_status == SwapchainStatus.MINIMIZED:
        return
    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    # Render a frame
    with window.frame() as frame:
        with frame.command_buffer as cmd:
            cmd.image_barrier(frame.image, ImageLayout.COLOR_ATTACHMENT_OPTIMAL, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.COLOR_ATTACHMENT)

            viewport = [0, 0, window.fb_width, window.fb_height]
            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.CLEAR,
                        store_op=StoreOp.STORE,
                        clear=clear,
                    ),
                ]):
                pass

            cmd.image_barrier(frame.image, ImageLayout.PRESENT_SRC, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.PRESENT)

# Input event
def mouse_button_event(p: Tuple[int, int], button: MouseButton, action: Action, mods: Modifiers):
    print("Mouse button event:", p, button, action, mods)

# Register window callbacks
window_0.set_callbacks(
    lambda: draw(window_0, (0.1, 0.2, 0.4, 1)),
    mouse_button_event=mouse_button_event,
)

window_1.set_callbacks(
    lambda: draw(window_1, (0.4, 0.2, 0.1, 1)),
    mouse_button_event=mouse_button_event,
)

# Main loop
while True:
    process_events(True)

    if window_0 is not None and window_0.should_close():
        window_0 = None

    if window_1 is not None and window_1.should_close():
        window_1 = None

    if window_0 is not None:
        draw(window_0, (0.1, 0.2, 0.4, 1))
    if window_1 is not None:
        draw(window_1, (0.4, 0.2, 0.1, 1))

    if window_0 is None and window_1 is None:
        break

device_0.wait_idle()
device_1.wait_idle()
