from pyxpg import *
from typing import Tuple

ctx = Context()
window = Window(ctx, "Minimal", 1280, 720)
gui = Gui(window)

# Draw function
def draw():
    # Update swapchain, this handles resizing the window buffers
    swapchain_status = window.update_swapchain()
    if swapchain_status == SwapchainStatus.MINIMIZED:
        return
    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    # Create GUI
    with gui.frame():
        if imgui.begin("Window"):
            imgui.text("Hello")
        imgui.end()

    # Render a frame
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
                # Render GUI
                gui.render(cmd)

            cmd.use_image(frame.image, ImageUsage.PRESENT)

# Input event
def mouse_button_event(p: Tuple[int, int], button: MouseButton, action: Action, mods: Modifiers):
    print("Mouse button event:", p, button, action, mods)

# Register window callbacks
window.set_callbacks(
    draw,
    mouse_button_event=mouse_button_event,
)

# Main loop
while True:
    process_events(True)

    if window.should_close():
        break

    draw()