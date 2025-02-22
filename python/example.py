from pyxpg import Context, Window, Gui, process_events, SwapchainStatus, begin_commands, end_commands
from pyxpg import imgui

ctx = Context()
window = Window(ctx, "Hello", 1280, 720)
gui = Gui(window)

def draw():
    # swapchain update
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    frame = window.begin_frame()

    gui.begin_frame()
    if imgui.begin("wow"):
        imgui.text("Hello")
        imgui.end()
    gui.end_frame()

    begin_commands(frame.command_pool, frame.command_buffer, ctx)
    gui.render(frame)
    end_commands(frame.command_buffer)

    window.end_frame(frame)

window.set_callbacks(draw)

while True:
    process_events(True)

    if window.should_close():
        break

    draw()

# if __name__ == "__main__":
#     run()