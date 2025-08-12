from pyxpg import *

from colors import *
from render import RenderParams, render
from pipelines import *


params = RenderParams()

ctx = Context()
window = Window(ctx, "Pipes", params.width, params.height)
gui = Gui(window)

@dataclass
class Costs:
    load: float = 4.0
    copy: float = 1.0
    cmd: float = 1.0
    draw: float = 3.0
    present: float = 1.0
    upload: float = 5.0

def double_buffered_async_gpu_upload_sync_submit_half_rate(costs: Costs):
    threads = {
        "load": Thread("CPU - LOAD", [Task("Load", costs.load, ORANGE)]),
        "cpu": Thread("CPU - MAIN", [Task("Copy", costs.copy, YELLOW), Task("Cmd", costs.cmd, YELLOW)]),
        "gpu": Thread("GPU - DRAW", [Task("Draw", costs.draw, GREEN), Task("Present", costs.present, VIOLET)]),
        "copy": Thread("GPU - COPY", [Task("Upload", costs.upload, BLUE)]),
    }
    p = Pipeline(threads)

    N = 2
    F = 2
    p.add("load", "Load", "cpubuf", count=N)
    p.add("cpu", "Copy", "cmd", count=N)

    p.produces(("load", "Load"), ("cpu", "Copy"), "copybuf", (F, 1))
    p.produces(("cpu", "Copy"), ("copy", "Upload"), "copybuf", (1, F))
    p.produces(("cpu", "Copy"), ("cpu", "Cmd"), "exec")
    p.produces(("gpu", "Present"), ("cpu", "Copy"), "cmd")
    p.produces(("copy", "Upload"), ("gpu", "Draw"), "copybuf", (F, 1))
    p.produces(("gpu", "Draw"), ("gpu", "Present"), "img")
    p.produces(("cpu", "Cmd"), ("gpu", "Draw"), "cmd")
    p.produces(("gpu", "Present"), ("load", "Load"), "cpubuf", (1, F))
    return p

costs = Costs()
pipe = double_buffered_async_gpu_upload_sync_submit_half_rate(costs)
rendered =  render(pipe, params)

def draw():
    global rendered, pipe

    status = window.update_swapchain()
    if status == SwapchainStatus.MINIMIZED:
        return
    if status == SwapchainStatus.RESIZED:
        pass


    with gui.frame():
        imgui.set_next_window_pos((0, 0), imgui.Cond.ALWAYS)
        imgui.set_next_window_size((window.fb_width, window.fb_height), imgui.Cond.ALWAYS)
        if imgui.begin("canvas",
            flags=
                imgui.WindowFlags.NO_RESIZE |
                imgui.WindowFlags.NO_TITLE_BAR |
                imgui.WindowFlags.NO_MOVE |
                imgui.WindowFlags.NO_BACKGROUND |
                imgui.WindowFlags.NO_SCROLLBAR |
                imgui.WindowFlags.NO_INPUTS |
                imgui.WindowFlags.NO_SAVED_SETTINGS |
                imgui.WindowFlags.NO_BRING_TO_FRONT_ON_FOCUS
        )[0]:
            dl = imgui.get_window_draw_list()

            def rect(text, x, y, w, h, color: ColorInfo):
                def hextocol(s: str) -> int:
                    s2 = s[4:6] + s[2:4] + s[0:2]
                    return int(s2, base=16) | 0xFF000000

                dl.add_rect_filled((x, y), (x + w, y + h), hextocol(color.face_color[1:]))
                dl.add_rect((x, y), (x + w, y + h), hextocol(color.border_color[1:]), thickness=2)
                sz = imgui.calc_text_size(text)
                dl.add_text((x + (w - sz.x) / 2, y + (h - sz.y)/2), 0xFF000000, text)

            def text(text, x, y):
                sz = imgui.calc_text_size(text)
                dl.add_text((x - sz.x / 2, y - sz.y / 2), 0xFF000000, text)

            def arrow(x0, y0, x1, y1, style):
                dl.add_line((x0, y0), (x1, y1), 0xFF000000, 2)

            for t in rendered.texts:
                text(t.text, t.x, t.y)
            for a in rendered.arrows:
                arrow(a.x0, a.y0, a.x1, a.y1, a.style)
            for r in rendered.rects:
                rect(r.text, r.x, r.y, r.w, r.h, r.color)
        imgui.end()

        if imgui.begin("params")[0]:
            for k in costs.__dataclass_fields__:
                changed, v = imgui.slider_float(f"{k}###{k}", getattr(costs, k), 1, 100)
                setattr(costs, k, v)
                if changed:
                    pipe = double_buffered_async_gpu_upload_sync_submit_half_rate(costs)
                    rendered =  render(pipe, params)
        imgui.end()

    with window.frame() as frame:
        with frame.command_buffer as cmd:
            cmd.image_barrier(frame.image, ImageLayout.COLOR_ATTACHMENT_OPTIMAL, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.COLOR_ATTACHMENT)
            viewport = [0, 0, window.fb_width, window.fb_height]
            with cmd.rendering(
                viewport,
                color_attachments= [
                    RenderingAttachment(
                        image=frame.image,
                        load_op=LoadOp.CLEAR,
                        store_op=StoreOp.STORE,
                        clear=(1, 1, 1, 1)
                    ),
                ],
            ):
                cmd.set_viewport(viewport)
                cmd.set_scissors(viewport)
                gui.render(cmd)
            cmd.image_barrier(frame.image, ImageLayout.PRESENT_SRC, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.PRESENT)


window.set_callbacks(draw=draw)
while True:
    process_events(True)
    if window.should_close():
        break
    draw()