from pyxpg import *
from pyxpg import imgui
from pathlib import Path
import numpy as np

ctx = Context()
window = Window(ctx, "Hello", 1280, 720)
gui = Gui(window)

rot = np.eye(4, dtype=np.float32)

buf = Buffer(ctx, 1024, BufferUsageFlags.VERTEX, AllocType.HOST)
ubuf = Buffer.from_data(ctx, rot.tobytes(), BufferUsageFlags.VERTEX, AllocType.HOST)

vert = Shader(ctx, Path("res/basic.vert.spirv").read_bytes())
frag = Shader(ctx, Path("res/basic.frag.spirv").read_bytes())

stage = PipelineStage(vert, Stage.VERTEX)

set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)

pipeline = GraphicsPipeline(
    ctx,
    stages = [
        PipelineStage(vert, Stage.VERTEX),
        PipelineStage(frag, Stage.FRAGMENT),
    ],
    vertex_bindings = [
        VertexBinding(0, 0, VertexInputRate.VERTEX),
    ],
    vertex_attributes = [
        VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
    ],
    input_assembly = InputAssembly(PrimitiveTopology.TRIANGLE_LIST),
    descriptor_sets = [ set ],
    attachments = [
        Attachment(format=window.swapchain_format)
    ]
)

v = 0

def draw():
    global v
    # swapchain update
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    with gui.frame():
        if imgui.begin("wow"):
            imgui.text("Hello")
            _, v = imgui.drag_float("Value", v)
        imgui.end()

    with window.frame() as frame:
        with frame.commands():
            gui.render(frame)

window.set_callbacks(draw)

while True:
    process_events(True)

    if window.should_close():
        break

    draw()

# if __name__ == "__main__":
#     run()