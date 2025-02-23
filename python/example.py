from pyxpg import *
from pyxpg import imgui
from pathlib import Path

ctx = Context()
window = Window(ctx, "Hello", 1280, 720)
gui = Gui(window)

buf = Buffer(ctx, 1024, BufferUsageFlags.VERTEX, AllocType.HOST)
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
)

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