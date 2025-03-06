from pyxpg import *
from pyxpg import imgui
# from pyxpg import slang
from pathlib import Path
import numpy as np

ctx = Context()
window = Window(ctx, "Hello", 1280, 720)
gui = Gui(window)

V = np.array([
    [-0.5, -0.5, 0],
    [ 0.5, -0.5, 0],
    [-0.5,  0.5, 0],
    [ 0.5,  0.5, 0],
], np.float32)

I = np.array([
    [0, 1, 2],
    [1, 3, 2],
], np.uint32)

rot = np.eye(4, dtype=np.float32)
rot[:3, :3] = np.eye(3)

rot2 = np.eye(4, dtype=np.float32)
rot2[:3, :3] = np.eye(3) * 0.4

rot3 = np.eye(4, dtype=np.float32)
rot3[:3, :3] = np.eye(3) * 2

push_constants = np.array([ 1.0, 0.0, 0.0], np.float32)

v_buf = Buffer.from_data(ctx, V.tobytes(), BufferUsageFlags.VERTEX, AllocType.DEVICE_MAPPED)
i_buf = Buffer.from_data(ctx, I.tobytes(), BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)
u_buf = Buffer.from_data(ctx, rot.tobytes(), BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
u_bu2 = Buffer.from_data(ctx, rot2.tobytes(), BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
u_bu3 = Buffer.from_data(ctx, rot3.tobytes(), BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)

# vert_prog = slang.compile("src/color.vert.slang")
# frag_prog = slang.compile("src/color.frag.slang")

# vert = Shader(ctx, vert_prog.code)
# frag = Shader(ctx, frag_prog.code)

# vert_refl = vert_prog.reflection()
# frag_refl = frag_prog.reflection()

vert = Shader(ctx, Path("res/color.vert.spirv").read_bytes())
frag = Shader(ctx, Path("res/color.frag.spirv").read_bytes())

set0 = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
        DescriptorSetEntry(2, DescriptorType.UNIFORM_BUFFER),
    ],
)
set1 = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)

set0.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)
set0.write_buffer(u_bu3, DescriptorType.UNIFORM_BUFFER, 1, 0)
set0.write_buffer(u_bu3, DescriptorType.UNIFORM_BUFFER, 1, 1)

set1.write_buffer(u_bu2, DescriptorType.UNIFORM_BUFFER, 0, 0)

pipeline = GraphicsPipeline(
    ctx,
    stages = [
        PipelineStage(vert, Stage.VERTEX),
        PipelineStage(frag, Stage.FRAGMENT),
    ],
    vertex_bindings = [
        VertexBinding(0, 12, VertexInputRate.VERTEX),
    ],
    vertex_attributes = [
        VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
    ],
    input_assembly = InputAssembly(PrimitiveTopology.TRIANGLE_LIST),
    push_constants_ranges = [
        PushConstantsRange(12),
    ],
    descriptor_sets = [ set0, set1 ],
    attachments = [
        Attachment(format=window.swapchain_format)
    ]
)

def draw():
    global push_constants

    # swapchain update
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    with gui.frame():
        if imgui.begin("wow"):
            imgui.text("Hello")
            try:
                push_constants = np.array(imgui.color_edit3("Value", tuple(push_constants))[1], np.float32)
            except Exception as e:
                print(e)
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
                cmd.bind_pipeline_state(
                    pipeline=pipeline,
                    descriptor_sets=[ set0, set1 ],
                    push_constants=push_constants.tobytes(),
                    vertex_buffers=[ v_buf ],
                    index_buffer=i_buf,
                    viewport=viewport,
                    scissors=viewport,
                )
                cmd.draw_indexed(I.size)
                gui.render(cmd)

            cmd.use_image(frame.image, ImageUsage.PRESENT)

window.set_callbacks(draw)

while True:
    process_events(True)

    if window.should_close():
        break

    draw()

# if __name__ == "__main__":
#     run()