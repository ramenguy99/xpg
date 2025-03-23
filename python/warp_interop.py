from pyxpg import *
from pyxpg import imgui
from pyxpg import slang
from pathlib import Path
import numpy as np
from pipelines import PipelineCache, Pipeline, compile
from reflection import to_dtype
from time import perf_counter

import warp as wp

wp.init()

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

# v_buf = Buffer.from_data(ctx, V.tobytes(), BufferUsageFlags.VERTEX, AllocType.DEVICE_MAPPED)
size = 4 * 3 * 4
v_buf = ExternalBuffer(ctx, size, BufferUsageFlags.VERTEX, AllocType.DEVICE)
i_buf = Buffer.from_data(ctx, I.tobytes(), BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)

set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)
pipeline: Pipeline = None

# a = wp.empty(shape=4, dtype=wp.vec3, device="cuda")
ext_buf = wp.ExternalMemoryBuffer(v_buf.handle, wp.ExternalMemoryBuffer.HANDLE_TYPE_OPAQUEWIN32, size)
a = ext_buf.map(wp.vec3, shape=4)

def create_pipeline():
    global pipeline
    global buf

    wait_idle(ctx)

    print("Rebuilding pipeline...", end="", flush=True)
    vert_prog = compile(Path("shaders/color.vert.slang"), "main")
    frag_prog = compile(Path("shaders/color.frag.slang"), "main")

    refl = vert_prog.reflection
    dt = to_dtype(refl.resources[0].type)

    u_buf = Buffer(ctx, dt.itemsize, BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
    set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

    buf = u_buf.view.view(dt)
    buf["transform"] = rot
    buf["nest1"]["val2"] = push_constants

    vert = Shader(ctx, vert_prog.code)
    frag = Shader(ctx, frag_prog.code)

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
        descriptor_sets = [ set ],
        attachments = [
            Attachment(format=window.swapchain_format)
        ]
    )

    print(" Done")

cache = PipelineCache([
    Pipeline(create_pipeline, ["shaders/color.vert.slang", "shaders/color.frag.slang"]),
])

@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3), t: wp.float32):
    # load two vec3s
    s = wp.sin(t) * 0.5 + 0.5
    a[0] = wp.vec3(-0.5, -0.5, 0.0) * s
    a[1] = wp.vec3( 0.5, -0.5, 0.0) * s
    a[2] = wp.vec3(-0.5,  0.5, 0.0) * s
    a[3] = wp.vec3( 0.5,  0.5, 0.0) * s

def draw():
    global push_constants

    cache.refresh()

    # swapchain update
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    t = perf_counter()
    wp.launch(kernel=simple_kernel, # kernel to launch
            dim=1,                # number of threads
            inputs=[a, t],        # parameters
            device="cuda")        # execution device

    with gui.frame():
        if imgui.begin("wow"):
            imgui.text("Hello")
            try:
                updated, v = imgui.color_edit3("Value", tuple(push_constants))
                if updated:
                    push_constants = np.array(v, np.float32)
                    buf["nest1"]["val2"] = v
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
                    descriptor_sets=[ set ],
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
    process_events(False)

    if window.should_close():
        break

    draw()

cache.stop()
# if __name__ == "__main__":
#     run()