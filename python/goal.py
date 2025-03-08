from pyxpg import *
from pyxpg import imgui
from pathlib import Path
import numpy as np
from time import perf_counter
from pipelines import PipelineCache, Pipeline
import hashlib
from platformdirs import user_cache_path

# frag_prog = slang.compile("shaders/color.frag.slang", "main")
# exit(1)

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


set0 = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
        # DescriptorSetEntry(2, DescriptorType.UNIFORM_BUFFER),
    ],
)
# set1 = DescriptorSet(
#     ctx,
#     [
#         DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
#     ],
# )

set0.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)
# set0.write_buffer(u_bu3, DescriptorType.UNIFORM_BUFFER, 1, 0)
# set0.write_buffer(u_bu3, DescriptorType.UNIFORM_BUFFER, 1, 1)

# set1.write_buffer(u_bu2, DescriptorType.UNIFORM_BUFFER, 0, 0)

pipeline: Pipeline = None

def compile(file: Path, entry: str):
    # TODO: 
    # [ ] This cache does not currently consider imported files / modules and slang target, compiler version, compilation defines / params (if any)
    # [ ] No obvious way to clear the cache, maybe should be have some LRU with max size? e.g. touch files when using them
    name = f"{hashlib.sha256(file.read_bytes()).digest().hex()}_{entry}.spirv"

    # Check cache
    cache_dir = user_cache_path("pyxpg")
    path = Path(cache_dir, name)
    if path.exists():
        return path.read_bytes()
    
    # Create prog
    prog = slang.compile(str(file), entry)
    # refl = prog.reflection()

    # Populate to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(prog.code)

    return prog.code

def create_pipeline():
    global pipeline

    wait_idle(ctx)

    print("Rebuilding pipeline...", end="", flush=True)
    vert_prog = compile(Path("shaders/color.vert.slang"), "main")
    frag_prog = compile(Path("shaders/color.frag.slang"), "main")

    # frag_refl = frag_prog.reflection()

    vert = Shader(ctx, vert_prog)
    frag = Shader(ctx, frag_prog)

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
        descriptor_sets = [ set0 ],
        # descriptor_sets = [ set0, set1 ],
        attachments = [
            Attachment(format=window.swapchain_format)
        ]
    )

    print(" Done")

cache = PipelineCache([
    Pipeline(create_pipeline, ["shaders/color.vert.slang", "shaders/color.frag.slang"]),
])

def draw():
    global push_constants

    cache.refresh()

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
                    # descriptor_sets=[ set0, set1 ],
                    descriptor_sets=[ set0 ],
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