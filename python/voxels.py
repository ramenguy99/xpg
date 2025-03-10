from pyxpg import *
from pyxpg import imgui
from pyxpg import slang
from pathlib import Path
import numpy as np
from time import perf_counter
from pipelines import PipelineCache, Pipeline
import hashlib
from platformdirs import user_cache_path

# TODO:
# [x] Fix indices (make small diagram)
# [x] Fix lighting
# [ ] Depth buffer
# [ ] Per voxel color
# [ ] Pack voxel data
# [ ] MSAA

scalar_to_np = {
    slang.ScalarKind.Float32: np.float32,
}

def to_dtype(typ: slang.Type) -> np.dtype:
    if   isinstance(typ, slang.Scalar):
        return scalar_to_np[typ.base]
    elif isinstance(typ, slang.Vector):
        return np.dtype((scalar_to_np[typ.base], (typ.count,)))
    elif isinstance(typ, slang.Matrix):
        return np.dtype((scalar_to_np[typ.base], (typ.rows, typ.columns)))
    elif isinstance(typ, slang.Array):
        return np.dtype((to_dtype(typ.type), (typ.count,)))
    elif isinstance(typ, slang.Struct):
        d = {}
        for f in typ.fields:
            d[f.name] = (to_dtype(f.type), f.offset)
        return np.dtype(d)
    else:
        raise TypeError("Unkown type")

S = 1
N = 10
R = 15
x, y = np.meshgrid(np.linspace(-R, R, N), np.linspace(-R, R, N))
# x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
voxels = np.vstack((x.flatten(), y.flatten(), np.zeros_like(y.flatten()), np.ones_like(y.flatten()))).T.astype(np.float32)
# voxels=np.array([[0, 0, 0, 1]])
print(voxels)

I = np.tile(np.array([
    [0, 1, 3],
    [3, 0, 2],
    [0, 4, 5],
    [1, 0, 5],
    [0, 2, 4],
    [2, 4, 6],
], np.uint32), (voxels.shape[0], 1))
I = (I.reshape((voxels.shape[0], -1)) + np.arange(voxels.shape[0]).reshape(voxels.shape[0], 1) * 8).astype(np.uint32)
print(I)

ctx = Context()
window = Window(ctx, "Voxels", 1280, 720)
gui = Gui(window)

ar = window.fb_width / window.fb_height
fov = 45
t = np.tan(fov / 2)
near = 0.1
far = 100.0

proj = np.array([
    [1 / (ar * t),            0,                               0,  0],
    [           0,        1 / t,                               0,  0],
    [           0,            0,     (near + far) / (near - far), -1],
    [           0,            0, (2 * near * far) / (near - far),  0],
], dtype=np.float32)
proj = proj.T

camera_pos = np.array([20, 20, 40], np.float32)
camera_target = np.array([0, 0, 0], np.float32)
up = np.array([0, 0, 1], np.float32)

def normalize(a):
    return a / np.linalg.norm(a)

f = normalize(camera_target - camera_pos)
s = normalize(np.cross(f, up))
u = np.cross(s, f)
view = np.eye(4, dtype=np.float32)
view[:3, 0] = s
view[:3, 1] = u
view[:3, 2] = f
view[3, 0] = -np.dot(s, camera_pos)
view[3, 1] = -np.dot(u, camera_pos)
view[3, 2] = np.dot(f, camera_pos)
view = view.T
# view = np.eye(4, dtype=np.float32)

index_buf = Buffer.from_data(ctx, I.tobytes(), BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)
voxels_buf = Buffer.from_data(ctx, voxels.tobytes(), BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)
depth = Image(ctx, window.fb_width, window.fb_height, Format.D32_SFLOAT, ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT, AllocType.DEVICE_DEDICATED)

set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
    ],
)

set.write_buffer(voxels_buf, DescriptorType.STORAGE_BUFFER, 1, 0)

pipeline: Pipeline = None
def create_pipeline():
    global pipeline
    global buf

    wait_idle(ctx)

    print("Rebuilding pipeline...", end="", flush=True)
    vert_prog = slang.compile("shaders/voxels.vert.slang", "main")
    frag_prog = slang.compile("shaders/voxels.frag.slang", "main")

    refl = vert_prog.reflection
    dt = to_dtype(refl.resources[0].type)

    u_buf = Buffer(ctx, dt.itemsize, BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
    set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

    buf = u_buf.view.view(dt)
    buf["projection"] = proj
    buf["view"] = view
    buf["camera_pos"] = camera_pos
    buf["size"] = S

    vert = Shader(ctx, vert_prog.code)
    frag = Shader(ctx, frag_prog.code)

    pipeline = GraphicsPipeline(
        ctx,
        stages = [
            PipelineStage(vert, Stage.VERTEX),
            PipelineStage(frag, Stage.FRAGMENT),
        ],
        input_assembly = InputAssembly(PrimitiveTopology.TRIANGLE_LIST),
        descriptor_sets = [ set ],
        attachments = [
            Attachment(format=window.swapchain_format)
        ],
        depth = Depth(format=Format.D32_SFLOAT, test=True, write=True, op=CompareOp.LESS),
    )

    print(" Done")

cache = PipelineCache([
    Pipeline(create_pipeline, ["shaders/color.vert.slang", "shaders/color.frag.slang"]),
])

def draw():
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
        imgui.end()

    with window.frame() as frame:
        with frame.command_buffer as cmd:
            cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)
            cmd.use_image(depth, ImageUsage.DEPTH_STENCIL_ATTACHMENT)

            viewport = [0, 0, window.fb_width, window.fb_height]

            # Render voxels
            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.CLEAR,
                        store_op=StoreOp.STORE,
                        clear=[0.9, 0.9, 0.9, 1],
                    ),
                ],
                depth = DepthAttachment(
                    depth,
                    load_op=LoadOp.CLEAR,
                    store_op=StoreOp.STORE,
                    clear=1.0,
                )):
                cmd.bind_pipeline_state(
                    pipeline=pipeline,
                    descriptor_sets=[ set ],
                    index_buffer=index_buf,
                    viewport=viewport,
                    scissors=viewport,
                )

                cmd.draw_indexed(I.size)

            # Render gui
            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.LOAD,
                        store_op=StoreOp.STORE,
                    ),
                ]):
                gui.render(cmd)

            cmd.use_image(frame.image, ImageUsage.PRESENT)

window.set_callbacks(draw)

while True:
    process_events(True)

    if window.should_close():
        break

    draw()

cache.stop()
# if __name__ == "__main__":
#     run()