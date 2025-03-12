from pyxpg import *
from pyxpg import imgui
from pyxpg import slang
from pathlib import Path
import numpy as np
from time import perf_counter
from pipelines import PipelineCache, Pipeline
import hashlib
from platformdirs import user_cache_path
from typing import Tuple

import gfxmath
from gfxmath import Vec3
import reflection
import renderutils
from camera import Camera

# TODO:
# [x] Fix indices (make small diagram)
# [x] Fix lighting
# [x] Depth buffer
# [x] MSAA
# [x] Triple buffering
# [x] Better camera utils
#     [x] Debug perspective / lookat
# [ ] Per voxel color
# [ ] Pack voxel data

# Scene

# S = 0.5
# N = 15
# R = 10
SAMPLES = 4

S = 2
N = 5
R = 10
voxels = gfxmath.grid3d(np.linspace(-R, R, N, dtype=np.float32), np.linspace(-R, R, N, dtype=np.float32), np.linspace(-R, R, N, dtype=np.float32))

I = np.tile(np.array([
    [0, 1, 3],
    [3, 0, 2],
    [0, 4, 5],
    [1, 0, 5],
    [0, 2, 4],
    [2, 4, 6],
], np.uint32), (voxels.shape[0], 1))
I = (I.reshape((voxels.shape[0], -1)) + np.arange(voxels.shape[0]).reshape(voxels.shape[0], 1) * 8).astype(np.uint32)


camera = Camera(Vec3(40, 40, 40), Vec3(0, 0, 0), Vec3(0, 0, 1), 45, 1, 0.1, 100.0)

# Init

ctx = Context()
window = Window(ctx, "Voxels", 1280, 720)
gui = Gui(window)

index_buf = Buffer.from_data(ctx, I.tobytes(), BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)
voxels_buf = Buffer.from_data(ctx, voxels.tobytes(), BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)

descriptor_sets = renderutils.PerFrameResource(DescriptorSet, window.num_frames,
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
    ],
)

for set in descriptor_sets.resources:
    set: DescriptorSet
    set.write_buffer(voxels_buf, DescriptorType.STORAGE_BUFFER, 1, 0)


# Pipeline
pipeline: Pipeline = None
def create_pipeline():
    global pipeline
    global dt
    global u_bufs

    wait_idle(ctx)

    print("Rebuilding pipeline...", end="", flush=True)
    vert_prog = slang.compile("shaders/voxels.vert.slang", "main")
    frag_prog = slang.compile("shaders/voxels.frag.slang", "main")

    refl = vert_prog.reflection
    dt = reflection.to_dtype(refl.resources[0].type)

    u_bufs = renderutils.PerFrameResource(Buffer, window.num_frames, ctx, dt.itemsize, BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
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
        samples=SAMPLES,
        attachments = [
            Attachment(format=window.swapchain_format)
        ],
        depth = Depth(format=Format.D32_SFLOAT, test=True, write=True, op=CompareOp.LESS),
    )

    print(" Done")

cache = PipelineCache([
    Pipeline(create_pipeline, ["shaders/color.vert.slang", "shaders/color.frag.slang"]),
])

first_frame: bool = True
depth: Image = None
msaa_target: Image = None

# Draw
def draw():
    global msaa_target
    global depth
    global first_frame
    global proj

    cache.refresh()

    # swapchain update
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    images_just_created = False
    if first_frame or swapchain_status == SwapchainStatus.RESIZED:
        first_frame = False

        # Refresh proj
        camera.ar = window.fb_width / window.fb_height

        # Resize depth
        if depth:
            depth.destroy()
        depth = Image(ctx, window.fb_width, window.fb_height, Format.D32_SFLOAT, ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT, AllocType.DEVICE_DEDICATED, samples=SAMPLES)
        if SAMPLES > 1:
            msaa_target = Image(ctx, window.fb_width, window.fb_height, window.swapchain_format, ImageUsageFlags.COLOR_ATTACHMENT, AllocType.DEVICE_DEDICATED, samples=SAMPLES)
        images_just_created = True

    # GUI
    with gui.frame():
        if imgui.begin("wow"):
            imgui.text("Hello")
        imgui.end()

    # Render
    with window.frame() as frame:
        # Per frame uploads
        u_buf: Buffer = u_bufs.get_current_and_advance()
        u_buf_view = u_buf.view.view(dt)
        u_buf_view["projection"] = camera.projection()
        u_buf_view["view"] = camera.view()
        u_buf_view["camera_pos"] = camera.position
        u_buf_view["size"] = S

        set: DescriptorSet = descriptor_sets.get_current_and_advance()
        set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        # Commands
        with frame.command_buffer as cmd:
            cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)

            if images_just_created:
                cmd.use_image(depth, ImageUsage.DEPTH_STENCIL_ATTACHMENT)
                if SAMPLES > 1:
                    cmd.use_image(msaa_target, ImageUsage.COLOR_ATTACHMENT)

            viewport = [0, 0, window.fb_width, window.fb_height]

            # Render voxels
            with cmd.rendering(viewport,
                color_attachments=[
                    (RenderingAttachment(msaa_target, load_op=LoadOp.CLEAR, store_op=StoreOp.STORE, clear=[0.9, 0.9, 0.9, 1], resolve_mode=ResolveMode.AVERAGE, resolve_image=frame.image)
                     if SAMPLES > 1 else
                     RenderingAttachment(frame.image, load_op=LoadOp.CLEAR, store_op=StoreOp.STORE, clear=[0.9, 0.9, 0.9, 1])),
                ],
                depth = DepthAttachment(depth, load_op=LoadOp.CLEAR, store_op=StoreOp.STORE, clear=1.0)
            ):
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
                    RenderingAttachment(frame.image, load_op=LoadOp.LOAD, store_op=StoreOp.STORE),
                ],
            ):
                gui.render(cmd)

            cmd.use_image(frame.image, ImageUsage.PRESENT)


drag_start = None

def mouse_move_event(p: Tuple[int, int]):
    global drag_start, camera
    if drag_start:
        delta = np.array((p[0] - drag_start[0], p[1] - drag_start[1]), dtype=np.float32)

        t = delta[0] * 1e-3
        s = np.sin(t)
        c = np.cos(t)

        v = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ], np.float32) @ (np.array(camera.position) - camera.target)
        camera.position = camera.target + v

        drag_start = p

def mouse_button_event(p: Tuple[int, int], button: MouseButton, action: Action, mods: Modifiers):
    global drag_start
    if button == MouseButton.LEFT:
        if action == Action.PRESS:
            drag_start = p
        elif action == Action.RELEASE:
            drag_start = None

window.set_callbacks(
    draw,
    mouse_move_event=mouse_move_event,
    mouse_button_event=mouse_button_event,
)

while True:
    process_events(True)

    if window.should_close():
        break

    draw()

cache.stop()
# if __name__ == "__main__":
#     run()