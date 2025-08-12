from pathlib import Path
import numpy as np
from typing import Tuple

from pyglm.glm import vec3
from pyxpg import *
from pyxpg import imgui
from pyxpg import slang

from utils.pipelines import PipelineWatch, Pipeline
from utils import reflection
from utils import render
from utils.camera import Camera
from utils.buffers import UploadableBuffer

def grid3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.meshgrid(x, y, z)
    return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

# Params
SAMPLES = 4

# Scene
S = 2
N = 5
R = 10
voxels: np.ndarray = grid3d(np.linspace(-R, R, N, dtype=np.float32), np.linspace(-R, R, N, dtype=np.float32), np.linspace(-R, R, N, dtype=np.float32))

I = np.tile(np.array([
    [0, 1, 3],
    [3, 0, 2],
    [0, 4, 5],
    [1, 0, 5],
    [0, 2, 4],
    [2, 4, 6],
], np.uint32), (voxels.shape[0], 1))
I: np.ndarray = (I.reshape((voxels.shape[0], -1)) + np.arange(voxels.shape[0]).reshape(voxels.shape[0], 1) * 8).astype(np.uint32)

camera = Camera(vec3(30, 30, -30), vec3(0, 0, 0), vec3(0, 0, 1), 45, 1, 0.1, 100.0)

# Init
ctx = Context(
    required_features=DeviceFeatures.SCALAR_BLOCK_LAYOUT | DeviceFeatures.DYNAMIC_RENDERING | DeviceFeatures.SYNCHRONIZATION_2,
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)
window = Window(ctx, "Voxels", 1280, 720)
gui = Gui(window)

index_buf = Buffer.from_data(ctx, I, BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED_WITH_FALLBACK)
voxels_buf = Buffer.from_data(ctx, voxels, BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED_WITH_FALLBACK)

descriptor_sets = render.PerFrameResource(DescriptorSet, window.num_frames,
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
    ],
)

for set in descriptor_sets.resources:
    set: DescriptorSet
    set.write_buffer(voxels_buf, DescriptorType.STORAGE_BUFFER, 1, 0)


SHADERS = Path(__file__).parent.joinpath("shaders")

# Pipeline
class VoxelPipeline(Pipeline):
    vert_prog = Path(SHADERS, "voxels.vert.slang")
    frag_prog = Path(SHADERS, "voxels.frag.slang")

    def create(self, vert_prog: slang.Shader, frag_prog: slang.Shader):
        global dt
        global u_bufs

        refl = vert_prog.reflection
        descs = reflection.DescriptorSetsReflection(refl)
        dt = reflection.to_dtype(descs.descriptors["u"].resource.type)

        u_bufs = render.PerFrameResource(UploadableBuffer, window.num_frames, ctx, dt.itemsize, BufferUsageFlags.UNIFORM)
        for set, u_buf in zip(descriptor_sets.resources, u_bufs.resources):
            set: DescriptorSet
            set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        vert = Shader(ctx, vert_prog.code)
        frag = Shader(ctx, frag_prog.code)

        self.pipeline = GraphicsPipeline(
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

voxels = VoxelPipeline()
cache = PipelineWatch([
    voxels,
], window)

first_frame: bool = True
depth: Image = None
msaa_target: Image = None

# Draw
def draw():
    global msaa_target
    global depth
    global first_frame

    cache.refresh(lambda: ctx.wait_idle())

    # swapchain update
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    images_just_created = False
    if first_frame or swapchain_status == SwapchainStatus.RESIZED:
        first_frame = False

        # Update aspect ratio
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
        imgui.set_next_window_pos((50, 50))
        imgui.set_next_window_size((400, 50))
        if imgui.begin(
            "transparent",
            flags=
                imgui.WindowFlags.NO_TITLE_BAR |
                imgui.WindowFlags.NO_BACKGROUND |
                imgui.WindowFlags.NO_MOUSE_INPUTS |
                imgui.WindowFlags.NO_RESIZE
        )[0]:
            imgui.text("Left-click and drag left and right to rotate!")
        imgui.end()

    # Render
    with window.frame() as frame:
        # Per frame uploads
        constants = np.zeros(1, dt)
        constants["projection"] = camera.projection()
        constants["view"] = camera.view()
        constants["camera_pos"] = camera.position
        constants["size"] = S

        u_buf: UploadableBuffer = u_bufs.get_current_and_advance()
        set: DescriptorSet = descriptor_sets.get_current_and_advance()

        # Commands
        with frame.command_buffer as cmd:
            u_buf.upload(cmd, MemoryUsage.VERTEX_SHADER_UNIFORM, constants.view(np.uint8).data)
            cmd.image_barrier(frame.image, ImageLayout.COLOR_ATTACHMENT_OPTIMAL, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.COLOR_ATTACHMENT)

            if images_just_created:
                cmd.image_barrier(depth, ImageLayout.DEPTH_STENCIL_ATTACHMENT_OPTIMAL, MemoryUsage.NONE, MemoryUsage.DEPTH_STENCIL_ATTACHMENT, aspect_mask=ImageAspectFlags.DEPTH)
                if SAMPLES > 1:
                    cmd.image_barrier(msaa_target, ImageLayout.COLOR_ATTACHMENT_OPTIMAL, MemoryUsage.NONE, MemoryUsage.COLOR_ATTACHMENT)

            viewport = [0, 0, window.fb_width, window.fb_height]

            # Render voxels
            with cmd.rendering(viewport,
                color_attachments=[
                    (RenderingAttachment(msaa_target, load_op=LoadOp.CLEAR, store_op=StoreOp.STORE, clear=[0.1, 0.1, 0.1, 1], resolve_mode=ResolveMode.AVERAGE, resolve_image=frame.image)
                     if SAMPLES > 1 else
                     RenderingAttachment(frame.image, load_op=LoadOp.CLEAR, store_op=StoreOp.STORE, clear=[0.1, 0.1, 0.1, 1])),
                ],
                depth = DepthAttachment(depth, load_op=LoadOp.CLEAR, store_op=StoreOp.STORE, clear=1.0)
            ):
                cmd.set_viewport(viewport)
                cmd.set_scissors(viewport)
                cmd.bind_graphics_pipeline(
                    pipeline=voxels.pipeline,
                    descriptor_sets=[ set ],
                    index_buffer=index_buf,
                )

                cmd.draw_indexed(I.size)

            # Render gui
            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(frame.image, load_op=LoadOp.LOAD, store_op=StoreOp.STORE),
                ],
            ):
                gui.render(cmd)

            cmd.image_barrier(frame.image, ImageLayout.PRESENT_SRC, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.PRESENT)


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
    process_events(False)

    if window.should_close():
        break

    draw()

cache.stop()