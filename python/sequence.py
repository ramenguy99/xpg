from pyxpg import *

import numpy as np
from pathlib import Path
from time import perf_counter
from pyglm.glm import perspectiveZO, lookAt, vec3

from utils.pipelines import PipelineWatch, Pipeline
from utils.reflection import to_dtype, DescriptorSetsReflection
from utils.render import PerFrameResource
from utils.buffered_stream import BufferedStream
from utils.utils import profile
import io
import struct

def read_exact_into(file: io.FileIO, view: memoryview):
    bread = 0
    while bread < len(view):
        n = file.readinto(view[bread:])
        if n == 0:
            raise EOFError()
        else:
            bread += n

def read_exact(file: io.FileIO, size: int):
    out = bytearray(size)
    view = memoryview(out)
    read_exact_into(file, view)
    return out

def read_exact_at_offset_into(file: io.FileIO, offset: int, view: memoryview):
    file.seek(offset, io.SEEK_SET)
    return read_exact_into(file, view)

def read_exact_at_offset(file: io.FileIO, offset: int, size: int):
    file.seek(offset, io.SEEK_SET)
    return read_exact(file, size)

# TODO:
# [x] Add buffer copy
# [x] Add memory barrier
# [ ] Sync issue still? Maybe wrong buffer in use?
# [ ] Visualize buffered stream state -> ImGui bindings
# [ ] Copy queue (synchronization likely similar to external buffer stuff?)
# [ ] Add buffer barriers with cross queue sync
# [ ] Advance buffer only if frame is new

WORKERS = 4
files = [open("N:\\scenes\\smpl\\all_frames_20.bin", "rb", buffering=0) for _ in range(WORKERS)]
file = files[0]

header = read_exact(file, 12)
N = struct.unpack("<I", header[0: 4])[0]
V = struct.unpack("<I", header[4: 8])[0]
I = struct.unpack("<I", header[8:12])[0]

indices = np.frombuffer(read_exact_at_offset(file, N * V * 12 + len(header), I * 4), np.uint32)

ctx = Context(
    device_features=DeviceFeatures.DYNAMIC_RENDERING | DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.PRESENTATION, 
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

window = Window(ctx, "Sequence", 1600, 900)
gui = Gui(window)

BUFFERS = 8
cpu_v_bufs = [Buffer(ctx, V * 12, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST) for _ in range(BUFFERS)]
gpu_v_bufs = PerFrameResource(Buffer, window.num_frames, ctx, V * 12, BufferUsageFlags.VERTEX | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE_MAPPED)
# gpu_v_bufs = PerFrameResource(Buffer, window.num_frames, ctx, V * 12, BufferUsageFlags.VERTEX, AllocType.DEVICE_MAPPED)

i_buf = Buffer.from_data(ctx, indices.tobytes(), BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)

sets = PerFrameResource(DescriptorSet, window.num_frames,
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)

SHADERS = Path(__file__).parent.joinpath("shaders")
class SequencePipeline(Pipeline):
    vert_prog = Path(SHADERS, "sequence.vert.slang")
    frag_prog = Path(SHADERS, "sequence.frag.slang")

    def create(self, vert_prog: slang.Shader, frag_prog: slang.Shader):
        global constants_dt, u_bufs

        # Reflection
        refl = vert_prog.reflection
        desc_refl = DescriptorSetsReflection(refl)
        constants_dt = to_dtype(desc_refl.descriptors["constants"].resource.type)

        # Create uniform buffers and write descriptors
        u_bufs = PerFrameResource(Buffer, window.num_frames, ctx, constants_dt.itemsize, BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
        for set, u_buf in zip(sets.resources, u_bufs.resources):
            set: DescriptorSet
            set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        # Turn SPIR-V code into vulkan shader modules
        vert = Shader(ctx, vert_prog.code)
        frag = Shader(ctx, frag_prog.code)

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = GraphicsPipeline(
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
            descriptor_sets = [ set ],
            attachments = [
                Attachment(format=window.swapchain_format)
            ],
            depth = Depth(format=Format.D32_SFLOAT, test=True, write=True, op=CompareOp.LESS),
        )

pipeline = SequencePipeline()
cache = PipelineWatch([
    pipeline,
], window=window)

depth: Image = None
first_frame = True
frame_index = 0
last_timestamp = perf_counter()
animation_time = 0
animation_fps = 25
animation_playing = False

from threading import Lock
lock = Lock()
def load(thread_index: int, buffer_index: int, frame_index: int):
    buf = cpu_v_bufs[buffer_index]
    file = files[thread_index]

    # with lock:
    # print(f"Thread {thread_index} loading {buffer_index} ({buf}) with frame {frame_index}")

    # with profile(f"Thread {thread_index} loading {buffer_index} ({buf}) with frame {frame_index}"):
    #     read_exact_at_offset_into(file, 12 + frame_index * V * 12, buf.view.view())
    read_exact_at_offset_into(file, 12 + frame_index * V * 12, buf.view.view())
    return buf

bufstream = BufferedStream(N, BUFFERS, WORKERS, load)

def draw():
    global depth
    global first_frame, frame_index, last_timestamp
    global animation_time, animation_playing

    timestamp = perf_counter()
    dt = timestamp - last_timestamp
    last_timestamp = timestamp

    animation_frame_index = int(animation_time * animation_fps) % N

    cache.refresh(lambda: ctx.wait_idle())

    # Update swapchain
    swapchain_status = window.update_swapchain()
    if swapchain_status == SwapchainStatus.MINIMIZED:
        return
    
    images_just_created = False
    if first_frame or swapchain_status == SwapchainStatus.RESIZED:
        first_frame = False

        if depth:
            depth.destroy()
        depth = Image(ctx, window.fb_width, window.fb_height, Format.D32_SFLOAT, ImageUsageFlags.DEPTH_STENCIL_ATTACHMENT, AllocType.DEVICE_DEDICATED)
        images_just_created = True

    # Camera
    camera_position = vec3(2, -8, 2) * 1.0
    camera_target = vec3(0, 0, 0)
    fov = 45.0
    ar = window.fb_width / window.fb_height
    transform = perspectiveZO(fov, ar, 0.01, 100.0) * lookAt(camera_position, camera_target, vec3(0, 1, 0))

    descriptor_set = sets.get_current_and_advance()
    u_buf: Buffer = u_bufs.get_current_and_advance()
    v_buf: Buffer = gpu_v_bufs.get_current_and_advance()

    cpu_v_buf: Buffer = bufstream.get_frame(animation_frame_index)

    # with lock:
    #     print(f"Drawing frame {frame_index}: {animation_frame_index} {cpu_v_buf} {v_buf}")

    constants: np.ndarray = np.zeros(1, dtype=constants_dt)
    constants["transform"] = transform

    u_buf.view.data[:] = constants.view(np.uint8).data

    # GUI
    with gui.frame():
        if imgui.begin("stats"):
            imgui.text(f"indices: {I} ({I * 4 / 1024 / 1024:.2f}MB)")
            imgui.text(f"vertices: {V} ({V * 12 / 1024 / 1024:.2f}MB)")
            imgui.text(f"vertices gpu buffers: {v_buf.view.size / 1024 / 1024 * len(gpu_v_bufs.resources):.2f}MB")
            imgui.text(f"vertices load buffers: {v_buf.view.size / 1024 / 1024 * len(cpu_v_bufs):.2f}MB")
        imgui.end()
        if imgui.begin("wow"):
            imgui.text(f"dt: {dt * 1000:.3f}ms")
            imgui.text(f"animation time: {animation_time:.1f}s")
            _, _ = imgui.slider_int("animation frame", animation_frame_index, 0, N - 1)
            _, animation_playing = imgui.checkbox("Playing", animation_playing)
        imgui.end()

    # Render
    with window.frame() as frame:
        with frame.command_buffer as cmd:
            cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)
            if images_just_created:
                cmd.use_image(depth, ImageUsage.DEPTH_STENCIL_ATTACHMENT)
            
            # cmd.copy_buffer(cpu_v_buf, v_buf)
            # cmd.memory_barrier(MemoryUsage.TRANSFER_WRITE, MemoryUsage.VERTEX_INPUT)
            v_buf.view.data[:] = cpu_v_buf.view.data

            viewport = [0, 0, window.fb_width, window.fb_height]
            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.CLEAR,
                        store_op=StoreOp.STORE,
                        clear=[0.1, 0.2, 0.4, 1],
                    ),
                ],
                depth=DepthAttachment(depth, load_op=LoadOp.CLEAR, store_op=StoreOp.STORE, clear=1.0)
            ):
                # Bind the pipeline
                cmd.bind_graphics_pipeline(
                    pipeline=pipeline.pipeline,
                    descriptor_sets=[ descriptor_set ],
                    vertex_buffers=[ v_buf ],
                    index_buffer=i_buf,
                    viewport=viewport,
                    scissors=viewport,
                )

                # Issue a draw
                cmd.draw_indexed(I)

            # Render gui
            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(frame.image, load_op=LoadOp.LOAD, store_op=StoreOp.STORE),
                ],
            ):
                # Render gui
                gui.render(cmd)

            cmd.use_image(frame.image, ImageUsage.PRESENT)

    frame_index += 1
    if animation_playing:
        animation_time += dt


window.set_callbacks(draw)

while True:
    process_events(False)

    if window.should_close():
        break

    draw()

cache.stop()
bufstream.stop()