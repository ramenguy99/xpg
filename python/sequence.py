from pyxpg import *

import numpy as np
from pathlib import Path
from time import perf_counter
from pyglm.glm import perspectiveZO, lookAt, vec3

from utils.pipelines import PipelineWatch, Pipeline
from utils.reflection import to_dtype, DescriptorSetsReflection
from utils.render import PerFrameResource
from utils.threadpool import ThreadPool
from threading import Event, Lock
import io
import struct
from utils.loaders import LRUPool
from utils.utils import profile
from typing import Optional

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
    preferred_frames_in_flight=2,
)

window = Window(ctx, "Sequence", 1600, 900)
gui = Gui(window)

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

NUM_WORKERS = 4
pool = ThreadPool(NUM_WORKERS)

class CpuBuffer:
    def __init__(self, size: int, name: Optional[str] = None):
        self.buf = Buffer(ctx, size, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST, name)
        self.event = Event()
    
    def __repr__(self):
        return self.buf.__repr__()

PREFETCH_SIZE = 2
BUFS = window.num_frames + PREFETCH_SIZE
cpu = LRUPool([CpuBuffer(V * 12, name=f"cpubuf{i}") for i in range(BUFS)], window.num_frames, PREFETCH_SIZE)

lock = Lock()
def load(thread_index: int, i: int, buf: CpuBuffer):
    file = files[thread_index]
    read_exact_at_offset_into(file, 12 + i * V * 12, buf.buf.view.view())
    buf.event.set()
    return buf

GPU_BUFFERS = window.num_frames
class GpuBuffer:
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, use_transfer_queue: bool, name: Optional[str] = None):
        self.buf = Buffer(ctx, size, usage_flags | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE_MAPPED, name=name)

        if use_transfer_queue:
            self.used = False
            self.use_done = Semaphore(ctx)
            self.upload_done = Semaphore(ctx)
        
    def __repr__(self):
        return self.buf.__repr__()

USE_TRANSFER_QUEUE = ctx.has_transfer_queue
gpu = LRUPool([GpuBuffer(ctx, V * 12, BufferUsageFlags.VERTEX, USE_TRANSFER_QUEUE, name=f"gpubuf{i}") for i in range(GPU_BUFFERS)], window.num_frames)

# TODO:
# [x] Implement copy on transfer queue
# [x] Try to implement prefetching on top of this
#   -> keep in mind RR prefetching as a goal, ideally pre-fetch policy is external / switchable
# [x] Take out this LRU implementation and make it somehow extendable? Lambdas?
# [x] Add optional names to resources, use in __repr__
# [ ] ImGui Profiler?

def draw():
    global depth
    global first_frame, frame_index, last_timestamp
    global animation_time, animation_playing

    global cpu, gpu

    timestamp = perf_counter()
    dt = timestamp - last_timestamp
    last_timestamp = timestamp
    # print(f"dt: {dt * 1000:.3f}ms")

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

    constants: np.ndarray = np.zeros(1, dtype=constants_dt)
    constants["transform"] = transform

    u_buf.view.data[:] = constants.view(np.uint8).data

    # GUI
    with gui.frame():
        if imgui.begin("stats"):
            imgui.text(f"indices: {I} ({I * 4 / 1024 / 1024:.2f}MB)")
            imgui.text(f"vertices: {V} ({V * 12 / 1024 / 1024:.2f}MB)")
            imgui.text(f"vertices gpu buffers: {V * 12 / 1024 / 1024 * GPU_BUFFERS:.2f}MB")
            imgui.text(f"vertices load buffers: {V * 12 / 1024 / 1024 * BUFS:.2f}MB")
        imgui.end()
        if imgui.begin("playback"):
            imgui.text(f"dt: {dt * 1000:.3f}ms")
            imgui.text(f"animation time: {animation_time:.1f}s")
            changed, idx = imgui.slider_int("animation frame", animation_frame_index, 0, N - 1)
            if changed:
                animation_frame_index = idx
                animation_time = idx / animation_fps
            _, animation_playing = imgui.checkbox("Playing", animation_playing)
            
            start = imgui.get_cursor_screen_pos()
            dl = imgui.get_window_draw_list()
            for i in range(N):
                cursor = imgui.Vec2(start.x + 5 * i, start.y)
                dl.add_rect(cursor, (cursor.x + 6, cursor.y + 20), 0xFFFFFFFF)
                dl.add_rect((cursor.x, cursor.y + 22), (cursor.x + 6, cursor.y + 42), 0xFFFFFFFF)

            for k, v in cpu.lookup.items():
                cursor = imgui.Vec2(start.x + 5 * k, start.y)
                color = 0xFF00FF00 if not v.prefetching else 0xFF00FFFF
                dl.add_rect_filled((cursor.x + 1, cursor.y + 1), (cursor.x + 5, cursor.y + 19), color)

            for i in gpu.lookup.keys():
                cursor = imgui.Vec2(start.x + 5 * i, start.y)
                dl.add_rect_filled((cursor.x + 1, cursor.y +23), (cursor.x + 5, cursor.y + 41), 0xFF00FF00)

        imgui.end()
        if imgui.begin("cache"):
            def drawpool(name: str, pool: LRUPool):
                imgui.separator_text(name)
                imgui.text(f"Map")
                imgui.indent()
                for k, v in pool.lookup.items():
                    imgui.text(f"{k:03d} {v}")
                imgui.unindent()

                imgui.text(f"LRU")
                imgui.indent()
                i = 0
                for k, v in pool.lru.items():
                    imgui.text(f"{k} {v}")
                    i += 1
                for _ in range(i, BUFS):
                    imgui.text(f"<EMPTY>")
                imgui.unindent()

                imgui.text(f"In Flight")
                imgui.indent()
                for v in pool.in_flight:
                    imgui.text(f"{v}")
                imgui.unindent()

            drawpool("CPU", cpu)
            drawpool("GPU", gpu)
        imgui.end()
    
    # Upload

    # Render
    frame = window.begin_frame()
    additional_wait_semaphores = []
    additional_signal_semaphores = []
    with frame.command_buffer as cmd:
        ####################################################################
        # Init

        cpu.new_frame(frame_index)
        gpu.new_frame(frame_index)

        ####################################################################
        # Get

        # Check if already uploaded
        def cpu_ensure_fetched(buf: CpuBuffer):
            # with profile("Wait"):
            if True:
                buf.event.wait()
            buf.event.clear()

        def cpu_load(k: int, buf: CpuBuffer):
            # with profile("Load"):
            if True:
                pool.submit(load, k, buf)
                buf.event.wait()
            buf.event.clear()

        def gpu_load(k: int, gpu_buf: GpuBuffer):
            cpu_buf = cpu.get(k, cpu_load, cpu_ensure_fetched)

            # Upload from CPU
            if False:
                # If using mapped buffer
                gpu_buf.buf.view.data[:] = cpu_buf.buf.view.data
                
                # Buffer is immediately not in use anymore. Add back to the LRU.
                # This moves back the buffer to the front of the LRU queue.
                cpu.give_back(k, cpu_buf)
            else:
                cpu.use(frame_index, k)
                if not USE_TRANSFER_QUEUE:
                    # Upload on gfx queue
                    cmd.copy_buffer(cpu_buf.buf, gpu_buf.buf)
                    cmd.memory_barrier(MemoryUsage.TRANSFER_WRITE, MemoryUsage.VERTEX_INPUT)
                else:
                    # Upload on copy queue
                    with frame.transfer_queue_commands(
                        wait_semaphores = [(gpu_buf.use_done, PipelineStageFlags.TRANSFER)] if gpu_buf.used else [],
                        signal_semaphores = [gpu_buf.upload_done],
                    ) as copy_cmd:
                        copy_cmd.copy_buffer(cpu_buf.buf, gpu_buf.buf)
                        copy_cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.TRANSFER_WRITE, MemoryUsage.NONE, ctx.transfer_queue_family_index, ctx.graphics_queue_family_index)
                    cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.NONE, MemoryUsage.VERTEX_INPUT, ctx.transfer_queue_family_index, ctx.graphics_queue_family_index)

                    # Add semaphores for graphics queue
                    additional_wait_semaphores.append((gpu_buf.upload_done, PipelineStageFlags.VERTEX_INPUT))
                    additional_signal_semaphores.append(gpu_buf.use_done)
                    gpu_buf.used = True

        gpu_buf = gpu.get(animation_frame_index, gpu_load)
        gpu.use(frame_index, animation_frame_index)

        ####################################################################
        # Prefetch
        
        prefetch_start = animation_frame_index + 1
        prefetch_end = prefetch_start + PREFETCH_SIZE
        def prefetch_check_loaded(buf: CpuBuffer):
            if buf.event.is_set():
                buf.event.clear()
                return True
            return False
        cpu.prefetch([i % N for i in range(prefetch_start, prefetch_end)], prefetch_check_loaded, lambda k, buf: pool.submit(load, k, buf))

        ####################################################################

        
        cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)
        if images_just_created:
            cmd.use_image(depth, ImageUsage.DEPTH_STENCIL_ATTACHMENT)
        
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
                vertex_buffers=[ gpu_buf.buf ],
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
    window.end_frame(frame, additional_wait_semaphores, additional_signal_semaphores)

    frame_index = (frame_index + 1) % window.num_frames
    if animation_playing:
        animation_time += dt

def on_key(k: Key, a: Action, m: Modifiers):
    global animation_time, animation_playing
    if a == Action.PRESS:
        if k == Key.SPACE:
            animation_playing = not animation_playing
    if a == Action.PRESS or a == Action.REPEAT:
        if k == Key.PERIOD:
            animation_time += 1 / animation_fps
        if k == Key.COMMA:
            animation_time -= 1 / animation_fps

window.set_callbacks(
    draw,
    key_event=on_key,
)

while True:
    process_events(False)

    if window.should_close():
        break

    draw()

cache.stop()