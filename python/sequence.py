from pyxpg import *

import numpy as np
from pathlib import Path
from time import perf_counter
from pyglm.glm import perspectiveZO, lookAt, vec3
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


from utils.pipelines import PipelineWatch, Pipeline
from utils.reflection import to_dtype, DescriptorSetsReflection
from utils.render import PerFrameResource
from utils.utils import profile
from utils.threadpool import ThreadPool
from dataclasses import dataclass
from threading import Event, Lock
import threading
from enum import Enum, auto
import io
import struct

from functools import lru_cache

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
# [x] Sync issue still? Maybe wrong buffer in use?
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

# CPU buffers:
# - buffers need to be kept in use during frames in flight
# - buffers can be used for pre-fetching if not in use
# - pre-fetching always wants to prefetch the next few images
#   - this is small enough that we can just reiterate that
#   - amount of buffers available for prefetching could be smaller if we have frames in flight or not
#   - maybe these concepts should be separate to avoid different behavior?
#   - e.g. total_buffers = num_frames + prefetch size. 
#   - get frame will return existing buffer, wait for prefetched, or schedule and then wait
#   - at least one buffer will always be available because we release the oldest frame in flight
# - pre-fetching should be cancelable if possible

NUM_WORKERS = 4
pool = ThreadPool(NUM_WORKERS)

class CpuBuffer:
    def __init__(self, size: int):
        self.buf = Buffer(ctx, size, BufferUsageFlags.TRANSFER_SRC, alloc_type=AllocType.HOST)
        self.event = Event()

PREFETCH_SIZE = 2
BUFS = window.num_frames + PREFETCH_SIZE
cpu_bufs_storage = [CpuBuffer(V * 12) for _ in range(BUFS)]
cpu_bufs_lru: OrderedDict[CpuBuffer, Optional[int]] = OrderedDict()
for b in cpu_bufs_storage:
    cpu_bufs_lru[b] = None
cpu_bufs_frame_lookup: OrderedDict[int, CpuBuffer] = OrderedDict()
cpu_bufs_in_flight: List[Optional[Tuple[CpuBuffer, int]]] = [None] * window.num_frames


def load(thread_index: int, buf: CpuBuffer, i: int):
    file = files[thread_index]
    read_exact_at_offset_into(file, 12 + i * V * 12, buf.buf.view.view())
    buf.event.set()
    return buf

# GPU buffers:
# - one per frame + LRU cache        -> works for both sequence and bigimage
# - no pre-fetching at first to GPU  -> not obvious what is the best to way to sync this with main thread
GPU_BUFFERS = window.num_frames
class GpuBuffer:
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, use_transfer_queue: bool):
        # NOTE: we are making the buffer and its upload the unit of
        # synchronization here.  This is ok for now, but maybe we want to make
        # this more generic in the future.
        self.buf = Buffer(ctx, size, usage_flags | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE_MAPPED)

        if use_transfer_queue:
            # Note: bufs that have been used once are different from bufs that were
            # never used, because we don't need to wait for their last user to be
            # done with them.
            #
            # It would be nice to make this difference transparent by initializing
            # them as they would be once a user is done using them, but this
            # is complicated by the fact that you cannot create a GPU semaphore
            # in the signaled state without timeline semaphores...
            #
            # Maybe should rework this by requiring timeline sempahores..
            self.used = False
            self.use_done = Semaphore(ctx)
            self.upload_done = Semaphore(ctx)

class RefCount:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1
    
    def dec(self) -> bool:
        self.count -= 1
        return self.count == 0

USE_TRANSFER_QUEUE = ctx.has_transfer_queue
gpu_bufs_storage = [GpuBuffer(ctx, V * 12, BufferUsageFlags.VERTEX, USE_TRANSFER_QUEUE) for _ in range(GPU_BUFFERS)]
gpu_bufs_lru: OrderedDict[GpuBuffer, Optional[int]] = OrderedDict()
for b in gpu_bufs_storage:
    gpu_bufs_lru[b] = None
gpu_bufs_frame_lookup: Dict[int, Tuple[GpuBuffer, RefCount]] = {}
gpu_bufs_in_flight: List[Optional[Tuple[GpuBuffer, RefCount, int]]] = [None] * window.num_frames

# NOTE:
# I really like this LRU cache buffer allocation mechanism
# - elegantly solves the problem of frames that can be in flight or not
# - allows reuse of GPU and CPU buffers if they are not evicted
# - can be used for both spatial and temporal caches transparently
#
# TODO:
# [x] Implement copy on transfer queue
# [ ] Try to implement prefetching on top of this
#   -> keep in mind RR prefetching as a goal, ideally pre-fetch policy is external / switchable
# [ ] Take out this LRU implementation and make it somehow extendable? Lambdas?
# [ ] Profile

def draw():
    global depth
    global first_frame, frame_index, last_timestamp
    global animation_time, animation_playing

    global cpu_bufs_in_flight, cpu_bufs_lru, cpu_bufs_frame_lookup
    global gpu_bufs_in_flight, gpu_bufs_lru, gpu_bufs_frame_lookup

    timestamp = perf_counter()
    dt = timestamp - last_timestamp
    last_timestamp = timestamp
    print(f"dt: {dt * 1000:.3f}ms")

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
                animation_time = idx / animation_fps
            _, animation_playing = imgui.checkbox("Playing", animation_playing)
            
            start = imgui.get_cursor_screen_pos()
            dl = imgui.get_window_draw_list()
            for i in range(N):
                cursor = imgui.Vec2(start.x + 5 * i, start.y)
                dl.add_rect(cursor, (cursor.x + 6, cursor.y + 20), 0xFFFFFFFF)
                dl.add_rect((cursor.x, cursor.y + 22), (cursor.x + 6, cursor.y + 42), 0xFFFFFFFF)

            for i in cpu_bufs_frame_lookup.keys():
                cursor = imgui.Vec2(start.x + 5 * i, start.y)
                dl.add_rect_filled((cursor.x + 1, cursor.y + 1), (cursor.x + 5, cursor.y + 19), 0xFF00FF00)

            for i in gpu_bufs_frame_lookup.keys():
                cursor = imgui.Vec2(start.x + 5 * i, start.y)
                dl.add_rect_filled((cursor.x + 1, cursor.y +23), (cursor.x + 5, cursor.y + 41), 0xFF00FF00)

        imgui.end()
        if imgui.begin("cache"):
            imgui.text(f"CPU buffers")
            imgui.indent()
            for k, v in cpu_bufs_frame_lookup.items():
                imgui.text(f"{k} {v}")
            imgui.unindent()

            imgui.text(f"CPU LRU")
            imgui.indent()
            i = 0
            for k, v in cpu_bufs_lru.items():
                imgui.text(f"{k} {v}")
                i += 1
            for _ in range(i, BUFS):
                imgui.text(f"<EMPTY>")
            imgui.unindent()

            imgui.text(f"CPU IN FLIGHT")
            imgui.indent()
            for v in cpu_bufs_in_flight:
                imgui.text(f"{v}")
            imgui.unindent()

            imgui.separator()

            imgui.text(f"GPU buffers")
            imgui.indent()
            for k, v in gpu_bufs_frame_lookup.items():
                imgui.text(f"{k} {v[0]} ({v[1].count})")
            imgui.unindent()
            imgui.text(f"GPU LRU")
            imgui.indent()
            i = 0
            for k, v in gpu_bufs_lru.items():
                imgui.text(f"{k} {v}")
                i += 1
            for _ in range(i, GPU_BUFFERS):
                imgui.text(f"<EMPTY>")
            imgui.unindent()

            imgui.text(f"GPU IN FLIGHT")
            imgui.indent()
            for v in gpu_bufs_in_flight:
                imgui.text(f"{v}")
            imgui.unindent()
        imgui.end()
    
    # Upload

    # Render
    frame = window.begin_frame()
    additional_wait_semaphores = []
    additional_signal_semaphores = []
    with frame.command_buffer as cmd:
        ####################################################################

        # Dec refcount on oldest GPU buffer
        if old := gpu_bufs_in_flight[frame_index]:
            buf, rc, key = old
            if rc.dec():
                # Insert in LRU as free buf, already populated with a key
                gpu_bufs_lru[buf] = key

            # Unnecessary because we overwrite this with a new GPU frame every frame.
            # It would be useful if the buffer could potentially get culled and
            # not used this frame.
            gpu_bufs_in_flight[frame_index] = None

        if old := cpu_bufs_in_flight[frame_index]:
            buf, key = old
            
            # Cpu bufs are not refcounted: if a CPU buffer is in use, then a GPU buffer with that
            # same key already exists, therefore we would not be trying to reuse this buffer
            # and the refcount would not grow above 1.
            cpu_bufs_lru[buf] = key

            # Necessary because we have frames that do not use any CPU buffer
            cpu_bufs_in_flight[frame_index] = None

        # Check if already uploaded
        cached = gpu_bufs_frame_lookup.get(animation_frame_index)
        if cached is None:
            cpu_buf = cpu_bufs_frame_lookup.get(animation_frame_index)

            # Check if already loaded
            if cpu_buf is None:
                # Grab a free buffer
                cpu_buf, key = cpu_bufs_lru.popitem(last=False)
                if key is not None:
                    # Remove the bufer from the lookup
                    cpu_bufs_frame_lookup.pop(key)

                # Submit and wait for the load to complete
                with profile("Load"):
                    cpu_buf.event.clear()
                    pool.submit(load, cpu_buf, animation_frame_index)
                    cpu_buf.event.wait()

                # Register buffer as loaded for future use
                cpu_bufs_frame_lookup[animation_frame_index] = cpu_buf
            else:
                # If the buffer is in the LRU remove it to mark it as in use
                try:
                    cpu_bufs_lru.pop(cpu_buf)
                except KeyError:
                    pass


            # Grab a free GPU buffer in FIFO order
            gpu_buf, key = gpu_bufs_lru.popitem(last=False)

            if key is not None:
                # Remove the buffer from the lookup
                gpu_bufs_frame_lookup.pop(key)

            # Upload from CPU
            if False:
                # If using mapped buffer
                gpu_buf.buf.view.data[:] = cpu_buf.buf.view.data
                
                # Buffer is immediately not in use anymore. Add back to the LRU.
                # This moves back the buffer to the front of the LRU queue.
                cpu_bufs_lru[cpu_buf] = animation_frame_index
            else:
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

                cpu_bufs_in_flight[frame_index] = (cpu_buf, animation_frame_index)
            
            # Register buffer as uploaded for future use
            rc = RefCount()
            gpu_bufs_frame_lookup[animation_frame_index] = (gpu_buf, rc)
        else:
            # If the buffer is in the LRU remove it to mark it as in use
            gpu_buf, rc = cached
            try:
                gpu_bufs_lru.pop(gpu_buf)
            except KeyError:
                pass

        # Mark GPU buffer as used by this frame
        rc.inc()
        gpu_bufs_in_flight[frame_index] = (gpu_buf, rc, animation_frame_index)

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


window.set_callbacks(draw)

while True:
    process_events(False)

    if window.should_close():
        break

    draw()

cache.stop()