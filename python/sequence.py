from pyxpg import *

import struct
from pathlib import Path
from typing import Optional, List, Dict
from time import perf_counter
from threading import Event
from enum import Enum, auto
from dataclasses import dataclass

from pyglm.glm import perspectiveZO, lookAt, vec3
import numpy as np

from utils.pipelines import PipelineWatch, Pipeline
from utils.reflection import to_dtype, DescriptorSetsReflection
from utils.render import PerFrameResource
from utils.threadpool import ThreadPool
from utils.profiler import Profiler, ProfilerFrame, gui_profiler_graph, gui_profiler_list
from utils.loaders import LRUPool
from utils.utils import profile, read_exact, read_exact_at_offset, read_exact_at_offset_into

VSYNC = False
WORKERS = 4
GPU_PREFETCH_SIZE = 2
PREFETCH_SIZE = 2

ctx = Context(
    device_features=DeviceFeatures.DYNAMIC_RENDERING | DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.PRESENTATION | DeviceFeatures.HOST_QUERY_RESET | DeviceFeatures.CALIBRATED_TIMESTAMPS, 
    enable_validation_layer=True,
    enable_synchronization_validation=True,
    preferred_frames_in_flight=2,
    vsync=VSYNC,
)

window = Window(ctx, "Sequence", 1600, 900)
gui = Gui(window)

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
total_frame_index = 0
last_timestamp = perf_counter()
animation_fps = 25
animation_time = 1 / animation_fps * 0.5
animation_playing = False

pool = ThreadPool(WORKERS)

USE_TRANSFER_QUEUE = ctx.has_transfer_queue
BUFS = window.num_frames + PREFETCH_SIZE + GPU_PREFETCH_SIZE
GPU_BUFFERS = window.num_frames + GPU_PREFETCH_SIZE

class CpuBuffer:
    def __init__(self, size: int, name: Optional[str] = None):
        self.buf = Buffer(ctx, size, BufferUsageFlags.TRANSFER_SRC, AllocType.HOST, name)
        self.event = Event()
    
    def __repr__(self):
        return self.buf.__repr__()

class GpuBufferState(Enum):
    EMPTY = auto()
    SYNC_LOAD = auto()
    PREFETCH = auto()
    FENCE_SIGNALED = auto()
    RENDERING = auto()

class GpuBuffer:
    def __init__(self, ctx: Context, size: int, usage_flags: BufferUsageFlags, use_transfer_queue: bool, name: Optional[str] = None):
        self.buf = Buffer(ctx, size, usage_flags | BufferUsageFlags.TRANSFER_DST, AllocType.DEVICE_MAPPED, name=name)

        if use_transfer_queue:
            self.state = GpuBufferState.EMPTY
            self.prefetch_done_semaphore = Semaphore(ctx)
            self.render_done = Semaphore(ctx)
            self.load_done = Semaphore(ctx)
        
    def __repr__(self):
        return self.buf.__repr__()

@dataclass
class PrefetchState:
    commands: CommandBuffer
    fence: Fence

path = Path("N:\\scenes\\smpl\\all_frames_20.bin")
class Sequence:
    def __init__(self, ctx: Context, path: Path, num_frames: int):
        self.files = [open(path, "rb", buffering=0) for _ in range(WORKERS)]
        file = self.files[0]

        header = read_exact(file, 12)
        N = struct.unpack("<I", header[0: 4])[0]
        V = struct.unpack("<I", header[4: 8])[0]
        I = struct.unpack("<I", header[8:12])[0]

        indices = np.frombuffer(read_exact_at_offset(file, N * V * 12 + len(header), I * 4), np.uint32)
        self.i_buf = Buffer.from_data(ctx, indices.tobytes(), BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)

        self.cpu = LRUPool([CpuBuffer(V * 12, name=f"cpubuf{i}") for i in range(BUFS)], num_frames, PREFETCH_SIZE)
        self.gpu = LRUPool([GpuBuffer(ctx, V * 12, BufferUsageFlags.VERTEX, USE_TRANSFER_QUEUE, name=f"gpubuf{i}") for i in range(GPU_BUFFERS + 1)], num_frames, GPU_PREFETCH_SIZE)
        self.N = N
        self.V = V
        self.I = I

        # Prefetching stuff
        self.prefetch_states = [
            PrefetchState(
                commands=CommandBuffer(ctx, queue_family_index=ctx.transfer_queue_family_index),
                fence=Fence(ctx, signaled=True),
            ) for _ in range(GPU_PREFETCH_SIZE)
        ]
        self.prefetch_states_lookup: Dict[GpuBuffer, PrefetchState] = {}

    def _load(self, i: int, buf: CpuBuffer, thread_index: int):
        file = self.files[thread_index]
        read_exact_at_offset_into(file, 12 + i * self.V * 12, buf.buf.view.view())
        buf.event.set()
        return buf

seq = Sequence(ctx, path, window.num_frames)

# Async upload queue
transfer_queue = ctx.transfer_queue

# TODO:
# [x] Implement copy on transfer queue
# [x] Try to implement prefetching on top of this
#   -> keep in mind RR prefetching as a goal, ideally pre-fetch policy is external / switchable
# [x] Take out this LRU implementation and make it somehow extendable? Lambdas?
# [x] Add optional names to resources, use in __repr__
# [x] ImGui Profiler?
# [x] Try async GPU upload?
#     [x] Figure out how to avoid incomplete queue ownership transfer
#     [x] Multi-frame GPU prefetch
#     [ ] Generalize this to multiple sequences with different frame rates and counts

profiler = Profiler(ctx, window.num_frames + 1)
profiler_max_frames = 20 if VSYNC else 60
profiler_results: List[ProfilerFrame] = []
    
def draw():
    global depth
    global first_frame, frame_index, last_timestamp, total_frame_index
    global animation_time, animation_playing
    global profiler, profiler_results

    timestamp = perf_counter()
    dt = timestamp - last_timestamp
    last_timestamp = timestamp

    animation_frame_index = int(animation_time * animation_fps) % seq.N

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

    # Render
    frame = window.begin_frame()
    additional_wait_semaphores = []
    additional_signal_semaphores = []
    with frame.command_buffer as cmd:
        prof = profiler.frame(cmd)

        if prof and len(profiler_results) < profiler_max_frames:
            profiler_results.append(prof)
        
        with profiler.zone(f"frame {animation_frame_index}"):
            # GUI
            with profiler.zone("gui"):
                with gui.frame():
                    if imgui.begin("stats")[0]:
                        imgui.text(f"indices: {seq.I} ({seq.I * 4 / 1024 / 1024:.2f}MB)")
                        imgui.text(f"vertices: {seq.V} ({seq.V * 12 / 1024 / 1024:.2f}MB)")
                        imgui.text(f"vertices gpu buffers: {seq.V * 12 / 1024 / 1024 * GPU_BUFFERS:.2f}MB")
                        imgui.text(f"vertices load buffers: {seq.V * 12 / 1024 / 1024 * BUFS:.2f}MB")
                    imgui.end()
                    if imgui.begin("playback")[0]:
                        imgui.text(f"animation time: {animation_time:.1f}s")
                        changed, idx = imgui.slider_int("animation frame", animation_frame_index, 0, seq.N - 1)
                        if changed:
                            animation_frame_index = idx
                            animation_time = idx / animation_fps
                        _, animation_playing = imgui.checkbox("Playing", animation_playing)
                        
                        start = imgui.get_cursor_screen_pos()
                        dl = imgui.get_window_draw_list()
                        for i in range(seq.N):
                            cursor = imgui.Vec2(start.x + 5 * i, start.y)
                            dl.add_rect(cursor, (cursor.x + 6, cursor.y + 20), 0xFFFFFFFF)
                            dl.add_rect((cursor.x, cursor.y + 22), (cursor.x + 6, cursor.y + 42), 0xFFFFFFFF)

                        for k, v in seq.cpu.lookup.items():
                            cursor = imgui.Vec2(start.x + 5 * k, start.y)
                            color = 0xFF00FF00 if not v.prefetching else 0xFF00FFFF
                            dl.add_rect_filled((cursor.x + 1, cursor.y + 1), (cursor.x + 5, cursor.y + 19), color)

                        for i in seq.gpu.lookup.keys():
                            cursor = imgui.Vec2(start.x + 5 * i, start.y)
                            dl.add_rect_filled((cursor.x + 1, cursor.y +23), (cursor.x + 5, cursor.y + 41), 0xFF00FF00)

                    imgui.end()
                    if imgui.begin("cache")[0]:
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

                            imgui.text(f"Prefetching")
                            imgui.indent()
                            i = 0
                            for k, v in pool.prefetch_store.items():
                                imgui.text(f"{k} {v}")
                                i += 1
                            for _ in range(i, pool.max_prefetch):
                                imgui.text(f"<EMPTY>")
                            imgui.unindent()


                        drawpool("CPU", seq.cpu)
                        drawpool("GPU", seq.gpu)
                    imgui.end()

                    gui_profiler_list(prof, dt, ctx.timestamp_period_ns)
                    gui_profiler_graph(profiler_results, ctx.timestamp_period_ns)
                

            with profiler.gpu_zone(f"frame {animation_frame_index}"):
                ####################################################################
                # Init
                seq.cpu.new_frame(frame_index)
                seq.gpu.new_frame(frame_index)

                ####################################################################
                # Get

                copy_wait_semaphores = []
                copy_signal_semaphores = []
                with frame.transfer_command_buffer as copy_cmd:
                    profiler.transfer_frame(copy_cmd)
                    # Check if already uploaded
                    def cpu_ensure_fetched(k: int, buf: CpuBuffer):
                        with profiler.zone("Wait"):
                            buf.event.wait()

                    def cpu_load(k: int, buf: CpuBuffer):
                        with profiler.zone("Load"):
                            pool.submit(seq._load, k, buf)
                            buf.event.wait()
                        buf.event.clear()

                    def gpu_load(k: int, gpu_buf: GpuBuffer):
                        cpu_buf = seq.cpu.get(k, cpu_load, cpu_ensure_fetched)

                        if False:
                            # Upload on CPU through PCIe BAR
                            with profiler.zone("upload"):
                                # If using mapped buffer
                                gpu_buf.buf.view.data[:] = cpu_buf.buf.view.data
                                
                                # Buffer is immediately not in use anymore. Add back to the LRU.
                                # This moves back the buffer to the front of the LRU queue.
                                cpu.give_back(k, cpu_buf)
                        else:
                            seq.cpu.use(frame_index, k)
                            if not USE_TRANSFER_QUEUE:
                                # Upload on gfx queue
                                with profiler.gpu_zone("copy"):
                                    cmd.copy_buffer(cpu_buf.buf, gpu_buf.buf)
                                    cmd.memory_barrier(MemoryUsage.TRANSFER_WRITE, MemoryUsage.VERTEX_INPUT)
                            else:
                                # Upload on copy queue
                                if gpu_buf.state == GpuBufferState.EMPTY:
                                    pass
                                elif gpu_buf.state == GpuBufferState.RENDERING:
                                    copy_wait_semaphores.append((gpu_buf.render_done, PipelineStageFlags.TRANSFER))
                                elif gpu_buf.state == GpuBufferState.FENCE_SIGNALED:
                                    copy_wait_semaphores.append((gpu_buf.prefetch_done_semaphore, PipelineStageFlags.TRANSFER))
                                else:
                                    assert False, gpu_buf.state
                                copy_signal_semaphores.append(gpu_buf.load_done)

                                with profiler.gpu_transfer_zone("copy"):
                                    copy_cmd.copy_buffer(cpu_buf.buf, gpu_buf.buf)
                                    copy_cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.TRANSFER_WRITE, MemoryUsage.NONE, ctx.transfer_queue_family_index, ctx.graphics_queue_family_index)

                                gpu_buf.state = GpuBufferState.SYNC_LOAD

                    def gpu_ensure_loaded(k: int, gpu_buf: GpuBuffer):
                        assert gpu_buf.state == GpuBufferState.PREFETCH, gpu_buf.state
                        # Promote CPU buffer from manually-managed to frame-managed
                        seq.cpu.use(frame_index, k)
                        seq.cpu.use_done_manual(k)

                    gpu_buf = seq.gpu.get(animation_frame_index, gpu_load, gpu_ensure_loaded)
                    seq.gpu.use(frame_index, animation_frame_index)

                    if gpu_buf.state == GpuBufferState.SYNC_LOAD or gpu_buf.state == GpuBufferState.PREFETCH or gpu_buf.state == GpuBufferState.FENCE_SIGNALED:
                        if gpu_buf.state == GpuBufferState.PREFETCH or gpu_buf.state == GpuBufferState.FENCE_SIGNALED:
                            with profiler.gpu_transfer_zone("prefetch barrier"):
                                copy_wait_semaphores.append((gpu_buf.prefetch_done_semaphore, PipelineStageFlags.TOP_OF_PIPE))
                                copy_signal_semaphores.append(gpu_buf.load_done)
                            copy_cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.TRANSFER_WRITE, MemoryUsage.NONE, ctx.transfer_queue_family_index, ctx.graphics_queue_family_index)
                        
                        cmd.buffer_barrier(gpu_buf.buf, MemoryUsage.NONE, MemoryUsage.VERTEX_INPUT, ctx.transfer_queue_family_index, ctx.graphics_queue_family_index)
                        additional_wait_semaphores.append((gpu_buf.load_done, PipelineStageFlags.VERTEX_INPUT))
                        additional_signal_semaphores.append(gpu_buf.render_done)
                        gpu_buf.state = GpuBufferState.RENDERING
                    
                    assert gpu_buf.state == GpuBufferState.RENDERING, gpu_buf.state
                
                if copy_signal_semaphores:
                    ctx.transfer_queue.submit(copy_cmd, copy_wait_semaphores, copy_signal_semaphores)

                ####################################################################
                # Prefetch
                
                def prefetch_cleanup(k: int, buf: CpuBuffer) -> bool:
                    if buf.event.is_set():
                        buf.event.clear()
                        return True
                    return False

                def prefetch(k: int, buf: CpuBuffer):
                    pool.submit(seq._load, k, buf)
                prefetch_start = animation_frame_index + 1
                prefetch_end = prefetch_start + PREFETCH_SIZE
                seq.cpu.prefetch([i % seq.N for i in range(prefetch_start, prefetch_end)], prefetch_cleanup, prefetch)

                if USE_TRANSFER_QUEUE:
                    def gpu_prefetch_cleanup(k: int, gpu_buf: GpuBuffer):
                        state = seq.prefetch_states_lookup[gpu_buf]
                        if state.fence.is_signaled():
                            # Prefetch state
                            seq.prefetch_states.append(state)
                            del seq.prefetch_states_lookup[gpu_buf]

                            if gpu_buf.state == GpuBufferState.RENDERING:
                                pass
                            else:
                                assert gpu_buf.state == GpuBufferState.PREFETCH, gpu_buf.state
                                gpu_buf.state = GpuBufferState.FENCE_SIGNALED
                                seq.cpu.use_done_manual(k)
                            return True
                        return False

                    def gpu_prefetch(k: int, gpu_buf: GpuBuffer):
                        cpu_next: CpuBuffer = seq.cpu.get(k, lambda x, y: None)
                        seq.cpu.use_manual(k)

                        # Prefetch state
                        state = seq.prefetch_states.pop()
                        seq.prefetch_states_lookup[gpu_buf] = state

                        with state.commands:
                            state.commands.copy_buffer(cpu_next.buf, gpu_buf.buf)

                        if gpu_buf.state == GpuBufferState.EMPTY:
                            wait_semaphores = []
                        elif gpu_buf.state == GpuBufferState.FENCE_SIGNALED:
                            wait_semaphores = [(gpu_buf.prefetch_done_semaphore, PipelineStageFlags.TRANSFER)]
                        elif gpu_buf.state == GpuBufferState.RENDERING:
                            wait_semaphores = [(gpu_buf.render_done, PipelineStageFlags.TRANSFER)]
                        else:
                            assert False, gpu_buf.state
                        
                        transfer_queue.submit(
                            state.commands,
                            wait_semaphores = wait_semaphores,
                            signal_semaphores = [gpu_buf.prefetch_done_semaphore],
                            fence=state.fence,
                        )
                        gpu_buf.state = GpuBufferState.PREFETCH

                    prefetch_start = animation_frame_index + 1
                    prefetch_end = prefetch_start + GPU_PREFETCH_SIZE
                    seq.gpu.prefetch([i % seq.N for i in range(prefetch_start, prefetch_end) if seq.cpu.is_available(i % seq.N)], gpu_prefetch_cleanup, gpu_prefetch)

                ####################################################################

                
                cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)
                if images_just_created:
                    cmd.use_image(depth, ImageUsage.DEPTH_STENCIL_ATTACHMENT)
                
                viewport = [0, 0, window.fb_width, window.fb_height]
                with profiler.gpu_zone("draw"):
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
                            index_buffer=seq.i_buf,
                            viewport=viewport,
                            scissors=viewport,
                        )

                        # Issue a draw
                        cmd.draw_indexed(seq.I)

                # Render gui
                with profiler.gpu_zone("gui"):
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
    total_frame_index += 1
    if animation_playing:
        animation_time += dt

def on_key(k: Key, a: Action, m: Modifiers):
    global animation_time, animation_playing, profiler_results
    if a == Action.PRESS:
        if k == Key.SPACE:
            animation_playing = not animation_playing
    if a == Action.PRESS or a == Action.REPEAT:
        if k == Key.PERIOD:
            animation_time += 1 / animation_fps
        if k == Key.COMMA:
            animation_time -= 1 / animation_fps
        if k == Key.P:
            profiler_results = []

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