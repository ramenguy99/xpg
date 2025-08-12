from pyxpg import *
from pyxpg import imgui
from pathlib import Path
import numpy as np
from utils.pipelines import PipelineWatch, Pipeline
from utils.reflection import to_dtype
from time import perf_counter
import os

import warp as wp

ctx = Context(
    version=(1, 1),
    required_features=DeviceFeatures.DYNAMIC_RENDERING | DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.EXTERNAL_RESOURCES,
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)
window = Window(ctx, "Warp Interop", 1280, 720)
gui = Gui(window)

# gfx
I = np.array([
    [0, 1, 2],
    [1, 3, 2],
], np.uint32)
rot = np.eye(4, dtype=np.float32)
push_constants = np.array([ 1.0, 0.0, 0.0], np.float32)
i_buf = Buffer.from_data(ctx, I, BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)
set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)


if os.name == 'nt':
    memory_handle_type = wp.ExternalMemoryBuffer.HANDLE_TYPE_OPAQUEWIN32
    semaphore_handle_type = wp.ExternalMemoryBuffer.HANDLE_TYPE_OPAQUEWIN32
else:
    memory_handle_type = wp.ExternalMemoryBuffer.HANDLE_TYPE_OPAQUEFD
    semaphore_handle_type = wp.ExternalMemoryBuffer.HANDLE_TYPE_OPAQUEFD

# Warp
wp.init()
size = 4 * 3 * 4
v_buf = ExternalBuffer(ctx, size, BufferUsageFlags.VERTEX, AllocType.DEVICE)
ext_buf = wp.ExternalMemoryBuffer(v_buf.handle, memory_handle_type, size)
a = ext_buf.map(wp.vec3, shape=4)

cuda_done = ExternalSemaphore(ctx)
vulkan_done = ExternalSemaphore(ctx)
warp_cuda_done = wp.ExternalSemaphore(cuda_done.handle, semaphore_handle_type)
warp_vulkan_done = wp.ExternalSemaphore(vulkan_done.handle, semaphore_handle_type)

@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3), t: wp.float32):
    s = wp.sin(t) * 0.5 + 1.0
    a[0] = wp.vec3(-0.5, -0.5, 0.0) * s
    a[1] = wp.vec3( 0.5, -0.5, 0.0) * s
    a[2] = wp.vec3(-0.5,  0.5, 0.0) * s
    a[3] = wp.vec3( 0.5,  0.5, 0.0) * s


# Pipeline
pipeline: Pipeline = None
class ColorPipeline(Pipeline):
    vert_prog = "shaders/color.vert.slang"
    frag_prog = "shaders/color.frag.slang"

    def create(self, vert_prog: slang.Shader, frag_prog: slang.Shader):
        global buf

        refl = vert_prog.reflection
        dt = to_dtype(refl.resources[0].type)

        u_buf = Buffer(ctx, dt.itemsize, BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
        set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        buf = u_buf.view.view(dt)
        buf["transform"] = rot
        buf["nest1"]["val2"] = push_constants

        vert = Shader(ctx, vert_prog.code)
        frag = Shader(ctx, frag_prog.code)

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
            push_constants_ranges = [
                PushConstantsRange(12),
            ],
            descriptor_sets = [ set ],
            attachments = [
                Attachment(format=window.swapchain_format)
            ]
        )

color = ColorPipeline()
cache = PipelineWatch([
    color,
])

# Main loop
first_frame = True
def draw():
    global push_constants, first_frame

    cache.refresh(lambda: ctx.wait_idle())

    # swapchain update
    swapchain_status = window.update_swapchain()

    if swapchain_status == SwapchainStatus.MINIMIZED:
        return

    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    # Launch warp
    t = perf_counter() * 2
    if first_frame:
        first_frame = False
    else:
        wp.wait_external_semaphore(warp_vulkan_done)
    wp.launch(kernel=simple_kernel, # kernel to launch
              dim=1,                # number of threads
              inputs=[a, t],        # parameters
              device="cuda")        # execution device
    wp.signal_external_semaphore(warp_cuda_done)

    # GUI
    with gui.frame():
        if imgui.begin("Window")[0]:
            imgui.text("Hello")
            try:
                updated, v = imgui.color_edit3("Value", tuple(push_constants))
                if updated:
                    push_constants = np.array(v, np.float32)
                    buf["nest1"]["val2"] = v
            except Exception as e:
                print(e)
        imgui.end()

    # Render
    with window.frame(
        additional_wait_semaphores=[(cuda_done, PipelineStageFlags.VERTEX_INPUT)],
        additional_signal_semaphores=[vulkan_done],
    ) as frame:
        with frame.command_buffer as cmd:
            cmd.image_barrier(frame.image, ImageLayout.COLOR_ATTACHMENT_OPTIMAL, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.COLOR_ATTACHMENT)

            # TODO: we technically need a buffer barrier here from external queue family type.

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
                cmd.set_viewport(viewport)
                cmd.set_scissors(viewport)
                cmd.bind_graphics_pipeline(
                    pipeline=color.pipeline,
                    descriptor_sets=[ set ],
                    push_constants=push_constants.tobytes(),
                    vertex_buffers=[ v_buf ],
                    index_buffer=i_buf,
                )
                cmd.draw_indexed(I.size)
                gui.render(cmd)

            cmd.image_barrier(frame.image, ImageLayout.PRESENT_SRC, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.PRESENT)

window.set_callbacks(draw)

while True:
    process_events(False)

    if window.should_close():
        break

    draw()

cache.stop()