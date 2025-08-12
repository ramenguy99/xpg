from pyxpg import *

import numpy as np
from pathlib import Path

from utils.buffers import UploadableBuffer
from utils.pipelines import PipelineWatch, Pipeline
from utils.reflection import to_dtype, DescriptorSetsReflection

ctx = Context(
    required_features=
        DeviceFeatures.DYNAMIC_RENDERING |
        DeviceFeatures.SYNCHRONIZATION_2,
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

window = Window(ctx, "App", 1280, 720)
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
color_value = np.array([ 1.0, 0.0, 0.0], np.float32)
v_buf = Buffer.from_data(ctx, V, BufferUsageFlags.VERTEX, AllocType.DEVICE_MAPPED_WITH_FALLBACK)
i_buf = Buffer.from_data(ctx, I, BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED_WITH_FALLBACK)

set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)

SHADERS = Path(__file__).parent.joinpath("shaders")

# Define a pipeline with a basic color only vertex and fragment shader.
#
# Any number of shaders can be defined as class fields. They will be compiled
# with slang  and passed automatically to the "create" method with matching
# parameter names. The slang.Shader object contains SPIR-V bytecode, reflection
# information and a list of file dependencies used to compile this shader.
class ColorPipeline(Pipeline):
    vert_prog = Path(SHADERS, "color.vert.slang")
    frag_prog = Path(SHADERS, "color.frag.slang")

    def create(self, vert_prog: slang.Shader, frag_prog: slang.Shader):
        global buf, u_buf

        # Get type reflection from vertex shader. We use this to create a
        # numpy dtype that represents the layout of the constant buffer.
        refl = vert_prog.reflection
        desc_refl = DescriptorSetsReflection(refl)
        dt = to_dtype(desc_refl.descriptors["u"].resource.type)

        # Create a buffer to hold the constants with the required size.
        u_buf = UploadableBuffer(ctx, dt.itemsize, BufferUsageFlags.UNIFORM)
        set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        # Write initial values into the buffer by creating a view over it
        # with the dtype generated from the reflection.
        buf = np.zeros(1, dt)
        buf["transform"] = rot
        buf["nest1"]["val2"] = color_value
        u_buf.upload_sync(buf.view(np.uint8).data)

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
            ]
        )

# Instantiate the color pipeline
color = ColorPipeline()

# Register the color pipeline for hot reloading. We pass the window
# so that event loop can be unblocked if a hot reloading event happens.
cache = PipelineWatch([
    color,
], window=window)

def draw():
    global color_value

    # Refresh shaders. If a shader needs to be recompiled we first wait
    # for the device to become idle. This is needed because as part of the
    # refresh the old pipeline is destroyed, and we need to ensure that no
    # previous frame is still using it.
    cache.refresh(lambda: ctx.wait_idle())

    # Update swapchain
    swapchain_status = window.update_swapchain()
    if swapchain_status == SwapchainStatus.MINIMIZED:
        return
    if swapchain_status == SwapchainStatus.RESIZED:
        pass

    # GUI
    updated = False
    with gui.frame():
        if imgui.begin("Window")[0]:
            imgui.text("Hello")
            try:
                updated, c = imgui.color_edit3("Color", tuple(color_value))
                if updated:
                    color_value = np.array(c, np.float32)
                    buf["nest1"]["val2"] = c
            except Exception as e:
                print(e)
        imgui.end()

    # Render
    with window.frame() as frame:
        with frame.command_buffer as cmd:
            if updated:
                u_buf.upload(cmd, MemoryUsage.VERTEX_SHADER_UNIFORM, buf.view(np.uint8).data)

            cmd.image_barrier(frame.image, ImageLayout.COLOR_ATTACHMENT_OPTIMAL, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.COLOR_ATTACHMENT)

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

                # Bind the pipeline
                cmd.bind_graphics_pipeline(
                    pipeline=color.pipeline,
                    descriptor_sets=[ set ],
                    vertex_buffers=[ v_buf ],
                    index_buffer=i_buf,
                )
                # Issue a draw
                cmd.draw_indexed(I.size)

                # Render gui
                gui.render(cmd)

            cmd.image_barrier(frame.image, ImageLayout.PRESENT_SRC, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.PRESENT)

window.set_callbacks(draw)

while True:
    process_events(True)

    if window.should_close():
        break

    draw()

cache.stop()