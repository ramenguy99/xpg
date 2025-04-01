from pyxpg import *
from pyxpg import imgui
from pyxpg import slang
import numpy as np
from pipelines import PipelineWatch, Pipeline
from reflection import to_dtype

ctx = Context(
    device_features=DeviceFeatures.DYNAMIC_RENDERING | DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.PRESENTATION, 
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

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
push_constants = np.array([ 1.0, 0.0, 0.0], np.float32)
v_buf = Buffer.from_data(ctx, V.tobytes(), BufferUsageFlags.VERTEX, AllocType.DEVICE_MAPPED)
i_buf = Buffer.from_data(ctx, I.tobytes(), BufferUsageFlags.INDEX, AllocType.DEVICE_MAPPED)

set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)

class ColorPipeline(Pipeline):
    vert_prog = "shaders/color.vert.slang"
    frag_prog = "shaders/color.frag.slang"

    def create(self, vert_prog: slang.Shader, frag_prog: slang.Shader):
        refl = vert_prog.reflection
        dt = to_dtype(refl.resources[0].type)

        u_buf = Buffer(ctx, dt.itemsize, BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
        set.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        global buf
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


def draw():
    global push_constants

    cache.refresh(lambda: wait_idle(ctx))

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
                updated, v = imgui.color_edit3("Value", tuple(push_constants))
                if updated:
                    push_constants = np.array(v, np.float32)
                    buf["nest1"]["val2"] = v
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
                    pipeline=color.pipeline,
                    descriptor_sets=[ set ],
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