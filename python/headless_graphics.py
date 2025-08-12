from pyxpg import *
import PIL.Image
import numpy as np

# Initialize without presentation for headless mode
print("Initializing context...")
ctx = Context(
    version=(1, 1),
    required_features=DeviceFeatures.SYNCHRONIZATION_2 | DeviceFeatures.DYNAMIC_RENDERING,
    presentation=False,
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

# Create an image for drawing on the GPU and a buffer for readback on the CPU
print("Creating resources...")
W = 256
H = 256
format = Format.R8G8B8A8_UNORM
img = Image(ctx, W, H, format,
            ImageUsageFlags.COLOR_ATTACHMENT | ImageUsageFlags.TRANSFER_SRC,
            AllocType.DEVICE)
buf = Buffer(ctx, W * H * 4, BufferUsageFlags.TRANSFER_DST, AllocType.HOST)

# Create vertices to draw a triangle
V = np.array([
    [-0.5, -0.5, 0], [ 1.0,  0.0, 0.0],
    [ 0.0,  0.5, 0], [ 0.0,  1.0, 0.0],
    [ 0.5, -0.5, 0], [ 0.0,  0.0, 1.0],
], np.float32)
v_buf = Buffer.from_data(ctx, V, BufferUsageFlags.VERTEX, AllocType.DEVICE_MAPPED_WITH_FALLBACK)

# Shaders
source = """
struct VSOutput
{
    float4 position: SV_Position;
    float3 color: COLOR;
};
struct VSInput
{
    [[vk::location(0)]]
    float3 position;

    [[vk::location(1)]]
    float3 color;
};

[shader("vertex")]
VSOutput vert_main(VSInput in)
{
    VSOutput out;
    out.position = float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

[shader("pixel")]
float4 frag_main(VSOutput in) : SV_Target0
{
    return float4(in.color, 1.0);
}
"""

print("Compiling shaders...")
vert = Shader(ctx, slang.compile_str(source, entry="vert_main", filename="vert.slang").code)
frag = Shader(ctx, slang.compile_str(source, entry="frag_main", filename="frag.slang").code)

pipeline = GraphicsPipeline(
    ctx,
    stages = [
        PipelineStage(vert, Stage.VERTEX),
        PipelineStage(frag, Stage.FRAGMENT),
    ],
    vertex_bindings = [
        VertexBinding(0, 24, VertexInputRate.VERTEX),
    ],
    vertex_attributes = [
        VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
        VertexAttribute(1, 0, Format.R32G32B32_SFLOAT, 12),
    ],
    input_assembly = InputAssembly(PrimitiveTopology.TRIANGLE_LIST),
    attachments = [
        Attachment(format)
    ]
)

# Record commands
print("Rendering...")
with ctx.sync_commands() as cmd:
    cmd.image_barrier(img, ImageLayout.COLOR_ATTACHMENT_OPTIMAL, MemoryUsage.NONE, MemoryUsage.COLOR_ATTACHMENT)
    viewport = [0, 0, W, H]
    with cmd.rendering(viewport,
        color_attachments=[
            RenderingAttachment(
                img,
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.STORE,
                clear=[0.1, 0.2, 0.4, 1],
            ),
        ]):
            cmd.set_viewport(viewport)
            cmd.set_scissors(viewport)
            cmd.bind_graphics_pipeline(
                pipeline,
                vertex_buffers=[ v_buf ],
            )
            cmd.draw(3)
    cmd.image_barrier(img, ImageLayout.TRANSFER_SRC_OPTIMAL, MemoryUsage.COLOR_ATTACHMENT, MemoryUsage.TRANSFER_SRC)
    cmd.copy_image_to_buffer(img, buf)

# Interpret buffer as image and save it to a file
print("Reading back...")
img = PIL.Image.frombuffer("RGBA", [W, H], buf.data, "raw", "RGBA", 0, -1).convert("RGB")

print("Saving to disk...")
img.save("headless_graphics.png")

print("Done.")
