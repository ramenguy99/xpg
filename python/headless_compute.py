from pyxpg import *
import PIL.Image
import numpy as np

# Initialize without DeviceFeatures.PRESENTATION for headless
print("Initializing context...")
ctx = Context(
    version=(1, 1),
    device_features=DeviceFeatures.SYNCHRONIZATION_2,
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

# Create an image for drawing on the GPU and a buffer for readback on the CPU
print("Creating resources...")
W = 512
H = 512
format = Format.R8G8B8A8_UNORM
img = Image(ctx, W, H, format,
            ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_SRC,
            AllocType.DEVICE)
buf = Buffer(ctx, W * H * 4, BufferUsageFlags.TRANSFER_DST, AllocType.HOST)

# Descriptors
set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.STORAGE_IMAGE),
    ],
)
set.write_image(img, ImageUsage.IMAGE_WRITE_ONLY, DescriptorType.STORAGE_IMAGE, 0, 0)

# Shaders
comp_source = """
[vk::binding(0, 0)]
[vk::image_format("rgba8i")]
RWTexture2D<float3> r_output;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint width, height;
    r_output.GetDimensions(width, height);

    if(all(threadId.xy < uint2(width, height))) {
        float2 uv = float2(threadId.x + 0.5, threadId.y + 0.5) / float2(width, height);
        float2 r = length(uv - 0.5);
        if(dot(r, r) < 0.5 * 0.5) {
            r_output[threadId.xy] = float3(0.8, 0.1, 0.1);
        } else {
            if (bool(((threadId.x ^ threadId.y) >> 4) & 1)) {
                r_output[threadId.xy] = float3(0.1, 0.1, 0.1);
            } else {
                r_output[threadId.xy] = float3(0.3, 0.3, 0.3);
            }
        }
    }
}
"""

print("Compiling shaders...")
comp = Shader(ctx, slang.compile_str(comp_source, "main", "comp.slang").code)

pipeline = ComputePipeline(ctx, comp, "main", descriptor_sets=[ set ])

# Record commands
print("Dispatching...")
with ctx.get_sync_command_buffer() as cmd:
    cmd.use_image(img, ImageUsage.IMAGE_WRITE_ONLY)
    cmd.bind_compute_pipeline(pipeline, descriptor_sets=[ set ])
    cmd.dispatch((W + 7) // 8, (H + 7) // 8)
    cmd.use_image(img, ImageUsage.TRANSFER_SRC)
    cmd.copy_image_to_buffer(img, buf)

# Submit commands and wait for them to complete
ctx.submit_sync()

# Interpret buffer as image and save it to a file
print("Reading back...")
img = PIL.Image.frombuffer("RGBA", [W, H], buf.view, "raw", "RGBA", 0, -1).convert("RGB")

print("Saving to disk...")
img.save("headless_compute.png")

print("Done.")
