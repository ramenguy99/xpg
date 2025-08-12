from pathlib import Path
from utils.scene import parse_scene, ImageFormat
from typing import List
from time import perf_counter

from pyxpg import *

scene = parse_scene(Path("res", "bistro.bin"))

ctx = Context(
    required_features=DeviceFeatures.SYNCHRONIZATION_2,
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

begin = perf_counter()
if True:
    largest = 0
    total_size = 0
    for image in scene.images:
        largest = max(image.data.size, largest)
        total_size += largest
    print(f"Total size: {total_size / 1024 / 1024}MB")

    def align_up(v, a):
        return (v + a - 1) & ~(a - 1)


    STAGING_SIZE = 32 * 1024 * 1024
    alignment = max(64, ctx.device_properties.limits.optimal_buffer_copy_offset_alignment)
    upload_size = align_up(max(largest, STAGING_SIZE), alignment)

    staging = Buffer(ctx, upload_size, BufferUsageFlags.TRANSFER_SRC, alloc_type=AllocType.HOST_WRITE_COMBINING)

    images: List[Image] = []
    i = 0
    while i < len(scene.images):
        # Batched upload
        with ctx.sync_commands() as cmd:
            offset = 0
            while i < len(scene.images) and offset + scene.images[i].data.size <= staging.size:
                image = scene.images[i]
                format = 0
                if image.format == ImageFormat.RGBA8: format = Format.R8G8B8A8_UNORM
                elif image.format == ImageFormat.SRGBA8: format = Format.R8G8B8A8_SRGB
                elif image.format == ImageFormat.RGBA8_BC7: format = Format.BC7_UNORM_BLOCK
                elif image.format == ImageFormat.SRGBA8_BC7: format = Format.BC7_SRGB_BLOCK
                else:
                    raise ValueError(f"Unhandled image format: {image.format}")

                # Create target image
                gpu_img = Image(ctx, image.width, image.height, format, ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST, AllocType.DEVICE)
                images.append(gpu_img)

                # Copy image data to staging buffer
                staging.data[offset:offset + len(image.data)] = image.data.data[:]

                # Upload
                cmd.image_barrier(gpu_img, ImageLayout.TRANSFER_DST_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_DST)
                cmd.copy_buffer_to_image(staging, gpu_img, buffer_offset=offset)
                cmd.image_barrier(gpu_img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.TRANSFER_DST, MemoryUsage.SHADER_READ_ONLY)

                # Advance image and buffer offset
                offset += align_up(image.data.size, alignment)
                i += 1
else:
    # Images
    images: List[Image] = []
    for image in scene.images:
        format = 0
        if image.format == ImageFormat.RGBA8: format = Format.R8G8B8A8_UNORM
        elif image.format == ImageFormat.SRGBA8: format = Format.R8G8B8A8_SRGB
        elif image.format == ImageFormat.RGBA8_BC7: format = Format.BC7_UNORM_BLOCK
        elif image.format == ImageFormat.SRGBA8_BC7: format = Format.BC7_SRGB_BLOCK
        else:
            raise ValueError(f"Unhandled image format: {image.format}")

        images.append(
            Image.from_data(
                ctx, image.data, ImageUsage.SHADER_READ_ONLY,
                image.width, image.height, format,
                ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST, AllocType.DEVICE
            )
        )
total_time = perf_counter() - begin
print(f"Total time: {total_time:.3f}s ({total_size / total_time / 1024 / 1024 / 1024:.3f}GB/s)")