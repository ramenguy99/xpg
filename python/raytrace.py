import numpy as np
from pathlib import Path
from typing import List

from pyxpg import *
from pyglm.glm import vec3, vec4, mat4, mat4x3, rotate, normalize
import tqdm
from time import perf_counter

from utils.pipelines import PipelineWatch, Pipeline
from utils.reflection import to_dtype
from utils.render import PerFrameResource
from utils.scene import parse_scene, MaterialParameter, MaterialParameterKind, ImageFormat

scene = parse_scene(Path("res", "bistro.bin"))

ctx = Context(
    device_features=
        DeviceFeatures.DYNAMIC_RENDERING |
        DeviceFeatures.SYNCHRONIZATION_2 |
        DeviceFeatures.PRESENTATION |
        DeviceFeatures.DESCRIPTOR_INDEXING |
        DeviceFeatures.SCALAR_BLOCK_LAYOUT |
        DeviceFeatures.RAY_QUERY, 
    enable_validation_layer=True,
    enable_synchronization_validation=True,
)

window = Window(ctx, "Raytrace", 1280, 720)
gui = Gui(window)

scene_descriptor_set = DescriptorSet(
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
        DescriptorSetEntry(1, DescriptorType.STORAGE_BUFFER),
        DescriptorSetEntry(1, DescriptorType.ACCELERATION_STRUCTURE),
        DescriptorSetEntry(1, DescriptorType.SAMPLER),
        DescriptorSetEntry(1, DescriptorType.STORAGE_IMAGE),
        DescriptorSetEntry(len(scene.images), DescriptorType.SAMPLED_IMAGE),
    ],
)

descriptor_sets = PerFrameResource(DescriptorSet, window.num_frames,
    ctx,
    [
        DescriptorSetEntry(1, DescriptorType.UNIFORM_BUFFER),
    ],
)

# Create resources
positions = np.vstack([m.positions for m in scene.meshes])
normals = np.vstack([m.normals for m in scene.meshes])
tangents = np.vstack([m.tangents for m in scene.meshes])
uvs = np.vstack([m.uvs for m in scene.meshes])
indices = np.vstack([m.indices for m in scene.meshes])

positions_buf = Buffer.from_data(ctx, positions.tobytes(), BufferUsageFlags.ACCELERATION_STRUCTURE_INPUT | BufferUsageFlags.SHADER_DEVICE_ADDRESS, AllocType.DEVICE_MAPPED)
normals_buf = Buffer.from_data(ctx, normals.tobytes(), BufferUsageFlags.ACCELERATION_STRUCTURE_INPUT | BufferUsageFlags.SHADER_DEVICE_ADDRESS | BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)
tangents_buf = Buffer.from_data(ctx, tangents.tobytes(), BufferUsageFlags.ACCELERATION_STRUCTURE_INPUT | BufferUsageFlags.SHADER_DEVICE_ADDRESS | BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)
uvs_buf = Buffer.from_data(ctx, uvs.tobytes(), BufferUsageFlags.ACCELERATION_STRUCTURE_INPUT | BufferUsageFlags.SHADER_DEVICE_ADDRESS | BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)
indices_buf = Buffer.from_data(ctx, indices.tobytes(), BufferUsageFlags.ACCELERATION_STRUCTURE_INPUT | BufferUsageFlags.SHADER_DEVICE_ADDRESS | BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)

SHADERS = Path(__file__).parent.parent.joinpath("shaders")

# Pipeline
class RaytracePipeline(Pipeline):
    rt_prog = Path(SHADERS, "raytrace.comp.slang")

    def create(self, rt_prog: slang.Shader):
        global u_bufs, constants_dt

        refl = rt_prog.reflection
        constants_dt = to_dtype(refl.resources[8].type)

        # Create a buffer to hold the constants with the required size.
        u_bufs = PerFrameResource(Buffer, window.num_frames, ctx, constants_dt.itemsize, BufferUsageFlags.UNIFORM, AllocType.DEVICE_MAPPED)
        for set_1, u_buf in zip(descriptor_sets.resources, u_bufs.resources):
            set_1: DescriptorSet
            set_1.write_buffer(u_buf, DescriptorType.UNIFORM_BUFFER, 0, 0)

        # Turn SPIR-V code into vulkan shader modules
        rt = Shader(ctx, rt_prog.code)

        # Instantiate the pipeline using the compiled shaders
        self.pipeline = ComputePipeline(
            ctx,
            shader=rt,
            name="main",
            descriptor_sets = [ scene_descriptor_set, descriptor_sets.get_current() ],
        )
        self.reflection = refl

# Instantiate the color pipeline
rt = RaytracePipeline()

# Get mesh instance type from reflection
as_meshes: List[AccelerationStructureMesh] = []
rot = rotate(0.75, vec3(0, 0, 1))
to_z_up = mat4(
    vec4(1., 0., 0., 0.),
    vec4(0., 0., 1., 0.),
    vec4(0., 1., 0., 0.),
    vec4(0., 0., 0., 1.),
)
vertices_address = positions_buf.address
indices_address = indices_buf.address

mesh_instances = np.zeros(len(scene.meshes), to_dtype(rt.reflection.resources[3].type))
vertex_offset = 0
index_offset = 0
for i, m in enumerate(scene.meshes):
    # Fill acceleration structure info
    as_meshes.append(AccelerationStructureMesh(
        vertices_address = vertices_address + vertex_offset * 12,
        vertices_stride = 12,
        vertices_count = m.positions.shape[0],
        vertices_format = Format.R32G32B32_SFLOAT,
        indices_address = indices_address + index_offset * 4,
        indices_type = IndexType.UINT32,
        primitive_count = m.indices.shape[0] // 3,
        transform = tuple(sum(mat4x3(rot @ to_z_up @ m.transform).to_tuple(), ())),
    )) 

    # Fill mesh instances
    mesh_instances[i]["transform"] = m.transform
    mesh_instances[i]["vertex_offset"] = vertex_offset
    mesh_instances[i]["index_offset"] = index_offset

    def mat_index(p: MaterialParameter):
        return p.value if p.kind == MaterialParameterKind.TEXTURE else 0xFFFFFFFF

    def mat_value(p: MaterialParameter):
        return vec4(p.value) if p.kind != MaterialParameterKind.TEXTURE and p.kind != MaterialParameterKind.NONE else vec4()

    mesh_instances[i]["albedo_index"] = mat_index(m.material.base_color)
    mesh_instances[i]["normal_index"] = mat_index(m.material.normal)
    mesh_instances[i]["specular_index"] = mat_index(m.material.specular)
    mesh_instances[i]["emissive_index"] = mat_index(m.material.emissive)

    mesh_instances[i]["albedo_value"] = mat_value(m.material.base_color)
    mesh_instances[i]["specular_value"] = mat_value(m.material.specular)
    mesh_instances[i]["emissive_value"] = mat_value(m.material.emissive)

    vertex_offset += m.positions.shape[0]
    index_offset += m.indices.shape[0]
assert vertex_offset == positions.shape[0]
assert index_offset == indices.shape[0]

images: List[Image] = []
for image in tqdm.tqdm(scene.images, "Uploading images"):
    format = 0
    if image.format == ImageFormat.RGBA8: format = Format.R8G8B8A8_UNORM
    elif image.format == ImageFormat.SRGBA8: format = Format.R8G8B8A8_SRGB
    elif image.format == ImageFormat.RGBA8_BC7: format = Format.BC7_UNORM_BLOCK
    elif image.format == ImageFormat.SRGBA8_BC7: format = Format.BC7_SRGB_BLOCK
    else:
        raise ValueError(f"Unhandled image format: {image.format}")

    images.append(
        Image.from_data(
            ctx, image.data.tobytes(), ImageUsage.SHADER_READ_ONLY, 
            image.width, image.height, format,
            ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST, AllocType.DEVICE
        )
    )
mesh_instances_buf = Buffer.from_data(ctx, mesh_instances.tobytes(), BufferUsageFlags.STORAGE, AllocType.DEVICE_MAPPED)

print("Creating acceleration structures... ", end="")
begin = perf_counter()
acceleration_structure = AccelerationStructure(ctx, as_meshes)
print(f" Took {perf_counter() - begin:.3f}s")


sampler = Sampler(
    ctx,
    min_filter=Filter.LINEAR,
    mag_filter=Filter.LINEAR,
)

# Write scene descriptos
scene_descriptor_set.write_buffer(normals_buf,        DescriptorType.STORAGE_BUFFER, 0)
scene_descriptor_set.write_buffer(uvs_buf,            DescriptorType.STORAGE_BUFFER, 1)
scene_descriptor_set.write_buffer(indices_buf,        DescriptorType.STORAGE_BUFFER, 2)
scene_descriptor_set.write_buffer(mesh_instances_buf, DescriptorType.STORAGE_BUFFER, 3)
scene_descriptor_set.write_acceleration_structure(acceleration_structure, 4)
scene_descriptor_set.write_sampler(sampler, 5)
for i, image in enumerate(images):
    scene_descriptor_set.write_image(image, ImageUsage.SHADER_READ_ONLY, DescriptorType.SAMPLED_IMAGE, 7, i)

# Register the color pipeline for hot reloading. We pass the window
# so that event loop can be unblocked if a hot reloading event happens.
cache = PipelineWatch([
    rt,
], window=window)

first_frame = True

def draw():
    global first_frame
    global output

    # Refresh shaders. If a shader needs to be recompiled we first wait
    # for the device to become idle. This is needed because as part of the
    # refresh the old pipeline is destroyed, and we need to ensure that no
    # previous frame is still using it.
    cache.refresh(lambda: ctx.wait_idle())

    # Update swapchain
    swapchain_status = window.update_swapchain()
    if swapchain_status == SwapchainStatus.MINIMIZED:
        return
    if first_frame or swapchain_status == SwapchainStatus.RESIZED:
        first_frame = False

        output = Image(ctx, window.fb_width, window.fb_height, Format.R32G32B32A32_SFLOAT, ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_SRC, AllocType.DEVICE_DEDICATED)
        scene_descriptor_set.write_image(output, ImageUsage.IMAGE, DescriptorType.STORAGE_IMAGE, 6)

    # GUI
    with gui.frame():
        if imgui.begin("wow"):
            imgui.text("Hello")
        imgui.end()

    # Render
    with window.frame() as frame:
        descriptor_set = descriptor_sets.get_current_and_advance()

        # TODO: I don't really like the idea of writing these constants
        # 1 by 1 into write-combining memory. I feel like we should probably
        # not return buffer views as numpy arrays, but memory_view if possible.
        #
        # I just want to memcpy bytes in there man!
        u_buf: Buffer = u_bufs.get_current_and_advance()
        constants: np.ndarray = u_buf.view.view(constants_dt)
        constants["width"] = window.fb_width
        constants["height"] = window.fb_height
        constants["camera_position"] = vec3(-9.9, -19.044, 4.352)
        constants["camera_direction"] = normalize(vec3(-1., 3., 0.3) - vec3(-9.9, -19.044, 4.352))
        constants["film_dist"] = 0.7

        with frame.command_buffer as cmd:
            cmd.use_image(output, ImageUsage.IMAGE)

            viewport = [0, 0, window.fb_width, window.fb_height]

            cmd.bind_compute_pipeline(rt.pipeline, descriptor_sets=[scene_descriptor_set, descriptor_set])
            cmd.dispatch((window.fb_width + 7) // 8, (window.fb_height + 7) // 8)

            cmd.use_image(output, ImageUsage.TRANSFER_SRC)
            cmd.use_image(frame.image, ImageUsage.TRANSFER_DST)

            cmd.blit_image(output, frame.image)

            cmd.use_image(frame.image, ImageUsage.COLOR_ATTACHMENT)

            with cmd.rendering(viewport,
                color_attachments=[
                    RenderingAttachment(
                        frame.image,
                        load_op=LoadOp.LOAD,
                        store_op=StoreOp.STORE,
                    ),
                ]):
                # Render gui
                gui.render(cmd)

            cmd.use_image(frame.image, ImageUsage.PRESENT)

window.set_callbacks(draw)

while True:
    process_events(True)

    if window.should_close():
        break

    draw()

cache.stop()