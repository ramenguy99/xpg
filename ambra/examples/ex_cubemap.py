import PIL.Image
import numpy as np

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.primitives3d import *
from ambra.viewer import Viewer
from ambra.geometry import create_cube
from ambra.lights import DirectionalLight, DirectionalShadowSettings
from ambra.utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer, create_descriptor_layout_pool_and_set, create_descriptor_layout_pool_and_sets
from ambra.utils.profile import profile

from pyglm.glm import normalize, quatLookAtRH, vec3

from pyxpg import *

import sys
import imageio.v3 as iio

class DebugCube(Object3D):
    def __init__(
        self,
        cubemap: Image,
        positions: BufferProperty,
        indices: Optional[BufferProperty] = None,
        name: Optional[str] = None,
    ):
        self.constants_dtype = np.dtype(
            {
                "transform": (np.dtype((np.float32, (3, 4))), 0),
                "level": (np.uint32, 48),
            }
        )  # type: ignore
        self.constants = np.zeros((1,), self.constants_dtype)
        self.constants["transform"] = mat4x3(1.0)
        self.constants["level"] = 1

        super().__init__(name)

        self.positions = self.add_buffer_property(positions, np.float32, (-1, 3), name="positions")
        self.indices = (
            self.add_buffer_property(indices, np.uint32, (-1,), name="indices") if indices is not None else None
        )
        self.cubemap = cubemap

    def create(self, r: Renderer) -> None:
        self.positions_buffer = r.add_gpu_buffer_property(
            self.positions,
            BufferUsageFlags.VERTEX,
            MemoryUsage.VERTEX_INPUT,
            PipelineStageFlags.VERTEX_INPUT,
            name=f"{self.name}-positions",
        )
        self.indices_buffer = (
            r.add_gpu_buffer_property(
                self.indices,
                BufferUsageFlags.INDEX,
                MemoryUsage.VERTEX_INPUT,
                PipelineStageFlags.VERTEX_INPUT,
                name=f"{self.name}-indices",
            )
            if self.indices is not None
            else None
        )

        vertex_bindings = [
            VertexBinding(0, 12, VertexInputRate.VERTEX),
        ]
        vertex_attributes = [
            VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
        ]

        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_sets = create_descriptor_layout_pool_and_sets_ringbuffer(r.ctx, [
            DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
        ], r.window.num_frames)
        self.sampler = Sampler(r.ctx, Filter.NEAREST, Filter.NEAREST)
        # self.sampler = Sampler(r.ctx, Filter.LINEAR, Filter.LINEAR)

        vert = r.get_builtin_shader("3d/cube.slang", "vertex_main")
        frag = r.get_builtin_shader("3d/cube.slang", "pixel_main")

        self.pipeline = GraphicsPipeline(
            r.ctx,
            stages=[
                PipelineStage(Shader(r.ctx, vert.code), Stage.VERTEX),
                PipelineStage(Shader(r.ctx, frag.code), Stage.FRAGMENT),
            ],
            vertex_bindings=vertex_bindings,
            vertex_attributes=vertex_attributes,
            input_assembly=InputAssembly(PrimitiveTopology.TRIANGLE_LIST),
            attachments=[Attachment(format=r.output_format)],
            depth=Depth(r.depth_format, True, True, r.depth_compare_op),
            descriptor_set_layouts=[
                r.scene_descriptor_set_layout,
                self.descriptor_set_layout,
            ],
            push_constants_ranges=[PushConstantsRange(self.constants_dtype.itemsize)],
        )

    def render(self, r: Renderer, frame: RendererFrame, scene_descriptor_set: DescriptorSet) -> None:
        index_buffer = self.indices_buffer.get_current() if self.indices_buffer is not None else None
        vertex_buffers = [
            self.positions_buffer.get_current(),
        ]

        self.constants["level"] = level

        descriptor_set = self.descriptor_sets.get_current_and_advance()
        descriptor_set.write_combined_image_sampler(self.cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, self.sampler, 0)

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=vertex_buffers,
            index_buffer=index_buffer,
            descriptor_sets=[
                scene_descriptor_set,
                descriptor_set,
            ],
            push_constants=self.constants.tobytes(),
        )

        if self.indices is not None:
            frame.cmd.draw_indexed(self.indices.get_current().shape[0])
        else:
            frame.cmd.draw(self.positions.get_current().shape[0])

level = 0
class CustomViewer(Viewer):
    def on_key(self, key: Key, action: Action, modifiers: Modifiers):
        global level
        if action == Action.PRESS:
            if key == Key.Z:
                level = max(level - 1, 0)
            if key == Key.X:
                level = min(level + 1, skybox_mip_levels - 1)
            if key == Key.C:
                cube.cubemap = skybox_cubemap
            if key == Key.V:
                cube.cubemap = irradiance_cubemap
            if key == Key.B:
                cube.cubemap = specular_cubemap
        return super().on_key(key, action, modifiers)

viewer = CustomViewer(
    "primitives",
    config=Config(
        playback=PlaybackConfig(
            enabled=True,
            playing=True,
        ),
        gui=GuiConfig(
            stats=True,
        ),
        world_up=(0, 0, 1),
        camera=CameraConfig(
            position=(3, 3, 3),
            target=(0, 0, 0),
        ),
    ),
)

positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    np.float32,
)

colors = np.array(
    [
        0xFF0000FF,
        0xFF0000FF,
        0xFF00FF00,
        0xFF00FF00,
        0xFFFF0000,
        0xFFFF0000,
    ],
    np.uint32,
)

ctx = viewer.ctx

# Load image
hdr_data = iio.imread(sys.argv[1])
hdr_height, hdr_width, hdr_channels = hdr_data.shape
assert hdr_channels == 3 and hdr_data.dtype == np.float32, f"Format not supported {hdr_channels} channels {hdr_data.dtype} type"
hdr_data = np.dstack((hdr_data, np.ones((hdr_height, hdr_width, 1), np.float32)))

# Upload image
hdr_img = Image.from_data(ctx, hdr_data, ImageLayout.SHADER_READ_ONLY_OPTIMAL, hdr_width, hdr_height, Format.R32G32B32A32_SFLOAT, ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST, AllocType.DEVICE)
hdr_sampler = Sampler(ctx, Filter.LINEAR, Filter.LINEAR)

skybox_size = 1024
skybox_mip_levels = skybox_size.bit_length()
skybox_cubemap = Image(
    ctx,
    skybox_size,
    skybox_size,
    Format.R16G16B16A16_SFLOAT,
    ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_SRC | ImageUsageFlags.TRANSFER_DST,
    AllocType.DEVICE,
    array_layers=6,
    mip_levels=skybox_mip_levels,
    is_cube=True,
)

def skybox():
    descriptor_set_layout, descriptor_pool, descriptor_set = create_descriptor_layout_pool_and_set(ctx, [
        DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
        DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
    ])

    constants_dtype = np.dtype(
        {
            "size": (np.uint32, 0),
        }
    )  # type: ignore
    constants = np.zeros((1,), constants_dtype)
    constants["size"] = skybox_size

    cubemap_array_view = ImageView(ctx, skybox_cubemap, ImageViewType.TYPE_2D_ARRAY, mip_level_count=1)
    descriptor_set.write_image(cubemap_array_view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)
    descriptor_set.write_combined_image_sampler(hdr_img, ImageLayout.SHADER_READ_ONLY_OPTIMAL, hdr_sampler, 1)

    shader = viewer.renderer.get_builtin_shader("3d/ibl.slang", "entry_skybox")
    compute_pipeline = ComputePipeline(ctx, Shader(ctx, shader.code), descriptor_set_layouts=[descriptor_set_layout], push_constants_ranges=[PushConstantsRange(constants_dtype.itemsize)])

    with viewer.ctx.sync_commands() as cmd:
        cmd.image_barrier(skybox_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, undefined=True)
        cmd.bind_compute_pipeline(compute_pipeline, descriptor_sets=[descriptor_set], push_constants=constants.tobytes())
        cmd.dispatch((skybox_size + 7) // 8, (skybox_size + 7) // 8, 6)
        cmd.image_barrier(skybox_cubemap, ImageLayout.GENERAL, MemoryUsage.IMAGE, MemoryUsage.ALL)
        for m in range(1, skybox_mip_levels):
            src_size = skybox_size >> m - 1
            dst_size = skybox_size >> m
            cmd.blit_image_range(skybox_cubemap, skybox_cubemap, src_size, src_size, dst_size, dst_size, Filter.LINEAR, src_mip_level=m - 1, dst_mip_level=m, src_array_layer_count=6, dst_array_layer_count=6)
            cmd.memory_barrier(MemoryUsage.ALL, MemoryUsage.ALL)
        cmd.image_barrier(skybox_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.IMAGE, MemoryUsage.ALL)

irradiance_size = 32
irradiance_cubemap = Image(
    ctx,
    irradiance_size,
    irradiance_size,
    Format.R16G16B16A16_SFLOAT,
    ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE,
    AllocType.DEVICE,
    array_layers=6,
    is_cube=True,
)

# TODO:
# - Try sampling cubemap instead of HDR directly
def irradiance():
    descriptor_set_layout, descriptor_pool, descriptor_set = create_descriptor_layout_pool_and_set(ctx, [
        DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
        DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
    ])

    constants_dtype = np.dtype(
        {
            "size": (np.uint32, 0),
            "samples_phi": (np.uint32, 4),
            "samples_theta": (np.uint32, 8),
        }
    )  # type: ignore
    constants = np.zeros((1,), constants_dtype)
    constants["size"] = irradiance_size
    constants["samples_phi"] = 256
    constants["samples_theta"] = 64

    cubemap_array_view = ImageView(ctx, irradiance_cubemap, ImageViewType.TYPE_2D_ARRAY, mip_level_count=1)
    descriptor_set.write_image(cubemap_array_view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)

    cubemap_sampler = Sampler(ctx, Filter.LINEAR, Filter.LINEAR, SamplerMipmapMode.LINEAR)
    descriptor_set.write_combined_image_sampler(skybox_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, cubemap_sampler, 1)

    shader = viewer.renderer.get_builtin_shader("3d/ibl.slang", "entry_irradiance")
    compute_pipeline = ComputePipeline(ctx, Shader(ctx, shader.code), descriptor_set_layouts=[descriptor_set_layout], push_constants_ranges=[PushConstantsRange(constants_dtype.itemsize)])

    with profile("irradiance"), viewer.ctx.sync_commands() as cmd:
        cmd.image_barrier(irradiance_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, undefined=True)
        cmd.bind_compute_pipeline(compute_pipeline, descriptor_sets=[descriptor_set], push_constants=constants.tobytes())
        cmd.dispatch((irradiance_size + 7) // 8, (irradiance_size + 7) // 8, 6)
        cmd.image_barrier(irradiance_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.IMAGE, MemoryUsage.ALL, undefined=False)

mip_levels = 5
specular_size = 1024
specular_cubemap = Image(
    ctx,
    specular_size,
    specular_size,
    Format.R16G16B16A16_SFLOAT,
    ImageUsageFlags.SAMPLED | ImageUsageFlags.STORAGE,
    AllocType.DEVICE,
    array_layers=6,
    mip_levels=mip_levels,
    is_cube=True,
)

def specular():
    cubemap_sampler = Sampler(ctx, Filter.LINEAR, Filter.LINEAR, SamplerMipmapMode.LINEAR)
    descriptor_set_layout, descriptor_pool, descriptor_sets = create_descriptor_layout_pool_and_sets(ctx, [
        DescriptorSetBinding(1, DescriptorType.STORAGE_IMAGE),
        DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
    ], mip_levels)

    constants_dtype = np.dtype(
        {
            "size": (np.uint32, 0),
            "samples": (np.uint32, 4),
            "roughness": (np.float32, 8),
            "skybox_resolution": (np.float32, 12),
        }
    )  # type: ignore
    constants = np.zeros((1,), constants_dtype)
    constants["samples"] = 1024
    constants["skybox_resolution"] = skybox_size

    shader = viewer.renderer.get_builtin_shader("3d/ibl.slang", "entry_specular")
    compute_pipeline = ComputePipeline(ctx, Shader(ctx, shader.code), descriptor_set_layouts=[descriptor_set_layout], push_constants_ranges=[PushConstantsRange(constants_dtype.itemsize)])

    views = []
    with profile("specular"), viewer.ctx.sync_commands() as cmd:
        cmd.image_barrier(specular_cubemap, ImageLayout.GENERAL, MemoryUsage.NONE, MemoryUsage.IMAGE, undefined=True)

        for m in range(0, mip_levels):
            descriptor_set = descriptor_sets[m]
            specular_mip_view = ImageView(ctx, specular_cubemap, ImageViewType.TYPE_2D_ARRAY, base_mip_level=m, mip_level_count=1)
            descriptor_set.write_image(specular_mip_view, ImageLayout.GENERAL, DescriptorType.STORAGE_IMAGE, 0)
            descriptor_set.write_combined_image_sampler(skybox_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, cubemap_sampler, 1)
            views.append(specular_mip_view)

            mip_size = specular_size >> m
            constants["size"] = mip_size
            constants["roughness"] = m / (mip_levels - 1)
            cmd.bind_compute_pipeline(compute_pipeline, descriptor_sets=[descriptor_set], push_constants=constants.tobytes())
            cmd.dispatch((mip_size + 7) // 8, (mip_size + 7) // 8, 6)

        cmd.image_barrier(specular_cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, MemoryUsage.ALL, MemoryUsage.ALL)

skybox()
irradiance()
specular()

cube_positions, cube_faces = create_cube()
cube = DebugCube(specular_cubemap, cube_positions, cube_faces)

line_width = 4
line = Lines(positions * 3, colors, line_width, translation=(-1, -1, -1))

light_position = vec3(5, 6, 7) * 4.0
light_target = vec3(0, 0, 0)
rotation = quatLookAtRH(normalize(light_target - light_position), vec3(0, 0, 1))
light = DirectionalLight(
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=10.0, z_near=1.0, z_far=100),
    translation=light_position,
    rotation=rotation,
)


viewer.viewport.scene.objects.extend([cube, line, light])

viewer.run()
