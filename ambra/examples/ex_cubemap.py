import os
import sys
from typing import Optional

import cv2
import numpy as np
from pyglm.glm import mat4x3, normalize, quatLookAtRH, vec3
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.geometry import create_cube
from ambra.lights import DirectionalLight, DirectionalShadowSettings, EnvironmentCubemaps
from ambra.primitives3d import Lines
from ambra.renderer import Renderer
from ambra.renderer_frame import RendererFrame
from ambra.scene import BufferProperty, Object3D
from ambra.utils.descriptors import create_descriptor_layout_pool_and_sets_ringbuffer
from ambra.viewer import Viewer

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


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

        self.positions = self.add_buffer_property(positions, np.float32, (-1, 3), name="positions").use_gpu(
            BufferUsageFlags.VERTEX, PipelineStageFlags.VERTEX_INPUT
        )
        self.indices = (
            self.add_buffer_property(indices, np.uint32, (-1,), name="indices").use_gpu(
                BufferUsageFlags.INDEX, PipelineStageFlags.VERTEX_INPUT
            )
            if indices is not None
            else None
        )
        self.cubemap = cubemap

    def create(self, r: Renderer) -> None:
        vertex_bindings = [
            VertexBinding(0, 12, VertexInputRate.VERTEX),
        ]
        vertex_attributes = [
            VertexAttribute(0, 0, Format.R32G32B32_SFLOAT),
        ]

        self.descriptor_set_layout, self.descriptor_pool, self.descriptor_sets = (
            create_descriptor_layout_pool_and_sets_ringbuffer(
                r.ctx,
                [
                    DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER),
                ],
                r.num_frames_in_flight,
            )
        )
        self.sampler = Sampler(r.ctx, Filter.NEAREST, Filter.NEAREST)
        # self.sampler = Sampler(r.ctx, Filter.LINEAR, Filter.LINEAR)

        vert = r.compile_builtin_shader("3d/cube.slang", "vertex_main")
        frag = r.compile_builtin_shader("3d/cube.slang", "pixel_main")

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
        vertex_buffers = [
            self.positions.get_current_gpu().buffer_and_offset(),
        ]

        self.constants["level"] = level

        descriptor_set = self.descriptor_sets.get_current_and_advance()
        descriptor_set.write_combined_image_sampler(
            self.cubemap, ImageLayout.SHADER_READ_ONLY_OPTIMAL, self.sampler, 0
        )

        frame.cmd.bind_graphics_pipeline(
            self.pipeline,
            vertex_buffers=vertex_buffers,
            index_buffer=self.indices.get_current_gpu().buffer_and_offset() if self.indices is not None else None,
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


def main():
    class CustomViewer(Viewer):
        def on_key(self, key: Key, action: Action, modifiers: Modifiers):
            global level
            if action == Action.PRESS:
                if key == Key.Z:
                    level = max(level - 1, 0)
                if key == Key.X:
                    level = min(level + 1, 10)
                if key == Key.V:
                    cube.cubemap = result.irradiance_cubemap
                if key == Key.B:
                    cube.cubemap = result.specular_cubemap
            return super().on_key(key, action, modifiers)

    viewer = CustomViewer(
        config=Config(
            playback=PlaybackConfig(
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
    hdr_data = cv2.imread(sys.argv[1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR_RGB)
    hdr_height, hdr_width, hdr_channels = hdr_data.shape
    assert hdr_channels == 3 and hdr_data.dtype == np.float32, (
        f"Format not supported {hdr_channels} channels {hdr_data.dtype} type"
    )
    hdr_data = np.dstack((hdr_data, np.ones((hdr_height, hdr_width, 1), np.float32)))
    hdr_img = Image.from_data(
        ctx,
        hdr_data,
        ImageLayout.SHADER_READ_ONLY_OPTIMAL,
        hdr_width,
        hdr_height,
        Format.R32G32B32A32_SFLOAT,
        ImageUsageFlags.SAMPLED | ImageUsageFlags.TRANSFER_DST,
        AllocType.DEVICE,
    )

    result = viewer.renderer.run_ibl_pipeline(hdr_img)

    # Test round trip from GPU to CPU to disk and back.
    cubemap_data = result.cpu(ctx)
    file = "cubemap_ibl.npz"
    cubemap_data.save(file)
    cubemap_data = EnvironmentCubemaps.load(file)
    result = cubemap_data.gpu(ctx)

    cube_positions, _, cube_faces = create_cube(extents=(2.0, 2.0, 2.0))
    cube = DebugCube(result.specular_cubemap, cube_positions, cube_faces)

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

    viewer.scene.objects.extend([cube, line, light])

    viewer.run()


if __name__ == "__main__":
    main()
