import numpy as np
from pyglm.glm import inverse, ivec2, normalize, quatLookAtRH, vec3
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.lights import DirectionalLight, DirectionalShadowSettings
from ambra.primitives3d import Lines, Mesh
from ambra.utils.descriptors import create_descriptor_layout_pool_and_set
from ambra.utils.hook import hook
from ambra.viewer import Viewer


class CustomViewer(Viewer):
    def __init__(self, title="ambra", config=None, key_map=None):
        super().__init__(title, config, key_map)
        self._texture = None

    @hook
    def on_gui(self):
        global scale
        if imgui.begin("shadow_map")[0]:
            if hasattr(light, "shadow_map"):
                if self._texture is None and light.shadow_map is not None:
                    sampler = Sampler(
                        viewer.ctx,
                        u=SamplerAddressMode.REPEAT,
                        v=SamplerAddressMode.REPEAT,
                    )
                    layout, pool, set = create_descriptor_layout_pool_and_set(
                        viewer.ctx,
                        [
                            DescriptorSetBinding(1, DescriptorType.COMBINED_IMAGE_SAMPLER, stage_flags=Stage.FRAGMENT),
                        ],
                    )
                    set.write_combined_image_sampler(
                        light.shadow_map, ImageLayout.SHADER_READ_ONLY_OPTIMAL, sampler, 0
                    )
                    self._sampler = sampler
                    self._texture = imgui.Texture(set)

                if self._texture is not None:
                    avail = imgui.get_content_region_avail()
                    available = ivec2(avail.x, avail.y)

                    ar = 1.0

                    # height = available.x / ar
                    height = available.y
                    view_size = ivec2(ar * height, height)
                    imgui.image(
                        self._texture,
                        imgui.Vec2(*view_size),
                        uv0=(0, 0),
                        uv1=(1, -1),
                    )
        imgui.end()


viewer = CustomViewer(
    config=Config(
        window_width=1920,
        window_height=1080,
        renderer=RendererConfig(
            msaa_samples=4,
        ),
        playback=PlaybackConfig(
            playing=True,
        ),
        camera=CameraConfig(
            position=vec3(10, 4, 10),
            target=vec3(0),
        ),
        world_up=(0, 0, 1),
        # enable_gpu_based_validation=True,
        gui=GuiConfig(
            stats=True,
        ),
    ),
)

cube_positions = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    np.float32,
)

cube_indices = np.array(
    [
        1,
        5,
        2,
        2,
        5,
        6,
        5,
        4,
        6,
        6,
        4,
        7,
        3,
        2,
        7,
        7,
        2,
        6,
        0,
        1,
        3,
        3,
        1,
        2,
        4,
        0,
        7,
        7,
        0,
        3,
        4,
        5,
        0,
        0,
        5,
        1,
    ],
)

plane_positions = np.array(
    [
        [-10.0, -10.0, -2.0],
        [10.0, -10.0, -2.0],
        [10.0, 10.0, -2.0],
        [-10.0, -10.0, -2.0],
        [10.0, 10.0, -2.0],
        [-10.0, 10.0, -2.0],
    ],
    np.float32,
)

origin_positions = (
    np.array(
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
    * 5.0
)

origin_colors = np.array(
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


translation = np.linspace((0, 0, 0.0), (5, 5, 0), num=50)

m = Mesh(
    cube_positions, cube_indices, translation=translation, cull_mode=CullMode.BACK, front_face=FrontFace.CLOCKWISE
)
p = Mesh(plane_positions)
o = Lines(origin_positions, origin_colors, 4.0)

light_position = vec3(5, 6, 7) * 4.0
light_target = vec3(0, 0, 0)
rotation = quatLookAtRH(normalize(light_target - light_position), vec3(0, 0, 1))
light = DirectionalLight(
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=10.0, z_near=1.0, z_far=100),
    translation=light_position,
    rotation=rotation,
)

viewer.scene.objects.extend([m, p, o, light])

viewer.run()
