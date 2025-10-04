import numpy as np
from pyglm.glm import inverse, ivec2, normalize, quatLookAtRH, vec3
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.lights import DirectionalLight, DirectionalShadowSettings
from ambra.primitives3d import Lines, Mesh
from ambra.utils.descriptors import create_descriptor_layout_pool_and_set
from ambra.utils.hook import hook
from ambra.viewer import Viewer
from ambra.materials import DiffuseMaterial, DiffuseSpecularMaterial, BaseColorMaterial
from ambra.geometry import create_sphere
from ambra.property import BufferProperty, as_buffer_property


class CustomViewer(Viewer):
    def __init__(self, title="ambra", config=None, key_map=None):
        super().__init__(title, config, key_map)
        self._texture = None

    @hook
    def on_gui(self):
        global scale
        if imgui.begin("Shadowmap"):
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
                        uv1=(1, 1),
                    )
        imgui.end()


viewer = CustomViewer(
    "primitives",
    config=Config(
        window_width=1600,
        window_height=900,
        playback=PlaybackConfig(
            enabled=True,
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

v, f = create_sphere(0.5)
n = v / np.linalg.norm(v, axis=1, keepdims=True)

p_v = as_buffer_property(v, np.float32, (-1, 3))
p_n = as_buffer_property(n, np.float32, (-1, 3))
p_f = as_buffer_property(f, np.uint32, (-1,))

spheres = []
for y in range(8):
    for x in range(8):
        t_x = ((x + 0.5) / 8 * 2.0 - 1.0) * 5
        t_y = ((y + 0.5) / 8 * 2.0 - 1.0) * 5
        # m = DiffuseMaterial((x / 8, y / 8, 0))
        # m = BaseColorMaterial((x / 8, y / 8, 0))
        m = DiffuseSpecularMaterial((x / 8, 0.0, 0.0), y / 8.0)
        sphere = Mesh(p_v, p_f, normals=p_n, translation=(t_x, t_y, 0), material=m)
        spheres.append(sphere)

translation = np.linspace((0, 0, 0.0), (5, 5, 0), num=50)

material = DiffuseMaterial((1.0, 1.0, 1.0))
p = Mesh(plane_positions, material=material)
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

viewer.viewport.scene.objects.extend(spheres + [p, o, light])

viewer.run()
