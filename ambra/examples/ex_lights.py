import numpy as np
from pyglm.glm import normalize, vec3, vec4
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.geometry import create_axis3d_lines_and_colors, create_cube, create_plane
from ambra.lights import DirectionalLight, DirectionalShadowSettings
from ambra.materials import DiffuseMaterial
from ambra.primitives3d import Lines, Mesh
from ambra.utils.gui import GuiImage
from ambra.utils.hook import hook
from ambra.viewer import Viewer


class CustomViewer(Viewer):
    def __init__(self, title="ambra", config=None, key_map=None):
        super().__init__(title, config, key_map)

    @hook
    def on_gui(self):
        for light in [light1, light2]:
            if imgui.begin(light.name)[0]:
                if hasattr(light, "shadow_map"):
                    u, t = imgui.drag_float3("Translation", light.translation.get_current(), 0.01)
                    if u:
                        light.translation.update_frame(0, np.array(t))

                    u, s = imgui.drag_float3("Scale", light.scale.get_current(), 0.01)
                    if u:
                        light.scale.update_frame(0, np.array(s))

                    u, r = imgui.drag_float4("Rotation", light.rotation.get_current(), 0.1)

                    _, light.shadow_settings.min_bias = imgui.drag_float(
                        "Min Bias", light.shadow_settings.min_bias, 0.0001, 0, 1.0, "%.4f"
                    )
                    _, light.shadow_settings.max_bias = imgui.drag_float(
                        "Max Bias", light.shadow_settings.max_bias, 0.0001, 0, 1.0, "%.4f"
                    )

                    if u:
                        light.rotation.update_frame(0, np.array(normalize(vec4(r))))
                    if light.shadow_image is None and light.shadow_map is not None:
                        light.shadow_image = GuiImage(viewer.device, light.shadow_map, sampler)

                    if light.shadow_image is not None:
                        light.shadow_image.draw_square(imgui.get_window_draw_list(), uv_max=(1, -1))

            imgui.end()


viewer = CustomViewer(
    config=Config(
        renderer=RendererConfig(
            msaa_samples=4,
            path_tracer_max_bounces=0,
        ),
        playback=PlaybackConfig(
            playing=False,
        ),
        camera=CameraConfig(
            position=vec3(10, 4, 10),
            target=vec3(0),
        ),
        world_up=(0, 0, 1),
        gui=GuiConfig(
            stats=True,
            # multiviewport=True,
            # initial_number_of_viewports=2,
            inspector=True,
        ),
    ),
)

translation = np.linspace((0, 0, 0.0), (5, 5, 0), num=50)
instance_transforms = np.array(
    [
        [
            [1, 0, 0, -2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        [
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
    ],
    np.float32,
)

# material = PBRMaterial((1, 1, 1), 1.0, 0.0)
material = DiffuseMaterial((1, 1, 1))

cube_positions, cube_normals, cube_indices = create_cube()
m = Mesh(
    cube_positions,
    cube_indices,
    translation=translation,
    cull_mode=CullMode.BACK,
    front_face=FrontFace.COUNTER_CLOCKWISE,
    instance_transforms=instance_transforms,
    material=material,
)

plane_positions, plane_normals, plane_indices = create_plane((0, 0, -0.51), (10, 10))
p = Mesh(plane_positions, plane_indices, plane_normals, material=material)
o = Lines(*create_axis3d_lines_and_colors(), 4.0)

light1 = DirectionalLight.look_at(
    # vec3(-2, 3, 4),
    vec3(2, 3, 4),
    vec3(0, 0, 0),
    vec3(0, 0, 1),
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=5.0, z_near=1.0, z_far=10),
)
light1.shadow_image = None

light2 = DirectionalLight.look_at(
    vec3(-2, 3, 4),
    vec3(0, 0, 0),
    vec3(0, 0, 1),
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=5.0, z_near=1.0, z_far=10),
)
light2.shadow_image = None

sampler = Sampler(
    viewer.device,
    u=SamplerAddressMode.REPEAT,
    v=SamplerAddressMode.REPEAT,
)

viewer.scene.objects.extend(
    [
        m,
        p,
        o,
        light1,
        light2,
    ]
)

viewer.run()
