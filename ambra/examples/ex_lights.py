import numpy as np
from pyglm.glm import inverse, ivec2, normalize, quatLookAtRH, vec3, vec4
from pyxpg import *

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.geometry import create_axis3d_lines_and_colors, create_cube, create_plane
from ambra.lights import DirectionalLight, DirectionalShadowSettings
from ambra.materials import DiffuseMaterial, PBRMaterial
from ambra.primitives3d import Lines, Mesh
from ambra.utils.descriptors import create_descriptor_layout_pool_and_set
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
                    if light._texture is None and light.shadow_map is not None:
                        layout, pool, set = create_descriptor_layout_pool_and_set(
                            viewer.device,
                            [
                                DescriptorSetBinding(
                                    1, DescriptorType.COMBINED_IMAGE_SAMPLER, stage_flags=Stage.FRAGMENT
                                ),
                            ],
                        )
                        set.write_combined_image_sampler(
                            light.shadow_map, ImageLayout.SHADER_READ_ONLY_OPTIMAL, sampler, 0
                        )
                        light._texture = imgui.Texture(set)

                    if light._texture is not None:
                        avail = imgui.get_content_region_avail()
                        available = ivec2(avail.x, avail.y)

                        ar = 1.0

                        # height = available.x / ar
                        height = available.y
                        view_size = ivec2(ar * height, height)
                        imgui.image(
                            light._texture,
                            imgui.Vec2(*view_size),
                            uv0=(0, 0),
                            uv1=(1, -1),
                        )

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
instance_positions = np.array([(-2, 0, 0), (2, 0, 0)])

# material = PBRMaterial((1, 1, 1), 1.0, 0.0)
material = DiffuseMaterial((1, 1, 1))

cube_positions, cube_normals, cube_indices = create_cube()
m = Mesh(
    cube_positions,
    cube_indices,
    translation=translation,
    cull_mode=CullMode.BACK,
    front_face=FrontFace.COUNTER_CLOCKWISE,
    instance_positions=instance_positions,
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
light1._texture = None

light2 = DirectionalLight.look_at(
    vec3(-2, 3, 4),
    vec3(0, 0, 0),
    vec3(0, 0, 1),
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=5.0, z_near=1.0, z_far=10),
)
light2._texture = None

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
