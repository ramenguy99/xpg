import numpy as np
from pyxpg import imgui

from ambra.config import (
    CameraConfig,
    Config,
    GuiConfig,
    RendererConfig,
)
from ambra.geometry import create_sphere
from ambra.lights import UniformEnvironmentLight
from ambra.materials import ColorMaterial, DiffuseMaterial, PBRMaterial
from ambra.primitives3d import Mesh, Spheres
from ambra.property import ArrayBufferProperty, ArrayImageProperty, UploadSettings
from ambra.viewer import Viewer

color = (0.8, 0.4, 0)
color_u8s = (*(int(c * 255.0) for c in color), 255)
color_u32 = color_u8s[0] << 0 | color_u8s[1] << 8 | color_u8s[2] << 16 | color_u8s[3] << 24


class CustomViewer(Viewer):
    def on_gui(self):
        global color
        super().on_gui()

        if imgui.begin("Material"):
            u, color = imgui.color_edit3("Color", color)
            if u:
                pbr_material.albedo[0].property.update_frame(0, (*color, 1.0))
                pbr_material.need_upload = True

                diffuse_material.diffuse[0].property.update_frame(0, (*color, 1.0))
                diffuse_material.need_upload = True

                color_material.color[0].property.update_frame(0, (*color, 1.0))
                color_material.need_upload = True

                img_data = np.zeros((1, 1, 1, 4), np.uint8)
                img_data[:, :, :, :] = (*(int(c * 255.0) for c in color), 1.0)

                img_pbr_material.albedo[0].property.update_frame(0, img_data)
                img_pbr_material.need_upload = True
                img_color_material.need_upload = True
                img_diffuse_material.need_upload = True
        imgui.end()


v, n, f = create_sphere(0.5, rings=32, sectors=64)
p_v = ArrayBufferProperty(v[np.newaxis], np.float32)
p_n = ArrayBufferProperty(n[np.newaxis], np.float32)
p_f = ArrayBufferProperty(f[np.newaxis], np.uint32)

pbr_material = PBRMaterial(color, 1.0, 0.0)
diffuse_material = DiffuseMaterial(color)
color_material = ColorMaterial(color)
buf_materials = [pbr_material, diffuse_material, color_material]

img_data = np.zeros((1, 1, 1, 4), np.uint8)
img_data[:, :, :, :] = (*(int(c * 255.0) for c in color), 1.0)
img = ArrayImageProperty(img_data, upload=UploadSettings(preupload=False))

img_pbr_material = PBRMaterial(img, 1.0, 0.0)
img_diffuse_material = DiffuseMaterial(img)
img_color_material = ColorMaterial(img)
img_materials = [img_pbr_material, img_diffuse_material, img_color_material]

spheres = []
for y, materials in enumerate([buf_materials, img_materials]):
    for x, m in enumerate(materials):
        spheres.append(
            Mesh(
                p_v,
                p_f,
                normals=p_n,
                translation=(x * 1, 0, y * 1),
                material=m,
            )
        )

spheres.append(
    Spheres(
        positions=np.array([[0, 0, 0]], np.float32),
        colors=np.array([color_u32], np.uint32),
        radius=0.5,
        translation=(0, 0, -1),
    )
)
p_c = np.full(p_v.shape[0], color_u32, np.uint32)
spheres.append(
    Mesh(
        p_v,
        p_f,
        normals=p_n,
        vertex_colors=p_c,
        translation=(1, 0, -1),
    )
)
spheres.append(
    Mesh(
        p_v,
        p_f,
        normals=p_n,
        vertex_colors=p_c,
        translation=(2, 0, -1),
        material=ColorMaterial((1, 1, 1, 1)),
    )
)

light = UniformEnvironmentLight(np.array((1, 1, 1)))

viewer = CustomViewer(
    config=Config(
        gui=GuiConfig(stats=True, inspector=True),
        world_up=(0, 0, 1),
        renderer=RendererConfig(
            msaa_samples=4,
            # background_color=(0, 0, 0, 1.0),
            background_color=(1.0, 1.0, 1.0, 1.0),
            # background_color=(*color, 1.0),
        ),
        camera=CameraConfig(
            position=(0, -10, 0),
            target=(0, 0, 0),
            z_near=0.1,
        ),
    )
)

viewer.scene.objects.extend(spheres + [light])
viewer.run()
