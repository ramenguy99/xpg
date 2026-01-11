import sys
from pathlib import Path

import numpy as np
from pyglm.glm import vec3
from pyxpg import AllocType, Buffer, BufferUsageFlags, Format, Image, ImageLayout, ImageUsageFlags, MemoryUsage, imgui

from ambra.config import Config, GuiConfig
from ambra.lights import DirectionalLight, DirectionalShadowSettings
from ambra.marching_cubes import MarchingCubesPipeline
from ambra.primitives3d import Grid, GridType, MarchingCubesMesh, Mesh
from ambra.utils.gpu import readback_buffer, view_bytes
from ambra.utils.hook import hook
from ambra.viewer import Viewer

if False:
    x, y, z = 128, 128, 128
    sdf = np.zeros((x, y, z), np.float32)
    size = (1.0, 1.0, 1.0)
else:
    vol_path = Path(sys.argv[1])
    sdf: np.ndarray = np.load(vol_path)["volume"]
    size = np.array(sdf.shape[::-1], np.float32) / max(sdf.shape)
    z, y, x = sdf.shape


class CustomViewer(Viewer):
    @hook
    def on_gui(self):
        global m
        if isinstance(m, MarchingCubesMesh):
            if imgui.begin("MC")[0]:
                u, m.level = imgui.slider_float("Level", m.level, -5, 5)
                if u:
                    m._need_marching = True
            imgui.end()


v = CustomViewer(config=Config(gui=GuiConfig(stats=True)))

print(v.ctx.device_features)
print(v.ctx.subgroup_size_control)
print(v.ctx.compute_full_subgroups)
print(v.ctx.device_properties.subgroup_size_control_properties.min_subgroup_size)
print(v.ctx.device_properties.subgroup_size_control_properties.max_subgroup_size)
print(v.ctx.device_properties.subgroup_size_control_properties.max_compute_workgroup_subgroups)
print(v.ctx.device_properties.subgroup_size_control_properties.required_subgroup_size_stages)

if False:
    # print(v.renderer.ctx.device_properties.subgroup_properties.subgroup_size)
    pipeline = MarchingCubesPipeline(v.renderer)

    sdf_buf = Buffer.from_data(v.ctx, view_bytes(sdf), BufferUsageFlags.TRANSFER_SRC, AllocType.HOST, name="sdf-buf")
    sdf_gpu = Image(
        v.ctx,
        x,
        y,
        depth=z,
        format=Format.R32_SFLOAT,
        usage_flags=ImageUsageFlags.STORAGE | ImageUsageFlags.TRANSFER_DST,
        alloc_type=AllocType.DEVICE,
        name="sdf",
    )

    with v.ctx.sync_commands() as cmd:
        cmd.image_barrier(sdf_gpu, ImageLayout.TRANSFER_DST_OPTIMAL, MemoryUsage.NONE, MemoryUsage.TRANSFER_DST)
        cmd.copy_buffer_to_image(sdf_buf, sdf_gpu)
        cmd.image_barrier(sdf_gpu, ImageLayout.GENERAL, MemoryUsage.TRANSFER_DST, MemoryUsage.COMPUTE_SHADER)

    res = pipeline.run_sync(v.renderer, sdf_gpu, size, 0.0)

    positions = readback_buffer(v.ctx, res.positions).view(np.float32).reshape((-1, 3))
    normals = readback_buffer(v.ctx, res.normals).view(np.float32).reshape((-1, 3))
    indices = readback_buffer(v.ctx, res.indices).view(np.uint32)

    print(positions.shape)
    print(normals.shape)
    print(indices.shape)
    m = Mesh(positions, indices, normals)
else:
    m = MarchingCubesMesh(sdf[:, :, :], size, max_vertices=16 * 1024 * 1024, max_indices=16 * 1024 * 1024)

l1 = DirectionalLight.look_at(
    vec3(5, 6, 7) * 4.0,
    vec3(0, 0, 0),
    vec3(0, 0, 1),
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=10.0, z_near=1.0, z_far=100),
)
l2 = DirectionalLight.look_at(
    vec3(-5, -6, 7) * 4.0,
    vec3(0, 0, 0),
    vec3(0, 0, 1),
    np.array([1.0, 1.0, 1.0]),
    shadow_settings=DirectionalShadowSettings(half_extent=10.0, z_near=1.0, z_far=100),
)

g = Grid.transparent_black_lines((100, 100), GridType.XY_PLANE)

v.scene.objects.extend([m, l1, l2, g])
v.run()
