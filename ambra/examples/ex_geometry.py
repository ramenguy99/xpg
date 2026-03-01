from pyglm.glm import vec3
from pyxpg import CullMode, PolygonMode

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig, RendererConfig
from ambra.geometry import (
    create_arrow,
    create_axis2d_lines_and_colors,
    create_axis3d_lines_and_colors,
    create_capped_cone,
    create_capped_cylinder,
    create_cone,
    create_cube,
    create_cube_edges,
    create_cylinder,
    create_disk,
    create_normal_lines,
    create_sphere,
)
from ambra.primitives3d import Lines, Mesh
from ambra.stages import two_directional_lights_and_uniform_environment_light
from ambra.viewer import Viewer

viewer = Viewer(
    config=Config(
        playback=PlaybackConfig(
            playing=True,
        ),
        camera=CameraConfig(
            position=vec3(3),
            target=vec3(0),
        ),
        gui=GuiConfig(
            playback=True,
            stats=True,
            inspector=True,
        ),
        renderer=RendererConfig(
            msaa_samples=4,
        ),
    ),
)

cull_mode = CullMode.BACK
polygon_mode = PolygonMode.FILL

cube_v, cube_n, cube_f = create_cube()
cube = Mesh(cube_v, cube_f, cube_n, cull_mode=cull_mode, polygon_mode=polygon_mode, translation=(-2.5, 0, 0))
cube_normals = Lines(*create_normal_lines(cube_v, cube_n, length=0.05), line_width=2, translation=(-2.5, 0, 0))

cone_v, cone_n, cone_f = create_cone()
cone = Mesh(cone_v, cone_f, cone_n, cull_mode=cull_mode, polygon_mode=polygon_mode, translation=(2.5, 0, 0))
cone_normals = Lines(*create_normal_lines(cone_v, cone_n, length=0.05), line_width=2, translation=(2.5, 0, 0))

capped_cone_v, capped_cone_n, capped_cone_f = create_capped_cone()
capped_cone = Mesh(
    capped_cone_v,
    capped_cone_f,
    capped_cone_n,
    cull_mode=cull_mode,
    polygon_mode=polygon_mode,
    translation=(2.5, -2.5, 0),
)
capped_cone_normals = Lines(
    *create_normal_lines(capped_cone_v, capped_cone_n, length=0.05), line_width=2, translation=(2.5, -2.5, 0)
)

disk_v, disk_n, disk_f = create_disk()
disk = Mesh(disk_v, disk_f, disk_n, cull_mode=cull_mode, polygon_mode=polygon_mode, translation=(0, 2.5, 0))
disk_normals = Lines(*create_normal_lines(disk_v, disk_n, length=0.05), line_width=2, translation=(0, 2.5, 0))

cylinder_v, cylinder_n, cylinder_f = create_cylinder()
cylinder = Mesh(
    cylinder_v, cylinder_f, cylinder_n, cull_mode=cull_mode, polygon_mode=polygon_mode, translation=(0, -2.5, 0)
)
cylinder_normals = Lines(
    *create_normal_lines(cylinder_v, cylinder_n, length=0.05), line_width=2, translation=(0, -2.5, 0)
)

cap_cylinder_v, cap_cylinder_n, cap_cylinder_f = create_capped_cylinder()
cap_cylinder = Mesh(
    cap_cylinder_v,
    cap_cylinder_f,
    cap_cylinder_n,
    cull_mode=cull_mode,
    polygon_mode=polygon_mode,
    translation=(-2.5, -2.5, 0),
)
cap_cylinder_normals = Lines(
    *create_normal_lines(cap_cylinder_v, cap_cylinder_n, length=0.05), line_width=2, translation=(-2.5, -2.5, 0)
)

sphere_v, sphere_n, sphere_f = create_sphere()
sphere = Mesh(sphere_v, sphere_f, sphere_n, cull_mode=cull_mode, polygon_mode=polygon_mode, translation=(-2.5, 2.5, 0))
sphere_normals = Lines(*create_normal_lines(sphere_v, sphere_n, length=0.05), line_width=2, translation=(-2.5, 2.5, 0))

arrow_v, arrow_n, arrow_f = create_arrow()
arrow = Mesh(arrow_v, arrow_f, arrow_n, cull_mode=cull_mode, polygon_mode=polygon_mode, translation=(2.5, 2.5, 0))
arrow_normals = Lines(*create_normal_lines(arrow_v, arrow_n, length=0.05), line_width=2, translation=(2.5, 2.5, 0))

ax3d = Lines(*create_axis3d_lines_and_colors(), line_width=2, translation=(0, 0, 0))
cube_edges = Lines(*create_cube_edges(), line_width=2, translation=(0, 0, 0))


viewer.scene.objects.extend(
    [
        cube,
        cube_normals,
        cone,
        cone_normals,
        capped_cone,
        capped_cone_normals,
        disk,
        disk_normals,
        cylinder,
        cylinder_normals,
        cap_cylinder,
        cap_cylinder_normals,
        sphere,
        sphere_normals,
        arrow,
        arrow_normals,
        ax3d,
        cube_edges,
        *two_directional_lights_and_uniform_environment_light(),
    ]
)

viewer.run()
