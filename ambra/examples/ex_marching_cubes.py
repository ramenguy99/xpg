from ambra.marching_cubes import MarchingCubesPipeline
from ambra.viewer import Viewer
from ambra.config import Config

v = Viewer(config=Config(window=False))

print(v.ctx.device_features)
print(v.ctx.subgroup_size_control)
print(v.ctx.compute_full_subgroups)
print(v.ctx.device_properties.subgroup_size_control_properties.min_subgroup_size)
print(v.ctx.device_properties.subgroup_size_control_properties.max_subgroup_size)
print(v.ctx.device_properties.subgroup_size_control_properties.max_compute_workgroup_subgroups)
print(v.ctx.device_properties.subgroup_size_control_properties.required_subgroup_size_stages)

# print(v.renderer.ctx.device_properties.subgroup_properties.subgroup_size)
# pipeline = MarchingCubesPipeline(v.renderer)
