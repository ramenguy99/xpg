import sys

import numpy as np
import PIL.Image
from pyxpg import Filter, PrimitiveTopology

from ambra.config import CameraConfig, Config, GuiConfig
from ambra.lights import UniformEnvironmentLight
from ambra.materials import DiffuseMaterial, MaterialProperty, MaterialSamplingOptions
from ambra.primitives3d import Mesh
from ambra.property import ArrayImageProperty, UploadSettings
from ambra.utils import profile
from ambra.viewer import Viewer

data = np.asarray(PIL.Image.open(sys.argv[1]))
if data.shape[2] == 3:
    data = np.dstack((data, np.full(data.shape[:2], 255, data.dtype)))

test_sequence = False
if test_sequence:
    data = data[:256, :256]
    with profile.profile("Interpolate"):
        data = np.linspace(data, np.full_like(data, 255), 2000, dtype=np.uint8)
else:
    data = data[np.newaxis]

img_data = ArrayImageProperty(
    data, upload=UploadSettings(preupload=False, async_load=True, cpu_prefetch_count=2, gpu_prefetch_count=2)
)
positions = (
    np.array(
        [
            -1.0,
            -1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            1.0,
            1.0,
            0.0,
        ]
    ).reshape((4, 3))
    * 10
)
uvs = np.array(
    [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
    ]
).reshape((4, 2))

material = DiffuseMaterial(
    MaterialProperty(img_data, 3, srgb=True, mips=True, name="albedo"),
    sampling_options=MaterialSamplingOptions(mag_filter=Filter.NEAREST),
)
mesh = Mesh(
    positions,
    uvs=uvs,
    primitive_topology=PrimitiveTopology.TRIANGLE_STRIP,
    material=material,
)

l = UniformEnvironmentLight((0.5, 0.5, 0.5))

v = Viewer(
    config=Config(
        camera=CameraConfig(position=(0, 0, 1.0)),
        world_up=(0, 1, 0),
        gui=GuiConfig(stats=True, playback=True, renderer=True),
    )
)
v.scene.objects.extend([mesh, l])
v.run()
