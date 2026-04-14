import numpy as np
from pyglm.glm import vec3

from ambra.config import CameraConfig, Config, GuiConfig, PlaybackConfig
from ambra.primitives3d import AxisGizmo
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
    ),
)

N = 100
transforms = np.zeros((N, 3, 4), np.float32)
transforms[:, :, 3] = np.linspace((-5, -5, -5), (5, 5, 5), 100, dtype=np.float32)
transforms[:, :, :3] = np.eye(3, dtype=np.float32)

obj = AxisGizmo(instance_transforms=transforms)
viewer.scene.objects.append(obj)
viewer.scene.objects.extend(two_directional_lights_and_uniform_environment_light())

viewer.run()
