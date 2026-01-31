from typing import Optional, Tuple

import numpy as np
from pyglm.glm import vec3

from .lights import DirectionalLight, DirectionalShadowSettings, UniformEnvironmentLight


def two_directional_lights_and_uniform_environment_light(
    front_position: Tuple[float, float, float] = (5, 6, 7),
    front_target: Tuple[float, float, float] = (0, 0, 0),
    front_world_up: Tuple[float, float, float] = (0, 0, 1),
    front_radiance: Tuple[float, float, float] = (1, 1, 1),
    front_shadow_settings: Optional[DirectionalShadowSettings] = None,
    back_position: Tuple[float, float, float] = (-5, -6, 7),
    back_target: Tuple[float, float, float] = (0, 0, 0),
    back_world_up: Tuple[float, float, float] = (0, 0, 1),
    back_radiance: Tuple[float, float, float] = (0.75, 0.75, 0.75),
    back_shadow_settings: Optional[DirectionalShadowSettings] = None,
    uniform_radiance: Tuple[float, float, float] = (0.2, 0.2, 0.2),
) -> Tuple[DirectionalLight, DirectionalLight, UniformEnvironmentLight]:
    front = DirectionalLight.look_at(
        vec3(front_position),
        vec3(front_target),
        vec3(front_world_up),
        np.array(front_radiance),  # type: ignore
        shadow_settings=front_shadow_settings or DirectionalShadowSettings(half_extent=10.0, z_near=1.0, z_far=100),
        name="Front light",
    )

    back = DirectionalLight.look_at(
        vec3(back_position),
        vec3(back_target),
        vec3(back_world_up),
        np.array(back_radiance),  # type: ignore
        shadow_settings=back_shadow_settings or DirectionalShadowSettings(half_extent=10.0, z_near=1.0, z_far=100),
        name="Back light",
    )

    uniform = UniformEnvironmentLight(uniform_radiance, name="Uniform light")

    return front, back, uniform
