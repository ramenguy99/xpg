from dataclasses import dataclass

from pyglm.glm import ivec2, ivec3, ivec4, vec2, vec3, vec4
from pyxpg import imgui

from ambra.config import Config, GuiConfig
from ambra.viewer import Viewer
from ambra.widgets import Annotated, CustomUI, CustomUpdate, Editor, Link, SliderFloat, SliderInt


@dataclass
class LData:
    linked_checkbox: bool


ld = LData(False)


@dataclass
class Data:
    checkbox_first: bool
    checkbox: Annotated[bool, Link(ld, "linked_checkbox")]
    floating: Annotated[float, CustomUpdate(print)]
    integer: Annotated[int, CustomUI(lambda d, t, n, v, m: imgui.checkbox(f"{n}##{id(d)}", bool(v)))]
    int_slider: Annotated[int, SliderInt(v_min=3, v_max=8)]
    float_slider: Annotated[float, SliderFloat(v_min=3.0, v_max=8.0)]
    vec2_item: vec2 = vec2()
    vec3_item: vec3 = vec3()
    vec4_item: vec4 = vec4()
    ivec2_item: ivec2 = ivec2()
    ivec3_item: Annotated[ivec3, SliderInt(3, 5)] = ivec3()
    ivec4_item: ivec4 = ivec4()


d = Data(False, True, 3.2, 4, 6, 5.3, vec2(12, 34))

viewer = Viewer(
    config=Config(
        gui=GuiConfig(
            stats=True,
        ),
    ),
)

viewer.scene.widgets.append(
    Editor(
        "Data editor",
        {
            "Data 1": d,
            "Data 2": ld,
        },
    )
)

viewer.run()
