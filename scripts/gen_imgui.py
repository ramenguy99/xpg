from dataclasses import dataclass
import json
import os
import re
from typing import Union

# TODO:
# [x] How to handle Refs (pyimgui used to do this with pass in and multiple return values, but I never really liked it that much, but probably no other option)
# [x] Default values
# [ ] Generate nanobind forwarders of the funcs
# [ ] Wrappers for ImGui objects (e.g. Context, FontAtlas, Style) -> see how to share those with c++
# [ ] Test this!

DEFINES = {
    # "IMGUI_DISABLE_OBSOLETE_FUNCTIONS",
    # "IMGUI_DISABLE_OBSOLETE_KEYIO",
}

OUT = True

data = json.load(open(os.path.join(os.path.dirname(__file__), "..", "ext", "dear_bindings", "xpg_imgui.json"), "r"))

if OUT:
    out_file = open(os.path.join(os.path.dirname(__file__), "pyimgui.py"), "w")

pascal_re = re.compile(r'(?<!^)(?=[A-Z])')
def pascal_to_scream_case(name: str):
    name = name.replace("IO", "Io")
    name = name.replace("ID", "Id")
    return pascal_re.sub('_', name).upper()
def pascal_to_snake_case(name: str):
    name = name.replace("HSV", "Hsv")
    name = name.replace("RGB", "Rgb")
    name = name.replace("IO", "Io")
    name = name.replace("ID", "Id")
    return pascal_re.sub('_', name).lower()

def out(*args, **kwargs):
    if OUT:
        print(*args, **kwargs, file=out_file)

out("from enum import IntEnum, IntFlag")
out("from typing import Tuple")
out("")
out("class ID(object): ...")
out("class Viewport(object): ...")
out("class ImDrawList(object): ...")
out("class ImFont(object): ...")
out("class WindowClass(object): ...")
out("class Payload(object): ...")
out("class Storage(object): ...")
out("class ImTextureID(object): ...")
out("class ImDrawListSharedData(object): ...")
out("class PlatformIO(object): ...")
out("")
# out("ImVec2: type = Tuple[float, float]")
# out("ImVec4: type = Tuple[float, float, float, float]")
# out("")

for enum in data["enums"]:
    flags: str = enum["is_flags_enum"]
    name: str = enum["name"]
    elems: str = enum["elements"]

    base_class = "IntFlag" if flags else "IntEnum"

    enum_name = name
    if name.startswith("ImGui"):
        enum_name = enum_name[5:]
    elif name.startswith("ImDraw") or name.startswith("ImFont"):
        enum_name = enum_name[2:]
    else:
        assert False, name

    if name.endswith("_"):
        enum_name = enum_name[:-1]
    out(f"class {enum_name}({base_class}):")

    for elem in elems:
        name = elem["name"]
        value = elem["value"]

        # Skip count elements
        if elem["is_count"]:
            continue

        # Skip if conditional not met
        if "conditionals" in elem:
            conds = elem["conditionals"]
            skip = False
            for c in conds:
                if c["condition"] == "ifndef":
                    if c["expression"] in DEFINES:
                        skip = True
                        break
                elif c["condition"] == "ifdef":
                    if c["expression"] not in DEFINES:
                        skip = True
                        break
                    pass
                else:
                    assert False
            if skip:
                continue

        if name.startswith("ImGui"):
            if enum_name == "Key":
                assert name.startswith("ImGui" + enum_name + "_") or name.startswith("ImGuiMod_"), f"{enum_name} {name}"
            else:
                assert name.startswith("ImGui" + enum_name + "_"), f"{enum_name} {name}"

            # Skip mods, we already have these as keys. Never used Shortcut and Mask before
            if name.startswith("ImGuiMod_"): continue

            elem_name = name[5 + len(enum_name) + 1:]
        elif name.startswith("ImDraw") or name.startswith("ImFont"):
            assert name.startswith("Im" + enum_name + "_"), f"{enum_name} {name}"
            elem_name = name[2 + len(enum_name) + 1:]
        else:
            assert False, name

        if elem_name in {str(i) for i in range(10)}:
            elem_name = "Key_" + elem_name
        elem_name = pascal_to_scream_case(elem_name)
        if flags:
            out(4 * " " + f"{elem_name} = 0x{value:x}")
        else:
            out(4 * " " + f"{elem_name} = {value}")
    out("")

    #     print(e["name"], e["original_fully_qualified_name"])


# Ignore typedefs for now, no real use yet. Maybe once function args type are missing?
for t in data["typedefs"]:
    name = t["name"]
    typ = t["type"]
    # print(name, typ)

# Will need to handle structs later
for s in data["structs"]:
    pass
    # print(s["name"])

@dataclass
class Ptr():
    t: str

def type_to_str(typ: dict) -> Union[str, Ptr]:
    type_decl = typ["declaration"]
    type_desc = typ["description"]

    if type_decl == "ImVec2":
        return "Tuple[float, float]"
    if type_decl == "ImVec4":
        return "Tuple[float, float, float, float]"
    if type_decl == "ImU32":
        return "Tuple[int, int, int, int]"
    if type_decl == "double":
        return "float"
    if type_decl == "const void*":
        return "memoryview"
    if type_decl == "const char*":
        return "str"
    if type_decl == "size_t":
        return "int"

    if type_desc["kind"] == "Pointer":
        inner = type_desc["inner_type"]
        if inner["kind"] == "Builtin":
            return Ptr(inner["builtin_type"])
        if inner["name"] == "size_t":
            return Ptr("int")
        elif "name" in inner:
            assert inner["name"] in {
                "ImDrawList",
                "ImGuiViewport",
                "ImFont",
                "ImGuiWindowClass",
                "ImGuiPayload",
                "ImDrawListSharedData",
                "ImGuiStorage",
                "ImGuiPlatformIO",
            }, inner
            type_str: str = inner["name"]
        else:
            assert False, f"{func_name} {typ}"
    elif type_desc["kind"] == "Array":
        inner = type_desc["inner_type"]
        assert inner["kind"] == "Builtin", inner
        builtin = inner["builtin_type"]
        return Ptr("Tuple[" + ", ".join([builtin] * int(type_desc["bounds"])) + "]")
    else:
        type_str: str = type_decl

    if type_str.startswith("ImGui"):
        type_str = type_str[5:]

    return type_str

# Functions
for f in data["functions"]:
    func_name: str = f["name"]
    # print(f["original_fully_qualified_name"])
    module = func_name.split("_")[0]

    # if "original_class" in f:
    #     # print(f["original_class"], name, f["original_fully_qualified_name"])
    #     continue

    if f["is_imstr_helper"]:
        continue
    if f["is_manual_helper"]:
        continue
    if f["is_unformatted_helper"]:
        continue
    if f["is_default_argument_helper"]:
        continue

    if "original_class" in f:
        continue

    # Skip specific functions we dont need
    if func_name in {
        # Context stuff
        "ImGui_CreateContext",
        "ImGui_DestroyContext",
        "ImGui_GetCurrentContext",
        "ImGui_SetCurrentContext",

        # IO
        "ImGui_GetIO",

        # InputText
        "ImGui_InputText",          # Requires special handling instead of callback
        "ImGui_InputTextMultiline", # Requires special handling instead of callback
        "ImGui_InputTextWithHint",  # Requires special handling instead of callback

        # Style
        "ImGui_GetStyle",
        "ImGui_StyleColorsDark",
        "ImGui_StyleColorsLight",
        "ImGui_StyleColorsClassic",

        # Frame handling
        "ImGui_NewFrame",
        "ImGui_Render",
        "ImGui_EndFrame",
        "ImGui_GetDrawData",

        # Show stuff
        "ImGui_ShowDemoWindow",
        "ImGui_ShowMetricsWindow",
        "ImGui_ShowDebugLogWindow",
        "ImGui_ShowStackToolWindow",
        "ImGui_ShowAboutWindow",
        "ImGui_ShowStyleEditor",
        "ImGui_ShowStyleSelector",
        "ImGui_ShowFontSelector",
        "ImGui_ShowUserGuide",

        # Weird API
        "ImGui_GetStyleColorVec4",
        "ImGui_ListBox",
        "ImGui_CalcListClipping",
        "ImGui_SetNextWindowSizeConstraints",
        "ImGui_TableGetSortSpecs",
        "ImGui_IsMousePosValid",
        "ImGui_ColorPicker4",
        "ImGui_ColorConvertHSVtoRGB",

        # Plot (can be made more ergonomic manually)
        "ImGui_PlotLines",
        "ImGui_PlotHistogram",

        # ID
        "ImGui_GetID",
        "ImGui_GetIDStr",
        "ImGui_BeginChildID",
        "ImGui_SetNextWindowViewport",
        "ImGui_OpenPopupID",
        "ImGui_TableSetupColumn",

        # Float is double
        "ImGui_InputDouble",

        # Memory stuff
        "ImGui_GetAllocatorFunctions",
        "ImGui_SetAllocatorFunctions",
        "ImGui_MemAlloc",
        "ImGui_MemFree",
    }:
        continue

    # Skip callbacks for now, this will need to be handled properly later
    if func_name.endswith("Callback"):
        continue

    # Skip varargs versions of stuff
    if func_name.endswith("V"):
        continue
    if func_name.endswith("Ptr"):
        continue

    # Also skip combo and list stuff, not sure best way to deal with yet
    if "Combo" in func_name:
        continue

    # Skip scalars, adds too much complexity right now
    if "Scalar" in func_name:
        continue

    # Skip range2, adds too much complexity right now
    if "Range2" in func_name:
        continue

    # Skip platform stuff, adds too much complexity right now
    if "Platform" in func_name:
        continue

    assert func_name.startswith("ImGui_"), func_name
    out(f"def {pascal_to_snake_case(func_name[6:])}(", end="")
    args = []

    additional_ret = None
    for a in f["arguments"]:
        arg_name = a["name"]
        if arg_name == "in":
            arg_name = "value"

        # Skip varargs, we expect those functions to preformat their strings
        if a["is_varargs"]:
            continue

        if "type" not in a:
            assert False, a

        type_str = type_to_str(a["type"])

        if isinstance(type_str, Ptr):
            assert additional_ret == None, additional_ret
            type_str = type_str.t
            additional_ret = type_str


        default_value = ""
        if "default_value" in a:
            default: str = a["default_value"]
            if default == "NULL":
                default_value = "None"
            elif default.startswith("ImVec2") or default.startswith("ImVec4"):
                nums = default.split("(")[1][:-1]
                nums = nums.replace("f", "")
                nums = nums.replace("FLT_MIN", "1.175494351e-38")
                default_value = f"({nums})"
            elif default == "false":
                default_value = "False"
            elif default == "true":
                default_value = "True"
            elif default.endswith("f"):
                default_value = default[:-1]
            else:
                default_value = str(default)

        arg_str = f'{arg_name}: {type_str}'
        if default_value:
            arg_str += f"={default_value}"
        args.append(arg_str)

    out(", ".join(args), end="")
    out(f")", end="")
    ret = f["return_type"]
    ret_str = ""
    if additional_ret != None:
        if ret["declaration"] == "void":
            ret_str = additional_ret
        else:
            ret_str = f"Tuple[{type_to_str(ret)}, {additional_ret}]"
    else:
        if ret["declaration"] != "void":
            ret_str = type_to_str(ret)

    if ret_str:
        out(f" -> {ret_str}", end="")

    out(f": ...")


    # if f["is_unformatted_helper"]:
    #     continue

    # if module == "ImGui":
    #     print(name)

    # if name.startswith("ImGui_"):
    #     pass
    # elif name.startswith("ImVector_"):
    #     # Skip ImVector for now, likely want a separate class for this + conversion helper from tuple/list/numpy array
    #     pass
    # elif name.startswith("ImStr_"):
    #     # Skip for now, not sure if we need this, ideally same as above with builtin helpers
    #     pass
    # elif name.startswith("ImGuiStyle_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.style.func
    #     pass
    # elif name.startswith("ImGuiIO_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImGuiInputTextCallbackData_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImGuiPayload_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImGuiTextFilter_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImGuiStorage_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImGuiListClipper_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImTextBuffer_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImColor_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImGuiTextBuffer_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImDrawCmd_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImDrawListSplitter_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImDrawList_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImDrawData_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImFontGlyphRangesBuilder"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImFontAtlas_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImFontGlypRanges_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImFontAtlasCustomRect_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImFont_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # elif name.startswith("ImGuiViewport_"):
    #     # Skip for now, ideally this goes to separate module, e.g. imgui.io.func
    #     pass
    # else:
    #     assert False, name