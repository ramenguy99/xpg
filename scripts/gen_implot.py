from dataclasses import dataclass, field as dataclass_field
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from enum import Flag, auto
import io
import sys

DEFINES = {}

OUT = True
PRINT = False

module = "mod_implot"

data = json.load(open(sys.argv[1], "r"))
if OUT:
    out_file = open(os.path.join(os.path.dirname(__file__), "..", "src", "python", "generated_implot.inc"), "w", newline="\n")

pascal_re = re.compile(r'(?<!^)(?=[A-Z])')

def subst_acronyms(name: str) -> str:
    name = name.replace("HSV", "Hsv")
    name = name.replace("RGB", "Rgb")
    name = name.replace("IO", "Io")
    name = name.replace("ID", "Id")
    return name

def pascal_to_scream_case(name: str):
    return pascal_re.sub('_', subst_acronyms(name)).upper()

def pascal_to_snake_case(name: str):
    return pascal_re.sub('_', subst_acronyms(name)).lower()

def out(*args, **kwargs):
    if PRINT:
        print(*args, **kwargs)
    if OUT:
        print(*args, **kwargs, file=out_file)

enum_typedefs = set()
colors = []

# Enums
for enum in data["enums"]:
    flags: str = enum["is_flags_enum"]
    name: str = enum["name"]
    elems: str = enum["elements"]
    orig: str = enum["original_fully_qualified_name"]

    base_class = "IntFlag" if flags else "IntEnum"

    # Trim prefix
    enum_name = name
    if enum_name.startswith("ImGui"):
        enum_name = enum_name[5:]
    elif enum_name.startswith("ImPlot") and name != "ImPlotFlags_":
        enum_name = enum_name[6:]
    elif enum_name.startswith("ImDraw") or enum_name.startswith("ImFont") or enum_name.startswith("ImTexture") or enum_name.startswith("ImAxis") or enum_name.startswith("ImPlot"):
        enum_name = enum_name[2:]
    else:
        assert False, name

    if enum_name.endswith("_"):
        enum_name = enum_name[:-1]
        enum_typedefs.add(name[:-1])

    if enum_name == "Col":
        python_enum_name = "Color"
    else:
        python_enum_name = enum_name

    extra_flag = ""
    if flags:
        extra_flag = ", nb::is_flag()"
    out(f'nb::enum_<{orig}>({module}, "{python_enum_name}", nb::is_arithmetic() {extra_flag})')

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

        # Adapt element name
        if name.startswith("ImGui"):
            # Skip mods, we already have these as keys. Never used Shortcut and Mask before
            if name.startswith("ImGuiMod_"): continue
            elem_name = name[5 + len(enum_name) + 1:]
        elif name.startswith("ImPlot") and enum_name != "PlotFlags":
            elem_name = name[6 + len(enum_name) + 1:]
        elif name.startswith("ImDraw") or name.startswith("ImFont") or name.startswith("ImTexture") or name.startswith("ImAxis") or name.startswith("ImPlot"):
            assert name.startswith("Im" + enum_name + "_"), f"{enum_name} {name}"
            elem_name = name[2 + len(enum_name) + 1:]
        else:
            assert False, name

        if enum_name == "Col":
            colors.append(name)

        # Handle number only names after adapting
        if elem_name in {str(i) for i in range(10)}:
            elem_name = "Key_" + elem_name
        elem_name = pascal_to_scream_case(elem_name)

        # Outut element
        out(" " * 4 + f'.value("{elem_name}", {name})')
    out(";")
    out("")
    out("")

# Ignore typedefs for now, no real use yet. Maybe once function args type are missing?
for t in data["typedefs"]:
    name = t["name"]
    typ = t["type"]
    # print(name, typ)

# Will need to handle structs later
enabled_structs = {
    "ImGuiIO",
    "ImPlotStyle",
    "ImPlotInputMap",
}

for struct in data["structs"]:
    if struct["name"] in enabled_structs:
        struct_name = struct["name"]
        if struct_name.startswith("ImGui"):
            python_struct_name = struct_name[5:]
        elif struct_name.startswith("ImPlot"):
            python_struct_name = struct_name[6:]
        else:
            python_struct_name = struct_name

        out(f'nb::class_<{struct_name}>({module}, "{python_struct_name}")')
        for field in struct["fields"]:
            field_name = field["name"]
            field_type = field["type"]
            if struct_name == "ImPlotStyle" and field_name.lower() == "colors":
                for c in colors:
                    out(" " * 4 + f'.def_prop_rw("color_{pascal_to_snake_case(c[10:])}", [](ImPlotStyle& s) -> ImVec4 {{ return s.{field_name}[{c}]; }}, [](ImPlotStyle& s, ImVec4 value){{ s.{field_name}[{c}] = value; }})')

            if field["is_array"]:
                continue
            # print(field_type["description"])
            if not field_type["description"]["kind"] == "Builtin" and not field_type["description"]["kind"] == "User":
                continue
            if field_name == "InputQueueCharacters":
                continue
            out(" " * 4 + f'.def_rw("{pascal_to_snake_case(field_name)}", &{struct_name}::{field_name})')
        out(";")
        out("")

class TypeFlag(Flag):
    IS_OUTPUT = auto()
    IS_USER_TYPE = auto()
    IS_PTR = auto()
    IS_REF = auto()
    IS_PTR_FROM_OPTION = auto()
    IS_REMOVED_PTR = auto()
    IS_OPTIONAL = auto()
    IS_BUILTIN = auto()
    IS_ENUM_TYPEDEF = auto()

@dataclass
class ArrayTypeInfo:
    typ: str
    count: int

@dataclass
class TypeInfo:
    name: str
    cpp_name: str
    flags: TypeFlag
    array: Optional[ArrayTypeInfo]

@dataclass
class Arg:
    name: str
    type: TypeInfo
    default: Optional[str]

def read_type(typ: dict) -> TypeInfo:
    type_decl: str = typ["declaration"]
    type_desc: str = typ["description"]

    flags = TypeFlag(0)
    array = None
    if type_decl == "ImU32":
        name = "Color"
    elif type_decl == "double":
        flags |= TypeFlag.IS_BUILTIN
        name = "float"
    elif type_decl == "const void*":
        name = "memoryview"
    elif type_decl == "const char*":
        name = "str"
    # elif type_decl == "size_t":
    #     flags |= TypeFlag.IS_BUILTIN
    #     name = "int"
    elif type_desc["kind"] == "Pointer":
        inner = type_desc["inner_type"]
        flags |= TypeFlag.IS_PTR
        if "is_reference" in type_desc:
            flags |= TypeFlag.IS_REF
            type_decl = type_decl.replace("*", "&")
        if inner["kind"] == "Builtin":
            name = inner["builtin_type"]
            flags |= TypeFlag.IS_BUILTIN
            flags |= TypeFlag.IS_OUTPUT
            flags |= TypeFlag.IS_REMOVED_PTR
        elif inner["name"] == "size_t":
            name = "int"
            flags |= TypeFlag.IS_BUILTIN
            flags |= TypeFlag.IS_OUTPUT
        elif inner["name"] == "ImU32":
            name = "unsigned int"
            flags |= TypeFlag.IS_BUILTIN
            flags |= TypeFlag.IS_OUTPUT
        elif "storage_classes" in inner and "const" in inner["storage_classes"]:
            name = inner["name"]
            flags |= TypeFlag.IS_USER_TYPE
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
                "ImGuiStyle",
                "ImGuiMultiSelectIO",
                "ImFontBaked",
                "ImPlotSpec",
                # "ImDrawListSplitter",
                # "ImGuiIO",
                # "ImGuiInputTextCallbackData",
                # "ImVector_ImGuiTextFilter_ImGuiTextRange",
                # "ImGuiTextFilter",
                # "ImGuiTextBuffer",
            }, (inner, typ)
            name = inner["name"]
            flags |= TypeFlag.IS_USER_TYPE
        else:
            assert False, f"{func_name} {typ}"
    elif type_desc["kind"] == "Array":
        inner = type_desc["inner_type"]
        if inner["kind"] == "Pointer" and inner["inner_type"]["kind"] == "Builtin" and inner["inner_type"]["builtin_type"] == "char":
            name = "Tuple[str, ...]"
            array = ArrayTypeInfo("nb::str", -1)
        else:
            assert inner["kind"] == "Builtin", inner
            builtin = inner["builtin_type"]
            count = int(type_desc["bounds"])
            name = "Tuple[" + ", ".join([builtin] * count) + "]"
            flags |= TypeFlag.IS_OUTPUT
            array = ArrayTypeInfo(builtin, count)
    else:
        name = type_decl
        flags |= TypeFlag.IS_USER_TYPE
        if name in enum_typedefs:
            flags |= TypeFlag.IS_ENUM_TYPEDEF

    if name.startswith("ImGui"):
        name = name[5:]
    elif name.startswith("ImPlot"):
        name = name[6:]
    elif name.startswith("Im"):
        name = name[2:]

    return TypeInfo(name, type_decl, flags, array)

all_funcs = set()
overloads = []

spec_started = False
spec_ended = False

# Types:
#-heatmap
#   - 2d values, [rows, cols, sclaemin, scalemax, format, bounds min, boundsmax]

@dataclass
class PlotOverload:
    arrays: List[str]
    scalars: List[Tuple[str, Any]] = dataclass_field(default_factory=list)

@dataclass
class PlotSpecField:
    name: str
    type: str
    default_value: str
    arr_type: Optional[str] = None

@dataclass
class PlotInfo:
    overloads: List[PlotOverload] = dataclass_field(default_factory=list)
    specs: List[PlotSpecField] = dataclass_field(default_factory=list)

Spec_LineColor = PlotSpecField("LineColor", "ImVec4", "IMPLOT_AUTO_COL", "ImU32")
Spec_LineWeight = PlotSpecField("LineWeight", "float", 1.0)
Spec_FillColor = PlotSpecField("FillColor", "ImVec4", "IMPLOT_AUTO_COL", "ImU32")
Spec_FillAlpha = PlotSpecField("FillAlpha", "float", 1.0)
Spec_Marker = PlotSpecField("Marker", "ImPlotMarker", "ImPlotMarker_None")
Spec_MarkerSize = PlotSpecField("MarkerSize", "float", 4, "float")
Spec_MarkerLineColor = PlotSpecField("MarkerLineColor", "ImVec4", "IMPLOT_AUTO_COL", "ImU32")
Spec_MarkerFillColor = PlotSpecField("MarkerFillColor", "ImVec4", "IMPLOT_AUTO_COL", "ImU32")
Spec_Size = PlotSpecField("Size", "float", 4)

plot_functions: Dict[str, PlotInfo] = {
    "PlotLine": PlotInfo(
        overloads=[
            PlotOverload(["values"], scalars=[("xscale", 1), ("xstart", "0")]),
            PlotOverload(["xs", "ys"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillAlpha, Spec_Marker, Spec_MarkerSize, Spec_MarkerLineColor, Spec_MarkerFillColor],
    ),
    "PlotLineInt": None,
    "PlotScatter": PlotInfo(
        overloads=[
            PlotOverload(["values"], scalars=[("xscale", 1), ("xstart", "0")]),
            PlotOverload(["xs", "ys"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillAlpha, Spec_Marker, Spec_MarkerSize, Spec_MarkerLineColor, Spec_MarkerFillColor]
    ),
    "PlotScatterInt": None,
    "PlotStairs": PlotInfo(
        overloads=[
            PlotOverload(["values"], scalars=[("xscale", 1), ("xstart", "0")]),
            PlotOverload(["xs", "ys"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillAlpha, Spec_Marker, Spec_MarkerSize, Spec_MarkerLineColor, Spec_MarkerFillColor],
    ),
    "PlotStairsInt": None,
    "PlotShaded": PlotInfo(
        overloads=[
            PlotOverload(["values"], scalars=[("yref", 0), ("xscale", 1), ("xstart", "0")]),
            PlotOverload(["xs", "ys"], scalars=[("yref", 0)]),
            PlotOverload(["xs", "ys1", "ys2"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha, Spec_Marker, Spec_MarkerSize, Spec_MarkerLineColor, Spec_MarkerFillColor],
    ),
    "PlotShadedInt": None,
    "PlotBars": PlotInfo(
        overloads = [
            PlotOverload(["values"], scalars=[("barsize", 0.67), ("shift", 0)]),
            PlotOverload(["xs", "ys"], scalars=[("barsize", None)]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha],
    ),
    "PlotBarGroups": PlotInfo(),
    "PlotErrorBars": PlotInfo(
        overloads = [
            PlotOverload(["xs", "ys", "err"]),
            PlotOverload(["xs", "ys", "neg", "pos"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha, Spec_Size],
    ),
    "PlotStems": PlotInfo(
        overloads=[
            PlotOverload(["values"], scalars=[("ref", 0), ("xscale", 1), ("xstart", "0")]),
            PlotOverload(["xs", "ys"], scalars=[("ref", 0)]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillAlpha, Spec_Marker, Spec_MarkerSize, Spec_MarkerLineColor, Spec_MarkerFillColor],
    ),
    "PlotStemsInt": None,
    "PlotInfLines": PlotInfo(
        overloads=[
            PlotOverload(["values"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight],
    ),
    "PlotPieChart": PlotInfo(),
    "PlotPieChartImPlotFormatter": None,
    "PlotHeatmap": PlotInfo(),
    "PlotHistogram": PlotInfo(
        overloads=[
            PlotOverload(["values"], scalars=[("bins", "ImPlotBin_Sturges", "ImPlotBin_"), ("bar_scale", 1), ("min_value", 0), ("max_value", 0)]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha],
    ),
    "PlotHistogram2D": PlotInfo(
        overloads=[
            PlotOverload(["xs", "ys"], scalars=[("x_bins", "ImPlotBin_Sturges", "ImPlotBin_"), ("y_bins", "ImPlotBin_Sturges", "ImPlotBin_"), ("min_x", 0), ("max_x", 0), ("min_y", 0), ("max_y", 0)]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha],
    ),
    "PlotDigital": PlotInfo(
        overloads=[
            PlotOverload(["xs", "ys"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha, Spec_Size],
    ),
    "PlotImage": None,
    "PlotBubblesInt": None,
    "PlotBubbles": PlotInfo(
        overloads=[
            PlotOverload(["values", "szs"], scalars=[("xscale", 1), ("xstart", "0")]),
            PlotOverload(["xs", "ys", "szs"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha]
    ),
    "PlotPolygon": PlotInfo(
        overloads=[
            PlotOverload(["xs", "ys"]),
        ],
        specs=[Spec_LineColor, Spec_LineWeight, Spec_FillColor, Spec_FillAlpha, Spec_Marker, Spec_MarkerSize, Spec_MarkerLineColor, Spec_MarkerFillColor]
    ),
    "PlotText": None,
    "PlotDummy": None,
}

# Functions
for f in data["functions"]:
    func_name: str = f["name"]
    # print(f["original_fully_qualified_name"])
    # module = func_name.split("_")[0]

    if f["is_imstr_helper"]:
        continue
    if f["is_manual_helper"]:
        continue
    if f["is_unformatted_helper"]:
        continue
    if f["is_default_argument_helper"]:
        continue

    # print(func_name)

    # Skip specific functions we dont need
    if func_name in {
        # Context stuff (don't need)
        "ImGui_CreateContext",
        "ImGui_DestroyContext",
        "ImGui_GetCurrentContext",
        "ImGui_SetCurrentContext",
        "ImPlotCreateContext",
        "ImPlotDestroyContext",
        "ImPlotGetCurrentContext",
        "ImPlotSetCurrentContext",
        "ImPlotSetImGuiContext",
        # Frame handling (don't need)
        "ImGui_NewFrame",
        "ImGui_Render",
        "ImGui_EndFrame",
        "ImGui_GetDrawData",
        # ID (don't need?)
        "ImGui_GetID",
        "ImGui_GetIDStr",
        "ImGui_BeginChildID",
        "ImGui_SetNextWindowViewport",
        "ImGui_OpenPopupID",
        "ImGui_TableSetupColumn",
        # Float is double
        "ImGui_InputDouble",
        # Memory stuff (don't need)
        "ImGui_GetAllocatorFunctions",
        "ImGui_SetAllocatorFunctions",
        "ImGui_MemAlloc",
        "ImGui_MemFree",
        # Done manually"
        "ImGui_GetIO",
        "ImPlotGetStyle",
        "ImPlotGetInputMap",
        "ImPlotMapInputDefault",
        "ImPlotMapInputReverse",

        # TODO:
        # Font
        "ImGui_GetFont",
        "ImGui_GetFontBaked",
        # InputText
        "ImGui_InputText",          # Requires special handling instead of callback
        "ImGui_InputTextMultiline", # Requires special handling instead of callback
        "ImGui_InputTextWithHint",  # Requires special handling instead of callback
        # Plot
        "ImGui_PlotLines",
        "ImGui_PlotHistogram",
        # .ini
        "ImGui_LoadIniSettingsFromMemory",
        "ImGui_SaveIniSettingsToMemory",
        # Weird API
        "ImGui_GetStyleColorVec4",
        "ImGui_ListBox",
        "ImGui_CalcListClipping",
        "ImGui_SetNextWindowSizeConstraints",
        "ImGui_TableGetSortSpecs",
        "ImGui_IsMousePosValid",
        "ImGui_ColorPicker4",
        "ImGui_ColorConvertHSVtoRGB",
        "ImGui_TextUnformatted",               # Substring of text, no need for this
        "ImGui_GetDrawListSharedData",         # Internal only struct def

        # ImPlot
        # Multiple return values
        "ImPlotSetupAxisLinks",
        "ImPlotSetNextAxisLinks",
        "ImPlotSetupAxisFormatImPlotFormatter",
        "ImPlotDragPoint",
        "ImPlotDragLineX",
        "ImPlotDragLineY",
        "ImPlotDragRect",
        "ImPlotColormapSlider",

        # Array input
        "ImPlotSetupAxisTicks",
        "ImPlotSetupAxisTicksDouble",

        # Function pointer
        "ImPlotSetupAxisScaleImPlotTransform",

        # Demo
        "ImPlotShowDemoWindow",
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
    if func_name.endswith("G"):
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

    # Skip ListBox for now, callback-based API needs manual wrappers likely
    if "ListBox" in func_name:
        continue

    # Check overloads
    cpp_name = f["original_fully_qualified_name"]
    if cpp_name in all_funcs:
        overloads.append(f)
    else:
        all_funcs.add(cpp_name)

    if not func_name.startswith("ImGui_"):
        # Skip unneeded implot struct methods
        if func_name.split("_")[0] in {
            "ImPlotRect", "ImPlotRange", "ImPlotStyle", "ImPlotInputMap",
        }:
            continue
        elif func_name.startswith("ImPlot"):
            func_name = func_name[6:]
        # Handle class methods, currently only ImDrawList is supported
        elif func_name.startswith("ImDrawList_"):
            pass
        else:
            continue

    py_func_name = pascal_to_snake_case(func_name)

    args: List[Arg] = []

    additional_ret = None
    additional_ret_type = None
    additional_ret_name = None

    for a in f["arguments"]:
        arg_name = a["name"]
        if arg_name == "in":
            arg_name = "value"

        if func_name == "BeginSubplots" and (arg_name == "row_ratios" or arg_name == "col_ratios"):
            continue

        # Skip arguments on implot style color functions (they have defaults)
        if func_name.startswith("StyleColors") or func_name.startswith("ShowStyleEditor"):
            continue

        # Skip varargs, we expect those functions to preformat their strings
        if a["is_varargs"]:
            continue

        if "type" not in a:
            assert False, a

        typ = read_type(a["type"])

        if typ.flags & TypeFlag.IS_OUTPUT:
            assert additional_ret == None, f"{cpp_name}: {additional_ret}"
            additional_ret = typ.name
            additional_ret_name = arg_name
            additional_ret_type = typ

        default_value = None
        if "default_value" in a:
            default: str = a["default_value"]
            if default == "NULL":
                default_value = "None"
                typ.name = f"Optional[{typ.name}]"
                typ.flags |= TypeFlag.IS_OPTIONAL
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
        args.append(Arg(arg_name, typ, default_value))

    args_str: List[str] = []
    for arg in args:
        arg_str = f'{arg.name}: {arg.type.name}'
        if arg.default is not None:
            arg_str += f"={arg.default}"
        args_str.append(arg_str)


    # Python fuc
    ret = f["return_type"]
    ret_type = read_type(ret)
    ret_str = ""

    has_ret: bool = True
    has_add_ret: bool = False

    if additional_ret is not None:
        has_add_ret = True
        if ret["declaration"] == "void":
            has_ret = False
            ret_str = additional_ret
        else:
            ret_str = f"Tuple[{ret_type.name}, {additional_ret}]"
    else:
        if ret["declaration"] == "void":
            has_ret = True
        else:
            ret_str = ret_type.name

    # CPP func
    def type_str_to_cpp(typ: str, ret: bool=False) -> str:
        if typ.startswith("Optional["):
            assert typ.endswith("]"), typ
            name = type_str_to_cpp(typ[9:-1])
            return f"std::optional<{name}>"
        elif typ.startswith("Tuple["):
            if ret:
                return "nb::tuple"
            else:
                assert typ.endswith("]"), typ
                return "std::tuple<" + ", ".join([type_str_to_cpp(t) for t in typ[6:-1].split(", ")]) + ">"
        elif typ == "memoryview":
            return "std::vector<uint8_t>"
        elif typ == "str":
            return "const char*"
        else:
            return typ

    def type_to_cpp(typ: TypeInfo, ret: bool=False):
        if typ.array is not None:
            return f"std::array<{type_str_to_cpp(typ.array.typ)}, {typ.array.count}>"
        elif typ.flags & TypeFlag.IS_ENUM_TYPEDEF:
            return typ.cpp_name + "_"
        elif typ.flags & TypeFlag.IS_USER_TYPE:
            if typ.name.startswith("Optional["):
                if typ.flags & TypeFlag.IS_PTR:
                    return f"std::optional<{typ.cpp_name[:-1]}*>"
                else:
                    return f"std::optional<{typ.cpp_name}>"
            else:
                if typ.cpp_name == "ImDrawList*" or typ.cpp_name == "const ImDrawList*":
                    return f"{typ.name}*"
                elif typ.cpp_name == "ImFont*":
                    return f"{typ.name}*"
                elif typ.cpp_name == "ImTextureRef":
                    return "const Texture&"
                else:
                    return typ.cpp_name
        else:
            return type_str_to_cpp(typ.name)

    if func_name in plot_functions.keys():
        # Skip int and formatter overloads
        if func_name.endswith("Int") or func_name.endswith("Formatter"):
            continue

        info = plot_functions[func_name]
        if info is None:
            continue

        # def out_check_1dim(name: str):
        #     out(" " * 8 + f'if ({name}.ndim() != 1) {{ nb::raise("Invalid array shape for parameter \\"{name}\\". Expected 1 dimension, got %zu", {name}.ndim()); }}')

        def out_count_stride_dtype(name: str):
            out(" " * 8 + f"size_t stride = {name}.stride(0);")
            out(" " * 8 + f"size_t count = {name}.shape(0);")
            out(" " * 8 + f"nb::dlpack::dtype dtype = {name}.dtype();")

        def out_check_count_stride_dtype(first: str, name: str):
            out(" " * 8 + f'if (dtype != {name}.dtype()) {{ nb::raise("{first} and {name} array must have the same dtype ({first} = %s%u, {name} = %s%u)", dtype_code_to_str(dtype.bits), dtype.bits, dtype_code_to_str({name}.dtype().code), {name}.dtype().bits); }}')
            out(" " * 8 + f'if (count != {name}.shape(0)) {{ nb::raise("{first} and {name} array must have the same count ({first} = %zu, {name} = %zu)", count, {name}.shape(0)); }}')
            out(" " * 8 + f'if (stride != {name}.stride(0)) {{ nb::raise("{first} and {name} array must have the same stride ({first} = %zu, {name} = %zu)", stride, {name}.stride(0)); }}')

        if func_name.endswith("2D"):
            flag_type_name = f"Im{func_name[:-2]}Flags" 
        else:
            flag_type_name = f"Im{func_name}Flags" 

        for ov in info.overloads:
            out(f"{module}.def(\"{py_func_name}\",")
            nb_args = [ "nb::str label_id" ]
            for a in ov.arrays:
                nb_args.append(f"nb::ndarray<nb::ndim<1>, nb::any_contig, nb::device::cpu> {a}")
            for x in ov.scalars:
                t = "double"
                if len(x) > 2:
                    t = x[2]
                nb_args.append(f"{t} {x[0]}")

            # Spec args
            for spec in info.specs:
                nb_args.append(f"{spec.type} {spec.name}")
                if spec.arr_type is not None:
                    nb_args.append(f"std::optional<nb::ndarray<nb::ndim<1>, nb::c_contig, {spec.arr_type}, nb::device::cpu>> {spec.name}s")
            nb_args.append(f"{flag_type_name}_ flags")

            out(" " * 4 + "[] (" + ", ".join(nb_args) + ") {")
            # for a in ov.arrays:
            #     out_check_1dim(a)
            out_count_stride_dtype(ov.arrays[0])
            for a in ov.arrays[1:]:
                out_check_count_stride_dtype(ov.arrays[0], a)

            out(" " * 8 + "ImPlotSpec spec;")
            out(" " * 8 + "spec.Flags = flags;")
            for spec in info.specs:
                out(" " * 8 + f"spec.{spec.name} = {spec.name};")
                if spec.arr_type is not None:
                    out(" " * 8 + f"if ({spec.name}s.has_value()) {{")
                    out(" " * 12 + f'if (count != {spec.name}s.value().shape(0)) {{ nb::raise("{ov.arrays[0]} and {pascal_to_snake_case(spec.name)}s array must have the same count ({ov.arrays[0]} = %zu, {spec.name}s = %zu)", count, {spec.name}s.value().shape(0)); }}')
                    out(" " * 12 + f"spec.{spec.name}s = {spec.name}s.value().data();")
                    out(" " * 8 + "}")

            # Dispatch
            out(" " * 8 + "switch (dtype.code) {")

            for t, sizes in [ ("Int", (8, 16, 32, 64)), ("UInt", [8, 16, 32, 64]), ("Float", [32, 64]) ]:
                out(" " * 12 + f"case (int)nb::dlpack::dtype_code::{t}: {{")
                out(" " * 16 + "switch (dtype.bits) {")
                for s in sizes:
                    if t == "Float":
                        im_t = "float" if s == 32 else "double"
                    else:
                        im_t = ("ImS" if t == "Int" else "ImU") + str(s)
                    disp_args = [ "label_id.c_str()" ]
                    for a in ov.arrays:
                        disp_args.append(f"({im_t}*){a}.data()")
                    disp_args.append("count")
                    for x in ov.scalars:
                        if x[0] == "min_value":
                            disp_args.append(f"ImPlotRange(min_value, max_value)")
                        elif x[0] == "min_x":
                            disp_args.append(f"ImPlotRect(min_x, max_x, min_y, max_y)")
                        elif x[0] == "max_value" or x[0] == "max_x" or x[0] == "min_y" or x[0] == "max_y":
                            continue
                        else:
                            disp_args.append(f"{x[0]}")
                    disp_args.append("spec")
                    out(" " * 20 + f"case {s:2d}: " + f"spec.Stride = stride * {s // 8}; ImPlot::{func_name}(" + ", ".join(disp_args) + "); break;")

                out(" " * 16 + "}")
                out(" " * 12 + "} break;")
            out(" " * 12 + f'default: {{ nb::raise("Invalid array dtype for parameter \\"{ov.arrays[0]}\\" with code %u and %u bits", dtype.code, dtype.bits); }} break;')
            out(" " * 8 + "}")

            params = ['nb::arg("label_id")']
            for a in ov.arrays:
                params.append(f'nb::arg("{a}")')
            for sc in ov.scalars:
                dv = ""
                if sc[1] is not None:
                    dv = f" = {sc[1]}"
                params.append(f'nb::arg("{sc[0]}"){dv}')
            for spec in info.specs:
                params.append(f'nb::arg("{pascal_to_snake_case(spec.name)}") = {spec.default_value}')
                if spec.arr_type is not None:
                    params.append(f'nb::arg("{pascal_to_snake_case(spec.name)}s") = nb::none()')
            params.append(f'nb::arg("flags") = {flag_type_name}_None')


            out(" " * 4 + "}, " + ", ".join(params) + ");")
            out("")
        continue

    # PlotSpec
    if py_func_name.startswith("spec__"):
        continue
        # For now spec will be fully manual
        if spec_started == False:
            spec_started = True
            out(f"""auto plotspec_class = nb::class_<PlotSpec>({module}, "PlotSpec",
    nb::intrusive_ptr<PlotSpec>([](PlotSpec *o, PyObject *po) noexcept {{ o->set_self_py(po); }}))
""")
        # Skip protected member functions
        if py_func_name.startswith("spec___"):
            continue

        out(" " * 4 + f".def(\"{py_func_name[6:]}\",")
    else:
        if spec_started and spec_ended == False:
            spec_ended = True
            out(";")
        out(f"{module}.def(\"{py_func_name}\",")


    out(" " * 4 + "[] (", end="")
    cpp_args_str: List[str] = []
    nb_args_str: List[str] = []
    for arg in args:
        if arg.name == "text_begin":
            arg.name = "text"
        if arg.name == "text_end":
            continue
        cpp_args_str.append(f"{type_to_cpp(arg.type)} {arg.name}")
        nb_str = f"nb::arg(\"{arg.name}\")"
        if arg.default:
            if arg.default == "None":
                default_value = "nb::none()"
            elif arg.default == "True":
                default_value = "true"
            elif arg.default == "False":
                default_value = "false"
            elif arg.default.startswith("("):
                assert arg.default.endswith(")")
                default_value = "nb::make_tuple" + arg.default
            else:
                default_value = arg.default

            nb_str += f" = {default_value}"
        nb_args_str.append(nb_str)
    if py_func_name.startswith("list__"):
        nb_args_str = nb_args_str[1:]
    out(", ".join(cpp_args_str), end="")

    if has_ret:
        if has_add_ret:
            out(f") -> std::tuple<{type_to_cpp(ret_type, True)}, {type_to_cpp(additional_ret_type)}> {{")
        else:
            out(f") -> {type_to_cpp(ret_type, True)} {{")
    else:
        if has_add_ret:
            out(f") -> {type_to_cpp(additional_ret_type, True)} {{")
        else:
            out(f") {{")


    # Output function call

    # Return value cases:
    # - Out parameter: [yes, no]
    # - Void return: [yes, no]
    #
    #  ret   out
    #  ---   ---
    #  yes | yes:  return make_tuple(call, out);
    #   no | yes:  call; return out;
    #  yes |  no:  return call;
    #   no |  no:  call;
    def out_call():
        global args
        if py_func_name.startswith("list__"):
            args = args[1:]
            func_call = "self->list->" + f["original_fully_qualified_name"]
        else:
            func_call = f["original_fully_qualified_name"]

        if ret_type.flags & TypeFlag.IS_ENUM_TYPEDEF:
            func_call = f"({ret_type.cpp_name}_){func_call}"

        arg_call_names = []
        for arg in args:
            if arg.name == "text_end":
                name = "nullptr"
            elif arg.type.array is not None:
                name = f"{arg.name}.data()"
            elif arg.name == "fmt" and py_func_name != "setup_axis_format":
                name = r'"%s", fmt'
            elif arg.type.name == "memoryview":
                name = f"{arg.name}.data()"
            elif arg.type.name == "TextureRef":
                name = f"{arg.name}.tex_ref"
            elif arg.type.name == "Font":
                name = f"{arg.name}->font"
            elif arg.type.flags & TypeFlag.IS_PTR and not arg.type.flags & TypeFlag.IS_REF:
                if arg.type.flags & TypeFlag.IS_OPTIONAL:
                    # name = f"&{arg.name}.value()"
                    if arg.type.flags & TypeFlag.IS_USER_TYPE:
                        name = f"{arg.name}.has_value() ? {arg.name}.value() : NULL"
                    else:
                        name = f"{arg.name}.has_value() ? &{arg.name}.value() : NULL"
                else:
                    if arg.type.flags & TypeFlag.IS_REMOVED_PTR:
                        name = f"&{arg.name}"
                    else:
                        name = f"{arg.name}"
            elif arg.type.flags & TypeFlag.IS_OPTIONAL:
                name = f"{arg.name}.has_value() ? {arg.name}.value() : NULL"
            else:
                name = arg.name
            arg_call_names.append(name)
        out(func_call + "(" + ", ".join(arg_call_names) + ")", end="")

    def out_add_ret():
        out(f"{additional_ret_name}", end="")


    if not has_ret:
        out(" " * 8, end="")
        out_call()
        out(";")
        if has_add_ret:
            out(" " * 8 + "return ", end="")
            out_add_ret()
            out(";")
    else:
        if has_add_ret:
            # Can't do inline because call needs to happen before ret, arg eval order to tuple is unspecified
            out(" " * 8 + "auto _call = ", end="")
            out_call()
            out(";")
            out(" " * 8 + "auto _ret = ", end="")
            out_add_ret()
            out(";")
            out(" " * 8 + "return std::make_tuple(_call, _ret);")
        else:
            out(" " * 8 + "return ", end="")
            # if ret_type.flags & TypeFlag.IS_USER_TYPE and ret_type.flags & TypeFlag.IS_PTR:
            if ret_type.cpp_name == "ImDrawList*":
                out(f"new {ret_type.name}(", end="")
                out_call()
                out(f")", end="")
            else:
                out_call()
            out(";")

    out(" " * 4 + f"}}", end="")
    if nb_args_str:
        out(", ", end="")
    out(", ".join(nb_args_str), end="")
    if  py_func_name.startswith("list__"):
        out(")")
    else:
        out(f");\n")
