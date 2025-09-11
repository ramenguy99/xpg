from dataclasses import dataclass
import json
import os
import re
from typing import List, Optional
from enum import Flag, auto
import io
import sys

DEFINES = {}

OUT = True
PRINT = True

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

    if name.startswith("ImGui"):
        name = name[5:]
    elif name.startswith("ImPlot"):
        name = name[6:]
    elif name.startswith("Im"):
        name = name[2:]

    return TypeInfo(name, type_decl, flags, array)

all_funcs = set()
overloads = []

drawlist_started = False
drawlist_ended = False

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

    print(func_name)
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

        # Plot functions
        "ImPlotPlotLine",
        "ImPlotPlotLineInt",
        "ImPlotPlotScatter",
        "ImPlotPlotScatterInt",
        "ImPlotPlotStairs",
        "ImPlotPlotStairsInt",
        "ImPlotPlotShaded",
        "ImPlotPlotShadedInt",
        "ImPlotPlotBars",
        "ImPlotPlotBarGroups",
        "ImPlotPlotErrorBars",
        "ImPlotPlotStems",
        "ImPlotPlotStemsInt",
        "ImPlotPlotInfLines",
        "ImPlotPlotPieChart",
        "ImPlotPlotPieChartImPlotFormatter",
        "ImPlotPlotHeatmap",
        "ImPlotPlotHistogram",
        "ImPlotPlotHistogram2D",
        "ImPlotPlotDigital",
        "ImPlotPlotImage",
        # "ImPlotPlotText",
        # "ImPlotPlotDummy",

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
        elif typ.flags & TypeFlag.IS_USER_TYPE:
            if typ.name.startswith("Optional["):
                if typ.flags & TypeFlag.IS_PTR:
                    print(typ.flags)
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

    if py_func_name.startswith("list__"):
        if drawlist_started == False:
            drawlist_started = True
            out(f"""auto drawlist_class = nb::class_<DrawList>({module}, "DrawList",
    nb::intrusive_ptr<DrawList>([](DrawList *o, PyObject *po) noexcept {{ o->set_self_py(po); }}))
""")
        # Skip protected member functions
        if py_func_name.startswith("list___"):
            continue

        out(" " * 4 + f".def(\"{py_func_name[6:]}\",")
    else:
        if drawlist_started and drawlist_ended == False:
            drawlist_ended = True
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
            out(f") -> nb::tuple {{")
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
            out(" " * 8 + "return nb::make_tuple(_call, _ret);")
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


if False:
    with open(os.path.join(os.path.dirname(__file__), "..", "src", "python", "module.cpp"), "r") as module_file:
        next_lines = []
        skip = False
        for l in module_file.readlines():
            if "!$" in l:
                skip = False

            if not skip:
                next_lines.append(l)

            if "$!" in l:
                skip = True
                next_lines.append(cpp_contents.getvalue())

    open(os.path.join(os.path.dirname(__file__), "..", "src", "python", "module.cpp"), "w").write("".join(next_lines))
