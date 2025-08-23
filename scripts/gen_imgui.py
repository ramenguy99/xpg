from dataclasses import dataclass
import json
import os
import re
from typing import List, Optional
from enum import Flag, auto
import io
import sys

# TODO:
# [x] How to handle Refs (pyimgui used to do this with pass in and multiple return values, but I never really liked it that much, but probably no other option)
# [x] Default values
# [x] Test this!
# [x] Generate nanobind forwarders of the funcs
#      [x] Nanobind enums
#      [x] Funcs -> can some wrappers be forwarded to imgui directly?
#                   careful about (basically wherever we do something special on the python function type):
#                       - return values with tuples
#                       - preformatted strings
#                       - optional / none to pointer (maybe nanobind handles this?)
#                       - structs?
# [ ] Wrappers for ImGui objects (e.g. Context, FontAtlas, Style) -> see how to share those with c++

# Notes:
# - store more info when converting types to python types for args (e.g. we would like to know original enum/flag stuff)
# - looks like we have some overloads with type suffix, check how to disable this during binding gen if possible, we would prefer overloading in python here.
# - also need to store metadata to know how to make the return value out of one of the params (e.g. pass by ref that param and then return it in tuple)
# - for types ideally we would work with a more AST like structure from the start, we now have too much string stuff going on

# Doable stuff
# [x] Fix opt return value to first
# [x] Tuple to float array to tuple back (don't think order guarantee holds for c++ tuple)
# [x] optional str_id to char* (just .has_value() also fix bool and ret type)
# [x] memoryview to bytes somehow (see if nanobind has something)
#
# [x] Skip remove references still in dear_bindings, currently no arg for that, can add and PR? Or use my own fork
# [ ] Generate structs (no fields at first, maybe some are useful, like IO)
# [ ] Implement some structs with method (DrawList comes to mind, rest never used (yet))
# [ ] ImVec2 to tuple conversion (or maybe just use imvec2 directly in signature and create helpers from python to it, to me makes more sense)
# [ ] ImU32 color too, probably conversion should also be on python side from float4 to u32 with some helper

DEFINES = {
    # "IMGUI_DISABLE_OBSOLETE_FUNCTIONS",
    # "IMGUI_DISABLE_OBSOLETE_KEYIO",
}

OUT_PYTHON = False
OUT = True
PRINT = False

data = json.load(open(sys.argv[1], "r"))

if OUT_PYTHON:
    out_file = open(os.path.join(os.path.dirname(__file__), "..", "src", "python", "pyxpg", "imgui", "__init__.pyi"), "w")
if OUT:
    out_cpp_file = open(os.path.join(os.path.dirname(__file__), "..", "src", "python", "generated_imgui.inc"), "w")

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
    if OUT_PYTHON:
        print(*args, **kwargs, file=out_file)

cpp_contents = io.StringIO()
def out_cpp(*args, **kwargs):
    print(*args, **kwargs, file=cpp_contents)
    if PRINT:
        print(*args, **kwargs)
    if OUT:
        print(*args, **kwargs, file=out_cpp_file)

out("from enum import IntEnum, IntFlag")
out("from typing import Tuple, Optional")
out("")
out("class ID: ...")
out("class Color: ...")
out("class Viewport: ...")
out("class DrawList: ...")
out("class Font: ...")
out("class WindowClass: ...")
out("class Payload: ...")
out("class Storage: ...")
out("class TextureID: ...")
out("class DrawListSharedData: ...")
out("class PlatformIO: ...")
out("class Style: ...")
out("class size_t: ...")
out("class double: ...")
out("")

for enum in data["enums"]:
    flags: str = enum["is_flags_enum"]
    name: str = enum["name"]
    elems: str = enum["elements"]
    orig: str = enum["original_fully_qualified_name"]

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

    extra_flag = ""
    if flags:
        extra_flag = ", nb::is_flag()"
    out_cpp(f'nb::enum_<{orig}>(mod_imgui, "{enum_name}", nb::is_arithmetic() {extra_flag})')

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
            out(" " * 4 + f"{elem_name} = 0x{value:x}")
        else:
            out(" " * 4 + f"{elem_name} = {value}")
        out_cpp(" " * 4 + f'.value("{elem_name}", {name})')
    out_cpp(" " * 4 + ";")
    out("")
    out_cpp("")

    #     print(e["name"], e["original_fully_qualified_name"])


# Ignore typedefs for now, no real use yet. Maybe once function args type are missing?
for t in data["typedefs"]:
    name = t["name"]
    typ = t["type"]
    # print(name, typ)

# Will need to handle structs later
enabled_structs = {
    "ImGuiIO"
}
for struct in data["structs"]:
    if struct["name"] in enabled_structs:
        struct_name = struct["name"]
        out_cpp(f'nb::class_<{struct_name}>(mod_imgui, "{struct_name[5:]}")')
        for field in struct["fields"]:
            field_name = field["name"]
            field_type = field["type"]
            if field["is_array"]:
                continue
            # print(field_type["description"])
            if not field_type["description"]["kind"] == "Builtin":
                continue
            out_cpp(" " * 4 + f'.def_rw("{pascal_to_snake_case(field_name)}", &{struct_name}::{field_name})')
        out_cpp(";")
        out_cpp("")

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
    # if type_decl == "ImVec2":
        # name = "Tuple[float, float]"
    # elif type_decl == "ImVec4":
        # name = "Tuple[float, float, float, float]"
    if type_decl == "ImU32":
        # name = "Tuple[int, int, int, int]"
        name = "Color"
    # elif type_decl == "double":
    #     flags |= TypeFlag.IS_BUILTIN
    #     name = "float"
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
    if name.startswith("Im"):
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
    module = func_name.split("_")[0]

    if f["is_imstr_helper"]:
        continue
    if f["is_manual_helper"]:
        continue
    if f["is_unformatted_helper"]:
        continue
    if f["is_default_argument_helper"]:
        continue


    # Skip specific functions we dont need
    if func_name in {
        # Context stuff (don't need)
        "ImGui_CreateContext",
        "ImGui_DestroyContext",
        "ImGui_GetCurrentContext",
        "ImGui_SetCurrentContext",
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

        # TODO:
        "ImGui_GetIO",
        # InputText
        "ImGui_InputText",          # Requires special handling instead of callback
        "ImGui_InputTextMultiline", # Requires special handling instead of callback
        "ImGui_InputTextWithHint",  # Requires special handling instead of callback
        # Plot
        "ImGui_PlotLines",
        "ImGui_PlotHistogram",
        # .ini
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
        # Handle class methods, currently only ImDrawList is supported
        if func_name.startswith("ImDrawList_"):
            pass
        else:
            continue

    py_func_name = pascal_to_snake_case(func_name[6:])

    out(f"def {py_func_name}(", end="")


    args: List[Arg] = []

    additional_ret = None
    additional_ret_type = None
    additional_ret_name = None

    for a in f["arguments"]:
        arg_name = a["name"]
        if arg_name == "in":
            arg_name = "value"

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
    out(", ".join(args_str), end="")
    out(f")", end="")
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

    if ret_str:
        out(f" -> {ret_str}", end="")
    out(f": ...")


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
                else:
                    return typ.cpp_name
        else:
            return type_str_to_cpp(typ.name)

    if py_func_name.startswith("list__"):
        if drawlist_started == False:
            drawlist_started = True
            out_cpp(f"""auto drawlist_class = nb::class_<DrawList>(mod_imgui, "DrawList",
    nb::intrusive_ptr<DrawList>([](DrawList *o, PyObject *po) noexcept {{ o->set_self_py(po); }}))
""")
        # Skip protected member functions
        if py_func_name.startswith("list___"):
            continue

        out_cpp(" " * 4 + f".def(\"{py_func_name[6:]}\",")
    else:
        if drawlist_started and drawlist_ended == False:
            drawlist_ended = True
            out_cpp(";")
        out_cpp(f"mod_imgui.def(\"{py_func_name}\",")

    out_cpp(" " * 4 + "[] (", end="")
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
    out_cpp(", ".join(cpp_args_str), end="")

    if has_ret:
        if has_add_ret:
            out_cpp(f") -> nb::tuple {{")
        else:
            out_cpp(f") -> {type_to_cpp(ret_type, True)} {{")
    else:
        if has_add_ret:
            out_cpp(f") -> {type_to_cpp(additional_ret_type, True)} {{")
        else:
            out_cpp(f") {{")


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
            elif arg.name == "fmt":
                name = r'"%s", fmt'
            elif arg.type.name == "memoryview":
                name = f"{arg.name}.data()"
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
        out_cpp(func_call + "(" + ", ".join(arg_call_names) + ")", end="")

    def out_add_ret():
        out_cpp(f"{additional_ret_name}", end="")


    if not has_ret:
        out_cpp(" " * 8, end="")
        out_call()
        out_cpp(";")
        if has_add_ret:
            out_cpp(" " * 8 + "return ", end="")
            out_add_ret()
            out_cpp(";")
    else:
        if has_add_ret:
            # Can't do inline because call needs to happen before ret, arg eval order to tuple is unspecified
            out_cpp(" " * 8 + "auto _call = ", end="")
            out_call()
            out_cpp(";")
            out_cpp(" " * 8 + "auto _ret = ", end="")
            out_add_ret()
            out_cpp(";")
            out_cpp(" " * 8 + "return nb::make_tuple(_call, _ret);")
        else:
            out_cpp(" " * 8 + "return ", end="")
            # if ret_type.flags & TypeFlag.IS_USER_TYPE and ret_type.flags & TypeFlag.IS_PTR:
            if ret_type.cpp_name == "ImDrawList*":
                out_cpp(f"new {ret_type.name}(", end="")
                out_call()
                out_cpp(f")", end="")
            else:
                out_call()
            out_cpp(";")

    out_cpp(" " * 4 + f"}}", end="")
    if nb_args_str:
        out_cpp(", ", end="")
    out_cpp(", ".join(nb_args_str), end="")
    if  py_func_name.startswith("list__"):
        out_cpp(")")
    else:
        out_cpp(f");\n")


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
