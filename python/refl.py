from pyxpg import *
from pyxpg import slang
import numpy as np
import json

if False:
    nested_typ = np.dtype({
        'd': (np.int8, 0),
        'e': (np.int32, 4),
    }, align=False)

    typ = np.dtype({
        'a': (np.int64, 0),
        'complex': (nested_typ, 8),
        'b': (np.int32, nested_typ.itemsize + 8),
    }, align=False)

    print(typ.itemsize)

    # Can assign to all fields at once (basically interleave data into array)
    N = 4

    # Backed allocation
    mem = bytearray(typ.itemsize * N)
    a = np.frombuffer(mem, typ)
    print(a.flags)

    # Owned alloation
    a = np.zeros((N,), typ)
    print(a.flags)

    a["b"] = 1
    a["a"] = 4
    a["complex"]["d"] = 2
    a["complex"]["e"] = 3

    print(mem.hex())
    print(a.tobytes().hex())

prog = slang.compile("shaders/color.vert.slang", "main")

def print_type(typ: slang.Type, indent = 0):
    if   isinstance(typ, slang.Scalar):
        print(" " * indent + f"{typ.base}")
    elif isinstance(typ, slang.Vector):
        print(" " * indent + f"{typ.base}_{typ.count}")
    elif isinstance(typ, slang.Matrix):
        print(" " * indent + f"{typ.base}_{typ.rows}x{typ.columns}")
    elif isinstance(typ, slang.Array):
        print(" " * indent + f"{typ.type}[{typ.count}]")
    elif isinstance(typ, slang.Struct):
        print(" " * indent + "Struct")
        for f in typ.fields:
            print(" " * (indent + 4) + f"{f.offset:3d} | {f.name}: ", end="")
            print_type(f.type, indent + 4)
    else:
        print(" " * indent + "UNKNOWN TYPE: ", typ)
    
scalar_to_np = {
    slang.ScalarKind.Bool: np.uint8,
    slang.ScalarKind.Float32: np.float32,
} 

def to_dtype(typ: slang.Type):
    if   isinstance(typ, slang.Scalar):
        return scalar_to_np[typ.base]
    elif isinstance(typ, slang.Vector):
        return np.dtype((scalar_to_np[typ.base], (typ.count,)))
    elif isinstance(typ, slang.Matrix):
        return np.dtype((scalar_to_np[typ.base], (typ.rows, typ.columns)))
    elif isinstance(typ, slang.Array):
        return np.dtype((to_dtype(typ.type), (typ.count,)))
    elif isinstance(typ, slang.Struct):
        d = {}
        for f in typ.fields:
            d[f.name] = (to_dtype(f.type), f.offset)
        return np.dtype(d)
    else:
        raise TypeError("Unkown type")

# for res in prog.reflection.resources:
#     print(res.name)
#     print(" ", res.set)
#     print(" ", res.binding)
#     print(" ", res.access)
#     print(" ", res.shape)
#     print(" ", res.kind)
#     print_type(res.type, 4)

dt = to_dtype(prog.reflection.resources[0].type)

N = 1
a = np.zeros((N,), dt)

a["transform"] = np.ones((4, 4)) * 1
a["nest1"]["val2"] = 2
a["nest2"]["val2"] = 4

print(a.tobytes().hex())