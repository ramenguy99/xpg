from pyxpg import slang
import numpy as np


scalar_to_np = {
    slang.ScalarKind.Bool: bool,
    slang.ScalarKind.Int8: np.int8,
    slang.ScalarKind.UInt8: np.uint8,
    slang.ScalarKind.Int16: np.int16,
    slang.ScalarKind.UInt16: np.uint16,
    slang.ScalarKind.Int32: np.int32,
    slang.ScalarKind.UInt32: np.uint32,
    slang.ScalarKind.Int64: np.int64,
    slang.ScalarKind.UInt64: np.uint64,
    slang.ScalarKind.Float16: np.float16,
    slang.ScalarKind.Float32: np.float32,
    slang.ScalarKind.Float64: np.float64,
}

def to_dtype(typ: slang.Type) -> np.dtype:
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
        raise TypeError(f"Unknown slang.Type: {type(typ)}")

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
        print(" " * indent + f"Unknown slang.Type: {type(typ)}")