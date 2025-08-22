from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from pyxpg import DescriptorType, slang


@dataclass
class DescriptorInfo:
    name: str
    binding: int
    set: int
    resource: slang.Resource
    image_format: slang.ImageFormat
    count: int = 1  # 1 for single element, 0 for unbounded


def to_descriptor_type(binding_type: slang.BindingType) -> DescriptorType:
    if binding_type == slang.BindingType.SAMPLER:
        return DescriptorType.SAMPLER
    elif binding_type == slang.BindingType.COMBINED_TEXTURE_SAMPLER:
        return DescriptorType.COMBINED_IMAGE_SAMPLER
    elif binding_type == slang.BindingType.TEXTURE:
        return DescriptorType.SAMPLED_IMAGE
    elif binding_type == slang.BindingType.MUTABLE_TEXTURE:
        return DescriptorType.STORAGE_IMAGE
    elif binding_type == slang.BindingType.TYPED_BUFFER:
        return DescriptorType.UNIFORM_TEXEL_BUFFER
    elif binding_type == slang.BindingType.MUTABLE_TYPED_BUFFER:
        return DescriptorType.STORAGE_TEXEL_BUFFER
    elif binding_type == slang.BindingType.RAW_BUFFER:
        return DescriptorType.STORAGE_BUFFER
    elif binding_type == slang.BindingType.MUTABLE_RAW_BUFFER:
        return DescriptorType.STORAGE_BUFFER
    elif binding_type == slang.BindingType.INPUT_RENDER_TARGET:
        return DescriptorType.INPUT_ATTACHMENT
    elif binding_type == slang.BindingType.INLINE_UNIFORM_DATA:
        return DescriptorType.INLINE_UNIFORM_BLOCK
    elif binding_type == slang.BindingType.RAYTRACING_ACCELERATION_STRUCTURE:
        return DescriptorType.ACCELERATION_STRUCTURE
    elif binding_type == slang.BindingType.CONSTANT_BUFFER:
        return DescriptorType.UNIFORM_BUFFER
    else:
        raise ValueError("Unknown slang.BindingType {}", binding_type)


def _flatten_descriptor_sets(
    sets: Dict[int, List[DescriptorInfo]],
    typ: slang.Type,
    name: str,
    binding: int,
    set: int,
    image_format: slang.ImageFormat,
) -> None:
    if isinstance(typ, (slang.Scalar, slang.Vector, slang.Matrix)):
        pass
    elif isinstance(typ, slang.Array):
        child_typ = typ.type
        if isinstance(child_typ, slang.Resource):
            sets.setdefault(set, []).append(DescriptorInfo(name, binding, set, child_typ, image_format, typ.count))
    elif isinstance(typ, slang.Struct):
        for f in typ.fields:
            _flatten_descriptor_sets(
                sets,
                f.type,
                f"{name}.{f.name}" if name else f.name,
                binding + f.binding,
                set + f.set,
                f.image_format,
            )
    elif isinstance(typ, slang.Resource):
        sets.setdefault(set, []).append(DescriptorInfo(name, binding, set, typ, image_format))
    else:
        raise TypeError(f"Unknown slang.Type: {type(typ)}")


class DescriptorSetsReflection:
    def __init__(self, refl: slang.Reflection):
        self.sets: Dict[int, List[DescriptorInfo]] = {}
        _flatten_descriptor_sets(self.sets, refl.object, "", 0, 0, slang.ImageFormat.UNKNOWN)

        self.descriptors: Dict[str, DescriptorInfo] = {}
        for s in self.sets.values():
            for r in s:
                self.descriptors[r.name] = r


_scalar_to_np = {
    slang.ScalarKind.BOOL: np.dtype(bool),
    slang.ScalarKind.INT8: np.dtype(np.int8),
    slang.ScalarKind.UINT8: np.dtype(np.uint8),
    slang.ScalarKind.INT16: np.dtype(np.int16),
    slang.ScalarKind.UINT16: np.dtype(np.uint16),
    slang.ScalarKind.INT32: np.dtype(np.int32),
    slang.ScalarKind.UINT32: np.dtype(np.uint32),
    slang.ScalarKind.INT64: np.dtype(np.int64),
    slang.ScalarKind.UINT64: np.dtype(np.uint64),
    slang.ScalarKind.FLOAT16: np.dtype(np.float16),
    slang.ScalarKind.FLOAT32: np.dtype(np.float32),
    slang.ScalarKind.FLOAT64: np.dtype(np.float64),
}


def to_dtype(typ: slang.Type) -> np.dtype:
    if isinstance(typ, slang.Scalar):
        return _scalar_to_np[typ.base]
    elif isinstance(typ, slang.Vector):
        return np.dtype((_scalar_to_np[typ.base], (typ.count,)))
    elif isinstance(typ, slang.Matrix):
        return np.dtype((_scalar_to_np[typ.base], (typ.rows, typ.columns)))
    elif isinstance(typ, slang.Array):
        return np.dtype((to_dtype(typ.type), (typ.count,)))
    elif isinstance(typ, slang.Struct):
        d = {}
        for f in typ.fields:
            d[f.name] = (to_dtype(f.type), f.offset)
        return np.dtype(d)  # type: ignore
    else:
        raise TypeError(f"Unknown slang.Type: {type(typ)}")


def print_type(typ: slang.Type, indent: int = 0) -> None:
    if isinstance(typ, slang.Scalar):
        print(" " * indent + f"{typ.base}")
    elif isinstance(typ, slang.Vector):
        print(" " * indent + f"{typ.base}_{typ.count}")
    elif isinstance(typ, slang.Matrix):
        print(" " * indent + f"{typ.base}_{typ.rows}x{typ.columns}")
    elif isinstance(typ, slang.Array):
        child_typ = typ.type
        if isinstance(child_typ, slang.Resource):
            print(" " * indent + f"{child_typ.kind}[{typ.count}]")
        else:
            print(" " * indent + f"{typ.type}[{typ.count}]")
    elif isinstance(typ, slang.Struct):
        print(" " * indent + "Struct")
        for f in typ.fields:
            print(
                " " * (indent + 4) + f"{f.offset:3d} ({f.binding}, {f.set}) | {f.image_format} | {f.name}: ",
                end="",
            )
            print_type(f.type, indent + 4)
    elif isinstance(typ, slang.Resource):
        print(" " * indent + f"Resource[{typ.kind}]")
        if typ.kind == slang.ResourceKind.SAMPLER:
            pass
        elif typ.kind == slang.ResourceKind.TEXTURE_2D:
            pass
        elif typ.kind == slang.ResourceKind.STRUCTURED_BUFFER:
            print_type(typ.type, indent + 4)
        elif typ.kind == slang.ResourceKind.CONSTANT_BUFFER:
            print_type(typ.type, indent + 4)
    else:
        print(" " * indent + f"Unknown slang.Type: {type(typ)}")
