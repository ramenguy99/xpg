import numpy as np
from enum import Enum, auto

import spirv_constants as spv

class Stage(Enum):
    VERTEX = auto()
    TESSELLATION_CONTROL = auto()
    TESSELLATION_EVALUATION = auto()
    GEOMETRY = auto()
    FRAGMENT = auto()
    COMPUTE = auto()

class Id():
    def __init__(self):
        self.set = None
        self.binding = None
        self.opcode = 0
        self.type_id = None
        self.storage_class = None

    def __repr__(self) -> str:
        l = []

        l.append(f"{spv.opcode_to_string[self.opcode]} ({self.opcode})")
        if self.set is not None:
            l.append(f"set: {self.set}")
        if self.binding is not None:
            l.append(f"binding: {self.binding}")

        return "\n    ".join(l)


class Shader:
    def __init__(self, path):
        words = np.frombuffer(open(path, "rb").read(), np.uint32)

        magic = words[0]
        assert magic == 0x07230203, f"Invalid magic number 0x{magic:08x}"

        _version = words[1]
        _generator = words[2]
        id_bound = words[3]
        _reserved_schema = words[4]

        ids = [Id() for i in range(id_bound)]

        i = 5

        local_size_id = None

        while i < len(words):
            w = words[i]

            opcode = w & 0xFFFF
            word_count = (w >> 16) & 0xFFFF

            inst = words[i: i+word_count]

            if opcode == spv.OpEntryPoint:
                model = inst[1]
                model_to_stage = {
                    spv.ExecutionModelVertex: Stage.VERTEX,
                    spv.ExecutionModelTessellationControl: Stage.TESSELLATION_CONTROL,
                    spv.ExecutionModelTessellationEvaluation: Stage.TESSELLATION_EVALUATION,
                    spv.ExecutionModelGeometry: Stage.GEOMETRY,
                    spv.ExecutionModelFragment: Stage.FRAGMENT,
                    spv.ExecutionModelGLCompute: Stage.COMPUTE,
                }
                self.stage = model_to_stage[model]
            elif opcode == spv.OpExecutionModeId:
                mode = inst[2]
                if mode == spv.ExecutionModeLocalSize:
                    self.local_size = (inst[3], inst[4], inst[5])
            elif opcode == spv.OpExecutionModeId:
                mode = inst[2]
                if mode == spv.ExecutionModeLocalSizeId:
                    local_size_id = (inst[3], inst[4], inst[5])
            elif opcode == spv.OpDecorate:
                id = inst[1]
                assert id < id_bound, f"Id {id} out of bounds ({id_bound})"
                kind = inst[2]
                if kind == spv.DecorationDescriptorSet:
                    ids[id].set = inst[3]
                elif kind == spv.DecorationBinding:
                    ids[id].binding = inst[3]
            elif opcode in [
                spv.OpTypeStruct,
                spv.OpTypeImage,
                spv.OpTypeSampler,
                spv.OpTypeSampledImage,
            ]:
                id = inst[1]
                assert ids[id].opcode == 0
                ids[id].opcode = opcode
            elif opcode == spv.OpTypePointer:
                id = inst[1]
                assert ids[id].opcode == 0
                ids[id].opcode = opcode
                ids[id].type_id = inst[3]
                ids[id].storage_class = inst[2]
            elif opcode == spv.OpConstant:
                id = inst[1]
                assert ids[id].opcode == 0
                ids[id].opcode = opcode
                ids[id].type_id = inst[1]
                ids[id].constant = inst[3]
            elif opcode == spv.OpVariable:
                id = inst[2]
                assert ids[id].opcode == 0, ids[id].opcode
                ids[id].opcode = opcode
                ids[id].type_id = inst[1]
                ids[id].storage_class = inst[3]
            i += word_count

        for id in ids:
            print(id)

        if self.stage == Stage.COMPUTE and local_size_id != None:
            def get_size(id):
                assert ids[id].opcode == spv.OpConstant
                return ids[id].constant

            self.local_size = (
                get_size(self.local_size_id[0]),
                get_size(self.local_size_id[1]),
                get_size(self.local_size_id[2]),
            )


Shader("../res/basic.vert.spirv")