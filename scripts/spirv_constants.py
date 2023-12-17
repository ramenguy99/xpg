# Opcodes
OpNop = 0
OpUndef = 1
OpSourceContinued = 2
OpSource = 3
OpSourceExtension = 4
OpName = 5
OpMemberName = 6
OpString = 7
OpLine = 8
OpExtension = 10
OpExtInstImport = 11
OpExtInst = 12
OpMemoryModel = 14
OpEntryPoint = 15
OpExecutionMode = 16
OpCapability = 17
OpTypeVoid = 19
OpTypeBool = 20
OpTypeInt = 21
OpTypeFloat = 22
OpTypeVector = 23
OpTypeMatrix = 24
OpTypeImage = 25
OpTypeSampler = 26
OpTypeSampledImage = 27
OpTypeArray = 28
OpTypeRuntimeArray = 29
OpTypeStruct = 30
OpTypeOpaque = 31
OpTypePointer = 32
OpTypeFunction = 33
OpTypeEvent = 34
OpTypeDeviceEvent = 35
OpTypeReserveId = 36
OpTypeQueue = 37
OpTypePipe = 38
OpTypeForwardPointer = 39
OpConstantTrue = 41
OpConstantFalse = 42
OpConstant = 43
OpConstantComposite = 44
OpConstantSampler = 45
OpConstantNull = 46
OpSpecConstantTrue = 48
OpSpecConstantFalse = 49
OpSpecConstant = 50
OpSpecConstantComposite = 51
OpSpecConstantOp = 52
OpFunction = 54
OpFunctionParameter = 55
OpFunctionEnd = 56
OpFunctionCall = 57
OpVariable = 59
OpImageTexelPointer = 60
OpLoad = 61
OpStore = 62
OpCopyMemory = 63
OpCopyMemorySized = 64
OpAccessChain = 65
OpInBoundsAccessChain = 66
OpPtrAccessChain = 67
OpArrayLength = 68
OpGenericPtrMemSemantics = 69
OpInBoundsPtrAccessChain = 70
OpDecorate = 71
OpMemberDecorate = 72
OpDecorationGroup = 73
OpGroupDecorate = 74
OpGroupMemberDecorate = 75
OpVectorExtractDynamic = 77
OpVectorInsertDynamic = 78
OpVectorShuffle = 79
OpCompositeConstruct = 80
OpCompositeExtract = 81
OpCompositeInsert = 82
OpCopyObject = 83
OpTranspose = 84
OpSampledImage = 86
OpImageSampleImplicitLod = 87
OpImageSampleExplicitLod = 88
OpImageSampleDrefImplicitLod = 89
OpImageSampleDrefExplicitLod = 90
OpImageSampleProjImplicitLod = 91
OpImageSampleProjExplicitLod = 92
OpImageSampleProjDrefImplicitLod = 93
OpImageSampleProjDrefExplicitLod = 94
OpImageFetch = 95
OpImageGather = 96
OpImageDrefGather = 97
OpImageRead = 98
OpImageWrite = 99
OpImage = 100
OpImageQueryFormat = 101
OpImageQueryOrder = 102
OpImageQuerySizeLod = 103
OpImageQuerySize = 104
OpImageQueryLod = 105
OpImageQueryLevels = 106
OpImageQuerySamples = 107
OpConvertFToU = 109
OpConvertFToS = 110
OpConvertSToF = 111
OpConvertUToF = 112
OpUConvert = 113
OpSConvert = 114
OpFConvert = 115
OpQuantizeToF16 = 116
OpConvertPtrToU = 117
OpSatConvertSToU = 118
OpSatConvertUToS = 119
OpConvertUToPtr = 120
OpPtrCastToGeneric = 121
OpGenericCastToPtr = 122
OpGenericCastToPtrExplicit = 123
OpBitcast = 124
OpSNegate = 126
OpFNegate = 127
OpIAdd = 128
OpFAdd = 129
OpISub = 130
OpFSub = 131
OpIMul = 132
OpFMul = 133
OpUDiv = 134
OpSDiv = 135
OpFDiv = 136
OpUMod = 137
OpSRem = 138
OpSMod = 139
OpFRem = 140
OpFMod = 141
OpVectorTimesScalar = 142
OpMatrixTimesScalar = 143
OpVectorTimesMatrix = 144
OpMatrixTimesVector = 145
OpMatrixTimesMatrix = 146
OpOuterProduct = 147
OpDot = 148
OpIAddCarry = 149
OpISubBorrow = 150
OpUMulExtended = 151
OpSMulExtended = 152
OpAny = 154
OpAll = 155
OpIsNan = 156
OpIsInf = 157
OpIsFinite = 158
OpIsNormal = 159
OpSignBitSet = 160
OpLessOrGreater = 161
OpOrdered = 162
OpUnordered = 163
OpLogicalEqual = 164
OpLogicalNotEqual = 165
OpLogicalOr = 166
OpLogicalAnd = 167
OpLogicalNot = 168
OpSelect = 169
OpIEqual = 170
OpINotEqual = 171
OpUGreaterThan = 172
OpSGreaterThan = 173
OpUGreaterThanEqual = 174
OpSGreaterThanEqual = 175
OpULessThan = 176
OpSLessThan = 177
OpULessThanEqual = 178
OpSLessThanEqual = 179
OpFOrdEqual = 180
OpFUnordEqual = 181
OpFOrdNotEqual = 182
OpFUnordNotEqual = 183
OpFOrdLessThan = 184
OpFUnordLessThan = 185
OpFOrdGreaterThan = 186
OpFUnordGreaterThan = 187
OpFOrdLessThanEqual = 188
OpFUnordLessThanEqual = 189
OpFOrdGreaterThanEqual = 190
OpFUnordGreaterThanEqual = 191
OpShiftRightLogical = 194
OpShiftRightArithmetic = 195
OpShiftLeftLogical = 196
OpBitwiseOr = 197
OpBitwiseXor = 198
OpBitwiseAnd = 199
OpNot = 200
OpBitFieldInsert = 201
OpBitFieldSExtract = 202
OpBitFieldUExtract = 203
OpBitReverse = 204
OpBitCount = 205
OpDPdx = 207
OpDPdy = 208
OpFwidth = 209
OpDPdxFine = 210
OpDPdyFine = 211
OpFwidthFine = 212
OpDPdxCoarse = 213
OpDPdyCoarse = 214
OpFwidthCoarse = 215
OpEmitVertex = 218
OpEndPrimitive = 219
OpEmitStreamVertex = 220
OpEndStreamPrimitive = 221
OpControlBarrier = 224
OpMemoryBarrier = 225
OpAtomicLoad = 227
OpAtomicStore = 228
OpAtomicExchange = 229
OpAtomicCompareExchange = 230
OpAtomicCompareExchangeWeak = 231
OpAtomicIIncrement = 232
OpAtomicIDecrement = 233
OpAtomicIAdd = 234
OpAtomicISub = 235
OpAtomicSMin = 236
OpAtomicUMin = 237
OpAtomicSMax = 238
OpAtomicUMax = 239
OpAtomicAnd = 240
OpAtomicOr = 241
OpAtomicXor = 242
OpPhi = 245
OpLoopMerge = 246
OpSelectionMerge = 247
OpLabel = 248
OpBranch = 249
OpBranchConditional = 250
OpSwitch = 251
OpKill = 252
OpReturn = 253
OpReturnValue = 254
OpUnreachable = 255
OpLifetimeStart = 256
OpLifetimeStop = 257
OpGroupAsyncCopy = 259
OpGroupWaitEvents = 260
OpGroupAll = 261
OpGroupAny = 262
OpGroupBroadcast = 263
OpGroupIAdd = 264
OpGroupFAdd = 265
OpGroupFMin = 266
OpGroupUMin = 267
OpGroupSMin = 268
OpGroupFMax = 269
OpGroupUMax = 270
OpGroupSMax = 271
OpReadPipe = 274
OpWritePipe = 275
OpReservedReadPipe = 276
OpReservedWritePipe = 277
OpReserveReadPipePackets = 278
OpReserveWritePipePackets = 279
OpCommitReadPipe = 280
OpCommitWritePipe = 281
OpIsValidReserveId = 282
OpGetNumPipePackets = 283
OpGetMaxPipePackets = 284
OpGroupReserveReadPipePackets = 285
OpGroupReserveWritePipePackets = 286
OpGroupCommitReadPipe = 287
OpGroupCommitWritePipe = 288
OpEnqueueMarker = 291
OpEnqueueKernel = 292
OpGetKernelNDrangeSubGroupCount = 293
OpGetKernelNDrangeMaxSubGroupSize = 294
OpGetKernelWorkGroupSize = 295
OpGetKernelPreferredWorkGroupSizeMultiple = 296
OpRetainEvent = 297
OpReleaseEvent = 298
OpCreateUserEvent = 299
OpIsValidEvent = 300
OpSetUserEventStatus = 301
OpCaptureEventProfilingInfo = 302
OpGetDefaultQueue = 303
OpBuildNDRange = 304
OpImageSparseSampleImplicitLod = 305
OpImageSparseSampleExplicitLod = 306
OpImageSparseSampleDrefImplicitLod = 307
OpImageSparseSampleDrefExplicitLod = 308
OpImageSparseSampleProjImplicitLod = 309
OpImageSparseSampleProjExplicitLod = 310
OpImageSparseSampleProjDrefImplicitLod = 311
OpImageSparseSampleProjDrefExplicitLod = 312
OpImageSparseFetch = 313
OpImageSparseGather = 314
OpImageSparseDrefGather = 315
OpImageSparseTexelsResident = 316
OpNoLine = 317
OpAtomicFlagTestAndSet = 318
OpAtomicFlagClear = 319
OpImageSparseRead = 320
OpSizeOf = 321
OpTypePipeStorage = 322
OpConstantPipeStorage = 323
OpCreatePipeFromPipeStorage = 324
OpGetKernelLocalSizeForSubgroupCount = 325
OpGetKernelMaxNumSubgroups = 326
OpTypeNamedBarrier = 327
OpNamedBarrierInitialize = 328
OpMemoryNamedBarrier = 329
OpModuleProcessed = 330
OpExecutionModeId = 331

# Execution model
ExecutionModelVertex = 0
ExecutionModelTessellationControl = 1
ExecutionModelTessellationEvaluation = 2
ExecutionModelGeometry = 3
ExecutionModelFragment = 4
ExecutionModelGLCompute = 5
ExecutionModelKernel = 6
ExecutionModelTaskNV = 5267
ExecutionModelMeshNV = 5268
ExecutionModelRayGenerationNV = 5313
ExecutionModelRayGenerationKHR = 5313
ExecutionModelIntersectionNV = 5314
ExecutionModelIntersectionKHR = 5314
ExecutionModelAnyHitNV = 5315
ExecutionModelAnyHitKHR = 5315
ExecutionModelClosestHitNV = 5316
ExecutionModelClosestHitKHR = 5316
ExecutionModelMissNV = 5317
ExecutionModelMissKHR = 5317
ExecutionModelCallableNV = 5318
ExecutionModelCallableKHR = 5318
ExecutionModelTaskEXT = 5364
ExecutionModelMeshEXT = 5365

#Execution mode
ExecutionModeInvocations = 0
ExecutionModeSpacingEqual = 1
ExecutionModeSpacingFractionalEven = 2
ExecutionModeSpacingFractionalOdd = 3
ExecutionModeVertexOrderCw = 4
ExecutionModeVertexOrderCcw = 5
ExecutionModePixelCenterInteger = 6
ExecutionModeOriginUpperLeft = 7
ExecutionModeOriginLowerLeft = 8
ExecutionModeEarlyFragmentTests = 9
ExecutionModePointMode = 10
ExecutionModeXfb = 11
ExecutionModeDepthReplacing = 12
ExecutionModeDepthGreater = 14
ExecutionModeDepthLess = 15
ExecutionModeDepthUnchanged = 16
ExecutionModeLocalSize = 17
ExecutionModeLocalSizeHint = 18
ExecutionModeInputPoints = 19
ExecutionModeInputLines = 20
ExecutionModeInputLinesAdjacency = 21
ExecutionModeTriangles = 22
ExecutionModeInputTrianglesAdjacency = 23
ExecutionModeQuads = 24
ExecutionModeIsolines = 25
ExecutionModeOutputVertices = 26
ExecutionModeOutputPoints = 27
ExecutionModeOutputLineStrip = 28
ExecutionModeOutputTriangleStrip = 29
ExecutionModeVecTypeHint = 30
ExecutionModeContractionOff = 31
ExecutionModeInitializer = 33
ExecutionModeFinalizer = 34
ExecutionModeSubgroupSize = 35
ExecutionModeSubgroupsPerWorkgroup = 36
ExecutionModeSubgroupsPerWorkgroupId = 37
ExecutionModeLocalSizeId = 38
ExecutionModeLocalSizeHintId = 39
ExecutionModeSubgroupUniformControlFlowKHR = 4421
ExecutionModePostDepthCoverage = 4446
ExecutionModeDenormPreserve = 4459
ExecutionModeDenormFlushToZero = 4460
ExecutionModeSignedZeroInfNanPreserve = 4461
ExecutionModeRoundingModeRTE = 4462
ExecutionModeRoundingModeRTZ = 4463
ExecutionModeStencilRefReplacingEXT = 5027
ExecutionModeOutputLinesNV = 5269
ExecutionModeOutputPrimitivesNV = 5270
ExecutionModeDerivativeGroupQuadsNV = 5289
ExecutionModeDerivativeGroupLinearNV = 5290
ExecutionModeOutputTrianglesNV = 5298
ExecutionModePixelInterlockOrderedEXT = 5366
ExecutionModePixelInterlockUnorderedEXT = 5367
ExecutionModeSampleInterlockOrderedEXT = 5368
ExecutionModeSampleInterlockUnorderedEXT = 5369
ExecutionModeShadingRateInterlockOrderedEXT = 5370
ExecutionModeShadingRateInterlockUnorderedEXT = 5371
ExecutionModeSharedLocalMemorySizeINTEL = 5618
ExecutionModeRoundingModeRTPINTEL = 5620
ExecutionModeRoundingModeRTNINTEL = 5621
ExecutionModeFloatingPointModeALTINTEL = 5622
ExecutionModeFloatingPointModeIEEEINTEL = 5623
ExecutionModeMaxWorkgroupSizeINTEL = 5893
ExecutionModeMaxWorkDimINTEL = 5894
ExecutionModeNoGlobalOffsetINTEL = 5895
ExecutionModeNumSIMDWorkitemsINTEL = 5896
ExecutionModeSchedulerTargetFmaxMhzINTEL = 5903
ExecutionModeNamedBarrierCountINTEL = 6417
ExecutionModeMax = 0x7fffffff


DecorationRelaxedPrecision = 0
DecorationSpecId = 1
DecorationBlock = 2
DecorationBufferBlock = 3
DecorationRowMajor = 4
DecorationColMajor = 5
DecorationArrayStride = 6
DecorationMatrixStride = 7
DecorationGLSLShared = 8
DecorationGLSLPacked = 9
DecorationCPacked = 10
DecorationBuiltIn = 11
DecorationNoPerspective = 13
DecorationFlat = 14
DecorationPatch = 15
DecorationCentroid = 16
DecorationSample = 17
DecorationInvariant = 18
DecorationRestrict = 19
DecorationAliased = 20
DecorationVolatile = 21
DecorationConstant = 22
DecorationCoherent = 23
DecorationNonWritable = 24
DecorationNonReadable = 25
DecorationUniform = 26
DecorationUniformId = 27
DecorationSaturatedConversion = 28
DecorationStream = 29
DecorationLocation = 30
DecorationComponent = 31
DecorationIndex = 32
DecorationBinding = 33
DecorationDescriptorSet = 34
DecorationOffset = 35
DecorationXfbBuffer = 36
DecorationXfbStride = 37
DecorationFuncParamAttr = 38
DecorationFPRoundingMode = 39
DecorationFPFastMathMode = 40
DecorationLinkageAttributes = 41
DecorationNoContraction = 42
DecorationInputAttachmentIndex = 43
DecorationAlignment = 44
DecorationMaxByteOffset = 45
DecorationAlignmentId = 46
DecorationMaxByteOffsetId = 47
DecorationNoSignedWrap = 4469
DecorationNoUnsignedWrap = 4470
DecorationExplicitInterpAMD = 4999
DecorationOverrideCoverageNV = 5248
DecorationPassthroughNV = 5250
DecorationViewportRelativeNV = 5252
DecorationSecondaryViewportRelativeNV = 5256
DecorationPerPrimitiveNV = 5271
DecorationPerViewNV = 5272
DecorationPerTaskNV = 5273
DecorationPerVertexKHR = 5285
DecorationPerVertexNV = 5285
DecorationNonUniform = 5300
DecorationNonUniformEXT = 5300
DecorationRestrictPointer = 5355
DecorationRestrictPointerEXT = 5355
DecorationAliasedPointer = 5356
DecorationAliasedPointerEXT = 5356
DecorationBindlessSamplerNV = 5398
DecorationBindlessImageNV = 5399
DecorationBoundSamplerNV = 5400
DecorationBoundImageNV = 5401
DecorationSIMTCallINTEL = 5599
DecorationReferencedIndirectlyINTEL = 5602
DecorationClobberINTEL = 5607
DecorationSideEffectsINTEL = 5608
DecorationVectorComputeVariableINTEL = 5624
DecorationFuncParamIOKindINTEL = 5625
DecorationVectorComputeFunctionINTEL = 5626
DecorationStackCallINTEL = 5627
DecorationGlobalVariableOffsetINTEL = 5628
DecorationCounterBuffer = 5634
DecorationHlslCounterBufferGOOGLE = 5634
DecorationHlslSemanticGOOGLE = 5635
DecorationUserSemantic = 5635
DecorationUserTypeGOOGLE = 5636
DecorationFunctionRoundingModeINTEL = 5822
DecorationFunctionDenormModeINTEL = 5823
DecorationRegisterINTEL = 5825
DecorationMemoryINTEL = 5826
DecorationNumbanksINTEL = 5827
DecorationBankwidthINTEL = 5828
DecorationMaxPrivateCopiesINTEL = 5829
DecorationSinglepumpINTEL = 5830
DecorationDoublepumpINTEL = 5831
DecorationMaxReplicatesINTEL = 5832
DecorationSimpleDualPortINTEL = 5833
DecorationMergeINTEL = 5834
DecorationBankBitsINTEL = 5835
DecorationForcePow2DepthINTEL = 5836
DecorationBurstCoalesceINTEL = 5899
DecorationCacheSizeINTEL = 5900
DecorationDontStaticallyCoalesceINTEL = 5901
DecorationPrefetchINTEL = 5902
DecorationStallEnableINTEL = 5905
DecorationFuseLoopsInFunctionINTEL = 5907
DecorationAliasScopeINTEL = 5914
DecorationNoAliasINTEL = 5915
DecorationBufferLocationINTEL = 5921
DecorationIOPipeStorageINTEL = 5944
DecorationFunctionFloatingPointModeINTEL = 6080
DecorationSingleElementVectorINTEL = 6085
DecorationVectorComputeCallableFunctionINTEL = 6087
DecorationMediaBlockIOINTEL = 6140
DecorationMax = 0x7fffffff


# Opcodes
opcode_to_string = {
    0: "OpNop",
    1: "OpUndef",
    2: "OpSourceContinued",
    3: "OpSource",
    4: "OpSourceExtension",
    5: "OpName",
    6: "OpMemberName",
    7: "OpString",
    8: "OpLine",
    10: "OpExtension",
    11: "OpExtInstImport",
    12: "OpExtInst",
    14: "OpMemoryModel",
    15: "OpEntryPoint",
    16: "OpExecutionMode",
    17: "OpCapability",
    19: "OpTypeVoid",
    20: "OpTypeBool",
    21: "OpTypeInt",
    22: "OpTypeFloat",
    23: "OpTypeVector",
    24: "OpTypeMatrix",
    25: "OpTypeImage",
    26: "OpTypeSampler",
    27: "OpTypeSampledImage",
    28: "OpTypeArray",
    29: "OpTypeRuntimeArray",
    30: "OpTypeStruct",
    31: "OpTypeOpaque",
    32: "OpTypePointer",
    33: "OpTypeFunction",
    34: "OpTypeEvent",
    35: "OpTypeDeviceEvent",
    36: "OpTypeReserveId",
    37: "OpTypeQueue",
    38: "OpTypePipe",
    39: "OpTypeForwardPointer",
    41: "OpConstantTrue",
    42: "OpConstantFalse",
    43: "OpConstant",
    44: "OpConstantComposite",
    45: "OpConstantSampler",
    46: "OpConstantNull",
    48: "OpSpecConstantTrue",
    49: "OpSpecConstantFalse",
    50: "OpSpecConstant",
    51: "OpSpecConstantComposite",
    52: "OpSpecConstantOp",
    54: "OpFunction",
    55: "OpFunctionParameter",
    56: "OpFunctionEnd",
    57: "OpFunctionCall",
    59: "OpVariable",
    60: "OpImageTexelPointer",
    61: "OpLoad",
    62: "OpStore",
    63: "OpCopyMemory",
    64: "OpCopyMemorySized",
    65: "OpAccessChain",
    66: "OpInBoundsAccessChain",
    67: "OpPtrAccessChain",
    68: "OpArrayLength",
    69: "OpGenericPtrMemSemantics",
    70: "OpInBoundsPtrAccessChain",
    71: "OpDecorate",
    72: "OpMemberDecorate",
    73: "OpDecorationGroup",
    74: "OpGroupDecorate",
    75: "OpGroupMemberDecorate",
    77: "OpVectorExtractDynamic",
    78: "OpVectorInsertDynamic",
    79: "OpVectorShuffle",
    80: "OpCompositeConstruct",
    81: "OpCompositeExtract",
    82: "OpCompositeInsert",
    83: "OpCopyObject",
    84: "OpTranspose",
    86: "OpSampledImage",
    87: "OpImageSampleImplicitLod",
    88: "OpImageSampleExplicitLod",
    89: "OpImageSampleDrefImplicitLod",
    90: "OpImageSampleDrefExplicitLod",
    91: "OpImageSampleProjImplicitLod",
    92: "OpImageSampleProjExplicitLod",
    93: "OpImageSampleProjDrefImplicitLod",
    94: "OpImageSampleProjDrefExplicitLod",
    95: "OpImageFetch",
    96: "OpImageGather",
    97: "OpImageDrefGather",
    98: "OpImageRead",
    99: "OpImageWrite",
    100: "OpImage",
    101: "OpImageQueryFormat",
    102: "OpImageQueryOrder",
    103: "OpImageQuerySizeLod",
    104: "OpImageQuerySize",
    105: "OpImageQueryLod",
    106: "OpImageQueryLevels",
    107: "OpImageQuerySamples",
    109: "OpConvertFToU",
    110: "OpConvertFToS",
    111: "OpConvertSToF",
    112: "OpConvertUToF",
    113: "OpUConvert",
    114: "OpSConvert",
    115: "OpFConvert",
    116: "OpQuantizeToF16",
    117: "OpConvertPtrToU",
    118: "OpSatConvertSToU",
    119: "OpSatConvertUToS",
    120: "OpConvertUToPtr",
    121: "OpPtrCastToGeneric",
    122: "OpGenericCastToPtr",
    123: "OpGenericCastToPtrExplicit",
    124: "OpBitcast",
    126: "OpSNegate",
    127: "OpFNegate",
    128: "OpIAdd",
    129: "OpFAdd",
    130: "OpISub",
    131: "OpFSub",
    132: "OpIMul",
    133: "OpFMul",
    134: "OpUDiv",
    135: "OpSDiv",
    136: "OpFDiv",
    137: "OpUMod",
    138: "OpSRem",
    139: "OpSMod",
    140: "OpFRem",
    141: "OpFMod",
    142: "OpVectorTimesScalar",
    143: "OpMatrixTimesScalar",
    144: "OpVectorTimesMatrix",
    145: "OpMatrixTimesVector",
    146: "OpMatrixTimesMatrix",
    147: "OpOuterProduct",
    148: "OpDot",
    149: "OpIAddCarry",
    150: "OpISubBorrow",
    151: "OpUMulExtended",
    152: "OpSMulExtended",
    154: "OpAny",
    155: "OpAll",
    156: "OpIsNan",
    157: "OpIsInf",
    158: "OpIsFinite",
    159: "OpIsNormal",
    160: "OpSignBitSet",
    161: "OpLessOrGreater",
    162: "OpOrdered",
    163: "OpUnordered",
    164: "OpLogicalEqual",
    165: "OpLogicalNotEqual",
    166: "OpLogicalOr",
    167: "OpLogicalAnd",
    168: "OpLogicalNot",
    169: "OpSelect",
    170: "OpIEqual",
    171: "OpINotEqual",
    172: "OpUGreaterThan",
    173: "OpSGreaterThan",
    174: "OpUGreaterThanEqual",
    175: "OpSGreaterThanEqual",
    176: "OpULessThan",
    177: "OpSLessThan",
    178: "OpULessThanEqual",
    179: "OpSLessThanEqual",
    180: "OpFOrdEqual",
    181: "OpFUnordEqual",
    182: "OpFOrdNotEqual",
    183: "OpFUnordNotEqual",
    184: "OpFOrdLessThan",
    185: "OpFUnordLessThan",
    186: "OpFOrdGreaterThan",
    187: "OpFUnordGreaterThan",
    188: "OpFOrdLessThanEqual",
    189: "OpFUnordLessThanEqual",
    190: "OpFOrdGreaterThanEqual",
    191: "OpFUnordGreaterThanEqual",
    194: "OpShiftRightLogical",
    195: "OpShiftRightArithmetic",
    196: "OpShiftLeftLogical",
    197: "OpBitwiseOr",
    198: "OpBitwiseXor",
    199: "OpBitwiseAnd",
    200: "OpNot",
    201: "OpBitFieldInsert",
    202: "OpBitFieldSExtract",
    203: "OpBitFieldUExtract",
    204: "OpBitReverse",
    205: "OpBitCount",
    207: "OpDPdx",
    208: "OpDPdy",
    209: "OpFwidth",
    210: "OpDPdxFine",
    211: "OpDPdyFine",
    212: "OpFwidthFine",
    213: "OpDPdxCoarse",
    214: "OpDPdyCoarse",
    215: "OpFwidthCoarse",
    218: "OpEmitVertex",
    219: "OpEndPrimitive",
    220: "OpEmitStreamVertex",
    221: "OpEndStreamPrimitive",
    224: "OpControlBarrier",
    225: "OpMemoryBarrier",
    227: "OpAtomicLoad",
    228: "OpAtomicStore",
    229: "OpAtomicExchange",
    230: "OpAtomicCompareExchange",
    231: "OpAtomicCompareExchangeWeak",
    232: "OpAtomicIIncrement",
    233: "OpAtomicIDecrement",
    234: "OpAtomicIAdd",
    235: "OpAtomicISub",
    236: "OpAtomicSMin",
    237: "OpAtomicUMin",
    238: "OpAtomicSMax",
    239: "OpAtomicUMax",
    240: "OpAtomicAnd",
    241: "OpAtomicOr",
    242: "OpAtomicXor",
    245: "OpPhi",
    246: "OpLoopMerge",
    247: "OpSelectionMerge",
    248: "OpLabel",
    249: "OpBranch",
    250: "OpBranchConditional",
    251: "OpSwitch",
    252: "OpKill",
    253: "OpReturn",
    254: "OpReturnValue",
    255: "OpUnreachable",
    256: "OpLifetimeStart",
    257: "OpLifetimeStop",
    259: "OpGroupAsyncCopy",
    260: "OpGroupWaitEvents",
    261: "OpGroupAll",
    262: "OpGroupAny",
    263: "OpGroupBroadcast",
    264: "OpGroupIAdd",
    265: "OpGroupFAdd",
    266: "OpGroupFMin",
    267: "OpGroupUMin",
    268: "OpGroupSMin",
    269: "OpGroupFMax",
    270: "OpGroupUMax",
    271: "OpGroupSMax",
    274: "OpReadPipe",
    275: "OpWritePipe",
    276: "OpReservedReadPipe",
    277: "OpReservedWritePipe",
    278: "OpReserveReadPipePackets",
    279: "OpReserveWritePipePackets",
    280: "OpCommitReadPipe",
    281: "OpCommitWritePipe",
    282: "OpIsValidReserveId",
    283: "OpGetNumPipePackets",
    284: "OpGetMaxPipePackets",
    285: "OpGroupReserveReadPipePackets",
    286: "OpGroupReserveWritePipePackets",
    287: "OpGroupCommitReadPipe",
    288: "OpGroupCommitWritePipe",
    291: "OpEnqueueMarker",
    292: "OpEnqueueKernel",
    293: "OpGetKernelNDrangeSubGroupCount",
    294: "OpGetKernelNDrangeMaxSubGroupSize",
    295: "OpGetKernelWorkGroupSize",
    296: "OpGetKernelPreferredWorkGroupSizeMultiple",
    297: "OpRetainEvent",
    298: "OpReleaseEvent",
    299: "OpCreateUserEvent",
    300: "OpIsValidEvent",
    301: "OpSetUserEventStatus",
    302: "OpCaptureEventProfilingInfo",
    303: "OpGetDefaultQueue",
    304: "OpBuildNDRange",
    305: "OpImageSparseSampleImplicitLod",
    306: "OpImageSparseSampleExplicitLod",
    307: "OpImageSparseSampleDrefImplicitLod",
    308: "OpImageSparseSampleDrefExplicitLod",
    309: "OpImageSparseSampleProjImplicitLod",
    310: "OpImageSparseSampleProjExplicitLod",
    311: "OpImageSparseSampleProjDrefImplicitLod",
    312: "OpImageSparseSampleProjDrefExplicitLod",
    313: "OpImageSparseFetch",
    314: "OpImageSparseGather",
    315: "OpImageSparseDrefGather",
    316: "OpImageSparseTexelsResident",
    317: "OpNoLine",
    318: "OpAtomicFlagTestAndSet",
    319: "OpAtomicFlagClear",
    320: "OpImageSparseRead",
    321: "OpSizeOf",
    322: "OpTypePipeStorage",
    323: "OpConstantPipeStorage",
    324: "OpCreatePipeFromPipeStorage",
    325: "OpGetKernelLocalSizeForSubgroupCount",
    326: "OpGetKernelMaxNumSubgroups",
    327: "OpTypeNamedBarrier",
    328: "OpNamedBarrierInitialize",
    329: "OpMemoryNamedBarrier",
    330: "OpModuleProcessed",
    331: "OpExecutionModeId",
}

# Execution model
execution_model_to_string = {
    0: "ExecutionModelVertex",
    1: "ExecutionModelTessellationControl",
    2: "ExecutionModelTessellationEvaluation",
    3: "ExecutionModelGeometry",
    4: "ExecutionModelFragment",
    5: "ExecutionModelGLCompute",
    6: "ExecutionModelKernel",
    5267: "ExecutionModelTaskNV",
    5268: "ExecutionModelMeshNV",
    5313: "ExecutionModelRayGenerationNV",
    5313: "ExecutionModelRayGenerationKHR",
    5314: "ExecutionModelIntersectionNV",
    5314: "ExecutionModelIntersectionKHR",
    5315: "ExecutionModelAnyHitNV",
    5315: "ExecutionModelAnyHitKHR",
    5316: "ExecutionModelClosestHitNV",
    5316: "ExecutionModelClosestHitKHR",
    5317: "ExecutionModelMissNV",
    5317: "ExecutionModelMissKHR",
    5318: "ExecutionModelCallableNV",
    5318: "ExecutionModelCallableKHR",
    5364: "ExecutionModelTaskEXT",
    5365: "ExecutionModelMeshEXT",
}

#Execution mode
execution_mode_to_string = {
    0: "ExecutionModeInvocations",
    1: "ExecutionModeSpacingEqual",
    2: "ExecutionModeSpacingFractionalEven",
    3: "ExecutionModeSpacingFractionalOdd",
    4: "ExecutionModeVertexOrderCw",
    5: "ExecutionModeVertexOrderCcw",
    6: "ExecutionModePixelCenterInteger",
    7: "ExecutionModeOriginUpperLeft",
    8: "ExecutionModeOriginLowerLeft",
    9: "ExecutionModeEarlyFragmentTests",
    10: "ExecutionModePointMode",
    11: "ExecutionModeXfb",
    12: "ExecutionModeDepthReplacing",
    14: "ExecutionModeDepthGreater",
    15: "ExecutionModeDepthLess",
    16: "ExecutionModeDepthUnchanged",
    17: "ExecutionModeLocalSize",
    18: "ExecutionModeLocalSizeHint",
    19: "ExecutionModeInputPoints",
    20: "ExecutionModeInputLines",
    21: "ExecutionModeInputLinesAdjacency",
    22: "ExecutionModeTriangles",
    23: "ExecutionModeInputTrianglesAdjacency",
    24: "ExecutionModeQuads",
    25: "ExecutionModeIsolines",
    26: "ExecutionModeOutputVertices",
    27: "ExecutionModeOutputPoints",
    28: "ExecutionModeOutputLineStrip",
    29: "ExecutionModeOutputTriangleStrip",
    30: "ExecutionModeVecTypeHint",
    31: "ExecutionModeContractionOff",
    33: "ExecutionModeInitializer",
    34: "ExecutionModeFinalizer",
    35: "ExecutionModeSubgroupSize",
    36: "ExecutionModeSubgroupsPerWorkgroup",
    37: "ExecutionModeSubgroupsPerWorkgroupId",
    38: "ExecutionModeLocalSizeId",
    39: "ExecutionModeLocalSizeHintId",
    4421: "ExecutionModeSubgroupUniformControlFlowKHR",
    4446: "ExecutionModePostDepthCoverage",
    4459: "ExecutionModeDenormPreserve",
    4460: "ExecutionModeDenormFlushToZero",
    4461: "ExecutionModeSignedZeroInfNanPreserve",
    4462: "ExecutionModeRoundingModeRTE",
    4463: "ExecutionModeRoundingModeRTZ",
    5027: "ExecutionModeStencilRefReplacingEXT",
    5269: "ExecutionModeOutputLinesNV",
    5270: "ExecutionModeOutputPrimitivesNV",
    5289: "ExecutionModeDerivativeGroupQuadsNV",
    5290: "ExecutionModeDerivativeGroupLinearNV",
    5298: "ExecutionModeOutputTrianglesNV",
    5366: "ExecutionModePixelInterlockOrderedEXT",
    5367: "ExecutionModePixelInterlockUnorderedEXT",
    5368: "ExecutionModeSampleInterlockOrderedEXT",
    5369: "ExecutionModeSampleInterlockUnorderedEXT",
    5370: "ExecutionModeShadingRateInterlockOrderedEXT",
    5371: "ExecutionModeShadingRateInterlockUnorderedEXT",
    5618: "ExecutionModeSharedLocalMemorySizeINTEL",
    5620: "ExecutionModeRoundingModeRTPINTEL",
    5621: "ExecutionModeRoundingModeRTNINTEL",
    5622: "ExecutionModeFloatingPointModeALTINTEL",
    5623: "ExecutionModeFloatingPointModeIEEEINTEL",
    5893: "ExecutionModeMaxWorkgroupSizeINTEL",
    5894: "ExecutionModeMaxWorkDimINTEL",
    5895: "ExecutionModeNoGlobalOffsetINTEL",
    5896: "ExecutionModeNumSIMDWorkitemsINTEL",
    5903: "ExecutionModeSchedulerTargetFmaxMhzINTEL",
    6417: "ExecutionModeNamedBarrierCountINTEL",
    0x7fffffff: "ExecutionModeMax",
}

decoration_to_string = {
    0: "DecorationRelaxedPrecision",
    1: "DecorationSpecId",
    2: "DecorationBlock",
    3: "DecorationBufferBlock",
    4: "DecorationRowMajor",
    5: "DecorationColMajor",
    6: "DecorationArrayStride",
    7: "DecorationMatrixStride",
    8: "DecorationGLSLShared",
    9: "DecorationGLSLPacked",
    10: "DecorationCPacked",
    11: "DecorationBuiltIn",
    13: "DecorationNoPerspective",
    14: "DecorationFlat",
    15: "DecorationPatch",
    16: "DecorationCentroid",
    17: "DecorationSample",
    18: "DecorationInvariant",
    19: "DecorationRestrict",
    20: "DecorationAliased",
    21: "DecorationVolatile",
    22: "DecorationConstant",
    23: "DecorationCoherent",
    24: "DecorationNonWritable",
    25: "DecorationNonReadable",
    26: "DecorationUniform",
    27: "DecorationUniformId",
    28: "DecorationSaturatedConversion",
    29: "DecorationStream",
    30: "DecorationLocation",
    31: "DecorationComponent",
    32: "DecorationIndex",
    33: "DecorationBinding",
    34: "DecorationDescriptorSet",
    35: "DecorationOffset",
    36: "DecorationXfbBuffer",
    37: "DecorationXfbStride",
    38: "DecorationFuncParamAttr",
    39: "DecorationFPRoundingMode",
    40: "DecorationFPFastMathMode",
    41: "DecorationLinkageAttributes",
    42: "DecorationNoContraction",
    43: "DecorationInputAttachmentIndex",
    44: "DecorationAlignment",
    45: "DecorationMaxByteOffset",
    46: "DecorationAlignmentId",
    47: "DecorationMaxByteOffsetId",
    4469: "DecorationNoSignedWrap",
    4470: "DecorationNoUnsignedWrap",
    4999: "DecorationExplicitInterpAMD",
    5248: "DecorationOverrideCoverageNV",
    5250: "DecorationPassthroughNV",
    5252: "DecorationViewportRelativeNV",
    5256: "DecorationSecondaryViewportRelativeNV",
    5271: "DecorationPerPrimitiveNV",
    5272: "DecorationPerViewNV",
    5273: "DecorationPerTaskNV",
    5285: "DecorationPerVertexKHR",
    5285: "DecorationPerVertexNV",
    5300: "DecorationNonUniform",
    5300: "DecorationNonUniformEXT",
    5355: "DecorationRestrictPointer",
    5355: "DecorationRestrictPointerEXT",
    5356: "DecorationAliasedPointer",
    5356: "DecorationAliasedPointerEXT",
    5398: "DecorationBindlessSamplerNV",
    5399: "DecorationBindlessImageNV",
    5400: "DecorationBoundSamplerNV",
    5401: "DecorationBoundImageNV",
    5599: "DecorationSIMTCallINTEL",
    5602: "DecorationReferencedIndirectlyINTEL",
    5607: "DecorationClobberINTEL",
    5608: "DecorationSideEffectsINTEL",
    5624: "DecorationVectorComputeVariableINTEL",
    5625: "DecorationFuncParamIOKindINTEL",
    5626: "DecorationVectorComputeFunctionINTEL",
    5627: "DecorationStackCallINTEL",
    5628: "DecorationGlobalVariableOffsetINTEL",
    5634: "DecorationCounterBuffer",
    5634: "DecorationHlslCounterBufferGOOGLE",
    5635: "DecorationHlslSemanticGOOGLE",
    5635: "DecorationUserSemantic",
    5636: "DecorationUserTypeGOOGLE",
    5822: "DecorationFunctionRoundingModeINTEL",
    5823: "DecorationFunctionDenormModeINTEL",
    5825: "DecorationRegisterINTEL",
    5826: "DecorationMemoryINTEL",
    5827: "DecorationNumbanksINTEL",
    5828: "DecorationBankwidthINTEL",
    5829: "DecorationMaxPrivateCopiesINTEL",
    5830: "DecorationSinglepumpINTEL",
    5831: "DecorationDoublepumpINTEL",
    5832: "DecorationMaxReplicatesINTEL",
    5833: "DecorationSimpleDualPortINTEL",
    5834: "DecorationMergeINTEL",
    5835: "DecorationBankBitsINTEL",
    5836: "DecorationForcePow2DepthINTEL",
    5899: "DecorationBurstCoalesceINTEL",
    5900: "DecorationCacheSizeINTEL",
    5901: "DecorationDontStaticallyCoalesceINTEL",
    5902: "DecorationPrefetchINTEL",
    5905: "DecorationStallEnableINTEL",
    5907: "DecorationFuseLoopsInFunctionINTEL",
    5914: "DecorationAliasScopeINTEL",
    5915: "DecorationNoAliasINTEL",
    5921: "DecorationBufferLocationINTEL",
    5944: "DecorationIOPipeStorageINTEL",
    6080: "DecorationFunctionFloatingPointModeINTEL",
    6085: "DecorationSingleElementVectorINTEL",
    6087: "DecorationVectorComputeCallableFunctionINTEL",
    6140: "DecorationMediaBlockIOINTEL",
    0x7fffffff: "DecorationMax",
}


