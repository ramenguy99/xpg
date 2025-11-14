// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "ffx_spd_resources.h"

#pragma warning(disable: 30081)  // conversion from larger type to smaller

#if defined(FFX_GPU)
#include "ffx_core.h"

struct SPDConstants
{
    FfxUInt32       mips;
    FfxUInt32       numWorkGroups;
    FfxUInt32x2     workGroupOffset;
    FfxFloat32x2    invInputSize;       // Only used for linear sampling mode
};

[vk::push_constant]
ConstantBuffer<SPDConstants> g_constants;

#if defined(FFX_SPD_BIND_SAMPLER_INPUT_LINEAR_CLAMP)
[vk::binding(FFX_SPD_BIND_SAMPLER_INPUT_LINEAR_CLAMP, 0)]
SamplerState                                                s_LinearClamp;
#endif

#if defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)
[vk::binding(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC, 0)]
Texture2DArray<FfxFloat32x4>                                r_input_downsample_src;
#endif

#if defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)
struct SpdGlobalAtomicBuffer { FfxUInt32 counter[6]; };

[vk::binding(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC, 0)]
globallycoherent RWStructuredBuffer<SpdGlobalAtomicBuffer>  rw_internal_global_atomic;
#endif

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
[vk::binding(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP, 0)]
globallycoherent RWTexture2DArray<FfxFloat32x4>             rw_input_downsample_src_mid_mip;
#endif

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
[vk::binding(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS, 0)]
RWTexture2DArray<FfxFloat32x4>                              rw_input_downsample_src_mips[SPD_MAX_MIP_LEVELS+1];
#endif

FfxUInt32 Mips()
{
    return g_constants.mips;
}

FfxUInt32 NumWorkGroups()
{
    return g_constants.numWorkGroups;
}

FfxUInt32x2  WorkGroupOffset()
{
    return g_constants.workGroupOffset;
}

FfxFloat32x2 InvInputSize()
{
    return g_constants.invInputSize;
}


#if FFX_HALF

#if defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)
    FfxFloat16x4 SampleSrcImageH(FfxFloat32x2 uv, FfxUInt32 slice)
    {
        FfxFloat32x2 textureCoord = FfxFloat32x2(uv) * InvInputSize() + InvInputSize();
        FfxFloat32x4 result = r_input_downsample_src.SampleLevel(s_LinearClamp, FfxFloat32x3(textureCoord, slice), 0);

        // Assume input image view is sRGB if downsampling an sRGB image.
        // No need for any conversion here because sRGB to linear happens in the sampler, if needed.
        return FfxFloat16x4(result.x, result.y, result.z, result.w);
    }
    #endif // defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    FfxFloat16x4 LoadSrcImageH(FfxFloat32x2 uv, FfxUInt32 slice)
    {
        FfxFloat16x4 value = FfxFloat16x4(rw_input_downsample_src_mips[0][FfxUInt32x3(uv, slice)]);
#if defined(FFX_SPD_SRGB)
        return FfxFloat16x4(ffxLinearFromSrgbHalf(value.x), ffxLinearFromSrgbHalf(value.y), ffxLinearFromSrgbHalf(value.z), value.w);
#else
        return value;
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    void StoreSrcMipH(FfxFloat16x4 value, FfxInt32x2 uv, FfxUInt32 slice, FfxUInt32 mip)
    {
        rw_input_downsample_src_mips[mip][FfxUInt32x3(uv, slice)] =
#if defined(FFX_SPD_SRGB)
        FfxFloat32x4(
            ffxSrgbFromLinearHalf(value.x),
            ffxSrgbFromLinearHalf(value.y),
            ffxSrgbFromLinearHalf(value.z),
            value.w
        );
#else
        FfxFloat32x4(value);
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    FfxFloat16x4 LoadMidMipH(FfxInt32x2 uv, FfxUInt32 slice)
    {
        FfxFloat16x4 value = FfxFloat16x4(rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)]);
#if defined(FFX_SPD_SRGB)
        return FfxFloat16x4(ffxLinearFromSrgbHalf(value.x), ffxLinearFromSrgbHalf(value.y), ffxLinearFromSrgbHalf(value.z), value.w);
#else
        return value;
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    void StoreMidMipH(FfxFloat16x4 value, FfxInt32x2 uv, FfxUInt32 slice)
    {
        rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)] =
#if defined(FFX_SPD_SRGB)
        FfxFloat32x4(
            ffxSrgbFromLinearHalf(value.x),
            ffxSrgbFromLinearHalf(value.y),
            ffxSrgbFromLinearHalf(value.z),
            value.w
        );
#else
        FfxFloat32x4(value);
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)

#else // FFX_HALF

#if defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)
    FfxFloat32x4 SampleSrcImage(FfxInt32x2 uv, FfxUInt32 slice)
    {
        FfxFloat32x2 textureCoord = FfxFloat32x2(uv) * InvInputSize() + InvInputSize();
        FfxFloat32x4 result = r_input_downsample_src.SampleLevel(s_LinearClamp, FfxFloat32x3(textureCoord, slice), 0);

        // Assume input image view is sRGB if downsampling an sRGB image.
        // No need for any conversion here because sRGB to linear happens in the sampler, if needed.
        return FfxFloat32x4(result.x, result.y, result.z, result.w);
    }
#endif // defined(FFX_SPD_BIND_SRV_INPUT_DOWNSAMPLE_SRC)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    FfxFloat32x4 LoadSrcImage(FfxInt32x2 uv, FfxUInt32 slice)
    {
        FfxFloat32x4 value = rw_input_downsample_src_mips[0][FfxUInt32x3(uv, slice)];
#if defined(FFX_SPD_SRGB)
        return FfxFloat32x4(ffxLinearFromSrgb(value.x), ffxLinearFromSrgb(value.y), ffxLinearFromSrgb(value.z), value.w);
#else
        return value;
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)
    void StoreSrcMip(FfxFloat32x4 value, FfxInt32x2 uv, FfxUInt32 slice, FfxUInt32 mip)
    {
        rw_input_downsample_src_mips[mip][FfxUInt32x3(uv, slice)] =
#if defined(FFX_SPD_SRGB)
            FfxFloat32x4(
                ffxSrgbFromLinear(value.x),
                ffxSrgbFromLinear(value.y),
                ffxSrgbFromLinear(value.z),
                value.w
            );
#else
            value;
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MIPS)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    FfxFloat32x4 LoadMidMip(FfxInt32x2 uv, FfxUInt32 slice)
    {
        FfxFloat32x4 value = rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)];
#if defined(FFX_SPD_SRGB)
        return FfxFloat32x4(ffxLinearFromSrgb(value.x), ffxLinearFromSrgb(value.y), ffxLinearFromSrgb(value.z), value.w);
#else
        return value;
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)

#if defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)
    void StoreMidMip(FfxFloat32x4 value, FfxInt32x2 uv, FfxUInt32 slice)
    {
        rw_input_downsample_src_mid_mip[FfxUInt32x3(uv, slice)] =
#if defined(FFX_SPD_SRGB)
            FfxFloat32x4(
                ffxSrgbFromLinear(value.x),
                ffxSrgbFromLinear(value.y),
                ffxSrgbFromLinear(value.z),
                value.w
            );
#else
            value;
#endif
    }
#endif // defined(FFX_SPD_BIND_UAV_INPUT_DOWNSAMPLE_SRC_MID_MIPMAP)

#endif // FFX_HALF

#if defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)
void IncreaseAtomicCounter(FFX_PARAMETER_IN FfxUInt32 slice, FFX_PARAMETER_INOUT FfxUInt32 counter)
{
    InterlockedAdd(rw_internal_global_atomic[0].counter[slice], 1, counter);
}
#endif // defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)

#if defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)
void ResetAtomicCounter(FFX_PARAMETER_IN FfxUInt32 slice)
{
    rw_internal_global_atomic[0].counter[slice] = 0;
}
#endif // defined(FFX_SPD_BIND_UAV_INTERNAL_GLOBAL_ATOMIC)

#endif // #if defined(FFX_GPU)
