// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";
import "colormap";
import "utils";

public struct ObjectConstants {
    public float3x4 transform; //  0
    public float point_size;   // 48
#ifdef COLORMAP
    uint32_t colormap_measure; // 52 - Colormap and measure (distance to point / plane)
    float range_min;           // 56
    float range_inv_delta;     // 60
    float3 point_or_normal;    // 64
    float alpha;               // 76
#elif defined(UNIFORM_COLOR)
    uint32_t color;            // 52
#endif
                               // 80
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float3 pos;

#ifdef VERTEX_COLORS
    [[vk::location(1)]]
    uint color;
#endif
};



struct VSOutput
{
    float4 position: SV_Position;
    float4 color;
    float point_size : SV_PointSize;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in, uint instance_id: SV_InstanceID)
{
    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(in.pos, 1.0));
    out.position = mul(g_constants.projection, (mul(g_constants.view, float4(world_position, 1.0))));

#ifdef VERTEX_COLORS
    out.color = uint32_srgb_to_float4_linear_fast(in.color);
#elif defined(COLORMAP)
    uint32_t measure = g_object_constants.colormap_measure >> 16;
    float value = 0.5;
    if (measure == 0) {
        value = distance(world_position, g_object_constants.point_or_normal);
    } else if (measure == 1) {
        value = dot(world_position, g_object_constants.point_or_normal);
    }
    float remapped_value = (value - g_object_constants.range_min) * g_object_constants.range_inv_delta;
    ColormapId colormap_id = (ColormapId)(g_object_constants.colormap_measure & 0xFFFF);

    out.color = float4(srgb_to_linear_fast(colormap(colormap_id, remapped_value)), g_object_constants.alpha);
#elif defined(UNIFORM_COLOR)
    out.color = uint32_srgb_to_float4_linear_fast(g_object_constants.color);
#endif

    out.point_size = g_object_constants.point_size;
    return out;
}


struct PSOutput {
#ifdef DUAL_DEPTH_PEELING
    float2 depth: SV_Target0;
    float4 front: SV_Target1;
    float4 back: SV_Target2;
#else
    float4 color: SV_Target0;
#ifdef WBOIT
    float revealage: SV_Target1;
#endif
#endif
}

[shader("pixel")]
PSOutput pixel_main(
    float4 position: SV_Position,
    float4 color,
#ifdef DUAL_DEPTH_PEELING_MSAA
    , uint sample_index:  SV_SampleIndex
#endif
)
{
    float4 color = color;

    PSOutput out;
#ifdef WBOIT
    out_wboit(color, out.color, out.revealage);
#elif defined(DUAL_DEPTH_PEELING)
    out_dual_depth_peeling(color, position.xyz,
#ifdef DUAL_DEPTH_PEELING_MSAA
        sample_index,
#endif
        out.depth, out.front, out.back);
#else
    out.color = color;
#endif


    return out;
}
