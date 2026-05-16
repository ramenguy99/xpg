// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT
import "scene";
import "utils";

public struct ObjectConstants {
    public float3x4 transform;
    public float3x3 normal_matrix;
    public float radius;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float3 position;

    [[vk::location(1)]]
    float3 center;

#ifdef COLORS
    [[vk::location(2)]]
    uint color;
#endif
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position;
    float3 world_normal;
#ifdef COLORS
    float4 color;
#endif
};

[shader("vertex")]
VSOutput vertex_main(VSInput in
    , uint vertex_index : SV_VertexID
    , uint instance_index : SV_InstanceID
) {
    float3 position = in.center + in.position * g_object_constants.radius;
    float3 normal = in.position;

    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.world_position = world_position;
    out.world_normal = mul(g_object_constants.normal_matrix, normal);

#ifdef COLORS
    out.color = uint32_srgb_to_float4_linear_fast(in.color);
#endif

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
PSOutput pixel_main(VSOutput in
#ifdef DUAL_DEPTH_PEELING_MSAA
    , uint sample_index:  SV_SampleIndex
#endif
)
{
    float3 N = normalize(in.world_normal);

    float3 T = float3(1, 0, 0);
    float2 uv = 0.0;

#ifdef COLORS
    float4 tint = in.color;
#else
    float4 tint = 1.0;
#endif

    float4 color = g_material.eval(in.world_position, uv, N, T, tint);

    PSOutput out;
#ifdef WBOIT
    out_wboit(color, out.color, out.revealage);
#elif defined(DUAL_DEPTH_PEELING)
    out_dual_depth_peeling(color, in.position.xyz,
#ifdef DUAL_DEPTH_PEELING_MSAA
        sample_index,
#endif
        out.depth, out.front, out.back);
#else
    out.color = color;
#endif


    return out;
}
