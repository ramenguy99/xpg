// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";
import "utils";

public struct ObjectConstants {
    public float3x4 transform;
    public float3x3 normal_matrix;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float3 position;

#ifdef VERTEX_NORMALS
    [[vk::location(VERTEX_NORMALS)]]
    float3 normal;
#endif

#ifdef VERTEX_TANGENTS
    [[vk::location(VERTEX_TANGENTS)]]
    float3 tangent;
#endif

#ifdef VERTEX_UVS
    [[vk::location(VERTEX_UVS)]]
    float2 uv;
#endif

#ifdef VERTEX_COLORS
    [[vk::location(VERTEX_COLORS)]]
    uint32_t color;
#endif

#ifdef INSTANCE_TRANSFORMS
    [[vk::location(INSTANCE_TRANSFORMS)]]
    float3x4 instance_transform;
#endif
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position;
#ifdef VERTEX_NORMALS
    float3 world_normal;
#endif
#ifdef VERTEX_TANGENTS
    float3 world_tangent;
#endif
#ifdef VERTEX_UVS
    float2 uv;
#endif
#ifdef VERTEX_COLORS
    float4 color;
#endif
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    float3 object_position = in.position;
#ifdef INSTANCE_TRANSFORMS
    object_position = mul(in.instance_transform, float4(object_position, 1.0));
#endif
    float3 world_position = mul(g_object_constants.transform, float4(object_position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.world_position = world_position;
#ifdef VERTEX_NORMALS
    float3 object_normal = in.normal;
#ifdef INSTANCE_TRANSFORMS
    // TODO: right now assuming no non-uniform scale here (would need to pass this as well or compute inverse here)
    object_normal = mul(float3x3(in.instance_transform), object_normal);
#endif
    out.world_normal = mul(g_object_constants.normal_matrix, object_normal);
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = mul(g_object_constants.transform, float4(in.tangent, 0.0));
#endif
#ifdef VERTEX_UVS
    out.uv = in.uv;
#endif
#ifdef VERTEX_COLORS
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
#ifdef VERTEX_NORMALS
    float3 N = normalize(in.world_normal);
#else
    float3 T = ddx(in.world_position);
    float3 B = ddy(in.world_position);
    float3 N = normalize(cross(B, T));
#endif

#ifdef VERTEX_TANGENTS
    float3 T = in.world_tangent;
#else
#ifdef VERTEX_NORMALS
    float3 T = ddx(in.world_position);
#endif
#endif
    // Re-orthogonalize T
    T = normalize(T - dot(T, N) * N);

#ifdef VERTEX_UVS
    float2 uv = in.uv;
#else
    float2 uv = 0.0;
#endif

#ifdef VERTEX_COLORS
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
