// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT
import "scene";

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

#ifdef VERTEX_UVS
    [[vk::location(VERTEX_UVS)]]
    float2 uv;
#endif
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position;
#ifdef VERTEX_NORMALS
    float3 world_normal;
#endif
#ifdef VERTEX_UVS
    float2 uv;
#endif
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(in.position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.world_position = world_position;
#ifdef VERTEX_NORMALS
    out.world_normal = mul(g_object_constants.normal_matrix, in.normal);
#endif
#ifdef VERTEX_UVS
    out.uv = in.uv;
#endif

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
#ifdef VERTEX_NORMALS
    float3 N = normalize(in.world_normal);
#else
    float3 t = ddx(in.world_position);
    float3 b = ddy(in.world_position);
    float3 N = normalize(cross(b, t));
#endif

#ifdef VERTEX_UVS
    float2 uv = in.uv;
#else
    float2 uv = 0.0;
#endif

    float3 color = g_material.eval(in.world_position, uv, N);

    return float4(color, 1.0);
}
