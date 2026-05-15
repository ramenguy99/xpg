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
    float3 l0;

    [[vk::location(2)]]
    float3 l1;

    [[vk::location(3)]]
    float3 normal;

#ifdef COLORS
    [[vk::location(4)]]
    uint c0;

    [[vk::location(5)]]
    uint c1;
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
    float3 p0 = in.l0;
    float3 p1 = in.l1;

    float3 p = in.position;
    p.xy *= g_object_constants.radius;

    float d = length(p1 - p0);
    p.z *= d;

    float3 v = normalize(p1 - p0);
    Frame tbn = frame_from_normal(v);

    float3 position = to_world(tbn, p) + p0;
    float3 normal = to_world(tbn, in.normal);

    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.world_position = world_position;
    out.world_normal = mul(g_object_constants.normal_matrix, normal);

#ifdef COLORS
    uint color = in.position.z < 0.5 ? in.c0 : in.c1;
    out.color = float4(
        (float)((color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((color >> 24) & 0xFF) * (1.0 / 255.0),
    );
#endif

    return out;
}

struct PSOutput {
    float4 color: SV_Target0;
#ifdef WBOIT
    float revealage: SV_Target1;
#endif
}

[shader("pixel")]
PSOutput pixel_main(VSOutput in)
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
    float w = 1.0;
    out.color = float4(color.rgb * color.a, color.a) * w;
    out.revealage = color.a;
#else
    out.color = color;
#endif

    return out;
}
