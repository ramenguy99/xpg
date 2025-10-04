// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";
import "object";

struct VSInput
{
    [[vk::location(0)]]
    float3 position;

#ifdef VERTEX_NORMALS
    [[vk::location(1)]]
    float3 normal;
#endif
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position;
#ifdef VERTEX_NORMALS
    float3 world_normal;
#endif
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    float4 world_position = mul(g_object_constants.transform, float4(in.position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, world_position));
    out.world_position = world_position.xyz;
#ifdef VERTEX_NORMALS
    out.world_normal = mul(g_object_constants.transform, float4(in.normal, 0.0)).xyz;
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

    float3 color = g_material.eval(in.world_position, 0.0, N);

    return float4(color, 1.0);
}
