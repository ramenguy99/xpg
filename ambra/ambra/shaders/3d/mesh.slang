// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";
import "object";

struct VSInput
{
    [[vk::location(0)]]
    float3 position;
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position: POSITION;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    float4 world_position = mul(g_object_constants.transform, float4(in.position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, world_position));
    out.world_position = world_position.xyz;
    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    float3 t = ddx(in.world_position);
    float3 b = ddy(in.world_position);
    float3 N = normalize(cross(t, b));

    float3 L = float3(0, 0, 0);
    for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
        LightEvaluation l = g_directional_lights[i].eval(in.world_position, N);
        L += l.L * max(dot(l.direction, N), 0.0);
    }

    float3 color = L;
    return float4(color, 1.0);
}
