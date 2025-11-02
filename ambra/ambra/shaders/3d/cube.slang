// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";
import "utils";

public struct ObjectConstants {
    public float3x4 transform;
    public uint level;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

[vk::binding(0, 1)]
public SamplerCube g_cubemap;

struct VSInput
{
    [[vk::location(0)]]
    float3 position;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 local_position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(in.position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.local_position = in.position;
    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    float3 N = normalize(in.local_position);
        // float3 color = N * 0.5 + 0.5;

    float3 color = g_cubemap.SampleLevel(N, g_object_constants.level).rgb;

    color = color / (color + 1);
    color = linear_to_srgb(color);


    return float4(color, 1.0);
}
