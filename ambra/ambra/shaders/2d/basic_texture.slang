// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";

public struct ObjectConstants {
    public float3x3 transform;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv;
};

[vk::binding(0, 1)]
SamplerState g_sampler;

[vk::binding(1, 1)]
Texture2D g_texture;

[shader("vertex")]
VSOutput vertex_main(uint vertex_id: SV_VertexID)
{
    int x = (vertex_id & 1);
    int y = (vertex_id >> 1);
    float2 uv = float2(x,  y);
    float2 position = uv * 2.0 - 1.0;

    VSOutput out;
    out.position = mul(g_constants.projection, (mul(g_constants.view, float4(mul(g_object_constants.transform, float3(position, 1.0)), 1.0))));
    out.uv = uv;

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    return g_texture.Sample(g_sampler, in.uv);
}
