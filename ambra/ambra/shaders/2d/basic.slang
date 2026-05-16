// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";
import "../3d/utils";

public struct ObjectConstants {
    public float3x3 transform;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float2 pos;

    [[vk::location(1)]]
    uint color;
};

struct VSOutput
{
    float4 position: SV_Position;
    float4 color;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in, uint instance_id: SV_InstanceID)
{
    VSOutput out;
    out.position = mul(g_constants.projection, (mul(g_constants.view, float4(mul(g_object_constants.transform, float3(in.pos, 1.0)).xy, 0.0, 1.0))));
    out.color = uint32_srgb_to_float4_linear_fast(in.color);

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    return in.color;
}
