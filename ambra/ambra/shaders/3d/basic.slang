// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";

public struct ObjectConstants {
    public float3x4 transform;
#if defined(UNIFORM_COLOR)
    uint32_t color;            // 48
#endif
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float3 pos;

#if defined(VERTEX_COLORS)
    [[vk::location(1)]]
    uint color;
#endif
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
    out.position = mul(g_constants.projection, (mul(g_constants.view, float4(mul(g_object_constants.transform, float4(in.pos, 1.0)), 1.0))));
#ifdef VERTEX_COLORS
    out.color = float4(
        (float)((in.color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 24) & 0xFF) * (1.0 / 255.0),
    );
#elif defined(UNIFORM_COLOR)
    out.color = float4(
        (float)((g_object_constants.color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((g_object_constants.color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((g_object_constants.color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((g_object_constants.color >> 24) & 0xFF) * (1.0 / 255.0),
    );
#else
#endif

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    return in.color;
}
