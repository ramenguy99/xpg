// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";

public struct ObjectConstants {
    public float3x4 transform; // 0
    public float point_size;   // 48

#ifdef COLORMAP
    uint32_t kind;    // 52 - Colormap and measure (distance to point / plane)
    float min_value;  // 56
    float max_value;  // 60
    vec4 plane;       // 64
                      // 80
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float3 pos;

#ifdef VERTEX_COLORS
    [[vk::location(1)]]
    uint color;
#endif
};



struct VSOutput
{
    float4 position: SV_Position;
    float4 color;
    float point_size : SV_PointSize;
};

float3 colormap_jet(float t)
{
    return clamp((float3(1.5) - abs(4.0*float3(t) + float3(-3,-2,-1))), 0., 1.);
}

[shader("vertex")]
VSOutput vertex_main(VSInput in, uint instance_id: SV_InstanceID)
{
    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(in.pos, 1.0));
    out.position = mul(g_constants.projection, (mul(g_constants.view, float4(world_position, 1.0))));
    out.color = float4(
        (float)((in.color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 24) & 0xFF) * (1.0 / 255.0),
    );
    out.point_size = g_object_constants.point_size;

    out.color.xyz = Map_Jet_JoshuaFraser(clamp((world_position.z + 0.0) / 0.4, 0, 1));

    return out;
}


[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    return in.color;
}
