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

[vk::binding(0, 1)]
#if defined(MATERIAL_COLOR)
public ConstantBuffer<ColorMaterial> g_material;
#elif defined(MATERIAL_DIFFUSE)
public ConstantBuffer<DiffuseMaterial> g_material;
#elif defined(MATERIAL_DIFFUSE_SPECULAR)
public ConstantBuffer<DiffuseSpecularMaterial> g_material;
#elif defined(MATERIAL_PBR)
public ConstantBuffer<PBRMaterial> g_material;
#else
public ConstantBuffer<EmptyMaterial> g_material;
#endif

struct VSInput
{
    [[vk::location(0)]]
    float3 position;

    [[vk::location(1)]]
    float3 center;

#ifdef COLORS
    [[vk::location(2)]]
    uint color;
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
    float3 position = in.center + in.position * g_object_constants.radius;
    float3 normal = in.position;

    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.world_position = world_position;
    out.world_normal = mul(g_object_constants.normal_matrix, normal);

#ifdef COLORS
    uint color = in.color;
    out.color = float4(
        (float)((color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((color >> 24) & 0xFF) * (1.0 / 255.0),
    );
#endif

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    float3 N = normalize(in.world_normal);

    float3 T = float3(1, 0, 0);
    float2 uv = 0.0;

#ifdef COLORS
    float3 tint = in.color.rgb;
#else
    float3 tint = 1.0;
#endif

    float3 color = g_material.eval(in.world_position, uv, N, T, tint);
    return float4(color, 1.0);
}
