// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT
import "scene";

public struct ObjectConstants {
    public float3x4 transform;
    public float3x3 normal_matrix;
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

#ifdef VERTEX_NORMALS
    [[vk::location(VERTEX_NORMALS)]]
    float3 normal;
#endif

#ifdef VERTEX_TANGENTS
    [[vk::location(VERTEX_TANGENTS)]]
    float3 tangent;
#endif

#ifdef VERTEX_UVS
    [[vk::location(VERTEX_UVS)]]
    float2 uv;
#endif

#ifdef VERTEX_COLORS
    [[vk::location(VERTEX_COLORS)]]
    uint32_t color;
#endif

#ifdef INSTANCE_POSITIONS
    [[vk::location(INSTANCE_POSITIONS)]]
    float3 instance_position;
#endif
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position;
#ifdef VERTEX_NORMALS
    float3 world_normal;
#endif
#ifdef VERTEX_TANGENTS
    float3 world_tangent;
#endif
#ifdef VERTEX_UVS
    float2 uv;
#endif
#ifdef VERTEX_COLORS
    float4 color;
#endif
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(in.position, 1.0));
#ifdef INSTANCE_POSITIONS
    world_position += in.instance_position;
#endif
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.world_position = world_position;
#ifdef VERTEX_NORMALS
    out.world_normal = mul(g_object_constants.normal_matrix, in.normal);
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = mul(g_object_constants.transform, float4(in.tangent, 0.0));
#endif
#ifdef VERTEX_UVS
    out.uv = in.uv;
#endif
#ifdef VERTEX_COLORS
    out.color = float4(
        (float)((in.color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 24) & 0xFF) * (1.0 / 255.0),
    );
#endif

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
#ifdef VERTEX_NORMALS
    float3 N = normalize(in.world_normal);
#else
    float3 T = ddx(in.world_position);
    float3 B = ddy(in.world_position);
    float3 N = normalize(cross(B, T));
#endif

#ifdef VERTEX_TANGENTS
    float3 T = in.world_tangent;
#else
#ifdef VERTEX_NORMALS
    float3 T = ddx(in.world_position);
#endif
#endif
    // Re-orthogonalize T
    T = normalize(T - dot(T, N) * N);

#ifdef VERTEX_UVS
    float2 uv = in.uv;
#else
    float2 uv = 0.0;
#endif

#ifdef VERTEX_COLORS
    float3 tint = in.color.rgb;
#else
    float3 tint = 1.0;
#endif

    float3 color = g_material.eval(in.world_position, uv, N, T, tint);

    return float4(color, 1.0);
}
