// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";

public struct ObjectConstants {
    public float3x4 transform;
    public float3x3 normal_matrix;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

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

#ifdef INSTANCE_TRANSFORMS
    [[vk::location(INSTANCE_TRANSFORMS)]]
    float3x4 instance_transform;
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
    float3 object_position = in.position;
#ifdef INSTANCE_TRANSFORMS
    object_position = mul(in.instance_transform, float4(object_position, 1.0));
#endif
    float3 world_position = mul(g_object_constants.transform, float4(object_position, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));
    out.world_position = world_position;
#ifdef VERTEX_NORMALS
    float3 object_normal = in.normal;
#ifdef INSTANCE_TRANSFORMS
    // TODO: right now assuming no non-uniform scale here (would need to pass this as well or compute inverse here)
    object_normal = mul(float3x3(in.instance_transform), object_normal);
#endif
    out.world_normal = mul(g_object_constants.normal_matrix, object_normal);
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

struct PSOutput {
#ifdef DUAL_DEPTH_PEELING
    float2 depth: SV_Target0;
    float4 front: SV_Target1;
    float4 back: SV_Target2;
#else
    float4 color: SV_Target0;
#ifdef WBOIT
    float revealage: SV_Target1;
#endif
#endif
}

[shader("pixel")]
PSOutput pixel_main(VSOutput in)
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
    float4 tint = in.color;
#else
    float4 tint = 1.0;
#endif

    float4 color = g_material.eval(in.world_position, uv, N, T, tint);

    PSOutput out;
#ifdef WBOIT
    // Depth and color based weight functions:
    //
    // float w =
    //     max(min(1.0, max(max(color.r, color.g), color.b) * color.a), color.a)  // color term
    //     * clamp(0.03 / (1e-5 + pow(z / 200, 4.0)), 1e-2, 3e3)                  // depth term
    // ;
    //
    // float w = clamp(pow(min(1.0, color.a * 10.0) + 0.01, 3.0) * 1e8 *  pow(1.0 - z * 0.9, 3.0), 1e-2, 3e3);
    //
    // float tmp = (premultipliedReflect.a * 8.0 + 0.01) *
    //              (-gl_FragCoord.z * 0.95 + 1.0);
    // If a lot of the scene is close to the far plane, then gl_FragCoord.z does not
    // provide enough discrimination. Add this term to compensate:
    // // tmp /= sqrt(abs(csZ));
    // float w = clamp(tmp * tmp * tmp * 1e3, 1e-2, 3e2);

    float w = 1.0;
    out.color = float4(color.rgb * color.a, color.a) * w;
    out.revealage = color.a;

#elif defined(DUAL_DEPTH_PEELING)
    static constexpr float bias = 0.000001;
    int3 C = int3(int2(in.position.xy), 0);
    float z = in.position.z;

    float2 depth_buffers = g_dual_depth_peeling_depth.Load(C).rg;
    float front_depth = -depth_buffers.r;
    float back_depth = depth_buffers.g;

    out.depth = float2(-1, -1);
    out.front = 0.0;
    out.back = 0.0;

    if (z < front_depth || z > back_depth) {
    // if (z + bias < front_depth || z - bias > back_depth) {
        // Closer than front, or furhter than back
        discard;
    }
    else if (z > front_depth && z < back_depth) {
    // else if (z - bias > front_depth && z + bias < back_depth) {
        // Between front and back
        out.depth = float2(-z, z);
    }
    else if (z == front_depth) {
        // else if (z == back_depth) {
        // else if (z < front_depth + bias && z > front_depth - bias) {
        // Exactly at front (or at both front and back for middle layer)
        out.front = float4(color.rgb * color.a, color.a);
    } else {
        // Exacty at back
        out.back = color;
    }
#else
    out.color = color;
#endif


    return out;
}
