// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene_depth";
import "utils";

public struct ObjectConstants {
    public float3x4 transform;
    public float3x3 normal_matrix;
    public float radius;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float3 position;

    [[vk::location(1)]]
    float3 l0;

    [[vk::location(2)]]
    float3 l1;
};

struct VSOutput
{
    float4 position: SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    float3 p0 = in.l0;
    float3 p1 = in.l1;

    float3 p = in.position;
    p.xy *= g_object_constants.radius;

    float d = length(p1 - p0);
    p.z *= d;

    float3 v = normalize(p1 - p0);
    Frame tbn = frame_from_normal(v);

    float3 position = to_world(tbn, p) + p0;

    VSOutput out;
    float3 world_position = mul(g_object_constants.transform, float4(position, 1.0));
    out.position = mul(g_constants.camera_matrix, float4(world_position, 1.0));
    return out;
}
