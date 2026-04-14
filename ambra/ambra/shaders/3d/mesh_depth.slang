// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene_depth";

public struct ObjectConstants {
    public float3x4 transform;
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSInput
{
    [[vk::location(0)]]
    float3 position;

#ifdef INSTANCE_TRANSFORMS
    [[vk::location(INSTANCE_TRANSFORMS)]]
    float3x4 instance_transform;
#endif
};

struct VSOutput
{
    float4 position: SV_Position;
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
    out.position = mul(g_constants.camera_matrix, float4(world_position, 1.0));
    return out;
}
