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

#ifdef INSTANCE_POSITIONS
    [[vk::location(INSTANCE_POSITIONS)]]
    float3 instance_position;
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
    float3 world_position = mul(g_object_constants.transform, float4(in.position, 1.0));
#ifdef INSTANCE_POSITIONS
    world_position += in.instance_position;
#endif
    out.position = mul(g_constants.camera_matrix, float4(world_position, 1.0));
    return out;
}
