// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene_depth";
import "object";

struct VSInput
{
    [[vk::location(0)]]
    float3 position;
};

struct VSOutput
{
    float4 position: SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    out.position = mul(g_constants.camera_matrix, mul(g_object_constants.transform, float4(in.position, 1.0)));
    return out;
}
