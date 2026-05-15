// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT


struct VSOutput
{
    float4 position: SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(uint vertex_id: SV_VertexID)
{
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);

    VSOutput output;
    output.position = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}

[vk::binding(0, 0)]
Texture2D g_opaque_depth;

[shader("pixel")]
float2 pixel_main(VSOutput in) : SV_Target0
{
    int3 C = int3(int2(in.position.xy), 0);
    float depth = g_opaque_depth.Load(C).r;
    return float2(1.0, depth);
}
