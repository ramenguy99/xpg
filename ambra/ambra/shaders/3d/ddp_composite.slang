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
Texture2D g_front_texture;

[vk::binding(1, 0)]
Texture2D g_back_texture;

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    int3 C = int3(int2(in.position.xy), 0);
    float4 back = g_back_texture.Load(C);
    float4 front = g_front_texture.Load(C);

    if (back.a == 0.0 && front.a == 1.0) {
        discard;
    }

    return float4(front.rgb + front.a * back.rgb, 1.0 - front.a * (1.0 - back.a));
}
