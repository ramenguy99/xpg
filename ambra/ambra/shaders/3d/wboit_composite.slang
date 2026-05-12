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
Texture2D g_accum_texture;

[vk::binding(1, 0)]
Texture2D g_revealage_texture;

float max_component(float3 v) {
    return max(max(v.x, v.y), v.z);
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    int3 C = int3(int2(in.position.xy), 0);
    float revealage = g_revealage_texture.Load(C).r;

    if (revealage >= 0.9999f) {
        discard;
    }

    float4 accum = g_accum_texture.Load(C);

    if (isinf(max_component(abs(accum.rgb)))) {
        accum.rgb = float3(accum.a, accum.a, accum.a);
    }

    float3 finalColor = accum.rgb / max(accum.a, 0.00001f);
    return float4(finalColor, 1.0 - revealage);
}
