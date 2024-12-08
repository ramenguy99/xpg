#version 450 core

struct VSInput
{
    [[vk::location(0)]]
    vec3 pos;

    [[vk::location(1)]]
    vec2 uv;
};

struct VSOutput
{
    float2 uv: TEXCOORD0;
    float4 position: SV_Position;
};

[shader("vertex")]
VSOutput main(VSInput in)
{
    VSOutput out;
    out.position = float4(in.pos, 1.0);
    out.uv = in.uv;

    return out;
}
