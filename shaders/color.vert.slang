#version 450 core

struct VSInput
{
    [[vk::location(0)]]
    vec3 position;
};

struct VSOutput
{
    float4 position: SV_Position;
};

struct Constants
{
    mat4 transform;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Constants> u;

[shader("vertex")]
VSOutput main(VSInput in)
{
    VSOutput out;
    out.position = u.transform * vec4(in.position, 1.0);
    return out;
}