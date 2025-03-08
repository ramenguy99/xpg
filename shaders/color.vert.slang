struct VSInput
{
    [[vk::location(0)]]
    float3 position;
};

struct VSOutput
{
    float4 position: SV_Position;
};

struct Constants
{
    float4x4 transform;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Constants> u;

[[vk::binding(0, 1)]]
StructuredBuffer<Constants> b;

[shader("vertex")]
VSOutput main(VSInput in)
{
    VSOutput out;
    out.position = mul(u.transform, float4(in.position, 1.0));
    return out;
}