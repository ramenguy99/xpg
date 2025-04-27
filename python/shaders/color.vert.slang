struct VSInput
{
    [[vk::location(0)]]
    float3 position;
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 color: COLOR;
};

struct Nested
{
    float3 val;
    float3 val2;
};

struct Constants
{
    float4x4 transform;
    Nested nest1;
    Nested nest2;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Constants> u;

[shader("vertex")]
VSOutput main(VSInput in)
{
    VSOutput out;
    out.position = mul(u.transform, float4(in.position, 1.0));
    out.color = u.nest1.val2;
    
    return out;
}