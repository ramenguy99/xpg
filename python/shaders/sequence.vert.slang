struct VSInput
{
    [[vk::location(0)]]
    float3 position;
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position: POSITION;
};

struct Constants
{
    float4x4 transform;
};

ConstantBuffer<Constants> constants;

[shader("vertex")]
VSOutput main(VSInput in)
{
    VSOutput out;
    out.position = mul(constants.transform, float4(in.position, 1.0));
    out.world_position = in.position;
    return out;
}