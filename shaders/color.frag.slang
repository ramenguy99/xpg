struct PushConstants {
    float3 color;
};

[[vk::push_constant]]
PushConstants ps;

struct VSOutput
{
    float4 position: SV_Position;
    float3 color: COLOR;
};

[shader("pixel")]
float4 main(VSOutput in) : SV_Target0
{
    return float4(in.color, 1.0);
}