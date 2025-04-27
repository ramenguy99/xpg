struct PushConstants {
    float3 color;
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 color: COLOR;
};

[shader("pixel")]
float4 main(VSOutput in) : SV_Target0
{
    return float4(in.color * 2.0, 1.0);
}