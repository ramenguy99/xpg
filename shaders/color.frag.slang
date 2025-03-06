struct PushConstants {
    float3 color;
};

[[vk::push_constant]]
PushConstants ps;

[shader("pixel")]
float4 main() : SV_Target0
{
    return float4(1, 0, 0, 1);
}