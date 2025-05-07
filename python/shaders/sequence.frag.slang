struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position: POSITION;
};

[shader("pixel")]
float4 main(VSOutput in) : SV_Target0
{
    float3 L = normalize(float3(-1, -1, 1));

    float3 t = ddx(in.world_position);
    float3 b = ddy(in.world_position);
    float3 N = normalize(cross(t, b));

    float kA = 0.3;
    float kD = max(0.0, dot(N, L));

    float3 color = min(kA + kD, 1.0);
    return float4(color, 1.0);
}