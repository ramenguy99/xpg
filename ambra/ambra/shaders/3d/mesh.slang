import "../common";
import "common_object";

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

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;
    out.position = mul(g_constants.projection, (mul(g_constants.view, mul(g_object_constants.transform, float4(in.position, 1.0)))));
    out.world_position = in.position;
    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    float3 L = normalize(float3(-1, 1, 1));

    float3 t = ddx(in.world_position);
    float3 b = ddy(in.world_position);
    float3 N = normalize(cross(t, b));

    float kA = 0.3;
    float kD = max(0.0, dot(N, L));

    float3 color = min(kA + kD, 1.0);
    return float4(color, 1.0);
}