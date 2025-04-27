struct VSOutput
{
    float4 position: SV_Position;
    float3 view_position : POSITION;
};

[shader("pixel")]
float4 main(VSOutput in) : SV_Target0
{
    float3 x_tangent = ddx_fine(in.view_position);
    float3 y_tangent = ddy_fine(in.view_position);
    
    float3 N = normalize(cross(y_tangent, x_tangent));
    float3 L = normalize(float3(0.2, 0.5, -1));

    float diffuse = clamp(dot(N, L) + 0.1, 0, 1);
    float3 color = float3(0.0, 1.0, 0.0);

    return float4(color * diffuse, 1.0);
}