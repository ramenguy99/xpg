struct VSInput
{
    [[vk::location(0)]]
    float2 pos;

    [[vk::location(1)]]
    uint color;
};

struct VSOutput
{
    float4 position: SV_Position;
    float4 color;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in, uint instance_id: SV_InstanceID)
{
    VSOutput out;
    out.position = float4(in.pos, 0, 1.0);
    out.color = float4(
        (float)((in.color >> 24) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >>  8) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >>  0) & 0xFF) * (1.0 / 255.0),
    );

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    return in.color;
}