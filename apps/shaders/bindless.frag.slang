
struct PushConstants {
    uint index;
};

[[vk::push_constant]]
PushConstants ps;

// Also works with this to be more explicit
// ConstantBuffer<PushConstants> ps;

struct VSOutput
{
    float2 uv: TEXCOORD0;
};

[[vk::binding(0, 0)]]
StructuredBuffer<float4> buffers[];

[shader("pixel")]
float4 computeMain(VSOutput in) : SV_Target0
{
    return buffers[ps.index][0];
    //return float4(in.uv, 0, 0);
}