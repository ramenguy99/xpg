
struct VSOutput
{
    float2 uv: TEXCOORD0;
    uint index: DESCINDEX;
};

struct PushConstants {
    float2 scale;
    float2 inv_window_size;
    float2 inv_scale;
    uint frame_id;
    uint flags;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> r_constants;

[vk::binding(1, 0)]
SamplerState r_sampler;

[vk::binding(2, 0)]
Texture2D<float4> r_textures[];

[shader("pixel")]
float4 computeMain(VSOutput in) : SV_Target0
{
    float4 color = r_textures[NonUniformResourceIndex(in.index)].SampleLevel(r_sampler, in.uv, 0).rgba;
    float2 inv_scale = r_constants.inv_scale * 2.0;
    bool show_grid = (bool)(r_constants.flags & 1);
    if(show_grid && (in.uv.x < inv_scale.x || in.uv.x > 1.0 - inv_scale.x || in.uv.y < inv_scale.y || in.uv.y > 1.0 - inv_scale.y)) {
        color = saturate(color + float4(0.5, 0, 0, 0));
    }
    return color;
    //return float4(in.uv, 0, 0);
}