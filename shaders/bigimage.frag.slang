
struct VSOutput
{
    float2 uv: TEXCOORD0;
    uint index: DESCINDEX;
};

[vk::binding(1, 0)]
SamplerState r_sampler;

[vk::binding(2, 0)]
Texture2D<float4> r_textures[];

[shader("pixel")]
float4 computeMain(VSOutput in) : SV_Target0
{
    return r_textures[NonUniformResourceIndex(in.index)].SampleLevel(r_sampler, in.uv, 0).rgba;
    //return float4(in.uv, 0, 0);
}