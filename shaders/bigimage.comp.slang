[vk::binding(0, 0)]
[vk::image_format("rgba8i")]
RWTexture2D<float4> r_output;

[vk::binding(1, 0)]
SamplerState r_sampler;

[vk::binding(2, 0)]
Texture2D<float4> r_texture;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint width, height;
    r_output.GetDimensions(width, height);

    if(all(threadId.xy < uint2(width, height))) {
        float2 uv = float2(threadId.x + 0.5, threadId.y + 0.5) / float2(width, height);
        // r_output[threadId.xy] = float4(uv.x, uv.y, 0, 1);
        r_output[threadId.xy] = r_texture.SampleLevel(r_sampler, uv, 0).rgba;
    }
}
