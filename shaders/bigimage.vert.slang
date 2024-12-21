#version 450 core

struct VSInput
{
    [[vk::location(0)]]
    vec2 pos;
};

struct VSOutput
{
    vec2 uv: TEXCOORD0;
    uint index: DESCINDEX;
    float4 position: SV_Position;
};

struct PushConstants {
    vec2 scale;
    vec2 inv_window_size;
    vec2 inv_scale;
    uint frame_id;
    uint flags;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> r_constants;

struct Chunk {
    vec2 position;
    uint desc_index;
    uint _padding;
};

[[vk::binding(0, 0)]]
StructuredBuffer<Chunk> r_chunks[];


[shader("vertex")]
VSOutput main(VSInput in, uint instance_id: SV_InstanceID)
{
    Chunk chunk = r_chunks[r_constants.frame_id][instance_id];
    vec2 chunk_position = chunk.position;

    VSOutput out;
    out.position = vec4((((in.pos * r_constants.scale + chunk_position) + 0.5) * r_constants.inv_window_size) * 2.0 - 1.0, 0.0, 1.0);
    //out.position = vec4((((in.pos * r_constants.scale) + 0.5) * r_constants.inv_window_size) * 2.0 - 1.0, 0.0, 1.0);
    out.uv = in.pos;
    out.index = chunk.desc_index;

    return out;
}
