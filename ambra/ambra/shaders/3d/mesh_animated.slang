import "../common";
import "common_object";

struct VSInput
{
    [[vk::location(0)]]
    float3 position;
    [[vk::location(1)]]
    float3 normal;
    [[vk::location(2)]]
    float3 tangent;
    [[vk::location(3)]]
    float2 uv;
    [[vk::location(4)]]
    uint4 joint_indices;
    [[vk::location(5)]]
    float4 weight;
};

struct VSOutput
{
    float4 position: SV_Position;
    float3 world_position: POSITION;
    float3 world_normal : NORMAL;
    float2 uv : TEXCOORD;
};

[vk::binding(0, 2)]
SamplerState g_sampler;

[vk::binding(1, 2)]
Texture2D g_texture;

[vk::binding(2, 2)]
StructuredBuffer<float4x4> g_joints;

[shader("vertex")]
VSOutput vertex_main(VSInput in)
{
    VSOutput out;

    float3 skinned_position =
        (mul(g_joints[in.joint_indices.x], float4(in.position, 1)) * in.weight.x).xyz +
        (mul(g_joints[in.joint_indices.y], float4(in.position, 1)) * in.weight.y).xyz +
        (mul(g_joints[in.joint_indices.z], float4(in.position, 1)) * in.weight.z).xyz +
        (mul(g_joints[in.joint_indices.w], float4(in.position, 1)) * in.weight.w).xyz;

    out.position = mul(g_constants.projection, (mul(g_constants.view, mul(g_object_constants.transform, float4(skinned_position, 1.0)))));
    out.world_normal = mul(g_object_constants.transform, float4(in.normal, 0.0)).xyz;
    out.world_position = in.position;
    out.uv = in.uv;
    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    float3 L = normalize(float3(1, 1, 1));
    float3 N = normalize(in.world_normal);

    float kA = 0.2;
    float kD = max(0.0, dot(N, L)) * 1.0;

    float3 color = min(kA + kD, 1.0) * g_texture.Sample(g_sampler, in.uv).rgb;
    return float4(color, 1.0);
}
