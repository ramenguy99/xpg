struct VSOutput
{
    float4 position: SV_Position;
    float3 view_position : POSITION;
};

struct Constants
{
    float4x4 projection;
    float4x4 view;
    float3 camera_pos;
    float size;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Constants> u;

struct Voxel
{
    float3 position;
};

[[vk::binding(1, 0)]]
StructuredBuffer<Voxel, ScalarDataLayout> voxels;

[shader("vertex")]
VSOutput main(uint vertex_index: SV_VertexID)
{
    uint voxel = vertex_index >> 3;

    float3 center = voxels[voxel].position;
    float3 dir = u.camera_pos - center;
    uint3 xyz = uint3((vertex_index & 0x4) >> 2, (vertex_index & 0x2) >> 1, (vertex_index & 0x1) >> 0);
    if (dir.x > 0) xyz.x = 1 - xyz.x;
    if (dir.y > 0) xyz.y = 1 - xyz.y;
    if (dir.z > 0) xyz.z = 1 - xyz.z;

    float3 uvw = float3(xyz);
    float3 world_position = center + (uvw * 2.0 - 1.0) * u.size;
    float4 view_position = mul(u.view, float4(world_position, 1.0));

    VSOutput out;
    out.position = mul(u.projection, view_position);
    out.view_position = view_position.xyz;

    return out;
}