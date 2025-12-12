import "scene";
import "colormap";

struct VSInput
{
    [[vk::location(0)]]
    float3 pos;
};

struct VSOutput
{
    float4 position: SV_Position;
    float4 color;
};

public struct ObjectConstants {
    public float3x4 transform; //  0
    public float voxel_size;   // 48
    uint32_t colormap_measure; // 52 - Colormap and measure (distance to point / plane)
    float range_min;           // 56
    float range_inv_delta;     // 60
    float3 point_or_normal;    // 64
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct Voxel
{
    float3 position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput in, uint vertex_index: SV_VertexID)
{
    VSOutput out;

    uint voxel = vertex_index >> 3;

    float3 center = in.pos;
    float3 dir = g_constants.world_camera_position - center;
    uint3 xyz = uint3((vertex_index & 0x4) >> 2, (vertex_index & 0x2) >> 1, (vertex_index & 0x1) >> 0);
    if (dir.x > 0) xyz.x = 1 - xyz.x;
    if (dir.y > 0) xyz.y = 1 - xyz.y;
    if (dir.z > 0) xyz.z = 1 - xyz.z;

    float3 uvw = float3(xyz);
    float3 world_position = center + (uvw * 2.0 - 1.0) * g_object_constants.voxel_size;

    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));

    uint32_t measure = g_object_constants.colormap_measure >> 16;
    float value = 0.5;
    if (measure == 0) {
        value = distance(center, g_object_constants.point_or_normal);
    } else if (measure == 1) {
        value = dot(center, g_object_constants.point_or_normal);
    }
    float remapped_value = (value - g_object_constants.range_min) * g_object_constants.range_inv_delta;
    ColormapId colormap_id = (ColormapId)(g_object_constants.colormap_measure & 0xFFFF);

    out.color = float4(colormap(colormap_id, remapped_value), 1.0);
    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    return in.color;
}