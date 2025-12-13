import "scene";
import "colormap";

struct VSInput
{
    [[vk::location(0)]]
    float3 pos;

#ifdef VERTEX_COLORS
    [[vk::location(1)]]
    uint color;
#endif
};

struct VSOutput
{
    float4 position: SV_Position;
    float4 color;
    float3 uvw;
};

public struct ObjectConstants {
    public float3x4 transform;     //  0
    float3 object_camera_position; // 48
    public float voxel_size;       // 60
    float border_size;             // 64
    float border_factor;           // 68
#ifdef COLORMAP
    uint32_t colormap_measure; // 72 - Colormap and measure (distance to point / plane)
    float range_min;           // 76
    float3 point_or_normal;    // 80
    float range_inv_delta;     // 92
#elif defined(UNIFORM_COLOR)
    uint32_t color;            // 72
#endif
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

[shader("vertex")]
VSOutput vertex_main(VSInput in, uint vertex_index: SV_VertexID)
{
    VSOutput out;

    float3 center = in.pos;
    float3 dir = g_object_constants.object_camera_position - center;
    uint3 xyz = uint3((vertex_index & 0x4) >> 2, (vertex_index & 0x2) >> 1, (vertex_index & 0x1) >> 0);
    if (dir.x > 0) xyz.x = 1 - xyz.x;
    if (dir.y > 0) xyz.y = 1 - xyz.y;
    if (dir.z > 0) xyz.z = 1 - xyz.z;

    float3 uvw = float3(xyz);
    float3 world_position = mul(g_object_constants.transform, float4(center + (uvw - 0.5) * g_object_constants.voxel_size, 1.0));
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));

#ifdef VERTEX_COLORS
    out.color = float4(
        (float)((in.color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((in.color >> 24) & 0xFF) * (1.0 / 255.0),
    );
#elif defined(COLORMAP)
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
#elif defined(UNIFORM_COLOR)
    out.color = float4(
        (float)((g_object_constants.color >> 0) & 0xFF) * (1.0 / 255.0),
        (float)((g_object_constants.color >> 8) & 0xFF) * (1.0 / 255.0),
        (float)((g_object_constants.color >> 16) & 0xFF) * (1.0 / 255.0),
        (float)((g_object_constants.color >> 24) & 0xFF) * (1.0 / 255.0),
    );
#else
#endif

    out.uvw = uvw - 0.5;
    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput in) : SV_Target0
{
    float3 abs_uvw = abs(in.uvw);
    float border_size = g_object_constants.border_size;
    if (border_size > 0.0) {
        float3 width = fwidth(in.uvw) + 0.01;
        float3 wir = smoothstep( 0.5 - width * 0.5 - border_size, 0.5 - border_size + width * 0.5, abs(in.uvw));
        float vvv = ((1.0-wir.x*wir.y)*(1.0-wir.x*wir.z)*(1.0-wir.y*wir.z) * (1.0 - g_object_constants.border_factor) + g_object_constants.border_factor);
        return float4(in.color.rgb * vvv, in.color.a);
    } else {
        return in.color;
    }
}
