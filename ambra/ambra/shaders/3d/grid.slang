// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "scene";

// Adapted from "Pristine Grid" by Ben Golus:
// https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8

public struct ObjectConstants {
    float4 major_line_color;     //   0
    float4 minor_line_color;     //  16
    float4 base_color;           //  32
    float2 size;                 //  48
    uint32_t grid_type;          //  56
    float inv_grid_scale;        //  60
    float major_grid_div;        //  64
    float axis_line_width;       //  68
    float major_line_width;      //  72
    float minor_line_width;      //  76
    float pos_axis_color_scale;  //  80
    float neg_axis_color_scale;  //  84
};

[vk::push_constant]
public ConstantBuffer<ObjectConstants> g_object_constants;

struct VSOutput
{
    float4 position: SV_Position;
    float4 uv;
};

static constexpr float3 _XAxisColor = float3(1, 0, 0);
static constexpr float3 _YAxisColor = float3(0, 1, 0);
static constexpr float3 _ZAxisColor = float3(0, 0, 1);

[shader("vertex")]
VSOutput vertex_main(uint vertex_id: SV_VertexID)
{
    int x = (vertex_id & 1);
    int y = (vertex_id >> 1);
    float2 uv = (float2(x, y) - 0.5) * g_object_constants.size;
    float3 world_position;

    if (g_object_constants.grid_type == 0) {
        world_position = float3(uv.x, uv.y, 0);
    } else if (g_object_constants.grid_type == 1) {
        world_position = float3(0, uv.x, uv.y);
    } else {
        world_position = float3(uv.x, 0, uv.y);
    }

    float div = max(2.0, round(g_object_constants.major_grid_div));

    VSOutput out;
    out.position = mul(g_constants.projection, mul(g_constants.view, float4(world_position, 1.0)));

    // trick to reduce visual artifacts when far from the world origin

    float3 cameraCenteringOffset = floor((g_constants.world_camera_position) / div) * div;
    float3 offset = (world_position - cameraCenteringOffset);

    if (g_object_constants.grid_type == 0) {
        out.uv.yx = offset.xy;
        out.uv.wz = world_position.xy;
    } else if (g_object_constants.grid_type == 1) {
        out.uv.yx = offset.yz;
        out.uv.wz = world_position.yz;
    } else {
        out.uv.yx = offset.xz;
        out.uv.wz = world_position.xz;
    }
    out.uv *= g_object_constants.inv_grid_scale;

    return out;
}

[shader("pixel")]
float4 pixel_main(VSOutput i) : SV_Target0
{
    float4 uvDDXY = float4(ddx(i.uv.xy), ddy(i.uv.xy));
    float2 uvDeriv = float2(length(uvDDXY.xz), length(uvDDXY.yw));

    float axis_line_width = max(g_object_constants.major_line_width, g_object_constants.axis_line_width);
    float2 axisDrawWidth = max(axis_line_width, uvDeriv);
    float2 axisLineAA = uvDeriv * 1.5;
    float2 axisLines2 = smoothstep(axisDrawWidth + axisLineAA, axisDrawWidth - axisLineAA, abs(i.uv.zw * 2.0));
    axisLines2 *= saturate(axis_line_width / axisDrawWidth);

    float div = max(2.0, round(g_object_constants.major_grid_div));
    float2 majorUVDeriv = uvDeriv / div;
    float major_line_width = g_object_constants.major_line_width / div;
    float2 majorDrawWidth = clamp(major_line_width, majorUVDeriv, 0.5);
    float2 majorLineAA = majorUVDeriv * 1.5;
    float2 majorGridUV = 1.0 - abs(frac(i.uv.xy / div) * 2.0 - 1.0);
    float2 majorAxisOffset = (1.0 - saturate(abs(i.uv.zw / div * 2.0))) * 2.0;
    majorGridUV += majorAxisOffset; // adjust UVs so center axis line is skipped
    float2 majorGrid2 = smoothstep(majorDrawWidth + majorLineAA, majorDrawWidth - majorLineAA, majorGridUV);
    majorGrid2 *= saturate(major_line_width / majorDrawWidth);
    majorGrid2 = saturate(majorGrid2 - axisLines2); // hack
    majorGrid2 = lerp(majorGrid2, major_line_width, saturate(majorUVDeriv * 2.0 - 1.0));

    float minor_line_width = min(g_object_constants.minor_line_width, g_object_constants.major_line_width);
    bool minorInvertLine = minor_line_width > 0.5;
    float minorTargetWidth = minorInvertLine ? 1.0 - minor_line_width : minor_line_width;
    float2 minorDrawWidth = clamp(minorTargetWidth, uvDeriv, 0.5);
    float2 minorLineAA = uvDeriv * 1.5;
    float2 minorGridUV = abs(frac(i.uv.xy) * 2.0 - 1.0);
    minorGridUV = minorInvertLine ? minorGridUV : 1.0 - minorGridUV;
    float2 minorMajorOffset = (1.0 - saturate((1.0 - abs(frac(i.uv.zw / div) * 2.0 - 1.0)) * div)) * 2.0;
    minorGridUV += minorMajorOffset; // adjust UVs so major division lines are skipped
    float2 minorGrid2 = smoothstep(minorDrawWidth + minorLineAA, minorDrawWidth - minorLineAA, minorGridUV);
    minorGrid2 *= saturate(minorTargetWidth / minorDrawWidth);
    minorGrid2 = saturate(minorGrid2 - axisLines2); // hack
    minorGrid2 = lerp(minorGrid2, minorTargetWidth, saturate(uvDeriv * 2.0 - 1.0));
    minorGrid2 = minorInvertLine ? 1.0 - minorGrid2 : minorGrid2;
    minorGrid2 = abs(i.uv.zw) > 0.5 ? minorGrid2 : 0.0;

    float minorGrid = lerp(minorGrid2.x, 1.0, minorGrid2.y);
    float majorGrid = lerp(majorGrid2.x, 1.0, majorGrid2.y);

    float4 major_line_color = g_object_constants.major_line_color;
    float4 minor_line_color = g_object_constants.minor_line_color;
    float4 base_color = g_object_constants.base_color;

    float4 aAxisColor;
    float4 bAxisColor;
    if (g_object_constants.grid_type == 0) {
        aAxisColor = float4(i.uv.w >= 0 ? _XAxisColor * g_object_constants.pos_axis_color_scale : _XAxisColor * g_object_constants.neg_axis_color_scale, 1.0);
        bAxisColor = float4(i.uv.z >= 0 ? _YAxisColor * g_object_constants.pos_axis_color_scale : _YAxisColor * g_object_constants.neg_axis_color_scale, 1.0);
    } else if (g_object_constants.grid_type == 1) {
        aAxisColor = float4(i.uv.w >= 0 ? _YAxisColor * g_object_constants.pos_axis_color_scale : _YAxisColor * g_object_constants.neg_axis_color_scale, 1.0);
        bAxisColor = float4(i.uv.z >= 0 ? _ZAxisColor * g_object_constants.pos_axis_color_scale : _ZAxisColor * g_object_constants.neg_axis_color_scale, 1.0);
    } else {
        aAxisColor = float4(i.uv.w >= 0 ? _XAxisColor * g_object_constants.pos_axis_color_scale : _XAxisColor * g_object_constants.neg_axis_color_scale, 1.0);
        bAxisColor = float4(i.uv.z >= 0 ? _ZAxisColor * g_object_constants.pos_axis_color_scale : _ZAxisColor * g_object_constants.neg_axis_color_scale, 1.0);
    }

    // center
    aAxisColor = lerp(aAxisColor, major_line_color, axisLines2.y);

    float4 axisLines = lerp(bAxisColor * axisLines2.y, aAxisColor, axisLines2.x);

    float4 col = lerp(base_color, minor_line_color, minorGrid *  minor_line_color.a);
    col = lerp(col, major_line_color, majorGrid * major_line_color.a);
    col = col * (1.0 - axisLines.a) + axisLines;

    return col;
}
