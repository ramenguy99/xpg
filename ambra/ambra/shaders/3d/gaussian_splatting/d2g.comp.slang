// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

struct PushConstants
{
    float fx, fy, cx, cy;
    float depth_threshold;
    float max_variance;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_constants;

[[vk::binding(0, 0)]] RWTexture2D<uint16_t> depth_img;
[[vk::binding(1, 0)]] RWTexture2D<float3> color_img;

[[vk::binding(2, 0)]] RWStructuredBuffer<float> positions;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> colors;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> covariances;

float3 unproject(
    float x,
    float y,
    float depth,
    float fx,
    float fy,
    float cx,
    float cy,
) {
    float px = (x - cx) * depth / fx;
    float py = (y - cy) * depth / fy;
    float pz = depth;
    return float3(px, py, pz);
}

float projected_sq_dist(float3 p, float3 c, float3 axis) {
    float3 v = p - c;
    float d = dot(v, axis);
    return d * d;
}

static constexpr float NAN = asfloat(0x7FFFFFFF);

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint32_t x = thread_id.x;
    uint32_t y = thread_id.y;

    uint32_t width, height;
    color_img.GetDimensions(width, height);

    uint32_t index = y * width + x;
    if (any(uint2(x, y) >= uint2(width, height))) {
        return;
    }

    float d_c = depth_img[uint2(x, y)] * 1e-3;
    float d_l = depth_img[uint2(x - 1, y)] * 1e-3;
    float d_r = depth_img[uint2(x + 1, y)] * 1e-3;
    float d_t = depth_img[uint2(x, y - 1)] * 1e-3;
    float d_b = depth_img[uint2(x, y + 1)] * 1e-3;

    if (
        abs(d_l - d_c) > g_constants.depth_threshold ||
        abs(d_r - d_c) > g_constants.depth_threshold ||
        abs(d_t - d_c) > g_constants.depth_threshold ||
        abs(d_b - d_c) > g_constants.depth_threshold
    ) {
        positions[index * 3 + 0] = NAN;
        return;
    }

    float fx = g_constants.fx;
    float fy = g_constants.fy;
    float cx = g_constants.cx;
    float cy = g_constants.cy;

    float3 p_c = unproject(x, y, d_c, fx, fy, cx, cy);
    float3 p_l = unproject(x - 1, y, d_l, fx, fy, cx, cy);
    float3 p_r = unproject(x + 1, y, d_r, fx, fy, cx, cy);
    float3 p_t = unproject(x, y - 1, d_t, fx, fy, cx, cy);
    float3 p_b = unproject(x, y + 1, d_b, fx, fy, cx, cy);

    float3 x_v = p_r - p_l;
    float3 y_v = p_t - p_b;

    float3 e1 = normalize(x_v);
    float3 e2 = normalize(y_v);
    e2 = normalize(e2 - dot(e2, e1) * e1);
    float3 e3 = normalize(cross(e1, e2));

    float var_x = min(
        projected_sq_dist(p_l, p_c, e1),
        projected_sq_dist(p_r, p_c, e1)
    );

    float var_y = min(
        projected_sq_dist(p_t, p_c, e2),
        projected_sq_dist(p_b, p_c, e2)
    );

    // # var_x = np.dot(x_v, x_v) * 0.25
    // # var_y = np.dot(y_v, y_v) * 0.25
    // # var_x = min(var_x, 0.05)
    // # var_y = min(var_y, 0.05)

    var_x = min(var_x, g_constants.max_variance);
    var_y = min(var_y, g_constants.max_variance);
    float var_z = 0.0; // z_variance_scale * (var_x + var_y) * 0.5

    float3x3 R = transpose(float3x3(e1, e2, e3));
    float3x3 D = float3x3(
        var_x, 0, 0,
        0, var_y, 0,
        0, 0, var_z,
    );
    float3x3 cov = mul(R, mul(D, transpose(R)));

    positions[index * 3 + 0] = p_c.x;
    positions[index * 3 + 1] = p_c.y;
    positions[index * 3 + 2] = p_c.z;

    float3 color = color_img[uint2(x, y)];
    colors[index * 4 + 0] = color.x;
    colors[index * 4 + 1] = color.y;
    colors[index * 4 + 2] = color.z;
    colors[index * 4 + 3] = 1.0;

    covariances[index * 6 + 0] = cov[0][0];
    covariances[index * 6 + 1] = cov[0][1];
    covariances[index * 6 + 2] = cov[0][2];
    covariances[index * 6 + 3] = cov[1][1];
    covariances[index * 6 + 4] = cov[1][2];
    covariances[index * 6 + 5] = cov[2][2];
}
