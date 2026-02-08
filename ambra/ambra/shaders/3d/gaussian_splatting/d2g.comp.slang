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
    float depth_threshold;
    float max_variance;
    uint use_eigen_decomposition;
    float relative_ratio_threshold;
    float var_z;
    uint z_mask;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> g_constants;

[[vk::binding(0, 0)]] RWTexture3D<uint16_t> depth_img;
[[vk::binding(1, 0)]] RWTexture3D<float3> color_img;

[[vk::binding(2, 0)]] RWStructuredBuffer<float> positions;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> colors;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> covariances;

struct Calibration
{
    float fx, fy, cx, cy;
    float3x4 extrinsics;
};

[[vk::binding(5, 0)]] StructuredBuffer<Calibration> calibration;

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

float3 ComputeEigenvector0(float3x3 A, float eval0) {
    float3 row0 = float3(A[0][0] - eval0, A[0][1], A[0][2]);
    float3 row1 = float3(A[0][1], A[1][1] - eval0, A[1][2]);
    float3 row2 = float3(A[0][2], A[1][2], A[2][2] - eval0);
    float3 r0xr1 = cross(row0, row1);
    float3 r0xr2 = cross(row0, row2);
    float3 r1xr2 = cross(row1, row2);
    float d0 = dot(r0xr1, r0xr1);
    float d1 = dot(r0xr2, r0xr2);
    float d2 = dot(r1xr2, r1xr2);

    float dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }

    if (imax == 0) {
        return r0xr1 / sqrt(d0);
    } else if (imax == 1) {
        return r0xr2 / sqrt(d1);
    } else {
        return r1xr2 / sqrt(d2);
    }
}

float3 ComputeEigenvector1(float3x3 A, float3 evec0, float eval1) {
    float3 U, V;
    if (abs(evec0[0]) > abs(evec0[1])) {
        float inv_length =
                1 / sqrt(evec0[0] * evec0[0] + evec0[2] * evec0[2]);
        U = float3(-evec0[2] * inv_length, 0, evec0[0] * inv_length);
    } else {
        float inv_length =
                1 / sqrt(evec0[1] * evec0[1] + evec0[2] * evec0[2]);
        U = float3(0, evec0[2] * inv_length, -evec0[1] * inv_length);
    }
    V = cross(evec0, U);

    float3 AU = float3(A[0][0] * U[0] + A[0][1] * U[1] + A[0][2] * U[2],
                       A[0][1] * U[0] + A[1][1] * U[1] + A[1][2] * U[2],
                       A[0][2] * U[0] + A[1][2] * U[1] + A[2][2] * U[2]);

    float3 AV = float3(A[0][0] * V[0] + A[0][1] * V[1] + A[0][2] * V[2],
                       A[0][1] * V[0] + A[1][1] * V[1] + A[1][2] * V[2],
                       A[0][2] * V[0] + A[1][2] * V[1] + A[2][2] * V[2]);

    float m00 = U[0] * AU[0] + U[1] * AU[1] + U[2] * AU[2] - eval1;
    float m01 = U[0] * AV[0] + U[1] * AV[1] + U[2] * AV[2];
    float m11 = V[0] * AV[0] + V[1] * AV[1] + V[2] * AV[2] - eval1;

    float absM00 = abs(m00);
    float absM01 = abs(m01);
    float absM11 = abs(m11);
    float max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            return m01 * U - m00 * V;
        } else {
            return U;
        }
    } else {
        max_abs_comp = max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            return m11 * U - m01 * V;
        } else {
            return U;
        }
    }
}

float max_coeff(float3x3 m)
{
    float3 row_max = float3(
        max(max(m[0].x, m[0].y), m[0].z),
        max(max(m[1].x, m[1].y), m[1].z),
        max(max(m[2].x, m[2].y), m[2].z)
      );
    return max(max(row_max.x, row_max.y), row_max.z);
}

void FastEigen3x3(float3x3 covariance,
    out float3 eigenvalues,
    out float3x3 eigenvectors,
) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    float3x3 A = covariance;
    float max_coeff = max_coeff(A);
    if (max_coeff == 0) {
        eigenvectors = float3x3(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        );
        eigenvalues = float3(0);
        return;
    }

    A /= max_coeff;

    float norm = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
    if (norm > 0) {
        float3 evals;
        float3x3 evecs;

        float q = (A[0][0] + A[1][1] + A[2][2]) / 3;

        float b00 = A[0][0] - q;
        float b11 = A[1][1] - q;
        float b22 = A[2][2] - q;

        float p = sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        float c00 = b11 * b22 - A[1][2] * A[1][2];
        float c01 = A[0][1] * b22 - A[1][2] * A[0][2];
        float c02 = A[0][1] * A[1][2] - b11 * A[0][2];
        float det = (b00 * c00 - A[0][1] * c01 + A[0][2] * c02) / (p * p * p);

        float half_det = det * 0.5;
        half_det = min(max(half_det, -1.0), 1.0);

        float angle = acos(half_det) / 3.0;
        float two_thirds_pi = 2.09439510239319549;
        float beta2 = cos(angle) * 2;
        float beta0 = cos(angle + two_thirds_pi) * 2;
        float beta1 = -(beta0 + beta2);

        evals.x = q + p * beta0;
        evals.y = q + p * beta1;
        evals.z = q + p * beta2;

        if (half_det >= 0) {
            evecs[2] = ComputeEigenvector0(A, evals[2]);
            evecs[1] = ComputeEigenvector1(A, evecs[2], evals[1]);
            evecs[0] = cross(evecs[1], evecs[2]);
        } else {
            evecs[0] = ComputeEigenvector0(A, evals[0]);
            evecs[1] = ComputeEigenvector1(A, evecs[0], evals[1]);
            evecs[2] = cross(evecs[0], evecs[1]);
        }
        eigenvalues = evals;
        eigenvectors = evecs;
    } else {
        eigenvectors = float3x3(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        );
        eigenvalues = float3(0);
    }
}

// Eigen decomposition for 3x3 symmetric matrix using Cardano's formula
void eigen_decomposition_symmetric3x3(
    float3x3 A,
    out float3 eigenvalues,
    out float3x3 eigenvectors
) {
    // For a symmetric 3x3 matrix, we use the analytical solution
    // Based on: https://www.mpi-hd.mpg.de/personalhomes/globes/3x3/

    float m = (A[0][0] + A[1][1] + A[2][2]) / 3.0;

    float3x3 K = A - float3x3(m, 0, 0, 0, m, 0, 0, 0, m);

    float q = determinant(K) / 2.0;

    float p = (K[0][0] * K[0][0] + K[1][1] * K[1][1] + K[2][2] * K[2][2] +
               2.0 * (K[0][1] * K[0][1] + K[0][2] * K[0][2] + K[1][2] * K[1][2])) / 6.0;

    float phi;
    if (p < 1e-10) {
        phi = 0.0;
    } else {
        float r = q / (p * sqrt(p));
        r = clamp(r, -1.0, 1.0);
        phi = acos(r) / 3.0;
    }

    float sqrt_p = sqrt(p);
    eigenvalues[0] = m + 2.0 * sqrt_p * cos(phi);
    eigenvalues[1] = m + 2.0 * sqrt_p * cos(phi - 2.0 * 3.14159265359 / 3.0);
    eigenvalues[2] = m + 2.0 * sqrt_p * cos(phi + 2.0 * 3.14159265359 / 3.0);

    // Sort eigenvalues in descending order
    if (eigenvalues[1] > eigenvalues[0]) {
        float tmp = eigenvalues[0];
        eigenvalues[0] = eigenvalues[1];
        eigenvalues[1] = tmp;
    }
    if (eigenvalues[2] > eigenvalues[0]) {
        float tmp = eigenvalues[0];
        eigenvalues[0] = eigenvalues[2];
        eigenvalues[2] = tmp;
    }
    if (eigenvalues[2] > eigenvalues[1]) {
        float tmp = eigenvalues[1];
        eigenvalues[1] = eigenvalues[2];
        eigenvalues[2] = tmp;
    }

    // Compute eigenvectors via (A - lambda*I) null space
    for (int i = 0; i < 3; i++) {
        float3x3 B = A - float3x3(eigenvalues[i], 0, 0, 0, eigenvalues[i], 0, 0, 0, eigenvalues[i]);

        // Find the eigenvector as cross product of two rows of B
        float3 r0 = float3(B[0][0], B[0][1], B[0][2]);
        float3 r1 = float3(B[1][0], B[1][1], B[1][2]);
        float3 r2 = float3(B[2][0], B[2][1], B[2][2]);

        float3 c01 = cross(r0, r1);
        float3 c02 = cross(r0, r2);
        float3 c12 = cross(r1, r2);

        float len01 = dot(c01, c01);
        float len02 = dot(c02, c02);
        float len12 = dot(c12, c12);

        float3 v;
        if (len01 >= len02 && len01 >= len12) {
            v = c01;
        } else if (len02 >= len01 && len02 >= len12) {
            v = c02;
        } else {
            v = c12;
        }

        float len = length(v);
        if (len > 1e-10) {
            v = v / len;
        } else {
            // Fallback for degenerate case
            if (i == 0) v = float3(1, 0, 0);
            else if (i == 1) v = float3(0, 1, 0);
            else v = float3(0, 0, 1);
        }

        eigenvectors[i] = v;
    }

    // Ensure right-handed coordinate system
    if (dot(cross(eigenvectors[0], eigenvectors[1]), eigenvectors[2]) < 0) {
        eigenvectors[2] = -eigenvectors[2];
    }
}

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint32_t x = thread_id.x;
    uint32_t y = thread_id.y;
    uint32_t z = thread_id.z;

    uint32_t width, height, depth;
    color_img.GetDimensions(width, height, depth);

    uint32_t index = ((z * height) + y) * width + x;
    if (any(uint2(x, y) <= uint2(1, 1)) || any(uint2(x, y) >= uint2(width - 1, height - 1)) || ((1 << z) & g_constants.z_mask) == 0) {
        positions[index * 3 + 0] = NAN;
        return;
    }

    float d_c = depth_img[uint3(x, y, z)] * 1e-3;
    float d_l = depth_img[uint3(x - 1, y, z)] * 1e-3;
    float d_r = depth_img[uint3(x + 1, y, z)] * 1e-3;
    float d_t = depth_img[uint3(x, y - 1, z)] * 1e-3;
    float d_b = depth_img[uint3(x, y + 1, z)] * 1e-3;

    if (
        abs(d_l - d_c) > g_constants.depth_threshold ||
        abs(d_r - d_c) > g_constants.depth_threshold ||
        abs(d_t - d_c) > g_constants.depth_threshold ||
        abs(d_b - d_c) > g_constants.depth_threshold
    ) {
        positions[index * 3 + 0] = NAN;
        return;
    }

    float fx = calibration[z].fx;
    float fy = calibration[z].fy;
    float cx = calibration[z].cx;
    float cy = calibration[z].cy;

    float3 p_c = unproject(x, y, d_c, fx, fy, cx, cy);
    float3 p_l = unproject(x - 1, y, d_l, fx, fy, cx, cy);
    float3 p_r = unproject(x + 1, y, d_r, fx, fy, cx, cy);
    float3 p_t = unproject(x, y - 1, d_t, fx, fy, cx, cy);
    float3 p_b = unproject(x, y + 1, d_b, fx, fy, cx, cy);

    float3 e1, e2, e3;
    float var_x, var_y, var_z;

    if (g_constants.use_eigen_decomposition != 0) {
        // Eigen decomposition method: compute basis from covariance of direction vectors
        float3 d0 = p_l - p_c;
        float3 d1 = p_r - p_c;
        float3 d2 = p_t - p_c;
        float3 d3 = p_b - p_c;

        // Build covariance matrix from the 4 directions: M = (1/4) * sum(d_i * d_i^T)
        float3x3 cov_directions = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
        cov_directions[0][0] = d0.x * d0.x + d1.x * d1.x + d2.x * d2.x + d3.x * d3.x;
        cov_directions[0][1] = d0.x * d0.y + d1.x * d1.y + d2.x * d2.y + d3.x * d3.y;
        cov_directions[0][2] = d0.x * d0.z + d1.x * d1.z + d2.x * d2.z + d3.x * d3.z;
        cov_directions[1][0] = cov_directions[0][1];
        cov_directions[1][1] = d0.y * d0.y + d1.y * d1.y + d2.y * d2.y + d3.y * d3.y;
        cov_directions[1][2] = d0.y * d0.z + d1.y * d1.z + d2.y * d2.z + d3.y * d3.z;
        cov_directions[2][0] = cov_directions[0][2];
        cov_directions[2][1] = cov_directions[1][2];
        cov_directions[2][2] = d0.z * d0.z + d1.z * d1.z + d2.z * d2.z + d3.z * d3.z;
        cov_directions = cov_directions * 0.25;

        // Eigen decomposition to get principal axes and variances
        float3 eigenvalues;
        float3x3 eigenvectors;
        eigen_decomposition_symmetric3x3(cov_directions, eigenvalues, eigenvectors);
        eigenvectors = transpose(eigenvectors);
        // FastEigen3x3(cov_directions, eigenvalues, eigenvectors);

        e1 = eigenvectors[0];
        e2 = eigenvectors[1];
        e3 = eigenvectors[2];

        // Eigenvalues are the variances along each principal axis
        var_x = eigenvalues[0];
        var_y = eigenvalues[1];
        var_z = eigenvalues[2];
    } else {
        // Gram-Schmidt method: compute basis from horizontal/vertical differences
        float3 x_v = p_r - p_l;
        float3 y_v = p_t - p_b;

        e1 = normalize(x_v);
        e2 = normalize(y_v);
        e2 = normalize(e2 - dot(e2, e1) * e1);
        e3 = normalize(cross(e1, e2));

        var_x = min(
            projected_sq_dist(p_l, p_c, e1),
            projected_sq_dist(p_r, p_c, e1)
        );
        var_y = min(
            projected_sq_dist(p_t, p_c, e2),
            projected_sq_dist(p_b, p_c, e2)
        );
        var_z = g_constants.var_z * g_constants.var_z;
    }

    if (min(var_x, var_y) / max(var_x, var_y) < g_constants.relative_ratio_threshold) {
        positions[index * 3 + 0] = NAN;
        return;
    }

    var_x = min(var_x, g_constants.max_variance);
    var_y = min(var_y, g_constants.max_variance);
    var_z = min(var_z, g_constants.max_variance);

    float3x4 extrinsics = calibration[z].extrinsics;

    float3 world_p = mul(extrinsics, float4(p_c, 1.0));
    positions[index * 3 + 0] = world_p.x;
    positions[index * 3 + 1] = world_p.y;
    positions[index * 3 + 2] = world_p.z;

    float3 color = color_img[uint3(x, y, z)];
    colors[index * 4 + 0] = color.x;
    colors[index * 4 + 1] = color.y;
    colors[index * 4 + 2] = color.z;
    colors[index * 4 + 3] = 1.0;

    // TODO: check if should be right-multiplied instead
    float3x3 R = mul(float3x3(extrinsics), transpose(float3x3(e1, e2, e3)));
    float3x3 D = float3x3(
        var_x, 0, 0,
        0, var_y, 0,
        0, 0, var_z,
    );
    float3x3 cov = mul(R, mul(D, transpose(R)));
    covariances[index * 6 + 0] = cov[0][0];
    covariances[index * 6 + 1] = cov[0][1];
    covariances[index * 6 + 2] = cov[0][2];
    covariances[index * 6 + 3] = cov[1][1];
    covariances[index * 6 + 4] = cov[1][2];
    covariances[index * 6 + 5] = cov[2][2];
}
