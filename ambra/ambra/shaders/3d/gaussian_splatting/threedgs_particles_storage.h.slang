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

#ifndef _THREEDGS_PARTICLE_STORAGE_H_
#define _THREEDGS_PARTICLE_STORAGE_H_

// clang-format off
// buffers describing the 3DGS model (alternative to textures)
[[vk::binding(BINDING_CENTERS_BUFFER, 1)]] StructuredBuffer<float> centersBuffer;
[[vk::binding(BINDING_COLORS_BUFFER, 1)]] StructuredBuffer<float> colorsBuffer;
[[vk::binding(BINDING_COVARIANCES_BUFFER, 1)]] StructuredBuffer<float> covariancesBuffer;
#if SH_FORMAT == FORMAT_FLOAT32
  [[vk::binding(BINDING_SH_BUFFER, 1)]] StructuredBuffer<float> sphericalHarmonicsBuffer;
#else
#if SH_FORMAT == FORMAT_FLOAT16
  [[vk::binding(BINDING_SH_BUFFER, 1)]] StructuredBuffer<float16_t> sphericalHarmonicsBuffer;
#else
#if SH_FORMAT == FORMAT_UINT8
  [[vk::binding(BINDING_SH_BUFFER, 1)]] StructuredBuffer<uint8_t> sphericalHarmonicsBuffer;
#else
  #error "Unsupported SH format"
#endif
#endif
#endif
// clang-format on

////////////
// constants

static const float sqrt8    = sqrt(8.0);
static const float SH_C1    = 0.4886025119029199f;
static const float SH_C2[5] = { 1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742 };
static const float SH_C3[7] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f };

// data texture accessors
int2 getDataPos(in uint splatIndex, in uint stride, in uint offset, in int2 dimensions)
{
  const uint fullOffset = splatIndex * stride + offset;

  return int2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

// data texture accessors
int2 getDataPosF(in uint splatIndex, in float stride, in uint offset, in int2 dimensions)
{
  const uint fullOffset = uint(float(splatIndex) * stride) + offset;

  return int2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

// fetch center value from data buffer
float3 fetchCenter(in uint splatIndex)
{
  return float3(centersBuffer[splatIndex * 3 + 0], centersBuffer[splatIndex * 3 + 1], centersBuffer[splatIndex * 3 + 2]);
}

// fetch center value from data buffer
float4 fetchColor(in uint splatIndex)
{
  return float4(colorsBuffer[splatIndex * 4 + 0], colorsBuffer[splatIndex * 4 + 1], colorsBuffer[splatIndex * 4 + 2],
                colorsBuffer[splatIndex * 4 + 3]);
}

// fetch from data buffers
void fetchSh(in uint    splatIndex,
             out float3 shd1[3]
#if MAX_SH_DEGREE >= 2
             ,
             out float3 shd2[5]
#endif
#if MAX_SH_DEGREE >= 3
             ,
             out float3 shd3[7]
#endif
)
{
  const uint splatStride = 45;

  const float SphericalHarmonics8BitCompressionRange     = 2.0;
  const float SphericalHarmonics8BitCompressionHalfRange = SphericalHarmonics8BitCompressionRange / 2.0;
  const float3 vec8BitSHShift = float3(SphericalHarmonics8BitCompressionHalfRange, SphericalHarmonics8BitCompressionHalfRange,
                                       SphericalHarmonics8BitCompressionHalfRange);
  const float SphericalHarmonics8BitScale = SphericalHarmonics8BitCompressionRange / 255.0f;

  // fetching degree 1
  const float3 sh1 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 2]);

  const float3 sh2 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 2]);

  const float3 sh3 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd1[0] = sh1;
  shd1[1] = sh2;
  shd1[2] = sh3;
#else
  shd1[0] = sh1 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd1[1] = sh2 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd1[2] = sh3 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif

  // fetching degree 2
#if MAX_SH_DEGREE >= 2
  const float3 sh4 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 2]);

  const float3 sh5 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 2]);

  const float3 sh6 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 2]);

  const float3 sh7 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 2]);

  const float3 sh8 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd2[0] = sh4;
  shd2[1] = sh5;
  shd2[2] = sh6;
  shd2[3] = sh7;
  shd2[4] = sh8;
#else
  shd2[0] = sh4 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[1] = sh5 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[2] = sh6 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[3] = sh7 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[4] = sh8 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif
#endif

  // fetching degree 3
#if MAX_SH_DEGREE >= 3
  const float3 sh9 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 2]);

  const float3 sh10 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 2]);

  const float3 sh11 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 2]);

  const float3 sh12 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 2]);

  const float3 sh13 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 2]);

  const float3 sh14 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 2]);

  const float3 sh15 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd3[0] = sh9;
  shd3[1] = sh10;
  shd3[2] = sh11;
  shd3[3] = sh12;
  shd3[4] = sh13;
  shd3[5] = sh14;
  shd3[6] = sh15;
#else
  shd3[0] = sh9 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[1] = sh10 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[2] = sh11 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[3] = sh12 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[4] = sh13 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[5] = sh14 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[6] = sh15 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif
#endif
}

float3x3 fetchCovariance(in uint splatIndex)
{
  // Use RGBA texture map to store sets of 3 elements requires some offset shifting depending on splatIndex
  const float3 cov3D_M11_M12_M13 = float3(covariancesBuffer[splatIndex * 6 + 0], covariancesBuffer[splatIndex * 6 + 1],
                                          covariancesBuffer[splatIndex * 6 + 2]);
  const float3 cov3D_M22_M23_M33 = float3(covariancesBuffer[splatIndex * 6 + 3], covariancesBuffer[splatIndex * 6 + 4],
                                          covariancesBuffer[splatIndex * 6 + 5]);

  return float3x3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
                  cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
}

float3 fetchViewDependentRadiance(in uint splatIndex, in float3 worldViewDir)
{
  // contribution is null if MAX_SH_DEGREE < 1
  float3 rgb = float3(0.0, 0.0, 0.0);

#if MAX_SH_DEGREE >= 1
  // SH coefficients for degree 1 (1,2,3)
  float3 shd1[3];
#if MAX_SH_DEGREE >= 2
  // SH coefficients for degree 2 (4 5 6 7 8)
  float3 shd2[5];
#endif
#if MAX_SH_DEGREE >= 3
  // SH coefficients for degree 3 (9,10,11,12,13,14,15)
  float3 shd3[7];
#endif
  // fetch the data (only what is needed according to degree)
  fetchSh(splatIndex, shd1
#if MAX_SH_DEGREE >= 2
          ,
          shd2
#endif
#if MAX_SH_DEGREE >= 3
          ,
          shd3
#endif
  );

  const float x = worldViewDir.x;
  const float y = worldViewDir.y;
  const float z = worldViewDir.z;

  // Degree 1 contributions
  rgb += SH_C1 * (-shd1[0] * y + shd1[1] * z - shd1[2] * x);

#if MAX_SH_DEGREE >= 2
  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float yz = y * z;
  const float xz = x * z;

  // Degree 2 contributions
  rgb += (SH_C2[0] * xy) * shd2[0] + (SH_C2[1] * yz) * shd2[1] + (SH_C2[2] * (2.0 * zz - xx - yy)) * shd2[2]
         + (SH_C2[3] * xz) * shd2[3] + (SH_C2[4] * (xx - yy)) * shd2[4];
#endif
#if MAX_SH_DEGREE >= 3

  const float xyy = x * yy;
  const float yzz = y * zz;
  const float zxx = z * xx;
  const float xyz = x * y * z;

  // Degree 3 contributions
  rgb += SH_C3[0] * shd3[0] * (3.0 * x * x - y * y) * y + SH_C3[1] * shd3[1] * x * y * z
         + SH_C3[2] * shd3[2] * (4.0 * z * z - x * x - y * y) * y
         + SH_C3[3] * shd3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
         + SH_C3[4] * shd3[4] * x * (4.0 * z * z - x * x - y * y) + SH_C3[5] * shd3[5] * (x * x - y * y) * z
         + SH_C3[6] * shd3[6] * x * (x * x - 3.0 * y * y);
#endif
#endif

  return rgb;
}

#endif
