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

// format for SH storage
#define FORMAT_FLOAT32 0
#define FORMAT_FLOAT16 1
#define FORMAT_UINT8 2

// type of frustum culling
#define FRUSTUM_CULLING_NONE 0
#define FRUSTUM_CULLING_AT_DIST 1
#define FRUSTUM_CULLING_AT_RASTER 2

// bindings for set 1
#define BINDING_DISTANCES_BUFFER 0
#define BINDING_INDICES_BUFFER 1
#define BINDING_INDIRECT_BUFFER 2
#define BINDING_CENTERS_BUFFER 3
#define BINDING_COLORS_BUFFER 4
#define BINDING_COVARIANCES_BUFFER 5
#define BINDING_SH_BUFFER 6

// location for vertex attributes
// (only for vertex shader mode)
#define ATTRIBUTE_LOC_POSITION 0
#define ATTRIBUTE_LOC_SPLAT_INDEX 1
// used for mesh rasterization
#define ATTRIBUTE_LOC_MESH_POSITION 0
#define ATTRIBUTE_LOC_MESH_NORMAL 1

enum RenderFlags: uint32_t
{
  DisableOpacity = 1 << 0,
  ShowSphericalHarmonicsOnly = 1 << 1,
  OrthographicMode = 1 << 2,
  PointCloudMode = 1 << 3,
}

// Push constant for raster
struct PushConstant
{
  // model transformation
  float3x4 modelMatrix;
  float3x4 modelMatrixInverse;

  // Splat count
  uint32_t splatCount;
  float alphaCullThreshold;
  float frustumDilation;
  float splatScale;
  RenderFlags flags;
};

// for vkCmdDrawIndexedIndirect
struct DrawIndirectParams
{
  // for vkCmdDrawIndexedIndirect
  uint32_t indexCount;  // always = 6 indices for the quad (2 triangles)
  uint32_t instanceCount;  // will be incremented by the distance compute shader
  uint32_t firstIndex;  // always zero
  uint32_t vertexOffset;  // always zero
  uint32_t firstInstance;  // always zero
};

// for vkCmdDrawMeshTasksIndirectEXT
struct DrawMeshTasksIndirectParams
{
  uint32_t groupCountX;   // will be incremented by the distance compute shader
  uint32_t groupCountY;   // always one workgroup on Y
  uint32_t groupCountZ;   // always one workgroup on Z
  uint32_t instanceCount; // will be incremented by the distance compute shader
};
