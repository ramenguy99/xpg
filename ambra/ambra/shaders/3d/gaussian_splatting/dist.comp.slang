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

#include "shaderio.h.slang"
#include "threedgs_particles_storage.h.slang"
#include "../scene.slang"

// clang-format off
[[vk::push_constant]]                        ConstantBuffer<PushConstant> pcRaster;
[[vk::binding(BINDING_DISTANCES_BUFFER, 1)]] RWStructuredBuffer<uint32_t> distances;
[[vk::binding(BINDING_INDICES_BUFFER, 1)]]   RWStructuredBuffer<uint32_t> indices;
#ifdef RASTER_MESH_WORKGROUP_SIZE
[[vk::binding(BINDING_INDIRECT_BUFFER, 1)]]  RWStructuredBuffer<DrawMeshTasksIndirectParams> indirectBuffer;
#else
[[vk::binding(BINDING_INDIRECT_BUFFER, 1)]]  RWStructuredBuffer<DrawIndirectParams> indirectBuffer;
#endif
// clang-format on

// encodes an fp32 into a uint32 that can be ordered
uint encodeMinMaxFp32(float val)
{
  uint bits = asuint(val);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  return bits;
}

[numthreads(DISTANCE_COMPUTE_WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const uint id = dispatchThreadID.x;
  // each workgroup (but the last one if splat count is not a multiple)
  // processes DISTANCE_COMPUTE_WORKGROUP_SIZE points
  if(id >= pcRaster.splatCount)
    return;

  float4 splatPos = float4(fetchCenter(id), 1.0);

  // Row-major: multiply in reverse order (vector * matrix * matrix)
  float4 viewPos      = mul(g_constants.view, float4(mul(pcRaster.modelMatrix, splatPos), 1.0));
  float4 ndcPos       = mul(g_constants.projection, viewPos);
  ndcPos            = ndcPos / ndcPos.w;
  const float depth = ndcPos.z;

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
  // valid only when center is inside NDC clip space.
  // Note: when culling between x=[-1,1] y=[-1,1], which is NDC extent,
  // the culling is not good since we only take into account
  // the center of each splat instead of its extent.

  const float clip = 1.0f + pcRaster.frustumDilation;
  if(abs(ndcPos.x) > clip || abs(ndcPos.y) > clip || ndcPos.z < 0.f - pcRaster.frustumDilation || ndcPos.z > 1.0)
    return;

  // increments the visible splat counter in the indirect buffer
  uint original;
  InterlockedAdd(indirectBuffer[0].instanceCount, 1, original);
  const uint instance_index = original;
#else
  const uint instance_index = id;
#endif

  // stores the distance
  distances[instance_index] = encodeMinMaxFp32(-depth);
  // stores the base index
  indices[instance_index] = id;

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
#ifdef RASTER_MESH_WORKGROUP_SIZE
  // set the workgroup count for the mesh shading pipeline
  if(instance_index % RASTER_MESH_WORKGROUP_SIZE == 0)
  {
    InterlockedAdd(indirectBuffer[0].groupCountX, 1);
  }
#endif
#endif
}
