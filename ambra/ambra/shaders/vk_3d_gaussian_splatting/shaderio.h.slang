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

#ifndef _DEVICE_HOST_H_
#define _DEVICE_HOST_H_

// type of method used for sorting
#define SORTING_GPU_SYNC_RADIX 0
#define SORTING_CPU_ASYNC_MONO 1
#define SORTING_CPU_ASYNC_MULTI 2

// type of model storage
#define STORAGE_BUFFERS 0
#define STORAGE_TEXTURES 1

// format for SH storage
#define FORMAT_FLOAT32 0
#define FORMAT_FLOAT16 1
#define FORMAT_UINT8 2

// type of pipeline used
#define PIPELINE_VERT 0
#define PIPELINE_MESH 1
#define PIPELINE_RTX 2
#define PIPELINE_HYBRID 3        // Hybrid rendering: raster primary rays (3DGS), raytrace secondary rays (3DGRT)
#define PIPELINE_MESH_3DGUT 4    // 3DGUT (Unscented Transform) rasterization using mesh shaders
#define PIPELINE_HYBRID_3DGUT 5  // Hybrid rendering: raster primary rays (3DGUT), raytrace secondary rays (3DGRT)

// visualization mode
#define VISUALIZE_FINAL 0
#define VISUALIZE_CLOCK 1
#define VISUALIZE_DEPTH 2
#define VISUALIZE_RAYHITS 3

// type of frustum culling
#define FRUSTUM_CULLING_NONE 0
#define FRUSTUM_CULLING_AT_DIST 1
#define FRUSTUM_CULLING_AT_RASTER 2

// method used to compute the 2D extent projection from the 3D covariance
#define EXTENT_EIGEN 0  // basis aligned rectangular extent
#define EXTENT_CONIC 1  // axis aligned rectangular extend as in original INRIA

// type of camera
#define CAMERA_PINHOLE 0
#define CAMERA_FISHEYE 1

// particle format (PF), RTX
// not used in shaders (using RTX_USE_AABBS compiler defined instead)
// used only by UI but here to be easier to find
#define PARTICLE_FORMAT_ICOSAHEDRON 0
#define PARTICLE_FORMAT_PARAMETRIC 1

// degree of the splat kernel, RTX
#define KERNEL_DEGREE_QUINTIC 5
#define KERNEL_DEGREE_TESSERACTIC 4
#define KERNEL_DEGREE_CUBIC 3
#define KERNEL_DEGREE_QUADRATIC 2
#define KERNEL_DEGREE_LAPLACIAN 1
#define KERNEL_DEGREE_LINEAR 0

// bindings for set 0 (common to Raster and RTX)
// #define BINDING_FRAME_INFO_UBO 0

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

#define DEFAULT(val)

// Push constant for raster
struct PushConstant
{
  // model transformation
  float3x4 modelMatrix;
  float3x4 modelMatrixInverse;

  // Splat count
  uint32_t splatCount;
  float alphaCullThreshold;
  float focal;
  float splatScale;

  float2 basisViewport;
  float inverseFocalAdjustment;
  float frustumDilation;
};

// indirect parameters for
// - vkCmdDrawIndexedIndirect (first 6 attr)
// - vkCmdDrawMeshTasksIndirectEXT (last 3 attr)
struct IndirectParams
{
  // for vkCmdDrawIndexedIndirect
  uint32_t indexCount    DEFAULT(6);  // allways = 6 indices for the quad (2 triangles)
  uint32_t instanceCount DEFAULT(0);  // will be incremented by the distance compute shader
  uint32_t firstIndex    DEFAULT(0);  // allways zero
  uint32_t vertexOffset  DEFAULT(0);  // allways zero
  uint32_t firstInstance DEFAULT(0);  // allways zero

  // for vkCmdDrawMeshTasksIndirectEXT
  uint32_t groupCountX DEFAULT(0);  // Will be incremented by the distance compute shader
  uint32_t groupCountY DEFAULT(1);  // Allways one workgroup on Y
  uint32_t groupCountZ DEFAULT(1);  // Allways one workgroup on Z

  // for debug info readback, TODO shall be in some other buffer
  int32_t particleID DEFAULT(-1);   // Will be set by the ID of the nearest splat on the ray path
  float particleDist DEFAULT(0.0);  // Will be set by the dist to the nearest splat on the ray path
  float val1         DEFAULT(0.0);
  float val2         DEFAULT(0.0);
  float val3         DEFAULT(0.0);
  float val4         DEFAULT(0.0);
  float val5         DEFAULT(0.0);
  float val6         DEFAULT(0.0);
  float val7         DEFAULT(0.0);
  float val8         DEFAULT(0.0);
};


#undef DEFAULT
#endif
