// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

module scene_depth;

public struct Constants {
    public float4x4 camera_matrix;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;
