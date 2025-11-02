// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "utils";

static const float3x3 k_face_rotations[6] = {
    float3x3(
        0, 0, 1,
        0, -1, 0,
        -1, 0, 0,
    ),
    float3x3(
        0, 0, -1,
        0, -1, 0,
        1, 0, 0,
    ),
    float3x3(
        1, 0, 0,
        0, 0, 1,
        0, 1, 0,
    ),
    float3x3(
        1, 0, 0,
        0, 0, -1,
        0, -1, 0,
    ),
    float3x3(
        1, 0, 0,
        0, -1, 0,
        0, 0, 1,
    ),
    float3x3(
        -1, 0, 0,
        0, -1, 0,
        0, 0, -1,
    ),
};

public struct SkyboxConstants {
    public uint size;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void entry_skybox(
    uint3 threadId : SV_DispatchThreadID,
    uniform SkyboxConstants g_constants, // Implicit pushconstants
    [vk::image_format("rgba16f")]
    RWTexture2DArray<float3> r_output,
    [vk::image_format("rgba32f")]
    Sampler2D<float3> r_equirectangular_map,
)
{
    uint size = g_constants.size;
    uint layer = threadId.z;

    if (all(threadId.xy < uint2(size, size))) {
        float2 face_uv = (threadId.xy + 0.5) / size * 2.0 - 1.0;
        float3 dir = normalize(mul(k_face_rotations[layer], float3(face_uv, 1.0)));
        float2 equirectangular_uv = direction_to_equirectangular(dir);

        r_output[uint3(threadId.xy, layer)] = r_equirectangular_map.SampleLevel(equirectangular_uv, 0);
    }
}

public struct IrradianceConstants {
    public uint size;
    public uint samples_phi;
    public uint samples_theta;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void entry_irradiance(
    uint3 threadId : SV_DispatchThreadID,
    uniform IrradianceConstants g_constants, // Implicit pushconstants
    [vk::image_format("rgba16f")]
    RWTexture2DArray<float3> r_output,
    [vk::image_format("rgba16f")]
    SamplerCube<float3> r_skybox,
)
{
    uint size = g_constants.size;
    uint layer = threadId.z;

    if (all(threadId.xy < uint2(size, size))) {
        float2 face_uv = (threadId.xy + 0.5) / size * 2.0 - 1.0;
        float3 normal = normalize(mul(k_face_rotations[layer], float3(face_uv, 1.0)));

        Frame frame = frame_from_normal(normal);

        float3 irradiance = 0.0;
        float inv_samples_phi = 1.0 / g_constants.samples_phi;
        float inv_samples_theta = 1.0 / g_constants.samples_theta;
        for(uint32_t phi_idx = 0; phi_idx < g_constants.samples_phi; phi_idx += 1)
        {
            float phi = (2.0 * PI) * (phi_idx + 0.5) * inv_samples_phi;
            for(float theta_idx = 0; theta_idx < g_constants.samples_theta; theta_idx += 1)
            {
                float theta = (0.5 * PI) * (theta_idx + 0.5) * inv_samples_theta;

                // spherical to cartesian
                float3 tangent_sample = float3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));

                // tangent space to world
                float3 dir = to_world(frame, tangent_sample);

                irradiance += r_skybox.SampleLevel(dir, 0) * cos(theta) * sin(theta);
            }
        }

        irradiance = PI * irradiance * (1.0 / float(g_constants.samples_phi * g_constants.samples_theta));
        r_output[uint3(threadId.xy, layer)] = irradiance;
    }
}

public struct SpecularConstants {
    public uint size;
    public uint samples;
    public float roughness;
    public float skybox_resolution;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void entry_specular(
    uint3 threadId : SV_DispatchThreadID,
    uniform SpecularConstants g_constants, // Implicit pushconstants
    [vk::image_format("rgba16f")]
    RWTexture2DArray<float3> r_output,
    [vk::image_format("rgba16f")]
    SamplerCube<float3> r_skybox,
)
{
    uint size = g_constants.size;
    uint layer = threadId.z;
    float roughness = g_constants.roughness;
    float alpha = square(roughness);

    if (all(threadId.xy < uint2(size, size))) {
        float2 face_uv = (threadId.xy + 0.5) / size * 2.0 - 1.0;
        float3 normal = normalize(mul(k_face_rotations[layer], float3(face_uv, 1.0)));

        if (roughness > 0) {
            Frame frame = frame_from_normal(normal);

            float total_weight = 0.0;
            float3 accumulator = 0.0;

            uint4 seed = uint4(threadId.xy, layer, 0);

            float4 sample;
            for(uint32_t i = 0; i < g_constants.samples; i++)
            {
                float2 u = hammersley(i, g_constants.samples);

                float3 half_v = sample_gtr2(u, alpha);
                float3 world_half = to_world(frame, half_v);
                float3 world_dir = normalize(2.0 * dot(normal, world_half) * world_half - normal);


                float cosine = max(dot(normal, world_dir), 0.0);
                if (cosine > 0.0) {
                    float pdf = pdf_gtr2(half_v, alpha);
                    float sa_texel = 4.0 * PI / (6.0 * square(g_constants.skybox_resolution));
                    float sa_sample = 1.0 / (g_constants.samples * pdf + 0.0001);
                    float mip_level = 0.5 * log2(sa_sample / sa_texel);

                    accumulator += r_skybox.SampleLevel(world_dir, mip_level) * cosine;
                    total_weight += cosine;
                }
            }

            r_output[uint3(threadId.xy, layer)] = accumulator / total_weight;
        } else {
            r_output[uint3(threadId.xy, layer)] = r_skybox.SampleLevel(normal, 0);
        }
    }
}

public struct LUTConstants {
    public uint samples;
};

// IMPORTANT: the k used for the LUT is different from the one used for direct lighting
public float geometry_schlick_ggx_lut(float NdotV, float alpha)
{
    float k = alpha / 2.0;

    float num   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return num / denom;
}

public float geometry_smith_lut(float NdotV, float NdotL, float alpha)
{
    float ggx2  = geometry_schlick_ggx_lut(NdotV, alpha);
    float ggx1  = geometry_schlick_ggx_lut(NdotL, alpha);
    return ggx1 * ggx2;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void entry_lut(
    uint3 threadId : SV_DispatchThreadID,
    uniform LUTConstants g_constants, // Implicit pushconstants
    [vk::image_format("rg16f")]
    RWTexture2D<float2> r_output,
)
{
    uint width, height;
    r_output.GetDimensions(width, height);

    if (all(threadId.xy < uint2(width, height))) {
        float NdotV = (threadId.x + 0.5) / width;
        float roughness = (threadId.y + 0.5) / height;
        float alpha = square(roughness);

        float3 V = float3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);

        float A = 0.0;
        float B = 0.0;

        float3 N = float3(0.0, 0.0, 1.0);

        for(uint i = 0u; i < g_constants.samples; ++i)
        {
            float2 u = hammersley(i, g_constants.samples);
            float3 H = sample_gtr2(u, alpha);
            float3 L = normalize(2.0 * dot(V, H) * H - V);

            float NdotL = max(L.z, 0.0);
            float NdotH = max(H.z, 0.0);
            float VdotH = max(dot(V, H), 0.0);

            if(NdotL > 0.0)
            {
                float G = geometry_smith_lut(NdotV, NdotL, alpha);
                float G_Vis = (G * VdotH) / (NdotH * NdotV);
                float Fc = pow(1.0 - VdotH, 5.0);

                A += (1.0 - Fc) * G_Vis;
                B += Fc * G_Vis;
            }
        }
        float inv_sample_count = 1.0 / g_constants.samples;
        A *= inv_sample_count;
        B *= inv_sample_count;

        r_output[threadId.xy] = float2(A, B);
    }
}
