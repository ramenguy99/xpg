// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "../utils";
import "brdf";

struct Material
{
    float3 albedo;
    uint albedo_texture;

    float roughness;
    uint roughness_texture;

    float metallic;
    uint metallic_texture;

    uint normal_texture;
    uint _padding;
    uint _padding1;
    uint _padding2;
};

struct MeshInstance {
    float3x3 normal_matrix;
    // float3x4 normal_matrix;

    float *positions;
    float *normals;
    float *tangents;
    float2 *uvs;
    uint *indices;

    uint material_index;
    uint _padding0;
};

struct PushConstants {
    uint32_t sample_index;
};

struct Constants {
    uint width;
    uint height;
    uint max_bounces;
    uint num_directional_lights;

    float3 camera_position;
    uint viewport_mask;
    float3 camera_forward;
    float film_dist;
    float3 camera_up;
    uint max_samples_per_pixel;
    float3 camera_right;
    float clip_value;

    float3 ambient_light;
    uint has_environment_light;

    float3 background_color;
    uint use_background_color;
};


public struct LightEvaluation {
    public float3 L;
    public float3 direction;
};

public interface ILight
{
    public LightEvaluation eval(float3 world_position, float3 world_normal);
}

public struct DirectionalLight : ILight
{
    public float4x4 _orthographic_camera;
    public float3 radiance;
    public uint32_t shadow_map_index;
    public float3 direction;
    public float _bias;

    public LightEvaluation eval(float3 world_position, float3 world_normal)
    {
        float3 L = radiance;
        LightEvaluation out;
        out.direction = -direction;
        out.L = L;
        return out;
    }
}

[vk::binding(0, 0)]
RaytracingAccelerationStructure acceleration_structure;

[vk::binding(1, 0)]
StructuredBuffer<MeshInstance> instances_buffer;

[vk::binding(2, 0)]
StructuredBuffer<Material> materials_buffer;

[vk::binding(3, 0)]
SamplerState sampler;

[vk::binding(4, 0)]
Texture2D<float4> textures[];

[vk::binding(0, 1)]
ConstantBuffer<Constants> constants;

[vk::binding(1, 1)]
StructuredBuffer<DirectionalLight> directional_lights;

[vk::binding(2, 1)]
RWTexture2D<float4> output;

[vk::binding(3, 1)]
TextureCube specular_cubemap;

[vk::binding(4, 1)]
RWTexture2D<float4> accumulation;

[vk::push_constant]
ConstantBuffer<PushConstants> push_constants;

inline void generate_camera_ray(float2 p, out float3 origin, out float3 direction)
{
    uint2 size = uint2(constants.width, constants.height);

    float2 offset = p / size.xy * 2.0 - 1.0;

    float3 camera_forward = constants.camera_forward;
    float3 camera_up = constants.camera_up;
    float3 camera_right = constants.camera_right;
    float3 camera_p = constants.camera_position;
    float film_dist = constants.film_dist;

    float2 film_size = float2(1.0, 1.0);
    film_size.x = (float)size.x / (float)size.y;
    float2 half_film = film_size * 0.5f;
    float3 film_center = camera_p - film_dist * camera_forward;

    origin = film_center + offset.x * half_film.x * camera_right
                         + offset.y * half_film.y * camera_up;

    direction = normalize(camera_p - origin);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 thread_idx : SV_DispatchThreadID)
{
    uint2 p = thread_idx.xy;
    if(any(p >= uint2(constants.width, constants.height))) {
        return;
    }

    float3 color = 0.0;
    bool render = push_constants.sample_index < constants.max_samples_per_pixel;
    if (render) {
        uint4 seed = uint4(p, uint(push_constants.sample_index), uint(p.x) + constants.width * uint(p.y));
        float2 jitter = rand2(seed);

        float3 dir;
        float3 origin;
        generate_camera_ray(p + jitter, origin, dir);

        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dir;
        ray.TMin = 0.001;
        ray.TMax = 100000.0;

        RayQuery<RAY_FLAG_NONE> q;
        uint rayFlags = RAY_FLAG_NONE;

        uint max_bounces = constants.max_bounces + 1;
        float3 throughput = 1.0;

        for(uint bounce = 0; bounce < max_bounces; bounce++) {
            q.TraceRayInline(acceleration_structure, rayFlags, constants.viewport_mask, ray);
            q.Proceed();

            if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                float t = q.CommittedRayT();

                // Hit position
                float3 position = origin + t * dir;


                // Mesh info
                uint mesh_index = q.CommittedRayInstanceCustomIndex();
                MeshInstance instance = instances_buffer[mesh_index];

                // Primitive info
                uint triangle_index = q.CommittedRayPrimitiveIndex();

                uint3 indices;
                if (instance.indices) {
                    indices = uint3(
                        instance.indices[triangle_index * 3 + 0],
                        instance.indices[triangle_index * 3 + 1],
                        instance.indices[triangle_index * 3 + 2],
                    );
                } else {
                    indices = uint3(
                        triangle_index * 3 + 0,
                        triangle_index * 3 + 1,
                        triangle_index * 3 + 2,
                    );
                }

                float2 barycentrics = q.CommittedRayBarycentrics();

                float3 a = float3(instance.positions[indices.x * 3 + 0], instance.positions[indices.x * 3 + 1], instance.positions[indices.x * 3 + 2]);
                float3 b = float3(instance.positions[indices.y * 3 + 0], instance.positions[indices.y * 3 + 1], instance.positions[indices.y * 3 + 2]);
                float3 c = float3(instance.positions[indices.z * 3 + 0], instance.positions[indices.z * 3 + 1], instance.positions[indices.z * 3 + 2]);
                float3 object_geom_normal = normalize(cross(b - a, c - a));
                float3 world_geom_normal = normalize(mul(instance.normal_matrix, object_geom_normal));
                float3 N = world_geom_normal;

                if (instance.normals) {
                    float3 object_shading_normal =
                        float3(instance.normals[indices.x * 3 + 0], instance.normals[indices.x * 3 + 1], instance.normals[indices.x * 3 + 2]) * (1 - barycentrics.x - barycentrics.y) +
                        float3(instance.normals[indices.y * 3 + 0], instance.normals[indices.y * 3 + 1], instance.normals[indices.y * 3 + 2]) * barycentrics.x +
                        float3(instance.normals[indices.z * 3 + 0], instance.normals[indices.z * 3 + 1], instance.normals[indices.z * 3 + 2]) * barycentrics.y;
                    float3 world_shading_normal = normalize(mul(instance.normal_matrix, object_shading_normal));

                    // Use geom normal if the shading normal is backfacing
                    if (dot(world_shading_normal, ray.Direction) >= 0.0) {
                        N = world_geom_normal;
                    } else {
                        N = world_shading_normal;
                    }
                }

                float2 uv;
                if (instance.uvs) {
                    uv =
                        instance.uvs[indices.x] * (1 - barycentrics.x - barycentrics.y) +
                        instance.uvs[indices.y] * barycentrics.x +
                        instance.uvs[indices.z] * barycentrics.y;
                } else {
                    uv = 0.0;
                }

                Material material = materials_buffer[instance.material_index];

                // Create shading frame, applying normal mapping only at the first full-throughput hit.
                Frame frame;

                // TODO: replace with check if first non-specular hit
                // TODO: potentially use similar logic for applying mip bias / roughness regularization
                if (all(throughput == 1.0)) {
                    float3 tangent;
                    if (instance.tangents) {
                        tangent =
                            float3(instance.tangents[indices.x * 3 + 0], instance.tangents[indices.x * 3 + 1],instance.tangents[indices.x * 3 + 2]) * (1 - barycentrics.x - barycentrics.y) +
                            float3(instance.tangents[indices.y * 3 + 0], instance.tangents[indices.y * 3 + 1],instance.tangents[indices.y * 3 + 2]) * barycentrics.x +
                            float3(instance.tangents[indices.z * 3 + 0], instance.tangents[indices.z * 3 + 1],instance.tangents[indices.z * 3 + 2]) * barycentrics.y;

                        float3 T = normalize(mul(instance.normal_matrix, tangent));
                        float3 B = normalize(cross(N, T));
                        if (any(isnan(B))) {
                            frame = frame_from_normal(N);
                        }
                        else {
                            frame = make_frame(T, B, N);
                            if (material.normal_texture != ~0) {
                                float2 n = textures[NonUniformResourceIndex(material.normal_texture)].SampleLevel(sampler, uv, 0.0f).rg * 2.0 - 1.0;
                                // Lerp towards local +Z when viewing at grazing angle
                                float weight = max(dot(N, -ray.Direction), 0.0);
                                n = lerp(0.0, n, weight);

                                float z = sqrt(1.0 - n.x * n.x - n.y * n.y);
                                N = to_world(frame, float3(n, z));
                                B = normalize(cross(N, T));
                                T = normalize(cross(B, N));
                                frame = make_frame(T, B, N);
                            }
                        }
                    } else {
                        frame = frame_from_normal(N);
                    }
                } else {
                    frame = frame_from_normal(N);
                }

                float3 albedo;
                if (material.albedo_texture != ~0) {
                    albedo = textures[NonUniformResourceIndex(material.albedo_texture)].SampleLevel(sampler, uv, 0).rgb;
                } else {
                    albedo = material.albedo;
                }

                float roughness;
                if (material.roughness_texture != ~0) {
                    roughness = textures[NonUniformResourceIndex(material.roughness_texture)].SampleLevel(sampler, uv, 0).r;
                } else {
                    roughness = material.roughness;
                }

                float metallic;
                if (material.metallic_texture != ~0) {
                    metallic = textures[NonUniformResourceIndex(material.metallic_texture)].SampleLevel(sampler, uv, 0).r;
                } else {
                    metallic = material.metallic;
                }

                // TODO: special case roughness < threshold (0.05) as full specular hit (with albedo)
                roughness = max(roughness, 0.05);

                float alpha = square(roughness);
                float3 wo = to_local(frame, -ray.Direction);

                // Delta light sampling
                float3 shadow_origin = position + N * 0.001;
                RayDesc shadow_ray;
                shadow_ray.Origin = shadow_origin;
                shadow_ray.Direction =
                shadow_ray.TMin = 0.001;
                shadow_ray.TMax = 100000.0;

                for (uint32_t i = 0; i < constants.num_directional_lights; i++) {
                    DirectionalLight light = directional_lights[i];

                    float3 wi = to_local(frame, -light.direction);
                    if (wi.z <= 0.0) {
                        continue;
                    }

                    bool shadow = false;
                    if (light.shadow_map_index != ~0) {
                        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadow_q;
                        uint shadow_ray_flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
                        shadow_ray.Direction = -light.direction;

                        shadow_q.TraceRayInline(acceleration_structure, shadow_ray_flags, constants.viewport_mask, shadow_ray);
                        shadow_q.Proceed();
                        shadow = shadow_q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
                    }

                    if (!shadow) {
                        float3 brdf = eval_principled_brdf(alpha, metallic, albedo, wo, wi);
                        color += throughput * brdf * light.radiance * wi.z;
                    }
                }

                // BRDF sampling for continuation ray
                float2 u = rand2(seed);

                float pdf;
                float3 sampled_dir;
                float3 brdf = sample_principled_brdf(alpha, metallic, albedo, wo, u, sampled_dir, pdf);
                if (pdf <= 0.0) {
                    break;
                }
                throughput *= brdf / pdf;

                ray.Origin = position;// + N * 0.001;
                ray.Direction = to_world(frame, sampled_dir);
            } else {
                if (bounce == 0 && constants.use_background_color != 0) {
                    color = constants.background_color;
                } else {
                    // TODO: MIS with BRDF sample environment light hit (once light sampling is implemented)
                    float3 ambient_radiance = constants.ambient_light;
                    if (constants.has_environment_light != 0) {
                        ambient_radiance += specular_cubemap.SampleLevel(sampler, ray.Direction, 0.0).rgb;
                    }
                    color += throughput * ambient_radiance;
                }

                break;
            }
        }
    }

    float4 out_color = float4(clamp(color, 0.0, constants.clip_value), 1.0);
    if(any(isnan(color))) {
        out_color = 0.0;
    }

    float4 prev_color = 0.0;
    if (push_constants.sample_index > 0) {
        prev_color = accumulation[thread_idx.xy];
    }

    float4 final_color = prev_color + out_color;
    if (render) {
        accumulation[thread_idx.xy] = final_color;
    }

    if (final_color.w > 0) {
        output[thread_idx.xy] = float4(linear_to_srgb(final_color.rgb / final_color.w), 1.0);
    } else {
        output[thread_idx.xy] = float4(0.0, 0.0, 0.0, 1.0);
    }
}
