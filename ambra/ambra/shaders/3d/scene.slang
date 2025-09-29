// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

module scene;

interface IMaterial
{
    float3 eval(float3 world_position, float2 uv, float3 world_normal);
}

struct ColorMaterial : IMaterial
{
    float3 color;

    float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        return color;
    }
}

public struct LightEvaluation {
    public float3 L;
    public float3 direction;
};

struct DiffuseMaterial : IMaterial
{
    float3 diffuse;
    bool has_diffuse_texture;
    Sampler2D diffuse_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        float3 diffuse;
        if (has_diffuse_texture) {
            diffuse = this.diffuse_texture.Sample(uv).rgb;
        } else {
            diffuse = this.diffuse;
        }

        float3 outgoing_radiance = float3(0, 0, 0);
        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, world_normal);
            float cosine = max(dot(light.direction, world_normal), 0.0);
            outgoing_radiance += cosine * light.L;
        }

        return outgoing_radiance;
    }
}

struct DiffuseSpecularMaterial : IMaterial
{
    float3 diffuse;
    bool has_diffuse_texture;

    float specular;
    bool has_specular_texture;

    Sampler2D diffuse_texture;
    Sampler2D specular_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        float3 diffuse;
        if (has_diffuse_texture) {
            diffuse = this.diffuse_texture.Sample(uv).rgb;
        } else {
            diffuse = this.diffuse;
        }

        float3 view = normalize(g_constants.world_camera_position - world_position);

        float3 outgoing_radiance = float3(0, 0, 0);
        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, world_normal);

            float3 half = normalize(view + light.direction);

            float3 specular = pow(max(dot(world_normal, half), 0.0f), specular);
            float cosine = max(dot(light.direction, world_normal), 0.0);

            outgoing_radiance += (cosine + specular) * light.L;
        }

        return outgoing_radiance;
    }
}


public interface ILight
{
    public LightEvaluation eval(float3 world_position, float3 world_normal);
}


public struct PointLight : ILight
{
    float3 position;
    float3 intensity;
    float radius_squared;

    public LightEvaluation eval(float3 world_position, float3 world_normal)
    {
        float3 dir = position - world_position;
        float d2 = dot(dir, dir);
        float d = sqrt(d2);

        // From "Point Light Attenuation Without Singularity - Cem Yuksel 2020"
        // https://www.cemyuksel.com/research/pointlightattenuation/
        float attenuation = 2 / (d2 + radius_squared + d * sqrt(d2 + radius_squared));

        LightEvaluation out;
        out.direction = dir * 1.0 / d;
        out.L = intensity * attenuation;
        return out;
    }
}

public struct DirectionalLight : ILight
{
    public float4x4 orthographic_camera;
    public float3 radiance;
    public uint32_t shadowmap_index;
    public float3 direction;
    public float bias;

    public LightEvaluation eval(float3 world_position, float3 world_normal)
    {
        float3 L = radiance;

        if (shadowmap_index != 0xffffffff) {
            // Compute fragment coordinates in light space
            float3 light_pos = mul(orthographic_camera, float4(world_position, 1.0)).xyz;

            // Assume not in shadow if outside frustum in z direction
            if (light_pos.z < 1.0)
            {
                // Compute value to compared with taking bias into account
                float z = light_pos.z;
                float b = max(bias * 0.5 * (1.0 + dot(direction, world_normal)), bias);
                float cmp_value = z - b;

                // Sample shadowmap
                float shadow = g_directional_shadowmaps[shadowmap_index].SampleCmpLevelZero(g_directional_shadowmap_sampler, light_pos.xy * 0.5 + 0.5, cmp_value);

                // Weight radiance by shadow term
                L *= shadow;
            }
        }

        LightEvaluation out;
        out.direction = -direction;
        out.L = L;
        return out;
    }
}

public struct UniformEnvironmentLight : ILight
{
    float3 radiance;
    public LightEvaluation eval(float3 world_position, float3 world_normal)
    {
        LightEvaluation out;
        out.direction = world_normal;
        out.L = radiance;
        return out;
    }
}

public struct Constants {
    public float4x4 view;
    public float4x4 projection;

    public float3 world_camera_position;
    public uint32_t directional_lights_count;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;

[vk::binding(1, 0)]
public StructuredBuffer<DirectionalLight> g_directional_lights;

[vk::binding(2, 0)]
public Texture2D g_directional_shadowmaps[];

[vk::binding(3, 0)]
public SamplerComparisonState g_directional_shadowmap_sampler;
