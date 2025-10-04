// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT
module scene;

public struct Constants {
    public float4x4 view;
    public float4x4 projection;

    public float3 world_camera_position;
    public uint32_t directional_lights_count;
};

public struct LightEvaluation {
    public float3 L;
    public float3 direction;
};

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

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;

[vk::binding(1, 0)]
public StructuredBuffer<DirectionalLight> g_directional_lights;

[vk::binding(2, 0)]
public Texture2D g_directional_shadowmaps[];

[vk::binding(3, 0)]
public SamplerComparisonState g_directional_shadowmap_sampler;

#define PI 3.1415926535897932384626433832795f

public interface IMaterial
{
    public float3 eval(float3 world_position, float2 uv, float3 world_normal);
}

public struct EmptyMaterial : IMaterial
{
    float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        return 0.0;
    }
}

struct ColorMaterial : IMaterial
{
    float3 color;
    bool has_color_texture;

    SamplerState sampler;
    Texture2D color_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        float3 color;
        if (has_color_texture) {
            color = this.color_texture.Sample(sampler, uv).rgb;
        } else {
            color = this.color;
        }
        return color;
    }
}

public struct DiffuseMaterial : IMaterial
{
    float3 diffuse;
    bool has_diffuse_texture;

    SamplerState sampler;
    Texture2D diffuse_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        float3 diffuse;
        if (has_diffuse_texture) {
            diffuse = this.diffuse_texture.Sample(sampler, uv).rgb;
        } else {
            diffuse = this.diffuse;
        }

        float3 outgoing_radiance = float3(0, 0, 0);
        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, world_normal);
            float cosine = max(dot(light.direction, world_normal), 0.0);
            outgoing_radiance += cosine * diffuse * light.L;
        }

        return outgoing_radiance;
    }
}

public struct DiffuseSpecularMaterial : IMaterial
{
    float3 diffuse;
    bool has_diffuse_texture;

    float specular_strength;
    bool has_specular_strength_texture;

    float specular_exponent;
    float specular_tint;

    SamplerState sampler;
    Texture2D diffuse_texture;
    Texture2D specular_strength_texture;

    public float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        float3 diffuse;
        if (has_diffuse_texture) {
            diffuse = this.diffuse_texture.Sample(sampler, uv).rgb;
        } else {
            diffuse = this.diffuse;
        }
        float specular_strength;
        if (has_diffuse_texture) {
            specular_strength = this.specular_strength_texture.Sample(sampler, uv).r;
        } else {
            specular_strength = this.specular_strength;
        }
        float specular_exponent;
        if (has_diffuse_texture) {
            specular_exponent = this.specular_exponent_texture.Sample(sampler, uv).r;
        } else {
            specular_exponent = this.specular_exponent;
        }

        float3 view = normalize(g_constants.world_camera_position - world_position);

        float3 outgoing_radiance = float3(0, 0, 0);
        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, world_normal);

            float3 half = normalize(view + light.direction);

            float specular = pow(max(dot(world_normal, half), 0.0f), specular_exponent) * specular_strength;
            float cosine = max(dot(light.direction, world_normal), 0.0);

            outgoing_radiance += (cosine * diffuse + specular) * light.L;
        }

        return outgoing_radiance;
    }
}

float distribution_ggx(float3 N, float3 H, float Roughness)
{
    float a      = Roughness*Roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;

    return num / denom;
}

float geometry_schlick_ggx(float NdotV, float Roughness)
{
    float r = (Roughness + 1.0f);
    float k = (r*r) / 8.0f;

    float num   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return num / denom;
}
float geometry_smith(float3 N, float3 V, float3 L, float Roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx2  = geometry_schlick_ggx(NdotV, Roughness);
    float ggx1  = geometry_schlick_ggx(NdotL, Roughness);

    return ggx1 * ggx2;
}

float3 fresnel_schlick(float CosTheta, float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - CosTheta, 5.0f);
}

float3 fresnel_schlick_roughness(float cosTheta, float3 F0, float Roughness)
{
    float invRough = 1.0f - Roughness;
    return F0 + (max(float3(invRough, invRough, invRough), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float3 eval_roughness_metallic_brdf(float3 N, float3 V, float3 L, float3 Albedo, float3 F0, float Roughness, float Metallic)
{
    float3 H = normalize(V + L);

    float NDF = distribution_ggx(N, H, Roughness);
    float G   = geometry_smith(N, V, L, Roughness);
    float3 F  = fresnel_schlick(clamp(dot(H, V), 0.0, 1.0), F0);

    float3 Numerator  = NDF * G * F;
    float Denominator = 4.0f * max(dot(N, V), 0.0f) * max(dot(N, L), 0.0f);
    float3 Specular   = Numerator / max(Denominator, 0.0001f);

    float3 kS = F;
    float3 kD = float3(1.0f, 1.0f, 1.0f) - kS;
    kD *= 1.0f - Metallic;

    float NdotL = max(dot(N, L), 0.0f);
    float3 brdf = (kD * Albedo / PI + Specular) * NdotL;

    return brdf;
}
struct PBRMaterial : IMaterial
{
    float3 albedo;
    bool has_albedo_texture;

    float roughness;
    bool has_roughness_texture;

    float metallic;
    bool has_metallic_texture;

    SamplerState sampler;
    Texture2D albedo_texture;
    Texture2D roughness_texture;
    Texture2D metallic_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal)
    {
        float3 albedo;
        if (has_albedo_texture) {
            albedo = this.albedo_texture.Sample(sampler, uv).rgb;
        } else {
            albedo = this.albedo;
        }

        float roughness;
        if (has_roughness_texture) {
            roughness = this.roughness_texture.Sample(sampler, uv).r;
        } else {
            roughness = this.roughness;
        }

        float metallic;
        if (has_metallic_texture) {
            metallic = this.metallic_texture.Sample(sampler, uv).r;
        } else {
            metallic = this.metallic;
        }

        float3 view = normalize(g_constants.world_camera_position - world_position);
        float3 outgoing_radiance = float3(0, 0, 0);

        float3 f0 = float3(0.04f, 0.04f, 0.04f);
        f0 = lerp(f0, albedo, metallic);

        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, world_normal);

            float3 brdf_cosine = eval_roughness_metallic_brdf(world_normal, view, light.direction, albedo, f0, roughness, metallic);
            outgoing_radiance += brdf_cosine * light.L;
        }

        return outgoing_radiance;
    }
}

public struct ObjectConstants {
    public float4x4 transform;

};

[vk::binding(0, 1)]
public ConstantBuffer<ObjectConstants> g_object_constants;

// TODO: think if we can use entry point arguments for this instead,
// but not sure if we have control over the descriptor set.
[vk::binding(0, 2)]
#if defined(MATERIAL_COLOR)
public ConstantBuffer<ColorMaterial> g_material;
#elif defined(MATERIAL_DIFFUSE)
public ConstantBuffer<DiffuseMaterial> g_material;
#elif defined(MATERIAL_DIFFUSE_SPECULAR)
public ConstantBuffer<DiffuseSpecularMaterial> g_material;
#elif defined(MATERIAL_PBR)
public ConstantBuffer<PBRMaterial> g_material;
#else
public ConstantBuffer<EmptyMaterial> g_material;
#endif