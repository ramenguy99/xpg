// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

module scene;

public static const float PI = 3.1415926535897932384626433832795f;

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
    public uint32_t shadow_map_index;
    public float3 direction;
    public float bias;

    public LightEvaluation eval(float3 world_position, float3 world_normal)
    {
        float3 L = radiance;

        if (shadow_map_index != 0xffffffff) {
            // Compute fragment coordinates in light space
            float3 light_pos = mul(orthographic_camera, float4(world_position, 1.0)).xyz;

            // Assume not in shadow if outside frustum in z direction
            if (light_pos.z < 1.0)
            {
                // Compute value to compared with taking bias into account
                float z = light_pos.z;
                float b = max(bias * 0.5 * (1.0 + dot(direction, world_normal)), bias);
                float cmp_value = z - b;

                // Sample shadow_map
                float shadow = g_directional_shadow_maps[shadow_map_index].SampleCmpLevelZero(g_directional_shadow_map_sampler, light_pos.xy * 0.5 + 0.5, cmp_value);

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

public interface IMaterial
{
    public float3 eval(float3 world_position, float2 uv, float3 world_normal, float3 world_tangent, float3 tint);
}

public struct EmptyMaterial : IMaterial
{
    float3 eval(float3 world_position, float2 uv, float3 world_normal, float3 world_tangent, float3 tint)
    {
        return 0.0;
    }
}

public struct ColorMaterial : IMaterial
{
    public float3 color;
    public bool has_color_texture;

    public SamplerState sampler;
    public Texture2D color_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal, float3 world_tangent, float3 tint)
    {
        float3 color;
        if (has_color_texture) {
            color = this.color_texture.Sample(sampler, uv).rgb;
        } else {
            color = this.color;
        }
        return color * tint.rgb;
    }
}

public struct DiffuseMaterial : IMaterial
{
    public float3 diffuse;
    public bool has_diffuse_texture;
    public bool has_normal_texture;

    public SamplerState sampler;
    public Texture2D diffuse_texture;
    public Texture2D normal_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal, float3 world_tangent, float3 tint)
    {
        float3 diffuse;
        if (has_diffuse_texture) {
            diffuse = this.diffuse_texture.Sample(sampler, uv).rgb;
        } else {
            diffuse = this.diffuse;
        }
        diffuse *= tint;

        float3 N = world_normal;
        if (has_normal_texture != 0) {
            float3 T = world_tangent;
            float3 B = cross(N, T);
            float3x3 world_from_tangent = transpose(float3x3(T, B, N));
            N = mul(world_from_tangent, this.normal_texture.Sample(sampler, uv).rgb * 2.0 - 1.0);
        }

        float3 outgoing_radiance = float3(0, 0, 0);
        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, N);
            float cosine = max(dot(light.direction, N), 0.0);
            outgoing_radiance += cosine * diffuse * light.L;
        }

        float3 irradiance = 0.0;
        if (g_constants.has_environment_light != 0) {
            irradiance += g_irradiance_cubemap.SampleLevel(g_linear_sampler, N, 0.0).rgb;
        }
        irradiance += g_constants.ambient_light;

        float3 color = outgoing_radiance + irradiance * diffuse;

        // Tonemapping
        color = color / (color + 1.0);

        // To sRGB
        color = pow(color, 1.0 / 2.2);

        return color;
    }
}

public struct DiffuseSpecularMaterial : IMaterial
{
    public float3 diffuse;
    public bool has_diffuse_texture;

    public float specular_strength;
    public bool has_specular_strength_texture;

    public float specular_exponent;
    public float specular_tint;

    public bool has_normal_texture;

    public SamplerState sampler;
    public Texture2D diffuse_texture;
    public Texture2D specular_strength_texture;
    public Texture2D normal_texture;

    public float3 eval(float3 world_position, float2 uv, float3 world_normal, float3 world_tangent, float3 tint)
    {
        float3 diffuse;
        if (has_diffuse_texture) {
            diffuse = this.diffuse_texture.Sample(sampler, uv).rgb;
        } else {
            diffuse = this.diffuse;
        }
        diffuse *= tint;

        float specular_strength;
        if (has_diffuse_texture) {
            specular_strength = this.specular_strength_texture.Sample(sampler, uv).r;
        } else {
            specular_strength = this.specular_strength;
        }

        float3 N = world_normal;
        if (has_normal_texture != 0) {
            float3 T = world_tangent;
            float3 B = cross(N, T);
            float3x3 world_from_tangent = transpose(float3x3(T, B, N));
            N = mul(world_from_tangent, this.normal_texture.Sample(sampler, uv).rgb * 2.0 - 1.0);
        }

        float3 view = normalize(g_constants.world_camera_position - world_position);

        float3 outgoing_radiance = float3(0, 0, 0);
        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, N);

            float3 half = normalize(view + light.direction);

            float specular = pow(max(dot(N, half), 0.0f), specular_exponent) * specular_strength;
            float cosine = max(dot(light.direction, N), 0.0);

            outgoing_radiance += ((cosine * diffuse) + lerp(1.0, diffuse, specular_tint) * specular) * light.L;
        }

        float3 irradiance = 0.0;
        if (g_constants.has_environment_light != 0) {
            irradiance += g_irradiance_cubemap.SampleLevel(g_linear_sampler, N, 0.0).rgb;
        }
        irradiance += g_constants.ambient_light;

        float3 color = outgoing_radiance + irradiance * diffuse;

        // Tonemapping
        color = color / (color + 1.0);

        // To sRGB
        color = pow(color, 1.0 / 2.2);

        return color;
    }
}

// TODO: Change to use sample_gtr2 in utils
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
    float3 F  = fresnel_schlick(max(dot(H, V), 0.0), F0);

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

public struct PBRMaterial : IMaterial
{
    public float3 albedo;
    public uint has_albedo_texture;

    public float roughness;
    public uint has_roughness_texture;
    public float metallic;
    public uint has_metallic_texture;

    public float ao;
    public uint has_ao_texture;
    public uint has_normal_texture;

    public SamplerState sampler;
    public Texture2D albedo_texture;
    public Texture2D roughness_texture;
    public Texture2D metallic_texture;
    public Texture2D ao_texture;
    public Texture2D normal_texture;

    float3 eval(float3 world_position, float2 uv, float3 world_normal, float3 world_tangent, float3 tint)
    {
        float3 albedo;
        if (has_albedo_texture != 0) {
            albedo = this.albedo_texture.Sample(sampler, uv).rgb;
        } else {
            albedo = this.albedo;
        }
        albedo *= tint;

        float roughness;
        if (has_roughness_texture != 0) {
            roughness = this.roughness_texture.Sample(sampler, uv).r;
        } else {
            roughness = this.roughness;
        }

        float metallic;
        if (has_metallic_texture != 0) {
            metallic = this.metallic_texture.Sample(sampler, uv).r;
        } else {
            metallic = this.metallic;
        }

        float ao;
        if (has_ao_texture != 0) {
            ao = this.ao_texture.Sample(sampler, uv).r;
        } else {
            ao = this.ao;
        }

        float3 N = world_normal;
        if (has_normal_texture != 0) {
            float3 T = world_tangent;
            float3 B = cross(N, T);
            float3x3 world_from_tangent = transpose(float3x3(T, B, N));
            N = mul(world_from_tangent, this.normal_texture.Sample(sampler, uv).rgb * 2.0 - 1.0);
        }

        float3 view = normalize(g_constants.world_camera_position - world_position);
        float3 outgoing_radiance = float3(0, 0, 0);

        float3 f0 = lerp(0.04, albedo, metallic);

        // Direct lights
        for (uint32_t i = 0; i < g_constants.directional_lights_count; i++) {
            LightEvaluation light = g_directional_lights[i].eval(world_position, N);

            float3 brdf_cosine = eval_roughness_metallic_brdf(N, view, light.direction, albedo, f0, roughness, metallic);
            outgoing_radiance += brdf_cosine * light.L;
        }

        // Environment light
        float cosine = max(dot(N, view), 0.0f);

        float3 f = fresnel_schlick_roughness(cosine, f0, roughness);
        float3 kS = f;
        float3 kD = (1.0 - kS) * (1.0 - metallic);

        float3 irradiance = 0.0;
        if (g_constants.has_environment_light != 0) {
            irradiance += g_irradiance_cubemap.SampleLevel(g_linear_sampler, N, 0.0).rgb;
        }
        irradiance += g_constants.ambient_light;

        float3 diffuse = irradiance * albedo;

        float3 prefiltered_specular = 0.0;
        if (g_constants.has_environment_light != 0) {
            float3 reflection = reflect(-view, N);
            prefiltered_specular += g_specular_cubemap.SampleLevel(g_linear_sampler, reflection, roughness * g_constants.max_specular_mip).rgb;
        }
        prefiltered_specular += g_constants.ambient_light;

        float2 lut = g_ggx_lut.SampleLevel(g_linear_sampler, float2(cosine, roughness), 0).rg;

        float3 specular = prefiltered_specular * (f * lut.x + lut.y);
        float3 environment = kD * diffuse + specular;

        float3 color = outgoing_radiance + environment * ao;

        // Tonemapping
        color = color / (color + 1.0);

        // To sRGB
        color = pow(color, 1.0 / 2.2);

        return color;
    }
}

public struct Constants {
    public float4x4 view;
    public float4x4 projection;

    public float3 world_camera_position;

    public float3 ambient_light;
    public uint32_t has_environment_light;

    public float2 focal;
    public float2 inverse_viewport_size;

    public float max_specular_mip;
    public uint32_t directional_lights_count;
};

[vk::binding(0, 0)]
public ConstantBuffer<Constants> g_constants;

[vk::binding(1, 0)]
public SamplerComparisonState g_directional_shadow_map_sampler;

[vk::binding(2, 0)]
public SamplerState g_linear_sampler;

[vk::binding(3, 0)]
public TextureCube g_irradiance_cubemap;

[vk::binding(4, 0)]
public TextureCube g_specular_cubemap;

[vk::binding(5, 0)]
public Texture2D g_ggx_lut;

[vk::binding(6, 0)]
public StructuredBuffer<DirectionalLight> g_directional_lights;

[vk::binding(7, 0)]
public Texture2D g_directional_shadow_maps[];
