// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

import "../utils";

module brdf;

float3 fresnel_schlick(float cos_theta, float3 F0) {
    return F0 + (1 - F0) * pow(1 - cos_theta, 5);
}

float tan_theta2(float3 v) {
    float sin2 = 1 - v.z * v.z;
    if (sin2 <= 0.0)
        return 0.0;
    return sin2 / (v.z * v.z);
}

float eval_geometry_ggx(float3 v, float alpha) {
    return 2.0 / (1.0 + sqrt(1.0 + square(alpha) * tan_theta2(v)));
}

float3 eval_principled_brdf_helper(float alpha, float metallic, float3 base_color,
                              float3 wo, float3 wh, float3 wi) {
    float cos_theta_h = wh.z;
    float cos_theta_d = dot(wh, wo);

    float3 diffuse = (1.0f - metallic) * base_color * INV_PI;

    float D = eval_gtr2(cos_theta_h, alpha);

    float3 specular_color = lerp(0.04, base_color, metallic);
    float3 F = fresnel_schlick(cos_theta_d, specular_color);

    float G = eval_geometry_ggx(wi, alpha) * eval_geometry_ggx(wo, alpha);
    float3 specular = (D * F * G) / (4.0 * wi.z * wo.z);

    return diffuse + specular;
}

public float3 eval_principled_brdf(float alpha, float metallic, float3 base_color,
                        float3 wo, float3 wi) {
    if (wi.z <= 0.0 || wo.z <= 0.0) {
        return 0.0;
    }

    float3 wh = normalize(wo + wi);
    return eval_principled_brdf_helper(alpha, metallic, base_color, wo, wh, wi);
}

float pdf_principled_brdf_helper(float alpha, float metallic, float3 wo, float3 wh) {
    float s_pdf = pdf_gtr2(wh, alpha) / (4 * dot(wh, wo));
    float d_pdf = pdf_cosine_weighted_hemisphere(wo);
    return d_pdf * (1.0 - metallic) +  metallic * s_pdf;
}

public float pdf_principled_brdf(float alpha, float metallic, float3 wo, float3 wi) {
    if (wo.z <= 0.0 || wi.z <= 0.0) {
        return 0.0;
    }
    float3 wh = normalize(wo + wi);
    return pdf_principled_brdf_helper(alpha, metallic, wo, wh);
}

public float3 sample_principled_brdf(float alpha, float metallic, float3 base_color,
                          float3 wi, float2 u, out float3 wo, out float pdf) {
    float diffuse_weight = 1.0 - metallic;
    float3 wh;
    if (u.x < diffuse_weight) {
        u.x /= diffuse_weight;
        wo = sample_cosine_weighted_hemisphere(u.xy);
        wh = normalize(wo + wi);
    } else {
        u.x = (u.x - diffuse_weight) / metallic;
        wh = sample_gtr2(u.xy, alpha);
        wo = 2 * dot(wi, wh) * wh - wi;
    }

    pdf = pdf_principled_brdf_helper(alpha, metallic, wo, wh);
    if(pdf > 0.0) {
        return eval_principled_brdf_helper(alpha, metallic, base_color, wo, wh, wi) * wo.z;
    } else {
        return 0.0;
    }
}
