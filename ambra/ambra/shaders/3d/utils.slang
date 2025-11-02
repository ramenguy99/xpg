// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

module utils;

public static const float PI = 3.1415926535897932384626433832795f;

public uint murmur_mix(uint Hash)
{
	Hash ^= Hash >> 16;
	Hash *= 0x85ebca6b;
	Hash ^= Hash >> 13;
	Hash *= 0xc2b2ae35;
	Hash ^= Hash >> 16;
	return Hash;
}

public float3 int_to_color(uint Index)
{
	uint Hash = murmur_mix(Index);

	float3 Color = float3
	(
		(Hash >>  0) & 255,
		(Hash >>  8) & 255,
		(Hash >> 16) & 255
	);

	return Color * (1.0f / 255.0f);
}

public float2 direction_to_equirectangular(float3 v)
{
    return float2(atan2(v.y, v.x), asin(v.z)) * float2(0.1591, -0.3183) + 0.5;
}

public struct Frame {
    public float3x3 world_from_local;
};

public float3 to_world(Frame f, float3 local_v) {
    return mul(f.world_from_local, local_v);
}

public float3 to_local(Frame f, float3 world_v) {
    float3x3 local_from_world = transpose(f.world_from_local);
    return mul(local_from_world, world_v);
}

public Frame frame_from_normal(float3 a) {
    float3 b, c;
    if (abs(a.x) > abs(a.y)) {
        float invLen = 1.0f / sqrt(a.x * a.x + a.z * a.z);
        c = float3(a.z * invLen, 0.0f, -a.x * invLen);
    } else {
        float invLen = 1.0f / sqrt(a.y * a.y + a.z * a.z);
        c = float3(0.0f, a.z * invLen, -a.y * invLen);
    }
    b = cross(c, a);
    Frame frame;
    frame.world_from_local = transpose(float3x3(b, c, a));
    return frame;
}

public Frame make_frame(float3 t, float3 b, float3 n) {
    Frame frame;
    frame.world_from_local = transpose(float3x3(t, b, n));
    return frame;
}


public float safe_sqrt(float x) {
    return sqrt(max(x, 0.0));
}

public float square(float x) {
    return x * x;
}

public float cube(float x) {
    return x * x * x;
}

public float linear_to_srgb(float v) {
    v = clamp(v, 0.0, 1.0);
    if(v > 0.0031308)
    {
        return 1.055 * pow(v, 1.0/2.4) - 0.055;
    }
    else
    {
        return v * 12.92;
    }
}

public float3 linear_to_srgb(float3 v) {
    return float3(
        linear_to_srgb(v.x),
        linear_to_srgb(v.y),
        linear_to_srgb(v.z)
    );
}

public float luminance(float3 x) {
    return dot(x, float3(0.212671, 0.715160, 0.072169));
}

public float balance_heuristic(float pa, float pb) {
    return 1.0f / (pa + pb);
}

public void pcg4d(inout uint4 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> 16u);
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
}

public float rand(inout uint4 seed)
{
    pcg4d(seed);
    return float(seed.x)/float(0xffffffffu);
}

public float2 rand2(inout uint4 seed)
{
    pcg4d(seed);
    return float2(seed.xy)/float(0xffffffffu);
}

public float3 rand3(inout uint4 seed)
{
    pcg4d(seed);
    return float3(seed.xyz)/float(0xffffffffu);
}

public float4 rand4(inout uint4 seed)
{
    pcg4d(seed);
    return float4(seed)/float(0xffffffffu);
}

public float radical_inverse_vdc(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
public float2 hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), radical_inverse_vdc(i));
}

public float3 sample_uniform_sphere(float2 u) {
    float z = 2.0 * u.x - 1.0;
    float r = sqrt(1.0 - z * z);
    float phi = (2.0 * PI) * u.y;
    float x = r * cos(phi);
    float y = r * sin(phi);
    return float3(x, y, z);
}

public float3 pdf_uniform_sphere() {
    return 1.0 / (4.0 * PI);
}


public float2 sample_uniform_disk(float2 u) {
    float r = sqrt(u.x);
    float theta = (2.0f * PI) * u.y;
    return float2(r * cos(theta), r * sin(theta));
}

public float3 sample_uniform_hemisphere(float2 u) {
    float3 d = sample_uniform_sphere(u);
    if(d.z < 0.0) {
        d.z = -d.z;
    }
    return d;
}

public float3 sample_cosine_weighted_hemisphere(float2 u) {
    float2 p = sample_uniform_disk(u);
    float z = sqrt(1 - p.x * p.x - p.y * p.y);
    return float3(p.x, p.y, z);
}

public float pdf_cosine_weighted_hemisphere(float3 v) {
    return v.z * (1.0 / PI);
}

public float3 pdf_uniform_hemisphere() {
    return 1.0 / (2.0 * PI);
}


public float3 sample_uniform_hemisphereN(inout float2 u, float3 n) {
    float3 d = sample_uniform_sphere(u);
    if(dot(d, n) < 0.0f) {
        d = -d;
    }
    return d;
}

public float eval_gtr2(float cos_theta, float alpha) {
    float alpha2 = square(alpha);
    return alpha2 / (PI * square((1 + (alpha2 - 1) * square(cos_theta))));
}

public float3 sample_gtr2(float2 u, float alpha) {
    float alpha2 = square(alpha);
    float cos_theta2 = (1 - u.x) / (1 + (alpha2 - 1) * u.x);

    float cos_theta = sqrt(cos_theta2);
    float sin_theta = sqrt(1.0f - cos_theta2);

    float phi = 2.0f * PI * u.y;

    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);
    float z = cos_theta;
    return float3(x, y, z);
}

public float pdf_gtr2(float3 m, float alpha) {
    return eval_gtr2(m.z, alpha) * m.z;
}

public float2 sampleTriangle(float2 u) {
    float su1 = sqrt(u.x);
    return float2(1.0 - su1, u.y * su1);
}

// Results in stronger highlights at view angles close to horizon than using the LUT
public void ggx_brdf_approximation(float cosine, float roughness, out float scale, out float bias) {
    float alpha = roughness * roughness;
    bias = pow(2, -(7 * cosine + 4 * alpha));
    scale = 1 - bias - alpha * max(bias, min(roughness, 0.739 + 0.323 * cosine) - 0.434);
}
