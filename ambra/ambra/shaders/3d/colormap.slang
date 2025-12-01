// Copyright Dario Mylonopoulos
// SPDX-License-Identifier: MIT

// Adapted from colormap-shaders: https://github.com/kbinani/colormap-shaders/tree/master
// MIT License
// Copyright (c) 2015 kbinani

module colormap;

public float3 colormap_autumn(float x) {
    float g = clamp(x, 0.0, 1.0);
    return float3(1.0, g, 0.0);
}

float colormap_bone_red(float x) {
    if (x < 0.75) {
        return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
    } else {
        return (13.0 + 8.0 / 9.0) / 10.0 * x - (3.0 + 8.0 / 9.0) / 10.0;
    }
}

float colormap_bone_green(float x) {
    if (x <= 0.375) {
        return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
    } else if (x <= 0.75) {
        return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 100.0;
    } else {
        return 8.0 / 9.0 * x + 1.0 / 9.0;
    }
}

float colormap_bone_blue(float x) {
    if (x <= 0.375) {
        return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 1000.0;
    } else {
        return 8.0 / 9.0 * x + 1.0 / 9.0;
    }
}

public float3 colormap_bone(float x) {
    float r = clamp(colormap_bone_red(x), 0.0, 1.0);
    float g = clamp(colormap_bone_green(x), 0.0, 1.0);
    float b = clamp(colormap_bone_blue(x), 0.0, 1.0);
    return float3(r, g, b);
}

public float3 colormap_cool(float x) {
    float r = clamp((1.0 + 1.0 / 63.0) * x - 1.0 / 63.0, 0.0, 1.0);
    float g = clamp(-(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0), 0.0, 1.0);
    float b = 1.0;
    return float3(r, g, b);
}

public float3 colormap_copper(float x) {
    float r = clamp(80.0 / 63.0 * x + 5.0 / 252.0, 0.0, 1.0);
    float g = clamp(0.7936 * x - 0.0124, 0.0, 1.0);
    float b = clamp(796.0 / 1575.0 * x + 199.0 / 25200.0, 0.0, 1.0);
    return float3(r, g, b);
}

public float3 colormap_hot(float x) {
    float r = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    float g = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    float b = clamp(4.0 * x - 3.0, 0.0, 1.0);
    return float3(r, g, b);
}

float colormap_hsv_red(float x) {
    if (x < 0.5) {
        return -6.0 * x + 67.0 / 32.0;
    } else {
        return 6.0 * x - 79.0 / 16.0;
    }
}

float colormap_hsv_green(float x) {
    if (x < 0.4) {
        return 6.0 * x - 3.0 / 32.0;
    } else {
        return -6.0 * x + 79.0 / 16.0;
    }
}

float colormap_hsv_blue(float x) {
    if (x < 0.7) {
       return 6.0 * x - 67.0 / 32.0;
    } else {
       return -6.0 * x + 195.0 / 32.0;
    }
}

public float3 colormap_hsv(float x) {
    float r = clamp(colormap_hsv_red(x), 0.0, 1.0);
    float g = clamp(colormap_hsv_green(x), 0.0, 1.0);
    float b = clamp(colormap_hsv_blue(x), 0.0, 1.0);
    return float3(r, g, b);
}

public float3 colormap_jet(float t)
{
    return clamp((float3(1.5) - abs(4.0*float3(t) + float3(-3,-2,-1))), 0., 1.);
}

public float3 colormap_spring(float x) {
    return float3(1.0, clamp(x, 0.0, 1.0), clamp(1.0 - x, 0.0, 1.0));
}

public float3 colormap_summer(float x) {
    return float3(clamp(x, 0.0, 1.0), clamp(0.5 * x + 0.5, 0.0, 1.0), 0.4);
}

public float3 colormap_winter(float x) {
    return float3(0.0, clamp(x, 0.0, 1.0), clamp(-0.5 * x + 1.0, 0.0, 1.0));
}

public enum ColormapId: uint32_t {
    AUTUMN,
    BONE,
    COOL,
    COPPER,
    HOT,
    HSV,
    JET,
    SPRING,
    SUMMER,
    WINTER,
};

public float3 colormap(ColormapId kind, float x) {
    switch (kind) {
    case ColormapId.AUTUMN: return colormap_autumn(x);
    case ColormapId.BONE:   return colormap_bone  (x);
    case ColormapId.COOL:   return colormap_cool  (x);
    case ColormapId.COPPER: return colormap_copper(x);
    case ColormapId.HOT:    return colormap_hot   (x);
    case ColormapId.HSV:    return colormap_hsv   (x);
    case ColormapId.JET:    return colormap_jet   (x);
    case ColormapId.SPRING: return colormap_spring(x);
    case ColormapId.SUMMER: return colormap_summer(x);
    case ColormapId.WINTER: return colormap_winter(x);
    }

    return 0.0;
}
