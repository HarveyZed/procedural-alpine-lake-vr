#version 450
// Horizon-based sky visibility baked from a height texture (with far-map fallback).
// ---------------------------------------------------------------------------
// common.glsl : constants, integer hashes, analytic noises (all procedural)
// ---------------------------------------------------------------------------
#define PI 3.14159265359
#define TAU 6.28318530718
#define saturate(x) clamp(x, 0.0, 1.0)

uint hashu(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u; return x;
}
uint hash2u(ivec2 c) { return hashu(uint(c.x) * 0x9E3779B1u + hashu(uint(c.y) + 0x632BE5ABu)); }

float u2f(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }
float hash12i(ivec2 c) { return u2f(hash2u(c)); }

// 2D gradient noise with analytic derivatives. returns (value, d/dx, d/dy), value ~[-0.7,0.7]

// 2D value noise [0,1]

// 3D value noise [0,1]

// 2D Voronoi: returns (F1 distance, F2 distance, cell id hash)

// voronoi with the offset to the nearest feature point: (dx, dy, F1, id)

layout(set=0, binding=0) uniform sampler2D uH;      // map being processed
layout(set=0, binding=1) uniform sampler2D uHFar;   // fallback beyond uH
// port(P1): the engine's loose pass uniforms, grouped into one block for Vulkan GLSL (unnamed instance -> members stay global)
layout(std140, set=1, binding=0) uniform _p1_pass {
    vec2 uCenter;
    float uHalf, uRes, uFarHalf;
    float uStep0;
};



      // first step length in m
layout(location=0) out vec4 fragColor;

float H(vec2 p) {
    vec2 uv = (p - uCenter) / (2.0 * uHalf) + 0.5;
    if (all(greaterThan(uv, vec2(0.002))) && all(lessThan(uv, vec2(0.998))))
        return texture(uH, uv).r;
    return texture(uHFar, p / (2.0 * uFarHalf) + 0.5).r;
}
void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = uCenter + (uv * 2.0 - 1.0) * uHalf;
    float h0 = H(p) + 0.6;
    float vis = 0.0;
    const int ND = 16;
    float jit = hash12i(ivec2(gl_FragCoord.xy));
    for (int d = 0; d < ND; d++) {
        float a = (float(d) + jit) / float(ND) * TAU;
        vec2 dir = vec2(cos(a), sin(a));
        float ms = 0.0;
        float t = uStep0;
        for (int i = 0; i < 30; i++) {
            vec2 q = p + dir * t;
            float dh = H(q) - h0;
            ms = max(ms, dh / sqrt(dh * dh + t * t));
            t *= 1.28;
        }
        vis += 1.0 - ms * ms;
    }
    fragColor = vec4(vis / float(ND), 0.0, 0.0, 1.0);
}
