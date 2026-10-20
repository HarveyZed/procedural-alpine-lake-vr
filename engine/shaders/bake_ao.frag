// Horizon-based sky visibility baked from a height texture (with far-map fallback).
#include "common.glsl"
uniform sampler2D uH;      // map being processed
uniform sampler2D uHFar;   // fallback beyond uH
uniform vec2 uCenter;
uniform float uHalf, uRes, uFarHalf;
uniform float uStep0;      // first step length in m
out vec4 fragColor;

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
