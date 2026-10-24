#version 450
// Height-field sun shadow map: for each texel the minimum height at which a point
// sees the sun (shadowTop) and the horizontal distance to the occluder.
// ---------------------------------------------------------------------------
// common.glsl : constants, integer hashes, analytic noises (all procedural)
// ---------------------------------------------------------------------------
#define PI 3.14159265359
#define TAU 6.28318530718
#define saturate(x) clamp(x, 0.0, 1.0)

// 2D gradient noise with analytic derivatives. returns (value, d/dx, d/dy), value ~[-0.7,0.7]

// 2D value noise [0,1]

// 3D value noise [0,1]

// 2D Voronoi: returns (F1 distance, F2 distance, cell id hash)

// voronoi with the offset to the nearest feature point: (dx, dy, F1, id)

layout(set=0, binding=0) uniform sampler2D uH;
layout(set=0, binding=1) uniform sampler2D uHFar;
// port(P1): the engine's loose pass uniforms, grouped into one block for Vulkan GLSL (unnamed instance -> members stay global)
layout(std140, set=1, binding=0) uniform _p1_pass {
    vec2 uCenter;
    float uHalf, uRes, uFarHalf;
    vec3 uSunDir;
};




layout(location=0) out vec4 fragColor;
float H(vec2 p) {
    vec2 uv = (p - uCenter) / (2.0 * uHalf) + 0.5;
    if (all(greaterThan(uv, vec2(0.001))) && all(lessThan(uv, vec2(0.999))))
        return texture(uH, uv).r;
    return texture(uHFar, p / (2.0 * uFarHalf) + 0.5).r;
}
void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = uCenter + (uv * 2.0 - 1.0) * uHalf;
    float texel = 2.0 * uHalf / uRes;
    float horiz = max(length(uSunDir.xz), 1e-4);
    vec2 d2 = uSunDir.xz / horiz;
    float tanE = uSunDir.y / horiz;
    float best = -1e9, bestS = 0.0;
    float s = texel * 0.75;
    for (int i = 0; i < 400; i++) {
        float v = H(p + d2 * s) - s * tanE;
        if (v > best) { best = v; bestS = s; }
        if (4200.0 - s * tanE < best) break;
        s += max(texel * 0.7, s * 0.018);
        if (s > 14000.0) break;
    }
    fragColor = vec4(best, bestS, 0.0, 1.0);
}
