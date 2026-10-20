// Height-field sun shadow map: for each texel the minimum height at which a point
// sees the sun (shadowTop) and the horizontal distance to the occluder.
#include "common.glsl"
uniform sampler2D uH;
uniform sampler2D uHFar;
uniform vec2 uCenter;
uniform float uHalf, uRes, uFarHalf;
uniform vec3 uSunDir;
out vec4 fragColor;
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
