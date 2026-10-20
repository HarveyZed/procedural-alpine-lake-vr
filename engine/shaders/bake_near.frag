// Near height map (1 m texels) = Catmull-Rom upsampled far map + fine analytic detail.
#include "common.glsl"
#include "heightfn.glsl"
uniform sampler2D uHFar;
uniform float uFarHalf, uFarRes;
uniform vec2 uCenter;
uniform float uHalf, uRes;
out vec4 fragColor;

float farTexel(ivec2 c) { return texelFetch(uHFar, clamp(c, ivec2(0), ivec2(int(uFarRes) - 1)), 0).r; }
float crW(float x, int i) {
    // Catmull-Rom weights for taps -1,0,1,2
    if (i == 0) return ((-0.5 * x + 1.0) * x - 0.5) * x;
    if (i == 1) return (1.5 * x - 2.5) * x * x + 1.0;
    if (i == 2) return ((-1.5 * x + 2.0) * x + 0.5) * x;
    return (0.5 * x - 0.5) * x * x;
}
float farCR(vec2 p) {
    vec2 g = (p / (2.0 * uFarHalf) + 0.5) * uFarRes - 0.5;
    vec2 i = floor(g); vec2 f = g - i; ivec2 c = ivec2(i);
    float s = 0.0;
    for (int y = 0; y < 4; y++) {
        float row = 0.0;
        for (int x = 0; x < 4; x++) row += crW(f.x, x) * farTexel(c + ivec2(x - 1, y - 1));
        s += crW(f.y, y) * row;
    }
    return s;
}
void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = uCenter + (uv * 2.0 - 1.0) * uHalf;
    float h = farCR(p);
    float hx = farCR(p + vec2(2.0, 0.0)), hz = farCR(p + vec2(0.0, 2.0));
    float slope = length(vec2(hx - h, hz - h) / 2.0);
    float fine = heightFine(p, slope);
    fine *= mix(0.35, 1.0, smoothstep(-1.0, 3.0, h));
    fragColor = vec4(h + fine, 0.0, 0.0, 1.0);
}
