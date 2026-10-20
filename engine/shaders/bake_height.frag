// Bakes the analytic height into a float texture.
#include "common.glsl"
#include "heightfn.glsl"
uniform vec2 uCenter;
uniform float uHalf;
uniform float uRes;
uniform int uFine;          // 1 = add fine detail (near map)
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = uCenter + (uv * 2.0 - 1.0) * uHalf;
    int n = uFine == 1 ? 3 : 1;
    float hh[3];
    vec2 offs[3] = vec2[3](vec2(0.0), vec2(2.0, 0.0), vec2(0.0, 2.0));
    for (int k = 0; k < n; k++) hh[k] = heightCoarse(p + offs[k]);
    float h = hh[0];
    if (uFine == 1) {
        float slope = length(vec2(hh[1] - h, hh[2] - h) / 2.0);
        float fine = heightFine(p, slope);
        fine *= mix(0.35, 1.0, smoothstep(-1.0, 3.0, h));
        h += fine;
    }
    fragColor = vec4(h, 0.0, 0.0, 1.0);
}
