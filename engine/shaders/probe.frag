// Computes lighting constants for a viewpoint: sun radiance after the
// atmosphere, sky irradiance on an up-facing surface, horizon radiance.
#include "common.glsl"
#include "world.glsl"
#include "sky.glsl"
uniform vec3 uProbePos;
uniform float uAmbBoost;
uniform float uTune[24];
out vec4 fragColor;
void main() {
    int i = int(gl_FragCoord.x);
    vec3 pp = toPlanet(uProbePos);
    vec3 r = vec3(0.0);
    if (i == 0) {
        r = uSunIntensity * transToSun(pp);
        // (r2b) uTune[9] = sun warmth: the sun's colour of a (1 + w) x longer path through the air at the same
        // luminance (golden light from a sun high enough to clear the valley walls; sky and haze unchanged)
        if (uTune[9] > 0.0) {
            vec3 T1 = transToSun(pp);
            vec3 T2 = pow(max(T1, vec3(1e-6)), vec3(1.0 + uTune[9]));
            r = uSunIntensity * T2 * (luma(T1) / max(luma(T2), 1e-6));
        }
    } else if (i == 1) {
        vec3 s = vec3(0.0);
        for (int a = 0; a < 8; a++)
        for (int b = 0; b < 16; b++) {
            float u1 = (float(a) + 0.5) / 8.0;
            float st = sqrt(u1), ct = sqrt(1.0 - u1);
            float ph = (float(b) + 0.5) / 16.0 * TAU;
            vec3 d = vec3(st * cos(ph), ct, st * sin(ph));
            vec3 tr;
            s += atmoScatter(uProbePos, d, 1e9, 12, tr);
        }
        r = PI * s / 128.0 * uAmbBoost;
    } else {
        // (r2b) the sky a steep face in the valley sees: azimuth average over 20 / 35 / 50 deg elevation,
        // weighted cos^2 (vertical-face cosine x solid angle); the 2 deg horizon band (pale, hazy) is
        // mostly hidden behind the opposite valley walls (lighting.glsl skyLight)
        vec3 s = vec3(0.0);
        float ws = 0.0;
        for (int e = 0; e < 3; e++) {
            float el = radians(20.0 + 15.0 * float(e));
            float we = cos(el) * cos(el);
            for (int b = 0; b < 16; b++) {
                float ph = (float(b) + 0.5) / 16.0 * TAU;
                vec3 d = vec3(cos(ph) * cos(el), sin(el), sin(ph) * cos(el));
                vec3 tr;
                s += atmoScatter(uProbePos, d, 1e9, 12, tr) * we;
                ws += we;
            }
        }
        r = s / ws * uAmbBoost;
    }
    fragColor = vec4(r, 1.0);
}
