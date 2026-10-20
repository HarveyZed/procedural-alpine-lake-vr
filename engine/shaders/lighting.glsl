// ---------------------------------------------------------------------------
// lighting.glsl : frame uniforms, hit kinds, shadows, sky/ambient light, BRDF helpers, aerial perspective
// ---------------------------------------------------------------------------
#ifndef QUALITY
#define QUALITY 1
#endif

uniform vec2 uRes;
uniform vec2 uJitter;
uniform mat3 uCamRot;
uniform float uTanHalfFov;
uniform float uAperture;
uniform float uFocusDist;
uniform int uSample;
uniform float uExposureHint;

layout(location = 0) out vec4 fragColor;
// lab-only runtime knobs for look-dev sweeps without a recompile: set with the environment variable
// ALPINE_LAB="i=v,i=v" (render.py); merged engine code must not read them (bake the chosen values in)
uniform float uLab[16];
// (r4 L3) direction of the shadow sun (render.py Look.shadow_dir(): = uSunDir unless Look.shadow_elevation is set): the
// near terrain march and the tree shadows use it like the baked terrain shadow maps; shading keeps uSunDir
uniform vec3 uShadowDir;

#define K_SKY 0
#define K_TERRAIN 1
#define K_TREE 2
#define K_ROCK 3
#define K_WATER 4
#define K_COVER 5
#define K_HERO 6

#define MAX_TERRAIN_H 4200.0
#define TREE_MAX_DIST 7000.0
#define ROCK_MAX_DIST 380.0

float gPixAng;   // radians per pixel
float gRand;     // per-pixel-per-sample random
uniform int uDebug;
bool gPrimary = true;
vec3 gShadeP = vec3(1e9);  // (r2d) point being shaded: set by terrainShadow / cloudShadow, which every shading path
                           // calls before skyLight (default: far away = the unscaled sky light)
float gTraceT0 = 0.0;    // distance already travelled before a secondary ray (footprint-based step size)   // false while tracing secondary (reflection / refraction) rays

struct Hit { float t; int kind; ivec2 cell; vec4 info; };

float detailWeight(float t) { return 1.0 - smoothstep(120.0, 260.0, t); }

// ---------------------------------------------------------------------------
// shadows
// ---------------------------------------------------------------------------
float terrainShadow(vec3 p, int steps) {
    gShadeP = p;
    vec2 sm = sunShadowMapD(p);
    // close to the camera the 1 m shadow map shows stair steps on steep ground: march the
    // bicubic heightfield toward the sun for the nearby occluders instead
    if (length(p - uCamPos) > 260.0 || steps < 40) return sm.x;
    float res = 1.0, t = 0.25;
    for (int i = 0; i < 24; i++) {
        vec3 q = p + uShadowDir * t;
        float dy = q.y - terrainH(q.xz, true);
        res = min(res, 8.0 * dy / t);
        if (res < 0.0) break;
        t += clamp(dy * 0.5, 0.25, 5.0);
        if (t > 70.0) break;
    }
    float march = smoothstep(0.0, 1.0, res);
    return sm.y < 70.0 ? march : min(sm.x, march);
}

float treeShadow(vec3 p) {
#ifdef NO_TSHADOW
    return 1.0;
#endif
    float res = 1.0;
    float t = 0.4;
    // (r2b) small, crisp sun flecks close to the camera: 0.15-0.5 m gaps between the shoots (uTune[14] = gap
    // amplitude in m) and a sharper penumbra (uTune[15] = k, ~ the sun's 0.5 deg disc) while the receiving
    // pixel is small enough to resolve them; both fade out with the pixel footprint (no shimmer)
    float fpx = length(p - uCamPos) * gPixAng;
    float fineA = uTune[14] * (1.0 - smoothstep(0.03, 0.12, fpx));
    float kS = mix(uTune[3], max(uTune[15], uTune[3]), 1.0 - smoothstep(0.02, 0.08, fpx));
    for (int i = 0; i < 18; i++) {
        vec3 q = p + uShadowDir * t;
        float dy = q.y - terrainH(q.xz, false);
        if (dy > 36.0) break;
        // (perf4) tree-free space: treesSDFShadow would return its cap (the rest of the ray too beyond 60 m: <= 18 x 3 m)
        float tf = treeFreeDist(q);
        if (tf > 60.0) break;
        float d = tf > 0.0 ? TREE_CELL * 0.55 : treesSDFShadow(q);
        if (d < TREE_CELL * 0.55 - 1e-3) {                               // ignore the "no tree near" cap
            // ragged, porous crown edge: gaps between the branch tiers let sun flecks through
            // (uTune[12] = gap amplitude in m; world-fixed noise, ~1.5-3 m features)
            if (uTune[12] > 0.0 && d < 1.5) d += uTune[12] * (texture(uNoise3, q * 0.37 + 0.13).g - 0.45) * 2.2;
            if (fineA > 0.0 && d < 0.9) d += fineA * (texture(uNoise3, q * 2.9 + 0.71).b - 0.47) * 2.0;
            res = min(res, kS * d / t);
        }
        if (res < 0.02) return 0.0;
        t += clamp(d, 0.4, 3.0);
    }
    return saturate(res);
}

float cloudShadow(vec3 p) {
    gShadeP = p;
    if (uCloudCover <= 0.001) return 1.0;
    // optical depth of the actual cloud bodies along the sun ray through the layer
    float t0 = (CL_BOT - p.y) / max(uSunDir.y, 0.05);
    float t1 = (CL_TOP - p.y) / max(uSunDir.y, 0.05);
    float od = 0.0;
    const int N = 6;
    float dt = (t1 - t0) / float(N);
    for (int i = 0; i < N; i++) {
        od += cloudDensity(p + uSunDir * (t0 + dt * (float(i) + 0.5)), false) * dt;
    }
    return mix(1.0, exp(-od * 0.012), 0.85);
}

// (r2d) near-field share of the sky fill. The steep-face band term (tune 0) and the boosted dome (amb_boost)
// were set so that the distant shaded faces read sky-blue and only ~0.75 stop under the lit ones;
// near the camera the same fill flattened every boulder and trunk: r1 boulder lit / shade 2.2-2.6 in the HDR
// (graded ~1.9) vs a target of 9 (boulder_lit / boulder_shadow 7-11), with a blue cast (target shade
// #40403c is neutral). At the lake shore the 20-50 deg sky band is largely hidden by the valley walls and
// the shade is lit by the warm ground instead. uTune[16] = band share near the camera, uTune[19] = dome
// share near the camera, both ramping to 1 between uTune[17] and uTune[18] m from the camera;
// uTune[20] = extra warm ground bounce near the camera (x the sunlit-ground term, 0 = off)
float nearSkyRamp() {
    return smoothstep(uTune[17], max(uTune[18], uTune[17] + 1.0), length(gShadeP - uCamPos));
}

vec3 skyLight(vec3 n, float ao) {
    // sky dome irradiance: brighter from above; bounce from the sunlit ground
    float up = 0.5 + 0.5 * n.y;
    float nr = nearSkyRamp();
    vec3 sky = uSkyAmb * mix(uTune[1], 1.0, up) * mix(uTune[19], 1.0, nr);
    // (r5 light5) the single-scattering dome is a saturated Rayleigh blue (b/r 3.7); multiple scattering, the Mie aureole
    // and the sunlit valley around make the real fill paler: 30 % toward its luminance (near shade grey,
    // #40403c / #494d4b; the distant shade keeps its blue from the in-scatter)
    sky = mix(sky, vec3(luma(sky)), 0.3);
    // steep faces see more of the sky's side than the up-facing dome average says. (r2b) uSkyHorizon is
    // now the probe's 20-50 deg elevation band (blue), not the pale 2 deg horizon: in a valley the low
    // horizon is hidden behind the opposite walls, and the shaded faces get their blue from this sky
    // (target #5c729d) instead of from the aerial-perspective veil
    sky += uSkyHorizon * uTune[0] * (1.0 - abs(n.y)) * mix(uTune[16], 1.0, nr);
    vec3 bounce = uSunCol * max(uSunDir.y, 0.0) * vec3(0.10, 0.11, 0.08) * (1.0 - up) * uTune[11];
    bounce += uSunCol * max(uSunDir.y, 0.0) * vec3(0.11, 0.10, 0.075) * (1.0 - abs(n.y)) * uTune[20] * (1.0 - nr);
    // light reflected by the surrounding sunlit valley walls fills what the sky can't reach
    vec3 env = uSunCol * max(uSunDir.y, 0.0) * vec3(0.10, 0.11, 0.095) * (1.0 - ao) * uTune[2];
    return (sky + bounce) * ao + env;
}

// (r2b) foliage: the sky light inside a crown comes mostly from above through the needles; the steep-face
// band term (for rock walls) turned the distant forest blue-grey / frosted (forest B 30-45 levels
// high). Keeps 30 % of it. For shade_trees.glsl (trees owner): skyLight(n, aoT) -> skyLightFoliage(n, aoT)
vec3 skyLightFoliage(vec3 n, float ao) {
    return skyLight(n, ao) - uSkyHorizon * (uTune[0] * 0.7 * (1.0 - abs(n.y)) * ao) * mix(uTune[16], 1.0, nearSkyRamp());
}

vec3 litSurface(vec3 alb, vec3 n, vec3 rd, float sun, float ao, float spec, float rough) {
    float ndl = max(dot(n, uSunDir), 0.0);
    if (uDebug == 62) return vec3(ndl * sun, luma(skyLight(n, ao)), ao);
    vec3 col = alb * (uSunCol * ndl * sun + skyLight(n, ao));
    if (spec > 0.0) {
        vec3 hv = normalize(uSunDir - rd);
        float a = rough * rough;
        float nh = max(dot(n, hv), 0.0);
        float dd = nh * nh * (a * a - 1.0) + 1.0;
        float D = a * a / (PI * dd * dd);
        float F = 0.04 + 0.96 * pow(1.0 - max(dot(hv, -rd), 0.0), 5.0);
        col += uSunCol * sun * ndl * D * F * spec * 0.25;
    }
    return col;
}

// fraction of the view path [0, t] whose air sees the sun (the terrain shadow map stores, per texel, the
// lowest height that sees the sun): the in-scatter of shadowed valley air lacks its direct-sun part.
// Matters at low sun (golden hour: the shaded valley side is not veiled by bright haze); at
// midday the shadow volumes are short. 8 shadow-map taps, denser near the camera, Mie-weighted by height.
float aerialLitFrac(vec3 ro, vec3 rd, float t) {
    float tl = min(t, 16000.0);
    float lit = 0.0, wsum = 0.0, prev = 0.0;
    for (int i = 0; i < 8; i++) {
        float f = (float(i) + 0.5 + 0.5 * (gRand - 0.5)) / 8.0;
        float s = tl * f * f;
        float ds = tl * ((float(i) + 1.0) * (float(i) + 1.0) - float(i) * float(i)) / 64.0;
        vec3 q = ro + rd * s;
        float w = ds * (0.35 + exp(-max(q.y, 0.0) / 1200.0));
        lit += w * sunShadowMap(q + vec3(0.0, 2.0, 0.0));
        wsum += w;
    }
    // beyond 16 km: lit
    float fl = lit / max(wsum, 1e-3);
    return mix(fl, 1.0, saturate((t - tl) / max(t, 1.0)) * 0.5);
}

vec3 applyAerial(vec3 col, vec3 ro, vec3 rd, float t, int n) {
    if (uDebug == 62) return col;
    if (t < 30.0) return uDebug == 64 ? vec3(0.0) : col;
    vec3 tr;
    vec3 ins = atmoScatter(ro, rd, t, n, tr);
    if (uTune[4] > 0.0 && n >= 8) {
        float lf = aerialLitFrac(ro, rd, t);
        ins *= mix(1.0, uTune[5] + (1.0 - uTune[5]) * lf, uTune[4]);
    }
    // (r2b) the finite-path in-scatter (sky.glsl: x uSkyBoost x 0.6, +55 % mid-distance lift) is ~2.5x out of
    // balance with its transmittance below ~8 km: at 3-5 km the air passes 96 % of the light, yet the glow
    // was 40-75 % of the lit faces and 75 % of the shade, turning the whole massif blue (#536c8e vs the
    // target #747981). uTune[13] = in-scatter scale up to 3 km, ramping to 1 (calibrated) at 15 km.
    // (r4 L9) ramp 2-9 km (was 3-15 km): the 3-8 km faces get the calibrated in-scatter and its full chroma
    float rNear = smoothstep(2000.0, 9000.0, t);
    ins *= mix(uTune[13], 1.0, rNear);
    // (r4 L9) thicker air toward the far ranges (target: 15+ km ranges ~#7c93b4 at ~0.7x the horizon sky, lit / shade
    // <= 3; the render kept a snow-bright, high-contrast massif): more in-scatter, less transmittance beyond 7 km
    float rFar = smoothstep(7000.0, 20000.0, t);
    // (r4 L9) ins x2 at 20 km: the far range was 0.58x the horizon sky (target 0.65-0.75x)
    ins *= 1.0 + 1.0 * rFar;
    tr = pow(tr, vec3(1.0 + 0.8 * rFar));
    // (r4 L15) contre-jour: the forward-scattered veil toward the sun lifts the 1-6 km massif (flat blue-grey,
    // lit ~= shade); side- and front-lit views unchanged (same 'mid' band as sky.glsl atmoScatter)
    float midB = smoothstep(700.0, 3000.0, t) * (1.0 - smoothstep(5000.0, 15000.0, t));
    float cjB = midB * smoothstep(0.35, 0.85, dot(rd, uSunDir));
    // (r5 light5) 1.7 -> 0.5: with the lift the 1-6 km massif of the toward-sun views (r1, shots 0 / 1) was 60-70 % veil
    ins *= 1.0 + 0.5 * cjB;
    // (r2d) uTune[21] = chroma of the in-scatter below ~8 km (1 = as computed): the near valley air is aerosol
    // (Mie) haze, near-white; the Rayleigh-blue glow turned the distant lit granite (0.75, 1, 1.46) vs the
    // target (0.89, 1, 1.07). With a whiter veil the haze can again do the distance compression (target: lit /
    // shade 1.7 at 3-5 km) instead of the grade's detail band
    float lI = dot(ins, vec3(0.2126, 0.7152, 0.0722));
    // (r4 L15) the contre-jour veil keeps most of its blue (massif target ~(118, 138, 166), was grey (113, 125, 142))
    ins = mix(vec3(lI), ins, mix(uTune[21], 1.0, max(rNear, 0.75 * cjB)));
    if (uDebug == 64) return ins;
    if (uDebug == 65) return vec3(t / 10000.0, tr.g, 0.0);
    return col * tr + ins;
}

#define SKY_VIEW_GAIN 0.5
vec3 skyWithClouds(vec3 ro, vec3 rd, int nAtmo, int nCloud) {
    vec3 col = skyRadiance(ro, rd, nAtmo);
    // (r5 light5) the directly seen sky x SKY_VIEW_GAIN (reflections keep the full sky: gPrimary is false in every other
    // pass). The intended look holds the sky at about the luminance of the sunlit rock (target: lit boulder
    // Y .42, zenith .385); in the HDR it was 2.5-5x the lit rock, so the grade had to compress everything bright (lit rock,
    // beach, snow) along with it: the grey, flat r4 frames. Clouds keep their radiance (cloud / sky up toward the target 2.2)
    if (gPrimary) col *= SKY_VIEW_GAIN;
    vec4 cl = renderClouds(ro, rd, 1e9, gRand, nCloud);
    if (cl.a < 0.999) {
        // distant clouds fade into the sky
        float tc = (CL_BOT - ro.y) / max(rd.y, 1e-3);
        float fade = exp(-max(tc - 6000.0, 0.0) / 30000.0);
        col = col * mix(1.0, cl.a, fade) + cl.rgb * fade;
    }
    return col;
}

