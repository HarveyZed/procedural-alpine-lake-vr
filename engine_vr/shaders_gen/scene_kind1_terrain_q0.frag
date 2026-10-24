#version 450
#define QUALITY 0
#define SCENE_PASS 1
// ---------------------------------------------------------------------------
// scene.frag : the full alpine-lake renderer (one sample per pixel per pass)
// ---------------------------------------------------------------------------
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
uint hash3u(ivec3 c) { return hashu(uint(c.x) * 0x9E3779B1u + hashu(uint(c.y) * 0x85EBCA77u + hashu(uint(c.z) + 0x27d4eb2fu))); }
float u2f(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }
float hash12i(ivec2 c) { return u2f(hash2u(c)); }
vec2 hash22i(ivec2 c) { uint h = hash2u(c); return vec2(u2f(h), u2f(hashu(h))); }
vec4 hash42i(ivec2 c) {
    uint a = hash2u(c); uint b = hashu(a); uint d = hashu(b); uint e = hashu(d);
    return vec4(u2f(a), u2f(b), u2f(d), u2f(e));
}
float hash13i(ivec3 c) { return u2f(hash3u(c)); }
float hash11(float x) { return u2f(hashu(floatBitsToUint(x))); }

// 2D gradient noise with analytic derivatives. returns (value, d/dx, d/dy), value ~[-0.7,0.7]
vec2 grad2(ivec2 c) { float a = hash12i(c) * TAU; return vec2(cos(a), sin(a)); }
vec3 gnoised(vec2 p) {
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec2 ga = grad2(c), gb = grad2(c + ivec2(1, 0)), gc = grad2(c + ivec2(0, 1)), gd = grad2(c + ivec2(1, 1));
    float va = dot(ga, f), vb = dot(gb, f - vec2(1, 0)), vc = dot(gc, f - vec2(0, 1)), vd = dot(gd, f - vec2(1, 1));
    float v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
    vec2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
           + du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va);
    return vec3(v, d);
}
float gnoise(vec2 p) { return gnoised(p).x; }

// 2D value noise [0,1]
float vnoise(vec2 p) {
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash12i(c), b = hash12i(c + ivec2(1, 0)), d = hash12i(c + ivec2(0, 1)), e = hash12i(c + ivec2(1, 1));
    return mix(mix(a, b, u.x), mix(d, e, u.x), u.y);
}

// 3D value noise [0,1]

float fbm2(vec2 p, int oct) {
    float s = 0.0, a = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < oct; i++) { s += a * gnoise(p); p = m * p; a *= 0.5; }
    return s;
}

// 2D Voronoi: returns (F1 distance, F2 distance, cell id hash)
vec3 voronoi2(vec2 p) {
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i);
    float d1 = 8.0, d2 = 8.0, id = 0.0;
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        ivec2 o = ivec2(x, y);
        vec2 r = vec2(o) + hash22i(c + o) * 0.85 + 0.075 - f;
        float d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; id = hash12i(c + o + ivec2(17, 31)); }
        else if (d < d2) { d2 = d; }
    }
    return vec3(sqrt(d1), sqrt(d2), id);
}

// voronoi with the offset to the nearest feature point: (dx, dy, F1, id)
vec4 voronoiV(vec2 p) {
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i);
    float d1 = 8.0; vec2 best = vec2(0.0); float id = 0.0;
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        ivec2 o = ivec2(x, y);
        vec2 r = vec2(o) + hash22i(c + o) * 0.8 + 0.1 - f;
        float d = dot(r, r);
        if (d < d1) { d1 = d; best = r; id = hash12i(c + o + ivec2(17, 31)); }
    }
    return vec4(-best, sqrt(d1), id);
}

mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float smin(float a, float b, float k) { float h = saturate(0.5 + 0.5 * (b - a) / k); return mix(b, a, h) - k * h * (1.0 - h); }
float smax(float a, float b, float k) { return -smin(-a, -b, k); }

// ---------------------------------------------------------------------------
// world.glsl : runtime access to the baked world (heights, AO, water, forest)
// ---------------------------------------------------------------------------
#define NEAR_HALF 2048.0
#define NEAR_RES 4096.0
#define FAR_HALF 16384.0
#define FAR_RES 8192.0
#define TREE_CELL 4.0

layout(set=0, binding=0) uniform sampler2D uHNear;    // R32F, 1 m texels, +-2048 m
layout(set=0, binding=1) uniform sampler2D uHFar;     // R32F, 4 m texels, +-16384 m
layout(set=0, binding=2) uniform sampler2D uAONear;   // sky visibility
layout(set=0, binding=3) uniform sampler2D uAOFar;
layout(set=0, binding=4) uniform sampler2D uForest;   // RGBA8 on the far grid: density, larch, scree, meadow
layout(set=0, binding=5) uniform sampler2D uWaterLvl; // R32F water surface level (bilinear), -1e4 = none
layout(set=0, binding=6) uniform sampler2D uWaterLvlNear; // R32F 1 m level map (lakes + stream channel)
layout(set=0, binding=7) uniform sampler2D uStreamNear;   // RGBA16F: flow dir xz, foam, stream mask
layout(set=0, binding=8) uniform sampler3D uNoise3;   // RGBA8 128^3 tileable: fbm, smooth, worley-fbm, worley
layout(set=0, binding=9) uniform sampler2D uTrees;    // RGBA32F per 4 m cell (+-8192 m): baseY, signed H (neg = larch), crown R, crown base
layout(set=0, binding=10) uniform sampler2D uShNear;   // RG32F sun shadow map: shadowTop (abs y), occluder distance
layout(set=0, binding=11) uniform sampler2D uShFar;
#define SH_FAR_HALF 16384.0

// port(P1): the engine's loose pass uniforms, grouped into one block for Vulkan GLSL (unnamed instance -> members stay global)
layout(std140, set=1, binding=0) uniform _p1_pass {
    vec3 uCamPos;
    float uTime;
    vec4 uTrail[32];
    int uTrailN;
    float uPixAng;
    vec3 uSunDir;
    float uSunIntensity;
    float uHaze;
    float uCloudCover;
    vec2 uWind;
    vec3 uSunCol;
    vec3 uSkyAmb;
    vec3 uSkyHorizon;
    float uSkyBoost;
    float uSkyDesat;
    vec2 uWaterWind;
    float uWaveStrength;
    vec3 uWaterIns;
    float uTune[24];
    vec4 uStreamCorrBox;
    vec2 uRes;
    vec2 uJitter;
    mat3 uCamRot;
    float uTanHalfFov;
    float uAperture;
    float uFocusDist;
    int uSample;
    float uExposureHint;
    float uLab[16];
    vec3 uShadowDir;
    int uDebug;
    vec4 uWF;
    vec2 uWFDir;
    vec4 uWFNotch;
    int uFallsOn;
    vec4 uFallK[48];
    vec4 uFallInfo[2];
    vec4 uFallBox[4];
    vec4 uStream[96];
    int uStreamN;
    vec4 uStreamBox;
    float uStreamS[96];
    vec4 uStreamMouth;
    vec4 uHeroA[6];
    vec4 uHeroB[6];
    int uHeroN;
    vec4 uW2K[8];
};



// forest trail polyline (features.json "trail"), used by ground.glsl and terrain_shade.glsl
#define GC_TRAIL 1
#define GC_SKYOCC 1


     // radians per pixel of the current camera

// beyond the map edge the far maps are mirrored (no clamped streaks toward the horizon)
vec2 farUV(vec2 p) { vec2 u = p / (2.0 * FAR_HALF) + 0.5; return 1.0 - abs(1.0 - mod(u, 2.0)); }
vec2 nearUV(vec2 p) { return p / (2.0 * NEAR_HALF) + 0.5; }

vec4 cubicW(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x, y = s.y - 4.0 * s.x, z = s.z - 4.0 * s.y + 6.0 * s.x;
    return vec4(x, y, z, 6.0 - x - y - z) * (1.0 / 6.0);
}
float texBicubic(sampler2D tex, vec2 uv, float res) {
    uv = uv * res - 0.5;
    vec2 fxy = fract(uv); uv -= fxy;
    vec4 xc = cubicW(fxy.x), yc = cubicW(fxy.y);
    vec4 c = uv.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xc.xz + xc.yw, yc.xz + yc.yw);
    vec4 off = (c + vec4(xc.yw, yc.yw) / s) / res;
    float s0 = texture(tex, off.xz).r, s1 = texture(tex, off.yz).r;
    float s2 = texture(tex, off.xw).r, s3 = texture(tex, off.yw).r;
    float sx = s.x / (s.x + s.y), sy = s.z / (s.z + s.w);
    return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
}

float nearBlend(vec2 p) {
    vec2 a = abs(p) / NEAR_HALF;
    return smoothstep(0.88, 0.97, max(a.x, a.y));
}

// base terrain height; hq = bicubic near map
float terrainH(vec2 p, bool hq) {
    float w = nearBlend(p);
    float hf = 0.0, hn = 0.0;
    if (w > 0.0) hf = texture(uHFar, farUV(p)).r;
    if (w < 1.0) hn = hq ? texBicubic(uHNear, nearUV(p), NEAR_RES) : texture(uHNear, nearUV(p)).r;
    return mix(hn, hf, w);
}

// sub-meter rock / ground relief near the camera
float microRelief(vec2 p) {
    float n = 0.22 * gnoise(p * 0.55 + 1.7) + 0.09 * gnoise(rot2(1.1) * p * 1.3 + 4.2)
            + 0.035 * gnoise(rot2(2.3) * p * 3.1);
    return n;
}

// ---------------------------------------------------------------------------
// (world3) granite relief of the steep rock faces (the walls are fractured into
// planar slabs and blocks with crisp edges - lit and shaded facets, V-shaped joints, notched crests). The 1-4 m
// height maps only hold the eroded shape (smooth folds + fall-line rills: the "melted wax" look), so the
// structure is added at run time as a height change D(x, z) <= 0 inside terrainHD (so the march, the normals
// and the shading all see it):
//  - planar joint facets: a 3D Voronoi of tilted slabs evaluated at the surface point (x, h, z) - in 3D the
//    cells are round ON the surface whatever its slope (2D cells in x-z stretch 2-4x down a 60-75 deg face:
//    curtains / organ pipes, t1-t5). Each cell = a plane through its point, capped by a gentle paraboloid;
//    the surface is the smooth max of the caps of the 2x2x2 neighbourhood (crisp V-creases at the cell
//    borders, smoothed over ~0.6 px); tilts from three joint families of the region (parallel facets repeat:
//    coherent lit / shaded planes, not a random mosaic)
//  - three octaves (cells 42 / 15 / 5.4 m), each faded before it is < ~4 px (no shimmer, the film moves)
//  - steep rock only (> ~37-49 deg; weaker on 27-42 deg beyond 350-500 m), > 30-70 m above the lake, no
//    forest (trees stand on the baked heights), not on scree < 45 deg, tarn shore / inflow stream guarded
//  - scree fans: 12 / 4 / 1.5 m block lumps (worley channel of the noise volume) beyond 300-450 m
// gRkW / gRkD = weight / height change of the last call (terrain_shade.glsl reads them at the shaded point).
// ---------------------------------------------------------------------------
float gRkW = 0.0, gRkD = 0.0;
bool gRkFine = false;                     // true while shading terrain (shadeTerrain): the 4th relief octave is on
vec2 gMzP = vec2(1e9); float gMzV = 0.0;  // cache of the massive-zone mask (rockRelief)
float hRawB(vec2 p) {                     // bilinear base height (cheap taps for the gradient)
    float w = nearBlend(p);
    if (w >= 1.0) return texture(uHFar, farUV(p)).r;
    float hn = texture(uHNear, nearUV(p)).r;
    return w > 0.0 ? mix(hn, texture(uHFar, farUV(p)).r, w) : hn;
}
float segDist(vec2 p, vec2 a, vec2 b) { vec2 ab = b - a; return length(p - a - ab * clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0)); }
// one octave of 3D slab facets (height change <= 0); F0..F2 = joint-family tilt directions
float rkFacets3(vec3 P, float C, float T, float fp, vec3 F0, vec3 F1, vec3 F2, uint seed) {
    vec3 q = P / C;
    vec3 b = floor(q - 0.5);
    float k = max(0.6 * fp, 0.03 * C);
    const float kc = 0.15;                    // (t10: flat slab caps, creases only at the borders)
    float best = -1e9;
    // (t20, perf) two hashes per cell (was 9 hashu: hash3u + 6): the cell's random values are bit fields of them
    ivec3 so = ivec3(int(seed) * 7919, int(seed) * 31, int(seed));
    for (int i = 0; i < 8; i++) {
        vec3 cell = b + vec3(float(i & 1), float((i >> 1) & 1), float(i >> 2));
        ivec3 ic = ivec3(cell) + so;
        uint hh = hashu(uint(ic.x) * 0x9E3779B1u ^ uint(ic.y) * 0x85EBCA77u ^ uint(ic.z) * 0xC2B2AE3Du);
        uint h2 = hashu(hh ^ 0x68E31DA4u);
        vec3 c = cell + 0.5 + 0.5 * (vec3(uvec3(hh, hh >> 10u, hh >> 20u) & 1023u) * (1.0 / 1023.0) - 0.5);
        vec3 r = (q - c) * C;
        uint fam = h2 % 3u;
        vec3 fd = fam == 0u ? F0 : (fam == 1u ? F1 : F2);
        float sg = (h2 & 0x400u) != 0u ? -1.0 : 1.0;
        vec3 tl = (T * (0.4 + 0.6 * float((h2 >> 12u) & 255u) * (1.0 / 255.0)) * sg) * fd;
        // (t20: the random slab offset 0.15 -> 0.08 C: deep pits between the slabs held flat sky-lit (blue) shadow
        // pockets on the frontally lit near wall, shot 0 left)
        float v = -0.08 * C * float(h2 >> 24u) * (1.0 / 255.0) + dot(tl, r) - kc * dot(r, r) / C;
        best = i == 0 ? v : smax(best, v, k);
    }
    return min(best - 0.25 * C, 0.0);
}
float rockRelief(vec2 p, float h) {
    gRkW = 0.0; gRkD = 0.0;
    if (h < 30.0) return 0.0;
    float dist = length(vec3(p.x, h, p.y) - uCamPos);
    float fp = max(dist * uPixAng, 0.02);
    const float C0 = 42.0;
    if (C0 < 4.2 * fp) return 0.0;
    // gradient over 16 m for the gates ((perf, session 4) forward differences from h: 2 taps instead of 4 - rockRelief
    // runs in every march step)
    vec2 g = vec2(hRawB(p + vec2(16.0, 0.0)) - h, hRawB(p + vec2(0.0, 16.0)) - h) / 16.0;
    float sl = length(g);
    float alt = smoothstep(30.0, 70.0, h);
    // (t23) the 24-42 deg band in ~300 m patches (smooth turf / slab slopes next to broken ones): at frame level every
    // lower slope carried the same even crumple (s7 lower right; target: the slopes alternate smooth and rocky ground)
    float wSt = smoothstep(0.75, 1.15, sl);
    float wG = 0.45 * smoothstep(0.45, 0.75, sl) * smoothstep(350.0, 500.0, dist);
    if (wG > wSt) wG *= mix(1.0, smoothstep(0.40, 0.62, texture(uNoise3, vec3(p * (1.0 / 330.0), 0.19)).g), 0.8);
    float w = max(wSt, wG) * alt;
    float wS = smoothstep(0.25, 0.5, sl) * (1.0 - smoothstep(0.95, 1.25, sl)) * smoothstep(300.0, 450.0, dist) * alt
             * smoothstep(1.2 * fp, 2.5 * fp, 12.0);
    if (w <= 0.0 && wS <= 0.0) return 0.0;
    // the tarn shore (y 520), the inflow stream corridor (the falls and the trail sit < 30 m above the lake)
    float gd = 1.0 - (1.0 - smoothstep(25.0, 50.0, abs(h - 520.0))) * (1.0 - smoothstep(700.0, 1000.0, length(p - vec2(-1640.0, -430.0))));
    if (p.y > 1400.0 && p.y < 2400.0) {
        float ds = min(min(segDist(p, vec2(191.0, 1545.0), vec2(263.0, 1809.0)), segDist(p, vec2(263.0, 1809.0), vec2(377.0, 1985.0))),
                       segDist(p, vec2(377.0, 1985.0), vec2(419.0, 2290.0)));
        gd *= smoothstep(70.0, 140.0, ds);
    }
    w *= gd; wS *= gd;
    if (w <= 0.0 && wS <= 0.0) return 0.0;
    vec4 fo = texture(uForest, farUV(p));
    float Ds = 0.0;
    wS *= smoothstep(0.3, 0.6, fo.b) * (1.0 - smoothstep(0.0, 0.05, fo.r));
    if (wS > 0.0) {
        float l0 = texture(uNoise3, vec3(p * (1.0 / 192.0), 0.41)).a * smoothstep(1.2 * fp, 2.5 * fp, 12.0);
        float l1 = texture(uNoise3, vec3(p * (1.0 / 64.0), 0.13)).a;
        float l2 = texture(uNoise3, vec3(p * (1.0 / 24.0), 0.57)).a * smoothstep(1.2 * fp, 2.5 * fp, 1.5);
        Ds = -wS * ((1.0 - l0) + 1.6 * (1.0 - l1) + 0.6 * (1.0 - l2));
    }
    gRkD = Ds;
    if (w <= 0.0) return Ds;
    // (scree only holds on < ~45 deg: the talus map also marks the walls above the fans)
    // (t23: 0.95..1.3 -> 0.75..1.0 on the 16 m slope: the 43-52 deg walls of the debris-mapped cirque (s5) had neither
    // the relief nor the scree albedo (gated at ~52 deg on the 64 m slope) - plaster with fine vertical flute stripes)
    w *= (1.0 - smoothstep(0.0, 0.025, fo.r)) * (1.0 - smoothstep(0.25, 0.6, fo.b) * (1.0 - smoothstep(0.75, 1.0, sl)));
    if (w <= 0.0) return Ds;
    // regional joint trend (~2 km) and patches of massive vs closely jointed rock (~300 m)
    vec4 nz = texture(uNoise3, vec3(p * (1.0 / 4096.0), 0.37));
    float ang = 0.5 + 2.4 * (nz.g - 0.5);
    vec3 F0 = vec3(cos(ang), 0.25, sin(ang)), F1 = vec3(-sin(ang), -0.2, cos(ang)), F2 = vec3(0.3 * cos(ang), 1.0, 0.3 * sin(ang));
    // (t11 tried 0.25 for more massive slabs: those patches then showed the base map's fall-line draperies
    // again - wax; 0.45 kept)
    // (t24) the massive zones of the bake (heightfn.glsl reliefPost: big planar faces + chimneys, ~40 % of the steep
    // high faces in ~500 m patches, same mask): the slab relief at 0.45 there (was an unrelated 640 m / 4 km mask)
    // (perf) the mask varies over ~500 m: cached per invocation and recomputed only > 16 m away from the last evaluation
    // (rockRelief runs ~20x per pixel: march steps, normal taps, cavity taps, relief-shadow steps; the two gradient
    // noises were ~15 % of the frame in shots 1 / 7)
    if (distance(p, gMzP) > 16.0) {
        gMzP = p;
        gMzV = smoothstep(0.0, 0.3, gnoise(p * 0.0013 + 44.0) + 0.5 * gnoise(p * 0.0035 + 9.0));
    }
    w *= mix(1.0, 0.45, gMzV);
    vec3 P = vec3(p.x, h, p.y);
    // (t11) gentler tilts on the 27-45 deg ground: there the steep side of a V-joint stood in the sun's shadow
    // as a crisp dark sliver - black dashes all over the grassy slopes (s7 lower left)
    // (t15: tilt 0.65 -> 0.8, variant Tp: crisper lit / shaded slab planes at 1:1, same frame-level look)
    // (t19: tilt floor 0.5 -> 0.35 on the 24-45 deg ground, with the turf now hiding the relief - terrain_shade.glsl)
    // (r4 M10) hierarchy on the 1-3 km faces: gentler facet tilts and the 15 / 5.4 / 1.9 m octaves x0.45 from 0.7-2 km (an
    // even crumple of hard lit / shaded slivers everywhere read as tinfoil: high-pass 1.41 vs a target of 0.61)
    float D = 0.0, C = C0, amp = 1.0, T = mix(0.8, 0.55, smoothstep(800.0, 2000.0, dist)) * mix(0.35, 1.0, smoothstep(0.75, 1.2, sl));
    float kF = mix(1.0, 0.45, smoothstep(700.0, 2000.0, dist));
    for (int k = 0; k < 4; k++) {
        float v = smoothstep(4.2 * fp, 8.5 * fp, C);
        // (session 4, perf) the finest octave (1.9 m, <= ~0.5 m deep, it only lowers the surface) only in the shading
        // (normal, cavity, relief shadow): the march hits the surface without it (< ~1 px at the footprints where it
        // exists) - s0's 1.2 km wall, where all four octaves run in every march step
        if (v <= 0.0 || (k == 3 && !gRkFine)) break;
        D += v * amp * rkFacets3(P + 37.0 * float(k), C, T, fp, F0, F1, F2, uint(5 + 13 * k));
        C *= 0.36; amp *= k == 0 ? 0.7 * kF : 0.7;
    }
    // (t20) bedding / sheeting ledges (the big faces carry one structural grain - parallel inclined ribs and
    // ledges 15-40 m apart running across the whole face and across its fall-line gullies; the slab crumple alone has
    // no direction). One plane family per region (dip ~15-50 deg, dip direction from the regional joint trend, both
    // turning over ~4 km); at every plane the face steps back: a sawtooth in the plane coordinate (a gentle ramp over
    // the layer, then a crisp riser >= ~1.5 px), lines warped over ~640 m, on ~half of the faces (~180 m patches)
    // (t20: 3.5 / 6 m variants: 6 m breaks the vertical fall-line grain of the walls best at frame level; 5 m kept)
    // (r4 sw1: 5 -> 2.5 m, x kF: the risers cut by the fluted / curved faces drew contour wood-grain terraces over the
    // shot 7 / shot 5 walls at 1-2 km; face high-pass 0.43 -> 0.37, shot 0's near-wall ledges unchanged. A 120 m / 18 m
    // structural octave (r4 M2 d) was tried and dropped: more of the same terracing)
    const float bA = 2.5;
    {
        const float SB = 24.0;
        // (t24: steep faces only - on the 24-42 deg band the parallel risers read as a washboard / wood grain, c7 0.8/0.6)
        float vb = smoothstep(0.9, 1.25, sl);
        float dd = ang + 1.3, dip = 0.55 + 1.6 * (nz.r - 0.5);
        vec3 Nd = vec3(cos(dd) * sin(dip), cos(dip), sin(dd) * sin(dip));
        // (t29) only where the planes CUT the face (> ~40 deg between the plane and the face): where they lay nearly
        // parallel to it, the face's own small undulations cut them into contour-line ripples - a fingerprint /
        // herringbone hatch over the far walls at 1:1 (crop c7 0.5/0.35, t20-t28); and only while a layer stays
        // > ~7 px on screen for the face's foreshortening
        vec3 nF = normalize(vec3(-g.x, 1.0, -g.y));
        float cutS = length(cross(Nd, nF));
        float fsh = max(abs(dot(nF, (P - uCamPos) / dist)), 0.05);
        vb *= smoothstep(0.62, 0.8, cutS) * smoothstep(3.0 * fp, 7.0 * fp, SB / cutS * fsh);
        // (t30) the cut / foreshortening gates alone did not remove the hatch (t29): it is the 5 m risers themselves -
        // nearly vertical heightfield steps of ~1.5 px, sampled by the march / the normal taps into 2 px diagonal dashes
        // once a footprint is > ~0.8 m. Bedding only where a layer is > 16-30 px (the near walls: s0's 1.3 km face,
        // where it gave real ledges, t20) and risers >= 12 % of the layer
        vb *= smoothstep(16.0 * fp, 30.0 * fp, SB);
        if (vb > 0.0) {
            vec4 wz = texture(uNoise3, vec3(p * (1.0 / 640.0), 0.71));
            vec4 bz = texture(uNoise3, P * (1.0 / 180.0) + 0.29);
            float s = dot(P, Nd) / SB + 1.5 * (wz.g - 0.5) + 0.6 * (bz.r - 0.5);
            float u = fract(s);
            float kr = clamp(1.5 * fp / SB, 0.12, 0.45);
            float r = min(u / (1.0 - kr), (1.0 - u) / kr);
            float on = smoothstep(0.42, 0.62, bz.g + 0.15 * (wz.a - 0.5));
            D -= vb * bA * kF * on * r;
        }
    }
    float sec = min(sqrt(1.0 + sl * sl), 3.0);
    // (t13: x0.8 at 4.5 km+: less frame-level crumple, variant farAmp)
    D *= w * mix(1.0, sec, 0.5) * (1.0 - 0.2 * smoothstep(2000.0, 4500.0, dist));
    gRkW = w; gRkD = D + Ds;
    return D + Ds;
}

// (round 3) the shading normal reads the 4 m far map bicubic: the bilinear map's derivative jumps at every texel
// border (a 4 m grid of creases) drew a fine diagonal hatch over every slope beyond the 1 m near map when the
// footprint is < ~1 m (the "brushed metal" scree of the hero cone, crop c5; speckle on the far turf). The march
// keeps the cheap bilinear surface (the two differ by centimetres)
float terrainHN(vec2 p) {
    float w = nearBlend(p);
    float hf = 0.0, hn = 0.0;
    if (w > 0.0) hf = texBicubic(uHFar, farUV(p), FAR_RES);
    if (w < 1.0) hn = texBicubic(uHNear, nearUV(p), NEAR_RES);
    return mix(hn, hf, w);
}

// (t22) the base surface's gradient over >= 3 m on the 4 m far map: its texel-scale erosion rills hung down every
// steep slope beyond the near map as dense vertical stripes at 1:1 final pixels (s5 cirque walls / cone, crop c5,
// debug 28); the slab relief and the micro relief keep the pixel-footprint taps. (Same as terrainHDN's sum.)
vec3 terrainNormal(vec2 p, float e, float detail) {
    float eB = mix(e, max(e, 3.0), nearBlend(p));
    float bx0 = terrainHN(p - vec2(eB, 0.0)), bx1 = terrainHN(p + vec2(eB, 0.0));
    float bz0 = terrainHN(p - vec2(0.0, eB)), bz1 = terrainHN(p + vec2(0.0, eB));
    vec2 g = vec2(bx1 - bx0, bz1 - bz0) / (2.0 * eB);
    float hc = 0.25 * (bx0 + bx1 + bz0 + bz1);
    vec2 px0 = p - vec2(e, 0.0), px1 = p + vec2(e, 0.0), pz0 = p - vec2(0.0, e), pz1 = p + vec2(0.0, e);
    float rx0 = rockRelief(px0, hc - g.x * e), rx1 = rockRelief(px1, hc + g.x * e);
    float rz0 = rockRelief(pz0, hc - g.y * e), rz1 = rockRelief(pz1, hc + g.y * e);
    if (detail > 0.001) {
        rx0 += detail * microRelief(px0); rx1 += detail * microRelief(px1);
        rz0 += detail * microRelief(pz0); rz1 += detail * microRelief(pz1);
    }
    g += vec2(rx1 - rx0, rz1 - rz0) / (2.0 * e);
    return normalize(vec3(-g.x, 1.0, -g.y));
}

float terrainAO(vec2 p) {
    float w = nearBlend(p);
    float af = w > 0.0 ? texture(uAOFar, farUV(p)).r : 0.0;
    float an = w < 1.0 ? texture(uAONear, nearUV(p)).r : 0.0;
    return mix(an, af, w);
}

float waterLevel(vec2 p) {
    float w = nearBlend(p);
    float lf = w > 0.0 ? texture(uWaterLvl, farUV(p)).r : 0.0;
    float ln = w < 1.0 ? texture(uWaterLvlNear, nearUV(p)).r : 0.0;
    return mix(ln, lf, w);
}

vec4 forestAt(vec2 p) { return texture(uForest, farUV(p)); }

ivec2 treeCellOf(vec2 p) { return ivec2(floor((p + FAR_HALF) / TREE_CELL)); }
vec2 treeCellOrigin(ivec2 c) { return vec2(c) * TREE_CELL - FAR_HALF; }

// sun visibility from the baked height-field shadow maps
float sunShadowMap(vec3 p) {
    float w = nearBlend(p.xz);
    vec2 sn = vec2(0.0), sf = vec2(0.0);
    if (w < 1.0) sn = texture(uShNear, nearUV(p.xz)).rg;
    if (w > 0.0) sf = texture(uShFar, farUV(p.xz)).rg;
    vec2 s = mix(sn, sf, w);
    float width = 0.35 + s.y * 0.011;
    return smoothstep(-width, width, p.y + 0.15 - s.x);
}
// same, plus the horizontal distance to the occluder that defines the shadow (m)
vec2 sunShadowMapD(vec3 p) {
    float w = nearBlend(p.xz);
    vec2 sn = vec2(0.0), sf = vec2(0.0);
    if (w < 1.0) sn = texture(uShNear, nearUV(p.xz)).rg;
    if (w > 0.0) sf = texture(uShFar, farUV(p.xz)).rg;
    vec2 s = mix(sn, sf, w);
    float width = 0.35 + s.y * 0.011;
    return vec2(smoothstep(-width, width, p.y + 0.15 - s.x), s.y);
}

// ---------------------------------------------------------------------------
// sky.glsl : single-scattering atmosphere (Rayleigh + Mie + ozone), aerial
// perspective and volumetric cumulus clouds.
// ---------------------------------------------------------------------------
#define R_E 6360000.0
#define R_A 6420000.0
#define LAKE_ALT 1800.0
const vec3 BETA_R = vec3(5.802e-6, 13.558e-6, 33.1e-6);
const vec3 BETA_O = vec3(0.650e-6, 1.881e-6, 0.085e-6);
#define H_R 8000.0
#define H_M 1200.0


   // radiometric scale of the sun
           // Mie density multiplier (1 = clear alpine air)
     // 0..1
            // cloud drift m/s
          // sun radiance reaching the valley (from probe)
          // sky irradiance on an up-facing surface (from probe)
      // average horizon radiance (from probe)
       // approx. multiple scattering / camera sky handling


vec3 betaM() { return vec3(3.996e-6) * uHaze; }

vec2 raySphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd), c = dot(ro, ro) - r * r;
    float d = b * b - c;
    if (d < 0.0) return vec2(1e30, -1e30);
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}
vec3 toPlanet(vec3 p) { return p + vec3(0.0, R_E + LAKE_ALT, 0.0); }

vec3 densAt(vec3 pp) {
    float h = length(pp) - R_E;
    float oz = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
    return vec3(exp(-h / H_R), exp(-h / H_M), oz);
}

vec3 transToSun(vec3 pp) {
    float h = length(pp) - R_E;
    vec3 up = pp / (R_E + h);
    float mu = dot(up, uSunDir);
    float e = degrees(asin(clamp(mu, -1.0, 1.0)));
    if (e < -1.0) return vec3(0.0);
    float am = 1.0 / (max(mu, 0.0) + 0.50572 * pow(max(e, -0.5) + 6.07995, -1.6364));
    vec3 od = BETA_R * H_R * exp(-h / H_R) + betaM() * 1.11 * H_M * exp(-h / H_M) + BETA_O * 15000.0;
    return exp(-od * am);
}

float phaseR(float mu) { return 3.0 / (16.0 * PI) * (1.0 + mu * mu); }

float phaseM(float mu) {
    float g = 0.76, g2 = g * g;
    return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

// in-scattered radiance along [0,tmax] and transmittance
vec3 atmoScatterRaw(vec3 ro, vec3 rd, float tmax, int n, out vec3 trans) {
    vec3 pp = toPlanet(ro);
    vec2 s = raySphere(pp, rd, R_A);
    vec2 g = raySphere(pp, rd, R_E);
    float tEnd = s.y;
    if (g.x > 0.0) tEnd = min(tEnd, g.x);
    tEnd = min(tEnd, tmax);
    float mu = dot(rd, uSunDir);
    float pr = phaseR(mu), pm = phaseM(mu);
    vec3 bm = betaM();
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    vec3 od = vec3(0.0);
    // non-uniform steps: denser near the camera
    float prevT = 0.0;
    for (int i = 0; i < n; i++) {
        float f = (float(i) + 1.0) / float(n);
        float t1 = tEnd * f * f;
        float dt = t1 - prevT;
        float tm = 0.5 * (t1 + prevT);
        prevT = t1;
        vec3 q = pp + rd * tm;
        vec3 d = densAt(q) * dt;
        od += d;
        vec3 tv = exp(-(BETA_R * od.x + bm * 1.11 * od.y + BETA_O * od.z));
        vec3 ts = transToSun(q);
        sumR += tv * ts * d.x;
        sumM += tv * ts * d.y;
    }
    trans = exp(-(BETA_R * od.x + bm * 1.11 * od.y + BETA_O * od.z));
    return uSunIntensity * (sumR * BETA_R * pr + sumM * bm * pm);
}

// Aerial perspective (finite paths) must use the same radiance calibration as the visible sky
// (skyRadiance scales the single-scattering result by uSkyBoost); without it distant ridges only
// lose blue to extinction and turn beige instead of fading into a sky-blue haze.
// Infinite paths (sky dome, lighting probe) keep the raw value: their callers apply their own gain.
#define AERIAL_GAIN 0.6
vec3 atmoScatter(vec3 ro, vec3 rd, float tmax, int n, out vec3 trans) {
    vec3 s = atmoScatterRaw(ro, rd, tmax, n, trans);
    // finite paths: the distant ranges should be a greyer blue than the raw single scattering
    // (far range #7c93b4, OKLCh h 257 / C 0.056)
    // (shaded faces 2-6 km out are already sky-blue, lit / shade only ~1.7x: the in-scatter
    // over the middle distance is lifted, the 15+ km ranges keep their calibrated value)
    if (tmax < 1e8) {
        float mid = smoothstep(700.0, 3000.0, tmax) * (1.0 - smoothstep(5000.0, 15000.0, tmax));
        // (session 4: 0.55 -> 0.25: the lift filled the shadows of the 1-6 km faces - rock p10 0.14 in shot 7
        // against a target of 0.05; the "melted wax" look of the head wall)
        // (session 5: 0.25 -> 0.12: the in-scatter alone was ~0.07 of the darkest rock's 0.12 in shot 7)
        s *= uSkyBoost * AERIAL_GAIN * (1.0 + 0.12 * mid * smoothstep(0.25, 0.5, uSunDir.y));
        // (session 6: less desaturation over the middle distance: shaded faces 1-6 km out read sky-blue,
        // target #5c729d, instead of grey-teal; the 15+ km ranges keep the calibrated 0.25)
        s = mix(s, vec3(luma(s)), 0.25 * (1.0 - 0.7 * mid));
    }
    return s;
}

// ---------------------------------------------------------------------------
// clouds : fair-weather cumulus. flat bases, cauliflower domes whose height
// follows the cell strength, billowy worley erosion, multi-octave scattering.
// ---------------------------------------------------------------------------
#define CL_BOT 2350.0
#define CL_TOP 4300.0

float remap(float v, float a, float b, float c, float d) { return c + (v - a) / (b - a) * (d - c); }

// large-scale weather: where clouds can exist (0..1)
float cloudCoverage(vec2 xz) {
    vec2 q = xz + uWind * uTime;
    float a = texture(uNoise3, vec3(q * 0.000028, 0.37)).r;
    float b = texture(uNoise3, vec3(q * 0.00009 + 0.3, 0.71)).r;
    float w = a * 0.6 + b * 0.4;
    return saturate(uCloudCover + (w - 0.5) * 0.9);
}

float cloudDensity(vec3 p, bool detail) {
    float hf = (p.y - CL_BOT) / (CL_TOP - CL_BOT);
    if (hf < 0.0 || hf > 1.0) return 0.0;
    float wc = cloudCoverage(p.xz);
    if (wc <= 0.01) return 0.0;
    vec3 q = p; q.xz += uWind * uTime;
    // low-frequency perlin-worley body (cells ~1.5 km)
    vec4 lo = texture(uNoise3, q * vec3(0.00011, 0.00016, 0.00011));
    float base = lo.b * 0.65 + lo.r * 0.35;
    // cumulus height gradient: flat base, rounded top
    float hg = smoothstep(0.0, 0.06, hf) * smoothstep(1.0, 0.3, hf);
    float bc = saturate(remap(base * hg, 1.0 - wc, 1.0, 0.0, 1.0)) * wc;
    if (bc <= 0.0) return 0.0;
    if (detail) {
        q.y -= uTime * 0.8;
        // (r4 L10) a 4th octave (~40 m) and stronger erosion (m x 0.32 -> 0.36): crisp cauliflower rims instead of
        // soft cotton blobs (target: 1-2 px rims). (0.5 shrank the r5 cumulus to a sixth of their area, 0.4 to a third)
        float hfb = (texture(uNoise3, q * 0.0009).b * 0.62 + texture(uNoise3, q * 0.0029 + 0.4).b * 0.26
                  + texture(uNoise3, q * 0.0085 + 0.7).r * 0.12 + texture(uNoise3, q * 0.024 + 0.2).r * 0.08) * (1.0 / 1.08);
        float m = mix(hfb, 1.0 - hfb, saturate(hf * 6.0));   // wispy base, billowy top
        bc = saturate(remap(bc, m * 0.36, 1.0, 0.0, 1.0));
    }
    return bc;
}

// returns premultiplied cloud radiance (rgb) and transmittance (a)

// ---------------------------------------------------------------------------
// water.glsl : lake surface waves (deep/capillary dispersion), gust patches,
// optical constants of a clear glacial lake.
// ---------------------------------------------------------------------------
      // wind direction over the lake (unit)
  // overall roughness multiplier

// absorption + scattering (1/m) and an effective single-scatter radiance factor
// (r4 W1: blue absorbed faster than green - the bed seen through 1-3 m turns green (drop-off #5f8061), not cyan-grey)
// (water5: clearer: (0.30, 0.055, 0.085) -> (0.225, 0.030, 0.038) - the camera's broad R / G bands see the low-absorption
// end of pure water's spectrum (a(570) .07, a(600) .24); with half the column scattering (water_shade wSca) the shallows
// are clear (the bed golden-olive, not green-grey), the deep in-scatter ceiling keeps its level, deep water a little bluer)
const vec3 W_ABS = vec3(0.225, 0.030, 0.038);
const vec3 W_SCA = vec3(0.095, 0.115, 0.118);
       // in-scatter tint scale (tunable look)
// look2 tunables (render.py Look.tune -> uTune[16]); see render.py for the meaning of each slot


// wave slopes filtered by the ANISOTROPIC pixel footprint: across the view it is t * pixAng, along the
// view t * pixAng / |rd.y| (up to ~300x longer at grazing angles). The old isotropic footprint (with
// |rd.y| clamped to 0.02) kept 0.3-3 m ripples unresolved-but-unfiltered on the far water: their
// back-facing facets were then flipped toward the viewer and the reflection jumped to ~20 deg up
// (the pale band along the far shore, film shots 0 / 1).
// fwd = horizontal view direction (unit), fa / fl = footprint across / along it (m).
// sv = slope variance of the filtered-out waves projected on fwd (the unresolved roughness that
// smears the reflection vertically), gust = wind patch strength.

// animated caustic pattern on the lake bed
// (r4 W6 / R8 / W3: a fine, soft net (~0.2-0.35 m cells) instead of ~1 m pool-floor cells; none at zero
// depth (the white kerb along the cliff foot), fading with depth; fp = pixel footprint (m): unresolved -> the mean)

// ---------------------------------------------------------------------------
// trees.glsl : procedural subalpine conifers (Norway spruce / subalpine fir and European larch)
// One tree per 4 m cell, occupancy + size from the baked per-cell texture uTrees; stand structure
// (gaps, groups, height cohorts, dominant trees) and crown shape are varied procedurally here.
//
// Every tree is built from whorls of branches. All levels of detail share the same whorl heights,
// branch directions and branch lengths, so they agree in silhouette. The model is picked per tree
// from its camera distance (not the pixel footprint, so the look does not depend on the render size)
// and dissolved per sample over a distance band, so a ray never mixes models:
//   near : individual drooping branches (10-16 per whorl, each with its own droop / rise, J-shaped up-turned
//          tips on ~45 % of the spruce branches, a sideways bend and small kinks); each carries a frond of side
//          branchlets in flat, layered fans that droop a little below the axis (spruce, a few comb curtains) or
//          spread flat (fir); thin larch twigs spread and hang (~58 deg) as chains of spiky needle rosettes
//          (sparse whorls, many missing twigs). Branchlet radius grows from the needle-brush size to a
//          gap-filling slab with the footprint, so the frond keeps its serrated outline and holes at every
//          distance. A dense inner crown (dark core) fills the interior from ~10 m on.
//   mid  : one drooping skirt per whorl, star-shaped from the per-branch lengths, jagged with noise
//   far  : spire envelope
// ---------------------------------------------------------------------------
struct Tree {
    vec3 base; float H; float R; float yb; float seed; float larch; float dens;
};

#define TREE_TEX_OFF 2048
#define TREE_TEX_RES 4096
#define TREE_LOD2_RANGE 1500.0
// near -> mid switch: camera distance (m, per-tree jitter x0.8..1.2), dissolved per sample over
// 0.8..1.25 x that distance (r3: 100 m, was 70: spiky silhouettes and self-shadowed spires at 70-120 m)
#define TREE_NEAR_D 100.0
// "look" footprint: radians per pixel of the 1920x1080 film camera (~44 deg vertical FOV). Look terms
// use dc * TREE_REF_PIXANG so the forest looks the same at every render size; only anti-aliasing terms
// (shoot widening, noise fades, geometric LOD) use the true footprint dc * uPixAng
#define TREE_REF_PIXANG 0.00072

// material info of the last sdTreeInfo() call
//   near: x part (0 foliage, 1 trunk, 2 wood/dead twig, 3 dense inner crown), y position along the
//         branch (0 inner .. 1 tip), z position along the branchlet (0 axis .. 1 tip), w side (-1 under .. +1 top)
//   mid : x part, y radial r/renv, z position in the skirt (-1 lower edge .. +1 top), w = 2 (marks mid)
vec4 gTreeInfo;
// tree cell to leave out of treesSDFShadow (the shaded tree's own crown is handled analytically)
ivec2 gTreeSkip = ivec2(-1000000);

// index of the current sample pass (defined in shade_trees.glsl, after the frame uniforms)
int treeSampleIdx();
// debug mode (20 = all trees near model, 21 = all mid model), defined in shade_trees.glsl
int treeDebugMode();
// 0 that the compiler cannot prove: loops starting here are not unrolled, so the large tree distance
// functions are not copied many times into every caller (the AMD driver needed ~13 GB / 17 min to build
// the unrolled program)
int treeZero() { return min(treeSampleIdx(), 0); }

// raw baked cell: position, base height, baked height / radius (before the stand structure below)
bool treeRaw(ivec2 cell, out vec4 d, out vec4 hs, out vec2 pos) {
    d = vec4(0.0); hs = vec4(0.0); pos = vec2(0.0);
    ivec2 tc = cell - ivec2(TREE_TEX_OFF);
    if (tc.x < 0 || tc.y < 0 || tc.x >= TREE_TEX_RES || tc.y >= TREE_TEX_RES) return false;
    d = texelFetch(uTrees, tc, 0);
    if (d.y == 0.0) return false;
    hs = hash42i(cell);
    pos = treeCellOrigin(cell) + TREE_CELL * (0.5 + (hs.yz - 0.5) * 0.6);
    return true;
}
// upper bounds of a raw tree's height and crown radius after treeFinish (only the per-cell hashes and
// the baked values: the stand-structure noise factors are taken at their maximum)
vec2 treeRawMax(ivec2 cell, vec4 d, vec4 hs) {
    float H0 = abs(d.y);
    float dens = saturate(d.w / (0.18 * H0 * max(hash12i(cell + ivec2(911, 37)), 0.02)));
    float hv = fract(hs.x * 13.7 + hs.w * 5.1);
    float yg = fract(hs.w * 31.7 + hs.y * 2.3);
    float ygF = mix(0.40, 0.16, smoothstep(0.20, 0.40, dens));
    float dom = step(0.89, fract(hs.y * 23.7 + hs.z * 5.3));
    float hsc = mix(0.74, 1.30, hv) * 1.12 * 1.12 * 1.10 * mix(0.55, 1.0, smoothstep(0.12, 0.50, dens));
    hsc = min(hsc * (yg < ygF ? mix(0.30, 0.62, yg / ygF) : mix(1.0, 1.30, dom)), 1.6);
    // (spruce converted to larch in treeFinish gets the larch crown radius)
    bool lr = d.y < 0.0;
    if (!lr) {
        float lp = 0.26 * (1.0 - smoothstep(60.0, 350.0, d.x)) * mix(1.4, 0.7, dens) + 0.05;
        lr = fract(hs.z * 17.3 + hs.x * 4.1) < lp;
    }
    float R = min(d.z * pow(hsc, 0.8) * (lr ? 1.25 : 0.92) * mix(0.76, 1.15, fract(hs.y * 9.3)), 3.6);
    // (r5 T14: open-grown skirt, crownR x1.15 at the base)
    R *= 1.0 + 0.15 * (1.0 - smoothstep(0.30, 0.45, dens));
    return vec2(min(H0 * hsc, 34.0), R);
}
// horizontal reach of a tree's branches / sprays (L <= 1.25 Lk + 0.4, Lk <= 1.32 R + 0.1, sprays <= 1.08 L),
// lean and a forked top

// conservative distance bound of a raw tree (snag of a broken top <= 2.2 m above the crown)

bool treeFinish(ivec2 cell, vec4 d, vec4 hs, vec2 pos, out Tree tr) {
    tr = Tree(vec3(0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    float H0 = abs(d.y);
    // stand density: the bake stores yb = H * 0.18 * density * hn with its own per-cell hash hn
    float hn = hash12i(cell + ivec2(911, 37));
    float dens = saturate(d.w / (0.18 * H0 * max(hn, 0.02)));
    float dk = smoothstep(0.45, 0.85, dens);           // closed stand: only small openings
    // stand structure (rotated so the 1.5 km noise tile does not line up with the grid):
    // r (fbm) -> small irregular gaps with ragged edges, g (smooth) -> patches of similar height
    vec4 st = textureLod(uNoise3, vec3(mat2(0.8, -0.6, 0.6, 0.8) * pos * (1.0 / 1500.0), 0.43), 0.0);
    float hx = fract(hs.x * 7.73 + hs.w * 3.17);
    if (st.r < 0.25 + 0.09 * hx - 0.07 * dk) return false;
    // stand structure at the 20-100 m scale: small openings with ragged edges, patches of old tall
    // trees and of young regrowth; a (worley, 16 m cells) -> tree groups
    vec4 s2 = textureLod(uNoise3, vec3(mat2(0.6, 0.8, -0.8, 0.6) * pos * (1.0 / 260.0), 0.17), 0.0);
    // (more openings on the slopes above the shore: open stands with grass and rock
    // between tree groups; the shore stands (hero trees of film shot 3) stay as they are)
    // (r5 T1: a fully closed stand (dens 1, e.g. a baked stand for a fixed camera framing) keeps all
    // its trees: no s2 openings)
    bool refS = dens > 0.995;
    if (!refS && s2.r < 0.30 + 0.06 * smoothstep(60.0, 160.0, d.x) + 0.08 * (hx - 0.5) - 0.08 * dk) return false;
    // clustered openings of 1-5 trees (worley centres, 8-32 m cells): no unbroken wall of spires
    // (r5 T5: more of them on the slopes > 60 m above the lake, .66 was .73: groups with rock / grass between)
    if (!refS && s2.b > mix(0.73, 0.66, smoothstep(60.0, 100.0, d.x)) + 0.05 * dk + 0.03 * (hx - 0.5)) return false;
    // open stands and treelines: trees stand in groups with open ground between them
    float grp = 0.55 * s2.a + 0.45 * st.a;          // groups at the 16 m and ~90 m scales
    if (dens < 0.6 && grp < mix(0.52, 0.0, smoothstep(0.12, 0.60, dens)) + 0.08 * (hx - 0.5)) return false;
    tr.base = vec3(pos.x, d.x, pos.y);
    tr.larch = d.y < 0.0 ? 1.0 : 0.0;
    tr.seed = fract(hs.w * 7.31 + hs.x * 3.7);
    tr.dens = dens;
    // height: cohorts + individual variety (a jagged canopy line), groups taller in the middle,
    // shorter trees toward the treeline (no hard edge)
    float hv = fract(hs.x * 13.7 + hs.w * 5.1);
    // (r5 T5: individual spread 0.55-1.30, was 0.66-1.18: mixed crown sizes, not a tree farm)
    float hsc = mix(0.55, 1.30, hv) * mix(0.70, 1.12, smoothstep(0.28, 0.72, st.g)) * mix(0.72, 1.12, smoothstep(0.25, 0.75, s2.g));
    hsc *= mix(0.82, 1.10, grp) * mix(0.55, 1.0, smoothstep(0.12, 0.50, dens));
    // a cohort of young trees (regeneration in gaps, along edges, many at the treeline), a few
    // dominant old trees
    float yg = fract(hs.w * 31.7 + hs.y * 2.3);
    float ygF = mix(0.40, 0.16, smoothstep(0.20, 0.40, dens));
    float dom = step(0.89, fract(hs.y * 23.7 + hs.z * 5.3));
    hsc *= yg < ygF ? mix(0.30, 0.62, yg / ygF) : mix(1.0, 1.30, dom);
    // a middle cohort (15 %, x0.65-0.85): an uneven, layered canopy line instead of equal spires
    float mc = fract(hs.x * 41.3 + hs.z * 7.9);
    float mcF = 0.25;
    if (yg >= ygF && dom < 0.5 && mc < mcF) hsc *= mix(0.65, 0.85, mc / mcF);
    hsc = min(hsc, 1.6);
    tr.H = min(H0 * hsc, 34.0);
    // (r5 T14: open-grown trees keep their branches down to the ground, skirts touch the grass)
    tr.yb = tr.H * mix(0.02, 0.25, pow(dens, 1.3)) * mix(1.0, 0.7, tr.larch) * smoothstep(0.30, 0.45, dens);
    tr.R = d.z * pow(hsc, 0.8) * mix(0.92, 1.25, tr.larch) * mix(0.76, 1.15, fract(hs.y * 9.3));
    // more larch than the bake (20-35 % light larches near the lake, at stand edges
    // and in open stands); H and yb stay those of the baked species, the crown gets the larch radius
    if (tr.larch < 0.5 && tr.H > 4.0) {
        float lp = 0.26 * (1.0 - smoothstep(60.0, 350.0, d.x)) * mix(1.4, 0.7, dens) + 0.05;
        if (fract(hs.z * 17.3 + hs.x * 4.1) < lp) {
            tr.larch = 1.0;
            tr.R *= 1.25 / 0.92;
        }
    }
    // (r4: spruce / fir crowns x0.75 from 4 to 20 m height: narrow spires, H:W ~4-5;
    // a reduction only, so treeRawMax stays an upper bound)
    tr.R *= mix(1.0, 0.75, smoothstep(4.0, 20.0, tr.H) * (1.0 - tr.larch));
    tr.R = min(tr.R, 3.6);
    return true;
}

// (trees6) height of a tree relative to its stand's canopy: only the per-tree factors of treeFinish
// (individual spread, young / middle cohort, dominant trees); ~1 for the main canopy, 0.3-0.6 for the young
// trees under it, up to ~1.6 for the old dominant ones (used by the shading: who gets the sun in a dense stand)

// (trees6) mid / far model only: per-tree crown slenderness (x0.78-1.0: narrow pencil spruce beside full
// cones, so the distant spires are not all one shape; dark gaps between the narrow ones). Ramped in by camera
// distance well beyond the near / mid switch (<= 150 m), a reduction only (all bounds stay valid)

bool getTree(ivec2 cell, out Tree tr) {
    vec4 d, hs; vec2 pos;
    if (!treeRaw(cell, d, hs, pos)) { tr = Tree(vec3(0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); return false; }
    return treeFinish(cell, d, hs, pos, tr);
}

// per-tree / per-whorl / per-branch random numbers (one integer hash each)

// direction (angle) of the open / light side of a crown: branches are longer there, and in 40 % of
// the trees the side facing the stand has lost many of its branches

// length factor of one branch: own variety (a narrow range: few long arms sticking out - those read as
// araucaria up close) and a crown asymmetry. Larch: irregular, very unequal branch lengths. Shared by the
// near and mid model (they agree in outline).

// openness: 0 = dense crown (shaded-stand spruce), 1 = open, airy crown (sunny subalpine tree)
float treeOpen(Tree tr) { return fract(tr.seed * 29.31) * (1.0 - 0.5 * tr.dens); }

// missing / broken branch (a gap in the crown); shared by the near and mid model
// sparse whorl: half its branches gone (a horizontal gap in an old, open crown)

// sparse, see-through larch: only the most open-grown ones; most larches are full soft cones
float larchSparse(Tree tr) { return tr.larch * smoothstep(0.62, 0.82, treeOpen(tr)); }

// dead branches in the lower live crown (open-grown old spruce / fir: grey, lichen-covered twigs on the
// lower trunk): probability for a whorl at relative crown height thk; shared by the near and mid model

// fir-ness: 0 = Norway spruce (broader, drooping), 1 = subalpine fir (narrow spire, flat sprays)
float treeFir(Tree tr) { return (1.0 - tr.larch) * step(0.62, fract(tr.seed * 13.17)); }

// top type: 0 normal, 1 broken top (4 %: the crown ends flat below a dead snag), 2 forked top (3 %)
int treeTop(Tree tr) {
    float h = fract(tr.seed * 83.71);
    return tr.H < 8.0 ? 0 : (h < 0.04 ? 1 : (h < 0.07 ? 2 : 0));
}
// height of the crown cone apex (above the actual top for a broken tree)
// (r5 T8: x1.06-1.15, was 1.18-1.35: the truncated cone read as a flat-topped cypress column on the fall lip)
float crownTop(Tree tr) { return treeTop(tr) == 1 ? tr.H * mix(1.06, 1.15, fract(tr.seed * 7.93)) : tr.H; }

// (r5 T10: treeFinish narrows spruce / fir crowns x0.75 from 4 to 20 m height; the lower crown (t < 0.5) keeps
// x0.9 instead: broad drooping lower branches, spire only in the top ~40 %, H:W ~3.5). Factor on tr.R
// below t 0.5 (<= 1.2); bounds that use tr.R multiply by it
float treeLowWiden(Tree tr) {
    float nK = smoothstep(4.0, 20.0, tr.H) * (1.0 - tr.larch);
    return mix(1.0, 0.9, nK) / mix(1.0, 0.75, nK) * (1.0 + 0.15 * (1.0 - smoothstep(0.30, 0.45, tr.dens)));
}
float treeForkY(Tree tr);
// crown envelope radius at local height y; per-tree profile from convex (columnar, old) to
// concave (drawn-out spire)
float crownR(Tree tr, float y, float fir) {
    float t = saturate((y - tr.yb) / max(crownTop(tr) - tr.yb, 1.0));
    float R = tr.R * mix(1.0, 0.80, fir) * mix(treeLowWiden(tr), 1.0, smoothstep(0.5, 0.6, t));
    // (r5 T8: forked top - two thin leaders above the fork, not two full-width crown halves)
    if (treeTop(tr) == 2) R *= mix(1.0, 0.55, smoothstep(-0.3, 0.5, y - treeForkY(tr)));
    float c = 1.0 - t;
    // (larch: broad, irregular crowns with long branches high up)
    // (r4: larch .85-1.0, was .55-.85: the convex domes read as round topiary / broadleaf crowns at 50-150 m)
    // (r4: spruce .90-1.20, was .70-1.0: with the narrower crowns the convex profiles read as columnar cypress
    // pencils at 50-300 m; target spires taper to a point)
    // (r5 T4: spruce 1.05-1.45: concave, drawn-out spires; no convex columns)
    float ex = mix(mix(1.05, 1.45, fract(tr.seed * 57.13)), mix(0.85, 1.00, fract(tr.seed * 57.13)), tr.larch);
    // (spruce: near-linear cone with a drawn-out spire tip (top ~35 %))
    float prof = pow(c, ex) * mix(mix(0.50, 1.0, smoothstep(0.0, 0.35, c)), 1.0, tr.larch);
    prof = mix(prof, c * c * (3.0 - 2.0 * c), 0.20 * fir);
    // in closed stands the shaded lowest branches stay short (the crown narrows toward its base)
    // (r5 T14: open-grown trees (dens < .3) x1.15 instead: broad skirts on the grass)
    float skirt = 1.0 + (0.15 * (1.0 - smoothstep(0.30, 0.45, tr.dens)) - 0.15 * tr.dens) * (1.0 - smoothstep(0.0, 0.30, t));
    return R * prof * skirt * (0.90 + 0.10 * smoothstep(0.0, 0.15, t)) * step(y, tr.H + 0.3);
}

// branches of 2-4 neighbouring whorls share a length factor: clumped, ragged crown outline at 50-400 m
// (the per-whorl variety averages out at range); shared by the near and mid model

float whorlSpacing(Tree tr) {
    // (r5 T2: larch whorls x0.8 closer - branches every 0.3-0.5 m)
    float lf = mix(1.10, 1.40, larchSparse(tr)) * 1.15 * 0.8;
    return clamp(tr.H * 0.022, 0.34, 0.62) * mix(0.88, 1.12, fract(tr.seed * 5.31)) * mix(0.70, lf, tr.larch) * mix(1.12, 0.70, treeFir(tr));
}
// whorl height jitter (fraction of the spacing) and missing whorls (larch: irregular, open crowns)
// (r4: spruce x1.12 spacing, jitter .45: fewer, more distinct tiers on spruce at 50-150 m)

// branch centre line: height offset at x = u/L (0 trunk .. 1 tip). droop D, upward angle e.
// mono 0: the quadratic sags to its lowest point at x = 0.6, then the tip curves up again (to ~55 % depth);
// mono 1: the outer branch droops more and more (no upturned 'monkey-tail' tips)

// droop shape: spruce / fir droop, larch keeps its horizontal branches with up-curved tips

// (larch: nearly horizontal branches with hanging twig curtains)

// camera distance of a tree (nearest point of its axis)

// near model or not: by distance (per-tree jitter), dissolved over a band: inside it each tree
// picks the model per sample pass (fixed for the whole pass, so a ray never mixes models)

// ---------------------------------------------------------------------------
// frond of one branch: herringbone of side branchlets, in branch-local coordinates
// u along the branch (m, 0 at the trunk), av = |lateral| (m), dy height above the sagging axis (m)
// Wm: max reach of the branchlets, hang: mean slope (tan) at which they hang below horizontal,
// x0: bare inner fraction. Every branchlet has its own hang, forward angle, length and droop, so the
// frond is a ragged 3D mass (some shoots spread flat, some hang) rather than a comb.
// rT: radius of the bare twig on the inner part of a branchlet (fraction bare), sh3: weight of the
// explicit side shoots (3rd order, length L3; once they are sub-pixel the shoot is widened to their
// reach instead), tuft: larch short-shoot tufts weight, sj: per-sample jitter of the needle-brush radii
// (soft, fuzzy shoot edges once the samples are averaged)
// fi: (position along the frond 0..1, along the branchlet 0..1, side -1 under .. +1 top)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// near model: individual branches with fronds
// fat: true pixel footprint (m) - sub-pixel shoots are widened so the sprays keep their coverage and
// the cost is cut once single shoots are sub-pixel; dc: camera distance; fine: needle detail weight
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// mid model: one drooping skirt per whorl, star-shaped from the per-branch lengths
// lod: 1 = whorl skirts .. 0 = envelope; fp: true pixel footprint (fades the sub-pixel noise)
// ---------------------------------------------------------------------------

// tree-local coordinates; crowns lean a little (wind, light seeking): shift grows with height.
// Forked top: above the fork the crown is mirrored about a plane through the axis and the two
// halves pushed apart, so two leaders diverge in a V
// (r5 T8: fork high in the crown, H x0.84-0.92, was 0.68-0.82: a short V of two leaders, not a two-lobed blob)
float treeForkY(Tree tr) { return tr.H * mix(0.84, 0.92, fract(tr.seed * 91.1)); }

// lod: 0 = plain cone, 1 = whorl skirts; lod2: 1 - distance / TREE_LOD2_RANGE (kept for callers;
// the model choice is per tree from its camera distance)

// same as sdTree, also fills gTreeInfo and env (used once per shaded hit)

// (perf4) tree-free space (render.py tree_tops): per 16 m block, R = the highest possible point of any tree in the block
// and its 8 neighbours (the 3 x 3 cells treesSDF tests around p always lie inside), G = horizontal clearance (m) to the
// crowns of all trees further out. Returns a distance below which no tree surface lies (<= 0: unknown, evaluate)
layout(set=0, binding=12) uniform sampler2D uTreeTop;
float treeFreeDist(vec3 p) {
    ivec2 b = (treeCellOf(p.xz) - ivec2(TREE_TEX_OFF)) >> 2;
    if (any(lessThan(b, ivec2(0))) || any(greaterThanEqual(b, ivec2(TREE_TEX_RES / 4))))
        return max(abs(p.x), abs(p.z)) - float(TREE_TEX_RES / 2) * TREE_CELL - 12.0;   // outside the tree map (r4 merge: reach <= ~11.6 m)
    vec2 tg = texelFetch(uTreeTop, b, 0).rg;
    return min(p.y - tg.r, tg.g);
}

// cheap version for shadow rays: shrunken tiered spire (outer sun-facing sprays stay lit);
// distances are scaled up so that the soft-shadow penumbra stays narrow
float treesSDFShadow(vec3 p) {
    ivec2 c0 = treeCellOf(p.xz);
    float dmin = TREE_CELL * 0.55;
    int z0 = treeZero();
    for (int j = z0 - 1; j <= 1; j++)
    for (int i = z0 - 1; i <= 1; i++) {
        ivec2 c = c0 + ivec2(i, j);
        if (c == gTreeSkip) continue;
        vec4 dr, hs; vec2 pos;
        if (!treeRaw(c, dr, hs, pos)) continue;
        vec2 mx = treeRawMax(c, dr, hs);
        if (max(length(p.xz - pos) - mx.y, p.y - dr.x - mx.x) * 0.96 > dmin) continue;
        Tree tr;
        if (!treeFinish(c, dr, hs, pos, tr)) continue;
        vec3 q = p - tr.base;
        float r = length(q.xz);
        float fir = treeFir(tr);
        float renv = crownR(tr, q.y, fir) * 0.80;
        float s = whorlSpacing(tr);
        float f = fract((q.y - tr.yb) / s - fract(tr.seed * 3.77));
        float rr = renv * (1.0 - 0.30 * smoothstep(0.2, 1.0, f));
        float d = (r - rr) * 0.6;
        d = max(d, max(q.y - tr.H, (tr.yb - q.y) * 0.8)) * 1.6;
        float dt = max(r - (0.03 + 0.0095 * tr.H), max(-q.y, q.y - tr.H));
        dmin = min(dmin, min(d, dt));
    }
    return dmin;
}

// length of the sun ray inside this tree's crown from local point q (self-shadowing); the cone
// encloses most branch tips (they reach ~1.1 x the envelope), so the outermost sprays on the sun side
// get a path ~0 and stay fully lit. Sprays sticking out on the far side still see the sun through the
// crown (ray enters and leaves the cone)

// ---------------------------------------------------------------------------
// rocks.glsl : granite boulders scattered on shores, scree and forest floor
// Main layer: 5 m cells with up to 6 rocks ("slots") each; small layer: 2.5 m cells with up to 3 (+2 beach cobbles)
// stones (R <= 0.35 m), within ~31 m of the camera. Placement is driven by the height above the water
// at the rock itself (cell-centre height + slope), so the waterline carries a band of blocks of every
// size, partly submerged; the band is dense along some stretches of shore and nearly clear along
// others; boulder fields (noise patches, the north end / inflow valley, scree near water) extend it
// inland; slab patches turn the biggest blocks into pale glacier-polished slabs that dip into the
// lake. Shape = irregular block: seven fracture planes (tilted top, four near-vertical sides, two
// corner chips) around a box, smooth max -> rounded edges, corners worn by an ellipsoid; slight
// taper; optional through-crack, fracture steps and low relief near the camera. Rocks are tilted to
// the terrain slope and partly buried.
// Placement is camera independent; only sizes shrink with the camera: sub-pixel rocks, small stones
// beyond 22-31 m, and blocks that would contain a low camera (continuous in the camera position).
// ---------------------------------------------------------------------------
#define ROCK_CELL 5.0
#define ROCK_OVER 1.45      // max distance a rock can reach beyond its own cell (m)
#define ROCK_RMAX 2.05      // max nominal size R (m); half extents are <= 0.94 R
#define ROCK_BOUND 1.46     // bounding-sphere radius / R
#define ROCK2_CELL 2.5
#define ROCK2_OVER 0.25
#define ROCK_OVER_S 0.4     // main slots >= 2 (R <= 1.2) stay within 0.4 m of their cell: skipped for points farther away
#define ROCK_RMAX_S 1.2
#define ROCK2_RMAX 0.35
#define ROCK2_D0 22.0       // small stones shrink away between these camera distances (geometry and shadows alike)
#define ROCK2_D1 31.0
#define ROCK2_TAG 4194304   // small-layer cells are tagged x + ROCK2_TAG
#define ROCK_SLOT_TAG 1048576   // slot k is tagged y + k * ROCK_SLOT_TAG
#define RK_NS 6
#define RK_NS2 20         // slots 3-19: beach cobbles (R 5-22 cm) within ~15 m of the camera only (i10: 7 slots, 3c: 11, rocks5: 17)
#define ROCK3_D0 10.5      // (rocks5: 9.5 / 14 -> 10.5 / 15)
#define ROCK3_D1 15.0
#define RK_OC0 6          // slots 6-7: jointed-outcrop blocks (own slots: they exist within ~180 m of the camera only)
#define RK_OCD 180.0

struct Rock {
    vec3 c;      // centre (world)
    vec3 r;      // half extents (local frame), r.x = largest
    float ang;   // yaw
    float seed;
    mat3 m;      // local frame columns (x, up, z); local q = (p - c) * m
    vec4 f0;     // fracture facet planes (local): normal, offset
    vec4 f1;
    vec4 f2;
    vec4 f3;
    vec4 f4;
    vec4 f5;
    vec4 f6;
    vec4 f7;     // (rocks3) three extra cuts across corners / top edges: more, smaller facets (subangular block,
    vec4 f8;     // not a prism / tombstone)
    vec4 f9;
    vec4 ck;     // crack plane (local): normal, offset
    float ckw;   // crack half width at the top (m), 0 = no crack
    vec2 nt;     // fracture step along the top / side-1 edge: depth below the top, depth into the side (0 = none)
    vec3 nt2;    // broken-off corner (top / side 2 / side 3): depths below the three faces (0 = none)
    vec2 nt3;    // second fracture step along the top / side-3 edge (0 = none)
    vec4 ch0;    // conchoidal chips knocked out of two edges: sphere centre (local), radius (0 = none)
    vec4 ch1;
    float ell;   // corner-trimming ellipsoid scale (1.55 = blocky, ~1.4 = worn round boulder)
    float rr;    // edge rounding radius (m)
    float taper; // narrowing toward the top
    float env;   // 0 open shore / scree, 1 forest floor (moss)
    float fld;   // 0 washed shore, 1 inland boulder field / meadow (older surfaces: more lichen, darker)
    float kind;  // 0 block, 1 slab
    float rel;   // height of the rock base above the water (m)
};

// random direction with elevation in [e0,e1] (radians) and azimuth u*2pi
vec3 rkDir(float u, float v, float e0, float e1) {
    float a = u * TAU, e = mix(e0, e1, v);
    return vec3(cos(a) * cos(e), sin(e), sin(a) * cos(e));
}
// facet plane at a fraction of the support distance of ellipsoid b along n
vec4 rkFacet(vec3 n, vec3 b, float frac) { return vec4(n, length(b * n) * frac); }
// chip on the edge of facets a and b: a point of the edge line at fraction u along it (ext = half length of the
// edge region to spread over) and the chip size r
vec4 rkChip(vec4 a, vec4 b, float u, float r, float ext) {
    float c = dot(a.xyz, b.xyz);
    float den = max(1.0 - c * c, 0.05);
    vec3 e = ((a.w - c * b.w) * a.xyz + (b.w - c * a.w) * b.xyz) / den;    // edge point nearest the centre
    vec3 dir = normalize(cross(a.xyz, b.xyz));
    return vec4(e + dir * (u - 0.5) * 2.0 * ext, r);
}
// flat conchoidal chip (round 2 review: the spherical scoops read as round craters): the piece of the ball
// (edge point, radius r) outside a cut plane RK_CHIP_D r inside the edge, plane normal = the direction from the
// block centre to the edge point (~ the edge bisector). Returns the SDF of the removed piece.
#define RK_CHIP_D 0.3
float rkChipD(vec3 q, vec4 ch) {
    vec3 dq = q - ch.xyz;
    return max(-(dot(dq, normalize(ch.xyz)) + RK_CHIP_D * ch.w), length(dq) - ch.w);
}

// context of a 5 m cell (shared by its slots and by the small-layer cells inside it)
struct RkCtx {
    vec4 f;      // forest map: tree density, larch, scree, meadow
    float rel;   // ground height above the water at the cell centre (99 = no water near)
    vec2 g;      // ground slope (dh/dx, dh/dz)
    vec2 cc;     // cell centre
    float bf;    // boulder-field weight
    float sl;    // slab-outcrop patch weight
    float em;    // boulder-strewn shallows patch weight (blocks in the water break the surface)
    float bm;    // shoreline density modulation (0.15 nearly clear .. 1.25 dense)
    float fg;    // boulder-field gaps (0 = grassy gap between the block clusters, 1 = blocks)
    float h0;    // ground height at the cell centre
    float cob;   // beach-cobble density (patchy)
    float oc;    // jointed-outcrop weight: steep bare ground -> half-buried joint blocks (ledges, kind 2)
    vec2 go;     // ground slope for the outcrops and the pre-bound (= g near the water; also fetched within ~180 m
                 // of the camera, where g stays 0 away from the water as before: no camera-dependent tilts)
    int ns;      // slots that can hold a rock (main / small layer): 1 away from shores and scree
    int ns2;
    float strm;  // stream corridor code (falls lab rockfix v3), 0 = none
};
layout(set=0, binding=13) uniform sampler2D uStreamCorr; 
float fallsWetness(vec3 p);    // falls.glsl (the waterfall's cliff and banks keep their own rock / vegetation)
float fallsBank(vec3 p);
// conservative 3D bound of an unbuilt rock: horizontal distance, and height above its highest possible top
// (ground from the cell plane + margin for curvature and the marcher's micro relief, top <= ground + 2.3 R)
float rkPreBound(vec3 p, vec2 pos, float R, RkCtx x, vec2 g) {
    float bxz = length(p.xz - pos) - ROCK_BOUND * R;
    float top = x.h0 + dot(g, pos - x.cc) + 1.0 + 0.12 * length(pos - x.cc) + (2.3 + length(g)) * R;
    return max(bxz, p.y - top);
}
bool rkCtx(ivec2 cell, out RkCtx x) {
    x.cc = (vec2(cell) + 0.5) * ROCK_CELL;
    x.f = vec4(0.0); x.rel = 99.0; x.g = vec2(0.0); x.bf = 0.0; x.sl = 0.0; x.em = 0.0; x.bm = 1.0; x.fg = 1.0; x.h0 = 0.0; x.ns = 0; x.ns2 = 0; x.cob = 0.0; x.oc = 0.0; x.go = vec2(0.0); x.strm = 0.0;
    // (rocks4 R12) 0.9 -> 0.985: a test camera stands at x -2018 (no rock cell within 175 m of it: its foreground slope had no blocks)
    if (max(abs(x.cc.x), abs(x.cc.y)) > NEAR_HALF * 0.985) return false;
    x.f = forestAt(x.cc);
    float h0 = terrainH(x.cc, false);
    x.h0 = h0;
    float wl = waterLevel(x.cc);
    vec2 cuS = (x.cc - uStreamCorrBox.xy) * uStreamCorrBox.zw;
    if (uStreamCorrBox.z > 0.0 && cuS.x > 0.0 && cuS.y > 0.0 && cuS.x < 1.0 && cuS.y < 1.0) {
        vec4 sc = texture(uStreamCorr, cuS);
        if (sc.x < 10.0) x.strm = 1.0 + floor((sc.x + 12.0) * 10.0 + 0.5) * 64.0 + mod(floor(atan(sc.z, sc.y) * (64.0 / 6.2832) + 64.5), 64.0);
    }
    // the level map only reaches ~12 m inland: beyond it, height above the main lake (y = 0)
    x.rel = wl > -1000.0 ? h0 - wl : (h0 > -0.5 && h0 < 30.0 ? h0 : 99.0);
    // (outcrop blocks have their own slots and have sunk away at 90-170 m: their slope is only needed within
    // RK_OCD of the camera; the cell plane g of the other rocks stays 0 away from the water, as before)
    bool ocR = length(x.cc - uCamPos.xz) < RK_OCD;
    if (x.rel < 30.0 || ocR) x.go = vec2(terrainH(x.cc + vec2(2.0, 0.0), false) - h0, terrainH(x.cc + vec2(0.0, 2.0), false) - h0) * 0.5;
    if (x.rel < 30.0) x.g = x.go;
    if (x.rel < 30.0) {
        vec4 pn = texture(uNoise3, vec3(x.cc * 0.0011, 0.29));
        // boulder fields: noise patches + the whole north end of the lake and the inflow valley
        // + scree reaching the water
        float geo = smoothstep(1100.0, 1220.0, x.cc.y) * mix(0.8, 1.0, pn.b);
        x.bf = max(max(smoothstep(0.42, 0.64, pn.g), geo), 0.7 * smoothstep(0.3, 0.6, x.f.b));
        x.sl = smoothstep(0.52, 0.66, pn.r) * (1.0 - 0.7 * x.bf);
        x.em = smoothstep(0.5, 0.7, pn.a);
        // stretches of shore that are nearly clear (grass / forest down to the water) vs packed with blocks
        // (noise offset chosen so the film shores are dense; ~38 % of the lake shore nearly clear)
        // boulder fields are not a uniform carpet: ~10 m grassy gaps between the clusters
        if (x.bf > 0.0) x.fg = smoothstep(0.35, 0.55, texture(uNoise3, vec3(x.cc * 0.0065 + vec2(0.37, 0.71), 0.47)).g);
        x.bm = max(mix(0.15, 1.25, smoothstep(0.4, 0.6, texture(uNoise3, vec3(x.cc * 0.0013 + vec2(0.6206, 0.5708), 0.1936)).g)), 1.25 * x.bf);
    }
    // can any slot >= 1 be dense here? (height range of the slot positions vs the band / field support)
    float dr = (abs(x.g.x) + abs(x.g.y)) * 2.5;
    bool near = x.rel + dr > -3.2 && x.rel - dr < (x.bf > 0.0 ? 12.0 : 5.0);
    bool many = near || x.f.b > 0.08;
    // (nearly clear stretches of shore away from scree / boulder fields: 3 slots, the others would be ~empty)
    x.ns = many ? (x.bm < 0.4 && x.bf <= 0.0 && x.f.b < 0.08 ? 3 : RK_NS) : 1;
    // beach cobbles (small slots 3-4) only where the cell reaches the waterline band
    bool beach = x.rel + dr > -0.6 && x.rel - dr < 1.9;
    // (rocks5 S2) boulder fields: the cobble slots carry loose 5-20 cm field stones between the blocks (shot 4)
    bool fldS = x.bf > 0.3 && x.rel < 10.0;
    x.ns2 = many ? (beach || fldS ? RK_NS2 : 3) : 1;
    if (beach) x.cob = mix(0.5, 1.0, smoothstep(0.3, 0.6, texture(uNoise3, vec3(x.cc * 0.023, 0.61)).g));
    // jointed outcrops: on steep bare ground the joint blocks stand out of the slope as ledges / steps
    // (the smooth 1 m heightfield alone reads as plaster); patchy, fewer in the forest and on meadows
    float slp = length(x.go);
    // (rocks4 R12 / R13) from ~19 deg (was 31), on meadows too (half), more patches: r7's foreground of jointed blocks,
    // r4's stepped basin ledges (the falls bank keeps half; fallsWetness darkens them)
    if (ocR && slp > 0.35 && x.rel > 1.0) {
        float pa = texture(uNoise3, vec3(x.cc * 0.0061 + vec2(0.13, 0.77), 0.83)).g;
        x.oc = smoothstep(0.35, 0.7, slp) * (1.0 - 0.55 * saturate(x.f.r * 1.2)) * (1.0 - 0.5 * saturate(x.f.a))
             * smoothstep(0.15, 0.35, pa);
        if (x.oc > 0.0) {
            vec3 pc = vec3(x.cc.x, h0, x.cc.y);
            x.oc *= (1.0 - saturate(4.0 * fallsWetness(pc))) * (1.0 - 0.5 * saturate(fallsBank(pc)));
        }
    }
    return true;
}
// per-cell / per-slot randoms: one hash per slot, four 8-bit uniforms
uint rkCellHash(ivec2 cell) { return hash2u(cell + ivec2(10007, 20011)); }
vec4 rkSlotHash(uint hc, int k) {
    uint h = hashu(hc + uint(k) * 0x9E3779B9u);
    return vec4(uvec4(h, h >> 8u, h >> 16u, h >> 24u) & 255u) * (1.0 / 256.0) + (0.5 / 256.0);
}
// (rocks5) slots >= 16 (the extra cobble slots of the small layer) are tagged x + 2^23, y + (k - 16) * 2^20: the G-buffer
// stores the cell as float (exact below 2^24); slots < 16 keep their tags (and the hashes of every existing rock)
#define ROCK_HI_TAG 8388608
ivec2 rkTagS(ivec2 cell, int k) { return k < 16 ? cell + ivec2(0, k * ROCK_SLOT_TAG) : cell + ivec2(ROCK_HI_TAG, (k - 16) * ROCK_SLOT_TAG); }
int rkSlotOf(ivec2 c) { return (c.y + ROCK_SLOT_TAG / 2) / ROCK_SLOT_TAG + (c.x >= ROCK_HI_TAG ? 16 : 0); }

// stage 1 (cheap, no fetches): does slot k of a cell (centre c0) hold a rock; where, how big,
// which kind. x = context of the parent 5 m cell.
bool rkSlot(RkCtx x, vec2 c0, bool small, int k, vec4 hs, vec4 hs0, out vec2 pos, out float R, out float kind) {
    pos = vec2(0.0); R = 0.0; kind = 0.0;
    float cs = small ? ROCK2_CELL : ROCK_CELL;
    float jf = k == 0 ? 0.3 : 0.48;
    vec2 pos0 = c0 + (hs.xy - 0.5) * (2.0 * jf * cs);
    // clusters: in about half of the cells the later blocks lie around the first one (jumbles of
    // touching blocks) instead of anywhere in the cell
    // (rocks4 R14) boulder fields: every later block is a satellite of the first one (touching / leaning on it at 0.7-1.1
    // of its estimated size, 0.2-0.7 of it), so the field is jumbled clusters with grass between, not evenly spaced props
    bool fcl = !small && x.bf > 0.5;
    bool clus = k > 0 && k < RK_OC0 && (fcl || fract(hs0.z * 5.31 + hs0.w * 2.17) < 0.5);
    float R0e = mix(0.32, 1.9, pow(hs0.w, 0.8));
    if (clus) {
        vec2 dir = normalize(vec2(hs.x - 0.5, fract(hs.x * 7.31 + hs.y * 3.7) - 0.5) + 1e-4);   // (no sin / cos)
        pos0 = fcl ? c0 + (hs0.xy - 0.5) * (2.0 * 0.3 * cs) + dir * (R0e * mix(0.8, 1.25, hs.y) + 0.2)
                   : c0 + (hs0.xy - 0.5) * (0.4 * cs) + dir * cs * (mix(0.12, 0.32, hs.y) + 0.1 * x.bf);
    }
    float relP = x.rel + dot(x.g, pos0 - x.cc);
    // band along the waterline (some blocks submerged in the shallows), wider in boulder fields
    // (the big blocks of slots 0-1 also lie deeper: submerged boulders seen through the clear shallows);
    // dense along some stretches of shore, nearly clear along others
    float band = smoothstep(k <= 1 ? -3.2 : (k == 2 ? -2.4 : -1.6), -0.2, relP) * (1.0 - smoothstep(2.5, 5.0 + 6.0 * x.bf, relP)) * x.bm;
    float field = x.bf * mix(1.0, x.fg, smoothstep(1.5, 4.0, relP)) * step(-0.6, relP) * (1.0 - smoothstep(3.0, 12.0, relP)) * (1.0 - 0.6 * saturate(x.f.r * 1.5) * (1.0 - 0.6 * x.bf));
    float sh = min(max(band, 0.8 * field), 1.0);
    float fk = float(k);
    float u = hs.w;
    float big = saturate(0.55 * x.f.b + 0.6 * sh + 0.2 * x.f.r);
    bool outc = !small && k >= RK_OC0;
    if (outc) {
        if (x.oc <= 0.0 || x.rel + dot(x.go, pos0 - x.cc) < 1.2 || hs.z > x.oc * (k == RK_OC0 ? 0.85 : 0.6)) return false;
        kind = 2.0;
        R = mix(1.0, 1.95, u);
    } else if (!small) {
        float dens = k == 0 ? max(0.04 + 0.80 * x.f.b + 0.95 * sh + 0.16 * x.f.r - 0.02 * x.f.a, 0.95 * field)
                            : (0.95 * sh + 0.3 * field + 0.6 * x.f.b - 0.05) * (1.0 - 0.07 * fk);
        if (hs.z > dens) return false;
        // boulder fields: the first two slots carry the big blocks
        bool bigS = k == 0 || (k == 1 && field > 0.3);
        R = bigS ? mix(0.32, 1.4 + 0.6 * max(big, field), pow(u, mix(2.2, 0.8, max(big, field))))
                 : mix(0.15, 0.7 + 1.1 * big, pow(u, 1.8));
        // boulder fields: the other slots hold mid-size blocks too (packed jumble)
        // (mostly 0.3-0.6 m with a few big ones)
        if (!bigS) R = mix(R, mix(0.3, 1.6, pow(u, 1.5)), smoothstep(0.2, 0.5, field));
        // (rocks4 R14) satellites (slot 1 keeps its big block, leaning on slot 0; b1: all 0.2-0.7 R0 = a lawn with pebbles, r8)
        if (fcl && k > 1) R = mix(R, max(0.22, R0e * mix(0.3, 0.95, pow(u, 1.2))), smoothstep(0.2, 0.5, field));
        // in the shallows many blocks are big enough to break the surface
        // (more so in boulder-strewn stretches; clean gravel shallows elsewhere)
        // (rocks3c) over a wider depth range, thinning with depth (r7: one depth band = a row of evenly spaced blocks
        // along the contour, a string of beads)
        if (relP < 0.0 && relP > -2.6 && fract(u * 5.7 + hs.x * 3.1) < mix(0.12, 0.55, x.em) * smoothstep(-2.6, mix(-1.0, -1.4, x.em), relP)) R = max(R, min((0.25 - relP) * 1.25, 1.85));
        // slab outcrops: the big blocks of slab patches near the waterline become flat slabs
        float slabP = x.sl * smoothstep(-1.8, -0.8, relP) * (1.0 - smoothstep(2.5, 5.0, relP));
        if (k <= 1 && fract(hs.z * 7.31 + u * 3.7) < slabP) { kind = 1.0; R = mix(1.25, 2.05, u); }
    } else {
        float shS = sh * smoothstep(-1.6, -0.4, relP);
        if (k >= 3) {
            // loose beach cobbles (3D: silhouettes / contact shadows at grazing angles),
            // near the waterline only, shrinking away at 8-12 m from the camera
            // (every cobble beach, not only the bouldery stretches: the terrain draws the gravel there too;
            // fewer on forest / meadow shores; patchy)
            float dens = x.cob * smoothstep(-0.3, -0.02, relP) * (1.0 - smoothstep(1.1, 1.6, relP))
                       * (1.0 - 0.4 * saturate(x.f.r * 1.5 + x.f.a));
            // (rocks5 S1 / S2) twice the slots (round 4: "far too little detail in the small stones"; a beach is packed
            // with 3D stones of 2-30 cm), a little smaller on average; field stones between the boulder-field blocks
            dens = max(dens, k < 14 ? 0.45 * field * step(0.3, x.bf) * (1.0 - 0.6 * saturate(x.f.r * 1.5)) : 0.0);
            if (hs.z > dens) return false;
            R = mix(0.05, 0.22, u * sqrt(u));
        } else {
            float dens = k == 0 ? 0.03 + 0.85 * x.f.b + 0.95 * shS + 0.1 * x.f.r - 0.05 * x.f.a
                                : (0.95 * shS + 0.5 * x.f.b - 0.05) * (1.0 - 0.12 * fk);
            // (rocks3c) fewer loose stones on the forest banks (shot 8: a scatter of pale chunks down the bank; target: blocks)
            dens *= 1.0 - 0.55 * saturate(x.f.r * 1.5);
            if (hs.z > dens) return false;
            R = k == 0 ? mix(0.1, ROCK2_RMAX, pow(u, 1.2)) : (k == 1 ? mix(0.08, 0.3, u) : mix(0.05, 0.22, u));
        }
    }
    R = min(R, small ? ROCK2_RMAX : (k >= 2 && !outc ? ROCK_RMAX_S : ROCK_RMAX));
    float over = small ? ROCK2_OVER : (k >= 2 && !outc ? ROCK_OVER_S : ROCK_OVER);
    float jm = clamp(0.5 * cs + over - ROCK_BOUND * R, 0.0, jf * cs);
    pos = clus ? clamp(pos0, c0 - jm, c0 + jm) : c0 + (hs.xy - 0.5) * 2.0 * jm;
    if (x.strm > 0.5) { float qS = x.strm - 1.0, anS = mod(qS, 64.0) * (6.2832 / 64.0);
        if (floor(qS / 64.0) * 0.1 - 12.0 + dot(vec2(cos(anS), sin(anS)), pos - x.cc) < 0.9 * R + 0.35) return false; }
    // ---- camera-dependent sizes (the position stays: it is valid for any smaller size)
    float dc = length(pos - uCamPos.xz);
    // small stones flatten into the ground with distance (rkSmallFade, in rkBuild); gone beyond it
    if (small && dc > (k >= 3 ? ROCK3_D1 : ROCK2_D1)) return false;
    // outcrop blocks sink into the slope far away (no pop at the rock distance limit)
    if (outc) R *= 1.0 - smoothstep(90.0, 170.0, dc);
    // (r5 qa5 B05) the other blocks shrink away before ROCK_MAX_DIST instead of popping off at it
    // (ROCK_MAX_DIST = 380 m is defined later, in lighting.glsl)
    else if (!small) R *= 1.0 - smoothstep(0.8 * 380.0, 0.97 * 380.0, dc);
    // blocks below ~2.5 pixels shrink away (they would only sparkle in the moving film; cheaper too)
    float px = R / max(dc * uPixAng, 1e-6);
    if (px < 0.6) return false;
    R *= smoothstep(0.6, 1.4, px);
    return true;
}
// small stones / beach cobbles flatten into the ground with the camera distance (round 2 review: shrinking made
// them visibly grow as the camera moved; their plan size stays, the top sinks flush): 0 = full, 1 = gone
float rkSmallFade(int k, vec2 pos) {
    float dc = length(pos - uCamPos.xz);
    return k >= 3 ? smoothstep(ROCK3_D0, ROCK3_D1, dc) : smoothstep(ROCK2_D0, ROCK2_D1, dc);
}
ivec2 rkParent(ivec2 cell2) { return ivec2(floor(vec2(cell2) * 0.5)); }
// small-layer cells re-anchor the parent's height plane at their own centre (height hc there): on a curved bank
// the 5 m parent's plane misses the narrow waterline band by a metre or more (no cobbles at the foot of a bank)
void rkSmallLocal(inout RkCtx x, vec2 cc, float hc) {
    if (x.rel < 98.0) x.rel = hc - (x.h0 - x.rel);
    x.h0 = hc; x.cc = cc;
    // (rocks3 i6) the beach-cobble slots / density from the small cell's own height: the parent's test missed every
    // beach whose 5 m cell centre lies on the bank above (c_box: no 3D cobbles at all within 14 m)
    if (x.ns2 > 1) {
        float dr = (abs(x.g.x) + abs(x.g.y)) * 1.25;
        bool beach = x.rel + dr > -0.6 && x.rel - dr < 1.9;
        x.ns2 = beach || (x.bf > 0.3 && x.rel < 10.0) ? RK_NS2 : 3;     // (rocks5 S2) + field stones
        x.cob = beach ? mix(0.6, 1.0, smoothstep(0.3, 0.6, texture(uNoise3, vec3(cc * 0.023, 0.61)).g)) : 0.0;
    }
}

// stage 2: full rock (ground fit, frame, facets, crack)
bool rkBuild(ivec2 cell, vec4 hs, vec2 pos, float R, float kind, RkCtx x, out Rock rk) {
    float h = terrainH(pos, false);
    bool small = cell.x >= ROCK2_TAG / 2;
    vec2 g = x.g;                                    // small stones: tilt from the cell plane (no fetches)
    if (!small && (R > 1.0 || kind > 1.5)) {
        float dd = 0.8 * R + 0.4;
        g = vec2(terrainH(pos + vec2(dd, 0.0), false) - h, terrainH(pos + vec2(0.0, dd), false) - h) / dd;
    }
    // sit on the rendered ground: the marcher adds the sub-metre relief near the camera (detailWeight);
    // its first octave here, the two small ones (+-0.12 m) covered by sinking the rock a little
    float det = 1.0 - smoothstep(120.0, 260.0, length(vec3(pos.x, h, pos.y) - uCamPos));
    if (small) {
        // (rocks3 i7) small stones / cobbles sit on exactly the ground the marcher draws (hq height + the whole micro
        // relief): with the first octave only and the 5 cm sink, most 6-24 cm cobbles were buried under the +-9 cm
        // small octaves (c_box beach: no 3D cobbles); near the camera they also take the local slope
        // (the parent plane misses a steep bank)
        h = terrainH(pos, true);
        if (det > 0.0) h += det * microRelief(pos) - 0.01;
        float wg = 1.0 - smoothstep(10.0, ROCK3_D1, length(pos - uCamPos.xz));
        if (wg > 0.0) {
            const float e = 0.35;
            g = mix(g, vec2(terrainH(pos + vec2(e, 0.0), false) - terrainH(pos - vec2(e, 0.0), false),
                            terrainH(pos + vec2(0.0, e), false) - terrainH(pos - vec2(0.0, e), false)) / (2.0 * e), wg);
        }
    } else if (det > 0.0) h += det * (0.22 * gnoise(pos * 0.55 + 1.7) - 0.05);
    float slope = length(g);
    vec3 nT = normalize(vec3(-g.x, 1.0, -g.y));
    vec4 h2 = hash42i(cell + ivec2(-7919, 3571));
    vec4 h3 = hash42i(cell + ivec2(4513, -9241));
    vec4 h4 = hash42i(cell + ivec2(1237, 7717));
    rk.kind = kind;
    rk.rel = x.rel + dot(x.g, pos - x.cc);
    bool slab = kind > 0.5 && kind < 1.5;
    bool outc = kind > 1.5;
    // proportions: blocky (some tall, some flattish); slabs thin and broad
    rk.r = slab ? R * vec3(0.94, mix(0.2, 0.32, h2.x), mix(0.55, 0.85, h2.y))
         : (outc ? R * vec3(0.94, mix(0.3, 0.46, h2.x), mix(0.6, 0.88, h2.y))
                 : R * vec3(0.9, 0.36 + 0.42 * h2.x + 0.1 * h2.y * h2.y, 0.55 + 0.35 * h2.y));
    if (small && R < 0.2) rk.r.y *= 0.75;                    // loose beach cobbles lie flat
    // (rocks4 R4) wave-washed band: low tabular / subrounded blocks (h/w 0.3-0.5), no gem shards / tombstones
    bool wwb = !small && !(kind > 0.5) && x.rel + dot(x.g, pos - x.cc) < 1.2;
    if (wwb) rk.r.y *= 0.7;
    // small stones far from the camera: flattened (plan size kept), sunk flush below (rkSmallFade)
    float sF = small ? rkSmallFade(rkSlotOf(cell), pos) : 0.0;
    rk.r.y *= mix(1.0, 0.12, sF);
    // local frame: up = mostly the terrain normal + random tilt (slabs follow the slope: they dip into the lake)
    // (outcrop blocks, round 2 review: tilted part-way toward level - a tread with a riser on the downhill side, not
    // a paving slab laid on the slope; j1: at 60 % the treads were buried down to pale slivers)
    vec3 up = slab ? normalize(mix(vec3(0.0, 1.0, 0.0), nT, 0.9) + (h2.zwx - 0.5) * vec3(0.12, 0.0, 0.12))
            : (outc ? normalize(mix(nT, vec3(0.0, 1.0, 0.0), 0.35) + (h2.zwx - 0.5) * vec3(0.1, 0.0, 0.1))
                    : normalize(mix(vec3(0.0, 1.0, 0.0), nT, 0.6) + (h2.zwx - 0.5) * vec3(0.4, 0.0, 0.4)));
    rk.ang = hs.x * 13.7 + hs.y * 5.1 + h4.x * 3.0;
    if (outc) {
        // outcrop blocks follow the joint set of the face around them (rkNearRock: the same slowly drifting strike),
        // so their edges line up with the painted joints; either the strike or the cross direction
        float r0 = 0.7 * gnoise(pos * 0.0011 + 4.0);                        // rot2(r0) * (0.83, 0.56) there
        rk.ang = atan(0.56, 0.83) + r0 + (h4.x < 0.5 ? 0.0 : 1.5708) + (h4.y - 0.5) * 0.16;
    }
    vec3 fw = vec3(cos(rk.ang), 0.0, sin(rk.ang));
    vec3 ax = normalize(fw - up * dot(fw, up));
    rk.m = mat3(ax, up, cross(ax, up));
    // burial: 12-50 % of the height below ground, deeper on slopes; slabs show their top only
    float bury = slab ? mix(0.22, 0.45, h3.x) : mix(0.12, 0.5, h3.x * h3.x);
    // (i11) loose beach cobbles rest ON the gravel: 4-20 % buried, no extra slope sink (at 12-50 % they stuck
    // out 1-6 cm and did not read as 3D stones)
    bool cobble = small && R < 0.25;
    if (cobble) bury = mix(0.04, 0.2, h3.x);
    // (rocks3c) small stones inland (grass / forest floor, shot 4 / 8) sink deeper: half-hidden, not eggs laid on the turf
    if (small && !cobble) bury = min(bury + 0.3 * smoothstep(0.8, 2.0, rk.rel), 0.68);   // (rocks4 G10: ~half sunk)
    else if (cobble) bury = min(bury + 0.25 * smoothstep(0.9, 1.6, rk.rel), 0.5);
    rk.c = vec3(pos.x, h + rk.r.y * (1.0 - 2.0 * bury) - slope * R * (slab ? 0.1 : (cobble ? 0.08 : 0.25)), pos.y);
    rk.c.y -= sF * rk.r.y * (1.95 - 2.0 * bury);             // flattened stones sink flush
    if (outc) {
        // outcrop block: the uphill edge of the top face flush with the slope (+0.05-0.15 m), the downhill edge a riser
        // standing proud (0.3 m .. 1.1 r.y; the uphill end sinks deeper where the slope would need more), so the
        // tread and the riser both show; the bottom stays buried
        vec2 dh = -g / max(slope, 1e-4);                                     // downhill (horizontal)
        float ex = 0.88 * (abs(dot(dh, ax.xz)) * rk.r.x + abs(dot(dh, rk.m[2].xz)) * rk.r.z);
        float topS = dot(up.xz, dh) / max(up.y, 0.2);                        // descent of the top plane per m downhill
        float s = min(max(2.0 * ex * (slope - topS) + mix(0.05, 0.15, h3.x), 0.3), 1.1 * rk.r.y) * min(R / 1.4, 1.0);
        float tc = h - slope * ex + s + topS * ex;                           // top-face centre height
        rk.c = vec3(pos.x, tc - 0.92 * rk.r.y * up.y, pos.y);
    }
    rk.seed = fract(hs.w * 13.7 + hs.z * 3.1 + h4.w);
    vec3 b = rk.r;
    // fracture facets: a flat-ish top, four near-vertical side faces, two corner chips
    // (outcrop blocks: faces along the local axes = the joint planes)
    float a0 = outc ? 0.0 : h3.x * TAU;
    vec2 e1 = vec2(cos(a0), sin(a0)), e2 = vec2(-e1.y, e1.x);
    // side faces lean inward up to ~22 deg; (rocks3) some lean outward a little: undercuts, no trapezoid tombstones
    vec4 el = slab ? mix(vec4(0.15), vec4(0.9), h4) : (outc ? mix(vec4(0.02), vec4(0.3), h4) : mix(vec4(-0.06), vec4(0.4), h4));
    vec4 jt = (h4.yzwx - 0.5) * (outc ? 0.25 : 0.8);                                       // azimuth jitter (~tan)
    // (rocks3) tilted tops; (rocks3b) strongly tilted on ~30 % only: tt 0.95 on every block + the steep corner cuts
    // made pointed "tent" roofs (crumpled-paper prisms in shot 4); target: mostly flattish tops
    float tt = slab || wwb ? 0.25 : (outc ? 0.2 : (fract(h3.w * 7.7 + h2.x * 3.1) < 0.3 ? 0.95 : 0.45));   // (rocks4 R4: washed band flat)
    rk.f0 = rkFacet(normalize(vec3((h3.y - 0.5) * tt, 1.0, (h3.z - 0.5) * tt)), b, mix(0.86, 0.97, h3.w));
    vec2 d1 = e1 + e2 * jt.x, d2 = e2 - e1 * jt.y, d3 = -e1 - e2 * jt.z, d4 = -e2 + e1 * jt.w;
    rk.f1 = rkFacet(normalize(vec3(d1.x, el.x, d1.y)), b, mix(0.82, 0.96, h4.z));
    rk.f2 = rkFacet(normalize(vec3(d2.x, el.y, d2.y)), b, mix(0.82, 0.96, h4.w));
    rk.f3 = rkFacet(normalize(vec3(d3.x, el.z, d3.y)), b, mix(0.82, 0.96, h4.x));
    rk.f4 = rkFacet(normalize(vec3(d4.x, el.w, d4.y)), b, mix(0.82, 0.96, h4.y));
    vec2 d5 = mix(e1, e2, h2.z) * sign(h2.w - 0.5), d6 = mix(-e1, e2, h3.z) * sign(0.5 - h2.y);
    // (round 2 review: corner planes cut deeper - more flat faces per block, less potato)
    rk.f5 = rkFacet(normalize(vec3(d5.x, wwb ? mix(0.2, 0.7, h2.w) : mix(0.4, 1.4, h2.w), d5.y)), b, mix(0.60, 0.75, h3.y));
    rk.f6 = rkFacet(normalize(vec3(d6.x, mix(-0.1, 0.7, h2.y), d6.y)), b, mix(0.60, 0.75, h3.w));
    // (rocks3) three more random cuts: across a top edge, a vertical edge and a corner, 74-90 % of the support
    // distance - granite blocks show 10-15 facets of every size, not 4 sides and a lid
    vec4 h5 = hash42i(cell + ivec2(-3301, 8761));
    float cw = slab ? 9.0 : (outc ? 0.06 : 0.0);          // (slabs: none; outcrop blocks: shallow)
    // (rocks3b) the top-corner cut f9 flatter / shallower (0.45-0.95 rad, 82-93 %): at 0.6-1.25 rad it replaced the
    // top by a second roof plane (ridge / tent tops)
    rk.f7 = rkFacet(rkDir(h5.x, h5.y, 0.3, 0.95), b, mix(0.76, 0.9, h5.z) + cw);
    rk.f8 = rkFacet(rkDir(fract(h5.x + 0.37 + 0.25 * h5.w), h5.z, -0.25, 0.45), b, mix(0.76, 0.9, h5.w) + cw);
    rk.f9 = rkFacet(rkDir(fract(h5.x + 0.71 + 0.2 * h5.y), h5.w, wwb ? 0.2 : 0.45, wwb ? 0.5 : 0.95), b, mix(0.82, 0.93, fract(h5.y * 7.1)) + cw);
    // fracture edges rounded by weathering (1.2-3.5 % of the size: crisp light / shadow edges); slabs smoother
    // (rocks4 R13: slabs 5-9 % -> 2-4 %, the rounded whaleback read as plaster; R4: washed blocks 3-6 %, subrounded)
    rk.rr = slab ? R * mix(0.02, 0.04, h3.y) : R * (outc ? mix(0.012, 0.03, h3.y) : (wwb ? mix(0.03, 0.06, h3.y) : mix(0.012, 0.035, h3.y)));
    rk.env = saturate(x.f.r * 1.6 - (1.0 - smoothstep(1.0, 4.0, rk.rel)) * 0.8);
    rk.fld = max(x.bf, saturate(x.f.a * 1.5)) * smoothstep(0.6, 2.5, rk.rel) * (1.0 - rk.env);
    // some blocks are worn, rounded boulders (glacial erratics) - but still blocky in outline (10 %)
    // (rocks3b) + up to 30 % more on the old inland surfaces (boulder fields, meadows, forest floor:
    // mostly subrounded blocks there; the wave-washed shore blocks stay angular)
    // (rocks3d) hardly any on the wave-washed band (r1: a smooth "bread loaf" among the angular shore blocks)
    bool worn = !slab && !outc && fract(h2.z * 5.7 + h4.w * 2.3) < mix(0.02, 0.1, smoothstep(0.8, 2.0, rk.rel)) + 0.3 * max(rk.fld, rk.env);
    rk.ell = worn ? mix(1.42, 1.5, h3.z) : (outc ? 2.2 : (wwb ? 1.6 : 1.75));       // (outcrop blocks: angular joint-bounded outline)
    if (worn) rk.rr = R * mix(0.05, 0.09, h3.y);
    // loose beach cobbles: water-worn, rounded (rounded 5-30 cm stones), a few still angular
    // (rocks3c) only at the waterline (beach cobble slots / the wave-washed band): the stones in the grass of the stream
    // valley and on the forest floor are angular granite chunks (shot 4: smooth round white eggs on the turf)
    bool washed = rkSlotOf(cell) >= 3 || rk.rel < 1.0;
    if (small && R < 0.25 && washed && fract(h4.y * 3.3 + h2.x) < 0.75) { rk.ell = mix(1.22, 1.36, h3.z); rk.rr = R * mix(0.1, 0.18, h3.y); }
    rk.taper = slab || outc ? 0.0 : mix(0.0, 0.08, fract(h2.w * 9.1));
    // some blocks are split by a sub-vertical crack (frost wedging)
    rk.ck = vec4(0.0, 1.0, 0.0, 1e3); rk.ckw = 0.0;
    // (rocks3d) blocks: across the LONG axis (local x +- ~25 deg), within 25 % of the half length of the middle - a split
    // block; 0.4-1.2 % R wide. A random azimuth let the wedge run parallel to a face just behind its edge: a black
    // "mail slot" with an overhanging lip across the top (r8, shot 4 / 8)
    // (rocks4 R10) a hairline joint, drawn by shadeRock as a dark line (no open SDF wedge: the 0.4-1.2 % R gap read as a black
    // "mail slot" / bunker window, r8 / shot 4 / shot 2); offset x0.12 (mid-block); none where it would run along a face
    if (R > 0.45 && h3.w < (slab ? 0.3 : 0.35)) {
        vec3 cn = slab ? rkDir(h2.y, 0.5, -0.1, 0.1) : normalize(vec3(1.0, (h3.z - 0.5) * 0.5, (h2.y - 0.5) * 0.9));
        rk.ck = vec4(cn, (h2.w - 0.5) * 0.12 * dot(abs(cn), b));
        rk.ckw = R * (slab ? mix(0.003, 0.008, h2.x) : mix(0.0015, 0.004, h2.x));
        float par = max(max(abs(dot(cn, rk.f1.xyz)), abs(dot(cn, rk.f2.xyz))), max(abs(dot(cn, rk.f3.xyz)), abs(dot(cn, rk.f4.xyz))));
        par = max(par, max(max(abs(dot(cn, rk.f5.xyz)), abs(dot(cn, rk.f6.xyz))), max(abs(dot(cn, rk.f7.xyz)), abs(dot(cn, rk.f8.xyz)))));
        if (!slab && par > 0.8) rk.ckw = 0.0;
    }
    // fracture steps: a block broken off one top edge, a broken-off corner on the opposite side
    // (rocks4 C5: 55 / 30 / 40 % with steps 0.6 b deep and square inner corners = cut cinder blocks / bunker windows (shot 2 / 4):
    // 15 / 10 / 10 %, at most ~0.2 b deep, sloping treads and leaning risers (sdRockLocal))
    rk.nt = vec2(0.0); rk.nt2 = vec3(0.0);
    if (!slab && R > 0.3 && fract(h4.z * 7.3 + h2.x) < 0.15)
        rk.nt = vec2(b.y * mix(0.12, 0.24, fract(h3.y * 5.1)), b.x * mix(0.08, 0.2, fract(h2.z * 3.7)));
    if (!slab && R > 0.3 && fract(h4.y * 5.9 + h3.z * 2.3) < 0.1)
        rk.nt2 = vec3(b.y * mix(0.12, 0.24, fract(h2.y * 7.7)), b.x * mix(0.1, 0.2, fract(h4.x * 3.3)), b.z * mix(0.1, 0.22, fract(h3.w * 6.1)));
    rk.nt3 = vec2(0.0);
    if (!slab && R > 0.3 && fract(h3.x * 6.1 + h4.z * 1.7) < 0.1)
        rk.nt3 = vec2(b.y * mix(0.1, 0.2, fract(h4.w * 4.3)), b.x * mix(0.08, 0.18, fract(h2.w * 5.9)));
    // conchoidal chips: flat pieces knocked off a top edge and a vertical edge (crisp rims; most blocks)
    rk.ch0 = vec4(0.0); rk.ch1 = vec4(0.0);
    if (!slab && R > 0.2 && rk.ell > 1.5) {
        float u0 = fract(h4.w * 3.7 + h2.y * 1.3);
        rk.ch0 = rkChip(rk.f0, u0 < 0.5 ? rk.f2 : rk.f4, fract(u0 * 7.9), R * mix(0.14, 0.32, fract(h3.x * 5.3)), 0.6 * b.x);
        float u1 = fract(h2.z * 3.1 + h3.y * 2.9);
        if (u1 < 0.7) rk.ch1 = rkChip(u1 < 0.35 ? rk.f1 : rk.f3, u1 < 0.35 ? rk.f2 : rk.f4, fract(u1 * 11.3), R * mix(0.1, 0.24, fract(h4.x * 7.7)), 0.6 * b.y);
    }
    if (!slab && !outc) rk.rr *= 1.0 + 1.6 * rk.fld * fract(h3.w * 3.7 + h2.y);   // (i5) older field blocks: rounder edges
    // the camera must not end up inside a block: a block whose footprint holds the camera sinks until only
    // a low flat top (<= 0.15 m above the ground) is left; continuous in the camera's horizontal position and
    // independent of its height (a rising camera does not make blocks grow); blocks more than 1.8 m away
    // from the camera keep their shape (foreground blocks stay)
    // distance from the camera to the block's footprint (yaw frame; ~1.1 r covers the tilt), fade 0.3-1.8 m
    vec2 dcv = uCamPos.xz - pos;
    vec2 rf = 1.1 * rk.r.xz;
    float dE = (length(vec2(dot(dcv, fw.xz), dot(dcv, vec2(-fw.z, fw.x))) / rf) - 1.0) * min(rf.x, rf.y);
    float wc = 1.0 - smoothstep(0.3, 1.8, dE);
    if (wc > 0.0) {
        vec3 upL = vec3(rk.m[0].y, rk.m[1].y, rk.m[2].y);          // world up in the local frame
        float top = rk.c.y + rk.ell * length(rk.r * upL) + 0.08 * R;   // the shape lies inside the ell * r ellipsoid
        rk.c.y -= max(top - (h + 0.15), 0.0) * wc;
    }
    return slope < 1.35 || (outc && slope < 3.0);           // no boulders glued to cliffs (outcrop blocks lie in them)
}

// rock of a (possibly small-layer- and slot-tagged) cell

// slot 0 of a main cell (the biggest block; used by the ground cover to clear grass)

float gRkN3 = 0.0;      // 1 while shadeRock takes its normal: adds the smallest SDF relief octave
float rkEllipsoid(vec3 q, vec3 r) {
    float k0 = length(q / r), k1 = length(q / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-5);
}

// signed distance in the rock's local frame (not yet Lipschitz-scaled);
// lod 0 = smooth blocks, 1 = full relief
float sdRockLocal(vec3 q, Rock rk, float lod) {
    // (rocks3) lod in (1, 2]: within ~12 m of the camera, the crumbly cm relief also enters the shape (silhouettes)
    float lodN = max(lod - 1.0, 0.0);
    lod = min(lod, 1.0);
    vec3 b = rk.r;
    float R = b.x;
    // taper: narrower toward the top (only shrinks, so the bound still holds)
    q.xz *= 1.0 + rk.taper * (clamp(q.y / b.y, -1.0, 1.0) * 0.5 + 0.5);
    float kf = rk.rr;
    float p0 = dot(q, rk.f0.xyz) - rk.f0.w, p1 = dot(q, rk.f1.xyz) - rk.f1.w;
    float p2 = dot(q, rk.f2.xyz) - rk.f2.w, p3 = dot(q, rk.f3.xyz) - rk.f3.w;
    float p4 = dot(q, rk.f4.xyz) - rk.f4.w, p5 = dot(q, rk.f5.xyz) - rk.f5.w, p6 = dot(q, rk.f6.xyz) - rk.f6.w;
    float p7 = dot(q, rk.f7.xyz) - rk.f7.w, p8 = dot(q, rk.f8.xyz) - rk.f8.w, p9 = dot(q, rk.f9.xyz) - rk.f9.w;
    // every edge weathered differently: some nearly sharp, some rounded (no uniform soap-bar bevel)
    vec4 km = mix(vec4(0.2), vec4(1.2), fract(rk.seed * vec4(17.13, 31.71, 47.37, 59.93)));
    float d = smax(p0, p1, kf * km.x);
    d = smax(d, p2, kf * km.y);
    d = smax(d, p3, kf * km.z);
    d = smax(d, p4, kf * km.w);
    d = smax(d, p5, kf * 0.4 * km.y);
    d = smax(d, p6, kf * 0.4 * km.z);
    d = smax(d, p7, kf * 0.5 * km.x);
    d = smax(d, p8, kf * 0.5 * km.w);
    d = smax(d, p9, kf * 0.35 * km.y);
    d = smax(d, -q.y - 0.95 * b.y, kf);                    // base (buried)
    d = smax(d, rkEllipsoid(q, b * rk.ell), 0.05 * R);     // trims the corners (more on worn boulders)
    d = max(d, length(q) - 1.45 * R);                      // keeps the corners (+ face bulges) inside ROCK_BOUND
    // (rocks4 C5) sloping treads / leaning risers (~20-25 deg) and softer inner corners: no square CSG notches
    if (rk.nt.x > 0.0) {                                     // fracture step on the top / side-1 edge
        float ta = p0 + rk.nt.x, tb = p1 + rk.nt.y;
        d = smax(d, min(ta + 0.42 * tb, tb + 0.36 * ta), kf * 0.9);
    }
    if (rk.nt2.x > 0.0) {                                    // broken-off corner (top / sides 2 and 3)
        float ta = p0 + rk.nt2.x, tb = p2 + rk.nt2.y, tc = p3 + rk.nt2.z;
        d = smax(d, min(min(ta + 0.3 * (tb + tc), tb + 0.3 * ta), tc + 0.3 * ta), kf * 0.9);
    }
    if (lod > 0.0) {
        if (rk.nt3.x > 0.0) {                                // second fracture step (top / side 3)
            float ta = p0 + rk.nt3.x, tb = p3 + rk.nt3.y;
            d = smax(d, min(ta + 0.5 * tb, tb + 0.3 * ta), kf * 0.6);
        }
        if (rk.ch0.w > 0.0) d = smax(d, -rkChipD(q, rk.ch0), 0.25 * kf);   // flat chips (crisp rims)
        if (rk.ch1.w > 0.0) d = smax(d, -rkChipD(q, rk.ch1), 0.25 * kf);
    }
    // (rocks4 R10: the crack is no longer cut into the shape - shadeRock draws it as a hairline joint)
    d *= 1.0 / (1.0 + rk.taper);                            // undo the taper stretch (Lipschitz)
    // weathered fracture faces are slightly convex (not planar prisms; round 2 review: 0.08 -> 0.03, flatter faces)
    // (rocks4 R15: 0.05 - gently warped faces, no ruler-flat concrete planes)
    d = mix(d, rkEllipsoid(q, b * 1.12), 0.05);
    // (rocks3) edge proximity (0 mid-face .. 1 on an edge): weathering eats edges and corners first - the crumbly
    // relief is deeper there, so the long fracture edges get chipped, irregular outlines instead of ruler lines
    float edge = 0.0;
    if (lod > 0.0) {
        float m1 = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, max(p7, max(p8, p9)))));
        float m2 = -1e9;
        m2 = max(m2, p0 < m1 ? p0 : -1e9); m2 = max(m2, p1 < m1 ? p1 : -1e9); m2 = max(m2, p2 < m1 ? p2 : -1e9);
        m2 = max(m2, p3 < m1 ? p3 : -1e9); m2 = max(m2, p4 < m1 ? p4 : -1e9); m2 = max(m2, p5 < m1 ? p5 : -1e9);
        m2 = max(m2, p6 < m1 ? p6 : -1e9); m2 = max(m2, p7 < m1 ? p7 : -1e9); m2 = max(m2, p8 < m1 ? p8 : -1e9);
        m2 = max(m2, p9 < m1 ? p9 : -1e9);
        edge = 1.0 - smoothstep(0.0, 0.14 * R, m1 - m2);
    }
    if (lod > 0.0 && R > 0.15) {
        vec3 s = q / R + rk.seed * 17.0;
        // (i5) undulating fracture faces: bulges / hollows ~0.5 R across, sd ~2.5 % R (i4: 0.75 % R - the faces were
        // planar, the blocks read as low-poly gems with ruler-straight edges)
        float n1 = texture(uNoise3, s * 0.11).g - 0.487;
        float n2 = R > 0.4 ? texture(uNoise3, s * 0.23 + 0.31).b - 0.525 : 0.0;    // crumbly relief: worley pits, crisp crests
        // small knobs / steps (big blocks): shading normals only (gRkN3), not in the march (2 % of R)
        float n3 = gRkN3 > 0.0 && R > 0.6 ? texture(uNoise3, s * 0.6 + 0.57).b - 0.525 : 0.0;
        d += lod * R * (0.2 * 0.55 * n1 + 0.055 * n2 * (0.25 + 1.45 * edge) + 0.02 * n3);   // (rocks4 R15: n1 0.16 -> 0.2)
    }
    if (lodN > 0.0 && R > 0.12) {
        // crumbly granite at 3-10 cm (close-up): worley cones, pits between sharp crests, ~1.4 cm deep;
        // shapes the silhouettes / contact lines near the camera (the shading adds the finer octaves)
        // (rocks4 R15) larger and sparser along the edges (7-15 cm chipped rims, up to 2.5 cm; r10: an even nibbled rim)
        vec3 s = q + rk.seed * 23.0;
        float m1 = texture(uNoise3, s * 0.7 + 0.29).b;
        // (i5) patchy: pitted / crumbly zones next to smoother ones (i4: evenly pitted all over = sponge)
        float pz = smoothstep(0.38, 0.62, texture(uNoise3, s * 0.37 + 0.71).r);
        d -= lodN * min(0.025, 0.06 * R) * mix(0.12, 1.0, max(pz * 0.5, smoothstep(0.5, 0.8, edge))) * (0.5 + 1.0 * edge) * (smoothstep(0.35, 0.9, m1) - 0.45);
    }
    return d;
}

float sdRock(vec3 p, Rock rk, float lod) {
    return sdRockLocal((p - rk.c) * rk.m, rk, lod) * 0.8;
}

// 0 far .. 1 full relief (< 45 m); + up to 1 more within ~12 m of the camera (near micro relief, sdRockLocal)
float rockLod(float t) { return 1.0 - smoothstep(45.0, 170.0, t) + 1.0 - smoothstep(6.0, 12.0, t); }

RkCtx gRkX[4];
float gRkSmallR = ROCK2_D1 + 1.5;   // small stones are evaluated within this distance of the camera
bool gRkOk[4];
ivec2 gRkBlk = ivec2(-2147483647);
ivec2 gRkSBlk = ivec2(-2147483647);
float gRkSH[4];
// nearest 2x2 cells only; the returned distance is capped so it never jumps into a cell that was not
// tested (rocks reach at most ROCK_OVER past their cell). Cells / slot classes whose reach cannot get
// near p are skipped as a whole (big slots 0-1 reach ROCK_OVER, slots >= 2 ROCK_OVER_S).
// The result is a conservative distance (every rock is at least that far), so a later query close to the
// last full evaluation can reuse it: d(p) >= dLast - |p - pLast| (minus a margin for the lod relief).
// Consecutive march steps in open ground (beach, field between blocks) then skip the cell / slot loops.
vec3 gRkLastP = vec3(1e9);
float gRkLastD = -1.0;
float rocksSDF(vec3 p, float lod, out ivec2 hitCell) {
    float dL = gRkLastD - length(p - gRkLastP) - 0.15;
    if (dL > 0.5) { hitCell = ivec2(-99999); return dL; }
    vec2 g = p.xz / ROCK_CELL;
    ivec2 c0 = ivec2(floor(g));
    vec2 fr = g - vec2(c0);
    ivec2 o = ivec2(fr.x < 0.5 ? -1 : 0, fr.y < 0.5 ? -1 : 0);
    float dmin = min(max(fr.x, 1.0 - fr.x), max(fr.y, 1.0 - fr.y)) * ROCK_CELL - ROCK_OVER;
    hitCell = ivec2(-99999);
    // contexts of the 2x2 cells, cached across calls (consecutive march / shadow steps mostly stay in one block)
    ivec2 blk = c0 + o;
    if (blk != gRkBlk) {
        gRkBlk = blk;
        for (int j = 0; j <= 1; j++)
        for (int i = 0; i <= 1; i++) gRkOk[i + 2 * j] = rkCtx(blk + ivec2(i, j), gRkX[i + 2 * j]);
    }
    for (int j = 0; j <= 1; j++)
    for (int i = 0; i <= 1; i++) {
        ivec2 c = c0 + o + ivec2(i, j);
        if (!gRkOk[i + 2 * j]) continue;
        RkCtx x = gRkX[i + 2 * j];
        float dcell = length(max(abs(p.xz - x.cc) - 0.5 * ROCK_CELL, 0.0));   // p to the cell rectangle
        float dB = dcell - ROCK_OVER, dS = dcell - ROCK_OVER_S;
        if (dB > min(dmin, 0.5)) { dmin = min(dmin, dB); continue; }
        uint hc = rkCellHash(c);
        vec4 hs0 = rkSlotHash(hc, 0);
        for (int k = 0; k < RK_NS; k++) {
            if (k >= x.ns) break;
            if (k == 2 && dS > min(dmin, 0.5)) { dmin = min(dmin, dS); break; }
            vec4 hs = rkSlotHash(hc, k);
            vec2 pos; float R, kind;
            if (!rkSlot(x, x.cc, false, k, hs, hs0, pos, R, kind)) continue;
            float bxz = rkPreBound(p, pos, R, x, x.g);         // bound without fetches
            if (bxz > min(dmin, 0.5)) { dmin = min(dmin, bxz); continue; }
            Rock rk;
            ivec2 ct = rkTagS(c, k);
            if (!rkBuild(ct, hs, pos, R, kind, x, rk)) continue;
            float bound = length(p - rk.c) - ROCK_BOUND * R;
            float d = bound > 0.4 ? bound : sdRock(p, rk, lod);
            if (d < dmin) { dmin = d; hitCell = ct; }
        }
        // jointed-outcrop blocks of steep bare cells (slots 6-7, near the camera only)
        if (x.oc > 0.0) {
            for (int k = RK_OC0; k < RK_OC0 + 2; k++) {
                vec4 hs = rkSlotHash(hc, k);
                vec2 pos; float R, kind;
                if (!rkSlot(x, x.cc, false, k, hs, hs0, pos, R, kind)) continue;
                float bxz = rkPreBound(p, pos, R, x, x.go);
                if (bxz > min(dmin, 0.5)) { dmin = min(dmin, bxz); continue; }
                Rock rk;
                ivec2 ct = rkTagS(c, k);
                if (!rkBuild(ct, hs, pos, R, kind, x, rk)) continue;
                float bound = length(p - rk.c) - ROCK_BOUND * R;
                float d = bound > 0.4 ? bound : sdRock(p, rk, lod);
                if (d < dmin) { dmin = d; hitCell = ct; }
            }
        }
    }
    // small stones (R <= 0.35) near the camera, only for points low enough to touch one
    // (parent-cell context from the cache: the small 2x2 block always lies inside the main 2x2 block)
    RkCtx xp = gRkX[(fr.x < 0.5 ? 1 : 0) + 2 * (fr.y < 0.5 ? 1 : 0)];   // the cell holding p
    float hgE = p.y - (xp.h0 + dot(xp.g, p.xz - xp.cc));
    // (not for refraction rays deeper than 0.3 m under the water: submerged pebbles are left to the bed texture)
    if (length(p.xz - uCamPos.xz) < gRkSmallR && hgE < 1.45 + 0.12 * length(p.xz - xp.cc) && p.y > xp.h0 - xp.rel - 0.3) {
        vec2 g2 = p.xz / ROCK2_CELL;
        ivec2 d0 = ivec2(floor(g2));
        vec2 f2 = g2 - vec2(d0);
        ivec2 o2 = ivec2(f2.x < 0.5 ? -1 : 0, f2.y < 0.5 ? -1 : 0);
        // ground heights at the centres of the 2x2 small cells (cached like the main contexts)
        if (d0 + o2 != gRkSBlk) {
            gRkSBlk = d0 + o2;
            for (int j = 0; j <= 1; j++)
            for (int i = 0; i <= 1; i++) gRkSH[i + 2 * j] = terrainH((vec2(gRkSBlk + ivec2(i, j)) + 0.5) * ROCK2_CELL, false);
        }
        dmin = min(dmin, min(max(f2.x, 1.0 - f2.x), max(f2.y, 1.0 - f2.y)) * ROCK2_CELL - ROCK2_OVER);
        for (int j = 0; j <= 1; j++)
        for (int i = 0; i <= 1; i++) {
            ivec2 c = d0 + o2 + ivec2(i, j);
            vec2 cc = (vec2(c) + 0.5) * ROCK2_CELL;
            float dc2 = length(max(abs(p.xz - cc) - 0.5 * ROCK2_CELL, 0.0)) - ROCK2_OVER;
            if (dc2 > min(dmin, 0.3)) { dmin = min(dmin, dc2); continue; }
            ivec2 pi = clamp(rkParent(c) - (c0 + o), ivec2(0), ivec2(1));
            int ix = pi.x + 2 * pi.y;
            if (!gRkOk[ix]) continue;
            RkCtx px = gRkX[ix];
            rkSmallLocal(px, cc, gRkSH[i + 2 * j]);
            uint hc = rkCellHash(c + ivec2(ROCK2_TAG + 3301, 1777));
            vec4 hs0 = rkSlotHash(hc, 0);
            // beach cobble slots (3-4) only in cells near the camera, for points low enough to touch one (same margin as rkPreBound)
            int n2 = length(cc - uCamPos.xz) < ROCK3_D1 + 2.2 && hgE < 0.9 + 0.12 * length(p.xz - xp.cc) ? px.ns2 : min(px.ns2, 3);
            for (int k = 0; k < RK_NS2; k++) {
                if (k >= n2) break;
                vec4 hs = rkSlotHash(hc, k);
                vec2 pos; float R, kind;
                if (!rkSlot(px, cc, true, k, hs, hs0, pos, R, kind)) continue;
                float bxz = rkPreBound(p, pos, R, px, px.g);
                if (bxz > min(dmin, 0.3)) { dmin = min(dmin, bxz); continue; }
                Rock rk;
                ivec2 ct = rkTagS(c + ivec2(ROCK2_TAG, 0), k);
                if (!rkBuild(ct, hs, pos, R, kind, px, rk)) continue;
                float bound = length(p - rk.c) - ROCK_BOUND * R;
                float d = bound > 0.25 ? bound : sdRock(p, rk, lod);
                if (d < dmin) { dmin = d; hitCell = ct; }
            }
        }
    }
    gRkLastP = p; gRkLastD = dmin;
    return dmin;
}

// ---------------------------------------------------------------------------
// boulder shadow / occlusion helpers (also usable by the ground shading)
// soft sun visibility through the boulders from point p toward l (unit)
float rocksShadow(vec3 p, vec3 l, float k, float lod) {
    float res = 1.0, t = 0.03;
    // small stones cast shadows only for points within 20 m of the camera (they shrink away at 25-35 m)
    gRkSmallR = length(p - uCamPos) < 20.0 ? ROCK2_D1 + 1.5 : 0.0;
    gRkLastD = -1.0;                                        // no reuse across differently gated queries
    ivec2 c;
    float g0 = terrainH(p.xz, false);
    float dcam = length(p - uCamPos);
    int ns = dcam < 40.0 ? 8 : (dcam < 60.0 ? 6 : 3);      // far points: shorter, coarser shadow march
    for (int i = 0; i < 8; i++) {
        if (i >= ns) break;
        vec3 q = p + l * t;
        if (q.y - g0 > 4.4) break;                          // above any boulder top
        float d = rocksSDF(q, i < 1 ? lod : 0.0, c);
        res = min(res, k * d / t);
        if (res < 0.02) { res = 0.0; break; }
        t += clamp(d, 0.08, 2.0);                           // d is a safe bound: long steps away from rocks
        if (t > 6.0) break;
    }
    gRkSmallR = ROCK2_D1 + 1.5;
    gRkLastD = -1.0;
    return saturate(res);
}
// sky occlusion of a ground point by nearby boulders (cheap: 1 SDF eval)
// (rocks3b) two probes: 8 cm up (the dark contact line at a block's foot: near-black where the
// block meets the ground) and 0.5 m up (the broad sky occlusion beside a block); the SDF is Lipschitz-scaled x0.8
float rocksGroundAO(vec3 p) {
    ivec2 c;
    gRkLastD = -1.0;                                        // needs the actual distance, not a reused bound
    float d1 = rocksSDF(p + vec3(0.0, 0.08, 0.0), 0.0, c) * 1.25;
    if (d1 > 1.6) { gRkLastD = -1.0; return 1.0; }          // nothing within reach of either probe
    gRkLastD = -1.0;
    float d2 = rocksSDF(p + vec3(0.0, 0.5, 0.0), 0.0, c) * 1.25;
    gRkLastD = -1.0;
    float a = mix(0.38, 1.0, smoothstep(0.0, 0.2, d1)) * mix(0.6, 1.0, smoothstep(0.0, 1.0, d2));
    return a;
}

// ---------------------------------------------------------------------------
// lighting.glsl : frame uniforms, hit kinds, shadows, sky/ambient light, BRDF helpers, aerial perspective
// ---------------------------------------------------------------------------










layout(location = 0) out vec4 fragColor;
// lab-only runtime knobs for look-dev sweeps without a recompile: set with the environment variable
// ALPINE_LAB="i=v,i=v" (render.py); merged engine code must not read them (bake the chosen values in)

// (r4 L3) direction of the shadow sun (render.py Look.shadow_dir(): = uSunDir unless Look.shadow_elevation is set): the
// near terrain march and the tree shadows use it like the baked terrain shadow maps; shading keeps uSunDir


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

// ---------------------------------------------------------------------------
// falls.glsl : the twin waterfall - a ~29 m step of dark wet, mossy rock set into the
// forested east-shore hillside. The main fall (A) slides and bounces down a chute in the face and
// fans out; the second (B) leaps from a cleft half way up. Water = a thin volumetric sheet that
// follows a path simulated on the baked height map (features.py -> uFallK knots), white and lacy,
// with spray at every ledge impact, mist at the base and a foam plume on the pool.
// The first part (shape) is shared with bake_features.frag, which includes this file with
// FALLS_SHAPE_ONLY defined, so the carved rock and the shading masks always agree.
// ---------------------------------------------------------------------------
          // cliff foot x,z, lip height of the main fall (m), along-shore half width
       // unit: from the cliff out over the lake
     // along-shore offsets of the two lips (vA, vB), lip width, lip notch depth

#define WF_LIPB 0.50       // fall B leaves its cleft at this fraction of the step height
#define WF_DRIFT_A 11.0    // fall A drifts this far along the shore (+v) over its whole drop (m, session 3)

// local frame: (u inland, v along the shore) ; +v = south for the default placement
vec2 wfUV(vec2 xz) {
    vec2 o = xz - uWF.xy;
    return vec2(-dot(o, uWFDir), dot(o, vec2(-uWFDir.y, uWFDir.x)));
}

// 1 across the bare rock cliff of the two falls, 0 where the step turns into steep forested banks
// that frame the falls on both sides (the falls sit in a shallow rock alcove)
float wfCliff(float v) { return smoothstep(-15.0, -9.5, v) * (1.0 - smoothstep(18.0, 30.0, v)); }

// inland position of the foot of the rock face at along-shore position v
float wfFaceLine(float v) {
    float n = 1.3 * gnoise(vec2(v * 0.085, 3.1)) + 0.6 * gnoise(vec2(v * 0.29, 7.3));
    // north of the falls the rocky bank swings out into the lake (the shore rocks)
    float north = -20.0 * smoothstep(-12.0, -40.0, v) * (1.0 - smoothstep(-55.0, -90.0, v));
    // south the forested bank juts out into the lake, closing the alcove
    float south = -7.0 * smoothstep(20.0, 48.0, v) * (1.0 - smoothstep(70.0, 120.0, v));
    // the dark buttress north of fall A's foot stands out into the pool
    float butt = -3.5 * smoothstep(-20.0, -14.0, v) * (1.0 - smoothstep(-6.0, -2.5, v));
    return 1.5 + n + north + south + butt;
}

// the buttress just north of fall A (the darkest, most massive rock, standing in the water left
// of the falls): bare, near-vertical blocky rock up to ~0.4 of the step height, ferns / shrubs above
float wfButt(float v) { return smoothstep(-22.0, -16.0, v) * (1.0 - smoothstep(-5.5, -2.0, v)); }
float wfButtH(float v, float hs) { return hs * (0.36 + 0.06 * gnoise(vec2(v * 0.13, 8.3))); }
#define WF_BUTT_W 1.6      // horizontal depth of the buttress wall (m)

// ledges of the two chutes (heights above the lake, m) where the water lands and bursts: fall A at
// 0.62 and 0.30 of its lip height, fall B at 0.45 (features.py LEDGES must agree)

#define WF_BENCH 1.0       // ledge width (m), with a low WF_LIP reverse lip at the outer edge
#define WF_LIP 0.15
// stepped chute profile: height at inland distance x (0 = foot, Wc = lip) - steep risers between
// ledges WF_BENCH wide with a 0.4 m reverse lip at their outer edge
// k1 widens the first riser (fall A: a ~60 deg slab the water slides down), the lower ones steepen

#define WF_CUT_A 2.5       // fall A's chute is cut back this far into the rim (recessed lip)
#define WF_SLAB_A 1.8      // fall A's first riser is this much wider than its share (a slab, ~60 deg)
#define WF_CUT_B 0.5       // ... and fall B's cleft is this much deeper

// height of the top of the rock step
float wfStepH(float v) {
    float H = uWF.z;
    float n = 0.08 * gnoise(vec2(v * 0.07, 4.0)) + 0.05 * gnoise(vec2(v * 0.23, 1.0));
    float north = mix(1.0, 0.5, smoothstep(-12.0, -45.0, v)) * (1.0 - smoothstep(-65.0, -100.0, v));
    float south = 1.0 - 0.8 * smoothstep(45.0, 110.0, v) - 0.2 * smoothstep(95.0, 140.0, v);
    return H * (v < 0.0 ? north : south) * (1.0 + n);
}

// width of the low rock apron in front of the north bank (the shore rocks)
float wfApronW(float v) { return 3.0 + 11.0 * smoothstep(-14.0, -40.0, v); }
// where the apron runs along the shore (open pool water at the buttress foot, south of it)
float wfApronZ(float v) { return smoothstep(-16.0, -24.0, v) * (1.0 - smoothstep(-62.0, -82.0, v)); }

// horizontal depth of the cliff face (foot to rim): ~70 deg on average
float wfFaceW(float hs) { return 2.0 + 0.30 * hs; }
// ... and of the step anywhere along the shore: the cliff, or a ~42 deg forested bank to the sides
// (within ~25 m of the cliff's ends the banks are steeper, ~55 deg: no tall bare flank beside the cliff)
float wfFaceWv(float hs, float v) {
    float nearC = 1.0 - smoothstep(0.0, 25.0, max(-12.0 - v, v - 24.0));
    return mix(2.5 + hs / mix(0.9, 1.5, nearC), wfFaceW(hs), wfCliff(v));
}

// chute axis (along-shore position) of fall k at inland position u, and its half width
float wfChuteV(int k, float u) {
    if (k == 1) return uWFNotch.y;
    float W = wfFaceW(uWF.z);                                   // (features.py chute_v)
    // (session 3) a straight diagonal down the whole face (fall A slides diagonally down the slab to
    // the pool), soft only at the ends; was smoothstep(0, 0.62): an S-bend, then a vertical drop
    float s = saturate((1.5 + W - u) / W);
    return uWFNotch.x + WF_DRIFT_A * (0.75 * s + 0.25 * s * s * (3.0 - 2.0 * s));
}
// (wider than the water fan + ledge widening: the sheet's lacy edges stay in view, not cut by the rock)
float wfChuteHW(int k, float u) {
    float W = wfFaceW(uWF.z);                                   // (features.py chute_hw)
    float f = saturate((1.5 + W - u) / W);
    return k == 0 ? mix(2.4, 5.8, f) : mix(2.6, 6.0, f);   // (review r2: B fans out wider, features.py FAN)
}

// voronoi for the rock joints: (offset of x from its cell point, F2 - F1, cell id)
vec4 wfVor(vec2 x) {
    vec2 i = floor(x); vec2 f = x - i; ivec2 c = ivec2(i);
    float d1 = 8.0, d2 = 8.0; vec2 o1 = vec2(0.0); float id = 0.0;
    for (int y = -1; y <= 1; y++)
    for (int xx = -1; xx <= 1; xx++) {
        ivec2 o = ivec2(xx, y);
        vec2 r = vec2(o) + hash22i(c + o) * 0.8 + 0.1 - f;
        float dd = dot(r, r);
        if (dd < d1) { d2 = d1; d1 = dd; o1 = r; id = hash12i(c + o + ivec2(17, 31)); }
        else if (dd < d2) d2 = dd;
    }
    return vec4(-o1, sqrt(d2) - sqrt(d1), id);
}

// angular rock slabs on the low apron north of the falls (the shore rocks):
// world-space cells ~3.3 m; returns the slab height above the shelf (0 in the joints / where none)
// and the cell (offset, edge distance, id) for the shading
float wfSlab(vec2 p, float dp, float aw, out vec4 sv) {
    sv = wfVor(p * 0.30 + 4.0);
    vec4 h = hash42i(ivec2(int(sv.w * 65536.0), 5));
    if (h.x < 0.22) return 0.0;
    float top = 0.06 + 0.18 * h.y + 0.8 * h.y * h.y * (1.0 - smoothstep(2.0, 5.0, dp));
    top += dot(sv.xy, (h.zw - 0.5)) * 0.3;
    return max(top, 0.0) * smoothstep(0.02, 0.12, sv.z);
}

// face coordinates (along shore, height) of point (v, y), warped so no joint is ruler-straight; B / hB =
// the big structure (~12 x 15 m buttresses / recessed slabs) and its hash
vec2 wfFaceC(float v, float y, out vec4 B, out vec4 hB) {
    vec2 fc = vec2(v, y);
    fc += vec2(gnoise(fc * 0.017 + 3.7), gnoise(fc * 0.017 + 9.1)) * 1.8;
    B = wfVor(vec2(fc.x * 0.08, fc.y * 0.065) + 21.0);
    hB = hash42i(ivec2(int(B.w * 65536.0), 3));
    return fc;
}

// irregular jointed rock (chunky, rounded-angular blocks, no masonry rows): anisotropic voronoi
// blocks of size ~S (m) in face coordinates turned to a slowly varying bedding dip (10-45 deg), joints
// wobbly; each block a rounded 'pillow' standing out by its own amount and tilted into its own facet.
// Returns (relief m outward, edge distance in cell units (0 at a joint .. ~0.6), block id, block hash)

// foliage cover of the rock face 0..1 (about half of the cliff lies under continuous masses of
// shrubs, ferns and moss, >= ~3 m across, densest over the upper ~30 % of the face and north of fall A,
// with dark rock between). Built only from ~10 m and ~5 m noise in face coordinates fc = (v, y).
// Shared by the bake (cushion relief, bake_features.frag) and the shading (wfRockShade mossW).
float wfVegRaw(vec2 fc, float hs) {
    float vN = gnoise(fc * vec2(0.085, 0.10) + 1.3);              // ~10 m
    float vN2 = gnoise(fc * vec2(0.17, 0.21) + 7.7);              // ~5 m
    float hy = fc.y / max(hs, 5.0);
    return 0.42 + 0.95 * vN + 0.45 * vN2 + 0.42 * smoothstep(0.60, 0.85, hy) + 0.22 * smoothstep(-8.0, -24.0, fc.x)
         - 0.30 * (1.0 - smoothstep(0.10, 0.30, hy));
}

// bush / fern cushions growing out of the face where it is covered (bake relief, m outward): domes
// ~2-3 m across, 0.6-1.8 m proud, ~70 % of the cells

// shrub masses on the rim shelf behind the bare face (bake relief, m up): clumps 2.5-4 m across, 1.2-3 m
// tall, so the cliff's outline against the forest is ragged, not a bald rounded edge


#define FF_MAX_DIST 380.0     // the falls flora (3D plants / blocks) is traced up to this distance
// falls lab timing toggles (uDebug 70 + part, 74 = all): 0 flora, 1 water volume, 2 rock / bank / apron
// shading, 3 stream (runtime only: same program, so the share of each part's work can be measured)
bool wfOff(int part) {
    // compile-time ablation (falls lab timing only: FALLS_ABL = bit mask of the parts compiled out)
    return uDebug == 70 + part || uDebug == 74;
}
// water path knots (features.py simulates the flow on the baked map): per fall WF_NK knots evenly
// spaced in height from the lip (k=0) down to the pool; each = (u, v, half width, impact 0..1)
#define WF_NK 24

   // lip height, relative volume, impact height (pool), unused
    // per fall k: [2k] = (u, y, v) min, [2k+1] = max of its water + mist (render.py)

vec3 fallLocal(vec3 p) {
    vec2 uv = wfUV(p.xz);
    return vec3(uv.x, p.y, uv.y);   // (u inland, y, v along shore)
}

// knot of fall k at height y, Catmull-Rom through the knots (no kinks): (u, v, half width, impact);
// dudy = slope of the sheet (du/dy) there
vec4 fallKnot(int k, float y, out float dudy) {
    float top = uFallInfo[k].x;
    float sc = float(WF_NK - 1) / (top + 0.6);
    float s = saturate((top - y) / (top + 0.6)) * float(WF_NK - 1);
    int i1 = min(int(s), WF_NK - 2);
    float f = s - float(i1);
    int b = k * WF_NK;
    vec4 p0 = uFallK[b + max(i1 - 1, 0)], p1 = uFallK[b + i1], p2 = uFallK[b + i1 + 1], p3 = uFallK[b + min(i1 + 2, WF_NK - 1)];
    vec4 c1 = 0.5 * (p2 - p0), c2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3, c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2);
    vec4 r = p1 + f * (c1 + f * (c2 + f * c3));
    dudy = -(c1.x + f * (2.0 * c2.x + 3.0 * f * c3.x)) * sc;
    r.z = max(r.z, 0.3);
    r.w = mix(p1.w, p2.w, f);
    return r;
}

float wfRemap(float x) { return saturate(0.5 + (x - 0.5) * 3.0); }   // noise channels have std ~0.14
float wfSoftAbs(float x, float w) { return sqrt(x * x + w * w); }   // |x| rounded over w (anti-aliased crease)

// droplet streaks (round 2): falling white water breaks into countless short white streaks
// (motion-blurred droplet trains, 3:1 .. 8:1) - one streak per cell of (across, along the flow) space
// (cell = size in m / flow-phase units), random width / length / brightness / presence (pres = share of
// the cells with a streak). Columns are staggered along the flow (no grid rows). Returns 0..1.
// Mean over a cell ~0.25 x pres x 0.8 (wfStreakMean).
float wfStreak(vec2 q, vec2 cell, float pres, int seed) {
    vec2 g = q / cell;
    float cx = floor(g.x);
    g.y += hash12i(ivec2(int(cx), seed)) * 7.0;
    vec2 c = floor(g);
    vec2 f = fract(g) - 0.5;
    vec4 h = hash42i(ivec2(c) + ivec2(seed * 7, 977));
    float wx = 0.22 + 0.22 * h.y;
    float x = (f.x - (h.x - 0.5) * (0.92 - 2.0 * wx)) / wx;
    float y = f.y / (0.42 + 0.08 * h.z);
    float sx = (1.0 - smoothstep(0.35, 1.0, abs(x))) * (1.0 - smoothstep(0.2, 1.0, abs(y)));
    return sx * step(h.w, pres) * mix(0.6, 1.0, fract(h.w * 13.7 + h.z));
}
float wfStreakMean(float pres) { return 0.25 * pres * 0.8; }
float gCoreD = 0.0;    // (round 3) fallDensity: distance to the sheet's dense core (0 inside it)
float gFFar = 0.0;     // (r4 C10) 0 near .. 1 beyond ~500 m (set per fall by fallsVolume): a thin soft thread far away

// density of falling water (per m) at local point q (u, y, v) of fall k. ld = ray direction (local
// frame): the optical depth is normalised to a face-on view so oblique views do not saturate.
// foamy = white aerated cover 0..1; dist = a safe step to the water's band when outside it
float fallDensity(int k, vec3 q, vec3 ld, float foot, out float foamy, out float dist) {
    foamy = 0.0;
    float top = uFallInfo[k].x;
    dist = max(q.y - top - 0.6, -0.6 - q.y);
    if (dist > 0.0) return 0.0;
    float dudy;
    vec4 kn = fallKnot(k, q.y, dudy);
    float drop = max(top - q.y, 0.0);
    float frac = drop / max(top, 1.0);                 // 0 at the lip .. 1 at the pool
    // fall B spills from its cleft: its top fades in over 1.5 m
    float w = kn.z * (k == 1 ? mix(0.6, 1.0, smoothstep(0.0, 1.5, drop)) : 1.0);
    // (round 3 i8) fall A comes over the lip as a narrow sheet hugging the rock and spreads only lower down
    // (a ribbon ~1/4 of its foot's width at the top; the r3 curtain was as broad at the lip as at the
    // pool) - the wet black chute rock shows either side of it
    if (k == 0) w *= mix(0.45, 1.0, smoothstep(0.05, 0.85, frac));
    // (review rv2) both falls ~0.6x narrower toward the pool (fall A ~0.2 x its height wide at the foot, B ~0.3;
    // the render fanned out to ~0.45 / 0.55 and the two merged into one white wall at the base) - dark wet rock
    // between them all the way down
    // (review rv7) fall B fans out from its narrow spill toward the pool (a lacy curtain widening over its steps)
    w *= k == 0 ? mix(1.0, 0.62, smoothstep(0.1, 0.9, frac)) : mix(0.55, 0.8, smoothstep(0.15, 1.0, frac));
    float dv = q.z - kn.y;
    // (the pool impact: a splash, not a white puff over the last metres)
    float imp = kn.w * mix(0.6, 1.0, smoothstep(0.3, 2.0, q.y));
    // (a thin sheet: a thick one viewed obliquely smeared the strands across)
    float th = 0.12 + 0.012 * drop + 0.8 * imp;
    float du = kn.x - q.x;                 // > 0 : outward of the water's back (rock side)
    float csl = inversesqrt(1.0 + dudy * dudy);
    // (session 4: + the spray veil out to 1.9 half widths / 1.1 m in front of the sheet)
    dist = max(max(abs(dv) - (w * 1.9 + 0.8), 0.0) * 0.65,
               max(-0.5 - du, du - (th * 2.0 + 1.1 + 1.8 * imp + 1.5 * kn.w * (1.0 - smoothstep(1.5, 3.5, q.y)))) * csl);   // (rv1: + the wider boil)
    if (dist > 0.0) return 0.0;
    dist = 0.0;
    // (round 3) distance to the sheet's core (the dense water, |du| ~ th): outside it, inside the band, only the
    // spray / veil / burst plume (smooth or granular, ~0.15 m+) - the march strides there (fallsVolume): the
    // band is up to ~5 m thick at the ledge impacts and the dtMin steps through it ran out of budget (horizontal
    // lines across fall B at its ledge in the tele view)
    gCoreD = max(max(du - (2.5 * th + 0.3), -0.6 - du) * csl, max(abs(dv) - w * 1.3, 0.0) * 0.65);
    // flow phase: (time the water has fallen since the lip - uTime) x 4 m/s. Speed v = 2.5 + 3.54 sqrt(drop),
    // capped at 9 m/s (the sheet slides on the rock), integrated in closed form: the streaks stretch as
    // the water accelerates and do not shear with time (q.y + uTime * v(q.y) compressed the noise
    // vertically ~uTime-fold -> horizontal hatching)
    float Sd = sqrt(min(drop, 3.37));
    float tfl = 0.565 * (Sd - 0.706 * log(1.0 + 1.416 * Sd)) + max(drop - 3.37, 0.0) / 9.0;
    float ty = (tfl - uTime) * 4.0;
    float ex = dv / w;
    float fk = float(k);
    // lacy white water - many fine strands (a few cm .. 0.25 m wide, long along the flow) whose
    // density comes and goes in packets and tears (holes where the dark rock shows through), grey and
    // translucent where thin, a denser core; the edges fray into spray. 5 octaves in (across, along the
    // flow) coordinates; ty advects with the water, so packets and strands fall at the flow speed.
    // The strands spread part of the way with the fan (no rigid comb of lines down the fan).
    // (the strand pattern is sheared along the ray through the sheet's thickness: every ray sees one
    // strand coordinate across the whole depth - an oblique view no longer averages ~0.6 m of strands
    // into a smooth blur; the pattern slides only by du x the change of the view angle)
    float aZ = clamp(ld.z / (abs(ld.x) + 0.05) * sign(ld.x), -1.8, 1.8);
    float aY = clamp(ld.y / (abs(ld.x) + 0.05) * sign(ld.x), -1.8, 1.8);
    float duS = clamp(du, -0.4, 0.8);
    // (round 3) the strands spread with the fall's mean fan (lip -> pool width, linear in the drop), not with the
    // local width: the knots' pinch / spread at each ledge bent the strands into S-curves (fall B)
    float wS = mix(uFallK[k * WF_NK].z, uFallK[k * WF_NK + WF_NK - 1].z, frac);
    float sx = (dv + duS * aZ) * inversesqrt(max(wS, 0.6) / 1.3) + fk * 3.7;
    ty -= duS * aY * 4.0 / min(2.5 + 3.54 * sqrt(drop), 9.0);
    // footprint visibility of each octave: unresolved -> its mean, its variance moves into the
    // threshold width (sl below), so the mean cover / brightness is the same near and far. (round 2: a
    // strand stays until it is ~1 px wide - box-filtered, a 1 px white streak still shows; fading at
    // 0.4 px smoothed the r4 / film views into blurred ribbons)
    float vH = 1.0 - smoothstep(0.7, 1.8, foot);
    float vP = 1.0 - smoothstep(0.16, 0.45, foot);
    float vR = 1.0 - smoothstep(0.08, 0.22, foot);
    float vS = 1.0 - smoothstep(0.034, 0.088, foot);
    float vF = 1.0 - smoothstep(0.02, 0.05, foot);
    // tears ~1.3 x 6 m, packets ~0.5 x 1.6 m, ropes ~0.22 x 1.4 m, strands ~0.09 x 0.7 m (clumpy, broken:
    // worley-fbm), striations ~3.5 cm x 0.6 m
    // (session 4: every octave ~2.3x shorter along the flow - the 15-45:1 fibres read as draped satin; the
    // white water is packets / lumps of ~3-6:1)
    float zH = (texture(uNoise3, vec3(sx * 0.048, ty * 0.018, 0.13 + fk * 0.31)).g - 0.487) * 6.5;
    vec4 tP = texture(uNoise3, vec3(sx * 0.125, ty * 0.035, du * 0.03 + 0.71 + fk * 0.13));
    float zP = (tP.a - 0.521) * 7.9;
    // (review rv1) every thread octave ~3x longer along the flow (kStr): the threads run dead straight down the flow
    // - their meander every 0.5-2 m read as combed / brushed hair; the along-flow variety now comes from
    // the falling clumps below (an amplitude modulation, the threads' position does not wander)
    float kStr = 0.35;
    float zR = (texture(uNoise3, vec3(sx * 0.80, ty * 0.26 * kStr, 0.29 + fk * 0.4)).r - 0.475) * 7.1;
    // (round 3 i8: strands / striations ~1.6x longer along the flow - straight threads, not wavy brush strokes)
    vec4 tS = texture(uNoise3, vec3(sx * 2.0, ty * 0.15 * kStr, du * 0.04 + 0.37));
    float zS = (tS.b - 0.525) * 8.0;
    float zF = vF > 0.0 ? (texture(uNoise3, vec3(sx * 3.6 + 0.3, ty * 0.20 * kStr, du * 0.08 + 0.9)).r - 0.475) * 7.1 : 0.0;
    // macro water amount (tears + packets + where the sheet is dense) -> the share of the strands that
    // carry water; the strands themselves are a crisp mask (fine white lines with grey
    // translucent gaps), their threshold widened by the variance the footprint filters away
    const float kMac = 0.4;                  // (round 3: macro tears / packets x0.4: one coherent sheet)
    float kBrk = 0.20;         // (round 3: breaks x0.3; i8 x0.15 - the packets read as horizontal smear bands; r4 W7: 0.06 -> 0.20, tears / packets back)
    float aH = 0.55 * kMac, aP = 0.50 * kMac;
    const float aR = 0.60, aS = 0.65, aF = 0.30;
    // (review r2, fix 5) fall B is an even lacy curtain fanning out over its steps, not a ribbon: flatter
    // lateral falloff, a little more water
    float lev = 0.75 - 0.25 * smoothstep(0.0, 0.45, frac) + 0.25 * smoothstep(0.75, 1.0, frac)
              - (k == 1 ? 0.6 : 1.6) * ex * ex + 0.45 * imp + (k == 1 ? -0.07 : 0.0)   // (round 3: B 0.9 -> 0.6, a broader curtain; rv7: B lacier -0.15)
              - 0.22 * smoothstep(0.08, 0.3, frac) * (1.0 - smoothstep(0.8, 0.95, frac))   // (session 3: more tears / holes mid-fall)
              + 0.3;
    float slM = sqrt(aH * aH * (1.0 - vH * vH) + aP * aP * (1.0 - vP * vP));
    float cm = smoothstep(-0.55 - slM, 0.55 + slM, aH * zH * vH + aP * zP * vP + lev);
    float S = aR * zR * vR + aS * zS * vS + aF * zF * vF;
    float slS = sqrt(aR * aR * (1.0 - vR * vR) + aS * aS * (1.0 - vS * vS) + aF * aF * (1.0 - vF * vF));
    float thS = mix(1.25, -1.0, cm);
    float cover = smoothstep(thS - 0.22 - 0.7 * slS, thS + 0.22 + 0.7 * slS, S);
    // (session 3) the strands break up along the flow into droplet trains (~0.3-0.8 m) with gaps between
    // them: continuous strands read as draped satin; unresolved -> the mean (x0.78)
    // (review r2, fix 4) its own visibility gate: the droplet trains are 0.3-0.8 m long, resolved long
    // after the strand width (vS ~0 at 640-1032 px left brk at its constant mean: smooth along the flow)
    float vB = 1.0 - smoothstep(0.12, 0.35, foot);
    float brkN = texture(uNoise3, vec3(sx * 0.85 + 0.3, ty * 0.38, 0.77 + fk * 0.2)).r;
    float brk = mix(0.78, smoothstep(0.36, 0.56, brkN + 0.25 * (cm - 0.5)), vB);
    cover *= mix(1.0, brk, kBrk);
    // lace edges: frayed, the strands at the sides end at different widths
    float edgeN = 0.45 * zP * vP + 0.3 * zR * vR + 0.3 * zH * vH;
    float feL = 2.0 * foot / max(w, 0.5) * gFFar;        // (r4 C10: edges feathered over >= 2 px far away)
    float lat = 1.0 - smoothstep(0.55 + 0.12 * edgeN - feL, 1.05 + 0.12 * edgeN + feL, abs(ex));
    float prof = exp(-pow(max(du, 0.0) / th, 2.0)) * smoothstep(-0.5, 0.0, du);
    float vol = uFallInfo[k].y;
    float cs = max(abs(dot(ld, normalize(vec3(-1.0, dudy, 0.0)))), 0.25);
    // optical depth across the sheet: a strand alone is translucent (~0.4), the closed core ~1.8 (the
    // veil shows the dark rock through it nearly everywhere but the core)
    float sb = 0.45 + 0.55 * smoothstep(-1.2, 1.2, 0.8 * zR * vR + 0.6 * zP * vP + 0.3);
    // droplet streaks: three sizes (~28 cm x 2.6 m flow phase: streak bundles; ~11 cm x 1.7; ~5 cm x 0.9),
    // staggered; the share of the cells holding a streak follows the sheet's amount (dense core: nearly
    // all; lace: a few). Each size fades to its mean once its streak is ~1 px wide.
    float pres = saturate(0.15 + 0.7 * cm + 0.25 * cover);
    vec2 qs = vec2(sx, ty);
    const float sw0 = 0.19, sw1 = 0.08, sw2 = 0.038;     // (session 3: x0.68, finer strands)
    float v0 = 1.0 - smoothstep(0.5, 1.2, foot / (0.66 * sw0));
    float v1 = 1.0 - smoothstep(0.5, 1.2, foot / (0.66 * sw1));
    float v2 = 1.0 - smoothstep(0.5, 1.2, foot / (0.66 * sw2));
    float mS = wfStreakMean(pres);
    float st0 = v0 > 0.0 ? mix(mS, wfStreak(qs + vec2(0.11, 1.3), vec2(sw0, 0.65), pres, 5 + k), v0) : mS;
    float st1 = v1 > 0.0 ? mix(mS, wfStreak(qs, vec2(sw1, 0.32), pres, 11 + k), v1) : mS;
    float st2 = v2 > 0.0 ? mix(mS, wfStreak(qs + vec2(0.03, 0.4), vec2(sw2, 0.16), pres, 23 + k), v2) : mS;
    float stk = max(max(st0, st1), st2) + 0.3 * (st0 + st1 + st2 - max(max(st0, st1), st2));
    float stkM = mS * 1.6;                                 // ~ its mean (unresolved)
    // the sheet is a veil of these streaks: thin (translucent, the dark rock shows) between them, the
    // streaks themselves white; the share of the streaks follows the sheet's amount (pres)
    float Fn = saturate(stk * 1.25);
    float stri = saturate(0.5 + (stk - stkM) * 1.4);
    // (session 4) a milky base between the strands where the sheet carries water (even the darkest
    // tenth of the fall's pixels is light grey, p10 ~144 - the w11 / s3c gaps showed the black rock, p10 ~90,
    // and the sheet read as dark-folded satin); the dark rock shows only in the tears and at the frayed edges
    // (review r2) the milk breaks up with the same droplet trains (x0.45-1.55): no uniform satin between them
    // (r4 W7: 0.08 + 0.42 -> 0.05 + 0.30, grey-white lace, not silk)
    float milk = (0.05 + 0.30 * cm * cm) * mix(1.0, mix(0.6, 1.4, smoothstep(0.3, 0.7, brkN)), vB * kBrk);
    // (round 3 i8) translucent toward the frayed edges and over the upper part (the dark rock shows
    // between the threads there; only the lower core is milky-white)
    milk *= mix(1.0, 0.12, smoothstep(0.35, 0.95, abs(ex))) * mix(0.55, 1.0, smoothstep(0.1, 0.6, frac));
    // (review rv1) a milky-white body in the lower core (below mid-fall the sheet is a soft bright mass with
    // streaks in it, not threads over black gaps)
    milk += 0.10 * cm * smoothstep(0.2, 0.7, frac) * (1.0 - smoothstep(0.45, 0.9, abs(ex)));   // (r4 W7: 0.18 -> 0.10)
    float dens = lat * prof * (milk + cover * sb * (0.05 + 2.8 * Fn) + 1.0 * stk * cm * (1.0 - cover)) * vol / max(th, 0.3) * cs;
    // (r3) packets: the falling water comes in lumps ~0.5 m across x 0.7 m along the flow (the veil is
    // lumpy and broken along the flow, not satin streaks); mean 1, unresolved -> 1
    float pkN = texture(uNoise3, vec3(sx * 0.25 + 0.5, ty * 0.18, 0.53 + fk * 0.17)).r;
    float pkS = smoothstep(0.35, 0.65, pkN + 0.15 * (brkN - 0.5));
    dens *= mix(1.0, 0.6 + 0.8 * pkS, vB * kBrk);
    // (r3c) coarse packets ~1 m across x 1.6 m along, resolved out to a ~0.8 m footprint (r4 / film shot 2)
    float vK = 1.0 - smoothstep(0.3, 0.8, foot);
    float pk2 = smoothstep(0.3, 0.7, texture(uNoise3, vec3(sx * 0.12 + 0.2, ty * 0.08, 0.91 + fk * 0.23)).r);
    dens *= mix(1.0, 0.75 + 0.5 * pk2, vK * kBrk);
    // spray halo: the edges and the face of the sheet shed a fine, thin, sparkling veil of droplets
    float spk = mix(0.35, smoothstep(0.55, 0.8, tS.b), vS);
    float halo = smoothstep(0.35, 0.9, abs(ex)) * (1.0 - smoothstep(1.0, 1.55, abs(ex) - 0.1 * edgeN))
               * exp(-max(du, 0.0) / 0.45) * smoothstep(-0.3, 0.1, du) * smoothstep(0.08, 0.35, frac);
    dens += halo * spk * 0.30 * vol * (1.0 - gFFar);     // (r4 C10: far away the plume is the channel's width)
    // (session 4) spray veil: a thin glowing envelope of fine droplets around the lower two thirds of the
    // sheet, wider than the water and up to ~1 m in front of it (the fall glows, its outline is
    // soft, the rock beside it is seen through a bright haze); fine-grained, falling with the water
    float veilN = mix(0.5, texture(uNoise3, vec3(sx * 1.3 + 0.7, ty * 0.21 * kStr, du * 0.5 + 0.15)).g, vR);
    float veil = (1.0 - smoothstep(0.9, 1.9, abs(ex) - 0.15 * edgeN)) * exp(-max(du, 0.0) / 0.55) * smoothstep(-0.4, 0.15, du)
               * smoothstep(0.15, 0.6, frac) * (0.4 + 0.6 * frac) * smoothstep(0.25, 0.75, veilN + 0.15);
    dens += veil * 0.17 * (2.6) * vol * (1.0 - gFFar);      // (round 3: 0.085 -> 0.17, a luminous veil; review rv1/rv3 x2.6: the glow)
    // spray plume bursting off each ledge impact: a looser, fine-grained cloud thrown out in front of
    // the sheet (~1-2 m), droplets streaking down (small white plumes where the water hits)
    float burst = imp * exp(-max(du - 0.1, 0.0) / (0.35 + 0.9 * imp)) * smoothstep(-0.4, 0.2, du)
                * (1.0 - smoothstep(0.7, 1.35, abs(ex) - 0.15 * edgeN))
                * smoothstep(-0.6, 0.9, 0.7 * zP * vP + 0.5 * mix(0.0, (tS.b - 0.525) * 8.0, vS) + 0.3);
    // (review r2, fix 4) granular: droplet clusters of 0.1-0.2 m (worley cells falling with the water), not
    // a smooth puff; unresolved -> their mean
    float vG = 1.0 - smoothstep(0.05, 0.16, foot);
    // (.a = worley 1 / 16 per unit: sx * 0.45 -> ~0.14 m cells across, ty * 0.4 -> ~0.16 m along)
    float gr = vG > 0.0 ? texture(uNoise3, vec3(sx * 0.45 + 0.13, ty * 0.4, du * 0.4 + 0.41 + fk * 0.3)).a : 0.5;
    burst *= mix(1.0, 2.2 * smoothstep(0.62, 0.38, gr), vG);
    dens += burst * 0.50 * vol;     // (session 3: 0.55 -> 0.3: puffs / skirts; review r2: granular, 0.5)
    // (review r2, fix 6) a granular white boil 0.3-1 m tall where the sheet hits the pool (the impact reads
    // as churning water and spray, not a flat foam sheet squashed by the grazing view)
    // (review rv1) kB: a taller, wider churn (a white boil at the foot, the curtain does not end in a line)
    float kB = 1.6;
    float boilV = smoothstep(1.4 * kB, 0.2, q.y) * smoothstep(-0.45, 0.05, q.y) * kn.w * smoothstep(0.75, 1.0, frac)
                * exp(-max(du + 0.2, 0.0) / ((0.9 + 0.5 * vol) * kB)) * smoothstep(-0.6, 0.0, du)
                * (1.0 - smoothstep(0.75, 1.25, abs(ex) - 0.2 * edgeN));
    if (boilV > 0.002) {
        float bg_ = vG > 0.0 ? texture(uNoise3, vec3(q.z * 0.3 + fk * 0.37, (q.y - uTime * 1.1) * 0.28, q.x * 0.3 + 0.2)).a : 0.5;
        float bh = q.y / ((0.35 + 0.65 * mix(0.5, texture(uNoise3, vec3(q.z * 0.09, uTime * 0.05, q.x * 0.09 + 0.6)).g, 1.0)) * kB);
        boilV *= mix(1.0, 2.0 * smoothstep(0.66, 0.40, bg_), vG) * (1.0 - smoothstep(0.6, 1.2, bh));
        dens += boilV * 1.6 * vol * (1.0 - 0.5 * gFFar);
    }
    foamy = max(max(cover * sb * mix(0.5, 1.15, stri), 0.8 * imp), boilV > 0.002 ? smoothstep(0.0, 0.3, boilV) : 0.0);
    // (r3) brightness along the streaks: sunlit droplet clusters and duller water (an along-flow sparkle)
    foamy *= mix(1.0, mix(0.65, 1.2, pkS) * mix(0.8, 1.15, smoothstep(0.3, 0.7, brkN)), vB * kBrk) * mix(1.0, mix(0.7, 1.15, pk2), vK * kBrk);
    // (round 3) fine striations along the flow (the white sheet is finely streaked - threads 2-8 cm
    // across x 0.5-2 m along, bright, with slightly greyer water between them - not smeared blotches);
    // resolved down to ~1 px, then their mean
    float vFS = 1.0 - smoothstep(0.03, 0.08, foot);
    if (vFS > 0.0) {
        float fs1 = texture(uNoise3, vec3(sx * 2.4 + 0.17, ty * 0.11 * kStr, 0.21 + fk * 0.3)).r;
        float fsm = smoothstep(0.40, 0.58, fs1);
        foamy *= mix(1.0, 0.62 + 0.66 * fsm, vFS);
        dens *= mix(1.0, 0.75 + 0.5 * fsm, vFS);
    }
    // (round 3 i8) long straight threads along the flow (the sheet is a bundle of bright threads 0.1-0.4 m
    // across and metres long, straight down the flow, darker translucent water between them; the r3 strands were
    // short wavy brush strokes, smeared at the r4 / film distances). Two widths (~0.25 m, ~0.08 m) x ~4-10 m along,
    // advected with the flow; each -> its mean once ~1 px wide (mean factor 1: same brightness far away)
    float vT1 = 1.0 - smoothstep(0.15, 0.45, foot), vT2 = 1.0 - smoothstep(0.05, 0.14, foot);
    if (vT1 > 0.0) {
        float th1 = smoothstep(0.36, 0.60, texture(uNoise3, vec3(sx * 0.42 + 0.61, ty * 0.014, 0.47 + fk * 0.21)).r);
        float th2 = vT2 > 0.0 ? smoothstep(0.38, 0.62, texture(uNoise3, vec3(sx * 1.3 + 0.13, ty * 0.035 * kStr, 0.83 + fk * 0.11)).g) : 0.5;
        float thr = mix(0.5, th1, vT1) * 0.6 + mix(0.5, th2, vT2) * 0.4 - 0.5;
        float kT = 0.8;          // (review rv1/rv3: 1.7 -> 0.8, softer - high-contrast threads over black read as hair)
        // (review rv5) the threads come in bundles: their contrast varies across the sheet over ~1-3 m (crisp lace in
        // places, a smooth milky band in others) - an even comb of equal threads read as a barcode (fall B, film shot 2)
        float bnd = texture(uNoise3, vec3(sx * 0.09 + 0.4, ty * 0.006, 0.61 + fk * 0.2)).g;
        kT *= mix(1.0, mix(0.25, 1.7, smoothstep(0.36, 0.62, bnd)), 1.0);
        foamy *= 1.0 + kT * 0.9 * thr;
        dens *= 1.0 + kT * 1.1 * thr;
    }
    // (review rv1) falling clumps: the water comes in elongated clumps (~0.4 m across x ~3 m along the flow, falling
    // with it) - denser and whiter, a thinner veil between them (the streaks start and stop along the flow).
    // An amplitude modulation (the thread positions stay straight); unresolved -> mean 1
    float vCl = 1.0 - smoothstep(0.15, 0.45, foot);
    if (vCl > 0.0) {
        float clN = texture(uNoise3, vec3(sx * 0.55 + 0.21, ty * 0.08, 0.19 + fk * 0.27)).r;
        float clS = smoothstep(0.33, 0.62, clN);
        float kCl = (0.6) * vCl;
        dens *= mix(1.0, 0.5 + 1.0 * clS, kCl);
        foamy *= mix(1.0, 0.8 + 0.4 * clS, kCl);
    }
    // (review rv7) droplet clusters ~0.4 m across x ~1.4 m along the flow (worley-fbm, falling with the water): the
    // sheet is granular - bright droplet clumps with darker translucent gaps (a fast shutter), the render's silky long
    // threads read as a long exposure. Mean ~1; unresolved -> 1
    float vDr = 1.0 - smoothstep(0.06, 0.18, foot);
    if (vDr > 0.0) {
        float drN = texture(uNoise3, vec3(sx * 0.30 + 0.33, ty * 0.09, 0.27 + fk * 0.19)).b;
        float dr = smoothstep(0.40, 0.66, drN);
        float kDr = 1.0 * vDr * smoothstep(0.1, 0.35, frac);
        dens *= mix(1.0, 0.35 + 1.3 * dr, kDr);
        foamy *= mix(1.0, 0.72 + 0.56 * dr, kDr);
    }
    // (session 3) the lip: the water comes over the edge as a thin glassy tongue that turns white over the
    // first ~1.5 m of its drop (the volume's flat top read as a white box from below)
    dens *= smoothstep(-0.6, 0.5, drop);
    foamy *= mix(0.35, 1.0, smoothstep(0.0, 1.6, drop));
    dens *= 1.0 - 0.4 * gFFar;                           // (r4 C10: a thin soft thread from 1 km, not a white sail)
    return dens;
}

// spray mist rising from where fall k hits the pool (bright, thin); dist = safe step outside it
float mistDensity(int k, vec3 q, out float dist) {
    vec4 kn = uFallK[k * WF_NK + WF_NK - 1];
    // (round 3 i9) the spray spreads out over the pool as a low bright haze (a luminous band hovering over
    // the water ~10-15 m out from the impact, the curtain's foot dissolving in it - no crisp waterline under the
    // sheet): centre further out, longer outward, lower toward its outer end
    vec2 c = vec2(kn.x - 3.5, kn.y);
    const float kMs = 1.3;                         // (round 3: x1.3)
    float oX = max(c.x - q.x, 0.0);
    float hy = q.y / ((k == 0 ? 3.8 : 2.8) * kMs * mix(1.0, 0.55, smoothstep(0.0, 10.0, oX)));   // a low haze at the impact, not a fog wall (round 2: ~1/6 of the drop)
    // (review r2, fix 7) narrower along the shore: a glowing plume at the impact, not a grey fog band
    float r2 = ((q.x - c.x) * (q.x - c.x) / (q.x < c.x ? 18.0 : 8.0) + (q.z - c.y) * (q.z - c.y) / (3.0 + 1.9 * kn.z)) / (kMs * kMs) + hy * hy;   // (review rv4: 4 + 3 kn.z -> 3 + 1.9 kn.z, with the narrower falls)
    dist = (sqrt(r2) - 2.0) * 2.4 * min(kMs, 1.0);
    if (dist > 0.0) return 0.0;
    dist = 0.0;
    float base = max(exp(-r2) - 0.0183, 0.0) * smoothstep(-0.5, 0.3, q.y) * uFallInfo[k].y;   // (round 3: 0 at the cut-off r2 = 4: no edge)
    float n = texture(uNoise3, vec3(q.x * 0.11, (q.y - uTime * 1.6) * 0.09, q.z * 0.11 + uTime * 0.03)).r;
    // (r4 W12: radii 40 / 14 -> 18 / 8, amplitude 0.105 -> 0.07: a small bright spray glow at each impact, not a grey
    // haze spreading over the dark rock)
    return base * smoothstep(0.2, 0.85, n) * 0.07;   // (review rv3 x0.7: a glowing spray, not grey fog; rv7 0.6: less fog over the rock right of B)   // (round 3: 0.07 -> 0.175, a bright haze at the base)     // (session 3: 0.16 -> 0.09, review r2: 0.06, r3c 0.07 - little mist)
}

// premultiplied colour (rgb) and transmittance (a) of the falls along [0,tmax]: each fall is marched
// in its own box with adaptive steps (fine inside the water's band, long strides outside it)
vec4 fallsVolume(vec3 ro, vec3 rd, float tmax, float jitter) {
    if (uFallsOn == 0) return vec4(0.0, 0.0, 0.0, 1.0);
    // falls lab mask renders (tools/lv.py --stats): 12 = water A / B / mist opacity, 13 / 15 = no water,
    // 14 = black (baseline of 12)
    if (uDebug == 13 || uDebug == 15) return vec4(0.0, 0.0, 0.0, 1.0);
    if (uDebug == 14) return vec4(0.0);
    // falls lab water harness (tools/fh.py): 48 = the frame without the falls' water (clean, unlike 13),
    // 49 / 50 / 51 = hit distance tmax / 0 / 1 in place of the volume (the aerial term applied after this
    // cancels in (49 - 50) / (51 - 50) -> tmax)
    if (uDebug >= 48 && uDebug <= 51) return uDebug == 48 ? vec4(0.0, 0.0, 0.0, 1.0) : vec4(vec3(uDebug == 49 ? tmax * 0.01 : (uDebug == 51 ? 0.01 : 0.0)), 0.0);
    if (wfOff(1)) return vec4(0.0, 0.0, 0.0, 1.0);
    vec3 lo = fallLocal(ro);
    vec2 alongD = vec2(-uWFDir.y, uWFDir.x);
    vec3 ld = vec3(-dot(rd.xz, uWFDir), rd.y, dot(rd.xz, alongD));
    vec3 inv = 1.0 / (ld + sign(ld) * 1e-6 + vec3(1e-9));
    vec2 seg[2];
    for (int k = 0; k < 2; k++) {
        vec3 t0 = (uFallBox[2 * k].xyz - lo) * inv, t1 = (uFallBox[2 * k + 1].xyz - lo) * inv;
        vec3 tn = min(t0, t1), tf = max(t0, t1);
        seg[k] = vec2(max(max(tn.x, tn.y), max(tn.z, 0.0)), min(min(tf.x, tf.y), min(tf.z, tmax)));
    }
    if (seg[0].y <= seg[0].x && seg[1].y <= seg[1].x) return vec4(0.0, 0.0, 0.0, uDebug == 12 ? 0.0 : 1.0);
    int first = seg[1].x < seg[0].x ? 1 : 0;
    vec3 col = vec3(0.0);
    float T = 1.0;
    vec3 Tdbg = vec3(1.0);                                     // uDebug 12: sheet A / sheet B / mist
    float mu = dot(rd, uSunDir);
    // (review r2, fix 7) mist / spray phase: normalised Henyey-Greenstein, g = 0.6 (x isotropic): the droplet
    // cloud glows toward the sun, dim away from it
    float fwd = 0.4375 / pow(1.5625 - 1.5 * mu, 1.5);    // (r4 W12: g 0.6 -> 0.75, the spray glows toward the sun only)
    // multiply scattered white water is lit by a near-grey sky (the saturated blue ambient alone turns
    // shaded water electric blue) plus light bounced off the sunlit rock / forest around it
    vec3 skyW = mix(vec3(dot(uSkyAmb, vec3(0.2126, 0.7152, 0.0722))), uSkyAmb, 0.6);
    for (int j = 0; j < 2; j++) {
        int k = j == 0 ? first : 1 - first;
        vec2 sg = seg[k];
        if (sg.y <= sg.x) continue;
        float dtMin = clamp(0.5 * gPixAng * sg.x + 0.06, 0.08, 0.25);
        gFFar = smoothstep(300.0, 500.0, sg.x);
        float t = sg.x + dtMin * jitter;
        float sunV = 1.0;
        int sinceSun = 99;
        bool inBand = false;
        for (int i = 0; i < 128; i++) {
            if (t >= sg.y || T < 0.01) break;
            vec3 q = lo + ld * t;
            float foot = max(t, 0.5) * gPixAng;
            float foamy, dF, dM;
            float df = fallDensity(k, q, ld, foot, foamy, dF);
            // entering the sheet's band after a distance stride: re-jitter the fine steps (the stride
            // lands at the same depth for neighbouring pixels -> the fixed sample grid showed as hatching
            // along the strands); the stretch stepped back over is empty by the distance bound
            if (dF <= 0.0 && !inBand) {
                inBand = true;
                t -= dtMin * fract(jitter * 7.31 + float(i) * 0.618);
                continue;
            }
            if (dF > 0.0) inBand = false;
            float dm = mistDensity(k, q, dM);
            // (round 3) inside the smooth mist but outside the sheet's band: 0.45 m strides (the mist's dM = 0
            // forced dtMin steps there - the 112-step budget ran out in the mist before the sheet was reached:
            // a horizontal seam across both falls at the mist's top, black arcs with a bigger mist)
            float dt = clamp(min(dF, dM > 0.0 ? dM : max(dF, 0.45)) * 0.85, dtMin, 1.5);
            float dd = df + dm;
            if (dd > 1e-4) {
                // fine steps in the water sheet; the smooth mist (~1.4 m features) takes 0.45 m strides
                dt = df > 1e-4 ? max(dtMin, min(gCoreD * 0.85, 0.35)) : max(dtMin, 0.6);   // (i9: mist strides 0.45 -> 0.6 m, the spread haze)
                // sun visibility: sampled on entering the water / mist, then every 4 steps
                if (sinceSun >= 4) { sunV = sunShadowMap(ro + rd * t + vec3(0.0, 1.0, 0.0)); sinceSun = 0; }
                sinceSun++;
                // white water: multiple scattering keeps it bright; thin lace stays grey-translucent
                vec3 L = uSunCol * (sunV * (0.44 + 0.14 * foamy) + 0.05) + skyW * (0.95 + 0.35 * foamy);
                vec3 Lm = uSunCol * (sunV * 0.9 * fwd + 0.06) + skyW * 0.7;    // (r4 W12: sky term 1.2 -> 0.7)
                // (r4 W7: 0.21 / 0.64 -> 0.17 / 0.52: grey-white translucent lace, ~7 % above 235, not a cool-white silk plateau)
                vec3 S = (df * L * vec3(0.97, 0.975, 0.98) * mix(0.17, 0.52, foamy) + dm * Lm) / dd;   // (r3: more contrast - white dense water, darker translucent lace; p95 ~240-248)
                float Ts = exp(-dd * dt);
                col += T * S * (1.0 - Ts);
                T *= Ts;
                if (uDebug == 12) { if (k == 0) Tdbg.x *= exp(-df * dt); else Tdbg.y *= exp(-df * dt); Tdbg.z *= exp(-dm * dt); }
            } else sinceSun = 99;
            t += dt;
        }
    }
    if (uDebug == 12) return vec4(1.0 - Tdbg, 0.0);
    return vec4(col, T);
}

// shrub / fern / grass / moss cover: clumps (~2 m and ~1 m domes, worley-fbm) with dark gaps between
// them, drooping fronds and leaf grain up close. lush 0..1 = wet spray zone (brighter lime moss,
// #9da136). Returns albedo, the clump normal nv, the clump field cl (0 gap .. 1 top) and an occlusion.
float gVegV = 1.0;      // (review rv5) vertical scale of the leaf-clump noise (< 1: hanging, set by wfRockShade on the face)
float gVegFine = 0.0;   // (falls6) 1 on the cliff face (wfRockShade): finer leaf / cushion grain, smaller dark gaps
vec3 wfVegetationC(vec3 p, vec3 n0, float fp, float lush, float con, out vec3 nv, out float cl, out float occ) {
    vec3 pq = p * 0.065;
    float c0 = texture(uNoise3, pq).b;
    float visC = 1.0 - smoothstep(0.10, 0.30, fp);
    vec3 gc = vec3(texture(uNoise3, pq + vec3(0.02, 0.0, 0.0)).b - c0, texture(uNoise3, pq + vec3(0.0, 0.02, 0.0)).b - c0,
                   texture(uNoise3, pq + vec3(0.0, 0.0, 0.02)).b - c0) * 50.0;
    float vF = 1.0 - smoothstep(0.03, 0.12, fp);
    float lf = mix(0.5, texture(uNoise3, p * 1.7).r, vF);
    // leaf clusters (~0.35 m): bright tufts and dark holes, the texture of foliage at 20-60 m
    float fF = gVegFine;      // (falls6) the 0.2-0.4 m leaf grain stays until ~1/3 of its size per pixel
    float lc = mix(0.5, texture(uNoise3, p * 0.35 * vec3(1.0, gVegV, 1.0) + 0.17).b, 1.0 - smoothstep(mix(0.04, 0.07, fF), mix(0.10, 0.15, fF), fp));
    // drooping fronds / hanging grass: streaks stretched down the slope, fading at 5-12 cm footprint
    float fr = mix(0.5, texture(uNoise3, p * vec3(0.9, 0.22, 0.9) + vec3(0.0, 0.05 * gnoise(p.xz * 0.7), 0.3)).r, 1.0 - smoothstep(0.03, 0.07, fp));
    // leaf blobs (~0.2 m, worley): bright sunlit leaves with dark holes between them
    float lb = (0.55 - texture(uNoise3, p * 0.32 * vec3(1.0, gVegV, 1.0) + 0.11).a) * 0.8 * (1.0 - smoothstep(mix(0.025, 0.06, fF), mix(0.06, 0.12, fF), fp));
    lf = 0.5 + (lf - 0.5) * 0.5 + (lc - 0.5) * 0.5 + lb + (wfRemap(fr) - 0.5) * 0.25;
    // bush groups (~5 m) that stay visible from far away: dark hollows between them
    vec4 nB = texture(uNoise3, p * 0.024 + 0.37);
    float cB = nB.b * 0.6 + nB.r * 0.4;
    float visC1 = 1.0 - smoothstep(0.04, 0.12, fp);
    float c1 = visC1 > 0.0 ? texture(uNoise3, p * 0.16 + 0.3).b : 0.525;
    // (review r2, fix 2) the 15-90 m modulation (c0 x0.8, cB x0.35) made lime camouflage blotches: the
    // variation now lives in leaf clumps of 0.15-0.6 m (k1 / k2) with hard edges and dark gaps
    // (uNoise3 .b = worley-fbm, cells 1 / (8 k) m: 0.4 m / 0.16 m leaf clumps)
    float visK1 = 1.0 - smoothstep(0.12, 0.30, fp), visK2 = 1.0 - smoothstep(0.05, 0.12, fp);
    // (review rv5) the leaf clumps hang: stretched ~2.5x down the slope (vertical; on a ledge the stretch does nothing) -
    // fronds / grass tongues draping down the face, not isotropic lime sponge
    vec3 kV = vec3(1.0, gVegV, 1.0);
    float k1 = visK1 > 0.0 ? texture(uNoise3, p * 0.31 * kV + 0.43).b : 0.525;
    float k2 = visK2 > 0.0 ? texture(uNoise3, p * 0.8 * kV + 0.19).b : 0.525;
    // (falls6) on the face: less of the 0.8-2 m gap blobs (dark camouflage patches), more of the 0.16-0.4 m clumps
    float fG = 1.6 * gVegFine;
    cl = saturate(0.5 + (c0 - 0.525) * 0.3 * visC * max(1.0 - 0.6 * fG, 0.0) + (c1 - 0.525) * 0.6 * visC1 * max(1.0 - 0.5 * fG, 0.1) + (cB - 0.51) * 0.1
                  + (k1 - 0.525) * 0.9 * visK1 * (1.0 + 0.2 * fG) + (k2 - 0.525) * 0.7 * visK2 * (1.0 + 0.4 * fG));
    float tone = texture(uNoise3, p * 0.011 + 0.7).g;
    // sunlit leaves #9da136 (yellow-lime), gaps #1a2010
    vec3 lit = mix(vec3(0.072, 0.098, 0.034), vec3(0.160, 0.188, 0.060), saturate(0.5 + (tone - 0.5) * 2.0));   // (r3a: x1.5; r3c less yellow)
    float slV = max((0.18 + 0.20 * (1.0 - visK1)) * (1.0 - 0.55 * gVegFine), 2.5 * fp);      // (r3d AA: >= the clump noise's change over ~2 px; falls6: crisper clumps on the face)
    float lv = smoothstep(0.5 - slV, 0.5 + slV, cl + 0.5 * (lf - 0.5));
    float fL = 0.6 * gVegFine;      // (falls6) the gaps between the clumps dark olive (shaded leaves), not black holes
    vec3 veg = mix(mix(vec3(0.014, 0.020, 0.006), lit * 0.33, fL), lit, mix(0.58, lv, con));   // (round 3: con < 1 on the face - the cushions carry the form)
    veg = mix(veg, vec3(0.15, 0.17, 0.030), lush * 0.3 * smoothstep(0.55, 0.85, cl + 0.3 * (lf - 0.5)));
    // (r3) leaf cushions ~0.6 m: domes (worley-fbm peaks) whose gradient tilts the normal - each cushion lit
    // on its sun side, shaded on the other, dark in the hollows between (3D, not spray paint); random leaf
    // tilt only at the leaf scale
    float visD = 1.0 - smoothstep(0.10, 0.28, fp);
    vec3 gD = vec3(0.0); float kd = 0.525;
    if (visD > 0.0) {
        vec3 pd = p * 0.2 + 0.43;
        kd = texture(uNoise3, pd).b;
        gD = vec3(texture(uNoise3, pd + vec3(0.025, 0.0, 0.0)).b - kd, texture(uNoise3, pd + vec3(0.0, 0.025, 0.0)).b - kd,
                  texture(uNoise3, pd + vec3(0.0, 0.0, 0.025)).b - kd) * 8.0;                // per m
    }
    vec3 nN = (texture(uNoise3, p * 0.9 + 0.2).rgb - 0.5) * 0.7 * (1.0 - smoothstep(0.02, 0.05, fp));
    vec3 nb0 = normalize(n0 * 0.45 + vec3(0.0, 0.75, 0.0) + gc * 0.22 * visC);
    nv = normalize(nb0 - (gD - nb0 * dot(gD, nb0)) * 0.9 * visD + nN);
    // gaps between the leaf clumps / hollows between the cushions: deep, ~0.1 of the sky (unresolved -> mean)
    occ = mix(mix(0.1, 0.4, fL), 1.0, mix(0.58, lv, con)) * mix(0.75, 1.0, smoothstep(0.3, 0.6, lf)) * mix(1.0, mix(0.35, 1.0, smoothstep(0.25, 0.6, kd)), visD);
    return veg;
}
vec3 wfVegetation(vec3 p, vec3 n0, float fp, float lush, out vec3 nv, out float cl, out float occ) {
    return wfVegetationC(p, n0, fp, lush, 1.0, nv, cl, occ);
}

// (round 3) water-worn rock: rounded-angular bosses instead of bedding joints (the r2 joint blocks read as a
// masonry wall of stacked slate slabs - the cliff is irregular knobbly rock: rounded bosses of very
// different sizes, black crevices between them, no parallel layering). One level of the boss field: per cell
// (size S) one boss - a superellipse footprint 0.45-0.95 S (own orientation, round .. squarish), a flat-ish
// top with steep sides (A (1 - s^2)), its own height and tilt - merged with its neighbours by a polynomial
// smooth max of width k (a wide k fuses neighbours into lumps, a narrow one leaves a crisp crease).
// q in m. Returns (height m, d/dq) and the id of the highest boss
float gBossFp = 0.0;     // (round 3 i12) pixel footprint (m) for wfBoss's crease width (set by wfRockShade)
float gFacet = 0.0;      // (review rv1) share of the polyhedral (planar-facet) boss profile (set by wfRockD; 0 = domes)
float gBossCrk = 1e3;    // (falls6) distance (m) to the nearest crease between two bosses (top-two height gap / slope gap)
vec3 wfBoss(vec2 q, float S, float amp, float seed, float kc, out float id) {
    // (each level's cell grid turned by its own angle: no rows / columns of bosses lining up)
    float ag = seed * 0.61 + 0.4;
    vec2 cg = vec2(cos(ag), sin(ag));
    vec2 g = vec2(dot(q, cg), dot(q, vec2(-cg.y, cg.x))) / S;
    vec2 i = floor(g), f = g - i;
    float h = -1e3, hb = -1e3, h2 = -1e3;
    vec2 gr = vec2(0.0), g1 = vec2(0.0), g2 = vec2(0.0);
    id = 0.0;
    ivec2 sd = ivec2(int(seed) * 7919 + 13, int(seed) * 3301 + 977);
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        vec2 o = vec2(x, y);
        ivec2 ci = ivec2(i + o);
        vec4 hc = hash42i(ci + sd);
        vec4 hd = hash42i(ci * 3 + sd.yx);
        vec2 r = f - (o + 0.1 + 0.8 * hc.xy);                     // from the boss centre (cell units)
        float an = hc.z * 3.1416;
        vec2 cs = vec2(cos(an), sin(an));
        vec2 R = vec2(mix(0.55, 0.95, hd.x), mix(0.45, 0.85, hd.y));
        vec2 l = vec2(dot(r, cs), dot(r, vec2(-cs.y, cs.x))) / R;
        float x2 = l.x * l.x, y2 = l.y * l.y;
        float q4 = sqrt(x2 * x2 + y2 * y2) + 1e-6;
        float sqr = sqrt(hd.z);                                     // (i8: mostly squarish - blocky, not pillows)
        float s = mix(x2 + y2, 1.25 * q4, sqr);                     // squared radius: round .. squarish
        vec2 ds = mix(2.0 * l, 2.5 * vec2(x2 * l.x, y2 * l.y) / q4, sqr);
        float sc = min(s, 1.5);                                     // the valley floor between bosses
        float A = amp * mix(0.4, 1.0, hc.w);
        float hi = A * (1.0 - sc * sc);
        vec2 gl = -A * 2.0 * sc * ds * step(s, 1.5) / R;
        // (review rv1) fractured, not water-worn: a polyhedral boss - 4 planar facets of different slopes around an
        // off-centre apex, linear profile (flat facets, each lit its own grey; ridges rounded over ~1.5 pixel
        // footprints) - the cliff is angular rock, the domes read as piled pillows. gFacet = share (wfRockD)
        if (gFacet > 0.0) {
            vec2 la = l - (hd.wx - 0.5) * 0.5;
            vec4 kk = 0.75 + 0.6 * fract(hd.zwxy * 7.31 + hc.yxwz * 3.7);
            float px = la.x > 0.0 ? la.x * kk.x : -la.x * kk.y, py = la.y > 0.0 ? la.y * kk.z : -la.y * kk.w;
            vec2 gpx = vec2(la.x > 0.0 ? kk.x : -kk.y, 0.0), gpy = vec2(0.0, la.y > 0.0 ? kk.z : -kk.w);
            float kr = max(0.06, 1.5 * gBossFp / (S * min(R.x, R.y)));
            float hr = clamp(0.5 + 0.5 * (px - py) / kr, 0.0, 1.0);
            float pm = mix(py, px, hr) + kr * hr * (1.0 - hr);
            vec2 gpm = mix(gpy, gpx, hr);
            float pmc = min(pm, 1.5);
            hi = mix(hi, A * (1.0 - 1.25 * pmc), gFacet);
            gl = mix(gl, -A * 1.25 * gpm * step(pm, 1.5) / R, gFacet);
        }
        vec2 gri = vec2(gl.x * cs.x - gl.y * cs.y, gl.x * cs.y + gl.y * cs.x);
        vec2 tl = (hd.wy - 0.5) * 1.6 * A;                   // each boss its own tilted facet (i8: 0.7 -> 1.1, angular)
        hi += dot(r, tl); gri += tl;
        // (i8: crisper creases x0.35; i12: never narrower than ~1.2 pixel footprints - a sub-pixel crease
        // aliased into frame-to-frame shimmer on the rock in film shot 2)
        float k = max(kc * A * mix(0.25, 1.7, fract(hc.w * 7.31)) * 0.35, 1.2 * gBossFp);
        float hk = clamp(0.5 + 0.5 * (hi - h) / k, 0.0, 1.0);
        h = mix(h, hi, hk) + k * hk * (1.0 - hk);
        gr = mix(gr, gri, hk);
        if (hi > hb) { h2 = hb; g2 = g1; hb = hi; g1 = gri; id = hc.w; }
        else if (hi > h2) { h2 = hi; g2 = gri; }
    }
    // (falls6) the crease where the two highest bosses meet: their height gap over their slope gap (m)
    gBossCrk = (hb - h2) * S / max(length(g1 - g2), 1e-3);
    gr /= S;
    return vec3(h, gr.x * cg.x - gr.y * cg.y, gr.x * cg.y + gr.y * cg.x);   // d/dq (back to q's frame)
}
// relief of the rock face (m, outward) at face coordinates fq: big bosses (S), medium (0.36 S) and knobs
// (0.13 S) on top, each level faded to its mean once unresolved (visM / vis2); smooth in the chute (water-worn,
// lower). Returns (height, d/dq); cav = 0 in the valleys between the big / medium bosses .. 1 on their tops
vec2 gCrk = vec2(1e3);   // (falls6) crease distance (m) of the big / medium boss levels (set by wfRockD)
vec3 wfRockD(vec2 fq, float S, float visM, float vis2, float chute, out float cav, out float id) {
    float kA = mix(1.0, 0.45, chute);
    const float kK = 1.0;
    gFacet = 1.0 - 0.8 * chute;                       // (review rv1: angular facets; the chutes stay water-worn; rocks4 W9: 0.85 -> 1)
    // (round 3) the relief varies over ~15 m: smoother slab faces beside knobbly areas (no even pillow field)
    kA *= mix(0.45, 1.2, smoothstep(-0.35, 0.35, gnoise(fq * 0.065 + 3.3)));
    float A1 = 0.30 * S * kA;
    vec2 wq = fq + S * 0.3 * vec2(gnoise(fq * (0.45 / S) + 1.7), gnoise(fq * (0.45 / S) + 8.2));
    vec3 r = wfBoss(wq, S, A1, 3.0, 0.30 * kK, id);
    gCrk = vec2(gBossCrk, 1e3);
    cav = smoothstep(-0.9, 0.45, r.x / A1);
    if (visM > 0.0) {
        float A2 = 0.26 * 0.36 * S * kA * (1.4), id2;      // (review rv2 x1.4: more, smaller facets)
        vec3 b2 = wfBoss(fq + vec2(7.3, 3.1), 0.36 * S, A2, 11.0, 0.45 * kK, id2);
        gCrk.y = gBossCrk;
        r += b2 * visM;
        cav *= mix(1.0, mix(0.55, 1.0, smoothstep(-0.9, 0.3, b2.x / A2)), visM);
    }
    if (vis2 > 0.0) {
        float A3 = 0.24 * 0.13 * S * kA * 1.5, id3;   // (review rv7 x1.5: crisper small facets)
        r += wfBoss(fq + vec2(3.7, 9.1), 0.13 * S, A3, 23.0, 0.7 * kK, id3) * vis2;
    }
    return r;
}

// the forested rim of the cliff (uf in [W-3, W+10] behind the bare face): a dark mass of shrubs and
// young spruce, only a few lime tips; 0..1
float wfRimW(float v, float uf, float W) {
    return smoothstep(W - 4.5, W - 2.0, uf) * (1.0 - smoothstep(W + 8.0, W + 13.0, uf))
         * smoothstep(-24.0, -14.0, v) * (1.0 - smoothstep(28.0, 38.0, v));
}

// the dark wet rock of the step: buttresses (~12 x 15 m) of chunky jointed blocks, each block a
// facet catching the light differently, bevelled edges, black joints and cast shadows between them;
// near-black where wet, grey with pale lichen patches on drier parts; ferns / grass growing from the
// joints and ledges, drapes of shrubs hanging below the ledges, lime moss in the spray.
// Returns albedo; bends n.
vec3 wfRockShade(vec3 p, vec3 n0, float t, out vec3 n, out float mossW, out float wetW, out float occ) {
    vec2 uv = wfUV(p.xz);
    float fp = t * gPixAng;
    gBossFp = fp;
    float inChute = 0.0, fringe = 0.0, sprayC = 0.0;
    for (int k = 0; k < 2; k++) {
        float hw = wfChuteHW(k, uv.x) * (k == 0 ? 0.8 : 0.75);
        // ragged wet edges (no ruler-straight stripes beside the water)
        float dv = abs(uv.y - wfChuteV(k, uv.x) + 0.9 * gnoise(vec2(p.y * 0.25, uv.x * 0.3 + float(k) * 7.0)));
        float below = 1.0 - smoothstep(uFallInfo[k].x, uFallInfo[k].x + 1.0, p.y);
        inChute = max(inChute, (1.0 - smoothstep(hw * 0.7, hw * 1.1, dv)) * below);
        fringe = max(fringe, (1.0 - smoothstep(hw, hw + 5.0, dv)) * below);
        sprayC = max(sprayC, (1.0 - smoothstep(hw * 0.9, hw + 2.2, dv)) * below);     // (review r2) wet black rock right beside the sheets
    }
    vec3 A = vec3(-uWFDir.y, 0.0, uWFDir.x);               // along the shore (+v)
    const vec3 U = vec3(0.0, 1.0, 0.0);
    float steep = 1.0 - smoothstep(0.55, 0.85, n0.y);      // 1 on the face, 0 on ledges / benches
    // the buttress north of fall A: bare, darkest rock below its fern-covered top
    float hsB = wfStepH(uv.y);
    float ybt = wfButtH(uv.y, hsB);
    float butt = wfButt(uv.y) * (1.0 - smoothstep(ybt - 1.5, ybt + 0.5, p.y));
    vec4 B, hB;
    vec2 fc = wfFaceC(uv.y, p.y, B, hB);
    // a big ledge on top of each (session 3: faded at the cell's edges - the per-buttress value switched
    // at the voronoi boundary and outlined sharp-edged rectangles of moss)
    float topB = smoothstep(0.18, 0.36, B.y + 0.05 * (hB.w - 0.5)) * smoothstep(0.0, 0.15, B.z);
    vec2 fj = fc + hB.xy * 7.0;
    float aB = 0.35 + 1.6 * gnoise(fc * 0.018 + 4.1);     // bedding dip of the blocks (r3: +-35 deg, ~50 m)
    vec2 cs = vec2(cos(aB), sin(aB));
    // (round 3) relief: water-worn rounded-angular bosses (wfRockD: big 1.8-3.2 m per buttress, medium ~0.36 x,
    // knobs ~0.13 x) - no bedding. Face coordinates (along the shore, up the face); on ledges / benches the
    // second one runs inland (y - 0.6 u: no stripes there); in the chutes the bosses are lower and stretched
    // down the flow (polished). Normal from the field's analytic gradient (world: dv -> A, d(y - 0.6 u) ->
    // U + 0.6 x outward-to-inland), cast shadows marched over the field toward the sun, black crevices (cav)
    vec3 Dout = vec3(uWFDir.x, 0.0, uWFDir.y);                    // outward (u decreases)
    float S = mix(1.8, 3.2, hB.z);
    float sY = mix(1.0, 0.55, inChute);
    vec2 fq = vec2(fj.x, (fj.y - 0.6 * uv.x) * sY);
    vec3 Ty = (U + 0.6 * Dout) * sY;                               // world gradient of fq.y
    float visR = 1.0 - smoothstep(0.10 * S, 0.30 * S, fp);
    float visM = 1.0 - smoothstep(0.036 * S, 0.11 * S, fp);
    float vis2 = 1.0 - smoothstep(0.013 * S, 0.04 * S, fp);
    float sh = 0.0, cav = 0.6, id1 = 0.5;
    vec3 nb = n0;
    vec2 crkD = vec2(1e3);
    if (visR > 0.0) {
        float cv_, id_;
        vec3 r0 = wfRockD(fq, S, visM, vis2, inChute, cav, id1);
        crkD = gCrk;
        vec3 gW = A * r0.y + Ty * r0.z;
        gW -= n0 * dot(gW, n0);
        nb = normalize(n0 - gW * steep * visR);
        cav = mix(0.6, cav, visR);
        vec2 Ls = vec2(dot(uSunDir, A), dot(uSunDir, Ty));
        float Lz = dot(uSunDir, n0), lL = length(Ls);
        if (Lz > 0.02 && lL > 0.05 && visR > 0.2) {
            float jt = 0.85 + 0.3 * gRand;                              // jittered start (no banding)
            for (int i = 0; i < 3; i++) {
                float sd = 0.07 * S * pow(2.6, float(i)) * jt;           // ~0.17, 0.45, 1.2 m (S 2.5)
                // (the far steps only see the big / medium bosses)
                vec3 rs = wfRockD(fq + Ls * sd, S, i < 2 ? visM : 0.0, i < 1 ? vis2 : 0.0, inChute, cv_, id_);
                sh = max(sh, smoothstep(0.0, 0.025 * S + 0.6 * fp, rs.x - r0.x - sd * Lz));
            }
            sh *= steep * visR * smoothstep(0.2, 0.45, visR);
        }
    }
    // (falls6) crisp fractures: a black crack line along the creases between the big and the medium
    // bosses, >= ~1 pixel footprint wide (anti-aliased), faded out once the bosses are ~10-25 px apart
    float wC1 = max(0.022 * S, 0.9 * fp), wC2 = max(0.010 * S, 0.9 * fp);
    float crk = (1.0 - smoothstep(0.35 * wC1, wC1 + 0.5 * fp, crkD.x)) * (1.0 - smoothstep(0.04 * S, 0.10 * S, fp));
    crk = max(crk, 0.8 * visM * (1.0 - smoothstep(0.35 * wC2, wC2 + 0.5 * fp, crkD.y)) * (1.0 - smoothstep(0.016 * S, 0.04 * S, fp)));
    crk *= visR * mix(0.5, 1.0, steep) * (1.0 - 0.7 * inChute);
    float fyR = (nb.y - n0.y) * steep * visR;          // (falls6) facet tilt of the boss relief (+ = faces up: a moss ledge)
    vec4 hr = hash42i(ivec2(int(id1 * 65536.0), 7));
    vec4 hb = mix(vec4(0.5), hr, visR);                    // per-boss values fade to their mean
    float jB = smoothstep(0.0, max(0.05, 0.25 * fp), B.z);
    nb = normalize(nb + steep * (A * (hB.x - 0.5) * 0.5 * jB + U * ((hB.y - 0.5) * 0.3 * jB + topB * 0.45)));
    // micro relief up close: ridged worley grain (cm-scale crystals / pits)
    float vis3 = 1.0 - smoothstep(0.008, 0.02, fp);
    if (vis3 > 0.0) {
        float w0 = texture(uNoise3, p * 0.8).a, wx = texture(uNoise3, p * 0.8 + vec3(0.01, 0.0, 0.0)).a, wy = texture(uNoise3, p * 0.8 + vec3(0.0, 0.01, 0.0)).a;
        vec2 g = vec2(wx - w0, wy - w0) * 35.0;
        nb = normalize(nb + vis3 * (A * g.x + U * g.y) * 0.05 + vis3 * (texture(uNoise3, p * 2.5 + 0.5).rgb - 0.5) * 0.12);
    }
    // rough, lumpy rock surface at 0.3-1 m (the noise channels' std is ~0.14: x1.6 / x1.2 tilts by ~10 deg)
    // (each scale fades out by a footprint of ~1/3 of its feature size: no sub-pixel normal noise, which
    // sparkles on the wet rock and crawls in motion)
    // (round 2: weaker - smooth noise normals lit from the side made soft dark blotches, a leopard /
    // camouflage pattern; the roughness is now mostly the angular facets below)
    nb = normalize(nb + (texture(uNoise3, p * 0.31).rgb - 0.5) * 0.3 * (1.0 - smoothstep(0.08, 0.18, fp))
                      + (texture(uNoise3, p * 0.7 + 0.2).rgb - 0.5) * 0.22 * (1.0 - smoothstep(0.02, 0.05, fp)));   // (round 3: 0.6 / 0.5 -> 0.3 / 0.22: the bosses carry the form)
    // ridged grain (round 2): creased, fractured roughness at ~0.25 m and ~0.1 m - crisp lit ridges
    // and dark creases (smooth noise normals made soft blotches); fades once a crease is ~1.5 px
    // (the crease of |n - m| is infinitely thin: it is rounded over the pixel footprint - sqrt(x^2 + w^2),
    // w = the noise's change across one pixel - else the creases alias / shimmer)
    float vR0 = (1.0 - smoothstep(0.12, 0.30, fp)) * steep;
    float vRa = (1.0 - smoothstep(0.04, 0.10, fp)) * steep, vRb = (1.0 - smoothstep(0.015, 0.035, fp)) * steep;
    const float eR = 0.03;
    if (vR0 > 0.0) {
        vec3 q0 = p * 0.17 + 7.7;
        float w0 = 3.0 * 0.17 * fp + 0.01;
        float c0 = wfSoftAbs(texture(uNoise3, q0).r - 0.475, w0), cx = wfSoftAbs(texture(uNoise3, q0 + A * eR * 0.17 * 4.0).r - 0.475, w0),
              cy = wfSoftAbs(texture(uNoise3, q0 + U * eR * 0.17 * 4.0).r - 0.475, w0);
        nb = normalize(nb - vR0 * (A * (cx - c0) + U * (cy - c0)) / (eR * 4.0) * 0.12);  // (session 3: 1.1 -> 0.7, session 4: 0.45, review r2: 0.25 - the blocks carry the structure now)
    }
    if (vRa > 0.0) {
        vec3 qa = p * 0.42 + 2.3;
        float wa = 3.0 * 0.42 * fp + 0.01;
        float r0 = wfSoftAbs(texture(uNoise3, qa).r - 0.475, wa), rx = wfSoftAbs(texture(uNoise3, qa + A * eR * 0.42).r - 0.475, wa),
              ry = wfSoftAbs(texture(uNoise3, qa + U * eR * 0.42).r - 0.475, wa);
        nb = normalize(nb - vRa * (A * (rx - r0) + U * (ry - r0)) / eR * 0.20);
        if (vRb > 0.0) {
            vec3 qb = p * 1.05 + 5.1;
            float wb = 3.0 * 1.05 * fp + 0.01;
            float s0 = wfSoftAbs(texture(uNoise3, qb).r - 0.475, wb), sx_ = wfSoftAbs(texture(uNoise3, qb + A * eR * 1.05).r - 0.475, wb),
                  sy_ = wfSoftAbs(texture(uNoise3, qb + U * eR * 1.05).r - 0.475, wb);
            nb = normalize(nb - vRb * (A * (sx_ - s0) + U * (sy_ - s0)) / eR * 0.22);
        }
    }
    nb = normalize(mix(nb, n0, inChute * 0.45));
    // crevices (round 3): the valleys between the bosses are black wet cracks where the sun does not reach;
    // unresolved bosses -> a mean darkening (their shadowed sides / crevices averaged into the pixel)
    float jnt = (1.0 - cav) * visR * steep;
    float occJ = mix(0.72, 1.0, visR) * mix(1.0, 0.22, jnt);
    // rock tone: dark wet rock, drier grey patches with pale lichen away from the spray, dark seepage
    // stains running down the face (sub-pixel tone -> its mean)
    float g = mix(0.48, texture(uNoise3, p * vec3(0.5, 0.22, 0.5)).r, 1.0 - smoothstep(0.05, 0.12, fp));
    float grain = mix(0.5, texture(uNoise3, p * 0.9 + 0.3).r, 1.0 - smoothstep(0.015, 0.05, fp));
    // (+ a 1-3 m mottle that stays resolved from the film's waterfall shot: no smooth grey slabs)
    float mot = mix(0.5, wfRemap(texture(uNoise3, p * vec3(0.12, 0.07, 0.12) + 0.37).r), 1.0 - smoothstep(0.25, 0.6, fp));
    // (round 2: darker, and no leopard mottle - the wet rock is near-black, its texture is light
    // and shadow on the blocks, not paint)
    // (rocks4 W9) tone noise <= +-15 % (the r3 face: light grey with hard black camouflage blots, median #515049 vs
    // #1f2215); fewer / darker dry patches, less pale lichen
    vec3 rock = vec3(0.016, 0.017, 0.015) * mix(0.92, 1.08, hb.w) * mix(0.94, 1.06, wfRemap(g)) * mix(0.9, 1.1, grain) * mix(0.93, 1.07, mot);
    float dry = smoothstep(0.72, 0.95, hb.w * 0.5 + 0.1 + 0.3 * hB.z + 0.55 * gnoise(fc * 0.03 + 1.3)) * (1.0 - fringe) * (1.0 - butt);
    vec3 dryR = vec3(0.030, 0.030, 0.027) * mix(0.92, 1.08, wfRemap(g)) * mix(0.92, 1.08, mot);
    float lich = smoothstep(0.55, 0.80, wfRemap(texture(uNoise3, p * 0.09 + 2.0).r) * 0.6 + 0.4 * texture(uNoise3, p * 0.31 + 1.0).g)
               * smoothstep(0.2, 0.7, nb.y + 0.5);
    dryR = mix(dryR, vec3(0.26, 0.265, 0.23), lich * 0.2);
    rock = mix(rock, dryR, dry);
    // (review r2, fix 5) pale granite steps right of fall B's foot: its water cascades over them
    float vB_ = uFallK[WF_NK + WF_NK - 1].y, hwB_ = uFallK[WF_NK + WF_NK - 1].z;
    float paleB = smoothstep(vB_ + 0.4 * hwB_, vB_ + 0.8 * hwB_, uv.y) * (1.0 - smoothstep(vB_ + hwB_ + 3.0, vB_ + hwB_ + 7.0, uv.y))
                * (1.0 - smoothstep(4.5, 7.5, p.y + 1.5 * gnoise(fc * 0.3)));
    rock = mix(rock, vec3(0.105, 0.102, 0.092) * mix(0.8, 1.2, wfRemap(g)) * mix(0.85, 1.15, grain), paleB * 0.85);
    // crustose lichen specks (5-20 cm, ~12 % cover) on the drier rock away from the spray; sub-pixel
    // specks -> their mean
    float spk = smoothstep(0.30, 0.22, texture(uNoise3, p * 0.55 + 0.71).a + 0.12 * (texture(uNoise3, p * 0.13 + 0.2).r - 0.5));
    spk = mix(0.025, spk * 0.3, 1.0 - smoothstep(0.02, 0.045, fp)) * (1.0 - fringe) * (1.0 - inChute) * mix(0.5, 1.0, dry);   // (rocks4 W9: 0.05 / 0.45)
    rock = mix(rock, vec3(0.18, 0.18, 0.165), spk);
    // the up-facing part of each block drier and paler (its lit top rim catches the sun: the p95),
    // its underside wet and near-black (-> mean far away)
    // (round 3 i9) more contrast: the buttress is mostly near-black (p50 ~30-50) with pale lit facet tops
    // (p90 ~150, p99 ~200-230); the r3 rock was an even mid grey (p50 ~65, p90 ~110)
    float topF = smoothstep(0.15, 0.7, nb.y);
    // (rocks4 W9: undersides x0.22 -> 0.55 - the darkening belongs in the joints (occJ), not in big black blots; lit tops <= 0.14)
    rock *= mix(1.0, mix(0.55, mix(2.2, 3.4, dry), topF), visR * steep);       // (r3a/b: dry tops x2.8, p95; wet x1.7)
    float stain = smoothstep(0.45, 0.75, texture(uNoise3, vec3(uv.y * 0.12, p.y * 0.012, 0.2)).r);
    rock *= mix(1.0, 0.65, stain * steep * (1.0 - paleB));
    rock *= mix(1.0, 0.70, butt);
    rock *= mix(1.0, 0.55, inChute * (1.0 - 0.6 * paleB));   // streaming wet rock
    // vegetation (review r2, fix 2): about half of the face under ferns / grass / moss (~50 % left of
    // fall A, densest over the upper face and the ledges, little in the spray right beside the sheets and
    // none under overhangs). Big masses (>= ~3 m) from wfVegRaw, broken at their edges and inside into leaf
    // clumps of 0.15-0.6 m with hard edges and dark rock showing in the gaps
    float vRaw = wfVegRaw(fc, hsB);
    vec3 nv; float cl, occV;
    float kVv = mix(1.0, 0.3, steep);           // (review rv5) hanging leaf clumps on the steep face (rv6 sweep: 0.4 / 0.25 -> 0.3)
    gVegV = kVv; gVegFine = 1.0;
    vec3 veg = wfVegetationC(p, n0, fp, 0.35 * fringe, mix(0.7, 0.5, smoothstep(0.03, 0.08, fp)), nv, cl, occV);   // (review rv4: con 0.45 -> 0.7, dark gaps between the leaf clumps; rv8: -> 0.5 from a 3-8 cm footprint - camouflage blotches in film shot 2)
    gVegV = 1.0; gVegFine = 0.0;
    // moss / undergrowth under the 3D plants: darker on the steep face, lit cushions on the ledges
    veg *= mix(0.72, 1.1, smoothstep(0.35, 0.8, n0.y + 0.4 * topB));
    veg *= mix(0.8, 1.0, smoothstep(0.45, 0.8, n0.y + 0.4 * topB));   // (review rv2: the steep face's cover darker)
    float upF = smoothstep(0.5, 0.85, p.y / max(hsB, 5.0)) * (1.0 - fringe);
    veg *= mix(1.0, 1.12, upF);
    float tn = texture(uNoise3, p * 0.55 + 0.33).g;
    // small tufts in the joints and on block tops: only up close (gone by a 4 cm footprint)
    float jtV = smoothstep(0.45, 0.8, 1.0 - cav) * smoothstep(0.1, 0.5, nb.y + 0.6) * smoothstep(0.4, 0.6, hr.w);   // (round 3: in the crevices)
    float tuftB = max(smoothstep(0.35 + 0.3 * tn, 0.6 + 0.3 * tn, nb.y) * step(hr.y, 0.45), jtV * 0.9) * visR;
    tuftB *= 1.0 - smoothstep(0.015, 0.04, fp);
    float clump = smoothstep(0.42, 0.62, texture(uNoise3, p * 0.28 + 0.6).b + 0.3 * (tn - 0.5));
    float tuft = tuftB * clump * smoothstep(0.1, 0.4, vRaw) * steep;
    // ragged masses draping down the face below the ledges (edges stretched vertically); a shadow
    // band under each drape's lower edge
    vec3 pr = p * vec3(0.06, 0.035, 0.06) + 0.4, pr2 = p * vec3(0.11, 0.06, 0.11) + 0.1;
    float rag = texture(uNoise3, pr).r + 0.35 * (texture(uNoise3, pr2).r - 0.5);
    // tiers: shrubs and ferns grow on the ledges of the (dipping) bedding, ~5.5 m apart, and hang down
    // the face below each ledge in ragged vertical tongues; bare rock between
    vec2 bN = vec2(-cs.y, cs.x);
    float ph = dot(fc, bN) / 5.5 + 0.35 * gnoise(fc * 0.04 + 2.0);
    float xr = mix(0.5, wfRemap(texture(uNoise3, vec3(fc.x * 0.09, fc.y * 0.012, 0.47)).r), 1.0 - smoothstep(0.08, 0.2, fp));
    float hang = 0.12 + 0.42 * xr * smoothstep(0.2, 0.6, vRaw);
    float tw = 0.06 + fp / 5.5;
    float tierM = smoothstep(0.25, 0.55, vRaw + 0.3 * (rag - 0.5));
    float drape = smoothstep(1.0 - hang - tw, 1.0 - hang + tw, fract(ph)) * tierM;
    float drapeU = smoothstep(1.0 - hang - tw, 1.0 - hang + tw, fract(ph + 0.3 * bN.y / 5.5)) * tierM;
    float dShadow = saturate(drapeU - drape) * steep;
    float bare = (1.0 - inChute) * smoothstep(-0.2, 0.8, p.y) * (1.0 - butt) * (1.0 - 0.9 * paleB);
    float ledge = smoothstep(0.40, 0.75, n0.y + 0.35 * topB);
    // (round 3 i8: less cover on the steep face - between the falls: black rock with green on the ledges
    // and in tongues hanging below them, not a lime sponge over the whole face)
    float vegM = vRaw + -0.05 + 0.28 * ledge + 0.22 * drape + 0.12 * (rag - 0.5) + 0.22 * upF
               - 0.16 * (1.0 - smoothstep(0.25, 0.65, n0.y + 0.35 * topB));
    // leaf clumps 0.15-0.6 m (hard edges; denser inside a mass, ragged at its edges); the threshold widens
    // by the variance the footprint filters away (unresolved -> the same mean cover)
    // (uNoise3 .b = worley-fbm, cells 1 / (8 k) m: p * 0.31 -> 0.4 m clumps, p * 0.8 -> 0.16 m)
    float visK1 = 1.0 - smoothstep(0.12, 0.30, fp), visK2 = 1.0 - smoothstep(0.05, 0.12, fp);
    vec3 kV3 = vec3(1.0, kVv, 1.0);
    float kc = (visK1 > 0.0 ? (texture(uNoise3, p * 0.31 * kV3 + 0.23).b - 0.525) * visK1 : 0.0)
             + (visK2 > 0.0 ? (texture(uNoise3, p * 0.8 * kV3 + 0.61).b - 0.525) * 0.6 * visK2 : 0.0);
    float slK = max(0.035 + 0.15 * (1.0 - visK1 * visK1), 2.5 * fp);   // (r3d AA)
    // (round 3) the cover grows as cushions / mats (lush masses of fern and grass with lit tops, shaded
    // undersides and dark hollows between them - the r2 leaf-clump threshold alone read as lime camouflage
    // paint): a boss field of ~0.9 m cushions in the face coordinates; the cover's edge follows the cushion
    // outlines (at a mass's edge only the tallest), each cushion bends the normal, the hollows are occluded.
    // Unresolved -> the field's mean (the threshold widens: the same mean cover)
    const float Sc = 0.9;
    float visC1 = 1.0 - smoothstep(0.10 * Sc, 0.28 * Sc, fp);
    float cn = 0.0, idC = 0.5;
    vec3 gC = vec3(0.0);
    if (visC1 > 0.0) {
        gFacet = 0.0;                                         // (cushions stay rounded)
        // (review rv5) cushions stretched down the face (hanging tongues, kVv)
        vec3 c1 = wfBoss(vec2(fq.x, fq.y * kVv) + vec2(17.3, 5.9), Sc, 0.30, 41.0, 0.9, idC);
        cn = c1.x / 0.30;                                     // ~ -1.25 (hollow) .. 1 (top)
        gC = (A * c1.y + Ty * c1.z * kVv) * visC1;
    }
    float cuM = mix(0.0, cn * 0.22, visC1);
    float slC = slK + 0.12 * (1.0 - visC1);
    // (falls6) the cover sits on the up-facing facets of the relief (ledges), the steep / down-facing ones stay bare;
    // its edge is the crisp clump / facet threshold, the 5-10 m cover field (vegM) only moves the threshold (the
    // cv ramp blended rock and moss over metres: soft lime / black camouflage patches)
    float argM = kc * mix(1.0, 0.5, visC1) + cuM + 0.10 * (rag - 0.5) + 1.2 * fyR;
    float thrN = mix(0.30, -0.20, smoothstep(0.30, 0.52, vegM));
    mossW = smoothstep(thrN - slC, thrN + slC, argM) * smoothstep(0.22, 0.36, vegM);
    // (falls6) moss / grass tufts along the cracks inside the vegetated areas
    float crkM = (1.0 - smoothstep(1.2 * wC1, 2.6 * wC1 + fp, crkD.x)) * (1.0 - smoothstep(0.04 * S, 0.10 * S, fp)) * visR;
    mossW = max(mossW, crkM * smoothstep(0.15, 0.45, vRaw) * smoothstep(-0.25, 0.1, fyR));
    mossW = max(mossW, tuft * 0.9) * bare;
    // cushion shading: normal over the dome, hollows dark, per-cushion tone (yellow-lime grass .. dark fern)
    vec3 gCt = gC - n0 * dot(gC, n0);
    nv = normalize(nv - gCt * 0.9);                            // (review rv1: 1.7 -> 0.9, lit domes read as lime broccoli)
    occV *= mix(1.0, mix(0.47, 1.0, smoothstep(-0.9, 0.55, cn)), visC1);
    vec4 hC = hash42i(ivec2(int(idC * 65536.0), 43));
    veg *= mix(vec3(1.0), mix(vec3(0.84, 0.88, 0.80), vec3(1.10, 1.08, 1.0), hC.x), visC1);   // (round 3 i7: milder)
    // none under overhangs / on the rolled undersides of the blocks, little in the spray core
    mossW *= (1.0 - 0.9 * smoothstep(0.0, -0.4, nb.y) * steep * visR) * (1.0 - 0.85 * sprayC);
    // (review r2, fix 12) beyond ~270 m the 3D plants fade out (FF_MAX_DIST): their cover moves into the moss
    mossW = max(mossW, 0.45 * smoothstep(0.7 * FF_MAX_DIST, FF_MAX_DIST, t) * smoothstep(0.3, 0.6, vRaw) * bare * (1.0 - sprayC));
    // the rim of the step: dark shrub / young spruce mass, few lime tips
    float rimW = wfRimW(uv.y, uv.x - wfFaceLine(uv.y), wfFaceW(hsB));
    veg = mix(veg, mix(vec3(0.016, 0.026, 0.014), veg, 0.15 + 0.2 * smoothstep(0.7, 0.9, cl)), rimW);
    nb = normalize(mix(nb, nv, mossW));
    // cast shadows of the blocks: turn the normal away from the sun (sky light stays)
    nb = normalize(nb - uSunDir * max(dot(nb, uSunDir), 0.0) * sh * (1.0 - mossW));
    n = nb;
    // wet sheen (terrain_shade: spec 0.7 x wetW): strongest on the up-facing facets the water runs over
    wetW = (max(inChute, fringe * 0.7) * 0.85 + 0.12) * (1.0 - mossW) * mix(0.65, 1.0, smoothstep(0.0, 0.5, nb.y));
    float under = butt * (1.0 - smoothstep(0.2, 1.8, p.y));    // overhanging foot of the buttress
    float downF = smoothstep(0.05, -0.45, nb.y) * steep * visR;   // undersides see the rock below, not the sky
    // undersides / shadowed hollows / open joints in shadow: almost no light reaches in (the ambient's
    // valley-wall fill must not lift them either -> albedo, not only sky occlusion). (review r2) the joints
    // darken only on their shadowed / down-facing side, not all round each block
    float cavA = mix(0.78, 1.0, visR) * mix(1.0, mix(1.0, 0.12, downF) * mix(1.0, 0.25, sh), visR * steep)
               * mix(1.0, 0.1, jnt * max(downF, sh));            // (r3a: darker - p5 ~4, render 15)
    vec3 alb = mix(rock * cavA * (1.0 - 0.85 * crk), veg, mossW);
    occ = mix(1.0, occJ * mix(1.0, 0.35, downF) * (1.0 - 0.6 * crk), 1.0 - mossW) * mix(1.0, occV, mossW)
        * mix(1.0, 0.3, dShadow * (1.0 - mossW)) * mix(1.0, 0.35, under);
    return alb;
}

// the rock material's flat mean (reflections in the pool: no relief, no joints; mossW = mean cover)
vec3 wfRockCheap(vec3 p, vec3 n0, out float mossW) {
    mossW = 0.30 * smoothstep(0.3, 0.8, n0.y + 0.3);
    return mix(vec3(0.019, 0.020, 0.018), vec3(0.055, 0.075, 0.018), mossW);
}

// where the falls' rock material applies (0..1): the cliff face of the step, the buttress and the
// spray zone; n0 (optional) = geometric normal: the flat tops of the rim stay vegetated
float fallsWetness(vec3 p, vec3 n0) {
    if (uFallsOn == 0) return 0.0;
    vec2 uv = wfUV(p.xz);
    if (uv.x < -50.0 || uv.x > 80.0 || uv.y < -100.0 || uv.y > 70.0) return 0.0;
    float cl = wfCliff(uv.y);
    float fl = wfFaceLine(uv.y);
    float hs = wfStepH(uv.y);
    float uf = uv.x - fl;
    float W = wfFaceW(hs);
    // the cliff face (not the forested rim above it), fading where the cliff turns into banks
    float step_ = smoothstep(-4.0, -1.0, uf) * (1.0 - smoothstep(W + 0.5, W + 4.0, uf)) * smoothstep(3.0, 9.0, hs) * cl;
    step_ *= 1.0 - smoothstep(hs + 0.5, hs + 2.5, p.y);
    step_ *= 1.0 - 0.85 * smoothstep(0.66, 0.9, p.y / hs + 0.3 * (texture(uNoise3, p * 0.025 + 0.4).r - 0.5));
    step_ *= 1.0 - 0.85 * smoothstep(0.46, 0.64, n0.y + 0.2 * (texture(uNoise3, p * 0.03 + 0.9).r - 0.5)) * smoothstep(4.0, 8.0, p.y);
    // the buttress wall
    float yb = wfButtH(uv.y, hs);
    float bt = wfButt(uv.y) * smoothstep(-5.0, -2.0, uf) * (1.0 - smoothstep(WF_BUTT_W + 2.5, WF_BUTT_W + 6.0, uf))
             * (1.0 - smoothstep(yb + 0.5, yb + 2.5, p.y));
    // spray zone around the two falls
    float spray = 0.0;
    for (int k = 0; k < 2; k++) {
        float vc = wfChuteV(k, uv.x);
        float dv = abs(uv.y - vc);
        spray = max(spray, (1.0 - smoothstep(wfChuteHW(k, uv.x) + 1.0, wfChuteHW(k, uv.x) + 9.0, dv)) * (1.0 - smoothstep(W, W + 8.0, uf)) * smoothstep(-8.0, -1.0, uf));
    }
    return max(max(spray, step_), bt);
}
float fallsWetness(vec3 p) { return fallsWetness(p, vec3(0.0, 0.0, 0.0)); }

// the angular slabs of the low apron north of the falls (weight 0..1; dp / aw for wfSlab)
float fallsApron(vec3 p, out float dp, out float aw) {
    dp = 0.0; aw = 0.0;
    if (uFallsOn == 0) return 0.0;
    vec2 uv = wfUV(p.xz);
    if (uv.y > -8.0 || uv.y < -95.0 || uv.x > 12.0 || uv.x < -60.0) return 0.0;
    float apron = wfApronZ(uv.y);
    float uf = uv.x - wfFaceLine(uv.y);
    dp = -uf;
    aw = wfApronW(uv.y) + 1.5 * gnoise(p.xz * 0.05);
    return apron * smoothstep(1.0, -1.0, uf) * (1.0 - smoothstep(aw + 4.0, aw + 7.0, dp));
}

vec3 pebbles(vec2 p, out vec3 bump);     // terrain_shade.glsl

// shore slabs: weathered dark-grey rock with sharp bevelled edges and sub-facets, pale and dark
// lichen, grass / moss tufts in the joints, wet and dark at the waterline. Returns albedo; bends n.
vec3 wfApronShade(vec3 p, vec3 n0, float t, float dp, float aw, float wgt, inout vec3 n, out float occ, out float wetA) {
    float fp = t * gPixAng;
    vec4 sv;
    float sh = wfSlab(p.xz, dp, aw, sv);
    vec4 h = hash42i(ivec2(int(sv.w * 65536.0), 5));
    float onSlab = step(0.22, h.x) * smoothstep(0.02, 0.07, sv.z) * smoothstep(0.10, 0.18, sh + 0.08 * (texture(uNoise3, p * 0.2).r - 0.5));
    // bevel toward the joint and sub-facets (~1.1 m)
    vec2 ed = normalize(sv.xy + 1e-4);
    float bev = 1.0 - smoothstep(0.04, 0.13, sv.z);
    vec4 s2 = wfVor(p.xz * 0.9 + 1.3);
    vec4 h2 = hash42i(ivec2(int(s2.w * 65536.0), 9));
    float vis = 1.0 - smoothstep(0.03, 0.10, fp);
    vec3 ns = normalize(n0 + vec3(ed.x, 0.0, ed.y) * bev * 1.2 * onSlab
                        + vec3(h2.x - 0.5, 0.0, h2.y - 0.5) * 0.7 * vis * onSlab
                        + (texture(uNoise3, p * 0.8).rgb - 0.5) * 0.35 * vis
                        + (texture(uNoise3, p * 2.3 + 0.6).rgb - 0.5) * 0.5 * (1.0 - smoothstep(0.01, 0.03, fp)));
    // rock tone: dark grey, lit tops paler with lichen; dark crustose speckles
    float g = texture(uNoise3, p * 0.45 + 0.2).r;
    vec3 rock = vec3(0.075, 0.074, 0.068) * mix(0.65, 1.45, h.y) * mix(0.8, 1.2, h2.w) * mix(0.8, 1.2, g);
    float lich = smoothstep(0.60, 0.68, texture(uNoise3, p * 0.11 + 1.7).r + 0.25 * (texture(uNoise3, p * 0.5).b - 0.5)) * smoothstep(0.6, 0.9, ns.y);
    rock = mix(rock, vec3(0.22, 0.225, 0.19), lich * 0.6);
    rock *= 1.0 - 0.7 * (1.0 - smoothstep(0.015, 0.05, sv.z));
    // fine fracture network and grain up close
    float cwA = max(0.012, fp * 0.8) * 0.9;
    rock *= 1.0 - 0.65 * (1.0 - smoothstep(cwA * 0.5, cwA * 1.5, s2.z)) * step(0.4, h2.z) * vis;
    rock *= mix(1.0, clamp(1.0 + 1.6 * (texture(uNoise3, p * 1.5 + 0.9).r - 0.48), 0.45, 1.5), 1.0 - smoothstep(0.01, 0.04, fp));
    // grass and moss in the joints and the low gaps, some moss on slab tops
    vec3 nv; float cl, occV;
    vec3 veg = wfVegetation(p, n0, fp, 0.3, nv, cl, occV);
    // between the slabs: warm golden-grey cobbles at the waterline, grass tufts higher up
    vec3 pb;
    vec3 cob = pebbles(p.xz, pb) * vec3(1.06, 0.98, 0.80);
    float gapV = (1.0 - onSlab) * smoothstep(0.30, 0.65, p.y + 0.25 * (cl - 0.5));
    float topV = onSlab * smoothstep(0.62, 0.70, texture(uNoise3, p * vec3(0.06, 0.1, 0.06) + 0.8).r + 0.2 * (h.w - 0.5));
    float vegW = max(gapV * smoothstep(0.3, 0.6, cl + 0.2), topV);
    vec3 alb = mix(mix(cob, rock, onSlab), veg, vegW);
    ns = normalize(ns + pb * (1.0 - onSlab) * (1.0 - vegW) * vis);
    ns = normalize(mix(ns, nv, vegW));
    n = normalize(mix(n, ns, wgt));
    wetA = 1.0 - smoothstep(0.0, 0.35, p.y);
    alb *= mix(1.0, 0.5, wetA);
    occ = mix(1.0, 0.25, (1.0 - smoothstep(0.015, 0.06, sv.z)) * step(0.22, h.x)) * mix(1.0, occV, vegW);
    return alb;
}

// ---------------------------------------------------------------------------------------------
// falls flora (round 2): real 3D shrubs, fern clumps and grass tufts on the ledges of the rock step,
// on its rim and on the steep banks beside it (plants with silhouettes against the dark rock,
// lit tops, dark undersides, cast shadows). One instance per FF_CELL cell of the falls frame (u, v)
// (or none), anchored on the near height map. scene.frag render(): after the primary trace, ffTrace
// marches ffSDF up to the primary hit (t < FF_MAX_DIST; hit kind K_FALLSV, info = plant centre + packed
// cell) and shadeFlora shades a flora hit. terrain_shade's falls block darkens the rock with ffShadow.
#define K_FALLSV 7
#define FF_CELL 3.6
#define FF_JIT 0.2
#define FF_REACH 2.7         // max horizontal reach of a plant from its cell's centre point (m)
#define FF_NLOOP (4 + min(uDebug, 0))   // (not a constant: keeps the loop rolled, compile time)

vec2 ffXZ(vec2 uv) { return uWF.xy - uWFDir * uv.x + vec2(-uWFDir.y, uWFDir.x) * uv.y; }
// (along shore, up, outward) components of a world vector
vec3 ffLocal(vec3 q) { return vec3(dot(q.xz, vec2(-uWFDir.y, uWFDir.x)), q.y, dot(q.xz, uWFDir)); }

// the step's along-shore profile at the centre of plant column cy (shared by the 2 cells of a column)
struct FfCol { float v, hs, fl, cl, W, ybt, bt; };
FfCol ffCol(int cy) {
    FfCol o;
    o.v = (float(cy) + 0.5) * FF_CELL;
    o.hs = wfStepH(o.v); o.fl = wfFaceLine(o.v); o.cl = wfCliff(o.v); o.W = wfFaceWv(o.hs, o.v);
    o.ybt = wfButtH(o.v, o.hs); o.bt = wfButt(o.v);
    return o;
}
// the plant of cell c (falls frame): false if none. ctr = centre (world), r = radii (along shore, up,
// outward), h = hashes; kind (h.w): < 0.42 shrub, < 0.8 fern rosette, < 1 grass tuft, 2..3 drape (fern /
// grass curtain on the steep face), 4..5 rock block (review r2, fix 8: base boulders at the cliff foot,
// blocks standing out of the face - real silhouettes and cast shadows)
bool ffPlantC(ivec2 c, FfCol col, float foot, out vec3 ctr, out vec3 r, out vec4 h) {
    ctr = vec3(0.0); r = vec3(1.0);
    h = hash42i(c + ivec2(7193, 3301));
    if (col.v < -56.0 || col.v > 46.0 || col.hs < 6.0) return false;
    // (review r2, fix 12) no hard cut at FF_MAX_DIST: the plants shrink away over its last 30 %
    float fd = 1.0 - smoothstep(0.7 * FF_MAX_DIST, FF_MAX_DIST, foot / gPixAng);
    if (fd <= 0.0) return false;
    vec2 uv = (vec2(c) + 0.5 + (h.xy - 0.5) * (2.0 * FF_JIT)) * FF_CELL;
    float v = uv.y;
    float uf = uv.x - col.fl;
    float W = col.W;
    vec2 xz = ffXZ(uv);
    vec2 dirO = vec2(uWFDir.x, uWFDir.y);
    // (review r2, fix 8) dark wet boulders (1.5-3 m) at the foot of the face: in front of the buttress and
    // at fall A's base north of its plunge; partly in the water, standing on a submerged rubble pile
    if (uf > -3.2 && uf < 1.4 && col.cl > 0.3 && v > -25.0 && v < 9.0) {
        float vA = uFallK[WF_NK - 1].y, hwA = uFallK[WF_NK - 1].z;
        bool plunge = abs(v - vA) < hwA + 1.2 || abs(v - uFallK[2 * WF_NK - 1].y) < uFallK[2 * WF_NK - 1].z + 1.0;
        if (!plunge && h.z < 0.55) {
            float R = mix(0.7, 1.15, fract(h.y * 5.3));
            r = R * vec3(mix(0.95, 1.25, h.x), mix(0.6, 0.85, fract(h.x * 3.7)), mix(0.8, 1.1, fract(h.y * 9.1)));
            float fs = (1.0 - smoothstep(0.15, 0.35, foot / R)) * fd;
            if (fs <= 0.0) return false;
            r *= fs;
            // (little jitter: the block's corners stay inside FF_REACH of the cell centre)
            vec2 xzB = ffXZ((vec2(c) + 0.5 + (h.xy - 0.5) * 0.1) * FF_CELL);
            float y0 = max(terrainH(xzB, false), -0.5 - 0.4 * h.x);
            ctr = vec3(xzB.x, y0 + mix(0.15, 0.5, fract(h.x * 3.1)) * r.y, xzB.y) + vec3(dirO.x, 0.0, dirO.y) * (0.25 * r.z);
            h.w = 4.0 + fract(h.w * 5.7) * 0.5;            // kind: boulder (4 .. 4.5)
            return true;
        }
        if (uf < 0.2) return false;
    }
    if (h.z > 0.9 || uf < 0.2 || uf > W + 7.0) return false;
    // none in the water: the chutes (and fall A's stream bed behind the lip)
    // (r3: only the water's own width + a margin - plants right beside the sheets)
    for (int k = 0; k < 2; k++)
        if (abs(v - wfChuteV(k, uv.x)) < wfChuteHW(k, uv.x) * (k == 0 ? 0.62 : 0.6) + (k == 0 ? 1.0 : 0.6)) return false;
    // planted: banks and the rim densely, the bare face in ~8 m patches (tiers, ledges)
    float nz = texture(uNoise3, vec3(uv.x * 0.012, uv.y * 0.0085, 0.57)).g;
    float pres = mix(0.90, 0.10 + 0.8 * smoothstep(0.42, 0.62, nz), col.cl);
    pres = max(pres, 0.8 * smoothstep(W - 2.5, W - 0.5, uf) * (1.0 - smoothstep(W + 4.0, W + 7.0, uf)));
    if (h.z > max(pres, 0.95 * col.cl)) return false;
    float y0 = terrainH(xz, false);
    if (y0 < 0.8) return false;
    // (session 3) plants grow in groups (~10-15 m) that merge into one ragged mass, with open ground /
    // bare rock between the groups: an even scatter of separate bushes read as a topiary garden (film
    // shot 2: rows of round lime balls on the banks)
    float grp = wfRemap(texture(uNoise3, vec3(uv.x * 0.055, uv.y * 0.045, 0.83)).r);
    float inG = smoothstep(0.22, 0.50, grp + 0.25 * (h.y - 0.5));
    pres *= mix(0.25, 1.0, inG);
    // on the face: also where the shading's foliage mask is (ledges / tiers of shrubs and ferns; not
    // thinned by the groups: the ledges are lush)
    float vr = wfVegRaw(vec2(v, y0), col.hs);
    if (col.cl > 0.05) pres = max(pres, 0.85 * col.cl * smoothstep(0.42, 0.70, vr));
    // (session 4) the steep face: curtains of fern / grass hang from most ledges where the face is grown over
    // (review r2: presence 0.12-0.88 -> 0.35-0.95: about half of the face is covered)
    float onFace = col.cl * smoothstep(0.8, 2.0, uf) * (1.0 - smoothstep(W - 1.4, W + 0.2, uf));
    if (onFace > 0.5) pres = max(pres, mix(0.35, 0.95, smoothstep(0.30, 0.62, vr)));
    // (review r2, fix 8) blocks standing out of the face (0.6-2.5 m, 0.3-1.2 m proud) where it is bare
    // (a few % of the face cells): real silhouettes and cast shadows on the cliff
    if (onFace > 0.5 && h.z > pres && h.z < pres + 0.14 * (1.0 - smoothstep(0.35, 0.6, vr)) && y0 > 1.5) {
        float R = mix(0.3, 1.25, fract(h.y * 7.9));
        float prot = mix(0.3, 1.2, fract(h.x * 5.1));
        r = vec3(R * mix(0.9, 1.5, fract(h.w * 3.3)), R * mix(0.55, 0.9, fract(h.w * 9.7)), max(prot, 0.5 * R) * 0.75 + 0.25);
        r = min(r, vec3(1.1, 1.1, 0.9));
        float fs = (1.0 - smoothstep(0.15, 0.35, foot / max(R, 0.3))) * fd;
        if (fs <= 0.0) return false;
        r *= fs;
        vec2 xzB = ffXZ((vec2(c) + 0.5 + (h.xy - 0.5) * 0.1) * FF_CELL);
        ctr = vec3(xzB.x, terrainH(xzB, false), xzB.y) + vec3(dirO.x, 0.0, dirO.y) * (prot - r.z);
        h.w = 4.5 + fract(h.w * 5.7) * 0.5;                // kind: face block (4.5 .. 5)
        return true;
    }
    // the buttress wall stays bare (ferns only on its top)
    if (h.z > pres * (1.0 - 0.9 * col.bt * (1.0 - smoothstep(col.ybt - 0.5, col.ybt + 0.5, y0)))) return false;
    float R;
    float gs = mix(0.7, 1.25, inG);                      // bigger plants inside a group (they overlap)
    // (session 4) on the steep face: hanging drapes of fern / grass (curtains of fronds hanging from
    // every ledge down the dark rock, lit on top, a fringed lower edge). Sheared along the face (ffPlantD),
    // anchored at the cell's ground point, hanging below it
    if (onFace > 0.5) {
        // (centred on the cell's face point: the shear keeps a 2 x 2.7 m curtain inside FF_REACH)
        // (review r2: bigger, taller curtains - the cells are 3.6 m horizontally, ~10 m of the 70 deg face: the
        // vertical extent costs no reach, so the curtains hang 2-5 m down the face)
        float kf = fract(h.w * 7.7);
        if (kf < 0.12) {   // (round 3: 0.65 -> 0.3; review rv5: 0.12 - the mats read as lime ovals stuck on the face, film shot 2)
            // (r3) a mat of ferns / grass / moss hugging the face: up to 3.7 x 7 m, a ragged lobed outline with
            // tongues hanging below, a bumpy cushioned surface 0.25-0.55 m proud (the cliff is half
            // covered by lush hanging vegetation; one mat per 3.6 m cell = ~10 m of the 70 deg face)
            R = mix(1.5, 2.2, fract(h.y * 5.1)) * mix(0.85, 1.0, inG);
            r = vec3(R, min(R * mix(1.2, 1.9, fract(h.x * 3.3)), 3.6), mix(0.16, 0.36, fract(h.z * 9.1)));   // (review rv5: flatter, 0.25-0.55 -> 0.16-0.36)
            float fs = (1.0 - smoothstep(0.20, 0.42, foot / R)) * fd;
            if (fs <= 0.0) return false;
            r *= fs;
            h.w = 3.0 + 0.5 * fract(h.w * 3.9);          // kind: mat (3 .. 3.5)
            vec2 xzM = ffXZ((vec2(c) + 0.5 + (h.xy - 0.5) * 0.06) * FF_CELL);
            float yM = terrainH(xzM, false);
            ctr = vec3(xzM.x, yM - 0.25 * r.y, xzM.y) + vec3(dirO.x, 0.0, dirO.y) * (0.05 + 0.09 * r.y);
            return true;
        }
        // (r3: the hanging strand drapes are retired - they read as haystacks / jellyfish)
        // (review r2) a fern rosette growing out of the face: fronds radiate around an axis tilted out of
        // the rock (up 37 deg), arch and droop over the dark rock (the cliff's lush fern tufts)
        R = mix(0.75, 1.3, fract(h.y * 5.1)) * mix(0.85, 1.0, inG);
        r = R * vec3(1.25, 1.0, 0.95);
        float fs = (1.0 - smoothstep(0.20, 0.42, foot / R)) * fd;
        if (fs <= 0.0) return false;
        r *= fs;
        h.w = 2.0 + 0.5 * fract(h.w * 3.9);          // kind: face fern (2 .. 2.5)
        ctr = vec3(xz.x, y0 + 0.1 * r.y, xz.y) + vec3(dirO.x, 0.0, dirO.y) * (0.15 + 0.45 * r.z);
        return true;
    }
    if (h.w < 0.42)      { R = mix(0.75, 1.25, fract(h.x * 7.3)) * gs; r = R * vec3(1.1, 0.72, 1.0); }
    else if (h.w < 0.8)  { R = mix(0.55, 0.95, fract(h.y * 5.1)) * gs; r = R * vec3(1.3, 0.45, 1.3); }
    else                 { R = mix(0.40, 0.65, fract(h.x * 3.7)) * gs; r = R * vec3(0.9, 1.1, 0.9); }
    R = min(R, 1.25); r = min(r, vec3(1.45));
    // sub-pixel plants shrink away (the terrain's foliage shading takes over)
    float fs = (1.0 - smoothstep(0.20, 0.42, foot / R)) * fd;
    if (fs <= 0.0) return false;
    r *= fs;
    ctr = vec3(xz.x, y0 + 0.30 * r.y, xz.y) + vec3(dirO.x, 0.0, dirO.y) * (0.3 * R);
    return true;
}

// (review r2, fix 3) blades: one ribbon per frond / blade, found by folding the angle around the plant's
// base into the frond's sector (the two nearest fronds are tested). qb = point relative to the base
// (along shore, up, outward). Fern fronds (rosette): arch up and droop (centre line y = L (rise s -
// droop s^2), s = 0..1 along the frond), 0.12 L wide tapering to the tip, leaflet notches when resolved.
// Grass blades (rosette = 0): nearly upright, curving out (r = 0.45 L s^2), thin. Widths never below
// ~half a pixel (no sparkling slivers); the gaps between the fronds show what is behind.

// (review r2, fix 5) pale granite steps right of fall B's foot (its water cascades over them): 4 blocks,
// deterministic, 0.5-1 m risers. Distance (1e9 outside their box); id = step index

// distance to one plant / block

// (round 2, perf) the flora is marched on its own AFTER the primary trace (scene.frag render()), from
// where the ray enters the flora's box up to the primary hit tmax: the hit distance (1e9 = none) and
// info as ffSDF. (Inside trace()'s march loop it was inlined into all three trace() call sites and its
// registers weighed on every march step of every view, ~x4 frame time.) Off for the prep / kind
// renders of the lab harnesses (uDebug 1, 48-51) and timing toggle 70.
#define FF_STEPS 72

// sun path length through a plant's ellipsoid from p (0 = none)
float ffSunPath(vec3 p, vec3 ctr, vec3 r) {
    vec3 s = ffLocal(p - ctr) / r, ds = ffLocal(uSunDir) / r;
    float a = dot(ds, ds), b = dot(s, ds), c = dot(s, s) - 1.0;
    float disc = b * b - a * c;
    if (disc <= 0.0) return 0.0;
    float sq = sqrt(disc);
    float tE = (-b + sq) / a, tN = (-b - sq) / a;
    return max(tE - max(tN, 0.0), 0.0);
}
// shadow the plants / blocks cast on p (0..1): the cells around two points toward the sun (0.8 m and
// 2.4 m: review r2 - plants / blocks up to ~2.7 m tall cast their whole shadow); skip = own cell
float ffShadowX(vec3 p, float t, ivec2 skip) {
    if (uFallsOn == 0 || t > FF_MAX_DIST) return 0.0;
    float foot = t * gPixAng;
    float sh = 0.0;
    ivec2 cPrev = ivec2(-99999);
    for (int s = 0; s < 1 + min(uDebug, 0); s++) {
        vec3 ps = p + uSunDir * 1.5;      // (r3d, perf: one point, 1.5 m toward the sun - the 2 x 2 cells around it)
        vec2 uv = wfUV(ps.xz);
        if (uv.y < -60.0 || uv.y > 50.0 || uv.x < -10.0 || uv.x > 52.0) continue;
        ivec2 c0 = ivec2(floor(uv / FF_CELL - 0.5));
        if (c0 == cPrev) continue;
        FfCol cA = ffCol(c0.y), cB = ffCol(c0.y + 1);
        for (int j = 0; j < FF_NLOOP; j++) {
            ivec2 c = c0 + ivec2(j & 1, j >> 1);
            if (c == skip) continue;
            // (the second set: only the cells the first did not test)
            if (s == 1 && abs(c.x - cPrev.x) <= 1 && abs(c.y - cPrev.y) <= 1 && c.x >= cPrev.x && c.y >= cPrev.y) continue;
            vec3 ctr, r; vec4 h;
            if (!ffPlantC(c, (j >> 1) == 0 ? cA : cB, foot, ctr, r, h)) continue;
            // (fronds / blades / strands are sparse: a lighter shadow than a leafy shrub or a rock block)
            float den = h.w >= 4.0 ? 9.0 : (h.w < 0.42 ? 3.5 : 2.0);
            sh = max(sh, 1.0 - exp(-den * ffSunPath(p, ctr, r)));
        }
        cPrev = c0;
    }
    return sh * 0.92;
}
float ffShadow(vec3 p, float t) { return ffShadowX(p, t, ivec2(-99999)); }

// the pale steps under fall B (ffStepsB): pale granite, wet; fall B's water threads over the treads

// ---------------------------------------------------------------------------------------------
// inflow stream: boulders of the stream bed and the open floodplain (bake_features.frag
// carves them: voronoiV(p * 0.36 + 13) cells, channel half width = 0.6 x the floodplain one)



                 // (round 3) arc length (m) of each polyline point from the first (render.py)
float gStreamS = 0.0;                       // streamDist: arc length of the nearest axis point (upstream +) and the
float gStreamD = 0.0;                       // signed distance across (+ = left of the downstream direction)
float gStreamHead = 0.0;                    // stream boulder heads in the water (water_shade.glsl stream block)
vec3 gStreamHeadN = vec3(0.0, 1.0, 0.0);
// (round 3) heaped 3D white water (water_shade.glsl stream block): hit flag, normal, hollow occlusion, weight
float gWWHit = 0.0, gWWOcc = 1.0, gWWVis = 0.0, gWWSh = 1.0, gWWRock = 0.0, gWWH = 0.0;
vec3 gWWN = vec3(0.0, 1.0, 0.0);
vec3 gWWP = vec3(0.0);     // (i12) world position of the 3D hit (the boulders' texture: not projected from the water plane)
float gWWBo = 1.0;         // (review rv3) share of the 3D boulders in wwField (water_shade: off where the parallax march fails)

// (round 3) the stream's tumbling white water as a heightfield above the water level (film shot 4: a
// cascade of heaped, lumpy aerated water broken by boulders, not white paint on a slope). pa = (along the flow
// downstream, across) in m - curvilinear stream coordinates (streamDist: arc length / signed distance), so the
// pattern is continuous across the polyline's segments and the same for every pixel. Heaps ~0.8 m along x 0.5 m
// across (worley bumps) with ~0.3 m bubbles (vis = their visibility), boiling slowly (the noise's third axis follows
// time), H = the local white-water height (m). Boulders (~45 % of the 3.2 x 2.6 m cells): rounded wet heads
// 0.7-1.9 m across standing 0.4-0.9 m out of the water, white water piled up on their upstream side, a dark trough
// in their lee. rock = 1 where the boulder is the surface
float gWWHw = 2.0;          // (review rv5) wet channel half width (m) at the pixel (water_shade.glsl stream block)
// (review rv5) the white water is a ribbon about half the channel wide winding between the boulders (its line
// meanders across the channel over ~10-20 m), dark glassy water and boulders either side - the r4 channel was white
// from bank to bank (a painted staircase in film shot 4). pa = (along, across) curvilinear (m); 1 inside .. 0 outside
// (rv6, perf) the ribbon's centre line / half width are evaluated once per pixel (wwRibbon) and reused by every wwField
// sample of the parallax march (it spans < ~2 m along the flow; the line bends over 6-18 m); gWWRib = 0 -> no ribbon
float gRibM = 0.0, gRibW = 1e3, gWWRib = 0.0;

// distance to the stream axis and the floodplain half width there (x = 1e9 when far away); the
// points are evenly spaced in z (features.place), so only the segments around p.z are tested
// (review r2) gStreamDir = (downstream direction xz, bed slope) of the nearest segment
vec3 gStreamDir = vec3(0.0, -1.0, 0.0);
vec2 streamDist(vec2 p) {
    if (uStreamN < 2 || p.x < uStreamBox.x || p.y < uStreamBox.y || p.x > uStreamBox.z || p.y > uStreamBox.w) return vec2(1e9, 1.0);
    float best = 1e9, hw = 1.0;
    float dz = uStream[1].y - uStream[0].y;
    int k0 = int(floor((p.y - uStream[0].y) / max(dz, 1.0)));
    for (int j = -2; j <= 2; j++) {
        int i = clamp(k0 + j, 0, uStreamN - 2);
        vec4 s0 = uStream[i], s1 = uStream[i + 1];
        vec2 ab = s1.xy - s0.xy;
        float tt = clamp(dot(p - s0.xy, ab) / dot(ab, ab), 0.0, 1.0);
        float d = length(p - (s0.xy + ab * tt));
        if (d < best) {
            best = d; hw = mix(s0.w, s1.w, tt);
            float lab = max(length(ab), 1e-3);
            gStreamDir = vec3(-ab / lab, max(s1.z - s0.z, 0.0) / max(lab, 1.0));
            gStreamS = uStreamS[i] + tt * lab;
            gStreamD = dot(p - (s0.xy + ab * tt), vec2(ab.y, -ab.x) / lab);
        }
    }
    return vec2(best, hw);
}

// pale granite boulders in and along the stream (1-2.4 m, in clusters): weight 0..1 (inside a boulder's
// footprint, only where the 1 m map carved them - same cells as bake_features.frag; fades out once a
// boulder is a few pixels wide); returns albedo, bends n
vec3 rkMidTex(vec3 q, float fp, float steep, out float pk);    // shade_rocks.glsl (rocks5)
vec3 rkFacetG(vec3 p, vec3 n, float fp);
float streamBoulder(vec3 p, vec3 n0, float t, inout vec3 n, out vec3 alb, out float occ) {
    alb = vec3(0.0); occ = 1.0;
    float nb = nearBlend(p.xz);
    if (nb >= 1.0 || wfOff(3)) return 0.0;
    vec2 sd = streamDist(p.xz);
    float hwC = sd.y * 0.6;
    if (sd.x > hwC * 4.3) return 0.0;
    float fp = t * gPixAng;
    vec4 c = voronoiV(p.xz * 0.25 + 13.0);
    float hsh = fract(c.w * 7.3);
    float clus = smoothstep(-0.12, 0.18, gnoise(p.xz * 0.035 + 1.3));
    float th = 1.0 - mix(0.10, 0.80, clus);
    if (hsh < th) return 0.0;
    // (round 2: few bed boulders inside the wet channel, ~0.48 x the floodplain half width; bake_features)
    if (fract(c.w * 11.3) < 0.15 * (1.0 - smoothstep(0.26 * sd.y, 0.53 * sd.y, sd.x))) return 0.0;   // (round 3: 0.38 -> 0.15, bake_features)
    float R = 0.25 + 0.35 * fract(c.w * 3.1);
    float e = c.z / R;
    float w = (1.0 - smoothstep(0.82, 1.02, e)) * (1.0 - smoothstep(hwC * 2.8, hwC * 4.2, sd.x)) * smoothstep(th, th + 0.04, hsh)
            * (1.0 - smoothstep(0.3, 0.8, fp * 0.25 / R)) * (1.0 - nb);
    if (w <= 0.0) return 0.0;
    vec4 h = hash42i(ivec2(int(c.w * 65536.0), 13));
    // rounded-angular block: steep sides (bake: profile exponent 0.25), a tilted top, and 2-4 broad
    // facets per boulder (~0.8 m, each its own plane; the tilt eases in from the facet edge)
    vec2 dir = c.xy / max(c.z, 1e-3);
    vec4 fv = wfVor(p.xz * 1.25 + h.zw * 37.0);
    vec4 fh = hash42i(ivec2(int(fv.w * 65536.0), 29));
    float visF = (1.0 - smoothstep(0.04, 0.12, fp)) * smoothstep(0.0, 0.06 + 2.0 * fp, fv.z);
    vec3 nbn = normalize(n0 + vec3(dir.x, 0.0, dir.y) * smoothstep(0.25, 1.0, e) * 2.2 + (vec3(h.x, 0.0, h.y) - vec3(0.5, 0.0, 0.5)) * 0.6
                        + vec3(fh.x - 0.5, 0.0, fh.y - 0.5) * 0.9 * visF
                        + (texture(uNoise3, p * 0.6 + h.z).rgb - 0.5) * 0.4 * (1.0 - smoothstep(0.03, 0.10, fp)));
    nbn = normalize(nbn + 1.6 * rkFacetG(p, nbn, fp));         // (rocks5 M4) chipped 25 cm facets (shot 4: smooth clay mounds)
    n = normalize(mix(n, nbn, w));
    float g = mix(0.5, texture(uNoise3, p * 0.5 + h.w).r, 1.0 - smoothstep(0.05, 0.15, fp));
    // pale granite (lit ~#c4bfb4), its shaded sides dark (~#494d4b: the steep sides + contact occlusion)
    alb = vec3(0.30, 0.292, 0.272) * mix(0.75, 1.15, h.z) * mix(0.82, 1.18, wfRemap(g));
    // grey-green and dark crustose lichen in irregular patches (no dots)
    float lich = smoothstep(0.58, 0.66, texture(uNoise3, p * 0.15 + 2.3).r + 0.3 * (h.w - 0.5)) * smoothstep(0.3, 0.8, nbn.y);
    alb = mix(alb, vec3(0.17, 0.185, 0.13), lich * 0.6);
    float dark = smoothstep(0.55, 0.85, wfRemap(texture(uNoise3, p * 0.21 + 0.4).g) * 0.7 + 0.3 * texture(uNoise3, p * 0.7).r);
    alb *= 1.0 - 0.35 * mix(dark, 0.3, smoothstep(0.03, 0.12, fp));
    { float pkS; alb *= rkMidTex(p, fp, 1.0 - smoothstep(0.35, 0.8, nbn.y), pkS); }   // (rocks5 M4) the boulders' granite speckle
    // contact shadow where the boulder meets the ground / its neighbours; the steep sides darker
    occ = mix(0.15, 1.0, smoothstep(1.0, 0.5, e));
    alb *= mix(1.0, 0.7, smoothstep(0.55, 0.95, e));
    return w;
}

// the steep forested banks beside the falls and the forest floor above the cliff (0..1): shrubs,
// ferns, grass and needle litter instead of bare rock
float fallsBank(vec3 p) {
    if (uFallsOn == 0) return 0.0;
    vec2 uv = wfUV(p.xz);
    if (uv.x < -40.0 || uv.x > 140.0 || uv.y < -110.0 || uv.y > 140.0) return 0.0;
    float hs = wfStepH(uv.y);
    float uf = uv.x - wfFaceLine(uv.y);
    float W = wfFaceWv(hs, uv.y);
    float band = smoothstep(-6.0, -2.0, uf) * (1.0 - smoothstep(W + 20.0, W + 50.0, uf));
    float ends = smoothstep(-110.0, -80.0, uv.y) * (1.0 - smoothstep(105.0, 140.0, uv.y));
    return band * ends * smoothstep(0.25, 1.0, p.y) * smoothstep(3.0, 9.0, hs);
}

// the forested rim behind the top of the bare cliff (0..1, wfRimW)
float fallsRim(vec3 p) {
    if (uFallsOn == 0) return 0.0;
    vec2 uv = wfUV(p.xz);
    if (uv.x < -10.0 || uv.x > 60.0 || uv.y < -30.0 || uv.y > 45.0) return 0.0;
    return wfRimW(uv.y, uv.x - wfFaceLine(uv.y), wfFaceW(wfStepH(uv.y)));
}

// the sunny banks right beside the falls (0..1): lush ferns and shrubs over the steep slope above the
// dark buttress (left of fall A), fewer bare outcrops
float fallsLushBank(vec3 p) {
    if (uFallsOn == 0) return 0.0;
    vec2 uv = wfUV(p.xz);
    float zone = smoothstep(-55.0, -40.0, uv.y) * (1.0 - smoothstep(-10.0, -6.0, uv.y))
               + smoothstep(14.0, 20.0, uv.y) * (1.0 - smoothstep(35.0, 50.0, uv.y));
    float yb = wfButtH(uv.y, wfStepH(uv.y));
    return saturate(zone) * smoothstep(yb + 0.5, yb + 3.5, p.y) * (1.0 - smoothstep(-5.0, 30.0, uv.x - wfFaceLine(uv.y) - wfFaceWv(wfStepH(uv.y), uv.y)));
}

// foam + aeration on the plunge pool (0..1) and an outward ring-wave normal tilt. A solid boil where
// each fall hits, then a plume of lacy foam (a network of foam lines around open-water holes, ~50 %
// cover) drifting out over the lake and southward with the outflow; gPoolAer is the bubbly,
// milky-turquoise water around it (water_shade.glsl). foot = pixel footprint on the water (m).
float gPoolAer = 0.0;
float gPoolTone = 1.0;     // brightness of the pool foam 0.65..1 (thin froth is greyer than the boil)

// where the inflow stream meets the lake (render.py uStreamMouth: x, z, downstream dir x, z)


// ---------------------------------------------------------------------------
// ground.glsl : near-field ground cover (grass blades, forbs, wildflowers, shrubs, stones)
//               and the albedo of vegetated ground at all distances
// ---------------------------------------------------------------------------
// Interface used by scene.frag:
//   float coverSDF(vec3 p, float t, out vec4 info)      : distance to ground-cover geometry
//          (only called when p is < COVER_MAX_H above the terrain and t < COVER_MAX_DIST)
//   vec3  shadeCover(vec3 p, vec3 rd, float t, vec4 info): shading of a ground-cover hit
//   vec3  groundVegAlbedo(vec3 p, inout vec3 n, float t, vec4 forest, float ao)
//          albedo of vegetated ground (meadow / forest floor); may perturb the normal n
//
// How the cover works
//   All cover lives within GC_FADE1 m of the camera and below GC_TOP m above the terrain. It is
//   made of analytic primitives in two hashed grids that only depend on world position:
//     fine cells (GC_C = 0.22 m)  : a grass patch (bent tapered ribbons) or a shore tussock, plus an
//                                   optional forb (broad leaves) carrying a wildflower (stem + head)
//     coarse cells (GC_C2 = 1.76 m): one dwarf shrub (ellipsoid hull with 3D-noise foliage) or a stone
//   Every primitive stays inside its own cell, so a 2D DDA along the ray is exact.
//   coverSDF reconstructs the ray it is marching (exactly for primary rays incl. the lens offset,
//   from two calls for reflection rays), walks the grid over a short segment and returns the
//   along-ray distance to the next cover hit or to the end of the tested segment. The result is
//   cached in globals, so the marcher's many small steps inside a segment cost almost nothing, and
//   the hit record (normal, material) is handed to shadeCover through globals as well.
//   LOD: blades narrower than ~1 pixel are widened and thinned by the same factor (coverage kept);
//   everything shrinks into the ground between GC_FADE0 and GC_FADE1 where the ground albedo
//   (groundVegAlbedo) takes over with tussock / flower / stone texture.
//   (round 3, veg3) Leaf blades of tussocks and long leaning tillers arch over and droop (gcArchFan);
//   tussocks carry flowering culms. Beyond ~4-7 m groundVegAlbedo draws the sward as a shell volume
//   (gcSward: 4-8 shells along the view ray through the canopy - clumps, blades, patch types, forb
//   whorls, stones, flower heads, with parallax and view-dependent depth) out to a ~6-10 cm pixel
//   footprint, the forest floor likewise (gcCarpet leafy dwarf-shrub carpet, dry-grass shells), and
//   past that a footprint-filtered texture of the same structure. A thin mat (GK_MAT, seed > 4.5) laid
//   over the outer part of the terrain's trail dirt carries the verge's floor (shaded through
//   groundVegAlbedo), so the trail reads ~1-1.5 m wide.
// ---------------------------------------------------------------------------
#define COVER_MAX_H 1.25
#define COVER_MAX_DIST 22.0

#define GC_TOP 0.74          // upper bound of cover height above the terrain (umbel stems, shrubs)
#define GC_GTOP 0.60         // upper bound of the grass layer
#define GC_C 0.22            // fine cell size (m)
#define GC_C2 1.76           // coarse cell size (m)
#define GC_CO 0.6            // coarse objects reach up to GC_CO beyond their cell
#define GC_FADE0 (uDebug == 30 ? 5.0 : 9.5)     // cover starts thinning out (by count; the ground takes over)
#define GC_FADE1 (uDebug == 30 ? 10.0 : 20.0)   // ... and is gone
#define GC_SEG 2.4           // ray segment tested per DDA run (m)
#define GC_MAXCELLS 20
#define GC_KMAX 18           // max grass blades per fine cell

#define GK_BLADE 1.0
#define GK_LEAF 2.0
#define GK_STEM 3.0
#define GK_HEAD 4.0
#define GK_SHRUB 5.0
#define GK_STONE 6.0
#define GK_BERRY 7.0
#define GK_FLEAF 8.0
#define GK_FDISC 9.0
#define GK_MAT 10.0
#define GC_CM 0.5            // medium cell size (m): forb rosettes, flower clumps, berry bushes
#define GC_MMAX 9

// ---- per-invocation ray cache and hit record ------------------------------------------------
float gcLastT = 1e30;
bool  gcRayOk = false;
vec3  gcRd = vec3(0.0, -1.0, 0.0);
bool  gcAnch = false;
vec3  gcAP = vec3(0.0);
float gcAT = 0.0;
float gcSafeT = -1.0;        // the ray is free of cover up to here (or hits it here if gcSegHit)
bool  gcSegHit = false;
vec4  gcInfo = vec4(0.0);    // kind, height above ground, seed, aux
vec3  gcN = vec3(0.0, 1.0, 0.0);
float gcTopH = 0.3;          // local canopy height at the hit
vec4  gcObj = vec4(0.0);     // shrub / stone: centre, seed
vec4  gcObjR = vec4(1.0);    // radii, angle / type
// boulders (rocks.glsl) around the current segment: centre xz, radii xz / rotation cos, sin
vec4  gcRk[8];               // (+ up to 4 small stones of the coarse cells, appended per segment)
vec2  gcRkA[8];
float gcStoneFoot = 1.0;     // ground-level footprint scale of the last stone from gcCoarseObj
int   gcNRk = 0;
vec3  gcT = vec3(1.0, 0.0, 0.0);   // leaf midrib direction at the hit
vec2  gcUV = vec2(0.0);            // leaf / disc coordinates at the hit
vec2  gcEnv = vec2(0.0);           // forest-floor weight, meadow weight at the hit
float gcSrc = 0.0;                 // look-dev (uDebug 18): which generator made the hit
vec2  gcTusEnv = vec2(0.0);        // gcEnv of the tussock last returned by gcTussockAt
float gcTusSh = 0.0;               // (veg4) grassy-shore / boulder-belt weight of that tussock (straw share)
// conifers near the current ray segment: base xz, crown radius used for the canopy footprint
vec3  gcTr[8];
int   gcNTr = 0;
float gcFadeK = 1.0;               // flower head colour weight (LOD-widened heads fade to green)
vec4  gcLo[3];                     // shrub lobes: centre offset (units of the shrub radii), radius
// sky visibility of the forest floor under the crowns / inside a stand, exported for shadeTerrain
// (integration: GC_SKYOCC, see NOTES.md); the albedo is no longer darkened for it
float gcGroundSkyOcc = 1.0;
float gcBladeU = 0.0;               // (i14) across-blade coordinate of the last blade hit (-1..1): keeled shading
float gcDensH = 1.0;                // (i17) sward density around the last blade hit: isolated tillers have no dark base

// distance LOD of the 3D cover: plants thin out by count (each keeps its full size; a plant with
// LOD hash rk lives while rk < keep and grows in over 0.12 of the keep range), and only the
// survivors shrink into the ground over the last 2 m. The ground albedo takes over the thinned
// fraction (groundVegAlbedo).
float gcKeep(float camD) { return 1.12 - 0.87 * smoothstep(GC_FADE0, GC_FADE1 - 2.0, camD); }
// (veg4 G2) the survivors shrink over the last ~7 m, not 2 m: the tall stems / flower heads ended along a
// "mowing line" at 18-20 m (r3, shot05)
float gcShrink(float camD) { float s = 1.0 - smoothstep(GC_FADE1 - 7.0, GC_FADE1, camD); return s * (2.0 - s); }

uint gcS;

// ---- habitat of a ground position (depends on position only, so every ray agrees) -----------
struct GBio { float veg; float forest; float meadow; float shore; float above; vec2 g; float wood; float trail; float sgr; float tedge; float peb; vec2 tdir; float tdist; float vegT; float sgrB; };
// grassy shore stretches (tussock grass and dwarf shrubs between boulders down to the
// water) vs. bare cobble beaches; ~30 m patches
float gcShoreGrass(vec2 x) {
    float s = smoothstep(0.22, 0.42, vnoise(x * 0.03 + 71.0));     // (veg4 G5: ~70 % of the shores grassy, was 0.40-0.62)
    // (veg3) the inflow stream's floodplain and delta (shot 4): lush grass between the boulders
    // right down to the water, never a bare cobble strip
    vec2 sd = streamDist(x);
    return max(s, 1.0 - smoothstep(3.0 * sd.y, 4.5 * sd.y, sd.x));
}
// stand density from the forest map averaged over ~20 x 20 m (smooth, ignores the bake's clearings)
float gcWoodAt5(vec2 x) {
    return (forestAt(x).r + forestAt(x + vec2(8.0, 2.0)).r + forestAt(x - vec2(8.0, 2.0)).r
          + forestAt(x + vec2(-2.0, 8.0)).r + forestAt(x - vec2(-2.0, 8.0)).r) * 0.2;
}
// ground height (= terrainHD(x, 1)) and its slope including all three micro-relief octaves: the same
// slope the terrain's shading normal has, so the cover's habitat masks match the terrain materials

// per ray segment [gcSegA, gcSegB]: the smooth stand field below at both ends (interpolated along
// the segment, so it does not step where the camera-dependent segments start), the tree occupancy
// of a 5x5 block of 4 m tree cells, the trail segments nearby
vec2 gcSegA = vec2(0.0), gcSegB = vec2(0.0);
float gcWoodA = 0.0, gcWoodB = 0.0;

ivec2 gcTB0 = ivec2(0);      // tree cell at bit 0 of gcTBits (bit i + 5 j = cell gcTB0 + (i, j))
uint gcTBits = 0u;
int gcTBN = 5;             // block size (5 per segment, 4 for one point)
bool gcTrailSeg = false;
// stand density at x (0 open .. 1 stand): 3x3 tree-cell occupancy counts of the 4 tree-cell centres
// around x blended bilinearly (separable weights 1-f, 1, 1, f over a 4x4 block) - continuous, and
// only a function of x (x is clamped into the gathered block)
float gcStandOf(vec2 x) {
    vec2 g = clamp((x + FAR_HALF) / TREE_CELL - 0.5, vec2(gcTB0 + 1), vec2(gcTB0 + gcTBN - 2) - 1e-3);
    ivec2 c = ivec2(floor(g));
    vec2 f = g - vec2(c);
    ivec2 o = c - 1 - gcTB0;
    float n = 0.0;
    for (int j = 0; j < 4; j++) {
        float wy = j == 0 ? 1.0 - f.y : (j == 3 ? f.y : 1.0);
        uint row = gcTBits >> uint(o.x + 5 * (o.y + j));
        n += wy * ((1.0 - f.x) * float(row & 1u) + float((row >> 1u) & 1u) + float((row >> 2u) & 1u) + f.x * float((row >> 3u) & 1u));
    }
    return smoothstep(1.0, 4.0, n);
}
// forest trail (shadeTerrain draws its dirt): no plants on it. Needs the trail uniforms declared
// before this file, signalled by GC_TRAIL.
vec4 gcTrS0 = vec4(0.0), gcTrS1 = vec4(0.0);   // the two trail segments nearest the current ray segment
uint gcTrMask = 0u;          // trail segments within reach of the cover around the camera (per pixel)
bool gcTrMaskOk = false;

// trail weight near the current ray segment (only the two segments found by gcTrailDist): x = dirt
// (same edge as shadeTerrain), y = 0.7 m verge outside the dirt (short, thin grass); dir = unit
// direction toward the trail's centre line
float gcTrD = 1e9;            // distance to the trail's centre line (set by gcTrailW)

// habitat at ground position x (height y, slope g, water level wl): a function of the position only,
// so every ray and pixel agrees on what grows where

// vegetation mask only (bare-segment test), with margins: > 0 wherever a plant rooted up to ~1.4 m
// beside the probed point could stand (slope ny 0.45, 0.5 m below the waterline, pebble shore 0.4 m
// lower), so plants leaning over rock or water next to the probed track are never skipped

// ~1-2 m patch field shared by the 3D grass and the ground texture (0 dark forb clump .. 1 light grass)
float gcPatch(vec2 x) {
    return smoothstep(0.25, 0.75, vnoise(x * 0.9 + 23.0) * 0.7 + vnoise(x * 2.3 + 5.0) * 0.3);
}

// ---- conifer canopy footprint (needle litter, shade-tolerant understory) ----------------------
// crown radius is capped so the tree-cell gathers below are exact
#define GC_CANR 2.9
float gcCanopyOne(vec2 x, vec3 tr) {
    float d = length(x - tr.xy);
    return 1.0 - smoothstep(0.45 * tr.z, 1.15 * tr.z + 0.4, d);
}
// per ray segment: 5x5 tree cells covering the 4x4 stand blocks of both segment ends (gcTBits);
// trees whose footprint can reach within 1.2 m of the segment [a, b] (gcTr)
bool gcTreesOk = false;

// same for one point (shading / albedo): the 4x4 stand block around x (sets gcTBits / gcTB0, so
// gcStandOf works for points near x); canopy from the same 16 trees (all trees within 6.8 m)
float gcCanopyAt(vec2 x, out float stand) {
    gcTB0 = ivec2(floor((x + FAR_HALF) / TREE_CELL - 0.5)) - 1;
    gcTBits = 0u; gcTBN = 4;
    float c = 0.0;
    for (int j = 0; j < 4; j++)
    for (int i = 0; i < 4; i++) {
        Tree tr;
        if (!getTree(gcTB0 + ivec2(i, j), tr)) continue;
        gcTBits |= 1u << uint(i + 5 * j);
        c = max(c, gcCanopyOne(x, vec3(tr.base.xz, min(tr.R, GC_CANR))));
    }
    stand = gcStandOf(x);
    return c;
}
// forest-floor weight: under a crown, or inside / at the edge of a stand (not on the meadows)
// (sharpened: the open trail corridor and small clearings inside a stand are forest floor too)
float gcFloorW(float w) { return smoothstep(0.08, 0.5, w); }

// boulders near x (2x2 nearest rock cells) for suppressing grass inside their footprints
// (cached per ray: consecutive segments mostly share the 2x2 rock cells; the stones appended by a
// segment are dropped again)
ivec2 gcRkC0 = ivec2(-99999);
int gcNRkB = 0;
vec2 gcRkY[8];               // (veg4 R6) centre height, half height of each gathered rock (sun shadow on the cover)

// (veg4 R6) sun visibility of a cover point past the gathered boulders / stones (ray vs their ellipsoids, soft
// edge) and their contact occlusion: the grass ran fully lit right up to every block (r8 0.33-0.40 vs open
// lawn 0.344; near-black slots between the blocks)

// 0 inside a boulder footprint; nearR = 1 right at a boulder's foot

// smallest footprint distance (1 = at 0.9 of a boulder's / stone's footprint radius)

// ---- analytic primitives ---------------------------------------------------------------------
// bent ribbon surface: root R, lean direction a (unit xz), height H, horizontal lean L of the top.
// Centre line x1 = L (y/H)^2 in the vertical plane through a; the surface extends sideways
// (perpendicular to a), so its face looks outward/up like an arching blade. Returns the two ray
// parameters of the surface crossings and the local ray terms (a2, b2, a3, b3).

// broad forb leaf (lanceolate) on a bent ribbon

// grass tiller: a fan of nb tapered blades in one bent surface (grass leaves are distichous).
// The fan widens from Wb at the root to Wt at the top; each blade fills fr of its slot and has
// its own length (hash hs). Returns v along the hit blade.

// (veg3) arching, drooping blades: the vertical plane through a holds the centre line
// y = c1 x - c2 x^2 (x = distance from the root along a): steep at the root, peak Hp at xp, then
// falling to the tip at x = Lx (Lx up to ~1.9 xp: the tip hangs down to ~1/3 of the peak). Fan of nb
// blades across the surface, as gcFan; v = x / Lx along the blade.

// leaning stem: cylinder of radius rad from R to R + (lean.x, H, lean.y) (sheared, so the 2D test is exact)

// axis-aligned ellipsoid, optionally rotated about y by (cos, sin) cs

// angular granite stone: an ellipsoid cut by a tilted top facet and two side facets (convex, so
// entry = max of the entries, exit = min of the exits); seed picks the facets

// dwarf shrub / understory bush: three seeded lobes (centres up to 0.4 R sideways and up to 0.15 h
// up) filled with 3D-noise foliage - ~18 cm lumps for a ragged outline, ~9 cm leaf clusters, ~3 cm
// sprigs where the march resolves them - porous at the rim, inside a hull 1.2x the shrub radii that
// the foliage never reaches (no smooth hull surface can show)

// leaves on a foliage surface: 3D cells (x in cell units; nearest of the 8 surrounding jittered
// feature points), each a leaf with its own tilt and tint. Returns (leaf randoms, F2 - F1: ~0 where
// two leaves meet - the dark gaps)

// shading version: plus a ~1 cm leaf term (weight lw, faded by the pixel footprint) for the normal

// cheap reject of a tiller (root R, height H, horizontal reach ext) that the ray cannot touch
vec2 gcDn = vec2(1.0, 0.0);   // normalized xz direction of the segment ray
float gcDl = 1.0;             // |d.xz|

// ---- coarse cell: one shrub or one stone ------------------------------------------------------
// returns 0 none, 1 shrub, 2 stone

// ---- leaves and flower heads as flat blades --------------------------------------------------------
// leaf frame from the midrib azimuth phi, its elevation above the horizontal and a roll about it

// flat leaf: centre c, normal n, midrib tu, half length a, half width b; tip > 0 makes it ovate
// (widest below the middle, pointed). uv = (along -1..1, across -1..1)

// round flower head facing n: returns radial coordinate (0..1) and angle

// ---- medium cell: blueberry bush / forb rosette / flower clump ----------------------------------
// blueberry / bilberry: a dome of leafy sprigs. Each sprig is a short twig with alternate small
// leaves and its own seed, so sprigs the ray misses are skipped without drawing their numbers.

// forb. species: 0 lady's-mantle (round, low), 1 tall leafy stem (broad pleated leaves),
// 2 lanceolate rosette, 3 broad ribbed leaves spiralling up a short stem (veratrum, adenostyles).
// big = 0 picks 0 / 2 (medium cells), big = 1 picks 1 / 3 (1 m cells). ctr: root, yR: ground there
// (veg4 G3a) leafy sprigs on a dwarf shrub's hull (coarse cells, within ~10 m): 5-9 cm leaves on short twigs
// sticking out of the foliage volume, so the bush reads as leaves (bilberry, alpenrose), not as a
// noise-hull "moss ball / hedgehog". Shaded as GK_BERRY leaves (lit lime when back-lit)

// flower species colours (linear albedo)
vec3 gcFlowerCol(float sp) {
    if (sp < 0.5) return vec3(0.40, 0.31, 0.014);     // buttercup / arnica yellow (i16: was 0.58 / 0.50 - clipped to cream facing the sun)
    if (sp < 1.5) return vec3(0.30, 0.105, 0.17);     // lousewort / betony: dusty magenta
    if (sp < 2.5) return vec3(0.70, 0.70, 0.58);      // yarrow / umbel white
    if (sp < 3.5) return vec3(0.13, 0.17, 0.52);      // gentian / campanula blue
    if (sp < 4.5) return vec3(0.24, 0.215, 0.12);     // grass panicle: straw
    return vec3(0.15, 0.10, 0.095);                   // grass panicle: purplish brown
}
// species chosen from a smooth field so colours grow in clusters
float gcSpecies(vec2 x, float r) {
    float a = vnoise(x * 0.45 + 77.0), b = vnoise(x * 0.37 - 21.0);
    // yellow composites most common, then magenta spikes; white umbels and blue bells few
    float s = a + 0.3 * (r - 0.5);
    float sp = s < 0.60 ? 0.0 : (b < 0.3 ? 2.0 : 1.0);
    if (r > 0.94) sp = 3.0;
    return sp;
}

// clump of 1-4 flowering stems of one species

// ---- fine cell: grass patch / tussock + forb + flower -----------------------------------------

// ---- tussock: a fountain of grass tillers around a centre (lake shore, between boulders, meadow)
#define GC_C3 1.2            // tussock cell size (m)   (veg4 G5: 1.0 -> 1.2, room for the 1 m shore tussocks)
#define GC_TR 0.6            // max tussock footprint radius (m)   (veg4 G5: 0.4 -> 0.6)
#define GC_TR0 0.4           // footprint radius of a size-1 tussock (m)

// ---- 1 m cells: a tussock and / or tall forbs (both may reach GC_TR beyond their cell) --------
// one habitat evaluation (at the cell centre) serves both; cheap bounds skip most empty cells
// before it

// ---- one DDA run over the ray segment [o, o + d GC_SEG] --------------------------------------
float gcSegLen = GC_SEG;     // (shadeCover's replay walks a short segment only)

// ---- shading ---------------------------------------------------------------------------------
// direct + sky light for thin vegetation: wrapped diffuse, translucency, canopy self-shadowing
// grassland dryness (0 lush hollows .. 1 dry straw on sunny south faces), shared by the 3D blades
// and the ground albedo so the cover and the far grass carry the same colours
float gcDryness(vec2 x, vec3 n, float ao, float meadow) {
    float d = 0.42 + 0.8 * fbm2(x * 0.011 + 3.0, 3) + 0.45 * gnoise(x * 0.047 + 11.0) + 0.22 * gnoise(x * 0.19 + 5.0)
            - 1.8 * n.z * (1.0 - n.y) - 0.7 * (1.0 - ao);
    return saturate(d - 0.15 * meadow);
}
// grass blade albedo (r: per-blade random); the mean over r for the grass seen from afar
// (veg3: yellow-green, high chroma (meadow #7b892b / #64751f): far less blue)

vec3 gcBladeMean(float dry) {
    vec3 c = mix(vec3(0.150, 0.195, 0.021), vec3(0.180, 0.166, 0.045), smoothstep(0.35, 0.9, dry) * 0.6);
    return mix(c, vec3(0.220, 0.185, 0.052), smoothstep(0.75, 1.1, dry) * 0.4);
}
vec3 gcSpecTint = vec3(1.0);

vec3 groundVegAlbedo(vec3 p, inout vec3 n, float t, vec4 f, float ao);     // (below)

// ---- forest-floor debris seen at 1-4 cm per pixel: twigs, cones, needle / leaf clumps -----------
// returns (albedo multiplier, darkening mix weight); filtered by the pixel footprint foot (m)
vec3 gcFloorDebris(vec2 x, float foot, vec3 col) {
    // needle / leaf-litter clumps: ~12 cm cells, each a bit lighter or darker, warm or grey
    float cw = 1.0 - smoothstep(0.012, 0.035, foot);
    float v1 = vnoise(x * 9.0 + 2.1), v2 = vnoise(x * 4.3 + 7.7);
    col *= mix(1.0, mix(0.60, 1.30, v1), cw);
    float gw = 1.0 - smoothstep(0.004, 0.012, foot);
    float v3 = vnoise(x * 31.0 + 4.4);
    col *= mix(1.0, mix(0.65, 1.25, v3), gw);
    col = mix(col, col * vec3(1.15, 1.0, 0.72), smoothstep(0.55, 0.8, v2) * 0.6 * cw);
    // litter pieces / clumps (~14 cm) with dark crevices between them, each a little lighter or
    // darker (faded once the cells are under ~4 px)
    float vw = 1.0 - smoothstep(0.02, 0.045, foot);
    if (vw > 0.0) {
        vec3 vc = voronoi2(x * 7.0 + 3.1);
        float crev = smoothstep(0.0, 0.16 + 4.0 * foot, vc.y - vc.x);
        float pw = vw * smoothstep(0.3, 0.6, vnoise(x * 1.7 + 11.0));      // in patches, not a paving
        col *= mix(1.0, mix(0.75, 1.0, crev) * mix(0.8, 1.2, vc.z), pw);
    }
    // twigs: 0-2 per 0.5 m cell, 12-40 cm long, 6-14 mm thick, dark grey-brown
    float tw = 1.0 - smoothstep(0.015, 0.04, foot);
    if (tw > 0.0) {
        ivec2 c0 = ivec2(floor(x / 0.5));
        for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            ivec2 c = c0 + ivec2(i, j);
            vec4 h = hash42i(c + ivec2(401, 77));
            if (h.w > 0.75) continue;
            vec2 ce = (vec2(c) + h.xy) * 0.5;
            float an = h.z * TAU;
            vec2 dr = vec2(cos(an), sin(an));
            float L = mix(0.07, 0.22, fract(h.w * 13.7));
            float wdt = mix(0.003, 0.007, fract(h.x * 9.1)) + 0.5 * foot;
            vec2 q = x - ce;
            float along = clamp(dot(q, dr), -L, L);
            float dd = length(q - dr * along);
            float m = (1.0 - smoothstep(wdt - 0.5 * foot, wdt + 0.5 * foot, dd)) * min(1.0, 2.0 * wdt / max(foot, 1e-4)) * 0.85;
            col = mix(col, vec3(0.050, 0.040, 0.028) * mix(0.8, 1.5, fract(h.y * 5.3)), m * tw);
        }
    }
    // cones: brown ellipses 3 x 6 cm in some 0.7 m cells
    float cn = 1.0 - smoothstep(0.01, 0.03, foot);
    if (cn > 0.0) {
        vec4 v = voronoiV(x / 0.7 + 5.5);
        if (fract(v.w * 17.1) < 0.18) {
            vec2 q = rot2(v.w * TAU) * v.xy * 0.7;
            float e = length(q / vec2(0.032, 0.016));
            float m = 1.0 - smoothstep(1.0 - foot / 0.016, 1.0 + foot / 0.016, e);
            col = mix(col, vec3(0.085, 0.050, 0.022) * (0.7 + 0.5 * (1.0 - e)), m * cn);
        }
    }
    return col;
}

// ---- vegetated ground albedo ------------------------------------------------------------------
// surface roots radiating from conifer trunks: mask and bump direction
float gcRoots(vec2 x, out vec2 bump) {
    bump = vec2(0.0);
    ivec2 c0 = treeCellOf(x);
    float best = 0.0;
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        Tree tr;
        if (!getTree(c0 + ivec2(i, j), tr)) continue;
        vec2 q = x - tr.base.xz;
        float r = length(q);
        float len = 0.7 + 0.06 * tr.H;
        float trunkR = 0.05 + 0.011 * tr.H;
        if (r > len || r < trunkR * 0.8) continue;
        float nR = 5.0 + floor(tr.seed * 4.0);
        float a = atan(q.y, q.x) * nR / TAU + tr.seed * 7.0 + 0.10 * sin(r * 5.0 + tr.seed * 20.0) * nR / TAU * 3.0;
        float fa = fract(a) - 0.5;
        float halfW = trunkR * 0.6 * exp(-r * 1.6) + 0.018;
        float wa = halfW / max(r * TAU / nR, 1e-3);
        float m = (1.0 - smoothstep(wa * 0.5, wa, abs(fa))) * (1.0 - smoothstep(len * 0.5, len, r));
        m *= step(0.5, fract(tr.seed * 17.0 + floor(a) * 0.618)) * 0.4 + 0.6;
        if (m > best) {
            best = m;
            vec2 tg = vec2(-q.y, q.x) / max(r, 1e-3);
            bump = tg * clamp(fa / max(wa, 1e-3), -1.0, 1.0) * m + q / max(r, 1e-3) * 0.3 * m;
        }
    }
    return best;
}

// low herb layer between the grass (lady's mantle, clover, plantain, young forbs): a jittered grid
// of overlapping ovate leaves, the highest one wins. Returns the albedo; nOff tilts the normal
// (leaf pose + midrib fold); cov = leaf coverage (0 in the dark gaps). Filtered by the pixel
// footprint foot (m): edges soften and the pattern dissolves into its mean when leaves get small.
vec3 gcLeafMat(vec2 x, float foot, float lushT, out vec2 nOff, out float cov) {
    const float S = 0.06;
    vec2 g = x / S;
    ivec2 c0 = ivec2(floor(g));
    float bestH = -1.0;
    vec3 gap = mix(vec3(0.040, 0.050, 0.014), vec3(0.070, 0.066, 0.030), saturate(0.5 + gnoise(x * 6.1)));
    vec3 col = gap;
    nOff = vec2(0.0); cov = 0.0;
    // clumping: leaves are missing in patches (bare thatch, grass bases)
    float pres = 0.55 + 0.45 * lushT + 0.35 * gnoise(x * 2.3 + 4.0);
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        ivec2 c = c0 + ivec2(i, j);
        vec4 h = hash42i(c + ivec2(911, 377));
        if (fract(h.x * 31.7 + h.w * 3.9) > pres) continue;
        float sp = fract(h.w * 7.31 + h.x * 3.3);                  // leaf type
        // round (lady's mantle, clover) / ovate / lanceolate (plantain) / grass blade
        float ty = sp < 0.2 ? 0.0 : (sp < 0.5 ? 1.0 : (sp < 0.75 ? 2.0 : 3.0));
        vec2 ctr = (vec2(c) + 0.15 + 0.7 * h.xy) * S;
        float ang = h.z * TAU;
        vec2 dir = vec2(cos(ang), sin(ang));
        float a = S * mix(0.45, 1.05, fract(h.y * 17.3)) * (ty > 2.5 ? 1.7 : 1.0);   // half length
        float b = a * (ty < 0.5 ? 0.85 : (ty < 1.5 ? 0.5 : (ty < 2.5 ? 0.28 : 0.07)));
        vec2 q = x - (ctr + dir * a * 0.6);
        float u = dot(q, dir) / a, v = (q.x * dir.y - q.y * dir.x) / b;
        float w = sqrt(max(1.0 - u * u, 0.0)) * (1.0 - 0.3 * u * step(0.5, ty));
        if (ty < 0.5) w *= 0.86 + 0.14 * cos(atan(v, u + 1.0) * 9.0);   // lobed margin
        float e = (abs(v) - w) * max(b, 0.5 * foot);                  // ~ signed distance to the margin (m)
        float inn = (1.0 - smoothstep(-0.6 * foot, 0.6 * foot, e)) * step(abs(u), 1.0);
        if (ty > 2.5) inn *= 1.0 - smoothstep(0.004, 0.012, foot);     // blades vanish when sub-pixel
        if (inn <= 0.0) continue;
        float hh = fract(h.x * 13.1 + h.y * 7.7);
        if (hh < bestH) continue;
        bestH = hh;
        // per-leaf tint: yellow-green to blue-green, some dull / yellowing leaves
        float r = fract(h.y * 5.7 + h.z * 3.1);
        vec3 lc = mix(vec3(0.090, 0.135, 0.024), vec3(0.150, 0.190, 0.034), r);
        if (ty < 0.5) lc = mix(vec3(0.070, 0.115, 0.040), vec3(0.095, 0.140, 0.048), r);   // bluish, hairy
        if (ty > 2.5) lc = mix(vec3(0.120, 0.165, 0.030), vec3(0.200, 0.200, 0.070), r * r); // grass, some straw
        lc = mix(lc, vec3(0.165, 0.160, 0.050), step(0.94, r) * 0.7);       // yellowing
        lc *= mix(0.78, 1.10, lushT) * (0.9 + 0.2 * hh) * 0.78;
        float vein = ty > 1.5 && ty < 2.5 ? 0.5 + 0.5 * cos(v * 9.0) : 1.0;  // parallel veins
        lc *= (0.88 + 0.16 * (1.0 - abs(v)) - 0.08 * smoothstep(0.3, 1.0, -u)) * mix(0.92, 1.0, vein);
        // pale midrib, darker curled margin, sunlit sheen on the upper blade half; fade when small
        float lw = 1.0 - smoothstep(0.002, 0.006, foot);
        lc *= mix(1.0, (1.0 + 0.22 * (1.0 - smoothstep(0.0, 0.12, abs(v))) * step(0.5, ty)) * mix(0.68, 1.0, 1.0 - smoothstep(0.7, 1.0, abs(v))), lw);
        lc = mix(lc, lc * vec3(1.12, 1.1, 1.25) + 0.012, lw * smoothstep(0.2, 0.9, u) * step(0.6, fract(h.w * 5.3)) * 0.6);
        col = mix(col, lc, inn);
        cov = max(cov, inn);
        // leaf pose (tilted towards its tip) + fold about the midrib
        vec2 side = vec2(-dir.y, dir.x);
        float tilt = (fract(h.z * 9.7) - 0.4) * 1.1;
        nOff = mix(nOff, dir * tilt + side * clamp(v, -1.0, 1.0) * 0.45, inn);
    }
    // dissolve into the mean as leaves fall below a few pixels
    float fw = smoothstep(0.003, 0.010, foot);
    vec3 mean = mix(gap, vec3(0.115, 0.155, 0.032) * mix(0.78, 1.10, lushT), 0.65);
    col = mix(col, mean, fw);
    nOff *= 1.0 - fw;
    cov = mix(cov, 0.65, fw);
    return col;
}

// ---- the sward seen from ~3-150 m: a shell-textured canopy volume --------------------------------
// Blades / tillers / tussock clumps as a density volume of height ~H over the ground, sampled on GC_SWK
// shells along the view ray between the canopy top and the ground hit and composited front to back:
// real parallax and view-dependent occlusion for a few texture fetches per shell (grazing views see the
// tips and the tops of the clumps, steep views look down between them into the dark base). Flower heads
// ride on a layer at the top. g = macro grass colour, base = what lies under the sward, flD = flower
// density (heads per 14 cm cell). Returns the albedo (canopy self-shadowing included).
#define GC_SWK 8
// granite stone poking through the sward (one per 0.9 m cell at most, probability stP): squashed dome
// 8-30 cm; returns its colour when the shell sample at (q, h) is inside it, alpha in .w
// (the shells of one pixel mostly stay in one stone / forb cell: its hash is cached)
ivec2 gcSwStC = ivec2(-99999); vec4 gcSwStH = vec4(1.0);
ivec2 gcSwFoC = ivec2(-99999); vec4 gcSwFoH = vec4(1.0);
vec4 gcSwardStone(vec2 q, float h, float stP, float foot) {
    ivec2 sc = ivec2(floor(q / 0.9));
    if (sc != gcSwStC) { gcSwStC = sc; gcSwStH = hash42i(sc + ivec2(733, 4127)); }
    vec4 sh = gcSwStH;
    if (sh.w >= stP) return vec4(0.0);
    float R = mix(0.08, 0.30, sh.z * sh.z);
    vec2 c = (vec2(sc) + 0.5) * 0.9 + (sh.xy - 0.5) * (0.9 - 2.0 * R);
    float hs = R * mix(0.45, 0.85, fract(sh.z * 7.1));
    vec2 d = (q - c) / R;
    float hn = h / hs;
    float e = dot(d, d) + hn * hn;
    float a = 1.0 - smoothstep(1.0 - 1.5 * foot / R, 1.0, e);
    if (a <= 0.0) return vec4(0.0);
    // lit top / sun-facing flank, shaded flank, salt-and-pepper grain, grey-green lichen
    vec3 nS = normalize(vec3(d.x, hn * 1.2 + 0.3, d.y));
    float lit = clamp(max(dot(nS, uSunDir), 0.0) / max(uSunDir.y, 0.3), 0.22, 1.5);   // (i18: relative to the flat ground the terrain lights)
    float gr = mix(0.5, texture(uNoise3, vec3(q * 3.0, sh.z * 7.0)).r, 1.0 - smoothstep(0.004, 0.012, foot));
    vec3 col = vec3(0.28, 0.27, 0.24) * mix(0.75, 1.12, fract(sh.z * 13.1)) * mix(0.8, 1.2, gr) * lit;   // (i17: was 0.25 - dark dots; veg4 G10: 0.36 -> 0.28, white popcorn)
    // dark crustose lichen (fades to its mean when sub-pixel)
    float cr = smoothstep(0.66, 0.72, texture(uNoise3, vec3(q * 1.7, 0.41)).a);
    col *= mix(1.0 - 0.18, 1.0 - 0.6 * cr, 1.0 - smoothstep(0.006, 0.02, foot));
    col = mix(col, vec3(0.25, 0.26, 0.19) * lit, smoothstep(0.55, 0.7, texture(uNoise3, vec3(q * 0.8, 0.77)).g) * 0.6);
    return vec4(col * mix(0.40, 1.0, smoothstep(0.0, 0.45, hn)), a);
}
// broad-leaved forbs inside the sward (lady's mantle, cranesbill, arnica rosettes): a whorl of 5-7
// leaves (r 4-11 cm) at one height per 0.3 m cell; the shell at that height hits it. Lighter, glossier
// than the grass. Returns colour, alpha in .w
vec4 gcSwardForb(vec2 q, float h, float dh, float H, float pF, vec3 g, float foot) {
    ivec2 fc = ivec2(floor(q / 0.3));
    if (fc != gcSwFoC) { gcSwFoC = fc; gcSwFoH = hash42i(fc + ivec2(2207, 613)); }
    vec4 fh = gcSwFoH;
    if (fh.w >= pF) return vec4(0.0);
    float hl = H * mix(0.2, 0.75, fh.z);
    if (abs(h - hl) > 0.5 * dh) return vec4(0.0);
    float r = mix(0.045, 0.11, fract(fh.z * 5.3));
    vec2 c = (vec2(fc) + 0.5) * 0.3 + (fh.xy - 0.5) * (0.3 - 2.0 * r);
    vec2 d = q - c;
    float ang = atan(d.y, d.x);
    float nl = 5.0 + floor(fract(fh.x * 9.1) * 3.0);
    float lobe = 0.62 + 0.38 * abs(cos(ang * nl * 0.5 + fh.y * 6.0));
    float rr = length(d) / (r * lobe);
    float a = 1.0 - smoothstep(1.0 - 1.2 * foot / r, 1.0, rr);
    a *= smoothstep(0.08, 0.25, length(d) / r);                 // the gap at the centre
    if (a <= 0.0) return vec4(0.0);
    float lw = 1.0 - smoothstep(0.004, 0.012, foot);
    // light, yellowish green leaves, veins / folds, the sunlit upper side brighter
    vec3 col = g * mix(vec3(1.05, 1.18, 0.85), vec3(1.35, 1.38, 0.9), fract(fh.z * 11.7));
    col *= mix(1.0, 0.8 + 0.3 * abs(cos(ang * nl)), lw) * mix(1.0, 0.75 + 0.35 * smoothstep(0.2, 1.0, rr), lw);
    return vec4(col, a);
}
vec3 gcSward(vec2 x, vec3 n0, vec3 V, float foot, float H, vec3 g, vec3 base, float dry, float flD, float stP, float pF) {
    float cosT = max(dot(V, n0), 0.03);
    vec2 vxz = V.xz / cosT;                  // xz shift per metre of height above the ground plane
    float lxz = length(vxz);
    float top = 1.3 * H;                     // the tallest clumps
    // grazing rays only reach the top of the canopy: the shells span the top ~0.9 m (xz) of the path
    float span = min(top, 0.9 / max(lxz, 1e-3));
    float nSh = foot < 0.01 ? float(GC_SWK) : (foot < 0.035 ? 6.0 : 4.0);   // fewer shells once blades are sub-pixel
    float dxz = span * lxz / nSh;  // xz spacing of the shell samples
    // (i19: both later - the tiller-scale grain is the meadow's texture at 20-60 m; the mean coverage
    //  read as a smooth lawn there, local contrast 13 vs the target ~25)
    float lodF = smoothstep(0.008, 0.024, foot);    // blade noise -> tiller noise
    float lodM = smoothstep(0.05, 0.12, foot);      // -> mean coverage
    vec3 acc = vec3(0.0);
    float T = 1.0;
    // flower heads: one per 14 cm cell at most, on a layer at the canopy top
    // (i14) in the drifts (flD > 0.12) a second, offset layer: clusters of heads that read at 10-40 m
    // (i22) a third layer: grass panicles / seed heads (straw, purplish brown) on the canopy top - the pale
    // speckle of a July meadow at 10-50 m (the turf read as a smooth lawn beyond ~25 m)
    for (int lf = 0; lf < 3; lf++) {
        float pL = lf == 0 ? flD : (lf == 1 ? (flD - 0.12) * 1.3 : 0.22 * (1.0 - 0.5 * smoothstep(0.03, 0.08, foot)));
        if (pL <= 0.002) continue;
        const float CF = 0.14;
        float hF = H * (lf == 0 ? 1.05 : (lf == 1 ? 0.9 : 1.12));
        vec2 oF = lf == 0 ? vec2(0.0) : (lf == 1 ? vec2(0.5 * CF, 0.37 * CF) : vec2(0.23 * CF, 0.71 * CF));
        vec2 qf = x + vxz * hF + oF;
        ivec2 fc = ivec2(floor(qf / CF));
        vec4 hh = hash42i(fc + (lf == 0 ? ivec2(5113, 977) : (lf == 1 ? ivec2(2711, 4441) : ivec2(919, 6007))));
        if (hh.w < pL) {
            float sp = lf == 2 ? (hh.z < 0.7 ? 4.0 : 5.0) : gcSpecies(vec2(fc) * CF - oF, hh.z);
            float r = sp < 0.5 ? mix(0.014, 0.030, hh.x * hh.x) : (sp < 1.5 ? mix(0.014, 0.02, hh.y) : (sp < 2.5 ? mix(0.026, 0.042, hh.y) : (sp < 3.5 ? 0.012 : mix(0.008, 0.014, hh.y))));
            vec2 ctr = (vec2(fc) + 0.5) * CF + (hh.xy - 0.5) * (CF - 2.0 * r - 0.01);
            // (spikes and bells are taller than wide: seen from the side they stretch along the view)
            vec2 dq = qf - ctr;
            float el = (sp > 0.5 && sp < 1.5) || sp > 3.5 ? 1.0 + (sp > 3.5 ? 2.5 : 1.5) * (1.0 - cosT) : 1.0;
            vec2 vn = lxz > 1e-3 ? vxz / lxz : vec2(1.0, 0.0);
            float d = length(vec2(dot(dq, vec2(-vn.y, vn.x)), dot(dq, vn) / el));
            float re = max(r, 0.6 * foot);
            float a = (1.0 - smoothstep(re - 0.5 * foot, re + 0.5 * foot, d)) * (r * r) / (re * re);
            vec3 fcol = gcFlowerCol(sp) * mix(0.8, 1.05, fract(hh.z * 7.3));
            acc += a * fcol;
            T *= 1.0 - a;
        }
    }
    float jit = gRand;
    float dh = span / nSh;
    float cdec = mix(0.25, 1.0, saturate(dxz / 0.06));   // decorrelation of the mean coverage between shells
    // pale waxy sheen of the lit blade tips (meadow p90 ~ #b1b77d: highlights pale, shade saturated)
    vec3 pale = vec3(0.40, 0.42, 0.17);      // (i18: was 0.34 / 0.35 / 0.20; i20 less blue)
    // (i23) sun self-shadow of the clump field (see below): unit xz direction to the sun, its slope; faded
    // when the 28 cm clumps fall below ~2-3 px
    float lsx = max(length(uSunDir.xz), 1e-3);
    vec2 sunH = uSunDir.xz / lsx;
    float sunT = uSunDir.y / lsx;
    float shW = 1.0 - smoothstep(0.08, 0.20, foot);     // (veg4: the clump groups' shade reads further out)
    for (int k = 0; k < GC_SWK; k++) {
        if (float(k) >= nSh) break;
        float h = top - span * (float(k) + jit) / nSh;
        if (h <= 0.0 || T < 0.03) break;
        vec2 q = x + vxz * h;
        if (stP > 0.0) {
            vec4 st = gcSwardStone(q, h, stP, foot);
            if (st.w > 0.0) { acc += T * st.w * st.rgb; T *= 1.0 - st.w; if (T < 0.03) break; }
        }
        if (pF > 0.0) {
            vec4 fo = gcSwardForb(q, h, dh, H, pF, g, foot);
            if (fo.w > 0.0) {
                float hnF = h / H;
                acc += T * fo.w * fo.rgb * mix(0.45, 1.1, saturate(hnF * 1.3));
                T *= 1.0 - fo.w;
                if (T < 0.03) break;
            }
        }
        vec4 c4 = texture(uNoise3, vec3(q * 0.22, 0.31));             // tussock clumps (~28 cm cells)
        // ~0.6-1.5 m patch types (fbm channel): lush dark forb clumps (taller) / glaucous fescue
        // and sedge tussocks / yellow-green grass between them (never one lawn colour)
        float forbP = smoothstep(0.55, 0.63, c4.r);
        float fescP = 1.0 - smoothstep(0.33, 0.41, c4.r);
        // (veg4 G1) clumps of clumps: the worley-fbm channel (~0.6 m domes) lifts whole clump groups and sinks
        // the gaps between them (dark, shaded) - plant-scale structure at 10-100 m instead of an even lawn
        float kcl = 1.0;
        float clB = mix(1.0 - 0.58 * kcl, 1.0 + 0.18 * kcl, smoothstep(0.30, 0.72, c4.b));
        float Hq = H * (0.45 + 0.85 * smoothstep(0.30, 0.80, c4.a)) * (0.8 + 0.4 * c4.g) * (1.0 + 0.3 * forbP) * clB;
        float hn = h / Hq;
        if (hn >= 1.0) continue;
        // (i23) the sample is in the shade of the clump 15 cm toward the sun when that clump stands above
        // the sun ray rising from it: lit tufts and shaded gaps (a plant-scale light / shade grain;
        // the shells were lit as one smooth sheet). Mean kept ~1 (the depth term holds the average shade)
        float shd = 0.0;
        if (shW > 0.0) {
            vec4 c4s = texture(uNoise3, vec3((q + sunH * 0.2) * 0.22, 0.31));
            float Hs = H * (0.45 + 0.85 * smoothstep(0.30, 0.80, c4s.a)) * (0.8 + 0.4 * c4s.g) * (1.0 + 0.3 * smoothstep(0.55, 0.63, c4s.r)) * mix(1.0 - 0.58 * kcl, 1.0 + 0.18 * kcl, smoothstep(0.30, 0.72, c4s.b));
            shd = smoothstep(-0.03, 0.05, Hs * 0.92 - (h + 0.2 * sunT));
        }
        float bl = 0.5, tb = 0.5;
        if (lodM < 1.0) {
            float bf = lodF < 1.0 ? texture(uNoise3, vec3(q * 2.0, h * 0.5 + 0.53)).r : 0.5;    // blades (~3 cm)
            vec4 t4 = texture(uNoise3, vec3(q * 0.7, h * 0.25 + 0.17));                           // tillers (~9 cm)
            bl = mix(bf, t4.r, lodF);
            tb = t4.g;
        }
        float thr = mix(0.40, 0.64, hn * hn);
        float w = 0.025 + 1.5 * foot;
        float a = smoothstep(thr - w, thr + w, bl);
        // (mean coverage stays high up to the clump tops: the clumps keep crisp silhouettes far away)
        a = mix(a, mix(0.72, 0.32, hn * hn) * cdec, lodM);
        // clump colour: blue-green sedge / yellow-green grass / brighter forb-rich clumps
        vec3 col = mix(g * vec3(0.82, 1.0, 1.15), g * vec3(1.2, 1.08, 0.62), smoothstep(0.3, 0.7, c4.g));   // (i20: less blue, meadow B/G ~0.33)
        col *= mix(0.55, 1.36, c4.b);
        col = mix(col, g * vec3(0.58, 0.84, 0.46), saturate(forbP) * 0.85);
        col = mix(col, vec3(luma(g)) * vec3(0.95, 1.05, 0.42) * 1.15, saturate(fescP) * 0.65);
        // dry straw tips on the dry slopes
        col = mix(col, vec3(0.20, 0.17, 0.07), smoothstep(0.55, 1.0, dry) * smoothstep(0.6, 1.0, hn) * step(0.6, fract(c4.r * 9.7)) * 0.6);
        // lit, glossy blade tips go pale (fine detail only; the mean keeps a little of it)
        float tipL = smoothstep(0.7, 1.0, hn) * mix(smoothstep(0.52, 0.66, bl), 0.3, lodM) * smoothstep(0.003, 0.008, foot);
        col = mix(col, pale, tipL * 0.9);                  // (veg4 G8: 0.75 -> 0.9)
        // depth in the canopy: little sky and sun reach the base (dark, and more saturated: the light
        // down there has passed through the leaves)
        col *= mix(0.09, 1.12, pow(hn, 0.85));             // (veg4 G8: 0.12 -> 0.09)
        col.b *= mix(0.45, 1.0, hn);
        // (i21) per-tiller brightness (lit / shaded tufts, sun angle per leaf): the plant-scale grain of the
        // meadow at 15-60 m (local contrast 25-31 vs the render's 17-19); faded while >= ~2 px
        col *= mix(1.0, mix(0.62, 1.34, smoothstep(0.30, 0.70, tb)), 1.0 - smoothstep(0.02, 0.06, foot));
        col *= mix(1.0, 1.16 - 0.58 * shd, shW);
        acc += T * a * col;
        T *= 1.0 - a;
    }
    // (veg4 G1) grazing rays sample only the top `span` of the canopy: what they would meet below that (in the
    // gaps between the clump groups) is the canopy's shaded lower part, not the bare soil / thatch
    float deep = saturate(1.0 - span / max(top, 1e-3));
    return acc + T * mix(base, g * vec3(0.42, 0.45, 0.32), saturate(deep * 0.8));
}

// ---- forest-floor understory carpet (bilberry / lingonberry / dwarf shrubs) ----------------------------
// a leafy shell volume like gcSward: mounds (~0.5-1 m) of small leaves, coverage cov (0..1) from the
// patch field; isotropic leaf-cluster noise (no blades), lit leaf tops yellow-green, the depths near
// black-green (#273d11 in shade). base = the floor under it.
#define GC_CPK 5
vec3 gcCarpet(vec2 x, vec3 n0, vec3 V, float foot, float H, float cov, vec3 base, float sunR) {
    float cosT = max(dot(V, n0), 0.03);
    vec2 vxz = V.xz / cosT;
    float lxz = length(vxz);
    float top = 1.15 * H;
    float span = min(top, 1.6 / max(lxz, 1e-3));     // (veg4 G11: 0.7 -> 1.6, grazing views see the whole canopy - no soil bands)
    float dxz = span * lxz / float(GC_CPK);
    float lodF = smoothstep(0.006, 0.02, foot);
    float lodM = smoothstep(0.035, 0.09, foot);
    float cdec = mix(0.3, 1.0, saturate(dxz / 0.05));
    float t0 = mix(0.80, 0.22, cov);
    vec3 acc = vec3(0.0);
    float T = 1.0;
    for (int k = 0; k < GC_CPK; k++) {
        float h = top - span * (float(k) + gRand) / float(GC_CPK);
        if (h <= 0.0 || T < 0.03) break;
        vec2 q = x + vxz * h;
        vec4 c4 = texture(uNoise3, vec3(q * 0.09, 0.61));          // mounds / patches (~0.7 m)
        float Hq = H * smoothstep(t0, t0 + 0.12, c4.g) * (0.55 + 0.6 * c4.a);
        float hn = h / max(Hq, 1e-3);
        if (hn >= 1.0) continue;
        // single leaves (~4 cm worley cells, 'a' channel) with their own tilt / tint ('g' channel at the
        // same scale: varies leaf to leaf), sprigs (~10 cm) when the leaves get small
        float lv = 0.5, lid = 0.5;
        if (lodM < 1.0) {
            vec4 lf4 = lodF < 1.0 ? texture(uNoise3, vec3(q * 1.6, h * 1.6 + 0.2)) : vec4(0.5);
            vec4 lc4 = lodF > 0.0 ? texture(uNoise3, vec3(q * 0.6, h * 0.6 + 0.7)) : vec4(0.5);
            lv = mix(lf4.a, lc4.a, lodF);
            lid = mix(lf4.g, lc4.g, lodF);
        }
        float thr = mix(0.50, 0.62, hn * hn);
        float w = 0.015 + 1.5 * foot;
        float a = smoothstep(thr - w, thr + w, lv);
        a = mix(a, mix(0.8, 0.45, hn * hn) * cdec, lodM);
        // leaf colour: deep green, lit tops yellow-green (where the sun reaches the floor), single
        // glossy leaves catching the light
        vec3 col = mix(vec3(0.026, 0.050, 0.007), vec3(0.058, 0.100, 0.014), smoothstep(0.3, 0.8, c4.b));
        col = mix(col, vec3(0.078, 0.135, 0.024), smoothstep(0.75, 1.0, hn) * (0.55 + 0.45 * sunR) * 0.5);   // (veg4 G3: a little less yellow-olive; r6's sunlit carpet read lime at 0.070 / 0.150)
        // single leaves: each tilted its own way - some catch the sun, most are in their neighbours'
        // shade (strong leaf-to-leaf contrast)
        float lvl = smoothstep(0.3, 0.7, lid);
        col *= mix(1.0, mix(0.6, 1.2, smoothstep(thr, thr + 0.12, lv)) * mix(0.4, 1.5, lvl), 1.0 - lodM);
        col = mix(col, col * vec3(1.5, 1.1, 0.7), step(0.88, fract(c4.r * 17.0)) * 0.5);   // reddening / yellowing sprigs
        col *= mix(0.10, 1.05, pow(hn, 0.8));
        acc += T * a * col;
        T *= 1.0 - a;
    }
    return acc + T * base;
}

#define GC_FAR 0.85         // grass seen from afar: the blades' mean colour x this (canopy shading)
#define GC_FARGAIN 1.6       // (old far palette -> blade canopy brightness, for the straw patches)
vec3 groundVegAlbedo(vec3 p, inout vec3 n, float t, vec4 f, float ao) {
    vec2 x = p.xz;
    float camD = length(p - uCamPos);
    float foot = max(camD * gPixAng, 0.0015);           // pixel footprint (m)
    float meadow = f.a;
    vec3 n0 = n;
    gcGroundSkyOcc = 1.0;
    // forest-floor weight (as for the 3D cover): under conifer crowns, or inside / next to a stand
    float stand = 0.0;
    float canopy = camD < 80.0 ? gcCanopyAt(x, stand) : 0.0;
    float nearW = 1.0 - smoothstep(55.0, 80.0, camD);
    canopy *= nearW; stand *= nearW;
    float wl = waterLevel(x);
    float shoreG = wl > -1000.0 ? 1.0 - smoothstep(0.6, 2.2, p.y - wl + 0.5 * gnoise(x * 0.45 + 1.0)) : 0.0;
    float wood = saturate(max(camD < 150.0 ? max(f.r * 1.3, gcWoodAt5(x) * 1.8) : f.r * 1.3, stand)) * (1.0 - 0.85 * meadow) * (1.0 - 0.85 * shoreG);
    if (uDebug == 12) return vec3(saturate(f.r * 1.3), stand, canopy);    // look-dev: habitat
    float forest = max(max(canopy, 0.9 * smoothstep(0.2, 0.6, wood)), saturate(f.r * 1.25));
    forest = max(forest, 0.95 * smoothstep(0.5, 0.9, stand) * (1.0 - shoreG));
    // (i17) the forest trail's corridor is forest floor (as gcFloor for the 3D cover)
    float dTr = 1e9;
    if (uTrailN > 1 && camD < 120.0) {
        for (int i = 0; i < 31; i++) {
            if (i + 1 >= uTrailN) break;
            vec2 a2 = uTrail[i].xy, ab2 = uTrail[i + 1].xy - a2;
            dTr = min(dTr, length(x - a2 - ab2 * clamp(dot(x - a2, ab2) / dot(ab2, ab2), 0.0, 1.0)));
        }
    }
    forest = max(forest, 0.85 * (1.0 - smoothstep(4.5, 8.0, dTr)) * (1.0 - shoreG) * nearW);
    forest = mix(forest, gcFloorW(forest), nearW);
    if (uDebug == 19) return vec3(forest, wood, meadow);                  // look-dev: floor weight
    // ---- grassland tone: lush hollows, olive slopes, dry straw on sunny south faces (several scales)
    // (the blades' mean colour times the canopy's average shading: base darkening, self-shadow)
    float dry = gcDryness(x, n, ao, meadow);
    vec3 straw = vec3(0.150, 0.128, 0.045) * GC_FARGAIN;
    vec3 g = gcBladeMean(dry) * GC_FAR;
    // 4-8 m patch mosaic: lime forb patches, straw patches, dark dwarf-shrub clumps, +-25 % luminance
    {
        float pa = vnoise(x * 0.17 + 7.0), pb = vnoise(x * 0.23 + 19.0), pc = vnoise(x * 0.13 + 31.0);
        // (i22: crisper, ragged-edged patches instead of soft gradients - the soft blends read as camouflage)
        float pe = 0.10 * (vnoise(x * 1.3 + 2.0) - 0.5);
        g = mix(g, g * vec3(1.15, 1.20, 0.90), smoothstep(0.66, 0.72, pa + pe));
        g = mix(g, straw, smoothstep(0.70, 0.76, pb + pe) * 0.6);
        g = mix(g, vec3(0.030, 0.040, 0.010) * GC_FARGAIN, smoothstep(0.80, 0.90, pc) * (1.0 - forest) * 0.85);
        g *= 0.86 + 0.28 * vnoise(x * 0.31 + 3.0);      // (i21: +-22 % -> +-14 %: soft camouflage blotches)
    }
    {
        float pz = gcPatch(x);
        g *= mix(0.78, 1.06, pz);
        g = mix(g, g * vec3(0.88, 1.0, 1.1), (1.0 - pz) * 0.3);
    }
    // lake and stream shores: olive, less saturated grass (#757a32 .. #8f9147)
    float shoreO = wl > -1000.0 ? 1.0 - smoothstep(1.0, 4.0, p.y - wl) : 0.0;
    // (i17) the inflow stream's grassy floodplain (shot 4): olive as well, not lime
    if (camD < 600.0) {
        vec2 sdF = streamDist(x);
        shoreO = max(shoreO, 0.85 * (1.0 - smoothstep(4.0 * sdF.y, 9.0 * sdF.y, sdF.x)));
    }
    g = mix(g, vec3(luma(g)) * vec3(1.06, 1.04, 0.58), 0.7 * shoreO * (1.0 - forest)) * mix(1.0, 0.85, shoreO * (1.0 - forest));   // (veg4 G1: shot04 floodplain L 0.63 C 0.086 vs target L 0.46-0.50 C <= 0.054)   // (i24: 0.35 -> 0.5; i26 greyer, not darker: target lit grass #8f9147 / (113, 122, 69))   // (i24: 0.35 -> 0.5, a little darker)   // (i20: olive, was greyish 0.66 blue)
    // the flowering bench by the tarn reads warmer from afar
    float fpat = vnoise(x * 0.31 + 41.0);
    g = mix(g, g * vec3(1.10, 0.98, 1.05), meadow * 0.5 * smoothstep(0.3, 0.75, fpat) * smoothstep(0.08, 0.3, foot));
    float midW = 1.0 - smoothstep(0.05, 0.3, foot);
    // fraction of the 3D cover still drawn here (it thins out by count, then shrinks)
    float k3 = min(gcKeep(camD), 1.0) * gcShrink(camD);
    float texIn = 1.0 - k3;                                     // texture takes over from the 3D cover
    if (midW > 0.0) {
        float l3 = gnoise(x * 0.23 + 5.0);
        g *= 1.0 + 0.12 * l3 * midW;
    }
    // (veg3) beyond the 3D cover: dwarf shrubs (juniper, alpenrose) and grey stones dotted through the
    // turf, ~3 m cells (the grass slopes are never a clean lawn); area-weighted when small
    bool waterPass = false;
    // (i17) micro-relief of the open turf: hummocks (~1.2 m) and tussock mounds (~0.45 m) tilt the shading
    // normal - lit and shaded sides at 10-150 m (the turf is never a smooth sheet). Faded by the
    // pixel footprint (features kept >= ~5 px): no shimmer
    {
        float mrA = (1.0 - forest) * (waterPass ? 0.0 : 1.0);
        float w0 = 1.0 - smoothstep(0.25, 0.7, foot);
        float w1 = 1.0 - smoothstep(0.08, 0.24, foot);
        float w2 = 1.0 - smoothstep(0.03, 0.09, foot);
        if (mrA * w0 > 0.0) {
            vec3 h0 = gnoised(rot2(2.1) * x * 0.29 + 41.0);
            vec2 gr = (h0.yz * rot2(2.1)) * (0.29 * 0.40) * w0;                 // ~3.5 m swells, 40 cm (i23: was 30)
            if (w1 > 0.0) {
                vec3 h1 = gnoised(x * 0.85 + 17.0);
                vec3 h2 = gnoised(rot2(0.7) * x * 2.2 + 3.0);
                gr += h1.yz * (0.85 * 0.14) * w1 + (h2.yz * rot2(0.7)) * (2.2 * 0.05) * w2;   // (i23: 10 / 3.5 cm -> 14 / 5 cm)
            }
            n = normalize(n - vec3(gr.x, 0.0, gr.y) * mrA);
        }
    }
    // (i27: also in sparse forest / stand zones - r7's treed grass slope read as felt; laid over the floor)
    float dotW = smoothstep(8.0, 16.0, camD) * (1.0 - smoothstep(0.35, 0.7, foot)) * (1.0 - 0.55 * forest);   // (veg4 G1: from 8 m, was 14)
    vec4 stF = vec4(0.0);          // (i17) stones / shrub cushions beyond the 3D cover: colour, coverage (laid over the sward)
    if (dotW > 0.0) {
        // (i20) analytic domes seen along the camera ray: granite stones (#b5b097, lichen) and dark
        // dwarf-shrub cushions (juniper, alpenrose, bilberry), 3 m cells. They stand up (silhouette,
        // lit / shaded flank, a ragged grass fringe hides the base) and cast their shadow on the turf. The
        // flat discs of i17 read as grey smears at grazing angles. Area-weighted when sub-pixel (no sparkle).
        vec3 Vd = normalize(uCamPos - p);
        vec2 ud = Vd.xz / max(dot(Vd, n0), 0.05);
        float lu = length(ud);
        if (lu > 4.0) ud *= 4.0 / lu;                         // (extreme grazing: flatter domes, <= 2 cells)
        vec2 us = uSunDir.xz / max(uSunDir.y, 0.25);
        if (length(us) > 3.0) us *= 3.0 / length(us);
        const float HMX = 0.7;
        vec2 bmin = min(x, min(x + ud * HMX, x + us * HMX)), bmax = max(x, max(x + ud * HMX, x + us * HMX));
        ivec2 c0 = ivec2(floor(bmin / 3.0)), c1 = min(ivec2(floor(bmax / 3.0)), c0 + 2);
        // (i21) clustered: stone litter in patches (~9 m noise), heath cushions merging into patches
        // (~11 m noise) - evaluated at the cell centre (no cut-off domes); the i20 even spread read as
        // polka dots / a sheep pasture
        float clu = smoothstep(0.25, 0.65, vnoise(x * 0.05 + 2.0) + 0.2 * (1.0 - dry));
        // (i26) steep turf: rock outcrops and shrubs poke through far more (r7 / s5 slopes read as felt)
        float stp = smoothstep(0.08, 0.35, 1.0 - n0.y);
        float pStB = (0.035 + 0.09 * meadow) * mix(0.5, 1.3, clu) * (1.0 + 2.2 * stp);
        float pShB = (0.24 - 0.20 * meadow) * (1.0 - 0.5 * shoreG) * (1.0 + 1.2 * shoreO) * (1.0 + 1.0 * stp);   // (i24: more on the floodplains)
        float pMax = pStB * 3.0 + pShB * 2.6;
        float bestH = -1.0, shw = 0.0;
        vec4 dHit = vec4(0.0);      // colour, coverage of the nearest dome hit
        float wF = 1.0 - smoothstep(0.03, 0.09, foot);        // grass fringe / grain while resolved
        for (int iy = 0; iy < 3; iy++) {
            for (int ix = 0; ix < 3; ix++) {
                ivec2 cc = c0 + ivec2(ix, iy);
                if (cc.x > c1.x || cc.y > c1.y) continue;
                vec4 hh = hash42i(cc + ivec2(3301, 7919));
                if (hh.w >= pMax) continue;
                vec2 ccx = (vec2(cc) + 0.5) * 3.0;
                float pSt = min(pStB * mix(0.15, 3.0, smoothstep(0.42, 0.80, vnoise(ccx * 0.11 + 7.0))), 0.6);
                float pSh = pShB * mix(0.05, 2.6, smoothstep(0.40, 0.78, vnoise(ccx * 0.09 + 31.0)));
                bool isSt = hh.w < pSt;
                if (!isSt && hh.w >= pSt + pSh) continue;
                float r = fract(hh.z * 7.13);
                float R = isSt ? mix(0.16, 0.95, r * r) : mix(0.40, 1.45, r);
                float hs = isSt ? R * mix(0.42, 0.85, fract(hh.z * 3.7)) : mix(0.20, 0.55, fract(hh.z * 3.7)) * min(R / 0.6, 1.0);
                // stones half buried (a cap shows: broader, flatter), cushions a little
                float z0 = hs * (isSt ? mix(0.3, 0.7, fract(hh.x * 9.3)) : 0.15);   // (veg4 G10: sunk 0.3-0.7)
                vec2 cen = (vec2(cc) + 0.5 + (hh.xy - 0.5) * (1.0 - 2.0 * R / 3.0)) * 3.0;
                // sub-pixel: grow to ~0.6 px and weight by the area ratio (constant energy)
                float Re = max(R, 0.6 * foot), kA = (R * R) / (Re * Re);
                float hsE = hs * Re / R, z0E = z0 * Re / R;
                vec2 d = x - cen;
                float iR2 = 1.0 / (Re * Re), iH2 = 1.0 / (hsE * hsE);
                // camera ray: f(h) = |d + u h|^2 / R^2 + (h + z0)^2 / hs^2 (min fm for the edge AA), first hit hH
                float A = dot(ud, ud) * iR2 + iH2, B = dot(d, ud) * iR2 + z0E * iH2, C = dot(d, d) * iR2 + z0E * z0E * iH2;
                float hm = max(-B / A, 0.0);
                float fm = C + 2.0 * B * hm + A * hm * hm;
                float wE = min(1.6 * foot / Re, 0.9);
                if (fm < 1.6 + wE) {
                    float disc = B * B - A * (C - 1.0);
                    float hH = disc > 0.0 ? (-B + sqrt(disc)) / A : hm;
                    if (hH > bestH) {
                        vec2 Pm = d + ud * hm;                         // closest approach: outline angle
                        float th = atan(Pm.y, Pm.x) + hh.x * TAU;
                        float thr;
                        if (isSt) {
                            // polygonal outline (5-8 sides), resolved only
                            float nf = 5.0 + floor(fract(hh.y * 7.7) * 4.0), sa = TAU / nf;
                            float pr = cos(0.5 * sa) / cos(mod(th, sa) - 0.5 * sa);
                            thr = mix(1.0, pr * pr, wF);
                        } else {
                            // ragged, lobed cushion
                            // (i23: the outline stays ragged while the lobes are a few px, not only near)
                            float wFo = 1.0 - smoothstep(0.10 * R, 0.30 * R, foot);
                            thr = 1.0 + 0.45 * (vnoise(x * (2.2 / R) + hh.z * 50.0) - 0.5) * wFo + 0.3 * (vnoise(x * (6.0 / R) + hh.y * 20.0) - 0.5) * wF;
                        }
                        float a = (1.0 - smoothstep(thr - wE, thr + wE, fm)) * kA;
                        // grass fringe in front of the base (resolved), its mean when small
                        float hg = (isSt ? 0.08 : 0.05) + 0.14 * meadow * vnoise(x * 9.0 + hh.z * 20.0);
                        a *= mix(0.8, smoothstep(hg - 0.6 * foot, hg + 0.6 * foot, hH), wF);
                        if (a > 0.0) {
                            bestH = hH;
                            vec2 P = d + ud * hH;
                            vec3 nd = normalize(vec3(P.x * iR2, (hH + z0E) * iH2, P.y * iR2));
                            float lw = saturate(kA * 1.5);
                            vec3 col;
                            if (isSt) {
                                // faceted: a flat-ish top facet, flat side facets per outline sector
                                float nf = 5.0 + floor(fract(hh.y * 7.7) * 4.0), sa = TAU / nf;
                                float thP = atan(P.y, P.x) + hh.x * TAU;
                                float phs = (floor(thP / sa) + 0.5) * sa - hh.x * TAU;
                                vec3 nSide = normalize(vec3(cos(phs), 0.55 + 0.6 * fract(hh.z * 13.3), sin(phs)));
                                vec3 nTop = normalize(vec3((fract(hh.y * 3.1) - 0.5) * 0.5, 1.0, (fract(hh.x * 5.9) - 0.5) * 0.5));
                                float rr = length(P) / Re;
                                vec3 nF = normalize(mix(nTop, nSide, smoothstep(0.50, 0.58, rr + 0.25 * (1.0 - (hH + z0E) / hsE))));
                                nd = normalize(mix(nd, nF, 0.8 * lw * wF));
                                col = vec3(0.27, 0.26, 0.23) * mix(0.75, 1.12, fract(hh.z * 7.7));   // (veg4 G10: 0.34 -> 0.27, white popcorn)
                                col = mix(col, vec3(0.25, 0.26, 0.19), step(0.6, fract(hh.z * 3.9)) * 0.4);   // grey-green lichen
                                col *= mix(1.0, 0.7 + 0.5 * vnoise(x * 6.0 + hh.x * 40.0), wF);          // lichen / grain
                            } else {
                                // juniper blue-green / alpenrose dark / bilberry bright green; lit tops
                                float sp = fract(hh.z * 5.31);
                                // (i22: between i20 (holes) and i21 (grass-coloured pads))
                                // (veg4 G9: lifted - the near-black cushions read as holes / puddles beyond 14 m)
                                col = sp < 0.4 ? vec3(0.055, 0.075, 0.035) : (sp < 0.75 ? vec3(0.050, 0.075, 0.022) : vec3(0.085, 0.120, 0.026));
                                col = mix(col, vec3(0.024, 0.048, 0.008), 0.7 * forest);      // understory bilberry / young spruce (#273d11)
                                col *= mix(1.0, 0.42 + 0.75 * vnoise(x * 7.0 + hh.y * 30.0) + 0.45 * vnoise(x * 19.0 + hh.x * 30.0), wF) * mix(0.7, 1.2, nd.y);   // leafy grain
                            }
                            float litR = clamp(dot(nd, uSunDir) / max(uSunDir.y, 0.25), isSt ? 0.30 : 0.45, 1.6);   // (veg4 G9: cushions 0.18 -> 0.45)
                            dHit = vec4(col * mix(1.0, litR, lw), a);
                        }
                    }
                }
                // its shadow on the turf (the sun ray from x through the dome)
                float As = dot(us, us) * iR2 + iH2, Bs = dot(d, us) * iR2 + z0E * iH2;
                float hms = max(-Bs / As, 0.0);
                float fs = C + 2.0 * Bs * hms + As * hms * hms;
                shw = max(shw, (1.0 - smoothstep(1.0 - wE - 0.15, 1.0 + wE + 0.1, fs)) * kA);
            }
        }
        // (veg4 G1b) grass / forb clumps on 1 m cells, 8 m out to a ~0.25 m footprint: 0.2-0.5 m clumps standing
        // 0.2-0.45 m out of the sward with a lit top, a shaded flank and a cast shadow - the plant-scale
        // light / shade at 20-100 m (r3 / shot05 read as a smooth lawn with mottling). Same ray-vs-dome test as
        // the stones / cushions above; area-weighted when sub-pixel, faded out by the footprint (no shimmer)
        float wCl = (1.0 - smoothstep(0.12, 0.25, foot)) * (1.0 - smoothstep(0.1, 0.4, forest)) * (1.0 - shoreO) * (waterPass ? 0.0 : 1.0);
        float pCl = (0.15 + 0.35 * meadow) * (1.0 - 0.5 * dry);
        if (wCl * pCl > 0.0) {
            const float HC = 0.5;
            vec2 bmn = min(x, min(x + ud * HC, x + us * HC)), bmx = max(x, max(x + ud * HC, x + us * HC));
            ivec2 k0 = ivec2(floor(bmn)), k1 = min(ivec2(floor(bmx)), k0 + 2);
            for (int iy = 0; iy < 3; iy++) {
                for (int ix = 0; ix < 3; ix++) {
                    ivec2 cc = k0 + ivec2(ix, iy);
                    if (cc.x > k1.x || cc.y > k1.y) continue;
                    vec4 hh = hash42i(cc + ivec2(6007, 1213));
                    if (hh.w >= pCl) continue;
                    float R = mix(0.2, 0.5, hh.z);
                    float hs = mix(0.3, 0.7, fract(hh.z * 5.3 + hh.x));
                    float z0 = hs * 0.35;                       // the lower part hides in the sward
                    vec2 cen = vec2(cc) + 0.5 + (hh.xy - 0.5) * (1.0 - 2.0 * R);
                    float Re = max(R, 0.6 * foot), kA = (R * R) / (Re * Re);
                    float hsE = hs * Re / R, z0E = z0 * Re / R;
                    vec2 d = x - cen;
                    float iR2 = 1.0 / (Re * Re), iH2 = 1.0 / (hsE * hsE);
                    float A = dot(ud, ud) * iR2 + iH2, B = dot(d, ud) * iR2 + z0E * iH2, C = dot(d, d) * iR2 + z0E * z0E * iH2;
                    float hm = max(-B / A, 0.0);
                    float fm = C + 2.0 * B * hm + A * hm * hm;
                    float wE = min(1.6 * foot / Re, 0.9);
                    if (fm < 1.6 + wE) {
                        float disc = B * B - A * (C - 1.0);
                        float hH = disc > 0.0 ? (-B + sqrt(disc)) / A : hm;
                        if (hH > bestH) {
                            // ragged leafy outline while resolved
                            float wFo = 1.0 - smoothstep(0.08 * R, 0.25 * R, foot);
                            float thr = 1.0 + 0.5 * (vnoise(x * (2.5 / R) + hh.z * 50.0) - 0.5) * wFo + 0.35 * (vnoise(x * (7.0 / R) + hh.y * 20.0) - 0.5) * wF;
                            float a = (1.0 - smoothstep(thr - wE, thr + wE, fm)) * kA * wCl;
                            if (a > 0.0) {
                                bestH = hH;
                                vec2 P = d + ud * hH;
                                vec3 nd = normalize(vec3(P.x * iR2, (hH + z0E) * iH2, P.y * iR2));
                                float lw = saturate(kA * 1.5);
                                // dark lush forb clumps / blue-green fescue / yellow-green grass (never one lawn colour;
                                // all green - B3's grey fescue tint and 0.35 flanks read as stones / molehills in r3)
                                float sp = fract(hh.z * 5.31 + hh.y);
                                vec3 col = sp < 0.4 ? g * vec3(0.72, 0.90, 0.58) : (sp < 0.7 ? g * vec3(0.86, 1.0, 1.10) : g * vec3(1.06, 1.05, 0.84));
                                col *= mix(1.0, 0.55 + 0.6 * vnoise(x * (9.0 / R) + hh.x * 30.0) + 0.3 * vnoise(x * 23.0 + hh.y * 30.0), wF);   // leafy grain
                                if (fract(hh.x * 7.9) < 0.3 * meadow) col = mix(col, gcFlowerCol(gcSpecies(cen, 0.5)), 0.2);   // flowering clump
                                float litR = dot(nd, uSunDir) / max(uSunDir.y, 0.25);
                                col *= mix(1.0, clamp(1.15 * litR, 0.45, 1.25), lw);
                                dHit = vec4(col, a);
                            }
                        }
                    }
                    float As = dot(us, us) * iR2 + iH2, Bs = dot(d, us) * iR2 + z0E * iH2;
                    float hms = max(-Bs / As, 0.0);
                    float fs = C + 2.0 * Bs * hms + As * hms * hms;
                    shw = max(shw, (1.0 - smoothstep(1.0 - wE - 0.15, 1.0 + wE + 0.1, fs)) * kA * wCl * 0.8);
                }
            }
        }
        g *= 1.0 - 0.45 * shw * dotW;          // (veg4 G9: 0.62 -> 0.45)
        stF = vec4(dHit.rgb, dHit.a * dotW);
    }
    // ---- the sward canopy (shell volume) out to ~150 m; beyond, its mean: the blade canopy, where
    //      the blades leave gaps (looking down on them) the dark sward base shows
    vec3 gS = g;
    // near field: under the shells lies a low leafy layer (lady's mantle, clover, plantain,
    // seedlings), thatch in the gaps; the shells fade in from 3.5-7 m
    float nearS = smoothstep(3.5, 7.0, camD);
    float under = k3 * (1.0 - forest) * (1.0 - nearS);
    vec3 leafB = vec3(0.0);
    if (under > 0.0) {
        vec2 nOff; float cov;
        float lushT = saturate(0.6 + 0.6 * meadow - 0.6 * dry + 0.3 * gnoise(x * 0.9));
        leafB = gcLeafMat(x, foot, lushT, nOff, cov);
        vec3 thatch = mix(vec3(0.050, 0.050, 0.020), vec3(0.090, 0.078, 0.036), saturate(0.5 + gnoise(x * 3.7)));
        leafB = mix(leafB, thatch, (1.0 - cov) * 0.5 * smoothstep(0.4, 0.9, dry)) * 0.75;
        // (i21) the floor of the 3D sward lies in the blades' shade, sun flecks through the gaps (c:mfg: a lit,
        // flat leaf "floor" between the plants read painted); flecks fade to their mean when sub-pixel
        float flk = mix(smoothstep(0.40, 0.75, vnoise(x * 9.0 + 5.0) * 0.7 + vnoise(x * 23.0 + 1.0) * 0.3), 0.35, smoothstep(0.004, 0.012, foot));
        leafB *= mix(0.5, 1.05, flk);
    }
    {
        vec3 V = normalize(uCamPos - p);
        float covV = 1.0 - exp(-1.2 / max(dot(V, n0), 0.12));
        // (the shells' mean: clump tops with pale tips over the dark depths)
        vec3 gFar = mix(g * 0.42, mix(g, vec3(0.34, 0.35, 0.15), 0.1) * 0.92, covV);
        // beyond the shells: their clump / patch-type structure as a footprint-filtered texture
        // (~0.5 m tussocks with dark gaps, ~1 m forb / fescue patches), no lawn
        float spW = 1.0 - smoothstep(0.15, 0.5, foot);
        if (spW > 0.0) {
            // (veg4 G1) the shells' own clump field (the same texture) instead of isotropic value-noise mottling,
            // which read as camouflage on a lawn: ~0.3-0.6 m clumps with dark gaps while >= ~4 px, ~2 m clump
            // groups / shrub-scale patches further out (no step where the shells hand over)
            vec4 cF = texture(uNoise3, vec3(x * 0.22, 0.31));
            float cG = texture(uNoise3, vec3(x * 0.075, 0.57)).b;
            float w0 = 1.0 - smoothstep(0.07, 0.16, foot);
            float wS = 1.0 - smoothstep(0.12, 0.3, foot);
            float kF = 1.0;
            gFar *= mix(1.0, mix(1.0 - 0.44 * kF, 1.0 + 0.36 * kF, smoothstep(0.30, 0.72, cF.b)) * mix(0.80, 1.16, smoothstep(0.30, 0.80, cF.a)), w0);
            gFar *= mix(1.0, mix(1.0 - 0.34 * kF, 1.0 + 0.22 * kF, smoothstep(0.25, 0.80, cG)), spW);
            // (i23) sun relief of the clump field beyond the shells (the shells' self-shadow, as an emboss: the
            // field rising toward the sun = shade side): 28 cm clumps, then ~0.9 m tussock mounds
            float eW0 = 1.0 - smoothstep(0.09, 0.20, foot), eW1 = 1.0 - smoothstep(0.30, 0.65, foot);
            vec2 sH = normalize(uSunDir.xz + 1e-4);
            float e0 = eW0 > 0.0 ? texture(uNoise3, vec3((x + sH * 0.15) * 0.22, 0.31)).a - cF.a : 0.0;
            float e1 = texture(uNoise3, vec3((x + sH * 0.45) * 0.07, 0.83)).a - texture(uNoise3, vec3(x * 0.07, 0.83)).a;
            gFar *= 1.0 + clamp(-2.0 * e0, -0.40, 0.28) * eW0 + clamp(-2.0 * e1, -0.40, 0.28) * eW1 * spW;
            // patch types as in the shells: dark forb clumps, glaucous fescue
            gFar = mix(gFar, gFar * vec3(0.72, 0.92, 0.6), wS * smoothstep(0.55, 0.63, cF.r) * 0.5);
            gFar = mix(gFar, vec3(luma(gFar)) * vec3(0.95, 1.04, 0.42), wS * (1.0 - smoothstep(0.33, 0.41, cF.r)) * 0.4);
        }
        // (i17) flower drifts seen from afar: their colour mixes into the sward (area-weighted, about the
        // shells' head coverage), species from the same cluster field (yellow / white / magenta washes)
        float flF = meadow * smoothstep(0.3, 0.7, fpat) * (1.0 - forest) * (1.0 - 0.6 * dry);
        if (flF > 0.0) gFar = mix(gFar, gcFlowerCol(gcSpecies(x, 0.5)) * 0.9, flF * 0.06);
        float shellW = (1.0 - smoothstep(0.08, 0.14, foot)) * (1.0 - smoothstep(0.6, 0.8, forest))     // (veg4 G1: 0.06-0.10 -> 0.08-0.14)
                     * smoothstep(0.45, 0.55, n0.y) * smoothstep(-0.2, 0.3, p.y - wl) * (waterPass ? 0.0 : 1.0);
        vec3 gNear = g;
        if (under > 0.0) {
            vec3 sb0 = mix(vec3(0.020, 0.025, 0.007), vec3(0.050, 0.042, 0.018), dry);
            gNear = mix(g, mix(sb0, leafB, under), 0.85);
        }
        if (shellW > 0.0 && nearS > 0.0) {
            float pat = vnoise(x * 0.55 + 13.0);
            float H = mix(0.13, 0.30, pat) * (1.0 + 0.45 * meadow) * (1.0 - 0.35 * dry);
            // flowers (beyond the 3D ones; 1-5 % of the meadow pixels, in drifts)
            float flD = (0.06 + 0.8 * meadow * smoothstep(0.3, 0.7, fpat)) * (1.0 - forest) * (0.4 + 0.6 * texIn) * (1.0 - 0.6 * dry);   // (i17: was 0.045 + 0.5)
            // stones (beyond the 3D ones), broad-leaved forbs (more on the meadows)
            float stP = (0.04 + 0.08 * meadow) * smoothstep(0.35, 0.6, vnoise(x * 0.12 + 5.0)) * (1.0 - forest) * (0.3 + 0.7 * texIn);
            float pF = (0.12 + 0.45 * meadow * (1.0 - gcPatch(x))) * (1.0 - forest) * (1.0 - 0.7 * dry) * smoothstep(0.004, 0.009, foot);
            vec3 sb = mix(vec3(0.020, 0.025, 0.007), vec3(0.050, 0.042, 0.018), dry);   // thatch, moss, soil
            sb = mix(sb, leafB, under);
            gS = mix(gNear, gcSward(x, n0, V, foot, H, g, sb, dry, flD, stP, pF), nearS);
        } else gS = gNear;
        g = mix(gFar, gS, shellW);
        // (veg4 G15) low sun: the far turf's chroma held back (the golden light warms it; shot08's forest-gap
        // meadows fluoresced lime-yellow, C p90 0.117)
        g = mix(g, vec3(luma(g)), 0.4 * smoothstep(0.2, 0.5, foot) * (1.0 - smoothstep(0.35, 0.6, uSunDir.y)));
        if (forest <= 0.01) g = mix(g, stF.rgb, stF.a);
    }
    // ---- forest floor: needle litter under the crowns, moss cushions, golden straw and dust where
    //      the sun reaches the floor, bilberry patches (3D bushes up close, texture beyond), roots
    if (forest > 0.01) {
        float m1 = gnoise(x * 0.37 + 2.0) + 0.5 * gnoise(x * 1.3 + 9.0);
        // (low blue: the floor is mostly seen in shade under a blue sky light, and reads olive / golden)
        vec3 moss = mix(vec3(0.030, 0.050, 0.006), vec3(0.066, 0.090, 0.012), saturate(0.5 + 0.7 * gnoise(x * 2.9)));
        vec3 litter = mix(vec3(0.075, 0.052, 0.020), vec3(0.160, 0.115, 0.042), saturate(0.5 + 0.8 * gnoise(x * 5.3 + 1.0)));
        litter = mix(litter, vec3(0.105, 0.056, 0.017), smoothstep(0.2, 0.6, gnoise(x * 1.7 + 4.0)) * 0.5);   // rusty needles
        if (foot < 0.02) {
            // needle litter: random orientation per 0.4 m cell, local coordinates (no swirl / aliasing)
            ivec2 nc = ivec2(floor(x / 0.4));
            vec2 xl = x - (vec2(nc) + 0.5) * 0.4;
            float fw = 1.0 - smoothstep(0.0012, 0.003, foot);
            if (fw > 0.0) {
                float nd = vnoise(rot2(hash12i(nc) * TAU) * xl * vec2(160.0, 22.0) + hash12i(nc + 7) * 31.0);
                litter *= mix(1.0, 0.75 + 0.5 * nd, fw);
            }
            moss *= mix(1.0, 0.8 + 0.4 * vnoise(x * 45.0), 1.0 - smoothstep(0.004, 0.015, foot));
            // twigs and cones
            vec4 tv = voronoiV(x * 3.0 + 1.7);
            float tw = (1.0 - smoothstep(0.05, 0.09, tv.z)) * step(0.82, fract(tv.w * 7.1)) * (1.0 - smoothstep(0.008, 0.02, foot));
            litter = mix(litter, vec3(0.055, 0.036, 0.014), tw * 0.8);
        }
        litter = gcFloorDebris(x, foot, litter);
        float mossM = smoothstep(0.15, 0.6, m1) * mix(0.45, 0.85, canopy);
        vec3 fl = mix(litter, moss, mossM);
        // mottling of the litter: dark debris / damp hollows and lighter patches at 0.3 - 2 m, and
        // dark deep litter between roots and stones (floor p10 is near black)
        fl *= 0.35 + 1.0 * vnoise(x * 1.1 + 3.0) + 0.3 * (vnoise(x * 3.7 + 8.0) - 0.5);
        fl *= mix(1.0, 0.45, smoothstep(0.62, 0.85, vnoise(x * 0.8 + 21.0)));
        // sun on the floor: outside the crowns' shadows (crowns ~8 m up, cast along the sun) and
        // outside the crowns themselves. There the floor is dry: straw (the 3D tussocks' patch field
        // densest) and dusty bare patches (floor luma p10 19 .. p90 167)
        float canopyS = canopy;
        if (camD < 50.0) {
            float dummy;
            canopyS = mix(gcCanopyAt(x + uSunDir.xz / max(uSunDir.y, 0.3) * 8.0, dummy), canopy, smoothstep(35.0, 50.0, camD));
        }
        float sunR = (1.0 - canopyS) * (1.0 - 0.7 * canopy);
        float gp = smoothstep(0.5, 0.75, vnoise(x * 0.6 + 3.0));
        float tuT = smoothstep(0.42, 0.72, vnoise(x * 2.2 + 9.0));
        vec3 strawF = mix(vec3(0.19, 0.145, 0.052), vec3(0.27, 0.205, 0.078), saturate(0.5 + gnoise(x * 4.1)));
        strawF *= 0.82 + 0.3 * vnoise(x * 1.3 + 5.0);
        vec3 dust = mix(vec3(0.25, 0.195, 0.135), vec3(0.35, 0.28, 0.20), saturate(0.5 + gnoise(x * 3.3 + 2.0)));
        float sw = sunR * saturate(0.25 + 0.75 * gp * mix(0.4, 1.0, tuT) + 0.3 * smoothstep(0.55, 0.8, vnoise(x * 0.45 + 13.0)));
        // (i22: the trail corridor's verges are planted, not dusty - r2 read as sand patches)
        float trvD = smoothstep(0.6, 1.1, dTr) * (1.0 - smoothstep(3.5, 7.0, dTr));   // (veg4 G4: narrower path)
        fl = mix(fl, dust, sunR * smoothstep(0.62, 0.8, vnoise(x * 0.8 + 61.0)) * (1.0 - gp) * 0.85 * (1.0 - 0.8 * trvD));
        fl = mix(fl, litter * 0.8, smoothstep(0.75, 1.0, canopy) * 0.4 * (1.0 - mossM * 0.5));
        if (camD < 32.0 && forest > 0.2) {
            vec2 rb;
            float rt = gcRoots(x, rb) * (1.0 - smoothstep(22.0, 32.0, camD));
            fl = mix(fl, mix(vec3(0.036, 0.029, 0.018), vec3(0.068, 0.056, 0.034), saturate(0.5 + gnoise(x * 9.0))), rt * 0.85);
            n = normalize(n + vec3(rb.x, 0.0, rb.y) * 0.6 * rt);
        }
        // (veg3) the floor's plants as shell volumes at every distance: dry golden grass tussocks
        // where the sun reaches the floor, a dense leafy bilberry / dwarf-shrub carpet in
        // patches, thickest along the stand edges and the trail verges; 3D bushes on top
        vec3 V = normalize(uCamPos - p);
        // (only where the floor carries real weight: scattered single trees on the meadows keep the
        // flat floor colour under their crowns - two shell volumes per pixel cost too much there)
        float fsW = (1.0 - smoothstep(0.05, 0.09, foot)) * smoothstep(0.42, 0.52, n0.y) * smoothstep(0.25, 0.45, forest) * (waterPass ? 0.0 : 1.0);
        float trv = smoothstep(0.6, 1.1, dTr) * (1.0 - smoothstep(3.5, 7.0, dTr));
        float bpc = vnoise(x * 0.33 + 5.0) + 0.25 * vnoise(x * 1.1 + 2.0) + 0.55 * trv + 0.2 * sunR - 0.2 * canopy * canopy;
        float cov = smoothstep(0.35, 0.75, bpc);
        if (uDebug == 33) return vec3(mossM, cov, sw) * 0.25;     // look-dev: floor layers (R moss, G carpet, B sunny straw)
        // (the grass shells only where the carpet leaves gaps: one shell volume per pixel mostly)
        if (fsW > 0.0 && sw > 0.05 && cov < 0.8) {
            float Hg = mix(0.12, 0.30, tuT) * smoothstep(0.05, 0.4, sw);
            vec3 gG = mix(strawF, gcBladeMean(0.5) * 0.9, 0.35 * (1.0 - sw));
            vec3 gr = gcSward(x, n0, V, foot, Hg, gG, fl, 0.9, 0.0, 0.03, 0.08 * sunR);
            fl = mix(fl, gr, fsW);
        }
        fl = mix(fl, strawF * 0.8, saturate(sw * 1.1) * (1.0 - fsW));
        if (fsW > 0.0 && cov > 0.02) {
            float Hc = mix(0.18, 0.38, vnoise(x * 0.7 + 1.0)) * (1.0 + 0.3 * trv);
            fl = mix(fl, gcCarpet(x, n0, V, foot, Hc, cov, fl, sunR), fsW);
        }
        // beyond the shells: the carpet's mean
        vec3 berry = mix(vec3(0.026, 0.052, 0.007), vec3(0.050, 0.088, 0.012), sunR);
        fl = mix(fl, berry, cov * 0.75 * (1.0 - fsW));
        g = mix(g, fl, forest);
        g = mix(g, stF.rgb, stF.a);
        // sky occluded by the crowns / the stand: exported for shadeTerrain (not baked into the
        // albedo, which would kill the sun flecks)
        gcGroundSkyOcc = mix(1.0, 0.75, canopy) * mix(1.0, 0.88, stand);    // (veg4 G14: 0.62 / 0.78 - with shadeTerrain's 0.45 the floor under a crown got ~0.22 sky: near black)
    }
    return g;
}

// ---------------------------------------------------------------------------
// hero.glsl : a few hand-placed granite blocks for specific shots (e.g. the
// rock the camera rises from behind in the opening reveal)
// (rocks3) the same construction as the scattered boulders (rocks.glsl) instead of a rounded box blended 38 % into
// an ellipsoid (a soft potato): a jointed block of 5 near-axis faces + 4 random cuts across edges / corners with
// crisp, unevenly weathered edges, gentle bulges, crumbly worley relief that is deeper along the edges, a fracture
// step on one top edge and a cm-scale octave near the camera
// ---------------------------------------------------------------------------
   // center xyz, size (m)
   // stretch xyz, yaw


// ---------------------------------------------------------------------------
// shade_rocks.glsl : boulder shading + granite cliff material (needs rocks.glsl + lighting.glsl)
// ---------------------------------------------------------------------------

// --- small analytic helpers -------------------------------------------------
vec3 rkGrad3(ivec3 c) {
    uint h = hash3u(c);
    return vec3(float(h & 1023u), float((h >> 10u) & 1023u), float((h >> 20u) & 1023u)) * (1.0 / 511.5) - 1.0;
}
// 3D gradient noise with analytic derivatives: (value, d/dx, d/dy, d/dz), value ~[-1,1]
vec4 rkNoised(vec3 x) {
    vec3 i = floor(x); vec3 f = x - i; ivec3 c = ivec3(i);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec3 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec3 ga = rkGrad3(c), gb = rkGrad3(c + ivec3(1, 0, 0)), gc = rkGrad3(c + ivec3(0, 1, 0)), gd = rkGrad3(c + ivec3(1, 1, 0));
    vec3 ge = rkGrad3(c + ivec3(0, 0, 1)), gf = rkGrad3(c + ivec3(1, 0, 1)), gg = rkGrad3(c + ivec3(0, 1, 1)), gh = rkGrad3(c + ivec3(1, 1, 1));
    float va = dot(ga, f), vb = dot(gb, f - vec3(1, 0, 0)), vc = dot(gc, f - vec3(0, 1, 0)), vd = dot(gd, f - vec3(1, 1, 0));
    float ve = dot(ge, f - vec3(0, 0, 1)), vf = dot(gf, f - vec3(1, 0, 1)), vg = dot(gg, f - vec3(0, 1, 1)), vh = dot(gh, f - vec3(1, 1, 1));
    float k7 = -va + vb + vc - vd + ve - vf - vg + vh;
    float v = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va) + u.x * u.y * (va - vb - vc + vd)
            + u.y * u.z * (va - vc - ve + vg) + u.z * u.x * (va - vb - ve + vf) + k7 * u.x * u.y * u.z;
    vec3 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga) + u.x * u.y * (ga - gb - gc + gd)
           + u.y * u.z * (ga - gc - ge + gg) + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z
           + du * (vec3(vb, vc, ve) - va + u.yzx * vec3(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
                   + u.zxy * vec3(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg) + u.yzx * u.zxy * k7);
    return vec4(v, d);
}
// relief of a few octaves: (height m, gradient), amplitude a0 (m) at frequency f0 (1/m), faded by footprint fp
vec4 rkRelief(vec3 q, float fp, float a0, float f0, int oct) {
    vec4 r = vec4(0.0);
    float a = a0, f = f0;
    for (int i = 0; i < oct; i++) {
        float vis = 1.0 - smoothstep(0.1, 0.35, fp * f);
        if (vis <= 0.0) break;
        vec4 n = rkNoised(q * f + float(i) * 17.3);
        r += vec4(n.x * a, n.yzw * (a * f)) * vis;
        a *= 0.45; f *= 2.3;
    }
    return r;
}
vec3 rkPerturb(vec3 n, vec3 g) { return normalize(n - (g - n * dot(g, n))); }

// C1-smooth sampling of the noise texture (removes polygonal contours of thresholded trilinear noise)
vec4 rkTexS(vec3 x) {
    vec3 p = x * 128.0 + 0.5;
    vec3 i = floor(p); vec3 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    return texture(uNoise3, (i + f - 0.5) * (1.0 / 128.0));
}

// --- granite surface (shared by boulders and terrain rock up close) -------------
// q: object/world position (m), up: normal.y in [-1,1], fp: pixel footprint (m),
// tone: per-rock brightness, steep: 0..1 (side faces). lich = lichen cover (for roughness/bump)
// 3D gradient noise, value only (~[-0.8,0.8], std ~0.2): smooth isotropic contours when thresholded

#define GRANITE_BASE vec3(0.19, 0.186, 0.177)
// mean of graniteDetail(tone 1, mapW 0, gRkCrust 0) over a block / face (all layers, any footprint; measured with
// tools/rkgdmean.py on the noise texture): terrain rock divides by it, so the detail never shifts the mean tone
#define GD_MEAN vec3(0.163, 0.157, 0.144)
vec3 rkGDMean(float steep) { return mix(vec3(0.167, 0.160, 0.147), vec3(0.156, 0.150, 0.138), steep); }
float gRkCrust = 0.0;    // per-block lichen-crust shift (shadeRock): > 0 dark-crusted blocks, < 0 fresh pale ones
float gRkMapC = 0.5;     // per-block map-lichen cover (0..1)
// thresholded noise with footprint filtering: v = fine noise (mean m, std sd), c = coarse part,
// fades to the expected coverage when the fine part is sub-pixel (vis -> 0)
float rkCov(float c, float v, float m, float sd, float thr, float e, float vis) {
    float x = c + mix(m, v, vis);
    float ee = e + (1.0 - vis) * sd * 1.6;
    return smoothstep(thr - ee, thr + ee, x);
}
// granite boulder surface: warm mid-grey granite. Smooth mottling at
// 5-40 cm (no thresholds: no marble / camouflage contours), pale crustose lichen blobs (~20 %, ragged
// edges) and dark lichen colonies (~12 %), small near-black crust spots clustered in the colonies,
// yellow-green map lichen patches on the upper faces (mapW = 0..1: per-rock amount), dark streaks down
// the sides, a fine salt-and-pepper grain. Every layer fades to its mean when it gets sub-pixel.
// Mean ~ GRANITE_BASE * tone.
vec3 graniteDetail(vec3 q, float up, float fp, float tone, float steep, float mapW, out float lich) {
    // scales and their footprint visibility (round 2 review: the 8 mm layer and the grain fade before they are
    // ~1 px - they shimmered in the moving film)
    float vK = 1.0 - smoothstep(0.04, 0.12, fp);       // 25 cm
    float vA = 1.0 - smoothstep(0.03, 0.09, fp);       // 14 cm
    float vB = 1.0 - smoothstep(0.012, 0.035, fp);     // 5 cm
    float vC = 1.0 - smoothstep(0.005, 0.014, fp);     // 2 cm
    float vD = 1.0 - smoothstep(0.002, 0.0045, fp);    // 8 mm
    float vS = 1.0 - smoothstep(0.004, 0.012, fp);     // black crust spots, small
    float sv = 1.0 - smoothstep(0.004, 0.01, fp);      // salt-and-pepper grain
    // (the four channels of one fetch serve different layers: ~10 fetches instead of 17)
    vec4 t0 = texture(uNoise3, q * 0.1 + vec3(0.61, 0.27, 0.83));      // r: broad weathering, g: clustering
    vec4 t1 = texture(uNoise3, q * 0.3 + vec3(0.13, 0.71, 0.37));      // r: 25 cm octave, g: exposure
    vec4 t2 = vB > 0.0 ? texture(uNoise3, q * 1.25 + 0.53) : vec4(0.475, 0.487, 0.5, 0.521);   // g: 5 cm, r: spot density, a: map lichen
    vec4 t3 = vC > 0.0 ? texture(uNoise3, q * 3.1 + 0.27) : vec4(0.475, 0.487, 0.5, 0.52);    // g: 2 cm, a: small spots
    vec4 t5 = sv > 0.0 ? texture(uNoise3, q * 7.7 + 0.61) : vec4(0.475, 0.487, 0.5, 0.52);    // g: 8 mm
    // broad weathering / exposure (0.5 - 3 m)
    float w1 = t0.r, w2 = t1.g;
    float cover = (w1 - 0.475) * 1.3 + (w2 - 0.487) * 0.7 + 0.14 * up - 0.12 * steep;
    vec3 alb = vec3(0.178, 0.172, 0.16) * (1.0 + 0.5 * (w2 - 0.487) + 0.3 * (w1 - 0.475));
    alb *= mix(vec3(1.0), vec3(1.06, 1.0, 0.9), smoothstep(0.56, 0.72, w1));      // iron-stained areas
    float gK = (t1.r - 0.475) * vK;
    float gA = vA > 0.0 ? (rkTexS(q * 0.45 + 0.11).g - 0.487) * vA : 0.0;
    float gB = (t2.g - 0.487) * vB;
    float gC = (t3.g - 0.487) * vC;
    float gD = (t5.g - 0.487) * vD;
    // smooth tonal mottling (+-~10 %)
    alb *= 1.0 + 0.5 * gK + 0.35 * gA + 0.25 * gB;
    // clustering field (40-80 cm): lichen crusts are dense in some areas, sparse in others
    float clu = (t0.g - 0.487) * 1.6 + 0.3 * cover;
    // dark lichen colonies: thresholded multi-octave noise dominated by the cm octaves (fine fractal crust,
    // no 10-30 cm camouflage blobs), ~22 % on average, 0-60 % depending on the cluster field; damp / shaded
    // sides more. The edge widens by the std of the octaves that are sub-pixel (keeps the mean)
    // (round 2 review: weight on the 14 cm octave, less on the 2-5 cm ones - crust colonies with fine ragged edges
    // instead of a 2-6 cm camouflage of colour patches; clustering x2 - whole zones crusted, others clean)
    float dn = 0.15 * gK + 0.6 * gA + 0.35 * gB + 0.35 * gC + 0.5 * gD;   // ~sd 0.14 when all octaves resolve
    float sdM = 0.153 * sqrt(0.36 * (1.0 - vA * vA) + 0.1225 * (1.0 - vB * vB) + 0.1225 * (1.0 - vC * vC) + 0.25 * (1.0 - vD * vD));
    float ew = 0.015 + 1.5 * fp + 1.6 * sdM;                         // (crisp colony edges)
    // (j2: ~17 % on an ordinary block - one dominant tone, the crust a minority pattern; ~65 % on the dark-crusted
    // blocks, gRkCrust +0.22; equal dark / pale shares read as camouflage)
    float dark = smoothstep(0.15 - ew, 0.15 + ew, dn + 0.25 * clu - 0.06 * cover + 0.03 * steep + gRkCrust);
    alb *= mix(1.0, 0.62, dark);
    // pale crustose lichen: small rosettes (2-3 cm) in the clustered colonies only, where the dark crust is not
    float vP = 1.0 - smoothstep(0.008, 0.025, fp);
    float pv = vP > 0.0 ? texture(uNoise3, q * 2.2 + 0.9).a + 0.25 * gC + 0.3 * gD : 0.521;
    float pale = rkCov(0.12 * clu + 0.12 * cover - 0.5 * gRkCrust, pv, 0.521, 0.15, 0.64, 0.006 + 1.2 * fp, vP) * (1.0 - dark);
    pale *= smoothstep(0.0, 0.25, clu);
    vec3 paleC = vec3(0.285, 0.288, 0.27) * (1.0 + 0.9 * gC);
    alb = mix(alb, paleC, pale * 0.3);
    // near-black crust spots (1-4 cm), clustered (inside colonies / dense patches)
    float dens = vB > 0.0 ? t2.r - 0.475 : 0.0;
    float inCol = mix(0.3, max(smoothstep(0.05, 0.2, dens), 0.7 * dark), vB);
    float spot = rkCov(0.9 * dens + 0.04 * steep - 0.08 * pale, t3.a, 0.52, 0.127, 0.77, 0.012 + 0.8 * fp, vS);
    float vS2 = 1.0 - smoothstep(0.01, 0.03, fp);
    float sB = vS2 > 0.0 ? texture(uNoise3, q * 1.9 + 0.33).a : 0.52;
    spot = max(spot, 0.85 * rkCov(0.9 * dens - 0.1 * pale, sB, 0.52, 0.127, 0.82, 0.02 + 0.6 * fp, vS2));
    spot *= inCol;
    // yellow-green map lichen (Rhizocarpon: patches ~#7c7b48, about as bright as the pale granite, on dark
    // and pale blocks alike): ragged 3-15 cm patches (two scales) on 15-30 % of the upper faces of the blocks that
    // carry it (mapW; gRkMapC = per-block cover), a thin black rim (prothallus) around them where resolved
    float mapl = 0.0;
    if (mapW > 0.0) {
        float vM1 = 1.0 - smoothstep(0.03, 0.08, fp);           // 12 cm octave
        float zr = vM1 > 0.0 ? (texture(uNoise3, q * 0.8 + 0.77).r - 0.475) * (0.62 / 0.14) * vM1 : 0.0;
        float mv = zr + (t2.a - 0.521) * (0.3 / 0.127) * vB + (t3.a - 0.52) * (0.2 / 0.127) * vC;   // sd ~0.72
        float sdm = sqrt(0.3844 * (1.0 - vM1 * vM1) + 0.09 * (1.0 - vB * vB) + 0.04 * (1.0 - vC * vC));
        float thr = mix(0.95, 0.6, gRkMapC) + 0.7 * (1.0 - smoothstep(-0.2, 0.6, up));      // 9-20 %, fewer on the sides
        float ewm = 0.03 + 30.0 * fp + 1.6 * sdm;
        mapl = smoothstep(thr - ewm, thr + ewm, mv) * mapW;
        float vR = 1.0 - smoothstep(0.002, 0.004, fp);                   // black rim, 2-4 mm outside the edge
        if (vR > 0.0) alb *= 1.0 - 0.6 * vR * mapW * (1.0 - smoothstep(0.0, 0.05, abs(mv - thr + 0.05)));
        alb = mix(alb, vec3(0.18, 0.185, 0.072) * (1.0 + 0.6 * gC), mapl * 0.8);
    }
    alb = mix(alb, vec3(0.055, 0.055, 0.05), spot * 0.75 * (1.0 - mapl));
    // dark water / cyanobacteria streaks hanging down the steep faces
    if (steep > 0.01) {
        float st = texture(uNoise3, vec3(q.x * 0.5, q.y * 0.05, q.z * 0.5) + 0.53).g;
        alb = mix(alb, vec3(0.07, 0.068, 0.062), smoothstep(0.6, 0.76, st) * steep * 0.45);
    }
    lich = max(pale, mapl);
    // salt-and-pepper grain: biotite specks / feldspar crystals (8-15 mm; at the final 1920 px / 12 spp
    // still resolved at 6-30 m), multiplier ~0.55-1.35
    if (sv > 0.0) {
        vec4 t6 = texture(uNoise3, q * 5.0 + 0.17);
        // (round 2: weaker - at 0.55-1.35 the whole block read as a uniform terrazzo / sponge speckle)
        float g1 = mix(1.0, 0.74, smoothstep(0.64, 0.74, t6.a)) * mix(1.0, 1.14, smoothstep(0.58, 0.68, t6.b)) * (1.0 + 0.9 * (t5.g - 0.487));
        alb *= mix(1.0, clamp(g1, 0.7, 1.2), sv * (1.0 - 0.5 * pale));
    }
    return alb * tone;
}

// --- crisp granite surface (rocks3) ------------------------------------------------------------------
// Round 3 review: the boulders read as soft camouflage blobs (thresholded smooth noise at 5-40 cm, a 0.7-1.2 grain
// multiplier of smooth noise). Real granite at 1-5 m shows crisp cell structures instead: a salt-and-pepper
// crystal mosaic (feldspar / quartz / biotite, 3-9 mm) and crustose lichen colonies built of 1-3 cm areoles, so the
// colony outlines are ragged at the areole scale and their edges are sharp. Both are 2D cell patterns in the plane
// of the dominant normal axis (the seams where that axis switches vanish in a random mosaic), filtered to their
// means by the pixel footprint.
vec2 rkPlaneUV(vec3 q, vec3 n) {
    vec3 a = abs(n);
    return a.x > a.y && a.x > a.z ? q.yz : (a.y > a.z ? q.xz : q.xy);
}
// crystal grain multiplier (mean 1): 6 mm crystals - 12 % biotite (near black), 30 % quartz (grey), 58 % feldspar;
// 2.2 cm clusters - 12 % pale K-feldspar megacrysts, 10 % dark mafic clots
float rkGrain(vec2 uv, float fp) {
    float v2 = 1.0 - smoothstep(0.008, 0.02, fp);
    float v1 = 1.0 - smoothstep(0.0025, 0.0065, fp);
    float g = 1.0;
    if (v2 > 0.0) {
        float id = voronoi2(uv * 45.0 + 11.3).z;
        g = mix(1.0, id < 0.12 ? 1.2 : (id > 0.9 ? 0.75 : 1.0), v2);      // (i4: 1.38 / 0.55 read as terrazzo)
    }
    if (v1 > 0.0) {
        float id = voronoi2(uv * 160.0 + 3.7).z;
        g *= mix(1.0, (id < 0.12 ? 0.38 : (id < 0.42 ? 0.78 : 1.17)) * (1.0 / 0.958), v1);   // (3c: biotite 0.28 -> 0.38)
    }
    return g;
}
// threshold of a ~N(0,1) field for a cover fraction c (logistic approximation of the inverse normal cdf)
float rkThr(float c) { c = clamp(c, 0.003, 0.997); return log((1.0 - c) / c) * (1.0 / 1.702); }
// lichen mosaic: per-block cover fractions cD (dark crust), cM (yellow-green map lichen), cP (pale grey-white crust).
// Colony fields ~N(0,1) at 11 / 35 cm; areoles = voronoi cells (2.4 cm) that are wholly in or out; black prothallus
// lines between the map-lichen areoles. lich = lichen cover.
// (rocks3b) crustose lichen as ROSETTES (round-lobed thalli at the sites of a jittered grid, cell s m, radius
// 0.3-0.65 cells, lobed rim from a fine noise on the site distance) instead of thresholded noise fields: merging
// rosettes build irregular colonies (pale grey-white crust in 2-15 cm rosettes, black crust
// spots, yellow-green map lichen with a black rim), and unresolved they fade to a soft expected cover - no
// thresholded 10-40 cm camouflage blobs (i3 / i6) and no uniform look (i8-i11: lichen nearly gone at 5-20 m).
// Union over the 3 x 3 neighbourhood (a thallus is never clipped by a Voronoi border). The rims are wobbly / lobed by
// a domain warp (~s / 7 lobes). The cluster density dens (0..1 thalli per cell) comes from rkRosZ (smooth ~15-45 cm
// zones); unresolved the layer fades to the Boolean-model cover 1 - exp(-0.758 dens) (0.758 = pi E[r^2]).
// Returns the coverage (0..1); rim = band inside the outline (0..1); id = hash of the covering thallus.

float gRkPaleK = 1.2, gRkKBig = 0.25;    // (rocks4 R11) pale-thallus brightness / big-rosette share (shadeRock sets them per block)

float gRkCovD = 0.15, gRkCovM = 0.0, gRkCovP = 0.15;     // per-block lichen covers (shadeRock)
// (rocks5 M1) mid-distance crisp granite speckle. Round 4 review: "stones = smooth grey plastic / clay, no texture". At 6-40 m
// (footprint 0.4-4 cm) the rosettes / crystal grain / dots / micro relief are sub-pixel and faded to their means: the blocks of
// shot 4 had a high-pass std of 11 on the rock pixels vs a target of ~25 at that scale, where granite still shows a crisp
// pixel-scale speckle (pale crust flecks, dark crust / pit clusters, light / dark mineral clots). Four octaves of value noise
// (16 / 8 / 4 / 2 cm features), each kept until its features are ~1.5 px, normalised by the std of the resolved octaves (the
// contrast stays constant over distance) and split into two-tone flecks; faded in where the near layers fade out.
// Returns an albedo multiplier (mean ~1); pk = pale-fleck weight (lichen share for the roughness).
vec3 rkMidTex(vec3 q, float fp, float steep, out float pk) {
    pk = 0.0;
    float k = smoothstep(0.003, 0.011, fp) * (1.0 - smoothstep(0.07, 0.12, fp));
    if (k <= 0.0) return vec3(1.0);
    float n = 0.0, s2 = 0.0, L = 0.16;
    for (int i = 0; i < 4; i++) {
        float v = 1.0 - smoothstep(0.45, 0.75, fp / L);
        if (v <= 0.0) break;
        float wo = v * (i == 0 ? 0.5 : (i == 1 ? 0.8 : 1.0));     // (b2: coarse octaves weaker - round 8-16 cm leopard spots)
        n += wo * (texture(uNoise3, q * (0.0625 / L) + vec3(0.17, 0.53, 0.29) * float(i + 1)).g - 0.487);
        s2 += wo * wo;
        L *= 0.5;
    }
    n /= max(sqrt(s2) * 0.153, 1e-3);                      // ~N(0,1)
    float kA = 1.3;
    float pale = smoothstep(0.8, 1.3, n + 0.15 * steep) * (1.0 - 0.4 * steep);   // ~15 % pale crust / feldspar flecks
    float dark = 1.0 - smoothstep(-1.5, -1.0, n);           // ~11 % dark crust / pit clusters
    vec3 m = vec3(1.0 + 0.1 * clamp(n, -2.0, 2.0));
    m = mix(m, vec3(1.4, 1.42, 1.36), pale);
    m = mix(m, vec3(0.5) * vec3(1.0, 1.0, 0.96), dark);
    pk = pale * k;
    // (rocks5 L2) yellow-green map-lichen flecks on the blocks that carry it (gRkCovM, set by shadeRock / shadeHeroRock)
    if (gRkCovM > 0.0) {
        float n2 = (texture(uNoise3, q * 0.78 + 0.31).a - 0.521) / 0.127;
        float mf = smoothstep(-0.3, 0.3, n2 - rkThr(min(gRkCovM * 1.0, 0.6)));
        m = mix(m, vec3(1.05, 1.12, 0.38), mf * (1.0 - 0.7 * steep));
    }
    return mix(vec3(1.0), m, k * kA);
}
// boulder surface: broad weathering tone, crystal grain, black lichen dots, lichen mosaic, water / algae streaks

// --- shore cobbles (used by pebbles() in terrain_shade.glsl) -----------------------
// cobble albedo by stone id: granite of every weathering tone (pale to grey), 15 % dark diorite / schist /
// lichen-blackened stones, a few rusty, a few quartz
vec3 rkCobCol(float id) {
    float v = fract(id * 7.13), k = fract(id * 13.7);
    // (rocks3d) narrower spread 0.22-0.6 -> 0.3-0.52: the palest stones clipped to white beads on the r1 beach; the eye tells the
    // chips apart by their shadows more than by their tones
    // (rocks4 R5) wider again (0.3-0.52 -> 0.22-0.62): the shingle mixes pale, mid and dark chips; 6 % white quartz (0.7),
    // 10 % near-black (0.10-0.19)
    vec3 c = vec3(mix(0.22, 0.62, v)) * vec3(1.0, 0.91, 0.83);      // (rocks3d: warmer, r1 beach b/r 0.89 / g/r 0.95 vs target 0.86 / 0.92)      // (shingle: pale pinkish to mid grey; rocks3b / 3c warmer, #b9ab9f b/r 0.86)
    // (rocks5 P1) per-stone hue families (round 4: "beach too grey"; shingle = warm grey, beige, pinkish, brown, dark
    // grey-green, white quartz, a few reddish stones, each its own colour) instead of +-4 % warm / cool on one grey
    float hf = fract(id * 31.7);
    float kH = 1.0;
    c *= mix(vec3(1.0), hf < 0.22 ? vec3(1.07, 0.97, 0.82) : (hf < 0.4 ? vec3(1.06, 0.94, 0.9) : (hf < 0.52 ? vec3(0.94, 0.98, 1.06)
       : (hf < 0.6 ? vec3(0.97, 1.0, 0.93) : vec3(1.0)))), kH);
    if (k > 0.9) c = vec3(mix(0.1, 0.19, v)) * mix(vec3(0.97, 0.98, 1.01), vec3(0.95, 1.0, 0.95), step(0.5, hf));
    else if (k < 0.04) c = mix(vec3(0.4, 0.33, 0.25), vec3(0.42, 0.27, 0.2), step(0.5, hf)) * mix(0.75, 1.1, v);   // brown / reddish
    else if (k < 0.1) c = vec3(0.7, 0.69, 0.66);
    return c;
}
// (rocks5 P2) unresolved gravel: crisp pixel-scale stone mosaic (8 / 4 / 2 cm octaves of value + cellular noise, each until
// ~1.5 px, normalised, two-tone), multiplier mean ~1. Beyond the resolved pebble layers the beach read as smooth pale sand.
vec3 rkGravelTex(vec3 q, float fp) {
    float n = 0.0, s2 = 0.0, L = 0.08;
    for (int i = 0; i < 3; i++) {
        float v = 1.0 - smoothstep(0.45, 0.75, fp / L);
        if (v <= 0.0) break;
        vec4 tx = texture(uNoise3, q * (0.0625 / L) + vec3(0.41, 0.13, 0.77) * float(i + 1));
        n += v * ((tx.g - 0.487) + 0.8 * (tx.b - 0.5255));
        s2 += v * v;
        L *= 0.5;
    }
    if (s2 <= 0.0) return vec3(1.0);
    n /= sqrt(s2) * 0.19;
    float k = 1.5 * sqrt(s2 / (1.0 + s2));
    vec3 m = vec3(1.0 + 0.18 * clamp(n, -2.0, 2.0));
    m = mix(m, vec3(1.35, 1.3, 1.24), smoothstep(0.9, 1.3, n));
    m = mix(m, vec3(0.5, 0.48, 0.46), 1.0 - smoothstep(-1.4, -1.0, n));
    return mix(vec3(1.0), m, k);
}
// (rocks3c) lake-bed stones (pebbles() under water): grey granite of every tone (darker on average than
// the dry shingle), ~25 % olive / golden-brown (algae / iron film), ~12 % dark stones; the bed tint (subT) comes on top
vec3 rkBedCol(float id) {
    float v = fract(id * 7.13), k = fract(id * 13.7);
    // (rocks4 R8) darker / browner and narrower (0.16-0.30, 15 % near-black): the bed read as cream-olive lentils (#727468)
    vec3 c = vec3(mix(0.16, 0.30, v)) * vec3(1.0, 0.96, 0.88);
    if (k < 0.25) c = mix(vec3(0.17, 0.155, 0.105), vec3(0.32, 0.27, 0.17), v);
    else if (k > 0.85) c = vec3(mix(0.07, 0.12, v)) * vec3(0.95, 0.97, 1.0);   // (rocks4 R8: 0.04-0.08 read as holes in the r10 bed)
    return c;
}
// packed angular pebbles: power diagram (weighted Voronoi) of a jittered grid with rounded corners
// (p in cell units). Random site weights (0..wv, cell units^2) give a broad mix of pebble sizes instead of a
// uniform paving. Returns x = distance to the rounded cell border, y = distance to the site, z = id hash;
// toS = p -> that site. kr = corner rounding (cell units).
vec3 rkPack(vec2 p, float kr, float wv, float el, out vec2 toS) {
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i);
    vec3 rs[9];
    float best = 9.0, bid = 0.0, mw = 0.0;
    vec2 mr = vec2(0.0);
    for (int k = 0; k < 9; k++) {
        ivec2 o = ivec2(k % 3 - 1, k / 3 - 1);
        uint h = hash2u(c + o + ivec2(7177, -3301));
        vec3 hv = vec3(uvec3(h, h >> 10u, h >> 20u) & 1023u) * (1.0 / 1023.0);
        vec2 r = vec2(o) + 0.08 + 0.84 * hv.xy - f;
        float w = wv * fract(hv.z * 5.31 + hv.x * 1.73);
        rs[k] = vec3(r, w);
        float d = dot(r, r) - w;
        if (d < best) { best = d; mr = r; mw = w; bid = hv.z; }
    }
    // two nearest power bisectors; polynomial smooth-min of them rounds the corners; dS = the site's distance to
    // its nearest bisector (inradius of its cell)
    float b1 = 9.0, b2 = 9.0, dS = 9.0;
    for (int k = 0; k < 9; k++) {
        vec2 dv = rs[k].xy - mr;
        float l2 = dot(dv, dv);
        if (l2 < 1e-8) continue;
        float il = inversesqrt(l2);
        float b = (dot(0.5 * (mr + rs[k].xy), dv) - 0.5 * (rs[k].z - mw)) * il;
        b2 = min(b2, max(b, b1));
        b1 = min(b1, b);
        dS = min(dS, 0.5 * l2 * il - 0.5 * (rs[k].z - mw) * il);
    }
    float hk = max(kr - (b2 - b1), 0.0) / kr;
    float bd = b1 - hk * hk * kr * 0.25;
    // (round 2 review) each pebble its own outline: an ellipse at a random angle (semi-axes 1.5-2.1 x the cell's
    // inradius, aspect 0.65-0.95) cut by its cell - rounded ends, straight sides only where a neighbour is close,
    // gaps of varying width (no shared-edge paving); el blends toward the plain cell (packing density)
    if (el < 0.0) el = -el * mix(0.3, 1.5, fract(bid * 71.9));   // (rocks4 R8) el < 0: a random blend per stone
    if (el > 0.0) {
        vec2 ax = normalize(fract(bid * vec2(37.13, 59.71)) - 0.5 + 1e-4);
        vec2 l = vec2(dot(-mr, ax), dot(-mr, vec2(-ax.y, ax.x)));
        float A = max(dS, 0.05) * mix(1.5, 2.1, fract(bid * 13.77));
        vec2 ab = A * vec2(1.0, mix(0.65, 0.95, fract(bid * 23.31)));
        float k0 = length(l / ab), k1 = length(l / (ab * ab));
        bd = mix(bd, min(bd, -k0 * (k0 - 1.0) / max(k1, 1e-5)), el);
    }
    toS = mr;
    return vec3(bd, length(mr), bid);
}
// shade a pebble: (coverage, dome height, self-shading); bmp = outward bump slope (xz); gapS = shading of the ground
// around it. Packed layers (sparse = 0): occlusion only at the bottom of the crevice between two pebbles,
// plus the pebble's short cast shadow on its anti-sun side (the gaps on the sun side stay lit: no dark
// mosaic network). Sparse stones (sparse = 1): contact ring at the stone's foot + cast shadow.
vec3 rkPackShade(vec3 v, vec2 toS, float g, float flt, float fpc, vec2 sunXZ, float sunT, float sparse, out vec2 bmp, out float gapS) {
    float aa = fpc * 0.7 + 0.015;
    float inside = smoothstep(g - aa, g + aa, v.x);
    float rad = length(toS);
    float u = saturate(rad / max(rad + v.x - g, 1e-3));
    // (rocks3d) packed gravel: ~70 % angular chips whose profile follows the polygon outline (flat top, bevel ~0.2 cell
    // wide along the edge) instead of the radial dome - the concentric dome shading drew every chip as an oval bean
    // (r1 / c_box: "white beans in grout"; target: flat angular chips)
    if (sparse < 0.5 && fract(v.z * 7.31) > 0.45) u = 1.0 - saturate((v.x - g) / 0.2);   // (k9: 70 -> 55 %, all-angular = terrazzo at 1 m)
    float u2 = u * u;
    float h = mix(sqrt(saturate(1.0 - u2)), 1.0 - u2 * u2, flt);      // round or flat-topped
    vec2 outw = -toS / max(rad, 1e-4);
    float sl = mix(u / max(sqrt(saturate(1.0 - u2)), 0.3), 4.0 * u2 * u, flt);
    // each pebble lies a little tilted (catches the light on its own: reads as loose 3D gravel)
    vec2 tilt = (fract(v.z * vec2(17.3, 29.1)) - 0.5) * (sparse > 0.5 ? 0.7 : 1.0);   // (rocks3c: chips 0.7 -> 1.0, own facet tone)
    bmp = (outw * min(sl, 2.0) + tilt) * inside;
    float eg = max(g - v.x, 0.0);                                       // distance from the pebble's edge
    // contact ring at the foot (sparse) / the narrow crevice along the pebble's edge (packed: the wider gaps of the
    // elliptical pebbles show the next layer, lit - j1: whole gaps dark read as white eggs on black)
    float crev = 1.0 - smoothstep(0.0, (sparse > 0.5 ? 0.06 : 0.05) + aa, eg);
    float L = (rad + v.x - g) * mix(0.45, 0.25, flt) * sunT * 2.0;     // pebble height x sun slope (rocks4 R5: x2, tiny cast shadows)
    float sh = smoothstep(0.1, -0.4, dot(outw, sunXZ)) * (1.0 - smoothstep(L, L + aa, eg));
    // (dark crevices / shadows; rocks3: packed gravel 0.55 / 0.3 - black gaps read as white eggs on black)
    // (rocks4 R5: packed 0.72 / 0.42 -> 0.62 / 0.32 - target shingle hp std 35 vs 22: darker crevices and chip shadows)
    gapS = mix(1.0, sparse > 0.5 ? 0.45 : 0.62, crev) * mix(1.0, sparse > 0.5 ? 0.15 : 0.32, sh);
    // self-shading of the rounded flank turned away from the sun (sub-pixel bump shading is lost, and the
    // lambert term of a 2-5 cm dome is too weak to read as a 3D stone): darker rim on the shadow side,
    // a little lift on the sun side; filtered away when the pebble is ~2 px
    float vS = 1.0 - smoothstep(0.15, 0.4, fpc);
    float sd = dot(outw, sunXZ);
    // (j2: a rim-only darkening read as white eggs - rounded stones shade across their whole flank, flat chips
    // only along the bevel)
    float rim = mix(u * u, smoothstep(0.6, 1.0, u), flt);
    // (rocks3b: packed chips 0.25 + crevice 0.84 / 5 % of a cell - the dark crescent and outline on every pebble drew the
    // egg paving of r1 / c_box; the eye tells the stones apart by tone, not by outlines)
    float kL = sparse > 0.5 ? 1.0 : 1.3;
    // (rocks3d) packed: sun-side lift 0.25 -> 0.1 (the lit domes clipped to white beads)
    float lit = 1.0 + vS * rim * kL * ((sparse > 0.5 ? 0.25 : 0.1) * saturate(sd) - (sparse > 0.5 ? 0.62 : 0.25) * saturate(0.2 - sd) * min(0.6 + sunT, 1.4));
    // (rocks4 R5) packed angular chips: two facets meeting at a ridge through the chip (random offset / tilt), the one facing the
    // sun lit x1.35, the other x0.55 - every chip catches the light on its own, not a flat stamped disc
    if (sparse < 0.5) {
        float fo = (fract(v.z * 41.3) - 0.5) * 0.9;
        float fs = smoothstep(-0.12, 0.12, rad * sd / max(rad + v.x - g, 1e-3) - 0.5 * fo + 0.3 * dot(tilt, sunXZ));
        lit *= mix(1.0, mix(0.55, 1.35, fs), 0.85 * vS);
    }
    return vec3(inside, h, lit);
}

// --- cliffs / outcrops: jointed granite -------------------------------------------
// one joint set: footprint-filtered joint-line coverage (0..1); id = slab index.
// Spacing is irregular (monotonic warp) and joints are only open in patches.
float rkJoint(vec3 p, vec3 a, float S, float wid, float fp, float seed, out float id) {
    float u = dot(p, a) / S;
    float aa = fp * 1.2 + 0.02;
    if (fp > 0.35 * S) { id = floor(u); return min(2.0 * wid / S, 1.0) * 0.5; }   // sub-pixel: mean only
    u += 0.3 * gnoise(vec2(u * 0.71, seed));
    id = floor(u);
    float dj = (0.5 - abs(fract(u) - 0.5)) * S;      // ~distance to the nearest joint (m)
    float open = smoothstep(0.38, 0.62, texture(uNoise3, p * (0.4 / S) + seed * 0.13).g);
    float w = wid * (0.4 + 1.2 * open);
    float line = (1.0 - smoothstep(w, w + aa, dj)) * open * saturate((2.0 * w + 0.02) / (2.0 * w + aa));
    float vis = 1.0 - smoothstep(0.12 * S, 0.35 * S, fp);
    float cov = min(2.0 * w / S, 1.0) * 0.5;
    return mix(cov, line, vis);
}

// --- lake-shore outcrops: glacier-polished granite slabs ------------
// terrain rock within a few metres of a lake / tarn level reads as smooth pale whaleback slabs with
// sheeting joints, sparse lichen and a darker stained band above the waterline (no cliff joint blocks)
bool gRkOnBlock = false;    // rockAlbedo called for an outcrop block (shadeRock): no gravel cover, no vegetation skip
vec3 gRkPebP = vec3(1e9);   // gravel cover of the shore outcrop: bump of the last rkSlabAlbedo point (for rockNormal)
vec3 gRkPebB = vec3(0.0);
vec3 pebbles(vec2 p, out vec3 bump);
float rkSlabWeight(vec3 p) {
    float wl = waterLevel(p.xz);
    if (wl < -1000.0) return 0.0;
    float above = p.y - wl;
    return 1.0 - smoothstep(7.0, 16.0, above + 6.0 * gnoise(p.xz * 0.017 + 5.0));
}
vec3 rkSlabAlbedo(vec3 p, vec3 n, float t, float fp) {
    float wl = waterLevel(p.xz);
    float above = p.y - wl;
    // pale, slightly warm granite with broad tone variation
    float w1 = texture(uNoise3, p * 0.035 + 0.2).r, w2 = texture(uNoise3, p * 0.13 + 0.6).r;
    vec3 alb = vec3(0.285, 0.274, 0.252) * mix(0.84, 1.1, saturate((w1 - 0.475) * 2.2 + (w2 - 0.475) * 1.2 + 0.55));
    // grey-brown lichen / weathering stains (0.5 - 3 m)
    alb *= mix(vec3(1.0), vec3(0.8, 0.78, 0.74), smoothstep(0.5, 0.68, texture(uNoise3, p * 0.05 + 0.2).r));
    // the boulders' granite surface at every distance (lichen crust, black spots, grain; it fades to its
    // mean by itself), a little cleaner and paler than on the boulders: crisp texture, not smooth plaster
    float lich;
    vec3 d = graniteDetail(p * 0.9, n.y - 0.3, fp, 1.0, 0.2, 0.0, lich);
    alb *= mix(vec3(1.0), d / GRANITE_BASE, 0.85);
    // (round 2 review: the noise iso-line cracks read as dried-mud / marble veins - the straight joint cracks of
    // rkNearRock, applied through rkNearAlbedo, carry the fracture network)
    // lake-level stain band (dark grey / brownish) and black streaks running down to the water
    float band = 1.0 - smoothstep(0.3, 1.8 + 0.6 * gnoise(p.xz * 0.2), above);
    alb *= mix(1.0, 0.62, band);
    float st = gnoise(vec2(dot(p.xz, vec2(0.7, 0.7)) * 0.9, p.y * 0.06));
    alb *= 1.0 - 0.35 * smoothstep(0.25, 0.5, st) * (1.0 - smoothstep(0.5, 4.0, above));
    // dark weathering pits (1-4 cm, ragged, in patches only - no even polka-dot pattern), ~3 %
    float vp = 1.0 - smoothstep(0.008, 0.025, fp);
    float pA = vp > 0.0 ? texture(uNoise3, p * 2.4 + 0.9).a + 0.25 * (texture(uNoise3, p * 6.0 + 0.3).g - 0.487) : 0.521;
    float pm = smoothstep(0.45, 0.65, texture(uNoise3, p * 0.11 + 0.47).g);
    alb *= 1.0 - 0.35 * pm * rkCov(0.0, pA, 0.521, 0.14, 0.74, 0.01 + 1.5 * fp, vp);
    // weathering streaks running down the slab (varying width / strength), everywhere on the outcrop
    vec2 dsl = normalize(n.xz + vec2(1e-4));
    float across = dot(p.xz, vec2(-dsl.y, dsl.x));
    float sv = texture(uNoise3, vec3(across * 0.3, p.y * 0.035, 0.61)).g + 0.25 * (texture(uNoise3, vec3(across * 1.1, p.y * 0.1, 0.13)).g - 0.487);
    alb *= 1.0 - 0.25 * smoothstep(0.56, 0.74, sv) * (1.0 - smoothstep(0.5, 2.0, fp));
    // gravel cover: gentle parts of the low shore outcrop (< ~3 m above the water) are buried under the
    // beach's cobbles in patches (bare whaleback slab next to packed gravel); terrain only (not heroes)
    // (the wave-washed zone < ~2 m above the water is gravel on steeper slopes too)
    float gw = max(smoothstep(0.72, 0.86, n.y) * (1.0 - smoothstep(2.5, 4.5, above + 1.2 * gnoise(p.xz * 0.35 + 2.0))),
                   smoothstep(0.55, 0.7, n.y) * (1.0 - smoothstep(1.2, 2.4, above + 0.6 * gnoise(p.xz * 0.5 + 7.0))))
             * (1.0 - smoothstep(25.0, 40.0, t)) * smoothstep(0.3, 0.45, texture(uNoise3, vec3(p.xz * 0.021, 0.43)).g);
    // (rocks3c) no shingle under the forest canopy (shot 8: a grey pebble mosaic on the shaded forest bank = crazy paving)
    if (gw > 0.0) gw *= 1.0 - saturate(forestAt(p.xz).r * 2.0);
    if (gw > 0.0 && !gRkOnBlock && abs(p.y - terrainH(p.xz, true)) < 0.4) {
        vec3 bb;
        vec3 pc = pebbles(p.xz, bb);
        alb = mix(alb, pc, gw);
        gRkPebP = p; gRkPebB = bb * gw;
    }
    return alb;
}

vec3 gRkCacheP = vec3(1e9);
vec4 gRkJ;      // x: small-block hash, y: big-panel hash, z: joint line darkness, w: small-block visibility
vec3 gRkTilt;   // per-block normal tilt

void rkCliffJoints(vec3 p, float fp) {
    if (p == gRkCacheP) return;
    gRkCacheP = p;
    // gentle domain warp so joints are not ruler-straight; joint strike drifts across the massif
    vec3 w = p + vec3(9.0 * gnoise(p.zy * 0.011 + 3.1), 0.0, 9.0 * gnoise(p.xy * 0.011 + 7.7));
    w.y += 5.0 * gnoise(p.xz * 0.013 + 1.3);
    w.xz = rot2(0.6 * gnoise(p.xz * 0.0006 + 2.0)) * w.xz;
    float iA, iB, iC, IA, IB;
    float jA = rkJoint(w, normalize(vec3(0.82, 0.05, 0.57)), 8.5, 0.10, fp, 1.0, iA);
    float jB = rkJoint(w, normalize(vec3(-0.55, 0.12, 0.83)), 12.0, 0.12, fp, 2.0, iB);
    float jC = rkJoint(w, normalize(vec3(0.25, 0.92, -0.3)), 6.5, 0.08, fp, 3.0, iC);
    float JA = rkJoint(w, normalize(vec3(0.80, 0.0, 0.60)), 47.0, 0.45, fp, 4.0, IA);
    float JB = rkJoint(w, normalize(vec3(-0.6, 0.25, 0.76)), 71.0, 0.6, fp, 5.0, IB);
    float hs = hash13i(ivec3(int(iA), int(iB), int(iC)));
    float hb = hash13i(ivec3(int(IA) + 911, int(IB) - 377, 5));
    float line = max(max(jA, jB), jC * 0.8);
    float bigLine = max(JA, JB);
    gRkJ = vec4(hs, hb, saturate(line * 0.8 + bigLine * 0.9), 1.0 - smoothstep(1.0, 3.0, fp));
    // block facet tilt (small blocks) + panel tilt (big)
    vec3 ts = vec3(hash13i(ivec3(int(iA), int(iB), int(iC) + 31)), hs, hash13i(ivec3(int(iA) + 7, int(iB), int(iC)))) - 0.5;
    vec3 tb = vec3(hb, hash13i(ivec3(int(IA), int(IB), 77)), hash13i(ivec3(int(IA), int(IB), 78))) - 0.5;
    float vs = 1.0 - smoothstep(0.9, 2.6, fp);
    float vb = 1.0 - smoothstep(6.0, 16.0, fp);
    gRkTilt = ts * 0.5 * vs + tb * 0.45 * vb;
    gRkTilt *= mix(0.15, 1.0, smoothstep(40.0, 220.0, length(p - uCamPos)));
}

// --- terrain granite up close (1-300 m): jointed outcrop ------------------------------------------
// The near heightfield is smooth (1 m map + soft micro relief): without structure the faces read as white
// plaster. Real outcrops are cut by joint sets into blocks: dark open cracks, small
// ledges / steps along the joints (a lit or shaded riser beside the crack), every block a slightly
// different plane and tone, fresh pale scars where blocks fell out, dark water streaks hanging below the
// sheeting joints, moss / grass in the cracks on the ledges, lichen colonies in patches.
// Computed once per shading point (cached) and used by rockAlbedo (albedo) and rockNormal (normal offset).
vec3 gRkNP = vec3(1e9);
vec3 gRkNG = vec3(0.0);      // normal offset (world)
vec4 gRkNA = vec4(0.0);      // x crack darkness, y moss / soil in the cracks, z water streak, w fresh scar
float gRkNT = 1.0;           // block tone
float gRkNW = 0.0;           // weight of the near model (0 far away)
float gRkNH = 0.0;           // weathering halo along the open joints
vec3 gRkMG = vec3(0.0);      // close range: crystalline / pitted micro relief, normal offset (world)
float gRkMP = 0.0;           // close range: micro pit depth (0..1, cavity darkening)
float rkMicroH(vec3 x, float a1, float a2, float a4);

// one joint set: x crack coverage (footprint filtered), y signed distance to the nearest joint (m, + above),
// z hash of the nearest joint, w block index along the set
vec4 rkJointN(vec3 w, vec3 a, float S, float wid, float fp, float seed, out float halo) {
    float u = dot(w, a) / S;
    float bid = floor(u);
    float fr = u - bid;
    float s = (fr < 0.5 ? fr : fr - 1.0) * S;
    float jh = hash11(bid + (fr < 0.5 ? 0.0 : 1.0) + seed * 17.31);
    // joints are open along part of their length only (they end, re-open, en-echelon)
    float open = smoothstep(0.42, 0.56, texture(uNoise3, w * (0.3 / S) + seed * 0.137).g + 0.18 * (jh - 0.5));
    float wd = wid * (0.35 + 1.3 * open) * mix(0.6, 1.5, jh);
    float aa = 0.9 * fp + 0.003;
    float line = (1.0 - smoothstep(wd, wd + aa, abs(s))) * open * saturate((2.0 * wd + 0.003) / (2.0 * wd + aa));
    float vis = 1.0 - smoothstep(0.06 * S, 0.25 * S, fp);
    line = mix(open * min(2.4 * wd / S, 1.0), line, vis);
    // weathering halo along the open joints (dirt, lichen, damp): a darker band 10-30 cm wide
    float hw = mix(0.08, 0.3, jh);
    halo = mix(open * min(2.0 * hw / S, 1.0) * 0.5, (1.0 - smoothstep(0.0, hw + aa, abs(s))) * open, vis);
    return vec4(line, s, jh, bid);
}

void rkNearRock(vec3 p, vec3 n, float t, float fp) {
    if (p == gRkNP) return;
    gRkNP = p;
    gRkNG = vec3(0.0); gRkNA = vec4(0.0); gRkNT = 1.0; gRkNH = 0.0; gRkMG = vec3(0.0); gRkMP = 0.0;
    // close range (footprint < ~2 cm): the boulders' crumbly pitted micro relief (worley pits 9 / 3 cm) and angular
    // crystal facets (voronoi in the face plane) - j1: within ~10 m the smooth heightfield read as stucco / plaster
    float vm = 1.0 - smoothstep(0.008, 0.025, fp);
    if (vm > 0.0) {
        float a1 = 0.018 * vm, a2 = 0.006 * (1.0 - smoothstep(0.003, 0.009, fp));
        const float em = 0.004;
        float h1 = rkMicroH(p + vec3(em, -em, -em), a1, a2, 0.0), h2 = rkMicroH(p + vec3(-em, -em, em), a1, a2, 0.0);
        float h3 = rkMicroH(p + vec3(-em, em, -em), a1, a2, 0.0), h4 = rkMicroH(p + vec3(em, em, em), a1, a2, 0.0);
        vec3 gm = (vec3(1.0, -1.0, -1.0) * h1 + vec3(-1.0, -1.0, 1.0) * h2 + vec3(-1.0, 1.0, -1.0) * h3 + vec3(h4)) / (4.0 * em);
        gRkMP = saturate(-0.25 * (h1 + h2 + h3 + h4) / max(a1 + a2, 1e-4) * 1.6 - 0.35) * vm;
        const vec3 FA = vec3(0.8513, 0.1277, 0.5087), FB = vec3(0.1213, 0.4852, 0.866);
        vec3 ta = normalize(cross(n, abs(dot(n, FA)) < 0.85 ? FA : FB));
        vec3 tb = cross(n, ta);
        vec3 v1 = voronoi2(vec2(dot(p, ta), dot(p, tb)) * 9.0 + 3.1);
        vec2 tf = (fract(v1.z * vec2(13.7, 29.3)) - 0.5) * 0.4 * (1.0 - smoothstep(0.012, 0.025, fp));
        gm += ta * tf.x + tb * tf.y;
        gRkMG = (gm - n * dot(gm, n)) * 0.8;
    }
    // joint blocks of 2-5 m: faded by the footprint only (gone once the blocks are ~10 px). rkNearAlbedo keeps the
    // mean tone (divides by the expected darkening), so there is no ring / bubble where it fades
    gRkNW = 1.0 - smoothstep(0.12, 0.3, fp);
    if (gRkNW <= 0.0) return;
    // wavy joints: two-scale warp (1 m at ~15 m, 0.25 m at ~4 m)
    vec3 w = p + (texture(uNoise3, p * 0.019 + 0.31).rgb - 0.48) * 2.2;
    if (fp < 0.12) w += (texture(uNoise3, p * 0.083 + 0.77).rgb - 0.48) * 0.55 * (1.0 - smoothstep(0.05, 0.12, fp));
    // joint orientations drift slowly across the massif
    float r0 = 0.7 * gnoise(p.xz * 0.0011 + 4.0);
    vec2 sa = rot2(r0) * vec2(0.83, 0.56);
    vec3 aA = normalize(vec3(sa.x, 0.08, sa.y));
    vec3 aB = normalize(vec3(-sa.y * 0.9 + sa.x * 0.3, 0.15, sa.x * 0.9 + sa.y * 0.3));
    vec3 aC = normalize(vec3(0.18 * sa.y, 1.0, -0.22 * sa.x));          // sheeting joints, nearly flat
    float hA, hB, hC, hD = 0.0;
    vec4 jA = rkJointN(w, aA, 3.3, 0.013, fp, 1.0, hA);
    vec4 jB = rkJointN(w, aB, 5.1, 0.016, fp, 2.0, hB);
    vec4 jC = rkJointN(w, aC, 2.1, 0.011, fp, 3.0, hC);
    // second order: short cross fractures inside the blocks (~1 m)
    vec4 jD = fp < 0.02 ? rkJointN(w + 0.37, normalize(aA + aB * 0.6 + vec3(0.0, 0.3, 0.0)), 1.15, 0.006, fp, 4.0, hD) : vec4(0.0);
    if (fp < 0.02) jD.x *= smoothstep(0.45, 0.6, texture(uNoise3, p * 0.11 + 0.5).r) * (1.0 - smoothstep(0.012, 0.02, fp));   // only in shattered patches
    float crack = max(max(jA.x, jB.x), max(jC.x * 0.9, jD.x * 0.8));
    gRkNH = max(max(hA, hB), hC * 0.8);
    // block: tone, plane tilt; fresh pale scars (a block fell out recently, ~6 %)
    ivec3 bi = ivec3(int(jA.w), int(jB.w), int(jC.w));
    vec4 hb = vec4(hash13i(bi), hash13i(bi + ivec3(71, 13, 5)), hash13i(bi + ivec3(-9, 47, 23)), hash13i(bi + ivec3(5, -3, 91)));
    float vb = 1.0 - smoothstep(0.12, 0.6, fp);
    gRkNT = mix(1.0, mix(0.84, 1.14, hb.x), vb);
    gRkNA.w = step(hb.y, 0.06) * vb;
    vec3 tilt = (hb.yzw - 0.5) * 0.5;
    // ledge risers: the block on one side of a joint stands proud by a few cm-dm; the riser beside the crack
    // faces the lower block (a lit or a shaded band, depending on the sun)
    vec3 g = (tilt - n * dot(tilt, n)) * vb;
    float vR0 = 1.0 - smoothstep(0.06, 0.1, fp);                  // risers <= ~1-5 px: faded out, loop skipped (cost)
    for (int i = 0; i < 3; i++) {
        if (vR0 <= 0.0) break;
        vec4 j = i == 0 ? jA : (i == 1 ? jB : jC);
        vec3 a = i == 0 ? aA : (i == 1 ? aB : aC);
        float S = i == 0 ? 3.3 : (i == 1 ? 5.1 : 2.1);
        float rb = mix(0.08, 0.3, fract(j.z * 7.3)) * S * 0.35;    // riser width
        float vr = (1.0 - smoothstep(0.4 * rb, 1.2 * rb, fp)) * vR0;
        if (vr <= 0.0) continue;
        float side = j.z < 0.5 ? 1.0 : -1.0;                       // which side stands proud
        float ss = j.y * side;
        float band = step(0.0, ss) * (1.0 - smoothstep(0.3 * rb, rb, ss));
        vec3 aT = a - n * dot(a, n);
        float la = length(aT);
        if (la < 0.15) continue;                                   // joint parallel to the surface: no step
        g -= (aT / la) * side * band * 1.4 * vr * smoothstep(0.42, 0.6, texture(uNoise3, w * (0.3 / S) + float(i + 1) * 0.137).g + 0.18 * (j.z - 0.5));
    }
    gRkNG = g * gRkNW;
    // water streaks (cyanobacteria / lichen tracks) hanging down the steep faces: stripes 0.15-0.5 m wide and
    // 3-10 m long, clustered in damp zones, starting at some of the sheeting joints
    float steep = 1.0 - smoothstep(0.45, 0.8, n.y);
    if (steep > 0.0) {
        vec2 hz = normalize(vec2(-n.z, n.x) + 1e-4);
        float acr = dot(p.xz, hz) + 0.3 * (w.x - p.x);
        float vs = 1.0 - smoothstep(0.06, 0.2, fp);
        float sA = vs > 0.0 ? texture(uNoise3, vec3(acr * 0.33, p.y * 0.011, 0.37)).g - 0.487 : 0.0;
        float sB = texture(uNoise3, vec3(acr * 0.11, p.y * 0.0055, 0.71)).g - 0.487;
        float wz = smoothstep(0.42, 0.62, texture(uNoise3, p * 0.017 + 0.23).r);
        float stp = 0.55 * sA * vs + 0.45 * sB + 0.1 * wz - 0.06 + 0.04 * smoothstep(0.0, 0.3, -jC.y);
        float ew = 0.012 + 0.06 * (1.0 - vs);
        gRkNA.z = smoothstep(0.075 - ew, 0.075 + ew, stp) * steep * mix(0.5, 1.0, wz);
    }
    // soil / moss in open cracks where they hold it (gentle faces, sheeting-joint ledges)
    gRkNA.y = crack * smoothstep(0.35, 0.7, n.y) * smoothstep(0.4, 0.6, texture(uNoise3, p * 0.21 + 0.13).b);
    gRkNA.x = crack * gRkNW;
}

// near-model albedo: block tones, lichen colonies (0.3-2 m), fresh scars, cracks, moss in cracks, streaks
// (k: strength, 1 cliffs, less on the polished lake-shore slabs)
vec3 rkNearAlbedo(vec3 alb, vec3 p, vec3 n, float fp, float k) {
    float W = gRkNW * k;
    if (W <= 0.0) return alb;
    float steep = 1.0 - smoothstep(0.5, 0.85, n.y);
    alb *= mix(1.0, gRkNT, W);
    // soft weathering mottling 0.5-3 m, drawn out down the fall line on steep faces (no crisp blotches:
    // those read as camouflage); the fine lichen crust itself is graniteDetail's
    vec3 pe = vec3(p.x, p.y * mix(1.0, 0.4, steep), p.z);
    float vm = 1.0 - smoothstep(0.1, 0.3, fp);
    float m = 0.6 * (texture(uNoise3, pe * 0.034 + 0.41).g - 0.487) + 0.4 * (texture(uNoise3, pe * 0.11 + 0.17).g - 0.487) * vm;
    float dz = smoothstep(0.45, 0.7, texture(uNoise3, pe * 0.05 + 0.73).r + 0.1 * steep);   // lichen-darkened zones
    float fresh = gRkNA.w * mix(0.5, 1.0, steep);                                          // fresh scars
    alb *= (1.0 + 1.1 * m * W) * (1.0 - 0.3 * dz * W * (1.0 - fresh));
    alb *= mix(vec3(1.0), vec3(1.3, 1.27, 1.2), fresh * W);
    // open cracks (dark) with their weathering halo, soil / moss in some, water streaks
    alb *= (1.0 - 0.85 * gRkNA.x * k) * (1.0 - 0.22 * gRkNH * W);
    alb = mix(alb, vec3(0.07, 0.085, 0.03) * (0.8 + 0.5 * (m + 0.5)), gRkNA.y * 0.7 * k);
    alb *= 1.0 - 0.55 * gRkNA.z * W * (1.0 - fresh);
    // (round 2 review) keep the mean tone: divide by the expected darkening of the terms above (lichen zones
    // ~0.925 / 0.85 steep, cracks 0.98, halo 0.98, streaks 0.9 steep, scars 1.01; all footprint-filtered to their
    // coverage, so the mean holds at every distance) -> no ring of darker rock where the model fades out
    return alb / (1.0 - W * mix(0.105, 0.26, steep));
}

// Terrain points whose rock albedo / normal terrainAlbedo (terrain_shade.glsl) cannot use: within 60 m its macro
// masks are off (far = 0) and the vegetation weight is vegOld * (1 - 0.9 f.b), exactly 1 where
// ny + 0.1 gnoise(p.xz * 0.11 + 7) >= 0.82, below ~950 m and without scree -> rockW = 0 there. Skipping them saves
// the near rock model on meadows / forest floor near the camera (round 2 review: ~25 % of the frame time).
// (not for outcrop blocks; the height test keeps hero rocks out)
bool rkVegSkip(vec3 p, vec3 n, float t) {
    if (gRkOnBlock || t > 59.0 || n.y < 0.72 || p.y > 700.0) return false;
    if (n.y + 0.1 * gnoise(p.xz * 0.11 + 7.0) < 0.825) return false;
    if (forestAt(p.xz).b > 0.0005) return false;
    return abs(p.y - terrainH(p.xz, false)) < 0.8;
}
// granite material for cliffs / outcrops of the terrain (world-space p, normal n)
vec3 rockAlbedo(vec3 p, vec3 n, float t) {
    if (rkVegSkip(p, n, t)) return vec3(0.205, 0.198, 0.182);      // (unused by the terrain shader)
    float fp = t * gPixAng;
    rkNearRock(p, n, t, fp);
    float slabW = rkSlabWeight(p);
    vec3 slabA = vec3(0.0);
    if (slabW > 0.0) {
        slabA = rkNearAlbedo(rkSlabAlbedo(p, n, t, fp), p, n, fp, 0.7) * (1.0 - 0.3 * gRkMP);
        float pkS;
        if (gRkPebP != p) slabA *= mix(vec3(1.0), rkMidTex(p, fp, 0.3, pkS), 0.8);   // (rocks5 M3)
        if (slabW >= 1.0) return slabA * rkGrain(rkPlaneUV(p, n), fp);
    }
    rkCliffJoints(p, fp);
    float steep = 1.0 - smoothstep(0.55, 0.85, n.y);
    // macro tone: large-scale variation of the massif + per-panel and per-block tones
    float n1 = fbm2(p.xz * 0.0045 + 3.0, 3);
    // (round 2 review) near rock in the hue of the granite (shaded faces read grey, not blue: #40403c),
    // back to the world lab's neutral (lit #80878b / shade #5c729d) over 150-300 m
    vec3 alb = mix(vec3(0.21, 0.198, 0.172), vec3(0.200, 0.205, 0.212), smoothstep(150.0, 300.0, t)) * (1.0 + 0.32 * n1);
    alb *= mix(0.86, 1.12, gRkJ.y);
    float blk = gRkJ.w * smoothstep(20.0, 120.0, t);          // block tones read as polygons up close
    alb *= mix(1.0, mix(0.90, 1.08, gRkJ.x), blk);
    // fresh rockfall scars: lighter, slightly warm blocks
    float scar = step(gRkJ.x, 0.07) * steep * blk;
    alb = mix(alb, vec3(0.40, 0.385, 0.35), scar * 0.85);
    // joints: dark crevice lines (softened up close where they would look painted)
    alb *= 1.0 - 0.6 * gRkJ.z * mix(0.25, 1.0, smoothstep(15.0, 80.0, t));
    // water / lichen streaks hanging down steep faces (triplanar on the vertical planes)
    vec3 w = p + vec3(0.0, 4.0 * gnoise(p.xz * 0.03), 0.0);
    float wx = n.x * n.x / max(n.x * n.x + n.z * n.z, 1e-4);
    float svis0 = 1.0 - smoothstep(4.0, 10.0, fp);
    float streak = 0.0;
    if (svis0 > 0.0 && steep > 0.0) {
        float sx = gnoise(vec2(w.z * 0.085, w.y * 0.0065 + 0.3 * gnoise(vec2(w.z * 0.03, 1.7))));
        float sz = gnoise(vec2(w.x * 0.085, w.y * 0.0065 + 0.3 * gnoise(vec2(w.x * 0.03, 5.1))));
        float s = mix(sz, sx, wx);
        float svis = 1.0 - smoothstep(0.6, 2.0, fp);
        float sfine = svis > 0.0 ? mix(gnoise(vec2(w.z * 0.3, w.y * 0.02)), gnoise(vec2(w.x * 0.3, w.y * 0.02)), wx) : 0.0;
        streak = mix(0.3, smoothstep(0.12, 0.42, s), svis0) + 0.6 * smoothstep(0.2, 0.45, sfine) * svis;
    } else streak = 0.3;
    alb *= 1.0 - 0.42 * saturate(streak) * steep * (1.0 - scar);
    // yellowish-green lichen tint on some blocks / upper faces
    float lt = smoothstep(0.55, 0.85, gRkJ.x * 0.6 + 0.4 * (gnoise(p.xz * 0.05) + 0.5));
    alb = mix(alb, alb * vec3(1.02, 1.05, 0.86), lt * 0.6);
    // close range: same granite surface detail as the boulders, divided by its own mean (GD_MEAN: no tone shift),
    // faded by the footprint only (round 2 review: the fixed 20-50 m fade made a second ring around the camera)
    float near = 1.0 - smoothstep(0.06, 0.15, fp);
    if (near > 0.0) {
        float lich;
        vec3 d = graniteDetail(p, n.y, fp, 1.0, steep, 0.0, lich);
        alb = mix(alb, alb * d / rkGDMean(steep), near * 0.85);
    }
    alb = rkNearAlbedo(alb, p, n, fp, 1.0);
    alb *= (1.0 - 0.4 * gRkMP) / (1.0 - 0.058 * (1.0 - smoothstep(0.008, 0.025, fp)));   // micro pits darker (mean 0.144 kept)
    // (rocks5 M3) terrain granite (outcrop knobs / mounds of shot 4, cliffs within ~100 m) the same mid-distance speckle as the
    // boulders: the stream-valley mounds read as smooth clay next to the textured blocks
    { float pkC; alb *= mix(vec3(1.0), rkMidTex(p, fp, steep, pkC), 0.8); }
    // (rocks3) crisp crystal grain up close (mean 1: no tone shift)
    return mix(alb, slabA, slabW) * rkGrain(rkPlaneUV(p, n), fp);
}

// (rocks5 M2 / M3) ~25 cm chipped facets (tilted voronoi cells in the face plane), kept until ~2 px: tangential normal offset
vec3 rkFacetG(vec3 p, vec3 n, float fp) {
    float vf0 = (1.0 - smoothstep(0.05, 0.12, fp)) * 0.35 * 1.7;
    if (vf0 <= 0.0) return vec3(0.0);
    const vec3 FA = vec3(0.8513, 0.1277, 0.5087), FB = vec3(0.1213, 0.4852, 0.866);
    vec3 ta = normalize(cross(n, abs(dot(n, FA)) < 0.85 ? FA : FB));
    vec3 tb = cross(n, ta);
    vec3 v0 = voronoi2(vec2(dot(p, ta), dot(p, tb)) * 4.0 + 1.7);
    vec2 t0 = (fract(v0.z * vec2(11.3, 23.7)) - 0.5) * vf0;
    return ta * t0.x + tb * t0.y;
}
// normal perturbation for rock faces: faceted joint blocks + panel tilt + surface relief
vec3 rockNormal(vec3 p, vec3 n, float t) {
    if (rkVegSkip(p, n, t)) return n;           // (unused by the terrain shader)
    float fp = t * gPixAng;
    rkNearRock(p, n, t, fp);                     // (cached: normally computed by rockAlbedo already)
    float slabW = rkSlabWeight(p);
    vec3 slabN = n;
    if (slabW > 0.0) {
        // polished slabs: broad whaleback undulation + low crystalline relief
        vec4 r = rkRelief(p, fp, 0.25, 0.35, 2);
        r += rkRelief(p + 3.3, fp, 0.012, 9.0, 2);
        // gentle 0.5 m undulation (round 2 review: the creased version drew noise contour lines across the slab)
        float vk = 1.0 - smoothstep(0.1, 0.35, fp * 1.6);
        if (vk > 0.0) {
            vec4 k1 = rkNoised(p * vec3(1.6, 2.4, 1.6) + 5.3);
            r.yzw += k1.yzw * vec3(1.6, 2.4, 1.6) * 0.025 * vk;
        }
        slabN = rkPerturb(n, r.yzw * (1.0 - smoothstep(60.0, 140.0, t)) + rkFacetG(p, n, fp));
        slabN = normalize(slabN + 0.7 * gRkNG + gRkMG);
        if (p == gRkPebP) slabN = normalize(slabN + gRkPebB * (1.0 - smoothstep(8.0, 25.0, t)));   // gravel cover
        if (slabW >= 1.0) return slabN;
    }
    rkCliffJoints(p, fp);
    float steep = smoothstep(0.35, 0.8, 1.0 - n.y);
    vec3 nn = normalize(n + gRkTilt * mix(0.12, 0.38, steep));
    if (t < 80.0 && (n.y < 0.84 || p.y > 900.0)) {
        // ridged relief: sharp creases (ledges, cracks) between flatter facets
        vec3 g = vec3(0.0);
        float a = 0.22, f = 0.42;
        for (int i = 0; i < 4; i++) {
            float vis = 1.0 - smoothstep(0.12, 0.45, fp * f);
            if (vis <= 0.0) break;
            vec4 nd = rkNoised(p * f + float(i) * 7.1);
            g += (i < 2 ? -sign(nd.x) : 1.0) * nd.yzw * (a * f * vis);
            a *= 0.45; f *= 2.25;
        }
        nn = rkPerturb(nn, g * (1.0 - smoothstep(45.0, 80.0, t)));
    }
    nn = normalize(nn + gRkNG + gRkMG + rkFacetG(p, n, fp));
    return normalize(mix(nn, slabN, slabW));
}

// --- boulders --------------------------------------------------------------------
// micro height (m, <= 0) of the crumbly granite surface at local point x: worley cones (texture b = inverted
// worley fbm, 1 at the cell sites) - pits with sharp crests between them; a1 ~9 cm cells, a2 ~3 cm cells;
// a4: 6-15 cm weathering pockets (the sites of a ~21 cm worley layer; a4 carries the patch mask)
float rkMicroH(vec3 x, float a1, float a2, float a4) {
    float b1 = texture(uNoise3, x * 1.4 + 0.17).b;
    float h = -a1 * (0.6 * b1 + 0.4 * smoothstep(0.68, 0.9, b1));
    if (a4 > 0.0) h -= a4 * smoothstep(0.6, 0.76, texture(uNoise3, x * 0.6 + 0.41).b);
    if (a2 > 0.0) {
        float b2 = texture(uNoise3, x * 3.9 + 0.53).b;
        h -= a2 * (0.4 * b2 + 0.6 * smoothstep(0.6, 0.86, b2));
    }
    return h;
}
// per-block tone and lichen-crust class (round 2 review): 25 % dark-crusted blocks (with their
// denser crust ~x0.55 of the mean), 15 % fresh pale ones (~x1.25), the rest x0.84-1.06; glacier slabs pale and
// clean, forest blocks darker (algae / moss film), inland field blocks a little darker (older surfaces)

// per-facet tone: every fracture face weathered differently (soft switch across the rounded edge)

// fresh surface of a flat chip (rkChipD): on its cut face, inside its ball

// --- hero blocks (hero.glsl) with the boulder material ----------------------------------------------------
// (rocks3b) PROPOSAL for scene.frag (not a rocks3 file): shade the K_HERO hits with shadeHeroRock(p, rd, t, i, cheap)
// instead of shadeHero, whose thresholded worley "lichen" draws evenly spaced yellow dots over the opening shot's
// block (shot 0) and whose grain is a smooth-noise multiplier. Same crisp granite / lichen rosettes / pitted relief
// with micro-shadows / contact + wet bands / falls spray / lighting as the scattered boulders (shadeRock).

// --- for the water shader: a rock seen through the water (K_ROCK hit of a refraction ray) ----------------
// returns the albedo (wet granite, olive algae film on the upper faces, dark contact zone) and the normal n
// (water_shade.glsl: if (hb.kind == K_ROCK) alb = rockBedAlbedo(bp, t, hb.cell, bn);)
// Round 2 review: the submerged part was brighter than the wet band above the water (pale pedestals): same wet
// darkening x0.62, block / facet tone like shadeRock, and the face's sky visibility (the water shader's ambient
// ignores the normal); smooth-shape normal with >= 2 cm taps (striped banding)

// ---------------------------------------------------------------------------
// shade_trees.glsl : conifer shading (needs trees.glsl + lighting.glsl)
// foliage: layered self-shadowing - near model: the tree's own (coarse) branches are marched toward the sun
// and toward the sky (treeOwnTrans: every spray layer passed lets ~55 % (sun) / 65 % (sky) through -> dark
// interior, light mostly on the outer sun-side twig tips), a clump-scale normal (sun side / shaded side of each
// spray clump, 4-20 m), mid model: soft shadow of the whorl skirts above (treeOwnTransMid) and dark tier
// undersides, plus the analytic path through the crown cone; neighbours' shadows, olive-green lit sprays
// with yellow-green fresh tips, translucency when backlit; bark: small grey-brown scales with lichen
// (spruce/fir) or red-brown plates (larch). Look terms use the 1080p reference footprint (dc * TREE_REF_PIXANG), so the
// look does not change with the render size; only anti-aliasing fades use the true footprint.
// ---------------------------------------------------------------------------

// sample pass index for trees.glsl (declared there before the frame uniforms exist)
int treeSampleIdx() { return uSample; }
int treeDebugMode() { return uDebug; }

// transmittance of this tree's own branches and sprays from p along L (unit), through the coarse near
// model (nearest branch per whorl, no side shoots): every spray layer the ray passes lets a fraction
// 'pass' through (porous needle mats), the trunk / branch wood a little less. Layered self-shadowing:
// the sprays of the whorls above shade the ones below, the outer sun-side tips stay lit, the interior
// behind several layers is near-black

// mid model: soft shadow of the whorl skirts above (each tier shades the one below it, the outer
// sun-side edge of a tier stays lit); the surface is the average spray surface, so 'res' only darkens
// to a floor (porous sprays)

// fissured bark: returns albedo, perturbs the normal (cylindrical frame around the trunk)

// ---------------------------------------------------------------------------
// terrain_shade.glsl : terrain materials (rock, scree, vegetation, snow, shore pebbles) and shading
// ---------------------------------------------------------------------------
// packed granite cobbles of mixed sizes; returns albedo and a bump normal offset
vec3 pebbles(vec2 p, out vec3 bump) {
    // shore of packed, angular-to-subrounded granite gravel (shingle of pale 1-10 cm chips with near-black
    // gaps): a dense layer of ~9 cm pebbles (60-80 % filled), ~4 cm pebbles in its gaps, ~2 cm pebbles and grit in
    // theirs, short cast shadows on the side away from the sun, and sparse 12-25 cm angular cobbles on top (beyond
    // ~14 m only: nearer, the 3D cobbles of rocks.glsl carry that size). Every pebble its own ellipse inside its
    // power-diagram cell (no shared-edge paving); each layer has its own rotated, warped grid (no tiling).
    // Under water an olive-brown algae film lowers the contrast (shallows #6e644a). Footprint filtered
    // toward the beach's mean (the film moves); the footprint grows at grazing angles.
    bump = vec3(0.0);
    float hT = terrainH(p, false);
    float dist = length(vec3(p.x, hT, p.y) - uCamPos);
    float cosT = clamp((uCamPos.y - hT) / max(dist, 1e-3), 0.15, 1.0);
    float fp = dist * gPixAng / pow(cosT, 0.35);      // pixel footprint on the ground (m) (rocks5: / cos^0.5 -> ^0.35: the grazing shore strip blurred to sand)
    // means of the fully resolved layers, gaps and shadows included (tools/rkpebmean.py, CPU replica)
    // (rocks3: re-measured for the finer layers; the composite is scaled x1.1 at the end -> beach mean ~0.2 as before)
    // (rocks4 R5: x0.92 - the smaller chips / longer cast shadows / chip facets darken the resolved layers; at x1.0 the
    // unresolved beach beyond ~8 m read as pale sand next to the near shingle)
    const vec3 AVG = vec3(0.256, 0.2348, 0.2145) * 0.92;          // (rocks3b: sparser chips, lighter grit; 3c: denser 2.5 cm layer)
    const vec3 AVGM = vec3(0.2422, 0.2223, 0.2027) * 0.94;          // the 2.5 cm layer over its gaps
    const vec3 AVGF = vec3(0.2071, 0.1903, 0.1724);                 // the 1.2 cm layer over grit
    // (rocks3) finer gravel (1-4 cm chips, a sparser 5 cm layer; the 9 cm plates read as crazy paving)
    const float SC = 3.4, SD = 24.0, SM = 52.0, SF = 100.0;   // (rocks3d: 19 / 40 / 80 -> 24 / 52 / 100: chips 1-4 cm at 3-6 m)
    float wl = waterLevel(p);
    float sub = wl > -1000.0 ? 1.0 - smoothstep(-0.12, -0.03, hT - wl) : 0.0;
    float zone = gnoise(p * 0.23 + 3.0) + 0.5 * gnoise(p * 0.9 + 1.3);   // sorting: coarse vs fine patches
    vec3 tint = vec3(1.0) + 0.05 * vec3(gnoise(p * 0.5 + 7.0), 0.0, -gnoise(p * 0.5 + 7.0));
    // sub-pixel beach: still mottled (25 cm and 8 cm tone patches of coarser / finer, paler / darker gravel, +-25 %)
    float fm = texture(uNoise3, vec3(p * 0.25, 0.71)).g - 0.487;
    if (fp < 0.05) fm += 0.6 * (texture(uNoise3, vec3(p * 0.75, 0.29)).g - 0.487) * (1.0 - smoothstep(0.025, 0.05, fp));
    vec3 far = AVG * (1.0 + 0.1 * zone) * (1.0 + 0.9 * fm);
    far *= 1.0 - 0.18 * sub;                      // (rocks3c) the bed is darker than the dry shingle
    // (rocks5 P2) the unresolved gravel keeps a crisp pixel-scale stone mosaic (rkGravelTex) instead of a smooth sand tone
    vec3 farT = far * rkGravelTex(vec3(p.x, hT, p.y), fp * (1.0 + 1.5 * sub));
    vec3 subT = vec3(0.78, 0.73, 0.47);           // (rocks3b: a little more olive, #6e644a)
    float visC = 1.0 - smoothstep(0.35, 0.8, fp * SC);
    float kB = mix(2.05, 1.15, sub);      // (rocks3d: 1.75 -> 1.95, r1 beach median 157 vs target 184 with the narrower stone tones; rocks4 R5: 2.2)
         // (rocks3c: dry 1.15 -> 1.75: r1 beach median sRGB 133 vs target 184, beach #b9ab9f ~ as bright as lit boulder tops)
    if (visC <= 0.0) return far * tint * kB * mix(vec3(1.0), subT, sub);   // (rocks3b: x1.15 on the new means = +33 %, the target beach #b9ab9f is paler)
    vec2 sunXZ = normalize(uSunDir.xz + vec2(1e-5));
    float sunT = length(uSunDir.xz) / max(uSunDir.y, 0.15);
    // slow domain warp shared by the layers (~1/4 of a small cell)
    vec2 wp = (texture(uNoise3, vec3(p * 0.35, 0.53)).rg - vec2(0.475, 0.487)) * 0.03;
    // sparse angular cobbles on top (polygonal, rounded corners, tilted); none within ~14 m of the camera
    const mat2 RC = mat2(0.8196, 0.5729, -0.5729, 0.8196);      // rot2(0.61)
    vec2 tC, bC = vec2(0.0); float gC = 1.0;
    vec3 C = vec3(0.0, 0.0, 1.0);
    // (rocks3c) under water: elliptical, rounded stones of a broad size mix (rocks3b: one size of rounded cells = an even
    // pavement of pale pillows / stepping-stone tiles in r10 / r1 / r4)
    // (rocks4 R8) per-stone elongation under water (el < 0: random 0.1-0.5) and sizes 0.4-1.4 (below)
    vec3 vC = rkPack(RC * p * SC + 0.37, 0.3, mix(0.25, 0.7, sub), -0.33 * sub, tC);
    if (sub > 0.0) { float rC = length(tC); vC.x = mix(vC.x, min(vC.x, (rC + vC.x) * mix(0.4, 1.4, fract(vC.z * 17.9)) - rC), sub); }
    // (rocks3b) under water a pavement of rounded cobbles at every distance (10-30 cm cobbles and boulders,
    // #6e644a - not sand / grit), only to ~3-6 m depth (deeper the water shader turns the bed to silt anyway - and every
    // lake pixel pays for it)
    float zc = saturate(zone + 0.5);
    float subP = sub * (1.0 - smoothstep(3.0, 6.0, wl - hT));
    // (rocks3c) in PATCHES (2-6 m): dense cobble fields, boulder clusters and finer gravel between them;
    // zB ~ N(0,1) patch fields
    float zB1 = 0.0, zB2 = 0.0;
    if (subP > 0.0) {
        zB1 = (texture(uNoise3, vec3(p * 0.045, 0.83)).g - 0.487) * 4.0 + (texture(uNoise3, vec3(p * 0.17, 0.17)).g - 0.487) * 2.5;
        zB2 = (texture(uNoise3, vec3(p * 0.06 + 0.31, 0.47)).g - 0.487) * 4.0 + (texture(uNoise3, vec3(p * 0.2 + 0.7, 0.61)).g - 0.487) * 2.5;
    }
    // (rocks3c) dry beach: the sparse 12-22 cm cobbles near the camera too (~5-10 stones > 8 cm per m2; the 3D
    // cobbles of rocks.glsl reach ~1 / m2)
    // (rocks4 R8: under water 0.5 + 0.75 zB1 -> 0.3 + 0.6 zB1, more fine gravel; R5: dry 0.03-0.15 from 12 m -> 0.1-0.32 from
    // 7 m - the beach beyond the resolved 4 cm layer read as smooth sand, cobbles belong there)
    // (rocks5 B1) bed: 0.3 -> 0.55 + 0.6 zB1 (shallows: packed 5-30 cm cobbles, not plates)
    float pC = mix(mix(0.1, 0.32, zc) * smoothstep(7.0, 10.0, dist), saturate(0.55 + 0.6 * zB1), subP);
    if (fract(vC.z * 11.3) < pC) {
        float gc = fract(vC.z * 4.1);
        C = rkPackShade(vC, tC, mix(mix(0.12, 0.3, gc), mix(0.04, 0.2, gc), subP), step(mix(0.8, 0.55, subP), fract(vC.z * 3.7)), fp * SC, RC * sunXZ, sunT, 1.0, bC, gC);
        // (rocks3c) under water softer self-shading / cast shadows (refracted, scattered light): the identical dark crescent
        // on every cobble embossed the bed like buttons
        C.z = mix(C.z, 1.0, 0.3 * subP); gC = mix(gC, 1.0, 0.35 * subP);   // (rocks3d: ring 0.35 -> 0.5; rocks5 B2: 0.3 / 0.35)
        bC = bC * RC;                                          // back to world xz (inverse rotation)
    }
    // (rocks3b) under water: boulders among the cobbles; (rocks3c) 0.5-1.4 m (1 m cells, broad weights), clustered
    vec3 B = vec3(0.0, 0.0, 1.0), vB = vec3(0.0);
    vec2 bB = vec2(0.0); float gB = 1.0;
    if (subP > 0.0) {
        const float SB = 0.72;                                  // (rocks3d: 1 -> 1.4 m cells, boulders 0.5-1.5 m)
        const mat2 RB = mat2(-0.8569, 0.5155, -0.5155, -0.8569);   // rot2(2.6)
        vec2 tB;
        vB = rkPack(RB * p * SB + 2.9, 0.15, 0.7, 0.25, tB);       // (rocks3d: kr 0.3 / el 0.4 -> 0.15 / 0.25: subangular blocks)
        if (fract(vB.z * 9.7) < saturate(0.4 + 0.5 * zB2) * subP) {   // (rocks5 B1: 0.8 -> 0.4)          // (rocks3d: 0.35 -> 0.8: near bed = mostly 0.5-1.5 m blocks; k8 sweep 0.65 / 0.9)
            B = rkPackShade(vB, tB, mix(0.05, 0.18, fract(vB.z * 4.1)), step(0.6, fract(vB.z * 2.9)), fp * SB, RB * sunXZ, sunT, 1.0, bB, gB);
            B.z = mix(B.z, 1.0, 0.15);                                // (rocks3d: 0.35 / gB lift 0.2 dropped: boulders keep their 3D shading and dark gaps)
            bB = bB * RB;
        }
    }
    // the packed layers
    // seen through the rippled water the small layers blur earlier (refraction distortion, absorption)
    float fpS = fp * (1.0 + 1.5 * sub);
    vec3 cD = farT;
    vec2 bD = vec2(0.0);
    float visD = 1.0 - smoothstep(0.4, 0.9, fpS * SD);       // (rocks4 R5: 0.3-0.7 -> 0.4-0.9, chips down to ~1.1 px: no "sand" at 8-15 m)
    if (visD > 0.0 && C.x < 0.999) {
        // ~9 cm pebbles, 60-80 % of the ground (fewer in fine patches)
        const mat2 RD = mat2(0.1987, 0.9801, -0.9801, 0.1987);  // rot2(1.37)
        vec2 tD, bmpD; float gD;
        vec2 qD = RD * (p - 1.7 * wp) * SD + 5.1;
        vec3 vD = rkPack(qD, 0.07, 0.6, 0.06, tD);           // (rocks3b: angular chips, kr 0.12 / el 0.15 = ovals)
        { float rD = length(tD); vD.x = min(vD.x, (rD + vD.x) * mix(0.45, 1.3, fract(vD.z * 17.9)) - rD); }   // (rocks4 R5) chip sizes x0.45-1.3
        // (i8: stronger sorting - coarse patches 15 % holes, fine ones 85 %)
        // (rocks3b: the shore is a matrix of 0.5-2 cm grit with SPARSE 3-8 cm flat chips - 15-85 % holes packed the 5 cm
        // layer into an egg paving; now 50-92 % holes; j3 sweep: 35-85 % read as paving again)
        // (rocks5 P3) denser 4 cm stones (holes x0.5): the shingle is packed stones, not grit with sparse chips
        float holeD = step(fract(vD.z * 7.7), mix(mix(0.5, 0.92, saturate(0.5 - zone)) * 0.5, 0.85, sub));   // (under water: mostly fine gravel)
        float shD = fract(vD.z * 5.9);
        vec3 D = rkPackShade(vD, tD, mix(0.02, 0.12, shD), 0.2 + 0.7 * fract(vD.z * 3.3), fp * SD, RD * sunXZ, sunT, 0.0, bmpD, gD);
        bmpD = bmpD * RD;
        D.x *= 1.0 - holeD;
        // ~4 cm pebbles in the gaps / holes of the 9 cm layer
        // (rocks3c) the unresolved 2.5 / 1.2 cm layers are not a flat mean (= "asphalt" between the pebbles at 3-6 m, r1):
        // a mean-preserving mottle at 6.6 / 3.3 cm until it is ~3 px
        vec3 cM = AVGM * (1.0 + 1.3 * (texture(uNoise3, vec3(p * 2.0, 0.59)).a - 0.5207) * (1.0 - smoothstep(0.012, 0.03, fp)));
        vec2 bM = vec2(0.0);
        float visM = 1.0 - smoothstep(0.3, 0.7, fpS * SM);
        if (visM > 0.0 && D.x < 0.999) {
            const mat2 RM = mat2(-0.4425, 0.8968, -0.8968, -0.4425);   // rot2(2.03)
            vec2 tM, bmp; float gS;
            vec2 qM = RM * (p + wp) * SM + 3.7;
            vec3 vM = rkPack(qM, 0.08, 0.45, 0.06, tM);
            { float rM = length(tM); vM.x = min(vM.x, (rM + vM.x) * mix(0.45, 1.3, fract(vM.z * 17.9)) - rM); }   // (rocks4 R5)
            float hole = step(fract(vM.z * 7.7), mix(0.15, 0.4, saturate(0.5 - zone)));   // (rocks3b: was 8-32 %, 3b 30-60 %; 3c 15-40 %)
            float sh = fract(vM.z * 5.9);
            // gap width varies along each pebble's outline (pebbles touch here, leave a gap there)
            float gvM = 0.25 + 1.5 * texture(uNoise3, vec3(qM * 0.12, 0.13)).g;
            vec3 M = rkPackShade(vM, tM, mix(0.02, 0.18, sh * sqrt(sh)) * gvM, 0.2 + 0.7 * fract(vM.z * 3.3), fp * SM, RM * sunXZ, sunT, 0.0, bmp, gS);
            bmp = bmp * RM;
            M.x *= 1.0 - hole;
            // small pebbles + grit in the gaps and holes (shadowed by the pebble beside them)
            float gA4 = texture(uNoise3, vec3(p * 4.0, 0.37)).a;
            vec3 cF = AVGF * (1.0 + 1.4 * (gA4 - 0.5207) * (1.0 - smoothstep(0.006, 0.014, fp)));
            vec2 bF = vec2(0.0);
            float visF = 1.0 - smoothstep(0.24, 0.5, fpS * SF);
            if (M.x < 0.999 && visF > 0.0) {
                const mat2 RF = mat2(0.3902, 0.9208, -0.9208, 0.3902);   // rot2(1.17)
                vec2 tF; float gF;
                vec2 qF = RF * (p - wp) * SF + 1.3;
                vec3 vF = rkPack(qF, 0.15, 0.4, 0.05, tF);
                float sf = fract(vF.z * 5.9);
                float gvF = 0.25 + 1.5 * texture(uNoise3, vec3(qF * 0.12, 0.61)).g;
                vec3 F = rkPackShade(vF, tF, mix(0.04, 0.25, sf) * gvF, 0.1 + 0.6 * fract(vF.z * 3.3), fp * SF, RF * sunXZ, sunT, 0.0, bF, gF);
                bF = bF * RF;
                vec3 grit = vec3(0.15, 0.139, 0.122) * (0.6 + 0.8 * gA4);   // (rocks3b: 0.1 = black gaps)
                vec3 st = rkCobCol(vF.z) * mix(0.8, 1.0, F.y) * 0.9 * F.z;
                cF = mix(cF, mix(grit * gF, st, F.x), visF);
                bF *= visF;
            }
            cF *= mix(gS, 0.92, hole);                         // crevice / cast shadow of the pebble beside it (rocks3b: hole 0.6 -> 0.92, many holes now)
            vec3 st = rkCobCol(vM.z) * mix(0.85, 1.0, M.y) * M.z * 1.04;
            cM = mix(cM, mix(cF, st, M.x), visM);
            bM = mix(bF, bmp, M.x) * visM;
        }
        cM *= mix(gD, 0.95, holeD);                            // crevice / cast shadow of the 5 cm pebble beside it (rocks3b: 0.75 -> 0.95)
        vec3 stD = rkCobCol(vD.z) * mix(0.85, 1.0, D.y) * D.z * 1.04;
        cD = mix(farT, mix(cM, stD, D.x), visD);
        bD = mix(bM, bmpD, D.x) * visD;
    }
    cD *= gC * mix(1.0, 0.7, subP);                            // the cobbles shade their surroundings (under water: shaded, algae-filmed gaps)
    // (rocks3c) bed stones: rkBedCol (darker, olive / brown / dark stones mixed in) instead of the dry shingle colours
    float kBed = 1.0;
    vec3 stC = mix(min(rkCobCol(vC.z), vec3(0.5)), rkBedCol(vC.z) * kBed, subP) * mix(0.85, 1.0, C.y) * C.z;
    vec3 c = mix(cD, stC, C.x);
    vec3 stB = rkBedCol(vB.z) * kBed * mix(0.85, 1.0, B.y) * B.z;
    // (rocks5 P4) the cobbles / bed boulders are textured stones (5 / 2 cm mottle + dark specks, +-30 %), not smooth soap pillows
    // (r1 / r10 shallows); faded by the footprint (through the water it blurs earlier)
    { float v5 = 1.0 - smoothstep(0.022, 0.04, fpS), v2 = 1.0 - smoothstep(0.009, 0.016, fpS);
      if (v5 > 0.0 && max(C.x, B.x) > 0.0) {
          vec4 s5 = texture(uNoise3, vec3(p * 1.25, fract(vC.z * 3.1 + vB.z)));
          float sm = (s5.g - 0.487) * 2.0 * v5 + (v2 > 0.0 ? (texture(uNoise3, vec3(p * 3.1, 0.37)).g - 0.487) * 1.6 * v2 : 0.0);
          float spk = v2 * smoothstep(0.66, 0.74, s5.a);
          float kS = 1.0;
          vec3 mt = vec3(1.0 + kS * sm) * (1.0 - 0.45 * kS * spk);
          stC *= mt; stB *= mt;
      } }
    c = mix(c * gB, stB, B.x);
    // granite grain / dark lichen crust on the stones (only where resolvable)
    float gv = 1.0 - smoothstep(0.004, 0.02, fp);
    if (gv > 0.0) {
        float gr = texture(uNoise3, vec3(p * 2.3, vC.z * 0.7 + 0.1)).b;
        float dk = smoothstep(0.66, 0.74, texture(uNoise3, vec3(p * 3.1, 0.77)).a);
        c *= 1.0 + gv * (0.24 * (gr - 0.5) - 0.35 * dk * C.x);
        // dark mineral / lichen specks on every stone (2-6 mm)
        float gs = 1.0 - smoothstep(0.0015, 0.005, fp);
        if (gs > 0.0) c *= 1.0 - 0.4 * gs * smoothstep(0.68, 0.76, texture(uNoise3, vec3(p * 9.0, 0.53)).a);
    }
    // (rocks4 R5) dry shingle: more tonal contrast around the beach mean (target p10 / p50 / p90 55 / 174 / 222 vs 101 / 188 / 223:
    // darker gaps / shaded chip facets, a little paler lit chips; the mean drops slightly; sweep 0.45 / 0.8); none under water
    { float kCt = 1.4 * (1.0 - sub);          // (rocks5: 0.8 -> 1.4, target shingle p10 55 / hp 35-46)
      float rL = max(dot(c, vec3(0.3333)) / max(dot(far, vec3(0.3333)), 1e-4), 1e-3);
      c *= pow(rL, rL < 1.0 ? kCt : 0.35 * kCt); }
    c = mix(farT, c, visC) * tint * kB;
    // under water: algae film, lower stone / gap contrast
    c = mix(c, mix(c, far * 1.15, 0.2) * subT, sub);           // (rocks3b: 0.3 -> 0.2, the cobbles keep their contrast)
    vec2 bm = mix(bC + bD * (1.0 - C.x), bB, B.x) * visC;
    bump = vec3(bm.x, 0.0, bm.y) * 0.75 * (1.0 - 0.45 * sub);    // (rocks3c: 0.3 -> 0.45, flatter lit domes under water)
    return c;
}

// (session 6) world2 look-dev knobs: 0 = the shipped values; the lab's wt/kv.py sweeps them inside one
// process (a shader edit costs a 15-20 min compile on this iGPU). Unset in the other passes -> 0.

float W2K(int i) { return uW2K[i >> 2][i & 3]; }
float gRkWS = 0.0, gRkDS = 0.0;             // (world3) joint-block relief weight / depth at the shaded point (shadeTerrain)
float gVegS = 0.0;                          // (world3) turf cover over the slab relief at the shaded point (terrainAlbedo -> shadeTerrain)

// (world3) sun shadows cast by the joint-block relief (world.glsl rockRelief): the baked shadow maps only
// know the base heightfield. March toward the sun until the ray is above the base surface (the relief only
// lowers the ground, so nothing above it can block); crisp, softened to the pixel footprint
float rockReliefShadow(vec3 p, float fp) {
    float res = 1.0;
    float t = max(0.25, 1.5 * fp);
    // (t20, perf: 16 -> 10 steps, no visible change in the 1:1 crops; t29: 8)
    for (int i = 0; i < 8; i++) {
        vec3 q = p + uSunDir * t;
        float hb = terrainH(q.xz, true);
        if (q.y > hb + 0.05) break;
        float dy = q.y - (hb + rockRelief(q.xz, hb));
        res = min(res, dy / max(0.012 * t, 0.8 * fp));
        if (res < -1.0) break;
        t += clamp(abs(dy) * 0.7, max(0.25, 0.7 * fp), 10.0);
        if (t > 160.0) break;
    }
    return smoothstep(-1.0, 1.0, res);
}

// fbm2 whose octaves fade out once they get finer than a few pixels: the macro masks below must not
// speckle or shimmer at 1-5 km (wl = wavelength of the first octave in metres)
float fbm2F(vec2 p, int oct, float wl, float fp) {
    float s = 0.0, a = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < oct; i++) {
        float v = smoothstep(1.6 * fp, 3.6 * fp, wl);
        if (v <= 0.0) break;
        s += a * v * gnoise(p); p = m * p; a *= 0.5; wl *= 0.5;
    }
    return s;
}

// Face coordinates for the mountain faces: of four vertical projection planes (every 45 deg of
// aspect) the two closest to the coarse aspect, blended. u runs along the plane (horizontally), v is
// the height, so v-elongated patterns hang down the fall line within ~15 deg for every aspect (two
// axis-aligned planes gave a diagonal hatch on the NE/NW faces) and nothing swirls with the normal.
vec2 gFa, gFb;            // (u, v) in plane A / B
vec3 gFTa, gFTb;          // horizontal unit axis of plane A / B (world)
float gFwa, gFwb;         // blend weights (sum 1)
void faceFrames(vec3 p, vec3 nC) {
    float a = atan(nC.z, nC.x) / (0.25 * PI);
    float k0 = floor(a), fr = a - k0;
    gFwb = smoothstep(0.3, 0.7, fr); gFwa = 1.0 - gFwb;
    float aa = k0 * 0.25 * PI, ab = aa + 0.25 * PI;
    gFTa = vec3(-sin(aa), 0.0, cos(aa)); gFTb = vec3(-sin(ab), 0.0, cos(ab));
    gFa = vec2(dot(p.xz, gFTa.xz), p.y); gFb = vec2(dot(p.xz, gFTb.xz), p.y);
}
// streak noise in face coordinates (fq: frequency along the face, down the face)
float faceNoise(vec2 fq, float off) {
    float a = 0.0;
    if (gFwa > 0.02) a += gFwa * gnoise(gFa * fq + off);
    if (gFwb > 0.02) a += gFwb * gnoise(gFb * fq + off + 17.3);
    return a;
}
// fbm in the same face coordinates, octaves faded by the footprint like fbm2F (wl: first octave, m)
float faceFbm(vec2 fq, int oct, float wl, float fp, float off) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < oct; i++) {
        float v = smoothstep(1.6 * fp, 3.6 * fp, wl);
        if (v <= 0.0) break;
        s += a * v * faceNoise(fq, off);
        fq *= 2.03; a *= 0.5; wl *= 0.5; off += 3.7;
    }
    return s;
}
// relief of the faces below the heightfield resolution: V-shaped gullies between rounded ribs, long
// down the face; octaves faded by the footprint; returns the world-space height gradient (m/m)
// (cav: the creases as dark crack / cleft lines, visible in flat frontal light too)
vec3 faceBump(float fp, out float cav) {
    vec3 g = vec3(0.0);
    cav = 0.0;
    // (session 4: 0.026 / 0.009 -> 0.022 / 0.014: the tall vertical ribs made the faces look like dripping
    // wax at 1720+ px)
    vec2 fq = vec2(0.022, 0.014);                             // ribs ~1.6x longer down the face than wide
    float a = 13.0, wl = 38.0, wc = 0.5;
    for (int i = 0; i < 4; i++) {
        float v = smoothstep(2.4 * fp, 5.0 * fp, wl);   // (session 6: 1.6..3.6 px shimmered)
        if (v <= 0.0) break;
        if (gFwa > 0.02) {
            vec3 nd = gnoised(gFa * fq + 4.1 + float(i));
            g += (gFwa * v * a * -sign(nd.x) * nd.y * fq.x) * gFTa + vec3(0.0, gFwa * v * a * -sign(nd.x) * nd.z * fq.y, 0.0);
            cav += gFwa * v * wc * (1.0 - smoothstep(0.0, 0.09, abs(nd.x)));
        }
        if (gFwb > 0.02) {
            vec3 nd = gnoised(gFb * fq + 9.3 + float(i));
            g += (gFwb * v * a * -sign(nd.x) * nd.y * fq.x) * gFTb + vec3(0.0, gFwb * v * a * -sign(nd.x) * nd.z * fq.y, 0.0);
            cav += gFwb * v * wc * (1.0 - smoothstep(0.0, 0.09, abs(nd.x)));
        }
        fq *= 2.1; a *= 0.5; wl *= 0.48; wc = (i == 0) ? 0.9 : wc * 0.8;
    }
    return g;
}

// (session 6) fall-line flutes of the granite faces (the walls are scored by narrow
// V-gullies with sharp, shadowed bottoms between broader ribs, broken by ledges along their length).
// Height ~ +|n| of an anisotropic (3:1 down the face) gradient noise: the crease at n = 0 is the gully
// floor. 3 octaves (28 / 12.6 / 5.7 m), each faded before it is < ~2.5-5 px; returns the height gradient
vec3 faceFlutes(float fp) {
    vec3 g = vec3(0.0);
    float wl = 28.0 * (1.0 + W2K(21)), a = 1.0;
    for (int i = 0; i < 3; i++) {
        float v = smoothstep(2.4 * fp, 5.0 * fp, wl);
        if (v <= 0.0) break;
        vec2 fq = vec2(1.0 / wl, 0.33 / wl);
        for (int k = 0; k < 2; k++) {
            float w = k == 0 ? gFwa : gFwb;
            if (w < 0.04) continue;
            vec2 F = k == 0 ? gFa : gFb;
            vec3 T = k == 0 ? gFTa : gFTb;
            vec3 nd = gnoised(F * fq + vec2(5.3 + 3.1 * float(i), 1.7 + 2.3 * float(k)));
            // flutes come and go along the face (ledges, blocky sections)
            float brk = smoothstep(-0.30, 0.20, gnoise(F * fq * vec2(0.45, 1.6) + vec2(11.0 + float(i), 3.0 * float(k))));
            float s = sign(nd.x) * a * 0.16 * wl * brk * w * v;
            g += s * (nd.y * fq.x * T + vec3(0.0, nd.z * fq.y, 0.0));
        }
        wl *= 0.45; a *= 0.85;
    }
    return g;
}

// voronoi2 (same cells) + bn: the unit vector from the nearest to the second-nearest feature point, i.e.
// the normal of the cell border between them (used to keep only the near-vertical / near-horizontal
// borders as cracks: joint sets, not a polygon network)
vec3 voronoi2n(vec2 p, out vec2 bn, out float id2) {
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i);
    float d1 = 8.0, d2 = 8.0, id = 0.0; vec2 q1 = vec2(0.0), q2 = vec2(1.0, 0.0);
    id2 = 0.0;
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        ivec2 o = ivec2(x, y);
        vec2 r = vec2(o) + hash22i(c + o) * 0.85 + 0.075 - f;
        float d = dot(r, r);
        float hid = hash12i(c + o + ivec2(17, 31));
        if (d < d1) { d2 = d1; q2 = q1; id2 = id; d1 = d; q1 = r; id = hid; }
        else if (d < d2) { d2 = d; q2 = r; id2 = hid; }
    }
    bn = normalize(q2 - q1 + vec2(1e-6, 0.0));
    return vec3(sqrt(d1), sqrt(d2), id);
}

// fractured granite below the heightfield resolution (at 0.3-5 km): planar facets with
// sharp edges, voronoi cells in face coordinates (taller than wide: vertical ribs and joints), each
// tilted at random so the sun picks them out light / dark; their borders are dark cracks. Two scales,
// each faded out before its cells get smaller than ~5 px. Returns the world-space normal offset.
vec3 faceFacets(float fp, out float crack, out float ftone) {
    vec3 g = vec3(0.0);
    crack = 0.0; ftone = 0.0;
    // (session 6: 30 m / tilt 3.2 (up to +-58 deg) read as crumpled foil at 0.5-3 km -> 55 m, 1.7)
    // (session 3: tilt 1.7 -> 1.2, crack 0.85 -> 0.6, cell tone 0.12 -> 0.05: at 3-5 km the cells read as
    // crazy paving - a mosaic of polygon tones; the faces should have smooth tone with crisp dark lines)
    // (session 4) blockier: cells 1.15x (was 1.8x) taller than wide, vertical tilt 0.35 -> 0.9 (lit ledge
    // tops, shaded risers), crisper cell edges (blend 0.12 -> 0.05 of a cell, never under ~1.5 px), and a
    // third octave (~7.5 m) for the 3440 px final, where the 4 m heightfield is soft
    // (session 6) the random cell tilts are OFF (amp 0, knob 0 brings them back): with grey albedo (debug 28)
    // they were the camouflage of the faces - crisp dark polygons (t19-t21) or, with soft borders, soft dark
    // blobs (t22b); the faces read as rock once the shading comes from the organised fall-line relief
    // (faceFlutes, faceBump, the heightfield). The joint-set cracks on the cell borders stay.
    float S = 55.0, amp = 1.5 * W2K(0), cd = 0.7;   // (session 5: 1.2 / 0.6 -> 1.5 / 0.7: rock 8-px contrast 0.047 vs 0.06-0.085)
    for (int i = 0; i < 3; i++) {
        float v = smoothstep(3.0 * fp, 7.0 * fp, S);          // (session 6: was 2..5 px, shimmered at 2-4 km)
        if (v <= 0.0) break;
        float eb = max(0.05, 1.5 * fp / S);
        for (int k = 0; k < 2; k++) {
            float w = k == 0 ? gFwa : gFwb;
            if (w < 0.04) continue;
            vec2 uv = (k == 0 ? gFa : gFb) / vec2(S, 1.15 * S) + vec2(float(i) * 7.3 + float(k) * 3.1, 0.0);
            vec2 bn; float id2;
            vec3 vr = voronoi2n(uv, bn, id2);
            float edge = vr.y - vr.x;
            vec2 tl = vec2(fract(vr.z * 13.7), fract(vr.z * 29.3)) - 0.5;
            vec2 tl2 = vec2(fract(id2 * 13.7), fract(id2 * 29.3)) - 0.5;
            vec3 T = k == 0 ? gFTa : gFTb;
            float ww = w * v;
            // (session 4) only the near-vertical (joints) and near-horizontal (sheeting) borders are cracks:
            // the full voronoi network read as a dark snakeskin at the final 3440 px
            float jset = max(smoothstep(0.74, 0.95, abs(bn.x)), 0.7 * smoothstep(0.80, 0.97, abs(bn.y)));
            // (session 6, knob 1) crisp creases only on the joint-set borders; the oblique borders blend the
            // two cells' tilts over a third of a cell (a crisp crease on every border = polygon mosaic, t19
            // shot 7); the border takes the mean tilt, so the normal stays continuous
            float ebk = mix(eb, max(eb, 0.35), W2K(1) * (1.0 - jset));
            g += ww * amp * mix(W2K(1) * 0.5 * (tl + tl2), tl, smoothstep(0.0, ebk, edge)).x * T
               + ww * amp * vec3(0.0, 0.9 * mix(W2K(1) * 0.5 * (tl.y + tl2.y), tl.y, smoothstep(0.0, ebk, edge)), 0.0);
            float cw = clamp(1.2 * fp / S, 0.03, 0.25);
            // only part of the cell borders are open cracks (a full network reads as crazy paving)
            float open = smoothstep(-0.05, 0.25, gnoise(uv * 2.3 + vec2(1.7, 4.1)));
            crack += ww * cd * open * jset * (1.0 - smoothstep(0.0, cw, edge));
            ftone += ww * (fract(vr.z * 7.1) - 0.5);
        }
        S *= 0.37; amp *= 0.6; cd *= 0.7;   // (t20: amp decay 0.75 -> 0.6: the 20 m cells read as crazy paving at 1720 px)
    }
    crack = saturate(crack);
    return g;
}

// (round 3) glacier-polished slabs / knobs where rock breaks through the turf on gentle ground (no rockRelief there):
// flat pale cut-outs at 1:1 (crop c5, s5 / s7 slopes) - slabs with their own tilts (lit / shaded), crisp joint cracks
// between them. x-z Voronoi (gentle ground: no fall-line stretch), 14 / 5 m, footprint-faded; returns the normal
// offset (x, 0, z), crack in [0, 1]
vec3 outcropSlabs(vec2 p, float fp, out float crack) {
    vec3 g = vec3(0.0);
    crack = 0.0;
    float S = 14.0, amp = 0.55;
    for (int i = 0; i < 2; i++) {
        float v = smoothstep(2.5 * fp, 5.0 * fp, S);
        if (v <= 0.0) break;
        vec2 bn; float id2;
        vec3 vr = voronoi2n(rot2(0.6 + 1.1 * float(i)) * p / S + vec2(5.1, 2.3) * float(i + 1), bn, id2);
        vec2 tl = vec2(fract(vr.z * 13.7), fract(vr.z * 29.3)) - 0.5;
        vec2 tl2 = vec2(fract(id2 * 13.7), fract(id2 * 29.3)) - 0.5;
        float edge = vr.y - vr.x;
        float eb = max(0.06, 1.2 * fp / S);
        vec2 t2 = mix(0.5 * (tl + tl2), tl, smoothstep(0.0, eb, edge));
        g += v * amp * vec3(t2.x, 0.0, t2.y);
        crack = max(crack, v * (1.0 - smoothstep(0.0, max(0.05, 0.9 * fp / S), edge)) * smoothstep(-0.1, 0.2, gnoise(p * (2.0 / S) + 3.7)));
        S *= 0.36; amp *= 0.7;
    }
    return g;
}

// (session 5) fine joint texture of the granite faces down to ~2.5 px: the faces carry
// a dense texture of thin dark joints in two sets - near-vertical joints and dipping sheet joints - on top of
// the facets; the renders had ~60 % of the target 8-px rock contrast. Each line is the zero crossing of an
// anisotropic (5:1) gradient noise in face coordinates, drawn with a constant width in metres from the
// noise's own gradient (|n| / |grad n|), at least ~1 px wide and faded by coverage below that; an octave
// goes before its line spacing is < ~2.5 px. Anisotropic, so the lines are joint sets, never the closed
// loops of an isotropic noise (the snakeskin). Per-plane lines blended by weight (no mixed zero crossings).
float jointSet(vec2 F, vec2 dir, float wl0, float fp, float off) {
    vec2 q = vec2(dot(F, vec2(-dir.y, dir.x)), dot(F, dir));
    float L = 0.0, wl = wl0, D = 0.55, wln = 0.9;   // (t19: 0.55 / 0.45 m -> 0.62 / 0.9 m: t18 barely showed; t20 D 0.55)
    for (int i = 0; i < 4; i++) {
        float v = smoothstep(1.8 * fp, 4.5 * fp, 0.5 * wl);
        if (v <= 0.0) break;
        vec2 fq = vec2(1.0 / wl, 0.2 / wl);
        vec3 nd = gnoised(q * fq + off + float(i) * 7.13);
        float d = abs(nd.x) / max(length(nd.yz * fq), 1e-5);   // metres to the zero crossing
        float hw = max(wln, 0.75 * fp);
        float line = (1.0 - smoothstep(0.5 * hw, hw, d)) * min(1.0, 1.4 * wln / max(0.75 * fp, 1e-4));
        // lines come and go along their length (open / closed joints)
        // (t20: segments ~2 wavelengths long instead of ~7: long continuous lines read as drawn contours)
        line *= smoothstep(-0.30, 0.10, gnoise(q * fq * vec2(0.5, 1.4) + off + 3.7 + float(i)));
        L = max(L, v * D * line);
        wl *= 0.45; D *= 0.9; wln *= 0.7;
    }
    return L;
}
float graniteJoints(vec2 pxz, float fp) {
    // set A ~ vertical (+-12 deg), set B dipping sheets (+-25 deg off horizontal), both turning slowly
    float aA = 0.21 * gnoise(pxz * 0.0011 + 61.0);
    float aB = 0.45 * gnoise(pxz * 0.0009 + 73.0);
    vec2 dA = vec2(sin(aA), cos(aA)), dB = vec2(cos(aB), sin(aB));
    float mA = smoothstep(-0.45, 0.10, gnoise(pxz * 0.0035 + 51.0));
    float mB = smoothstep(-0.40, 0.15, gnoise(pxz * 0.0042 + 57.0));
    float j = 0.0;
    for (int k = 0; k < 2; k++) {
        float w = k == 0 ? gFwa : gFwb;
        if (w < 0.04) continue;
        vec2 F = k == 0 ? gFa : gFb;
        float lA = mA > 0.01 ? mA * jointSet(F, dA, 34.0, fp, 11.0 + 5.0 * float(k)) : 0.0;
        float lB = mB > 0.01 ? mB * jointSet(F, dB, 26.0, fp, 29.0 + 5.0 * float(k)) : 0.0;
        j += w * max(lA, 0.85 * lB);
    }
    return j;
}

// Macro materials of the mountains: above the forest the slopes are a mosaic of
// grass ledges, dark dwarf-pine patches and granite outcrops; pale scree fans run down below the
// cliffs; summer snow survives as sharp-edged patches in couloirs and on benches, and as fields near
// the summits. Close to the camera the vegetation mask stays the one the 3D ground cover uses.
vec3 terrainAlbedo(vec3 p, inout vec3 n, float t, vec4 f, float wl, float ao, out float wetness, out float snowM, out float spec) {
    gVegS = 0.0;
    if (uDebug == 44) { spec = 0.0; wetness = 0.0; snowM = 0.0; return vec3(0.18); }
    float h = p.y;
    float above = wl > -1000.0 ? h - wl : 999.0;
    float ny = n.y;
    float fp = t * gPixAng;                                   // pixel footprint (m)
    // (round 3) the footprint ALONG an obliquely seen slope is fp / cos(view, normal): the fine debris / turf grain
    // filtered by the isotropic fp aliased into a brushed hatch on the cones seen at 50-70 deg (crop c5)
    float fpO = fp / clamp(abs(dot(n, normalize(p - uCamPos))), 0.25, 1.0);
    float n1 = fbm2(p.xz * 0.012 + 3.0, 4);
    float n2 = gnoise(p.xz * 0.11 + 7.0);
    float n3 = gnoise(p.xz * 0.45 + 1.0);
    float n2f = n2 * (1.0 - smoothstep(3.0, 9.0, fpO));      // footprint-filtered small scales
    float n2ff = n2 * (1.0 - smoothstep(2.0, 4.5, fpO));     // (for the masks: gone before it is < 2 px)
    float n3f = n3 * (1.0 - smoothstep(0.8, 2.5, fpO));
    float nb = fbm2(p.xz * 0.0021 + 11.0, 3);                 // ~500 m: snow / grass limits
    float nm = fbm2F(p.xz * 0.021 + 5.0, 3, 48.0, fp);       // ~50 m: outcrops, patch shapes
    float far = smoothstep(60.0, 160.0, t);                   // new macro masks only away from the camera
    // (t11) fine mosaic noise (9 / 4.5 / 2.2 m, footprint-faded) for the material edges: the soft 4 m-map masks
    // gave 10-30 m blurred margins - turf islands on the scree / rock read as pasted, smeared stickers at 1:1.
    // Added to a soft mask before a narrow threshold it turns the transition into a crisp interlocking mosaic
    // whose density follows the soft mask (grass in the joints, rock knobs in the turf); faded back to the
    // soft mask once the cells are < ~2-5 px (no sparkle)
    float dV = 1.0 - smoothstep(2.0, 5.0, fp);
    float nD = dV > 0.0 ? fbm2F(p.xz * (1.0 / 9.0) + 3.1, 3, 9.0, fpO) / 0.15 : 0.0;
    float nD2 = dV > 0.0 ? fbm2F(rot2(1.3) * p.xz * (1.0 / 7.0) + 8.7, 3, 7.0, fpO) / 0.15 : 0.0;

    // coarse geometry (>= 36 m, wider with the footprint) for everything that decides snow, scree and
    // turf far away: gully / rib curvature (ring minus a small centre ring, so the crag relief at the
    // pixel itself does not pick every micro-gully), gradient and normal from the same taps
    float curv = 0.0, curvB = 0.0, curvG = 0.0, curvX = 0.0, curvBX = 0.0;
    vec3 nC = n;
    float aoS = ao;                                           // sky visibility averaged over ~30 m
    if (far > 0.0 && h > 100.0) {
        float rc = max(64.0, 3.0 * fp);
        float tE = terrainH(p.xz + vec2(rc, 0.0), false), tW = terrainH(p.xz - vec2(rc, 0.0), false);
        float tN = terrainH(p.xz + vec2(0.0, rc), false), tS = terrainH(p.xz - vec2(0.0, rc), false);
        // centre: an 8-tap ring at 0.3 rc (session 6: the 4-tap 16 m ring still picked up the 4 m crag
        // relief, so the curvature - and the snow edges cut by it - jumped from pixel to pixel)
        float r4 = 0.3 * rc, r5 = 0.212 * rc;
        float hc = 0.125 * (terrainH(p.xz + vec2(r4, 0.0), false) + terrainH(p.xz - vec2(r4, 0.0), false)
                          + terrainH(p.xz + vec2(0.0, r4), false) + terrainH(p.xz - vec2(0.0, r4), false)
                          + terrainH(p.xz + vec2(r5, r5), false) + terrainH(p.xz - vec2(r5, r5), false)
                          + terrainH(p.xz + vec2(r5, -r5), false) + terrainH(p.xz - vec2(r5, -r5), false));
        curv = (tE + tW + tN + tS - 4.0 * hc) / rc;
        // the same at ~160 m (only the big couloir systems are concave at both scales)
        float rb = max(160.0, 6.0 * fp);
        curvB = (terrainH(p.xz + vec2(rb, 0.0), false) + terrainH(p.xz - vec2(rb, 0.0), false)
               + terrainH(p.xz + vec2(0.0, rb), false) + terrainH(p.xz - vec2(0.0, rb), false) - 4.0 * hc) / rb;
        // (session 4) the ~900 m spur-and-gully systems of the relief pass: curvature at ~400 m
        float rg = max(400.0, 10.0 * fp);
        curvG = (terrainH(p.xz + vec2(rg, 0.0), false) + terrainH(p.xz - vec2(rg, 0.0), false)
               + terrainH(p.xz + vec2(0.0, rg), false) + terrainH(p.xz - vec2(0.0, rg), false) - 4.0 * hc) / rg;
        // (session 4: 4 taps at 16 m -> 8 taps at >= 32 m: the snow edges cut by the hollow test traced the
        // 8 m AO texels - straight-edged, square chips at 1720 px)
        float r3 = max(32.0, 3.0 * fp), r6 = 0.7071 * r3;
        aoS = mix(ao, 0.125 * (terrainAO(p.xz + vec2(r3, 0.0)) + terrainAO(p.xz - vec2(r3, 0.0))
                             + terrainAO(p.xz + vec2(0.0, r3)) + terrainAO(p.xz - vec2(0.0, r3))
                             + terrainAO(p.xz + vec2(r6, r6)) + terrainAO(p.xz - vec2(r6, r6))
                             + terrainAO(p.xz + vec2(r6, -r6)) + terrainAO(p.xz - vec2(r6, -r6))), far);
        vec2 gC = vec2(tE - tW, tN - tS) / (2.0 * rc);
        nC = normalize(mix(n, normalize(vec3(-gC.x, 1.0, -gC.y)), far));
        // (world3) on the steep faces the couloir test uses the CROSS-slope curvature, averaged over three points
        // ~0.9 rc apart down the fall line: the x-z ring also sampled the face's own profile 2-4x further up /
        // down the slope, so the couloir fills broke into a series of flame-shaped flakes (debug 27, t8);
        // gullies are concave across, and the snow in them runs as continuous strips
        curvX = curv; curvBX = curvB;
        float sgC = length(gC);
        if (sgC > 0.3) {
            vec2 fl = gC / sgC, tg = vec2(-fl.y, fl.x);
            float cx = 0.0, cxB = 0.0;
            for (int k = -1; k <= 1; k++) {
                vec2 q = p.xz + fl * (float(k) * 0.9 * rc);
                float hq = terrainH(q, false);
                cx += (terrainH(q + tg * rc, false) + terrainH(q - tg * rc, false) - 2.0 * hq) / rc;
                cxB += (terrainH(q + tg * rb, false) + terrainH(q - tg * rb, false) - 2.0 * hq) / rb;
            }
            float wX = smoothstep(0.35, 0.8, sgC);
            curvX = mix(curv, 2.0 * cx / 3.0, wX); curvBX = mix(curvB, 2.0 * cxB / 3.0, wX);
        }
    }
    float nyC = nC.y;
    // turf / scree slope mask (fine detail up close). (round 3) far away it keeps part of the fine (footprint-
    // filtered) normal: the risers of the slab relief turn to rock, its treads hold turf - the turf / rock mosaic
    // follows the ground (it was a dither noise unrelated to the relief: camouflage, s7 lower left / s5)
    // (t19: 0.7 -> 0.5 + knob 4: the fine normal salted the 25-45 deg slopes with 2-10 m rock / turf speckle - camouflage)
    float nyV = mix(ny, mix(nyC, ny, 0.5), smoothstep(1.5, 4.0, fp));
    float gullyC = smoothstep(0.35, 1.1, curv) * far;
    float ribC = smoothstep(0.35, 1.1, -curv) * far;
    faceFrames(p, nC);
    // far away: granite base tone without rockAlbedo's axis-aligned streaks (they read as a diagonal
    // hatch on the oblique faces); the panel tones of the cliff joint sets stay
    // (session 4) the steep high walls hand over at 120-400 m instead of 300-900 m: rockAlbedo's warped
    // joint lines (6.5-71 m sets) drew a dark snakeskin network over the faces 0.3-2 km out (shot 0 left
    // face, shot 7) - at 3440 px it would cover every face within several km. The far granite has no
    // joint lines (the facet cracks below carry the structure); its base tone is a little warmer
    // (rock B-R +24..+30 in shots 0/1/7 against a target of +10..+25)
    float wallF = smoothstep(150.0, 250.0, h) * (1.0 - smoothstep(0.75, 0.85, nyC)) * far;
    float farA = mix(smoothstep(300.0, 900.0, t), smoothstep(120.0, 400.0, t), wallF);
    vec3 rock = farA < 1.0 ? rockAlbedo(p, n, t) : vec3(0.0);
    if (farA > 0.0) {
        rkCliffJoints(p, fp);                                 // (cached: rockAlbedo calls it too)
        // (session 6: the ~220 m tone mottling 0.32 -> 0.15 (soft camouflage patches); near grey granite, the warm
        // base only beyond ~2-4 km, where the haze turns it blue (near faces read beige, t22 shot 0 left face))
        vec3 rf = mix(vec3(0.218, 0.212, 0.203), vec3(0.226, 0.207, 0.186), smoothstep(1500.0, 4000.0, t))
                * (1.0 + (0.15 + W2K(4)) * fbm2(p.xz * 0.0045 + 3.0, 3)) * mix(0.92, 1.08, gRkJ.y);
        rock = mix(rock, rf * (1.0 + W2K(20)), farA);
    }
    // sub-heightfield relief of the faces (normal perturbation applied below) and its clefts
    float cav = 0.0;
    vec3 gb = vec3(0.0);
    // (world3) where the joint-block relief (world.glsl rockRelief) carries real geometry, the normal-only
    // substitutes (faceBump / faceFlutes / facet tilts / coarse-normal pull) fade out: their fall-line ribs
    // were the "melted wax" look
    float rkF = gRkWS;
    // (t23) where the footprint is < ~1-3 m the slab relief / micro-texture carry the walls: the fall-line ribs and
    // flutes (2-6 m octaves) drew dense vertical stripes at 1:1 final pixels on the walls without relief (s5 cirque)
    float flN = mix(0.3, 1.0, smoothstep(1.0, 3.0, fp));
    if (t > 70.0 && h > 150.0) gb = (0.5 + W2K(6)) * flN * faceBump(fp, cav);   // (session 4: 0.35 -> 0.25, "drip" ribs; session 6: 0.5 with the facets off)
    // (session 6) fall-line flutes: the organised sub-heightfield relief that replaces the random facet tilts
    float flA = (2.5 + W2K(7)) * flN;
    if (flA > 0.0 && t > 70.0 && h > 150.0) gb += flA * faceFlutes(fp);
    gb *= 1.0 - rkF;
    // (session 5) the creases of faceBump follow the zero contours of a gradient noise: meandering closed
    // loops - as dark lines they were the snakeskin network on the near walls at 1:1 final pixels (shot 0
    // left face, t16 crop c0). The dark lines now come from the facet joint sets only
    cav = 0.0;
    float crack = 0.0, ftone = 0.0;
    vec3 gF = vec3(0.0);
    if (t > 70.0 && h > 120.0) gF = faceFacets(fp, crack, ftone);
    // distant granite: mid-grey (lit #80878b), darker and streaked in the gullies
    // (wet rock, lichen, shadowed clefts), pale on the ribs and on fresh exfoliation slabs, banded by
    // sheet joints; keeps the faces from reading as pale plaster once the close-range detail is gone
    float farR = smoothstep(120.0, 800.0, t);
    // (session 4) the 900 m gully systems: dark (wet rock, lichen, dwarf pine below ~1100 m), so they
    // still read in sky-only shade (shot 6's left massif)
    // (band pass: the 160 m curvature minus 0.4 x the 400 m one - the 400 m term alone is dominated by
    // the valleys / slope feet (+-1.1 at p10/p90 on the faces), wt/curvmap.py; on slopes only)
    float gulG = smoothstep(0.35, 0.95, curvB - 0.4 * curvG) * (1.0 - smoothstep(0.82, 0.92, nyC)) * far;
    // the detail of the far granite (gullies, stains, facet cracks) takes over with the far tone, on the
    // walls already from ~100-300 m; the brightness calibration (x0.55) keeps the old 120-800 m ramp
    float wD = max(farR, wallF * smoothstep(100.0, 300.0, t));
    if (wD > 0.0) {
        // (session 4: gully darkening 0.66 -> 0.5: the faces lacked their darkest tones, p10 0.14 vs 0.05)
        float gul = max(smoothstep(0.76, 0.52, aoS + 0.05 * nm / 0.12), 0.8 * gullyC);
        // water / lichen stains hanging down the faces (sharp-edged dark stripes)
        float fsv = 1.0 - smoothstep(8.0, 25.0, fp);
        float fstreak = faceNoise(vec2(0.030, 0.012), 7.7)
                      + 0.45 * (1.0 - smoothstep(4.0, 10.0, fp)) * faceNoise(vec2(0.07, 0.025), 3.3);
        float fine = (1.0 - smoothstep(2.0, 5.0, fp)) * faceNoise(vec2(0.12, 0.025), 1.9);
        // (edge width follows the footprint: a fixed 0.04 edge aliased into crawling dashes at 1-3 km)
        float sew = 0.02 + 0.08 * smoothstep(2.0, 10.0, fp);
        float stain = smoothstep(0.32 - sew, 0.32 + sew, fstreak + 0.10 * fine) * fsv;
        stain *= 1.0 - 0.85 * rkF;               // (world3) the relief carries the structure
        // pale exfoliation slabs (fresh granite) on the open faces, only on the ribs / convex ground
        // (at x1.55 everywhere they were the pale blotches of the camouflage look)
        float slab = smoothstep(-0.08, 0.16, faceNoise(vec2(0.016, 0.011), 9.1) + 0.04 * nm / 0.12)
                   * (1.0 - smoothstep(25.0, 60.0, fp)) * (1.0 - gul) * max(ribC, smoothstep(0.0, 0.3, -curv))
                   * (1.0 - 0.6 * rkF);   // (t12: soft pale vertical smears on the relief walls)
        // sheet joints: dark ledge lines ~18 m apart across the steep faces
        float jw = (1.0 - smoothstep(1.8, 4.5, fp)) * (1.0 - smoothstep(0.55, 0.8, nyC));
        float joint = 0.0;
        if (jw > 0.0) {
            // (session 4) a regular line every 18 m over the whole face read as a ruled grid at the final
            // 3440 px (shot 0 left face): ~1 band in 3 is an open joint, 23 m base spacing, the lines ~0.7 m
            // wide (antialiased by the footprint, not 1-2 m bands)
            float jv = h / 23.0 + 0.6 * faceNoise(vec2(0.010, 0.004), 4.4);
            float jk = step(0.64, hash11(floor(jv) * 0.7548 + 0.113));
            float jd = abs(fract(jv) - 0.5) * 23.0;
            joint = jk * (1.0 - smoothstep(0.35, 0.35 + 1.2 * fp, jd)) * saturate(0.7 / (0.7 + 1.2 * fp))
                  * smoothstep(-0.15, 0.1, faceNoise(vec2(0.03, 0.02), 2.6)) * 1.6;
        }
        // (no more blotchy fbm mottling: the facets and their cracks carry the mid-scale structure)
        // (session 3: ribs x1.4 -> x1.5, fresh slabs x1.2 -> x1.35: the lit faces should reach luminance
        // 0.69-0.72 at p90 of the rock against 0.58-0.64 in the renders, at the same median)
        // (session 4: stains 0.78 -> 0.68, facet cracks 0.35 -> 0.5 (fewer of them: joint sets only), sheet
        // joints 0.7 -> 0.6 (sparse): the faces' darkest tones are crisp dark lines / streaks, not blotches)
        // (t19) on the slab-relief walls the gullies are real geometry now: their albedo darkening 0.68 -> 0.80 (dark
        // vertical smears = the "drips" at frame level, s7 / s1)
        float tone = mix(1.0, mix(0.68, 0.80, rkF) + W2K(2), gul) * (1.0 + 0.5 * ribC) * mix(1.0, 0.78 + W2K(23), stain) * (1.0 + 0.04 * fstreak * fsv)
                   * (0.90 + 0.20 * smoothstep(-0.15, 0.15, nb)) * (1.0 + (0.35 + W2K(5)) * slab) * mix(1.0, 0.6, saturate(joint) * jw * (0.3 + W2K(25))) * (1.0 - 0.3 * saturate(cav));
        tone *= (1.0 - (0.45 + W2K(13)) * crack) * (1.0 + W2K(14) * ftone) * 1.02 * mix(1.0, 0.78 + W2K(3), gulG);   // (session 5: 0.5 / 0.72 -> 0.62 / 0.62)
        // (session 6: AO gullies 0.44 -> 0.68, big gullies 0.62 -> 0.78, stains 0.68 -> 0.78, cracks 0.62 -> 0.45, cell
        // tone off, sheet joints x0.5: soft dark albedo blotches read as camouflage; t22b isolated each term)
        // (session 5) fine joint sets on the steep faces (not on benches / gentle ground)
        // (t20: only the steep walls (> ~45 deg) seen from > ~1 km: on the 40 deg knobs and the
        // near walls (shot 0 left) the lines read as a net drawn on a smooth surface)
        // (session 6: from 1.5-3 km - on the 1.4 km face of shot 0 they read as a scribbled net at 1:1 final pixels)
        float jW = (1.0 - smoothstep(0.62, 0.80, nyC)) * smoothstep(1500.0, 3000.0, t);
        if (jW > 0.0 && t > 70.0) tone *= 1.0 - jW * (0.6 + W2K(12)) * graniteJoints(p.xz, fp);
        // (session 6: 0.70 -> 0.55: lit rock / zenith sky was 0.7-0.95 against a target of 0.62)
        rock *= mix(1.0, tone, wD) * mix(1.0, 0.78, farR);   // (t12: 0.65 -> 0.72: rock medians 10-15 % below target, high-key granite)   // (session 6: 0.55 -> 0.65: lit-rock p90 0.33-0.36 vs a target of 0.39-0.47)
        // 8-16 km: lit faces darker (the far ranges should read nearly flat blue with white fields on them)
        rock *= mix(1.0, 0.72, smoothstep(8000.0, 20000.0, t));
    }
    // (t14) weathering mosaic of the granite walls (at 1-3 km: lichen-darkened slabs, fresh pale
    // scars, at 6-60 m): volume noise in (x, y, z), isotropic on any slope (no fall-line smear), faded by the
    // footprint; on the relief walls / steep faces only
    float lmW = max(rkF, wallF) * (1.0 - smoothstep(3.0, 8.0, fp));
    if (lmW > 0.0) {
        vec4 lz = texture(uNoise3, p * (1.0 / 256.0) + 0.53);
        float lich = smoothstep(0.56, 0.70, lz.b + 0.4 * (lz.r - 0.475));
        float fresh = smoothstep(0.64, 0.78, texture(uNoise3, p * (1.0 / 150.0) + 0.29).a);
        rock *= mix(vec3(1.0), mix(vec3(1.0), mix(vec3(1.0), vec3(0.74, 0.76, 0.70), 0.6), lich) * (1.0 + 0.12 * fresh * (1.0 - lich)), lmW);
    }
    // (t21) granite micro-texture where the footprint resolves 1-8 m: at 1:1 final pixels the 0.5-1.5 km walls and
    // knobs (s5 cirque, s0 left wall: footprint 0.1-0.3 m) were smooth plaster - the walls should show weathering tone
    // patches, dark lichen / moss blotches and fine grain at those scales. 3D noise (isotropic on any slope), each
    // octave faded before it is < ~5 px of the oblique footprint (no sparkle)
    // (t21 variants: x2 gives the s0 near wall its weathering blotches, x1 barely shows -> 1.6)
    float mvA = (1.0 - smoothstep(0.6, 1.6, fpO)) * far * 1.6;
    if (mvA > 0.0) {
        float m8 = clamp((texture(uNoise3, p * (1.0 / 8.0) + 0.11).r - 0.48) / 0.1, -2.0, 2.0);
        float m3 = clamp((texture(uNoise3, p * (1.0 / 3.0) + 0.47).b - 0.52) / 0.09, -2.0, 2.0) * (1.0 - smoothstep(0.25, 0.6, fpO));
        float lk = smoothstep(0.63, 0.71, texture(uNoise3, p * (1.0 / 5.0) + 0.71).a);
        rock *= max(1.0 + mvA * (0.07 * m8 + 0.06 * m3 - 0.3 * lk), 0.3);
    }
    // (world5 R1) pixel-scale granite grain of the faces at 0.15-9 km: the walls carry
    // crisp tone structure down to the pixel - lichen-darkened and fresh pale slabs, dark wet / lichen streaks hanging
    // down the faces, crack shadows; the renders' faces were smooth plaster below the ~40 m relief (DoG band contrast
    // 0.11-0.14 log2 at 1-8 px against a target of 0.19-0.22). Octaves of the smooth single-octave value noise
    // (uNoise3.g, features ~wl), 1.8x taller than wide (fall-line streaks), kept to the 2.5-40 px band of the oblique
    // footprint at any distance (sw1: the 30-60 px octaves only added soft camouflage blotches) and faded in / out
    // with the footprint (no shimmer in the moving film); sparse crisp dark streak patches (>= 4-8 px)
    float grW = far * (1.0 - smoothstep(9000.0, 14000.0, t)) * max(max(rkF, wallF), 1.0 - smoothstep(0.62, 0.85, nyC));
    if (grW > 0.0) {
        float lg = 0.0, dkS = 0.0, wlG = 64.0;
        for (int i = 0; i < 7; i++) {
            if (wlG < 3.0 * fpO) break;
            // (shimmer test 17:17, shot 7 temporal / spatial 1.54 / 1.39 at 2.5 px: 3-6 px)
            float v = smoothstep(3.0 * fpO, 6.0 * fpO, wlG) * (1.0 - smoothstep(14.0 * fpO, 40.0 * fpO, wlG));
            vec3 q = vec3(p.x, 0.55 * p.y, p.z) * (1.0 / (16.0 * wlG)) + vec3(0.137, 0.291, 0.419) * float(i + 1);
            if (v > 0.0) lg += v * (texture(uNoise3, q).g - 0.487) / 0.153;
            float vS = smoothstep(5.0 * fpO, 10.0 * fpO, wlG) * (1.0 - smoothstep(24.0 * fpO, 60.0 * fpO, wlG));
            if (vS > 0.0) dkS = max(dkS, vS * smoothstep(0.71, 0.74, texture(uNoise3, q * vec3(1.7, 0.6, 1.7) + 0.31).g));
            wlG *= 0.5;
        }
        rock *= exp2(grW * ((0.30 + W2K(27)) * lg - (0.80 + W2K(28)) * dkS));   // (sw2 / sw4: 0.30 / 0.80)
    }
    // ---- scree / talus: pale, nearly neutral granite debris (#a4a7a3), lighter than the cliffs above,
    // with debris streaks down the fall line and a boulder grain at mid range
    float sF = faceNoise(vec2(0.02, 0.0045), 5.3) * (1.0 - smoothstep(8.0, 20.0, fp));   // fan-scale streaks
    // (round 3) isotropic debris tone: the face-coordinate streaks (2-4:1 down the fall line) read as brushed
    // metal on the cones at 1:1 (hero cone, crop c5)
    float sT = saturate(0.5 + 0.55 * n3f + 0.35 * n2f + 0.25 * nm);
    // (t19: far 0.48 -> 0.58 + knob 2: the fans are the brightest ground after the snow, #a4a7a3 vs lit granite
    // #80878b; the 0.48 fans read as lavender-grey smears level with the rock)
    vec3 scree = mix(vec3(0.255, 0.258, 0.250), vec3(0.42, 0.425, 0.41), sT) * mix(1.0, 0.64, farR);   // (t23: 0.58 -> 0.64)
    float sOld = 0.0, sShrub = 0.0;
    vec3 scrB = vec3(0.0);
    // (t22, perf) the debris texture only where the talus map has debris (6-8 Voronoi evaluations per pixel)
    if (f.b > 0.02) {
    // block-size sorting / shadowed gaps: mottled at 15-60 m
    float sM = fbm2F(rot2(0.7) * p.xz * (1.0 / 22.0) + 8.2, 3, 22.0, fp);
    float sFi = fbm2F(p.xz * (1.0 / 50.0) + 5.3, 2, 50.0, fp) / 0.12 * 0.2;
    scree *= mix(1.0, (0.85 + 0.4 * sFi) * (1.0 + 0.5 * sM), far);
    // (session 6) 4-12 m block / gap grain at 0.3-3 km: the fans read as flat pale paint without it
    // (session 4: the value-noise grain read as smooth pale concrete (the hero cone is a coarse block field):
    // ~4.5 m Voronoi blocks, per-block tone +-20 %, dark gaps x0.6 (gap width >= ~1.2 px), faded once a
    // block is < ~2.5 px; beyond that 15-30 m block clusters +-12 % (sorted patches, shadowed hollows))
    // (t22) the block grain at EVERY distance: the octave of Voronoi blocks (4.5, 9, 18, 36 m) whose cells are 2.4-4.8
    // px of the oblique footprint, cross-faded between two octaves. The 4.5 m grain alone faded out beyond fpO ~1.9 m,
    // so the fans of shot 7 (2-3 km, 1.3 m final pixels, fpO 2-4 m) were smooth plaster bands; scree fans at 1-3
    // km are a salt-and-pepper of lit blocks and shadowed gaps down to the pixel. Coarser octaves = block clusters:
    // lower contrast
    if (far > 0.0 && fpO < 60.0) {
        float lc = log2(max(2.4 * fpO / 4.5, 1.0));
        float o0 = floor(lc), fo = lc - o0;
        float grain = 0.0;
        for (int k = 0; k < 2; k++) {
            float ok = o0 + float(k);
            float S = 4.5 * exp2(ok);
            vec3 vb = voronoi2(p.xz * (1.0 / S) + 7.0 + 3.1 * ok);
            // (round 3: +-20 % / gaps x0.6 -> +-26 % / x0.45, wider gaps: the cone read as smooth grey clay at 1 km, it should
            // be a coarse salt-and-pepper block field, in-box std ~20)
            float ewB = max(0.10, 1.2 * fpO / S);
            float gapB = (1.0 - smoothstep(0.0, ewB, vb.y - vb.x)) * (0.10 / ewB);   // (wider gap, weaker: same mean)
            float ct = mix(1.3, 0.7, saturate(ok / 3.0));   // (t22 variants: x1.6 a little crunchy, x0 = smooth paint)
            grain += (k == 0 ? 1.0 - fo : fo) * (1.0 + ct * ((1.0 + 0.52 * (vb.z - 0.5)) * (1.0 - 0.55 * gapB) - 1.0));
        }
        scree *= mix(1.0, grain, far * (1.0 - smoothstep(30.0, 60.0, fpO)));
    }
    float cwM = smoothstep(0.8, 2.0, fpO) * (1.0 - smoothstep(8.0, 16.0, fpO));
    if (cwM > 0.0) {
        vec3 vc = voronoi2(p.xz * (1.0 / 22.0) + 3.0);
        float mc = 0.5 * (2.0 * vc.z - 1.0) + 0.5 * clamp(gnoise(p.xz * 0.05 + 2.0) / 0.35, -1.0, 1.0);
        scree *= 1.0 + 0.12 * cwM * mc;
    }
    // (session 6) the 0.5-3 km fans read as pale paint smears (t19 shot 7): crisp dark debris
    // furrows / channel shadows down the fall line, ~16 m and finer, drawn like the joint lines (>= ~1 px)
    float dsw = 0.25 * far * smoothstep(1.0, 2.5, fp) * (1.0 - smoothstep(14.0, 28.0, fp));   // (t13: 1 -> 0.5, brushed lines at 1:1)
    if (dsw > 0.0) {
        float fur = 0.0;
        if (gFwa > 0.04) fur += gFwa * jointSet(gFa, vec2(0.0, 1.0), 36.0, fp, 41.0);
        if (gFwb > 0.04) fur += gFwb * jointSet(gFb, vec2(0.0, 1.0), 36.0, fp, 47.0);
        scree *= 1.0 - dsw * fur;
    }
    float bw = 1.0 - smoothstep(0.25, 1.2, fpO);
    if (bw > 0.0) {
        vec4 bv = voronoiV(p.xz * 0.9 + 13.0);
        float gap = smoothstep(0.25, 0.5, bv.z);
        scree *= mix(1.0, mix(1.08, 0.62, gap) * mix(0.85, 1.12, bv.w), bw);
    }
    // (t11) the hero cone at 1:1: not an even grey asphalt grain but sorted debris - ~9 m boulders (to ~3.8 m
    // footprint), lobes of older lichen-grey / olive debris between fresh pale tongues (30-80 m), and the
    // older lobes colonised by dwarf shrubs / grass tufts (dark dots, cells > ~2.5 px)
    if (far > 0.0) {
        // (t13) the blocks as lit / shaded faces (dome normals, applied with the other normal terms below): flat
        // tone cells read as asphalt; ~9 m blocks on ~45 % of the cells (to ~2.5 px), 3.5 m blocks (to ~2 px)
        float bw9 = 1.0 - smoothstep(2.2, 4.5, fpO);
        if (bw9 > 0.0) {
            vec4 v9 = voronoiV(p.xz * (1.0 / 9.0) + 2.0);
            float big = step(0.55, v9.w);                       // ~45 % of the cells hold a big block
            scree *= mix(1.0, mix(1.0, 0.86 + 0.32 * fract(v9.w * 7.3), big), bw9 * far);
            scrB += bw9 * far * big * 0.75 * smoothstep(0.12, 0.6, v9.z) * vec3(v9.x, 0.0, v9.y) / max(v9.z, 1e-3);
        }
        float bw3 = 1.0 - smoothstep(0.9, 1.9, fpO);
        if (bw3 > 0.0) {
            vec4 v3 = voronoiV(rot2(0.5) * p.xz * (1.0 / 3.5) + 41.0);
            vec2 o3 = rot2(-0.5) * (v3.xy / max(v3.z, 1e-3));
            scrB += bw3 * far * 0.85 * smoothstep(0.1, 0.62, v3.z) * vec3(o3.x, 0.0, o3.y);
        }
        sOld = smoothstep(-0.05, 0.30, fbm2F(p.xz * 0.016 + 27.0, 3, 62.0, fp) / 0.12 * 0.25 + 0.35 * (0.62 - f.b) / 0.3);
        scree *= mix(vec3(1.0), vec3(0.76, 0.78, 0.72), sOld * far);
        float shV = 1.0 - smoothstep(1.4, 2.8, fpO);
        if (shV > 0.0) {
            vec3 vs = voronoi2(p.xz * (1.0 / 4.0) + 17.0);
            float rS = 0.28 + 0.16 * fract(vs.z * 5.1);
            float clump = smoothstep(-0.05, 0.30, gnoise(p.xz * 0.035 + 41.0));   // (t12: ~30 m clumps, not an even polka-dot grid)
            sShrub = step(fract(vs.z * 11.3), 0.45 * sOld * clump * smoothstep(250.0, 650.0, 1300.0 - h))
                   * (1.0 - smoothstep(rS - 0.4 * fp / 4.0, rS + 0.4 * fp / 4.0, vs.x)) * shV * far;
        }
    }
    }   // (f.b > 0.02)
    // cool, never clipping (#ced4dc lit); a little brighter on the far ranges, where the haze would
    // otherwise wash the snowfields into the blue rock (crisp white fields at 10-20 km)
    // (r4 M4: neutral albedo - the sky ambient alone gives the cool cast (lit snow was #c1cfe8, B-R +35; target +2..+14);
    // no far boost: the far ranges read as snow-mantled hero peaks)
    vec3 snow = vec3(0.70, 0.70, 0.69);
    snow *= 1.0 + 0.06 * clamp(fbm2F(p.xz * 0.025 + 41.0, 2, 40.0, fp) / 0.12, -1.5, 1.5);   // +-6 % (40 m)
    vec3 pebble = mix(vec3(0.19, 0.187, 0.18), vec3(0.27, 0.265, 0.255), saturate(0.5 + 0.8 * n3));

    // ---- vegetation: the original slope mask near the camera (shared with the 3D cover) ...
    float vegOld = smoothstep(0.60, 0.82, ny + 0.10 * n2) * (1.0 - smoothstep(950.0, 1250.0, h + 160.0 * n1));
    // ... and away from it the alpine mosaic: steeper ground turns to rock with altitude, granite
    // knobs break the grass above the forest, the highest grass ledges end ~1000-1300 m
    float hv = h + 150.0 * nb + 50.0 * nm;
    // (world3) the turf / rock / dwarf-shrub belt ~150 m higher (green slopes with rock knobs up to the
    // foot of the walls; the 600-1000 m slopes of shot 5 read as a pale grey wash with dark stickers)
    float hvV = hv - 150.0;
    // (session 6: 0.64..0.80 left the 45-deg slopes right above the forest bare grey; there should be a
    // belt of turf, dwarf shrubs and scree between the forest and the walls)
    // (session 6: 0.54 @450 -> 0.50 @550 m: the 35-45 deg slopes around the tarn read as bare pale clay)
    float ty = mix(0.50, 0.80, smoothstep(550.0, 1150.0, hvV)) + 0.1 * W2K(22);
    float nT = fbm2F(p.xz * 0.0067 + 5.0, 3, 150.0, fp);     // ~150 m turf / rock mosaic
    // (session 4: edge 0.16 -> 0.28 - bright olive "stickers" with hard edges on 35-45 deg rock; above
    // ~900 m turf only on the benches)
    // (round 3) ledge bands: turf on strata ledges running along the slope (strike), ~10-15 m thick and 50-80 m long,
    // dipping with the region; rock on the ribs, turf in the gullies - instead of isotropic
    // speckle
    float ledge = 0.0;
    if (far > 0.0 && dV > 0.0) {
        float dip = 0.35 * gnoise(p.xz * 0.0013 + 91.0) / 0.25;
        mat2 rd2 = rot2(dip);
        float v6 = 1.0 - smoothstep(1.2, 2.4, fpO);
        for (int k = 0; k < 2; k++) {
            float w = k == 0 ? gFwa : gFwb;
            if (w < 0.04) continue;
            vec2 F = rd2 * (k == 0 ? gFa : gFb);
            ledge += w * (0.65 * gnoise(F * vec2(1.0 / 70.0, 1.0 / 14.0) + vec2(3.3, 7.1 * float(k)))
                        + 0.35 * v6 * gnoise(F * vec2(1.0 / 30.0, 1.0 / 6.0) + vec2(9.7, 1.3 + 5.0 * float(k))));
        }
        ledge *= dV / 0.22;
    }
    float vegIn = nyV + 0.06 * n2ff / 0.22 + 0.03 * n3f / 0.22 + 0.08 * nT / 0.12
                + 0.12 * ledge + 0.08 * (gullyC - ribC);   // (t19: ledges 0.07 -> 0.12, ribs 0.05 -> 0.08: the slopes = turf in strata bands, rock on the ribs)
    float vegN = smoothstep(ty, ty + 0.28, vegIn) * (1.0 - smoothstep(950.0, 1350.0, hvV));
    vegN *= mix(1.0, smoothstep(0.62, 0.78, nyC), smoothstep(850.0, 1000.0, hvV) * far);   // (t16: 650-800 m stripped the slopes)
    // (t11) crisp turf / rock mosaic instead of a blurred margin (see nD)
    float kD = 0.12;   // (round 3: 0.30 -> 0.12, the geometry decides - see nyV)
    vegN = mix(vegN, smoothstep(0.42, 0.58, vegN + kD * nD * smoothstep(0.0, 0.2, vegN)), dV * far);
    float outc = smoothstep(0.02, 0.22, nm + 0.10 * n2ff + 0.25 * (ao - 0.85)) * smoothstep(380.0, 620.0, hvV) * (1.0 - f.a);
    // granite knobs and slabs breaking through the open stands of the upper forest: the
    // same noises / thresholds as the outcrops that bake_forest clears of trees
    float slopeC = sqrt(max(1.0 - nyC * nyC, 0.0)) / max(nyC, 0.05);
    float nmO = fbm2F(p.xz * 0.042 + 5.0, 3, 24.0, fp);       // ~25 m (bake_forest: same noise)
    float oS = 0.55 * nmO / 0.12 + 0.45 * nT / 0.12 + 0.8 * ribC + 1.5 * (slopeC - 0.85);
    float openA = smoothstep(90.0, 280.0, h);
    // (only the bare knobs: the gaps the bake opens between the trees are mostly grass and shrubs, and
    // the open belt higher up keeps its turf / rock mosaic below)
    // (r4 M11: = bake_forest outR: threshold -0.3, up to ~700 m)
    float outLow = smoothstep(1.2 - 1.2 * openA, 1.7 - 1.2 * openA, oS) * smoothstep(40.0, 100.0, h)
                 * (1.0 - smoothstep(600.0, 750.0, h)) * (1.0 - f.a) * far;
    outc = max(outc, outLow);
    vegN *= 1.0 - 0.9 * outc;
    vegN = max(vegN, saturate(f.r * 2.5) * smoothstep(0.42, 0.6, nyV));     // forest floor holds on steep ground
    // scree: soft map value up close (shared with the 3D cover), far away a hard, interlocking edge
    // broken by the fan streaks
    // (t19) crisp, lobed fan margins: the 0.40-0.50 band of the 4 m map's soft debris value was a 10-20 m wide
    // grey fringe (the fans read as soft smears at frame level, s7 lower left); a 0.04 band + 12-35 m lobes /
    // tongues (debris tongues and turf tongues interlock with sharp edges)
    float ew2 = 0.025 + 0.08 * smoothstep(3.0, 16.0, fp);
    float sLob = 0.07 * fbm2F(rot2(0.4) * p.xz * (1.0 / 35.0) + 2.9, 2, 35.0, fpO) / 0.12;
    // (t21) the steepness limits of the debris (the bake keeps it to <= ~38 deg on the 40 m slope; safety net: no pale
    // sheets on > ~45 deg faces on the 64 m slope, none on the summit zone) are part of the threshold argument
    // instead of multiplying the mask: multiplied, their 50-100 m soft ramps left half-transparent debris over the
    // rock / turf - the "grey smears" of s7's lower slopes (t19 scD variant: they dim with the scree albedo)
    // (t24) ... as a CRISP gate (1-2 % of ny, lobed by the fan noise), not an offset of the debris value: with the
    // debris map at 1.0 on the cirque walls an offset of -0.06 still passed, and the x-z debris grain on 50-70 deg walls
    // hung down as pixel-column stripes (s5 crop c5, t21-t23)
    float sArg = f.b + 0.25 * sF + sLob * smoothstep(0.15, 0.3, f.b);
    float scrN = smoothstep(0.43 - ew2, 0.43 + ew2, sArg);
    scrN = mix(scrN, smoothstep(0.43 - ew2, 0.43 + ew2, sArg + 0.35 * kD * nD2 * smoothstep(0.1, 0.3, f.b)), dV);   // (t11) interlocking fan margins
    float sGw = 0.012 + 0.02 * smoothstep(3.0, 16.0, fp);
    float sGate = smoothstep(-sGw, sGw, min(nyC - 0.615 + 0.5 * sLob, 0.6 * (nyV - 0.40) + 0.3 * sLob))
                * (1.0 - smoothstep(2000.0, 2400.0, h));
    float scr = mix(saturate(f.b * 1.3) * smoothstep(0.30, 0.50, nyV), scrN * sGate, far);
    float veg = mix(vegOld * (1.0 - f.b * 0.9), vegN * (1.0 - scr), far);
    // (session 4) the hero talus fan (bake.HERO_CONES) is bare blocks: no turf on it
    float heroFan = (1.0 - smoothstep(460.0, 640.0, length(p.xz - vec2(-2680.0, -320.0)))) * smoothstep(0.1, 0.35, f.b) * far;
    veg *= 1.0 - heroFan;
    // the outcrops in the forest band are pale exfoliated slabs, lighter than the cliffs above
    rock *= 1.0 + 0.15 * outLow * (1.0 - veg) * (1.0 - smoothstep(450.0, 800.0, h));

    // ---- late-summer snow: sharp-edged patches in couloirs and gullies and on benches/ledges, more
    // and larger with altitude, fields on the gentler ground near the summits; steep faces bare.
    // Expressed as an effective altitude: hollows, ledges and shady aspects hold snow lower down.
    // All terms use the coarse geometry and >= ~50 m noise: clustered patches, no flecks.
    float north = saturate(nC.z * 1.1 + 0.45);                // shaded aspects keep their snow
    // concave, sky-occluded couloirs: AO well below what is typical for that altitude
    float aoExp = mix(0.80, 0.58, smoothstep(0.0, 2400.0, h));
    float hollow = smoothstep(aoExp - 0.02, aoExp - 0.20, aoS);
    // the ranges 8-16 km out are the high, glaciated ones (snowfields on every far
    // summit): their snow limits sit lower (position-based, so it cannot change with the view)
    float rng = smoothstep(7000.0, 13000.0, length(p.xz));
    float rngB = smoothstep(0.4, 0.7, rng);
    // > ~65 deg (on the 64 m slope) sheds everything; on the far glaciated ranges up to ~75 deg (hanging
    // snowfields / ice on the faces)
    // (session 4: near walls 0.30..0.48 -> 0.40..0.56 (bare above ~57 deg): white paint on 60 deg faces read pasted)
    float shed = smoothstep(mix(0.40, 0.18, rngB), mix(0.56, 0.36, rngB), nyC + 0.05 * nm / 0.12);
    // ledges / gentler ground (< ~40 deg; on the far ranges, where only the 64 m+ shape survives the
    // haze, up to ~55 deg: their snowfields sit on every shoulder)
    // (r4 M3: near benches from ~37 deg (0.72 / 0.88 -> 0.62 / 0.80): late-summer snow lies in horizontal bands / bowl fields)
    float bench = smoothstep(mix(0.62, 0.55, rngB), mix(0.80, 0.75, rngB), nyC + 0.06 * nm / 0.12);
    float nS = fbm2(p.xz * 0.0009 + 17.0, 2) / 0.12;             // ~1 km: snowy vs bare faces
    float nL = fbm2(p.xz * 0.005 + 2.0, 2) / 0.12;               // ~200 m: fewer, larger patch clusters
    // ragged, lobed snow limits: ~25 m noise on the effective height and on the couloir test (a
    // domain-style perturbation of the inputs: ragged edges without the isolated flecks a noise added
    // to the final mask makes); filtered by the footprint
    float nE = fbm2F(p.xz * 0.04 + 7.0, 2, 25.0, fp) / 0.12;
    // (world3) the same edge noise in 3D (x, y, z): on the 60-75 deg walls the x-z noise hung down the face as
    // flames; a volume noise is isotropic on any slope (fbm channel, ~40 m with a ~130 m fallback far away)
    {
        float nA = (texture(uNoise3, p * (1.0 / 320.0) + 0.21).r - 0.5) / 0.1;
        float nB = (texture(uNoise3, p * (1.0 / 1000.0) + 0.67).r - 0.5) / 0.1;
        nE = mix(nE, mix(nA, nB, smoothstep(2.0, 4.0, fp)), saturate(0.6 * (1.0 - nyC) + 0.6) * far);
    }
    // (session 3: 25 m term 55 -> 30 m: the field edges were zig-zag dashes)
    float hN = h + 300.0 * nb + 120.0 * nm + 45.0 * nL + 20.0 * nE + 100.0 * W2K(19);   // patchy snow limits (t20: nE 30 -> 20)
    // (7-11 km: limits ~1100 m lower; the 11+ km ranges carry fields on their upper half only - with the
    // full offset they were salted with snow from 800 m up, which read as a pale comb)
    hN += (1100.0 * rng - 800.0 * rngB) * smoothstep(450.0, 1000.0, h);
    float fL = 700.0 * rngB;                                     // couloir / bench limits on the 11+ km ranges
    // couloirs: concave at both 64 m and 160 m (session 6: the 64 m test alone salted every gully
    // with 2-6 px dashes); about half of the couloir systems hold snow, the others are bare
    float cBig = min(curvX, 1.4 * curvBX) + 0.06 * nE;   // (t20: 0.10 -> 0.06: flame-shaped couloir fills)
    // (round 2: wider couloir fills, about two thirds of the couloir systems hold snow)
    // (round 3) couloir snow is avalanche deposit: a narrow strip on the steep gully floor, widening where the floor
    // flattens toward its foot (the cone), lasting lower than the open-slope snow. The old altitude ramp narrowed every
    // strip downward into a flame (t17 debug 27: the couloir term = the flames of shot 7)
    float cW = 0.30 * (nyC - 0.45);
    float sC = smoothstep(0.18, 0.60, cBig + cW) * far
             * smoothstep(480.0 + fL, 800.0 + fL, hN + 150.0 * (north - 0.5) + 150.0 * smoothstep(0.40, 0.65, nyC));
    // (session 3: on the near walls only ~1/3 of the couloir systems hold snow - a dash in every gully
    // read as a pattern from the lake, shots 0/1; the far ranges keep the old gate)
    sC *= smoothstep(mix(-0.45, -0.8, rngB), mix(0.35, 0.1, rngB), 0.4 * nL + 0.8 * nS);   // (t14: ~45 %)
    // (r4 M3) on the near ranges the couloir strip holds mainly where the gully floor flattens (its cone / foot): the
    // continuous strips down the steep faces were the paint drips / flames (blob h/w 1.1-1.9, target 0.67-0.87)
    sC *= mix(mix(0.25, 1.0, smoothstep(0.55, 0.75, nyC)), 1.0, rngB);   // (b2 shot 7: blob h/w still 1.3 at 0.4 / 0.72)
    float sBn = bench * mix(0.6, 1.0, max(smoothstep(0.05, 0.5, cBig), hollow))     // ledges, aprons at the
              * smoothstep(mix(800.0, 680.0, hollow) + fL, mix(1080.0, 960.0, hollow) + fL, hN + 250.0 * (north - 0.5)); // foot of the faces
    float sHi = smoothstep(mix(1650.0, 1800.0, rngB), mix(2250.0, 2300.0, rngB), hN + 300.0 * (north - 0.5) + 250.0 * hollow)
              * mix(mix(0.75, 1.0, bench), mix(0.15, 1.0, bench) * smoothstep(-0.7, 0.5, nS + 0.6 * nL), rngB);                                                  // summit fields
    // (far ranges: fields on the gentler shoulders and in ~1 km patches only, steep upper faces bare -
    // with the full summit term every crest above the limit was white)
    // (edge noise 0.07 -> 0.04 and faded out from 4 px: below that it only made flecks)
    // (round 2) a few LARGE fields instead of a fleck in every gully (snow lies in the high
    // hollows below the faces, on the aprons at their foot and on the upper shoulders; 3-6 % of a high
    // view against ~1-2 %): concave at the 160 m scale, < ~50 deg, from ~1150-1400 m (lower on north
    // aspects), in ~1 km clusters
    float sHo = smoothstep(0.10, 0.45, 1.4 * curvB + 0.06 * nE) * smoothstep(0.48, 0.64, nyC + 0.04 * nm / 0.12)   // (r4 M3: 0.55 / 0.70)
              * smoothstep(880.0 + fL, 1180.0 + fL, hN + 260.0 * (north - 0.5) + 120.0 * hollow) * far   // (session 5: -70 m)
              * smoothstep(-0.6, 0.25, 0.6 * nL + 0.7 * nS);
    // upper north / east shoulders of the near walls: open faces up to ~50 deg hold snowfields above
    // ~1500-1800 m
    float sSh = smoothstep(0.50, 0.64, nyC + 0.05 * nm / 0.12) * smoothstep(1350.0 + fL, 1650.0 + fL, hN + 350.0 * (north - 0.5))
              * smoothstep(-0.2, 0.5, nS + 0.5 * nL) * far * (1.0 - rngB);
    // (the hollow / shoulder fields broken into tongues hanging down the fall line - round blobs read as
    // cut-outs; face-coordinate streaks ~100 m wide, ~300 m long)
    // (session 4: 2:1 instead of 3:1 and weaker - long white drips read as paint runs)
    // (r4 M3: the tongues elongated along strike (~220 m across, ~110 m down the fall line), was 0.008 / 0.0055 = flames)
    float tng = smoothstep(-0.25, 0.20, faceNoise(vec2(0.0045, 0.009), 23.0)   // (session 5: 2:1 -> 1.45:1, "flames")
                                      + 0.12 * (1.0 - smoothstep(4.0, 10.0, fp)) * faceNoise(vec2(0.022, 0.012), 31.0));
    sHo *= mix(0.55, 1.0, tng);
    sSh *= mix(0.45, 1.0, tng);
    // the glaciated far ranges (shot 6): the upper faces and every shoulder white above a ragged
    // limit, glaciers in their hollows; only the steepest rock (> ~65 deg) bare
    float hF = h + 180.0 * nb + 90.0 * nL + 220.0 * (north - 0.5) + 300.0 * saturate(1.4 * curvB);
    // (session 3: 1750-2050 -> 1380-1680 m: the down-valley range tops out at 2300-2800 m, so only its
    // summit tips were white; the upper ~half of every far massif should be under snow and ice)
    // (r4 M8: 1380 / 1680 -> 1750 / 2050: ~30 % patchy snow on the far range - it was a snow mantle)
    float sFar = rngB * smoothstep(1750.0, 2050.0, hF) * smoothstep(0.30, 0.50, nyC + 0.05 * nm / 0.12)
               * smoothstep(-0.9, 0.0, nS + 0.4 * nL);
    // ledges across the steep high faces (the strata bands of the relief pass): snow lines where the
    // footprint-filtered normal is flat on a steep 64 m face (they fade out by themselves once the
    // ledges are below the pixel footprint)
    float sLd = 1.0 * smoothstep(0.70, 0.86, mix(ny, nyC, 0.55)) * (1.0 - smoothstep(0.62, 0.80, nyC)) * far   // (r4 M3: 0.85, 0.76 / 0.90)
              * smoothstep(1250.0 + fL, 1550.0 + fL, hN + 250.0 * (north - 0.5)) * smoothstep(-0.5, 0.4, nL + 0.5 * nS);
    // (session 4: the additive ~200 m / ~1 km terms (0.07 nL + 0.08 nS) are gone - they only gate the terms
    // above: added to sv they left large half-transparent "fog" fields with soft edges)
    // (round 3) melt pattern of the ground itself: facets turned to the sun (against the 64 m surface) melt out
    // first, the shaded backs of the slabs / joints and the lee hollows keep their snow - field margins that
    // follow the micro-topography (noise-cut margins read as torn paper / cartoon blobs, crops c7 / c5). (The
    // relief-tread term of t11 (treads hold, risers shed) is gone: it tore the couloir strips into teeth.)
    float melt = (dot(nC, uSunDir) - dot(n, uSunDir)) * far * (1.0 - smoothstep(6.0, 14.0, fp));
    // (round 3) rocky, ragged margins at 1-9 m when the footprint resolves them (a smooth cartoon outline at 1:1, s5 /
    // crop c5): volume noise, isotropic on the walls
    float vMg = far * (1.0 - smoothstep(1.0, 2.5, fpO));
    // (t19) couloir strips vs open fields: the strips (a narrow cross-gully profile) were torn into teeth and holes by
    // the margin noises (crops c7 / c1: "torn paper"); couloir snow runs as continuous ribbons with
    // smooth, wall-following margins -> the margin noises x(1 - 0.65) on the couloir-only snow. The open fields on
    // gentler ground (s5's cone: a smooth cartoon outline at 1:1) get MORE break-up: a 14 m octave (rocks
    // poking through, lobed melt margins)
    float sOpen = max(max(sBn, sHi), max(sHo, sSh));
    float isC = smoothstep(0.25, 0.6, sC) * (1.0 - smoothstep(0.2, 0.5, sOpen));
    float mgK = 1.0 - 0.65 * isC;
    float gK = smoothstep(0.62, 0.8, nyC) * (1.0 - isC);
    float sv = max(max(max(sC, sBn), max(sHi, sHo)), max(max(sSh, sFar), sLd)) + 0.05 * nm / 0.12
             + 0.3 * melt
             + 0.08 * mgK * vMg * (texture(uNoise3, p * (1.0 / 40.0) + 0.61).r - 0.5) / 0.1
             + 0.09 * gK * far * (1.0 - smoothstep(0.5, 1.2, fpO)) * (texture(uNoise3, p * (1.0 / 14.0) + 0.83).r - 0.5) / 0.1
             + 0.05 * mgK * far * (1.0 - smoothstep(2.5, 6.0, fpO)) * (texture(uNoise3, p * (1.0 / 130.0) + 0.17).r - 0.5) / 0.1
             // (t11) broken, partly melted margins at 2-20 m (volume noise: isotropic on the walls too)
             + 0.04 * mgK * dV * far * (texture(uNoise3, p * (1.0 / 90.0) + 0.37).r - 0.5) / 0.1
             + 0.015 * (1.0 - smoothstep(4.0, 10.0, fp)) * fbm2F(p.xz * 0.033 + 13.0, 3, 30.0, fp) / 0.12;   // (session 3: 0.04 -> 0.015, fewer flecks)
    // (t26) rock steps in the couloirs: the continuous strips ran down whole faces as white paint drips (r10, c1 at
    // 1.5-3 km; gully snow is interrupted where the gully floor steepens into a step). Gaps in altitude bands
    // (~40 m, the band pattern drifting over ~600 m: anisotropic lookup of the noise volume), ~1 in 5 bands, on the
    // couloir-only snow; faded once a band is < ~6 px
    {
        // (the fbm channel's features are ~1/8 of its period: 40 m vertical / 600 m horizontal)
        // (t26 variants: 0.6 = straight horizontal cuts through every strip (man-made); 0.3 + ragged band edges (+-6 m
        // of the 40 m edge noise) kept)
        float gB = texture(uNoise3, vec3(p.x * (1.0 / 4800.0) + 0.13, (h + 6.0 * nE) * (1.0 / 320.0) + 0.41, p.z * (1.0 / 4800.0))).r;
        sv -= 0.30 * isC * smoothstep(0.58, 0.68, gB) * (1.0 - smoothstep(4.0, 7.0, fp));
    }
    // edge width: the inputs of sv are already smooth over >= 3 px (64 m / 3 fp taps, footprint-filtered
    // noise), so a narrow threshold still gives 1-2 px edges (session 3: 0.12 + 0.30 washed the far
    // fields into a grey veil: sv 0.52 gave 50 % snow)
    // (t13: 0.07 + 0.11 -> 0.04 + 0.12 from 1.5-10 m: the fields' soft fringes read as cotton wool at 1:1)
    float ew = (0.04 + 0.12 * smoothstep(1.5, 10.0, fp)) * (1.0 + W2K(18));   // (session 5: 0.10 -> 0.07 near: crisper field edges)
    // (round 3) the couloir floors hold snow ~10 deg steeper than the open faces (the strips broke into holes where the
    // gully steepened)
    float shedC = max(shed, smoothstep(mix(0.30, 0.18, rngB), mix(0.46, 0.36, rngB), nyC + 0.05 * nm / 0.12) * smoothstep(0.2, 0.6, cBig));
    float snowNew = smoothstep(0.52 - ew, 0.52 + ew, sv) * shedC * smoothstep(740.0 - 250.0 * rng, 860.0 - 250.0 * rng, h);
    // (round 2: only the sharp crests / rib tops; 0.05..0.45 bared every shoulder)
    snowNew *= 1.0 - 0.85 * smoothstep(0.15, 0.70, -curv) * far * (1.0 - 0.75 * rngB);   // crests, ribs and spire tips bare (near ranges)

    // previous snow (kept at the camera so nothing near pops)
    float snowOld = smoothstep(820.0, 1150.0, h + 240.0 * n1 + 60.0 * n2) * smoothstep(0.35, 0.6, ny + 0.2 * n2);
    snowOld *= mix(0.25, 1.0, saturate(n.z * 1.2 + 0.4)) * mix(0.35, 1.0, 1.0 - smoothstep(0.55, 0.85, ao));
    snowOld = smoothstep(0.18, 0.42, snowOld + 0.12 * n3);
    snowM = mix(snowOld, snowNew, far);
    // (session 3) thin, dirty snow at the rims of the fields, clean in their cores: the uniform white cut-outs
    // read as pasted on (grey margins, dust streaks down the fall line)
    float sCore = smoothstep(0.0, 0.28, sv - 0.52);
    float sDust = (1.0 - smoothstep(8.0, 25.0, fp)) * faceNoise(vec2(0.05, 0.012), 11.3);
    // (session 5: the dust streaks and the 0.80 rims read as brush strokes / smears at 1:1 final pixels)
    snow *= mix(1.0, mix(0.88, 1.0, sCore), far);

    if (uDebug == 21) scr = 0.0;
    if (uDebug == 23) { veg = 0.0; scr = 0.0; snowM = 0.0; }
    vec3 alb = rock;
    // fans have a body: thin debris over rock at the margins, the pale core along the middle (session 6:
    // flat cut-outs of one tone read as camouflage blotches at 1-2 km)
    alb = mix(alb, scree * mix(0.86, 1.04, smoothstep(0.40, 0.80, f.b)), scr);
    alb = mix(alb, vec3(0.034, 0.044, 0.022) * (0.8 + 0.5 * saturate(0.5 + n2ff)), sShrub * scr * (1.0 - f.a));   // (t11) shrubs on the old debris
    vec3 nv = n;
    // (t19) turf covers the slab relief: on the relief's turf the facet normal (V-joints in the sun's shadow =
    // crisp black dashes all over the grassy 30-45 deg slopes, crop c7 lower left) gives way to the base surface;
    // the rock between the turf keeps its facets
    float tvW = saturate(rkF * 0.8);
    if (tvW > 0.01 && veg > 0.01) {
        float eb = max(3.0, 1.5 * fp);
        vec3 nB = normalize(vec3(terrainH(p.xz - vec2(eb, 0.0), true) - terrainH(p.xz + vec2(eb, 0.0), true), 2.0 * eb,
                                 terrainH(p.xz - vec2(0.0, eb), true) - terrainH(p.xz + vec2(0.0, eb), true)));
        nv = normalize(mix(n, nB, tvW));
    }
    gVegS = veg * tvW;
    vec3 vegAlb = groundVegAlbedo(p, nv, t, f, ao);
    // dark dwarf-pine / alpenrose patches through the open belt above the forest
    // (session 4: a bigger share of the 320-1100 m belt (-0.2 -> -0.45), and the big gullies hold it)
    float dwarf = smoothstep(-0.45, 0.45, (nm * 0.8 + 0.4 * n2ff - 0.25 * nb) / 0.12 + 1.5 * gulG + 0.5 + W2K(17)) * smoothstep(320.0, 520.0, hvV) * (1.0 - smoothstep(900.0, 1100.0, hvV));
    dwarf *= far * (1.0 - f.a) * (1.0 - saturate(f.r * 2.0));
    dwarf = mix(dwarf, smoothstep(0.40, 0.60, dwarf + kD * nD2) * far * (1.0 - f.a) * (1.0 - saturate(f.r * 2.0)), dV);   // (t11) clumps with crisp edges, not soft blobs
    vegAlb = mix(vegAlb, vec3(0.040, 0.056, 0.026) * (0.8 + 0.4 * saturate(0.5 + n2ff)), dwarf * 0.8);
    // far alpine turf reads darker and more olive than the lit meadow up close (tussocks, shrubs, rock)
    // (session 4: 0.65 -> 0.5, the far turf read as bright olive patches)
    vegAlb *= mix(1.0, 0.55 + W2K(16), smoothstep(80.0, 700.0, t) * (1.0 - f.a));
    // (t12) the far turf is not smooth paint: tussocks, dwarf shrubs, stones and bare soil at 2-9 m (cells and
    // the mosaic fbm, faded with the footprint)
    if (dV * far > 0.0) {
        float tg = dV * far * (1.0 - f.a);
        vec3 vt = voronoi2(rot2(2.1) * p.xz * (1.0 / 3.2) + 5.0);
        float cellT = (fract(vt.z * 9.7) - 0.5) * (1.0 - smoothstep(0.7, 1.4, fp));
        vegAlb *= 1.0 + tg * (0.16 * clamp(0.6 * nD2 - 0.4 * nD, -1.5, 1.5) + 0.35 * cellT);
    }
    alb = mix(alb, vegAlb, veg);
    // (session 6) the steep faces of the forest belt are not bare monoliths (t21 shot 0 left face: a smooth
    // triangular slab in the forest): irregular turf / dwarf-shrub patches, elongated along the face (ledges),
    // ~10-30 m, clustered with the ~50 m outcrop noise, more on the ledgier faces. (t23: thin strips + joint
    // shrubs read as drawn dashes / glyphs at 1:1 final pixels.) Footprint-faded fbm: no shimmer
    float lvW = far * (1.0 - veg) * (1.0 - scr) * (1.0 - smoothstep(850.0, 1250.0, hv)) * smoothstep(0.20, 0.42, nyC)
              * (1.0 - smoothstep(0.86, 0.95, nyC)) * smoothstep(60.0, 140.0, h);
    if (lvW > 0.0) {
        float lp = faceFbm(vec2(0.035, 0.10), 3, 28.0, fp, 61.0) / 0.15
                 + 0.6 * clamp(nm / 0.12, -1.5, 1.5) + 1.2 * (nyC - 0.55);
        float m = smoothstep(0.55, 0.95, lp + 0.6 * W2K(26));
        // (world3) on the relief walls: turf / shrubs on the ledge treads (flat fine normal), patchy
        m = mix(m, smoothstep(0.55, 0.85, ny) * smoothstep(0.0, 0.6, lp + 0.6), rkF);
        vec3 lgc = mix(vec3(0.034, 0.046, 0.024), vec3(0.080, 0.084, 0.042), saturate(0.5 + n2ff + 0.5 * n3f));
        alb = mix(alb, lgc, m * lvW);
    }
    float rockW = (1.0 - veg) * (1.0 - scr);
    // (session 4) on the walls rockNormal's joint-panel tilts (47 / 71 m warped panels) hand over to the
    // face facets below with the albedo (polygon shading 0.2-2 km out)
    float rkNW = rockW * (1.0 - wallF * smoothstep(150.0, 400.0, t));
    n = normalize(mix(mix(n, nv, veg), rkNW > 0.0 ? rockNormal(p, n, t) : n, rkNW));
    // far away (where rockNormal has no relief left): ribs / gullies / ledges below the heightfield
    // resolution, so the faces shade like fractured granite instead of smooth clay
    float wB = smoothstep(70.0, 180.0, t) * (rockW + 0.35 * scr * (1.0 - veg)) * (1.0 - 0.85 * snowM);
    if (wB > 0.01) {
        // the 4 m heightfield's erosion rills read as a fine diagonal hatch on the far faces: pull the
        // shading part-way to the coarse (>= 64 m) normal, then add the facets / ribs below it
        // (session 3: 0.55 -> 0.4; with the fins of the relief pass gone the facets read crisper)
        // (t25 variants: no pull / no faceBump or x1.6 - all ~ the same at 1:1; kept)
        n = normalize(mix(n, nC, 0.3 * wB * smoothstep(150.0, 500.0, t) * (1.0 - rkF)));   // (session 5: 0.4 -> 0.3, more of the real relief)
        n = normalize(n - wB * (gb - dot(gb, n) * n));
        float wF = wB * (1.0 - 0.65 * scr) * (1.0 - rkF);
        n = normalize(n + wF * (gF - dot(gF, n) * n));
    }
    // (round 3) slabs / knobs where the rock shows through the turf on gentle ground (no slab relief there)
    float wO = rockW * far * smoothstep(0.52, 0.70, nyC) * (1.0 - rkF) * (1.0 - snowM) * (1.0 - smoothstep(1300.0, 1600.0, h));
    if (wO > 0.01) {
        float ocr;
        vec3 go = outcropSlabs(p.xz, fpO, ocr);
        n = normalize(n - wO * (go - dot(go, n) * n));
        alb *= mix(1.0, 1.0 - 0.35 * ocr, wO);
    }
    // (t13) block faces of the debris (scree section): lit sun sides, shaded lee sides
    float wSB = scr * (1.0 - veg) * (1.0 - snowM);
    if (wSB > 0.01) n = normalize(n + wSB * (scrB - dot(scrB, n) * n));
    // snow fills the crevices: its surface follows the coarse ground shape, not the crag relief
    // (also keeps the bright patches from shimmering on the spires)
    // (session 4: 0.6 -> 0.3: the fields take their shading from the ground under them, not pasted paper)
    // (t11) on the slab-relief walls the snow fills the joints: its surface follows the base heightfield, not the
    // facets (facet shading and V-joint shadows inside the fields read as torn paper, t10 crop c7)
    if (snowM > 0.01 && rkF > 0.01) {
        float eb = max(8.0, 2.0 * fp);   // (t12: 1.5 m taps kept the base map's 4 m rills - vertical streaks on the snow)
        vec3 nB = normalize(vec3(terrainH(p.xz - vec2(eb, 0.0), true) - terrainH(p.xz + vec2(eb, 0.0), true), 2.0 * eb,
                                 terrainH(p.xz - vec2(0.0, eb), true) - terrainH(p.xz + vec2(0.0, eb), true)));
        n = normalize(mix(n, nB, snowM * rkF * 0.6));
    }
    n = normalize(mix(n, nC, (0.45 + W2K(10)) * snowM * far * (1.0 - 0.7 * rkF)));   // (session 5: 0.3 -> 0.45: the 4-10 m rills streaked the fields like flames)
    float shore = (1.0 - smoothstep(0.35, 1.3, above + 0.4 * n3)) * (1.0 - 0.7 * saturate(f.a * 1.5));
    // (veg4 W4 / W11 / G13) shore band: (a) the waterline itself is stone / wet gravel within ~0.25 m above the
    // water whatever the meadow weight (meadow paint reached the water on the flats: lime "rafts" in r1 /
    // shot01), not on banks steeper than ~10-15 deg; (b) under the forest a narrow dark wet band (roots,
    // dark stones; no pale pebble kerb along the wooded shores); (c) on the grassy shore stretches (the 3D
    // cover's gcShoreGrass) the sward reaches down to ~0.3 m with a noise-broken sparse-grass / gravel
    // ecotone, so the far shore beyond the 3D cover agrees with the near one
    vec3 pebW = pebble;
    if (above < 1.8) {
        float eco = (vnoise(p.xz * 1.3 + 7.0) - 0.5) * (1.0 - smoothstep(0.25, 0.8, fp));
        float wEdge = (1.0 - smoothstep(0.16, 0.32, above + 0.10 * n3 + 0.12 * eco)) * smoothstep(0.955, 0.985, ny);
        float fo = smoothstep(0.3, 0.55, f.r) * (1.0 - 0.6 * saturate(f.a * 1.5));
        float sgS = t < 900.0 ? gcShoreGrass(p.xz) * (1.0 - fo) : 0.0;
        float shN = 1.0 - smoothstep(0.15, 0.5 + 0.25 * eco, above + 0.12 * n3);       // narrow band
        shore = mix(shore, shN, max(fo, 0.75 * sgS));
        shore = max(shore, wEdge);
        pebW = mix(pebble, pebble * vec3(0.50, 0.50, 0.44), fo);                     // wet dark stones under the trees
    }
    alb = mix(alb, pebW, shore);
    alb = mix(alb, snow, snowM);

    // rounded pebbles up close on shores
    if (shore > 0.01 && t < 40.0) {
        vec3 bumpN;
        vec3 pc = pebbles(p.xz, bumpN);
        float w = shore * (1.0 - smoothstep(22.0, 40.0, t));
        alb = mix(alb, pc, w);
        n = normalize(n + bumpN * w * (1.0 - smoothstep(8.0, 25.0, t)));
    }
    // wet band at the waterline, darker submerged stones
    wetness = 1.0 - smoothstep(0.0, 0.35 + 0.15 * n3, above);
    alb *= mix(1.0, 0.55, wetness);
    spec = mix(0.02, 0.25, wetness) + 0.2 * snowM;
    // look-dev views (uDebug): 21 no scree, 22 black (aerial in-scatter only), 23 bare rock,
    // 24 material ids (red scree, green vegetation, white snow), 27 snow terms (final, couloir, bench),
    // 26 (couloir curvature, summit fields, effective snow height), 29 = 24 inside the 350-700 m belt only, 28 grey albedo (shading
    // only), 44 grey albedo + unperturbed normal (heightfield shading), 46 facet cracks / tone / plane blend
    if (uDebug == 22) { spec = 0.0; return vec3(0.0); }
    if (uDebug == 49) { spec = 0.0; return vec3(rkF, gVegS, scr) * 0.5; }   // (world3) relief weight / turf on relief / scree
    if (uDebug == 46) { spec = 0.0; return vec3(crack, ftone + 0.5, gFwb) * 0.4; }
    if (uDebug == 28) { spec = 0.0; return vec3(0.18); }
    if (uDebug == 27) { spec = 0.0; return vec3(snowNew, sC, sBn) * 0.5; }
    if (uDebug == 47) { spec = 0.0; return vec3(sHo, sSh, sFar) * 0.5; }
    if (uDebug == 48) { spec = 0.0; return vec3(gulG, saturate(curvG * 2.0), wallF) * 0.5; }
    if (uDebug == 26) { spec = 0.0; return vec3(saturate(cBig), sHi, saturate((hN - 500.0) / 2500.0)) * 0.5; }
    if (uDebug == 29) { spec = 0.0; if (h < 350.0 || h > 700.0 || f.r > 0.25) return vec3(0.0); return mix(mix(mix(vec3(0.12), vec3(0.35, 0.02, 0.02), scr), vec3(0.03, 0.15, 0.03), veg * (1.0 - scr)), vec3(0.45, 0.02, 0.5), snowM); }
    if (uDebug == 24) { spec = 0.0; return mix(mix(mix(vec3(0.12), vec3(0.35, 0.02, 0.02), scr), vec3(0.03, 0.15, 0.03), veg * (1.0 - scr)), vec3(0.45, 0.02, 0.5), snowM); }
    return alb;
}

float gFallsMoss = 0.0;      // falls lab debug 13
vec3 shadeTerrain(vec3 p, vec3 rd, float t, bool cheap) {
    gFallsMoss = 0.0;
    gRkFine = true;                                            // (world3) all relief octaves for the shading
    float det = detailWeight(t);
    float e = max(0.03, t * gPixAng * 1.2);
    vec3 n = terrainNormal(p.xz, e, det);
    // (world3) joint-block relief at this point: its weight steers the shading, its depth lifts the point
    // back to the base surface for the baked shadow maps
    rockRelief(p.xz, terrainH(p.xz, true));
    float rkW = gRkW, rkD = gRkD;
    gRkWS = rkW; gRkDS = rkD;
    // (world5 R4) the slab facets' share of the shading normal x(1 + kB) beyond ~0.4-1.5 km (shading only, the march
    // keeps the relief as it is): the 42 / 15 m facets were faded to soft undulations there (r4 M10 kF / T), the walls
    // read as smeared plaster at 1:1 - the faces should show crisp lit / shaded planes at 1-5 km. Relief gradient =
    // the normal's gradient minus the base map's (the same taps as terrainNormal's base part)
    {
        float kB = (1.5 + W2K(31)) * smoothstep(400.0, 1500.0, t) * rkW;   // (sw2 / sw4: 1.5)
        if (kB > 0.0 && !cheap) {
            float eBb = mix(e, max(e, 3.0), nearBlend(p.xz));
            vec2 gB = vec2(terrainHN(p.xz + vec2(eBb, 0.0)) - terrainHN(p.xz - vec2(eBb, 0.0)),
                           terrainHN(p.xz + vec2(0.0, eBb)) - terrainHN(p.xz - vec2(0.0, eBb))) / (2.0 * eBb);
            vec2 gT = -n.xz / max(n.y, 0.05);
            vec2 gR = gT - gB;
            n = normalize(vec3(-(gB.x + (1.0 + kB) * gR.x), 1.0, -(gB.y + (1.0 + kB) * gR.y)));
        }
    }
    // (world5 R5) the finer slab-facet octaves (15 / 5.4 / 1.9 m, world.glsl rockRelief) in the SHADING normal down to
    // ~2.2-4.5 px per cell (the march fades them at 4.2-8.5 px): crisp lit / shaded facets on the 0.3-4 km walls, which
    // kept only the 42 m (+15 m) facets - soft folds at 1:1. Only the part the march lacks (fade difference), same joint
    // families / tilts / seeds as rockRelief, gradient by central differences along the surface over ~0.7 px
    {
        float fpx = t * gPixAng;
        float kR = (1.8 + W2K(8)) * rkW * smoothstep(250.0, 600.0, t) * (1.0 - smoothstep(5000.0, 7000.0, t));   // (sw4: 1.8)
        if (kR > 0.0 && !cheap) {
            vec4 nz = texture(uNoise3, vec3(p.xz * (1.0 / 4096.0), 0.37));
            float ang = 0.5 + 2.4 * (nz.g - 0.5);
            vec3 F0 = vec3(cos(ang), 0.25, sin(ang)), F1 = vec3(-sin(ang), -0.2, cos(ang)), F2 = vec3(0.3 * cos(ang), 1.0, 0.3 * sin(ang));
            vec2 gN = -n.xz / max(n.y, 0.05);
            float slN = length(gN);
            float T = mix(0.8, 0.55, smoothstep(800.0, 2000.0, t)) * mix(0.35, 1.0, smoothstep(0.75, 1.2, slN));
            float kF = mix(1.0, 0.45, smoothstep(700.0, 2000.0, t));
            float es = max(0.7 * fpx, 0.05);
            vec3 P = vec3(p.x, terrainH(p.xz, true), p.z);
            vec3 dX = vec3(es, gN.x * es, 0.0), dZ = vec3(0.0, gN.y * es, es);
            vec2 gE = vec2(0.0);
            float C = 15.12, amp = 0.7 * kF;
            for (int k = 1; k < 4; k++) {
                float wk = smoothstep(3.0 * fpx, 6.0 * fpx, C) - smoothstep(4.2 * fpx, 8.5 * fpx, C);   // (shimmer test: 2.2 / 4.5 -> 3 / 6 px)
                if (wk > 0.001) {
                    vec3 Q = P + 37.0 * float(k);
                    uint sd = uint(5 + 13 * k);
                    float ex1 = rkFacets3(Q + dX, C, T, fpx, F0, F1, F2, sd), ex0 = rkFacets3(Q - dX, C, T, fpx, F0, F1, F2, sd);
                    float ez1 = rkFacets3(Q + dZ, C, T, fpx, F0, F1, F2, sd), ez0 = rkFacets3(Q - dZ, C, T, fpx, F0, F1, F2, sd);
                    gE += wk * amp * vec2(ex1 - ex0, ez1 - ez0) / (2.0 * es);
                }
                C *= 0.36; amp *= 0.7;
            }
            float sec = min(sqrt(1.0 + slN * slN), 3.0);
            gE *= kR * mix(1.0, sec, 0.5);
            n = normalize(vec3(-(gN.x + gE.x), 1.0, -(gN.y + gE.y)));
        }
    }
    vec4 f = forestAt(p.xz);
    float ao = terrainAO(p.xz);
    float wl = waterLevel(p.xz);
    float wet, snowM, spec;
    vec3 n0 = n;
    vec3 alb = terrainAlbedo(p, n, t, f, wl, ao, wet, snowM, spec);
    // (world3) cavity of the slab relief: its height against the mean of 8 taps 2-5 px around (a Laplacian of
    // D, the base slope cancels): the V-joints between the slabs and the recesses hold moisture, lichen,
    // debris and shade - dark crisp joint lines and pockets; the slab edges stay clean and catch the light.
    // Carries the structure in frontal light (shot 0's 1.3 km face: aspect = sun azimuth), where facet
    // tilts barely change the shading
    if (rkW > 0.01 && !cheap) {
        float fpx = t * gPixAng;
        float ec = max(1.5, 2.0 * fpx);
        // (t20, perf: 8 taps at 1 / 2.3 ec -> 4 taps at 1.6 ec, same look in the 1:1 crops)
        float s4 = 0.0;
        for (int i = 0; i < 4; i++) {
            float a = float(i) * 1.5708 + 0.3;
            vec2 q = p.xz + vec2(cos(a), sin(a)) * (1.6 * ec);
            s4 += rockRelief(q, terrainH(q, true));
        }
        float cav = rkD - s4 * 0.25;
        float dk = smoothstep(0.15, 2.5, -cav) * rkW * (1.0 - snowM) * (1.0 - gVegS);
        float br = smoothstep(0.15, 2.5, cav) * rkW * (1.0 - snowM);
        // (round 3: 0.45 -> 0.35, x0.6 once the taps are > ~3 m apart: at frame scale the 6-14 m recesses darkened
        // into blotches on top of the facet shading - s7 final vs its grey-albedo view)
        // (world5 R2: 0.35 / x0.6 -> 0.60 / x0.90 (sw4; lab knobs 29 / 24): crisp dark joints between the slabs at 0.5-3 km)
        alb *= (1.0 - (0.60 + W2K(29)) * mix(1.0, 0.90 + W2K(24), smoothstep(1.5, 4.0, fpx)) * dk) * (1.0 + 0.14 * br);
        gRkWS = rkW; gRkDS = rkD;                                   // (rockRelief overwrote the globals)
    }
    // dusty forest trail with roots and needles
    // (veg4 G4) a narrow (~0.7-1.5 m) compacted grey-brown dirt path (#797169, R-B ~16, luma p10 / p90
    // ~55 / 227 with the dappled shade) with exposed roots crossing it every 0.5-1.5 m, embedded stones and
    // needle litter; it was a 2-4 m smooth pinkish sand road (#aa8f70, R-B 58) that read as a beach
    if (uTrailN > 1 && t < 400.0) {
        float td = 1e9;
        vec2 tdir = vec2(1.0, 0.0);
        for (int i = 0; i < 31; i++) {
            if (i + 1 >= uTrailN) break;
            vec2 a2 = uTrail[i].xy, b2 = uTrail[i + 1].xy, ab2 = b2 - a2;
            float tt = clamp(dot(p.xz - a2, ab2) / dot(ab2, ab2), 0.0, 1.0);
            float dd = length(p.xz - (a2 + ab2 * tt));
            if (dd < td) { td = dd; tdir = ab2; }
        }
        float tw = 1.0 - smoothstep(0.35, 0.75 + 0.25 * gnoise(p.xz * 0.7), td);
        if (tw > 0.0) {
            float fpT = max(t * gPixAng, 0.002);
            float g = texture(uNoise3, vec3(p.xz * 0.9, 0.4)).r;
            float g3 = mix(texture(uNoise3, vec3(p.xz * 2.6, 0.63)).g, 0.5, smoothstep(0.01, 0.03, fpT));   // ~3 cm grain
            vec3 dirt = mix(vec3(0.10, 0.085, 0.065), vec3(0.30, 0.26, 0.20), saturate((g - 0.5) * 1.6 + 0.5 + 0.5 * (g3 - 0.5)));
            // trampled centre a little paler and smoother, the edges darker with litter
            dirt *= mix(0.72, 1.06, 1.0 - smoothstep(0.1, 0.6, td));
            // exposed roots: borders of Voronoi cells stretched across the path (~0.9 m along, ~3 m across), 4-8 cm
            vec2 tu = normalize(tdir), tv = vec2(-tu.y, tu.x);
            vec2 qr = vec2(dot(p.xz, tu), dot(p.xz, tv));
            vec3 vr = voronoi2(qr * vec2(1.1, 0.33) + 3.7);
            float rw = 0.020 + 0.018 * fract(vr.z * 13.1);                  // half width (m)
            float dB = 0.5 * (vr.y - vr.x) / 1.1;                            // ~ distance to the cell border (m)
            float fwR = max(fpT, 0.006);
            float root = (1.0 - smoothstep(rw - fwR, rw + fwR, dB)) * step(0.35, fract(vr.z * 7.3)) * min(1.0, 2.0 * rw / fwR);
            vec3 rootC = mix(vec3(0.06, 0.05, 0.04), vec3(0.15, 0.12, 0.09), saturate(1.0 - dB / max(rw, 1e-3)) * 0.6);   // lit crown of the root
            dirt = mix(dirt, rootC, root * 0.9);
            // embedded stones (~30 %), grey, 4-12 cm; needle specks (both fade to their mean when sub-pixel)
            vec4 vs = voronoiV(p.xz * 7.0 + 1.3);
            float sR = 0.38 + 0.25 * fract(vs.w * 5.3);
            float sw = max(fpT * 7.0, 0.04);
            float stn = (1.0 - smoothstep(sR - sw, sR + sw, vs.z)) * step(0.62, fract(vs.w * 3.1)) * (1.0 - root);
            dirt = mix(dirt, vec3(0.22, 0.215, 0.20) * mix(0.7, 1.2, fract(vs.w * 9.7)), stn * 0.85);
            float nd = mix(texture(uNoise3, vec3(p.xz * 6.0, 0.21)).a, 0.5, smoothstep(0.004, 0.012, fpT));
            dirt = mix(dirt, vec3(0.11, 0.065, 0.03), smoothstep(0.62, 0.8, nd) * 0.5 * (1.0 - stn));
            alb = mix(alb, dirt, tw);
        }
    }
    // the waterfall's rock step (falls.glsl wfRockShade): dark wet jointed rock, moss / fern tufts. The
    // rock material is evaluated ONCE for the step (fw) and the bank outcrops (outc) - (round 2, perf: two
    // inlined copies x two shadeTerrain call sites); reflections (cheap) get its flat mean (wfRockCheap)
    float fw = wfOff(2) ? 0.0 : fallsWetness(p, n0);
    float dpA, awA;
    // (review rv5) the foot of the cliff / banks at the waterline: dark wet rock into the water, not the lake
    // shore's pale pebble band - a pale strip ~1 m tall ran along the whole cliff foot (film shot 2). Where
    // the falls rock / bank zone stands just above (y 2 m), its rock reaches down into the water (not on the apron)
    if (fw < 1.0 && !wfOff(2) && p.y < 1.6) {
        float apr = fallsApron(p, dpA, awA);
        vec3 pu = vec3(p.x, 2.0, p.z);
        float zf = max(fallsWetness(pu), fallsBank(pu)) * (1.0 - apr);
        fw = max(fw, zf * (1.0 - smoothstep(0.7, 1.5, p.y)));
    }
    float fa = fw < 1.0 && !wfOff(2) ? fallsApron(p, dpA, awA) * (1.0 - fw) : 0.0;
    float fb = fw < 1.0 && !wfOff(2) ? fallsBank(p) * (1.0 - fw) * (1.0 - fa) : 0.0;
    // dark rock outcrops (with their own moss / shrubs) where the bank is steepest; on the bank next to
    // the falls only low down (ferns and shrubs above the dark buttress)
    float outc = fb > 0.0 ? fb * smoothstep(0.70, 0.56, n0.y + 0.25 * (texture(uNoise3, p * 0.013 + 0.6).r - 0.5)
                                            + 0.18 * fallsLushBank(p)) : 0.0;
    vec3 ra = vec3(0.0), nR = n;
    float mossR = 0.0, wetR = 0.0, occR = 1.0;
    if (max(fw, outc) > 0.0) {
        if (cheap) ra = wfRockCheap(p, n0, mossR);
        else ra = wfRockShade(p, n0, t, nR, mossR, wetR, occR);
    }
    if (fw > 0.0) {
        gFallsMoss = mossR;
        alb = mix(alb, ra, fw);
        n = normalize(mix(n, nR, fw));
        ao *= mix(1.0, occR, fw);
        spec = max(spec, max(1.33 * fw * wetR, 0.12 * fw * mossR));     // (r3c: leaf sheen on the face's cover)
        wet = max(wet, fw * wetR);
    }
    // the angular shore slabs on the apron north of the falls
    if (fa > 0.0) {
        float occA, wetA;
        vec3 aa = wfApronShade(p, n0, t, dpA, awA, fa, n, occA, wetA);
        alb = mix(alb, aa, fa);
        ao *= mix(1.0, occA, fa);
        spec = max(spec, 0.4 * fa * wetA);
        wet = max(wet, fa * wetA);
    }
    // the inflow stream's banks and its mouth delta (meadow grass right up to the boulders). The
    // stream's level map makes terrainAlbedo's lake-shore rule paint a pebble beach along the whole
    // stream: back to grass wherever the ground stands > 0.15 m above the water (falls lab)
    if (uStreamN > 1 && t < 900.0 && !wfOff(3)) {
        vec2 sdb = streamDist(p.xz);
        float hwc = 0.45 * sdb.y;                                // wet channel half width (bake: 0.35-0.5 x)
        float dmo = dot(uStreamMouth.zw, uStreamMouth.zw) > 0.5 ? length(p.xz - uStreamMouth.xy) : 1e9;
        float gw = max(smoothstep(1.0, 1.5, sdb.x / hwc) * (1.0 - smoothstep(3.5, 4.0, sdb.x / hwc)), 1.0 - smoothstep(30.0, 44.0, dmo));
        float abv = wl > -1000.0 ? p.y - wl : 99.0;
        // (grass also on the cut banks of the delta: no pale scree walls either side of the mouth)
        float dlt = 1.0 - smoothstep(30.0, 44.0, dmo);
        gw *= smoothstep(0.15, 0.35, abv) * smoothstep(mix(0.55, 0.30, dlt), mix(0.75, 0.55, dlt), n0.y) * (1.0 - snowM);
        if (gw > 0.01) {
            vec3 ng = n0;
            vec3 ga = groundVegAlbedo(p, ng, t, f, ao);
            alb = mix(alb, ga, gw);
            n = normalize(mix(n, ng, gw));
            wet *= 1.0 - gw;
            spec = mix(spec, 0.02, gw);
        }
        // (review r2, fix 9) beyond the near map (z > ~1800 m) the stream is painted onto the ground from its
        // polyline - the 4 m far map's channel holds almost no water surface (its level lies under the far
        // terrain over most of the channel): white water tumbling between boulders, dark wet rock at its
        // edges, pale granite boulders on the banks (film shot 4: a white cascade from 150-500 m)
        float nbF = nearBlend(p.xz);
        if (nbF > 0.0 && sdb.x < 3.0 * sdb.y) {
            float fpS = t * gPixAng;
            vec2 fdS = gStreamDir.xy, fxS = vec2(-fdS.y, fdS.x);
            vec2 pa = vec2(dot(p.xz, fdS), dot(p.xz, fxS));
            float hwW = 0.48 * sdb.y;
            float eN = 0.30 * hwW * gnoise(pa * vec2(0.07, 0.25) + 1.7);
            float wmS = 1.0 - smoothstep(0.8 * hwW - fpS, hwW + fpS, sdb.x + eN);
            float wtS = 1.0 - smoothstep(hwW, 1.5 * hwW + fpS, sdb.x + eN);
            float vS = 1.0 - smoothstep(0.15, 0.6, fpS);
            float str = mix(0.55, texture(uNoise3, vec3(pa.x * 0.05, pa.y * 0.35, 0.83)).r + 0.3 * (texture(uNoise3, vec3(pa.x * 0.02, pa.y * 0.12, 0.29)).g - 0.5), vS);
            // boulders (~1.5-2.5 m cells): wet dark heads in the white water, pale granite on the banks
            vec4 bc = wfVor(pa * vec2(0.42, 0.55) + 31.0);
            float bR = 0.30 + 0.16 * fract(bc.w * 5.3);
            float inW = wmS;
            float hasB = step(fract(bc.w * 13.7), mix(0.45, 0.28, inW)) * (1.0 - smoothstep(1.4 * hwW, 2.6 * hwW, sdb.x));
            float bvis = 1.0 - smoothstep(0.4, 1.2, fpS);
            float boul = hasB * mix(0.35, 1.0 - smoothstep(bR - 0.5 * fpS, bR + 0.5 * fpS, length(bc.xy)), bvis);
            float steepF = smoothstep(0.05, 0.14, gStreamDir.z);
            vec3 wwA = vec3(0.80, 0.82, 0.83) * mix(0.68, 1.05, smoothstep(0.32, 0.66, str));
            vec3 boA = mix(vec3(0.075, 0.074, 0.068), vec3(0.30, 0.292, 0.272), 1.0 - inW) * mix(0.8, 1.2, fract(bc.w * 3.7));
            float whiteW = wmS * mix(0.55, 1.0, steepF) * smoothstep(0.25, 0.45, str + 0.25 * steepF) * (1.0 - boul);
            vec3 sa = mix(mix(alb * mix(1.0, 0.45, wtS), boA, boul), wwA, whiteW);
            float wS = nbF * max(wtS, boul * (1.0 - smoothstep(1.6 * hwW, 2.6 * hwW, sdb.x)));
            alb = mix(alb, sa, wS);
            n = normalize(mix(n, normalize(n0 * 0.5 + vec3(0.0, 1.0, 0.0) + vec3(fdS.x, 0.0, fdS.y) * (str - 0.5) * 0.9), nbF * wmS * (1.0 - boul)));
            spec = mix(spec, 0.3, nbF * wtS * (1.0 - whiteW));
            wet = max(wet, nbF * wtS * (1.0 - whiteW));
        }
    }
    // boulders of the inflow stream bed and its floodplain (fade out with the footprint)
    {
        vec3 ba; float occB;
        float fbo = streamBoulder(p, n0, t, n, ba, occB);
        if (fbo > 0.0) {
            float wetS = 1.0 - smoothstep(0.0, 0.25, p.y - wl);
            alb = mix(alb, ba * mix(1.0, 0.5, wetS), fbo);
            ao *= mix(1.0, occB, fbo);
            wet = max(wet, wetS * fbo);
            spec = max(spec, 0.3 * wetS * fbo);
        }
    }
    // ... and the forested banks / rim around it: shrubs, ferns, grass and needle litter, no bare rock
    if (fb > 0.0) {
        vec3 nv; float clv, occV;
        vec3 va = wfVegetation(p, n0, t * gPixAng, 0.0, nv, clv, occV);
        float lit = texture(uNoise3, p * 0.02 + 0.3).r;
        float floorW = smoothstep(0.2, 0.5, f.r + 0.5 * (lit - 0.5)) * (1.0 - smoothstep(0.45, 0.75, clv));
        float lush = fallsLushBank(p);
        floorW *= 1.0 - 0.7 * lush;
        va = mix(va * mix(0.62, 0.95, lush), vec3(0.052, 0.050, 0.032) * mix(0.7, 1.25, lit), floorW);
        // the cliff's rim: a dark mass of shrubs / young spruce, few lime tips
        float rim = fallsRim(p);
        va = mix(va, mix(vec3(0.016, 0.026, 0.014), va, 0.15 + 0.2 * smoothstep(0.7, 0.9, clv)), rim);
        alb = mix(alb, va, fb);
        n = normalize(mix(n, nv, fb * (1.0 - 0.5 * floorW)));
        ao *= mix(1.0, occV, fb * (1.0 - floorW));
        if (outc > 0.0) {
            alb = mix(alb, ra, outc);
            n = normalize(mix(n, nR, outc));
            ao *= mix(1.0, occR, outc);
        }
    }
    // shadows of the falls flora (3D shrubs / ferns, falls.glsl ffShadow) on the rock, banks and apron
    if (fw + fb + fa > 0.01 && !cheap) {
        float ffs = ffShadow(p, t);
        if (ffs > 0.0) { n = normalize(n - uSunDir * max(dot(n, uSunDir), 0.0) * ffs); ao *= 1.0 - 0.35 * ffs; }
    }
    // forest floor / understory darkness
    float canopy = saturate(f.r * 1.2);
    float occ = ao * mix(1.0, mix(0.55, 0.45, smoothstep(80.0, 200.0, t)), canopy * (1.0 - snowM));   // (veg4 G14: near floor 0.45 -> 0.55, target shaded floor L 0.32 vs 0.24)
    occ *= gcGroundSkyOcc;
    // (session 6) far open faces: the heightfield AO under-counts the sky a 1-5 km face sees (distant
    // shade is sky-blue at only ~1.7x below the lit faces); less of the warm valley fill there too
    // (session 4: floor 0.8 -> 0.65 - it erased the AO of the gullies / hollows far away: soft "wax" faces)
    occ = mix(occ, max(occ, 0.5 + W2K(11)), smoothstep(500.0, 2000.0, t) * (1.0 - canopy));   // (session 5: 0.65 -> 0.5: shot 6's shaded face read as one flat plane)
    // (world3) recesses of the joint-block relief (chimneys, the backs of the ledges) see less sky
    occ *= 1.0 - 0.2 * saturate(-rkD / 14.0) * (1.0 - snowM);   // (t11: snow fills them)
    float sun = terrainShadow(p - vec3(0.0, rkD, 0.0) + n * 0.3, cheap ? 40 : 90) * cloudShadow(p);
    // (perf: the relief shadow marches the three coarse octaves only; the 1.9 m one casts sub-pixel shadows)
    gRkFine = false;
    // (r4 M10: floor 0.3 -> 0.6 at 0.8-2.5 km: the hard lit / shaded slivers of every slab crumpled the 1-3 km faces like foil)
    // (world5 R3) relief-shadow floor 0.3 / 0.6 -> 0.13 / 0.33 (sw4; lab knob 30): crisper shaded facets at 1-3 km
    if ((rkW > 0.0 || rkD < -0.02) && sun > 0.0 && !cheap) sun *= mix(mix(max(mix(0.13, 0.33, smoothstep(800.0, 2500.0, t)) - W2K(30), 0.0), 1.0, max(0.85 * snowM, 0.8 * gVegS)), 1.0, rockReliefShadow(p + n * 0.05, t * gPixAng));
    if (!cheap && t < 700.0 && sun > 0.0) sun *= mix(treeShadow(p + n * 0.2), mix(1.0, 0.45, canopy), smoothstep(450.0, 700.0, t));
    else sun *= mix(1.0, 0.45, canopy);
    // (t21) light reflected onto shaded snow by the sunlit snow / rock around it (couloirs, bowls): the shaded snow
    // read saturated blue and dark (s7 / c7: ~106,137,180, B-R +74) against a target of #a8b6d1 (B-R +41) -
    // a neutral-warm lift of its albedo where the sun does not reach (the shared lighting only has sky fill there)
    if (snowM > 0.01) alb *= mix(vec3(1.0), vec3(1.22, 1.13, 1.0), snowM * (1.0 - sun * smoothstep(0.0, 0.25, dot(n, uSunDir))));
    // boulder contact shadows + occlusion on the ground
    if (!cheap && t < 120.0) {
        occ *= rocksGroundAO(p);
        if (sun > 0.0) sun *= rocksShadow(p + n * 0.03, uSunDir, 8.0, rockLod(t));
    }
    if (uDebug == 11) return vec3(fw, fb, fa);          // falls lab: material weights
    if (uDebug == 13) return vec3(fw * (1.0 - gFallsMoss), fw * gFallsMoss, 0.0);   // falls lab: bare rock / cover
    if (uDebug == 14 || uDebug == 15) return vec3(0.0);                              // falls lab: mask baselines
    vec3 colT = litSurface(alb, n, rd, sun, occ, spec, mix(0.6, 0.25, wet) - 0.048 * fw * wetR);
    // (falls6) wet sheen of the falls rock: the film of water mirrors the (mist-bright) sky - glistening up-facing
    // facets / ledge lips near the sheets; Schlick fresnel, the sky seen along the reflected ray
    if (fw > 0.0 && !cheap) {
        vec3 rr = reflect(rd, n);
        float Fw = 0.02 + 0.98 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
        colT += skyLight(rr, occ) * Fw * smoothstep(-0.05, 0.35, rr.y) * fw * wetR;
    }
    return colT;
}

// ---------------------------------------------------------------------------
// ray marcher : terrain heightfield (+ water surfaces) with tree and rock SDFs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// water_shade.glsl : lake surface shading: reflection, refraction, bed, scattering, glints (needs trace/shadeHit)
// ---------------------------------------------------------------------------
vec2 gDbgA = vec2(0.0);     // falls lab debug 66 / 67 (stream masks)
float gWWFoot = 0.0;        // (review rv8) footprint of the 3D white-water hit (lateral / along the view on the surface)
vec3 gDbgB = vec3(0.0);

// ---------------------------------------------------------------------------
// split renderer (engine/render.py Renderer): compiled as ONE program, the whole renderer cost the AMD
// compiler ~45-50 GB of commit memory and 13-36 min per fresh compile, while each part alone compiles in
// 1-210 s with < 1 GB (tools/split_probe.py, 2026-10-18). So every sample runs, per tile:
//   SCENE_PASS == 99 : the primary trace -> G-buffer: fragColor = (t, kind, cell), fragInfo = info
//   SCENE_PASS == K  : one program per hit kind K (K_SKY .. K_FALLSV): shades the pixels whose primary hit
//                      is of kind K (+ falls water volume, aerial perspective) into the accumulator and
//                      discards the others, so each pixel gets exactly one contribution per sample
// ---------------------------------------------------------------------------

void cameraRay(out vec3 ro, out vec3 rd) {
    vec2 frag = gl_FragCoord.xy + uJitter;
    // (r5 qa5 B13) + a frame term: the sample noise no longer sits fixed on the screen while the scene moves
    int iFrame = int(floor(uTime * 24.0 + 0.5));
    gRand = hash12i(ivec2(gl_FragCoord.xy) + ivec2(uSample * 7919 + iFrame * 31337, uSample * 104729 + iFrame * 7121));
    gPixAng = 2.0 * uTanHalfFov / uRes.y;
    vec2 ndc = (2.0 * frag - uRes) / uRes.y;
    vec3 rdc = normalize(vec3(ndc * uTanHalfFov, 1.0));
    ro = uCamPos;
    rd = uCamRot * rdc;
    if (uAperture > 0.0) {
        vec3 fp = ro + rd * (uFocusDist / rdc.z);
        vec2 lr = hash22i(ivec2(gl_FragCoord.xy) * 3 + ivec2(uSample * 31, uSample * 17));
        float ang = lr.x * TAU, rad = sqrt(lr.y) * uAperture;
        ro += uCamRot * vec3(cos(ang) * rad, sin(ang) * rad, 0.0);
        rd = normalize(fp - ro);
    }
}

layout(set=0, binding=14) uniform sampler2D uGBuf0;     // (t, kind, cell) of the primary hit (SCENE_PASS 99)
layout(set=0, binding=15) uniform sampler2D uGBuf1;     // its info

vec3 render(Hit h, vec3 ro, vec3 rd) {
    if (uDebug == 1) {
        vec3 kc[8] = vec3[8](vec3(0.2, 0.3, 1.0), vec3(0.6, 0.5, 0.4), vec3(0.0, 0.6, 0.0), vec3(0.5, 0.5, 0.5), vec3(0.0, 0.8, 0.9), vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0));
        return kc[h.kind];
    }
    vec3 p = ro + rd * h.t;
    vec3 col;
    gPrimary = false;
    col = shadeTerrain(p, rd, h.t, false);
    vec4 fv = fallsVolume(ro, rd, h.t, gRand);
    col = col * fv.a + fv.rgb;
    return applyAerial(col, ro, rd, h.t, 12);
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 g0 = texelFetch(uGBuf0, px, 0);
    if (int(g0.y + 0.5) != SCENE_PASS) discard;
    vec3 ro, rd;
    cameraRay(ro, rd);
    Hit h;
    h.t = g0.x; h.kind = SCENE_PASS; h.cell = ivec2(g0.zw); h.info = texelFetch(uGBuf1, px, 0);
    vec3 col = render(h, ro, rd);
    col = max(col, vec3(0.0));
    if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);
    fragColor = vec4(col, 1.0);
}
