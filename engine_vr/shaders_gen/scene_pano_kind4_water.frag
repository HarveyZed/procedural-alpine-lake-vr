#version 450
#define QUALITY 1
#define SCENE_PASS 4
#define VR_PANO 1
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
// T3 (port/PHASE_4.md): equirectangular panorama mode. uPixAng is a
// per-camera CONSTANT, but an equirect texel's angular footprint varies
// with latitude (port/RUNBOOK.md §6): fp(lat) = max(dLat, dLon*cos lat).
// panoFoot is the 1 x H R32F strip vr.py uploads from
// engine_vr/pano.texel_footprint(); row y carries the exact footprint of
// latitude band y. Redefining uPixAng here makes EVERY uPixAng consumer
// (this file, rocks.glsl, trees.glsl, shade_trees.glsl) sample the
// per-texel footprint without those files being touched. The footprint is
// symmetric in latitude, so the GL/Vulkan frag-y order vs the numpy row
// order of the strip is irrelevant. Every added line FIXME(port); the
// scene_pano_* programs are the only ones compiled with -DVR_PANO - all
// base programs resolve this whole branch away before vulkanize, so their
// generated sources and SPIR-V are unchanged (build_p4.py parity check).
layout(set=0, binding=12) uniform sampler2D panoFoot;   // FIXME(port)
float panoFootAt(float y) {   // FIXME(port)
    return texelFetch(panoFoot, ivec2(0, int(clamp(y, 0.0, uRes.y - 1.0))), 0).r;   // FIXME(port)
}                             // FIXME(port)
#define uPixAng panoFootAt(gl_FragCoord.y)   // FIXME(port) per-texel footprint, not a constant

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

float terrainHD(vec2 p, float detail) {
    float h = terrainH(p, true);
    h += rockRelief(p, h);
    if (detail > 0.001) h += detail * microRelief(p);
    return h;
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
vec4 streamAt(vec2 p) {
    if (nearBlend(p) >= 1.0) return vec4(0.0);
    return texture(uStreamNear, nearUV(p));
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
float phaseHG(float mu, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}
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

vec3 skyRadiance(vec3 ro, vec3 rd, int n) {
    vec3 tr;
    vec3 c = atmoScatter(ro, rd, 1e9, n, tr) * uSkyBoost;
    c = mix(c, vec3(luma(c)), uSkyDesat);
    float mu = dot(rd, uSunDir);
    float sd = smoothstep(0.999966, 0.999985, mu);
    c += uSunCol * sd * 600.0;
    return c;
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

float lightOD(vec3 p) {
    float od = 0.0;
    // (r4 L10) first step 45 -> 25 m (self-shadowed flanks and bases), x1.9 keeps the ~1.3 km reach; detail on the first
    // 2 steps (was 3) pays for the 4th detail octave
    float ls = 25.0, t = 0.0;
    for (int j = 0; j < 6; j++) {
        t += ls;
        od += cloudDensity(p + uSunDir * t, j < 2) * ls;
        ls *= 1.9;
    }
    return od;
}

// returns premultiplied cloud radiance (rgb) and transmittance (a)
vec4 renderClouds(vec3 ro, vec3 rd, float tmax, float jitter, int nsteps) {
    if (uCloudCover <= 0.001 || abs(rd.y) < 1e-4) return vec4(0.0, 0.0, 0.0, 1.0);
    float ta = (CL_BOT - ro.y) / rd.y, tb = (CL_TOP - ro.y) / rd.y;
    float t0 = max(min(ta, tb), 0.0), t1 = min(max(ta, tb), tmax);
    t1 = min(t1, t0 + 40000.0);
    if (t1 <= t0) return vec4(0.0, 0.0, 0.0, 1.0);
    float dt = (t1 - t0) / float(nsteps);
    float t = t0 + dt * jitter;
    vec3 col = vec3(0.0);
    float T = 1.0;
    float mu = dot(rd, uSunDir);
    const float SIG = 0.05;
    for (int i = 0; i < nsteps; i++) {
        vec3 p = ro + rd * t;
        float d = cloudDensity(p, true);
        if (d > 0.003) {
            float od = lightOD(p) * SIG;
            float hf = saturate((p.y - CL_BOT) / (CL_TOP - CL_BOT));
            // multiple scattering octaves (Wrenninge)
            vec3 S = vec3(0.0);
            float a = 1.0, b = 1.0, cph = 1.0;
            for (int k = 0; k < 3; k++) {
                float ph = mix(phaseHG(mu, 0.65 * cph), phaseHG(mu, -0.2 * cph), 0.25) * 4.0 * PI;
                S += uSunCol * a * ph * exp(-od * b);
                a *= 0.55; b *= 0.3; cph *= 0.5;
            }
            float powder = 1.0 - 0.55 * exp(-d * 12.0);
            S *= powder;
            // (r4 L10) (0.9 + 1.6 hf) x 1.4 -> (0.8 + 2.2 hf) x 1.45: greyer bases under bright tops (target: base / top
            // 0.70-0.85); (0.45 + 1.9 hf) x 1.2 turned the whole cloud grey (r5 / r10 cloud / sky 1.5 vs 2.0-2.8)
            S += uSkyAmb * (0.8 + 2.2 * hf) * 1.45 + uSunCol * 0.035;
            float sig = SIG * d;
            float Ts = exp(-sig * dt);
            col += T * S * (1.0 - Ts);
            T *= Ts;
            if (T < 0.01) break;
        }
        t += dt;
    }
    return vec4(col, T);
}

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


float waterGust(vec2 p) {
    vec2 w = uWaterWind;
    vec2 gp = p - w * uTime * 2.2;
    // (r4 C6: + cat's-paw patches ~30-80 m (0.018 / m): the near water is not one flat teal)
    float g = texture(uNoise3, vec3(gp * 0.0016, 0.21)).r * 0.55 + texture(uNoise3, vec3(gp * 0.0051, 0.63)).r * 0.27
            + texture(uNoise3, vec3(gp * 0.018 - uTime * 0.01, 0.47)).r * 0.18;
    return smoothstep(0.45, 0.75, g);     // (r4 W5: 0.38 / 0.72 -> fewer, smaller gust patches)
}

// wave slopes filtered by the ANISOTROPIC pixel footprint: across the view it is t * pixAng, along the
// view t * pixAng / |rd.y| (up to ~300x longer at grazing angles). The old isotropic footprint (with
// |rd.y| clamped to 0.02) kept 0.3-3 m ripples unresolved-but-unfiltered on the far water: their
// back-facing facets were then flipped toward the viewer and the reflection jumped to ~20 deg up
// (the pale band along the far shore, film shots 0 / 1).
// fwd = horizontal view direction (unit), fa / fl = footprint across / along it (m).
// sv = slope variance of the filtered-out waves projected on fwd (the unresolved roughness that
// smears the reflection vertically), gust = wind patch strength.
vec3 waterNormalA(vec2 p, vec2 fwd, float fa, float fl, out float gust, out float sv) {
    vec2 w = uWaterWind;
    gust = waterGust(p);
    vec2 side = vec2(-fwd.y, fwd.x);
    vec2 grad = vec2(0.0);
    float amp = mix(0.12, 0.85, gust) * uWaveStrength;     // (r4 W5: gust floor 0.18 / 1.0 -> 0.12 / 0.85)
    sv = 0.0;
    for (int i = 0; i < 22; i++) {
        float fi = float(i);
        // (r4 W10: wavelength jitter 0.9-1.1 - the exact geometric ladder of near-aligned sines made a grating)
        float lambda = 5.0 * pow(0.80, fi) * (0.9 + 0.2 * hash11(fi * 5.1));   // 5 m .. ~4 cm
        float k = TAU / lambda;
        // (r4 W10: +-75 deg spread below 1.5 m (was +-46): no regular diagonal comb of sky streaks)
        float ang = (hash11(fi * 7.31 + 0.5) - 0.5) * mix(1.6, 2.6, smoothstep(1.6, 1.4, lambda));
        vec2 d = rot2(ang) * w;
        float A = lambda * mix(0.0035, 0.0075, smoothstep(1.5, 0.15, lambda)) * (0.55 + 0.9 * hash11(fi * 3.17 + 1.0));
        // (r4 W5: calmer shore views - waves over 1.5 m at 60 % (the marbled swell from above; 35 % left the mid water glassy, hp 0.19 vs a 0.34-0.44 target), 0.15-0.6 m ripples x0.7
        // (the crumpled-foil chop near the shore))
        A *= mix(0.6, 1.0, smoothstep(1.5, 0.6, lambda)) * mix(1.0, 0.7, smoothstep(0.7, 0.55, lambda) * smoothstep(0.12, 0.17, lambda));
        float ca = dot(d, fwd), cs = dot(d, side);
        float footD = sqrt(cs * cs * fa * fa + ca * ca * fl * fl);
        // (r2b: filter from 2 footprints, full at 6: the 1.2..3.5 ramp kept waves down to ~1 px, which
        // flickered frame to frame (shot 8: 10.5 % of the water pixels changed > 20 levels per frame))
        float filt = smoothstep(2.0, 6.0, lambda / footD);
        float sl = A * k;
        sv += 0.5 * sl * sl * (1.0 - filt * filt) * ca * ca;
        if (filt <= 0.0) continue;
        float omega = sqrt(9.81 * k + 7.28e-5 * k * k * k);
        float ph = k * dot(d, p) - omega * uTime + hash11(fi * 1.7) * TAU;
        float s = sin(ph), c = cos(ph);
        grad += d * (A * k * (c + 0.25 * sin(2.0 * ph))) * filt;
    }
    // irregular small-scale chop (isotropic noise: filtered by the larger, along-view footprint)
    float fm = max(fa, min(fl, fa * 40.0));
    vec2 cp = p * 1.3 - w * uTime * 0.7;
    float f1 = smoothstep(2.0, 6.0, 0.85 / fm), f2 = smoothstep(2.0, 6.0, 0.33 / fm);
    if (f1 > 0.0) {
        vec3 g1 = gnoised(cp * 0.9 + vec2(0.0, uTime * 0.06));
        vec3 g2 = f2 > 0.0 ? gnoised(cp * 2.3 + vec2(uTime * 0.09, 3.7)) : vec3(0.0);
        grad += (g1.yz * (1.17 * 0.010) * f1 + g2.yz * (2.99 * 0.004 * 0.7) * f2) * mix(0.3, 1.0, gust);
    }
    sv += (0.5 * (0.0117 * 0.0117) * (1.0 - f1 * f1) + 0.5 * (0.012 * 0.012 * 0.7 * 0.7) * (1.0 - f2 * f2)) * 0.5 * mix(0.09, 1.0, gust);
    grad *= amp;
    sv *= amp * amp;
    return normalize(vec3(-grad.x, 1.0, -grad.y));
}

// animated caustic pattern on the lake bed
// (r4 W6 / R8 / W3: a fine, soft net (~0.2-0.35 m cells) instead of ~1 m pool-floor cells; none at zero
// depth (the white kerb along the cliff foot), fading with depth; fp = pixel footprint (m): unresolved -> the mean)
float caustics(vec2 p, float depth, float fp) {
    float vis = 1.0 - smoothstep(0.025, 0.05, fp);
    float c = 2.0 * 0.095;                              // (the mean of the two octaves)
    if (vis > 0.0) {
        vec2 q = p * 3.0;
        float cc = 0.0;
        for (int i = 0; i < 2; i++) {
            vec2 o = vec2(uTime * 0.33, uTime * 0.21) * (i == 0 ? 1.0 : -1.3);
            vec3 v = voronoi2(q * (i == 0 ? 1.0 : 1.6) + o);
            cc += pow(saturate(1.0 - (v.y - v.x) * 2.2), 10.0);
        }
        c = mix(c, cc, vis);
    }
    return c * 0.3 * smoothstep(0.05, 0.5, depth) * exp(-depth * 0.4);
}

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
float treeReach(float R, float H, float seed) {
    float fh = fract(seed * 83.71);
    return R * 1.42 + 0.7 + 0.012 * H + ((fh >= 0.04 && fh < 0.07) ? 0.07 * H : 0.0);
}
// conservative distance bound of a raw tree (snag of a broken top <= 2.2 m above the crown)
float treeRawBound(vec3 p, ivec2 cell, vec4 d, vec4 hs, vec2 pos) {
    vec2 m = treeRawMax(cell, d, hs);
    float reach = treeReach(m.y, m.x, fract(hs.w * 7.31 + hs.x * 3.7));
    return max(length(p.xz - pos) - reach, max(p.y - d.x - m.x - 2.4, d.x - 1.0 - p.y));
}

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
float treeRelH(ivec2 cell, float dens) {
    vec4 hs = hash42i(cell);
    float f = mix(0.55, 1.30, fract(hs.x * 13.7 + hs.w * 5.1)) / 0.925;
    float yg = fract(hs.w * 31.7 + hs.y * 2.3);
    float ygF = mix(0.40, 0.16, smoothstep(0.20, 0.40, dens));
    float dom = step(0.89, fract(hs.y * 23.7 + hs.z * 5.3));
    f *= yg < ygF ? mix(0.30, 0.62, yg / ygF) : mix(1.0, 1.30, dom);
    float mc = fract(hs.x * 41.3 + hs.z * 7.9);
    if (yg >= ygF && dom < 0.5 && mc < 0.25) f *= mix(0.65, 0.85, mc / 0.25);
    return f;
}
// (trees6) mid / far model only: per-tree crown slenderness (x0.78-1.0: narrow pencil spruce beside full
// cones, so the distant spires are not all one shape; dark gaps between the narrow ones). Ramped in by camera
// distance well beyond the near / mid switch (<= 150 m), a reduction only (all bounds stay valid)
float treeMidSlim(Tree tr, float dc) {
    float h = fract(tr.seed * 151.3);
    return 1.0 - 0.22 * smoothstep(150.0, 300.0, dc) * h * sqrt(h) * (1.0 - 0.5 * tr.larch);
}

bool getTree(ivec2 cell, out Tree tr) {
    vec4 d, hs; vec2 pos;
    if (!treeRaw(cell, d, hs, pos)) { tr = Tree(vec3(0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0); return false; }
    return treeFinish(cell, d, hs, pos, tr);
}

// per-tree / per-whorl / per-branch random numbers (one integer hash each)
uint treeKey(Tree tr) { return uint(tr.seed * 16777215.0); }
vec4 whorlRand(uint key, float k) {
    uint h = hashu(key ^ (uint(int(k) + 8192) * 0x9E3779B1u));
    return vec4(uvec4(h, h >> 8u, h >> 16u, h >> 24u) & 255u) * (1.0 / 255.0);
}
vec3 branchRand(uint key, float k, float j) {
    uint h = hashu(key ^ (uint(int(k) + 8192) * 0x9E3779B1u) ^ (uint(int(j) + 64) * 0x85EBCA77u));
    return vec3(uvec3(h, h >> 10u, h >> 20u) & 1023u) * (1.0 / 1023.0);
}

// direction (angle) of the open / light side of a crown: branches are longer there, and in 40 % of
// the trees the side facing the stand has lost many of its branches
float treeOpenDir(Tree tr) { return fract(tr.seed * 19.7) * TAU; }

// length factor of one branch: own variety (a narrow range: few long arms sticking out - those read as
// araucaria up close) and a crown asymmetry. Larch: irregular, very unequal branch lengths. Shared by the
// near and mid model (they agree in outline).
float branchLen(Tree tr, vec3 br, float ba) {
    float asy = fract(tr.seed * 47.9) * 0.35;
    float l = mix(0.70, 1.15, br.y) + 0.10 * step(0.88, fract(br.y * 17.3 + br.z * 3.1));
    l = mix(l, mix(0.35, 1.30, br.y), tr.larch);
    return l * (1.0 + asy * cos(ba - treeOpenDir(tr)));
}

// openness: 0 = dense crown (shaded-stand spruce), 1 = open, airy crown (sunny subalpine tree)
float treeOpen(Tree tr) { return fract(tr.seed * 29.31) * (1.0 - 0.5 * tr.dens); }

// missing / broken branch (a gap in the crown); shared by the near and mid model
// sparse whorl: half its branches gone (a horizontal gap in an old, open crown)
float whorlGap(Tree tr, vec4 wr) { return step(fract(wr.x * 5.31 + wr.w * 2.73), 0.04 + 0.07 * treeOpen(tr)) * 0.5; }
// sparse, see-through larch: only the most open-grown ones; most larches are full soft cones
float larchSparse(Tree tr) { return tr.larch * smoothstep(0.62, 0.82, treeOpen(tr)); }
bool branchMissing(Tree tr, vec3 br, float ba, float op, float wg) {
    float pm = (0.02 + 0.03 * op) + 0.04 * larchSparse(tr) + wg;
    if (fract(tr.seed * 61.7) < 0.40) pm += 0.35 * smoothstep(0.1, 0.8, -cos(ba - treeOpenDir(tr)));
    return br.z < pm;
}

// dead branches in the lower live crown (open-grown old spruce / fir: grey, lichen-covered twigs on the
// lower trunk): probability for a whorl at relative crown height thk; shared by the near and mid model
float branchDeadP(Tree tr, float thk) {
    // (r5 T2 / T14: x0.7, was x1.4 - the long dead zone read as black whips; none on open-grown trees < 15 m)
    float dz = (0.05 + 0.22 * treeOpen(tr)) * step(mix(15.0, 9.0, smoothstep(0.30, 0.50, tr.dens)), tr.H) * (1.0 - tr.larch) * 0.7;
    return 0.85 * (1.0 - smoothstep(0.0, max(dz, 1e-3), thk)) * step(1e-3, dz);
}
bool branchDead(vec3 br, float dP) { return fract(br.x * 31.7 + br.y * 7.3) < dP; }

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
float whorlGroup(Tree tr, float k) {
    float g = floor((k + floor(fract(tr.seed * 7.7) * 3.0)) / 3.0);
    float h = hash11(g * 1.37 + tr.seed * 91.0);
    return mix(0.72, 1.22, h);
}

float whorlSpacing(Tree tr) {
    // (r5 T2: larch whorls x0.8 closer - branches every 0.3-0.5 m)
    float lf = mix(1.10, 1.40, larchSparse(tr)) * 1.15 * 0.8;
    return clamp(tr.H * 0.022, 0.34, 0.62) * mix(0.88, 1.12, fract(tr.seed * 5.31)) * mix(0.70, lf, tr.larch) * mix(1.12, 0.70, treeFir(tr));
}
// whorl height jitter (fraction of the spacing) and missing whorls (larch: irregular, open crowns)
// (r4: spruce x1.12 spacing, jitter .45: fewer, more distinct tiers on spruce at 50-150 m)
float whorlJit(Tree tr) { return mix(0.45, 0.9, tr.larch); }
bool whorlSkip(Tree tr, vec4 wr) { return tr.larch > 0.5 && fract(wr.y * 7.7 + wr.z * 3.3) < 0.04 + 0.08 * larchSparse(tr); }
float whorlBranches(Tree tr, vec4 wr) { return mix(floor(mix(14.0, 10.0, treeOpen(tr))), floor(mix(9.0, 6.0, larchSparse(tr)) * 0.8), tr.larch) + floor(wr.z * 3.0); }

// branch centre line: height offset at x = u/L (0 trunk .. 1 tip). droop D, upward angle e.
// mono 0: the quadratic sags to its lowest point at x = 0.6, then the tip curves up again (to ~55 % depth);
// mono 1: the outer branch droops more and more (no upturned 'monkey-tail' tips)
float branchSag(float x, float L, float D, float e, float mono) {
    return L * (e * x - D * mix(2.16 * x - 1.8 * x * x, 0.20 * x + 0.45 * x * x, mono));
}
// droop shape: spruce / fir droop, larch keeps its horizontal branches with up-curved tips
float branchMono(Tree tr) { return 1.0 - tr.larch; }
float branchDroop(Tree tr, float th, float fir) {
    float D = mix(0.25, 0.05, th) * 0.85;
    D = mix(D, mix(0.12, 0.03, th), fir);
    return mix(D, mix(0.10, 0.03, th), tr.larch);
}
// (larch: nearly horizontal branches with hanging twig curtains)
float branchRise(Tree tr, float th) { return mix(0.0, 0.32, th * th) + tr.larch * 0.10; }

// camera distance of a tree (nearest point of its axis)
float treeCamDist(Tree tr) {
    vec3 c = uCamPos - tr.base;
    return max(length(vec3(c.x, c.y - clamp(c.y, 0.0, tr.H), c.z)) - tr.R * 0.5, 0.5);
}
// near model or not: by distance (per-tree jitter), dissolved over a band: inside it each tree
// picks the model per sample pass (fixed for the whole pass, so a ray never mixes models)
bool treeIsNear(Tree tr, float dc) {
    int dm = treeDebugMode();
    if (dm == 20 || (dm >= 30 && dm <= 32)) return true;
    if (dm == 21 || (dm >= 33 && dm <= 35)) return false;
    float x = dc / (TREE_NEAR_D * mix(0.8, 1.2, fract(tr.seed * 71.3)));
    if (x <= 0.8) return true;
    if (x >= 1.25) return false;
    float h = fract(fract(tr.seed * 613.7) + float(treeSampleIdx()) * 0.6180340);
    return h >= smoothstep(0.8, 1.25, x);
}

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
float vnoise1(float x) {
    float i = floor(x), f = x - i;
    return mix(hash11(i), hash11(i + 1.0), f * f * (3.0 - 2.0 * f));
}

float sdFrond(float u, float v, float dy, float L, float Wm, float hang, float x0, float P, float rS,
              float rT, float bare, float gapP, float seed, float sh3, float L3, float tuft, float sj, float curlK, float spread, float clk, float larchF, out vec3 fi) {
    // the two sides of the spray alternate (half a period apart) and use their own random numbers
    float side = v < 0.0 ? 1.0 : 0.0;
    float av = abs(v);
    seed += 0.5 * side;
    // index of the branchlet through this point: branchlets leave the axis at the mean forward angle;
    // the radial distance from the axis does not depend on the (individual) plane of a branchlet
    const float cot = 0.78;                           // mean forward angle ~52 deg
    float rho = length(vec2(av, dy));
    float fi0 = (u - rho * cot) / P - seed;
    float i0 = floor(fi0 + 0.5);
    float i1 = i0 + (fi0 > i0 ? 1.0 : -1.0);
    float d = 1e9;
    fi = vec3(0.0, 0.0, 1.0);
    uint sk = uint(seed * 65535.0) ^ (uint(side) * 0x68E31DA4u);
    float sIdx = float(treeSampleIdx()) * 0.6180340;
    for (int k = treeZero(); k < 2; k++) {
        float ii = k == 0 ? i0 : i1;
        uint hh = hashu(uint(int(ii) + 4096) * 0x9E3779B1u ^ sk);
        vec4 hr = vec4(uvec4(hh, hh >> 8u, hh >> 16u, hh >> 24u) & 255u) * (1.0 / 255.0);
        float u0 = (ii + seed + (hr.w - 0.5) * 1.6) * P;
        float xs = (u0 / L - x0) / (1.0 - x0);
        if (xs <= 0.0 || xs >= 1.0) continue;
        if (hr.y < gapP * (1.0 + 1.5 * xs)) continue;     // missing branchlet -> hole (more toward the tip)
        // spray outline: widest at ~35 %, tapering to a point at the tip
        // (r5 T2: full sprays from ~20 % of the branch on, was 50 %: no bare inner half / tip-only tufts)
        float env = smoothstep(0.0, 0.20, xs) * pow(saturate((1.0 - xs) * 2.3), 0.55);
        // ragged outline: the needles bunch into tufts (0.2-0.4 m) with thin stretches between them;
        // clk: clumps of foliage (~0.4 m) with nearly bare stretches of the branch between them
        float cn = vnoise1(u0 * 2.6 + seed * 37.0);
        env = min(env * mix(0.55 + 0.9 * vnoise1(xs * 6.0 + seed * 37.0), 1.5 * smoothstep(0.22, 0.75, cn), clk), 1.1);
        // own plane (slope below horizontal, tan): around the mean hang, some shoots rise, some hang
        vec2 ci = normalize(vec2(1.0, hang + (hr.z - 0.55) * spread));
        float wi = av * ci.x - dy * ci.y;             // along the branchlet's direction (in its plane)
        float hi = av * ci.y + dy * ci.x;             // above its plane
        vec2 fa = normalize(vec2(mix(0.25, 1.50, hr.w) * mix(1.0, 0.35, larchF), 1.0));   // (cos, sin) of the forward angle
        // (larch: very unequal shoots, a few long ones and many short tufts)
        // (spruce: many short shoots, a few long ones)
        float li = Wm * env * mix(mix(0.45, 1.15, hr.x), mix(0.30, 1.10, hr.x), larchF) / fa.y;
        if (li < 0.03) continue;
        float du = u - u0;
        float sp = du * fa.x + wi * fa.y;             // along the branchlet (unclamped)
        float s = clamp(sp, 0.0, li);
        float sf = s / li;
        // branchlets curve up toward their tips (the fresh shoots turn to the light)
        float hz = hi - curlK * li * sf * sf * (0.3 + hr.x);
        float lat = wi * fa.x - du * fa.y;            // in-plane offset from the branchlet axis
        // branchlets bend sideways in their plane (not straight ribs)
        lat += (hr.z - 0.5) * 0.80 * li * sf * sf;
        // needle brush thinning to the tip; bare twig on the inner part
        float bs = bare * (1.0 - 0.6 * xs);
        float ndl = smoothstep(bs - 0.10, bs + 0.10, sf);
        float jr = 1.0 + sj * (fract(hr.x * 7.13 + hr.y * 3.71 + sIdx) - 0.5);
        float rr = mix(rT, (rS * (1.0 - 0.40 * sf) + (1.0 - sh3) * L3 * 0.16 * (1.0 - 0.6 * sf)) * jr, ndl);
        // larch: needle rosettes on short shoots, in tufts every 0.15-0.25 m
        // (r5 T2: rosettes every 3-4 cm, was 6-8: dense tufts along the whole twig)
        float tri = 1.0 - abs(fract(s * mix(24.0, 32.0, hr.y) + hr.z) - 0.5) * 2.0;
        rr *= mix(1.0, 0.30 + 1.4 * tri * tri * tri, tuft);
        {
            // (spruce / fir close up: needle tufts, the shoots are serrated rather than smooth sausages)
            // (r5 T11: deeper serration, 0.25 + 1.4 tz^2: lacy shoots, not seaweed straps)
            float tz = 1.0 - abs(fract(s * 36.0 + hr.y * 3.0) - 0.5) * 2.0;
            rr *= mix(1.0, 0.25 + 1.40 * tz * tz, (1.0 - larchF) * sh3);
        }
        float dk = length(vec3(sp - s, lat, hz)) - rr;
        if (sh3 > 0.0 && L3 > 0.0) {
            // side shoots (3rd order) in the branchlet plane: alternating, swept forward ~50 deg,
            // shorter toward the tip of the branchlet -> lacy, serrated sprays
            float al = abs(lat);
            float so = lat < 0.0 ? 0.5 : 0.0;
            float Q = 0.025 * mix(0.65, 1.35, fract(hr.z * 13.7 + hr.w * 3.1));
            float j3 = floor((sp - al * 0.83) / Q - so + 0.5);
            float s0 = (j3 + so + (fract(j3 * 0.618 + hr.y * 7.1) - 0.5) * 0.5) * Q;
            float f0 = s0 / li;
            float l3 = L3 * sh3 * (1.0 - 0.65 * f0) * (0.55 + 0.9 * fract(j3 * 0.618 + hr.z * 5.3))
                     * smoothstep(bs, bs + 0.12, f0) * step(f0, 0.94);
            if (l3 > 0.006) {
                float ds = sp - s0;
                float t3 = clamp(ds * 0.64 + al * 0.77, 0.0, l3);
                float d3 = length(vec3(ds - t3 * 0.64, al - t3 * 0.77, hz)) - rS * 0.62 * jr * (1.0 - 0.4 * t3 / l3);
                dk = min(dk, d3);
            }
        }
        // (|fi.z| 0.5..1: per-branchlet tone, so the sprays are not a regular comb of equal shoots)
        if (dk < d) { d = dk; fi = vec3(xs, sf, (hi >= 0.0 ? 1.0 : -1.0) * (0.5 + 0.5 * fract(hr.x * 7.31 + hr.z * 3.17 + hr.w * 1.93))); }
    }
    return d;
}

// ---------------------------------------------------------------------------
// near model: individual branches with fronds
// fat: true pixel footprint (m) - sub-pixel shoots are widened so the sprays keep their coverage and
// the cost is cut once single shoots are sub-pixel; dc: camera distance; fine: needle detail weight
// ---------------------------------------------------------------------------
float sdTreeNear(vec3 q, Tree tr, float fat, float dc, float fine, bool info, out vec4 inf) {
    inf = vec4(0.0);
    float fir = treeFir(tr);
    float r = length(q.xz);
    float y = q.y;
    float op = treeOpen(tr);
    int top = treeTop(tr);
    float wide = 0.12 * max(fat - 0.015, 0.0);
    // (larch: fuzzy rosette ropes keep their coverage once the rosettes are sub-pixel)
    wide = mix(wide, 0.18 * max(fat - 0.010, 0.0), tr.larch);
    // (r5 T4: half the sub-pixel widening in the spruce / fir spire - at 60-120 m the widened sprays filled the
    // top into a solid column, spire tops should be thin drooping branches with sky between them)
    wide *= 1.0 - 0.5 * smoothstep(0.60, 0.90, y / tr.H) * (1.0 - tr.larch);
    // trunk with root flare, tapering into the leader (a broken tree keeps its girth up to the break)
    // (r5 T9: ~0.48 m at the base of a 25 m tree, was 0.32 - target trunks 0.45-0.6 m; tapers faster
    // inside the crown)
    float rt = max((0.03 + 0.0085 * tr.H) * pow(saturate(1.0 - y / crownTop(tr)), 1.8) * mix(1.0, 1.1, tr.larch), 0.012);
    rt += (0.03 + 0.009 * tr.H) * 0.9 * exp(-max(y, 0.0) * 2.2);
    float dt = max(r - rt, max(-y - 1.0, y - tr.H + 0.15)) * 0.9;
    if (info) inf = vec4(1.0, 0.0, 0.0, 0.0);
    if (top == 1) {
        // broken top: a bare grey snag above the last live whorls
        float hsn = mix(0.6, 2.2, fract(tr.seed * 31.3));
        float ds = max(r - rt * (1.0 - 0.7 * saturate((y - tr.H) / hsn)), max(tr.H - 0.3 - y, y - tr.H - hsn)) * 0.9;
        if (ds < dt) { dt = ds; if (info) inf = vec4(2.0, 0.0, 0.0, 0.0); }
    } else {
        // leader: needle-covered top shoot above the last whorl
        float yl = saturate((y - tr.H + 1.3) / 1.3);
        float dl = max(r - mix(0.06, 0.02, yl) * mix(1.0, 0.8, tr.larch) - wide, max(tr.H - 1.3 - y, y - tr.H - 0.05)) * 0.9;
        if (dl < dt) { dt = dl; if (info) inf = vec4(0.0, 1.0, 1.0, 1.0); }
    }
    // early out: well outside the reach of the branches (L <= 1.25 Lk + 0.4, sprays <= 1.08 L)
    float renv = crownR(tr, y, fir);
    float Lb = max(crownR(tr, y - 0.8, fir), crownR(tr, y + 2.0, fir));
    float dEnv = max((r - Lb * 1.45 - 0.7) * 0.9, max(y - tr.H - 0.3, min(tr.yb, 1.2) - 1.6 - y));
    if (dEnv > 0.4) return min(dEnv, dt);
    float d = dt;
    uint key = treeKey(tr);
    float ang = atan(q.z, q.x);
    float hb = max(tr.H - tr.yb, 1.0);
    float s = whorlSpacing(tr);
    float jit = whorlJit(tr);
    float phase = fract(tr.seed * 3.77);
    // un-sag: estimate the whorl height the branches passing here grew from
    float th = saturate((y - tr.yb) / hb);
    float Lest = max(renv, 0.25);
    float xe = saturate(r / Lest);
    float ys = y - branchSag(xe, Lest, branchDroop(tr, th, fir), branchRise(tr, th), branchMono(tr));
    float kf = (ys - tr.yb) / s - phase;
    float k0 = floor(kf - 0.25);
    // frond parameters per species
    float P = mix(mix(0.062, 0.058, fir), 0.075, tr.larch);
    // (r4: larch .017, was .026: thin twigs with fine rosettes instead of fuzzy ropes)
    // (r5 T2: larch .021 - the r4 twigs read as bare dead whips)
    float rS = mix(mix(0.013, 0.012, fir), 0.021, tr.larch) + wide;
    float rT = 0.0045 + 0.25 * max(fat - 0.004, 0.0);
    float bare = 0.25 * (1.0 - smoothstep(0.006, 0.02, fat));
    float lsp = larchSparse(tr);
    // (r5 T2: larch .06 + .03 op, was .14-.24: tufts along the whole twig)
    float gapP = mix(0.05 + 0.09 * op, 0.06 + 0.03 * op, tr.larch);
    float sh3 = 1.0 - smoothstep(0.007, 0.016, fat);
    // (r5 T11: longer side shoots, .075 / .062, was .055 / .045: lacy herringbone sprays)
    float L3 = mix(mix(0.075, 0.062, fir), 0.0, tr.larch);
    float tuft = mix(0.7, 1.0, fine) * tr.larch;
        float curlK = mix(mix(0.22, 0.18, fir), 0.10, tr.larch);
    // comb spruce: the branchlets hang from the branch as a curtain
    // (r4: few comb curtains, x0.3 - spruce carry flat layered fans)
    float comb = (1.0 - fir) * (1.0 - tr.larch) * smoothstep(0.55, 0.85, fract(tr.seed * 37.3)) * 0.3;
    // spread of the branchlet planes about the mean hang: layered sprays, flat fans (r4: spruce x0.6), not
    // bottle-brush ropes; larch twigs all hang (curtains)
    // (fir 2.4: thicker fans, shot 3 fir less of a flat-plate fishbone)
    // (r5 T2: larch 1.0, was 1.6)
    float spread = mix(mix(mix(1.0, 1.25, comb) * 0.6, 2.4, fir), 1.0, tr.larch);
    // clumped foliage along the branches (open old crowns: thin bare twig stretches between clumps)
    // (r5 T2: spruce <= .20, larch .35, was .42 / 1.0: the bare stretches read as dead branches)
    float clk = mix(mix(0.07, 0.20, op), 0.35, tr.larch);
    // once single branches are only a few pixels wide only the nearest branch of a whorl is evaluated;
    // dissolved per tree and sample pass over a footprint band (no pop)
    float hS = fract(fract(tr.seed * 917.3) + float(treeSampleIdx()) * 0.6180340);
    float nmF = mix(0.018, 0.012, tr.larch);
    int nm = fat > nmF + 0.012 * hS ? 1 : 2;
    for (int i = treeZero(); i < 3; i++) {
        // the whorl just above first (its sprays hang down through here), so the bounds cull the rest
        float k = k0 + (i == 0 ? 1.0 : (i == 1 ? 2.0 : 0.0));
        vec4 wr = whorlRand(key, k);
        if (whorlSkip(tr, wr)) continue;
        float yk = tr.yb + (k + phase + (wr.x - 0.5) * jit) * s;
        if (yk > tr.H - 0.2) continue;
        bool deadW = yk < tr.yb;
        if (deadW && (yk < 1.2 || tr.larch > 0.5)) continue;
        float thk = saturate((yk - tr.yb) / hb);
        float dP = branchDeadP(tr, thk);
        float Lk = deadW ? mix(0.35, 1.6, wr.y) * saturate((yk - 1.0) / 2.0 + 0.3)
                         : crownR(tr, yk, fir) * mix(0.88, 1.12, wr.y) * whorlGroup(tr, k) + 0.10;
        float wg = whorlGap(tr, wr);
        float nb = whorlBranches(tr, wr);
        float w = TAU / nb;
        float phi = wr.w * TAU;
        float a = (ang - phi) / w;
        float j0 = floor(a + 0.5);
        float j1 = j0 + (a > j0 ? 1.0 : -1.0);
        float Dl = branchDroop(tr, thk, fir);
        float El = branchRise(tr, thk);
        // hang angle of the branchlets: spruce sprays hang below the axis as a fringe (more low in the
        // crown), comb spruce curtains, flat fir sprays, near-vertical larch twigs
        // (r4: spruce x0.5 - flatter, layered fans instead of hanging fringes)
        float hang0 = mix(mix(0.85, 0.30, thk) * 0.5, 0.10, fir);
        hang0 = mix(hang0, mix(0.9, 0.4, thk), comb);
        // (r5 T2: larch twigs spread (0.5), was 1.6 - a hanging curtain seen edge-on from the side)
        hang0 = mix(hang0, 0.5, tr.larch);
        float x0 = mix(min((mix(0.20, 0.08, thk) + 0.06) * 0.6, 0.6), 0.18, tr.larch) * mix(1.0, 0.6, fir) * (1.0 - smoothstep(0.62, 0.78, thk));
        float wfB = mix(mix(0.68, 0.55, fir), 0.45, tr.larch) * mix(1.0, 0.90, op * (1.0 - tr.larch));
        for (int m = 0; m < nm; m++) {
            float j = m == 0 ? j0 : j1;
            vec3 br = branchRand(key, k, mod(j, nb));
            float ba = phi + (j + (br.x - 0.5) * 0.45) * w;
            bool dead = deadW || branchDead(br, dP);
            // (r5 T4: a third of the branches missing toward the top - a ragged, see-through spire)
            if (dead ? br.z < (deadW ? 0.45 : 0.15) : branchMissing(tr, br, ba, op, wg + 0.35 * smoothstep(0.55, 0.92, thk) * (1.0 - tr.larch))) continue;   // missing / broken branch -> gap
            float D = dead ? 0.28 : Dl;
            float E = dead ? 0.0 : El;
            vec2 dir = vec2(cos(ba), sin(ba));
            float u = dot(q.xz, dir);
            float v = dot(q.xz, vec2(-dir.y, dir.x));
            // kinks: branch axes wander a few cm every 0.3-0.6 m (no straight rods)
            float wA = 0.14 * saturate(u * 2.0);
            v += wA * (vnoise1(u * 2.3 + br.x * 31.0) - 0.5);
            // (r5 T2: short dead stubs, x0.20-0.45, was 0.30-0.70)
            float L = dead ? Lk * (deadW ? mix(0.80, 1.12, br.y) : mix(0.20, 0.45, br.y)) : min(Lk * branchLen(tr, br, ba), min(mix(1.25 * Lk + 0.4, 1.08 * Lk + 0.15, smoothstep(0.45, 0.80, thk)), 4.2));
            // (no long arms in the upper crown: the reach stays within the mid model's 1.25 x envelope
            // cap (+ a little), so spruce tops stay narrow spires)
            // (r5 T10: lower crown x1.60 + 0.3, was x1.40 + 0.25: 1.5-2.5 m drooping lower branches)
            if (!dead) L = min(L, crownR(tr, yk, fir) * mix(1.60, 1.05, smoothstep(0.30, 0.75, thk)) * mix(1.0, 1.25, tr.larch) + 0.3);
            float x = u / L;
            float xc = saturate(x);
            // every branch its own droop, rise and tip shape: some sag and turn their tips up (J), some
            // hang straight, some rise; branches curve sideways in plan (no straight radial spokes)
                        float Db = D * (1.0 + (fract(br.z * 7.1) - 0.5) * mix(1.15, 0.65, tr.larch));
            float Eb = E + (fract(br.y * 3.7 + br.z * 1.3) - 0.5) * 0.22 * (1.0 - tr.larch);
            float monoB = branchMono(tr) * (fract(br.x * 5.3 + br.z * 1.7) < 0.45 ? 0.25 : 1.0);
            float yc = yk + (fract(br.x * 13.1) - 0.5) * mix(0.85, 1.0, tr.larch) * s + branchSag(xc, L, Db, Eb, monoB);
            v -= L * (fract(br.y * 11.3 + br.x * 2.9) - 0.5) * 2.0 * 0.16 * xc * xc;
            float dy = y - yc - wA * (vnoise1(u * 2.9 + br.y * 17.0) - 0.5);
            {
                // the frond is rolled about the branch axis (+-35 deg spruce, +-60 deg larch): sprays are
                // seen edge-on, face-on and from below, not as a stack of identical chevrons
                float ro = (fract(br.y * 5.7 + br.x * 2.3) - 0.5) * mix(1.2, 2.1, tr.larch);
                float cr = cos(ro), sr = sin(ro);
                float v2 = v * cr + dy * sr;
                dy = dy * cr - v * sr;
                v = v2;
            }
            float hang = hang0 * mix(0.75, 1.25, br.x);
            // (r5 T4: narrower fronds in the spruce / fir spire - see-through top)
            float Wm = min(L * wfB, 1.15) * (1.0 - 0.35 * smoothstep(0.55, 0.90, thk) * (1.0 - tr.larch));
            // reach of the sprays below / above the axis: steepest hanging / rising branchlet plane
            // (reach <= 1.27 Wm) plus the upward curl of the branchlet tips
            float sD = hang + 0.45 * spread, sU = max(0.55 * spread - hang, 0.0);
            float cdep = Wm * 1.27 * sD * inversesqrt(1.0 + sD * sD) + 0.08;
            float cup = Wm * (1.27 * sU * inversesqrt(1.0 + sU * sU) + 2.2 * curlK) + 0.08;
            // cheap bound of everything this branch carries
            float bnd = max(max(abs(v) - Wm * 1.3 - 0.06, max(-u, u - L) - 0.15), max(dy - cup, -dy - cdep)) * 0.7 - 0.05;
            if (bnd > d) continue;
            // wood: thin tapering branch axis
            // (r5 T11: x1.5 - the grey branch wood reads inside the sprays)
            float rb = (0.006 + 0.005 * L) * (1.0 - 0.75 * xc) * (dead ? 0.55 : 1.0) * 1.08;
            float dw = length(vec3(max(max(-u, u - (dead ? 1.0 : 0.9) * L), 0.0), v, dy)) - rb;
            if (dead) {
                // short dead spurs angled toward the branch tip
                float sp = x * L - abs(v) * 1.3;
                float spd = (fract(sp / 0.18 + br.x) - 0.5) * 0.18 * 0.62;
                float tw = max(length(vec2(spd, dy)) - max(rb, 0.004) * 0.7, abs(v) - 0.22 * L * (1.0 - xc));
                tw = max(tw, max(0.12 - x, x - 0.92) * L);
                dw = min(dw, tw);
                if (dw < d) { d = dw; if (info) inf = vec4(2.0, xc, 1.0, 0.0); }
                continue;
            }
            vec3 fi;
            float df;
            df = sdFrond(u, v, dy, L, Wm, hang, x0, P * mix(0.9, 1.1, br.y), rS, rT, bare, gapP,
                         fract(br.y * 0.73 + br.z * 0.19), sh3, L3, tuft, 0.15, curlK, spread, clk, tr.larch, fi) * 0.85;
            // leading shoot: needle brush along the whole branch axis beyond its bare inner part, so
            // the wood shows only close to the trunk
            float rl = max(rS * 1.4, rS * 1.3 + (1.0 - sh3) * L3 * 0.15);
            // tufted, not a tube; in clumped crowns the axis is bare (thin dark twig) between the clumps
            rl *= mix(0.55 + 0.9 * vnoise1(u * 4.0 + br.y * 17.0),
                      1.3 * smoothstep(0.30, 0.75, vnoise1(u * 2.6 + fract(br.y * 0.73 + br.z * 0.19) * 37.0)), clk);
            float dl = (length(vec3(max(max(x0 * L - u, u - L), 0.0), v, dy)) - rl * (1.0 - 0.35 * xc)) * 0.9;
            if (dl < df) { df = dl; fi = vec3(xc, mix(0.3, 0.9, smoothstep(0.85, 1.0, xc)), 0.75); }
            float dd = min(df, dw);
            if (dd < d) {
                d = dd;
                if (info) inf = df < dw ? vec4(0.0, fi) : vec4(2.0, xc, 0.0, 0.0);
            }
        }
    }
    // dense inner crown (old shoots and twigs, nearly opaque): seen through the gaps between the outer
    // sprays as near-black shade. Small in open crowns, almost none in larch
    float coreK = smoothstep(10.0, 25.0, dc) * mix(0.55, 0.15, op) * (1.0 - tr.larch);
    float hbc = max(tr.H - tr.yb, 1.0);
    float thc = (y - tr.yb) / hbc;
    // (r5 T4: at most 0.45 x the envelope and none in the upper 40 %: the spire top stays see-through;
    // was up to 0.9 renv up to thc 0.95 - opaque cypress columns at 40-150 m)
    float rcMax = min(renv * coreK * 1.65, 0.45 * renv);
    if (coreK > 0.02 && max(r - rcMax, max((thc - 0.60) * hbc, tr.yb + 0.4 - y)) * 0.6 < d) {
        // lumpy (clumps of old shoots), so it never shows as a smooth dark column through the gaps
        float cnz = textureLod(uNoise3, q * vec3(0.55, 0.40, 0.55) + tr.seed * 5.0, 0.0).r;
        float rc = min(renv * coreK * (1.0 - smoothstep(0.35, 0.60, thc)) * (0.35 + 1.3 * cnz * cnz), rcMax);
        float dco = max(r - rc, max((thc - 0.60) * hbc, tr.yb + 0.4 - y)) * 0.6;
        if (dco < d) { d = dco; if (info) inf = vec4(3.0, 0.0, 0.0, -1.0); }
    }
    return d;
}

// ---------------------------------------------------------------------------
// mid model: one drooping skirt per whorl, star-shaped from the per-branch lengths
// lod: 1 = whorl skirts .. 0 = envelope; fp: true pixel footprint (fades the sub-pixel noise)
// ---------------------------------------------------------------------------
float sdTreeMid(vec3 q, Tree tr, float lod, float fp, bool info, out vec4 inf) {
    float fir = treeFir(tr);
    float r = length(q.xz);
    float y = q.y;
    float hb = max(tr.H - tr.yb, 1.0);
    float th = saturate((y - tr.yb) / hb);
    float renv = crownR(tr, y, fir);
    float trr = 0.03 + 0.0085 * tr.H;
    float dt = max(r - trr * (1.0 - 0.75 * saturate(y / tr.H)), max(-y - 1.0, y - tr.H * 0.9));
    float cap = max(y - tr.H, (tr.yb - y) * 0.8);
    // jagged-outline noise amplitudes, faded out before their features get sub-pixel
    float nA = 0.18 * (1.0 - smoothstep(0.12, 0.40, fp));
    // (fine outline noise: 7-28 cm worley cells, faded while they are still >= ~3 px; amplitude kept low
    // so the distance stays near 1-Lipschitz)
    float n2A = 0.12 * (1.0 - smoothstep(0.02, 0.06, fp));
    // seen from afar the sub-pixel outer sprays still cover part of the pixels around the crown: the
    // spires read broader (ramped in slowly, well beyond the near/mid switch)
    float dcm = fp / max(uPixAng, 1e-6);
    float wid = 1.0 + 0.10 * smoothstep(80.0, 220.0, dcm) + -0.20 * smoothstep(220.0, 600.0, dcm);
    float slim = treeMidSlim(tr, dcm);
    wid *= slim;
    // early out well outside the (lumpy) envelope (skirt radius <= 1.25 renv * wid + 0.08)
    float dEnv = (r - renv * 1.25 * wid - 0.1) * 0.55 - 0.6 * nA;
    if (dEnv > 0.5) {
        inf = vec4(0.0, 1.0, -0.22, 2.0);
        return min(max(dEnv, cap), dt);
    }
    float rr = renv;
    float tierV = -0.22;
    float s = whorlSpacing(tr);
    float phase = fract(tr.seed * 3.77);
    float op = treeOpen(tr);
    if (lod <= 0.0) {
        // far: branch lobes and whorl tiers are sub-pixel, keep their average
        rr = renv * 0.95 + 0.08;
    } else {
        uint key = treeKey(tr);
        float Le = max(renv, 0.25);
        float xe = saturate(r / Le);
        float ys = y - branchSag(xe, Le, branchDroop(tr, th, fir), branchRise(tr, th), branchMono(tr));
        float kf = (ys - tr.yb) / s - phase;
        // whorl heights jittered like the near model's
        kf -= (whorlRand(key, floor(kf + 0.5)).x - 0.5) * whorlJit(tr);
        float f = kf - floor(kf);                // 0 at the (un-sagged) whorl below, 1 at the one above
        // the region just above a whorl belongs to its branches, the rest to the skirt of the next
        float kw = floor(kf) + (f < 0.3 ? 0.0 : 1.0);
        vec4 wr = whorlRand(key, kw);
        float nb = whorlBranches(tr, wr);
        float w = TAU / nb;
        float a = (atan(q.z, q.x) - wr.w * TAU) / w;
        float j0 = floor(a), fa = a - j0;
        vec3 b0 = branchRand(key, kw, mod(j0, nb));
        vec3 b1 = branchRand(key, kw, mod(j0 + 1.0, nb));
        // branch tips: lengths per branch, angular falloff between them gives a star outline
        float ph = wr.w * TAU;
        float a0 = ph + (j0 + (b0.x - 0.5) * 0.45) * w, a1 = ph + (j0 + 1.0 + (b1.x - 0.5) * 0.45) * w;
        float wg = whorlGap(tr, wr);
        // (dead lower branches: bare twigs in the near model, sub-pixel here)
        float dP = branchDeadP(tr, saturate((tr.yb + (kw + phase) * s - tr.yb) / hb));
        float wgT = wg + 0.35 * smoothstep(0.55, 0.92, saturate((kw + phase) * s / hb)) * (1.0 - tr.larch);
        float l0 = (branchMissing(tr, b0, a0, op, wgT) || branchDead(b0, dP)) ? 0.0 : min(branchLen(tr, b0, a0), 1.25 + 0.4 / max(renv, 0.3));
        float l1 = (branchMissing(tr, b1, a1, op, wgT) || branchDead(b1, dP)) ? 0.0 : min(branchLen(tr, b1, a1), 1.25 + 0.4 / max(renv, 0.3));
        float c0 = fa - (b0.x - 0.5) * 0.45, c1 = fa - 1.0 - (b1.x - 0.5) * 0.45;
        // pointed branch tips (serrated outline), flat valleys between them
        float lob = max(l0 * (1.0 - 1.2 * abs(c0)), l1 * (1.0 - 1.2 * abs(c1)));
        float lsp = larchSparse(tr);
        // (r5 T4: lower floor in the upper crown - sky between the branch tips of the spire)
        lob = max(lob, mix(mix(0.72, 0.54, op), mix(0.55, 0.42, lsp), tr.larch) * (1.0 - 0.8 * dP) * (1.0 - mix(0.6, 0.4, tr.larch) * smoothstep(0.45, 0.9, th)));
        if (whorlSkip(tr, wr)) lob = 0.35;       // larch: missing whorl -> open band in the crown
        lob = mix(1.0, lob, lod) * mix(0.88, 1.12, wr.y) * whorlGroup(tr, kw);
        // skirt profile between two whorls: full at the branch plane, a thin layer of upward
        // shoots just above it (pinched), then the hanging curtain of the whorl above widens again.
        // The tiers fade to their average before they are < ~2.5 px apart (no moire in motion)
        float tierK = smoothstep(2.5, 6.0, s / max(fp, 1e-4));
        // (r5 T4: +0.22 above th 0.6 - open tiers with sky between them in the spire; b2 r4 tops still solid)
        float gap = (mix(mix(0.22, 0.25, fir), mix(0.32, 0.48, lsp), tr.larch) + 0.10 * op + 0.22 * smoothstep(0.55, 0.70, th)) * mix(0.6, 1.4, tierK);
        gap *= mix(0.6, 1.3, wr.x);
        // each branch sits at its own height (like the near model's): the tiers break up per branch
        // sector instead of stacking as regular discs
        float fbj = fract(f - (fract((abs(c0) < abs(c1) ? b0.x : b1.x) * 13.1) - 0.5) * mix(0.85, 1.0, tr.larch));
        float pin = smoothstep(0.0, 0.22, fbj) * (1.0 - smoothstep(mix(0.42, 0.55, max(fir, tr.larch)), 0.95, fbj));
        float layer = 1.0 - gap * mix(0.5, pin, tierK);
        // (x1.1: the near model's branch tips and sprays reach ~0.9-1.1 x the envelope)
        rr = renv * mix(0.95, min(lob * layer * 1.1, 1.25), lod) + 0.08;
        float tierS = smoothstep(1.5, 4.0, s / max(fp, 1e-4));
        tierV = mix(-0.22, fbj < 0.3 ? 1.0 - fbj * 3.0 : mix(0.6, -1.0, (fbj - 0.3) / 0.7), tierS * lod);
    }
    {
        // (trees6) far crowns (lod -> 0, whorl lobes gone): keep their coarse effect - whorl-group lumps ~1.1 m
        // tall and the lopsided crown (longer branches on the open side, as branchLen) - while the lumps are
        // >= ~1.5 px; <= +28 % of the far radius, inside the early-out bound (1.25 renv)
        float rgK = (1.0 - lod) * (1.0 - smoothstep(0.35, 0.75, fp));
        if (rgK > 0.0) {
            float lump = vnoise1(y * 0.9 + tr.seed * 61.0) - 0.5;
            float oa = treeOpenDir(tr);
            float asy = fract(tr.seed * 47.9) * 0.175 * dot(q.xz, vec2(cos(oa), sin(oa))) / max(r, 1e-3);
            rr *= 1.0 + rgK * (0.22 * lump + asy);
        }
    }
    rr *= wid;
    float d = (r - rr) * 0.55;
    if (nA > 0.0 && d < 0.8) {
        // lumpy crown (branch clumps) and a jagged outline
        float n = texture(uNoise3, q * vec3(0.16, 0.10, 0.16) + tr.seed * 17.0).r - 0.5;
        float n2 = n2A > 0.0 ? texture(uNoise3, q * 0.45 + tr.seed * 7.0).b - 0.5 : 0.0;
        d += (n * nA + n2 * n2A) * smoothstep(0.0, 0.6, renv);
    }
    {
        // holes through airy crowns (larch, open old trees): the sky shows through. Holes ~0.6 x 0.8 m
        // (value noise, 16 cells per 9 m tile), faded out before they are ~3 px; the hole term is kept
        // 1-Lipschitz (|grad hn| <~ 1.7 / m)
        // (r5 T4: spruce / fir spires (upper ~40 %) get holes too - sky through the top)
        float hK = ((0.35 + 0.35 * larchSparse(tr)) * tr.larch + max(0.45 * op, 0.45 * smoothstep(0.55, 0.85, th)) * (1.0 - tr.larch)) * (1.0 - smoothstep(0.10, 0.22, fp)) * lod;
        if (hK > 0.01 && d < 0.6) {
            float hn = texture(uNoise3, q * vec3(0.11, 0.08, 0.11) + tr.seed * 11.0).g;
            d = max(d, (hn - mix(0.95, 0.55, hK)) * 0.55);
        }
    }
    d = max(d, cap);
    inf = vec4(dt < d ? 1.0 : 0.0, saturate(r / max(renv * slim, 0.05)), tierV, 2.0);
    return min(d, dt);
}

// tree-local coordinates; crowns lean a little (wind, light seeking): shift grows with height.
// Forked top: above the fork the crown is mirrored about a plane through the axis and the two
// halves pushed apart, so two leaders diverge in a V
// (r5 T8: fork high in the crown, H x0.84-0.92, was 0.68-0.82: a short V of two leaders, not a two-lobed blob)
float treeForkY(Tree tr) { return tr.H * mix(0.84, 0.92, fract(tr.seed * 91.1)); }
vec3 treeLocal(vec3 p, Tree tr) {
    vec3 q = p - tr.base;
    vec2 lean = (vec2(fract(tr.seed * 23.1), fract(tr.seed * 41.7)) - 0.5) * 0.08;
    q.xz -= lean * max(q.y, 0.0);
    if (treeTop(tr) == 2) {
        float yf = treeForkY(tr);
        float wf = smoothstep(yf - 0.5, yf + 0.5, q.y);
        if (wf > 0.0) {
            vec2 fd = vec2(cos(tr.seed * 40.0), sin(tr.seed * 40.0));
            float sd = dot(q.xz, fd);
            float off = 0.06 + 0.10 * max(q.y - yf, 0.0);
            q.xz += fd * (mix(sd, abs(sd) - off, wf) - sd);
        }
    }
    return q;
}

float sdTreeAny(vec3 p, Tree tr, float lod, float lod2, bool info, out float env, out vec4 inf) {
    vec3 q = treeLocal(p, tr);
    float dc = treeCamDist(tr);
    float fp = dc * uPixAng;
    env = 1.0;
    if (info) env = length(q.xz) / max(crownR(tr, q.y, treeFir(tr)), 0.05);
    float d;
    if (treeIsNear(tr, dc)) {
        // branchlets keep their physical needle-brush radius; the trace hits at d < ~0.8 px, so
        // undo most of that dilation (thin shoots stay ~0.3 px wider than they are, never vanish)
        float fine = 1.0 - smoothstep(0.002, 0.006, fp);
        d = sdTreeNear(q, tr, fp, dc, fine, info, inf) + 0.5 * fp;
    } else {
        float lodM = 1.0 - smoothstep(0.25, 0.55, fp);
        d = sdTreeMid(q, tr, lodM, fp, info, inf);
        // the trace hits at d < eps (~0.8 px): undo the resulting ~1.5 px crown dilation at range
        d += 0.58 * fp;
    }
    // the fork fold stretches space a little
    if (treeTop(tr) == 2) d *= 0.75;
    return d;
}

// lod: 0 = plain cone, 1 = whorl skirts; lod2: 1 - distance / TREE_LOD2_RANGE (kept for callers;
// the model choice is per tree from its camera distance)
float sdTree(vec3 p, Tree tr, float lod, float lod2, out float env) {
    vec4 inf;
    return sdTreeAny(p, tr, lod, lod2, false, env, inf);
}

// same as sdTree, also fills gTreeInfo and env (used once per shaded hit)
float sdTreeInfo(vec3 p, Tree tr, float lod, float lod2, out float env) {
    vec4 inf;
    float d = sdTreeAny(p, tr, lod, lod2, true, env, inf);
    gTreeInfo = inf;
    return d;
}

float treeLod(float t) { return 1.0 - smoothstep(900.0, 2200.0, t); }
float treeLod2(float t) { return saturate(1.0 - t / TREE_LOD2_RANGE); }

// (perf4) tree-free space (render.py tree_tops): per 16 m block, R = the highest possible point of any tree in the block
// and its 8 neighbours (the 3 x 3 cells treesSDF tests around p always lie inside), G = horizontal clearance (m) to the
// crowns of all trees further out. Returns a distance below which no tree surface lies (<= 0: unknown, evaluate)
layout(set=0, binding=13) uniform sampler2D uTreeTop;
float treeFreeDist(vec3 p) {
    ivec2 b = (treeCellOf(p.xz) - ivec2(TREE_TEX_OFF)) >> 2;
    if (any(lessThan(b, ivec2(0))) || any(greaterThanEqual(b, ivec2(TREE_TEX_RES / 4))))
        return max(abs(p.x), abs(p.z)) - float(TREE_TEX_RES / 2) * TREE_CELL - 12.0;   // outside the tree map (r4 merge: reach <= ~11.6 m)
    vec2 tg = texelFetch(uTreeTop, b, 0).rg;
    return min(p.y - tg.r, tg.g);
}

// (perf4) the marcher's distance to everything else (and its hit threshold): trees whose bound lies beyond it can
// neither be hit nor shorten the step, so they are not evaluated (trace() / coneTrace() set it before treesSDF)
float gTreeDMax = 1e9;
float treesSDF(vec3 p, float lod, float lod2, out ivec2 hitCell) {
    ivec2 c0 = treeCellOf(p.xz);
    // "no tree near" cap; trees whose bound is beyond it are skipped before their stand-structure
    // lookups (raw baked cell first)
    float dmin = TREE_CELL * 0.55 + length(p - uCamPos) * 0.0012;
    dmin = min(dmin, gTreeDMax);
    hitCell = ivec2(-1);
    int z0 = treeZero();
    for (int j = z0 - 1; j <= 1; j++)
    for (int i = z0 - 1; i <= 1; i++) {
        ivec2 c = c0 + ivec2(i, j);
        vec4 d, hs; vec2 pos;
        if (!treeRaw(c, d, hs, pos)) continue;
        if (treeRawBound(p, c, d, hs, pos) > dmin) continue;
        Tree tr;
        if (!treeFinish(c, d, hs, pos, tr)) continue;
        vec3 q = p - tr.base;
        float bound = max(length(q.xz) - treeReach(tr.R * treeLowWiden(tr), tr.H, tr.seed), max(q.y - tr.H - 2.4, -q.y - 1.0));
        if (bound > dmin) continue;
        float env;
        float dd = sdTree(p, tr, lod, lod2, env);
        if (dd < dmin) { dmin = dd; hitCell = c; }
    }
    return dmin;
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
float crownSunPath(vec3 q, Tree tr, float fir) {
    float top = crownTop(tr) + 0.6;
    float hb = max(top - 0.6 - tr.yb, 1.0);
    float k = tr.R * mix(1.0, 0.80, fir) * 1.12 / hb;       // cone slope (radius per metre)
    vec3 L = uSunDir;
    float A = dot(L.xz, L.xz) - k * k * L.y * L.y;
    float B = 2.0 * (dot(q.xz, L.xz) + k * k * (top - q.y) * L.y);
    float C = dot(q.xz, q.xz) - k * k * (top - q.y) * (top - q.y);
    float disc = B * B - 4.0 * A * C;
    float s;
    if (A < 1e-4) {
        // sun almost overhead: only points inside the envelope are shaded
        if (C >= 0.0) return 0.0;
        s = -C / max(B, 1e-4);
    } else {
        if (disc <= 0.0) return 0.0;
        float sq = sqrt(disc);
        float s1 = (-B - sq) / (2.0 * A), s2 = (-B + sq) / (2.0 * A);
        if (C < 0.0) s = s2;                                 // inside: path to the exit
        else {
            // outside: through the crown if the ray crosses the (lower) cone ahead
            if (s1 <= 0.0 || q.y + L.y * 0.5 * (s1 + s2) > top) return 0.0;
            s = s2 - s1;
        }
    }
    return clamp(s - 0.15, 0.0, 2.5 * tr.R + 2.0);
}

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
layout(set=0, binding=14) uniform sampler2D uStreamCorr; 
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

void rkInit(out Rock rk) {
    rk.c = vec3(0.0); rk.r = vec3(1.0); rk.ang = 0.0; rk.seed = 0.0; rk.m = mat3(1.0);
    rk.f0 = rk.f1 = rk.f2 = rk.f3 = rk.f4 = rk.f5 = rk.f6 = rk.f7 = rk.f8 = rk.f9 = rk.ck = vec4(0.0, 1.0, 0.0, 1e3);
    rk.ckw = 0.0; rk.rr = 0.1; rk.taper = 0.0; rk.env = 0.0; rk.fld = 0.0; rk.kind = 0.0; rk.rel = 99.0; rk.nt = vec2(0.0); rk.nt2 = vec3(0.0); rk.ell = 1.55;
    rk.nt3 = vec2(0.0); rk.ch0 = vec4(0.0); rk.ch1 = vec4(0.0);
}
vec4 rkHashOf(ivec2 c, bool small, int k) { return rkSlotHash(rkCellHash(small ? c + ivec2(ROCK2_TAG + 3301, 1777) : c), k); }
// rock of a (possibly small-layer- and slot-tagged) cell
bool rkGet(ivec2 cell, float lod, out Rock rk) {
    rkInit(rk);
    int k = rkSlotOf(cell);
    ivec2 cb = cell - (k < 16 ? ivec2(0, k * ROCK_SLOT_TAG) : ivec2(ROCK_HI_TAG, (k - 16) * ROCK_SLOT_TAG));
    bool small = cb.x >= ROCK2_TAG / 2;
    ivec2 c = small ? cb - ivec2(ROCK2_TAG, 0) : cb;
    RkCtx x;
    if (!rkCtx(small ? rkParent(c) : c, x)) return false;
    if (small) { vec2 c2 = (vec2(c) + 0.5) * ROCK2_CELL; rkSmallLocal(x, c2, terrainH(c2, false)); }
    vec4 hs = rkHashOf(c, small, k);
    vec2 pos; float R, kind;
    float cs = small ? ROCK2_CELL : ROCK_CELL;
    if (!rkSlot(x, (vec2(c) + 0.5) * cs, small, k, hs, rkHashOf(c, small, 0), pos, R, kind)) return false;
    return rkBuild(cell, hs, pos, R, kind, x, rk);
}
// slot 0 of a main cell (the biggest block; used by the ground cover to clear grass)
bool getRock(ivec2 cell, out Rock rk) { return rkGet(cell, 1.0, rk); }

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
float wwRibbon(vec2 pa, float hw, float aa) {
    gRibM = hw * (0.55 * gnoise(vec2(pa.x * 0.055, 3.1)) + 0.2 * gnoise(vec2(pa.x * 0.17, 7.7)));
    gRibW = hw * 0.5 * mix(0.65, 1.35, 0.5 + 0.7 * gnoise(vec2(pa.x * 0.09, 1.3)));
    return 1.0 - smoothstep(gRibW * 0.6 - aa, gRibW * 1.2 + aa, abs(pa.y - gRibM));
}
float wwRibbonG(float y, float aa) { return mix(1.0, 1.0 - smoothstep(gRibW * 0.6 - aa, gRibW * 1.2 + aa, abs(y - gRibM)), gWWRib); }
float wwField(vec2 pa, float H, float vis, out float rock) {
    float a = texture(uNoise3, vec3(pa.x * 0.075, pa.y * 0.12, 0.31 + uTime * 0.03)).a;
    float b = vis > 0.0 ? texture(uNoise3, vec3(pa * vec2(0.35, 0.45), 0.67 + uTime * 0.07)).b : 0.5;
    float l = 0.7 * a + 0.3 * mix(0.5, b, vis);
    // (i10: 3.2 x 2.6 m cells, centre jitter 0.2: a boulder up to 2 m across stays inside its cell - the old
    // 0.95 m boulders with 0.36 jitter were cut off straight at the cell edges)
    const vec2 BC = vec2(3.2, 2.6);
    // (review rv1) every column of cells (across the flow) shifted along the flow by its own amount: the boulders of
    // neighbouring columns no longer line up in rows straight across the channel (weirs / terraces in film shot 4)
    vec2 pc = pa;
    pc.x += hash12i(ivec2(int(floor(pa.y / BC.y)), 4111)) * BC.x;
    vec2 ci = floor(pc / BC);
    vec4 hb = hash42i(ivec2(ci) + ivec2(733, 1999));
    vec2 o = pc - (ci + 0.5 + (hb.yz - 0.5) * 0.2) * BC;
    // (round 3 i10/i12: rounded heads 0.7-1.9 m across standing 0.4-0.9 m out of the water in ~45 % of the cells -
    // the i10 2 m domes 0.3-0.8 m tall read as flat pucks from film shot 4's low camera)
    float R = mix(0.35, 0.8, hb.w);
    // (review rv5) the boulders crowd the sides of the white ribbon (it winds between them): ~60 % of the cells off
    // the ribbon, ~20 % in it
    float has = 0.0;
    if (hb.x < 0.6 && gWWBo > 0.0) {
        float ribC = wwRibbonG((ci.y + 0.5) * BC.y, 0.3);
        has = step(hb.x, mix(mix(0.35, 0.6, gWWRib), 0.2, ribC)) * gWWBo;
    }
    vec2 on = o / vec2(R * 1.2, R);
    // (review rv5) a lobed outline (3 + 5 lobes, own phase): angular boulders, not elliptical eggs
    float ang = atan(on.y, on.x);
    float lob = 1.0 + 0.3 * (0.6 * sin(3.0 * ang + hb.y * 6.28) + 0.4 * sin(5.0 * ang + hb.z * 6.28));
    float r2 = dot(on, on) * lob;
    // (i11: a rounded dome, tilted, not a flat-topped puck - pow 0.45 made flat tops with a steep dark rim)
    // (review rv1: rounder heads, 0.5-1.1 m out of the water - still read as pucks)
    float hB = has * mix(0.5, 1.1, fract(hb.w * 7.31)) * (1.0) * pow(max(1.0 - r2, 0.0), 0.6) * (1.0 + 0.35 * dot(on, hb.yz - 0.5));
    // (review rv4) an angular block, not a smooth loaf: 4 planar facets around an off-centre apex (+ the dome's round
    // flanks), a lumpy outline (smooth ellipsoids read as chrome pucks in film shot 4)
    if (hB > 0.0) {
        vec2 la = on - (hb.zy - 0.5) * 0.6;
        vec4 kk = 0.8 + 0.5 * fract(hb.wxyz * 5.73 + 0.31);
        float pm = max(la.x > 0.0 ? la.x * kk.x : -la.x * kk.y, la.y > 0.0 ? la.y * kk.z : -la.y * kk.w);
        float bump = vis > 0.0 ? (texture(uNoise3, vec3(pa * 0.9, 0.83)).r - 0.475) * 0.6 * vis : 0.0;
        hB = max(hB * mix(1.0, saturate(1.15 - 0.6 * pm) * 1.2, 0.7) + bump * hB, 0.0);
    }
    float up = has * smoothstep(0.3, -0.6, o.x / R) * (1.0 - smoothstep(1.0, 2.6, sqrt(r2)));
    float lee = has * smoothstep(0.0, 0.8, o.x / R) * (1.0 - smoothstep(0.7, 1.3, abs(on.y))) * exp(-max(o.x, 0.0) / 2.2);
    // (r4 C4 / W8: heaps only on ~30 % of the ribbon and none beside it (was 0.30 / 12 %): at film shot 4's grazing view even
    // low heaps hid all the water between them - a bank-to-bank white sheet; now dark glassy water shows between the tongues)
    float hF = H * smoothstep(0.55, 0.85, l + 0.35 * up - 0.45 * lee) * wwRibbonG(pa.y, 0.15);
    rock = step(1e-4, hB) * step(hF, hB);
    return max(hF, hB);
}

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
float poolFoamF(vec2 xz, float foot, out vec2 tilt) {
    tilt = vec2(0.0);
    gPoolAer = 0.0;
    gPoolTone = 1.0;
    if (uFallsOn == 0) return 0.0;
    vec2 q = wfUV(xz);
    if (q.x < -110.0 || q.x > 25.0 || q.y < -60.0 || q.y > 95.0) return 0.0;
    float f = 0.0, aer = 0.0;
    vec2 alongD = vec2(-uWFDir.y, uWFDir.x);
    float fineV = 1.0 - smoothstep(0.3, 1.2, foot);
    for (int k = 0; k < 2; k++) {
        vec4 kn = uFallK[k * WF_NK + WF_NK - 1];
        float vol = uFallInfo[k].y;
        // (centred ~1 m out from the foot of the face: the boil must show in front of the curtain)
        vec2 c = vec2(min(kn.x, wfFaceLine(kn.y) - 0.5) - 1.2, kn.y);
        vec2 d = q - c;
        float out_ = max(-d.x, 0.0);
        float dy = d.y - 0.30 * out_;
        float spread = kn.z * 1.1 + 0.32 * out_;
        // (round 2: the pool is clear turquoise a few metres out: a shorter drifting plume)
        // (review r2, fix 6: decay 11 + 5 vol -> 6 + 3 vol: at the grazing view a long plume squashed into a white
        // tablecloth / a ruler-straight line along the cliff foot)
        float plume = exp(-abs(dy) / spread) * exp(-out_ / ((6.0 + 3.0 * vol) * 2.0)) * smoothstep(1.5, -0.5, d.x);   // (round 3: x2, the foam plume spreads over the pool)
        vec2 adv = vec2(q.x + uTime * 0.7, dy);
        // boil (~3-4 m out, about as wide as the fall) where the water hits: a torn rim, broken into
        // bubbly froth (worley cells ~0.2 m; unresolved -> their mean), never a flat white slab
        float rr = length(d * vec2(1.0, 0.75));
        float nr = texture(uNoise3, vec3(adv * 0.12, 0.2 + uTime * 0.05)).r;
        // (session 3: a bigger, whiter boil - the impact foam is solid white a few metres around each fall)
        // (review r2: a smaller boil - the 3D churn / spray volume of fallsVolume stands on it)
        float eb = length(vec2(d.x / (3.0 + 1.5 * vol), d.y / (kn.z * 1.1 + 1.2))) / (1.5 * (1.3));   // (round 3: x1.5, a broad white boil; rv1 x1.3)
        float bubN = texture(uNoise3, vec3(adv * 0.35, 0.6 + uTime * 0.1)).a;
        float vBub = 1.0 - smoothstep(0.05, 0.2, foot);
        float bub = mix(0.8, smoothstep(0.25, 0.65, bubN), vBub);
        float boil = (1.0 - smoothstep(0.8, 1.4, eb + 0.9 * (nr - 0.5))) * vol * mix(0.75, 1.0, bub);
        float m = max(plume, exp(-0.5 * eb * eb)) * vol;
        // near the falls (< ~6 m): lacy foam, a network of curved foam lines around open-water holes
        float lw = mix(0.02, 0.16, m);
        float r1 = texture(uNoise3, vec3(adv * 0.050, uTime * 0.012 + float(k) * 0.5)).r;
        float r2 = texture(uNoise3, vec3(adv * 0.13 + 0.3, 0.5 + uTime * 0.02)).g;
        float lace = max(1.0 - smoothstep(lw * 0.5, lw, abs(r1 - 0.475)), 0.8 * (1.0 - smoothstep(lw * 0.4, lw * 0.8, abs(r2 - 0.487))));
        lace = mix(saturate(lw * 5.0), lace, fineV);
        // patch noise stretched ~3:1 along the drift (the lace comes in drifting patches)
        vec2 ra = vec2(out_ - uTime * 0.7, dy);         // (along the drift, across it)
        float rn = wfRemap(texture(uNoise3, vec3(ra.x * 0.035, ra.y * 0.105, 0.37 + float(k) * 0.3 + uTime * 0.004)).r) * 0.7
                 + 0.3 * mix(0.5, texture(uNoise3, vec3(ra * vec2(0.09, 0.27), 0.81)).g, fineV);
        float rw = 0.06 + 0.5 * foot / 3.0;
        // the lace network comes in drifting patches (streaks ~3:1 along the drift), dense near the falls
        float patchL = smoothstep(0.35 - 0.3 * m - rw, 0.65 - 0.3 * m + rw, rn);
        // (review r2: the drifting patches everywhere, also in the dense part - no uniform white sheet)
        float fl = lace * smoothstep(0.02, 0.3, m) * mix(0.55, 0.9, m) * mix(0.35, 1.0, patchL);
        f = max(f, max(boil, fl));
        // foam tone: thick froth white, thin drifting foam greyer (0.65-1)
        float tn = mix(0.5, texture(uNoise3, vec3(adv * 0.2 + 0.4, 0.9 + uTime * 0.03)).g, fineV);
        gPoolTone = min(gPoolTone, mix(1.0, saturate(0.65 + 0.35 * wfRemap(tn) + 0.3 * boil), smoothstep(0.02, 0.2, max(boil, fl))));
        // aeration: ~10 m core + the plume
        aer = max(aer, max(exp(-rr * rr / 100.0), plume * exp(-out_ / 18.0)) * vol);
        // outward-travelling ring waves (unresolved -> none)
        float rl = length(d);
        vec2 dirL = d / max(rl, 0.3);
        float w = sin(rl * 3.2 - uTime * 7.0) * exp(-rl * 0.14) * 0.10 * vol * (1.0 - smoothstep(0.5, 1.0, foot));
        tilt += (-uWFDir * dirL.x + alongD * dirL.y) * w;
        // (review r2) churn: the boil's bubbly surface is heaped and broken (normal tilt from the bubble field)
        if (vBub > 0.0 && boil > 0.02) {
            float bx = texture(uNoise3, vec3((adv + vec2(0.12, 0.0)) * 0.35, 0.6 + uTime * 0.1)).a - bubN;
            float bz = texture(uNoise3, vec3((adv + vec2(0.0, 0.12)) * 0.35, 0.6 + uTime * 0.1)).a - bubN;
            tilt += (-uWFDir * bx + alongD * bz) * 9.0 * boil * vBub;
        }
    }
    gPoolAer = saturate(aer);
    return f;
}

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
float gcGrow(float keep, float rk) { return saturate((keep - rk) / 0.12); }

uint gcS;
float gcRnd() {
    gcS = gcS * 747796405u + 2891336453u;
    uint w = ((gcS >> ((gcS >> 28u) + 4u)) ^ gcS) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w >> 8u) * (1.0 / 16777216.0);
}

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
float gcGroundG(vec2 x, out vec2 g) {
    vec3 m1 = gnoised(x * 0.55 + 1.7);
    vec3 m2 = gnoised(rot2(1.1) * x * 1.3 + 4.2);
    vec3 m3 = gnoised(rot2(2.3) * x * 3.1);
    float hx = terrainH(x + vec2(0.5, 0.0), false) - terrainH(x - vec2(0.5, 0.0), false);
    float hz = terrainH(x + vec2(0.0, 0.5), false) - terrainH(x - vec2(0.0, 0.5), false);
    g = vec2(hx, hz) + 0.121 * m1.yz + 0.117 * (m2.yz * rot2(1.1)) + 0.1085 * (m3.yz * rot2(2.3));
    return terrainH(x, true) + 0.22 * m1.x + 0.09 * m2.x + 0.035 * m3.x;
}
// per ray segment [gcSegA, gcSegB]: the smooth stand field below at both ends (interpolated along
// the segment, so it does not step where the camera-dependent segments start), the tree occupancy
// of a 5x5 block of 4 m tree cells, the trail segments nearby
vec2 gcSegA = vec2(0.0), gcSegB = vec2(0.0);
float gcWoodA = 0.0, gcWoodB = 0.0;
float gcWoodOf(vec2 x) {
    vec2 ab = gcSegB - gcSegA;
    return mix(gcWoodA, gcWoodB, saturate(dot(x - gcSegA, ab) / max(dot(ab, ab), 1e-6)));
}
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
float gcTrailDist(vec2 x) {        // search over the trail segments near the camera; keeps the two nearest
    if (uTrailN < 2) return 1e9;
    if (!gcTrMaskOk) {
        gcTrMaskOk = true;
        for (int i = 0; i < 31; i++) {
            if (i + 1 >= uTrailN) break;
            vec2 a2 = uTrail[i].xy, ab2 = uTrail[i + 1].xy - a2;
            vec2 q = uCamPos.xz - a2;
            if (length(q - ab2 * clamp(dot(q, ab2) / dot(ab2, ab2), 0.0, 1.0)) < GC_FADE1 + 10.0) gcTrMask |= 1u << uint(i);
        }
    }
    float d0 = 1e9, d1 = 1e9;
    for (int i = 0; i < 31; i++) {
        if (i + 1 >= uTrailN) break;
        if ((gcTrMask & (1u << uint(i))) == 0u) continue;
        vec2 a2 = uTrail[i].xy, b2 = uTrail[i + 1].xy, ab2 = b2 - a2;
        float tt = clamp(dot(x - a2, ab2) / dot(ab2, ab2), 0.0, 1.0);
        float d = length(x - (a2 + ab2 * tt));
        if (d < d0) { d1 = d0; gcTrS1 = gcTrS0; d0 = d; gcTrS0 = vec4(a2, b2); }
        else if (d < d1) { d1 = d; gcTrS1 = vec4(a2, b2); }
    }
    return d0;
}
vec2 gcSegNear(vec2 x, vec4 sg) {
    vec2 ab = sg.zw - sg.xy;
    return sg.xy + ab * clamp(dot(x - sg.xy, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
}
// trail weight near the current ray segment (only the two segments found by gcTrailDist): x = dirt
// (same edge as shadeTerrain), y = 0.7 m verge outside the dirt (short, thin grass); dir = unit
// direction toward the trail's centre line
float gcTrD = 1e9;            // distance to the trail's centre line (set by gcTrailW)
vec2 gcTrailW(vec2 x, out vec2 dir) {
    vec2 p0 = gcSegNear(x, gcTrS0), p1 = gcSegNear(x, gcTrS1);
    float d0 = length(x - p0), d1 = length(x - p1);
    float d = min(d0, d1);
    gcTrD = d;
    dir = ((d0 < d1 ? p0 : p1) - x) / max(d, 1e-4);
    // (veg3) plants reach in over the outer part of the terrain's dirt band (0.9-1.9 m): the trail
    // reads ~1-1.5 m wide between dense verges, not a 3-4 m dirt road
    // (veg4 G4: the terrain's dirt is now 0.35 - 0.75 m from the centre line: a 0.7-1.5 m path)
    float e = 0.75 + 0.25 * gnoise(x * 0.7);
    return vec2(1.0 - smoothstep(0.35, e, d), 1.0 - smoothstep(e, e + 0.6, d));
}
// habitat at ground position x (height y, slope g, water level wl): a function of the position only,
// so every ray and pixel agrees on what grows where
GBio gcBiome(vec2 x, float y, vec2 g, float wl) {
    GBio b;
    vec4 f = forestAt(x);
    b.above = wl > -1000.0 ? y - wl : 999.0;
    b.g = g;
    float ny = inversesqrt(1.0 + dot(g, g));
    float n2 = gnoise(x * 0.11 + 7.0);
    float n3 = gnoise(x * 0.45 + 1.0);
    // terrainAlbedo's vegetation mask (same slope, same noise) ...
    b.veg = smoothstep(0.60, 0.82, ny + 0.10 * n2) * (1.0 - smoothstep(800.0, 1050.0, y)) * (1.0 - f.b * 0.9);
    // ... and its cobble shore, where the terrain draws pebbles
    b.peb = 1.0 - smoothstep(0.35, 1.3, b.above + 0.4 * n3);
    b.shore = 1.0 - smoothstep(0.6, 2.2, b.above + 0.5 * n3);
    float sgr0 = gcShoreGrass(x);
    b.sgr = sgr0 * b.shore;
    // (i25) the grassy shore's boulder belt up to ~6 m above the water (dwarf shrubs)
    b.sgrB = mix(0.6, 1.0, sgr0) * smoothstep(0.25, 0.6, b.above) * (1.0 - smoothstep(3.5, 6.5, b.above));   // (i27: not only in the grassy-shore patches)
    // grassy shores: grass also on the steeper banks right by the water, and over the cobbles (the
    // sward's own base mat hides them there)
    float matW = smoothstep(0.3, 0.5, b.sgr);
    // (vegT: what the terrain itself draws as vegetated - the grass of the far end of the cover,
    //  where the shore mat breaks up, follows it)
    b.vegT = b.veg * (1.0 - b.peb) * smoothstep(0.25, 0.45, b.above);
    b.veg = mix(b.veg, max(b.veg, 0.9 * smoothstep(0.35, 0.6, ny)), matW);
    b.veg *= mix(1.0 - b.peb, 1.0, matW);
    b.veg *= mix(smoothstep(0.25, 0.45, b.above), smoothstep(0.0, 0.06, b.above), b.sgr);
    b.forest = saturate(f.r * 1.3);
    b.meadow = f.a;
    // inside or right next to a forest stand (the trail corridor and small clearings included)
    // (the lake shore gets the light from the open water side: grass, not forest floor)
    float stand = gcStandOf(x);
    b.wood = saturate(max(max(f.r * 1.3, gcWoodOf(x) * 1.8), stand)) * (1.0 - 0.85 * f.a) * (1.0 - 0.85 * b.shore);
    // a real stand around (also where the bake calls it meadow): forest floor
    b.wood = max(b.wood, smoothstep(0.5, 0.9, stand) * (1.0 - 0.85 * b.shore));
    b.tdir = vec2(0.0);
    gcTrD = 1e9;
    vec2 tw = gcTrailSeg ? gcTrailW(x, b.tdir) : vec2(0.0);
    b.trail = tw.x; b.tedge = tw.y; b.tdist = gcTrD;
    return b;
}
// vegetation mask only (bare-segment test), with margins: > 0 wherever a plant rooted up to ~1.4 m
// beside the probed point could stand (slope ny 0.45, 0.5 m below the waterline, pebble shore 0.4 m
// lower), so plants leaning over rock or water next to the probed track are never skipped
float gcVegProbe(vec2 x) {
    float y = terrainH(x, false);
    vec4 f = forestAt(x);
    float wl = waterLevel(x);
    float above = wl > -1000.0 ? y - wl : 999.0;
    float hx = terrainH(x + vec2(0.5, 0.0), false) - terrainH(x - vec2(0.5, 0.0), false);
    float hz = terrainH(x + vec2(0.0, 0.5), false) - terrainH(x - vec2(0.0, 0.5), false);
    vec2 g = vec2(hx, hz) + 0.121 * gnoised(x * 0.55 + 1.7).yz;
    float ny = inversesqrt(1.0 + dot(g, g));
    float n3 = gnoise(x * 0.45 + 1.0);
    float veg = smoothstep(0.45, 0.82, ny + 0.10 * gnoise(x * 0.11 + 7.0)) * (1.0 - smoothstep(800.0, 1050.0, y)) * (1.0 - f.b * 0.85);
    float shore = 1.0 - smoothstep(0.6, 2.2, above + 0.5 * n3);
    float sgr = gcShoreGrass(x) * shore;
    veg = mix(veg, max(veg, smoothstep(0.30, 0.6, ny)), sgr);
    veg *= mix(smoothstep(0.35, 1.3, above + 0.4 * n3 + 0.4), 1.0, smoothstep(0.2, 0.5, sgr));
    return veg * mix(smoothstep(-0.5, 0.45, above), smoothstep(-0.5, 0.06, above), sgr);
}

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
void gcGatherTrees(vec2 a, vec2 b) {
    gcNTr = 0;
    gcTreesOk = true;
    gcTB0 = ivec2(floor((min(a, b) + FAR_HALF) / TREE_CELL - 0.5)) - 1;
    gcTBits = 0u; gcTBN = 5;
    vec2 ab = b - a;
    float l2 = max(dot(ab, ab), 1e-6);
    for (int j = 0; j < 5; j++)
    for (int i = 0; i < 5; i++) {
        // occupancy from the raw texel; the full tree only for cells that can reach the segment
        ivec2 cell = gcTB0 + ivec2(i, j), tc = cell - ivec2(TREE_TEX_OFF);
        if (any(lessThan(tc, ivec2(0))) || any(greaterThanEqual(tc, ivec2(TREE_TEX_RES)))) continue;
        if (texelFetch(uTrees, tc, 0).y == 0.0) continue;
        gcTBits |= 1u << uint(i + 5 * j);
        vec2 qc = treeCellOrigin(cell) + 0.5 * TREE_CELL - a;
        if (length(qc - ab * clamp(dot(qc, ab) / l2, 0.0, 1.0)) > 1.15 * GC_CANR + 0.4 + 1.2 + 1.7 || gcNTr >= 8) continue;
        Tree tr;
        getTree(cell, tr);
        vec2 q = tr.base.xz - a;
        float dd = length(q - ab * clamp(dot(q, ab) / l2, 0.0, 1.0));
        float R = min(tr.R, GC_CANR);
        if (dd > 1.15 * R + 0.4 + 1.2 || gcNTr >= 8) continue;
        gcTr[gcNTr++] = vec3(tr.base.xz, R);
    }
}
float gcCanopy(vec2 x) {
    if (!gcTreesOk) gcGatherTrees(gcSegA, gcSegB);
    float c = 0.0;
    for (int i = 0; i < gcNTr; i++) c = max(c, gcCanopyOne(x, gcTr[i]));
    return c;
}
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
float gcFloor(GBio b, float canopy) {
    // (i17) the forest trail's corridor (~6 m either side) is forest floor even where the stand opens up
    // (leafy understory both sides of the path, no lawn strip)
    float corr = 0.85 * (1.0 - smoothstep(4.5, 8.0, b.tdist)) * (1.0 - b.shore);
    return gcFloorW(max(max(canopy, 0.9 * smoothstep(0.2, 0.6, b.wood)), corr));
}

// boulders near x (2x2 nearest rock cells) for suppressing grass inside their footprints
// (cached per ray: consecutive segments mostly share the 2x2 rock cells; the stones appended by a
// segment are dropped again)
ivec2 gcRkC0 = ivec2(-99999);
int gcNRkB = 0;
vec2 gcRkY[8];               // (veg4 R6) centre height, half height of each gathered rock (sun shadow on the cover)
void gcGatherRocks(vec2 x) {
    ivec2 c0 = ivec2(floor(x / ROCK_CELL - 0.5));
    if (c0 == gcRkC0) { gcNRk = gcNRkB; return; }
    gcRkC0 = c0;
    gcNRk = 0;
    // (veg4 R6: slots 0-1 of each cell, not only the biggest block - grass grew through the second blocks)
    for (int k = 0; k < 2; k++)
    for (int j = 0; j < 2; j++)
    for (int i = 0; i < 2; i++) {
        Rock rk;
        if (!getRock(c0 + ivec2(i, j) + ivec2(0, k * ROCK_SLOT_TAG), rk)) continue;
        gcRk[gcNRk] = vec4(rk.c.x, rk.c.z, rk.r.x, rk.r.z);
        gcRkA[gcNRk] = vec2(cos(rk.ang), sin(rk.ang));
        gcRkY[gcNRk] = vec2(rk.c.y, rk.r.y);
        gcNRk++;
    }
    gcNRkB = gcNRk;
}
// (veg4 R6) sun visibility of a cover point past the gathered boulders / stones (ray vs their ellipsoids, soft
// edge) and their contact occlusion: the grass ran fully lit right up to every block (r8 0.33-0.40 vs open
// lawn 0.344; near-black slots between the blocks)
float gcRockSun(vec3 p, out float aoR) {
    float s = 1.0;
    aoR = 1.0;
    vec3 l = uSunDir;
    for (int i = 0; i < gcNRk; i++) {
        mat2 R = mat2(gcRkA[i].x, gcRkA[i].y, -gcRkA[i].y, gcRkA[i].x);
        vec2 oxz = R * (p.xz - gcRk[i].xy), lxz = R * l.xz;
        vec3 rr = vec3(gcRk[i].z, max(gcRkY[i].y, 0.05), gcRk[i].w);
        vec3 o = vec3(oxz.x, p.y - gcRkY[i].x, oxz.y) / rr;
        vec3 d = vec3(lxz.x, l.y, lxz.y) / rr;
        float a = dot(d, d), b = dot(o, d), c = dot(o, o);
        float tm = max(-b / a, 0.0);
        float dm = sqrt(max(c + 2.0 * b * tm + a * tm * tm, 0.0));      // closest approach of the sun ray (unit sphere)
        if (b < 0.0) s = min(s, smoothstep(0.82, 1.08, dm));
        // contact: ground beside the block (horizontal footprint distance, ~1 at the rim)
        float e = length(oxz / rr.xz);
        aoR = min(aoR, mix(0.45, 1.0, smoothstep(0.92, 1.55, e)));
    }
    return s;
}
// 0 inside a boulder footprint; nearR = 1 right at a boulder's foot
float gcRockMask(vec2 x, out float nearR) {
    float m = 1.0; nearR = 0.0;
    for (int i = 0; i < gcNRk; i++) {
        vec2 q = x - gcRk[i].xy;
        q = mat2(gcRkA[i].x, gcRkA[i].y, -gcRkA[i].y, gcRkA[i].x) * q;
        float e = length(q / (gcRk[i].zw * 0.9));
        m = min(m, smoothstep(0.9, 1.04, e));
        nearR = max(nearR, 1.0 - smoothstep(1.0, 1.7, e));
    }
    return m;
}
// smallest footprint distance (1 = at 0.9 of a boulder's / stone's footprint radius)
float gcRockE(vec2 x) {
    float m = 9.0;
    for (int i = 0; i < gcNRk; i++) {
        vec2 q = mat2(gcRkA[i].x, gcRkA[i].y, -gcRkA[i].y, gcRkA[i].x) * (x - gcRk[i].xy);
        m = min(m, length(q / (gcRk[i].zw * 0.9)));
    }
    return m;
}

// ---- analytic primitives ---------------------------------------------------------------------
// bent ribbon surface: root R, lean direction a (unit xz), height H, horizontal lean L of the top.
// Centre line x1 = L (y/H)^2 in the vertical plane through a; the surface extends sideways
// (perpendicular to a), so its face looks outward/up like an arching blade. Returns the two ray
// parameters of the surface crossings and the local ray terms (a2, b2, a3, b3).
bool gcBent(vec3 o, vec3 d, vec3 R, vec2 a, float H, float L, out vec2 ss, out vec4 ab, out float k) {
    vec3 q = o - R;
    vec2 bd = vec2(-a.y, a.x);
    float a1 = dot(q.xz, a), b1 = dot(d.xz, a);
    ab = vec4(q.y, d.y, dot(q.xz, bd), dot(d.xz, bd));
    k = L / (H * H);
    float A = k * ab.y * ab.y, B = 2.0 * k * ab.x * ab.y - b1, C = k * ab.x * ab.x - a1;
    if (abs(A) < 1e-7) {
        if (abs(B) < 1e-9) return false;
        ss = vec2(-C / B, -1.0);
        return true;
    }
    float D = B * B - 4.0 * A * C;
    if (D < 0.0) return false;
    float sq = sqrt(D);
    float qq = -0.5 * (B + (B < 0.0 ? -sq : sq));
    ss = vec2(qq / A, C / qq);
    if (ss.x > ss.y) ss = ss.yx;
    return true;
}

// broad forb leaf (lanceolate) on a bent ribbon

// grass tiller: a fan of nb tapered blades in one bent surface (grass leaves are distichous).
// The fan widens from Wb at the root to Wt at the top; each blade fills fr of its slot and has
// its own length (hash hs). Returns v along the hit blade.
bool gcFan(vec3 o, vec3 d, vec3 R, vec2 a, float H, float L, float Wb, float Wt, float nb, float fr, float hs,
           float sMax, out float sHit, out float vHit, out vec3 nHit) {
    vec2 ss; vec4 ab; float k;
    if (!gcBent(o, d, R, a, H, L, ss, ab, k)) return false;
    // fan seen edge-on: its blades overlap in projection (no ladder of slots and gaps)
    fr = mix(fr, 1.0, smoothstep(0.55, 0.9, abs(ab.w) / max(length(d.xz), 1e-4)));
    for (int j = 0; j < 2; j++) {
        float s = j == 0 ? ss.x : ss.y;
        if (s < 0.0 || s >= sMax) continue;
        float y = ab.x + ab.y * s;
        float v = y / H;
        if (v < 0.0 || v > 1.0) continue;
        float u = (ab.z + ab.w * s) / mix(Wb, Wt, v) + 0.5;
        if (u < 0.0 || u > 1.0) continue;
        float js = floor(u * nb);
        float uu = fract(u * nb) - 0.5;
        float vj = v / (0.5 + 0.5 * hash11(js * 1.618 + hs));
        if (vj > 1.0) continue;
        if (abs(uu) > 0.5 * fr * min(1.0, (1.0 - vj) * 3.0)) continue;
        sHit = s; vHit = vj; gcBladeU = uu / max(0.5 * fr, 1e-3);
        nHit = normalize(vec3(a.x, -2.0 * k * y, a.y));
        return true;
    }
    return false;
}

// (veg3) arching, drooping blades: the vertical plane through a holds the centre line
// y = c1 x - c2 x^2 (x = distance from the root along a): steep at the root, peak Hp at xp, then
// falling to the tip at x = Lx (Lx up to ~1.9 xp: the tip hangs down to ~1/3 of the peak). Fan of nb
// blades across the surface, as gcFan; v = x / Lx along the blade.
bool gcArchFan(vec3 o, vec3 d, vec3 R, vec2 a, float Hp, float xp, float Lx, float Wb, float Wt, float nb, float fr, float hs,
               float sMax, out float sHit, out float vHit, out float yHit, out vec3 nHit) {
    vec3 q = o - R;
    vec2 bd = vec2(-a.y, a.x);
    float a1 = dot(q.xz, a), b1 = dot(d.xz, a);
    float az = dot(q.xz, bd), bz = dot(d.xz, bd);
    float c2 = Hp / (xp * xp), c1 = 2.0 * c2 * xp;
    float A = c2 * b1 * b1, B = 2.0 * c2 * a1 * b1 - c1 * b1 + d.y, C = c2 * a1 * a1 - c1 * a1 + q.y;
    vec2 ss;
    if (abs(A) < 1e-9) {
        if (abs(B) < 1e-9) return false;
        ss = vec2(-C / B, -1.0);
    } else {
        float D = B * B - 4.0 * A * C;
        if (D < 0.0) return false;
        float sq = sqrt(D);
        float qq = -0.5 * (B + (B < 0.0 ? -sq : sq));
        ss = vec2(qq / A, C / qq);
        if (ss.x > ss.y) ss = ss.yx;
    }
    fr = mix(fr, 1.0, smoothstep(0.55, 0.9, abs(bz) / max(length(d.xz), 1e-4)));
    for (int j = 0; j < 2; j++) {
        float s = j == 0 ? ss.x : ss.y;
        if (s < 0.0 || s >= sMax) continue;
        float x = a1 + b1 * s;
        if (x < 0.0 || x > Lx) continue;
        float v = x / Lx;
        float u = (az + bz * s) / mix(Wb, Wt, min(v * 1.6, 1.0)) + 0.5;
        if (u < 0.0 || u > 1.0) continue;
        float js = floor(u * nb);
        float uu = fract(u * nb) - 0.5;
        float vj = v / (0.6 + 0.4 * hash11(js * 1.618 + hs));
        if (vj > 1.0) continue;
        if (abs(uu) > 0.5 * fr * min(1.0, (1.0 - vj) * 2.5)) continue;
        sHit = s; vHit = vj; yHit = q.y + d.y * s; gcBladeU = uu / max(0.5 * fr, 1e-3);
        float sl = c1 - 2.0 * c2 * x;
        nHit = normalize(vec3(-sl * a.x, 1.0, -sl * a.y));
        return true;
    }
    return false;
}

// leaning stem: cylinder of radius rad from R to R + (lean.x, H, lean.y) (sheared, so the 2D test is exact)
bool gcStem(vec3 o, vec3 d, vec3 R, vec2 lean, float H, float rad, float sMax, out float sHit, out vec3 nHit) {
    vec3 q = o - R;
    vec2 qx = q.xz - lean * (q.y / H);
    vec2 dx = d.xz - lean * (d.y / H);
    float A = dot(dx, dx), B = dot(qx, dx), C = dot(qx, qx) - rad * rad;
    float D = B * B - A * C;
    if (D < 0.0 || A < 1e-10) return false;
    float s = (-B - sqrt(D)) / A;
    if (s < 0.0 || s >= sMax) return false;
    float y = q.y + d.y * s;
    if (y < 0.0 || y > H) return false;
    vec2 r = qx + dx * s;
    sHit = s;
    nHit = normalize(vec3(r.x, 0.0, r.y));
    return true;
}

// axis-aligned ellipsoid, optionally rotated about y by (cos, sin) cs
bool gcEllipsoid(vec3 o, vec3 d, vec3 c, vec3 r, vec2 cs, float sMax, out float sHit, out vec3 nHit) {
    mat2 M = mat2(cs.x, cs.y, -cs.y, cs.x);
    vec3 oc = o - c; oc.xz = M * oc.xz;
    vec3 dd = d; dd.xz = M * dd.xz;
    oc /= r; dd /= r;
    float A = dot(dd, dd), B = dot(oc, dd), C = dot(oc, oc) - 1.0;
    float D = B * B - A * C;
    if (D < 0.0) return false;
    float s = (-B - sqrt(D)) / A;
    if (s < 0.0 || s >= sMax) return false;
    vec3 n = (oc + dd * s) / r;
    n.xz = transpose(M) * n.xz;
    sHit = s;
    nHit = normalize(n);
    return true;
}

// angular granite stone: an ellipsoid cut by a tilted top facet and two side facets (convex, so
// entry = max of the entries, exit = min of the exits); seed picks the facets
bool gcStone(vec3 o, vec3 d, vec3 c, vec3 r, vec2 cs, float seed, float sMax, out float sHit, out vec3 nHit) {
    mat2 M = mat2(cs.x, cs.y, -cs.y, cs.x);
    vec3 oc = o - c; oc.xz = M * oc.xz;
    vec3 dd = d; dd.xz = M * dd.xz;
    oc /= r; dd /= r;
    float A = dot(dd, dd), B = dot(oc, dd), C = dot(oc, oc) - 1.0;
    float D = B * B - A * C;
    if (D < 0.0) return false;
    float sq = sqrt(D);
    float sIn = (-B - sq) / A, sOut = (-B + sq) / A;
    vec3 mIn = oc + dd * sIn;            // normal (unit-sphere space) at the entry
    vec4 h = hash42i(ivec2(int(seed * 65536.0), 911));
    for (int k = 0; k < 3; k++) {
        float a = TAU * fract(h.x + float(k) * 0.37);
        vec3 m = k == 0 ? normalize(vec3((h.y - 0.5) * 0.9, 1.0, (h.z - 0.5) * 0.9))
                        : normalize(vec3(cos(a), 0.35 * (fract(h.w * 7.0 + float(k)) - 0.3), sin(a)));
        float cc = k == 0 ? mix(0.45, 0.78, h.w) : mix(0.55, 0.85, fract(h.y * 5.3 + float(k) * 0.71));
        float dm = dot(dd, m), om = dot(oc, m);
        if (abs(dm) < 1e-7) { if (om > cc) return false; continue; }
        float sp = (cc - om) / dm;
        if (dm < 0.0) { if (sp > sIn) { sIn = sp; mIn = m; } }
        else sOut = min(sOut, sp);
    }
    if (sIn >= sOut || sIn < 0.0 || sIn >= sMax) return false;
    vec3 n = mIn / r;
    n.xz = transpose(M) * n.xz;
    sHit = sIn;
    nHit = normalize(n);
    return true;
}

// dwarf shrub / understory bush: three seeded lobes (centres up to 0.4 R sideways and up to 0.15 h
// up) filled with 3D-noise foliage - ~18 cm lumps for a ragged outline, ~9 cm leaf clusters, ~3 cm
// sprigs where the march resolves them - porous at the rim, inside a hull 1.2x the shrub radii that
// the foliage never reaches (no smooth hull surface can show)
void gcLobes(float seed) {
    vec4 h = hash42i(ivec2(int(seed * 65536.0), 313));
    float a0 = h.x * TAU;
    for (int k = 0; k < 3; k++) {
        float fk = float(k);
        float a = a0 + fk * 2.094 + (fract(h.y * 7.0 + fk * 0.37) - 0.5) * 1.2;
        float rr = 0.45 * mix(0.4, 1.0, fract(h.z * 5.0 + fk * 0.61));
        float lr = mix(0.48, 0.66, fract(h.w * 3.0 + fk * 0.29));
        gcLo[k] = vec4(cos(a) * rr, 0.2 * fract(h.y * 11.0 + fk * 0.43), sin(a) * rr, lr);
    }
}
float gcFoliage(vec3 x, vec3 cen, vec3 rad, float seed, float fs, float fine) {
    vec3 u = (x - cen) / rad;
    float rho = 9.0;
    for (int k = 0; k < 3; k++) { vec3 q = (u - gcLo[k].xyz) / gcLo[k].w; rho = min(rho, dot(q, q)); }
    vec3 off = vec3(seed * 37.0, seed * 11.0, seed * 23.0);
    vec3 xs = x * fs;                                           // feature scale (small bushes: finer)
    float lump = texture(uNoise3, xs * 0.35 + off).g;           // ragged, lumpy outline
    float nv = texture(uNoise3, xs * 1.4 + off.yzx).r;          // leaf clusters
    float v = 0.20 * (1.0 - rho) + 1.0 * (lump - 0.487) + 0.75 * (nv - 0.475) - 0.06;
    if (fine > 0.0) v += fine * 0.45 * (texture(uNoise3, xs * 4.5 + off.zxy).b - 0.525);   // sprigs
    return v - 0.9 * smoothstep(0.85, 1.44, dot(u, u));
}
// leaves on a foliage surface: 3D cells (x in cell units; nearest of the 8 surrounding jittered
// feature points), each a leaf with its own tilt and tint. Returns (leaf randoms, F2 - F1: ~0 where
// two leaves meet - the dark gaps)
vec4 gcLeafCell(vec3 x) {
    vec3 g = floor(x - 0.5);
    float d1 = 9.0, d2 = 9.0; uint id = 0u;
    for (int k = 0; k < 8; k++) {
        vec3 c = g + vec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        uint h = hash3u(ivec3(c));
        vec3 r = c + 0.25 + 0.5 * vec3(uvec3(h, h >> 8u, h >> 16u) & 255u) * (1.0 / 255.0) - x;
        float d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; id = h; } else if (d < d2) d2 = d;
    }
    uint hv = hashu(id ^ 0x5bd1e995u);
    return vec4(vec3(uvec3(hv, hv >> 8u, hv >> 16u) & 255u) * (1.0 / 255.0), sqrt(d2) - sqrt(d1));
}
// shading version: plus a ~1 cm leaf term (weight lw, faded by the pixel footprint) for the normal
float gcFoliageS(vec3 x, vec3 cen, vec3 rad, float seed, float fs, float lw) {
    float v = gcFoliage(x, cen, rad, seed, fs, 1.0);
    return v + lw * 0.55 * (texture(uNoise3, x * fs * 6.5 + vec3(seed * 13.0, seed * 29.0, seed * 7.0)).a - 0.521);
}
bool gcShrub(vec3 o, vec3 d, vec3 cen, vec3 rad, float seed, float fs, float sMax, out float sHit) {
    vec3 hr = rad * 1.2;
    vec3 oc = (o - cen) / hr, dd = d / hr;
    float A = dot(dd, dd), B = dot(oc, dd), C = dot(oc, oc) - 1.0;
    float D = B * B - A * C;
    if (D < 0.0) return false;
    float sq = sqrt(D);
    float s0 = max((-B - sq) / A, 0.0), s1 = min((-B + sq) / A, sMax);
    if (s0 >= s1) return false;
    gcLobes(seed);
    // steps of 2-6 cm (/fs): up to 22 within 8 m of the camera, 14 beyond (a bush stops the ray
    // within its first ~0.8 m anyway); the 3 cm sprig octave only where the steps resolve it
    int N = length(cen - uCamPos) < 8.0 ? 22 : 14;
    float st = clamp((s1 - s0) / float(N), 0.02 / fs, 0.06 / fs);
    float fine = 1.0 - smoothstep(0.025, 0.04, st * fs);
    for (int i = 0; i <= 22; i++) {
        float s = s0 + float(i) * st;
        if (i > N || s > s1) break;
        if (gcFoliage(o + d * s, cen, rad, seed, fs, fine) > 0.0) {
            float a = max(s - st, s0), b = s;
            if (i > 0) {
                for (int k = 0; k < 2; k++) {
                    float m = 0.5 * (a + b);
                    if (gcFoliage(o + d * m, cen, rad, seed, fs, fine) > 0.0) b = m; else a = m;
                }
            }
            sHit = b;
            return true;
        }
    }
    return false;
}

// cheap reject of a tiller (root R, height H, horizontal reach ext) that the ray cannot touch
vec2 gcDn = vec2(1.0, 0.0);   // normalized xz direction of the segment ray
float gcDl = 1.0;             // |d.xz|
bool gcMiss(vec3 o, vec3 d, vec3 R, float H, float ext) {
    if (gcDl < 0.05) return false;
    vec2 rel = R.xz - o.xz;
    if (abs(rel.x * gcDn.y - rel.y * gcDn.x) > ext) return true;
    float sC = dot(rel, gcDn) / gcDl;
    float yR = o.y + d.y * sC - R.y;
    float dy = ext * abs(d.y) / gcDl;
    return yR - dy > H || yR + dy < -0.05 || sC < -ext / gcDl;
}

// ---- coarse cell: one shrub or one stone ------------------------------------------------------
// returns 0 none, 1 shrub, 2 stone
int gcCoarseObj(ivec2 c, out vec3 cen, out vec3 rad, out float ang, out float seed) {
    gcS = hash2u(c + ivec2(3301, 9203));
    float r0 = gcRnd(), r1 = gcRnd(), r2 = gcRnd(), r3 = gcRnd(), r4 = gcRnd(), r5 = gcRnd();
    seed = r5; ang = r4 * TAU;
    cen = vec3(0.0); rad = vec3(1.0);
    // (objects reach at most GC_CO beyond their cell: the hull of the largest shrub is 1.2 Rmax)
    float Rmax = mix(0.30, 0.90, r3);
    float jit = 0.5 * GC_C2 + GC_CO - 1.2 * Rmax;
    vec2 pos = (vec2(c) + 0.5) * GC_C2 + (vec2(r1, r2) * 2.0 - 1.0) * jit;
    // cheap upper bound of pShrub + pStone before the habitat
    float wn = smoothstep(0.2, 0.5, vnoise(pos * 0.21 + 3.0));
    float stP = step(0.45, vnoise(pos * 0.12 + 5.0));
    // (i24) + the grassy-shore cushions (below) where a lake / stream is near: the bound cut them to 3 %
    float wl0 = waterLevel(pos);
    float shN = wl0 > -1000.0 ? smoothstep(0.3, 0.6, vnoise(pos * 0.35 + 17.0)) : 0.0;
    if (r0 >= 0.03 + 0.7 * wn + 0.55 * stP + 0.8 * shN + (gcTrailSeg ? 0.75 : 0.0)) return 0;
    vec2 g;
    float y = gcGroundG(pos, g);
    float camD = length(vec3(pos.x, y, pos.y) - uCamPos);
    if (camD > GC_FADE1) return 0;
    float wl = wl0;
    if (wl > y - 0.1) return 0;
    GBio b = gcBiome(pos, y, g, wl);
    float open = 1.0 - b.forest;
    float sink = smoothstep(GC_FADE1 - 3.5, GC_FADE1, camD);        // into the ground at the far end
    float slope = length(g);
    // dwarf shrubs: rare in the open; on the forest floor a dark leafy understory, in patches
    // and along the stand edges
    float woodS = smoothstep(0.2, 0.6, b.wood) * wn;
    float pShrub = b.veg * (1.0 - b.shore) * (0.03 * open * (1.0 - b.meadow) + 0.7 * woodS) * (1.0 - b.trail);
    // (veg4 T15) a shrub band along the stand edges (both sides), where the light comes in
    float edgeS = smoothstep(0.05, 0.3, b.wood) * (1.0 - smoothstep(0.55, 0.9, b.wood));
    pShrub += 0.35 * edgeS * wn * b.veg * (1.0 - b.shore) * (1.0 - b.trail);
    // the trail through the forest: a dense leafy understory right up to the verge, kept
    // off the dirt
    float vergeS = smoothstep(0.7, 1.2, b.tdist - 0.75 * Rmax) * (1.0 - smoothstep(3.5, 6.0, b.tdist)) * (1.0 - b.shore);   // (i17: also where the stand opens up; veg4 G4: the narrower path)
    pShrub += 0.75 * vergeS * b.veg * (1.0 - b.shore);
    pShrub *= smoothstep(0.55, 0.85, b.tdist - 1.15 * Rmax);        // never reaching over the dirt
    // (i21) juniper / alpenrose cushions on the grassy shores and floodplains between the boulders
    // (shot 4) - pShrub excluded the shore band entirely
    pShrub += 0.8 * b.veg * b.sgrB * shN * (1.0 - b.trail);      // (veg4 G6: 0.45 -> 0.8, bilberry / alpenrose / juniper between the boulders)
    float pStone = b.veg * (1.0 - b.shore) * (0.03 + 0.17 * b.meadow * open + 0.02 * b.forest) * stP * 2.5 * (1.0 - b.trail);
    if (r0 < pShrub) {
        // on the forest floor about half of them are low bilberry / dwarf-shrub carpets
        float carpet = step(0.5, fract(r5 * 7.7)) * smoothstep(0.2, 0.6, b.wood);
        float h = min(Rmax * mix(mix(0.5, 0.9, r1), mix(0.24, 0.38, r1), carpet), 0.48);
        rad = vec3(Rmax, h, Rmax * mix(0.65, 1.0, r2));
        // (veg4 G6) in the shore's boulder belt a third are upright juniper / dwarf-pine / sapling columns
        if (fract(r5 * 3.3) > 0.66 && b.sgrB > 0.3 && woodS < 0.3) {
            h = mix(0.36, 0.50, r1);
            rad = vec3(mix(0.20, 0.30, r2), h, mix(0.18, 0.28, r3));
        }
        // (sunk on slopes: no floating downhill side)
        cen = vec3(pos.x, y + h * 0.42 - 0.8 * Rmax * slope - sink * h * 1.6, pos.y);
        return 1;
    }
    if (r0 < pShrub + pStone) {
        // granite stones; on the meadows also flat half-buried slabs up to ~1.2 m
        float mo = smoothstep(0.3, 0.6, b.meadow * open);
        float R = mix(mix(0.07, 0.26, pow(r3, 1.4)), mix(0.2, 0.62, pow(r3, 0.8)), mo);
        rad = vec3(R, R * mix(0.5, 0.8, r1) * mix(1.0, 0.6, smoothstep(0.25, 0.5, R)), R * mix(0.65, 1.0, r2));
        float k = mix(mix(-0.45, 0.05, r2), mix(-0.35, 0.15, r2), mo);       // meadow slabs half buried
        cen = vec3(pos.x, y + rad.y * k - sink * rad.y * 1.4 - 0.02 - 0.5 * R * slope, pos.y);
        float dy = (y - cen.y) / rad.y;
        gcStoneFoot = sqrt(max(1.0 - dy * dy, 0.0));
        return 2;
    }
    return 0;
}

// ---- leaves and flower heads as flat blades --------------------------------------------------------
// leaf frame from the midrib azimuth phi, its elevation above the horizontal and a roll about it
void gcLeafFrame(float phi, float elev, float roll, out vec3 tu, out vec3 n) {
    float cp = cos(phi), sp = sin(phi), ce = cos(elev), se = sin(elev);
    tu = vec3(ce * cp, se, ce * sp);
    vec3 side = vec3(-sp, 0.0, cp);
    n = vec3(-cp * se, ce, -sp * se) * cos(roll) + side * sin(roll);
}
// flat leaf: centre c, normal n, midrib tu, half length a, half width b; tip > 0 makes it ovate
// (widest below the middle, pointed). uv = (along -1..1, across -1..1)
bool gcLeafDisc(vec3 o, vec3 d, vec3 c, vec3 n, vec3 tu, float a, float b, float tip, float sMax,
                out float s, out vec2 uv) {
    s = 0.0; uv = vec2(0.0);
    float dn = dot(d, n);
    if (abs(dn) < 1e-6) return false;
    s = dot(c - o, n) / dn;
    if (s <= 0.0 || s >= sMax) return false;
    vec3 q = o + d * s - c;
    float u = dot(q, tu) / a;
    if (abs(u) > 1.0) return false;
    float v = dot(q, cross(n, tu)) / b;
    float w = sqrt(1.0 - u * u) * (1.0 - tip * u);
    if (abs(v) > w) return false;
    uv = vec2(u, v / max(w, 1e-3));
    return true;
}
// round flower head facing n: returns radial coordinate (0..1) and angle
bool gcHeadDisc(vec3 o, vec3 d, vec3 c, vec3 n, float R, float sMax, out float s, out vec2 ra) {
    s = 0.0; ra = vec2(0.0);
    float dn = dot(d, n);
    if (abs(dn) < 1e-6) return false;
    s = dot(c - o, n) / dn;
    if (s <= 0.0 || s >= sMax) return false;
    vec3 q = o + d * s - c;
    float r = length(q) / R;
    if (r > 1.0) return false;
    vec3 e1 = normalize(cross(n, abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    ra = vec2(r, atan(dot(q, cross(n, e1)), dot(q, e1)));
    return true;
}
float gcRayDist(vec3 o, vec3 d, vec3 c) {          // distance of c from the ray (d normalized)
    vec3 q = c - o;
    return length(q - d * max(dot(q, d), 0.0));
}

// ---- medium cell: blueberry bush / forb rosette / flower clump ----------------------------------
// blueberry / bilberry: a dome of leafy sprigs. Each sprig is a short twig with alternate small
// leaves and its own seed, so sprigs the ray misses are skipped without drawing their numbers.
bool gcBerry(vec3 o, vec3 d, vec2 cc, float yC, vec2 g, float fade, float wPix, float scale, inout float sBest) {
    float R = min(mix(0.17, 0.235, gcRnd()) * scale, 0.215);  // sprigs + leaves stay inside the cell
    float H = mix(0.22, 0.42, gcRnd()) * fade * scale;
    vec2 ctr = cc + (vec2(gcRnd(), gcRnd()) - 0.5) * max(GC_CM - 2.3 * R - 0.02, 0.0);
    float sd = gcRnd();
    uint key = gcS;
    if (H < 0.03) return false;
    vec3 C = vec3(ctr.x, yC + dot(g, ctr - cc) - 0.03, ctr.y);
    if (gcMiss(o, d, C, H + 0.04, R + 0.01)) return false;
    float a = 0.025, bw = 0.015;
    float k = max(1.0, 0.75 * wPix / (2.0 * bw));
    a *= k; bw *= k;
    float keep = 1.0 / (k * k);
    bool hit = false;
    // dense leafy core (bilberry / understory mass): noise foliage at a fine scale
    {
        vec3 cc3 = C + vec3(0.0, H * 0.24, 0.0), cr3 = vec3(R * 0.95, H * 0.36, R * 0.85);   // low, wide cushion
        float s;
        if (gcShrub(o, d, cc3, cr3, sd, 3.2, sBest, s)) {
            sBest = s; hit = true;
            gcInfo = vec4(GK_SHRUB, (o.y + d.y * s) - C.y, sd, 1.0);
            gcObj = vec4(cc3, sd); gcObjR = vec4(cr3, 3.2);
            gcTopH = H;
        }
    }
    int nSp = wPix > 0.005 ? 6 : 9;                  // sprigs sticking out of the core
    float spS = wPix > 0.005 ? 1.2 : 1.0;
    for (int j = 0; j < 13; j++) {
        if (j >= nSp) break;
        gcS = hashu(key + uint(j) * 0x9E3779B9u);
        float q0 = gcRnd(), q1 = gcRnd(), q2 = gcRnd(), q3 = gcRnd();
        float rr = sqrt(q0) * (R - 0.07);
        float ang = q1 * TAU;
        float hy = H * mix(0.42, 1.0, q2) * sqrt(max(1.0 - 0.6 * (rr / R) * (rr / R), 0.0));
        vec3 S = C + vec3(cos(ang) * rr, hy, sin(ang) * rr);
        // twig: centred on S, mostly horizontal, pointing outwards
        float ta = ang + (q3 - 0.5) * 1.6;
        vec3 tw = normalize(vec3(cos(ta), 0.35 * (q2 - 0.3), sin(ta)));
        float tl = 0.065;
        if (gcRayDist(o, d, S) > tl + 2.0 * a + 0.004) continue;
        vec3 sideV = normalize(cross(vec3(0.0, 1.0, 0.0), tw));
        for (int i = 0; i < 9; i++) {
            float w0 = gcRnd(), w1 = gcRnd(), w2 = gcRnd(), w3 = gcRnd();
            if (w3 > keep) continue;
            float u = (float(i) + 0.5) / 9.0 * 2.0 - 1.0;
            float side = (i % 2 == 0) ? 1.0 : -1.0;
            vec3 base = S + tw * (u * tl) + vec3(0.0, 0.006 * w2, 0.0);
            vec3 sv = sideV * side + tw * 0.6;
            float phi = atan(sv.z, sv.x) + (w0 - 0.5) * 0.8;
            vec3 tu, n;
            gcLeafFrame(phi, (w1 - 0.35) * 1.0, (w2 - 0.5) * 0.9, tu, n);
            float sz = mix(0.7, 1.2, w0) * (1.0 - 0.25 * abs(u)) * spS;
            float s; vec2 uv;
            if (gcLeafDisc(o, d, base + tu * (a * sz), n, tu, a * sz, bw * sz, 0.2, sBest, s, uv)) {
                sBest = s; hit = true;
                gcInfo = vec4(GK_BERRY, base.y - C.y, fract(sd * 3.1 + w2 * 0.37), w1);
                gcN = n; gcT = tu; gcUV = uv; gcTopH = H;
            }
        }
    }
    return hit;
}

// forb. species: 0 lady's-mantle (round, low), 1 tall leafy stem (broad pleated leaves),
// 2 lanceolate rosette, 3 broad ribbed leaves spiralling up a short stem (veratrum, adenostyles).
// big = 0 picks 0 / 2 (medium cells), big = 1 picks 1 / 3 (1 m cells). ctr: root, yR: ground there
// (veg4 G3a) leafy sprigs on a dwarf shrub's hull (coarse cells, within ~10 m): 5-9 cm leaves on short twigs
// sticking out of the foliage volume, so the bush reads as leaves (bilberry, alpenrose), not as a
// noise-hull "moss ball / hedgehog". Shaded as GK_BERRY leaves (lit lime when back-lit)
bool gcShrubSprigs(vec3 o, vec3 d, vec3 cen, vec3 rad, float seed, float camD, inout float sBest) {
    float wPix = camD * gPixAng * 0.9;
    float ss = clamp(rad.x / 0.7, 0.5, 1.0);               // (small bushes: smaller sprigs, inside the cell's reach)
    float a = 0.032 * ss, bw = 0.018 * ss;
    float k = max(1.0, 0.75 * wPix / (2.0 * bw));
    a *= k; bw *= k;
    float keep = 1.0 / (k * k);
    float yG = cen.y - rad.y * 0.42;                       // ~ the ground under the bush
    uint key = hash2u(ivec2(int(seed * 65536.0), 977));
    int nSp = camD < 6.0 ? 16 : 11;
    bool hit = false;
    for (int j = 0; j < 16; j++) {
        if (j >= nSp) break;
        gcS = hashu(key + uint(j) * 0x9E3779B9u);
        float q0 = gcRnd(), q1 = gcRnd(), q2 = gcRnd(), q3 = gcRnd();
        float u = mix(-0.15, 0.92, q0);
        float az = q1 * TAU;
        vec3 dir = vec3(cos(az) * sqrt(1.0 - u * u), u, sin(az) * sqrt(1.0 - u * u));
        vec3 S = cen + dir * rad * mix(0.78, 0.95, q2);
        if (S.y < yG + 0.04) continue;
        float tl = mix(0.05, 0.09, q3) * ss;
        if (gcRayDist(o, d, S) > tl + 2.0 * a + 0.006) continue;
        float ta = az + (q3 - 0.5) * 1.4;
        vec3 tw = normalize(vec3(cos(ta), 0.5 * u + 0.25 * (q2 - 0.5), sin(ta)));
        vec3 sideV = normalize(cross(vec3(0.0, 1.0, 0.0), tw) + vec3(1e-4, 0.0, 0.0));
        for (int i = 0; i < 7; i++) {
            float w0 = gcRnd(), w1 = gcRnd(), w2 = gcRnd(), w3 = gcRnd();
            if (w3 > keep) continue;
            float uu = (float(i) + 0.5) / 7.0 * 2.0 - 1.0;
            float side = (i % 2 == 0) ? 1.0 : -1.0;
            vec3 base = S + tw * (uu * tl);
            vec3 sv = sideV * side + tw * 0.5;
            float phi = atan(sv.z, sv.x) + (w0 - 0.5) * 0.9;
            vec3 tu, n;
            gcLeafFrame(phi, (w1 - 0.3) * 1.1, (w2 - 0.5) * 1.0, tu, n);
            float sz = mix(0.75, 1.3, w0) * (1.0 - 0.3 * abs(uu));
            float s; vec2 uv;
            if (gcLeafDisc(o, d, base + tu * (a * sz), n, tu, a * sz, bw * sz, 0.25, sBest, s, uv)) {
                sBest = s; hit = true;
                gcInfo = vec4(GK_BERRY, base.y - yG, fract(seed * 3.1 + w2 * 0.37), w1);
                gcN = n; gcT = tu; gcUV = uv; gcTopH = rad.y * 1.4;
            }
        }
    }
    return hit;
}
bool gcForbAt(vec3 o, vec3 d, vec2 ctr, float yR, float big, float maxReach, float fade, float wPix, float tall, inout float sBest) {
    float r0 = gcRnd(), r1 = gcRnd(), r2 = gcRnd();
    float sp = big < 0.5 ? (r0 < 0.55 ? 0.0 : 2.0) : (r0 < 0.5 ? 1.0 : 3.0);
    float plantR = fract(r1 * 7.7 + r2 * 3.1);
    float a = sp < 0.5 ? mix(0.03, 0.05, r1) : (sp < 1.5 ? mix(0.05, 0.095, r1) : (sp < 2.5 ? mix(0.06, 0.10, r1) : mix(0.07, 0.12, r1)));
    a = min(a, 0.5 * (maxReach - 0.02));
    float bw = a * (sp < 0.5 ? 0.95 : (sp < 1.5 ? 0.55 : (sp < 2.5 ? 0.24 : 0.48)));
    float Hs = sp < 0.5 ? mix(0.03, 0.10, r2) : (sp < 1.5 ? mix(0.16, 0.42, r2) * tall : (sp < 2.5 ? 0.0 : mix(0.06, 0.26, r2) * tall));
    Hs *= fade;
    float reach = 2.0 * a + 0.02;
    vec3 C = vec3(ctr.x, yR - 0.015, ctr.y);
    float top = Hs + 2.0 * a + 0.02;
    if (gcMiss(o, d, C, top, reach)) return false;
    if (bw < 0.5 * wPix) { float k = 0.5 * wPix / bw; bw *= k; a *= min(k, 1.4); }
    int nl = sp < 0.5 ? 6 : (sp < 1.5 ? 7 : (sp < 2.5 ? 6 : 7));
    float a0 = gcRnd() * TAU;
    bool hit = false;
    for (int i = 0; i < nl; i++) {
        float w0 = gcRnd(), w1 = gcRnd(), w2 = gcRnd(), w3 = gcRnd();
        float phi = a0 + float(i) * 2.4 + (w0 - 0.5) * 0.6;          // golden-angle phyllotaxis
        float el, y0, sz = 1.0;
        float lv = float(i) / float(nl);
        if (sp < 0.5)      { el = mix(0.05, 0.45, w1); y0 = Hs * mix(0.5, 1.0, w2); }
        else if (sp < 1.5) { el = mix(0.15, 0.75, w1) + 0.3 * lv; y0 = Hs * lv * 0.85 + 0.02; sz = mix(1.0, 0.55, lv); }
        else if (sp < 2.5) { el = mix(0.25, 0.95, w1); y0 = 0.005; }
        else               { el = mix(0.20, 0.65, w1) + 0.4 * lv; y0 = Hs * lv * 0.9 + 0.015; sz = mix(1.0, 0.6, lv); }
        el *= mix(1.0, 0.6, 1.0 - fade);
        vec3 tu, n;
        gcLeafFrame(phi, el, (w3 - 0.5) * 0.5, tu, n);
        float pet = sp < 0.5 ? 0.015 : 0.008;
        vec3 base = C + vec3(0.0, y0, 0.0) + vec3(cos(phi), 0.0, sin(phi)) * pet;
        float s; vec2 uv;
        if (gcLeafDisc(o, d, base + tu * (a * sz), n, tu, a * sz, bw * sz, sp < 0.5 ? 0.0 : 0.25, sBest, s, uv)) {
            sBest = s; hit = true;
            gcInfo = vec4(GK_FLEAF, (o.y + d.y * s) - C.y, fract(0.7 * plantR + 0.3 * fract(w2 * 7.1 + w0)) + sp * 2.0, uv.x);
            gcN = n; gcT = tu; gcUV = uv; gcTopH = max(0.3 * tall * fade, 0.05);
        }
    }
    // the stem of the tall forbs
    if ((sp > 0.5 && sp < 1.5 || sp > 2.5) && Hs > 0.05) {
        float s; vec3 n;
        if (gcStem(o, d, C, vec2(0.0), Hs, max(sp > 2.5 ? 0.005 : 0.003, wPix * 0.8), sBest, s, n)) {
            sBest = s; hit = true;
            gcInfo = vec4(GK_STEM, (o.y + d.y * s) - C.y, 0.5, 0.0);
            gcN = n; gcTopH = max(0.3 * tall * fade, 0.05);
        }
    }
    return hit;
}
bool gcForb(vec3 o, vec3 d, vec2 cc, float yC, vec2 g, float fade, float wPix, float tall, inout float sBest) {
    vec2 j = vec2(gcRnd(), gcRnd()) - 0.5;
    vec2 ctr = cc + j * (GC_CM - 0.26);
    float mr = 0.5 * GC_CM - max(abs(j.x), abs(j.y)) * (GC_CM - 0.26) - 0.01;
    return gcForbAt(o, d, ctr, yC + dot(g, ctr - cc), 0.0, mr, fade, wPix, tall, sBest);
}

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
bool gcFlowers(vec3 o, vec3 d, vec2 cc, float yC, vec2 g, float fade, float wPix, float tall, inout float sBest) {
    float sp = gcSpecies(cc, gcRnd());
    float r1 = gcRnd(), r2 = gcRnd(), r3 = gcRnd(), r4 = gcRnd();
    int nh = sp < 0.5 ? 3 + int(r1 * 4.0) : (sp < 1.5 ? 2 + int(r1 * 3.0) : (sp < 2.5 ? 1 + int(r1 * 1.6) : 1 + int(r1 * 3.0)));
    float hs = (sp < 0.5 ? mix(0.26, 0.52, r2) : (sp < 1.5 ? mix(0.18, 0.38, r2) : (sp < 2.5 ? mix(0.34, 0.58, r2) : mix(0.10, 0.24, r2)))) * fade;   // (i18: +15-30 %)
    vec2 ctr = cc + (vec2(r3, r4) - 0.5) * 0.18;
    vec3 R = vec3(ctr.x, yC + dot(g, ctr - cc) - 0.01, ctr.y);
    if (hs < 0.04) return false;
    if (gcMiss(o, d, R, hs + 0.08, 0.16)) return false;
    bool hit = false;
    for (int i = 0; i < nh; i++) {
        float w0 = gcRnd(), w1 = gcRnd(), w2 = gcRnd(), w3 = gcRnd(), w4 = gcRnd();
        float Hs = hs * (1.0 - 0.18 * float(i)) * (0.85 + 0.3 * w0);
        vec2 lean = (vec2(w1, w2) - 0.5) * (0.10 + 0.05 * float(i));
        vec3 hc = R + vec3(lean.x, Hs, lean.y);
        float s; vec3 n;
        if (gcStem(o, d, R, lean, Hs, max(0.0018, wPix * 0.8), sBest, s, n)) {
            sBest = s; hit = true;
            gcInfo = vec4(GK_STEM, (o.y + d.y * s) - R.y, w3, sp);
            gcN = n; gcTopH = max(0.25 * tall * fade, 0.05);
        }
        if (sp < 0.5 || (sp > 1.5 && sp < 2.5)) {
            // flat heads: yellow composites / buttercups, white umbels
            float Rh0 = sp < 0.5 ? mix(0.015, 0.038, w3 * w3) : mix(0.032, 0.052, w3);   // buttercups .. arnica
            // LOD: at least ~1.8 px wide, its colour faded by the area ratio (constant energy)
            float Rh = max(Rh0, 0.9 * wPix);
            float tl = sp < 0.5 ? 1.2 : 0.4;
            vec3 hn = normalize(vec3(lean.x * 2.0 + (w4 - 0.5) * tl, 1.0, lean.y * 2.0 + (w0 - 0.5) * tl));
            vec2 ra;
            if (gcHeadDisc(o, d, hc + vec3(0.0, 0.004, 0.0), hn, Rh, sBest, s, ra)) {
                float wr = Rh / max(wPix, 1e-4);
                bool keepH = true;
                if (sp < 0.5) {      // 5 broad petals (buttercup) or many rays (arnica)
                    float np = w2 < 0.5 ? 5.0 : 12.0;
                    float pet = 0.5 + 0.5 * cos(ra.y * np + w1 * 6.0);
                    float edge = np < 6.0 ? mix(0.62, 1.0, sqrt(pet)) : mix(0.55, 1.0, pet);
                    keepH = ra.x < mix(1.0, edge, saturate(wr / 4.0 - 0.5)) || ra.x < 0.32;
                } else {             // umbel: rays ending in round floret clusters
                    float sub = 0.5 + 0.5 * cos(ra.y * 9.0 + w1 * 6.0);
                    keepH = ra.x < mix(1.0, 0.72 + 0.28 * sub, saturate(wr / 5.0 - 0.5));
                }
                if (keepH) {
                    sBest = s; hit = true;
                    gcInfo = vec4(GK_FDISC, Hs, fract(w1 * 7.7 + float(i) * 0.31), sp);
                    gcN = hn; gcUV = ra; gcTopH = max(0.25 * tall * fade, 0.05);
                    gcObj = vec4(hc, Rh); gcFadeK = (Rh0 / Rh) * (Rh0 / Rh);
                }
            }
        } else {
            // spikes of small florets (lousewort, betony) / bells (gentian, campanula)
            vec3 hr = sp < 1.5 ? vec3(0.018, 0.046, 0.018) * (0.8 + 0.4 * w3) : vec3(0.011, 0.014, 0.011) * (0.9 + 0.3 * w3);
            vec3 hr0 = hr;
            hr = max(hr, vec3(wPix * 0.7));
            vec3 hcc = hc + vec3(0.0, hr.y * 0.6, 0.0);
            if (gcEllipsoid(o, d, hcc, hr, vec2(1.0, 0.0), sBest, s, n)) {
                sBest = s; hit = true;
                gcInfo = vec4(GK_HEAD, Hs, fract(w1 * 7.7 + float(i) * 0.31), sp);
                gcN = n; gcTopH = max(0.25 * tall * fade, 0.05);
                gcObj = vec4(hcc, 0.0); gcObjR = vec4(hr, sp); gcFadeK = hr0.x * hr0.y / (hr.x * hr.y);
            }
        }
    }
    return hit;
}

bool gcMid(ivec2 c, vec3 o, vec3 d, inout float sBest) {
    vec2 cc = (vec2(c) + 0.5) * GC_CM;
    vec2 g;
    float yC = gcGroundG(cc, g);
    float camDc = length(vec3(cc.x, yC, cc.y) - uCamPos);
    // distance LOD: the cell's plants live while their hash is below gcKeep (thinning by count)
    float keep = gcKeep(camDc), rk = hash12i(c + ivec2(907, 3301));
    if (rk >= keep || camDc > GC_FADE1) return false;
    float fade = gcGrow(keep, rk) * gcShrink(camDc);
    float wl = waterLevel(cc);
    if (wl > yC - 0.25) return false;
    GBio b = gcBiome(cc, yC, g, wl);
    if (b.veg < 0.02) return false;
    float nearR;
    float rm = gcRockMask(cc, nearR) * (1.0 - b.trail);
    if (rm < 0.05) return false;
    float canopy = gcCanopy(cc);
    float ff = gcFloor(b, canopy);
    float open = 1.0 - ff;
    float shoreS = smoothstep(0.25, 0.6, b.shore);
    float wPix = camDc * gPixAng * 0.9;
    gcS = hash2u(c + ivec2(4421, 6113));
    float r0 = gcRnd(), r1 = gcRnd(), r2 = gcRnd(), r3 = gcRnd();
    float fpatch = vnoise(cc * 0.31 + 41.0);
    float pat = vnoise(cc * 0.55 + 13.0);
    float tall = (0.75 + 0.5 * pat) * mix(1.0, 0.7, ff);
    float veg = b.veg * rm * (1.0 - shoreS);
    // (bilberry / understory patches; thickest along the trail verge, where the light comes in)
    // (dense, merged carpets in ~1/4 of the floor - not a planted pattern of single bushes)
    float vergeB = smoothstep(0.7, 1.15, b.tdist) * (1.0 - smoothstep(3.0, 5.5, b.tdist));   // (veg4 G4: narrower path)
    float pBerry = 0.4 * veg * ff * smoothstep(0.46, 0.60, vnoise(cc * 0.33 + 5.0) + 0.3 * vergeB) * (1.0 - 0.45 * canopy * canopy);
    // dwarf shrubs (lingonberry, alpine rose, willow) on the grassy shores, mostly by the boulders
    // (i12: bilberry / alpine-rose / juniper cushions in most gaps of the boulder field)
    pBerry += b.veg * rm * max(b.sgr * smoothstep(0.15, 0.45, b.above), 0.8 * b.sgrB) * (0.20 + 0.7 * nearR) * smoothstep(0.25, 0.50, vnoise(cc * 0.5 + 61.0) + 0.35 * nearR);   // (i17: x2)
    // broad-leaved forbs dominate the meadow foreground; flowers 1-5 % of the meadow pixels
    float pForb = veg * (0.10 + 0.9 * b.meadow * open + 0.10 * open + 0.12 * ff);
    float pFlow = veg * open * (0.04 + 0.25 * b.meadow + 0.5 * b.meadow * smoothstep(0.3, 0.7, fpatch));
    bool hit = false;
    uint key = gcS;
    // (irregular: gaps and size variety inside the patches, no planted rows)
    if (r0 < pBerry * mix(0.6, 1.0, vnoise(cc * 1.9 + 7.0))) {
        gcS = hashu(key + 11u);
        hit = gcBerry(o, d, cc, yC, b.g, fade, wPix, mix(0.75, 1.15, r3) * mix(0.8, 1.0, pBerry), sBest);
    } else {
        if (r1 < pForb) {
            gcS = hashu(key + 23u);
            hit = gcForb(o, d, cc, yC, b.g, fade, wPix, tall, sBest);
        }
        if (r3 < pForb * b.meadow * 0.5) {          // a second rosette on the meadows
            gcS = hashu(key + 29u);
            hit = gcForb(o, d, cc, yC, b.g, fade, wPix, tall, sBest) || hit;
        }
    }
    if (r2 < pFlow) {
        gcS = hashu(key + 37u);
        hit = gcFlowers(o, d, cc, yC, b.g, fade, wPix, tall, sBest) || hit;
    }
    if (fract(r2 * 7.31 + r0) < pFlow - 0.55) {      // dense flower patches: a second clump
        gcS = hashu(key + 41u);
        hit = gcFlowers(o, d, cc, yC, b.g, fade, wPix, tall, sBest) || hit;
    }
    if (hit) gcEnv = vec2(ff, b.meadow);
    return hit;
}

// ---- fine cell: grass patch / tussock + forb + flower -----------------------------------------
bool gcFine(ivec2 c, vec3 o, vec3 d, float yC, vec2 gC, inout float sBest) {
    vec2 c0 = vec2(c) * GC_C;
    vec2 cc = c0 + 0.5 * GC_C;
    float camDc = length(vec3(cc.x, yC, cc.y) - uCamPos);
    if (camDc > GC_FADE1) return false;
    float wl = waterLevel(cc);
    if (wl > yC - 0.05) return false;          // (veg4 G15: no tillers standing in the water - shot08's flooded lawn)
    GBio b = gcBiome(cc, yC, gC, wl);
    if (b.veg < 0.02) return false;
    float eR = gcRockE(cc);
    float nearR = 1.0 - smoothstep(1.0, 1.7, eR);
    // (cells deep inside a boulder / stone footprint are skipped; elsewhere each tiller root is tested)
    float rm = smoothstep(0.75, 1.0, eR) * (1.0 - b.trail);
    if (rm < 0.05) return false;
    gcS = hash2u(c + ivec2(7919, 1031));
    float rc = gcRnd();
    float pat = vnoise(cc * 0.55 + 13.0);
    float canopy = gcCanopy(cc);
    float ff = gcFloor(b, canopy);
    float keep = gcKeep(camDc), shr = gcShrink(camDc);
    float k3c = min(keep, 1.0) * shr;          // fraction of the 3D cover drawn here
    // grassy shores: grass down to ~15 cm above the water, thinning to stony gaps away from boulders
    float sgGap = smoothstep(0.12, 0.38, vnoise(cc * 0.7 + 17.0) + 0.35 * nearR);
    float tufts = smoothstep(0.3, 0.7, vnoise(cc * 2.2 + 9.0));
    float vegE = mix(b.vegT, b.veg, k3c);
    float dens = vegE * rm * (1.0 - smoothstep(0.25, 0.6, b.shore)) * (0.5 + 0.65 * pat) * (1.0 - 0.3 * b.meadow);
    dens *= mix(0.45, 1.45, tufts);                                                // tufts and gaps
    // forest floor: dry-grass tussocks 0.3-0.6 m across, only in the sunny gaps (~35 %
    // cover there, < 5 % under the crowns), none in the bilberry patches; litter between them
    float bpF = smoothstep(0.42, 0.58, vnoise(cc * 0.33 + 5.0));
    float gpF = smoothstep(0.5, 0.75, vnoise(cc * 0.6 + 3.0));
    float tusF = smoothstep(0.58, 0.66, vnoise(cc * 1.6 + 9.0)) * gpF * (1.0 - canopy) * (1.0 - 0.85 * bpF);
    dens = mix(dens, b.veg * rm * 0.35 * tusF, ff);
    // (i19: strongly clumped - dense tufts and thin gaps, was an even field at the tiller cap)
    float clS = smoothstep(0.25, 0.75, vnoise(cc * 1.6 + 29.0) * 0.65 + vnoise(cc * 4.1 + 3.0) * 0.35);
    float sgD = vegE * rm * smoothstep(0.01, 0.08, b.above) * mix(0.6, 1.3, sgGap) * mix(0.8, 1.4, clS);   // (veg3: lush; veg4 G6: 0.3-1.6 -> 0.8-1.4, continuous cover, not islands)
    dens = mix(dens, sgD, b.sgr);
    // trail verge (0.7 m outside the dirt): shorter, thinner grass leaning over the trail
    float tv = b.tedge * (1.0 - b.trail);
    dens *= mix(1.0, 0.6, tv);
    // (i17) a thin sward where the terrain draws bare ground (scree, rock slabs) grows in tufts: tillers
    // gather in ~0.3 m clumps (scattered single tillers read as stubble)
    float thin = (1.0 - smoothstep(0.3, 0.6, dens)) * (1.0 - smoothstep(0.3, 0.7, b.vegT)) * (1.0 - b.sgr);
    if (thin > 0.0) dens *= mix(1.0, 2.4 * smoothstep(0.52, 0.72, vnoise(cc * 3.3 + 41.0)), thin);
    // (i17: shore heights clumped - short turf next to tall flopping tufts)
    float tall = mix(1.0, 0.7, ff) * mix(0.68 + 0.6 * pat, 0.45 + 0.75 * pat, b.meadow) * (1.0 + 0.12 * b.meadow)   // (i18: meadow grass a bit lower, flowers stand above it)
               * mix(1.0, mix(0.42, 1.30, smoothstep(0.2, 0.8, vnoise(cc * 1.3 + 5.0))), b.sgr) * mix(1.0, 0.35, tv);
    float wPix = camDc * gPixAng * 0.9;
    // (no scattered single tillers where the sward is thin: they read as sticks and streaks)
    float rK = gcRnd();
    // (i15: up to 1.33 GC_KMAX tillers where the sward is densest - the shores - for the narrower blades)
    int K = dens < 0.12 ? 0 : int(float(GC_KMAX) * min(dens, 1.33) + rK);
    if (uDebug == 15) K = GC_KMAX;
    if (uDebug == 17 && b.sgr > 0.5) K = GC_KMAX;
    bool hit = false;
    if (uDebug == 16 || uDebug == 20) {          // look-dev: density (R), grassy shore (G), habitat veg (B) on a plane at the ground; 20: K, floor weight, floor tussock field
        float sg = abs(d.y) > 1e-5 ? (yC - o.y) / d.y : -1.0;
        vec2 hp = o.xz + d.xz * sg;
        if (sg > 0.0 && sg < sBest && all(greaterThanEqual(hp, c0)) && all(lessThan(hp, c0 + GC_C))) {
            sBest = sg; gcInfo = uDebug == 16 ? vec4(99.0, float(K) / float(GC_KMAX), b.sgr, b.veg) : vec4(99.0, smoothstep(1.7, 2.4, b.tdist) * (1.0 - smoothstep(3.5, 6.0, b.tdist)), b.wood, b.shore); return true;
        }
        return false;
    }
    // grassy shores: the terrain there is drawn as cobbles, so the sward's own base (thatch, soil,
    // moss) is a thin mat 1-2.5 cm above the ground in the cell
    if (b.sgr > 0.05 && abs(d.y) > 1e-4) {
        // (thinnest at the water's edge, sinks into the ground at the far end of the cover)
        float lift = mix(0.01, 0.025, smoothstep(0.05, 0.3, b.above));
        float y0p = yC - 0.5 * GC_C * (b.g.x + b.g.y) + lift;          // plane height at c0
        float sg = (y0p - o.y + dot(b.g, o.xz - c0)) / (d.y - dot(b.g, d.xz));
        // where the ray enters the cell: if it is already below the plane there, it hit the side of
        // this cell's mat (closes the cracks between the per-cell planes)
        vec2 iv = 1.0 / vec2(abs(d.x) > 1e-7 ? d.x : 1e-7, abs(d.z) > 1e-7 ? d.z : 1e-7);
        vec2 ta = (c0 - o.xz) * iv, tb = (c0 + GC_C - o.xz) * iv;
        float sEn = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), 0.0);
        vec2 he = o.xz + d.xz * sEn;
        if (o.y + d.y * sEn < y0p + dot(b.g, he - c0) && sg < sEn) sg = sEn;
        // (the marcher can start a segment a few mm past the previous one's end: a ray that is
        // already inside the thin mat there hits it at s = 0 - no cracks along segment boundaries)
        sg = sg > -0.05 ? max(sg, 0.0) : sg;
        vec2 hp = o.xz + d.xz * sg;
        if (sg >= 0.0 && sg < sBest && all(greaterThanEqual(hp, c0 - 1e-4)) && all(lessThan(hp, c0 + GC_C + 1e-4))) {
            float mn = vnoise(hp * 3.1 + 2.0) * 0.6 + vnoise(hp * 9.0 + 5.0) * 0.4;
            float cover = mix(0.35, 1.0, sgGap);
            // per-pixel edge: an irregular line 3-15 cm above the water (5-15 cm wiggles, no cell steps)
            float aboveP = o.y + d.y * sg - waterLevel(hp);
            bool on = b.sgr * rm > 0.3 + 0.2 * mn && aboveP > 0.03 + 0.12 * (vnoise(hp * 9.0 + 1.0) * 0.6 + vnoise(hp * 21.0 + 3.0) * 0.4);
            // far end of the cover: the mat breaks up into patches (the terrain beyond draws cobbles there)
            float camDp = length(vec3(hp.x, o.y + d.y * sg, hp.y) - uCamPos);
            on = on && vnoise(hp * 1.3 + 7.0) * 0.75 + 0.25 * mn < min(gcKeep(camDp), 1.0) * gcShrink(camDp) * 1.25;
            if (on && aboveP < 0.45) {
                // the terrain's larger cobbles (the same Voronoi as pebbles()) poke through near the water
                vec4 va = voronoiV(hp * mix(2.6, 4.2, vnoise(hp * 0.35 + 3.0)));
                float ra = 0.40 + 0.10 * fract(va.w * 7.3);
                on = !(va.z < ra - 0.06 && aboveP < 0.10 + 0.35 * fract(va.w * 3.7));
            }
            if (on) {
                sBest = sg; hit = true;
                gcInfo = vec4(GK_MAT, 0.0, mn, cover);
                gcN = normalize(vec3(-b.g.x, 1.0, -b.g.y));
                gcTopH = 0.05; gcEnv = vec2(0.0);
            }
        }
    }
    // (veg3) forest trail: the terrain paints its dirt 0.9-2.4 m from the centre line; the verge's
    // floor (litter, carpet, grass, as groundVegAlbedo draws it) is laid over the outer part as a mat
    // 1.5 cm up, so the trail reads ~1-1.5 m wide between the plants. Its inner edge moves out
    // to the terrain's dirt edge at the far end of the cover (no step where the cover ends).
    if (gcTrailSeg && b.tdist < 2.9 && b.tdist > 0.3 && abs(d.y) > 1e-4) {
        float y0p = yC - 0.5 * GC_C * (b.g.x + b.g.y) + 0.015;
        float sg = (y0p - o.y + dot(b.g, o.xz - c0)) / (d.y - dot(b.g, d.xz));
        vec2 iv = 1.0 / vec2(abs(d.x) > 1e-7 ? d.x : 1e-7, abs(d.z) > 1e-7 ? d.z : 1e-7);
        vec2 ta = (c0 - o.xz) * iv, tb = (c0 + GC_C - o.xz) * iv;
        float sEn = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), 0.0);
        vec2 he = o.xz + d.xz * sEn;
        if (o.y + d.y * sEn < y0p + dot(b.g, he - c0) && sg < sEn) sg = sEn;
        sg = sg > -0.05 ? max(sg, 0.0) : sg;
        vec2 hp = o.xz + d.xz * sg;
        if (sg >= 0.0 && sg < sBest && all(greaterThanEqual(hp, c0 - 1e-4)) && all(lessThan(hp, c0 + GC_C + 1e-4))) {
            vec2 tdir;
            gcTrailW(hp, tdir);
            float dP = gcTrD;
            float camDp = length(vec3(hp.x, o.y + d.y * sg, hp.y) - uCamPos);
            float ein = mix(0.45, 0.9, smoothstep(GC_FADE0, GC_FADE1 - 1.0, camDp))     // (veg4 G4: was 1.05 - 2.3)
                      + 0.22 * (vnoise(hp * 2.3 + 4.0) - 0.5) + 0.08 * (vnoise(hp * 9.0 + 1.0) - 0.5);
            if (dP > ein && b.veg * smoothstep(0.75, 1.0, eR) > 0.1) {
                sBest = sg; hit = true;
                gcInfo = vec4(GK_MAT, 0.0, 5.0 + vnoise(hp * 3.1 + 2.0), 1.0);
                gcN = normalize(vec3(-b.g.x, 1.0, -b.g.y));
                gcTopH = 0.05; gcEnv = vec2(1.0, 0.0);
            }
        }
    }
    // within ~0.3 m of the water's edge the blades hang over it; on the verge they lean over the trail
    float wlean = b.sgr * (1.0 - smoothstep(0.15, 0.35, b.above / max(length(b.g), 0.05))) * step(1e-4, dot(b.g, b.g));
    vec2 down = -b.g / max(length(b.g), 1e-4);
    // grassy shore where the terrain itself draws rock / cobbles and the shore mat is patchy: blades
    // only where the mat is drawn (same test as the mat, at the root)
    bool matChk = b.sgr > 0.05 && b.sgr * rm < 0.52 && b.vegT < 0.5;
    for (int i = 0; i < K; i++) {
        float r0 = gcRnd(), r1 = gcRnd(), r2 = gcRnd(), r3 = gcRnd(), r4 = gcRnd(), r5 = gcRnd(), r6 = gcRnd(), r7 = gcRnd();
        if (r7 >= keep) continue;                   // distance LOD: thinned by count
        // tiller: fan of 3-5 blades
        float Wt = mix(0.016, 0.038, r3);           // (i14: blades 2-10 mm, was 3-14)
        float nb = 3.0 + floor(r6 * 3.0);
        float fr = 0.6 + 0.25 * r1;
        // LOD: merge blades narrower than a pixel (coverage kept); thin whole tillers when even
        // the merged fan is sub-pixel
        float cov = fr * Wt;
        // (blades kept >= ~1.4 px: thinner slanted lines alias into dashed ladders at 2 spp)
        float wB = wPix * 1.6;
        if (cov < wB) {
            if (r4 * wB > cov) continue;
            Wt = wB / fr; nb = 1.0;
        } else nb = min(nb, max(1.0, floor(cov / wB)));
        float Wb = max(0.004, min(Wt, 1.2 * wB / fr));      // root width: >= a pixel (no dashed blades)
        // leaf blades arch over (horizontal reach 0.3-1 H), ~15 % stay upright stems
        float ang = r5 * TAU;
        vec2 a = vec2(cos(ang), sin(ang));
        float lf = r4 < 0.15 ? 0.06 : mix(0.3, 1.0, (r4 - 0.15) / 0.85);
        if (wlean > 0.05) { a = normalize(mix(a, down, 0.8 * wlean) + down * 1e-3); lf = mix(lf, 0.9, wlean); }
        if (tv > 0.05) { a = normalize(mix(a, b.tdir, tv) + b.tdir * 1e-3); lf = mix(lf, 0.9, tv); }
        // (i12) the root sits back from the lean, on the far side of the cell: the cell-bound reach
        // no longer clips long leaning blades into upright reeds (they arch and flop).
        // The shrunk square + offset stays inside [0.03, 0.97] GC_C for any lean direction.
        float kS = 0.5 * lf;
        vec2 lr = (0.5 + (vec2(r0, r1) - 0.5) * 0.94 * (1.0 - kS)) * GC_C - a * (kS * 0.47 * GC_C);
        // grass grows right up to the stones and boulders (roots outside 0.9 of their footprints)
        if (nearR > 0.0 && gcRockE(c0 + lr) < 1.0) continue;
        if (matChk) {
            vec2 xr = c0 + lr;
            if (b.sgr * rm <= 0.3 + 0.2 * (vnoise(xr * 3.1 + 2.0) * 0.6 + vnoise(xr * 9.0 + 5.0) * 0.4)) continue;
        }
        // meadows / slopes 8-42 cm; grassy shores 10-42 cm (patchy, not a hay field)
        // (veg4 G7: meadow heights 6-61 cm, mostly low with a few tall ones - was a leek field of equal spikes)
        float H = min(mix(0.06 + 0.55 * pow(r2, 1.6), 0.10 + 0.32 * pow(r2, 1.3), b.sgr) * tall, 0.58) * gcGrow(keep, r7) * shr;   // (<= GC_GTOP)
        if (H < 0.02) continue;
        vec2 lim = vec2(a.x > 0.0 ? GC_C - lr.x : lr.x, a.y > 0.0 ? GC_C - lr.y : lr.y) - 0.5 * Wt;
        float Lmax = max(min(lim.x / max(abs(a.x), 1e-3), lim.y / max(abs(a.y), 1e-3)), 0.0);
        float L = min(H * lf, Lmax);
        vec3 R = vec3(c0.x + lr.x, yC + dot(b.g, lr - 0.5 * GC_C) - 0.035, c0.y + lr.y);
        if (gcMiss(o, d, R, H, L + 0.5 * Wt + 0.005)) continue;
        float s, v, yh; vec3 n;
        // (veg3) long leaning leaves arch over and droop instead of standing as stiff spikes
        bool arch = L > 0.45 * H && fract(r6 * 5.3 + r2) < 0.9;     // (veg4 G7: 0.7 -> 0.9)
        bool got = arch ? gcArchFan(o, d, R, a, H, L * 0.58, L, Wb, Wt, nb, fr, r2 * 7.0, sBest, s, v, yh, n)
                        : gcFan(o, d, R, a, H, L, Wb, Wt, nb, fr, r2 * 7.0, sBest, s, v, n);
        if (got) {
            sBest = s; hit = true;
            gcUV = vec2(v, 0.0);                    // (along the blade: the pale tip)
            if (arch) v = max(v, yh / H);
            // dry blades: 7 % in the meadows, a few more on the shores and the forest floor
            // (veg4 G6: ~30 % straw blades on the grassy shores / boulder belt, #686538 olive-straw mix)
            float dry = step(0.93 - 0.02 * b.shore - 0.23 * max(b.sgr, 0.8 * b.sgrB) - 0.05 * ff, fract(r0 * 13.7 + r1 * 7.3));
            gcInfo = vec4(GK_BLADE, (o.y + d.y * s) - yC, fract(r2 * 17.3 + r3 * 5.1) + dry * 2.0, v);
            gcN = n;
            gcTopH = max(0.33 * tall * shr, 0.04);
            gcEnv = vec2(ff, b.meadow); gcSrc = 1.0;
            gcDensH = saturate(dens * 1.8) * mix(1.0, min(keep, 1.0), 0.7);   // (thinned far end: sparse, lit bases; i20: x1.8 - the meadow's forbs fill the canopy)
        }
    }
    return hit;
}

// ---- tussock: a fountain of grass tillers around a centre (lake shore, between boulders, meadow)
#define GC_C3 1.2            // tussock cell size (m)   (veg4 G5: 1.0 -> 1.2, room for the 1 m shore tussocks)
#define GC_TR 0.6            // max tussock footprint radius (m)   (veg4 G5: 0.4 -> 0.6)
#define GC_TR0 0.4           // footprint radius of a size-1 tussock (m)
bool gcTussock(vec3 o, vec3 d, vec3 c, vec4 prm, float sMax, inout float sBest) {
    // bounding cylinder
    float Rf = GC_TR0 * (prm.x - 0.03) / 0.04;     // footprint radius (sz * GC_TR0)
    vec2 q = o.xz - c.xz;
    float A = dot(d.xz, d.xz), B = dot(q, d.xz), C = dot(q, q) - Rf * Rf;
    float D = B * B - A * C;
    if (D < 0.0 || A < 1e-8) return false;
    float sq = sqrt(D);
    float s0 = (-B - sq) / A, s1 = (-B + sq) / A;
    // clip with the height slab
    float ylo = c.y - 0.08, yhi = c.y + min(prm.y * 1.6 + 0.07, 0.74);     // (culms stand above the leaves)
    if (abs(d.y) > 1e-6) {
        float ta = (ylo - o.y) / d.y, tb = (yhi - o.y) / d.y;
        s0 = max(s0, min(ta, tb)); s1 = min(s1, max(ta, tb));
    } else if (o.y < ylo || o.y > yhi) return false;
    s0 = max(s0, 0.0); s1 = min(s1, min(sMax, sBest));
    if (s0 >= s1) return false;
    float camD = length(c - uCamPos);
    float wPix = camD * gPixAng * 0.9;
    gcS = hash2u(ivec2(floor(c.xz * 97.0)) + ivec2(77, 13));
    int N = int(prm.z);
    bool hit = false;
    for (int i = 0; i < N; i++) {
        float r0 = gcRnd(), r1 = gcRnd(), r2 = gcRnd(), r3 = gcRnd(), r4 = gcRnd(), r5 = gcRnd();
        float u = sqrt(r0);
        float th = r1 * TAU;
        vec2 dir = vec2(cos(th), sin(th));
        vec2 lr = dir * u * prm.x;
        float th2 = th + (r5 - 0.5) * 0.9;
        vec2 a = vec2(cos(th2), sin(th2));
        float H = prm.y * (0.55 + 0.45 * r2) * (1.0 - 0.3 * u);
        float Wt = mix(0.022, 0.046, r3);
        float nb = 3.0 + floor(r4 * 4.0);
        float fr = 0.55 + 0.25 * r5;
        float cov = fr * Wt;
        // (blades kept >= ~1.4 px: thinner slanted lines alias into dashed ladders at 2 spp)
        float wB = wPix * 1.6;
        if (cov < wB) {
            if (r4 * wB > cov) continue;
            Wt = wB / fr; nb = 1.0;
        } else nb = min(nb, max(1.0, floor(cov / wB)));
        float Wb = max(0.004, min(Wt, 1.2 * wB / fr));
        float L = min(H * (0.45 + 1.1 * u) * (0.8 + 0.4 * r3), Rf - prm.x - 0.5 * Wt);
        vec3 R = vec3(c.x + lr.x, c.y - 0.04, c.z + lr.y);
        if (gcMiss(o, d, R, H, L + 0.5 * Wt + 0.005)) continue;
        float s, v, yh; vec3 n;
        // (veg3) the outer tillers arch over and droop (the fountain of a tussock); the
        // inner ones stand up
        bool arch = fract(r2 * 7.31 + r5 * 3.1) < smoothstep(0.25, 0.7, u) && L > 0.4 * H;
        bool got;
        if (arch) {
            float xp = L * mix(0.52, 0.7, r3);
            got = gcArchFan(o, d, R, a, H, xp, L, Wb, Wt, nb, fr, r2 * 5.0 + prm.w, min(sBest, s1 + 0.05), s, v, yh, n);
        } else {
            got = gcFan(o, d, R, a, H, L, Wb, Wt, nb, fr, r2 * 5.0 + prm.w, min(sBest, s1 + 0.05), s, v, n);
            yh = v * H;
        }
        if (got) {
            sBest = s; hit = true;
            float dry = step(0.92 - 0.20 * gcTusSh, fract(r0 * 13.7 + r2 * 7.3));
            gcInfo = vec4(GK_BLADE, yh * 0.9, fract(r2 * 17.3 + r3 * 5.1) * 0.7 + 0.15 + dry * 2.0, arch ? max(v, yh / H) : v);
            gcUV = vec2(v, 0.0);
            gcN = n;
            gcTopH = prm.y * 0.55;
            gcEnv = gcTusEnv; gcSrc = 2.0; gcDensH = 1.0;
        }
    }
    // flowering culms: 0-3 thin straight stems standing above the leaves with a narrow panicle
    // (straw / purplish brown), the grass's summer look
    {
        uint key = hash2u(ivec2(floor(c.xz * 97.0)) + ivec2(311, 59));
        int nc = int(fract(float(key & 1023u) / 1024.0) * 3.4 * prm.w + 0.3);
        for (int i = 0; i < 3; i++) {
            if (i >= nc) break;
            gcS = hashu(key + uint(i) * 0x9E3779B9u);
            float q0 = gcRnd(), q1 = gcRnd(), q2 = gcRnd(), q3 = gcRnd();
            vec2 lr = (vec2(q0, q1) - 0.5) * prm.x;
            float Hc = min(prm.y * mix(1.15, 1.55, q2) + 0.05, 0.72);
            vec2 lean = (vec2(q3, q2) - 0.5) * 0.12;
            vec3 R = vec3(c.x + lr.x, c.y - 0.04, c.z + lr.y);
            float camD2 = length(R - uCamPos);
            float wP = camD2 * gPixAng * 0.9;
            float s; vec3 n;
            if (gcStem(o, d, R, lean, Hc, max(0.0015, wP * 0.8), min(sBest, s1 + 0.05), s, n)) {
                sBest = s; hit = true;
                gcInfo = vec4(GK_STEM, (o.y + d.y * s) - R.y, q3, 4.0);
                gcN = n; gcTopH = prm.y * 0.55;
            }
            vec3 hr = max(vec3(0.007, 0.035, 0.007) * (0.8 + 0.4 * q0), vec3(wP * 0.7));
            vec3 hcc = R + vec3(lean.x, Hc - hr.y * 0.7, lean.y);
            if (gcEllipsoid(o, d, hcc, hr, vec2(1.0, 0.0), min(sBest, s1 + 0.05), s, n)) {
                sBest = s; hit = true;
                gcInfo = vec4(GK_HEAD, Hc, q1, 4.0 + step(0.6, q2));
                gcN = n; gcTopH = prm.y * 0.55;
                gcObj = vec4(hcc, 0.0); gcObjR = vec4(hr, 4.0 + step(0.6, q2)); gcFadeK = 1.0;
            }
        }
    }
    return hit;
}

// ---- 1 m cells: a tussock and / or tall forbs (both may reach GC_TR beyond their cell) --------
// one habitat evaluation (at the cell centre) serves both; cheap bounds skip most empty cells
// before it
void gcCell1(ivec2 cell, vec3 o, vec3 d, float sEnd, inout float sBest) {
    vec2 cc = (vec2(cell) + 0.5) * GC_C3;
    gcS = hash2u(cell + ivec2(6007, 1213));
    float rN = gcRnd();
    uint keyF = gcS;
    gcS = hash2u(cell + ivec2(5113, 2027));
    float r0 = gcRnd(), r1 = gcRnd(), r2 = gcRnd(), r3 = gcRnd(), r4 = gcRnd();
    // tall forbs need 0.25 + 2.8 meadow + rN >= 1; a tussock needs r0 <= 0.9, and r0 < 0.52 away
    // from the shore
    vec4 f = forestAt(cc);
    float wl = waterLevel(cc);
    bool forbs = 1.44 * (0.25 + 3.2 * f.a) + rN >= 1.0;
    bool tus = r0 <= 0.9 && (r0 < 0.68 || terrainH(cc, false) - wl < 3.2);
    if (!forbs && !tus) return;
    vec2 g;
    float yc = gcGroundG(cc, g);
    float camD = length(vec3(cc.x, yc, cc.y) - uCamPos);
    if (camD > GC_FADE1 + 0.6) return;
    GBio b = gcBiome(cc, yc, g, wl);
    if (b.veg < 0.02) return;
    float canopy = gcCanopy(cc);
    float ff = gcFloor(b, canopy);
    float keep = gcKeep(camD), shr = gcShrink(camD);
    if (forbs && wl < yc - 0.25) {
        float open = 1.0 - ff;
        float nf = b.veg * open * (1.0 - smoothstep(0.25, 0.6, b.shore)) * (1.0 - b.trail)
                 * (0.25 + 3.2 * b.meadow * smoothstep(0.25, 0.65, 1.0 - gcPatch(cc) + 0.3));
        int N = min(int(nf * 1.44 + rN), 6);     // (veg4: x1.44 / cap 4 -> 6 for the 1.2 m cells, same density per m2)
        float wPix = camD * gPixAng * 0.9;
        float tall = 0.8 + 0.4 * vnoise(cc * 0.55 + 13.0);
        bool hit = false;
        for (int k = 0; k < N; k++) {
            gcS = hashu(keyF + uint(k) * 0x9E3779B9u);
            vec2 pos = cc + (vec2(gcRnd(), gcRnd()) - 0.5) * GC_C3;
            float rk = gcRnd();
            if (rk >= keep) continue;
            if (gcMiss(o, d, vec3(pos.x, yc + dot(g, pos - cc) - 0.2, pos.y), 1.05, GC_TR + 0.02)) continue;
            float nearR;
            if (gcRockMask(pos, nearR) < 0.5) continue;
            if (gcForbAt(o, d, pos, terrainHD(pos, 1.0), 1.0, GC_TR, gcGrow(keep, rk) * shr, wPix, tall, sBest)) hit = true;
        }
        if (hit) gcEnv = vec2(ff, b.meadow);
    }
    if (tus) {
        // size first: small tussocks get more positional jitter (no visible grid)
        // (veg4 G5: the grassy shores carry big 1-1.1 m fountains)
        float sz = 0.55 + 0.45 * r3 + 0.4 * smoothstep(0.3, 0.8, b.sgr) * r3 * r3;
        vec2 pos = cc + (vec2(r1, r2) - 0.5) * (GC_C3 - 2.0 * GC_TR0 * sz);
        float nearR;
        float rm = gcRockMask(pos, nearR);
        float pt = sqrt(mix(b.vegT, b.veg, min(keep, 1.0) * shr)) * rm * (1.0 - b.trail) * (b.shore * mix(0.55 + 0.45 * nearR, 0.6 + 0.3 * nearR, b.sgr / max(b.shore, 1e-3)) * smoothstep(mix(0.3, 0.12, b.sgr), mix(0.7, 0.4, b.sgr), b.above) + (1.0 - b.shore) * (0.06 + 0.06 * b.meadow + 0.4 * nearR) * 1.3 * (1.0 - b.forest));   // (veg4: x1.3 for the 1.2 m cells)
        float rk = fract(r4 * 7.13 + r1 * 3.7);
        if (r0 <= pt && rk < keep) {
            gcTusEnv = vec2(ff, b.meadow);
            gcTusSh = max(b.sgr, 0.8 * b.sgrB);
            vec3 c = vec3(pos.x, terrainHD(pos, 1.0), pos.y);
            vec4 prm = vec4(0.03 + 0.04 * sz, (0.2 + 0.36 * sz) * (0.85 + 0.3 * r4) * mix(0.85, 1.1, b.shore) * mix(1.0, 1.35, b.sgr) * gcGrow(keep, rk) * shr,
                            floor(36.0 + 60.0 * sz + 30.0 * b.sgr), fract(r1 * 7.3 + r4));
            prm.y = min(prm.y, 0.72);            // (inside GC_TOP: the march skips cover above it)
            gcTussock(o, d, c, prm, sEnd, sBest);
        }
    }
}

// ---- one DDA run over the ray segment [o, o + d GC_SEG] --------------------------------------
float gcSegLen = GC_SEG;     // (shadeCover's replay walks a short segment only)
void gcSegment(vec3 o, float t0, vec3 d) {
    float sEnd = gcSegLen;
    gcDl = length(d.xz);
    gcDn = d.xz / max(gcDl, 1e-6);
    float sBest = 1e9;
    vec2 midP = o.xz + d.xz * (0.5 * sEnd);
    // bare segment (rock, scree, cobble beach, water): nothing can grow along it -> skip the gathers
    // and all the cell walks (every generator needs b.veg > 0). 5 probes 0.4 m apart, each with a
    // margin (gcVegProbe > 0 a little before b.veg), the first vegetated one ends the test.
    {
        bool bare = true;
        for (int k = 0; k < 5; k++) {
            if (gcVegProbe(o.xz + d.xz * (sEnd * 0.25 * float(k))) > 0.0) { bare = false; break; }
        }
        if (bare) { gcSegHit = false; gcSafeT = t0 + sEnd; return; }
    }
    gcGatherRocks(midP);
    gcSegA = o.xz; gcSegB = o.xz + d.xz * sEnd;
    gcGatherTrees(gcSegA, gcSegB);
    gcWoodA = gcWoodAt5(gcSegA); gcWoodB = gcWoodAt5(gcSegB);
    // (generous: trail verges up to ~3.1 m from the centre line, coarse cells up to ~2.6 m off the ray)
    gcTrailSeg = gcTrailDist(midP) < 6.5;
    // coarse objects (shrubs, stones) whose reach (cell grown by GC_CO) the segment passes over
    vec2 e = o.xz + d.xz * sEnd;
    vec2 ivS = 1.0 / max(abs(d.xz), vec2(1e-7)) * vec2(d.x >= 0.0 ? 1.0 : -1.0, d.z >= 0.0 ? 1.0 : -1.0);
    ivec2 ca = ivec2(floor((min(o.xz, e) - GC_CO) / GC_C2)), cb = ivec2(floor((max(o.xz, e) + GC_CO) / GC_C2));
    for (int j = ca.y; j <= cb.y; j++)
    for (int i = ca.x; i <= cb.x; i++) {
        vec2 bmin = vec2(i, j) * GC_C2 - GC_CO, bmax = bmin + GC_C2 + 2.0 * GC_CO;
        vec2 t1 = (bmin - o.xz) * ivS, t2 = (bmax - o.xz) * ivS;
        if (max(max(min(t1.x, t2.x), min(t1.y, t2.y)), 0.0) > min(min(max(t1.x, t2.x), max(t1.y, t2.y)), sEnd)) continue;
        vec3 cen, rad; float ang, seed;
        int ty = gcCoarseObj(ivec2(i, j), cen, rad, ang, seed);
        if (ty == 0) continue;
        float s; vec3 n;
        if (ty == 1) {
            if (gcShrub(o, d, cen, rad, seed, 1.0, min(sBest, sEnd), s)) {
                sBest = s;
                vec3 hp = o + d * s;
                gcInfo = vec4(GK_SHRUB, hp.y - (cen.y - rad.y * 0.42), seed, 0.0);
                gcObj = vec4(cen, seed); gcObjR = vec4(rad, 1.0);
                gcTopH = rad.y * 1.4;
            }
            // (veg4 G3a) leafy sprigs breaking the hull up close (not on the upright juniper / sapling columns)
            float camDs = length(cen - uCamPos);
            if (camDs < 10.0 && rad.y < rad.x * 1.2) gcShrubSprigs(o, d, cen, rad, seed, camDs, sBest);
        } else {
            // keep grass and forbs out of the stone (blades would poke through it)
            if (gcNRk < 8 && gcStoneFoot > 0.2) {
                gcRk[gcNRk] = vec4(cen.xz, rad.xz * gcStoneFoot);
                gcRkA[gcNRk] = vec2(cos(ang), sin(ang));
                gcRkY[gcNRk] = vec2(cen.y, rad.y);
                gcNRk++;
            }
            if (gcStone(o, d, cen, rad, vec2(cos(ang), sin(ang)), seed, min(sBest, sEnd), s, n)) {
                sBest = s;
                gcInfo = vec4(GK_STONE, (o.y + d.y * s) - cen.y + rad.y, seed, 0.0);
                gcN = n; gcObj = vec4(cen, seed); gcObjR = vec4(rad, ang);
                gcTopH = 0.12;
            }
        }
    }
    // tussocks in the 1 m cells the segment crosses
    {
        vec2 lo = min(o.xz, e) - GC_TR, hi = max(o.xz, e) + GC_TR;
        ivec2 ta = ivec2(floor(lo / GC_C3)), tb = ivec2(floor(hi / GC_C3));
        vec2 iv = ivS;
        for (int j = ta.y; j <= tb.y; j++)
        for (int i = ta.x; i <= tb.x; i++) {
            // skip cells whose footprint (cell grown by the tussock radius) the ray does not pass over
            vec2 bmin = vec2(i, j) * GC_C3 - GC_TR, bmax = bmin + GC_C3 + 2.0 * GC_TR;
            vec2 t1 = (bmin - o.xz) * iv, t2 = (bmax - o.xz) * iv;
            float tn = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), 0.0);
            float tf = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), sEnd);
            if (tn > tf || tn > sBest) continue;
            gcCell1(ivec2(i, j), o, d, sEnd, sBest);
        }
    }
    // medium grid DDA: forb rosettes, flower clumps, berry bushes
    vec2 dxz = d.xz;
    vec2 sg = vec2(dxz.x >= 0.0 ? 1.0 : -1.0, dxz.y >= 0.0 ? 1.0 : -1.0);
    vec2 inv = 1.0 / max(abs(dxz), vec2(1e-6));
    float sDoneM = sEnd;
    sDoneM = 0.0;
    {
        ivec2 cell = ivec2(floor(o.xz / GC_CM));
        vec2 tMax = abs((vec2(cell) + max(sg, 0.0)) * GC_CM - o.xz) * inv;
        vec2 tDel = GC_CM * inv;
        float sIn = 0.0;
        for (int i = 0; i < GC_MMAX; i++) {
            float sOut = min(min(tMax.x, tMax.y), sEnd);
            vec2 cc = (vec2(cell) + 0.5) * GC_CM;
            float yC = terrainH(cc, true);
            float y0 = o.y + d.y * sIn, y1 = o.y + d.y * sOut;
            if (max(y0, y1) - yC < -0.45) { sDoneM = sIn; break; }
            if (min(y0, y1) - yC < GC_TOP + 0.15) gcMid(cell, o, d, sBest);
            sDoneM = sOut;
            if (sBest <= sOut || sOut >= sEnd) break;
            sIn = sOut;
            if (tMax.x < tMax.y) { tMax.x += tDel.x; cell.x += int(sg.x); }
            else { tMax.y += tDel.y; cell.y += int(sg.y); }
        }
    }
    // fine grid DDA
    ivec2 cell = ivec2(floor(o.xz / GC_C));
    vec2 tMax = abs((vec2(cell) + max(sg, 0.0)) * GC_C - o.xz) * inv;
    vec2 tDel = GC_C * inv;
    float sIn = 0.0;
    float sDone = 0.0;
    for (int i = 0; i < GC_MAXCELLS; i++) {
        float sOut = min(min(tMax.x, tMax.y), sEnd);
        vec2 cc = (vec2(cell) + 0.5) * GC_C;
        vec2 gC;
        float yC = gcGroundG(cc, gC);
        float y0 = o.y + d.y * sIn, y1 = o.y + d.y * sOut;
        if (max(y0, y1) - yC < -0.3) { sDone = sIn; break; }       // below ground: terrain wins
        if (min(y0, y1) - yC < GC_GTOP + 0.12) gcFine(cell, o, d, yC, gC, sBest);
        sDone = sOut;
        if (sBest <= sOut || sOut >= sEnd) break;
        sIn = sOut;
        if (tMax.x < tMax.y) { tMax.x += tDel.x; cell.x += int(sg.x); }
        else { tMax.y += tDel.y; cell.y += int(sg.y); }
    }
    sDone = min(sDone, sDoneM);
    gcSegHit = sBest <= sDone;
    gcSafeT = t0 + (gcSegHit ? sBest : max(sDone, 0.0));
}

float coverSDF(vec3 p, float t, out vec4 info) {
    info = vec4(0.0);
    if (t < gcLastT) { gcRayOk = false; gcAnch = false; gcSafeT = -1.0; gcSegHit = false; }
    gcLastT = t;
    float camD = length(p - uCamPos);
    if (camD > GC_FADE1 + 0.3) return camD - GC_FADE1;
    float epsC = max(0.002, t * gPixAng * 3.0);
    // the marcher steps exactly onto the cached hit; float rounding can land it a hair beyond, where
    // a fresh segment would start behind the thin surface and miss it (dithered holes in leaves/blades)
    if (gcSegHit && t < gcSafeT + 1e-3 + t * 1e-5) { info = gcInfo; return max(gcSafeT - t, 0.0); }
    if (t < gcSafeT) {
        float r = gcSafeT - t;
        if (gcSegHit) { info = gcInfo; return r; }
        if (r > epsC) return r;
    }
    float wl = waterLevel(p.xz);
    float thp = terrainHD(p.xz, detailWeight(t));
    float yr = p.y - max(thp, wl);
    if (yr > GC_TOP + 0.08) return (yr - GC_TOP) * 0.6;
    if (uDebug == 14) {          // look-dev: habitat on the ground (veg, shore, grassy shore)
        if (yr > 0.03) return yr * 0.7;
        vec2 g14;
        float y14 = gcGroundG(p.xz, g14);
        GBio b = gcBiome(p.xz, y14, g14, wl);
        float nearR;
        gcGatherRocks(p.xz);
        float rm = gcRockMask(p.xz, nearR);
        float sgGap = smoothstep(0.35, 0.65, vnoise(p.xz * 0.7 + 17.0) + 0.35 * nearR);
        float shoreK = mix(1.0 - smoothstep(0.25, 0.6, b.shore), 2.6 * smoothstep(0.04, 0.22, b.above) * mix(0.1, 1.0, sgGap), b.sgr);
        info = vec4(99.0, b.veg * rm, shoreK, b.above);
        return 0.0;
    }
    // just above open water: nothing to find (over the shore the low grass by the waterline is)
    if (p.y < wl + 0.2 && thp < wl + 0.03) return max(wl + 0.2 - p.y, epsC);
    // the ray being marched
    if (!gcRayOk) {
        vec3 ro = uCamPos;
        if (uAperture > 0.0) {
            vec2 lr = hash22i(ivec2(gl_FragCoord.xy) * 3 + ivec2(uSample * 31, uSample * 17));
            float ang = lr.x * TAU, rad = sqrt(lr.y) * uAperture;
            ro += uCamRot * vec3(cos(ang) * rad, sin(ang) * rad, 0.0);
        }
        vec3 v = p - ro;
        float lv = length(v);
        if (abs(lv - t) < 2e-3 + 2e-5 * t) { gcRd = v / lv; gcRayOk = true; }
        else if (!gcAnch || t - gcAT < 0.03) {
            if (!gcAnch) { gcAnch = true; gcAP = p; gcAT = t; }
            return max(epsC, 0.02);
        } else { gcRd = normalize(p - gcAP); gcRayOk = true; }
    }
    gcSegment(p, t, gcRd);
    float r = gcSafeT - t;
    if (gcSegHit) { info = gcInfo; return r; }
    return max(r, epsC);
}

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
vec3 gcBladeCol(float r, float dry) {
    // (i21: blue halved - target meadow #7b892b is linear B/G 0.10; the render read B/G ~0.2 (sRGB 0.46))
    // (i22: R -9 %, B +25 %: i21 read too yellow (R/G lin 0.91 vs target 0.79))
    vec3 c = mix(vec3(0.104, 0.160, 0.016), vec3(0.152, 0.198, 0.021), smoothstep(0.0, 0.6, r));
    c = mix(c, vec3(0.194, 0.232, 0.029), smoothstep(0.6, 1.0, r) * 0.8);
    c = mix(c, vec3(0.180, 0.166, 0.045), smoothstep(0.35, 0.9, dry) * 0.6);
    return mix(c, vec3(0.220, 0.185, 0.052), smoothstep(0.75, 1.1, dry) * 0.4);
}
vec3 gcBladeMean(float dry) {
    vec3 c = mix(vec3(0.150, 0.195, 0.021), vec3(0.180, 0.166, 0.045), smoothstep(0.35, 0.9, dry) * 0.6);
    return mix(c, vec3(0.220, 0.185, 0.052), smoothstep(0.75, 1.1, dry) * 0.4);
}
vec3 gcSpecTint = vec3(1.0);
vec3 gcLight(vec3 p, vec3 rd, vec3 n, vec3 alb, vec3 trans, float ao, float depth, float spec, float rough, float wrap) {
    // (i18) the shadow march compares against the smooth terrainH: cover rooted in a micro-relief hollow
    // (up to ~0.3 m below it) was in "shadow" - black stubble on scree (r7), dark grass bases. Tested from
    // >= 0.3 m above that surface, like the terrain's own p + 0.3 n
    vec3 pS = vec3(p.x, max(p.y, terrainH(p.xz, true) + 0.3), p.z);
    float sun = terrainShadow(pS, 40) * cloudShadow(p);
    if (uDebug == 32) return vec3(sun, saturate((dot(n, uSunDir) + wrap) / (1.0 + wrap)), ao);
    if (sun > 0.0) sun *= treeShadow(p + vec3(0.0, 0.03, 0.0));
    {
        // (veg4 R6) the boulders' cast shadow and contact occlusion on the cover
        float aoR;
        sun *= gcRockSun(p, aoR);
        ao *= aoR;
    }
    if (depth > 0.0 && sun > 0.0) {
        // light reaching into the grass canopy: mostly blocked, with sun flecks through gaps
        // (fleck noise at ~5 / 12 cm, faded to its mean when it would speckle at pixel scale)
        float ds = depth / max(uSunDir.y, 0.2);
        vec2 sp = p.xz + uSunDir.xz * ds;
        float pf = length(p - uCamPos) * gPixAng;
        float fl = mix(vnoise(sp * 8.0) * 0.6 + vnoise(sp * 19.0 + 7.0) * 0.4, 0.5, smoothstep(0.004, 0.012, pf));
        sun *= mix(exp(-ds * 5.0), 1.0, smoothstep(0.40, 0.72, fl) * exp(-ds * 1.5));
    }
    float ndl = dot(n, uSunDir);
    float diff = saturate((ndl + wrap) / (1.0 + wrap));
    vec3 col = alb * uSunCol * sun * diff;
    col += trans * uSunCol * sun * saturate(-ndl);
    // foliage: sky light that reached it through / off other leaves is greener and warmer
    col += alb * skyLight(normalize(n + vec3(0.0, 0.6, 0.0)), ao) * (wrap > 0.0 ? vec3(1.06, 1.06, 0.62) : vec3(1.0));
    if (spec > 0.0) {
        vec3 hv = normalize(uSunDir - rd);
        float a = rough * rough;
        float nh = max(dot(n, hv), 0.0);
        float dd = nh * nh * (a * a - 1.0) + 1.0;
        float D = a * a / (PI * dd * dd);
        float F = 0.04 + 0.96 * pow(1.0 - max(dot(hv, -rd), 0.0), 5.0);
        col += uSunCol * gcSpecTint * sun * max(ndl, 0.0) * D * F * spec * 0.25;
    }
    return col;
}

vec3 groundVegAlbedo(vec3 p, inout vec3 n, float t, vec4 f, float ao);     // (below)
vec3 shadeCover(vec3 p, vec3 rd, float t, vec4 info) {
    // (i13) split renderer: the shading program only gets the G-buffer's info, not the hit record the
    // trace left in its globals (normal, leaf frame / uv, object centre and radii, habitat, LOD fade).
    // Replay a short segment along the same camera ray from 5 cm before the hit to restore them (without
    // this every blade, leaf and bush was lit with the default normal (0, 1, 0): flat, cardboard-like).
    if (info.x < 98.0) {
        gcSegLen = 0.12;
        gcSegment(p - rd * 0.05, t - 0.05, rd);
        gcSegLen = GC_SEG;
        if (gcSegHit) info = gcInfo;
    }
    float kind = info.x, yr = info.y, seed = info.z, aux = info.w;
    if (uDebug == 11) {     // look-dev: colour by cover kind
        vec3 kc[9] = vec3[9](vec3(0.05, 0.35, 0.05), vec3(0.05, 0.10, 0.6), vec3(0.6, 0.6, 0.0), vec3(0.7, 0.0, 0.0), vec3(0.6, 0.0, 0.6), vec3(0.6),
                             vec3(0.0, 0.5, 0.5), vec3(0.2, 0.3, 0.9), vec3(1.0, 0.3, 0.0));
        return kc[clamp(int(kind + 0.5) - 1, 0, 8)] * (0.4 + 0.6 * saturate(abs(dot(gcN, uSunDir))));
    }
    if (kind > 98.0) return vec3(yr, seed, aux);
    if (uDebug == 13) return vec3(gcEnv.x, gcEnv.y, kind < 1.5 ? 0.0 : 1.0);   // look-dev: floor / meadow weight
    if (uDebug == 18) return kind > 1.5 ? vec3(0.3) : gcSrc < 0.5 ? vec3(0.9) : (gcSrc < 1.5 ? vec3(0.9, 0.1, 0.1) : vec3(0.1, 0.2, 0.9));
    vec3 n = gcN;
    float top = gcTopH;
    // crowns / stand of the ray segment that produced the hit (still in the globals)
    float stand = gcStandOf(p.xz);
    float canC = gcCanopy(p.xz);
    // sky hidden by the crowns overhead / the surrounding stand
    float ao0 = terrainAO(p.xz) * mix(1.0, 0.5, canC) * mix(1.0, 0.78, stand);
    vec4 f = forestAt(p.xz);
    float forest = saturate(f.r * 1.3);
    // grassland tone of the neighbourhood (same field as the ground albedo)
    vec3 alb, trans = vec3(0.0);
    float spec = 0.0, rough = 0.5, wrap = 0.3;
    float hfrac = saturate(yr / max(top, 0.03));
    float ao = ao0 * mix(0.4, 1.0, pow(hfrac, 0.7));
    float depth = max(top - yr, 0.0);

    if (kind > 9.5 && seed > 4.5) {                       // trail verge mat: the floor as the terrain draws it
        vec3 nV = n;
        vec4 fV = forestAt(p.xz);
        float aoT = terrainAO(p.xz);
        vec3 albV = groundVegAlbedo(p, nV, t, fV, aoT);
        float aoV = aoT * mix(1.0, 0.45, saturate(fV.r * 1.2)) * gcGroundSkyOcc;
        return gcLight(p, rd, nV, albV, vec3(0.0), aoV, 0.0, 0.02, 0.6, 0.0);
    }
    if (kind > 9.5) {                                     // shore sward base: thatch, soil, moss
        vec2 x = p.xz;
        float pf = max(t * gPixAng, 0.002);
        float f1 = vnoise(x * 23.0), f2 = vnoise(x * 61.0 + 3.0);
        float fw = 1.0 - smoothstep(0.003, 0.008, pf);
        // dense sward base seen between the blades: dark olive-green grass bases and moss,
        // some straw; no bare soil patches
        // (i17: lighter, warmer - dead blades, straw and litter between the tillers, was near black)
        vec3 thatch = mix(vec3(0.092, 0.086, 0.040), vec3(0.165, 0.146, 0.068), mix(0.5, f1, fw));
        vec3 moss = mix(vec3(0.045, 0.066, 0.016), vec3(0.088, 0.112, 0.028), mix(0.5, f2, fw));
        float mm = smoothstep(0.30, 0.65, vnoise(x * 1.7 + 9.0));
        alb = mix(thatch, moss, mm * 0.6);
        // thin sward: a little darker (shade of the surrounding tussocks)
        alb *= mix(1.0, 0.8, smoothstep(0.75, 1.1, seed / max(aux, 0.05)));
        alb *= mix(1.0, 0.75 + 0.5 * f2, fw);
        // lying dead blades: fine streaks along a direction per ~0.3 m patch
        float fwS = 1.0 - smoothstep(0.0015, 0.004, pf);
        if (fwS > 0.0) {
            ivec2 sc = ivec2(floor(x / 0.3));
            vec2 xl = x - (vec2(sc) + 0.5) * 0.3;
            float st = vnoise(rot2(hash12i(sc + 3) * TAU) * xl * vec2(110.0, 9.0) + hash12i(sc + 5) * 17.0);
            alb *= mix(1.0, 0.62 + 0.7 * st, fwS);
        }
        vec3 nb = normalize(n + vec3(f1 - 0.5, 0.0, f2 - 0.5) * 0.5 * fw);
        // (deep in the sward: less sky, sun mostly through gaps)
        return gcLight(p, rd, nb, alb * 0.9, vec3(0.0), ao0 * 0.68, 0.10, 0.05, 0.6, 0.1);
    }
    if (kind > 6.5) {
        float under = step(0.0, dot(n, rd)) * step(0.2, n.y);   // seeing a leaf from below
        if (dot(n, rd) > 0.0) n = -n;                     // thin two-sided blades face the viewer
        vec2 uv = gcUV;
        if (kind < 8.5) {
            vec3 tu = gcT;
            vec3 sv = cross(n, tu);
            float r = fract(seed);
            float hf = saturate(yr / max(top, 0.05));
            if (kind < 7.5) {                             // blueberry leaf: small, cupped, glossy
                vec3 nb = normalize(n + sv * (uv.y * 0.45) - tu * (uv.x * 0.2));
                alb = mix(vec3(0.030, 0.068, 0.010), vec3(0.050, 0.098, 0.016), r);
                alb = mix(alb, vec3(0.120, 0.120, 0.030), step(0.94, r) * 0.7);
                alb = mix(alb, vec3(0.150, 0.050, 0.030), step(0.975, fract(seed * 31.0)) * 0.6);
                alb *= 0.9 + 0.2 * (1.0 - abs(uv.y));
                trans = alb * vec3(0.55, 0.85, 0.3);
                ao = ao0 * mix(0.35, 1.0, hf);
                return gcLight(p, rd, nb, alb, trans, ao, max(top - yr, 0.0) * 0.8, 0.5, 0.33, 0.25);
            }
            float sp = floor(seed * 0.5);                 // forb leaf
            vec3 nb;
            float spec = 0.25, rough = 0.45;
            if (sp < 0.5) {                               // lady's mantle: round, pleated fan, hairy
                float ang = atan(uv.y, uv.x + 1.0);
                nb = normalize(n + sv * 0.25 * sin(ang * 9.0) + tu * 0.15 * uv.x);
                alb = mix(vec3(0.080, 0.125, 0.038), vec3(0.100, 0.145, 0.045), r);
                spec = 0.15; rough = 0.6;
            } else if (sp < 1.5) {                        // broad pleated leaves, glossy
                nb = normalize(n + sv * 0.32 * sin(uv.y * 10.0) - tu * 0.2 * uv.x);
                alb = mix(vec3(0.085, 0.140, 0.028), vec3(0.120, 0.170, 0.036), r);
                spec = 0.35; rough = 0.45;                // (i17: was 0.7 / 0.32 - plastic)
            } else if (sp > 2.5) {                        // broad ribbed leaves (veratrum, adenostyles)
                nb = normalize(n + sv * 0.30 * sin(uv.y * 7.0 + 0.5 * uv.x) - tu * 0.15 * uv.x);
                alb = mix(vec3(0.080, 0.130, 0.030), vec3(0.115, 0.165, 0.036), r);
                alb *= 0.9 + 0.15 * cos(uv.y * 14.0);
                spec = 0.3; rough = 0.42;
            } else {                                      // lanceolate rosette with a pale midrib
                nb = normalize(n + sv * uv.y * 0.3);
                alb = mix(vec3(0.065, 0.110, 0.024), vec3(0.090, 0.135, 0.030), r);
                alb *= 1.0 + 0.35 * (1.0 - smoothstep(0.0, 0.12, abs(uv.y)));
                spec = 0.25; rough = 0.42;
            }
            alb *= 0.85 + 0.2 * (1.0 - abs(uv.y));
            // cupped blade, darker (shaded, curled) margins, some yellowing leaves
            nb = normalize(nb + sv * uv.y * 0.6);
            alb *= mix(1.0, 0.7, smoothstep(0.65, 1.0, abs(uv.y)));
            alb = mix(alb, vec3(0.205, 0.195, 0.050), step(0.82, fract(seed * 37.1 + 0.13)) * 0.75);
            // (i19) glaucous, grey-green plants (valerian, hawkweed, lupine-like hairy leaves): the meadow
            // foreground is full of pale bluish-grey-green leaves
            alb = mix(alb, vec3(0.120, 0.150, 0.060), smoothstep(0.58, 0.64, vnoise(p.xz * 2.6 + 33.0)) * 0.6);   // (i21: less blue / weaker - read as rubber)
            alb = mix(alb, vec3(luma(alb)) * vec3(1.05, 1.2, 0.8), under * 0.35);   // paler, greyer underside
            trans = alb * vec3(0.6, 0.85, 0.3);
            ao = ao0 * mix(0.45, 1.0, hf);
            return gcLight(p, rd, nb, alb, trans, ao, max(top - yr, 0.0) * 0.6, spec, rough, 0.3);
        }
        // flat flower head (yellow composite / buttercup, white umbel); LOD-widened heads fade to green
        float sp = aux;
        alb = mix(vec3(0.10, 0.14, 0.03), gcFlowerCol(sp), gcFadeK);
        float spec = 0.0, rough = 0.5;
        if (sp < 0.5) {
            alb = mix(alb * vec3(0.85, 0.6, 0.4), alb, smoothstep(0.22, 0.36, uv.x));
            alb *= 0.85 + 0.2 * uv.x;
            spec = 0.2; rough = 0.4;                  // (i14: 0.6 / 0.25 washed the heads out white)
        } else {
            vec3 q = p - gcObj.xyz;
            float fl = texture(uNoise3, q * 70.0 + seed * 9.0).a;
            alb *= mix(0.62, 1.08, fl);
            alb = mix(alb, vec3(0.30, 0.34, 0.16), (1.0 - smoothstep(0.0, 0.25, uv.x)) * 0.5);
        }
        trans = alb * 0.5;
        ao = ao0 * mix(0.6, 1.0, saturate(yr / max(top * 1.4, 0.05)));
        return gcLight(p, rd, n, alb, trans, ao, 0.0, spec, rough, 0.4);
    }
    if (kind < 6.5 && kind > 5.5) {                       // stone
        vec3 q = p - gcObj.xyz;
        float pf = t * gPixAng;
        float gw = 1.0 - smoothstep(0.004, 0.012, pf);
        float n1 = texture(uNoise3, q * 2.0 + gcObj.w * 9.0).r;
        float n2 = texture(uNoise3, q * 3.0 + gcObj.w * 3.0).b;
        // granite grain only (~1 cm): the cut facets stay crisp
        vec3 nb3 = texture(uNoise3, q * 6.0 + gcObj.w * 5.0).rgb - vec3(0.475, 0.487, 0.525);
        n = normalize(n + nb3 * 0.25 * mix(0.4, 1.0, gw));
        // (i12: as dark as the boulders around them - the old 0.31 top read as white eggs in the sun)
        // (i17: paler again - meadow stones #b5b097, boulders lit #c4bfb4; the r8 "white egg" was a rock-module hit)
        alb = vec3(mix(0.15, 0.33, saturate(n1 * 1.4 - 0.2))) * mix(0.85, 1.08, n2) * vec3(1.0, 0.97, 0.88);
        alb *= mix(1.0, mix(0.72, 1.15, texture(uNoise3, q * 17.0 + gcObj.w).a), gw);   // salt and pepper
        float lich = smoothstep(0.5, 0.7, texture(uNoise3, q * 0.9 + gcObj.w).g) * saturate(n.y + 0.3);
        alb = mix(alb, vec3(0.34, 0.35, 0.24), lich * 0.6);
        // dark crustose lichen spots, pale grey-green map lichen (faded out when sub-pixel)
        float cr = smoothstep(0.70, 0.75, texture(uNoise3, q * 2.3 + 3.1).a) * (1.0 - smoothstep(0.006, 0.02, pf));
        alb = mix(alb, vec3(0.06, 0.06, 0.05), cr * 0.6);
        float ml = smoothstep(0.55, 0.62, texture(uNoise3, q * 0.6 + 7.7).g) * saturate(n.y + 0.2);
        alb = mix(alb, vec3(0.27, 0.30, 0.18), ml * 0.45);
        alb = mix(alb, vec3(0.05, 0.08, 0.025), forest * smoothstep(0.3, 0.9, n.y) * smoothstep(0.45, 0.6, n1));
        // faces turned sideways / down hold dirt and their own shade; soil stain at the foot
        alb *= mix(0.6, 1.0, smoothstep(0.05, 0.3, n.y));
        alb = mix(alb, vec3(0.06, 0.06, 0.03), (1.0 - smoothstep(0.0, 0.05, yr)) * 0.6);
        // contact occlusion where the stone sits in the sward
        ao = ao0 * mix(0.25, 1.0, smoothstep(0.0, 0.12, yr)) * mix(0.6, 1.0, saturate(n.y * 0.5 + 0.5));
        if (dot(n, rd) > 0.0) n = normalize(n - rd * dot(n, rd) * 1.05);
        return gcLight(p, rd, n, alb, vec3(0.0), ao, 0.0, 0.08, 0.55, 0.0);
    }
    if (kind > 4.5) {                                     // dwarf shrub / understory bush
        vec3 cen = gcObj.xyz, rad = gcObjR.xyz;
        float sd = gcObj.w, fs = gcObjR.w;
        gcLobes(sd);
        float pf = t * gPixAng;
        // normal: gradient of the foliage incl. the sprig octave and a ~1 cm leaf term (bump only)
        float lw = 1.0 - smoothstep(0.01, 0.03, pf * fs);
        float e = clamp(pf * 1.5, 0.004, 0.02) / sqrt(fs);
        vec3 gN = vec3(gcFoliageS(p + vec3(e, 0, 0), cen, rad, sd, fs, lw) - gcFoliageS(p - vec3(e, 0, 0), cen, rad, sd, fs, lw),
                       gcFoliageS(p + vec3(0, e, 0), cen, rad, sd, fs, lw) - gcFoliageS(p - vec3(0, e, 0), cen, rad, sd, fs, lw),
                       gcFoliageS(p + vec3(0, 0, e), cen, rad, sd, fs, lw) - gcFoliageS(p - vec3(0, 0, e), cen, rad, sd, fs, lw));
        vec3 u = (p - cen) / rad;
        // nearest lobe: its outward direction (the bush's lit and shaded sides) and the depth in it
        float rr = 9.0; vec3 nl = vec3(0.0, 1.0, 0.0);
        for (int k = 0; k < 3; k++) {
            vec3 qk = (u - gcLo[k].xyz) / gcLo[k].w;
            float dk = length(qk);
            if (dk < rr) { rr = dk; nl = qk / rad; }
        }
        n = normalize(-gN / max(length(gN), 1e-5) + normalize(nl) * 0.35 + vec3(0.0, 0.15, 0.0));
        // single leaves (~4.5 cm, ~2.5 cm on the small bilberry cores): flat facets with their own
        // tilt, so lit leaves flash bright and dark gaps open between them; faded when < ~3 px
        float S = 0.045 / pow(fs, 0.5);
        vec4 lc = gcLeafCell(p / S + sd * 17.0);
        float lwc = 1.0 - smoothstep(0.2, 0.45, pf / S);
        vec3 nLeaf = normalize(n * 0.6 + (lc.xyz - 0.5) * 1.5 + vec3(0.0, 0.45, 0.0));
        n = normalize(mix(n, nLeaf, lwc));
        float gap = smoothstep(0.0, 0.14, lc.w);
        // bilberry / understory (saturated deep green, #273d11 in shade) in the forest,
        // juniper / alpine rose (dark, duller) in the open
        vec3 berry = mix(vec3(0.036, 0.076, 0.010), vec3(0.060, 0.112, 0.016), fract(sd * 7.3));
        vec3 jun = mix(vec3(0.022, 0.036, 0.016), vec3(0.034, 0.050, 0.016), fract(sd * 5.1));
        alb = mix(jun, berry, saturate(max(forest * 1.5, stand) + step(0.8, fract(sd * 3.7))));
        // leaf clusters (~4 cm) and single leaves (~1.3 cm), faded by the pixel footprint
        vec3 xs = p * fs;
        float c1 = texture(uNoise3, xs * 1.6 + sd).g;
        alb *= mix(1.0, mix(0.7, 1.25, c1), 1.0 - smoothstep(0.012, 0.035, pf * fs));
        float c2 = texture(uNoise3, xs * 4.8 + sd * 3.0).a;
        alb *= mix(1.0, mix(0.55, 1.3, smoothstep(0.3, 0.72, c2)), 1.0 - smoothstep(0.004, 0.012, pf * fs));
        alb = mix(alb, alb * vec3(1.35, 1.15, 0.8), step(0.86, fract(sd * 11.3)) * 0.6);   // some yellower bushes
        // (i22) when the leaves are sub-pixel their mean keeps the leafy surface's self-shadowing (darker, less
        // glow): the low bilberry carpets read as smooth lime "pads" in shot 8 / r2
        alb *= mix(0.72, (0.75 + 0.5 * lc.y) * mix(0.45, 1.0, gap), lwc);
        alb = mix(alb, alb * vec3(1.4, 1.2, 0.7), step(0.93, lc.z) * 0.5 * lwc);           // a few yellowing leaves
        trans = alb * vec3(0.80, 1.12, 0.40) * mix(0.6, 1.0, lwc);   // back-lit leaves glow lime
        ao = ao0 * mix(0.22, 1.0, smoothstep(0.25, 1.05, rr)) * mix(0.55, 1.0, saturate(u.y * 0.5 + 0.5)) * mix(0.8, mix(0.5, 1.0, gap), lwc);
        return gcLight(p, rd, n, alb, trans, ao, 0.0, 0.45, 0.33, 0.4);
    }
    if (dot(n, rd) > 0.0) n = -n;                         // thin two-sided surfaces face the viewer
    if (kind > 3.5) {                                     // flower head
        vec3 q = p - gcObj.xyz;
        vec3 hr = gcObjR.xyz;
        float sp = gcObjR.w;
        alb = mix(vec3(0.10, 0.14, 0.03), gcFlowerCol(sp), gcFadeK);
        float ang = atan(q.z, q.x);
        float rr = length(q.xz) / max(hr.x, 1e-3);
        if (sp < 0.5) {                                   // five glossy petals, darker centre
            float pet = abs(cos(ang * 2.5 + seed * 6.0));
            alb *= mix(0.7, 1.05, pet) * mix(0.75, 1.0, smoothstep(0.15, 0.45, rr));
            spec = 0.2; rough = 0.4;
        } else if (sp < 1.5) {                            // spike of small florets
            // whorls of hooded florets: dark gaps between them, each floret a small dome
            float yy = q.y / hr.y * 3.2 + seed * 3.0;
            float aa = (ang / TAU) * 5.0 + floor(yy) * 0.5;
            vec2 fq = vec2(fract(yy) - 0.5, fract(aa) - 0.5);
            float fd = length(fq * vec2(1.0, 1.3));
            float flo = 1.0 - smoothstep(0.30, 0.48, fd);
            alb *= mix(0.30, 1.1, flo);
            alb = mix(alb, vec3(0.30, 0.20, 0.22), step(0.85, fract(floor(yy) * 7.3 + floor(aa) * 3.1)) * 0.5);
            n = normalize(n + vec3(fq.y, fq.x, -fq.y) * 1.2 * flo);
        } else if (sp < 2.5) {                            // umbel of tiny florets
            float fl = texture(uNoise3, q * 90.0 + seed * 7.0).a;
            alb *= mix(0.6, 1.1, fl);
            alb = mix(alb, vec3(0.45, 0.42, 0.22), smoothstep(0.75, 0.95, fl) * 0.5);
        } else if (sp < 3.5) {
            alb *= mix(0.7, 1.1, smoothstep(-0.5, 0.8, q.y / hr.y));
        } else {                                          // grass panicle: spikelets
            float sk = texture(uNoise3, q * 60.0 + seed * 7.0).a;
            alb *= mix(0.5, 1.2, smoothstep(0.35, 0.7, sk)) * mix(0.8, 1.05, smoothstep(-1.0, 1.0, q.y / hr.y));
        }
        trans = alb * 0.45;
        ao = ao0 * mix(0.55, 1.0, saturate(yr / max(top * 1.4, 0.05)));
        return gcLight(p, rd, n, alb, trans, ao, max(top - yr, 0.0) * 0.5, spec, rough, 0.5);
    }
    if (kind > 2.5) {                                     // flower stem / grass culm
        alb = aux > 3.5 ? mix(vec3(0.075, 0.095, 0.030), vec3(0.20, 0.18, 0.09), seed) : vec3(0.055, 0.085, 0.022);
        trans = alb * 0.4;
        return gcLight(p, rd, n, alb, trans, ao, depth, 0.1, 0.5, 0.5);
    }
    if (kind > 1.5) {                                     // forb leaf
        float r = fract(seed * 5.3);
        alb = mix(vec3(0.070, 0.112, 0.024), vec3(0.110, 0.145, 0.036), r);
        alb = mix(alb, vec3(0.085, 0.090, 0.032), step(0.85, r));
        // midrib and slightly paler margins
        alb *= 0.85 + 0.25 * aux;
        trans = alb * vec3(0.55, 0.8, 0.3);
        return gcLight(p, rd, normalize(n + vec3(0.0, 0.4, 0.0)), alb, trans, ao, depth, 0.35, 0.32, 0.35);
    }
    // grass blade
    float dry = step(1.5, seed);
    float r = fract(seed);
    // on the forest floor (i14: not within ~2-4 m above the lake / stream: lush shore grass, no golden straw)
    float wlF = waterLevel(p.xz);
    float ffl = smoothstep(0.5, 0.8, gcEnv.x) * (wlF > -1000.0 ? smoothstep(1.5, 4.0, p.y - wlF) : 1.0);
    {
        // terrain slope / aspect at the blade (as the ground albedo uses it)
        vec2 tg = vec2(terrainH(p.xz + vec2(0.5, 0.0), false) - terrainH(p.xz - vec2(0.5, 0.0), false),
                       terrainH(p.xz + vec2(0.0, 0.5), false) - terrainH(p.xz - vec2(0.0, 0.5), false));
        alb = gcBladeCol(r, gcDryness(p.xz, normalize(vec3(-tg.x, 1.0, -tg.y)), terrainAO(p.xz), f.a));
        // (veg3) species mix per tiller: blue-green sedges / fescue next to yellow-green grasses
        float hv = fract(seed * 3.71 + 0.29);
        alb *= mix(vec3(0.86, 1.0, 1.25), vec3(1.04, 1.04, 0.76), smoothstep(0.15, 0.85, hv)) * mix(0.62, 1.15, fract(seed * 11.3));   // (i23: yellow grasses R 1.12 -> 1.04)
        alb = mix(alb, vec3(luma(alb)), 0.2);     // (veg4 G7: the near blades C 0.097 vs target 0.071)
        // lake / stream shores: olive, sun-bleached tussock grass with many dead blades
        float wlB = waterLevel(p.xz);
        float shB = wlB > -1000.0 ? 1.0 - smoothstep(0.8, 2.5, p.y - wlB) : 0.0;
        // (i24: 0.35 -> 0.55 on half the blades: the gap grass is a yellow-olive / straw mix, the render's lime)
        alb = mix(alb, mix(vec3(0.140, 0.150, 0.050), vec3(0.190, 0.185, 0.075), fract(seed * 5.9)), shB * step(0.5, fract(seed * 2.33)) * 0.45);   // (i25: 0.55 read as wheat in c:shorew)
        // (i12) shore sedges / bleached grass: less chroma than the meadow (lit grass B/G ~0.57)
        // (i17: 0.3 -> 0.4, bluer: target lit (113, 122, 69), render was (100, 111, 52))
        // (i20: 0.15 toward an olive, not grey: target grass #5b641c / highlights #acb267 are saturated - the
        //  grey mix made pale cream "wheat" tops)
        // (i26: veg_lit r8 (95, 106, 46) vs target (113, 122, 69), r1 / r7 alike: shore grass lighter, greyer)
        alb = mix(alb, vec3(luma(alb)) * vec3(1.0, 1.05, 0.80), shB * 0.32) * (1.0 + 0.14 * shB);
        // (veg4 G6) the boulder belt up to ~6 m above the water: olive / khaki grass (#686538, h 105;
        // the render's belt read lime, h 120)
        float belt = wlB > -1000.0 ? 1.0 - smoothstep(3.5, 6.5, p.y - wlB) : 0.0;
        alb = mix(alb, vec3(0.17, 0.16, 0.06) * mix(0.8, 1.15, fract(seed * 4.7)), 0.4 * belt);
    }
    float shS = 0.0;
    {
        float wlS = waterLevel(p.xz);
        shS = wlS > -1000.0 ? 1.0 - smoothstep(0.8, 2.5, p.y - wlS) : 0.0;
    }
    // forest floor: golden, half-dry tussocks in the sunny gaps (#ad994f in full sun)
    alb = mix(alb, mix(vec3(0.25, 0.19, 0.06), vec3(0.19, 0.15, 0.05), r), 0.9 * ffl);
    alb = mix(alb, mix(vec3(0.30, 0.25, 0.14), vec3(0.27, 0.20, 0.08), ffl), dry);
    // dark base deep in the sward (i17: only where the sward is dense - isolated tillers on scree / rock are
    // lit down to the ground, they went black as stubble)
    float dk = smoothstep(0.15, 0.7, gcDensH);
    alb *= mix(mix(0.85, 0.30, dk), 1.0, smoothstep(0.0, 0.5, aux));
    // pale tip on a quarter of the blades only (no hay field; i17: fewer on the shores - wheat-field look)
    float tip = step(0.6, fract(seed * 7.31 + 0.17)) * (1.0 - dry) * (1.0 - ffl) * (1.0 - 0.7 * shS);    // (veg4 G8: 25 -> 40 % of the blades)
    // (i13: along the blade, gcUV.x - aux is the height, whose maximum on an arching blade is its crest)
    alb = mix(alb, vec3(0.25, 0.24, 0.11), smoothstep(0.75, 1.0, gcUV.x) * 0.35 * tip);
    // ~1 m patches (same field as the ground texture): darker, bluer forb-rich clumps vs light grass
    float pz = gcPatch(p.xz);
    alb *= mix(0.72, 1.08, pz);
    alb = mix(alb, alb * vec3(0.85, 1.0, 1.3), (1.0 - pz) * 0.35 * (1.0 - ffl));
    trans = alb * vec3(0.80, 0.95, 0.62) * (1.0 - dry * 0.6);
    ao = ao0 * mix(mix(0.7, 0.15, dk), 1.0, pow(hfrac, 0.7));
    depth *= dk;
    // broad waxy sheen, pale (lit grass highlights p90 [175,174,99], r ~ g)
    // (i19: paler, meadow p90 (174, 182, 112): the lit highlights are nearly neutral)
    gcSpecTint = mix(vec3(0.92, 1.0, 0.72), vec3(0.85, 0.95, 0.55), shS);     // (i15: yellow-green sheen - white made pale crooks)
    // (i14) keeled / rolled blade: the normal turns across the blade (gcBladeU), so each blade has a lit
    // and a shaded half instead of one flat tone (flat fans read as cardboard, their backs went black)
    vec3 acr = normalize(cross(n, vec3(0.0, 1.0, 0.0)) + vec3(1e-4, 0.0, 0.0));
    // (i15: a soft keel + a per-tiller twist; a full +-0.8 keel striped narrow blades into ladders, and
    // wrap 0.4: thin blades pass light, their shaded sides never go black)
    // (i19) isolated tillers (sparse sward on scree / rock, low gcDensH): no canopy around them - light from
    // the open sky and bounced off the bright ground on every side: normal further up, wider wrap, more
    // translucency (they were black stubs on pale scree with the sun on their far side, r7)
    float up = mix(1.6, 0.6, dk);
    vec3 nK = normalize(n + acr * (0.35 * clamp(gcBladeU, -1.0, 1.0) + 0.8 * (fract(seed * 5.7) - 0.5)) + vec3(0.0, up, 0.0));
    trans *= mix(1.6, 1.0, dk);
    return gcLight(p, rd, nK, alb, trans, ao, depth, mix(1.5, 0.9, shS), 0.42, mix(0.9, 0.4, dk));
}

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
    waterPass = true;            // terrain seen in reflections / through the water: no shell volumes
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


float sdHero(vec3 p, int i) {
    vec3 q = p - uHeroA[i].xyz;
    q.xz = rot2(uHeroB[i].w) * q.xz;
    vec3 r = uHeroA[i].w * uHeroB[i].xyz;
    vec3 b = r * 0.78;                                        // half extents (the old rounded box: r * 0.74)
    float R = max(b.x, max(b.y, b.z));
    float sd = float(i) * 7.31 + 1.7;
    vec4 h1 = vec4(hash11(sd), hash11(sd + 1.13), hash11(sd + 2.31), hash11(sd + 3.77));
    vec4 h2 = vec4(hash11(sd + 4.19), hash11(sd + 5.53), hash11(sd + 6.71), hash11(sd + 7.97));
    // faces: tilted top, four sides leaning in / out a little, buried base
    float p0 = dot(q, normalize(vec3((h1.x - 0.5) * 0.5, 1.0, (h1.y - 0.5) * 0.5))) - b.y * mix(0.9, 0.98, h1.z);
    float p1 = dot(q, normalize(vec3(1.0, mix(-0.1, 0.35, h1.w), (h2.x - 0.5) * 0.4))) - b.x;
    float p2 = dot(q, normalize(vec3(-1.0, mix(-0.1, 0.35, h2.y), (h2.z - 0.5) * 0.4))) - b.x * mix(0.85, 1.0, h2.w);
    float p3 = dot(q, normalize(vec3((h1.z - 0.5) * 0.4, mix(-0.1, 0.35, h2.x), 1.0))) - b.z;
    float p4 = dot(q, normalize(vec3((h2.w - 0.5) * 0.4, mix(-0.1, 0.35, h1.x), -1.0))) - b.z * mix(0.85, 1.0, h1.w);
    // cuts across edges / corners (74-90 % of the support distance of the b ellipsoid)
    vec3 n5 = rkDir(h1.x, h1.y, 0.35, 1.0), n6 = rkDir(fract(h1.x + 0.4), h2.z, -0.2, 0.4);
    vec3 n7 = rkDir(fract(h1.x + 0.65), h2.w, 0.5, 1.2), n8 = rkDir(fract(h2.x + 0.13), h1.w, 0.1, 0.7);
    float p5 = dot(q, n5) - length(b * n5) * mix(0.74, 0.86, h2.x);
    float p6 = dot(q, n6) - length(b * n6) * mix(0.76, 0.9, h2.y);
    float p7 = dot(q, n7) - length(b * n7) * mix(0.78, 0.9, h1.z);
    float p8 = dot(q, n8) - length(b * n8) * mix(0.74, 0.88, h1.y);
    float kf = R * 0.025;                                     // crisp, unevenly weathered edges
    float d = smax(p0, p1, kf * 1.2);
    d = smax(d, p2, kf * 0.5);
    d = smax(d, p3, kf * 1.0);
    d = smax(d, p4, kf * 0.4);
    d = smax(d, p5, kf * 0.6);
    d = smax(d, p6, kf * 0.8);
    d = smax(d, p7, kf * 0.4);
    d = smax(d, p8, kf * 0.7);
    d = smax(d, -q.y - b.y, kf);                              // base (buried)
    // fracture step along the top / side-1 edge
    d = smax(d, min(p0 + b.y * mix(0.2, 0.4, h2.z), p1 + b.x * mix(0.12, 0.3, h2.w)), kf * 0.4);
    d = smax(d, rkEllipsoid(q, b * 1.6), 0.05 * R);          // trims the far corners
    // edge proximity (largest vs second-largest face distance)
    float m1 = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, max(p7, p8))));
    float m2 = -1e9;
    m2 = max(m2, p0 < m1 ? p0 : -1e9); m2 = max(m2, p1 < m1 ? p1 : -1e9); m2 = max(m2, p2 < m1 ? p2 : -1e9);
    m2 = max(m2, p3 < m1 ? p3 : -1e9); m2 = max(m2, p4 < m1 ? p4 : -1e9); m2 = max(m2, p5 < m1 ? p5 : -1e9);
    m2 = max(m2, p6 < m1 ? p6 : -1e9); m2 = max(m2, p7 < m1 ? p7 : -1e9); m2 = max(m2, p8 < m1 ? p8 : -1e9);
    float edge = 1.0 - smoothstep(0.0, 0.12 * R, m1 - m2);
    // relief: gentle bulges, crumbly worley pits with crisp crests (deeper on the edges), a through-crack
    vec3 s = q / R + float(i) * 3.1;
    float n1 = texture(uNoise3, s * 0.11).g - 0.487;
    float n2 = texture(uNoise3, s * 0.23 + 0.31).b - 0.525;
    float n3 = texture(uNoise3, s * 0.6 + 0.57).b - 0.525;
    d += R * (0.14 * n1 + 0.035 * n2 * (0.55 + 1.15 * edge) + 0.006 * n3);
    float ck = abs(dot(q, normalize(vec3(h2.x - 0.5, 0.15, 1.0))) - (h1.y - 0.5) * 0.6 * b.z) - R * 0.006 * saturate(0.4 + q.y / b.y);
    d = smax(d, -ck, R * 0.008);
    // cm octave near the camera (silhouettes / contact lines)
    float tc = length(p - uCamPos);
    if (tc < 14.0) {
        float m = texture(uNoise3, q * 1.45 + 0.29 + float(i)).b;
        float pz = smoothstep(0.38, 0.62, texture(uNoise3, q * 0.37 + 0.71 + float(i)).r);
        d -= (1.0 - smoothstep(8.0, 14.0, tc)) * 0.011 * mix(0.2, 1.0, max(pz, edge)) * (0.7 + edge) * (smoothstep(0.25, 0.9, m) - 0.55);
    }
    return d * 0.75;
}

float heroSDF(vec3 p, out int idx) {
    float dm = 1e9; idx = -1;
    for (int i = 0; i < 6; i++) {
        if (i >= uHeroN) break;
        float b = length(p - uHeroA[i].xyz) - uHeroA[i].w * max(uHeroB[i].x, max(uHeroB[i].y, uHeroB[i].z)) * 1.3;
        if (b > dm) continue;
        float d = b > 1.0 ? b : sdHero(p, i);
        if (d < dm) { dm = d; idx = i; }
    }
    return dm;
}

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
float rkRosette(vec2 uv, float s, float dens, vec2 w, float fp, int seed, out float rim, out float id) {
    rim = 0.0; id = 0.5;
    float vis = 1.0 - smoothstep(0.09 * s * 1.6, 0.26 * s * 1.6, fp);   // (j2 sweep: x1.6 - rosettes stay down to ~3 px)
    float m = 1.0 - exp(-0.758 * dens);
    if (vis <= 0.0 || dens <= 0.004) return m;
    vec2 p = (uv + w * s) / s;
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i) + ivec2(4513 * seed, 1291 * seed);
    float best = -1.0;
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        uint h = hash2u(c + ivec2(x, y));
        vec4 u = vec4(uvec4(h, h >> 8u, h >> 16u, h >> 24u) & 255u) * (1.0 / 255.0);
        if (u.z >= dens) continue;
        vec2 r = vec2(float(x), float(y)) + 0.1 + 0.8 * u.xy - f;
        float sd = mix(0.3, 0.66, u.w) - length(r);         // > 0 inside the thallus
        if (sd > best) { best = sd; id = u.w; }
    }
    float aa = fp / s + 0.012;
    float cv = smoothstep(-aa, aa, best);
    rim = cv * (1.0 - smoothstep(0.02, 0.07 + aa, best));
    return mix(m, cv, vis);
}
float gRkPaleK = 1.2, gRkKBig = 0.25;    // (rocks4 R11) pale-thallus brightness / big-rosette share (shadeRock sets them per block)
float erfA(float x) { float x2 = x * x; return sign(x) * sqrt(1.0 - exp(-x2 * (1.2732 + 0.147 * x2) / (1.0 + 0.147 * x2))); }
vec3 rkLichen(vec3 alb, vec3 q, vec2 uv, float up, float fp, float cD, float cM, float cP, out float lich) {
    // cluster zones (shared by the layers: one channel each) and the rim warps (big / small thalli)
    vec4 za = texture(uNoise3, vec3(uv * 0.25, 0.13));
    vec4 zb = texture(uNoise3, vec3(uv * 0.75, 0.41));
    vec4 z = ((za - vec4(0.475, 0.487, 0.525, 0.521)) + 0.6 * (zb - vec4(0.475, 0.487, 0.525, 0.521))) * vec4(5.1, 4.7, 5.7, 5.6);
    float wv1 = 1.0 - smoothstep(0.01, 0.03, fp), wv2 = 1.0 - smoothstep(0.004, 0.012, fp);
    vec2 w1 = wv1 > 0.0 ? (texture(uNoise3, vec3(uv * 9.0, 0.71)).rg - vec2(0.475, 0.487)) * (1.4 * wv1) : vec2(0.0);
    vec2 w2 = wv2 > 0.0 ? (texture(uNoise3, vec3(uv * 22.0, 0.29)).rg - vec2(0.475, 0.487)) * (1.4 * wv2) : vec2(0.0);
    float kC = 1.6;                                   // (j5 sweep: pale crust cover x1.6, crusty blocks)
    // thalli per cell for a target cover c: dens = -ln(1 - c) / 0.758, clustered by the zone field (x 0 .. ~2.5)
    // (rocks3c) zoned colonies: the zone field gates the thalli density harder (x max(0, 1 + 1.6 z), mean kept by the
    // 1.06 / 1.26 factor): crusted zones next to bare granite instead of an even countertop speckle
    float zk = 1.2;
    float zm = 1.06 / (0.5 * (1.0 + erfA(1.0 / (zk * 1.4142))) + zk * 0.3989 * exp(-0.5 / (zk * zk)));
    float dP = -log(1.0 - min(cP * kC, 0.9)) * 1.32 * zm, dD = -log(1.0 - min(cD, 0.9)) * 1.32 * zm, dM = -log(1.0 - min(cM, 0.9)) * 1.32 * zm;
    float rimP, idP, rimP2, idP2, rimD, idD, rimM = 0.0, idM = 0.5;
    // pale grey-white crust: big (9 cm cells: 5-12 cm thalli) + small (3.5 cm cells) rosettes, union
    // (rocks3c) more of the crust in small thalli (dense crisp white flecks, fewer big pale blots)
    // (rocks3d) still more in the small thalli (knob 15 = big share; 0.55 / 0.8 painted the near blocks of shot 4 white in
    // 10-50 cm zones; near blocks: dark grey granite with dense small white flecks)
    float kBig = gRkKBig;                     // (k8 sweep: 0.35 / 0.2; rocks4 R11: 0.15 on the forest-shore blocks, finer flecks)
    float cP1 = rkRosette(uv, 0.09, saturate(kBig * dP * max(1.0 + zk * z.g, 0.0)), w1, fp, 1, rimP, idP);
    float cP2 = rkRosette(uv, 0.035, saturate((1.35 - kBig) * dP * max(1.0 + zk * z.g, 0.0)), w2, fp, 2, rimP2, idP2);
    float kP = 1.0 - (1.0 - cP1) * (1.0 - cP2);
    // black / dark grey crust thalli, 3.5 cm cells (3c: 5 cm cells read as dark camouflage blots)
    // (rocks3d) dark crust only weakly zoned (zk 1.6 -> 0.7): dark 10-30 cm blots next to pale zones = camouflage (r8)
    float kD = rkRosette(uv, 0.035, saturate(dD * max(1.0 + 0.7 * z.r, 0.0)), w1, fp, 3, rimD, idD);
    // yellow-green map lichen: 6 cm cells, black prothallus rim
    float kM = dM > 0.0 ? rkRosette(uv, 0.06, saturate(dM * max(1.0 + zk * z.b, 0.0)), w1, fp, 4, rimM, idM) : 0.0;
    kM *= 1.0 - kD;
    kP *= (1.0 - kD) * (1.0 - kM);
    float vA = 1.0 - smoothstep(0.004, 0.012, fp);           // per-thallus tone / rim lines resolved
    float hP = cP1 > cP2 ? idP : idP2;
    vec3 pale = vec3(0.30, 0.305, 0.29) * gRkPaleK * mix(1.0, 0.86 + 0.28 * fract(hP * 7.13), vA);
    pale *= mix(vec3(1.0), vec3(0.93, 1.0, 0.86), step(0.45, fract(hP * 3.1)) * mix(0.6, 1.0, vA));    // greener-grey thalli (3c: 30 -> 55 %, grey-green crust; kept at distance)
    pale *= 1.0 - 0.18 * vA * max(rimP, rimP2);              // slightly darker margin
    // (rocks5 L1) brighter, yellower (map lichen ~#a5aa5a lit, as bright as the pale granite; was a dull olive)
    vec3 map = vec3(0.235, 0.255, 0.075) * mix(1.0, 0.8 + 0.4 * fract(idM * 7.13), vA);
    map *= 1.0 - 0.8 * vA * rimM;                             // black prothallus rim
    alb = mix(alb, vec3(0.07, 0.069, 0.064) * mix(1.0, 0.7 + 0.6 * fract(idD * 7.1), vA), kD * 0.9);
    alb = mix(alb, map, kM);
    alb = mix(alb, pale, kP);
    lich = kM + kP;
    return alb;
}
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
vec3 rkGraniteCrisp(vec3 q, vec3 nl, float fp, float tone, float steep, out float lich) {
    vec2 uv = rkPlaneUV(q, nl);
    vec4 t0 = texture(uNoise3, q * 0.1 + vec3(0.61, 0.27, 0.83));
    vec4 t1 = texture(uNoise3, q * 0.3 + vec3(0.13, 0.71, 0.37));
    // (i8 sweep: x1.1 - lit boulder tops are among the brightest surfaces)
    // (rocks3b) the granite between the lichen thalli darker on crust-rich blocks (dark grey rock, pale crust)
    // (rocks3b) less warm: (0.2,0.194,0.179) lit by the sun came out b/r 0.82 (f4); target lit #b5ac99 / #c4bfb4 / #aea59e
    // = b/r 0.85-0.92
    // (rocks3c) x1.1: r8 rock median (100,108,101) vs target (117,124,117); lit tops #b5ac99 / #c4bfb4
    // (rocks4 R1 / R2) the soft 30 cm - 3 m albedo blotches cut (0.7 / 0.5 / 0.8 -> 0.3 / 0.25 / 0.3: the faces read as painted
    // clay - the variation now comes from the relief and the per-block tones); base x1.1 -> x1.2 (target lit granite p90 211-219)
    // (rocks4 C5) ... but at mid distance (footprint > ~2-4 cm: the lichen rosettes / relief are sub-pixel) the 20-60 cm tone
    // zones are what real lichen / weathering patches look like there (s02 / s05: blocks read as plaster, p10-p90
    // 1 stop vs a target of ~2): x3 there
    float kBl = mix(1.0, 3.0, smoothstep(0.015, 0.05, fp));
    // (rocks5) x1.15: lit block tops L 0.64-0.69 (r1 / r10 / r8 / s4, top quartile) vs a target boulder_lit L 0.72-0.81
    vec3 alb = vec3(0.2, 0.196, 0.187) * 1.2 * 1.25 * (1.0 - 0.3 * gRkCovP) * (1.0 + kBl * (0.3 * (t1.g - 0.487) + 0.25 * (t0.r - 0.475)));
    // (i10) soft weathering tone at ~30 cm, +-8 % (no threshold: tonal depth without camouflage edges), mean 1
    float vW = 1.0 - smoothstep(0.03, 0.08, fp);
    if (vW > 0.0) alb *= 1.0 + 0.3 * kBl * vW * (texture(uNoise3, q * 1.1 + vec3(0.37, 0.19, 0.53)).r - 0.475);
    alb *= mix(vec3(1.0), vec3(1.05, 1.0, 0.92), smoothstep(0.58, 0.72, t0.r));       // iron-stained areas
    alb *= rkGrain(uv, fp);
    // black lichen dots (Lecidea, 2-6 mm), clustered; mean darkening kept when unresolved
    float vB = 1.0 - smoothstep(0.002, 0.005, fp);
    float dens = smoothstep(0.42, 0.62, t1.a);
    float dotC = 0.036 * dens;                     // (rocks3c: x0.6 - r8 blocks read as a black-speckled countertop)
    if (vB > 0.0) {
        vec3 v = voronoi2(uv * 75.0 + 1.9);
        float on = step(fract(v.z * 13.1), 0.27 * dens) * (1.0 - smoothstep(0.17, 0.22, v.x));
        alb *= 1.0 - 0.85 * mix(dotC, on, vB);
    } else alb *= 1.0 - 0.85 * dotC;
    alb = rkLichen(alb, q, uv, nl.y, fp, gRkCovD, gRkCovM * smoothstep(-0.2, 0.5, nl.y), gRkCovP, lich);
    float pkM;
    alb *= rkMidTex(q, fp, steep, pkM);                // (rocks5 M1)
    lich = max(lich, pkM);
    // dark water / cyanobacteria streaks hanging down the steep faces
    if (steep > 0.01) {
        // (rocks3b: 0.25 / 0.035 and x0.3 - the 5 cm x 50 cm streaks read as brushed scratches on the inclined faces, r1)
        float st = texture(uNoise3, vec3(q.x * 0.25, q.y * 0.035, q.z * 0.25) + 0.53).g;
        alb = mix(alb, vec3(0.06, 0.058, 0.052), smoothstep(0.6, 0.76, st) * steep * 0.3);
    }
    return alb * tone;
}

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
float rkBlockTone(Rock rk, out float crust) {
    float u = fract(rk.seed * 11.3), v = fract(rk.seed * 23.1);
    bool slab = rk.kind > 0.5 && rk.kind < 1.5;
    float tone = mix(0.84, 1.06, v);
    crust = 0.0;
    // (rocks3d) the wave-washed band (< ~1.5 m above the water) has no dark-crusted blocks (clean grey waterline blocks;
    // k5 r1 / c_box: black-blotched blocks and cobbles at the water)
    float wsh = 1.0 - smoothstep(0.6, 1.6, rk.rel);
    // (rocks4 R2 / C5) a wider per-block spread: 35 % dark-bodied blocks inland (x0.52-0.68, #424642 next to pale flecked
    // ones), fresh pale ones x1.15-1.35 (fewer on the washed band: R4, the waterline blocks read as pale tombstones)
    float dS = 0.35 * (1.0 - wsh);
    if (!slab && u < dS) { tone = mix(0.52, 0.68, v); crust = 0.22; }    // (rocks3d: 0.58-0.74 = near-black blocks in r1)
       // (rocks3: darker base, fewer black patches)
    else if (u < dS + mix(0.15, 0.08, wsh)) { tone = mix(1.15, 1.35, v); crust = -0.14; }
    // (rocks3c) field blocks no darker (x0.84 -> 0.96): the boulder field is the palest rock (#c4bfb4)
    // (rocks4 R13: slabs x1.2 -> 1.0, plaster; R12: outcrop blocks x1.12 -> 1.25, r7 pale jointed granite #b5b2a7)
    return tone * mix(1.0, 0.85, rk.env) * mix(1.0, 0.96, rk.fld) * (slab ? 1.0 : (rk.kind > 1.5 ? 1.25 : 1.0));
}
// per-facet tone: every fracture face weathered differently (soft switch across the rounded edge)
float rkFacetTone(vec3 q, Rock rk, float fp) {
    float pf[10];
    pf[0] = dot(q, rk.f0.xyz) - rk.f0.w; pf[1] = dot(q, rk.f1.xyz) - rk.f1.w; pf[2] = dot(q, rk.f2.xyz) - rk.f2.w;
    pf[3] = dot(q, rk.f3.xyz) - rk.f3.w; pf[4] = dot(q, rk.f4.xyz) - rk.f4.w; pf[5] = dot(q, rk.f5.xyz) - rk.f5.w;
    pf[6] = dot(q, rk.f6.xyz) - rk.f6.w; pf[7] = dot(q, rk.f7.xyz) - rk.f7.w; pf[8] = dot(q, rk.f8.xyz) - rk.f8.w;
    pf[9] = dot(q, rk.f9.xyz) - rk.f9.w;
    float b1 = -1e9, b2 = -1e9; int i1 = 0, i2 = 0;
    for (int i = 0; i < 10; i++) {
        if (pf[i] > b1) { b2 = b1; i2 = i1; b1 = pf[i]; i1 = i; }
        else if (pf[i] > b2) { b2 = pf[i]; i2 = i; }
    }
    // (rocks3b) narrower (0.84-1.1, was 0.7-1.18): a different albedo on every face on top of its lighting made the
    // blocks read as low-poly papercraft (shot 4)
    float fr = 0.13;
    float t1 = mix(0.97 - fr, 0.97 + fr, hash11(rk.seed * 91.7 + float(i1)));
    float t2 = mix(0.97 - fr, 0.97 + fr, hash11(rk.seed * 91.7 + float(i2)));
    float wf = smoothstep(0.0, max(rk.rr * 0.6, 2.0 * fp), b1 - b2);
    float ft = mix(0.5 * (t1 + t2), t1, wf);
    return mix(1.0, ft, rk.kind > 0.5 && rk.kind < 1.5 ? 0.3 : (rk.ell < 1.5 ? 0.5 : 1.0));
}
// fresh surface of a flat chip (rkChipD): on its cut face, inside its ball
float rkChipFresh(vec3 q, vec4 ch, float R) {
    vec3 dq = q - ch.xyz;
    float pl = dot(dq, normalize(ch.xyz)) + RK_CHIP_D * ch.w;
    return (1.0 - smoothstep(-0.02 * R, 0.01 * R, length(dq) - ch.w)) * (1.0 - smoothstep(0.015 * R, 0.045 * R, abs(pl)));
}
vec3 shadeRock(vec3 p, vec3 rd, float t, ivec2 cell, bool cheap) {
    Rock rk;
    float lod = rockLod(t);
    rkGet(cell, lod, rk);
    float fp = t * gPixAng;
    float R = rk.r.x;
    vec3 q = (p - rk.c) * rk.m;
    // tetrahedral gradient in the local frame
    float e = max(0.004, fp * 0.8);
    vec2 k = vec2(1.0, -1.0);
    gRkN3 = 1.0;
    vec3 nl = normalize(k.xyy * sdRockLocal(q + k.xyy * e, rk, lod) + k.yyx * sdRockLocal(q + k.yyx * e, rk, lod)
                      + k.yxy * sdRockLocal(q + k.yxy * e, rk, lod) + k.xxx * sdRockLocal(q + k.xxx * e, rk, lod));
    gRkN3 = 0.0;
    vec3 n = rk.m * nl;
    vec3 qs = q + rk.seed * 37.0;
    // height above the ground the marcher draws (i7: + the micro relief - the contact zones were misplaced by up to 20 cm)
    float hg = p.y - terrainHD(p.xz, detailWeight(t));
    float wl = waterLevel(p.xz);
    float above = wl > -1000.0 ? p.y - wl : 99.0;
    float steep = 1.0 - smoothstep(0.35, 0.8, nl.y);
    vec4 f = forestAt(p.xz);
    float canopy = saturate(f.r * 1.2);
    // ---- albedo
    float crust;
    float tone = rkBlockTone(rk, crust) * rkFacetTone(q, rk, fp);
    // fresh fracture surfaces (flat chips, fracture-step risers): paler, unweathered granite, few lichens
    float fresh = 0.0;
    if (rk.ch0.w > 0.0) fresh = max(fresh, rkChipFresh(q, rk.ch0, R));
    if (rk.ch1.w > 0.0) fresh = max(fresh, rkChipFresh(q, rk.ch1, R));
    if (rk.nt3.x > 0.0) {
        float pa = dot(q, rk.f0.xyz) - rk.f0.w + rk.nt3.x, pb = dot(q, rk.f3.xyz) - rk.f3.w + rk.nt3.y;
        fresh = max(fresh, 0.7 * (1.0 - smoothstep(0.0, 0.04 * R, abs(min(pa, pb)))) * step(0.0, max(pa, pb)));
    }
    tone *= 1.0 + 0.22 * fresh;
    // (rocks4 R4) the far waterline blocks (40-120 m, r1 far shore) darker: a thin broken lit rim (#8a877e), not pale tombstones
    if (!(rk.kind > 0.5)) tone *= mix(1.0, 0.75, smoothstep(30.0, 60.0, t) * (1.0 - smoothstep(1.0, 1.4, rk.rel)));
    // small stones / beach cobbles: beach-stone colours, as pale as the shingle around them (#b9ab9f;
    // round 2 review: as granite blocks they came out 2x darker than the gravel)
    // (rocks3b) only where the terrain draws the shingle (terrainAlbedo's shore band, not on meadows): the stones in
    // the grass of the stream valley (shot 4) came out sugar-white eggs
    float cobW = (1.0 - smoothstep(0.16, 0.3, R)) * (1.0 - smoothstep(0.35, 1.3, rk.rel)) * (1.0 - saturate(f.a * 2.5));
    // (rocks3c) from the ground height above the water AT the stone (rk.rel extrapolates the cell plane: on the steep
    // forest bank of shot 8 stones 2 m above the lake got shingle colours = sugar eggs), not on the forest floor
    if (wl > -1000.0) cobW *= 1.0 - smoothstep(0.35, 1.1, above - hg);
    cobW *= 1.0 - saturate(rk.env * 1.5);
    vec3 cobC = min(rkCobCol(fract(rk.seed * 3.37)) * 1.25, vec3(0.6)) * 0.95;   // (3c: x1.25 with the paler beach)
    float lich = 0.0;
    float crk = 0.0;                  // (rocks4) joint / crack line coverage (darkening + cavity)
    vec3 alb;
    if (cheap) alb = mix(GRANITE_BASE * tone, cobC, cobW) * mix(0.75, 1.05, texture(uNoise3, qs * 0.06).r);   // reflections etc.
    else if (rk.kind > 1.5) {
        // outcrop blocks are the terrain's own rock standing proud: the same material as the face around them
        // (joints, streaks, lichen zones), with the block / facet tone on top; sampled at a per-block offset so the
        // painted joints do not run across the block like a decal
        gRkOnBlock = true;
        vec2 ho = fract(rk.seed * vec2(71.3, 37.9)) - 0.5;
        alb = rockAlbedo(p + vec3(ho.x, 0.0, ho.y) * 14.6, n, t) * mix(1.0, tone / 1.12, 0.5);
        gRkOnBlock = false;
    } else {
        // map lichen on ~35 % of the open-shore / field blocks, not where the lake washes them
        // map lichen: on ~45 % of the inland boulder-field blocks, few on the wave-washed shore
        // (rocks5 L1) map lichen on more blocks (most field / shore blocks carry a few patches)
        float mapW = (1.0 - rk.env) * step(fract(rk.seed * 5.3), mix(0.3, 0.75, saturate(rk.fld * 1.5)))
                   * smoothstep(0.3, 1.0, above) * (1.0 - cobW);
        // crust: the waterline blocks are washed cleaner, the field blocks older and crustier
        // (rocks3) lichen covers per block: dark crust (crusted blocks ~45 %, fresh ones none, the wave-washed band
        // cleaner), pale grey-white crust (field / old blocks up to ~45 %), map lichen
        float wash = 1.0 - smoothstep(0.2, 1.2, rk.rel);
        float hp = fract(rk.seed * 31.3);
        // (rocks3d: x0.9 -> 0.65 and x(1 - 0.6 wash): r8 pale / dark 10-30 cm blots, black-blotched waterline blocks)
        gRkCovD = max(0.035 + 0.3 * crust - 0.02 * wash, 0.01) * mix(1.0, 0.3, cobW) * 0.65 * (1.0 - 0.6 * wash) * 0.4;   // (3c: x1.4 -> 0.9, less black)   // (j4 sweep: x1.8 / weathering 1.0 a little closer to the target 45-60 stdY)
        // (rocks3b) pale crust also on the forest-shore blocks (heavily crusted), more on the old field blocks
        // (rocks3c) field blocks crustier (+0.14 -> +0.22 fld): the boulder field is pale with grey / white crust on nearly every block
        gRkCovP = (0.06 + 0.3 * hp * hp + 0.14 * rk.fld + 0.15 * rk.env) * (1.0 - 0.75 * wash) * (1.0 + 0.25 * rk.env) * (1.0 - cobW);
        gRkCovM = mapW * mix(0.04, 0.16, fract(rk.seed * 17.9)) * 2.0;
        // (rocks4 R11) forest-shore blocks: coated in pale grey-green crustose lichen, fine 1-5 cm flecks ~2.5x brighter
        // than the rock between (moss only at the bases / on low ledges, below)
        // (the shore blocks under the forest edge too - env is cut near the water, the canopy is not)
        float fsh = smoothstep(0.3, 0.7, max(rk.env, canopy * (1.0 - 0.5 * wash)));
        gRkCovP += 0.3 * fsh * (1.0 - cobW);
        gRkPaleK = mix(1.2, 1.45, fsh); gRkKBig = mix(0.25, 0.15, fsh);
        // (rocks4 R15) no black crust band along the risers of the slabs
        if (rk.kind > 0.5 && rk.kind < 1.5) gRkCovD = mix(gRkCovD, min(gRkCovD, 0.1), steep);
        alb = rkGraniteCrisp(qs, nl, fp, tone, steep, lich);
        // (rocks4 C11) blocks on the forest / falls banks <= ~0.3 lit albedo: near-white blobs on the moss in shot 2 (234,231,220)
        // vs the target mid-grey mossy rocks (p90 ~#9e9f83)
        alb *= mix(1.0, 0.78, max(fsh, rk.env));
        // (i9: relative to rkGraniteCrisp's own mean ~(0.195,0.189,0.175); the stones in the grass came out sugar-white)
        if (cobW > 0.0) alb = mix(alb, cobC * 0.85 * min(alb / (vec3(0.195, 0.191, 0.183) * tone), vec3(1.3)), cobW);
    }
    // (rocks3d) loose stones off the shingle a little darker (shot 8 bank / shot 4 grass: pale sugar chunks next to the
    // lichen-darkened blocks)
    // (rocks4 G10 / C11) inland: grey, not sugar-white eggs in the grass (#9e9f83 lit), <= ~0.22 albedo
    float smI = (1.0 - smoothstep(0.25, 0.4, R)) * (1.0 - cobW);
    alb *= mix(1.0, 0.68, smI);
    alb = mix(alb, min(alb, vec3(0.2)), smI);
    // (rocks3c) polished slabs: weathering stains / grey crust zones 0.2-1 m (+-20 %) and a darker film toward the water -
    // at 1-3 m the clean pale slab read as plaster / concrete (r10 foreground)
    if (rk.kind > 0.5 && rk.kind < 1.5 && !cheap) {
        float vZ = 1.0 - smoothstep(0.05, 0.15, fp);
        float zs = (texture(uNoise3, qs * 0.35 + 0.19).g - 0.487) + 0.5 * (texture(uNoise3, qs * 1.1 + 0.67).g - 0.487) * vZ;
        alb *= 1.0 + 1.1 * zs;
        alb *= mix(1.0, 0.85, 1.0 - smoothstep(0.3, 1.5, above));
        // (rocks4 R13) sheeting / cross joints (dark hairline cracks 1-2 m apart): no smooth plaster whaleback (r10 / r4)
        float jI;
        float jS = max(rkJoint(qs, normalize(vec3(1.0, 0.05, 0.3)), 1.3, 0.006, fp, 1.0, jI),
                       rkJoint(qs + 0.37, normalize(vec3(-0.4, 0.1, 1.0)), 2.1, 0.008, fp, 2.0, jI));
        alb *= 1.0 - 0.7 * jS;
        crk = max(crk, jS);
    }
    // forest floor: olive algae / moss film over the granite (moss-spotted boulders #595745)
    alb *= mix(vec3(1.0), vec3(0.95, 0.98, 0.88), rk.env * 0.35);       // (rocks3b: was (0.9,0.97,0.74): tan)
    // per-rock hue: a few slightly pinkish (K-feldspar) or cooler blocks
    float hue = fract(rk.seed * 3.71);
    alb *= hue < 0.2 ? vec3(1.04, 0.99, 0.95) : (hue > 0.85 ? vec3(0.96, 0.99, 1.03) : vec3(1.0));
    // moss cushions on tops (forest floor, shady shores)
    float mn = texture(uNoise3, qs * 0.38 + 0.21).g;
    float mfv = 1.0 - smoothstep(0.01, 0.04, fp);
    float mf = rkTexS(qs * 2.3 + 0.4).r;
    float mossCov = rk.env * 0.55 + 0.03;
    float mv = nl.y + 0.55 * (mn - 0.5) + 0.35 * mfv * (mf - 0.475) + 0.6 * (mossCov - 0.5);
    float me = 0.08 + 2.0 * fp;                                     // (rocks4 R11: softer cushion edges, 0.02 = a cut hat rim)
    float H = 2.0 * rk.r.y;
    // (rocks4 R11) moss only at the bases and on low ledges, not as a cap over the top (shot 3: a lime hat)
    float lowZ = 1.0 - smoothstep(0.3 * H, 0.8 * H, hg + 0.25 * H * (mn - 0.5));
    float moss = smoothstep(0.82 - me, 0.82 + me, mv) * saturate(mossCov * 1.6) * smoothstep(0.5, 0.6, nl.y) * mix(0.25, 1.0, lowZ);
    float baseM = (1.0 - smoothstep(0.08 * H, 0.3 * H, hg + 0.3 * H * (mn - 0.5) + 0.12 * H * mfv * (mf - 0.475)))
                * saturate(rk.env * 1.4) * smoothstep(-0.2, 0.3, nl.y);
    moss = max(moss, baseM);
    moss *= smoothstep(0.0, 0.25, above);
    vec3 mossC = mix(vec3(0.06, 0.075, 0.022), vec3(0.14, 0.15, 0.05), smoothstep(0.3, 0.75, mn * 0.6 + mf * 0.6));
    mossC *= mix(0.6, 1.15, mix(0.5, mf, mfv));
    // dirt / soil splash and dark contact band near the ground; both zones scale with the stone height H (round 2
    // review: a 6-24 cm stone sat entirely inside the fixed-size zones and rendered black)
    float oc = rk.kind > 1.5 ? 0.35 : 1.0;                         // outcrop blocks: thin soil line only
    float contact = smoothstep(-0.02, min((0.06 + 0.12 * R) * oc, 0.25 * H), hg);
    alb *= mix(vec3(0.55, 0.5, 0.45), vec3(1.0), contact);
    // (rocks4 G13) 0-15 cm ground-contact band: soil / moss stain (~60 %) where the block stands in turf / forest floor
    float cb = (1.0 - smoothstep(0.0, min(0.15 * oc, 0.35 * H), hg + 0.04 * (mn - 0.5))) * smoothstep(0.4, 1.2, above);
    alb = mix(alb, mix(vec3(0.075, 0.064, 0.048), vec3(0.05, 0.065, 0.02), smoothstep(0.45, 0.6, mn) * saturate(0.3 + rk.env)), 0.6 * cb);
    // (rocks4 R10) hairline joint (rkBuild: no open wedge any more): a 1-2 px dark line with a little wobble, faded out
    // when sub-pixel at ~25 m
    if (rk.ckw > 0.0 && !cheap) {
        float vcr = 1.0 - smoothstep(0.012, 0.03, fp);
        if (vcr > 0.0) {
            float dcr = abs(dot(q, rk.ck.xyz) - rk.ck.w + 0.06 * R * (texture(uNoise3, qs * 0.5 + 0.3).r - 0.475));
            float wcr = max(rk.ckw * (0.4 + saturate(0.55 + 0.6 * q.y / rk.r.y)), 0.5 * fp);
            float ck1 = (1.0 - smoothstep(wcr, wcr + fp, dcr)) * vcr;
            alb *= 1.0 - 0.75 * ck1;
            crk = max(crk, ck1);
        }
    }
    // wet band at the waterline: dark and glossy
    // (rocks3d) narrower / lighter (wet 8-30 cm x0.62 + splash 0.25 m x0.7: low waterline blocks were dark all over (c_box),
    // the slab foot of r10 a jagged black fringe)
    float wb = 0.05 + 0.14 * texture(uNoise3, qs * 0.6).g;                 // wet band ~5-15 cm
    float wet = 1.0 - smoothstep(0.0, wb, above);
    alb *= mix(0.82, 1.0, smoothstep(0.0, 0.18, above));     // splash zone (bottom 0.18 m): darker, stained
    alb = mix(alb, alb * vec3(0.68, 0.68, 0.64), wet);
    alb = mix(alb, mossC, moss);
    // spray zone of the waterfall: dark wet granite, lime moss on the upper faces
    float fw = fallsWetness(p);
    if (fw > 0.0) {
        float mN = texture(uNoise3, qs * 0.45 + 0.17).r;
        alb = mix(alb, alb * 0.38, fw);
        float mw = fw * smoothstep(0.25, 0.7, nl.y + 0.5 * (mN - 0.475));
        alb = mix(alb, mix(vec3(0.045, 0.06, 0.012), vec3(0.12, 0.14, 0.025), smoothstep(0.35, 0.65, mN)), mw * 0.85);
        wet = max(wet, fw * (1.0 - mw) * 0.8);
        moss = max(moss, mw * 0.85);
    }
    // (rocks4 C11) blocks on the falls banks (r4: near-white 246,241,236 blobs on the dark wet bank; target #3f3d38-#878462):
    // damp, darker granite with moss on the upper faces, as the cliff around them
    float fbk = fallsBank(p) * (1.0 - fw);
    if (fbk > 0.0) {
        float mN2 = texture(uNoise3, qs * 0.45 + 0.53).r;
        alb *= mix(1.0, 0.42, fbk);
        float mb = fbk * smoothstep(0.35, 0.75, nl.y + 0.5 * (mN2 - 0.475)) * 0.7;
        alb = mix(alb, mix(vec3(0.05, 0.065, 0.015), vec3(0.13, 0.15, 0.03), smoothstep(0.35, 0.65, mN2)), mb);
        moss = max(moss, mb);
    }
    // ---- surface relief: pitted, crystalline granite (bump + cavity darkening)
    float cav = 1.0, grainPit = 0.0, microSh = 1.0;
    // (rocks3c) mid-scale bump (5-30 cm undulation, knobs, 9 cm pits, facets) weaker: at full strength every face read as
    // crumpled foil / sponge (r1, shot 4); real granite shows flat-ish fracture planes with a fine crisp grain
    // (rocks4 R1) near the camera (footprint < ~1 cm) 0.85: knobby, pitted faces at 2-6 m (target hp std 25-33)
    float RLF = mix(0.85, 0.5, smoothstep(0.006, 0.016, fp));
    if (!cheap) {
        vec4 r = rkRelief(qs, fp, 0.022 * RLF, 5.0, 4);
        r *= 1.0 - 0.4 * lich;
        // knobby mid-scale relief: rounded bulges and dents (5-30 cm), in patches (round 2 review: no gradient
        // sign flip / crease darkening - their noise contour lines read as marble veins)
        float vk = 1.0 - smoothstep(0.1, 0.35, fp * 3.2);
        if (vk > 0.0) {
            vec4 k1 = rkNoised(qs * 3.2 + 9.7);
            float cm = smoothstep(0.42, 0.62, texture(uNoise3, qs * 0.3 + 0.77).g);
            r.yzw += k1.yzw * (0.03 * 3.2 * RLF * 1.4) * vk * (1.0 - 0.5 * lich) * mix(0.3, 1.0, cm);   // (rocks4 R1: x1.4)
        }
        // crumbly, pitted granite: worley cones - pits at the cell sites, sharp crests
        // along the cell borders - at ~9 cm and ~3 cm, 6-15 cm weathering pockets on ~30 % of the surface, with
        // self-shadowing along the sun direction (height taps toward the sun) and cavity darkening; normal-only
        float vm1 = 1.0 - smoothstep(0.006, 0.02, fp), vm2 = 1.0 - smoothstep(0.003, 0.009, fp);
        if (vm1 > 0.0) {
            // pitting varies over the block: heavily pitted patches next to smoother, flaking ones
            vec4 tm = texture(uNoise3, qs * 0.15 + 0.63);
            float pm = smoothstep(0.32, 0.66, tm.g);
            // (rocks3d) knob 5 = crumbly micro relief (9 / 3 cm worley) strength: real near blocks are lumpy at 2-8 cm
            // with crisp sunlit knobs and shadowed pits; ours read flat / painted at 2-4 m (shot 4 big block)
            float kMR = 1.8;                                     // (rocks4 R1: 1.3 -> 1.8; 2.2 read as pumice on r8's block)
            float a1 = 0.024 * vm1 * (1.0 - 0.5 * lich) * (1.0 - moss) * mix(0.3 * RLF, 1.35, pm * pm) * kMR, a2 = 0.008 * vm2 * (1.0 - 0.5 * lich) * (1.0 - moss) * kMR;
            // (30 % of the blocks are heavily pocketed - tafoni: 6-10 cm deep on ~45 % of the
            // surface; the others 3-5 cm on ~25 %)
            // (rocks3: round pockets on 30 % of the blocks read as cheese / dalmatian - 8 %, shallower)
            float tf = step(fract(rk.seed * 29.7), 0.04);                 // (rocks3d: 8 -> 4 %, r1 black-pocketed blocks)
            float a4 = mix(mix(0.008, 0.02, fract(rk.seed * 41.3)), mix(0.035, 0.06, fract(rk.seed * 41.3)), tf)
                     * smoothstep(mix(0.55, 0.43, tf), mix(0.65, 0.53, tf), tm.r) * vm1 * (1.0 - moss) * (1.0 - fresh)
                     * smoothstep(0.25, 0.5, R);
            // (rocks3d) wave-washed blocks / beach cobbles are smoother: no pockets, shallower pits
            float wsh = 1.0 - smoothstep(0.6, 1.6, rk.rel);
            a4 *= 1.0 - 0.75 * wsh;
            a1 *= (1.0 - 0.12 * wsh) * (1.0 - 0.6 * cobW);   // (rocks4 R1: 0.35 -> 0.12, waterline boulders are knobbly)
            vec3 xm = qs;
            const float em = 0.0035;
            float h1 = rkMicroH(xm + vec3(em, -em, -em), a1, a2, a4), h2 = rkMicroH(xm + vec3(-em, -em, em), a1, a2, a4);
            float h3 = rkMicroH(xm + vec3(-em, em, -em), a1, a2, a4), h4 = rkMicroH(xm + vec3(em, em, em), a1, a2, a4);
            vec3 gm = (vec3(1.0, -1.0, -1.0) * h1 + vec3(-1.0, -1.0, 1.0) * h2 + vec3(-1.0, 1.0, -1.0) * h3 + vec3(h4)) / (4.0 * em);
            float hc = 0.25 * (h1 + h2 + h3 + h4);
            r.yzw += gm;
            grainPit = saturate(-hc / max(a1 + a2 + 0.5 * a4, 1e-4) * 1.6 - 0.35);
            // pocket interiors: deep, near-black cavities (sky occluded)
            if (a4 > 0.004) cav *= 1.0 - 0.7 * smoothstep(0.62, 0.78, texture(uNoise3, xm * 0.6 + 0.41).b) * saturate(a4 / 0.06);
            // self-shadow of the pits / pockets / behind the crests: is the surface 3 cm (and 8 cm, pockets) toward
            // the sun higher than the sun ray at that distance?
            vec3 sl = uSunDir * rk.m;                                   // sun in the local frame
            float se = dot(sl, nl);
            vec3 lt = sl - nl * se;
            float ll = length(lt);
            if (se > 0.0 && ll > 0.05) {
                vec3 dl = lt / ll;
                float hs = rkMicroH(xm + dl * 0.03, a1, a2, a4);
                float s1 = smoothstep(0.0, 0.006, hs - hc - 0.03 * se / ll);
                // (rocks4 R1) second tap on every block (6 cm; 8 cm on the pocketed ones): the 9 cm pits / knobs cast shadows
                float dl2 = a4 > 0.004 ? 0.08 : 0.06;
                float hs2 = rkMicroH(xm + dl * dl2, a1, a2, a4);
                s1 = max(s1, smoothstep(0.0, 0.01, hs2 - hc - dl2 * se / ll));
                // (rocks3c) weaker where the sun grazes the face (sun-grazed crack walls: leopard blotches)
                microSh = 1.0 - s1 * 0.95 * mix(0.4, 1.0, smoothstep(0.12, 0.45, se));
            }
        }
        // angular micro-facets (chipped, crystalline faces): 2D voronoi cells in the plane of
        // the face, each tilted at random -> crisp facet edges; ~9 cm and ~3.5 cm cells
        float vf1 = 1.0 - smoothstep(0.012, 0.035, fp), vf2 = 1.0 - smoothstep(0.005, 0.014, fp);
        // (rocks5 M2) ~25 cm chipped facets, kept until ~2 px (fp 5-12 cm): at 10-40 m the faces of shot 4 / r8 were smooth
        // clay; real granite breaks every face into small lit / shaded facets at that scale
        float vf0 = (1.0 - smoothstep(0.05, 0.12, fp)) * 0.35;
        if (vf0 > 0.0) {
            const vec3 FA = vec3(0.8513, 0.1277, 0.5087), FB = vec3(0.1213, 0.4852, 0.866);
            vec3 ta = normalize(cross(nl, abs(dot(nl, FA)) < 0.85 ? FA : FB));
            vec3 tb = cross(nl, ta);
            vec2 uv = vec2(dot(qs, ta), dot(qs, tb));
            vec3 fo = vec3(0.0);
            vec3 v0 = voronoi2(uv * 4.0 + 1.7);
            vec2 t0 = (fract(v0.z * vec2(11.3, 23.7)) - 0.5) * vf0;
            vec3 fo0 = ta * t0.x + tb * t0.y;
            if (vf1 > 0.0) {
            vec3 v1 = voronoi2(uv * 11.0 + 3.1);
            vec2 t1 = (fract(v1.z * vec2(13.7, 29.3)) - 0.5) * 0.5 * vf1;
            fo += ta * t1.x + tb * t1.y;
            if (vf2 > 0.0) {
                vec3 v2 = voronoi2(uv * 29.0 + 7.7);
                vec2 t2 = (fract(v2.z * vec2(17.1, 23.9)) - 0.5) * 0.3 * vf2;
                fo += ta * t2.x + tb * t2.y;
            }
            }
            r.yzw += (fo0 + fo * RLF) * (1.0 - moss) * (1.0 - 0.4 * lich);
        }
        if (moss > 0.0) r = mix(r, rkRelief(qs, fp, 0.01, 25.0, 2) + rkRelief(qs * 1.3, fp, 0.012, 7.0, 1), moss);   // cushions
        cav = mix(1.0, 1.0 - 0.65 * saturate(-r.x / 0.02), 1.0 - smoothstep(0.008, 0.04, fp));
        cav *= 1.0 - 0.55 * grainPit * (1.0 - moss);    // (rocks4 R1: 0.45 -> 0.55; 0.7 + kMR 2.2 = pumice / sponge, r8)
        cav *= 1.0 - 0.6 * crk;                         // (rocks4 R10) the joint is occluded
        n = rkPerturb(n, rk.m * r.yzw * 1.5 * 1.15);   // (i4 sweep: x1.5 crisper pitting; rocks4 R1: x1.15)
    }
    // ---- occlusion: ground contact, self-occlusion (cracks, under-cuts), sky visibility, canopy
    float ao = terrainAO(p.xz) * mix(1.0, 0.5, canopy);
    float cAO = smoothstep(-0.03, min((0.25 + 0.3 * R) * oc, 0.6 * H), hg);
    ao *= mix(R < 0.35 ? 0.5 : 0.42, 1.0, cAO) * mix(0.7, 1.0, saturate(nl.y * 0.5 + 0.5));
    float occN = 0.0;
    if (!cheap) {
        // first probe 3 cm out on the full-relief shape (the gaps between the knobs and steps darken), second on the
        // smooth shape (under-cuts, the crack, fracture steps)
        const float hp1 = 0.03;
        float hp2 = 0.22 * R + 0.05;
        gRkN3 = 1.0;
        float d1 = sdRockLocal(q + nl * hp1, rk, lod);
        gRkN3 = 0.0;
        float d2 = sdRockLocal(q + nl * hp2, rk, 0.0);
        float occ = saturate((hp1 - d1) / hp1 * 0.5 + (hp2 - d2) / hp2 * 0.5);
        cav *= 1.0 - 0.7 * occ;
        // (rocks3b) occlusion by the NEIGHBOURING blocks (the probes above see only this block): two probes along the
        // smooth normal into the boulder field SDF (self included, Lipschitz x0.8 undone): crevices between touching
        // blocks, faces turned toward a neighbour, the gap under an overhang go dark (near-black crevices;
        // i11: lit / shade ~5.8 vs a target of 9)
        if (t < 90.0) {
            vec3 nG = rk.m * nl;
            float ha = clamp(0.35 * R, 0.08, 0.35), hb = clamp(1.1 * R, 0.25, 1.2);
            ivec2 cN;
            gRkLastD = -1.0;
            float da = rocksSDF(p + nG * ha, 0.0, cN) * 1.25;
            gRkLastD = -1.0;
            float db = rocksSDF(p + nG * hb, 0.0, cN) * 1.25;
            gRkLastD = -1.0;
            occN = saturate(0.55 * saturate(1.0 - da / ha) + 0.45 * saturate(1.0 - db / hb)) * (1.0 - smoothstep(60.0, 90.0, t));
            occN = saturate(occN * 2.0);
        }
    }
    // ---- sun visibility
    float ndl = max(dot(n, uSunDir), 0.0);
    float sun = terrainShadow(p + n * 0.05, cheap ? 30 : 70) * cloudShadow(p);
    if (cheap) sun *= mix(1.0, 0.45, canopy);
    else if (sun > 0.0 && dot(rk.m * nl, uSunDir) > -0.05) {
        sun *= treeShadow(p + n * 0.1);
        if (sun > 0.0) sun *= rocksShadow(p + rk.m * nl * 0.015, uSunDir, 10.0, lod);
    } else if (dot(rk.m * nl, uSunDir) <= -0.05) sun = 0.0;
    // ---- lighting: scene sky/ambient model, cavity applied to all ambient, warm ground bounce
    float spec = mix(0.05, 0.4, wet) * (1.0 - moss);
    float rough = mix(mix(0.5, 0.62, lich), 0.16, wet);
    // (rocks3d) floor 0.3: x2 occN saturated in every concave notch (fracture steps, the gap under an overhang) -> ambient
    // AND bounce 0 = pure black "mail slots" (r8 right block, shot 4 centre blocks); shaded steps should stay readable
    float vN = 1.0 - 0.8 * occN;                                    // (rocks3b) neighbour occlusion
    // (rocks4 R2) ambient 0.85 -> 0.6, bounce 0.75 -> 0.55: lit / shade 3-4x -> 6-9x on a block
    vec3 amb = skyLight(n, ao) * cav * mix(0.5, 1.0, contact) * vN * 0.6;
    // shaded granite is grey, not blue (#40403c): the sky fill is warmed by the light bounced from
    // the sunlit gravel / grass / other blocks around; under the canopy it is filtered green-yellow
    // (rocks3b) the warm ground bounce fades with a low sun: golden-hour shade is cool blue-grey (#4f5660)
    // (rocks3b: canopy tint (1.15,1,0.62) made the forest-shore blocks of shot 8 tan cardboard instead of cool grey)
    amb *= mix(mix(vec3(1.0), vec3(1.03, 1.0, 0.93), smoothstep(0.25, 0.6, uSunDir.y)), vec3(1.02, 1.0, 0.9), canopy);
    // (j2: shaded shore faces came out near black, [26,29,31] vs a target ~[60,65,62]: the sunlit beach / lake
    // around them bounce more light, most within ~3 m of the water)
    vec3 bounce = uSunCol * max(uSunDir.y, 0.0) * (vec3(0.21, 0.2, 0.178) * saturate(0.6 - 0.5 * n.y)
                + vec3(0.05, 0.06, 0.025) * saturate(0.3 - n.y)) * (1.0 - 0.7 * canopy) * cav
                * (1.0 + 0.6 * (1.0 - smoothstep(0.0, 3.0, above))) * vN * 0.55;
    vec3 col = alb * (uSunCol * ndl * sun * microSh * mix(1.0, cav, 0.6) + amb + bounce);
    col += litSurface(vec3(0.0), n, rd, sun * microSh, ao, spec, rough);
    return col;
}

// --- hero blocks (hero.glsl) with the boulder material ----------------------------------------------------
// (rocks3b) PROPOSAL for scene.frag (not a rocks3 file): shade the K_HERO hits with shadeHeroRock(p, rd, t, i, cheap)
// instead of shadeHero, whose thresholded worley "lichen" draws evenly spaced yellow dots over the opening shot's
// block (shot 0) and whose grain is a smooth-noise multiplier. Same crisp granite / lichen rosettes / pitted relief
// with micro-shadows / contact + wet bands / falls spray / lighting as the scattered boulders (shadeRock).
vec3 shadeHeroRock(vec3 p, vec3 rd, float t, int i, bool cheap) {
    float fp = t * gPixAng;
    float e = max(0.004, fp * 0.8);
    vec2 k = vec2(1.0, -1.0);
    vec3 n = normalize(k.xyy * sdHero(p + k.xyy * e, i) + k.yyx * sdHero(p + k.yyx * e, i)
                     + k.yxy * sdHero(p + k.yxy * e, i) + k.xxx * sdHero(p + k.xxx * e, i));
    mat2 ry = rot2(uHeroB[i].w);                              // world -> local (yaw only, as sdHero)
    vec3 q = p - uHeroA[i].xyz; q.xz = ry * q.xz;
    vec3 nl = n; nl.xz = ry * nl.xz;
    float R = uHeroA[i].w * max(uHeroB[i].x, max(uHeroB[i].y, uHeroB[i].z)) * 0.78;
    float hs = hash11(float(i) * 7.31 + 9.1);
    vec3 qs = q + float(i) * 37.0 + 11.0;
    float steep = 1.0 - smoothstep(0.35, 0.8, nl.y);
    float hg = p.y - terrainHD(p.xz, detailWeight(t));
    float wl = waterLevel(p.xz);
    float above = wl > -1000.0 ? p.y - wl : 99.0;
    float canopy = saturate(forestAt(p.xz).r * 1.2);
    float fw = fallsWetness(p);
    // an old block: dark crust spots, pale grey-white crust, some map lichen on the top
    gRkCovD = 0.1; gRkCovP = 0.24 * (1.0 - fw); gRkCovM = 0.07 * (1.0 - fw) * smoothstep(0.3, 1.0, above);
    float lich;
    vec3 alb = rkGraniteCrisp(qs, nl, fp, mix(0.94, 1.06, hs), steep, lich);
    alb *= mix(vec3(1.0), vec3(0.95, 0.98, 0.88), canopy * 0.35);
    // moss cushions on the shaded top near the forest
    float mn = texture(uNoise3, qs * 0.38 + 0.21).g;
    // (j6: canopy * 0.55 as for the forest blocks covered the whole top of the shot-0 block in a green film)
    float moss = smoothstep(0.8, 0.84, nl.y + 0.55 * (mn - 0.5) + 0.6 * (canopy * 0.3 - 0.5)) * canopy * 0.7 * smoothstep(0.0, 0.25, above);
    vec3 mossC = mix(vec3(0.035, 0.048, 0.014), vec3(0.11, 0.12, 0.04), smoothstep(0.3, 0.75, mn));
    // dark contact band at the foot, wet / stained band at the waterline
    float contact = smoothstep(-0.02, 0.06 + 0.12 * R, hg);
    alb *= mix(vec3(0.55, 0.5, 0.45), vec3(1.0), contact);
    float wb = 0.08 + 0.22 * texture(uNoise3, qs * 0.6).g;
    float wet = 1.0 - smoothstep(0.0, wb, above);
    alb *= mix(0.7, 1.0, smoothstep(0.0, 0.25, above));
    alb = mix(alb, alb * vec3(0.62, 0.62, 0.58), wet);
    alb = mix(alb, mossC, moss);
    if (fw > 0.0) {
        float mN = texture(uNoise3, qs * 0.45 + 0.17).r;
        // (rocks4 C10) far away (shot 0 / 1, ~1 km) the wet block at the falls was a black box beside the white plume:
        // the wet darkening x0.38 -> x0.7 beyond ~300 m, more moss on the faces
        float fF = smoothstep(250.0, 400.0, t);
        alb = mix(alb, alb * mix(0.38, 0.7, fF), fw);
        float mw = fw * smoothstep(0.25, 0.7, nl.y + 0.5 * (mN - 0.475) + 0.35 * fF);
        alb = mix(alb, mix(vec3(0.045, 0.06, 0.012), vec3(0.12, 0.14, 0.025), smoothstep(0.35, 0.65, mN)), mw * 0.85);
        wet = max(wet, fw * (1.0 - mw) * 0.8);
        moss = max(moss, mw * 0.85);
    }
    // relief: pitted crystalline surface (bump + cavity + sun micro-shadows), as shadeRock
    float cav = 1.0, microSh = 1.0;
    if (!cheap) {
        vec4 r = rkRelief(qs, fp, 0.022, 5.0, 4) * (1.0 - 0.4 * lich);
        float grainPit = 0.0;
        float vm1 = 1.0 - smoothstep(0.006, 0.02, fp), vm2 = 1.0 - smoothstep(0.003, 0.009, fp);
        if (vm1 > 0.0) {
            vec4 tm = texture(uNoise3, qs * 0.15 + 0.63);
            float pm = smoothstep(0.32, 0.66, tm.g);
            float a1 = 0.024 * vm1 * (1.0 - 0.5 * lich) * (1.0 - moss) * mix(0.3, 1.35, pm);
            float a2 = 0.008 * vm2 * (1.0 - 0.5 * lich) * (1.0 - moss);
            float a4 = 0.014 * smoothstep(0.55, 0.65, tm.r) * vm1 * (1.0 - moss);
            const float em = 0.0035;
            float h1 = rkMicroH(qs + vec3(em, -em, -em), a1, a2, a4), h2 = rkMicroH(qs + vec3(-em, -em, em), a1, a2, a4);
            float h3 = rkMicroH(qs + vec3(-em, em, -em), a1, a2, a4), h4 = rkMicroH(qs + vec3(em, em, em), a1, a2, a4);
            r.yzw += (vec3(1.0, -1.0, -1.0) * h1 + vec3(-1.0, -1.0, 1.0) * h2 + vec3(-1.0, 1.0, -1.0) * h3 + vec3(h4)) / (4.0 * em);
            float hc = 0.25 * (h1 + h2 + h3 + h4);
            grainPit = saturate(-hc / max(a1 + a2 + 0.5 * a4, 1e-4) * 1.6 - 0.35);
            vec3 sl = uSunDir; sl.xz = ry * sl.xz;
            float se = dot(sl, nl);
            vec3 lt = sl - nl * se;
            float ll = length(lt);
            if (se > 0.0 && ll > 0.05) {
                float hsun = rkMicroH(qs + lt / ll * 0.03, a1, a2, a4);
                microSh = 1.0 - 0.95 * smoothstep(0.0, 0.006, hsun - hc - 0.03 * se / ll);   // (rocks4 R1: 0.75 -> 0.95)
            }
        }
        cav = mix(1.0, 1.0 - 0.65 * saturate(-r.x / 0.02), 1.0 - smoothstep(0.008, 0.04, fp));
        cav *= 1.0 - 0.55 * grainPit * (1.0 - moss);   // (rocks4 R1: 0.45 -> 0.55)
        vec3 g = r.yzw * 1.5;
        g.xz = g.xz * ry;                                     // local -> world (inverse yaw)
        n = rkPerturb(n, g);
        // self-occlusion of the steps / crack / under-cuts: probe out along the normal on the block's own SDF
        float hp2 = 0.22 * R + 0.05;
        float occ = saturate((hp2 - sdHero(p + n * hp2, i) / 0.75) / hp2);
        cav *= 1.0 - 0.6 * occ;
    }
    float ao = terrainAO(p.xz) * mix(1.0, 0.5, canopy);
    ao *= mix(0.42, 1.0, smoothstep(-0.03, 0.25 + 0.3 * R, hg)) * mix(0.7, 1.0, saturate(nl.y * 0.5 + 0.5));
    float ndl = max(dot(n, uSunDir), 0.0);
    float sun = sunShadowMap(p + n * 0.05) * cloudShadow(p);
    if (!cheap && sun > 0.0) sun *= treeShadow(p + n * 0.1);
    float spec = mix(0.05, 0.4, wet) * (1.0 - moss);
    float rough = mix(mix(0.5, 0.62, lich), 0.16, wet);
    // (rocks4 R2) ambient x0.75, bounce 0.75 -> 0.55 (as the boulders: lit / shade 6-9x)
    vec3 amb = skyLight(n, ao) * cav * mix(0.5, 1.0, contact) * 0.75;
    amb *= mix(mix(vec3(1.0), vec3(1.03, 1.0, 0.93), smoothstep(0.25, 0.6, uSunDir.y)), vec3(1.02, 1.0, 0.9), canopy);
    vec3 bounce = uSunCol * max(uSunDir.y, 0.0) * (vec3(0.21, 0.2, 0.178) * saturate(0.6 - 0.5 * n.y)
                + vec3(0.05, 0.06, 0.025) * saturate(0.3 - n.y)) * (1.0 - 0.7 * canopy) * cav
                * (1.0 + 0.6 * (1.0 - smoothstep(0.0, 3.0, above))) * 0.55;
    vec3 col = alb * (uSunCol * ndl * sun * microSh * mix(1.0, cav, 0.6) + amb + bounce);
    col += litSurface(vec3(0.0), n, rd, sun * microSh, ao, spec, rough);
    return col;
}
// --- for the water shader: a rock seen through the water (K_ROCK hit of a refraction ray) ----------------
// returns the albedo (wet granite, olive algae film on the upper faces, dark contact zone) and the normal n
// (water_shade.glsl: if (hb.kind == K_ROCK) alb = rockBedAlbedo(bp, t, hb.cell, bn);)
// Round 2 review: the submerged part was brighter than the wet band above the water (pale pedestals): same wet
// darkening x0.62, block / facet tone like shadeRock, and the face's sky visibility (the water shader's ambient
// ignores the normal); smooth-shape normal with >= 2 cm taps (striped banding)
vec3 rockBedAlbedo(vec3 p, float t, ivec2 cell, out vec3 n) {
    Rock rk;
    float lod = rockLod(t);
    rkGet(cell, lod, rk);
    vec3 q = (p - rk.c) * rk.m;
    float fp = t * gPixAng;
    float e = max(0.02, fp * 1.5);
    vec2 k = vec2(1.0, -1.0);
    vec3 nl = normalize(k.xyy * sdRockLocal(q + k.xyy * e, rk, 0.0) + k.yyx * sdRockLocal(q + k.yyx * e, rk, 0.0)
                      + k.yxy * sdRockLocal(q + k.yxy * e, rk, 0.0) + k.xxx * sdRockLocal(q + k.xxx * e, rk, 0.0));
    n = rk.m * nl;
    vec3 qs = q + rk.seed * 37.0;
    float crust;
    float tone = rkBlockTone(rk, crust) * rkFacetTone(q, rk, fp) * (1.0 - 1.5 * max(crust, -0.06));
    float cobW = (1.0 - smoothstep(0.16, 0.3, rk.r.x)) * (1.0 - smoothstep(1.0, 2.5, rk.rel));
    vec3 alb = mix(GD_MEAN * tone, rkCobCol(fract(rk.seed * 3.37)) * 0.95, cobW) * mix(0.7, 1.12, texture(uNoise3, qs * 0.3).r);
    alb *= mix(1.0, 0.55, smoothstep(0.62, 0.72, texture(uNoise3, qs * 1.7 + 0.4).a) * (1.0 - smoothstep(0.01, 0.04, fp)));
    alb *= vec3(0.62, 0.62, 0.58) * mix(vec3(0.86, 0.86, 0.76), vec3(1.0, 0.98, 0.8), saturate(nl.y * 0.7 + 0.3));
    alb *= mix(0.75, 1.0, saturate(n.y * 0.5 + 0.5));               // (rocks3d: 0.55 -> 0.75, submerged blocks = dark plates in r10)
    // (i7) crystal grain + mottled algae film (submerged blocks read as smooth cut stone)
    alb *= rkGrain(rkPlaneUV(qs, nl), fp * 1.5);
    float vf = 1.0 - smoothstep(0.02, 0.06, fp);
    if (vf > 0.0) alb *= 1.0 + 0.35 * vf * (texture(uNoise3, qs * 0.9 + 0.3).g - 0.487) * 2.0;
    float hg = p.y - terrainHD(p.xz, detailWeight(t));
    alb *= mix(0.45, 1.0, smoothstep(0.0, 0.15 + 0.3 * rk.r.y, hg));                        // contact zone
    // (rocks3c) submerged tops lighter (r7: the blocks along the bay came out black dots; target: pale blocks / slabs
    // under the water)
    // (rocks4 R7: x2.0 -> x1.4 - the near-surface flat tops turned into uniform plates / an oil puddle in r10)
    alb *= mix(1.0, 1.4, saturate(nl.y * 0.8 + 0.2));
    // (rocks4 R7 / R3-tint) olive-brown algae film fading in over the first ~0.3 m of depth (#918a68) with a
    // 10-30 cm olive mottle +-25 %
    float dep = saturate((waterLevel(p.xz) - p.y) / 0.3);
    float om = texture(uNoise3, qs * 0.45 + 0.61).g - 0.487;
    alb *= mix(vec3(1.0), vec3(0.95, 0.9, 0.62) * (1.0 + 1.6 * om), dep * 0.8) * 1.25;
    float pkB;
    alb *= rkMidTex(qs, fp * 1.3, 0.0, pkB);          // (rocks5 M1) submerged blocks textured, not soap
    return alb;
}

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
float treeOwnTrans(vec3 p, vec3 L, Tree tr, float dc, int steps, float pass) {
    float T = 1.0;
    float t = 0.03 + 0.05 * gRand;
    vec4 inf;
    float reach = treeReach(tr.R * treeLowWiden(tr), tr.H, tr.seed) + 0.3;
    for (int i = treeZero(); i < steps; i++) {
        vec3 q = treeLocal(p + L * t, tr);
        if (q.y > tr.H + 0.4 || q.y < -0.5 || dot(q.xz, q.xz) > reach * reach) break;
        float d = sdTreeNear(q, tr, 0.03, dc, 0.0, false, inf);
        float w = 0.02 + 0.012 * t;
        if (d < w) { T *= mix(pass, 1.0, max(d, 0.0) / w); t += w; }
        else t += min(d, 0.6);
        if (T < 0.02) break;
    }
    return T;
}
// mid model: soft shadow of the whorl skirts above (each tier shades the one below it, the outer
// sun-side edge of a tier stays lit); the surface is the average spray surface, so 'res' only darkens
// to a floor (porous sprays)
float treeOwnTransMid(vec3 p, vec3 L, Tree tr, float fp) {
    float res = 1.0;
    float t = 0.10 + 0.12 * gRand;
    vec4 inf;
    float reach = treeReach(tr.R, tr.H, tr.seed) + 0.3;
    for (int i = treeZero(); i < 12; i++) {
        vec3 q = treeLocal(p + L * t, tr);
        if (q.y > tr.H + 0.4 || dot(q.xz, q.xz) > reach * reach) break;
        float d = sdTreeMid(q, tr, 1.0, fp, false, inf);
        res = min(res, saturate(3.0 * d / t));
        if (res < 0.01) break;
        t += clamp(d, 0.08, 0.8);
    }
    return res;
}

// fissured bark: returns albedo, perturbs the normal (cylindrical frame around the trunk)
vec3 barkAlbedo(vec3 q, Tree tr, float t, inout vec3 n) {
    float ang = atan(q.z, q.x);
    float rr = max(length(q.xz), 0.05);
    float fpb = t * gPixAng;                           // pixel footprint (m): detail fades once sub-pixel
    float aF = 1.0 - smoothstep(0.004, 0.02, fpb);
    // unrolled bark coordinates: one noise tile around the trunk (seamless at the wrap), ~0.95 tile per metre
    // up: spruce / fir small roundish flaking scales (worley, ~4-7 cm on a mature trunk, finer up the
    // tapering stem), larch larger plates between deep furrows
    vec3 bc = vec3(ang / TAU, q.y * mix(0.95, 0.50, tr.larch), tr.seed * 3.0);
    float plates = texture(uNoise3, bc).a;
    float fis = texture(uNoise3, bc * vec3(1.0, 0.6, 1.0) + 0.37).b;
    float fine = texture(uNoise3, bc * vec3(4.0, 4.0, 1.0) + 0.71).r;
    float edge = 1.0 - smoothstep(0.25, 0.55, plates);           // gaps between the scales
    float crack = saturate(smoothstep(0.45, 0.65, fis) * mix(0.5, 1.0, tr.larch) + 0.6 * edge * (1.0 - 0.5 * tr.larch));
    crack = mix(0.30, crack, aF);
    // spruce / fir: grey-brown scales (trunks #282621 in shade, grey on the lit side),
    // larch: red-brown plates
    // (r5 T9: plate contrast x1.5 - scaly bark reads at 1:1)
    vec3 spruceB = mix(vec3(0.033, 0.031, 0.028), vec3(0.111, 0.107, 0.097), plates);
    vec3 larchB = mix(vec3(0.056, 0.044, 0.034), vec3(0.110, 0.088, 0.068), plates);
    vec3 alb = mix(spruceB, larchB, tr.larch);
    alb *= mix(1.0, 0.30, crack) * mix(1.0, 0.55 + 0.9 * fine, aF);
    // crustose lichen: pale grey-green patches (3-15 cm), more on the shaded north side and low on the trunk;
    // dark wet / soiled foot
    float ln = texture(uNoise3, vec3(ang / TAU * 2.0, q.y * 1.4, tr.seed * 9.0)).r;
    float lich = smoothstep(0.56, 0.68, ln) * saturate(0.55 - q.z / rr * 0.5) * (1.0 - smoothstep(2.0, 0.45 * tr.H, q.y));
    alb = mix(alb, vec3(0.17, 0.18, 0.145) * mix(0.8, 1.1, fine), lich * mix(0.15, 0.40, aF) * (1.0 - 0.7 * tr.larch));
    alb *= mix(0.65, 1.0, smoothstep(0.0, 0.7, q.y));
    if (fpb < 0.03) {
        // scale relief: tilt the normal by the plate height field (around / up the stem)
        vec3 radial = vec3(q.x, 0.0, q.z) / rr;
        vec3 tang = vec3(-radial.z, 0.0, radial.x);
        const float e = 0.004;
        float hx = texture(uNoise3, bc + vec3(e, 0.0, 0.0)).a - texture(uNoise3, bc - vec3(e, 0.0, 0.0)).a;
        float hy = texture(uNoise3, bc + vec3(0.0, e, 0.0)).a - texture(uNoise3, bc - vec3(0.0, e, 0.0)).a;
        float k = 2.5 * (1.0 - smoothstep(0.008, 0.03, fpb));
        n = normalize(n - (tang * hx + vec3(0.0, hy, 0.0)) * k);
    }
    return alb;
}

vec3 shadeTree(vec3 p, vec3 rd, float t, ivec2 cell, bool cheap) {
    Tree tr;
    getTree(cell, tr);
    float lod = treeLod(t), lod2 = treeLod2(t);
    float e = max(0.006, t * gPixAng * 0.8);
    float env;
    // tetrahedral normal (4 taps, (+-1,+-1,+-1) with an odd number of +1); a loop that is not unrolled, so
    // the tree distance function is compiled once here instead of four times
    vec3 n = vec3(1e-6);
    for (int i = treeZero(); i < 4; i++) {
        vec3 kk = 2.0 * vec3(((i + 3) >> 1) & 1, (i >> 1) & 1, i & 1) - 1.0;
        n += kk * sdTree(p + kk * e, tr, lod, lod2, env);
    }
    n = normalize(n);
    sdTreeInfo(p, tr, lod, lod2, env);
    vec4 inf = gTreeInfo;
    vec3 q = treeLocal(p, tr);
    float fir = treeFir(tr);
    float th = saturate(q.y / tr.H);
    float rid = hash12i(cell + ivec2(3, 9));
    float dc = treeCamDist(tr);
    float fp = dc * uPixAng;              // true footprint: anti-aliasing fades only
    float fpL = dc * TREE_REF_PIXANG;     // look footprint (1080p reference)
    bool near = inf.w < 1.5 && treeIsNear(tr, dc);
    // crowns about a pixel wide or less: per-tree contrast (and tip glitter) fades out
    float sub = 1.0 - smoothstep(0.5, 2.0, 2.0 * tr.R / max(fp, 1e-4));
    // several crowns per pixel: their lit / shaded sides, gaps and shadows average out (no sparkle)
    // (r5 T3 / T12: from fpL 2-6, was 0.8-3: distant crowns keep their lit side / shaded side and lit tops)
    float far2 = near ? 0.0 : max(smoothstep(2.0, 6.0, fpL), sub);
    float op = treeOpen(tr);
    float r = length(q.xz);
    float renv = max(crownR(tr, q.y, fir), 0.05);
    // (trees6) per-tree variety of the mid / distant forest (sun exposure from the height above the
    // neighbours, value / hue, lit brightness, grey old trees): per-tree constants (no shimmer), ramped in by
    // camera distance from 60 to 140 m (the same on both models across the near / mid switch; the close-up
    // hero trees unchanged) and faded with the crowns' per-tree contrast once they are ~1 px (sub)
    float kV = smoothstep(60.0, 140.0, dc) * (1.0 - sub);
    float relH = treeRelH(cell, tr.dens);
    if (uDebug == 11) return vec3(inf.x / 3.0, inf.y, inf.z * 0.5 + 0.5) * (near ? 1.0 : 0.3);
    if (uDebug == 10) return vec3(tr.H / 40.0, tr.R / 5.0, tr.larch * 0.5 + fir * 0.25 + (near ? 0.2 : 0.0));
    if (uDebug == 12) return (dc > 30.0 && dc < 300.0) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 13) return (dc > 40.0 && dc < 120.0) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 14) return dc > 600.0 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 15) return dc < 60.0 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 16) return (dc >= 60.0 && dc <= 600.0) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);

    // ---- sun visibility: terrain, clouds, neighbouring crowns, this crown (analytic) ----
    float sun = terrainShadow(p, cheap ? 30 : 70) * cloudShadow(p);
    float fd = saturate(forestAt(tr.base.xz).r * 1.3 - 0.2);           // stand density
    // the side of the crown we see faces the camera: at a forest edge (open ground toward the camera)
    // that side gets the open sky, inside the stand it does not
    vec2 toCam = normalize(uCamPos.xz - tr.base.xz + 1e-3);
    float fdC = saturate(forestAt(tr.base.xz + toCam * 12.0).r * 1.3 - 0.2);
    float fdV = min(fd, mix(fdC, fd, 0.25));
    // statistical shade from the neighbours: lower crowns inside dense stands get little sun
    // (low sun: the neighbours shade all but the crown tops - golden hour)
    float sunLow = 1.0 - smoothstep(0.15, 0.60, uSunDir.y);
    // (trees6: by the height relative to the stand canopy - the dominant trees' crowns are sunlit far down,
    // the young / middle cohort stays in the canopy's shade; deeper shade low in the stand: dark gaps)
    float thE = th * mix(1.0, clamp(relH, 0.35, 1.45), 1.3 * kV);
    float lowS = 0.35 * (1.0 - 0.8 * sunLow) * mix(1.0, 0.63, kV);
    float tstat = mix(1.0, mix(lowS, 1.0, smoothstep(0.15 + 0.55 * sunLow, 0.80 + 0.15 * sunLow, thE)), mix(fd, 1.0, 0.8 * sunLow));
    float tsh = tstat;
    if (!cheap && t < 600.0 && sun > 0.0) {
        gTreeSkip = cell;
        float ts = treeShadow(p + n * 0.12);
        gTreeSkip = ivec2(-1000000);
        tsh = mix(ts, tstat, smoothstep(350.0, 600.0, t));
    }
    // sun flecks: the neighbours' crowns are porous, light falls through their gaps in patches
    // (30 cm patches: replaced by their mean 0.19 before they are sub-pixel)
    float dap = mix(smoothstep(0.50, 0.78, texture(uNoise3, p * 0.21 + 0.3).g), 0.19, smoothstep(0.08, 0.25, fp));
    tsh = max(tsh, dap * mix(0.75, 0.35, fd) * smoothstep(0.1, 0.6, th + 0.3));
    // (r5 T6: larch .30, was .45 - thin crowns, lit through)
    float sig = mix(mix(1.00, 0.70, op), 0.30, tr.larch);              // foliage extinction (1/m)
    // (the mid model's surface is the average spray surface, ~15 % inside the reach of the outer
    // sprays: their sun path is taken at the spray tips like the near model's)
    vec3 qS = q;
    if (inf.w > 1.5) qS.xz *= 1.0 / (0.85 * treeMidSlim(tr, dc));
    float Tself = exp(-sig * crownSunPath(qS, tr, fir));
    // near model: march the tree's own (coarse) branches toward the sun and toward the sky above
    // (layered self-shadowing / sky occlusion of the spray layers); the analytic cone path keeps a
    // small share for the unresolved needles and the branches the coarse model leaves out
    float Tsky = 1.0;
    bool marchK = near && !cheap;
    if (marchK) {
        // (r4 b1: .38 / .48 left more than half the crown pixels without sun - a 3-5 cm spray is crossed
        // in 2-3 march steps, so the effective layer transmittance was ~.38^2)
        // (r5 T3: .65 - more lit sprays inside the near crowns, target p95 ~190)
        // (r5 b4: .55 - with the higher near sun gain the r6 / r2 crowns keep more dark interior, pct 20/42/74/121/187)
        float pass = 0.55;
        float Ts = sun * tsh > 0.0 ? treeOwnTrans(p + n * 0.01, uSunDir, tr, dc, 18, pass) : 0.0;
        Tself = Ts * exp(-sig * 0.30 * crownSunPath(qS, tr, fir));
        vec3 rad = normalize(vec3(q.x, 0.0, q.z) + 1e-4);
        Tsky = treeOwnTrans(p + n * 0.01, normalize(vec3(rad.x * 0.6, 1.0, rad.z * 0.6)), tr, dc, 12, 0.65);
    }

    if (!near && !cheap && inf.w > 1.5) {
        float mK = 1.0 - smoothstep(0.8, 1.6, fpL);
        if (mK > 0.0) {
            float Tm = treeOwnTransMid(p + n * 0.02, uSunDir, tr, fp);
            Tself *= mix(1.0, mix(0.22, 1.0, Tm), mK);
        }
    }

    // twigs: bare inner part of the branchlets (close up only, like the geometry)
    float twig = 0.0;
    if (near && inf.x < 0.5) {
        float bare = 0.25 * (1.0 - smoothstep(0.006, 0.02, fp)) * (1.0 - 0.6 * inf.y);
        twig = 1.0 - smoothstep(bare - 0.04, bare + 0.06, inf.z);
    }
    if ((inf.x > 0.5 && inf.x < 2.5) || twig > 0.5) {
        // ---- trunk / branch wood ----
        vec3 alb;
        // (r4: x1.5 and warmer - shot 3 trunks measured 29,28,24 / 66,64,45 vs target 45,39,25 / 81,72,53)
        // (r5 T9: spruce / fir grey-brown x(1.45,1.35,1.10), r/g ~1.12 lit (target 81,72,53); the r4 x(1.85,1.55)
        // read as orange pine poles; larch keeps its red-brown)
        // (r5 b3: x(1.40,1.26,1.04) and less lichen - the b2 r6 trunks read as flat neutral-grey poles (80,80,72),
        // target trunk cores (34,25,14)-(53,42,33), lit edges (114,104,89))
        if (inf.x > 0.5 && inf.x < 1.5) alb = barkAlbedo(q, tr, t, n) * mix(vec3(1.40, 1.26, 1.04), vec3(1.85, 1.55, 1.10), tr.larch);
        else if (inf.x > 1.5 && inf.z > 0.5) {
            // dead lower branches: silver-grey wood with grey-green beard / crust lichen
            // (r5 T2: >= .12 grey, was .085 - black whips in shade)
            float lc = texture(uNoise3, q * 2.3 + tr.seed * 4.0).g;
            alb = mix(vec3(0.12, 0.12, 0.108), vec3(0.17, 0.172, 0.15), lc) * mix(0.85, 1.1, rid);
        }
        else alb = mix(vec3(0.060, 0.058, 0.052), vec3(0.090, 0.086, 0.076), rid);  // grey-brown twigs (r5 T11: was .028-.05)
        float ao = mix(0.30, 0.8, smoothstep(0.0, 0.5, th)) * terrainAO(tr.base.xz) * mix(0.75, 1.0, saturate(q.y * 0.5));
        ao *= mix(0.55, 1.0, 1.0 - tr.dens * 0.5) * mix(0.70, 1.0, saturate(r / renv)) * 1.6;
        // (r5 T9: near trunks keep readable bark in shade, ~(40,40,35) - not a featureless black cylinder)
        if (near) ao = max(ao, 0.42);
        return litSurface(alb, n, rd, sun * tsh * Tself, ao, 0.0, 0.8);
    }

    // ---- foliage ----
    // species colour (linear albedo; green, red:green ~0.65), slight per-tree variation
    vec3 spruce = mix(vec3(0.072, 0.092, 0.034), vec3(0.088, 0.110, 0.038), rid);
    vec3 firC = mix(vec3(0.062, 0.080, 0.034), vec3(0.074, 0.095, 0.040), rid);
    // (larch: muted yellow-green, lit #8c965b, never lime)
    // (r5 T6: ~1.5x the spruce albedo, h ~117 - the r4 larch was the spruce colour and never read as larch)
    vec3 larchC = mix(vec3(0.115, 0.140, 0.045), vec3(0.130, 0.155, 0.050), rid);
    vec3 alb = mix(mix(spruce, firC, fir), larchC, tr.larch);
    // (r4 b4: near crowns were ~.6 stop too dark in the mid-tones; x1.3 within 30 m, fading
    // out by 80 m so the mid model's level is unchanged)
    alb *= mix(1.3, 1.0, smoothstep(30.0, 80.0, t));
    // (close up spruce foliage is olive: red ~ green in sRGB)
    alb.r *= mix(1.10, 1.0, max(smoothstep(20.0, 45.0, t), tr.larch));
    float rv = hash12i(cell + ivec2(17, 5));
    alb *= mix(1.0, mix(0.70, 1.20, rv), (1.0 - sub) * (1.0 - 0.35 * smoothstep(0.8, 3.0, fpL)));
    // (r5 T5: per-crown hue - yellow-olive and blue-green crowns side by side, forest G-R std >= 4.3)
    float rh = hash12i(cell + ivec2(29, 61));
    alb *= mix(vec3(1.0), mix(vec3(1.07, 1.0, 0.86), vec3(0.93, 1.0, 1.10), rh), (1.0 - sub));
    if (kV > 0.0) {
        // (trees6) dark blue-green old spruce / fir beside lighter yellow-olive crowns (value and hue go
        // together), a few grey old / dying trees (thin needles, lichen)
        float v2 = hash12i(cell + ivec2(67, 23));
        float bl = saturate(1.25 * (1.0 - rv) + 0.5 * (v2 - 0.5)) * (1.0 - tr.larch);
        alb *= mix(vec3(1.0), mix(vec3(1.05, 1.02, 0.84), vec3(0.86, 0.96, 1.24), bl) * mix(0.82, 1.14, v2), kV * 1.4);
        float gr = step(hash12i(cell + ivec2(41, 77)), 0.035);
        alb = mix(alb, vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722))) * vec3(1.06, 1.0, 0.90) * 1.05, 0.55 * gr * kV);
    }
    // stand-scale patches (age, vigour, exposure): lighter olive and darker blue-green parts of the forest
    float pv = smoothstep(0.30, 0.70, textureLod(uNoise3, vec3(tr.base.xz * (1.0 / 420.0), 0.61), 0.0).g);
    alb *= mix(vec3(0.80, 0.80, 0.84), vec3(1.20, 1.15, 1.02), pv);
    vec3 radial = normalize(vec3(q.x, 0.0, q.z) + 1e-4);
    vec3 nc = normalize(radial + vec3(0.0, mix(0.30, 0.5, fir), 0.0));    // crown (cone) normal
    float depth = max(renv - r, 0.0);                                    // metres inside the envelope
    float aoC, lit;
    float farF = 0.0;
    float far = 0.0;
    // clump-scale normal: gradient of the tree distance over ~0.15-0.45 m (the normal of the foliage mass
    // of a branch / spray clump, not of a single shoot): each clump is lit on its sun side
    // and dark on the other, the crown reads as 3D masses instead of an evenly lit fuzz
    vec3 nCl = nc;
    float wCl = near && inf.x < 0.5 ? 0.9 * smoothstep(4.0, 20.0, t) : 0.0;
    if (wCl > 0.0 && !cheap) {
        float e2 = clamp(0.18 + 0.006 * t, 0.18, 0.75);
        vec3 m = vec3(0.0);
        for (int i = treeZero(); i < 4; i++) {
            vec3 kk = 2.0 * vec3(((i + 3) >> 1) & 1, (i >> 1) & 1, i & 1) - 1.0;
            m += kk * sdTree(p + kk * e2, tr, lod, lod2, env);
        }
        nCl = normalize(m + nc * 1e-3);
    }
    if (near) {
        // inf.y along the branch (0 inner .. 1 tip), inf.z along the branchlet, inf.w side
        float fx = inf.y, fs = inf.z;
        float up = inf.w;
        float bt = saturate(abs(up) * 2.0 - 1.0);     // per-branchlet tone
        up = up >= 0.0 ? 1.0 : -1.0;
        bool core = inf.x > 2.5;
        // fresh light growth at the branchlet tips (more on the outer part of the branch)
        // in patches (this year's shoots on the vigorous outer sprays), mostly on the upper side
        float tn = texture(uNoise3, q * 0.8 + tr.seed * 5.0).g;
        // (r5 T3: wider noise gate .30-.60, was .55-.80 - more of the outer sprays carry fresh tips)
        float tip = smoothstep(0.35, 1.0, fs) * smoothstep(0.35, 0.95, fx) * smoothstep(0.30, 0.60, tn) * mix(0.35, 1.0, step(0.0, up));
        // (r4: saturated yellow-green fresh tips, the old x(1.45,1.55,1.35) read as pale cream "moss")
        // (r5 T3: a little warmer, x(1.55,1.48,0.95): lit r/g ~0.95, still saturated)
        // (r5 b5: (1.50,1.44,1.08) - the (1.55,1.48,.95) tips read as lime streaks in shot03; b/g still < sprays)
        alb = mix(alb, alb * vec3(1.50, 1.44, 1.12), tip * (1.0 - 0.7 * tr.larch));
        // needle-scale self shadowing (faded out once it is sub-pixel)
        float ns = texture(uNoise3, q * 6.0 + tr.seed * 3.0).b;
        float nA = (1.0 - smoothstep(0.006, 0.02, fp)) * 0.5;
        float speck = mix(1.0, mix(0.35, 1.25, smoothstep(0.25, 0.75, ns)), nA);
        // sky occlusion: deep inside the crown, under a frond, close to the axis of a branchlet
        // the underside of a spray gets no direct sun (only what leaks through the needles)
        // (the crowns are mostly mid-tones: sky reaches well into the porous sprays; x0.92 beyond
        // 40 m matches the mid model's brightness at the switch)
        aoC = exp(-depth * (marchK ? 0.06 : 0.35)) * mix(0.43, 1.0, Tsky) * mix(0.70, 1.0, step(0.0, up)) * mix(0.80, 1.0, fs) * mix(0.68, 1.0, fx) * speck;
        lit = mix(0.22, 1.0, step(0.0, up)) * mix(0.63, 1.0, fx) * speck * mix(1.0, 0.92, smoothstep(35.0, 50.0, dc));
        if (core) {
            // dense inner crown: old needles in deep shade, a little light through the gaps
            float cn = texture(uNoise3, q * 1.3 + tr.seed * 2.0).r;
            aoC = mix(0.35, 0.60, cn) * exp(-depth * 0.4) * 1.5;
            lit = 0.15 * cn;
            n = nc;
        } else {
            // shoots are fuzzy needle brushes: their shading follows the spray (top lit, underside
            // dark) more than the round branchlet cylinder
            n = normalize(n * 0.55 + nc * 0.25 + vec3(0.0, 0.35 * up, 0.0));
            n = normalize(mix(n, normalize(nCl + nc * 0.25 + vec3(0.0, 0.15, 0.0)), wCl));
        }
        // (the deep interior seen through the gaps stays near-black; sprays keep 0.45)
        if (!core) { lit *= mix(0.62, 1.30, bt); aoC *= mix(0.80, 1.16, bt); alb *= mix(0.90, 1.10, bt); }
        if (!core) {
            // needle tufts: the lit spray tops break into small lit tufts and shaded needles between
            // them (~4 cm), replaced by their mean before they are sub-pixel
            float tf = texture(uNoise3, q * 1.6 + tr.seed * 13.0).b;
            float tfA = (1.0 - smoothstep(0.004, 0.014, fp)) * 0.5;
            lit *= mix(1.0, mix(0.35, 1.45, smoothstep(0.30, 0.70, tf)), tfA);
            // shoot clusters (~20 cm): the hanging branchlets are not evenly lit ropes; kept until the
            // clusters are ~4 px
            float tf2 = texture(uNoise3, q * 0.55 + tr.seed * 7.0).b;
            float tf2A = (1.0 - smoothstep(0.025, 0.06, fp)) * 0.4;
            lit *= mix(1.0, mix(0.40, 1.30, smoothstep(0.30, 0.70, tf2)), tf2A);
        }
        if (!core) {
            // shadows of the sprays above: the lit spray tops break into lit and shaded patches
            // (0.5-0.8 m), faded to their mean before they are ~6 px
            float sm = texture(uNoise3, q * vec3(0.10, 0.14, 0.10) + tr.seed * 6.0).r;
            float smA = (1.0 - smoothstep(0.05, 0.14, fp)) * (1.0 - 0.5 * tr.larch) * (marchK ? 0.0 : 1.0);
            // (mean ~0.93: the near model's lit level stays that of the mid model at the switch)
            lit *= mix(1.0, mix(0.42, 1.40, smoothstep(0.30, 0.70, sm)), smA);
        }
        aoC = max(aoC, marchK ? 0.10 : (core ? 0.14 : 0.38));
        lit *= mix(0.86, 1.0, smoothstep(20.0, 45.0, dc));
        // larch: open, thin-needled crown - sky reaches far into it
        aoC = mix(aoC, max(aoC, 0.65), 0.6 * tr.larch);
    } else {
        // mid / far: inf.y radial r/renv, inf.z position in the skirt (+1 top .. -1 lower edge)
        far = smoothstep(0.25, 1.2, fpL);
        float outer = smoothstep(0.35, 0.95, inf.y);
        float under = saturate(saturate(-inf.z) * (1.0 - far) * 2.5);
        // branch-scale clumps: lit spray tops / dark holes (faded out once they are sub-pixel)
        float cl = texture(uNoise3, q * 0.30 + tr.seed * 7.0).b;
        float clA = (1.0 - smoothstep(0.45, 0.90, fpL)) * (1.0 - smoothstep(0.25, 0.6, fp));
        float clump = mix(0.6, smoothstep(0.25, 0.75, cl), clA);
        // spray clumps vs gaps between them (mean ~1, like the near model's lit tips and dark holes)
        float cK = mix(1.0, mix(0.25, 1.50, smoothstep(0.25, 0.75, cl)), clA);
        // coarser clumps (1.5-3 m) once the fine ones are gone: the lit flank of a distant crown is broken
        // into lit branch clumps and dark holes, not a smooth cone (faded before they are ~2 px)
        float clB = (1.0 - clA) * (1.0 - smoothstep(0.45, 0.95, fpL)) * (1.0 - smoothstep(0.45, 0.95, fp));
        if (clB > 0.0) {
            float c2 = texture(uNoise3, q * 0.11 + tr.seed * 3.0).b;
            cK *= mix(1.0, mix(0.55, 1.30, smoothstep(0.3, 0.7, c2)), clB);
        }
        alb = mix(alb, alb * vec3(1.32, 1.30, 1.26), clump * outer * (1.0 - far) * 0.3 * (1.0 - sub));
        // (x0.86 / x0.92 near the near/mid switch: the near model shows its dark interior through the
        // gaps between the sprays, so its average sky access and lit fraction are lower)
        // (r4: darker undersides of the whorl skirts, 2x: tiers read at 60-300 m)
        const float uK = 2.0;
        aoC = mix(0.8, 1.0, outer) * mix(1.0, saturate(1.0 - 0.2 * uK), under) * mix(1.0, cK, 0.7) * 0.93 * mix(0.85, 1.0, far) * mix(0.86, 1.0, far);
        // porous crown seen from afar: part of the lit side is shadowed gaps (needle clumping)
        lit = mix(0.9, 1.0, outer) * mix(1.0, saturate(1.0 - 0.45 * uK), under) * cK * 0.93 * mix(0.85, 0.9, far) * mix(0.92, 1.0, far);
        // (r4: the crown interior seen between the outer clumps is dark, spires at 40-300 m)
        aoC *= mix(0.55, 1.0, outer * (1.0 - far));
        lit *= mix(0.775, 1.0, outer * (1.0 - far));
        // (r4 b3: the shaded flank of a 60-300 m spire is near-black: less sky fill)
        aoC *= mix(0.6, 1.0, far);
        // (r5 b4: lit mid / far crown flanks, slope forest p95 142 / 149)
        lit *= 1.55;
        aoC *= 1.25;
        if (uDebug == 36) return vec3(outer, under, clump);
        // the visible surfaces are mostly spray tops (like the near model's): normals tilt up
        n = normalize(n * mix(0.65, 0.15, far) + nc * mix(0.35, 0.85, far) + vec3(0.0, 0.30 * (1.0 - far), 0.0));
        // distant crowns: shade dominates the mix, the skylit shade is blue-green
        farF = far;
    }
    // distant forest (by distance, not footprint): unresolved skylit shade and needle sheen make it
    // grey / blue-green, not yellow-green
    float distF = smoothstep(200.0, 2000.0, t);
    // beyond a few tens of metres a pixel mixes lit sprays with skylit gaps: lit flanks read green /
    // blue-green rather than yellow-olive, and the shade picks up more sky (same for both models)
    float midF = smoothstep(15.0, 150.0, t);
    alb *= mix(vec3(1.0), vec3(0.97, 1.0, 1.04), midF);
    alb *= mix(1.0, 0.85, smoothstep(30.0, 120.0, t) * (1.0 - smoothstep(600.0, 1500.0, t)));
    // (larch at range: soft light yellow-green, never lime)
    // (trees6: x(1.10,1.15,1.00) beyond ~140 m, was (1.15,1.15,.95) - distant larch read lime, lit larch #8c965b)
    alb *= mix(vec3(1.0), mix(vec3(1.15, 1.15, 0.95), vec3(1.10, 1.15, 1.00), kV), tr.larch * smoothstep(20.0, 120.0, t));
    // (the grey / blue cast of distant forest is its unresolved sky-lit shade and sheen: the sunlit needles
    // keep their own colour - golden hour: saturated yellow-green lit flanks, not cream)
    vec3 albS = alb;
    if (kV > 0.0) {
        // (trees6) aerial perspective on the sunlit flanks too (not at golden hour): the distant lit crowns
        // greyer / bluer, not the 200 m yellow-green
        float dsK = 0.15 * distF * (1.0 - smoothstep(0.4, 0.6, sunLow)) * kV;
        albS = mix(albS, vec3(dot(albS, vec3(0.2126, 0.7152, 0.0722))) * vec3(0.96, 1.0, 1.10), dsK);
    }
    alb = mix(alb, vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722))), 0.22 * distF);
    // (the canopy seen from afar shows mostly its sunlit, sky-facing upper crowns: lighter)
    // (r5 T13: (1,1,1.04), was (0.97,1,1.15): distant shade green, b <= g - 3, not blue-grey)
    alb *= mix(vec3(1.0), vec3(1.0, 1.0, 1.04), distF);
    // stand-level sky occlusion (lower crowns inside dense stands)
    // (low sun: the low sky behind the camera is dim, the far stands' lower crowns get little sky light)
    aoC *= mix(1.0, mix(mix(0.50, 0.36, kV), 1.0, smoothstep(0.05, 0.75, thE)), max(fdV, 0.6 * sunLow * smoothstep(150.0, 500.0, t)));
    // (crowns at 20-300 m are mostly mid-tones: more sky light into the shade)
    aoC *= mix(1.0, mix(1.30, 1.0, smoothstep(150.0, 400.0, t)), smoothstep(10.0, 40.0, t));
    { float mr = smoothstep(20.0, 60.0, t) * (1.0 - smoothstep(300.0, 600.0, t)); aoC *= 1.0 + 0.18 * mr; lit *= 1.0 + 0.12 * mr; }
    aoC *= mix(1.0, 0.6, sunLow * smoothstep(300.0, 1000.0, t));
    float aoT = mix(terrainAO(tr.base.xz), 1.0, th * 0.6);
    float sunV = sun * mix(tsh, tstat, far2) * mix(Tself, mix(0.6, 0.35, sunLow), far2);
    // (r5 T6: larch x1.0, was .86 - light larch crowns)
    sunV *= mix(1.0, mix(0.76, 1.0, tr.larch), smoothstep(25.0, 180.0, t));
    // (trees6) per-tree lit brightness (needle density, crown orientation): not the same lit tip on every tree
    sunV *= 1.0 + 0.30 * (2.0 * hash12i(cell + ivec2(53, 13)) - 1.0) * kV;
    // (r5 T12: low sun lights the crown tops - the far shore's gold-topped trees over dark lower crowns)
    float gold = smoothstep(0.4, 0.6, sunLow);
    sunV *= mix(1.0, 1.4, smoothstep(0.70, 0.95, th) * gold);
    // forest seen from afar follows the terrain relief: slopes facing the sun are lit more
    if (far > 0.0) {
        vec3 nT = terrainNormal(tr.base.xz, 6.0, 0.0);
        float relief = saturate(0.30 + 0.85 * dot(nT, uSunDir)) / max(0.30 + 0.85 * uSunDir.y, 0.3);
        sunV *= mix(1.0, 0.45 + 0.55 * relief, far);
    }
    if (uDebug == 7) return vec3(sun, tsh, Tself);
    if (uDebug == 9) return vec3(crownSunPath(q, tr, fir) / 3.0, r / renv * 0.5, q.y / tr.H);
    if (uDebug == 8) return vec3(aoC, lit, near ? 1.0 : 0.0);
    float ndl = dot(n, uSunDir);
    // (mid model: the shaded side of a crown is mostly gaps into the dark interior -> less wrap)
    float wrapD = near ? saturate(ndl * 0.85 + 0.15) : saturate(ndl * mix(0.85, 0.65, far) + mix(0.22, 0.15, far));
    // mid model near the switch: its smooth skirt faces the camera and the sun together, the near model's
    // sprays point every way -> flatten its lit / shaded contrast to the near model's
    // (r4 b3: only 40 % of that flattening - the mid spires need their dark shaded flank)
    if (!near) wrapD = mix(mix(wrapD, 0.10 + 0.75 * wrapD, 0.4), wrapD, farF);
    if (!near) wrapD = mix(wrapD, saturate(ndl * 0.95 + 0.05), 0.6 * smoothstep(100.0, 250.0, t) * (1.0 - far2));
    wrapD = mix(wrapD, 0.65, 0.6 * far2);
    lit = mix(lit, 0.8, 0.5 * far2);
    aoC = mix(aoC, 0.9, 0.5 * far2);
    // sky light inside the crown has passed through needles: green, little blue
    // (r5 T13: distF (0.95,1,1.06), was (0.90,1,1.20): green, not blue-grey distant shade)
    vec3 skyT = mix(vec3(0.98, 1.0, 0.95), vec3(0.95, 1.0, 1.06), distF) * mix(vec3(1.0), vec3(1.0, 1.0, 0.92), (1.0 - aoC) * (1.0 - distF));
    skyT *= mix(vec3(1.0), vec3(1.12, 1.12, 1.04), midF);
    // (r4: the shaded interior of near crowns is olive-brown, r >= g: old needles, twigs, bark)
    // (r5 b3 / b4: (1.22,1.05,1.0), was (1.30,1.05,.70) - b2 near darks (16,18,6) vs target (11,16,13) / (16,14,7))
    skyT *= mix(vec3(1.0), vec3(1.22, 1.05, 1.0), saturate(1.0 - aoC) * (1.0 - midF));
    // sunlit needles: diffuse (olive) plus a broad, nearly white waxy sheen that greys the lit flanks
    // (r4: x1 close up and at low sun - x2 creamed the near tips and the golden-hour forest; x2 at range at
    // midday: paler lit flanks)
    // (r5 T3: near sun gain 1.45 fading to 1.20 by 80 m - near crowns p95 ~190)
    // (b4: near 1.90, was 1.45 - shot03 lit sprays p95 148 vs target 189, lit band 7 % vs 17 %)
    vec3 col = albS * uSunCol * sunV * wrapD * lit * mix(1.90, 1.20, smoothstep(20.0, 80.0, t));
    // (r5 b3: near sunlit sprays less yellow - target lit10 (180,178,135) / (168,176,136), b2 (167,177,122):
    // r/g ~1.0, b/g ~.76; the needles' gloss greys their sunlit side)
    // (b4: mid range too - target lit flanks b/g ~.79-.83, b3 r4 / shot02 ~.62; faded out by 250-600 m)
    // (trees6: beyond 600 m keep a green / blue-green lit flank (midday), not the yellow-lime of the bare albedo:
    // aerial perspective goes the other way, lit spruce r/g ~0.89, b/g ~0.70)
    vec3 farTint = mix(vec3(1.0), vec3(0.91, 1.0, 1.32), kV * (1.0 - smoothstep(0.4, 0.6, sunLow)));
    col *= mix(mix(vec3(1.08, 1.0, 1.70), vec3(1.03, 1.0, 1.25), smoothstep(30.0, 90.0, t)), farTint, smoothstep(250.0, 600.0, t));
    vec3 colA = alb * skyLightFoliage(n, aoT) * aoC * skyT;
    // (r5 T13: sheen (.010,.011,.009) x <= 1, was (.016,.017,.018) x 1.4: frosted blue-grey crowns at 50-150 m)
    float sheenK = mix(1.0, saturate(1.4 * saturate(1.0 - 1.6 * sunLow)), smoothstep(40.0, 150.0, t));
    col += uSunCol * sunV * wrapD * lit * vec3(0.010, 0.011, 0.009) * sheenK * mix(0.35, 1.0, smoothstep(15.0, 90.0, t));
    if (uDebug == 30 || uDebug == 33) return vec3(sunV, wrapD, lit);
    if (uDebug == 37) return vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722)) * 10.0, dot(skyLight(n, aoT), vec3(0.2126, 0.7152, 0.0722)), wrapD);
    if (uDebug == 38) return vec3(sunV, aoC, lit);
    if (uDebug == 31 || uDebug == 34) return vec3(aoC, Tself, tsh);
    if (uDebug == 32 || uDebug == 35) return vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722)) * 10.0, dot(skyLight(n, aoT), vec3(0.2126, 0.7152, 0.0722)), n.y * 0.5 + 0.5);
    // diffuse transmission through sunlit needles (larch is thin and glows more)
    col += albS * uSunCol * sunV * saturate(-ndl * 0.6 + 0.2) * mix(0.22, 0.45, tr.larch) * 1.3;
    // needle translucency when backlit: thin outer shoots / silhouette rims glow
    float bl = pow(saturate(dot(rd, uSunDir)), 2.5);
    float rim = pow(1.0 - abs(dot(n, rd)), 2.0);
    // (r5 T3 / T12: near x1.6, was 1.3; x2 at low sun - gold rim light on the sun-side sprays)
    // (r5 b5: (.95,1.12,.58), was (.9,1.15,.45) - backlit sprays glowed lime in shot03)
    col += uSunCol * sunV * bl * albS * vec3(0.95, 1.12, 0.58) * (0.15 + 0.85 * rim) * mix(0.8, 0.2, farF) * mix(1.3, 1.6, near ? 1.0 : 0.0) * mix(1.0, 2.0, gold);
    // waxy needle sheen
    vec3 hv = normalize(uSunDir - rd);
    col += uSunCol * sunV * pow(max(dot(n, hv), 0.0), 16.0) * 0.008 * lit;
    // distant forest: the forested slopes are dark (forest / water luminance 0.25-0.35)
    // (r5 T3: on the sky-lit part only - the sunlit crowns keep their lit side, slope forest p95 ~140)
    float dD = mix(1.0, 0.72, smoothstep(120.0, 500.0, t)) * mix(1.0, 0.80, smoothstep(600.0, 1800.0, t));
    return col + colA * dD;
}

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
Hit trace(vec3 ro, vec3 rd, float tmin, float tmax, bool withWater, bool withTrees, bool withRocks, int maxSteps, float pixScale) {
    Hit h; h.t = tmax; h.kind = K_SKY; h.cell = ivec2(0); h.info = vec4(0.0);
    float t = tmin, tPrev = tmin;
    bool escaped = false;
    for (int i = 0; i < maxSteps; i++) {
        vec3 p = ro + rd * t;
        if (p.y > MAX_TERRAIN_H && rd.y >= 0.0) { escaped = true; break; }
        float det = detailWeight(t);
        float th = terrainHD(p.xz, det);
        float wl = withWater ? waterLevel(p.xz) : -1e4;
        float dyT = p.y - th;            // height above the ground / lake bed
        float dy = p.y - max(th, wl);    // height above the visible surface
        if (dy < 0.0) {
            float a = tPrev, b = t;
            for (int k = 0; k < 9; k++) {
                float m = 0.5 * (a + b);
                vec3 q = ro + rd * m;
                float s = max(terrainHD(q.xz, det), withWater ? waterLevel(q.xz) : -1e4);
                if (q.y < s) b = m; else a = m;
            }
            h.t = b;
            vec3 q = ro + rd * b;
            h.kind = (withWater && waterLevel(q.xz) > terrainHD(q.xz, det)) ? K_WATER : K_TERRAIN;
            return h;
        }
        float eps = max(0.002, (t + gTraceT0) * gPixAng * pixScale);
        float d = dyT * 0.42;
        // the lake surface is a plane: step exactly onto it (no creeping along grazing rays)
        if (wl > -1000.0 && wl > th) d = rd.y < -1e-5 ? min(d, (p.y - wl) / -rd.y + 0.01) : d;
        if (withTrees && dyT < 36.0 && t < TREE_MAX_DIST) {
            // (perf4) in tree-free space (trees.glsl treeFreeDist) treesSDF would only return its "no tree near" cap
            float tf = treeFreeDist(p);
            if (tf > TREE_CELL * 0.55 + length(p - uCamPos) * 0.0012) d = min(d, tf);
            else {
            ivec2 c;
            gTreeDMax = max(d, eps);
            float dt = treesSDF(p, treeLod(t), treeLod2(t), c);
            if (dt < eps && c.x >= 0) { h.t = t; h.kind = K_TREE; h.cell = c; return h; }
            d = min(d, dt);
            }
        }
        if (withRocks && gPrimary && dyT < COVER_MAX_H && t < COVER_MAX_DIST) {
            vec4 ci;
            float dc = coverSDF(p, t, ci);
            if (dc < eps) { h.t = t; h.kind = K_COVER; h.info = ci; return h; }
            d = min(d, dc);
        }
        if (withRocks && uHeroN > 0 && t < 600.0) {
            int hi;
            float dh = heroSDF(p, hi);
            if (dh < eps && hi >= 0) { h.t = t; h.kind = K_HERO; h.cell = ivec2(hi, 0); return h; }
            d = min(d, max(dh, 0.0) + (hi < 0 ? 1e9 : 0.0));
        }
        if (withRocks && dyT < 4.5 && t < ROCK_MAX_DIST) {
            ivec2 c;
            float dr = rocksSDF(p, rockLod(t), c);
            if (dr < min(eps, ROCK_CELL * 0.5 - 0.01)) { h.t = t; h.kind = K_ROCK; h.cell = c; return h; }
            d = min(d, dr);
        }
        tPrev = t;
        t += max(d, eps * 0.8);
        if (t > tmax) { escaped = true; break; }
    }
    // out of steps while still close to the ground: treat as an (unresolved) terrain hit, not sky
    if (!escaped) {
        vec3 q = ro + rd * t;
        if (q.y < terrainH(q.xz, false) + 60.0) { h.t = t; h.kind = K_TERRAIN; }
    }
    return h;
}

vec3 shadeHit(Hit h, vec3 ro, vec3 rd, bool cheap) {
    vec3 p = ro + rd * h.t;
    if (h.kind == K_TREE) return shadeTree(p, rd, h.t, h.cell, cheap);
    if (h.kind == K_ROCK) return shadeRock(p, rd, h.t, h.cell, cheap);
    if (h.kind == K_COVER) return shadeCover(p, rd, h.t, h.info);
    if (h.kind == K_HERO) return shadeHeroRock(p, rd, h.t, h.cell.x, cheap);   // (r3: rocks3 boulder material, shade_rocks.glsl)
    return shadeTerrain(p, rd, h.t, cheap);
}

// ---------------------------------------------------------------------------
// water_shade.glsl : lake surface shading: reflection, refraction, bed, scattering, glints (needs trace/shadeHit)
// ---------------------------------------------------------------------------
vec2 gDbgA = vec2(0.0);     // falls lab debug 66 / 67 (stream masks)
float gWWFoot = 0.0;        // (review rv8) footprint of the 3D white-water hit (lateral / along the view on the surface)
vec3 gDbgB = vec3(0.0);
vec3 shadeWater(vec3 ro, vec3 rd, float t) {
    if (uDebug >= 13 && uDebug <= 15) return vec3(0.0);          // falls lab mask renders (tools/lv.py --stats)
    vec3 p = ro + rd * t;
    float level = waterLevel(p.xz);
    p.y = level;
    float foot = t * gPixAng / max(abs(rd.y), 0.02);
    float gust, sv;
    // anisotropic footprint for the wave normals (water.glsl waterNormalA)
    vec2 fwdH = normalize(rd.xz + vec2(1e-6, 0.0));
    float fA = t * gPixAng;
    float fL = fA / max(abs(rd.y), uTune[8]);
    vec3 n = waterNormalA(p.xz, fwdH, fA, fL, gust, sv);
    // flowing stream: flow-map advected turbulence and whitewater (fading out where it meets the lake)
    float foamW = 0.0, foamT = 1.0;
    vec3 nW = vec3(0.0, 1.0, 0.0); float wW = 0.0;     // stream: white-water relief normal and its weight
    gStreamHead = 0.0;
    float headE = 1.0;            // (r4) radial position on the flat stream head (0 top .. 1 waterline)
    gWWHit = 0.0; gWWVis = 0.0; gWWOcc = 1.0; gWWSh = 1.0; gWWRock = 0.0; gWWRib = 0.0;
    vec4 st = streamAt(p.xz);
    // (review r2, fix 9) beyond the near map (z > ~1800 m, nearBlend > 0) the stream's flow / white water
    // come from its polyline (the 4 m far map has no usable stream data there): steep reaches white
    float nbS = uStreamN > 1 ? nearBlend(p.xz) : 0.0;
    if (nbS > 0.0) {
        vec2 sdF = streamDist(p.xz);
        float wfm = 1.0 - smoothstep(0.50 * sdF.y, 0.75 * sdF.y, sdF.x);
        st = mix(st, vec4(gStreamDir.xy, 0.95 * smoothstep(0.06, 0.14, gStreamDir.z), wfm), nbS);
    }
    float mf = 0.0;
    if (st.w > 0.02 && !wfOff(3)) {
        // (round 2) the stream mask is binary on the 1 m grid: its bilinear 0.3-0.9 contour cut the white
        // water off in 1 m stair steps along the banks. The mask now only has to be > 0 (its contour lies
        // under the banks, 1.2-1.9 x the half width), and the white thins out over the last ~12 cm of
        // depth at the true (smooth) water's edge instead
        float edgeD = level - terrainH(p.xz, true);
        mf = smoothstep(0.02, 0.15, st.w) * smoothstep(0.02, 0.30, level) * smoothstep(0.0, 0.12, edgeD);
        vec2 fd = normalize(st.xy + 1e-5);
        float el = 0.5;
        float wsl = (waterLevel(p.xz - fd * el) - waterLevel(p.xz + fd * el)) / (2.0 * el);
        // (round 3) the pixel footprint on the SLOPING stream surface: the cascade climbs away from a low camera
        // (film shot 4: incidence ~15-20 deg, not the ~3 deg of the horizontal lake plane, which faded every
        // stream detail into a flat white slide)
        float ddS = max(-(rd.y + wsl * dot(fd, rd.xz)), 0.02);
        float footW = t * gPixAng / ddS;
        float spd = 1.3 + 4.0 * st.z;
        float ph0 = fract(uTime * 0.4), ph1 = fract(uTime * 0.4 + 0.5);
        float wb = abs(ph0 - 0.5) * 2.0;
        vec2 q0 = p.xz - fd * spd * ph0 * 2.5, q1 = p.xz - fd * spd * ph1 * 2.5 + 3.7;
        float e = 0.06;
        vec2 g0 = vec2(texture(uNoise3, vec3((q0 + vec2(e, 0.0)) * 0.35, 0.3)).r - texture(uNoise3, vec3(q0 * 0.35, 0.3)).r,
                       texture(uNoise3, vec3((q0 + vec2(0.0, e)) * 0.35, 0.3)).r - texture(uNoise3, vec3(q0 * 0.35, 0.3)).r) / e;
        vec2 g1 = vec2(texture(uNoise3, vec3((q1 + vec2(e, 0.0)) * 0.35, 0.6)).r - texture(uNoise3, vec3(q1 * 0.35, 0.6)).r,
                       texture(uNoise3, vec3((q1 + vec2(0.0, e)) * 0.35, 0.6)).r - texture(uNoise3, vec3(q1 * 0.35, 0.6)).r) / e;
        vec2 g = mix(g0, g1, wb) * (0.10 + 0.25 * st.z);
        n = normalize(mix(n, normalize(vec3(-g.x, 1.0, -g.y) + n * 0.3), mf));
        // whitewater: streaks stretched along the flow (~2 m x 0.6 m), finer froth up close; ~80 %
        // white over the drops, ~15 % on the pools; edges no sharper than the pixel footprint
        vec2 fx = vec2(-fd.y, fd.x);
        // the streaks are long along the flow: where the stream runs toward / away from the camera only
        // the lateral footprint (not the grazing-angle one) has to resolve them
        float alV = abs(dot(fd, normalize(rd.xz + 1e-5)));
        float footS = mix(footW, max(t * gPixAng, footW * 0.27), alV * alV);
        float fineV = 1.0 - smoothstep(0.05, 0.15, footW);    // isotropic froth: full footprint
        float f0 = texture(uNoise3, vec3(dot(q0, fd) * 0.06, dot(q0, fx) * 0.22, 0.8)).r + (texture(uNoise3, vec3(q0 * 0.5, 0.8)).b - 0.5) * 0.6 * fineV;
        float f1 = texture(uNoise3, vec3(dot(q1, fd) * 0.06, dot(q1, fx) * 0.22, 0.2)).r + (texture(uNoise3, vec3(q1 * 0.5, 0.2)).b - 0.5) * 0.6 * fineV;
        float ft = wfRemap(mix(f0, f1, wb));
        float cov = mix(0.10, 0.50, smoothstep(0.04, 0.60, st.z));   // (round 2: more dark water between the white; r4 C4 / W8: 0.78 -> 0.50)
        // (session 4) where the water surface is nearly flat (the delta, the pools) the white is broken: runs of
        // dark glassy water between foam lines, not a white slide (mouth view / film shot 4)
        float wslF = (waterLevel(p.xz - fd * 0.5) - waterLevel(p.xz + fd * 0.5));
        cov *= mix(0.55, 1.0, smoothstep(0.03, 0.15, wslF));
        // (session 4) the white streaks are long along the flow (~2 m): seen along the flow at a grazing angle
        // only the lateral footprint has to resolve them (footS's 0.27 x footW faded them to a flat white
        // slide in the low mouth / film-shot-4 views); the along-view smear stays below a streak's length
        float footL = mix(footW, max(t * gPixAng * 1.5, footW * 0.05), alV * alV);
        float wd = 0.14 + 1.2 * footL;     // (round 3 i10: 0.08 -> 0.14, softer aerated edges - no white paint blotches)
        ft = mix(ft, 0.5, smoothstep(0.4, 1.5, footL));
        foamW = mf * smoothstep(1.0 - cov - wd, 1.0 - cov + wd, ft) * 0.95;
        // tumbling white water is not flat paint: grey-shaded streaks, thinner (darker) froth at its edges
        float tn = mix(0.5, texture(uNoise3, vec3(dot(q0, fd) * 0.15, dot(q0, fx) * 0.6, 0.35)).r, fineV);
        // + lane tone across the flow (~0.5-1 m lanes of brighter / greyer white water, resolved laterally)
        float lane = mix(0.5, texture(uNoise3, vec3(dot(q0, fd) * 0.02, dot(q0, fx) * 0.32, 0.55)).g, 1.0 - smoothstep(0.08, 0.3, footL));
        tn = 0.5 + (tn - 0.5) + 1.6 * (lane - 0.5);
        foamT = mix(0.42, 1.0, saturate(1.4 * (ft - (1.0 - cov)) / max(cov, 0.1) + 0.7 * (tn - 0.5) + 0.3));   // (r4 W8: grey flanks 0.55 -> 0.42)
        // steep water (level slope along the flow > ~0.35: over the lips, in the rapids) is white water
        // throughout, streaked along the flow (tone 0.7-1); along the flow only: the level's drop under
        // the banks does not count
        float stW = smoothstep(0.12, 0.30, wsl) * mf;
        // (round 2) tumbling white water has relief: mounds and standing waves (~1.5 m along x 0.7 m
        // across the flow) fixed to the bed (they do not drift with the foam), lit on their sunny side,
        // grey in the troughs, the troughs opening into dark holes on the rapids; unresolved -> the mean
        // (the same mean whiteness from 150-500 m: a white cascade)
        float visW = 1.0 - smoothstep(0.10, 0.40, footS);
        float rel = 0.5, gA = 0.0, gB = 0.0;
        if (visW > 0.0) {
            vec2 pa = vec2(dot(p.xz, fd), dot(p.xz, fx));
            const float eW = 0.18;
            rel = texture(uNoise3, vec3(pa.x * 0.10, pa.y * 0.24, 0.61)).g * 0.65 + 0.35 * texture(uNoise3, vec3(pa.x * 0.26, pa.y * 0.40, 0.13)).g;
            gA = (texture(uNoise3, vec3((pa.x + eW) * 0.10, pa.y * 0.24, 0.61)).g * 0.65 + 0.35 * texture(uNoise3, vec3((pa.x + eW) * 0.26, pa.y * 0.40, 0.13)).g - rel) / eW;
            gB = (texture(uNoise3, vec3(pa.x * 0.10, (pa.y + eW) * 0.24, 0.61)).g * 0.65 + 0.35 * texture(uNoise3, vec3(pa.x * 0.26, (pa.y + eW) * 0.40, 0.13)).g - rel) / eW;
            rel = mix(0.5, wfRemap(rel), visW);
        }
        vec3 fd3 = vec3(fd.x, 0.0, fd.y), fx3 = vec3(fx.x, 0.0, fx.y);
        nW = normalize(vec3(0.0, 1.0, 0.0) - (fd3 * gA + fx3 * gB) * 2.0 * visW);
        wW = mf * visW;
        foamT *= mix(1.0, mix(0.45, 1.08, smoothstep(0.15, 0.75, rel)), visW);
        foamW *= mix(1.0, mix(0.55, 1.0, smoothstep(0.12, 0.4, rel)), visW);
        if (stW > 0.0) {
            // (session 4) steep white water in lanes (resolved across the flow): dense white tongues and thinner
            // grey-green runs between them, not one white slide; unresolved -> ~0.82 x
            float laneV = 1.0 - smoothstep(0.08, 0.3, footL);
            // (review r2, fix 10: unresolved 0.82 / 0.92 -> 0.95 / 1.0: near-white #e9ecea from 150-500 m)
            // (r4 C4: unresolved 0.95 -> 0.6, the cascade is a white thread with grey-green water beside it)
            float lw = mix(0.6, mix(0.42, 1.0, smoothstep(0.32, 0.68, lane + 0.35 * (ft - 0.5))), laneV);
            foamW = max(foamW, stW * lw * mix(1.0, mix(0.45, 0.97, smoothstep(0.18, 0.5, rel)), visW));
            foamT = mix(foamT, mix(0.62, 1.04, saturate(ft + 0.7 * (tn - 0.5))) * mix(1.0, mix(0.6, 1.08, smoothstep(0.15, 0.75, rel)), visW), stW);
        }
        // (session 3) a tumbling cascade, not a white slide (film shot 4 / mouth view): on the steep
        // reaches the water drops over a lip every ~2.5-5 m (a ragged line across the channel); below each
        // lip a band of churned white water, then a short run of darker glassy water streaked with foam
        // lines before the next lip. Wet boulder heads break the surface, white water piled on their
        // upstream side, a dark wake below. Unresolved (footprint > ~0.5 m) -> the previous mean whiteness
        // (the 150-500 m views: a white cascade)
        float steepS = smoothstep(0.05, 0.22, wsl) * mf;
        gDbgA = vec2(stW, steepS);
        float visC = 1.0 - smoothstep(0.12, 0.5, mix(t * gPixAng, footW, alV * alV));   // (review r2: along-flow footprint)
        if (mf * visC > 0.0) {
            vec2 pa = vec2(dot(p.xz, fd), dot(p.xz, fx));
            // (review r2, fix 11) the churn follows the bake's real level drops (step pools 4.5-11 m: the old
            // synthetic 3.4 m phase did not line up with them): white water just below each drop, decaying
            // over ~2.5 m into a darker glassy run; a glassy tongue right at the lip. Filtered with the
            // footprint ALONG the flow (footF; footS is the lateral one, up to 3.7x too small here)
            float footF = mix(t * gPixAng, footW, alV * alV);
            // (review rv1) the probe distances vary across the channel (+-0.8 m): the white below each drop starts along a
            // ragged line, not a ruled one (the terraces of film shot 4)
            float jg = mix(0.0, (texture(uNoise3, vec3(pa.y * 0.35 + 0.7, pa.x * 0.02, 0.41)).r - 0.475) * 7.0, 1.0 - smoothstep(0.15, 0.5, footS));
            float o1 = max(1.2 + 0.8 * jg, 0.35), o2 = max(2.6 + 1.2 * jg, 0.8);
            float dr1 = (waterLevel(p.xz - fd * o1) - level) / o1, dr2 = (waterLevel(p.xz - fd * o2) - level) / o2;
            float dLip = (level - waterLevel(p.xz + fd * 0.6)) / 0.6;
            float aw = 0.05 + 0.6 * footF;
            float churnR = max(smoothstep(0.10 - aw * 0.2, 0.32 + aw * 0.2, dr1), 0.75 * smoothstep(0.12, 0.35, dr2));
            float churn = saturate(churnR + 0.35 * (ft - 0.5) * (1.0 - smoothstep(0.1, 0.4, footF)));
            // (round 3 i10) the churn below each drop comes in tongues along the flow (~1-2 m wide, many metres
            // long) with darker glassy lanes between them: the white water tumbles in tongues between the
            // boulders, not in bands straight across the channel (the terraces of film shot 4)
            float tg = mix(0.5, texture(uNoise3, vec3(pa.y * 0.45 + 0.3, pa.x * 0.045, 0.66)).r, 1.0 - smoothstep(0.12, 0.45, footS));
            churn *= mix(1.0, smoothstep(0.30, 0.62, tg) * 1.35, 0.85);
            float lipD = smoothstep(0.15, 0.45, dLip) * (1.0 - churnR) * (1.0 - smoothstep(0.06, 0.2, footF));   // glassy tongue at the lip
            float runF = smoothstep(0.60, 0.82, ft) * 0.7 * (1.0 - lipD);
            float wC = max(churn * mix(0.8, 1.0, ft), runF);
            // (review rv1) on the steep reaches the runs between the drops stay white in tongues (one continuous
            // tumbling ribbon between the boulders, not white bands with glassy water between them)
            wC = max(wC, stW * (0.6) * smoothstep(0.25, 0.6, tg) * mix(0.8, 1.0, ft));
            float k = steepS * visC;
            foamW = mix(foamW, wC, k);
            gDbgB = vec3(lipD, churn, wC);
            foamT = mix(foamT, mix(0.62, 1.05, churn) * mix(0.85, 1.05, ft), k * 0.7);
            // boulder heads (~2 m cells, ~1 in 4): dark wet granite domes 0.3-0.6 m across
            vec2 bs = vec2(0.42, 0.55);
            vec4 bc = wfVor(pa * bs + 11.0);
            vec2 bd = bc.xy / bs;                                 // (along, across) from the head's centre (m)
            float bR = 0.28 + 0.30 * fract(bc.w * 5.31);
            float rH = length(bd);
            // (r4 C4 / R9: an irregular lobed outline (3 + 5 lobes, own phase), not a smooth ellipse - the manhole covers)
            float angH = atan(bd.y, bd.x);
            bR *= 1.0 + 0.22 * (0.6 * sin(3.0 * angH + bc.w * 40.0) + 0.4 * sin(5.0 * angH + bc.w * 71.0));
            headE = rH / bR;
            float hasH = step(fract(bc.w * 13.7), 0.26);
            float aH = 0.03 + max(footF, footS);
            k = mf * visC;                                       // (session 4: heads on every reach, not only the steep)
            float head = hasH * (1.0 - smoothstep(bR - aH, bR + aH, rH)) * k;
            // white water piled against the upstream side, a dark trough / wake downstream
            float side = bd.x / max(rH, 1e-3);                    // -1 upstream .. +1 downstream
            float collar = hasH * (1.0 - smoothstep(bR + 0.15, bR + 0.6 + aH, rH)) * smoothstep(0.2, -0.5, side) * k;   // (r4: wider pillow 0.45 -> 0.6)
            float wake = hasH * (1.0 - smoothstep(bR * 0.8, bR * 1.3 + aH, abs(bd.y))) * smoothstep(0.0, bR, bd.x) * (1.0 - smoothstep(bR, bR + 2.2, bd.x)) * k;
            // (r4 R9) a thin white contact ring where the water wraps the head (~4 cm; unresolved -> faint)
            float ring = hasH * (1.0 - smoothstep(0.015 + 0.5 * aH, 0.04 + aH, abs(rH - bR - 0.02))) * k * mix(0.85, 0.3, smoothstep(0.03, 0.12, aH));
            foamW = max(max(foamW * (1.0 - 0.75 * wake), collar), ring);
            foamW *= 1.0 - head;
            gStreamHead = head;
            vec3 fd3h = vec3(fd.x, 0.0, fd.y), fx3h = vec3(fx.x, 0.0, fx.y);
            gStreamHeadN = normalize(vec3(0.0, 1.0, 0.0) + (fd3h * bd.x + fx3h * bd.y) / max(bR, 0.1) * 1.3);
        }
        // (round 3) the white water in 3D where it is resolved (lateral footprint < ~0.15 m: film shot 4, the
        // close stream views): heaps of aerated water standing above the water level (falls.glsl wwField),
        // up to ~0.5 m on the steep reaches, found by marching back along the ray from where it enters the
        // layer (parallax occlusion: heaps hide the water behind them, lit tops / shaded flanks). A hit is
        // shaded as opaque white water at the end; the 2D foam between the heaps thins out
        // (review rv5) the white water is a ribbon about half the channel wide winding between the boulders (the
        // channel was white from bank to bank - a painted staircase in film shot 4): dark glassy water beside it.
        // Resolved laterally only (footS); unresolved -> unchanged (the far views keep their white thread)
        vec2 sdR = streamDist(p.xz);
        gWWHw = 0.20 * sdR.y;     // (r4 C4 / W8: 0.45 -> 0.20, a ribbon ~1/4 of the bed)
        // (rv6) only on the steep reaches (the cascade): at the flat mouth the white fans out over the whole width
        // - the ribbon there left a salt-and-pepper of heap tops
        // (c1: also on the gently sloping reaches where the 3D heaps are drawn (bed slope > ~5 %): there the heaps covered the
        // channel bank to bank - film shot 4's near stream, debug 63)
        float ribV = (1.0 - smoothstep(0.2, 0.6, footS / max(gWWHw, 0.5))) * max(max(stW, steepS), smoothstep(0.03, 0.08, gStreamDir.z));
        float rib = mix(1.0, wwRibbon(vec2(-gStreamS, gStreamD), gWWHw, footS), ribV);
        gWWRib = ribV;
        foamW *= mix(1.0, 0.25 + 0.75 * rib, ribV);
        // (review rv7) the 3D heaps / boulders only on the sloping reaches (bed slope > ~5 %): on the flat mouth the grazing
        // march hit the low heap tops at random - a salt-and-pepper of white dots (film shot 4); the 2D foam stays there
        float visP = (1.0 - smoothstep(0.07, 0.22, footS)) * mf * smoothstep(0.03, 0.08, gStreamDir.z);
        float Hm = mix(0.24, 0.72, max(stW, steepS)) * saturate(foamW * 1.3);   // (i12: x1.2, more volume)
        gStreamHead *= 1.0 - visP * smoothstep(0.10, 0.22, max(ddS, 0.03));   // (the 3D boulders take over from the flat heads)
        if (visP > 0.0) {
            // curvilinear stream coordinates of this pixel (continuous across segments, same for every pixel),
            // stepped along the ray in the segment's frame
            vec2 fdS = gStreamDir.xy, fxS = vec2(-fdS.y, fdS.x);
            vec2 paP = vec2(-gStreamS, gStreamD);
            // the ray's descent toward the (locally planar, sloping) water surface per unit of t
            float dd = max(ddS, 0.03);
            // (review rv3) the 3D boulders only where the ray meets the (locally planar) surface steeply enough for the short
            // parallax march to reach their tops: at grazing views over the pools they were cut into flat horizontal
            // 'chrome pucks' along every lip (film shot 4, debug 63); there the flat 2D heads stay
            gWWBo = smoothstep(0.10, 0.22, dd);
            // (r4 R9: the march started at most 8 % of t before the surface - at film shot 4's ~10 m it saw only the lowest
            // ~0.12 m of the boulders, cut flat: dark discs / pucks lying in the white water; now up to 25 %)
            float tE = max(t - max(Hm, 1.45) / dd, t * 0.75);     // (i10/11: 0.62 -> 1.1, the boulders stand up to ~1 m; rv1 1.45: up to 1.3 m)
            // (review rv8) the heaps' fine bubbles / normals filtered by the footprint ALONG the view on the surface too (footW):
            // at grazing views their 4-30 cm detail is sub-pixel vertically - it was shaded into a grey salt-and-pepper
            float footG = max(footS, 0.35 * footW);
            gWWFoot = footG;
            float visB = 1.0 - smoothstep(0.03, 0.08, footG);
            const int NW = 14;
            float ds = (t - tE) / float(NW);
            float ti = tE + ds * fract(gRand * 7.13 + 0.37), a0 = tE, hit = -1.0, rk;
            // (review rv7) closest approach of the ray to the field: a near miss (within the march's vertical resolution)
            // counts as a partial hit there - soft heap outlines instead of random white / dark dots
            float minD = 1e9, tMin = tE;
            for (int i = 0; i < NW; i++) {
                vec2 dq = rd.xz * (ti - t);
                float dH = (t - ti) * dd - wwField(paP + vec2(dot(dq, fdS), dot(dq, fxS)), Hm, visB, rk);
                if (dH < 0.0) { hit = ti; break; }
                if (dH < minD) { minD = dH; tMin = ti; }
                a0 = ti;
                ti += ds;
            }
            float covW = 1.0;
            bool nearM = false;
            if (hit < 0.0) {
                covW = 1.0 - smoothstep(0.0, max(0.03, 1.6 * ds * dd), minD);   // (rv9: 0.8 -> 1.6 x the march resolution)
                if (covW > 0.02) { hit = tMin; nearM = true; }
            }
            if (hit > 0.0) {
                float ba = nearM ? hit : a0, bb = hit;
                for (int k = 0; k < (nearM ? 0 : 4); k++) {
                    float m = 0.5 * (ba + bb);
                    vec2 dq = rd.xz * (m - t);
                    if ((t - m) * dd < wwField(paP + vec2(dot(dq, fdS), dot(dq, fxS)), Hm, visB, rk)) bb = m; else ba = m;
                }
                vec2 dq = rd.xz * (bb - t);
                vec2 pq = paP + vec2(dot(dq, fdS), dot(dq, fxS));
                gWWP = ro + rd * bb;
                float eH = max(0.03, max(bb * gPixAng, 0.5 * footG));
                float rkx, rkz;
                float h0 = wwField(pq, Hm, visB, gWWRock);
                gWWH = h0;
                float hx = wwField(pq + vec2(eH, 0.0), Hm, visB, rkx), hz = wwField(pq + vec2(0.0, eH), Hm, visB, rkz);
                // gradient (along, across) -> world xz
                vec2 gA = vec2(hx - h0, hz - h0) / eH;
                vec2 gW = fdS * gA.x + fxS * gA.y;
                vec3 ns = normalize(vec3(wsl * fd.x, 1.0, wsl * fd.y));
                gWWN = normalize(ns - vec3(gW.x, 0.0, gW.y));
                gWWOcc = mix(0.28, 1.0, saturate(h0 / max(Hm, 0.3)));   // (review rv5: 0.45 -> 0.28, darker troughs)
                // heaps / boulders shade each other (3 steps toward the sun over the field, relative to the surface)
                float rise = uSunDir.y + wsl * dot(fd, uSunDir.xz);
                vec2 Lp = vec2(dot(uSunDir.xz, fdS), dot(uSunDir.xz, fxS));
                float shW = 0.0;
                for (int k = 1; k <= 3; k++) {
                    float sd = 0.1 * float(k * k);
                    shW = max(shW, smoothstep(0.0, 0.06, wwField(pq + Lp * sd, Hm, visB, rkx) - h0 - sd * rise));
                }
                gWWSh = 1.0 - shW;
                gWWHit = 1.0;
            }
            gWWVis = visP * covW;
            // (review rv9) the water between the heaps stays aerated (0.4 -> 0.75): the troughs the march slips through showed as
            // a grey salt-and-pepper over the white cascade (film shot 4)
            foamW *= mix(1.0, 0.4, visP);     // (r4 C4: 0.75 -> 0.4, dark glassy water between the tongues)
        }
        foamT *= 0.8;
    }
    // falls lab debug 63: stream layers (r = 3D boulder hit, g = 3D white-water hit, b = 2D foam cover)
    if (uDebug == 63) return vec3(gWWHit * gWWRock * gWWVis, gWWHit * (1.0 - gWWRock) * gWWVis, foamW);
    if (uDebug == 66) return vec3(mf, gDbgA.x, gDbgA.y);
    if (uDebug == 67) return gDbgB;
    // the stream's outflow into the lake: a few soft foam streaks drifting out, gone after ~15 m
    // (the lake stays clear turquoise right up to the cascade)
    if (dot(uStreamMouth.zw, uStreamMouth.zw) > 0.5) {
        vec2 dm = p.xz - uStreamMouth.xy;
        float along = dot(dm, uStreamMouth.zw), lat = dot(dm, vec2(-uStreamMouth.w, uStreamMouth.z));
        if (along > -8.0 && along < 70.0 && abs(lat) < 30.0) {
            float wdt = 2.0 + 0.2 * max(along, 0.0);
            float pl = exp(-max(along, 0.0) / 6.0) * exp(-lat * lat / (wdt * wdt)) * smoothstep(-8.0, 0.0, along) * (1.0 - mf);
            vec2 qa = vec2(along - uTime * 0.45, lat + 0.5 * gnoise(dm * 0.08));
            float nn = wfRemap(texture(uNoise3, vec3(qa.x * 0.05, qa.y * 0.14, 0.45 + uTime * 0.01)).r) * 0.65
                     + 0.35 * wfRemap(texture(uNoise3, vec3(qa * 0.12, 0.2)).a);
            nn = mix(nn, 0.5, smoothstep(0.4, 1.5, foot));
            float wdp = 0.1 + 1.0 * foot;
            foamW = max(foamW, smoothstep(0.82 - 0.35 * pl - wdp, 0.82 - 0.35 * pl + wdp, nn) * smoothstep(0.05, 0.5, pl) * 0.6);
            n = normalize(n + vec3(gnoise(qa * 0.7), 0.0, gnoise(qa * 0.7 + 3.1)) * 0.12 * pl);
        }
    }
    // waterfall plunge pool: foam and ring waves
    vec2 tilt;
    float pf = poolFoamF(p.xz, foot, tilt);
    n = normalize(n + vec3(tilt.x, 0.0, tilt.y));
    if (pf > foamW) foamT = mix(foamT, gPoolTone, saturate((pf - foamW) / max(pf, 1e-3)));
    foamW = max(foamW, pf);
    if (uDebug == 9) return vec3(pf, gPoolAer, foamW);
    // unresolved wave slopes (sv): their spread smears the reflection vertically (uTune[6] x sigma).
    // (r2c) Two parts of unit variance: (1) a world-space ripple-group pattern (wind-aligned, crests ~3x longer
    // than wide, drifting and evolving with uTime) at the finest scales the along-view footprint resolves
    // (cells >= 2-5 px): broken horizontal streaks, coherent in space and time (no grain, moves with
    // the water); (2) the stochastic per-sample part for what is still below the pixel (converges to a soft
    // vertical blur with the film's spp). uTune[7] = share of (1) where it is resolved. (The r2 masking tilt
    // toward the viewer, the old slot 7, is gone: it brought the pale far-water band back.)
    {
        float sg = sqrt(sv);
        int fr = int(floor(uTime * 30.0 + 0.5));
        float h1 = hash12i(ivec2(gl_FragCoord.xy) * 5 + ivec2(fr * 389 + 3, fr * 7741 + 11));
        float h2 = hash12i(ivec2(gl_FragCoord.xy) * 3 + ivec2(fr * 1543 + 17, fr * 97 + 5));
        float gs = (fract(h1 + float(uSample) * 0.6180340) + fract(h2 + float(uSample) * 0.7548777) - 1.0) * 2.45;
        float gc = 0.0, gq = 0.0, rem = 1.0;
        float tu7 = uTune[7];
        if (tu7 > 0.0) {
            vec2 wd = normalize(uWaterWind + vec2(1e-5, 0.0));
            vec3 nTs = vec3(textureSize(uNoise3, 0));
            // (r4 C6 / W2: crests 1.5x (was 3x) longer than wide and cells only up to 1.8 m (was 16 m): the long parallel
            // world-space strips converged on a vanishing point (the speed-line fan, shot 1) and shifted whole strips of
            // the far-shore reflection out into the lake (the venetian blinds, shot 0))
            vec2 q0 = vec2(dot(p.xz, wd), dot(p.xz, vec2(-wd.y, wd.x)) / 1.5) - vec2(uTime * 0.35, 0.0);
            float cs = 0.2;                                   // cell size (m) of the finest octave
            for (int j = 0; j < 3; j++) {
                float wj = smoothstep(1.5, 4.0, cs / fL) * rem;
                if (wj > 0.0) {
                    // (water5) C1-smooth (cubic) interpolation between the noise texels: the trilinear cells drew blocky
                    // facets into the far-shore reflection (shot 0's mid water)
                    vec3 nq = vec3(q0 / (16.0 * cs) + float(j) * 0.37, uTime * 0.05 + float(j) * 0.29) * nTs - 0.5;
                    vec3 ni = floor(nq), nf = nq - ni;
                    nf.xy = nf.xy * nf.xy * (3.0 - 2.0 * nf.xy);
                    float nz = texture(uNoise3, (ni + nf + 0.5) / nTs).g;
                    gc += wj * (nz - 0.5) * 5.0;              // value noise std ~0.2 -> ~unit
                    gq += wj * wj;
                    rem -= wj;
                }
                cs *= 3.0;
            }
        }
        float sh = (1.0 - rem) * tu7;                         // coherent share of the variance
        // (the per-sample part is temporal noise in the film (shot 0: 13.8 % of the water pixels changed > 20
        // levels per frame at 2 spp); with the ripple groups on it is kept at 35 %: the far, unresolved water
        // keeps a soft smear, the rest of the variance is carried by the coherent pattern or dropped)
        // (r2d) uTune[22] = per-sample share at steep views (|rd.y| 0.08 -> 0.3): looking down on the lake (flyover,
        // high overlooks) the unresolved smear only scattered the reflected sky / cloud boundary into pixel speckle
        // (shot 6: reflection high-pass std 0.24 vs the water body 0.26, ~10 % sparkle at 2 spp); at grazing views
        // it still draws the vertical streaks
        float gsS = mix(1.0, uTune[22], smoothstep(0.08, 0.3, -rd.y));
        float g = (gq > 1e-4 ? gc * inversesqrt(gq) : 0.0) * sqrt(sh) + gs * sqrt(1.0 - sh) * mix(1.0, 0.35, tu7) * gsS;
        float tilt = sg * uTune[6] * g;
        // (r4 W2: a tilt away from the viewer flattens the reflected ray onto the far beach: clamp 0.6 -> 0.35 |rd.y|)
        tilt = max(tilt, -0.35 * abs(rd.y));
        n = normalize(n - vec3(fwdH.x, 0.0, fwdH.y) * tilt);
    }
    // back-facing facet left after the filtering: project the normal just far enough to face the viewer
    // (the old fix added 0.2 x the view vector, which tilted it ~10 deg and sent the reflection ~20 deg up)
    // (r2b: always the minimal fix; uTune[9] is now the sun warmth, probe.frag)
    float nv = dot(n, -rd);
    if (nv < 0.02) n = normalize(n + (-rd) * (0.02 - nv));
    float cosi = saturate(dot(-rd, n));
    // (r4 W1: roughness-aware Fresnel - at grazing views the visible ripple facets lean toward the viewer (masking), by about the
    // slope spread: the filtered-out ripples (sv) plus a floor for the resolved ones; the effective incidence is |rd.y| + that tilt,
    // which lets the teal body through (r4 mid F 0.64 -> ~0.4; far band <= ~0.55) instead of a grey / khaki mirror)
    // (the floor for the resolved ripples fades out toward the far band, |rd.y| < ~0.01: there all the ripples are in sv and the far
    // shore's reflection keeps dominating)
    // (c1: both terms fade out below |rd.y| ~0.03: the unresolved sv alone had killed the far shore's forest reflection in the shore
    // views - flat green paint, mid-water hp 0.08-0.10 vs a 0.42 target (shots 0 / 1))
    // (the lean is weaker in the midday shore views (k 0.6: their reflections carry the ripple texture - k 1.5 left shots 0 / 1 flat
    // green paint, hp 0.08-0.10 vs 0.42) than at a low sun (k 1.5: turquoise water, not a khaki mirror of the lit wall))
    float lowSunF = 1.0 - smoothstep(0.35, 0.6, uSunDir.y);
    // (water5: midday k 0.6 -> 0.78 and the lean kept down to the horizon (it faded out below |rd.y| 0.03): the far band was
    // a yellow-green copy of the hazed far-shore forest (shots 0 / 1 h158-163, F .85-.9) - it should read teal-grey
    // (h183-200), the water body through the ripple facets; the low sun keeps the fade: there the lake mirrors the lit far forest)
    float cosF = max(cosi, abs(rd.y) + mix(0.78, 1.5, lowSunF) * (sqrt(sv) + mix(0.03, 0.06, lowSunF) * uWaveStrength) * mix(1.0, smoothstep(0.006, 0.03, abs(rd.y)), lowSunF));
    float F = 0.02 + 0.98 * pow(1.0 - cosF, 5.0);

    // reflection
    vec3 rr = reflect(rd, n);
    rr.y = max(rr.y, 0.003);
    rr = normalize(rr);
    vec3 ro2 = p + vec3(0.0, 0.05, 0.0);
    gTraceT0 = t;
    Hit hr = trace(ro2, rr, 0.1, 25000.0, false, true, t < 300.0, QUALITY == 1 ? 320 : 160, 2.5);
    gTraceT0 = 0.0;
    if (uDebug == 60) return vec3(hr.kind == K_SKY ? 1.0 : 0.0, rr.y * 10.0, t / 4000.0);
    if (uDebug == 61) return vec3(hr.t / 4000.0, float(hr.kind) / 6.0, F);
    if (uDebug == 3) {
        vec3 kc[7] = vec3[7](vec3(0.2, 0.3, 1.0), vec3(0.6, 0.5, 0.4), vec3(0.0, 0.6, 0.0), vec3(0.5, 0.5, 0.5), vec3(0.0, 0.8, 0.9), vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 0.0));
        return kc[hr.kind] * (hr.kind == K_TERRAIN ? clamp(hr.t / 3000.0, 0.2, 1.0) : 1.0);
    }
    vec3 refl;
    if (hr.kind == K_SKY) refl = skyWithClouds(ro2, rr, 10, QUALITY == 1 ? 40 : 20);
    else refl = applyAerial(shadeHit(hr, ro2, rr, true), ro2, rr, hr.t, 8);

    // sun glint: GGX with Smith masking-shadowing (grazing views must not glitter everywhere)
    vec3 hv = normalize(uSunDir - rd);
    float a = mix(0.03, 0.08, gust) + min(foot * 0.02, 0.05);
    float nh = max(dot(n, hv), 0.0);
    float dd = nh * nh * (a * a - 1.0) + 1.0;
    float D = a * a / (PI * dd * dd);
    float nl = max(dot(n, uSunDir), 0.0);
    float cv2 = cosi * cosi, cl2 = nl * nl;
    float G1v = 2.0 / (1.0 + sqrt(1.0 + a * a * (1.0 - cv2) / max(cv2, 1e-6)));
    float G1l = 2.0 / (1.0 + sqrt(1.0 + a * a * (1.0 - cl2) / max(cl2, 1e-6)));
    float sunVis = terrainShadow(p + vec3(0, 0.5, 0), 50) * cloudShadow(p);
    float Fh = 0.02 + 0.98 * pow(1.0 - max(dot(hv, -rd), 0.0), 5.0);
    vec3 glint = uSunCol * sunVis * D * Fh * G1v * G1l / (4.0 * max(cosi, 1e-3)) * nl;
    // (r4 W14) sub-pixel sun glints: the facets of the unresolved ripples (slope spread sqrt(sv)) catch the sun in tiny
    // points - the mean glint over that wider lobe is drawn as rare per-sample sparkles of a fixed peak radiance (the
    // same mean over the samples: 1-3 px clipped white points near the sun azimuth, no flat pale dashes / glare path)
    {
        float aS = sqrt(a * a + 0.3 * sv) + 0.03;     // (0.3 sv: the sparkles stay near the glitter path, none off it)
        float ddS = nh * nh * (aS * aS - 1.0) + 1.0;
        float DS = aS * aS / (PI * ddS * ddS);
        float G1vS = 2.0 / (1.0 + sqrt(1.0 + aS * aS * (1.0 - cv2) / max(cv2, 1e-6)));
        float G1lS = 2.0 / (1.0 + sqrt(1.0 + aS * aS * (1.0 - cl2) / max(cl2, 1e-6)));
        vec3 gM = uSunCol * sunVis * DS * Fh * G1vS * G1lS / (4.0 * max(cosi, 1e-3)) * nl;
        float lS = dot(uSunCol, vec3(0.2126, 0.7152, 0.0722)) * 120.0;
        float P = dot(gM, vec3(0.2126, 0.7152, 0.0722)) / max(lS, 1e-4);
        int frG = int(floor(uTime * 24.0 + 0.5));
        float uG = hash12i(ivec2(gl_FragCoord.xy) * 7 + ivec2(uSample * 7919 + frG * 131, uSample * 3511 + frG * 977));
        glint = P >= 1.0 ? gM : (uG < P ? gM / max(P, 1e-6) : vec3(0.0));
    }

    // refraction into the lake
    vec3 tdir = refract(rd, n, 1.0 / 1.333);
    if (dot(tdir, tdir) < 0.5) tdir = normalize(rd + vec3(0.0, -0.5, 0.0));
    Hit hb = trace(p - vec3(0.0, 0.01, 0.0), tdir, 0.0, 90.0, false, false, t < 300.0, 120, 3.0);
    float path = hb.kind == K_SKY ? 90.0 : hb.t;
    vec3 bp = p + tdir * path;
    float bedDepth = max(level - bp.y, 0.0);
    vec3 bedCol = vec3(0.0);
    // (look3 r3) clearer water: 0.45 x the round-2 scattering (stones on the shelf visible 2-4 m down),
    // the in-scatter scale x1.45 keeps the deep turquoise
    // (water5: 0.45 -> 0.225 with the clearer W_ABS: the deep in-scatter ceiling wSca / (2 sext) stays, its build-up per metre
    // halves - clear shallows, the bed visible through twice the depth)
    vec3 wSca = W_SCA * 0.225;
    vec3 sext = W_ABS + wSca;
    // downwelling light at depth z (sun refracted, sky diffuse)
    float sunCosW = sqrt(1.0 - (1.0 - uSunDir.y * uSunDir.y) / (1.333 * 1.333));
    vec3 Tsun = exp(-sext * bedDepth / max(sunCosW, 0.1));
    vec3 Tsky = exp(-sext * bedDepth * 1.25);
    if (hb.kind != K_SKY) {
        // simplified bed shading (stones, silt, algae) with caustics
        vec3 bn = hb.kind == K_ROCK ? vec3(0.0, 1.0, 0.0) : terrainNormal(bp.xz, max(0.05, t * gPixAng * 2.0), detailWeight(t));
        vec3 alb;
        float n3 = gnoise(bp.xz * 0.45);
        if (hb.kind == K_ROCK) alb = rockBedAlbedo(bp, t, hb.cell, bn);
        else {
            vec3 bb;
            alb = pebbles(bp.xz, bb) * 0.95 * vec3(1.0, 0.95, 0.75);   // (r4 W6: golden-olive bed)
            bn = normalize(bn + bb * (1.0 - smoothstep(6.0, 20.0, t)));
            vec3 silt = mix(vec3(0.20, 0.21, 0.17), vec3(0.28, 0.28, 0.23), saturate(0.5 + n3));
            silt *= vec3(1.08, 1.0, 0.62) * 1.6;    // (water5: paler, golden-olive rock-flour silt, not grey: the luminous band over the shelf)
            alb = mix(alb, silt, smoothstep(1.5, 6.0, bedDepth) * 0.8);
            // (water5) the deeper bed in broad patches (~15-40 m) of pale rock-flour sand and darker cobble / rock: the turquoise
            // over it is luminous over the pale patches, deeper jade over the dark ones, not one flat sheet
            float bpN = texture(uNoise3, vec3(bp.xz * 0.021, 0.37)).r * 0.65 + texture(uNoise3, vec3(bp.xz * 0.055, 0.81)).r * 0.35;
            alb *= mix(1.0, clamp(1.0 + 2.4 * (bpN - 0.475), 0.4, 1.6), smoothstep(1.0, 4.0, bedDepth));
            alb *= mix(vec3(1.0), vec3(0.95, 1.0, 0.85), smoothstep(0.5, 4.0, bedDepth));   // algae film (r4: from 0.5 m; water5: (.85, 1, .8) -> (.95, 1, .85), less green)
            // (r4 W3) steep submerged faces (the cliff foot) are dark wet rock, not pale pebbles
            alb = mix(alb, vec3(0.018, 0.019, 0.016), smoothstep(0.6, 0.45, bn.y));
        }
        // (r4 C4 / W8) the stream's bed is wet, algae-dark stone (dark glassy water between the white tongues), not the
        // lake shelf's pale pebbles - through the few cm of stream water they made it a pale plaster sheet (film shot 4)
        alb *= mix(vec3(1.0), vec3(0.36, 0.40, 0.33), mf);
        // (water5) submerged stones are wet and carry a golden-brown periphyton film (target shallows #6e644a /
        // #6d684e / #68685e: the bed through the water was green-grey, h150 vs h90-130); not at the waterline, not in the stream
        alb = mix(alb, alb * vec3(1.0, 0.78, 0.60) * 0.85, smoothstep(0.03, 0.35, bedDepth) * (1.0 - mf));
        float sh = terrainShadow(bp + vec3(0.0, level - bp.y + 0.3, 0.0), 40);
        float ca = caustics(bp.xz - uSunDir.xz / max(uSunDir.y, 0.2) * bedDepth * 0.75, bedDepth, (t + path) * gPixAng / max(-tdir.y, 0.25));
        float ndl = max(dot(bn, uSunDir), 0.0);
        // (water5: caustic gain 0.9 -> 1.8: a bright net over the near stones)
        vec3 E = uSunCol * (1.0 - 0.03) * ndl * sh * Tsun * (0.85 + 1.8 * ca) + uSkyAmb * 0.8 * Tsky * terrainAO(bp.xz);
        // (r4 W3) contact occlusion where the bed meets the surface (no bright kerb along the cliff foot / boulder skirts)
        bedCol = alb * E * mix(0.35, 1.0, smoothstep(0.0, 0.4, bedDepth));
    }
    vec3 Tv = exp(-sext * path);
    // single scattering in the water column (approx. downwelling at mid depth)
    // single scattering: light entering at the surface decays on its way down and back up along
    // the view path (combined extinction ~ 2 x sext); deep water saturates at W_SCA / (2 sext)
    // (the water column under a terrain / cloud shadow only gets skylight: golden-hour shaded lake)
    vec3 E0 = uSunCol * max(uSunDir.y, 0.0) * 0.97 * mix(1.0, sunVis, uTune[10]) + uSkyAmb;
    // (r4 L5) low sun: the sky-lit water column keeps the shaded lake turquoise (near water #2c5b62, mid #40908e;
    // the gold look's low ambient left a khaki mirror of the lit wall)
    // (warm-grey: the low sun's light; the blue sky alone turned the shaded water ink-blue)
    // (water5: 1.5 -> 2.4 with the clearer, less scattering column: target mid #40908e C.078, ours C.052-.056)
    E0 += vec3(1.0, 1.0, 0.65) * dot(uSkyAmb, vec3(0.2126, 0.7152, 0.0722)) * 2.4 * (1.0 - smoothstep(0.35, 0.6, uSunDir.y));
    vec3 ins = E0 * uWaterIns * 1.45 * (wSca / (2.0 * sext)) * (1.0 - exp(-2.0 * sext * path));
    vec3 refr = bedCol * Tv + ins;

    if (uDebug == 4) return refl;
    if (uDebug == 5) return refr;
    if (uDebug == 68) return bedCol * Tv;
    if (uDebug == 69) return ins;
    if (uDebug == 6) return vec3(F);
    if (uDebug == 7) return glint;
    vec3 col = mix(refr, refl, F) + glint;
    if (foamW > 0.0) {
        vec3 nf = normalize(mix(normalize(n + vec3(0.0, 1.5, 0.0)), nW, wW));
        vec3 foamC = vec3(0.86, 0.89, 0.90) * (uSunCol * max(dot(nf, uSunDir), 0.0) * sunVis + skyLight(nf, 1.0) * 1.1);
        col = mix(col, foamC * foamT, foamW);
    }
    // stream boulder heads breaking the surface (stream block above): dark wet granite, a wet sheen
    if (gStreamHead > 0.0) {
        vec3 nr = gStreamHeadN;
        // (r4 C4 / W8 / R9: the same granite ramp as the 3D boulders - wet and near-black at the waterline, pale dry granite
        // on the top (was flat 0.075: black holes punched in the white water); grain + lichen up close)
        float fpH = t * gPixAng;
        float grH = mix(0.475, texture(uNoise3, vec3(p.xz * 0.9, 0.3)).r, 1.0 - smoothstep(0.03, 0.1, fpH)) * 0.6
                  + 0.4 * mix(0.475, texture(uNoise3, vec3(p.xz * 0.3, 0.7)).r, 1.0 - smoothstep(0.1, 0.3, fpH)) + 0.025;
        vec3 raH = mix(vec3(0.075, 0.074, 0.069), vec3(0.26, 0.252, 0.235), smoothstep(0.85, 0.35, headE)) * mix(0.75, 1.25, grH);
        vec3 rc = raH * (uSunCol * max(dot(nr, uSunDir), 0.0) * sunVis + skyLight(nr, 1.0))
                + uSunCol * sunVis * mix(0.3, 0.12, smoothstep(0.85, 0.35, headE)) * pow(max(dot(reflect(rd, nr), uSunDir), 0.0), 24.0);
        col = mix(col, rc, gStreamHead);
    }
    // (round 3) heaped 3D white water of the stream (stream block): opaque aerated water - bright diffuse, light
    // bleeding through (the shaded flanks are grey-blue, not black), the hollows between the heaps occluded
    if (gWWVis > 0.0 && gWWHit > 0.0) {
        vec3 nw = gWWN;
        float ndl = max(dot(nw, uSunDir), 0.0);
        // (i10) lit by a near-grey sky (aerated water scatters many times: the saturated blue ambient alone turned the
        // shaded flanks of the heaps blue-speckled)
        vec3 skW = skyLight(nw, 1.0);
        skW = mix(vec3(dot(skW, vec3(0.2126, 0.7152, 0.0722))), skW, 0.5);
        // (review rv5) tone: aerated lumps bright, thinner glassy water between them greyer / greener (the 3D white water
        // was an even white, p25-p75 224-234 in film shot 4 - flat paint; the cascade shades grey on its flanks)
        float fpW = max(length(gWWP - ro) * gPixAng, gWWFoot);
        float wt1 = mix(0.5, texture(uNoise3, vec3(gWWP.xz * vec2(0.55, 0.55), 0.13 + uTime * 0.05).xzy).b, 1.0 - smoothstep(0.08, 0.25, fpW));
        float wt2 = mix(0.5, texture(uNoise3, vec3(gWWP.xz * 0.45, 0.71 + uTime * 0.11).xzy).a, 1.0 - smoothstep(0.05, 0.15, fpW));   // (rv6: 1.6 -> 0.45, ~14 cm bubbles - the 4 cm ones were salt-and-pepper)
        float wtn = saturate(0.5 + 1.8 * (wt1 - 0.525) + 1.2 * (0.5 - wt2));
        vec3 wAlb = mix(vec3(0.40, 0.45, 0.44), vec3(0.68, 0.69, 0.70), smoothstep(0.15, 0.75, wtn));
        vec3 wc = wAlb * (uSunCol * sunVis * (ndl * gWWSh * mix(0.75, 1.0, gWWOcc) * 0.92 + 0.06) + skW * 0.95 * gWWOcc);
        // a boulder breaking the surface: dark wet granite, a wet sheen
        if (gWWRock > 0.5) {
            // (i12: textured at the hit point - the water plane's p smeared it into horizontal streaks at grazing views)
            float fpR = length(gWWP - ro) * gPixAng;
            float gr = mix(0.475, texture(uNoise3, gWWP * 0.9 + 0.3).r, 1.0 - smoothstep(0.03, 0.1, fpR)) * 0.6
                     + 0.4 * mix(0.475, texture(uNoise3, gWWP * 0.3 + 0.7).r, 1.0 - smoothstep(0.1, 0.3, fpR)) + 0.025;
            // (wet and near-black at the waterline, its top above the splash dry pale granite)
            // (r4 R9: near-black only in the splash band (< ~6 cm), grey wet granite 0.20-0.26 above - not black holes)
            vec3 ra = mix(vec3(0.075, 0.074, 0.069), vec3(0.26, 0.252, 0.235), smoothstep(0.02, 0.14, gWWH)) * mix(0.75, 1.25, gr);
            // (review rv5) granite, not a smooth egg: rough facets (normal noise ~0.5 / 0.15 m), grey-green lichen
            // patches on the dry top, dark streaks; resolved only (footprint)
            float vRk = 1.0 - smoothstep(0.03, 0.10, fpR);
            nw = normalize(nw + ((texture(uNoise3, gWWP * 0.75 + 0.4).rgb - 0.5) * 0.9 + (texture(uNoise3, gWWP * 2.2 + 0.1).rgb - 0.5) * 0.45) * vRk);
            ndl = max(dot(nw, uSunDir), 0.0);
            float lchR = smoothstep(0.56, 0.66, texture(uNoise3, gWWP * 0.35 + 2.3).r) * smoothstep(0.2, 0.5, gWWH) * vRk;
            ra = mix(ra, vec3(0.15, 0.16, 0.12), lchR * 0.6);
            ra *= mix(1.0, mix(0.55, 1.1, smoothstep(0.35, 0.65, texture(uNoise3, vec3(gWWP.x * 0.8, gWWP.y * 0.25, gWWP.z * 0.8) + 0.9).g)), vRk);
            wc = ra * (uSunCol * sunVis * ndl * gWWSh + skyLight(nw, 1.0) * gWWOcc)
               + uSunCol * sunVis * gWWSh * 0.3 * pow(max(dot(reflect(rd, nw), uSunDir), 0.0), 24.0);
        }
        col = mix(col, wc, gWWVis);
    }
    // aerated, milky turquoise water around the plunge pool (falls.glsl poolFoam)
    if (gPoolAer > 0.0) {
        vec3 aerC = vec3(0.32, 0.66, 0.62) * (uSunCol * max(uSunDir.y, 0.0) * sunVis * 0.55 + skyLight(vec3(0.0, 1.0, 0.0), 1.0));
        col = mix(col, aerC, gPoolAer * 0.5 * (1.0 - foamW));
    }
    return col;
}

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
    // T3 (port/RUNBOOK.md §6, BINDING): equirect texel centre -> direction.
    // lon = -pi + (frag.x/W)*2pi, lat = (frag.y/H - 0.5)*pi with frag the
    // (jittered) pixel CENTRE coords - at frag = (x+0.5, y+0.5) this is
    // exactly the pixel-centre formula of engine_vr/pano.equirect_directions
    // (frag y grows up = lat grows; the readback flip at the PNG step
    // reproduces the §6 top-down row order, same place as the engine's
    // engine/render.py:505 display flip). gPixAng becomes the per-texel
    // footprint from the panoFoot strip. The DOF aperture branch below is
    // NOT copied: a pano has no lens (vr.py always sets uAperture = 0).
    vec2 frag = gl_FragCoord.xy + uJitter;   // FIXME(port)
    int iFrame = int(floor(uTime * 24.0 + 0.5));   // FIXME(port)
    gRand = hash12i(ivec2(gl_FragCoord.xy) + ivec2(uSample * 7919 + iFrame * 31337, uSample * 104729 + iFrame * 7121));   // FIXME(port)
    float lon = (frag.x / uRes.x * 2.0 - 1.0) * PI;   // FIXME(port)
    float lat = (frag.y / uRes.y - 0.5) * PI;   // FIXME(port)
    gPixAng = panoFootAt(frag.y);   // FIXME(port)
    ro = uCamPos;   // FIXME(port)
    rd = uCamRot * vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));   // FIXME(port)
}

layout(set=0, binding=15) uniform sampler2D uGBuf0;     // (t, kind, cell) of the primary hit (SCENE_PASS 99)
layout(set=0, binding=16) uniform sampler2D uGBuf1;     // its info

vec3 render(Hit h, vec3 ro, vec3 rd) {
    if (uDebug == 1) {
        vec3 kc[8] = vec3[8](vec3(0.2, 0.3, 1.0), vec3(0.6, 0.5, 0.4), vec3(0.0, 0.6, 0.0), vec3(0.5, 0.5, 0.5), vec3(0.0, 0.8, 0.9), vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0));
        return kc[h.kind];
    }
    if (uDebug == 2) {
        vec3 p = ro + rd * h.t;
        return vec3(sunShadowMap(vec3(p.x, waterLevel(p.xz) + 0.3, p.z)), cloudShadow(p), fract(waterLevel(p.xz)));
    }
    vec3 p = ro + rd * h.t;
    vec3 col;
    gPrimary = false;
    col = shadeWater(ro, rd, h.t);
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
