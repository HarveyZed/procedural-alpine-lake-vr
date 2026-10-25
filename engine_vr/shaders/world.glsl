// ---------------------------------------------------------------------------
// world.glsl : runtime access to the baked world (heights, AO, water, forest)
// ---------------------------------------------------------------------------
#define NEAR_HALF 2048.0
#define NEAR_RES 4096.0
#define FAR_HALF 16384.0
#define FAR_RES 8192.0
#define TREE_CELL 4.0

uniform sampler2D uHNear;    // R32F, 1 m texels, +-2048 m
uniform sampler2D uHFar;     // R32F, 4 m texels, +-16384 m
uniform sampler2D uAONear;   // sky visibility
uniform sampler2D uAOFar;
uniform sampler2D uForest;   // RGBA8 on the far grid: density, larch, scree, meadow
uniform sampler2D uWaterLvl; // R32F water surface level (bilinear), -1e4 = none
uniform sampler2D uWaterLvlNear; // R32F 1 m level map (lakes + stream channel)
uniform sampler2D uStreamNear;   // RGBA16F: flow dir xz, foam, stream mask
uniform sampler3D uNoise3;   // RGBA8 128^3 tileable: fbm, smooth, worley-fbm, worley
uniform sampler2D uTrees;    // RGBA32F per 4 m cell (+-8192 m): baseY, signed H (neg = larch), crown R, crown base
uniform sampler2D uShNear;   // RG32F sun shadow map: shadowTop (abs y), occluder distance
uniform sampler2D uShFar;
#define SH_FAR_HALF 16384.0

uniform vec3 uCamPos;
uniform float uTime;
// forest trail polyline (features.json "trail"), used by ground.glsl and terrain_shade.glsl
#define GC_TRAIL 1
#define GC_SKYOCC 1
uniform vec4 uTrail[32];
uniform int uTrailN;
uniform float uPixAng;     // radians per pixel of the current camera
#ifdef VR_PANO
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
uniform sampler2D panoFoot;   // FIXME(port)
float panoFootAt(float y) {   // FIXME(port)
    return texelFetch(panoFoot, ivec2(0, int(clamp(y, 0.0, uRes.y - 1.0))), 0).r;   // FIXME(port)
}                             // FIXME(port)
#if SCENE_PASS == 98
// FIXME(port): the cone pass (SCENE_PASS 98) draws into a QUARTER-res rect
// ((w+3)/4 x (h+3)/4 in vr.py; engine analogue engine/render.py:472-479),
// so gl_FragCoord.y here is a CONE row while panoFoot rows carry FULL-res
// latitudes. Scale by 4.0 so the uPixAng consumers inside the shared
// SDF/LOD helpers (terrainHD :156, treesSDF, rocksSDF) read the same
// latitude band as this pass's main(), which already samples the block
// centre via fc = gl_FragCoord.xy * 4.0; gPixAng = panoFootAt(fc.y).
// Without the scale the entire cone pass indexes the bottom-quarter band
// of the strip whenever W < 2H (port/reports/review_p4.md M1); the
// trace/kind passes keep the unscaled row lookup on the #else side.
#define uPixAng panoFootAt(gl_FragCoord.y * 4.0)   // FIXME(port) cone rows are quarter-res
#else
#define uPixAng panoFootAt(gl_FragCoord.y)   // FIXME(port) per-texel footprint, not a constant
#endif
#endif

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
float terrainHDN(vec2 p, float detail) {
    float h = terrainHN(p);
    h += rockRelief(p, h);
    if (detail > 0.001) h += detail * microRelief(p);
    return h;
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

float tnoise3(vec3 p) { return texture(uNoise3, p).r; }
vec4 tnoise3v(vec3 p) { return texture(uNoise3, p); }

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
