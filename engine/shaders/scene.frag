// ---------------------------------------------------------------------------
// scene.frag : the full alpine-lake renderer (one sample per pixel per pass)
// ---------------------------------------------------------------------------
#include "common.glsl"
#include "world.glsl"
#include "sky.glsl"
#include "water.glsl"
#include "trees.glsl"
#include "rocks.glsl"
#include "lighting.glsl"
#include "falls.glsl"
#include "ground.glsl"
#include "hero.glsl"
#include "shade_rocks.glsl"
#include "shade_trees.glsl"
#include "terrain_shade.glsl"

vec3 shadeHero(vec3 p, vec3 rd, float t, int i, bool cheap) {
    float e = max(0.006, t * gPixAng * 1.2);
    vec3 n = normalize(vec3(sdHero(p + vec3(e, 0, 0), i) - sdHero(p - vec3(e, 0, 0), i),
                            sdHero(p + vec3(0, e, 0), i) - sdHero(p - vec3(0, e, 0), i),
                            sdHero(p + vec3(0, 0, e), i) - sdHero(p - vec3(0, 0, e), i)) + 1e-6);
    vec3 alb = rockAlbedo(p, n, t);
    vec3 q = p - uHeroA[i].xyz;
    // salt-and-pepper granite grain, lichen rosettes, darker weathered base
    float g = texture(uNoise3, q * 3.1).r;
    alb *= mix(0.82, 1.12, smoothstep(0.3, 0.7, g));
    float lich = smoothstep(0.66, 0.72, texture(uNoise3, q * 0.7 + 4.0).a) * saturate(n.y + 0.4);
    alb = mix(alb, vec3(0.38, 0.40, 0.27), lich * 0.55);
    float ground = p.y - terrainH(p.xz, true);
    alb *= mix(0.6, 1.0, smoothstep(0.0, 0.6, ground));
    float wl = waterLevel(p.xz);
    float wet = 1.0 - smoothstep(0.0, 0.3, p.y - wl);
    float fwH = fallsWetness(p);
    if (fwH > 0.0) {
        float mN = texture(uNoise3, p * 0.3).r;
        alb = mix(alb, alb * 0.38, fwH);
        alb = mix(alb, mix(vec3(0.04, 0.06, 0.012), vec3(0.10, 0.12, 0.02), mN), fwH * smoothstep(0.25, 0.7, n.y + 0.4 * (mN - 0.5)) * 0.9);
        wet = max(wet, fwH * 0.6);
    }
    alb *= mix(1.0, 0.5, wet);
    float ao = terrainAO(p.xz) * mix(0.35, 1.0, smoothstep(0.0, 0.8, ground)) * mix(0.6, 1.0, n.y * 0.5 + 0.5);
    float sun = sunShadowMap(p + n * 0.05) * cloudShadow(p);
    if (!cheap && sun > 0.0) sun *= treeShadow(p + n * 0.1);
    return litSurface(alb, n, rd, sun, ao, mix(0.03, 0.3, wet), mix(0.5, 0.2, wet));
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

#include "water_shade.glsl"

// ---------------------------------------------------------------------------
// split renderer (engine/render.py Renderer): compiled as ONE program, the whole renderer cost the AMD
// compiler ~45-50 GB of commit memory and 13-36 min per fresh compile, while each part alone compiles in
// 1-210 s with < 1 GB (tools/split_probe.py, 2026-10-18). So every sample runs, per tile:
//   SCENE_PASS == 99 : the primary trace -> G-buffer: fragColor = (t, kind, cell), fragInfo = info
//   SCENE_PASS == K  : one program per hit kind K (K_SKY .. K_FALLSV): shades the pixels whose primary hit
//                      is of kind K (+ falls water volume, aerial perspective) into the accumulator and
//                      discards the others, so each pixel gets exactly one contribution per sample
// ---------------------------------------------------------------------------
#ifndef SCENE_PASS
#error scene.frag needs SCENE_PASS: 99 = primary trace, else the hit kind (K_*) to shade
#endif

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

#if SCENE_PASS == 98
// (perf4) cone pre-pass (render.py, before the primary trace of each tile): one cone per 4 x 4 pixel block, wide
// enough for every primary ray of the block (any jitter, the lens), marched through empty space with the trace's own
// distance bounds (terrain / water x 0.42, the boulder / cover zone <= 4.5 m above the ground counted as surface, the
// trees' and hero blocks' SDFs) until a surface comes within ~2 cone radii; the block's primary rays start there
// (the empty spheres along the cone hold every one of them). Output: that distance.
float coneTrace(vec3 ro, vec3 rd, float th) {
    float t = 0.05, tSafe = 0.05;     // (qa5 fix) last cone position known to be outside
    const float sk = 2.0;      // stop: a surface within sk cone radii
    for (int i = 0; i < 400; i++) {
        vec3 p = ro + rd * t;
        if (p.y > MAX_TERRAIN_H && rd.y >= 0.0) break;
        float r = t * th + uAperture * (1.0 + t / uFocusDist);
        float det = detailWeight(t);
        float h = terrainHD(p.xz, det);
        float dyT = p.y - h;
        float D = dyT * 0.42;
        D = min(D, p.y - waterLevel(p.xz));
        // ground cover (<= COVER_MAX_H above the ground, < COVER_MAX_DIST): its zone counts as surface (coverSDF
        // marches its cells along one ray only); boulders (< 4.5 m above the ground): their SDF, below the zone's bound
        if (t < COVER_MAX_DIST + 5.0) D = min(D, (dyT - COVER_MAX_H - 0.05) * 0.42);
        if (D > sk * r && t < ROCK_MAX_DIST + 20.0 && (dyT - 4.5) * 0.42 < D) {
            ivec2 c;
            D = min(D, max((dyT - 4.5) * 0.42, rocksSDF(p, rockLod(t), c)));
        }
        if (D > sk * r && t < TREE_MAX_DIST) {
            float tf = treeFreeDist(p);
            if (tf < D) {
                ivec2 c;
                gTreeDMax = D;
                D = min(D, max(tf, treesSDF(p, treeLod(t), treeLod2(t), c)));
            }
        }
        if (D > sk * r && uHeroN > 0 && t < 700.0) { int hi; D = min(D, heroSDF(p, hi)); }
        if (D < sk * r) { if (D < 0.0) t = tSafe; break; }   // (qa5) inside a surface (steep face): back off
        tSafe = t;
        t += (D - r) / (1.0 + th);
        if (t > 60000.0) break;
    }
    return t;
}
void main() {
    gPixAng = 2.0 * uTanHalfFov / uRes.y;
    vec2 ndc = (2.0 * (gl_FragCoord.xy * 4.0) - uRes) / uRes.y;     // the block's centre
    vec3 rd = uCamRot * normalize(vec3(ndc * uTanHalfFov, 1.0));
    // half-diagonal of the block (2.83 px) + margin
    fragColor = vec4(coneTrace(uCamPos, rd, 3.0 * gPixAng), 0.0, 0.0, 1.0);
}

#elif SCENE_PASS == 99
layout(location = 1) out vec4 fragInfo;
uniform sampler2D uCone;      // (perf4) cone pre-pass: the block's primary rays start there

void main() {
    vec3 ro, rd;
    cameraRay(ro, rd);
    float t0 = max(texelFetch(uCone, ivec2(gl_FragCoord.xy) >> 2, 0).r, 0.05);
    // (qa5 fix) the cone bound (terrain dy x 0.42) is not conservative on the steep falls cliff: never start inside the
    // terrain (trace() would bisect [t0, t0] and return the block-constant t0), and start 2 cone radii early so a ray
    // grazing a tree crown is not counted as a hit at its first step (4-px dashes on branch tops)
    t0 = max(t0 * (1.0 - 6.0 * gPixAng), 0.05);
    for (int k = 0; k < 6 && t0 > 0.05; k++) {
        vec3 q0 = ro + rd * t0;
        if (q0.y > terrainHD(q0.xz, detailWeight(t0))) break;
        t0 = k == 5 ? 0.05 : max(t0 * 0.75, 0.05);
    }
    Hit h = trace(ro, rd, t0, 60000.0, true, true, true, QUALITY == 1 ? 700 : 380, 0.8);
    // falls flora hook (falls.glsl): 3D shrubs / ferns on the waterfall's rock step and banks, marched
    // by ffTrace up to the primary hit (primary rays only), shaded by shadeFlora (SCENE_PASS K_FALLSV)
    vec4 ffI;
    float tFF = ffTrace(ro, rd, h.t, ffI);
    if (tFF < h.t) { h.t = tFF; h.kind = K_FALLSV; h.info = ffI; }
    fragColor = vec4(h.t, float(h.kind), vec2(h.cell));
    fragInfo = h.info;
}

#else
uniform sampler2D uGBuf0;     // (t, kind, cell) of the primary hit (SCENE_PASS 99)
uniform sampler2D uGBuf1;     // its info

vec3 render(Hit h, vec3 ro, vec3 rd) {
    if (uDebug == 1) {
        vec3 kc[8] = vec3[8](vec3(0.2, 0.3, 1.0), vec3(0.6, 0.5, 0.4), vec3(0.0, 0.6, 0.0), vec3(0.5, 0.5, 0.5), vec3(0.0, 0.8, 0.9), vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0));
        return kc[h.kind];
    }
#if SCENE_PASS == K_SKY
    return skyWithClouds(ro, rd, 16, QUALITY == 1 ? 72 : 36);
#else
#if SCENE_PASS == K_WATER
    if (uDebug == 2) {
        vec3 p = ro + rd * h.t;
        return vec3(sunShadowMap(vec3(p.x, waterLevel(p.xz) + 0.3, p.z)), cloudShadow(p), fract(waterLevel(p.xz)));
    }
#endif
    vec3 p = ro + rd * h.t;
    vec3 col;
    gPrimary = false;
#if SCENE_PASS == K_WATER
    col = shadeWater(ro, rd, h.t);
#elif SCENE_PASS == K_FALLSV
    col = shadeFlora(p, rd, h.t, h.info);     // falls flora hook
#elif SCENE_PASS == K_TREE
    col = shadeTree(p, rd, h.t, h.cell, false);
#elif SCENE_PASS == K_ROCK
    col = shadeRock(p, rd, h.t, h.cell, false);
#elif SCENE_PASS == K_COVER
    col = shadeCover(p, rd, h.t, h.info);
#elif SCENE_PASS == K_HERO
    col = shadeHeroRock(p, rd, h.t, h.cell.x, false);   // (r3: rocks3 boulder material)
#else
    col = shadeTerrain(p, rd, h.t, false);
#endif
    vec4 fv = fallsVolume(ro, rd, h.t, gRand);
    col = col * fv.a + fv.rgb;
    return applyAerial(col, ro, rd, h.t, 12);
#endif
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
#endif
