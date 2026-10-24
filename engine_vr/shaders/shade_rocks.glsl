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
float rkN(vec3 x) {
    vec3 i = floor(x); vec3 f = x - i; ivec3 c = ivec3(i);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float va = dot(rkGrad3(c), f), vb = dot(rkGrad3(c + ivec3(1, 0, 0)), f - vec3(1, 0, 0));
    float vc = dot(rkGrad3(c + ivec3(0, 1, 0)), f - vec3(0, 1, 0)), vd = dot(rkGrad3(c + ivec3(1, 1, 0)), f - vec3(1, 1, 0));
    float ve = dot(rkGrad3(c + ivec3(0, 0, 1)), f - vec3(0, 0, 1)), vf = dot(rkGrad3(c + ivec3(1, 0, 1)), f - vec3(1, 0, 1));
    float vg = dot(rkGrad3(c + ivec3(0, 1, 1)), f - vec3(0, 1, 1)), vh = dot(rkGrad3(c + ivec3(1, 1, 1)), f - vec3(1, 1, 1));
    return mix(mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y), mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y), u.z);
}

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
#ifdef RK_DBG_TERRAIN
    return vec3(0.25);
#endif
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
#ifdef RK_DBG_TERRAIN
    return n;
#endif
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
#ifdef RK_DBG_SHADE
    cheap = true;
#endif
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
