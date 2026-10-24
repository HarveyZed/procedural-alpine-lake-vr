#version 450
// Bakes the analytic height into a float texture.
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

float u2f(uint h) { return float(h >> 8u) * (1.0 / 16777216.0); }
float hash12i(ivec2 c) { return u2f(hash2u(c)); }
vec2 hash22i(ivec2 c) { uint h = hash2u(c); return vec2(u2f(h), u2f(hashu(h))); }

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

// 3D value noise [0,1]

float fbm2(vec2 p, int oct) {
    float s = 0.0, a = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < oct; i++) { s += a * gnoise(p); p = m * p; a *= 0.5; }
    return s;
}

// 2D Voronoi: returns (F1 distance, F2 distance, cell id hash)

// voronoi with the offset to the nearest feature point: (dx, dy, F1, id)

mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }

float smin(float a, float b, float k) { float h = saturate(0.5 + 0.5 * (b - a) / k); return mix(b, a, h) - k * h * (1.0 - h); }

// ---------------------------------------------------------------------------
// heightfn.glsl : the analytic alpine world (bake-time only, cost irrelevant)
// World units are meters. y = 0 is the main lake surface. x east, z north.
// The main lake sits in a glacial trough running north-south; the trough
// closes into a cirque at the north head and opens to a lower valley south.
// A hanging valley with a small tarn (y=520) notches the west wall.
// ---------------------------------------------------------------------------
#define HEAD_Z 1550.0
#define DLO_NEAR 0.40
#define DLO_FAR 0.40
#define ER_OCT 4
#define ER_AMP 34.0
#define MASSIF_AMP 1.0
#define FAR_OCT 7
#define FAR_CRAG 70.0
#define TARN_C vec2(-1640.0, -430.0)
#define TARN_LEVEL 520.0
#define HV_A vec2(-900.0, -300.0)
#define HV_B vec2(-3500.0, -850.0)     // (r4 M5: -2750 / -720 - the cirque headwall stood 1.7-1.9 km from the tarn, no skyline)

float tanhs(float x) { float e = exp(-2.0 * abs(x)); return sign(x) * (1.0 - e) / (1.0 + e); }
float softplus(float x, float k) { return k * log(1.0 + exp(clamp(x / k, -60.0, 60.0))); }

float axisX(float z) { return 150.0 * sin(z * 0.00105 + 0.5) + 45.0 * sin(z * 0.0029 + 2.1); }

// ridged multifractal split into a low (massif) part and a high (crag) part.
// detail is suppressed in the valleys of the previous octave.
// (session 6: dlo = amplitude decay of the low (massif) octaves; 0.48 = the original. A faster decay
// gives fewer, bigger summits with planar faces instead of a comb of equal sub-peaks)
vec2 ridged2d(vec2 p, int oct, int split, float dlo) {
    float lo = 0.0, hi = 0.0, a = 0.5, prev = 1.0, nlo = 0.0, nhi = 0.0;
    mat2 m = mat2(1.62, 1.18, -1.18, 1.62);
    for (int i = 0; i < oct; i++) {
        float n = 1.0 - abs(gnoise(p) * 1.45);
        n = n * n;
        float v = n * a * (0.3 + 0.7 * prev);
        if (i < split) { lo += v; nlo += a; } else { hi += v; nhi += a; }
        prev = saturate(n * 1.3);
        p = m * p + vec2(13.7, 7.3);
        a *= i < split ? dlo : 0.48;
    }
    return vec2(lo / nlo, hi / max(nhi, 1e-4));
}

vec2 warp(vec2 p) {
    return p + 420.0 * vec2(fbm2(p * 0.00032 + vec2(3.1, 7.7), 4), fbm2(p * 0.00032 + vec2(8.3, 1.9), 4));
}

// (valleyA) concave wall profile: forestable lower slopes (a gentler parabola, ~20-28 deg, ~450 m at
// ~1.1 km from the shore), then a steep rock wall (softplus term, ~40-45 deg) up to the crest; the ridged
// massif / crag relief starts on the rock wall (spurs no longer reach the water)
#define VA_LOW 40.0        // lower-slope parabola (m per s^2 above the shore, s = 1)
// (r4 M1: 3.9 / 220 / 1350 -> 3.0 / 340 / 2000: the side walls were low forested hills (800-1150 m within 2 km of the
// axis); target: bare walls rising 1500-2000 m within 2-3 km, forest on the lowest 15-35 %)
#define VA_ROCK_S 3.0      // s where the steep rock wall starts (it1-3: 3.5, r3: 3.9)
#define VA_ROCK_A 340.0    // its extra rise per unit s (r3: 220)
#define VA_CAP 2000.0      // soft crest limit of the trough profile (r3: 1350)
// 1 = the inflow valley continues north as a V-valley. (it1: the V started right above the
// cascade with a nearly flat floor (~330-350 m from z 2300 to 2600): the erosion cut the cascade into a
// 250 m gorge (the stream path moved ~300 m west and lost its cascade) and the head summits' pyramid faces
// stood as a 1300 m sheer wall at its end. it2: the old cirque geometry is kept up to z ~2450 (the
// stream's top is at z ~2270), the V floor rises steadily (~0.2 m/m, no flat reach, no lower-slope
// noise on it) and the head pyramids do not emerge from the V floor (macroRelief)). it2 result: the
// erosion again cut the cascade down (axis z 2200: 66 m, baseline 161; z 2400: 94 / 288) and the stream
// tracker followed the axis channel (NW route, bed 58-74 m from z 1985 to 2315, no cascade): the larger
// V catchment draining through the cascade is the cause, not the raw shape -> off (it3)
#define VA_NORTH 0
// (valleyA it3) profile power of the three head pyramids (was 1.15 / 1.2 / 1.2 = planar faces): concave
// flanks - the summits keep their height and position, the steep rock is concentrated near the top and
// the lower flanks fall back toward the cirque terrain (pointed summits on a lower plinth)
#define VA_PK_K 1.85       // (it3-4: 1.6)
#define VA_HEAD2 3900.0    // z of the V-valley's head (the trough closes radially beyond it)
// (valleyA it2) distinct summits: between the summits of a ~2.5 km summit field the upper rock wall is
// compressed toward VA_SAD_H by this fraction (pointed summits with deep notches, sky between)
#define VA_SAD_K 0.6       // (it2: 0.5)
#define VA_SAD_H 650.0
#define VA_HSAD_K 0.8      // head (macroRelief): the crest between the three head summits, compressed above
#define VA_HSAD_H 1450.0   // this height by this fraction (was 0.55 above 2050 m)
#define VA_CSAD_K 0.75     // (it5) post-erosion saddle compression of the cirque bowl walls (it4: VA_SAD_K 0.6)
float heightPre(vec2 p) {
    vec2 q = warp(p);
    float z = q.y;
    float ax = axisX(z);
    float dx = q.x - ax;
    float dzh = max(z - HEAD_Z, 0.0);
    float hw = 440.0 + 60.0 * sin(z * 0.0021 + 1.0) + 35.0 * sin(z * 0.0052 + 0.3);
    hw += 300.0 * smoothstep(-1900.0, -4800.0, z);
    float sOld = length(vec2(dx, dzh * 1.05)) / hw;
    float s = sOld;
    float vFloor = 0.0;

    // U-shaped trough, soft-limited crest height; (valleyA) concave above the shore
    float s2 = s * s - 1.0;
    float base = s2 < 0.0 ? 46.0 * s2
               : VA_LOW * s2 + VA_ROCK_A * softplus(s - VA_ROCK_S, 0.3);
    base = base > 0.0 ? VA_CAP * tanhs(base / VA_CAP) : base;
    // the hanging valley's mouth, the tarn's east bench and its meadow (shot 5) keep the old wall
    vec2 kA = vec2(-1150.0, -440.0), kB = vec2(-2000.0, -450.0);
    float kt = clamp(dot(p - kA, kB - kA) / dot(kB - kA, kB - kA), 0.0, 1.0);
    float keepHV = 1.0 - smoothstep(320.0, 720.0, length(p - kA - (kB - kA) * kt));
    // ... and so does the inflow plain / the stream's cascade / the cirque floor (the erosion re-routes the
    // stream on any change of the floor relief there)
    float keepN = (1.0 - smoothstep(350.0, 650.0, abs(dx))) * smoothstep(1300.0, 1500.0, z) * (1.0 - smoothstep(2700.0, 3100.0, z));
    // (it4) ... and so does the whole cirque bowl above the stream's top (its catchment): in bake2 / bake3 the
    // lower (saddle-compressed / concave) cirque walls let the erosion cut the cascade ~150-200 m down and
    // the stream took a NW route without a cascade; its saddles are cut after the erosion (macroRelief)
    float keepC = smoothstep(1700.0, 2200.0, z) * (1.0 - smoothstep(2400.0, 3200.0, abs(dx)));
    float keepOld = max(max(keepHV, keepN), keepC);
    float baseOld = -46.0 + 46.0 * sOld * sOld;
    baseOld = baseOld > 0.0 ? 1450.0 * tanhs(baseOld / 1450.0) : baseOld;
    base = mix(base, baseOld, keepOld);
    // floor level along the valley: lake section at 0, morainic sill closing the
    // lake to the south then a descending lower valley; inflow plain to the north
    float floorZ = -0.085 * softplus(-2000.0 - z, 300.0)
                 + 78.0 * exp(-pow((z + 1520.0) / 210.0, 2.0)) * (1.0 - smoothstep(1.2, 2.2, s))
                 + 44.0 * smoothstep(1050.0, 1650.0, z);
    float floorOld = 540.0 * smoothstep(1750.0, 3300.0, z);   // cirque head rises
    floorZ += floorOld;

    // (valleyA) a little relief on the lower slopes (knobs, spurs), the full massif from the rock wall up
    float mS = (0.2 * smoothstep(1.2, 3.0, s)) * (1.0 - vFloor) + 0.8 * smoothstep(3.3, 5.6, s);   // (it1-3: 2.9 .. 5.4)
    mS = mix(mS, smoothstep(1.1, 4.4, sOld), keepOld);
    float massif = 1050.0 * mS + 280.0 * smoothstep(1700.0, 3200.0, z) * mix(smoothstep(1.6, 3.2, s) * (1.0 - vFloor), 1.0, keepOld);
    // (session 6: 8 octaves - the two finest (~11 m / ~5 m) were below the 4 m grid's Nyquist and
    // only added needle noise to the crests)
    vec2 R = ridged2d(q * 0.00036 + vec2(1.3, 4.1), 8, 4, DLO_NEAR);
    float h = base + floorZ + massif * (R.x - 0.42);
    // crags only on the main crests, moderate amplitude: 3-6 distinct summits per view and planar
    // faces 300-800 m across instead of a comb of 20-30 needles (session 6: 170 -> 90 m, gated by the
    // massif ridge value R.x 0.55..0.8 instead of 0.3..0.65)
    // (r4 M1: gate 650 / 1350 -> 500 / 1100 m and 160 m on the side walls (s < ~5): serrated wall crests against the sky)
    float high = mix(smoothstep(500.0, 1100.0, h), smoothstep(650.0, 1350.0, h), keepOld);
    h += (R.y - 0.35) * mix(mix(160.0, 90.0, smoothstep(4.5, 5.5, s)), 90.0, keepOld) * high * mix(smoothstep(3.2, 5.0, s), smoothstep(0.9, 3.0, s), keepOld) * smoothstep(0.55, 0.8, R.x);
    // (valleyA it2) distinct summits: a ~2.5 km summit field (sf 1 at the summits); between them the
    // upper rock wall is compressed toward VA_SAD_H, so the skyline seen from the shore drops into saddles
    // and notches while the summit heights stay. Not on the lower slopes (s < 2.4, and below VA_SAD_H
    // anyway), not around the hanging valley / tarn cirque (shot 5) or the inflow cascade.
    {
        float sf = smoothstep(-0.02, 0.2, fbm2(p * 0.00042 + vec2(5.3, 1.7), 2));
        vec2 hab = HV_B - HV_A;
        float dHV = length(p - HV_A - hab * clamp(dot(p - HV_A, hab) / dot(hab, hab), 0.0, 1.0));
        float sadW = VA_SAD_K * (1.0 - sf) * smoothstep(2.4, 3.6, s) * (1.0 - keepOld) * smoothstep(800.0, 1500.0, dHV);
        h -= sadW * max(h - VA_SAD_H, 0.0);
    }

    // distant big ranges (seen down-valley to the south and over the ridges)
    float far = smoothstep(5000.0, 9500.0, length(p));
    vec2 R2 = ridged2d(q * 0.00017 + vec2(7.0, 2.0), FAR_OCT, 4, DLO_FAR);
    // a few dominant massifs instead of a comb of equal serrations: halved crag amplitude and a
    // ~20 km modulation of the whole far relief
    // (session 5: pow 1.5 on the massif term, wider dom range, crags 250 -> 120 m)
    float dom = 0.30 + 1.1 * smoothstep(-0.15, 0.25, fbm2(p * 0.00005 + 3.0, 2));
    h += far * dom * (1100.0 * (pow(max(R2.x, 0.0), 1.5) - 0.24) + FAR_CRAG * (R2.y - 0.3) * smoothstep(800.0, 1800.0, h));
    // (session 6) 2-3 dominant glaciated massifs (~12 km mask) standing 600-900 m above the rest of
    // the far ranges (shot 6: a few big snowy massifs, not an even comb of equal spikes); the
    // lift follows the ridge value so the massif keeps its faces and does not become a raised plateau
    float mM = smoothstep(0.35, 1.3, fbm2(p * 0.000083 + vec2(11.2, 4.9), 2) / 0.125);
    h += far * mM * MASSIF_AMP * (330.0 + 520.0 * saturate(R2.x * 1.5 - 0.15));

    // shoreline irregularity: bays, points, a few rocky promontories
    float nearShore = 1.0 - smoothstep(1.3, 2.4, s);
    h += nearShore * (26.0 * fbm2(p * 0.0058 + 11.0, 4) + 8.0 * fbm2(p * 0.021 + 3.0, 3));

    // (r4 M9) the lower valley beyond the moraine (z < -2500): a flat ~600 m floodplain (2-4 m terrace relief) instead of
    // the U-trough's noisy floor (target: a braided river through meadows and forest patches; ours was one dark mush)
    {
        float fpW = (1.0 - smoothstep(0.40, 0.80, s)) * smoothstep(-2500.0, -3300.0, z);
        float fl = floorZ - 43.0 + 1.5 * fbm2(p * 0.004 + 5.0, 2) / 0.125 + 2.0 * smoothstep(0.15, 0.40, s);
        h = mix(h, fl, fpW);
    }

    // ---- hanging valley on the west wall (U profile, floor rising to a cirque)
    vec2 ab = HV_B - HV_A;
    float u = dot(p - HV_A, ab) / dot(ab, ab);
    vec2 cp = HV_A + ab * clamp(u, 0.0, 1.0);
    float dh = length(p - cp);
    float uc = clamp(u, 0.0, 1.0);
    float hvFloor = 548.0 + 50.0 * uc + 650.0 * smoothstep(0.75, 1.15, u);   // (r4 M5: 0.62 / 1.08)
    float hv = hvFloor + 0.00105 * dh * dh * (1.0 + 0.4 * gnoise(p * 0.003));
    hv += 18.0 * fbm2(p * 0.006 + 2.0, 3);
    float hvMask = smoothstep(-0.25, 0.02, u);
    h = mix(h, smin(h, hv, 70.0), hvMask);
    return h;
}

// gully / couloir pattern aligned with the slope (after Clay John's erosion noise)
vec3 erosionCell(vec2 p, vec2 dir) {
    vec2 ip = floor(p); vec2 fp = p - ip; ivec2 c = ivec2(ip);
    vec3 va = vec3(0.0); float wt = 0.0;
    for (int i = -2; i <= 1; i++)
    for (int j = -2; j <= 1; j++) {
        ivec2 o = ivec2(i, j);
        vec2 hh = hash22i(c - o) * 0.5;
        vec2 pp = fp + vec2(o) - hh;
        float d = dot(pp, pp);
        float w = exp(-d * 2.0);
        wt += w;
        float mag = dot(pp, dir);
        va += vec3(cos(mag * TAU), -sin(mag * TAU) * dir) * w;
    }
    return va / wt;
}

float erosion(vec2 p, vec2 g, float cellSize, int oct) {
    float slope = length(g);
    vec2 dir0 = vec2(g.y, -g.x) / max(slope, 1e-5);
    vec3 acc = vec3(0.0);
    float a = 0.5, f = 1.0 / cellSize, norm = 0.0;
    for (int i = 0; i < oct; i++) {
        vec2 dir = dir0 * 1.6 + acc.zy * vec2(1.0, -1.0) * 2.4;
        vec3 e = erosionCell(p * f, dir);
        acc += e * a;
        norm += a;
        a *= 0.5; f *= 2.0;
    }
    return acc.x / norm; // ~[-1,1], negative = gully
}

// complete coarse height (features >= ~8 m)
float heightCoarse(vec2 p) {
    float hs[5];
    vec2 offs[5] = vec2[5](vec2(0.0), vec2(12.0, 0.0), vec2(-12.0, 0.0), vec2(0.0, 12.0), vec2(0.0, -12.0));
    for (int k = 0; k < 5; k++) hs[k] = heightPre(p + offs[k]);
    float h0 = hs[0];
    vec2 g = vec2(hs[1] - hs[2], hs[3] - hs[4]) / 24.0;
    float slope = length(g);
    float above = saturate(h0 / 40.0);
    // gullies, couloirs, avalanche chutes: strongest on steep slopes
    // (session 6: 6 -> 4 octaves, 48 -> 34 m: the 14 m / 7 m gully octaves sat at the 4 m grid's limit
    // and, with the full amplitude, combed every wall into the same dense feather of parallel rills)
    // (r4 M2 a) gullies along a regional joint direction (+-43 deg off the fall line, ~2 km patches) and shallower on the
    // steep walls (ER_AMP x0.53 above slope ~0.9): every gully / rib ran straight down the fall line - organ pipes
    // (head-wall ex/ey 1.6-1.9, target 0.65-0.97)
    // (not around the inflow stream / cascade and the hanging-valley mouth: heightPre's keepN / keepHV zones)
    float kpS;
    {
        vec2 q = warp(p);
        float dxk = q.x - axisX(q.y);
        float kN = (1.0 - smoothstep(350.0, 650.0, abs(dxk))) * smoothstep(1300.0, 1500.0, q.y) * (1.0 - smoothstep(2700.0, 3100.0, q.y));
        vec2 kA = vec2(-1150.0, -440.0), kB = vec2(-2000.0, -450.0);
        float kt = clamp(dot(p - kA, kB - kA) / dot(kB - kA, kB - kA), 0.0, 1.0);
        kpS = 1.0 - max(kN, 1.0 - smoothstep(320.0, 720.0, length(p - kA - (kB - kA) * kt)));
    }
    vec2 gR = rot2(0.75 * kpS * clamp(fbm2(p * 0.0005 + 3.0, 2) / 0.05, -1.0, 1.0)) * g;
    float er = erosion(p, gR, 220.0, ER_OCT);
    float h = h0 + er * ER_AMP * mix(1.0, 18.0 / 34.0, smoothstep(0.75, 1.0, slope) * kpS) * smoothstep(0.12, 0.9, slope) * above;
    float er2 = erosion(p + 37.0, gR, 48.0, 4);
    h += er2 * 5.0 * smoothstep(0.5, 1.4, slope) * above;
    // cliff bands: steep rock above the forest is stepped into ledges and risers
    float steepW = smoothstep(0.9, 1.5, slope) * smoothstep(220.0, 600.0, h0);
    if (steepW > 0.0) {
        float S = 34.0 + 12.0 * gnoise(p * 0.004);
        float wv = 0.6 * gnoise(p * 0.0075 + 4.0);
        float tt = h / S + wv;
        float f = fract(tt);
        float ht = (floor(tt) + smoothstep(0.22, 0.78, f) - wv) * S;
        h = mix(h, ht, steepW * mix(0.32, 0.5, smoothstep(800.0, 1000.0, h0)));
    }

    // ---- tarn basin inside the hanging valley: irregular lobed shoreline, steeper west side
    vec2 tp = p - TARN_C;
    tp += 85.0 * vec2(fbm2(p * 0.0042, 4), fbm2(p * 0.0042 + 5.0, 4));
    float ang = atan(tp.y, tp.x);
    float lobes = 1.0 + 0.16 * sin(ang * 3.0 + 1.3) + 0.09 * sin(ang * 5.0 + 0.4);
    float r = length(tp * vec2(1.0, 0.85)) / lobes;
    float west = saturate(0.5 - 0.5 * tp.x / max(length(tp), 1.0));
    // (r4 M6: west exponent 2.6 -> 1.7, fade 240..430 -> 260..900 m, smin 30 -> 110: the bowl's rim crease at r ~430-460 m
    // stood as a 20-40 m pale "dam" wall on the west shore and a contour-parallel "road" bench 160 m above the tarn)
    // (on the west side only: on the east the wider / softer bowl lowered the sill below the tarn level - 258 ha)
    float wW = smoothstep(0.5, 0.85, west);
    float bowl = TARN_LEVEL - 18.0 + 18.0 * pow(r / 205.0, mix(1.5, mix(2.6, 1.7, wW), west));
    float tw = 1.0 - smoothstep(mix(240.0, 260.0, wW), mix(430.0, 900.0, wW), r);
    h = mix(h, smin(h, bowl, mix(30.0, 110.0, wW)), tw);

    // ---- lake bed shaping: shallow shelf, then a turquoise drop-off
    if (h < 0.0) {
        float d = -h;
        d = 0.30 * d + 0.95 * max(d - 4.0, 0.0);
        h = -d;
    }
    return h;
}

// fine scale detail 2..8 m (added in the near bake)
float heightFine(vec2 p, float slope) {
    float rock = smoothstep(0.35, 1.0, slope);
    float n = 0.0;
    n += 0.60 * gnoise(p * 0.16 + 3.1);
    n += 0.30 * gnoise(rot2(0.7) * p * 0.37 + 7.7);
    n += 0.14 * gnoise(rot2(1.9) * p * 0.81 + 1.3);
    return n * mix(0.5, 2.4, rock);
}

// ---------------------------------------------------------------------------
// macro relief (round 2; bake.relief_post pass 1, on the eroded far map). The eroded ridged-noise
// massif is statistically uniform: every crest is a level comb of equal summits (shot 7, shot 6
// far range). A few dominant pyramidal massifs are grown out of it - polygonal cones (planar faces,
// aretes at the corners) combined with the macro terrain by a smooth max, so saddles stay at the old
// crest level and the summits stand 400-900 m above it. Where a massif emerges, the old 40-400 m relief
// (the comb of spurs and sub-peaks, oriented to the old slopes) is mostly removed; the relief pass
// adds new fall-line ribs / couloirs / cliff bands on the new faces.
//   h = eroded height, L = its ~400 m blur, B = its ~40 m blur (bake.relief_post, numpy)
// ---------------------------------------------------------------------------

// distance to the lake capsule / tarn: nothing below changes within ~2 km of the lake or the tarn

// regular N-gon "radius": planar faces normal to rot + k * TAU / N, aretes between them

// pyramid surface: summit Hs, drops by A over the radius R with a concave (power k) profile

#define NPK 19
// x, z, summit height, radius | faces N, rotation (rad), profile power, drop A | east-west stretch
// (session 4: the down-valley range read as a row of 4-5 equal pyramids (shot 6): now ONE dominant
// glaciated massif (3300 m, broad, x2 east-west) with neighbours ~1000 m lower, saddles at the plinth
// level (<= ~1950 m), N varied 3..5, and a lower ~1900 m layer at z ~ -13 km for depth, either side of
// the lower valley so the view down it stays open)
// (r4 C1) the head: three UNEQUAL summits (3100 / 2550 / 2650 m), irregular spacing (x 500 / -2700 / 2100), the west one
// set back (z 5800), the east one a long E-W ridge (stretch 1.6), N 4 / 5 / 4 and profile powers 1.3 / 1.6 / 1.45 (was
// three near-equal concave shark teeth 2850-3100 m, k 1.85, ~2.3 km apart); joined by a serrated crest (macroRelief)
// (r4 M8) the down-valley range lower and smaller (dominant massif 3300 -> 2500 m, the others 2150-2250 -> 1850-1950)
// (r4 M1 / M5, index 12+) side-wall summits (head group: pW-gated) - 3 on the east wall, 4 around the hanging valley /
// tarn cirque (the M1 and M5 lists merged where they overlapped)
const vec4 PKA[NPK] = vec4[NPK](
    vec4(  500.0,   4700.0, 3100.0, 3000.0),   // head: main horn (shots 0/7), big south face
    vec4(-2700.0,   5800.0, 2550.0, 2600.0),   // head: west summit, set back
    vec4( 2100.0,   4300.0, 2650.0, 2400.0),   // head: east summit, a long ridge
    vec4(-5900.0, -16384.0, 1900.0, 4200.0),   // down-valley range (shot 6), on the map edge
    vec4(  700.0, -16384.0, 2500.0, 6500.0),   //   the dominant massif (the far map is mirrored there:
    vec4( 7600.0, -16384.0, 1950.0, 4400.0),   //   only the north half of each massif is real)
    vec4(-10800.0, -16384.0, 1850.0, 4200.0),
    vec4(12000.0, -16384.0, 1900.0, 4200.0),
    vec4(-3000.0, -14700.0, 1850.0, 2800.0),   //   fore-summits: saddles at different levels,
    vec4( 4600.0, -15000.0, 1950.0, 3000.0),   //   no symmetric pair of cones
    vec4(-5400.0, -12900.0, 1700.0, 2500.0),   //   lower layer (depth), clear of the valley axis
    vec4( 6000.0, -12500.0, 1750.0, 2600.0),
    vec4( 2900.0,  -2200.0, 2300.0, 1600.0),   // east wall (M1)
    vec4( 3400.0,   -300.0, 1950.0, 1500.0),   // (b2: 3100 / -200 / 2150 and 2700 / 1600 / 2400 stood 28-30 deg over the
    vec4( 3500.0,   1900.0, 1950.0, 1500.0),   //  r1 / shot 0 shore - a dark wall hiding the head; target ~15 deg)
    vec4(-3000.0,  -1800.0, 2350.0, 1700.0),   // west wall, south of the hanging valley (M1)
    vec4(-3700.0,  -1250.0, 2150.0, 1000.0),   // tarn cirque crest (M5)
    vec4(-4000.0,   -450.0, 2250.0, 1100.0),
    vec4(-3300.0,    350.0, 2050.0, 1000.0));
const vec4 PKB[NPK] = vec4[NPK](
    vec4(4.0, -1.5708, 1.30, 2900.0),
    vec4(5.0,  0.35,   1.60, 2000.0),
    vec4(4.0, -2.618,  1.45, 2100.0),
    // (far faces turned toward the sun, az 118: a lit ENE face and a shaded NNW face toward the lake)
    vec4(4.0,  0.30,   1.5, 3000.0),
    vec4(5.0,  0.52,   1.2, 4400.0),     // (A 5200: the lower valley stays open to z ~ -11.5 km)
    vec4(3.0,  0.70,   1.5, 3050.0),
    vec4(5.0,  0.15,   1.5, 2950.0),
    vec4(4.0,  0.60,   1.5, 3000.0),
    vec4(3.0,  0.95,   1.4, 3000.0),     // (their feet reach the -900 m lower valley: a smaller drop
    vec4(5.0,  0.20,   1.4, 3150.0),     //  left a flat 400 m plateau damming the valley)
    vec4(4.0,  0.40,   1.4, 2800.0),
    vec4(3.0,  1.10,   1.4, 2850.0),
    vec4(3.0,  0.40,   1.6, 1450.0),
    vec4(4.0,  1.10,   1.6, 1350.0),
    vec4(3.0, -0.30,   1.6, 1500.0),
    vec4(4.0,  0.80,   1.6, 1500.0),
    vec4(3.0,  0.20,   1.4, 1100.0),
    vec4(3.0, -0.70,   1.4, 1200.0),
    vec4(3.0,  1.30,   1.4, 1000.0));
const float PKS[NPK] = float[NPK](1.0, 1.0, 1.6, 1.45, 2.0, 1.45, 1.45, 1.45, 1.3, 1.3, 1.45, 1.45,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

// ---------------------------------------------------------------------------
// post-erosion relief (bake.relief_post, applied to the eroded far map before the hand-made
// features are carved). The hillslope erosion relaxes every face to a smooth plane at its
// critical angle; granite walls are split into fall-line ribs / couloirs and
// cliff bands with ledges. Only above ~300 m, away from the tarn and the inflow stream.
//   h      eroded height at p, g = its gradient over +-32 m, w = protection weight (0..1),
//   crest  0..1 near the local summit / ridge line (within ~60 m of the highest point 200 m around)
// returns the new height; dbg = (couloir term, ledge term, steep weight, 0)
// ---------------------------------------------------------------------------
// erosion() output has p10/p90 ~ +-0.31: normalise, then V-shaped gullies and broad round ribs (zero mean)

// (world3) a second, smaller facet layer (~75 m) on the big faces: a 3D Voronoi of tilted slabs evaluated at
// the surface point (x, h, z) - round cells ON the face at any slope (2D x-z cells hang 2-4x long down a steep
// face: organ pipes) - the surface the smooth max of flat caps (crisp V creases between the slabs), tilts from
// three joint families of the region (parallel slabs repeat: coherent lit / shaded planes). Height change
// <= 0. The run-time relief (world.glsl rockRelief) adds the 42 / 15 / 5 m octaves of the same structure.
// (r4 M2 c) sa = strike angle of the face (atan of its contour direction): the two lateral joint families F0 / F1 at
// +-40 deg to strike (diagonal joint-controlled creases / ledges instead of a regional, slope-blind direction)

// port(P1): the engine's loose pass uniforms, grouped into one block for Vulkan GLSL (unnamed instance -> members stay global)
layout(std140, set=0, binding=0) uniform _p1_pass {
    vec2 uCenter;
    float uHalf;
    float uRes;
    int uFine;
};




          // 1 = add fine detail (near map)
layout(location=0) out vec4 fragColor;
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
