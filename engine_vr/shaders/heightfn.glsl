// ---------------------------------------------------------------------------
// heightfn.glsl : the analytic alpine world (bake-time only, cost irrelevant)
// World units are meters. y = 0 is the main lake surface. x east, z north.
// The main lake sits in a glacial trough running north-south; the trough
// closes into a cirque at the north head and opens to a lower valley south.
// A hanging valley with a small tarn (y=520) notches the west wall.
// ---------------------------------------------------------------------------
#define HEAD_Z 1550.0
#ifndef DLO_NEAR
#define DLO_NEAR 0.40
#endif
#ifndef DLO_FAR
#define DLO_FAR 0.40
#endif
#ifndef ER_OCT
#define ER_OCT 4
#endif
#ifndef ER_AMP
#define ER_AMP 34.0
#endif
#ifndef MASSIF_AMP
#define MASSIF_AMP 1.0
#endif
#ifndef FAR_OCT
#define FAR_OCT 7
#endif
#ifndef FAR_CRAG
#define FAR_CRAG 70.0
#endif
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
vec2 ridged2(vec2 p, int oct, int split) { return ridged2d(p, oct, split, 0.48); }

vec2 warp(vec2 p) {
    return p + 420.0 * vec2(fbm2(p * 0.00032 + vec2(3.1, 7.7), 4), fbm2(p * 0.00032 + vec2(8.3, 1.9), 4));
}

// (valleyA) concave wall profile: forestable lower slopes (a gentler parabola, ~20-28 deg, ~450 m at
// ~1.1 km from the shore), then a steep rock wall (softplus term, ~40-45 deg) up to the crest; the ridged
// massif / crag relief starts on the rock wall (spurs no longer reach the water)
#ifndef VA_LOW
#define VA_LOW 40.0        // lower-slope parabola (m per s^2 above the shore, s = 1)
#endif
// (r4 M1: 3.9 / 220 / 1350 -> 3.0 / 340 / 2000: the side walls were low forested hills (800-1150 m within 2 km of the
// axis); target: bare walls rising 1500-2000 m within 2-3 km, forest on the lowest 15-35 %)
#ifndef VA_ROCK_S
#define VA_ROCK_S 3.0      // s where the steep rock wall starts (it1-3: 3.5, r3: 3.9)
#endif
#ifndef VA_ROCK_A
#define VA_ROCK_A 340.0    // its extra rise per unit s (r3: 220)
#endif
#ifndef VA_CAP
#define VA_CAP 2000.0      // soft crest limit of the trough profile (r3: 1350)
#endif
#ifndef VA_NORTH
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
#endif
#ifndef VA_PK_K
// (valleyA it3) profile power of the three head pyramids (was 1.15 / 1.2 / 1.2 = planar faces): concave
// flanks - the summits keep their height and position, the steep rock is concentrated near the top and
// the lower flanks fall back toward the cirque terrain (pointed summits on a lower plinth)
#define VA_PK_K 1.85       // (it3-4: 1.6)
#endif
#ifndef VA_HEAD2
#define VA_HEAD2 3900.0    // z of the V-valley's head (the trough closes radially beyond it)
#endif
#ifndef VA_SAD_K
// (valleyA it2) distinct summits: between the summits of a ~2.5 km summit field the upper rock wall is
// compressed toward VA_SAD_H by this fraction (pointed summits with deep notches, sky between)
#define VA_SAD_K 0.6       // (it2: 0.5)
#endif
#ifndef VA_SAD_H
#define VA_SAD_H 650.0
#endif
#ifndef VA_HSAD_K
#define VA_HSAD_K 0.8      // head (macroRelief): the crest between the three head summits, compressed above
#endif                     // (it2-4: 0.7 above 1650 m)
#ifndef VA_HSAD_H
#define VA_HSAD_H 1450.0   // this height by this fraction (was 0.55 above 2050 m)
#endif
#ifndef VA_CSAD_K
#define VA_CSAD_K 0.75     // (it5) post-erosion saddle compression of the cirque bowl walls (it4: VA_SAD_K 0.6)
#endif
float heightPre(vec2 p) {
    vec2 q = warp(p);
    float z = q.y;
    float ax = axisX(z);
    float dx = q.x - ax;
    float dzh = max(z - HEAD_Z, 0.0);
    float hw = 440.0 + 60.0 * sin(z * 0.0021 + 1.0) + 35.0 * sin(z * 0.0052 + 0.3);
    hw += 300.0 * smoothstep(-1900.0, -4800.0, z);
    float sOld = length(vec2(dx, dzh * 1.05)) / hw;
#if VA_NORTH == 1
    // (valleyA) the trough closes around the lake head as before over ~1000 m (inflow plain + the
    // stream's cascade, unchanged up to z ~2450), then the valley continues north, narrowing into a V
    // (the valley head: from VA_HEAD2 the trough closes radially again, below the head summits)
    float dzc = dzh - softplus(dzh - 1000.0, 120.0) + softplus(z - VA_HEAD2, 90.0);
    float hwV = hw * mix(1.0, 0.65, smoothstep(2500.0, 3600.0, z));
    float s = length(vec2(dx / hwV, dzc * 1.05 / hw));
    // the V floor zone (no lower-slope knobs / spurs on it: they dam it into depressions that the
    // erosion's breach step then drains by cutting through the cascade)
    float vFloor = (1.0 - smoothstep(250.0, 520.0, abs(dx))) * smoothstep(2350.0, 2700.0, z) * (1.0 - smoothstep(VA_HEAD2 - 200.0, VA_HEAD2 + 300.0, z));
#else
    float s = sOld;
    float vFloor = 0.0;
#endif

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
#if VA_NORTH == 1
    float keepN = (1.0 - smoothstep(350.0, 650.0, abs(dx))) * smoothstep(1300.0, 1500.0, z) * (1.0 - smoothstep(2300.0, 2550.0, z));
#else
    float keepN = (1.0 - smoothstep(350.0, 650.0, abs(dx))) * smoothstep(1300.0, 1500.0, z) * (1.0 - smoothstep(2700.0, 3100.0, z));
#endif
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
#if VA_NORTH == 1
    // the stream's cascade as before up to z ~2400, then the V-valley floor keeps rising at ~0.2 m/m to
    // the valley head (the old cirque term rose 0.5 m/m and closed the valley); with the V base (~+150-190 m,
    // s ~2.2-2.4 on its axis) the raw axis is ~300 (z 2300), 420 (2550), 540 (3000), 640 (3500), 720 (3900)
    float floorV = 540.0 * smoothstep(1750.0, 3300.0, min(z, 2400.0)) + 0.2 * softplus(z - 2400.0, 120.0);
    floorZ += mix(floorV, floorOld, keepN);
#else
    floorZ += floorOld;
#endif

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
float smaxk(float a, float b, float k) { return -smin(-a, -b, k); }
// distance to the lake capsule / tarn: nothing below changes within ~2 km of the lake or the tarn
float lakeGuard(vec2 p) {
    float zc = clamp(p.y, -1700.0, 2300.0);   // (the inflow stream ends at z ~2270)
    float dL = length(vec2(p.x - axisX(zc), p.y - zc));
    return smoothstep(1200.0, 1900.0, dL) * smoothstep(1300.0, 1900.0, length(p - TARN_C));
}
// regular N-gon "radius": planar faces normal to rot + k * TAU / N, aretes between them
float ngon(vec2 d, float N, float rot) {
    float seg = TAU / N;
    float a = atan(d.y, d.x) - rot;
    float ak = mod(a + 0.5 * seg, seg) - 0.5 * seg;
    return length(d) * cos(ak);
}
// pyramid surface: summit Hs, drops by A over the radius R with a concave (power k) profile
float pyramid(vec2 p, vec2 c, float Hs, float A, float R, float N, float rot, float k, float sx) {
    // (session 3: sx stretches the massif east-west - broad massifs with long summit ridges and planar
    // faces instead of isolated cones; an affine map keeps the faces planar)
    vec2 d = (p - c) * vec2(1.0 / sx, 1.0);
    // irregular faces: the polygon distance is bent by ~1.4 km noise (no perfectly planar geometry, each
    // face its own steepness); multiplicative, so the summit stays a point (an additive bend cut it flat)
    float r = ngon(d, N, rot);
    float u = r / R * (1.0 + 0.28 * gnoise(p * 0.0007 + c * 0.001)) + 0.05 * gnoise(p * 0.0021 + c * 0.003) * saturate(3.0 * r / R);
    // (beyond the radius the face keeps falling, C1: no flat plateau at Hs - A around the massif)
    return Hs - A * (1.0 - pow(saturate(1.0 - u), k)) - A * max(u - 1.0, 0.0) * max(u - 1.0, 0.0);
}
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

#ifdef MACRO_PASS
// the macro pass samples the two blurred maps itself (bake.MACRO_FRAG defines macroL / macroB)
float macroL(vec2 p);
float macroB(vec2 p);
// crest regularisation: along a crest the eroded relief is a comb of equal towers 300-500 m apart
// (shot 7, the far ranges). The 40 m relief is averaged ALONG the crest line (the axis of least
// curvature of the 250 m map), so the cross-section of the arete stays sharp while the comb of
// towers merges into a few long ridges; returns the height change, crest weight in w
float crestComb(vec2 p, float L, float B, out float w) {
    const float r = 160.0;
    float lxp = macroL(p + vec2(r, 0.0)), lxm = macroL(p - vec2(r, 0.0));
    float lzp = macroL(p + vec2(0.0, r)), lzm = macroL(p - vec2(0.0, r));
    float Lxx = (lxp + lxm - 2.0 * L) / (r * r), Lzz = (lzp + lzm - 2.0 * L) / (r * r);
    float Lxz = (macroL(p + vec2(r, r)) - macroL(p + vec2(r, -r)) - macroL(p + vec2(-r, r)) + macroL(p - vec2(r, r))) / (4.0 * r * r);
    // eigenvalues of the Hessian: l1 <= l2; the crest runs along the eigenvector of l2
    float tr = 0.5 * (Lxx + Lzz), df = sqrt(0.25 * (Lxx - Lzz) * (Lxx - Lzz) + Lxz * Lxz);
    float l1 = tr - df, l2 = tr + df;
    vec2 ax = abs(Lxz) > 1e-9 ? normalize(vec2(l2 - Lzz, Lxz)) : (Lxx > Lzz ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    // a sharp crest across (strongly convex), not a summit dome (convex both ways)
    w = smoothstep(0.0007, 0.0018, -l1) * (1.0 - smoothstep(0.45, 0.85, l2 / min(l1, -1e-6))) * smoothstep(1200.0, 1700.0, L);
    if (w <= 0.0) return 0.0;
    float s = 0.0, ws = 0.0;
    for (int i = -4; i <= 4; i++) {
        float k = 1.0 - abs(float(i)) / 5.0;
        s += k * macroB(p + ax * (80.0 * float(i)));
        ws += k;
    }
    return w * (s / ws - B);
}

float macroRelief(vec2 p, float h, float L, float B, out vec4 dbg) {
    float guard = lakeGuard(p);
    float wc;
    float dC = crestComb(p, L, B, wc) * guard;   // (session 3: x0.85 -> x1)
    float P = -1e4, PS = -1e4;            // head massifs (0-2), down-valley range (3+)
    for (int i = 0; i < NPK; i++) {
        vec4 a = PKA[i], b = PKB[i];
        // (the down-valley range: x1.45 broader massifs - at x1 they read as a row of isolated cones)
        float sx = PKS[i];
        if (length((p - a.xy) * vec2(1.0 / sx, 1.0)) > 1.6 * a.w) continue;
        // (r4 C1: own profile powers for the head summits, b.z; the side-wall summits (12+) join the head group)
        float v = pyramid(p, a.xy, a.z, b.w, a.w, b.x, b.y, b.z, sx);
        if (i < 3 || i >= 12) P = max(P, v); else PS = max(PS, v);
    }
    // the down-valley range stands on a plinth (~1700 m) along the south map edge: lower saddles
    // between the massifs instead of separate cones rising out of the valley floor
    float dz = abs(p.y + 16384.0) + 700.0 * fbm2(p * 0.00022 + 3.3, 3);
    float Hp = 1350.0 + 260.0 * clamp(fbm2(p * 0.00016 + 7.1, 3) / 0.125, -1.5, 1.5);   // (r4 M8: 1650)
    PS = max(PS, Hp - 0.55 * max(dz - 900.0, 0.0));
    // head: the cirque floor keeps its shape (no pyramid filling it); the down-valley range rises
    // straight out of the lower valley (it closes the view down the valley)
    float pW = smoothstep(700.0, 1100.0, L) * guard;
#if VA_NORTH == 1
    // (valleyA it2) the head pyramids do not emerge from the V-valley's floor and lower flanks (in it1
    // their south faces stood there as a 1300 m sheer wall): only from ~1300 m up, and further up the
    // nearer the V axis
    float vax = abs(p.x - axisX(p.y));
    float vZ = smoothstep(2300.0, 2800.0, p.y) * (1.0 - smoothstep(VA_HEAD2 + 300.0, VA_HEAD2 + 1300.0, p.y));
    pW *= mix(1.0, smoothstep(1300.0, 1900.0, L) * smoothstep(300.0, 900.0, vax), vZ);
#endif
    // (valleyA it4) the cirque bowl kept its old raw walls (heightPre keepC, for the cascade): its saddles
    // are cut here, after the erosion, with the same summit field (the eroded detail h - L stays)
    float Lc = L;
    {
        float wC = smoothstep(1700.0, 2200.0, p.y) * (1.0 - smoothstep(2400.0, 3200.0, abs(p.x - axisX(p.y)))) * guard;
        float sf = smoothstep(-0.02, 0.2, fbm2(p * 0.00042 + vec2(5.3, 1.7), 2));
        Lc -= VA_CSAD_K * (1.0 - sf) * wC * max(L - VA_SAD_H, 0.0);
    }
    float Pm = mix(Lc, smaxk(Lc, P, 90.0), pW);
    Pm = mix(Pm, smaxk(Pm, PS, 120.0), guard);
    // the head wall (shots 0/7): the comb of equal 300-500 m spurs and crest towers is
    // damped over the whole wall, so the summits and the big couloirs / buttresses read
    float hz = smoothstep(1300.0, 1900.0, L) * smoothstep(2600.0, 3400.0, p.y) * (1.0 - smoothstep(7500.0, 9000.0, p.y))
             * (1.0 - smoothstep(4500.0, 6000.0, abs(p.x - 300.0))) * guard;
    // (session 3) lower saddles between the three head summits: away from them the macro surface above
    // ~2150 m is compressed by 40 % (the crest between them was a level 2300-2450 m comb)
    float nearS = 0.0;
    // (valleyA it2: the summits kept, the saddles between them lowered further: distinct
    // pointed summits with deep notches; was 0.55 above 2050 m outside 0.22..0.48 R)
    for (int i = 0; i < 3; i++) nearS = max(nearS, 1.0 - smoothstep(0.18 * PKA[i].w, 0.42 * PKA[i].w, length(p - PKA[i].xy)));
    Pm -= VA_HSAD_K * max(Pm - VA_HSAD_H, 0.0) * (1.0 - nearS) * hz;   // (t13: 0.4 / 2150 -> 0.55 / 2050)
    // (r4 C1) one connected massif: an arete crest at ~2380-2480 m joins the west summit, the main horn and the east ridge
    // (the saddles were compressed to ~1650 m: three isolated shark teeth). ~42 deg flanks; its 60-150 m serration (towers
    // and notches) is cut by reliefPost; the summits keep 100-700 m of sky notch above it
    {
        vec2 a0 = PKA[1].xy, a1 = PKA[0].xy, a2 = PKA[2].xy;
        vec2 e0 = a1 - a0, e1 = a2 - a1;
        float t0u = dot(p - a0, e0) / dot(e0, e0), t1u = dot(p - a1, e1) / dot(e1, e1);
        float t0 = clamp(t0u, 0.0, 1.0), t1 = clamp(t1u, 0.0, 1.0);
        float d0 = length(p - a0 - e0 * t0), d1 = length(p - a1 - e1 * t1);
        float sA = d0 < d1 ? t0 * length(e0) : length(e0) + t1 * length(e1);
        float dR = min(d0, d1);
        float tS = d0 < d1 ? t0 : t1, tU = d0 < d1 ? t0u : t1u;
        // the crest height wanders along the ridge (~0.9 km) and sags into a col mid-way between two summits (~2300 m:
        // b1's level 2420 m crest read as a wall, the summits barely standing out; target cols sit 250-700 m down);
        // no end caps (b2: the crest's rounded end cone at ~2500 m swallowed the west / east summit tips - flat-topped towers)
        float Hr = mix(2290.0, 2410.0, (1.0 - 4.0 * tS * (1.0 - tS)) * (1.0 - 4.0 * tS * (1.0 - tS))) + 55.0 * gnoise(vec2(sA * 0.0011, 3.7)) + 30.0 * gnoise(vec2(sA * 0.0037, 8.1));
        float rv = Hr - 0.9 * dR - 0.00012 * dR * dR - 400.0 * (1.0 - smoothstep(0.0, 0.10, min(tU, 1.0 - tU)));
        Pm = mix(Pm, smaxk(Pm, rv, 70.0), pW);
    }
    // (r4, b2 -> b3) the east wall north of z ~0 (below the head) stays a lower side ridge: above 1200 m compressed by 35 %
    // (b2's M1 walls of 1800-2100 m within 3-4 km stood 28-30 deg over the r1 / shot 0 shore and hid the head summits;
    // target: forested side slopes ~15 deg, the peaks set back). The west wall (r5 / r6, shot 6) keeps its height
    {
        float eN = smoothstep(800.0, 1400.0, p.x - axisX(p.y)) * smoothstep(-600.0, 0.0, p.y) * (1.0 - smoothstep(2400.0, 3000.0, p.y));
        Pm -= 0.35 * max(Pm - 1200.0, 0.0) * eN;
        // (r4 M5) the hanging valley's head: crest ~1700-1900 m (b2: 2000-2400 m 3 km from the tarn, 28-34 deg up - the shot 5
        // frame had no skyline); with the M5 summits it now sits ~20-23 deg over the tarn meadow
        float hvH = 1.0 - smoothstep(1300.0, 2300.0, length(p - vec2(-3800.0, -800.0)));
        Pm -= 0.5 * max(Pm - 1450.0, 0.0) * hvH;
    }
    float emerge = smoothstep(0.0, 220.0, Pm - L);
    // (the down-valley range keeps more of the old ridges: its faces are otherwise bare cones at 15-20 km)
    float kM = mix(1.0, mix(0.35, 0.75, smoothstep(-9000.0, -12000.0, p.y)), emerge);   // (session 3: 0.6 -> 0.75)
    kM = min(kM, mix(1.0, 0.2, hz));   // (session 3: 0.35 -> 0.2)
    dbg = vec4(Pm - L, emerge, kM, dC);
    return h + dC + (Pm - L) + (kM - 1.0) * (B - L);
}
#endif

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
float vprof(float e) { return 2.0 * sqrt(saturate(0.5 + 0.5 * clamp(e / 0.42, -1.0, 1.0))) - 1.36; }

#ifdef RELIEF_PASS
float reliefH(vec2 p);          // the pass's input map (bake.RELIEF_FRAG)
// planar joint facets (round 2): the hillslope erosion leaves every high face a smooth 40-45 deg plane
// with fine parallel rills; granite walls are a mosaic of planar joint faces. Cells of ~90-160 m (taller
// down the fall line), each replaced by a plane through its centre with its own random tilt; neighbouring
// planes are blended over ~14 m, so cell borders become creases and short steps (joints), not walls.
float facetH(vec2 p, vec2 gD, out float edge) {
    // (an axis-aligned lattice: rotating it to the local fall line scrambled the cells far from the origin)
    // (session 3: 120 -> 170 m cells, 14 -> 18 m joints, gradient over +-60 m: bigger slabs)
    const float CS = 170.0;
    vec2 q = p / CS;
    vec2 ip = floor(q), fq = q - ip;
    float d1 = 1e9, d2 = 1e9; vec2 c1 = vec2(0.0), c2 = vec2(0.0); float s1 = 0.0, s2 = 0.0;
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec2 o = vec2(float(i), float(j));
        ivec2 ci = ivec2(ip + o);
        vec2 r = o + hash22i(ci) * 0.8 + 0.1 - fq;
        float d = dot(r, r);
        float sd = hash12i(ci + ivec2(17, 3));
        if (d < d1) { d2 = d1; c2 = c1; s2 = s1; d1 = d; c1 = q + r; s1 = sd; }
        else if (d < d2) { d2 = d; c2 = q + r; s2 = sd; }
    }
    vec2 w1 = c1 * CS, w2 = c2 * CS;
    float h1 = reliefH(w1), h2 = reliefH(w2);
    vec2 g1 = vec2(reliefH(w1 + vec2(60.0, 0.0)) - reliefH(w1 - vec2(60.0, 0.0)), reliefH(w1 + vec2(0.0, 60.0)) - reliefH(w1 - vec2(0.0, 60.0))) / 120.0;
    vec2 g2 = vec2(reliefH(w2 + vec2(60.0, 0.0)) - reliefH(w2 - vec2(60.0, 0.0)), reliefH(w2 + vec2(0.0, 60.0)) - reliefH(w2 - vec2(0.0, 60.0))) / 120.0;
    // random tilt: mostly sideways (lit / shaded facets across the face), a little along the fall line
    vec2 f1 = length(g1) > 1e-4 ? normalize(g1) : vec2(0.0, 1.0), f2 = length(g2) > 1e-4 ? normalize(g2) : vec2(0.0, 1.0);
    // (session 5: lateral 0.55 -> 0.8, along 0.25 -> 0.3: big planar lit / shaded slabs)
    g1 += vec2(-f1.y, f1.x) * (fract(s1 * 13.7) - 0.5) * 0.8 * length(g1) + f1 * (fract(s1 * 7.3) - 0.5) * 0.3;
    g2 += vec2(-f2.y, f2.x) * (fract(s2 * 13.7) - 0.5) * 0.8 * length(g2) + f2 * (fract(s2 * 7.3) - 0.5) * 0.3;
    float P1 = h1 + dot(g1, p - w1), P2 = h2 + dot(g2, p - w2);
    // distance to the cell border (metres)
    vec2 m = 0.5 * (c1 + c2), n = normalize(c2 - c1);
    edge = dot(m - q, n) * CS;
    // (world3: 18 -> 8 m: crisp creases between the big planar faces - soft 18 m folds read as melted wax)
    float bl = smoothstep(0.0, 12.0, edge);
    return mix(0.5 * (P1 + P2), P1, bl);
}
#endif

// (world3) a second, smaller facet layer (~75 m) on the big faces: a 3D Voronoi of tilted slabs evaluated at
// the surface point (x, h, z) - round cells ON the face at any slope (2D x-z cells hang 2-4x long down a steep
// face: organ pipes) - the surface the smooth max of flat caps (crisp V creases between the slabs), tilts from
// three joint families of the region (parallel slabs repeat: coherent lit / shaded planes). Height change
// <= 0. The run-time relief (world.glsl rockRelief) adds the 42 / 15 / 5 m octaves of the same structure.
// (r4 M2 c) sa = strike angle of the face (atan of its contour direction): the two lateral joint families F0 / F1 at
// +-40 deg to strike (diagonal joint-controlled creases / ledges instead of a regional, slope-blind direction)
float bkFacets3(vec3 P, float C, float T, float k, float sa) {
    vec3 q = P / C;
    vec3 ip = floor(q), fq = q - ip;
    float ang = 0.5 + 2.4 * (vnoise(P.xz * (1.0 / 4096.0) + 3.7) - 0.5);
    vec3 F0 = vec3(cos(sa + 0.7), 0.25, sin(sa + 0.7)), F1 = vec3(cos(sa - 0.7), -0.2, sin(sa - 0.7)), F2 = vec3(0.3 * cos(ang), 1.0, 0.3 * sin(ang));
    float best = -1e9;
    for (int z = -1; z <= 1; z++)
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec3 o = vec3(float(i), float(j), float(z));
        uint hh = hash3u(ivec3(ip + o) + ivec3(9173, 2711, 4409));
        vec3 c = o + 0.15 + 0.7 * vec3(u2f(hh), u2f(hashu(hh + 1u)), u2f(hashu(hh + 2u)));
        vec3 r = (fq - c) * C;
        float fam = 3.0 * u2f(hashu(hh + 3u));
        vec3 fd = fam < 1.0 ? F0 : (fam < 2.0 ? F1 : F2);
        float sg = u2f(hashu(hh + 6u)) < 0.5 ? -1.0 : 1.0;
        vec3 tl = (T * (0.4 + 0.6 * u2f(hashu(hh + 4u))) * sg) * fd;
        float v = -0.12 * C * u2f(hashu(hh + 5u)) + dot(tl, r) - 0.12 * dot(r, r) / C;
        best = (z == -1 && j == -1 && i == -1) ? v : smax(best, v, k);
    }
    return min(best - 0.2 * C, 0.0);
}

float reliefPost(vec2 p, float h, vec2 g, float w, float crest, out vec4 dbg) {
    float slope = length(g);
    float alt = smoothstep(300.0, 650.0, h) * w;
    float steep = smoothstep(0.45, 0.95, slope);            // 24 .. 43 deg
    // patchiness: some faces are clean slabs, others broken into bands and towers
    float pn = gnoise(p * 0.0016 + 21.0) + 0.5 * gnoise(p * 0.0041 + 5.0);
    float patchy = smoothstep(-0.22, 0.25, pn);
    float mzR = smoothstep(0.0, 0.3, gnoise(p * 0.0013 + 44.0) + 0.5 * gnoise(p * 0.0035 + 9.0));   // (t24) massive zones
    float hb = h;
#ifdef RELIEF_PASS
    // ---- planar joint facets on the steep high faces (round 2)
    float edge;
    float wf = alt * smoothstep(0.55, 0.95, slope) * smoothstep(450.0, 800.0, h) * mix(0.85, 1.0, patchy);
    // (session 5: 0.85 -> 0.95: the eroded map's dense fall-line rills left under the facets read as dripping wax)
    if (wf > 0.0) hb = mix(h, facetH(p, g, edge), wf * 0.95);
    // (world3) ~75 m slabs on the steep high faces (crests at half strength: notched, not planed)
    // (t24) massive vs broken zones (~40 % massive, ~500 m patches; world.glsl rockRelief uses the same mask): the
    // granite walls alternate big smooth planar faces cut by a few deep dark chimneys with broken, blocky zones - the
    // even 75 m slab crumple everywhere read as one texture at frame level. Massive: the 75 m slabs x0.25
    // (r4 M2 c: 75 -> 110 m slabs at full weight (0.7), massive zones x0.6 (x0.25), joint families +-40 deg to strike: the
    // big planar triangular facets bounded by aretes)
    float wf2 = alt * smoothstep(0.7, 1.1, slope) * smoothstep(350.0, 650.0, h) * mix(1.0, 0.5, crest) * mix(1.0, 0.6, mzR);
    if (wf2 > 0.0) {
        vec2 gS = vec2(reliefH(p + vec2(200.0, 0.0)) - reliefH(p - vec2(200.0, 0.0)), reliefH(p + vec2(0.0, 200.0)) - reliefH(p - vec2(0.0, 200.0)));
        hb += 1.0 * wf2 * bkFacets3(vec3(p.x, h, p.y), 110.0, 0.4, 4.0, length(gS) > 1e-3 ? atan(gS.x, -gS.y) : 0.0);
    }
    // (r4 C1) serrated high crests: V-notches 60-150 m apart, 40-90 m deep, within ~70 m below the crest line (zero lines of
    // two noise octaves crossing the crest; subtractive, so no needles for the spike clamp); was averaged smooth
    float cW = smoothstep(0.35, 1.0, crest) * smoothstep(1600.0, 2100.0, h) * w;
    if (cW > 0.0) {
        float aN = mix(40.0, 90.0, saturate(0.5 + 1.5 * gnoise(p * 0.0011 + 61.0)));
        float z1 = abs(gnoise(rot2(0.6) * p * (1.0 / 150.0) + 13.0)), z2 = abs(gnoise(rot2(2.1) * p * (1.0 / 70.0) + 29.0));
        float notch = max(1.0 - smoothstep(0.0, 0.12, z1), 0.6 * (1.0 - smoothstep(0.0, 0.09, z2)));
        hb -= cW * aN * notch;
    }
#endif
    // ---- cliff bands: tilted strata (dip ~7 deg, slowly turning), only on ~40 % of the steep faces in
    // 300-600 m patches (round 2: the old bands ran as contour stripes over every wall), steps 34..70 m
    // with steep risers and narrow ledges
    float da = 3.0 * gnoise(p * 0.00025 + 9.0);
    float hh = h + 0.12 * dot(p, vec2(cos(da), sin(da)));
    // (session 4) the band coordinate was h / S with S = 48 +- 18 m varying over ~500 m: at h ~ 2000 m the
    // gradient of h / S was ~4x that of the slope, so the bands followed the contours of S - evenly spaced
    // diagonal stripes across whole faces ("washboard", shot 7) and man-made benches. Now a fixed
    // 44 m base step, and each band its own amplitude: only ~1 band in 3 is a ledge (0.6..1 strength), so
    // the ledges sit at irregular 44-180 m intervals. The profile is c at both band ends, so
    // (q - c) * a_k + c stays continuous when the amplitude changes from band to band.
    const float S = 44.0;
    float u = hh / S + 0.9 * gnoise(p * 0.0042 + 3.0);
    float f = fract(u);
    float kb = floor(u);
    float rs = mix(0.50, 0.70, saturate(0.5 + 2.0 * gnoise(p * 0.006 + 1.0)));
    float c = 0.5 * rs;
    float q = smoothstep(rs, 0.96, f) - f + c;                        // ledge then riser (~zero mean)
    float hk = hash11(kb * 0.6180339 + 0.37);
    float ak = step(0.67, hk) * mix(0.6, 1.0, fract(hk * 7.31));
    // (session 3: on ~55 % of the steep faces; session 4: only on the steep (> ~40 deg) faces above
    // ~700 m - on 29-40 deg slopes they read as benches)
    float bm = smoothstep(-0.10, 0.25, gnoise(p * 0.0019 + 4.0) + 0.5 * gnoise(p * 0.0052 + 7.0));
    float tw = alt * smoothstep(0.85, 1.25, slope) * smoothstep(700.0, 1100.0, h) * bm;
    float led = tw * S * ((q - c) * ak + c);
    // ---- fall-line buttresses and couloirs (two scales), V-shaped gullies; deeper on the big walls
    // (r4 M2) joint-controlled: the gully systems follow the regional joint direction (+-34 deg off the fall line, ~2 km
    // patches, the heightCoarse field), the 900 m system +-20 deg - parallel fall-line gullies were the organ pipes
    float jA = clamp(fbm2(p * 0.0005 + 3.0, 2) / 0.05, -1.0, 1.0);
    vec2 gJ = rot2(0.6 * jA) * g;
    float r1 = erosion(p, gJ, 170.0, 3), r2 = erosion(p + 517.0, gJ, 430.0, 2);
    float e1 = vprof(r1);
    float e2 = vprof(r2);
    // (session 6: the 170 m couloirs 20 -> 8 m: at 20 m they combed every wall into parallel needles)
    // (session 5: 9 -> 6 m and patchy: the 42-170 m grooves combed every face like dripping wax, debug 44)
    // (world3: x0.35 - their broad round ribs every 40-170 m hung down the faces as draperies; the run-time
    // slab relief carries the small-scale structure now)
    float a1 = 0.35 * 6.0 * mix(0.6, 1.2, smoothstep(700.0, 1600.0, h)) * mix(1.0, 0.35, patchy) * mix(1.0, 0.3, mzR);
    // (session 3) the 430 m couloirs cut a fin every 430 m into every wall (shot 7: organ pipes, needle
    // tops): now ~40 % of the faces carry deep couloir systems, the rest only shallow ones, and the ribs
    // between the gullies are flattened (planar buttress faces with the gullies cut into them)
    float mBig = smoothstep(-0.05, 0.30, gnoise(p * 0.00105 + 31.0) + 0.45 * gnoise(p * 0.0026 + 8.0));
    float a2 = mix(18.0, 92.0, mBig) * smoothstep(600.0, 1200.0, h);
    float e2c = e2 < 0.0 ? e2 : 0.3 * e2;
    // (crests: a quarter strength - massive summits and ridges, not a comb of notches)
    float cou = alt * steep * (a1 * e1 + a2 * e2c * mix(0.5, 1.0, patchy)) * mix(1.0, 0.25, crest);
    // (session 4) one large spur-and-gully scale on EVERY face above ~500 m (shot 6's left massif and the
    // shot 0 left face were featureless planes: the deep couloirs are on ~40 % of the
    // faces only): ~900 / 450 m systems, 45-70 m deep V gullies, flattened spurs (x0.3), a quarter
    // strength on the crests; oriented by the 400 m slope (the 64 m gradient wanders with the facets)
    vec2 gL = g;
#ifdef RELIEF_PASS
    gL = vec2(reliefH(p + vec2(200.0, 0.0)) - reliefH(p - vec2(200.0, 0.0)), reliefH(p + vec2(0.0, 200.0)) - reliefH(p - vec2(0.0, 200.0))) / 400.0;
#endif
    float e3 = vprof(erosion(p + 1301.0, rot2(0.35 * jA) * gL, 900.0, 2));
    float e3c = e3 < 0.0 ? e3 : 0.3 * e3;
    float a3 = mix(45.0, 70.0, saturate(0.5 + 1.2 * gnoise(p * 0.0005 + 12.0))) * smoothstep(500.0, 900.0, h);
    float big = alt * smoothstep(0.35, 0.65, length(gL)) * a3 * e3c * mix(1.0, 0.25, crest);
    // (t24) chimneys of the massive zones: narrow (~15-25 m), 20-38 m deep V clefts down the fall line every ~150 m
    // (the most negative band of a 2-octave erosion pattern on the 400 m slope: clefts only, no round ribs between them -
    // the ribs were the draperies), meandering a little, broken along their length; a third strength on the crests
    float chim = 0.0;
    if (mzR > 0.0 && alt > 0.0) {
        float ec = erosion(p + 913.0, rot2(0.6 * jA) * gL, 150.0, 2);
        float on = smoothstep(-0.15, 0.2, gnoise(p * 0.004 + 71.0));
        float cd = mix(20.0, 38.0, saturate(0.5 + 1.5 * gnoise(p * 0.0021 + 17.0)));
        chim = -alt * mzR * on * smoothstep(0.75, 1.15, slope) * smoothstep(600.0, 950.0, h) * mix(1.0, 0.35, crest)
             * cd * (1.0 - smoothstep(-0.50, -0.28, ec));   // (erosion() p10 ~ -0.31: ~10 % of the face)
    }
    dbg = vec4(cou + big + chim, led, hb - h, alt);
    return hb + led + cou + big + chim;
}
