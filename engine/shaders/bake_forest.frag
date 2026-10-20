// Bakes the forest / material map on the far grid (4 m cells):
//   r = tree density, g = larch fraction, b = scree / talus, a = meadow
//
// Macro look: closed spruce/larch forest only on the lower slopes (to ~250-400 m
// above the lake), an irregular upper edge with tongues climbing the ribs and receding in the
// gullies, then a wide belt of clumps and lone trees on grassy / rocky ground, and above ~650 m
// open alpine terrain: grass ledges, granite, pale scree fans below the cliffs, snow higher up.
#include "common.glsl"
#define FEX0 0.42
#define FEX1 0.62
#define LOBEN 0.27
#define TRK0 0.05
#define TRK1 0.10
#ifndef VA_TL
#define VA_TL 420.0      // (it1-3: 440, r3: 520; r4 M1: forest on <= ~35 % of the new, higher side walls)
#endif
uniform sampler2D uH;          // far heights
uniform sampler2D uLvl;        // water level (nearest)
uniform sampler2D uTalus;      // optional erosion talus map (0..1)
uniform sampler2D uFlow;       // optional erosion flow map (normalized log flux 0..1)
uniform sampler2D uCones;      // (r4 M7) talus-cone deposits of bake.talus_cones (0..1)
uniform int uHasErosion;
uniform float uHalf, uRes;
uniform vec2 uTarn;
uniform float uTarnLevel;
uniform vec4 uStream[96];
uniform int uStreamN;
uniform vec4 uTrail[32];
uniform int uTrailN;
out vec4 fragColor;

float H(vec2 p) { return texture(uH, p / (2.0 * uHalf) + 0.5).r; }
vec2 G(vec2 p, float e) {
    return vec2(H(p + vec2(e, 0.0)) - H(p - vec2(e, 0.0)), H(p + vec2(0.0, e)) - H(p - vec2(0.0, e))) / (2.0 * e);
}
float lapAt(vec2 p, float h, float r) {
    return (H(p + vec2(r, 0.0)) + H(p - vec2(r, 0.0)) + H(p + vec2(0.0, r)) + H(p - vec2(0.0, r)) - 4.0 * h) / (r * r);
}
// same, centre averaged over a ring of r/4 (band-limited: no cell-scale speckle)
float lapS(vec2 p, float r) {
    float q = 0.25 * r;
    float hc = 0.25 * (H(p + vec2(q, 0.0)) + H(p - vec2(q, 0.0)) + H(p + vec2(0.0, q)) + H(p - vec2(0.0, q)));
    return (H(p + vec2(r, 0.0)) + H(p - vec2(r, 0.0)) + H(p + vec2(0.0, r)) + H(p - vec2(0.0, r)) - 4.0 * hc) / (r * r);
}

// debris potential of a point: a rock face close by (talus apron) or up the fall line (fan, up to
// ~250 m below it). Every test uses >= 16 m gradients and soft thresholds, so the crag relief of
// the 4 m heightfield cannot flip it cell by cell (the map is sampled without mipmaps far away).
// erosion flow (normalised log flux) averaged over ~30 m: couloir / channel strength at q
float flowB(vec2 q) {
    if (uHasErosion != 1) return 0.0;
    vec2 u = q / (2.0 * uHalf) + 0.5; float k = 30.0 / (2.0 * uHalf);
    return 0.2 * (texture(uFlow, u).r + texture(uFlow, u + vec2(k, 0.0)).r + texture(uFlow, u - vec2(k, 0.0)).r
                + texture(uFlow, u + vec2(0.0, k)).r + texture(uFlow, u - vec2(0.0, k)).r);
}
float potAt(vec2 x, float hx) {
    float cl = 0.0;
    for (int i = 0; i < 12; i++) {
        float a = float(i) / 12.0 * TAU;
        vec2 d = vec2(cos(a), sin(a));
        for (int k = 1; k <= 3; k++) {
            vec2 q = x + d * (22.0 * float(k));
            float hq = H(q);
            float s2 = length(G(q, 16.0));
            cl = max(cl, smoothstep(5.0, 15.0, hq - hx) * smoothstep(1.0, 1.5, s2) * (1.0 - float(k - 1) * 0.2));
        }
    }
    float cu = 0.0;
    vec2 q = x; float hq = hx; float run = 0.0, fex = 0.0;
    for (int k = 0; k < 14; k++) {
        vec2 gq = G(q, 20.0);
        float sq = length(gq);
        if (sq < 0.08) break;
        q += gq / sq * 32.0;
        float hn = H(q);
        float rise = (hn - hq) / 32.0;
        hq = hn;
        float st = smoothstep(0.88, 1.24, rise);
        // (session 6) the fan apex: where the walk first meets the face, a couloir exit (high flow)
        // feeds a big fan, an open wall only a small apron
        if (run < 0.5 && st > 0.5) fex = flowB(q);
        run = run * st + st;                  // consecutive steep steps = height of the face
        cu = max(cu, st * smoothstep(0.8, 2.5, run) * (1.0 - float(k) * 0.045)
                     * mix(0.55, 1.25, smoothstep(FEX0, FEX1, fex)));   // fans reach ~450 m
    }
    // (world3 t12: couloir exits 0.75..1.2 -> 0.55..1.25, aprons 0.75 -> 0.6: the aprons along the foot of every
    // wall contoured the whole cirque as one pale band (s5 / r3: 13 % of the 150-1500 m ground); target:
    // cones below the couloirs and thin, vegetated aprons between them)
    return max(cu, 0.6 * cl);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = (uv * 2.0 - 1.0) * uHalf;
    float cell = 2.0 * uHalf / uRes;
    float h = H(p);
    float e = cell;
    vec2 g = G(p, e);
    float slope = length(g);
    vec2 g3 = G(p, 12.0);                     // smoother gradient (aspect, uphill direction)
    float slope3 = length(g3);
    // curvature (positive = concave gully, negative = convex rib) at ~25 m and ~90 m
    float lap = lapAt(p, h, 24.0);
    float lapB = lapAt(p, h, 90.0);
    float lapBs = lapS(p, 90.0);              // (smoothed centre: avalanche tracks, valley floors)
    float lapV = lapS(p, 260.0);              // valley floors
    float slope16 = length(G(p, 16.0));
    // cliffs uphill within ~70 m -> talus right below them
    float cliff = 0.0;
    for (int i = 0; i < 12; i++) {
        float a = float(i) / 12.0 * TAU;
        vec2 d = vec2(cos(a), sin(a));
        for (int k = 1; k <= 3; k++) {
            float rr = 22.0 * float(k);
            vec2 q = p + d * rr;
            float hq = H(q);
            float s2 = length(G(q, 10.0));
            if (hq > h + 5.0) cliff = max(cliff, smoothstep(1.05, 1.6, s2) * (1.0 - float(k - 1) * 0.2));
        }
    }
    float wl = texture(uLvl, uv).r;
    float above = wl > -1000.0 ? h - wl : 999.0;
    // (world5 F1) denser forest everywhere EXCEPT around the film cameras of the shots framed close to trees (the
    // proven r4 layouts there: the opening forest dolly on the trail looking west through the trunks, the reveal /
    // waterfall / stream / finale shore positions, the shot 5 meadow foreground): kD = 0 -> exactly the r4 bake
    float kD = 1.0;
    {
        const vec4 KC[7] = vec4[7](vec4(15.0, -1395.0, 90.0, 170.0),     // reveal (south-shore block)
                                   vec4(441.0, -590.0, 90.0, 170.0),     // waterfall (low over the east shore)
                                   vec4(698.0, 340.0, 170.0, 260.0),     // forest dolly (trail, view west to the lake)
                                   vec4(220.0, 1676.0, 90.0, 170.0),     // stream mouth
                                   vec4(-503.0, 295.0, 90.0, 170.0),     // finale (west shore)
                                   vec4(-2053.0, -40.0, 95.0, 150.0),    // meadow (shot 5: its open foreground meadow; closed forest from ~150 m to the tarn)
                                   vec4(-150.0, -1020.0, 60.0, 120.0));  // skim (low over the lake)
        for (int i = 0; i < 7; i++) kD = min(kD, smoothstep(KC[i].z, KC[i].w, length(p - KC[i].xy)));
        // the reveal's camera search (shots_film make_reveal: x = -120 +-195 m along the south shore, first clear path
        // wins) must find the r4 position again: the whole candidate strip keeps the r4 trees
        vec2 ra = vec2(-345.0, -1425.0), rb = vec2(105.0, -1395.0), rab = rb - ra;
        float rd = length(p - ra - rab * clamp(dot(p - ra, rab) / dot(rab, rab), 0.0, 1.0));
        kD = min(kD, smoothstep(100.0, 180.0, rd));
    }
    // the tarn: everything near it is relative to its level
    float dT = length((p - uTarn) * vec2(1.0, 0.82));
    float relH = h;

    float n1 = fbm2(p * 0.0028 + 3.7, 4);          // ~350 m
    float n2 = fbm2(p * 0.018 + 9.1, 3);           // ~55 m
    float n3 = gnoise(p * 0.06 + 1.3);             // ~17 m
    float n4 = fbm2(p * 0.0075 + 21.3, 3);         // ~130 m: edge tongues / clump fields
    float n5 = gnoise(p * 0.11 + 5.7);             // ~9 m: single-tree scale

    float sd = 1e9; float shw = 1.0;
    for (int i = 0; i < 95; i++) {
        if (i + 1 >= uStreamN) break;
        vec4 s0 = uStream[i], s1 = uStream[i + 1];
        vec2 ab = s1.xy - s0.xy;
        float tt = clamp(dot(p - s0.xy, ab) / dot(ab, ab), 0.0, 1.0);
        float dd = length(p - (s0.xy + ab * tt));
        if (dd < sd) { sd = dd; shw = mix(s0.w, s1.w, tt); }
    }
    // ---- treeline -------------------------------------------------------------------------------
    // sunny (south-facing) slopes carry the forest higher, shady north faces lower; ribs higher,
    // gullies (avalanche tracks) lower
    float south = slope3 > 0.05 ? g3.y / slope3 : 0.0;
    float rib = saturate(-lapB / 0.0022) - saturate(lapB / 0.0022);
    // (fbm2 has std ~0.125 and gnoise ~0.22: normalise so the amplitudes below are in metres / sigmas)
    float N1 = n1 / 0.125, N2 = n2 / 0.125, N4 = n4 / 0.125, N3 = n3 / 0.22, N5 = n5 / 0.22;
    // (valleyA) the treeline follows the new concave wall profile: the forested lower slopes reach ~450 m
    // (~1.1 km from the shore), the steep rock wall starts above them (dense dark forest
    // up to 6-12 deg above the far waterline). Was 305 m.
    // (r4 M11: N1 65 -> 110 m, ribs 110 -> 160 m: a ragged treeline with forest tongues up the ribs, not a contour line)
    float TL = VA_TL + 110.0 * N1 + 55.0 * south * saturate(slope3 * 2.0) + 160.0 * rib + 60.0 * N4 + 28.0 * N2;
    // (session 6) ~1.2 km term: whole hillsides carry the forest higher or lower (not a uniform ring)
    TL += 70.0 * clamp(fbm2(p * 0.00083 + 41.3, 2) / 0.125, -1.3, 1.3);
    // sheltered valley floors and the stream banks carry the trees higher (larch up the inflow
    // valley towards the cirque) instead of two straight diagonals where the valley crosses the treeline
    float vfloor = smoothstep(0.0004, 0.0018, lapV) * (1.0 - smoothstep(0.30, 0.55, slope3));
    // the tarn bench sits in a sheltered hanging valley: spruce groups around the water
    float tarnW = 1.0 - smoothstep(650.0, 1100.0, dT);
    TL += 120.0 * vfloor * (1.0 - tarnW) + 150.0 * (1.0 - smoothstep(60.0, 150.0, sd));   // (tarn bench: own treeline below)
    // (shot 5: dense spruce-fir right down to the west, north and south shores of the tarn,
    // thinning out ~150-200 m above it; the open flower meadow is the east bench only)
    TL = mix(TL, max(TL, uTarnLevel + 170.0 + 40.0 * kD + 60.0 * N1 + 25.0 * N4), tarnW);   // (world5 F1: +40 m outside the camera zones)
    // closed canopy only on the lowest slopes; from ~TL-200 the stands open up: thinner
    // canopy, granite knobs and slabs breaking through, grass glades; above TL clumps and lone trees
    // (valleyA: the stands open up only over the last ~130 m below the treeline; was TL-210 .. TL-20)
    // (world5 F1, outside the camera zones: the closed canopy reaches ~40 m higher, the stands open up later; the
    // target is closed dark conifer forest with touching crowns on the lower slopes)
    TL += 30.0 * kD;
    float openS = smoothstep(TL - mix(130.0, 90.0, kD), TL + mix(-10.0, 20.0, kD), relH);
    float closed = 1.0 - smoothstep(TL + mix(-40.0, -10.0, kD), TL + mix(50.0, 70.0, kD), relH);
    // closed-canopy core of the lower slopes (dark, ~0.9 density: from across the lake the open
    // 0.4-0.6 stands over pale ground read as bare rock)
    // (it5: up to TL-40 and denser - from across the lake the upper half of the forest still read grey-blue;
    // it1-4: TL-170 .. TL-60)
    float core = (1.0 - smoothstep(TL - mix(150.0, 110.0, kD), TL - mix(40.0, 10.0, kD), relH)) * smoothstep(mix(40.0, 15.0, kD), mix(90.0, 55.0, kD), relH);
    float belt = 1.0 - smoothstep(TL - 20.0, TL + 330.0, relH);              // clumps / lone trees above
    belt *= 1.0 - smoothstep(800.0, 950.0, relH);
    // outcrops: the same ~50 m / ~150 m noises as the outcrop mask in terrainAlbedo (so the bare spots
    // render as granite there); convex, steeper ground first, more of it with altitude
    float oN = fbm2(p * 0.042 + 5.0, 3) / 0.12, oL = fbm2(p * 0.0067 + 5.0, 3) / 0.12;
    float cvx = saturate(-lapB / 0.0025);
    float oS = 0.55 * oN + 0.45 * oL + 0.8 * cvx + 1.5 * (slope3 - 0.85);
    float outc = smoothstep(1.5 - 1.6 * openS, 2.0 - 1.6 * openS, oS);
    float stand = closed * mix(1.0, mix(0.55, 0.7, kD), openS) * (1.0 - 0.95 * outc);   // (world5 F6: 0.7 outside the camera zones)
    // clumps: islands of trees whose size and number shrink with altitude through the belt
    float up = smoothstep(TL - 40.0, TL + 330.0, relH);
    float clumpN = (0.65 * N2 + 0.35 * N4 + 0.30 * N3) / 0.8;
    float clumps = smoothstep(-0.35 + 1.8 * up, 0.35 + 1.8 * up, clumpN) * mix(0.8, 0.45, up) * (1.0 - 0.7 * outc);
    float lone = 0.12 * step(1.0 + 0.4 * up, N5 + 0.4 * N3);
    float alt = max(stand, belt * max(clumps, lone));
    // slope: trees cling to steep ground but cliff bands are bare
    // (r4 M1: 1.05 / 1.45 -> 0.85 / 1.20: faces over ~45 deg bare)
    // (world5 F5: outside the camera zones 0.92 / 1.28 - the closed forest climbs the ~40-45 deg lower walls)
    float sl = 1.0 - smoothstep(mix(0.85, 0.92, kD), mix(1.20, 1.28, kD), slope16 + 0.25 * n3 + 0.15 * n2);
    // (r4 M11) cliff bands through the forest on ~40 % of the slopes (~300 m patches): bare rock steps between the stands
    float cbm = smoothstep(-0.05, 0.20, fbm2(p * 0.0032 + 77.0, 3) / 0.125 * 0.25 + 0.1 * n2 / 0.125);
    sl *= 1.0 - cbm * smoothstep(0.70, 0.95, slope16 + 0.2 * n3) * smoothstep(80.0, 150.0, relH);
    float shore = saturate((above - 0.8) / 2.0);
    // avalanche tracks: the concave fall lines carry grass and shrubs, not trees
    float gully = 1.0 - saturate((lapBs - 0.0016) / 0.0035) * saturate((slope3 - 0.35) / 0.3);
    // ---- scree / talus: debris only collects on repose-angle ground (~20-40 deg) at the foot of
    // rock faces, in concave positions; ribs, shoulders and the faces themselves stay rock.
    float concave = smoothstep(-0.0004, 0.0012, lapB + 0.4 * lap);
    // debris potential: a rock face directly up the fall line (fans, up to ~250 m below it) or close
    // by (talus aprons); cohesive patches with lobed ~50-130 m edges, so the map stays smooth at
    // the 4 m cell scale (it is sampled without mipmaps far away)
    float concB = smoothstep(-0.0012, 0.0008, lapB);
    // debris potential and fan-scale slope / fall line evaluated on a 24 m lattice and interpolated:
    // band-limited, so thresholding gives cohesive fans instead of per-cell speckle (the 12 m and
    // even the 40 m gradient jitter on the crag relief of the 4 m heightfield)
    // (session 6: 48 m lattice - the 24 m one still let the crag relief break fans into flecks)
    vec2 lp = p / 48.0 - 0.5;
    vec2 li = floor(lp), lf = lp - li;
    lf = lf * lf * (3.0 - 2.0 * lf);
    float potS = 0.0;
    vec2 gF = vec2(0.0);
    for (int j = 0; j < 4; j++) {
        vec2 o = vec2(float(j & 1), float(j >> 1));
        vec2 q = (li + o + 0.5) * 48.0;
        float w = (o.x > 0.5 ? lf.x : 1.0 - lf.x) * (o.y > 0.5 ? lf.y : 1.0 - lf.y);
        potS += w * potAt(q, H(q));
        gF += w * G(q, 40.0);
    }
    float slopeF = length(gF);
    vec2 fall = slopeF > 1e-3 ? -gF / slopeF : vec2(0.0, 1.0);
    // repose: ~20-38 deg on the 40 m slope, but right below a strong debris source up to ~47 deg (the
    // 450-1000 m terrain has a median slope of 45 deg: the 38 deg cap left almost no fans)
    float kRep = smoothstep(0.40, 0.55, potS);
    float reposeS = smoothstep(0.25, 0.40, slopeF) * (1.0 - smoothstep(mix(0.60, 0.95, kRep), mix(0.78, 1.15, kRep), slopeF));
    // lobe noise stretched ~1:3 along the fall line -> downslope tongues instead of round blobs
    float lobeS = 0.0;
    for (int j = -5; j <= 5; j++) {
        vec2 q = p + fall * (22.0 * float(j));
        lobeS += (0.6 * fbm2(q * 0.018 + 9.1, 2) + 0.4 * fbm2(q * 0.0075 + 21.3, 3)) * (1.0 - abs(float(j)) * 0.15);
    }
    lobeS = lobeS / LOBEN;                                              // ~unit variance
    float pot = potS * reposeS * mix(0.55, 1.0, concB);
    float scree = smoothstep(0.24, 0.46, pot + 0.14 * lobeS + 0.45 * n1);   // (world3 t12: 0.20 / 0.10 -> 0.24 / 0.14: tongues, fewer bands)
    // (no flecks from weak sources; fans from ~60-140 m above the lake up to ~1300-1700 m)
    scree *= smoothstep(0.3, 0.6, potS) * smoothstep(60.0, 140.0, relH) * (1.0 - smoothstep(1300.0, 1700.0, relH));
    // (world3 t15) the aprons of the weaker sources only in ~0.5 km sections (debug 24 of shot 5: one continuous
    // pale band around the whole cirque); the big couloir fans (strong sources) stay. Elsewhere the foot of the
    // walls carries turf, rock knobs and (lower) trees
    float gateN = fbm2(p * 0.0019 + 63.0, 3) / 0.125;
    scree *= smoothstep(-0.6, 0.4, gateN + 2.5 * (potS - 0.75));
    float flow = 0.0, trackD = 0.0;   // (trackD, flwD: debug views)
    vec2 flwD = vec2(0.0);
    if (uHasErosion == 1) {
        // erosion talus, averaged over ~20 m (the raw map is per-cell speckle)
        float tal = 0.0;
        for (int j = 0; j < 9; j++) {
            vec2 o = j == 0 ? vec2(0.0) : 14.0 * vec2(cos(float(j) * 0.785), sin(float(j) * 0.785));
            tal += texture(uTalus, uv + o / (2.0 * uHalf)).r / 9.0;
        }
        scree = max(scree, smoothstep(0.15, 0.45, tal) * saturate((slope3 - 0.3) / 0.2) * reposeS * (1.0 - smoothstep(2000.0, 2400.0, relH)));
        // (world3 t12) the hero cone (bake.HERO_CONES, R 820 m) is bare debris over its whole fan, also on
        // its flatter toe (the slope / repose gates above left turf islands in it: pasted stickers, s5)
        // (t13: bare to ~520-700 m of the 820 m radius: the toe is colonised (fully bare it filled shot 5's
        // middle ground as one grey hillside))
        // (round 3) ... and only as a TONGUE down the fan axis (~+-35-45 deg of its ~140 deg fan), its flanks colonised
        // with debris tongues between turf and trees: bare over the whole fan it was the pale grey wash filling shot
        // 5's middle ground (target: a pale cone between forested / green flanks)
        vec2 hA = vec2(-2680.0, -320.0), hD = normalize(uTarn - hA);
        vec2 hv = p - hA;
        float hd = length(hv);
        float hang = acos(clamp(dot(hv / max(hd, 1.0), hD), -1.0, 1.0));
        float hCore = (1.0 - smoothstep(800.0, 1000.0, hd + 40.0 * lobeS)) * (1.0 - smoothstep(0.55, 0.80, hang + 0.12 * lobeS));   // (r4 M6: 480 / 660)
        float hFlank = smoothstep(0.45, 0.75, hang) * (1.0 - smoothstep(1150.0, 1300.0, hd));
        scree *= 1.0 - hFlank * (1.0 - smoothstep(0.25, 0.65, lobeS));
        scree = max(scree, hCore * smoothstep(0.3, 0.6, tal));
        flow = texture(uFlow, uv).r;
        // wide, straight avalanche tracks along the main channels (flow averaged over +-24 m: 25-60 m
        // wide tracks instead of single-cell threads), pale debris and grass down them
        float flw = 0.0;
        for (int j = 0; j < 9; j++) {
            vec2 o = j == 0 ? vec2(0.0) : 24.0 * vec2(cos(float(j) * 0.785), sin(float(j) * 0.785));
            flw += texture(uFlow, uv + o / (2.0 * uHalf)).r / 9.0;
        }
        // channel cores only: the flow above its ~100 m neighbourhood (the log flux is high over whole
        // valley sides, a plain threshold gave 200-400 m wide pale sheets)
        float flwW = 0.0;
        for (int j = 0; j < 8; j++) {
            vec2 o = 96.0 * vec2(cos(float(j) * 0.785 + 0.39), sin(float(j) * 0.785 + 0.39));
            flwW += texture(uFlow, uv + o / (2.0 * uHalf)).r / 8.0;
        }
        flwD = vec2(flw, flwW);
        // (r4 M11) only ~half of the channels carry a track (~1 km gate), their width +-50 % (~300 m noise), broken along
        // their length (fall-line stretched lobes), with young trees / shrubs left in them: straight, parallel, even lime
        // stripes read as ski runs (r6, shot 6)
        float wN = clamp(fbm2(p * 0.0033 + 17.0, 2) / 0.125, -1.5, 1.5);
        float track = smoothstep(TRK0 + 0.025 * wN, TRK1 + 0.025 * wN, flw - flwW + 0.02 * N4) * smoothstep(0.40, 0.50, flw)
                    * saturate((slope3 - 0.25) / 0.3) * smoothstep(60.0, 160.0, relH);
        track *= smoothstep(-0.25, 0.15, fbm2(p * 0.0009 + 51.0, 2) / 0.125) * smoothstep(-0.7, 0.3, lobeS);
        gully *= 1.0 - 0.8 * track;
        trackD = track;
        scree = max(scree, 0.6 * track * smoothstep(150.0, 300.0, relH) * (1.0 - smoothstep(1300.0, 1700.0, relH)));
    }
    // open, boulder-strewn floodplain along the inflow stream
    float plain = 1.0 - smoothstep(shw * 2.2 + 8.0 + 10.0 * n2, shw * 4.0 + 26.0, sd);
    // small glades inside the closed forest
    float clear = saturate(0.62 + 1.5 * n2 + 0.8 * n1);
    // closed spruce-fir on the tarn shores (W, N, S): few glades or avalanche gaps right at the water
    float eastB = smoothstep(0.5, 0.82, normalize(p - uTarn).x);
    float tS = tarnW * (1.0 - smoothstep(70.0, 190.0, h - uTarnLevel + 40.0 * N2)) * (1.0 - eastB);
    clear = mix(clear, 0.8 + 0.2 * clear, tS);
    float dens = alt * sl * shore * mix(1.0, gully, 0.9 * (1.0 - 0.75 * tS)) * (1.0 - 0.92 * saturate(scree * 1.2))
               * mix(mix(0.3, 0.5, kD) + mix(0.7, 0.5, kD) * clear, mix(0.82, 0.95, kD) + mix(0.18, 0.05, kD) * clear, core);   // (it1-4: 0.72 + 0.28 clear)
    // stand-scale structure (~250 m): denser and thinner stands, so the forest is not a uniform carpet
    float N6 = fbm2(p * 0.004 + 31.7, 3) / 0.125;
    dens *= mix(mix(mix(mix(mix(0.65, 0.82, kD), mix(0.94, 1.0, kD), core), mix(0.4, 0.5, kD), openS), 1.0, smoothstep(-1.0, 0.8, N6)), 1.0, tS);   // (it1-4: 0.88)
    // (session 6) glades of 50-150 m (grass, rock, old avalanche debris) in the forest above ~100-220 m:
    // from across the lake the closed forest read as one uniform carpet up to the treeline
    // (valleyA: fewer and smaller in the closed core)
    // (r4 M11: the ~130 m glade noise -> the fall-line stretched lobes (~1:3, lobed edges): rectangular patches before)
    float glade = smoothstep(mix(0.50, 0.85, core) + 0.18 * kD, 0.95 + 0.15 * core + 0.18 * kD, (0.7 * lobeS + 0.5 * N2 + 0.3 * N6) / 0.92) * smoothstep(80.0, 180.0, relH) * (1.0 - tarnW);
    dens *= 1.0 - mix(0.9, 0.8, kD) * glade;   // (world5 F6: 0.8 outside the camera zones)
    // granite slabs and knobs breaking through the forest above ~100-200 m: the trees are
    // cleared where terrainAlbedo draws its pale low outcrops (same noises / ramp as its outLow mask)
    // (valleyA: knobs only on convex / steeper ground inside the core, the open slabs higher up)
    float openA = smoothstep(90.0, 280.0, relH) * (1.0 - 0.6 * core);
    // (r4 M11: threshold -0.3, active over 100-700 m (was below VA_TL + 150..350): granite knobs / outcrops through the
    // stands; terrain_shade.glsl outLow matches)
    float outR = smoothstep(1.2 - 1.2 * openA + 0.25 * kD * core, 1.7 - 1.2 * openA + 0.25 * kD * core, oS) * smoothstep(100.0, 200.0, relH)
               * (1.0 - smoothstep(600.0, 750.0, relH)) * (1.0 - tarnW);
    dens *= 1.0 - mix(0.85, 0.65, kD) * outR;   // (world5 F6: 0.65 outside the camera zones - the 150-300 m band: 31 % of it < 0.5 here)
    // the lake-side forest (film paths, trail) keeps its proven layout:
    // exactly the previous formulation below ~60 m, blended into the new structure up to ~150 m
    float screeOld = cliff * saturate((slope - 0.42) / 0.18) * (1.0 - saturate((slope - 1.2) / 0.25));
    screeOld *= saturate((relH - 80.0) / 120.0);
    float gullyOld = 1.0 - saturate((lap - 0.0012) / 0.004) * saturate((slope - 0.35) / 0.3);
    if (uHasErosion == 1) {
        screeOld = max(screeOld, texture(uTalus, uv).r * saturate((slope - 0.3) / 0.2));
        gullyOld *= 1.0 - smoothstep(0.55, 0.8, flow) * saturate((slope - 0.3) / 0.3);
    }
    float altOld = 1.0 - saturate((relH - (720.0 + 110.0 * n1)) / 160.0);
    float slOld = 1.0 - saturate((slope - 1.15) / 0.55);
    float clearOld = saturate(0.68 + 1.5 * n2 + 0.8 * n1);
    // (world5 F1: the shore forest outside the camera zones closes up: a dark conifer wall)
    float densOld = altOld * slOld * shore * gullyOld * (1.0 - 0.9 * screeOld) * (mix(0.3, 0.62, kD) + mix(0.7, 0.38, kD) * clearOld);
    float wNew = smoothstep(60.0, 150.0, relH);
    dens = mix(densOld, dens, wNew);
    scree = mix(screeOld, scree, wNew);
    dens *= 1.0 - 0.92 * plain;
    // (r4 M7) the talus fans (bake.talus_cones deposits): bare pale cores, tree tongues only on their thin flanks / toes
    {
        float k = 12.0 / (2.0 * uHalf);
        float cn = 0.2 * (texture(uCones, uv).r + texture(uCones, uv + vec2(k, 0.0)).r + texture(uCones, uv - vec2(k, 0.0)).r
                        + texture(uCones, uv + vec2(0.0, k)).r + texture(uCones, uv - vec2(0.0, k)).r);
        float cc = smoothstep(0.3, 0.6, cn + 0.12 * lobeS);
        dens *= 1.0 - 0.95 * cc;
        scree = max(scree, 0.9 * cc * (1.0 - smoothstep(1700.0, 2100.0, relH)));
    }
    // (r4 M9) the lower valley's floodplain (z < -2500, flat, below the lake): meadow clearings between forest patches
    {
        float fpl = smoothstep(-2500.0, -3200.0, p.y) * (1.0 - smoothstep(0.06, 0.14, slope3)) * (1.0 - smoothstep(-20.0, 10.0, relH));
        float mo = smoothstep(-0.15, 0.25, fbm2(p * 0.0067 + 13.0, 3) / 0.125 * 0.25 + 0.08 * N2);
        dens *= 1.0 - 0.9 * fpl * mo;
    }
    // (world5 F7) shot 5: a closed dark spruce-fir band on the W / N / S tarn shores (tS: up to ~70-190 m above
    // the tarn, not on the east meadow bench); the r4 stands there were spaced spires over bright turf
    dens = mix(dens, max(dens, 0.95 * sl * shore * (1.0 - saturate(scree * 1.5)) * (1.0 - 0.9 * plain)), kD * tS);
    // (world5 F2) closed canopy on the lower slopes outside the camera zones: the dense stands saturate to 1 (crowns
    // touching); trees.glsl treeFinish keeps every tree of a dens-1 stand (no s2
    // openings), so the cover rises with the tree count; the gaps (glades, gullies, scree, outcrops) stay < 1
    {
        float lowB = 1.0 - smoothstep(TL - 120.0, TL - 30.0, relH);
        dens = mix(dens, saturate(dens * 1.32), kD * lowB);
    }
    // keep the forest trail open (trees right up to its edges)
    float td = 1e9;
    for (int i = 0; i < 31; i++) {
        if (i + 1 >= uTrailN) break;
        vec2 a2 = uTrail[i].xy, b2 = uTrail[i + 1].xy, ab2 = b2 - a2;
        float tt = clamp(dot(p - a2, ab2) / dot(ab2, ab2), 0.0, 1.0);
        td = min(td, length(p - (a2 + ab2 * tt)));
    }
    dens *= smoothstep(5.5, 9.0, td);
    scree = max(scree, plain * 0.35 * smoothstep(0.1, 0.6, n3 + n2));
    // meadow bench around the tarn: scattered trees only
    float meadow = saturate(1.0 - (dT - 300.0) / 450.0) * saturate((h - 450.0) / 60.0);
    meadow *= eastB;
    // (open flower meadow with spruce groups, not an even scatter)
    float mgrp = smoothstep(0.55, 1.15, (0.7 * N2 + 0.3 * N3) / 0.76);
    dens *= mix(1.0, 0.02 + 0.40 * mgrp, meadow);
    // a few lone trees in the open (low slopes only)
    float loneLow = 0.06 * step(0.8, n3 * 0.5 + 0.5) * shore;
    dens = max(dens, loneLow * mix(altOld * slOld, closed * sl, wNew));
    // larch: the treeline tree; lighter crowns in the belt and in open stands near the shore
    float larchOld = saturate(0.12 + 0.55 * saturate((h - 250.0) / 320.0) + 0.35 * n2 + 0.2 * (1.0 - dens));
    float larch = saturate(0.12 + 0.45 * saturate((h - 150.0) / 320.0) + 0.35 * n2 + 0.25 * (1.0 - dens) + 0.25 * up + 0.30 * N4);
    larch *= 1.0 - 0.45 * tarnW * (1.0 - up);             // dark spruce-fir around the tarn
    larch = mix(larchOld, larch, wNew);
    fragColor = vec4(saturate(dens), larch, saturate(scree), saturate(meadow));
}
