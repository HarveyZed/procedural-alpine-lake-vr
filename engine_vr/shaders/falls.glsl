// ---------------------------------------------------------------------------
// falls.glsl : the twin waterfall - a ~29 m step of dark wet, mossy rock set into the
// forested east-shore hillside. The main fall (A) slides and bounces down a chute in the face and
// fans out; the second (B) leaps from a cleft half way up. Water = a thin volumetric sheet that
// follows a path simulated on the baked height map (features.py -> uFallK knots), white and lacy,
// with spray at every ledge impact, mist at the base and a foam plume on the pool.
// The first part (shape) is shared with bake_features.frag, which includes this file with
// FALLS_SHAPE_ONLY defined, so the carved rock and the shading masks always agree.
// ---------------------------------------------------------------------------
uniform vec4 uWF;          // cliff foot x,z, lip height of the main fall (m), along-shore half width
uniform vec2 uWFDir;       // unit: from the cliff out over the lake
uniform vec4 uWFNotch;     // along-shore offsets of the two lips (vA, vB), lip width, lip notch depth

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
vec2 wfLedges(int k, float hs) {
    float yl = k == 0 ? hs - uWFNotch.w : hs * WF_LIPB;
    return k == 0 ? vec2(0.62, 0.30) * yl : vec2(0.45 * yl, -99.0);
}
#define WF_BENCH 1.0       // ledge width (m), with a low WF_LIP reverse lip at the outer edge
#define WF_LIP 0.15
// stepped chute profile: height at inland distance x (0 = foot, Wc = lip) - steep risers between
// ledges WF_BENCH wide with a 0.4 m reverse lip at their outer edge
// k1 widens the first riser (fall A: a ~60 deg slab the water slides down), the lower ones steepen
float wfStepped(float x, float Wc, float yTop, float base, vec2 led, float k1) {
    float n = led.y > base + 0.5 ? 2.0 : 1.0;
    float rW = max(Wc - n * WF_BENCH, 1.0);
    float drop = max(yTop - base, 1.0);
    float u = Wc - x;                                  // outward from the lip
    float d1 = yTop - led.x, r1 = min(rW * d1 / drop * k1, rW * 0.8);
    float rs = (rW - r1) / max(rW * (1.0 - d1 / drop), 0.1);
    if (u < r1) return yTop - d1 * u / r1;
    u -= r1;
    if (u < WF_BENCH) return led.x + WF_LIP * smoothstep(0.35, 1.0, u / WF_BENCH);
    u -= WF_BENCH;
    float l2 = n > 1.5 ? led.y : base;
    float d2 = led.x + WF_LIP - l2, r2 = rW * (led.x - l2) / drop * rs;
    if (u < r2 || n < 1.5) return led.x + WF_LIP - d2 * saturate(u / r2);
    u -= r2;
    if (u < WF_BENCH) return l2 + WF_LIP * smoothstep(0.35, 1.0, u / WF_BENCH);
    u -= WF_BENCH;
    float r3 = max(rW * (l2 - base) / drop * rs, 0.5);
    return l2 + WF_LIP - (l2 + WF_LIP - base) * saturate(u / r3);
}
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
vec4 wfPillow(vec2 fc, float S, float amp) {
    // (round 3) no bedding dip / elongated cells (the 1.45 x 0.8 cells along a 25 deg dip laid the face out
    // in diagonal layers): near-equant cells, a little taller than wide (vertical jointing), each a
    // rounded boss (dome around its cell point) with a worn crevice at the cell edges
    float a = 0.15 * gnoise(fc * 0.011 + 4.1);
    vec2 cs = vec2(cos(a), sin(a));
    vec2 q = vec2(dot(fc, cs), dot(fc, vec2(-cs.y, cs.x)));
    q += S * 0.22 * vec2(gnoise(q * (1.2 / S) + 2.0), gnoise(q * (1.2 / S) + 6.0));
    vec4 c = wfVor(q / vec2(S * 0.95, S * 1.15));
    vec4 h = hash42i(ivec2(int(c.w * 65536.0), 23));
    float r = 1.0 - smoothstep(0.0, 0.30, c.z);
    float d = amp * ((h.x - 0.5) * 1.0 + 0.3 * dot(c.xy, h.yz - 0.5) - 1.1 * dot(c.xy, c.xy) - 0.45 * r * r);
    return vec4(d, c.z, c.w, h.w);
}

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
float wfVegCover(vec2 fc, float hs) { return smoothstep(0.40, 0.62, wfVegRaw(fc, hs)); }
// bush / fern cushions growing out of the face where it is covered (bake relief, m outward): domes
// ~2-3 m across, 0.6-1.8 m proud, ~70 % of the cells
float wfCushion(vec2 fc) {
    vec4 c = wfVor(fc * vec2(0.36, 0.42) + 17.0);
    vec4 h = hash42i(ivec2(int(c.w * 65536.0), 51));
    float r = length(c.xy) / (0.62 + 0.25 * h.x);
    return step(0.3, h.z) * (0.6 + 1.2 * h.y) * sqrt(saturate(1.0 - r * r));
}
// shrub masses on the rim shelf behind the bare face (bake relief, m up): clumps 2.5-4 m across, 1.2-3 m
// tall, so the cliff's outline against the forest is ragged, not a bald rounded edge
float wfRimShrub(vec2 p) {
    vec4 c = wfVor(p * 0.28 + 5.3);
    vec4 h = hash42i(ivec2(int(c.w * 65536.0), 61));
    float r = length(c.xy) / (0.6 + 0.3 * h.x);
    return step(0.2, h.z) * (1.2 + 1.8 * h.y) * pow(saturate(1.0 - r * r), 0.5);
}

#ifndef FALLS_SHAPE_ONLY
uniform int uFallsOn;
#define FF_MAX_DIST 380.0     // the falls flora (3D plants / blocks) is traced up to this distance
// falls lab timing toggles (uDebug 70 + part, 74 = all): 0 flora, 1 water volume, 2 rock / bank / apron
// shading, 3 stream (runtime only: same program, so the share of each part's work can be measured)
bool wfOff(int part) {
    // compile-time ablation (falls lab timing only: FALLS_ABL = bit mask of the parts compiled out)
#ifdef FALLS_ABL
    if (((FALLS_ABL >> part) & 1) != 0) return true;
#endif
    return uDebug == 70 + part || uDebug == 74;
}
// water path knots (features.py simulates the flow on the baked map): per fall WF_NK knots evenly
// spaced in height from the lip (k=0) down to the pool; each = (u, v, half width, impact 0..1)
#define WF_NK 24
uniform vec4 uFallK[48];
uniform vec4 uFallInfo[2];   // lip height, relative volume, impact height (pool), unused
uniform vec4 uFallBox[4];    // per fall k: [2k] = (u, y, v) min, [2k+1] = max of its water + mist (render.py)

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
bool ffPlant(ivec2 c, float foot, out vec3 ctr, out vec3 r, out vec4 h) { return ffPlantC(c, ffCol(c.y), foot, ctr, r, h); }

// (review r2, fix 3) blades: one ribbon per frond / blade, found by folding the angle around the plant's
// base into the frond's sector (the two nearest fronds are tested). qb = point relative to the base
// (along shore, up, outward). Fern fronds (rosette): arch up and droop (centre line y = L (rise s -
// droop s^2), s = 0..1 along the frond), 0.12 L wide tapering to the tip, leaflet notches when resolved.
// Grass blades (rosette = 0): nearly upright, curving out (r = 0.45 L s^2), thin. Widths never below
// ~half a pixel (no sparkling slivers); the gaps between the fronds show what is behind.
float ffBlades(vec3 qb, float N, float L, float wid, float foot, float seed, bool fern) {
    float a = atan(qb.z, qb.x);
    float sec = 6.2832 / N;
    float g = a / sec;
    float f0 = floor(g + 0.5);
    float best = 1e9;
    for (int j = 0; j < 2; j++) {
        float idx = f0 + (j == 0 ? 0.0 : (g > f0 ? 1.0 : -1.0));
        vec4 hf = hash42i(ivec2(int(mod(idx + 64.0, N)) + 7, int(seed * 4096.0)));
        float ac = (idx + (hf.x - 0.5) * 0.5) * sec;
        vec2 dir = vec2(cos(ac), sin(ac));
        float Lf = L * mix(0.7, 1.15, hf.y);
        float rr = dot(qb.xz, dir), lat = dot(qb.xz, vec2(-dir.y, dir.x));
        // centre line y(r) over the radial reach Lr: ferns arch and droop (rise 0.85, droop 0.55-0.95), grass
        // shoots up steeply and bends over (a fountain: rise 1.5, droop 0.9 over a short reach)
        float Lr = fern ? Lf : Lf * mix(0.45, 0.65, hf.w);
        float ri = fern ? 0.85 : 1.5, dr = fern ? mix(0.55, 0.95, hf.z) : mix(0.75, 1.05, hf.z);
        float s = clamp(rr / Lr, 0.0, 1.0);
        float yc = Lf * (ri * s - dr * s * s);
        float sl = Lf / Lr * (ri - 2.0 * dr * s);                   // dy / dr: distance across the slanted ribbon
        float w = wid * Lf * (1.0 - 0.8 * s) * (fern ? smoothstep(0.0, 0.15, s) : 1.0);
        float visP = fern ? 1.0 - smoothstep(0.008, 0.02, foot) : 0.0;
        w *= mix(1.0, 0.55 + 0.45 * abs(sin(rr * 55.0 + hf.w * 6.0)), visP);       // leaflets
        w = max(w, 0.85 * foot);
        float th = max(0.012 * Lf, 0.45 * foot);
        vec2 dd = vec2(abs(lat) - w, abs(qb.y - yc) * inversesqrt(1.0 + sl * sl) - th);
        float d = length(max(dd, 0.0)) + min(max(dd.x, dd.y), 0.0);
        d = max(d, max(rr - Lr, -rr - 0.03));
        best = min(best, d);
    }
    return best;
}

// (review r2, fix 5) pale granite steps right of fall B's foot (its water cascades over them): 4 blocks,
// deterministic, 0.5-1 m risers. Distance (1e9 outside their box); id = step index
float ffStepsB(vec3 p, out int id) {
    id = -1;
    vec4 kb = uFallK[2 * WF_NK - 1];
    vec2 uv = wfUV(p.xz);
    float v0 = kb.y + 0.55 * kb.z;
    if (uv.y < v0 - 1.5 || uv.y > v0 + 9.0 || p.y > 5.0 || p.y < -1.0) return 1e9;   // (rocks4: +-40 % blocks)
    float fl = wfFaceLine(uv.y) + 0.2;
    float d = 1e9;
    for (int i = 0; i < 4; i++) {
        vec4 hs = hash42i(ivec2(i, 5151));
        float yT = 0.55 + 0.85 * float(i) + 0.45 * (hs.x - 0.3);         // tread height (rv6: jitter 0.25 -> 0.45)
        float uF = fl + 0.37 * yT - 0.75 - 0.3 * hs.y;                   // front edge (out of the 70 deg face)
        float vc = v0 + 1.2 + 1.3 * float(i) + 1.3 * (hs.z - 0.5);      // (rv6: jitter 0.6 -> 1.3 m, not a staircase)
        float hwS = 1.3 + 0.5 * hs.w;
        vec3 c = vec3(uF + 1.4, yT - 0.45, vc);
        vec3 q = vec3(uv.x, p.y, uv.y) - c;
        q.z += 0.18 * q.x;                                               // a little skew: not a staircase
        // (review rv5) every block tilted its own way (+-10 deg): the treads no longer stack as level plates (film shot 2)
        q.y += 0.18 * (hs.z - 0.5) * q.z + 0.14 * (hs.w - 0.5) * q.x;
        float yw = (hs.y - 0.5) * 0.9;                                   // (rv6) and yawed +-25 deg
        q.xz = mat2(cos(yw), sin(yw), -sin(yw), cos(yw)) * q.xz;
        // (rocks4 W15) sizes +-40 % and two more random cuts across the edges: polyhedral boulders, not a masonry stack
        vec4 hz = hash42i(ivec2(i, 7373));
        vec3 b = vec3(1.4, 0.45, hwS) * mix(vec3(0.6), vec3(1.4), hz.xyz) - 0.2;
        vec3 qa = abs(q) - b;
        float di = length(max(qa, 0.0)) + min(max(qa.x, max(qa.y, qa.z)), 0.0) - 0.2;
        // a worn nose and a slanted end: a natural ledge, not a concrete step
        di = max(di, dot(q, normalize(vec3(-0.75, 1.0, 0.25 * (hs.x - 0.5)))) - 0.62 - 0.2 * hs.z);
        di = max(di, dot(q, normalize(vec3(-0.3, 0.2, sign(hs.w - 0.5)))) - hwS * 0.85);
        di = max(di, dot(q, normalize(vec3(0.6 * (hz.w - 0.5), 1.0, 0.9 * (hz.x - 0.5)))) - (b.y + 0.2) * mix(0.75, 0.95, hz.y));
        di = max(di, dot(q, normalize(vec3(0.8, 0.5 + hz.z, 1.2 * (hz.w - 0.5)))) - length(b + 0.2) * mix(0.6, 0.75, hz.x));
        // (round 3 i9: + a finer octave - knobbly worn rock, not smooth boxes)
        if (di < 0.35) di -= (texture(uNoise3, p * 0.45 + float(i) * 0.37).r - 0.475) * 0.55
                            + (texture(uNoise3, p * 1.4 + float(i) * 0.71).g - 0.487) * 0.16;   // (review rv5: 0.35 -> 0.55, lumpier)
        if (di < d) { d = di; id = i; }
    }
    return d;
}

// distance to one plant / block
float ffPlantD(vec3 p, vec3 ctr, vec3 r, vec4 h, float foot) {
    vec3 ql = ffLocal(p - ctr);
    float rm = min(r.x, min(r.y, r.z));
    if (h.w >= 4.0) {
        // rock block: a chamfered rounded box, yawed / tilted a little; knobbly when resolved
        float yaw = (fract(h.w * 13.1) - 0.5) * 0.9, tl = (fract(h.w * 7.7) - 0.5) * 0.35;
        vec3 qb = ql;
        qb.xz = mat2(cos(yaw), sin(yaw), -sin(yaw), cos(yaw)) * qb.xz;
        qb.xy = mat2(cos(tl), sin(tl), -sin(tl), cos(tl)) * qb.xy;
        float rr = 0.32 * rm;
        vec3 qa = abs(qb) - (r - rr);
        float d = length(max(qa, 0.0)) + min(max(qa.x, max(qa.y, qa.z)), 0.0) - rr;
        // two chamfer cuts (angular, not a brick)
        vec3 c1 = normalize(vec3(fract(h.x * 3.1) - 0.5, 0.8, fract(h.y * 2.3) + 0.2));
        vec3 c2 = normalize(vec3(-(fract(h.x * 3.1) - 0.5) * 1.5, -0.2, 1.0));
        d = max(d, dot(qb, c1) - dot(r, abs(c1)) * 0.72);
        d = max(d, dot(qb, c2) - dot(r, abs(c2)) * 0.80);
        float vis1 = 1.0 - smoothstep(0.03, 0.10, foot);
        if (vis1 > 0.0 && d < 0.3) d -= ((texture(uNoise3, p * 0.35 + h.xyz * 7.0).r - 0.475) * 0.6 + (texture(uNoise3, p * 0.9 + h.zxy).r - 0.475) * 0.25) * rm * vis1;
        return d * 0.8;
    }
    if (h.w >= 3.0 && h.w < 4.0) {
        // mat: in the frame sheared to the ~70 deg face (z out of the face); a slab from deep in the rock out to
        // a cushioned surface th proud, cut by a lobed outline with tongues hanging below
        vec3 qd = ql;
        qd.z += 0.36 * qd.y;
        vec2 e2 = qd.xy / r.xy;
        float rad = length(e2);
        if (rad > 1.9) return (rad - 1.6) * min(r.x, r.y);
        vec2 dirE = e2 / max(rad, 1e-3);
        float lob = 0.62 + 0.55 * texture(uNoise3, vec3(dirE * 0.18 + h.xy * 3.0, 0.31)).r
                  + 0.30 * smoothstep(0.0, -0.8, dirE.y) * texture(uNoise3, vec3(qd.x * 0.35 + h.z * 7.0, 0.62, 0.17)).r;
        float edge = (rad - lob) * min(r.x, r.y);                       // < 0 inside the outline (m)
        float vis1 = 1.0 - smoothstep(0.05, 0.16, foot);
        float cu = vis1 > 0.0 ? texture(uNoise3, vec3(qd.xy * 0.24 + h.zx * 5.0, 0.43)).b - 0.525 : 0.0;   // ~0.5 m cushions
        float th = r.z * (0.75 + 1.4 * cu * vis1) * (0.35 + 0.65 * smoothstep(0.0, -0.6, edge));
        float dz = max(-(qd.z + 1.4), qd.z - th);
        float d = max(dz, edge);
        if (vis1 > 0.0 && d < 0.2) d -= (texture(uNoise3, p * 0.6 + h.xyz * 3.0).b - 0.525) * 0.12 * vis1;
        return d * 0.6;
    }
    if (h.w >= 2.0 && h.w < 2.55) {
        // face fern: the rosette frame tilted out of the face (axis up 37 deg / out), base at the rock
        vec3 qa = ql + vec3(0.0, 0.1 * r.y, 0.45 * r.z);
        const vec3 ax = vec3(0.0, 0.8, 0.6), az = vec3(0.0, -0.6, 0.8);
        vec3 qb = vec3(qa.x, dot(qa, ax), dot(qa, az));
        float bnd = length(qa / (r * 1.6)) - 1.0;
        if (bnd > 0.25) return bnd * rm;
        float N = floor(9.0 + 4.99 * fract(h.z * 3.7));
        return ffBlades(qb, N, r.x * 0.95, 0.12, foot, h.x, true) * 0.6;
    }
    if (h.w >= 0.42) {
        // fern rosette (7-11 fronds) / grass tuft (15-25 blades): the base sits on the ground (ctr.y = ground
        // + 0.3 r.y); unresolved fronds -> a flat leafy mass (the blades widen with the footprint, so the
        // rosette closes into a disc before it shrinks away)
        bool fern = h.w < 0.8;
        vec3 qb = ql + vec3(0.0, 0.30 * r.y, 0.0);
        float L = fern ? r.x * 0.95 : r.y * 0.92;
        float N = fern ? floor(7.0 + 4.99 * fract(h.z * 3.7)) : floor(12.0 + 8.99 * fract(h.z * 5.3));
        float bnd = length(qb / vec3(r.x * 1.15, r.y * 1.6, r.z * 1.15)) - 1.0;
        if (bnd > 0.25) return bnd * rm;
        float d = ffBlades(qb, N, L, fern ? 0.12 : 0.035, foot, h.x, fern);
        return d * 0.6;
    }
    // shrub: a leafy ellipsoid + lobes, ragged leaf masses
    float k0 = length(ql / r), k1 = length(ql / (r * r));
    float d = k0 * (k0 - 1.0) / max(k1, 1e-4);
    // a second lobe (55-75 % size) off to one side and up, and a third low on the other side:
    // irregular masses, not balls
    vec3 r2 = r * mix(0.55, 0.75, h.z);
    vec3 q2 = ql - vec3((h.x - 0.5) * 1.2 * r.x, 0.35 * r.y, (h.y - 0.5) * 0.6 * r.z);
    float k2 = length(q2 / r2), k3 = length(q2 / (r2 * r2));
    d = smin(d, k2 * (k2 - 1.0) / max(k3, 1e-4), 0.15 * r.y);
    vec3 r3 = r * vec3(0.6, 0.5, 0.6) * mix(0.8, 1.2, fract(h.x * 9.1));
    vec3 q3 = ql - vec3((0.5 - h.x) * 1.0 * r.x + 0.25 * r.x * sign(0.5 - h.x), -0.15 * r.y, (0.5 - h.y) * 0.9 * r.z);
    float k4 = length(q3 / r3), k5 = length(q3 / (r3 * r3));
    d = smin(d, k4 * (k4 - 1.0) / max(k5, 1e-4), 0.12 * r.y);
    if (d > 0.5 * r.y + 0.3) return d - 0.6 * rm;
    float vis1 = 1.0 - smoothstep(0.07, 0.22, foot);
    float vis2 = 1.0 - smoothstep(0.025, 0.08, foot);
    vec3 pn = p + h.xyz * 17.0;
    // (round 2) ragged leaf masses: deep lumps (~0.3 m) and clumps of leaves (~0.1 m) break the outline
    float n1 = vis1 > 0.0 ? texture(uNoise3, pn * 0.33).b - 0.525 : 0.0;
    float n2 = vis2 > 0.0 ? texture(uNoise3, pn * 1.05).b - 0.525 : 0.0;
    d -= (n1 * 2.7 * vis1 + n2 * 1.0 * vis2) * rm * 0.9;
    // shallow notches between ~5 radial branches: a ragged outline
    float frW = 6.2832 * r.x / 5.0;
    float visF = 1.0 - smoothstep(0.18, 0.4, foot / frW);
    if (visF > 0.0) {
        float ang = atan(ql.z, ql.x) + 0.25 * ql.y / r.y;
        float fr = abs(fract(ang * 5.0 / 6.2832 + h.z * 3.0) - 0.5) * 2.0;
        d += fr * fr * 0.18 * smoothstep(0.35, 0.95, k0) * min(r.x, r.z) * visF;
    }
    return d * 0.40;
}

float ffSDF(vec3 p, float t, out vec4 info) {
    info = vec4(0.0);
    if (uFallsOn == 0) return 1e9;
    vec2 uv = wfUV(p.xz);
    // outside the planted (u, v) range: a step to it (ffTrace's box is wider by the reach)
    float ob = max(max(-10.0 - uv.x, uv.x - 52.0), max(-60.0 - uv.y, uv.y - 50.0));
    if (ob > 0.0) { info.w = -1.0; return ob + 0.02; }
    // (r3a, perf) in front of the cliff foot beyond every plant's / boulder's reach (the pool / lake in
    // front of the face: rays over the water marched the whole box at full cost - r4 x3): a safe step
    // (margin 2 m for the face line's wiggle over the reach, its gradient <= ~1.7 -> x0.55)
    float gapF = -3.2 - FF_REACH - 2.0 - (uv.x - wfFaceLine(uv.y));
    if (gapF > 0.0) { info.w = -1.0; return gapF * 0.55 + 0.02; }
    // (r3d, perf) high above the local ground (> 12 m: on a <= 70 deg slope >= 4.4 m from it horizontally, beyond
    // every plant's reach + thickness): a safe step (the sky / far rays over the banks and the rim)
    float hiG = p.y - terrainH(p.xz, false) - 12.0;
    if (hiG > 0.0) { info.w = -1.0; return hiG * 0.3 + 0.02; }
    float foot = t * gPixAng;
    vec2 cu = uv / FF_CELL - 0.5;
    vec2 cf = floor(cu);
    ivec2 c0 = ivec2(cf);
    vec2 f = cu - cf;
    float d = 1e9;
    FfCol cA = ffCol(c0.y), cB = ffCol(c0.y + 1);
    for (int j = 0; j < FF_NLOOP; j++) {
        ivec2 c = c0 + ivec2(j & 1, j >> 1);
        vec3 ctr, r; vec4 h;
        if (!ffPlantC(c, (j >> 1) == 0 ? cA : cB, foot, ctr, r, h)) continue;
        float bs = length(p - ctr) - max(r.x, max(r.y, r.z)) * 1.75;
        float dj = bs > 0.35 ? bs : ffPlantD(p, ctr, r, h, foot);
        if (dj < d) info = vec4(ctr, float((c.x + 64) * 512 + (c.y + 256)));
        d = h.w >= 0.42 ? min(d, dj) : smin(d, dj, 0.25);      // (blades / blocks: no blending)
    }
    // pale steps under fall B (review r2, fix 5): cell code x = -60 + step
    int sid;
    float dS = ffStepsB(p, sid);
    if (dS < d) { d = dS; info = vec4(p, float((sid - 60 + 64) * 512 + 256)); }
    // the plants of the cells outside this 2 x 2 block are at least this far away
    // (only a step size: info.w < 0 = no plant surface here, the caller must not count it as a hit)
    float bound = FF_CELL * (1.0 - FF_JIT + min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y))) - FF_REACH;
    if (bound < d) { info.w = -1.0; return max(bound, 0.05); }
    return d;
}

ivec2 ffCellOf(vec4 info) { int pk = int(info.w + 0.5); return ivec2(pk / 512 - 64, pk % 512 - 256); }

// (round 2, perf) the flora is marched on its own AFTER the primary trace (scene.frag render()), from
// where the ray enters the flora's box up to the primary hit tmax: the hit distance (1e9 = none) and
// info as ffSDF. (Inside trace()'s march loop it was inlined into all three trace() call sites and its
// registers weighed on every march step of every view, ~x4 frame time.) Off for the prep / kind
// renders of the lab harnesses (uDebug 1, 48-51) and timing toggle 70.
#define FF_STEPS 72
float ffTrace(vec3 ro, vec3 rd, float tmax, out vec4 info) {
    info = vec4(0.0);
    if (uFallsOn == 0 || uDebug == 1 || (uDebug >= 48 && uDebug <= 51) || wfOff(0)) return 1e9;
    // box of the plants in the falls frame (u inland, y, v along the shore): ffSDF's (u, v) range + reach
    vec3 lo = fallLocal(ro);
    vec3 ld = vec3(-dot(rd.xz, uWFDir), rd.y, dot(rd.xz, vec2(-uWFDir.y, uWFDir.x)));
    vec3 inv = 1.0 / (ld + sign(ld) * 1e-6 + vec3(1e-9));
    vec3 t0 = (vec3(-10.0 - FF_REACH, -1.5, -60.0 - FF_REACH) - lo) * inv;   // (r3b: u from -10: the base boulders in front of the buttress)
    vec3 t1 = (vec3(52.0 + FF_REACH, uWF.z + 22.0, 50.0 + FF_REACH) - lo) * inv;
    vec3 tn = min(t0, t1), tf = max(t0, t1);
    float t = max(max(tn.x, tn.y), max(tn.z, 0.05));
    float tb = min(min(tf.x, tf.y), min(tf.z, min(tmax, FF_MAX_DIST)));
    if (tb <= t) return 1e9;
    for (int i = 0; i < FF_STEPS; i++) {
        vec3 p = ro + rd * t;
        vec4 fi;
        float d = ffSDF(p, t, fi);
        float eps = max(0.002, t * gPixAng * 0.6);
        if (d < eps && fi.w > 0.0) { info = fi; return t; }
        t += max(d, eps * 0.7);
        if (t > tb) break;
    }
    return 1e9;
}

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
vec3 shadeStepsB(vec3 p, vec3 rd, float t, int sid) {
    float e = max(0.01, t * gPixAng);
    int i_;
    vec2 k = vec2(1.0, -1.0);
    vec3 n = normalize(k.xyy * ffStepsB(p + k.xyy * e, i_) + k.yyx * ffStepsB(p + k.yyx * e, i_)
                     + k.yxy * ffStepsB(p + k.yxy * e, i_) + k.xxx * ffStepsB(p + k.xxx * e, i_));
    float fp = t * gPixAng;
    n = normalize(n + (texture(uNoise3, p * 1.3).rgb - 0.5) * 0.5 * (1.0 - smoothstep(0.02, 0.06, fp)));
    float g = texture(uNoise3, p * 0.6 + 0.3).r;
    // (review rv5) + darker joints / weathering streaks down the faces: worn granite, not pale plates
    float wst = mix(0.5, texture(uNoise3, vec3(p.x * 0.9, p.y * 0.25, p.z * 0.9) + 0.21).r, 1.0 - smoothstep(0.03, 0.09, t * gPixAng));
    // (rocks4 W15: 0.068 -> 0.045, the dark wet rock of the cliff after the wet darkening, #20221e; + a moss cap below)
    vec3 alb = vec3(0.045, 0.045, 0.040) * mix(0.75, 1.2, g) * mix(0.85, 1.1, hash12i(ivec2(sid, 77))) * mix(0.6, 1.15, wfRemap(wst));   // (round 3: 0.13 -> 0.10; rv5 0.082)
    vec4 kb = uFallK[2 * WF_NK - 1];
    vec2 uv = wfUV(p.xz);
    // the fall's water fans over the left part of the steps: white threads running down the risers and
    // across the treads, wet dark rock between
    float inF = 1.0 - smoothstep(kb.z * 1.05, kb.z * 1.25 + 0.8, abs(uv.y - kb.y));
    float th = texture(uNoise3, vec3(uv.y * 1.8, (p.y + uTime * 2.5) * 0.25, 0.3)).r;
    float wt = inF * smoothstep(0.45, 0.6, th + 0.15 * (inF - 0.5)) * mix(0.35, 0.9, 1.0 - smoothstep(0.03, 0.1, fp));   // (round 3: 0.6 -> 0.35 unresolved)
    float wetS = max(inF, 0.35);
    alb *= mix(1.0, 0.45, wetS);
    alb *= mix(0.55, 1.0, smoothstep(-0.2, 0.6, n.y));                       // the risers' undersides darker
    // (rocks4 W15) lime moss cap on the tops (as the cliff's ledges in the spray)
    float mN = texture(uNoise3, p * 0.45 + 0.17).r;
    alb = mix(alb, mix(vec3(0.045, 0.06, 0.012), vec3(0.12, 0.14, 0.025), smoothstep(0.35, 0.65, mN)),
              0.85 * smoothstep(0.55, 0.8, n.y + 0.5 * (mN - 0.475)) * (1.0 - inF));
    alb = mix(alb, vec3(0.62, 0.64, 0.65), wt);   // (review rv8: 0.80 -> 0.62, the threads over the steps read as white plates)
    float occ = mix(0.45, 1.0, smoothstep(0.0, 0.9, p.y - terrainH(p.xz, false) + 0.6)) * terrainAO(p.xz);
    float sun = sunShadowMap(p + n * 0.1) * cloudShadow(p) * (1.0 - ffShadow(p, t));
    if (sun > 0.0) sun *= treeShadow(p + n * 0.2);
    return litSurface(alb, n, rd, sun, occ, 0.6 * wetS * (1.0 - wt), 0.25);
}

vec3 shadeFlora(vec3 p, vec3 rd, float t, vec4 info) {
    ivec2 c = ffCellOf(info);
    if (c.x <= -50) return shadeStepsB(p, rd, t, c.x + 60);
    float foot = t * gPixAng;
    vec3 ctr, r; vec4 h;
    ffPlant(c, foot, ctr, r, h);
    float e = max(0.008, foot * 1.0);
    // tetrahedral normal in a rolled loop (one inlined copy of ffPlantD instead of four: compile time)
    vec3 n = vec3(0.0);
    for (int i = 0; i < FF_NLOOP; i++) {
        vec3 kk = vec3(((i + 3) >> 1) & 1, (i >> 1) & 1, i & 1) * 2.0 - 1.0;
        n += kk * ffPlantD(p + kk * e, ctr, r, h, foot);
    }
    n = normalize(n);
    vec3 ql = ffLocal(p - ctr);
    // falls lab debug 16: plant kinds (shrub red, fern green, grass blue, face fern yellow, drape magenta, block white)
    if (uDebug == 16) return h.w >= 4.0 ? vec3(1.0) : h.w >= 3.0 ? vec3(0.0, 1.0, 1.0) : (h.w >= 2.55 ? vec3(1.0, 0.0, 1.0) : (h.w >= 2.0 ? vec3(1.0, 1.0, 0.0) : (h.w >= 0.8 ? vec3(0.0, 0.0, 1.0) : (h.w >= 0.42 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0))))) * (0.3 + 0.7 * saturate(dot(n, uSunDir)));
    if (h.w >= 4.0) {
        // (review r2, fix 8) rock block: the cliff's dark wet rock - lit top facets, near-black undersides,
        // pale lichen on dry tops of the face blocks, moss tufts on top; the base boulders wet to the waterline
        float vR = 1.0 - smoothstep(0.02, 0.06, foot);
        n = normalize(n + (texture(uNoise3, p * 1.4 + h.xyz * 3.0).rgb - 0.5) * 0.5 * vR);
        float g = mix(0.5, texture(uNoise3, p * 0.7 + h.zxy).r, 1.0 - smoothstep(0.04, 0.12, foot));
        vec3 alb = vec3(0.024, 0.025, 0.022) * mix(0.75, 1.3, wfRemap(g)) * mix(0.85, 1.2, fract(h.x * 11.3));
        bool base = h.w < 4.5;
        float dryT = base ? 0.0 : smoothstep(0.3, 0.8, n.y) * smoothstep(0.35, 0.6, fract(h.y * 7.1));
        float lich = smoothstep(0.55, 0.7, texture(uNoise3, p * 0.45 + h.xyz).g) * dryT;
        alb = mix(alb, vec3(0.20, 0.205, 0.18), lich * 0.5);
        alb *= mix(1.0, 1.9, smoothstep(0.1, 0.7, n.y));                     // lit tops paler, undersides black
        float mo = smoothstep(0.55, 0.85, n.y) * smoothstep(0.45, 0.6, texture(uNoise3, p * 1.1 + h.yzx).b + 0.2 * (fract(h.z * 3.3) - 0.5));
        alb = mix(alb, vec3(0.09, 0.105, 0.018) * mix(0.7, 1.2, g), mo * (base ? 0.5 : 0.8));
        float wl = waterLevel(p.xz);
        float wet = base ? max(0.6, 1.0 - smoothstep(0.0, 0.5, p.y - max(wl, 0.0))) : 0.3;
        alb *= mix(1.0, 0.55, wet * (1.0 - mo));
        float hy = ql.y / max(r.y, 0.1);
        float occ = mix(0.35, 1.0, smoothstep(-1.0, 0.3, hy)) * mix(0.5, 1.0, smoothstep(-0.6, 0.2, n.y)) * terrainAO(p.xz);
        float sun = sunShadowMap(p + n * 0.1) * cloudShadow(p) * (1.0 - ffShadowX(p, t, c));
        if (sun > 0.0) sun *= treeShadow(p + n * 0.2);
        return litSurface(alb, n, rd, sun, occ, 0.55 * wet * (1.0 - mo), 0.3);
    }
    // leaf-cluster normals: each clump of leaves faces its own way (lit and shaded clumps all over the
    // plant, not one terminator across a ball); unresolved -> the smooth normal. (blades: their own
    // ribbon normal, only a little leaf tilt)
    float vN = 1.0 - smoothstep(0.04, 0.12, foot);
    bool mat = h.w >= 3.0;
    n = normalize(n + (texture(uNoise3, p * 0.85 + h.zxy * 3.0).rgb - 0.5) * (h.w < 0.42 ? 2.4 : (mat ? 1.6 : 0.8)) * vN);
    // (drapes: the sheared frame of ffPlantD - pS is the point in the unsheared ellipsoid)
    vec3 pS = p + (h.w >= 2.55 ? vec3(uWFDir.x, 0.0, uWFDir.y) * 0.36 * ql.y : vec3(0.0));
    ql = ffLocal(pS - ctr);
    float hy = ql.y / r.y;                                  // -1 bottom .. +1 top of the plant
    float k0 = length(ql / r);                              // ~1 at the outer leaves, < 1 inside
    float shell = smoothstep(0.55, 1.0, k0);
    // leaves: sunlit leaf clusters and dark holes between them (unresolved -> their mean)
    float vL = 1.0 - smoothstep(0.02, 0.06, foot), vC = 1.0 - smoothstep(0.05, 0.15, foot);
    float lf = mix(0.525, texture(uNoise3, p * 2.4 + h.xyz * 5.0).b, vL);
    float lc = mix(0.525, texture(uNoise3, p * 0.9 + h.yzx * 3.0).b, vC);
    float lm = mix(0.525, texture(uNoise3, p * 0.42 + h.zyx * 2.0).b, 1.0 - smoothstep(0.12, 0.35, foot));
    float lv = smoothstep(0.2, 0.8, 0.5 + 2.2 * (lf - 0.525) + 1.8 * (lc - 0.525) + 2.0 * (lm - 0.525) + 0.15 * hy);
    float strand = 1.0;
    bool blade = h.w >= 0.42 && !mat;
    if (blade) {
        // fronds / blades / strands: each its own tone, lit toward the tip / the upper side, the base and
        // the lower ends in the plant's own shade (gaps show the rock: no dark holes needed here)
        float rad = length(ql.xz) / max(r.x, 0.1);
        float bt = hash12i(ivec2(int(atan(ql.z, ql.x) * 3.0 + 64.0), int(h.x * 4096.0)));
        strand = mix(0.7, 1.2, bt);
        lv = h.w >= 2.0 ? smoothstep(-0.6, 0.6, hy + 0.4 * (lf - 0.5)) : smoothstep(0.05, 0.6, (h.w < 0.8 ? rad : hy * 0.5 + 0.5) + 0.3 * (lf - 0.5));
        lv = mix(lv, 0.5 + 0.5 * lv, 1.0 - vC);
    }
    float tn = fract(h.y * 13.1);
    vec3 dk, lt;
    if (h.w < 0.42)     { dk = vec3(0.014, 0.024, 0.009); lt = mix(vec3(0.056, 0.086, 0.022), vec3(0.112, 0.140, 0.032), tn); }  // shrub (review rv5: x0.8 - lime balls on the rim, film shot 2)
    else if (h.w < 0.8) { dk = vec3(0.025, 0.036, 0.007); lt = mix(vec3(0.112, 0.144, 0.024), vec3(0.180, 0.208, 0.032), tn); }  // fern (#9da136 lit; review r2: ~x1.4 with real gaps)
    else if (h.w >= 3.0) { dk = vec3(0.012, 0.022, 0.007); lt = mix(vec3(0.080, 0.118, 0.030), vec3(0.150, 0.185, 0.048), tn); }   // (round 3: x0.75)  // mat (fern / grass / moss; r3a x1.4, r3c less yellow)
    else if (h.w >= 2.0 && h.w < 2.55) { dk = vec3(0.020, 0.030, 0.007); lt = mix(vec3(0.104, 0.140, 0.024), vec3(0.162, 0.195, 0.032), tn); }  // face fern (review rv8: x0.65 - isolated lime pom-poms on the face, film shot 2)
    else                { dk = vec3(0.034, 0.040, 0.014); lt = mix(vec3(0.105, 0.125, 0.034), vec3(0.165, 0.175, 0.048), tn); }  // grass
    // (round 2) a plant only a few pixels wide is a speck of mixed leaves and shadow, not a lit lime ball
    lt *= mix(1.0, blade ? 0.8 : 0.68, smoothstep(0.04, 0.14, foot / max(r.y, 0.3)));
    // (session 3) every plant its own tint: bluish dark shrubs, yellow-green ferns, brownish grass
    float tv = fract(h.x * 17.3 + h.z * 3.1);
    lt *= mix(vec3(0.78, 0.88, 1.10), vec3(1.15, 1.04, 0.78), tv) * mix(0.75, 1.1, fract(h.y * 7.7));
    vec3 alb = mix(dk, lt, lv) * strand;
    // depth into the foliage: the inside and the underside see little sky
    float occ = (blade || mat ? mix(mat ? 0.3 : 0.45, 1.0, smoothstep(-0.8, 0.6, hy)) : mix(0.18, 1.0, shell) * mix(0.35, 1.0, smoothstep(-0.9, 0.5, hy)))
              * mix(0.55, 1.0, lv) * terrainAO(p.xz);
    // sun: own self-shadow (path through the ellipsoid), the neighbours' shadows, terrain / trees
    float self_ = exp(-(blade ? 1.4 : 3.0) * ffSunPath(pS - n * 0.05, ctr, r));
    float sun = sunShadowMap(p + n * 0.1) * cloudShadow(p) * self_ * (1.0 - ffShadowX(p, t, c));
    if (sun > 0.0) sun *= treeShadow(p + n * 0.2);
    // in the spray: wet, darker leaves
    float wetF = fallsWetness(p) * 0.4;
    alb *= 1.0 - 0.3 * wetF;
    // (thin blades: lit from both sides - the sun through a frond lights its shaded face too)
    vec3 nl = blade && dot(n, uSunDir) < 0.0 ? normalize(n - 1.4 * uSunDir * dot(n, uSunDir)) : n;
    vec3 col = litSurface(alb, nl, rd, sun * (blade && dot(n, uSunDir) < 0.0 ? 0.45 : 1.0), occ, 0.14 + 0.2 * wetF, 0.42);
    // back-lit leaves glow (translucent ferns / grass)
    float bl = pow(saturate(dot(rd, uSunDir)), 3.0);
    col += lt * vec3(1.0, 1.05, 0.6) * uSunCol * sun * bl * 0.4 * (blade ? 1.0 : shell);   // (review rv9: 0.55 -> 0.4)
    return col;
}

// ---------------------------------------------------------------------------------------------
// inflow stream: boulders of the stream bed and the open floodplain (bake_features.frag
// carves them: voronoiV(p * 0.36 + 13) cells, channel half width = 0.6 x the floodplain one)
uniform vec4 uStream[96];
uniform int uStreamN;
uniform vec4 uStreamBox;
uniform float uStreamS[96];                 // (round 3) arc length (m) of each polyline point from the first (render.py)
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
float poolFoam(vec2 xz, out vec2 tilt) { return poolFoamF(xz, 0.0, tilt); }

// where the inflow stream meets the lake (render.py uStreamMouth: x, z, downstream dir x, z)
uniform vec4 uStreamMouth;
#endif
