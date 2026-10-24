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
uniform sampler2D uStreamCorr; uniform vec4 uStreamCorrBox;
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
