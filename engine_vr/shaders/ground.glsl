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
#ifndef GC_FADE0
#define GC_FADE0 (uDebug == 30 ? 5.0 : 9.5)     // cover starts thinning out (by count; the ground takes over)
#define GC_FADE1 (uDebug == 30 ? 10.0 : 20.0)   // ... and is gone
#endif
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
#ifdef GC_TRAIL
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
#else
    return 1e9;
#endif
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
bool gcLeaf(vec3 o, vec3 d, vec3 R, vec2 a, float H, float L, float w, float sMax,
            out float sHit, out float vHit, out vec3 nHit) {
    vec2 ss; vec4 ab; float k;
    if (!gcBent(o, d, R, a, H, L, ss, ab, k)) return false;
    for (int j = 0; j < 2; j++) {
        float s = j == 0 ? ss.x : ss.y;
        if (s < 0.0 || s >= sMax) continue;
        float y = ab.x + ab.y * s;
        float v = y / H;
        if (v < 0.0 || v > 1.0) continue;
        float wv = w * pow(max(sin(PI * pow(v, 0.75)), 0.0), 0.7) + 0.001;
        if (abs(ab.z + ab.w * s) > 0.5 * wv) continue;
        sHit = s; vHit = v;
        nHit = normalize(vec3(a.x, -2.0 * k * y, a.y));
        return true;
    }
    return false;
}

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
#ifdef GCP_SEGONLY
    gcSegHit = false; gcSafeT = t0 + sEnd; return;
#endif
    // coarse objects (shrubs, stones) whose reach (cell grown by GC_CO) the segment passes over
    vec2 e = o.xz + d.xz * sEnd;
    vec2 ivS = 1.0 / max(abs(d.xz), vec2(1e-7)) * vec2(d.x >= 0.0 ? 1.0 : -1.0, d.z >= 0.0 ? 1.0 : -1.0);
    ivec2 ca = ivec2(floor((min(o.xz, e) - GC_CO) / GC_C2)), cb = ivec2(floor((max(o.xz, e) + GC_CO) / GC_C2));
#ifndef GCP_NOCOARSE
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
#endif
    // tussocks in the 1 m cells the segment crosses
#ifndef GCP_NOTUS
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
#endif
    // medium grid DDA: forb rosettes, flower clumps, berry bushes
    vec2 dxz = d.xz;
    vec2 sg = vec2(dxz.x >= 0.0 ? 1.0 : -1.0, dxz.y >= 0.0 ? 1.0 : -1.0);
    vec2 inv = 1.0 / max(abs(dxz), vec2(1e-6));
    float sDoneM = sEnd;
#ifndef GCP_NOMID
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
#endif
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
#ifndef GCP_NOFINE
        if (min(y0, y1) - yC < GC_GTOP + 0.12) gcFine(cell, o, d, yC, gC, sBest);
#endif
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
#ifdef GCP_NOCOVER
    return 1e9;
#endif
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
#if defined(SCENE_PASS) && SCENE_PASS != 99
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
#endif
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

#ifndef GC_FAR
#define GC_FAR 0.85         // grass seen from afar: the blades' mean colour x this (canopy shading)
#endif
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
#ifdef GCP_NOALB
    return vec3(0.08, 0.1, 0.02);
#endif
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
#ifdef GC_TRAIL
    if (uTrailN > 1 && camD < 120.0) {
        for (int i = 0; i < 31; i++) {
            if (i + 1 >= uTrailN) break;
            vec2 a2 = uTrail[i].xy, ab2 = uTrail[i + 1].xy - a2;
            dTr = min(dTr, length(x - a2 - ab2 * clamp(dot(x - a2, ab2) / dot(ab2, ab2), 0.0, 1.0)));
        }
    }
#endif
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
#if defined(SCENE_PASS) && SCENE_PASS == K_WATER
    waterPass = true;            // terrain seen in reflections / through the water: no shell volumes
#endif
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
#ifndef GC_SKYOCC
        g *= mix(1.0, 0.8, canopy) * mix(1.0, 0.9, stand);
#endif
    }
    return g;
}
