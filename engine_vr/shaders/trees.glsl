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
uniform sampler2D uTreeTop;
float treeFreeDist(vec3 p) {
    ivec2 b = (treeCellOf(p.xz) - ivec2(TREE_TEX_OFF)) >> 2;
    if (any(lessThan(b, ivec2(0))) || any(greaterThanEqual(b, ivec2(TREE_TEX_RES / 4))))
        return max(abs(p.x), abs(p.z)) - float(TREE_TEX_RES / 2) * TREE_CELL - 12.0;   // outside the tree map (r4 merge: reach <= ~11.6 m)
    vec2 tg = texelFetch(uTreeTop, b, 0).rg;
    return min(p.y - tg.r, tg.g);
}

#if SCENE_PASS == 98 || SCENE_PASS == 99 || SCENE_PASS == 4
// (perf4) the marcher's distance to everything else (and its hit threshold): trees whose bound lies beyond it can
// neither be hit nor shorten the step, so they are not evaluated (trace() / coneTrace() set it before treesSDF)
float gTreeDMax = 1e9;
#endif
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
