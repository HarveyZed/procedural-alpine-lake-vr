#version 450
// Near height map (1 m texels) = Catmull-Rom upsampled far map + fine analytic detail.
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

// 2D Voronoi: returns (F1 distance, F2 distance, cell id hash)

// voronoi with the offset to the nearest feature point: (dx, dy, F1, id)

mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }

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

// ridged multifractal split into a low (massif) part and a high (crag) part.
// detail is suppressed in the valleys of the previous octave.
// (session 6: dlo = amplitude decay of the low (massif) octaves; 0.48 = the original. A faster decay
// gives fewer, bigger summits with planar faces instead of a comb of equal sub-peaks)

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

layout(set=0, binding=0) uniform sampler2D uHFar;
// port(P1): the engine's loose pass uniforms, grouped into one block for Vulkan GLSL (unnamed instance -> members stay global)
layout(std140, set=1, binding=0) uniform _p1_pass {
    float uFarHalf, uFarRes;
    vec2 uCenter;
    float uHalf, uRes;
};




layout(location=0) out vec4 fragColor;

float farTexel(ivec2 c) { return texelFetch(uHFar, clamp(c, ivec2(0), ivec2(int(uFarRes) - 1)), 0).r; }
float crW(float x, int i) {
    // Catmull-Rom weights for taps -1,0,1,2
    if (i == 0) return ((-0.5 * x + 1.0) * x - 0.5) * x;
    if (i == 1) return (1.5 * x - 2.5) * x * x + 1.0;
    if (i == 2) return ((-1.5 * x + 2.0) * x + 0.5) * x;
    return (0.5 * x - 0.5) * x * x;
}
float farCR(vec2 p) {
    vec2 g = (p / (2.0 * uFarHalf) + 0.5) * uFarRes - 0.5;
    vec2 i = floor(g); vec2 f = g - i; ivec2 c = ivec2(i);
    float s = 0.0;
    for (int y = 0; y < 4; y++) {
        float row = 0.0;
        for (int x = 0; x < 4; x++) row += crW(f.x, x) * farTexel(c + ivec2(x - 1, y - 1));
        s += crW(f.y, y) * row;
    }
    return s;
}
void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = uCenter + (uv * 2.0 - 1.0) * uHalf;
    float h = farCR(p);
    float hx = farCR(p + vec2(2.0, 0.0)), hz = farCR(p + vec2(0.0, 2.0));
    float slope = length(vec2(hx - h, hz - h) / 2.0);
    float fine = heightFine(p, slope);
    fine *= mix(0.35, 1.0, smoothstep(-1.0, 3.0, h));
    fragColor = vec4(h + fine, 0.0, 0.0, 1.0);
}
