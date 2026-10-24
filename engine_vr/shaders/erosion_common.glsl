// ---------------------------------------------------------------------------
// erosion_common.glsl : shared helpers for the GPU erosion compute passes
// Grid: texel (x = column = world x, y = row = world z), square cells of uCell m.
// Height state texture: rg = (bedrock height, loose layer thickness) in meters.
// All passes work on 16x16 tiles staged in shared memory with a 1-2 cell halo;
// uShift alternates the tiling between dispatches so tile seams move around.
// ---------------------------------------------------------------------------
#define TS 16
layout(local_size_x = TS, local_size_y = TS) in;

uniform ivec2 uRes;      // grid size
uniform ivec2 uOffset;   // dispatch offset in tiles (for split dispatches)
uniform ivec2 uShift;    // tiling shift in cells
uniform float uCell;     // cell size in meters

const ivec2 OFF[8] = ivec2[8](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1),
                              ivec2(1, 1), ivec2(-1, 1), ivec2(1, -1), ivec2(-1, -1));
const float DST[8] = float[8](1.0, 1.0, 1.0, 1.0, 1.41421356, 1.41421356, 1.41421356, 1.41421356);

bool inb(ivec2 p) { return p.x >= 0 && p.y >= 0 && p.x < uRes.x && p.y < uRes.y; }
bool onBorder(ivec2 p) { return p.x == 0 || p.y == 0 || p.x == uRes.x - 1 || p.y == uRes.y - 1; }
float Htot(sampler2D t, ivec2 p) { vec2 v = texelFetch(t, p, 0).xy; return v.x + v.y; }
// world cell of shared-tile entry (lx, ly) for a tile with halo `halo`
ivec2 tileOrigin(int halo) { return (ivec2(gl_WorkGroupID.xy) + uOffset) * TS - uShift - halo; }
