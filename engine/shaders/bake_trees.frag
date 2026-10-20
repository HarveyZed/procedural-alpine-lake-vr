// Bakes per-cell tree parameters (4 m cells, +-8192 m): baseY, signed height (neg = larch), crown radius, crown base
#include "common.glsl"
#include "world.glsl"
#define TREE_TEX_OFF 2048
out vec4 fragColor;
void main() {
    ivec2 cell = ivec2(gl_FragCoord.xy) + ivec2(TREE_TEX_OFF);
    vec4 f = texelFetch(uForest, cell, 0);
    float dens = f.r;
    vec4 hs = hash42i(cell);
    fragColor = vec4(0.0);
    if (dens < 0.03 || hs.x > dens * 0.97) return;
    vec2 pos = treeCellOrigin(cell) + TREE_CELL * (0.5 + (hs.yz - 0.5) * 0.6);
    float h = terrainH(pos, true);
    if (h < waterLevel(pos) + 0.6) return;
    // steep rock under the tree position -> no tree
    float hx = terrainH(pos + vec2(2.0, 0.0), true), hz = terrainH(pos + vec2(0.0, 2.0), true);
    float slope = length(vec2(hx - h, hz - h) / 2.0);
    if (slope > 1.6) return;
    float hn = hash12i(cell + ivec2(911, 37));
    float larch = step(hash12i(cell + ivec2(5, 77)), f.g * 0.55);
    float alt = smoothstep(250.0, 800.0, h);
    float open = 1.0 - dens;
    float H = mix(13.0, 31.0, hs.w * hs.w * 0.4 + hs.w * 0.6) * mix(1.0, 0.55, alt) * mix(1.0, 0.8, open);
    float R = H * mix(0.115, 0.16, hn) * (larch > 0.5 ? 1.15 : 1.0) * mix(1.0, 1.25, open);
    float yb = H * mix(0.0, 0.18, dens * hn);
    fragColor = vec4(h - 0.25, larch > 0.5 ? -H : H, R, yb);
}
