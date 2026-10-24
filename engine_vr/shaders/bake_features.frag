// Carves hand-placed world features into a height map (run on the far and the near grid):
//  - the waterfall rock step: a ~29 m stepped, blocky face set into the east-shore hillside, with a
//    chute for each fall, a cleft for the lower fall B, base boulders, a deep plunge pool and a low
//    rock apron on the bank north of it (shape functions shared with falls.glsl)
//  - a boulder-choked inflow stream channel at the north end of the lake
// Outputs: r = new height, g = stream water level (-1e4 = none), b = foam factor, a = flow angle
#include "common.glsl"
#define FALLS_SHAPE_ONLY
#include "falls.glsl"
uniform sampler2D uH;
uniform vec2 uCenter;
uniform float uHalf, uRes;
// stream: polyline points (x, z, bed height, half width)
uniform vec4 uStream[96];
uniform int uStreamN;
out vec4 fragColor;

float Hs(vec2 uv) { return texture(uH, uv).r; }

// staircase rise 0..1 over x = 0..1: three tall steep risers between ~2.5 m sloping ledges
// (the near map is 1 m texels read through a B-spline: anything finer is smoothed away)
float wfRise(float x, float v) {
    x = saturate(x);
    float acc = 0.12 * x;
    float amp[3] = float[3](0.34, 0.30, 0.24);
    for (int k = 0; k < 3; k++) {
        float c = (float(k) + 0.45) / 3.0 + 0.06 * gnoise(vec2(v * 0.09, float(k) * 3.7 + 1.0));
        acc += amp[k] * smoothstep(c - 0.07, c + 0.05, x);
    }
    return acc;
}

// forested bank profile 0..1 over x = 0..1 (~45 deg) with two benches ~5 m wide that hold the
// forest; the risers between them steepen to ~55 deg (dark outcrops show there)
float wfBank(float x, float v, float W) {
    float bw = 5.0 / max(W, 12.0);
    float acc = 0.0;
    for (int k = 0; k < 2; k++) {
        float c = (k == 0 ? 0.36 : 0.70) + 0.05 * gnoise(vec2(v * 0.045, float(k) * 5.1 + 2.0));
        acc += clamp(x - (c - 0.5 * bw), 0.0, bw);
    }
    return saturate((x - 0.88 * acc) / (1.0 - 1.76 * bw));
}

// height of the step's face / rim at (shifted) inland distance uf from the foot line
float wfFaceProfile(vec2 p, float u, float v, float vEff, float uf, float W, float hs, float base, float cl, float bt,
                float cA, float clB) {
    float hp;
    float x = uf / W;
    // cliff: stepped risers; forested banks: ~45 deg with two benches
    float prof = mix(wfBank(x, vEff, W), wfRise(x, vEff), cl);
    if (uf < W) hp = base + (hs - base) * prof;
    // behind the rim: a nearly level shelf for ~8 m (the forest stands right at the edge, no rounded
    // dome), then the hillside climbs again
    else hp = hs + 0.05 * min(uf - W, 8.0) + 0.22 * max(uf - W - 8.0, 0.0) + 0.9 * gnoise(p * 0.055) * smoothstep(3.0, 14.0, uf - W);
    // buttress: a near-vertical wall up to ~0.4 of the step, the bank / cliff profile above it
    if (bt > 0.0 && uf < W) {
        float yb = wfButtH(v, hs);
        // (max: the block fixed-point pass can hand in uf < 0 -> pow(negative) = NaN -> a needle spike)
        float hb = uf < WF_BUTT_W ? base + (yb - base) * pow(max(uf, 0.0) / WF_BUTT_W, 0.6)
                 : yb + (hs - yb) * mix(wfBank((uf - WF_BUTT_W) / (W - WF_BUTT_W), vEff, W), wfRise((uf - WF_BUTT_W) / (W - WF_BUTT_W), vEff), cl);
        hp = mix(hp, hb, bt);
    }
    // fall A: notch in the rim and a shallow stream bed inland
    float nA = 1.0 - smoothstep(uWFNotch.z * 0.5, uWFNotch.z * 0.5 + 1.6, abs(v - wfChuteV(0, u)));
    hp -= nA * uWFNotch.w * smoothstep(W - 3.0, W - 0.5, uf) * (1.0 - smoothstep(W + 25.0, W + 45.0, uf));
    // fall A's chute: cut back into the rim, steep risers between two ledges where the water lands
    if (cA > 0.0 && uf < W + WF_CUT_A + 1.0) {
        float Wc = W + WF_CUT_A;
        float yT = hs - uWFNotch.w;
        float hc = wfStepped(min(uf, Wc), Wc, yT, base, wfLedges(0, hs), WF_SLAB_A);
        hp = mix(hp, min(hp, hc), cA);
    }
    // fall B: cleft with one ledge
    if (clB > 0.0) {
        float hB = hs * WF_LIPB;
        float WB = W * 0.45 + WF_CUT_B;
        float hc = uf < WB ? wfStepped(uf, WB, hB, base, wfLedges(1, hs), 1.0) : hB + 0.95 * (uf - WB);
        hp = mix(hp, min(hp, hc), clB);
    }
    // shrub masses on the rim shelf behind the bare face (a ragged outline, not a bald dome; the forest
    // bake's slope test puts no trees there); none in fall A's notch or fall B's cleft
    float rimS = smoothstep(W - 2.5, W - 0.5, uf) * (1.0 - smoothstep(W + 6.0, W + 9.0, uf))
               * smoothstep(-24.0, -14.0, v) * (1.0 - smoothstep(28.0, 38.0, v))
               * smoothstep(2.5, 4.5, abs(v - wfChuteV(0, u))) * (1.0 - clB);
    if (rimS > 0.0) hp += rimS * wfRimShrub(p);
    return hp;
}

float wfCarve(vec2 p, float h) {
    vec2 uv = wfUV(p);
    float u = uv.x, v = uv.y;
    if (u < -90.0 || u > 260.0 || v < -140.0 || v > 150.0) return h;
    float cl = wfCliff(v);
    // chutes: inside a fall's chute the face is laterally uniform and cut back a little
    float vEff = v, chute = 0.0, nearC = 0.0;
    float cK[2], cut[2];
    for (int k = 0; k < 2; k++) {
        float vc = wfChuteV(k, u);
        float hw = wfChuteHW(k, u);
        float c = 1.0 - smoothstep(hw, hw + 1.8, abs(v - vc));
        vEff = mix(vEff, vc, c);
        chute = max(chute, c);
        cK[k] = c;
        cut[k] = 1.0 - smoothstep(hw + 0.5, hw + 7.0, abs(v - vc));   // the chute cut eases out sideways (no side walls)
        // beside the chute the face carries no big protruding blocks: nothing hides the sheet's edges
        nearC = max(nearC, 1.0 - smoothstep(hw + 1.8, hw + 7.0, abs(v - vc)));
    }
    float fl = wfFaceLine(vEff);
    // blocky jointing: per-cell offsets of the face -> angular buttresses and recesses
    vec3 vb = voronoi2(vec2(vEff * 0.10, u * 0.14) + 11.0);
    float blk = (vb.z - 0.5) * 3.5 * (1.0 - 0.8 * nearC) * mix(0.25, 1.0, cl);
    float rough = 0.55 * gnoise(vec2(vEff * 0.55, u * 0.55) + 3.0) * (1.0 - 0.6 * nearC);
    float uf = u - fl - blk - rough - 0.5 * chute;
    // forested banks: an irregular foot line (+-2 m), not a ruled slope
    uf += 3.0 * gnoise(p * 0.06) * (1.0 - cl);
    // (review r2, fix 8) the cliff's foot: +-1.5 m in and out along the shore over the lowest ~2 m (coves and
    // spurs, not a straight shoreline); the face above is unchanged (a smooth remap of uf, monotonic)
    float footN = 1.5 * clamp(1.15 * gnoise(vec2(v * 0.21, 4.7)) + 0.5 * gnoise(vec2(v * 0.55, 9.3)), -1.0, 1.0);
    uf -= footN * cl * (1.0 - smoothstep(-1.5, 1.5, uf));
    float hs = wfStepH(vEff);
    float W = wfFaceWv(hs, vEff);
    float bt = wfButt(v);
    // fall B leaves a cleft half way up: lower step there, the cleft floor climbs back steeply inland
    float hwB = wfChuteHW(1, u);
    float clB = 1.0 - smoothstep(hwB + 0.5, hwB + 2.0, abs(v - uWFNotch.y));
    // apron of low shore rocks on the north bank, plunge pool in front of the falls
    float apron = wfApronZ(v);
    float wpool = smoothstep(-32.0, -12.0, v) * (1.0 - smoothstep(16.0, 30.0, v));
    float base = mix(-2.5, 0.25, apron);
    float hf;
    if (uf < 0.0) {
        float dp = -uf;
        float aw = wfApronW(v) + 1.5 * gnoise(p * 0.05);
        float pool = -2.2 - 5.5 * smoothstep(0.0, 9.0, dp);
        float shelf = 0.25 + 0.22 * gnoise(p * 0.31) + 0.12 * gnoise(p * 1.1)
                    - 1.0 * smoothstep(aw - 1.5, aw + 1.0, dp) - 1.2 * smoothstep(aw + 2.0, aw + 14.0, dp);
        // angular rock slabs on the apron, some standing in the shallows
        if (apron > 0.0 && dp < aw + 5.0) {
            vec4 sv;
            shelf += wfSlab(p, dp, aw, sv) * (1.0 - smoothstep(aw + 2.0, aw + 5.0, dp));
        }
        float tgt = mix(min(h, pool), shelf, apron);
        float wl = max(wpool * (1.0 - smoothstep(16.0, 34.0, dp)), apron * (1.0 - smoothstep(aw + 16.0, aw + 26.0, dp)));
        hf = mix(h, tgt, wl);
        // the forested banks keep their slope down under the water (a steep drop-off, no concave
        // foot: the forest bake keeps trees out of concave hollows)
        float bz = (1.0 - cl) * (1.0 - apron) * smoothstep(3.0, 9.0, hs);
        if (bz > 0.0) hf = mix(hf, min(hf, max(base - dp * (hs - base) / W, -26.0)), bz);
        // the buttress drops straight into deep water
        if (bt > 0.0) hf = mix(hf, min(hf, max(base - 3.0 * dp, -7.5)), bt * (1.0 - apron) * (1.0 - smoothstep(6.0, 12.0, dp)));
    } else {
        float hp = wfFaceProfile(p, u, v, vEff, uf, W, hs, base, cl, bt, cut[0], clB);
        // big rounded-angular blocks (~6 x 4 m, tilted bedding, wfPillow) stand out of the cliff /
        // buttress face by up to ~1 m, recessed joints between them: real ledges and overhang-like
        // steps, no masonry rows (two fixed-point passes find the block at this height)
        float bw = max(cl, bt) * (1.0 - 0.8 * chute) * (1.0 - 0.5 * nearC) * (1.0 - smoothstep(W - 1.5, W + 0.5, uf));
        if (bw > 0.0) {
            for (int it = 0; it < 2; it++) {
                vec4 B_, hB_;
                vec2 fc_ = wfFaceC(v, hp, B_, hB_);
                // + multi-scale roughness of the face (bulges and recesses 1-8 m)
                float rgh = 0.9 * gnoise(fc_ * vec2(0.12, 0.16)) + 0.45 * gnoise(fc_ * vec2(0.25, 0.33) + 3.0) + 0.22 * gnoise(fc_ * 0.6 + 7.0);
                float blkO = wfPillow(fc_ + hB_.xy * 7.0 + 31.0, 4.4, 1.5).x * mix(0.6, 1.0, smoothstep(0.30, 0.60, hB_.z));
                // bush / fern cushions where the face is covered (same mask as the shading's foliage)
                float cu = wfVegCover(fc_, hs) * wfCushion(fc_) * (1.0 - chute);
                hp = wfFaceProfile(p, u, v, vEff, uf + (blkO + rgh + cu) * bw, W, hs, base, cl, bt, cut[0], clB);
            }
        }
        // blend back into the natural hillside inland; never dig into higher natural ground
        hp = mix(hp, h, smoothstep(25.0, 110.0, uf - W));
        hf = max(h, hp);
    }
    // big angular blocks fallen to the foot of the face, in clusters of a few: at the buttress foot and
    // south of fall B, partly in the water (none in the falls' plunge zones, none alone in open water)
    float zoneW = smoothstep(-26.0, -18.0, v) * (1.0 - smoothstep(-6.5, 0.0, v))
                + smoothstep(13.0, 19.0, v) * (1.0 - smoothstep(25.0, 33.0, v));
    if (zoneW > 0.0 && uf > -7.0 && uf < 6.0) {
        vec2 bq = vec2(v, u - fl);
        vec4 c = voronoiV(bq * 0.24 + 5.0);
        float hsh = fract(c.w * 13.7);
        // accepted by the block's centre (a whole block or none: a footprint clipped by the zone edge
        // left single-texel needles standing in the water)
        vec2 bc = bq - c.xy / 0.24;
        float zoneB = smoothstep(-22.0, -18.0, bc.x) * (1.0 - smoothstep(-6.5, -3.5, bc.x))
                    + smoothstep(16.5, 19.0, bc.x) * (1.0 - smoothstep(25.0, 29.0, bc.x));
        float clus = smoothstep(-0.15, 0.2, gnoise(bc * 0.09 + 2.0));
        if (hsh < mix(0.25, 0.9, clus) * zoneB && bc.y > -3.0 && bc.y < 2.0) {
            vec4 hh = hash42i(ivec2(int(c.w * 65536.0), 41));
            vec2 o = rot2(hh.x * 3.14) * (c.xy / 0.24);              // offset from the block centre (m)
            vec2 R = vec2(1.5 + 1.0 * hh.y, 1.2 + 0.9 * hh.z);
            vec2 a = abs(o) / R;
            float e = max(max(a.x, a.y), (a.x + a.y) * 0.78);         // chamfered box
            float top = 0.7 + 2.0 * hh.w - 0.35 * max(-uf - 2.5, 0.0) + dot(o, hh.zy - 0.5) * 0.45;   // tilted top
            float bh = top * (1.0 - smoothstep(0.88, 1.0, e)) - 3.0 * smoothstep(1.0, 1.15, e);
            hf = max(hf, bh);
        }
    }
    return hf;
}

void main() {
    vec2 uv = gl_FragCoord.xy / uRes;
    vec2 p = uCenter + (uv * 2.0 - 1.0) * uHalf;
    float h = Hs(uv);
    float level = -1e4, foam = 0.0, flowAng = 0.0;

    // ---------------- waterfall rock step ----------------
    h = wfCarve(p, h);

    // ---------------- inflow stream ----------------
    // a boulder-choked mountain stream: step-pool staircase (flat pools, short white drops at
    // boulder lips) where it is gentle, one continuous white rapid where it is steep; boulders (1-2.4 m)
    // in clusters in the bed and on the banks; a low boulder delta where it meets the lake
    if (uStreamN > 1) {
        float cell = 2.0 * uHalf / uRes;
        // step-pool phase = integral of ds / Ls along the stream (continuous across the polyline's
        // vertices: a per-segment s / Ls jumped there and left 0.6-1 m water walls)
        float best = 1e9; vec4 a = uStream[0], b = uStream[1]; float bt = 0.0, phBest = 0.0, phAcc = 0.0, sBest = 0.0, sAcc = 0.0;
        int ib = 0;
        vec2 mouthP = vec2(1e9);
        for (int i = 0; i < 95; i++) {
            if (i + 1 >= uStreamN) break;
            vec4 s0 = uStream[i], s1 = uStream[i + 1];
            vec2 ab = s1.xy - s0.xy;
            float L = length(ab);
            float LsI = mix(11.0, 4.5, saturate(max(s1.z - s0.z, 0.0) / max(L, 1.0) * 3.0));
            float tt = clamp(dot(p - s0.xy, ab) / dot(ab, ab), 0.0, 1.0);
            float d = length(p - (s0.xy + ab * tt));
            if (d < best) { best = d; a = s0; b = s1; bt = tt; phBest = phAcc + tt * L / LsI; sBest = sAcc + tt * L; ib = i; }
            if (s0.z <= 0.0 && s1.z > 0.0) mouthP = mix(s0.xy, s1.xy, -s0.z / (s1.z - s0.z));
            phAcc += L / LsI;
            sAcc += L;
        }
        float bed = mix(a.z, b.z, bt);
        // bed slope, continuous along the stream (vertex slopes = mean of the two segments, linear between)
        float sl[3];
        for (int j = 0; j < 3; j++) {
            int i0 = clamp(ib - 1 + j, 0, uStreamN - 2);
            vec4 q0 = uStream[i0], q1 = uStream[i0 + 1];
            sl[j] = max(q1.z - q0.z, 0.0) / max(length(q1.xy - q0.xy), 1.0);
        }
        float slope = bt < 0.5 ? mix(0.5 * (sl[0] + sl[1]), sl[1], bt * 2.0) : mix(sl[1], 0.5 * (sl[1] + sl[2]), bt * 2.0 - 1.0);
        float hwP = mix(a.w, b.w, bt) * (1.0 + 0.25 * gnoise(p * 0.05));   // floodplain half width
        // STREAM_CHAN: wet channel = 0.5 x the floodplain (water ~4-5 m wide), 0.35 x on steep reaches
        // (the boulders choke it there: thin white threads between them)
        float hw = hwP * mix(0.52, 0.44, smoothstep(0.06, 0.15, slope));
        float hwB = hwP * 0.6;              // boulder field (falls.glsl streamBoulder uses the same)
        float d = best;
        float dMouth = length(p - mouthP);
        if (d < hwB * 4.5 || dMouth < 44.0) {
            // step pools: pool level flat, rising over a lip at the upstream end of each step; the lips
            // curve across the flow; steep reaches blend into one continuous rapid (no water walls)
            float Ls = mix(11.0, 4.5, saturate(slope * 3.0));
            float rapid = smoothstep(0.12, 0.25, slope);
            // (review rv4) + ragged lips on the steeper reaches: the phase shifts by +-0.3 step across the channel over ~2 m
            // (tongues of water between boulder lips) - the lips ran straight across the channel and the cascade read as
            // white terraces / stairs in film shot 4
            float sx = phBest + 0.35 * gnoise(vec2(sBest * 0.03, 1.7)) + 0.15 * gnoise(p * 0.08) + 0.18 * gnoise(p * 0.21 + 5.0)
                     + 0.45 * gnoise(p * 0.12) + 0.32 * gnoise(p * 0.45 + 2.0) * smoothstep(0.04, 0.12, slope);
            float fs = fract(sx);
            // (a wide lip ramp: no lip drops more than ~0.45 m per 1 m texel -> no glassy water walls)
            // (round 3) the steep reaches keep part of the steps: a cascade of drops over boulder lips with short
            // runs between them, not one smooth inclined slide (the white water piles up below each drop)
            // (review rv4: the rapids keep 40 % of the steps, was 60 %: more continuous tumbling water, fewer terraces)
            float lip = mix(smoothstep(0.45, 0.97, fs), mix(fs, smoothstep(0.30, 0.90, fs), 0.25), rapid);   // (rv7: 0.4 -> 0.25, a tumbling rapid, not stairs)
            float stepK = smoothstep(0.03, 0.10, slope) * smoothstep(0.2, 0.8, bed);
            float bedS = bed + slope * Ls * (lip - fs) * stepK;
            float r = d / hw;
            float prof = bedS + 0.8 * r * r + 0.25 * gnoise(p * 0.5) * (1.0 - smoothstep(0.0, 1.0, r));
            // boulders: ~4 m cells, R 1.0-2.4 m, 0.5-1.8 m tall, in clusters (~80 % of the cells inside a
            // cluster, ~10 % outside); only on the 1 m grid (falls.glsl streamBoulder: same cells)
            float boul = 0.0;
            if (cell < 2.0) {
                vec4 c = voronoiV(p * 0.25 + 13.0);
                float hsh = fract(c.w * 7.3);
                float R = 0.25 + 0.35 * fract(c.w * 3.1);
                float e = c.z / R;
                float clus = smoothstep(-0.12, 0.18, gnoise(p * 0.035 + 1.3));
                float present = step(1.0 - mix(0.10, 0.80, clus), hsh) * (1.0 - smoothstep(hwB * 2.8, hwB * 4.2, d));
                // (round 2) the wet channel is mostly open white water: few bed boulders in it
                present *= step(0.15 * (1.0 - smoothstep(0.55 * hw, 1.1 * hw, d)), fract(c.w * 11.3));   // (0.62 -> 0.38 -> round 3 0.15: a boulder-choked bed)
                // (1.1-1.8 m tall, steep sides: dark-sided blocks the water threads between)
                float bTop = (0.6 + 2.0 * R) * mix(0.8, 1.25, fract(c.w * 5.7)) * present;
                boul = bTop * pow(saturate(1.0 - e * e), 0.25);
                boul *= mix(0.6, 1.0, smoothstep(0.6 * hw, 1.3 * hw, d));
            }
            float wbl = smoothstep(hw * 1.0, hw * 2.4, d);
            float chanH = min(h, prof) + boul;
            float bankH = h + boul * 0.85;
            h = mix(chanH, bankH, wbl);
            float wl = bedS + 0.32;
            // banks contain the water (no sheet spilling over the low ground at the mouth)
            float lev = smoothstep(0.75 * hw, hw, d) * (1.0 - smoothstep(1.4 * hw, 2.0 * hw, d)) * smoothstep(0.0, 0.2, wl);
            // (round 2: a bank rising from the waterline - a flat shelf 0.25 m above the water read as
            // flat green tabs lying along the white water)
            h = max(h, mix(h, wl + 0.10 + 0.95 * saturate(d / hw - 0.75), lev));
            // a low boulder delta at the mouth: no shore humps hiding the cascade from the lake (the channel
            // ~0.4 m above the water, its grassy banks ~1.8 m: not a pebble beach)
            float capW = 1.0 - smoothstep(20.0, 42.0, dMouth);
            if (capW > 0.0) h = mix(h, min(h, max(wl, 0.0) + mix(0.4, 1.4, smoothstep(1.2 * hw, 3.0 * hw, d)) + boul), capW);
            // the level is written well beyond the water's edge (under the banks): its bilinear edge
            // never shows as 1 m texel steps. The true level only inside the channel; beyond 1.2 x the
            // half width it stays 0.4 m under the ground (no sheet spilling where the levee fades)
            if (d < hw * 1.9 && wl > -0.05) {
                level = d < 1.2 * hw ? wl : min(wl, h - 0.4);
                vec2 dir = normalize(a.xy - b.xy);        // downstream (points ordered from the lake upstream)
                float shallow = 1.0 - smoothstep(0.05, 0.45, wl - h);
                // white over the drop, a foamy tail decaying ~3.5 m into the pool below it
                float drop = smoothstep(0.48, 0.66, fs) * (1.0 - smoothstep(0.97, 1.0, fs));
                float tail = 0.55 * exp(-max(0.52 - fs, 0.0) * Ls / 3.5);
                foam = saturate(0.02 + stepK * (drop + tail) + slope * 0.6 * stepK + 0.45 * shallow * smoothstep(0.04, 0.12, slope)
                                + 0.08 * smoothstep(0.5, 1.1, r));
                // steep reaches: a white rapid, its core white, darker faster water along the edges
                foam = max(foam, 0.95 * smoothstep(0.08, 0.14, slope) * (1.0 - 0.35 * smoothstep(0.5, 0.95, r)));
                // fades out where the stream meets the lake (no foam / turbulence on open water)
                foam *= smoothstep(-0.05, 0.25, wl);
                // (round 2) the foam map doubles as the wet-channel mask (> 0.01 only inside the water, 1.2 x
                // the half width): rocks.glsl keeps its SDF boulders out of the channel with it
                foam = d < 1.2 * hw ? max(foam, 0.03) : 0.0;
                flowAng = atan(dir.y, dir.x);
            }
        }
    }
    fragColor = vec4(h, level, foam, flowAng);
}
