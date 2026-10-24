// ---------------------------------------------------------------------------
// terrain_shade.glsl : terrain materials (rock, scree, vegetation, snow, shore pebbles) and shading
// ---------------------------------------------------------------------------
// packed granite cobbles of mixed sizes; returns albedo and a bump normal offset
vec3 pebbles(vec2 p, out vec3 bump) {
    // shore of packed, angular-to-subrounded granite gravel (shingle of pale 1-10 cm chips with near-black
    // gaps): a dense layer of ~9 cm pebbles (60-80 % filled), ~4 cm pebbles in its gaps, ~2 cm pebbles and grit in
    // theirs, short cast shadows on the side away from the sun, and sparse 12-25 cm angular cobbles on top (beyond
    // ~14 m only: nearer, the 3D cobbles of rocks.glsl carry that size). Every pebble its own ellipse inside its
    // power-diagram cell (no shared-edge paving); each layer has its own rotated, warped grid (no tiling).
    // Under water an olive-brown algae film lowers the contrast (shallows #6e644a). Footprint filtered
    // toward the beach's mean (the film moves); the footprint grows at grazing angles.
    bump = vec3(0.0);
    float hT = terrainH(p, false);
    float dist = length(vec3(p.x, hT, p.y) - uCamPos);
    float cosT = clamp((uCamPos.y - hT) / max(dist, 1e-3), 0.15, 1.0);
    float fp = dist * gPixAng / pow(cosT, 0.35);      // pixel footprint on the ground (m) (rocks5: / cos^0.5 -> ^0.35: the grazing shore strip blurred to sand)
    // means of the fully resolved layers, gaps and shadows included (tools/rkpebmean.py, CPU replica)
    // (rocks3: re-measured for the finer layers; the composite is scaled x1.1 at the end -> beach mean ~0.2 as before)
    // (rocks4 R5: x0.92 - the smaller chips / longer cast shadows / chip facets darken the resolved layers; at x1.0 the
    // unresolved beach beyond ~8 m read as pale sand next to the near shingle)
    const vec3 AVG = vec3(0.256, 0.2348, 0.2145) * 0.92;          // (rocks3b: sparser chips, lighter grit; 3c: denser 2.5 cm layer)
    const vec3 AVGM = vec3(0.2422, 0.2223, 0.2027) * 0.94;          // the 2.5 cm layer over its gaps
    const vec3 AVGF = vec3(0.2071, 0.1903, 0.1724);                 // the 1.2 cm layer over grit
    // (rocks3) finer gravel (1-4 cm chips, a sparser 5 cm layer; the 9 cm plates read as crazy paving)
    const float SC = 3.4, SD = 24.0, SM = 52.0, SF = 100.0;   // (rocks3d: 19 / 40 / 80 -> 24 / 52 / 100: chips 1-4 cm at 3-6 m)
    float wl = waterLevel(p);
    float sub = wl > -1000.0 ? 1.0 - smoothstep(-0.12, -0.03, hT - wl) : 0.0;
    float zone = gnoise(p * 0.23 + 3.0) + 0.5 * gnoise(p * 0.9 + 1.3);   // sorting: coarse vs fine patches
    vec3 tint = vec3(1.0) + 0.05 * vec3(gnoise(p * 0.5 + 7.0), 0.0, -gnoise(p * 0.5 + 7.0));
    // sub-pixel beach: still mottled (25 cm and 8 cm tone patches of coarser / finer, paler / darker gravel, +-25 %)
    float fm = texture(uNoise3, vec3(p * 0.25, 0.71)).g - 0.487;
    if (fp < 0.05) fm += 0.6 * (texture(uNoise3, vec3(p * 0.75, 0.29)).g - 0.487) * (1.0 - smoothstep(0.025, 0.05, fp));
    vec3 far = AVG * (1.0 + 0.1 * zone) * (1.0 + 0.9 * fm);
    far *= 1.0 - 0.18 * sub;                      // (rocks3c) the bed is darker than the dry shingle
    // (rocks5 P2) the unresolved gravel keeps a crisp pixel-scale stone mosaic (rkGravelTex) instead of a smooth sand tone
    vec3 farT = far * rkGravelTex(vec3(p.x, hT, p.y), fp * (1.0 + 1.5 * sub));
    vec3 subT = vec3(0.78, 0.73, 0.47);           // (rocks3b: a little more olive, #6e644a)
    float visC = 1.0 - smoothstep(0.35, 0.8, fp * SC);
    float kB = mix(2.05, 1.15, sub);      // (rocks3d: 1.75 -> 1.95, r1 beach median 157 vs target 184 with the narrower stone tones; rocks4 R5: 2.2)
         // (rocks3c: dry 1.15 -> 1.75: r1 beach median sRGB 133 vs target 184, beach #b9ab9f ~ as bright as lit boulder tops)
    if (visC <= 0.0) return far * tint * kB * mix(vec3(1.0), subT, sub);   // (rocks3b: x1.15 on the new means = +33 %, the target beach #b9ab9f is paler)
    vec2 sunXZ = normalize(uSunDir.xz + vec2(1e-5));
    float sunT = length(uSunDir.xz) / max(uSunDir.y, 0.15);
    // slow domain warp shared by the layers (~1/4 of a small cell)
    vec2 wp = (texture(uNoise3, vec3(p * 0.35, 0.53)).rg - vec2(0.475, 0.487)) * 0.03;
    // sparse angular cobbles on top (polygonal, rounded corners, tilted); none within ~14 m of the camera
    const mat2 RC = mat2(0.8196, 0.5729, -0.5729, 0.8196);      // rot2(0.61)
    vec2 tC, bC = vec2(0.0); float gC = 1.0;
    vec3 C = vec3(0.0, 0.0, 1.0);
    // (rocks3c) under water: elliptical, rounded stones of a broad size mix (rocks3b: one size of rounded cells = an even
    // pavement of pale pillows / stepping-stone tiles in r10 / r1 / r4)
    // (rocks4 R8) per-stone elongation under water (el < 0: random 0.1-0.5) and sizes 0.4-1.4 (below)
    vec3 vC = rkPack(RC * p * SC + 0.37, 0.3, mix(0.25, 0.7, sub), -0.33 * sub, tC);
    if (sub > 0.0) { float rC = length(tC); vC.x = mix(vC.x, min(vC.x, (rC + vC.x) * mix(0.4, 1.4, fract(vC.z * 17.9)) - rC), sub); }
    // (rocks3b) under water a pavement of rounded cobbles at every distance (10-30 cm cobbles and boulders,
    // #6e644a - not sand / grit), only to ~3-6 m depth (deeper the water shader turns the bed to silt anyway - and every
    // lake pixel pays for it)
    float zc = saturate(zone + 0.5);
    float subP = sub * (1.0 - smoothstep(3.0, 6.0, wl - hT));
    // (rocks3c) in PATCHES (2-6 m): dense cobble fields, boulder clusters and finer gravel between them;
    // zB ~ N(0,1) patch fields
    float zB1 = 0.0, zB2 = 0.0;
    if (subP > 0.0) {
        zB1 = (texture(uNoise3, vec3(p * 0.045, 0.83)).g - 0.487) * 4.0 + (texture(uNoise3, vec3(p * 0.17, 0.17)).g - 0.487) * 2.5;
        zB2 = (texture(uNoise3, vec3(p * 0.06 + 0.31, 0.47)).g - 0.487) * 4.0 + (texture(uNoise3, vec3(p * 0.2 + 0.7, 0.61)).g - 0.487) * 2.5;
    }
    // (rocks3c) dry beach: the sparse 12-22 cm cobbles near the camera too (~5-10 stones > 8 cm per m2; the 3D
    // cobbles of rocks.glsl reach ~1 / m2)
    // (rocks4 R8: under water 0.5 + 0.75 zB1 -> 0.3 + 0.6 zB1, more fine gravel; R5: dry 0.03-0.15 from 12 m -> 0.1-0.32 from
    // 7 m - the beach beyond the resolved 4 cm layer read as smooth sand, cobbles belong there)
    // (rocks5 B1) bed: 0.3 -> 0.55 + 0.6 zB1 (shallows: packed 5-30 cm cobbles, not plates)
    float pC = mix(mix(0.1, 0.32, zc) * smoothstep(7.0, 10.0, dist), saturate(0.55 + 0.6 * zB1), subP);
    if (fract(vC.z * 11.3) < pC) {
        float gc = fract(vC.z * 4.1);
        C = rkPackShade(vC, tC, mix(mix(0.12, 0.3, gc), mix(0.04, 0.2, gc), subP), step(mix(0.8, 0.55, subP), fract(vC.z * 3.7)), fp * SC, RC * sunXZ, sunT, 1.0, bC, gC);
        // (rocks3c) under water softer self-shading / cast shadows (refracted, scattered light): the identical dark crescent
        // on every cobble embossed the bed like buttons
        C.z = mix(C.z, 1.0, 0.3 * subP); gC = mix(gC, 1.0, 0.35 * subP);   // (rocks3d: ring 0.35 -> 0.5; rocks5 B2: 0.3 / 0.35)
        bC = bC * RC;                                          // back to world xz (inverse rotation)
    }
    // (rocks3b) under water: boulders among the cobbles; (rocks3c) 0.5-1.4 m (1 m cells, broad weights), clustered
    vec3 B = vec3(0.0, 0.0, 1.0), vB = vec3(0.0);
    vec2 bB = vec2(0.0); float gB = 1.0;
    if (subP > 0.0) {
        const float SB = 0.72;                                  // (rocks3d: 1 -> 1.4 m cells, boulders 0.5-1.5 m)
        const mat2 RB = mat2(-0.8569, 0.5155, -0.5155, -0.8569);   // rot2(2.6)
        vec2 tB;
        vB = rkPack(RB * p * SB + 2.9, 0.15, 0.7, 0.25, tB);       // (rocks3d: kr 0.3 / el 0.4 -> 0.15 / 0.25: subangular blocks)
        if (fract(vB.z * 9.7) < saturate(0.4 + 0.5 * zB2) * subP) {   // (rocks5 B1: 0.8 -> 0.4)          // (rocks3d: 0.35 -> 0.8: near bed = mostly 0.5-1.5 m blocks; k8 sweep 0.65 / 0.9)
            B = rkPackShade(vB, tB, mix(0.05, 0.18, fract(vB.z * 4.1)), step(0.6, fract(vB.z * 2.9)), fp * SB, RB * sunXZ, sunT, 1.0, bB, gB);
            B.z = mix(B.z, 1.0, 0.15);                                // (rocks3d: 0.35 / gB lift 0.2 dropped: boulders keep their 3D shading and dark gaps)
            bB = bB * RB;
        }
    }
    // the packed layers
    // seen through the rippled water the small layers blur earlier (refraction distortion, absorption)
    float fpS = fp * (1.0 + 1.5 * sub);
    vec3 cD = farT;
    vec2 bD = vec2(0.0);
    float visD = 1.0 - smoothstep(0.4, 0.9, fpS * SD);       // (rocks4 R5: 0.3-0.7 -> 0.4-0.9, chips down to ~1.1 px: no "sand" at 8-15 m)
    if (visD > 0.0 && C.x < 0.999) {
        // ~9 cm pebbles, 60-80 % of the ground (fewer in fine patches)
        const mat2 RD = mat2(0.1987, 0.9801, -0.9801, 0.1987);  // rot2(1.37)
        vec2 tD, bmpD; float gD;
        vec2 qD = RD * (p - 1.7 * wp) * SD + 5.1;
        vec3 vD = rkPack(qD, 0.07, 0.6, 0.06, tD);           // (rocks3b: angular chips, kr 0.12 / el 0.15 = ovals)
        { float rD = length(tD); vD.x = min(vD.x, (rD + vD.x) * mix(0.45, 1.3, fract(vD.z * 17.9)) - rD); }   // (rocks4 R5) chip sizes x0.45-1.3
        // (i8: stronger sorting - coarse patches 15 % holes, fine ones 85 %)
        // (rocks3b: the shore is a matrix of 0.5-2 cm grit with SPARSE 3-8 cm flat chips - 15-85 % holes packed the 5 cm
        // layer into an egg paving; now 50-92 % holes; j3 sweep: 35-85 % read as paving again)
        // (rocks5 P3) denser 4 cm stones (holes x0.5): the shingle is packed stones, not grit with sparse chips
        float holeD = step(fract(vD.z * 7.7), mix(mix(0.5, 0.92, saturate(0.5 - zone)) * 0.5, 0.85, sub));   // (under water: mostly fine gravel)
        float shD = fract(vD.z * 5.9);
        vec3 D = rkPackShade(vD, tD, mix(0.02, 0.12, shD), 0.2 + 0.7 * fract(vD.z * 3.3), fp * SD, RD * sunXZ, sunT, 0.0, bmpD, gD);
        bmpD = bmpD * RD;
        D.x *= 1.0 - holeD;
        // ~4 cm pebbles in the gaps / holes of the 9 cm layer
        // (rocks3c) the unresolved 2.5 / 1.2 cm layers are not a flat mean (= "asphalt" between the pebbles at 3-6 m, r1):
        // a mean-preserving mottle at 6.6 / 3.3 cm until it is ~3 px
        vec3 cM = AVGM * (1.0 + 1.3 * (texture(uNoise3, vec3(p * 2.0, 0.59)).a - 0.5207) * (1.0 - smoothstep(0.012, 0.03, fp)));
        vec2 bM = vec2(0.0);
        float visM = 1.0 - smoothstep(0.3, 0.7, fpS * SM);
        if (visM > 0.0 && D.x < 0.999) {
            const mat2 RM = mat2(-0.4425, 0.8968, -0.8968, -0.4425);   // rot2(2.03)
            vec2 tM, bmp; float gS;
            vec2 qM = RM * (p + wp) * SM + 3.7;
            vec3 vM = rkPack(qM, 0.08, 0.45, 0.06, tM);
            { float rM = length(tM); vM.x = min(vM.x, (rM + vM.x) * mix(0.45, 1.3, fract(vM.z * 17.9)) - rM); }   // (rocks4 R5)
            float hole = step(fract(vM.z * 7.7), mix(0.15, 0.4, saturate(0.5 - zone)));   // (rocks3b: was 8-32 %, 3b 30-60 %; 3c 15-40 %)
            float sh = fract(vM.z * 5.9);
            // gap width varies along each pebble's outline (pebbles touch here, leave a gap there)
            float gvM = 0.25 + 1.5 * texture(uNoise3, vec3(qM * 0.12, 0.13)).g;
            vec3 M = rkPackShade(vM, tM, mix(0.02, 0.18, sh * sqrt(sh)) * gvM, 0.2 + 0.7 * fract(vM.z * 3.3), fp * SM, RM * sunXZ, sunT, 0.0, bmp, gS);
            bmp = bmp * RM;
            M.x *= 1.0 - hole;
            // small pebbles + grit in the gaps and holes (shadowed by the pebble beside them)
            float gA4 = texture(uNoise3, vec3(p * 4.0, 0.37)).a;
            vec3 cF = AVGF * (1.0 + 1.4 * (gA4 - 0.5207) * (1.0 - smoothstep(0.006, 0.014, fp)));
            vec2 bF = vec2(0.0);
            float visF = 1.0 - smoothstep(0.24, 0.5, fpS * SF);
            if (M.x < 0.999 && visF > 0.0) {
                const mat2 RF = mat2(0.3902, 0.9208, -0.9208, 0.3902);   // rot2(1.17)
                vec2 tF; float gF;
                vec2 qF = RF * (p - wp) * SF + 1.3;
                vec3 vF = rkPack(qF, 0.15, 0.4, 0.05, tF);
                float sf = fract(vF.z * 5.9);
                float gvF = 0.25 + 1.5 * texture(uNoise3, vec3(qF * 0.12, 0.61)).g;
                vec3 F = rkPackShade(vF, tF, mix(0.04, 0.25, sf) * gvF, 0.1 + 0.6 * fract(vF.z * 3.3), fp * SF, RF * sunXZ, sunT, 0.0, bF, gF);
                bF = bF * RF;
                vec3 grit = vec3(0.15, 0.139, 0.122) * (0.6 + 0.8 * gA4);   // (rocks3b: 0.1 = black gaps)
                vec3 st = rkCobCol(vF.z) * mix(0.8, 1.0, F.y) * 0.9 * F.z;
                cF = mix(cF, mix(grit * gF, st, F.x), visF);
                bF *= visF;
            }
            cF *= mix(gS, 0.92, hole);                         // crevice / cast shadow of the pebble beside it (rocks3b: hole 0.6 -> 0.92, many holes now)
            vec3 st = rkCobCol(vM.z) * mix(0.85, 1.0, M.y) * M.z * 1.04;
            cM = mix(cM, mix(cF, st, M.x), visM);
            bM = mix(bF, bmp, M.x) * visM;
        }
        cM *= mix(gD, 0.95, holeD);                            // crevice / cast shadow of the 5 cm pebble beside it (rocks3b: 0.75 -> 0.95)
        vec3 stD = rkCobCol(vD.z) * mix(0.85, 1.0, D.y) * D.z * 1.04;
        cD = mix(farT, mix(cM, stD, D.x), visD);
        bD = mix(bM, bmpD, D.x) * visD;
    }
    cD *= gC * mix(1.0, 0.7, subP);                            // the cobbles shade their surroundings (under water: shaded, algae-filmed gaps)
    // (rocks3c) bed stones: rkBedCol (darker, olive / brown / dark stones mixed in) instead of the dry shingle colours
    float kBed = 1.0;
    vec3 stC = mix(min(rkCobCol(vC.z), vec3(0.5)), rkBedCol(vC.z) * kBed, subP) * mix(0.85, 1.0, C.y) * C.z;
    vec3 c = mix(cD, stC, C.x);
    vec3 stB = rkBedCol(vB.z) * kBed * mix(0.85, 1.0, B.y) * B.z;
    // (rocks5 P4) the cobbles / bed boulders are textured stones (5 / 2 cm mottle + dark specks, +-30 %), not smooth soap pillows
    // (r1 / r10 shallows); faded by the footprint (through the water it blurs earlier)
    { float v5 = 1.0 - smoothstep(0.022, 0.04, fpS), v2 = 1.0 - smoothstep(0.009, 0.016, fpS);
      if (v5 > 0.0 && max(C.x, B.x) > 0.0) {
          vec4 s5 = texture(uNoise3, vec3(p * 1.25, fract(vC.z * 3.1 + vB.z)));
          float sm = (s5.g - 0.487) * 2.0 * v5 + (v2 > 0.0 ? (texture(uNoise3, vec3(p * 3.1, 0.37)).g - 0.487) * 1.6 * v2 : 0.0);
          float spk = v2 * smoothstep(0.66, 0.74, s5.a);
          float kS = 1.0;
          vec3 mt = vec3(1.0 + kS * sm) * (1.0 - 0.45 * kS * spk);
          stC *= mt; stB *= mt;
      } }
    c = mix(c * gB, stB, B.x);
    // granite grain / dark lichen crust on the stones (only where resolvable)
    float gv = 1.0 - smoothstep(0.004, 0.02, fp);
    if (gv > 0.0) {
        float gr = texture(uNoise3, vec3(p * 2.3, vC.z * 0.7 + 0.1)).b;
        float dk = smoothstep(0.66, 0.74, texture(uNoise3, vec3(p * 3.1, 0.77)).a);
        c *= 1.0 + gv * (0.24 * (gr - 0.5) - 0.35 * dk * C.x);
        // dark mineral / lichen specks on every stone (2-6 mm)
        float gs = 1.0 - smoothstep(0.0015, 0.005, fp);
        if (gs > 0.0) c *= 1.0 - 0.4 * gs * smoothstep(0.68, 0.76, texture(uNoise3, vec3(p * 9.0, 0.53)).a);
    }
    // (rocks4 R5) dry shingle: more tonal contrast around the beach mean (target p10 / p50 / p90 55 / 174 / 222 vs 101 / 188 / 223:
    // darker gaps / shaded chip facets, a little paler lit chips; the mean drops slightly; sweep 0.45 / 0.8); none under water
    { float kCt = 1.4 * (1.0 - sub);          // (rocks5: 0.8 -> 1.4, target shingle p10 55 / hp 35-46)
      float rL = max(dot(c, vec3(0.3333)) / max(dot(far, vec3(0.3333)), 1e-4), 1e-3);
      c *= pow(rL, rL < 1.0 ? kCt : 0.35 * kCt); }
    c = mix(farT, c, visC) * tint * kB;
    // under water: algae film, lower stone / gap contrast
    c = mix(c, mix(c, far * 1.15, 0.2) * subT, sub);           // (rocks3b: 0.3 -> 0.2, the cobbles keep their contrast)
    vec2 bm = mix(bC + bD * (1.0 - C.x), bB, B.x) * visC;
    bump = vec3(bm.x, 0.0, bm.y) * 0.75 * (1.0 - 0.45 * sub);    // (rocks3c: 0.3 -> 0.45, flatter lit domes under water)
    return c;
}

// (session 6) world2 look-dev knobs: 0 = the shipped values; the lab's wt/kv.py sweeps them inside one
// process (a shader edit costs a 15-20 min compile on this iGPU). Unset in the other passes -> 0.
uniform vec4 uW2K[8];
float W2K(int i) { return uW2K[i >> 2][i & 3]; }
float gRkWS = 0.0, gRkDS = 0.0;             // (world3) joint-block relief weight / depth at the shaded point (shadeTerrain)
float gVegS = 0.0;                          // (world3) turf cover over the slab relief at the shaded point (terrainAlbedo -> shadeTerrain)

// (world3) sun shadows cast by the joint-block relief (world.glsl rockRelief): the baked shadow maps only
// know the base heightfield. March toward the sun until the ray is above the base surface (the relief only
// lowers the ground, so nothing above it can block); crisp, softened to the pixel footprint
float rockReliefShadow(vec3 p, float fp) {
    float res = 1.0;
    float t = max(0.25, 1.5 * fp);
    // (t20, perf: 16 -> 10 steps, no visible change in the 1:1 crops; t29: 8)
    for (int i = 0; i < 8; i++) {
        vec3 q = p + uSunDir * t;
        float hb = terrainH(q.xz, true);
        if (q.y > hb + 0.05) break;
        float dy = q.y - (hb + rockRelief(q.xz, hb));
        res = min(res, dy / max(0.012 * t, 0.8 * fp));
        if (res < -1.0) break;
        t += clamp(abs(dy) * 0.7, max(0.25, 0.7 * fp), 10.0);
        if (t > 160.0) break;
    }
    return smoothstep(-1.0, 1.0, res);
}

// fbm2 whose octaves fade out once they get finer than a few pixels: the macro masks below must not
// speckle or shimmer at 1-5 km (wl = wavelength of the first octave in metres)
float fbm2F(vec2 p, int oct, float wl, float fp) {
    float s = 0.0, a = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < oct; i++) {
        float v = smoothstep(1.6 * fp, 3.6 * fp, wl);
        if (v <= 0.0) break;
        s += a * v * gnoise(p); p = m * p; a *= 0.5; wl *= 0.5;
    }
    return s;
}

// Face coordinates for the mountain faces: of four vertical projection planes (every 45 deg of
// aspect) the two closest to the coarse aspect, blended. u runs along the plane (horizontally), v is
// the height, so v-elongated patterns hang down the fall line within ~15 deg for every aspect (two
// axis-aligned planes gave a diagonal hatch on the NE/NW faces) and nothing swirls with the normal.
vec2 gFa, gFb;            // (u, v) in plane A / B
vec3 gFTa, gFTb;          // horizontal unit axis of plane A / B (world)
float gFwa, gFwb;         // blend weights (sum 1)
void faceFrames(vec3 p, vec3 nC) {
    float a = atan(nC.z, nC.x) / (0.25 * PI);
    float k0 = floor(a), fr = a - k0;
    gFwb = smoothstep(0.3, 0.7, fr); gFwa = 1.0 - gFwb;
    float aa = k0 * 0.25 * PI, ab = aa + 0.25 * PI;
    gFTa = vec3(-sin(aa), 0.0, cos(aa)); gFTb = vec3(-sin(ab), 0.0, cos(ab));
    gFa = vec2(dot(p.xz, gFTa.xz), p.y); gFb = vec2(dot(p.xz, gFTb.xz), p.y);
}
// streak noise in face coordinates (fq: frequency along the face, down the face)
float faceNoise(vec2 fq, float off) {
    float a = 0.0;
    if (gFwa > 0.02) a += gFwa * gnoise(gFa * fq + off);
    if (gFwb > 0.02) a += gFwb * gnoise(gFb * fq + off + 17.3);
    return a;
}
// fbm in the same face coordinates, octaves faded by the footprint like fbm2F (wl: first octave, m)
float faceFbm(vec2 fq, int oct, float wl, float fp, float off) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < oct; i++) {
        float v = smoothstep(1.6 * fp, 3.6 * fp, wl);
        if (v <= 0.0) break;
        s += a * v * faceNoise(fq, off);
        fq *= 2.03; a *= 0.5; wl *= 0.5; off += 3.7;
    }
    return s;
}
// relief of the faces below the heightfield resolution: V-shaped gullies between rounded ribs, long
// down the face; octaves faded by the footprint; returns the world-space height gradient (m/m)
// (cav: the creases as dark crack / cleft lines, visible in flat frontal light too)
vec3 faceBump(float fp, out float cav) {
    vec3 g = vec3(0.0);
    cav = 0.0;
    // (session 4: 0.026 / 0.009 -> 0.022 / 0.014: the tall vertical ribs made the faces look like dripping
    // wax at 1720+ px)
    vec2 fq = vec2(0.022, 0.014);                             // ribs ~1.6x longer down the face than wide
    float a = 13.0, wl = 38.0, wc = 0.5;
    for (int i = 0; i < 4; i++) {
        float v = smoothstep(2.4 * fp, 5.0 * fp, wl);   // (session 6: 1.6..3.6 px shimmered)
        if (v <= 0.0) break;
        if (gFwa > 0.02) {
            vec3 nd = gnoised(gFa * fq + 4.1 + float(i));
            g += (gFwa * v * a * -sign(nd.x) * nd.y * fq.x) * gFTa + vec3(0.0, gFwa * v * a * -sign(nd.x) * nd.z * fq.y, 0.0);
            cav += gFwa * v * wc * (1.0 - smoothstep(0.0, 0.09, abs(nd.x)));
        }
        if (gFwb > 0.02) {
            vec3 nd = gnoised(gFb * fq + 9.3 + float(i));
            g += (gFwb * v * a * -sign(nd.x) * nd.y * fq.x) * gFTb + vec3(0.0, gFwb * v * a * -sign(nd.x) * nd.z * fq.y, 0.0);
            cav += gFwb * v * wc * (1.0 - smoothstep(0.0, 0.09, abs(nd.x)));
        }
        fq *= 2.1; a *= 0.5; wl *= 0.48; wc = (i == 0) ? 0.9 : wc * 0.8;
    }
    return g;
}

// (session 6) fall-line flutes of the granite faces (the walls are scored by narrow
// V-gullies with sharp, shadowed bottoms between broader ribs, broken by ledges along their length).
// Height ~ +|n| of an anisotropic (3:1 down the face) gradient noise: the crease at n = 0 is the gully
// floor. 3 octaves (28 / 12.6 / 5.7 m), each faded before it is < ~2.5-5 px; returns the height gradient
vec3 faceFlutes(float fp) {
    vec3 g = vec3(0.0);
    float wl = 28.0 * (1.0 + W2K(21)), a = 1.0;
    for (int i = 0; i < 3; i++) {
        float v = smoothstep(2.4 * fp, 5.0 * fp, wl);
        if (v <= 0.0) break;
        vec2 fq = vec2(1.0 / wl, 0.33 / wl);
        for (int k = 0; k < 2; k++) {
            float w = k == 0 ? gFwa : gFwb;
            if (w < 0.04) continue;
            vec2 F = k == 0 ? gFa : gFb;
            vec3 T = k == 0 ? gFTa : gFTb;
            vec3 nd = gnoised(F * fq + vec2(5.3 + 3.1 * float(i), 1.7 + 2.3 * float(k)));
            // flutes come and go along the face (ledges, blocky sections)
            float brk = smoothstep(-0.30, 0.20, gnoise(F * fq * vec2(0.45, 1.6) + vec2(11.0 + float(i), 3.0 * float(k))));
            float s = sign(nd.x) * a * 0.16 * wl * brk * w * v;
            g += s * (nd.y * fq.x * T + vec3(0.0, nd.z * fq.y, 0.0));
        }
        wl *= 0.45; a *= 0.85;
    }
    return g;
}

// voronoi2 (same cells) + bn: the unit vector from the nearest to the second-nearest feature point, i.e.
// the normal of the cell border between them (used to keep only the near-vertical / near-horizontal
// borders as cracks: joint sets, not a polygon network)
vec3 voronoi2n(vec2 p, out vec2 bn, out float id2) {
    vec2 i = floor(p); vec2 f = p - i; ivec2 c = ivec2(i);
    float d1 = 8.0, d2 = 8.0, id = 0.0; vec2 q1 = vec2(0.0), q2 = vec2(1.0, 0.0);
    id2 = 0.0;
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        ivec2 o = ivec2(x, y);
        vec2 r = vec2(o) + hash22i(c + o) * 0.85 + 0.075 - f;
        float d = dot(r, r);
        float hid = hash12i(c + o + ivec2(17, 31));
        if (d < d1) { d2 = d1; q2 = q1; id2 = id; d1 = d; q1 = r; id = hid; }
        else if (d < d2) { d2 = d; q2 = r; id2 = hid; }
    }
    bn = normalize(q2 - q1 + vec2(1e-6, 0.0));
    return vec3(sqrt(d1), sqrt(d2), id);
}

// fractured granite below the heightfield resolution (at 0.3-5 km): planar facets with
// sharp edges, voronoi cells in face coordinates (taller than wide: vertical ribs and joints), each
// tilted at random so the sun picks them out light / dark; their borders are dark cracks. Two scales,
// each faded out before its cells get smaller than ~5 px. Returns the world-space normal offset.
vec3 faceFacets(float fp, out float crack, out float ftone) {
    vec3 g = vec3(0.0);
    crack = 0.0; ftone = 0.0;
    // (session 6: 30 m / tilt 3.2 (up to +-58 deg) read as crumpled foil at 0.5-3 km -> 55 m, 1.7)
    // (session 3: tilt 1.7 -> 1.2, crack 0.85 -> 0.6, cell tone 0.12 -> 0.05: at 3-5 km the cells read as
    // crazy paving - a mosaic of polygon tones; the faces should have smooth tone with crisp dark lines)
    // (session 4) blockier: cells 1.15x (was 1.8x) taller than wide, vertical tilt 0.35 -> 0.9 (lit ledge
    // tops, shaded risers), crisper cell edges (blend 0.12 -> 0.05 of a cell, never under ~1.5 px), and a
    // third octave (~7.5 m) for the 3440 px final, where the 4 m heightfield is soft
    // (session 6) the random cell tilts are OFF (amp 0, knob 0 brings them back): with grey albedo (debug 28)
    // they were the camouflage of the faces - crisp dark polygons (t19-t21) or, with soft borders, soft dark
    // blobs (t22b); the faces read as rock once the shading comes from the organised fall-line relief
    // (faceFlutes, faceBump, the heightfield). The joint-set cracks on the cell borders stay.
    float S = 55.0, amp = 1.5 * W2K(0), cd = 0.7;   // (session 5: 1.2 / 0.6 -> 1.5 / 0.7: rock 8-px contrast 0.047 vs 0.06-0.085)
    for (int i = 0; i < 3; i++) {
        float v = smoothstep(3.0 * fp, 7.0 * fp, S);          // (session 6: was 2..5 px, shimmered at 2-4 km)
        if (v <= 0.0) break;
        float eb = max(0.05, 1.5 * fp / S);
        for (int k = 0; k < 2; k++) {
            float w = k == 0 ? gFwa : gFwb;
            if (w < 0.04) continue;
            vec2 uv = (k == 0 ? gFa : gFb) / vec2(S, 1.15 * S) + vec2(float(i) * 7.3 + float(k) * 3.1, 0.0);
            vec2 bn; float id2;
            vec3 vr = voronoi2n(uv, bn, id2);
            float edge = vr.y - vr.x;
            vec2 tl = vec2(fract(vr.z * 13.7), fract(vr.z * 29.3)) - 0.5;
            vec2 tl2 = vec2(fract(id2 * 13.7), fract(id2 * 29.3)) - 0.5;
            vec3 T = k == 0 ? gFTa : gFTb;
            float ww = w * v;
            // (session 4) only the near-vertical (joints) and near-horizontal (sheeting) borders are cracks:
            // the full voronoi network read as a dark snakeskin at the final 3440 px
            float jset = max(smoothstep(0.74, 0.95, abs(bn.x)), 0.7 * smoothstep(0.80, 0.97, abs(bn.y)));
            // (session 6, knob 1) crisp creases only on the joint-set borders; the oblique borders blend the
            // two cells' tilts over a third of a cell (a crisp crease on every border = polygon mosaic, t19
            // shot 7); the border takes the mean tilt, so the normal stays continuous
            float ebk = mix(eb, max(eb, 0.35), W2K(1) * (1.0 - jset));
            g += ww * amp * mix(W2K(1) * 0.5 * (tl + tl2), tl, smoothstep(0.0, ebk, edge)).x * T
               + ww * amp * vec3(0.0, 0.9 * mix(W2K(1) * 0.5 * (tl.y + tl2.y), tl.y, smoothstep(0.0, ebk, edge)), 0.0);
            float cw = clamp(1.2 * fp / S, 0.03, 0.25);
            // only part of the cell borders are open cracks (a full network reads as crazy paving)
            float open = smoothstep(-0.05, 0.25, gnoise(uv * 2.3 + vec2(1.7, 4.1)));
            crack += ww * cd * open * jset * (1.0 - smoothstep(0.0, cw, edge));
            ftone += ww * (fract(vr.z * 7.1) - 0.5);
        }
        S *= 0.37; amp *= 0.6; cd *= 0.7;   // (t20: amp decay 0.75 -> 0.6: the 20 m cells read as crazy paving at 1720 px)
    }
    crack = saturate(crack);
    return g;
}

// (round 3) glacier-polished slabs / knobs where rock breaks through the turf on gentle ground (no rockRelief there):
// flat pale cut-outs at 1:1 (crop c5, s5 / s7 slopes) - slabs with their own tilts (lit / shaded), crisp joint cracks
// between them. x-z Voronoi (gentle ground: no fall-line stretch), 14 / 5 m, footprint-faded; returns the normal
// offset (x, 0, z), crack in [0, 1]
vec3 outcropSlabs(vec2 p, float fp, out float crack) {
    vec3 g = vec3(0.0);
    crack = 0.0;
    float S = 14.0, amp = 0.55;
    for (int i = 0; i < 2; i++) {
        float v = smoothstep(2.5 * fp, 5.0 * fp, S);
        if (v <= 0.0) break;
        vec2 bn; float id2;
        vec3 vr = voronoi2n(rot2(0.6 + 1.1 * float(i)) * p / S + vec2(5.1, 2.3) * float(i + 1), bn, id2);
        vec2 tl = vec2(fract(vr.z * 13.7), fract(vr.z * 29.3)) - 0.5;
        vec2 tl2 = vec2(fract(id2 * 13.7), fract(id2 * 29.3)) - 0.5;
        float edge = vr.y - vr.x;
        float eb = max(0.06, 1.2 * fp / S);
        vec2 t2 = mix(0.5 * (tl + tl2), tl, smoothstep(0.0, eb, edge));
        g += v * amp * vec3(t2.x, 0.0, t2.y);
        crack = max(crack, v * (1.0 - smoothstep(0.0, max(0.05, 0.9 * fp / S), edge)) * smoothstep(-0.1, 0.2, gnoise(p * (2.0 / S) + 3.7)));
        S *= 0.36; amp *= 0.7;
    }
    return g;
}

// (session 5) fine joint texture of the granite faces down to ~2.5 px: the faces carry
// a dense texture of thin dark joints in two sets - near-vertical joints and dipping sheet joints - on top of
// the facets; the renders had ~60 % of the target 8-px rock contrast. Each line is the zero crossing of an
// anisotropic (5:1) gradient noise in face coordinates, drawn with a constant width in metres from the
// noise's own gradient (|n| / |grad n|), at least ~1 px wide and faded by coverage below that; an octave
// goes before its line spacing is < ~2.5 px. Anisotropic, so the lines are joint sets, never the closed
// loops of an isotropic noise (the snakeskin). Per-plane lines blended by weight (no mixed zero crossings).
float jointSet(vec2 F, vec2 dir, float wl0, float fp, float off) {
    vec2 q = vec2(dot(F, vec2(-dir.y, dir.x)), dot(F, dir));
    float L = 0.0, wl = wl0, D = 0.55, wln = 0.9;   // (t19: 0.55 / 0.45 m -> 0.62 / 0.9 m: t18 barely showed; t20 D 0.55)
    for (int i = 0; i < 4; i++) {
        float v = smoothstep(1.8 * fp, 4.5 * fp, 0.5 * wl);
        if (v <= 0.0) break;
        vec2 fq = vec2(1.0 / wl, 0.2 / wl);
        vec3 nd = gnoised(q * fq + off + float(i) * 7.13);
        float d = abs(nd.x) / max(length(nd.yz * fq), 1e-5);   // metres to the zero crossing
        float hw = max(wln, 0.75 * fp);
        float line = (1.0 - smoothstep(0.5 * hw, hw, d)) * min(1.0, 1.4 * wln / max(0.75 * fp, 1e-4));
        // lines come and go along their length (open / closed joints)
        // (t20: segments ~2 wavelengths long instead of ~7: long continuous lines read as drawn contours)
        line *= smoothstep(-0.30, 0.10, gnoise(q * fq * vec2(0.5, 1.4) + off + 3.7 + float(i)));
        L = max(L, v * D * line);
        wl *= 0.45; D *= 0.9; wln *= 0.7;
    }
    return L;
}
float graniteJoints(vec2 pxz, float fp) {
    // set A ~ vertical (+-12 deg), set B dipping sheets (+-25 deg off horizontal), both turning slowly
    float aA = 0.21 * gnoise(pxz * 0.0011 + 61.0);
    float aB = 0.45 * gnoise(pxz * 0.0009 + 73.0);
    vec2 dA = vec2(sin(aA), cos(aA)), dB = vec2(cos(aB), sin(aB));
    float mA = smoothstep(-0.45, 0.10, gnoise(pxz * 0.0035 + 51.0));
    float mB = smoothstep(-0.40, 0.15, gnoise(pxz * 0.0042 + 57.0));
    float j = 0.0;
    for (int k = 0; k < 2; k++) {
        float w = k == 0 ? gFwa : gFwb;
        if (w < 0.04) continue;
        vec2 F = k == 0 ? gFa : gFb;
        float lA = mA > 0.01 ? mA * jointSet(F, dA, 34.0, fp, 11.0 + 5.0 * float(k)) : 0.0;
        float lB = mB > 0.01 ? mB * jointSet(F, dB, 26.0, fp, 29.0 + 5.0 * float(k)) : 0.0;
        j += w * max(lA, 0.85 * lB);
    }
    return j;
}

// Macro materials of the mountains: above the forest the slopes are a mosaic of
// grass ledges, dark dwarf-pine patches and granite outcrops; pale scree fans run down below the
// cliffs; summer snow survives as sharp-edged patches in couloirs and on benches, and as fields near
// the summits. Close to the camera the vegetation mask stays the one the 3D ground cover uses.
vec3 terrainAlbedo(vec3 p, inout vec3 n, float t, vec4 f, float wl, float ao, out float wetness, out float snowM, out float spec) {
    gVegS = 0.0;
    if (uDebug == 44) { spec = 0.0; wetness = 0.0; snowM = 0.0; return vec3(0.18); }
    float h = p.y;
    float above = wl > -1000.0 ? h - wl : 999.0;
    float ny = n.y;
    float fp = t * gPixAng;                                   // pixel footprint (m)
    // (round 3) the footprint ALONG an obliquely seen slope is fp / cos(view, normal): the fine debris / turf grain
    // filtered by the isotropic fp aliased into a brushed hatch on the cones seen at 50-70 deg (crop c5)
    float fpO = fp / clamp(abs(dot(n, normalize(p - uCamPos))), 0.25, 1.0);
    float n1 = fbm2(p.xz * 0.012 + 3.0, 4);
    float n2 = gnoise(p.xz * 0.11 + 7.0);
    float n3 = gnoise(p.xz * 0.45 + 1.0);
    float n2f = n2 * (1.0 - smoothstep(3.0, 9.0, fpO));      // footprint-filtered small scales
    float n2ff = n2 * (1.0 - smoothstep(2.0, 4.5, fpO));     // (for the masks: gone before it is < 2 px)
    float n3f = n3 * (1.0 - smoothstep(0.8, 2.5, fpO));
    float nb = fbm2(p.xz * 0.0021 + 11.0, 3);                 // ~500 m: snow / grass limits
    float nm = fbm2F(p.xz * 0.021 + 5.0, 3, 48.0, fp);       // ~50 m: outcrops, patch shapes
    float far = smoothstep(60.0, 160.0, t);                   // new macro masks only away from the camera
    // (t11) fine mosaic noise (9 / 4.5 / 2.2 m, footprint-faded) for the material edges: the soft 4 m-map masks
    // gave 10-30 m blurred margins - turf islands on the scree / rock read as pasted, smeared stickers at 1:1.
    // Added to a soft mask before a narrow threshold it turns the transition into a crisp interlocking mosaic
    // whose density follows the soft mask (grass in the joints, rock knobs in the turf); faded back to the
    // soft mask once the cells are < ~2-5 px (no sparkle)
    float dV = 1.0 - smoothstep(2.0, 5.0, fp);
    float nD = dV > 0.0 ? fbm2F(p.xz * (1.0 / 9.0) + 3.1, 3, 9.0, fpO) / 0.15 : 0.0;
    float nD2 = dV > 0.0 ? fbm2F(rot2(1.3) * p.xz * (1.0 / 7.0) + 8.7, 3, 7.0, fpO) / 0.15 : 0.0;

    // coarse geometry (>= 36 m, wider with the footprint) for everything that decides snow, scree and
    // turf far away: gully / rib curvature (ring minus a small centre ring, so the crag relief at the
    // pixel itself does not pick every micro-gully), gradient and normal from the same taps
    float curv = 0.0, curvB = 0.0, curvG = 0.0, curvX = 0.0, curvBX = 0.0;
    vec3 nC = n;
    float aoS = ao;                                           // sky visibility averaged over ~30 m
    if (far > 0.0 && h > 100.0) {
        float rc = max(64.0, 3.0 * fp);
        float tE = terrainH(p.xz + vec2(rc, 0.0), false), tW = terrainH(p.xz - vec2(rc, 0.0), false);
        float tN = terrainH(p.xz + vec2(0.0, rc), false), tS = terrainH(p.xz - vec2(0.0, rc), false);
        // centre: an 8-tap ring at 0.3 rc (session 6: the 4-tap 16 m ring still picked up the 4 m crag
        // relief, so the curvature - and the snow edges cut by it - jumped from pixel to pixel)
        float r4 = 0.3 * rc, r5 = 0.212 * rc;
        float hc = 0.125 * (terrainH(p.xz + vec2(r4, 0.0), false) + terrainH(p.xz - vec2(r4, 0.0), false)
                          + terrainH(p.xz + vec2(0.0, r4), false) + terrainH(p.xz - vec2(0.0, r4), false)
                          + terrainH(p.xz + vec2(r5, r5), false) + terrainH(p.xz - vec2(r5, r5), false)
                          + terrainH(p.xz + vec2(r5, -r5), false) + terrainH(p.xz - vec2(r5, -r5), false));
        curv = (tE + tW + tN + tS - 4.0 * hc) / rc;
        // the same at ~160 m (only the big couloir systems are concave at both scales)
        float rb = max(160.0, 6.0 * fp);
        curvB = (terrainH(p.xz + vec2(rb, 0.0), false) + terrainH(p.xz - vec2(rb, 0.0), false)
               + terrainH(p.xz + vec2(0.0, rb), false) + terrainH(p.xz - vec2(0.0, rb), false) - 4.0 * hc) / rb;
        // (session 4) the ~900 m spur-and-gully systems of the relief pass: curvature at ~400 m
        float rg = max(400.0, 10.0 * fp);
        curvG = (terrainH(p.xz + vec2(rg, 0.0), false) + terrainH(p.xz - vec2(rg, 0.0), false)
               + terrainH(p.xz + vec2(0.0, rg), false) + terrainH(p.xz - vec2(0.0, rg), false) - 4.0 * hc) / rg;
        // (session 4: 4 taps at 16 m -> 8 taps at >= 32 m: the snow edges cut by the hollow test traced the
        // 8 m AO texels - straight-edged, square chips at 1720 px)
        float r3 = max(32.0, 3.0 * fp), r6 = 0.7071 * r3;
        aoS = mix(ao, 0.125 * (terrainAO(p.xz + vec2(r3, 0.0)) + terrainAO(p.xz - vec2(r3, 0.0))
                             + terrainAO(p.xz + vec2(0.0, r3)) + terrainAO(p.xz - vec2(0.0, r3))
                             + terrainAO(p.xz + vec2(r6, r6)) + terrainAO(p.xz - vec2(r6, r6))
                             + terrainAO(p.xz + vec2(r6, -r6)) + terrainAO(p.xz - vec2(r6, -r6))), far);
        vec2 gC = vec2(tE - tW, tN - tS) / (2.0 * rc);
        nC = normalize(mix(n, normalize(vec3(-gC.x, 1.0, -gC.y)), far));
        // (world3) on the steep faces the couloir test uses the CROSS-slope curvature, averaged over three points
        // ~0.9 rc apart down the fall line: the x-z ring also sampled the face's own profile 2-4x further up /
        // down the slope, so the couloir fills broke into a series of flame-shaped flakes (debug 27, t8);
        // gullies are concave across, and the snow in them runs as continuous strips
        curvX = curv; curvBX = curvB;
        float sgC = length(gC);
        if (sgC > 0.3) {
            vec2 fl = gC / sgC, tg = vec2(-fl.y, fl.x);
            float cx = 0.0, cxB = 0.0;
            for (int k = -1; k <= 1; k++) {
                vec2 q = p.xz + fl * (float(k) * 0.9 * rc);
                float hq = terrainH(q, false);
                cx += (terrainH(q + tg * rc, false) + terrainH(q - tg * rc, false) - 2.0 * hq) / rc;
                cxB += (terrainH(q + tg * rb, false) + terrainH(q - tg * rb, false) - 2.0 * hq) / rb;
            }
            float wX = smoothstep(0.35, 0.8, sgC);
            curvX = mix(curv, 2.0 * cx / 3.0, wX); curvBX = mix(curvB, 2.0 * cxB / 3.0, wX);
        }
    }
    float nyC = nC.y;
    // turf / scree slope mask (fine detail up close). (round 3) far away it keeps part of the fine (footprint-
    // filtered) normal: the risers of the slab relief turn to rock, its treads hold turf - the turf / rock mosaic
    // follows the ground (it was a dither noise unrelated to the relief: camouflage, s7 lower left / s5)
    // (t19: 0.7 -> 0.5 + knob 4: the fine normal salted the 25-45 deg slopes with 2-10 m rock / turf speckle - camouflage)
    float nyV = mix(ny, mix(nyC, ny, 0.5), smoothstep(1.5, 4.0, fp));
    float gullyC = smoothstep(0.35, 1.1, curv) * far;
    float ribC = smoothstep(0.35, 1.1, -curv) * far;
    faceFrames(p, nC);
    // far away: granite base tone without rockAlbedo's axis-aligned streaks (they read as a diagonal
    // hatch on the oblique faces); the panel tones of the cliff joint sets stay
    // (session 4) the steep high walls hand over at 120-400 m instead of 300-900 m: rockAlbedo's warped
    // joint lines (6.5-71 m sets) drew a dark snakeskin network over the faces 0.3-2 km out (shot 0 left
    // face, shot 7) - at 3440 px it would cover every face within several km. The far granite has no
    // joint lines (the facet cracks below carry the structure); its base tone is a little warmer
    // (rock B-R +24..+30 in shots 0/1/7 against a target of +10..+25)
    float wallF = smoothstep(150.0, 250.0, h) * (1.0 - smoothstep(0.75, 0.85, nyC)) * far;
    float farA = mix(smoothstep(300.0, 900.0, t), smoothstep(120.0, 400.0, t), wallF);
    vec3 rock = farA < 1.0 ? rockAlbedo(p, n, t) : vec3(0.0);
    if (farA > 0.0) {
        rkCliffJoints(p, fp);                                 // (cached: rockAlbedo calls it too)
        // (session 6: the ~220 m tone mottling 0.32 -> 0.15 (soft camouflage patches); near grey granite, the warm
        // base only beyond ~2-4 km, where the haze turns it blue (near faces read beige, t22 shot 0 left face))
        vec3 rf = mix(vec3(0.218, 0.212, 0.203), vec3(0.226, 0.207, 0.186), smoothstep(1500.0, 4000.0, t))
                * (1.0 + (0.15 + W2K(4)) * fbm2(p.xz * 0.0045 + 3.0, 3)) * mix(0.92, 1.08, gRkJ.y);
        rock = mix(rock, rf * (1.0 + W2K(20)), farA);
    }
    // sub-heightfield relief of the faces (normal perturbation applied below) and its clefts
    float cav = 0.0;
    vec3 gb = vec3(0.0);
    // (world3) where the joint-block relief (world.glsl rockRelief) carries real geometry, the normal-only
    // substitutes (faceBump / faceFlutes / facet tilts / coarse-normal pull) fade out: their fall-line ribs
    // were the "melted wax" look
    float rkF = gRkWS;
    // (t23) where the footprint is < ~1-3 m the slab relief / micro-texture carry the walls: the fall-line ribs and
    // flutes (2-6 m octaves) drew dense vertical stripes at 1:1 final pixels on the walls without relief (s5 cirque)
    float flN = mix(0.3, 1.0, smoothstep(1.0, 3.0, fp));
    if (t > 70.0 && h > 150.0) gb = (0.5 + W2K(6)) * flN * faceBump(fp, cav);   // (session 4: 0.35 -> 0.25, "drip" ribs; session 6: 0.5 with the facets off)
    // (session 6) fall-line flutes: the organised sub-heightfield relief that replaces the random facet tilts
    float flA = (2.5 + W2K(7)) * flN;
    if (flA > 0.0 && t > 70.0 && h > 150.0) gb += flA * faceFlutes(fp);
    gb *= 1.0 - rkF;
    // (session 5) the creases of faceBump follow the zero contours of a gradient noise: meandering closed
    // loops - as dark lines they were the snakeskin network on the near walls at 1:1 final pixels (shot 0
    // left face, t16 crop c0). The dark lines now come from the facet joint sets only
    cav = 0.0;
    float crack = 0.0, ftone = 0.0;
    vec3 gF = vec3(0.0);
    if (t > 70.0 && h > 120.0) gF = faceFacets(fp, crack, ftone);
    // distant granite: mid-grey (lit #80878b), darker and streaked in the gullies
    // (wet rock, lichen, shadowed clefts), pale on the ribs and on fresh exfoliation slabs, banded by
    // sheet joints; keeps the faces from reading as pale plaster once the close-range detail is gone
    float farR = smoothstep(120.0, 800.0, t);
    // (session 4) the 900 m gully systems: dark (wet rock, lichen, dwarf pine below ~1100 m), so they
    // still read in sky-only shade (shot 6's left massif)
    // (band pass: the 160 m curvature minus 0.4 x the 400 m one - the 400 m term alone is dominated by
    // the valleys / slope feet (+-1.1 at p10/p90 on the faces), wt/curvmap.py; on slopes only)
    float gulG = smoothstep(0.35, 0.95, curvB - 0.4 * curvG) * (1.0 - smoothstep(0.82, 0.92, nyC)) * far;
    // the detail of the far granite (gullies, stains, facet cracks) takes over with the far tone, on the
    // walls already from ~100-300 m; the brightness calibration (x0.55) keeps the old 120-800 m ramp
    float wD = max(farR, wallF * smoothstep(100.0, 300.0, t));
    if (wD > 0.0) {
        // (session 4: gully darkening 0.66 -> 0.5: the faces lacked their darkest tones, p10 0.14 vs 0.05)
        float gul = max(smoothstep(0.76, 0.52, aoS + 0.05 * nm / 0.12), 0.8 * gullyC);
        // water / lichen stains hanging down the faces (sharp-edged dark stripes)
        float fsv = 1.0 - smoothstep(8.0, 25.0, fp);
        float fstreak = faceNoise(vec2(0.030, 0.012), 7.7)
                      + 0.45 * (1.0 - smoothstep(4.0, 10.0, fp)) * faceNoise(vec2(0.07, 0.025), 3.3);
        float fine = (1.0 - smoothstep(2.0, 5.0, fp)) * faceNoise(vec2(0.12, 0.025), 1.9);
        // (edge width follows the footprint: a fixed 0.04 edge aliased into crawling dashes at 1-3 km)
        float sew = 0.02 + 0.08 * smoothstep(2.0, 10.0, fp);
        float stain = smoothstep(0.32 - sew, 0.32 + sew, fstreak + 0.10 * fine) * fsv;
        stain *= 1.0 - 0.85 * rkF;               // (world3) the relief carries the structure
        // pale exfoliation slabs (fresh granite) on the open faces, only on the ribs / convex ground
        // (at x1.55 everywhere they were the pale blotches of the camouflage look)
        float slab = smoothstep(-0.08, 0.16, faceNoise(vec2(0.016, 0.011), 9.1) + 0.04 * nm / 0.12)
                   * (1.0 - smoothstep(25.0, 60.0, fp)) * (1.0 - gul) * max(ribC, smoothstep(0.0, 0.3, -curv))
                   * (1.0 - 0.6 * rkF);   // (t12: soft pale vertical smears on the relief walls)
        // sheet joints: dark ledge lines ~18 m apart across the steep faces
        float jw = (1.0 - smoothstep(1.8, 4.5, fp)) * (1.0 - smoothstep(0.55, 0.8, nyC));
        float joint = 0.0;
        if (jw > 0.0) {
            // (session 4) a regular line every 18 m over the whole face read as a ruled grid at the final
            // 3440 px (shot 0 left face): ~1 band in 3 is an open joint, 23 m base spacing, the lines ~0.7 m
            // wide (antialiased by the footprint, not 1-2 m bands)
            float jv = h / 23.0 + 0.6 * faceNoise(vec2(0.010, 0.004), 4.4);
            float jk = step(0.64, hash11(floor(jv) * 0.7548 + 0.113));
            float jd = abs(fract(jv) - 0.5) * 23.0;
            joint = jk * (1.0 - smoothstep(0.35, 0.35 + 1.2 * fp, jd)) * saturate(0.7 / (0.7 + 1.2 * fp))
                  * smoothstep(-0.15, 0.1, faceNoise(vec2(0.03, 0.02), 2.6)) * 1.6;
        }
        // (no more blotchy fbm mottling: the facets and their cracks carry the mid-scale structure)
        // (session 3: ribs x1.4 -> x1.5, fresh slabs x1.2 -> x1.35: the lit faces should reach luminance
        // 0.69-0.72 at p90 of the rock against 0.58-0.64 in the renders, at the same median)
        // (session 4: stains 0.78 -> 0.68, facet cracks 0.35 -> 0.5 (fewer of them: joint sets only), sheet
        // joints 0.7 -> 0.6 (sparse): the faces' darkest tones are crisp dark lines / streaks, not blotches)
        // (t19) on the slab-relief walls the gullies are real geometry now: their albedo darkening 0.68 -> 0.80 (dark
        // vertical smears = the "drips" at frame level, s7 / s1)
        float tone = mix(1.0, mix(0.68, 0.80, rkF) + W2K(2), gul) * (1.0 + 0.5 * ribC) * mix(1.0, 0.78 + W2K(23), stain) * (1.0 + 0.04 * fstreak * fsv)
                   * (0.90 + 0.20 * smoothstep(-0.15, 0.15, nb)) * (1.0 + (0.35 + W2K(5)) * slab) * mix(1.0, 0.6, saturate(joint) * jw * (0.3 + W2K(25))) * (1.0 - 0.3 * saturate(cav));
        tone *= (1.0 - (0.45 + W2K(13)) * crack) * (1.0 + W2K(14) * ftone) * 1.02 * mix(1.0, 0.78 + W2K(3), gulG);   // (session 5: 0.5 / 0.72 -> 0.62 / 0.62)
        // (session 6: AO gullies 0.44 -> 0.68, big gullies 0.62 -> 0.78, stains 0.68 -> 0.78, cracks 0.62 -> 0.45, cell
        // tone off, sheet joints x0.5: soft dark albedo blotches read as camouflage; t22b isolated each term)
        // (session 5) fine joint sets on the steep faces (not on benches / gentle ground)
        // (t20: only the steep walls (> ~45 deg) seen from > ~1 km: on the 40 deg knobs and the
        // near walls (shot 0 left) the lines read as a net drawn on a smooth surface)
        // (session 6: from 1.5-3 km - on the 1.4 km face of shot 0 they read as a scribbled net at 1:1 final pixels)
        float jW = (1.0 - smoothstep(0.62, 0.80, nyC)) * smoothstep(1500.0, 3000.0, t);
        if (jW > 0.0 && t > 70.0) tone *= 1.0 - jW * (0.6 + W2K(12)) * graniteJoints(p.xz, fp);
        // (session 6: 0.70 -> 0.55: lit rock / zenith sky was 0.7-0.95 against a target of 0.62)
        rock *= mix(1.0, tone, wD) * mix(1.0, 0.78, farR);   // (t12: 0.65 -> 0.72: rock medians 10-15 % below target, high-key granite)   // (session 6: 0.55 -> 0.65: lit-rock p90 0.33-0.36 vs a target of 0.39-0.47)
        // 8-16 km: lit faces darker (the far ranges should read nearly flat blue with white fields on them)
        rock *= mix(1.0, 0.72, smoothstep(8000.0, 20000.0, t));
    }
    // (t14) weathering mosaic of the granite walls (at 1-3 km: lichen-darkened slabs, fresh pale
    // scars, at 6-60 m): volume noise in (x, y, z), isotropic on any slope (no fall-line smear), faded by the
    // footprint; on the relief walls / steep faces only
    float lmW = max(rkF, wallF) * (1.0 - smoothstep(3.0, 8.0, fp));
    if (lmW > 0.0) {
        vec4 lz = texture(uNoise3, p * (1.0 / 256.0) + 0.53);
        float lich = smoothstep(0.56, 0.70, lz.b + 0.4 * (lz.r - 0.475));
        float fresh = smoothstep(0.64, 0.78, texture(uNoise3, p * (1.0 / 150.0) + 0.29).a);
        rock *= mix(vec3(1.0), mix(vec3(1.0), mix(vec3(1.0), vec3(0.74, 0.76, 0.70), 0.6), lich) * (1.0 + 0.12 * fresh * (1.0 - lich)), lmW);
    }
    // (t21) granite micro-texture where the footprint resolves 1-8 m: at 1:1 final pixels the 0.5-1.5 km walls and
    // knobs (s5 cirque, s0 left wall: footprint 0.1-0.3 m) were smooth plaster - the walls should show weathering tone
    // patches, dark lichen / moss blotches and fine grain at those scales. 3D noise (isotropic on any slope), each
    // octave faded before it is < ~5 px of the oblique footprint (no sparkle)
    // (t21 variants: x2 gives the s0 near wall its weathering blotches, x1 barely shows -> 1.6)
    float mvA = (1.0 - smoothstep(0.6, 1.6, fpO)) * far * 1.6;
    if (mvA > 0.0) {
        float m8 = clamp((texture(uNoise3, p * (1.0 / 8.0) + 0.11).r - 0.48) / 0.1, -2.0, 2.0);
        float m3 = clamp((texture(uNoise3, p * (1.0 / 3.0) + 0.47).b - 0.52) / 0.09, -2.0, 2.0) * (1.0 - smoothstep(0.25, 0.6, fpO));
        float lk = smoothstep(0.63, 0.71, texture(uNoise3, p * (1.0 / 5.0) + 0.71).a);
        rock *= max(1.0 + mvA * (0.07 * m8 + 0.06 * m3 - 0.3 * lk), 0.3);
    }
    // (world5 R1) pixel-scale granite grain of the faces at 0.15-9 km: the walls carry
    // crisp tone structure down to the pixel - lichen-darkened and fresh pale slabs, dark wet / lichen streaks hanging
    // down the faces, crack shadows; the renders' faces were smooth plaster below the ~40 m relief (DoG band contrast
    // 0.11-0.14 log2 at 1-8 px against a target of 0.19-0.22). Octaves of the smooth single-octave value noise
    // (uNoise3.g, features ~wl), 1.8x taller than wide (fall-line streaks), kept to the 2.5-40 px band of the oblique
    // footprint at any distance (sw1: the 30-60 px octaves only added soft camouflage blotches) and faded in / out
    // with the footprint (no shimmer in the moving film); sparse crisp dark streak patches (>= 4-8 px)
    float grW = far * (1.0 - smoothstep(9000.0, 14000.0, t)) * max(max(rkF, wallF), 1.0 - smoothstep(0.62, 0.85, nyC));
    if (grW > 0.0) {
        float lg = 0.0, dkS = 0.0, wlG = 64.0;
        for (int i = 0; i < 7; i++) {
            if (wlG < 3.0 * fpO) break;
            // (shimmer test 17:17, shot 7 temporal / spatial 1.54 / 1.39 at 2.5 px: 3-6 px)
            float v = smoothstep(3.0 * fpO, 6.0 * fpO, wlG) * (1.0 - smoothstep(14.0 * fpO, 40.0 * fpO, wlG));
            vec3 q = vec3(p.x, 0.55 * p.y, p.z) * (1.0 / (16.0 * wlG)) + vec3(0.137, 0.291, 0.419) * float(i + 1);
            if (v > 0.0) lg += v * (texture(uNoise3, q).g - 0.487) / 0.153;
            float vS = smoothstep(5.0 * fpO, 10.0 * fpO, wlG) * (1.0 - smoothstep(24.0 * fpO, 60.0 * fpO, wlG));
            if (vS > 0.0) dkS = max(dkS, vS * smoothstep(0.71, 0.74, texture(uNoise3, q * vec3(1.7, 0.6, 1.7) + 0.31).g));
            wlG *= 0.5;
        }
        rock *= exp2(grW * ((0.30 + W2K(27)) * lg - (0.80 + W2K(28)) * dkS));   // (sw2 / sw4: 0.30 / 0.80)
    }
    // ---- scree / talus: pale, nearly neutral granite debris (#a4a7a3), lighter than the cliffs above,
    // with debris streaks down the fall line and a boulder grain at mid range
    float sF = faceNoise(vec2(0.02, 0.0045), 5.3) * (1.0 - smoothstep(8.0, 20.0, fp));   // fan-scale streaks
    // (round 3) isotropic debris tone: the face-coordinate streaks (2-4:1 down the fall line) read as brushed
    // metal on the cones at 1:1 (hero cone, crop c5)
    float sT = saturate(0.5 + 0.55 * n3f + 0.35 * n2f + 0.25 * nm);
    // (t19: far 0.48 -> 0.58 + knob 2: the fans are the brightest ground after the snow, #a4a7a3 vs lit granite
    // #80878b; the 0.48 fans read as lavender-grey smears level with the rock)
    vec3 scree = mix(vec3(0.255, 0.258, 0.250), vec3(0.42, 0.425, 0.41), sT) * mix(1.0, 0.64, farR);   // (t23: 0.58 -> 0.64)
    float sOld = 0.0, sShrub = 0.0;
    vec3 scrB = vec3(0.0);
    // (t22, perf) the debris texture only where the talus map has debris (6-8 Voronoi evaluations per pixel)
    if (f.b > 0.02) {
    // block-size sorting / shadowed gaps: mottled at 15-60 m
    float sM = fbm2F(rot2(0.7) * p.xz * (1.0 / 22.0) + 8.2, 3, 22.0, fp);
    float sFi = fbm2F(p.xz * (1.0 / 50.0) + 5.3, 2, 50.0, fp) / 0.12 * 0.2;
    scree *= mix(1.0, (0.85 + 0.4 * sFi) * (1.0 + 0.5 * sM), far);
    // (session 6) 4-12 m block / gap grain at 0.3-3 km: the fans read as flat pale paint without it
    // (session 4: the value-noise grain read as smooth pale concrete (the hero cone is a coarse block field):
    // ~4.5 m Voronoi blocks, per-block tone +-20 %, dark gaps x0.6 (gap width >= ~1.2 px), faded once a
    // block is < ~2.5 px; beyond that 15-30 m block clusters +-12 % (sorted patches, shadowed hollows))
    // (t22) the block grain at EVERY distance: the octave of Voronoi blocks (4.5, 9, 18, 36 m) whose cells are 2.4-4.8
    // px of the oblique footprint, cross-faded between two octaves. The 4.5 m grain alone faded out beyond fpO ~1.9 m,
    // so the fans of shot 7 (2-3 km, 1.3 m final pixels, fpO 2-4 m) were smooth plaster bands; scree fans at 1-3
    // km are a salt-and-pepper of lit blocks and shadowed gaps down to the pixel. Coarser octaves = block clusters:
    // lower contrast
    if (far > 0.0 && fpO < 60.0) {
        float lc = log2(max(2.4 * fpO / 4.5, 1.0));
        float o0 = floor(lc), fo = lc - o0;
        float grain = 0.0;
        for (int k = 0; k < 2; k++) {
            float ok = o0 + float(k);
            float S = 4.5 * exp2(ok);
            vec3 vb = voronoi2(p.xz * (1.0 / S) + 7.0 + 3.1 * ok);
            // (round 3: +-20 % / gaps x0.6 -> +-26 % / x0.45, wider gaps: the cone read as smooth grey clay at 1 km, it should
            // be a coarse salt-and-pepper block field, in-box std ~20)
            float ewB = max(0.10, 1.2 * fpO / S);
            float gapB = (1.0 - smoothstep(0.0, ewB, vb.y - vb.x)) * (0.10 / ewB);   // (wider gap, weaker: same mean)
            float ct = mix(1.3, 0.7, saturate(ok / 3.0));   // (t22 variants: x1.6 a little crunchy, x0 = smooth paint)
            grain += (k == 0 ? 1.0 - fo : fo) * (1.0 + ct * ((1.0 + 0.52 * (vb.z - 0.5)) * (1.0 - 0.55 * gapB) - 1.0));
        }
        scree *= mix(1.0, grain, far * (1.0 - smoothstep(30.0, 60.0, fpO)));
    }
    float cwM = smoothstep(0.8, 2.0, fpO) * (1.0 - smoothstep(8.0, 16.0, fpO));
    if (cwM > 0.0) {
        vec3 vc = voronoi2(p.xz * (1.0 / 22.0) + 3.0);
        float mc = 0.5 * (2.0 * vc.z - 1.0) + 0.5 * clamp(gnoise(p.xz * 0.05 + 2.0) / 0.35, -1.0, 1.0);
        scree *= 1.0 + 0.12 * cwM * mc;
    }
    // (session 6) the 0.5-3 km fans read as pale paint smears (t19 shot 7): crisp dark debris
    // furrows / channel shadows down the fall line, ~16 m and finer, drawn like the joint lines (>= ~1 px)
    float dsw = 0.25 * far * smoothstep(1.0, 2.5, fp) * (1.0 - smoothstep(14.0, 28.0, fp));   // (t13: 1 -> 0.5, brushed lines at 1:1)
    if (dsw > 0.0) {
        float fur = 0.0;
        if (gFwa > 0.04) fur += gFwa * jointSet(gFa, vec2(0.0, 1.0), 36.0, fp, 41.0);
        if (gFwb > 0.04) fur += gFwb * jointSet(gFb, vec2(0.0, 1.0), 36.0, fp, 47.0);
        scree *= 1.0 - dsw * fur;
    }
    float bw = 1.0 - smoothstep(0.25, 1.2, fpO);
    if (bw > 0.0) {
        vec4 bv = voronoiV(p.xz * 0.9 + 13.0);
        float gap = smoothstep(0.25, 0.5, bv.z);
        scree *= mix(1.0, mix(1.08, 0.62, gap) * mix(0.85, 1.12, bv.w), bw);
    }
    // (t11) the hero cone at 1:1: not an even grey asphalt grain but sorted debris - ~9 m boulders (to ~3.8 m
    // footprint), lobes of older lichen-grey / olive debris between fresh pale tongues (30-80 m), and the
    // older lobes colonised by dwarf shrubs / grass tufts (dark dots, cells > ~2.5 px)
    if (far > 0.0) {
        // (t13) the blocks as lit / shaded faces (dome normals, applied with the other normal terms below): flat
        // tone cells read as asphalt; ~9 m blocks on ~45 % of the cells (to ~2.5 px), 3.5 m blocks (to ~2 px)
        float bw9 = 1.0 - smoothstep(2.2, 4.5, fpO);
        if (bw9 > 0.0) {
            vec4 v9 = voronoiV(p.xz * (1.0 / 9.0) + 2.0);
            float big = step(0.55, v9.w);                       // ~45 % of the cells hold a big block
            scree *= mix(1.0, mix(1.0, 0.86 + 0.32 * fract(v9.w * 7.3), big), bw9 * far);
            scrB += bw9 * far * big * 0.75 * smoothstep(0.12, 0.6, v9.z) * vec3(v9.x, 0.0, v9.y) / max(v9.z, 1e-3);
        }
        float bw3 = 1.0 - smoothstep(0.9, 1.9, fpO);
        if (bw3 > 0.0) {
            vec4 v3 = voronoiV(rot2(0.5) * p.xz * (1.0 / 3.5) + 41.0);
            vec2 o3 = rot2(-0.5) * (v3.xy / max(v3.z, 1e-3));
            scrB += bw3 * far * 0.85 * smoothstep(0.1, 0.62, v3.z) * vec3(o3.x, 0.0, o3.y);
        }
        sOld = smoothstep(-0.05, 0.30, fbm2F(p.xz * 0.016 + 27.0, 3, 62.0, fp) / 0.12 * 0.25 + 0.35 * (0.62 - f.b) / 0.3);
        scree *= mix(vec3(1.0), vec3(0.76, 0.78, 0.72), sOld * far);
        float shV = 1.0 - smoothstep(1.4, 2.8, fpO);
        if (shV > 0.0) {
            vec3 vs = voronoi2(p.xz * (1.0 / 4.0) + 17.0);
            float rS = 0.28 + 0.16 * fract(vs.z * 5.1);
            float clump = smoothstep(-0.05, 0.30, gnoise(p.xz * 0.035 + 41.0));   // (t12: ~30 m clumps, not an even polka-dot grid)
            sShrub = step(fract(vs.z * 11.3), 0.45 * sOld * clump * smoothstep(250.0, 650.0, 1300.0 - h))
                   * (1.0 - smoothstep(rS - 0.4 * fp / 4.0, rS + 0.4 * fp / 4.0, vs.x)) * shV * far;
        }
    }
    }   // (f.b > 0.02)
    // cool, never clipping (#ced4dc lit); a little brighter on the far ranges, where the haze would
    // otherwise wash the snowfields into the blue rock (crisp white fields at 10-20 km)
    // (r4 M4: neutral albedo - the sky ambient alone gives the cool cast (lit snow was #c1cfe8, B-R +35; target +2..+14);
    // no far boost: the far ranges read as snow-mantled hero peaks)
    vec3 snow = vec3(0.70, 0.70, 0.69);
    snow *= 1.0 + 0.06 * clamp(fbm2F(p.xz * 0.025 + 41.0, 2, 40.0, fp) / 0.12, -1.5, 1.5);   // +-6 % (40 m)
    vec3 pebble = mix(vec3(0.19, 0.187, 0.18), vec3(0.27, 0.265, 0.255), saturate(0.5 + 0.8 * n3));

    // ---- vegetation: the original slope mask near the camera (shared with the 3D cover) ...
    float vegOld = smoothstep(0.60, 0.82, ny + 0.10 * n2) * (1.0 - smoothstep(950.0, 1250.0, h + 160.0 * n1));
    // ... and away from it the alpine mosaic: steeper ground turns to rock with altitude, granite
    // knobs break the grass above the forest, the highest grass ledges end ~1000-1300 m
    float hv = h + 150.0 * nb + 50.0 * nm;
    // (world3) the turf / rock / dwarf-shrub belt ~150 m higher (green slopes with rock knobs up to the
    // foot of the walls; the 600-1000 m slopes of shot 5 read as a pale grey wash with dark stickers)
    float hvV = hv - 150.0;
    // (session 6: 0.64..0.80 left the 45-deg slopes right above the forest bare grey; there should be a
    // belt of turf, dwarf shrubs and scree between the forest and the walls)
    // (session 6: 0.54 @450 -> 0.50 @550 m: the 35-45 deg slopes around the tarn read as bare pale clay)
    float ty = mix(0.50, 0.80, smoothstep(550.0, 1150.0, hvV)) + 0.1 * W2K(22);
    float nT = fbm2F(p.xz * 0.0067 + 5.0, 3, 150.0, fp);     // ~150 m turf / rock mosaic
    // (session 4: edge 0.16 -> 0.28 - bright olive "stickers" with hard edges on 35-45 deg rock; above
    // ~900 m turf only on the benches)
    // (round 3) ledge bands: turf on strata ledges running along the slope (strike), ~10-15 m thick and 50-80 m long,
    // dipping with the region; rock on the ribs, turf in the gullies - instead of isotropic
    // speckle
    float ledge = 0.0;
    if (far > 0.0 && dV > 0.0) {
        float dip = 0.35 * gnoise(p.xz * 0.0013 + 91.0) / 0.25;
        mat2 rd2 = rot2(dip);
        float v6 = 1.0 - smoothstep(1.2, 2.4, fpO);
        for (int k = 0; k < 2; k++) {
            float w = k == 0 ? gFwa : gFwb;
            if (w < 0.04) continue;
            vec2 F = rd2 * (k == 0 ? gFa : gFb);
            ledge += w * (0.65 * gnoise(F * vec2(1.0 / 70.0, 1.0 / 14.0) + vec2(3.3, 7.1 * float(k)))
                        + 0.35 * v6 * gnoise(F * vec2(1.0 / 30.0, 1.0 / 6.0) + vec2(9.7, 1.3 + 5.0 * float(k))));
        }
        ledge *= dV / 0.22;
    }
    float vegIn = nyV + 0.06 * n2ff / 0.22 + 0.03 * n3f / 0.22 + 0.08 * nT / 0.12
                + 0.12 * ledge + 0.08 * (gullyC - ribC);   // (t19: ledges 0.07 -> 0.12, ribs 0.05 -> 0.08: the slopes = turf in strata bands, rock on the ribs)
    float vegN = smoothstep(ty, ty + 0.28, vegIn) * (1.0 - smoothstep(950.0, 1350.0, hvV));
    vegN *= mix(1.0, smoothstep(0.62, 0.78, nyC), smoothstep(850.0, 1000.0, hvV) * far);   // (t16: 650-800 m stripped the slopes)
    // (t11) crisp turf / rock mosaic instead of a blurred margin (see nD)
    float kD = 0.12;   // (round 3: 0.30 -> 0.12, the geometry decides - see nyV)
    vegN = mix(vegN, smoothstep(0.42, 0.58, vegN + kD * nD * smoothstep(0.0, 0.2, vegN)), dV * far);
    float outc = smoothstep(0.02, 0.22, nm + 0.10 * n2ff + 0.25 * (ao - 0.85)) * smoothstep(380.0, 620.0, hvV) * (1.0 - f.a);
    // granite knobs and slabs breaking through the open stands of the upper forest: the
    // same noises / thresholds as the outcrops that bake_forest clears of trees
    float slopeC = sqrt(max(1.0 - nyC * nyC, 0.0)) / max(nyC, 0.05);
    float nmO = fbm2F(p.xz * 0.042 + 5.0, 3, 24.0, fp);       // ~25 m (bake_forest: same noise)
    float oS = 0.55 * nmO / 0.12 + 0.45 * nT / 0.12 + 0.8 * ribC + 1.5 * (slopeC - 0.85);
    float openA = smoothstep(90.0, 280.0, h);
    // (only the bare knobs: the gaps the bake opens between the trees are mostly grass and shrubs, and
    // the open belt higher up keeps its turf / rock mosaic below)
    // (r4 M11: = bake_forest outR: threshold -0.3, up to ~700 m)
    float outLow = smoothstep(1.2 - 1.2 * openA, 1.7 - 1.2 * openA, oS) * smoothstep(40.0, 100.0, h)
                 * (1.0 - smoothstep(600.0, 750.0, h)) * (1.0 - f.a) * far;
    outc = max(outc, outLow);
    vegN *= 1.0 - 0.9 * outc;
    vegN = max(vegN, saturate(f.r * 2.5) * smoothstep(0.42, 0.6, nyV));     // forest floor holds on steep ground
    // scree: soft map value up close (shared with the 3D cover), far away a hard, interlocking edge
    // broken by the fan streaks
    // (t19) crisp, lobed fan margins: the 0.40-0.50 band of the 4 m map's soft debris value was a 10-20 m wide
    // grey fringe (the fans read as soft smears at frame level, s7 lower left); a 0.04 band + 12-35 m lobes /
    // tongues (debris tongues and turf tongues interlock with sharp edges)
    float ew2 = 0.025 + 0.08 * smoothstep(3.0, 16.0, fp);
    float sLob = 0.07 * fbm2F(rot2(0.4) * p.xz * (1.0 / 35.0) + 2.9, 2, 35.0, fpO) / 0.12;
    // (t21) the steepness limits of the debris (the bake keeps it to <= ~38 deg on the 40 m slope; safety net: no pale
    // sheets on > ~45 deg faces on the 64 m slope, none on the summit zone) are part of the threshold argument
    // instead of multiplying the mask: multiplied, their 50-100 m soft ramps left half-transparent debris over the
    // rock / turf - the "grey smears" of s7's lower slopes (t19 scD variant: they dim with the scree albedo)
    // (t24) ... as a CRISP gate (1-2 % of ny, lobed by the fan noise), not an offset of the debris value: with the
    // debris map at 1.0 on the cirque walls an offset of -0.06 still passed, and the x-z debris grain on 50-70 deg walls
    // hung down as pixel-column stripes (s5 crop c5, t21-t23)
    float sArg = f.b + 0.25 * sF + sLob * smoothstep(0.15, 0.3, f.b);
    float scrN = smoothstep(0.43 - ew2, 0.43 + ew2, sArg);
    scrN = mix(scrN, smoothstep(0.43 - ew2, 0.43 + ew2, sArg + 0.35 * kD * nD2 * smoothstep(0.1, 0.3, f.b)), dV);   // (t11) interlocking fan margins
    float sGw = 0.012 + 0.02 * smoothstep(3.0, 16.0, fp);
    float sGate = smoothstep(-sGw, sGw, min(nyC - 0.615 + 0.5 * sLob, 0.6 * (nyV - 0.40) + 0.3 * sLob))
                * (1.0 - smoothstep(2000.0, 2400.0, h));
    float scr = mix(saturate(f.b * 1.3) * smoothstep(0.30, 0.50, nyV), scrN * sGate, far);
    float veg = mix(vegOld * (1.0 - f.b * 0.9), vegN * (1.0 - scr), far);
    // (session 4) the hero talus fan (bake.HERO_CONES) is bare blocks: no turf on it
    float heroFan = (1.0 - smoothstep(460.0, 640.0, length(p.xz - vec2(-2680.0, -320.0)))) * smoothstep(0.1, 0.35, f.b) * far;
    veg *= 1.0 - heroFan;
    // the outcrops in the forest band are pale exfoliated slabs, lighter than the cliffs above
    rock *= 1.0 + 0.15 * outLow * (1.0 - veg) * (1.0 - smoothstep(450.0, 800.0, h));

    // ---- late-summer snow: sharp-edged patches in couloirs and gullies and on benches/ledges, more
    // and larger with altitude, fields on the gentler ground near the summits; steep faces bare.
    // Expressed as an effective altitude: hollows, ledges and shady aspects hold snow lower down.
    // All terms use the coarse geometry and >= ~50 m noise: clustered patches, no flecks.
    float north = saturate(nC.z * 1.1 + 0.45);                // shaded aspects keep their snow
    // concave, sky-occluded couloirs: AO well below what is typical for that altitude
    float aoExp = mix(0.80, 0.58, smoothstep(0.0, 2400.0, h));
    float hollow = smoothstep(aoExp - 0.02, aoExp - 0.20, aoS);
    // the ranges 8-16 km out are the high, glaciated ones (snowfields on every far
    // summit): their snow limits sit lower (position-based, so it cannot change with the view)
    float rng = smoothstep(7000.0, 13000.0, length(p.xz));
    float rngB = smoothstep(0.4, 0.7, rng);
    // > ~65 deg (on the 64 m slope) sheds everything; on the far glaciated ranges up to ~75 deg (hanging
    // snowfields / ice on the faces)
    // (session 4: near walls 0.30..0.48 -> 0.40..0.56 (bare above ~57 deg): white paint on 60 deg faces read pasted)
    float shed = smoothstep(mix(0.40, 0.18, rngB), mix(0.56, 0.36, rngB), nyC + 0.05 * nm / 0.12);
    // ledges / gentler ground (< ~40 deg; on the far ranges, where only the 64 m+ shape survives the
    // haze, up to ~55 deg: their snowfields sit on every shoulder)
    // (r4 M3: near benches from ~37 deg (0.72 / 0.88 -> 0.62 / 0.80): late-summer snow lies in horizontal bands / bowl fields)
    float bench = smoothstep(mix(0.62, 0.55, rngB), mix(0.80, 0.75, rngB), nyC + 0.06 * nm / 0.12);
    float nS = fbm2(p.xz * 0.0009 + 17.0, 2) / 0.12;             // ~1 km: snowy vs bare faces
    float nL = fbm2(p.xz * 0.005 + 2.0, 2) / 0.12;               // ~200 m: fewer, larger patch clusters
    // ragged, lobed snow limits: ~25 m noise on the effective height and on the couloir test (a
    // domain-style perturbation of the inputs: ragged edges without the isolated flecks a noise added
    // to the final mask makes); filtered by the footprint
    float nE = fbm2F(p.xz * 0.04 + 7.0, 2, 25.0, fp) / 0.12;
    // (world3) the same edge noise in 3D (x, y, z): on the 60-75 deg walls the x-z noise hung down the face as
    // flames; a volume noise is isotropic on any slope (fbm channel, ~40 m with a ~130 m fallback far away)
    {
        float nA = (texture(uNoise3, p * (1.0 / 320.0) + 0.21).r - 0.5) / 0.1;
        float nB = (texture(uNoise3, p * (1.0 / 1000.0) + 0.67).r - 0.5) / 0.1;
        nE = mix(nE, mix(nA, nB, smoothstep(2.0, 4.0, fp)), saturate(0.6 * (1.0 - nyC) + 0.6) * far);
    }
    // (session 3: 25 m term 55 -> 30 m: the field edges were zig-zag dashes)
    float hN = h + 300.0 * nb + 120.0 * nm + 45.0 * nL + 20.0 * nE + 100.0 * W2K(19);   // patchy snow limits (t20: nE 30 -> 20)
    // (7-11 km: limits ~1100 m lower; the 11+ km ranges carry fields on their upper half only - with the
    // full offset they were salted with snow from 800 m up, which read as a pale comb)
    hN += (1100.0 * rng - 800.0 * rngB) * smoothstep(450.0, 1000.0, h);
    float fL = 700.0 * rngB;                                     // couloir / bench limits on the 11+ km ranges
    // couloirs: concave at both 64 m and 160 m (session 6: the 64 m test alone salted every gully
    // with 2-6 px dashes); about half of the couloir systems hold snow, the others are bare
    float cBig = min(curvX, 1.4 * curvBX) + 0.06 * nE;   // (t20: 0.10 -> 0.06: flame-shaped couloir fills)
    // (round 2: wider couloir fills, about two thirds of the couloir systems hold snow)
    // (round 3) couloir snow is avalanche deposit: a narrow strip on the steep gully floor, widening where the floor
    // flattens toward its foot (the cone), lasting lower than the open-slope snow. The old altitude ramp narrowed every
    // strip downward into a flame (t17 debug 27: the couloir term = the flames of shot 7)
    float cW = 0.30 * (nyC - 0.45);
    float sC = smoothstep(0.18, 0.60, cBig + cW) * far
             * smoothstep(480.0 + fL, 800.0 + fL, hN + 150.0 * (north - 0.5) + 150.0 * smoothstep(0.40, 0.65, nyC));
    // (session 3: on the near walls only ~1/3 of the couloir systems hold snow - a dash in every gully
    // read as a pattern from the lake, shots 0/1; the far ranges keep the old gate)
    sC *= smoothstep(mix(-0.45, -0.8, rngB), mix(0.35, 0.1, rngB), 0.4 * nL + 0.8 * nS);   // (t14: ~45 %)
    // (r4 M3) on the near ranges the couloir strip holds mainly where the gully floor flattens (its cone / foot): the
    // continuous strips down the steep faces were the paint drips / flames (blob h/w 1.1-1.9, target 0.67-0.87)
    sC *= mix(mix(0.25, 1.0, smoothstep(0.55, 0.75, nyC)), 1.0, rngB);   // (b2 shot 7: blob h/w still 1.3 at 0.4 / 0.72)
    float sBn = bench * mix(0.6, 1.0, max(smoothstep(0.05, 0.5, cBig), hollow))     // ledges, aprons at the
              * smoothstep(mix(800.0, 680.0, hollow) + fL, mix(1080.0, 960.0, hollow) + fL, hN + 250.0 * (north - 0.5)); // foot of the faces
    float sHi = smoothstep(mix(1650.0, 1800.0, rngB), mix(2250.0, 2300.0, rngB), hN + 300.0 * (north - 0.5) + 250.0 * hollow)
              * mix(mix(0.75, 1.0, bench), mix(0.15, 1.0, bench) * smoothstep(-0.7, 0.5, nS + 0.6 * nL), rngB);                                                  // summit fields
    // (far ranges: fields on the gentler shoulders and in ~1 km patches only, steep upper faces bare -
    // with the full summit term every crest above the limit was white)
    // (edge noise 0.07 -> 0.04 and faded out from 4 px: below that it only made flecks)
    // (round 2) a few LARGE fields instead of a fleck in every gully (snow lies in the high
    // hollows below the faces, on the aprons at their foot and on the upper shoulders; 3-6 % of a high
    // view against ~1-2 %): concave at the 160 m scale, < ~50 deg, from ~1150-1400 m (lower on north
    // aspects), in ~1 km clusters
    float sHo = smoothstep(0.10, 0.45, 1.4 * curvB + 0.06 * nE) * smoothstep(0.48, 0.64, nyC + 0.04 * nm / 0.12)   // (r4 M3: 0.55 / 0.70)
              * smoothstep(880.0 + fL, 1180.0 + fL, hN + 260.0 * (north - 0.5) + 120.0 * hollow) * far   // (session 5: -70 m)
              * smoothstep(-0.6, 0.25, 0.6 * nL + 0.7 * nS);
    // upper north / east shoulders of the near walls: open faces up to ~50 deg hold snowfields above
    // ~1500-1800 m
    float sSh = smoothstep(0.50, 0.64, nyC + 0.05 * nm / 0.12) * smoothstep(1350.0 + fL, 1650.0 + fL, hN + 350.0 * (north - 0.5))
              * smoothstep(-0.2, 0.5, nS + 0.5 * nL) * far * (1.0 - rngB);
    // (the hollow / shoulder fields broken into tongues hanging down the fall line - round blobs read as
    // cut-outs; face-coordinate streaks ~100 m wide, ~300 m long)
    // (session 4: 2:1 instead of 3:1 and weaker - long white drips read as paint runs)
    // (r4 M3: the tongues elongated along strike (~220 m across, ~110 m down the fall line), was 0.008 / 0.0055 = flames)
    float tng = smoothstep(-0.25, 0.20, faceNoise(vec2(0.0045, 0.009), 23.0)   // (session 5: 2:1 -> 1.45:1, "flames")
                                      + 0.12 * (1.0 - smoothstep(4.0, 10.0, fp)) * faceNoise(vec2(0.022, 0.012), 31.0));
    sHo *= mix(0.55, 1.0, tng);
    sSh *= mix(0.45, 1.0, tng);
    // the glaciated far ranges (shot 6): the upper faces and every shoulder white above a ragged
    // limit, glaciers in their hollows; only the steepest rock (> ~65 deg) bare
    float hF = h + 180.0 * nb + 90.0 * nL + 220.0 * (north - 0.5) + 300.0 * saturate(1.4 * curvB);
    // (session 3: 1750-2050 -> 1380-1680 m: the down-valley range tops out at 2300-2800 m, so only its
    // summit tips were white; the upper ~half of every far massif should be under snow and ice)
    // (r4 M8: 1380 / 1680 -> 1750 / 2050: ~30 % patchy snow on the far range - it was a snow mantle)
    float sFar = rngB * smoothstep(1750.0, 2050.0, hF) * smoothstep(0.30, 0.50, nyC + 0.05 * nm / 0.12)
               * smoothstep(-0.9, 0.0, nS + 0.4 * nL);
    // ledges across the steep high faces (the strata bands of the relief pass): snow lines where the
    // footprint-filtered normal is flat on a steep 64 m face (they fade out by themselves once the
    // ledges are below the pixel footprint)
    float sLd = 1.0 * smoothstep(0.70, 0.86, mix(ny, nyC, 0.55)) * (1.0 - smoothstep(0.62, 0.80, nyC)) * far   // (r4 M3: 0.85, 0.76 / 0.90)
              * smoothstep(1250.0 + fL, 1550.0 + fL, hN + 250.0 * (north - 0.5)) * smoothstep(-0.5, 0.4, nL + 0.5 * nS);
    // (session 4: the additive ~200 m / ~1 km terms (0.07 nL + 0.08 nS) are gone - they only gate the terms
    // above: added to sv they left large half-transparent "fog" fields with soft edges)
    // (round 3) melt pattern of the ground itself: facets turned to the sun (against the 64 m surface) melt out
    // first, the shaded backs of the slabs / joints and the lee hollows keep their snow - field margins that
    // follow the micro-topography (noise-cut margins read as torn paper / cartoon blobs, crops c7 / c5). (The
    // relief-tread term of t11 (treads hold, risers shed) is gone: it tore the couloir strips into teeth.)
    float melt = (dot(nC, uSunDir) - dot(n, uSunDir)) * far * (1.0 - smoothstep(6.0, 14.0, fp));
    // (round 3) rocky, ragged margins at 1-9 m when the footprint resolves them (a smooth cartoon outline at 1:1, s5 /
    // crop c5): volume noise, isotropic on the walls
    float vMg = far * (1.0 - smoothstep(1.0, 2.5, fpO));
    // (t19) couloir strips vs open fields: the strips (a narrow cross-gully profile) were torn into teeth and holes by
    // the margin noises (crops c7 / c1: "torn paper"); couloir snow runs as continuous ribbons with
    // smooth, wall-following margins -> the margin noises x(1 - 0.65) on the couloir-only snow. The open fields on
    // gentler ground (s5's cone: a smooth cartoon outline at 1:1) get MORE break-up: a 14 m octave (rocks
    // poking through, lobed melt margins)
    float sOpen = max(max(sBn, sHi), max(sHo, sSh));
    float isC = smoothstep(0.25, 0.6, sC) * (1.0 - smoothstep(0.2, 0.5, sOpen));
    float mgK = 1.0 - 0.65 * isC;
    float gK = smoothstep(0.62, 0.8, nyC) * (1.0 - isC);
    float sv = max(max(max(sC, sBn), max(sHi, sHo)), max(max(sSh, sFar), sLd)) + 0.05 * nm / 0.12
             + 0.3 * melt
             + 0.08 * mgK * vMg * (texture(uNoise3, p * (1.0 / 40.0) + 0.61).r - 0.5) / 0.1
             + 0.09 * gK * far * (1.0 - smoothstep(0.5, 1.2, fpO)) * (texture(uNoise3, p * (1.0 / 14.0) + 0.83).r - 0.5) / 0.1
             + 0.05 * mgK * far * (1.0 - smoothstep(2.5, 6.0, fpO)) * (texture(uNoise3, p * (1.0 / 130.0) + 0.17).r - 0.5) / 0.1
             // (t11) broken, partly melted margins at 2-20 m (volume noise: isotropic on the walls too)
             + 0.04 * mgK * dV * far * (texture(uNoise3, p * (1.0 / 90.0) + 0.37).r - 0.5) / 0.1
             + 0.015 * (1.0 - smoothstep(4.0, 10.0, fp)) * fbm2F(p.xz * 0.033 + 13.0, 3, 30.0, fp) / 0.12;   // (session 3: 0.04 -> 0.015, fewer flecks)
    // (t26) rock steps in the couloirs: the continuous strips ran down whole faces as white paint drips (r10, c1 at
    // 1.5-3 km; gully snow is interrupted where the gully floor steepens into a step). Gaps in altitude bands
    // (~40 m, the band pattern drifting over ~600 m: anisotropic lookup of the noise volume), ~1 in 5 bands, on the
    // couloir-only snow; faded once a band is < ~6 px
    {
        // (the fbm channel's features are ~1/8 of its period: 40 m vertical / 600 m horizontal)
        // (t26 variants: 0.6 = straight horizontal cuts through every strip (man-made); 0.3 + ragged band edges (+-6 m
        // of the 40 m edge noise) kept)
        float gB = texture(uNoise3, vec3(p.x * (1.0 / 4800.0) + 0.13, (h + 6.0 * nE) * (1.0 / 320.0) + 0.41, p.z * (1.0 / 4800.0))).r;
        sv -= 0.30 * isC * smoothstep(0.58, 0.68, gB) * (1.0 - smoothstep(4.0, 7.0, fp));
    }
    // edge width: the inputs of sv are already smooth over >= 3 px (64 m / 3 fp taps, footprint-filtered
    // noise), so a narrow threshold still gives 1-2 px edges (session 3: 0.12 + 0.30 washed the far
    // fields into a grey veil: sv 0.52 gave 50 % snow)
    // (t13: 0.07 + 0.11 -> 0.04 + 0.12 from 1.5-10 m: the fields' soft fringes read as cotton wool at 1:1)
    float ew = (0.04 + 0.12 * smoothstep(1.5, 10.0, fp)) * (1.0 + W2K(18));   // (session 5: 0.10 -> 0.07 near: crisper field edges)
    // (round 3) the couloir floors hold snow ~10 deg steeper than the open faces (the strips broke into holes where the
    // gully steepened)
    float shedC = max(shed, smoothstep(mix(0.30, 0.18, rngB), mix(0.46, 0.36, rngB), nyC + 0.05 * nm / 0.12) * smoothstep(0.2, 0.6, cBig));
    float snowNew = smoothstep(0.52 - ew, 0.52 + ew, sv) * shedC * smoothstep(740.0 - 250.0 * rng, 860.0 - 250.0 * rng, h);
    // (round 2: only the sharp crests / rib tops; 0.05..0.45 bared every shoulder)
    snowNew *= 1.0 - 0.85 * smoothstep(0.15, 0.70, -curv) * far * (1.0 - 0.75 * rngB);   // crests, ribs and spire tips bare (near ranges)

    // previous snow (kept at the camera so nothing near pops)
    float snowOld = smoothstep(820.0, 1150.0, h + 240.0 * n1 + 60.0 * n2) * smoothstep(0.35, 0.6, ny + 0.2 * n2);
    snowOld *= mix(0.25, 1.0, saturate(n.z * 1.2 + 0.4)) * mix(0.35, 1.0, 1.0 - smoothstep(0.55, 0.85, ao));
    snowOld = smoothstep(0.18, 0.42, snowOld + 0.12 * n3);
    snowM = mix(snowOld, snowNew, far);
    // (session 3) thin, dirty snow at the rims of the fields, clean in their cores: the uniform white cut-outs
    // read as pasted on (grey margins, dust streaks down the fall line)
    float sCore = smoothstep(0.0, 0.28, sv - 0.52);
    float sDust = (1.0 - smoothstep(8.0, 25.0, fp)) * faceNoise(vec2(0.05, 0.012), 11.3);
    // (session 5: the dust streaks and the 0.80 rims read as brush strokes / smears at 1:1 final pixels)
    snow *= mix(1.0, mix(0.88, 1.0, sCore), far);

    if (uDebug == 21) scr = 0.0;
    if (uDebug == 23) { veg = 0.0; scr = 0.0; snowM = 0.0; }
    vec3 alb = rock;
    // fans have a body: thin debris over rock at the margins, the pale core along the middle (session 6:
    // flat cut-outs of one tone read as camouflage blotches at 1-2 km)
    alb = mix(alb, scree * mix(0.86, 1.04, smoothstep(0.40, 0.80, f.b)), scr);
    alb = mix(alb, vec3(0.034, 0.044, 0.022) * (0.8 + 0.5 * saturate(0.5 + n2ff)), sShrub * scr * (1.0 - f.a));   // (t11) shrubs on the old debris
    vec3 nv = n;
    // (t19) turf covers the slab relief: on the relief's turf the facet normal (V-joints in the sun's shadow =
    // crisp black dashes all over the grassy 30-45 deg slopes, crop c7 lower left) gives way to the base surface;
    // the rock between the turf keeps its facets
    float tvW = saturate(rkF * 0.8);
    if (tvW > 0.01 && veg > 0.01) {
        float eb = max(3.0, 1.5 * fp);
        vec3 nB = normalize(vec3(terrainH(p.xz - vec2(eb, 0.0), true) - terrainH(p.xz + vec2(eb, 0.0), true), 2.0 * eb,
                                 terrainH(p.xz - vec2(0.0, eb), true) - terrainH(p.xz + vec2(0.0, eb), true)));
        nv = normalize(mix(n, nB, tvW));
    }
    gVegS = veg * tvW;
    vec3 vegAlb = groundVegAlbedo(p, nv, t, f, ao);
    // dark dwarf-pine / alpenrose patches through the open belt above the forest
    // (session 4: a bigger share of the 320-1100 m belt (-0.2 -> -0.45), and the big gullies hold it)
    float dwarf = smoothstep(-0.45, 0.45, (nm * 0.8 + 0.4 * n2ff - 0.25 * nb) / 0.12 + 1.5 * gulG + 0.5 + W2K(17)) * smoothstep(320.0, 520.0, hvV) * (1.0 - smoothstep(900.0, 1100.0, hvV));
    dwarf *= far * (1.0 - f.a) * (1.0 - saturate(f.r * 2.0));
    dwarf = mix(dwarf, smoothstep(0.40, 0.60, dwarf + kD * nD2) * far * (1.0 - f.a) * (1.0 - saturate(f.r * 2.0)), dV);   // (t11) clumps with crisp edges, not soft blobs
    vegAlb = mix(vegAlb, vec3(0.040, 0.056, 0.026) * (0.8 + 0.4 * saturate(0.5 + n2ff)), dwarf * 0.8);
    // far alpine turf reads darker and more olive than the lit meadow up close (tussocks, shrubs, rock)
    // (session 4: 0.65 -> 0.5, the far turf read as bright olive patches)
    vegAlb *= mix(1.0, 0.55 + W2K(16), smoothstep(80.0, 700.0, t) * (1.0 - f.a));
    // (t12) the far turf is not smooth paint: tussocks, dwarf shrubs, stones and bare soil at 2-9 m (cells and
    // the mosaic fbm, faded with the footprint)
    if (dV * far > 0.0) {
        float tg = dV * far * (1.0 - f.a);
        vec3 vt = voronoi2(rot2(2.1) * p.xz * (1.0 / 3.2) + 5.0);
        float cellT = (fract(vt.z * 9.7) - 0.5) * (1.0 - smoothstep(0.7, 1.4, fp));
        vegAlb *= 1.0 + tg * (0.16 * clamp(0.6 * nD2 - 0.4 * nD, -1.5, 1.5) + 0.35 * cellT);
    }
    alb = mix(alb, vegAlb, veg);
    // (session 6) the steep faces of the forest belt are not bare monoliths (t21 shot 0 left face: a smooth
    // triangular slab in the forest): irregular turf / dwarf-shrub patches, elongated along the face (ledges),
    // ~10-30 m, clustered with the ~50 m outcrop noise, more on the ledgier faces. (t23: thin strips + joint
    // shrubs read as drawn dashes / glyphs at 1:1 final pixels.) Footprint-faded fbm: no shimmer
    float lvW = far * (1.0 - veg) * (1.0 - scr) * (1.0 - smoothstep(850.0, 1250.0, hv)) * smoothstep(0.20, 0.42, nyC)
              * (1.0 - smoothstep(0.86, 0.95, nyC)) * smoothstep(60.0, 140.0, h);
    if (lvW > 0.0) {
        float lp = faceFbm(vec2(0.035, 0.10), 3, 28.0, fp, 61.0) / 0.15
                 + 0.6 * clamp(nm / 0.12, -1.5, 1.5) + 1.2 * (nyC - 0.55);
        float m = smoothstep(0.55, 0.95, lp + 0.6 * W2K(26));
        // (world3) on the relief walls: turf / shrubs on the ledge treads (flat fine normal), patchy
        m = mix(m, smoothstep(0.55, 0.85, ny) * smoothstep(0.0, 0.6, lp + 0.6), rkF);
        vec3 lgc = mix(vec3(0.034, 0.046, 0.024), vec3(0.080, 0.084, 0.042), saturate(0.5 + n2ff + 0.5 * n3f));
        alb = mix(alb, lgc, m * lvW);
    }
    float rockW = (1.0 - veg) * (1.0 - scr);
    // (session 4) on the walls rockNormal's joint-panel tilts (47 / 71 m warped panels) hand over to the
    // face facets below with the albedo (polygon shading 0.2-2 km out)
    float rkNW = rockW * (1.0 - wallF * smoothstep(150.0, 400.0, t));
    n = normalize(mix(mix(n, nv, veg), rkNW > 0.0 ? rockNormal(p, n, t) : n, rkNW));
    // far away (where rockNormal has no relief left): ribs / gullies / ledges below the heightfield
    // resolution, so the faces shade like fractured granite instead of smooth clay
    float wB = smoothstep(70.0, 180.0, t) * (rockW + 0.35 * scr * (1.0 - veg)) * (1.0 - 0.85 * snowM);
    if (wB > 0.01) {
        // the 4 m heightfield's erosion rills read as a fine diagonal hatch on the far faces: pull the
        // shading part-way to the coarse (>= 64 m) normal, then add the facets / ribs below it
        // (session 3: 0.55 -> 0.4; with the fins of the relief pass gone the facets read crisper)
        // (t25 variants: no pull / no faceBump or x1.6 - all ~ the same at 1:1; kept)
        n = normalize(mix(n, nC, 0.3 * wB * smoothstep(150.0, 500.0, t) * (1.0 - rkF)));   // (session 5: 0.4 -> 0.3, more of the real relief)
        n = normalize(n - wB * (gb - dot(gb, n) * n));
        float wF = wB * (1.0 - 0.65 * scr) * (1.0 - rkF);
        n = normalize(n + wF * (gF - dot(gF, n) * n));
    }
    // (round 3) slabs / knobs where the rock shows through the turf on gentle ground (no slab relief there)
    float wO = rockW * far * smoothstep(0.52, 0.70, nyC) * (1.0 - rkF) * (1.0 - snowM) * (1.0 - smoothstep(1300.0, 1600.0, h));
    if (wO > 0.01) {
        float ocr;
        vec3 go = outcropSlabs(p.xz, fpO, ocr);
        n = normalize(n - wO * (go - dot(go, n) * n));
        alb *= mix(1.0, 1.0 - 0.35 * ocr, wO);
    }
    // (t13) block faces of the debris (scree section): lit sun sides, shaded lee sides
    float wSB = scr * (1.0 - veg) * (1.0 - snowM);
    if (wSB > 0.01) n = normalize(n + wSB * (scrB - dot(scrB, n) * n));
    // snow fills the crevices: its surface follows the coarse ground shape, not the crag relief
    // (also keeps the bright patches from shimmering on the spires)
    // (session 4: 0.6 -> 0.3: the fields take their shading from the ground under them, not pasted paper)
    // (t11) on the slab-relief walls the snow fills the joints: its surface follows the base heightfield, not the
    // facets (facet shading and V-joint shadows inside the fields read as torn paper, t10 crop c7)
    if (snowM > 0.01 && rkF > 0.01) {
        float eb = max(8.0, 2.0 * fp);   // (t12: 1.5 m taps kept the base map's 4 m rills - vertical streaks on the snow)
        vec3 nB = normalize(vec3(terrainH(p.xz - vec2(eb, 0.0), true) - terrainH(p.xz + vec2(eb, 0.0), true), 2.0 * eb,
                                 terrainH(p.xz - vec2(0.0, eb), true) - terrainH(p.xz + vec2(0.0, eb), true)));
        n = normalize(mix(n, nB, snowM * rkF * 0.6));
    }
    n = normalize(mix(n, nC, (0.45 + W2K(10)) * snowM * far * (1.0 - 0.7 * rkF)));   // (session 5: 0.3 -> 0.45: the 4-10 m rills streaked the fields like flames)
    float shore = (1.0 - smoothstep(0.35, 1.3, above + 0.4 * n3)) * (1.0 - 0.7 * saturate(f.a * 1.5));
    // (veg4 W4 / W11 / G13) shore band: (a) the waterline itself is stone / wet gravel within ~0.25 m above the
    // water whatever the meadow weight (meadow paint reached the water on the flats: lime "rafts" in r1 /
    // shot01), not on banks steeper than ~10-15 deg; (b) under the forest a narrow dark wet band (roots,
    // dark stones; no pale pebble kerb along the wooded shores); (c) on the grassy shore stretches (the 3D
    // cover's gcShoreGrass) the sward reaches down to ~0.3 m with a noise-broken sparse-grass / gravel
    // ecotone, so the far shore beyond the 3D cover agrees with the near one
    vec3 pebW = pebble;
    if (above < 1.8) {
        float eco = (vnoise(p.xz * 1.3 + 7.0) - 0.5) * (1.0 - smoothstep(0.25, 0.8, fp));
        float wEdge = (1.0 - smoothstep(0.16, 0.32, above + 0.10 * n3 + 0.12 * eco)) * smoothstep(0.955, 0.985, ny);
        float fo = smoothstep(0.3, 0.55, f.r) * (1.0 - 0.6 * saturate(f.a * 1.5));
        float sgS = t < 900.0 ? gcShoreGrass(p.xz) * (1.0 - fo) : 0.0;
        float shN = 1.0 - smoothstep(0.15, 0.5 + 0.25 * eco, above + 0.12 * n3);       // narrow band
        shore = mix(shore, shN, max(fo, 0.75 * sgS));
        shore = max(shore, wEdge);
        pebW = mix(pebble, pebble * vec3(0.50, 0.50, 0.44), fo);                     // wet dark stones under the trees
    }
    alb = mix(alb, pebW, shore);
    alb = mix(alb, snow, snowM);

    // rounded pebbles up close on shores
    if (shore > 0.01 && t < 40.0) {
        vec3 bumpN;
        vec3 pc = pebbles(p.xz, bumpN);
        float w = shore * (1.0 - smoothstep(22.0, 40.0, t));
        alb = mix(alb, pc, w);
        n = normalize(n + bumpN * w * (1.0 - smoothstep(8.0, 25.0, t)));
    }
    // wet band at the waterline, darker submerged stones
    wetness = 1.0 - smoothstep(0.0, 0.35 + 0.15 * n3, above);
    alb *= mix(1.0, 0.55, wetness);
    spec = mix(0.02, 0.25, wetness) + 0.2 * snowM;
    // look-dev views (uDebug): 21 no scree, 22 black (aerial in-scatter only), 23 bare rock,
    // 24 material ids (red scree, green vegetation, white snow), 27 snow terms (final, couloir, bench),
    // 26 (couloir curvature, summit fields, effective snow height), 29 = 24 inside the 350-700 m belt only, 28 grey albedo (shading
    // only), 44 grey albedo + unperturbed normal (heightfield shading), 46 facet cracks / tone / plane blend
    if (uDebug == 22) { spec = 0.0; return vec3(0.0); }
    if (uDebug == 49) { spec = 0.0; return vec3(rkF, gVegS, scr) * 0.5; }   // (world3) relief weight / turf on relief / scree
    if (uDebug == 46) { spec = 0.0; return vec3(crack, ftone + 0.5, gFwb) * 0.4; }
    if (uDebug == 28) { spec = 0.0; return vec3(0.18); }
    if (uDebug == 27) { spec = 0.0; return vec3(snowNew, sC, sBn) * 0.5; }
    if (uDebug == 47) { spec = 0.0; return vec3(sHo, sSh, sFar) * 0.5; }
    if (uDebug == 48) { spec = 0.0; return vec3(gulG, saturate(curvG * 2.0), wallF) * 0.5; }
    if (uDebug == 26) { spec = 0.0; return vec3(saturate(cBig), sHi, saturate((hN - 500.0) / 2500.0)) * 0.5; }
    if (uDebug == 29) { spec = 0.0; if (h < 350.0 || h > 700.0 || f.r > 0.25) return vec3(0.0); return mix(mix(mix(vec3(0.12), vec3(0.35, 0.02, 0.02), scr), vec3(0.03, 0.15, 0.03), veg * (1.0 - scr)), vec3(0.45, 0.02, 0.5), snowM); }
    if (uDebug == 24) { spec = 0.0; return mix(mix(mix(vec3(0.12), vec3(0.35, 0.02, 0.02), scr), vec3(0.03, 0.15, 0.03), veg * (1.0 - scr)), vec3(0.45, 0.02, 0.5), snowM); }
    return alb;
}


float gFallsMoss = 0.0;      // falls lab debug 13
vec3 shadeTerrain(vec3 p, vec3 rd, float t, bool cheap) {
    gFallsMoss = 0.0;
    gRkFine = true;                                            // (world3) all relief octaves for the shading
    float det = detailWeight(t);
    float e = max(0.03, t * gPixAng * 1.2);
    vec3 n = terrainNormal(p.xz, e, det);
    // (world3) joint-block relief at this point: its weight steers the shading, its depth lifts the point
    // back to the base surface for the baked shadow maps
    rockRelief(p.xz, terrainH(p.xz, true));
    float rkW = gRkW, rkD = gRkD;
    gRkWS = rkW; gRkDS = rkD;
    // (world5 R4) the slab facets' share of the shading normal x(1 + kB) beyond ~0.4-1.5 km (shading only, the march
    // keeps the relief as it is): the 42 / 15 m facets were faded to soft undulations there (r4 M10 kF / T), the walls
    // read as smeared plaster at 1:1 - the faces should show crisp lit / shaded planes at 1-5 km. Relief gradient =
    // the normal's gradient minus the base map's (the same taps as terrainNormal's base part)
    {
        float kB = (1.5 + W2K(31)) * smoothstep(400.0, 1500.0, t) * rkW;   // (sw2 / sw4: 1.5)
        if (kB > 0.0 && !cheap) {
            float eBb = mix(e, max(e, 3.0), nearBlend(p.xz));
            vec2 gB = vec2(terrainHN(p.xz + vec2(eBb, 0.0)) - terrainHN(p.xz - vec2(eBb, 0.0)),
                           terrainHN(p.xz + vec2(0.0, eBb)) - terrainHN(p.xz - vec2(0.0, eBb))) / (2.0 * eBb);
            vec2 gT = -n.xz / max(n.y, 0.05);
            vec2 gR = gT - gB;
            n = normalize(vec3(-(gB.x + (1.0 + kB) * gR.x), 1.0, -(gB.y + (1.0 + kB) * gR.y)));
        }
    }
    // (world5 R5) the finer slab-facet octaves (15 / 5.4 / 1.9 m, world.glsl rockRelief) in the SHADING normal down to
    // ~2.2-4.5 px per cell (the march fades them at 4.2-8.5 px): crisp lit / shaded facets on the 0.3-4 km walls, which
    // kept only the 42 m (+15 m) facets - soft folds at 1:1. Only the part the march lacks (fade difference), same joint
    // families / tilts / seeds as rockRelief, gradient by central differences along the surface over ~0.7 px
    {
        float fpx = t * gPixAng;
        float kR = (1.8 + W2K(8)) * rkW * smoothstep(250.0, 600.0, t) * (1.0 - smoothstep(5000.0, 7000.0, t));   // (sw4: 1.8)
        if (kR > 0.0 && !cheap) {
            vec4 nz = texture(uNoise3, vec3(p.xz * (1.0 / 4096.0), 0.37));
            float ang = 0.5 + 2.4 * (nz.g - 0.5);
            vec3 F0 = vec3(cos(ang), 0.25, sin(ang)), F1 = vec3(-sin(ang), -0.2, cos(ang)), F2 = vec3(0.3 * cos(ang), 1.0, 0.3 * sin(ang));
            vec2 gN = -n.xz / max(n.y, 0.05);
            float slN = length(gN);
            float T = mix(0.8, 0.55, smoothstep(800.0, 2000.0, t)) * mix(0.35, 1.0, smoothstep(0.75, 1.2, slN));
            float kF = mix(1.0, 0.45, smoothstep(700.0, 2000.0, t));
            float es = max(0.7 * fpx, 0.05);
            vec3 P = vec3(p.x, terrainH(p.xz, true), p.z);
            vec3 dX = vec3(es, gN.x * es, 0.0), dZ = vec3(0.0, gN.y * es, es);
            vec2 gE = vec2(0.0);
            float C = 15.12, amp = 0.7 * kF;
            for (int k = 1; k < 4; k++) {
                float wk = smoothstep(3.0 * fpx, 6.0 * fpx, C) - smoothstep(4.2 * fpx, 8.5 * fpx, C);   // (shimmer test: 2.2 / 4.5 -> 3 / 6 px)
                if (wk > 0.001) {
                    vec3 Q = P + 37.0 * float(k);
                    uint sd = uint(5 + 13 * k);
                    float ex1 = rkFacets3(Q + dX, C, T, fpx, F0, F1, F2, sd), ex0 = rkFacets3(Q - dX, C, T, fpx, F0, F1, F2, sd);
                    float ez1 = rkFacets3(Q + dZ, C, T, fpx, F0, F1, F2, sd), ez0 = rkFacets3(Q - dZ, C, T, fpx, F0, F1, F2, sd);
                    gE += wk * amp * vec2(ex1 - ex0, ez1 - ez0) / (2.0 * es);
                }
                C *= 0.36; amp *= 0.7;
            }
            float sec = min(sqrt(1.0 + slN * slN), 3.0);
            gE *= kR * mix(1.0, sec, 0.5);
            n = normalize(vec3(-(gN.x + gE.x), 1.0, -(gN.y + gE.y)));
        }
    }
    vec4 f = forestAt(p.xz);
    float ao = terrainAO(p.xz);
    float wl = waterLevel(p.xz);
    float wet, snowM, spec;
    vec3 n0 = n;
    vec3 alb = terrainAlbedo(p, n, t, f, wl, ao, wet, snowM, spec);
    // (world3) cavity of the slab relief: its height against the mean of 8 taps 2-5 px around (a Laplacian of
    // D, the base slope cancels): the V-joints between the slabs and the recesses hold moisture, lichen,
    // debris and shade - dark crisp joint lines and pockets; the slab edges stay clean and catch the light.
    // Carries the structure in frontal light (shot 0's 1.3 km face: aspect = sun azimuth), where facet
    // tilts barely change the shading
    if (rkW > 0.01 && !cheap) {
        float fpx = t * gPixAng;
        float ec = max(1.5, 2.0 * fpx);
        // (t20, perf: 8 taps at 1 / 2.3 ec -> 4 taps at 1.6 ec, same look in the 1:1 crops)
        float s4 = 0.0;
        for (int i = 0; i < 4; i++) {
            float a = float(i) * 1.5708 + 0.3;
            vec2 q = p.xz + vec2(cos(a), sin(a)) * (1.6 * ec);
            s4 += rockRelief(q, terrainH(q, true));
        }
        float cav = rkD - s4 * 0.25;
        float dk = smoothstep(0.15, 2.5, -cav) * rkW * (1.0 - snowM) * (1.0 - gVegS);
        float br = smoothstep(0.15, 2.5, cav) * rkW * (1.0 - snowM);
        // (round 3: 0.45 -> 0.35, x0.6 once the taps are > ~3 m apart: at frame scale the 6-14 m recesses darkened
        // into blotches on top of the facet shading - s7 final vs its grey-albedo view)
        // (world5 R2: 0.35 / x0.6 -> 0.60 / x0.90 (sw4; lab knobs 29 / 24): crisp dark joints between the slabs at 0.5-3 km)
        alb *= (1.0 - (0.60 + W2K(29)) * mix(1.0, 0.90 + W2K(24), smoothstep(1.5, 4.0, fpx)) * dk) * (1.0 + 0.14 * br);
        gRkWS = rkW; gRkDS = rkD;                                   // (rockRelief overwrote the globals)
    }
    // dusty forest trail with roots and needles
    // (veg4 G4) a narrow (~0.7-1.5 m) compacted grey-brown dirt path (#797169, R-B ~16, luma p10 / p90
    // ~55 / 227 with the dappled shade) with exposed roots crossing it every 0.5-1.5 m, embedded stones and
    // needle litter; it was a 2-4 m smooth pinkish sand road (#aa8f70, R-B 58) that read as a beach
    if (uTrailN > 1 && t < 400.0) {
        float td = 1e9;
        vec2 tdir = vec2(1.0, 0.0);
        for (int i = 0; i < 31; i++) {
            if (i + 1 >= uTrailN) break;
            vec2 a2 = uTrail[i].xy, b2 = uTrail[i + 1].xy, ab2 = b2 - a2;
            float tt = clamp(dot(p.xz - a2, ab2) / dot(ab2, ab2), 0.0, 1.0);
            float dd = length(p.xz - (a2 + ab2 * tt));
            if (dd < td) { td = dd; tdir = ab2; }
        }
        float tw = 1.0 - smoothstep(0.35, 0.75 + 0.25 * gnoise(p.xz * 0.7), td);
        if (tw > 0.0) {
            float fpT = max(t * gPixAng, 0.002);
            float g = texture(uNoise3, vec3(p.xz * 0.9, 0.4)).r;
            float g3 = mix(texture(uNoise3, vec3(p.xz * 2.6, 0.63)).g, 0.5, smoothstep(0.01, 0.03, fpT));   // ~3 cm grain
            vec3 dirt = mix(vec3(0.10, 0.085, 0.065), vec3(0.30, 0.26, 0.20), saturate((g - 0.5) * 1.6 + 0.5 + 0.5 * (g3 - 0.5)));
            // trampled centre a little paler and smoother, the edges darker with litter
            dirt *= mix(0.72, 1.06, 1.0 - smoothstep(0.1, 0.6, td));
            // exposed roots: borders of Voronoi cells stretched across the path (~0.9 m along, ~3 m across), 4-8 cm
            vec2 tu = normalize(tdir), tv = vec2(-tu.y, tu.x);
            vec2 qr = vec2(dot(p.xz, tu), dot(p.xz, tv));
            vec3 vr = voronoi2(qr * vec2(1.1, 0.33) + 3.7);
            float rw = 0.020 + 0.018 * fract(vr.z * 13.1);                  // half width (m)
            float dB = 0.5 * (vr.y - vr.x) / 1.1;                            // ~ distance to the cell border (m)
            float fwR = max(fpT, 0.006);
            float root = (1.0 - smoothstep(rw - fwR, rw + fwR, dB)) * step(0.35, fract(vr.z * 7.3)) * min(1.0, 2.0 * rw / fwR);
            vec3 rootC = mix(vec3(0.06, 0.05, 0.04), vec3(0.15, 0.12, 0.09), saturate(1.0 - dB / max(rw, 1e-3)) * 0.6);   // lit crown of the root
            dirt = mix(dirt, rootC, root * 0.9);
            // embedded stones (~30 %), grey, 4-12 cm; needle specks (both fade to their mean when sub-pixel)
            vec4 vs = voronoiV(p.xz * 7.0 + 1.3);
            float sR = 0.38 + 0.25 * fract(vs.w * 5.3);
            float sw = max(fpT * 7.0, 0.04);
            float stn = (1.0 - smoothstep(sR - sw, sR + sw, vs.z)) * step(0.62, fract(vs.w * 3.1)) * (1.0 - root);
            dirt = mix(dirt, vec3(0.22, 0.215, 0.20) * mix(0.7, 1.2, fract(vs.w * 9.7)), stn * 0.85);
            float nd = mix(texture(uNoise3, vec3(p.xz * 6.0, 0.21)).a, 0.5, smoothstep(0.004, 0.012, fpT));
            dirt = mix(dirt, vec3(0.11, 0.065, 0.03), smoothstep(0.62, 0.8, nd) * 0.5 * (1.0 - stn));
            alb = mix(alb, dirt, tw);
        }
    }
    // the waterfall's rock step (falls.glsl wfRockShade): dark wet jointed rock, moss / fern tufts. The
    // rock material is evaluated ONCE for the step (fw) and the bank outcrops (outc) - (round 2, perf: two
    // inlined copies x two shadeTerrain call sites); reflections (cheap) get its flat mean (wfRockCheap)
    float fw = wfOff(2) ? 0.0 : fallsWetness(p, n0);
    float dpA, awA;
    // (review rv5) the foot of the cliff / banks at the waterline: dark wet rock into the water, not the lake
    // shore's pale pebble band - a pale strip ~1 m tall ran along the whole cliff foot (film shot 2). Where
    // the falls rock / bank zone stands just above (y 2 m), its rock reaches down into the water (not on the apron)
    if (fw < 1.0 && !wfOff(2) && p.y < 1.6) {
        float apr = fallsApron(p, dpA, awA);
        vec3 pu = vec3(p.x, 2.0, p.z);
        float zf = max(fallsWetness(pu), fallsBank(pu)) * (1.0 - apr);
        fw = max(fw, zf * (1.0 - smoothstep(0.7, 1.5, p.y)));
    }
    float fa = fw < 1.0 && !wfOff(2) ? fallsApron(p, dpA, awA) * (1.0 - fw) : 0.0;
    float fb = fw < 1.0 && !wfOff(2) ? fallsBank(p) * (1.0 - fw) * (1.0 - fa) : 0.0;
    // dark rock outcrops (with their own moss / shrubs) where the bank is steepest; on the bank next to
    // the falls only low down (ferns and shrubs above the dark buttress)
    float outc = fb > 0.0 ? fb * smoothstep(0.70, 0.56, n0.y + 0.25 * (texture(uNoise3, p * 0.013 + 0.6).r - 0.5)
                                            + 0.18 * fallsLushBank(p)) : 0.0;
    vec3 ra = vec3(0.0), nR = n;
    float mossR = 0.0, wetR = 0.0, occR = 1.0;
    if (max(fw, outc) > 0.0) {
        if (cheap) ra = wfRockCheap(p, n0, mossR);
        else ra = wfRockShade(p, n0, t, nR, mossR, wetR, occR);
    }
    if (fw > 0.0) {
        gFallsMoss = mossR;
        alb = mix(alb, ra, fw);
        n = normalize(mix(n, nR, fw));
        ao *= mix(1.0, occR, fw);
        spec = max(spec, max(1.33 * fw * wetR, 0.12 * fw * mossR));     // (r3c: leaf sheen on the face's cover)
        wet = max(wet, fw * wetR);
    }
    // the angular shore slabs on the apron north of the falls
    if (fa > 0.0) {
        float occA, wetA;
        vec3 aa = wfApronShade(p, n0, t, dpA, awA, fa, n, occA, wetA);
        alb = mix(alb, aa, fa);
        ao *= mix(1.0, occA, fa);
        spec = max(spec, 0.4 * fa * wetA);
        wet = max(wet, fa * wetA);
    }
    // the inflow stream's banks and its mouth delta (meadow grass right up to the boulders). The
    // stream's level map makes terrainAlbedo's lake-shore rule paint a pebble beach along the whole
    // stream: back to grass wherever the ground stands > 0.15 m above the water (falls lab)
    if (uStreamN > 1 && t < 900.0 && !wfOff(3)) {
        vec2 sdb = streamDist(p.xz);
        float hwc = 0.45 * sdb.y;                                // wet channel half width (bake: 0.35-0.5 x)
        float dmo = dot(uStreamMouth.zw, uStreamMouth.zw) > 0.5 ? length(p.xz - uStreamMouth.xy) : 1e9;
        float gw = max(smoothstep(1.0, 1.5, sdb.x / hwc) * (1.0 - smoothstep(3.5, 4.0, sdb.x / hwc)), 1.0 - smoothstep(30.0, 44.0, dmo));
        float abv = wl > -1000.0 ? p.y - wl : 99.0;
        // (grass also on the cut banks of the delta: no pale scree walls either side of the mouth)
        float dlt = 1.0 - smoothstep(30.0, 44.0, dmo);
        gw *= smoothstep(0.15, 0.35, abv) * smoothstep(mix(0.55, 0.30, dlt), mix(0.75, 0.55, dlt), n0.y) * (1.0 - snowM);
        if (gw > 0.01) {
            vec3 ng = n0;
            vec3 ga = groundVegAlbedo(p, ng, t, f, ao);
            alb = mix(alb, ga, gw);
            n = normalize(mix(n, ng, gw));
            wet *= 1.0 - gw;
            spec = mix(spec, 0.02, gw);
        }
        // (review r2, fix 9) beyond the near map (z > ~1800 m) the stream is painted onto the ground from its
        // polyline - the 4 m far map's channel holds almost no water surface (its level lies under the far
        // terrain over most of the channel): white water tumbling between boulders, dark wet rock at its
        // edges, pale granite boulders on the banks (film shot 4: a white cascade from 150-500 m)
        float nbF = nearBlend(p.xz);
        if (nbF > 0.0 && sdb.x < 3.0 * sdb.y) {
            float fpS = t * gPixAng;
            vec2 fdS = gStreamDir.xy, fxS = vec2(-fdS.y, fdS.x);
            vec2 pa = vec2(dot(p.xz, fdS), dot(p.xz, fxS));
            float hwW = 0.48 * sdb.y;
            float eN = 0.30 * hwW * gnoise(pa * vec2(0.07, 0.25) + 1.7);
            float wmS = 1.0 - smoothstep(0.8 * hwW - fpS, hwW + fpS, sdb.x + eN);
            float wtS = 1.0 - smoothstep(hwW, 1.5 * hwW + fpS, sdb.x + eN);
            float vS = 1.0 - smoothstep(0.15, 0.6, fpS);
            float str = mix(0.55, texture(uNoise3, vec3(pa.x * 0.05, pa.y * 0.35, 0.83)).r + 0.3 * (texture(uNoise3, vec3(pa.x * 0.02, pa.y * 0.12, 0.29)).g - 0.5), vS);
            // boulders (~1.5-2.5 m cells): wet dark heads in the white water, pale granite on the banks
            vec4 bc = wfVor(pa * vec2(0.42, 0.55) + 31.0);
            float bR = 0.30 + 0.16 * fract(bc.w * 5.3);
            float inW = wmS;
            float hasB = step(fract(bc.w * 13.7), mix(0.45, 0.28, inW)) * (1.0 - smoothstep(1.4 * hwW, 2.6 * hwW, sdb.x));
            float bvis = 1.0 - smoothstep(0.4, 1.2, fpS);
            float boul = hasB * mix(0.35, 1.0 - smoothstep(bR - 0.5 * fpS, bR + 0.5 * fpS, length(bc.xy)), bvis);
            float steepF = smoothstep(0.05, 0.14, gStreamDir.z);
            vec3 wwA = vec3(0.80, 0.82, 0.83) * mix(0.68, 1.05, smoothstep(0.32, 0.66, str));
            vec3 boA = mix(vec3(0.075, 0.074, 0.068), vec3(0.30, 0.292, 0.272), 1.0 - inW) * mix(0.8, 1.2, fract(bc.w * 3.7));
            float whiteW = wmS * mix(0.55, 1.0, steepF) * smoothstep(0.25, 0.45, str + 0.25 * steepF) * (1.0 - boul);
            vec3 sa = mix(mix(alb * mix(1.0, 0.45, wtS), boA, boul), wwA, whiteW);
            float wS = nbF * max(wtS, boul * (1.0 - smoothstep(1.6 * hwW, 2.6 * hwW, sdb.x)));
            alb = mix(alb, sa, wS);
            n = normalize(mix(n, normalize(n0 * 0.5 + vec3(0.0, 1.0, 0.0) + vec3(fdS.x, 0.0, fdS.y) * (str - 0.5) * 0.9), nbF * wmS * (1.0 - boul)));
            spec = mix(spec, 0.3, nbF * wtS * (1.0 - whiteW));
            wet = max(wet, nbF * wtS * (1.0 - whiteW));
        }
    }
    // boulders of the inflow stream bed and its floodplain (fade out with the footprint)
    {
        vec3 ba; float occB;
        float fbo = streamBoulder(p, n0, t, n, ba, occB);
        if (fbo > 0.0) {
            float wetS = 1.0 - smoothstep(0.0, 0.25, p.y - wl);
            alb = mix(alb, ba * mix(1.0, 0.5, wetS), fbo);
            ao *= mix(1.0, occB, fbo);
            wet = max(wet, wetS * fbo);
            spec = max(spec, 0.3 * wetS * fbo);
        }
    }
    // ... and the forested banks / rim around it: shrubs, ferns, grass and needle litter, no bare rock
    if (fb > 0.0) {
        vec3 nv; float clv, occV;
        vec3 va = wfVegetation(p, n0, t * gPixAng, 0.0, nv, clv, occV);
        float lit = texture(uNoise3, p * 0.02 + 0.3).r;
        float floorW = smoothstep(0.2, 0.5, f.r + 0.5 * (lit - 0.5)) * (1.0 - smoothstep(0.45, 0.75, clv));
        float lush = fallsLushBank(p);
        floorW *= 1.0 - 0.7 * lush;
        va = mix(va * mix(0.62, 0.95, lush), vec3(0.052, 0.050, 0.032) * mix(0.7, 1.25, lit), floorW);
        // the cliff's rim: a dark mass of shrubs / young spruce, few lime tips
        float rim = fallsRim(p);
        va = mix(va, mix(vec3(0.016, 0.026, 0.014), va, 0.15 + 0.2 * smoothstep(0.7, 0.9, clv)), rim);
        alb = mix(alb, va, fb);
        n = normalize(mix(n, nv, fb * (1.0 - 0.5 * floorW)));
        ao *= mix(1.0, occV, fb * (1.0 - floorW));
        if (outc > 0.0) {
            alb = mix(alb, ra, outc);
            n = normalize(mix(n, nR, outc));
            ao *= mix(1.0, occR, outc);
        }
    }
    // shadows of the falls flora (3D shrubs / ferns, falls.glsl ffShadow) on the rock, banks and apron
    if (fw + fb + fa > 0.01 && !cheap) {
        float ffs = ffShadow(p, t);
        if (ffs > 0.0) { n = normalize(n - uSunDir * max(dot(n, uSunDir), 0.0) * ffs); ao *= 1.0 - 0.35 * ffs; }
    }
    // forest floor / understory darkness
    float canopy = saturate(f.r * 1.2);
    float occ = ao * mix(1.0, mix(0.55, 0.45, smoothstep(80.0, 200.0, t)), canopy * (1.0 - snowM));   // (veg4 G14: near floor 0.45 -> 0.55, target shaded floor L 0.32 vs 0.24)
    occ *= gcGroundSkyOcc;
    // (session 6) far open faces: the heightfield AO under-counts the sky a 1-5 km face sees (distant
    // shade is sky-blue at only ~1.7x below the lit faces); less of the warm valley fill there too
    // (session 4: floor 0.8 -> 0.65 - it erased the AO of the gullies / hollows far away: soft "wax" faces)
    occ = mix(occ, max(occ, 0.5 + W2K(11)), smoothstep(500.0, 2000.0, t) * (1.0 - canopy));   // (session 5: 0.65 -> 0.5: shot 6's shaded face read as one flat plane)
    // (world3) recesses of the joint-block relief (chimneys, the backs of the ledges) see less sky
    occ *= 1.0 - 0.2 * saturate(-rkD / 14.0) * (1.0 - snowM);   // (t11: snow fills them)
    float sun = terrainShadow(p - vec3(0.0, rkD, 0.0) + n * 0.3, cheap ? 40 : 90) * cloudShadow(p);
    // (perf: the relief shadow marches the three coarse octaves only; the 1.9 m one casts sub-pixel shadows)
    gRkFine = false;
    // (r4 M10: floor 0.3 -> 0.6 at 0.8-2.5 km: the hard lit / shaded slivers of every slab crumpled the 1-3 km faces like foil)
    // (world5 R3) relief-shadow floor 0.3 / 0.6 -> 0.13 / 0.33 (sw4; lab knob 30): crisper shaded facets at 1-3 km
    if ((rkW > 0.0 || rkD < -0.02) && sun > 0.0 && !cheap) sun *= mix(mix(max(mix(0.13, 0.33, smoothstep(800.0, 2500.0, t)) - W2K(30), 0.0), 1.0, max(0.85 * snowM, 0.8 * gVegS)), 1.0, rockReliefShadow(p + n * 0.05, t * gPixAng));
    if (!cheap && t < 700.0 && sun > 0.0) sun *= mix(treeShadow(p + n * 0.2), mix(1.0, 0.45, canopy), smoothstep(450.0, 700.0, t));
    else sun *= mix(1.0, 0.45, canopy);
    // (t21) light reflected onto shaded snow by the sunlit snow / rock around it (couloirs, bowls): the shaded snow
    // read saturated blue and dark (s7 / c7: ~106,137,180, B-R +74) against a target of #a8b6d1 (B-R +41) -
    // a neutral-warm lift of its albedo where the sun does not reach (the shared lighting only has sky fill there)
    if (snowM > 0.01) alb *= mix(vec3(1.0), vec3(1.22, 1.13, 1.0), snowM * (1.0 - sun * smoothstep(0.0, 0.25, dot(n, uSunDir))));
    // boulder contact shadows + occlusion on the ground
    if (!cheap && t < 120.0) {
        occ *= rocksGroundAO(p);
        if (sun > 0.0) sun *= rocksShadow(p + n * 0.03, uSunDir, 8.0, rockLod(t));
    }
    if (uDebug == 11) return vec3(fw, fb, fa);          // falls lab: material weights
    if (uDebug == 13) return vec3(fw * (1.0 - gFallsMoss), fw * gFallsMoss, 0.0);   // falls lab: bare rock / cover
    if (uDebug == 14 || uDebug == 15) return vec3(0.0);                              // falls lab: mask baselines
    vec3 colT = litSurface(alb, n, rd, sun, occ, spec, mix(0.6, 0.25, wet) - 0.048 * fw * wetR);
    // (falls6) wet sheen of the falls rock: the film of water mirrors the (mist-bright) sky - glistening up-facing
    // facets / ledge lips near the sheets; Schlick fresnel, the sky seen along the reflected ray
    if (fw > 0.0 && !cheap) {
        vec3 rr = reflect(rd, n);
        float Fw = 0.02 + 0.98 * pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
        colT += skyLight(rr, occ) * Fw * smoothstep(-0.05, 0.35, rr.y) * fw * wetR;
    }
    return colT;
}

