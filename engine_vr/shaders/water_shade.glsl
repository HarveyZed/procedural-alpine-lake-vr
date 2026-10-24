// ---------------------------------------------------------------------------
// water_shade.glsl : lake surface shading: reflection, refraction, bed, scattering, glints (needs trace/shadeHit)
// ---------------------------------------------------------------------------
vec2 gDbgA = vec2(0.0);     // falls lab debug 66 / 67 (stream masks)
float gWWFoot = 0.0;        // (review rv8) footprint of the 3D white-water hit (lateral / along the view on the surface)
vec3 gDbgB = vec3(0.0);
vec3 shadeWater(vec3 ro, vec3 rd, float t) {
    if (uDebug >= 13 && uDebug <= 15) return vec3(0.0);          // falls lab mask renders (tools/lv.py --stats)
    vec3 p = ro + rd * t;
    float level = waterLevel(p.xz);
    p.y = level;
    float foot = t * gPixAng / max(abs(rd.y), 0.02);
    float gust, sv;
    // anisotropic footprint for the wave normals (water.glsl waterNormalA)
    vec2 fwdH = normalize(rd.xz + vec2(1e-6, 0.0));
    float fA = t * gPixAng;
    float fL = fA / max(abs(rd.y), uTune[8]);
    vec3 n = waterNormalA(p.xz, fwdH, fA, fL, gust, sv);
    // flowing stream: flow-map advected turbulence and whitewater (fading out where it meets the lake)
    float foamW = 0.0, foamT = 1.0;
    vec3 nW = vec3(0.0, 1.0, 0.0); float wW = 0.0;     // stream: white-water relief normal and its weight
    gStreamHead = 0.0;
    float headE = 1.0;            // (r4) radial position on the flat stream head (0 top .. 1 waterline)
    gWWHit = 0.0; gWWVis = 0.0; gWWOcc = 1.0; gWWSh = 1.0; gWWRock = 0.0; gWWRib = 0.0;
    vec4 st = streamAt(p.xz);
    // (review r2, fix 9) beyond the near map (z > ~1800 m, nearBlend > 0) the stream's flow / white water
    // come from its polyline (the 4 m far map has no usable stream data there): steep reaches white
    float nbS = uStreamN > 1 ? nearBlend(p.xz) : 0.0;
    if (nbS > 0.0) {
        vec2 sdF = streamDist(p.xz);
        float wfm = 1.0 - smoothstep(0.50 * sdF.y, 0.75 * sdF.y, sdF.x);
        st = mix(st, vec4(gStreamDir.xy, 0.95 * smoothstep(0.06, 0.14, gStreamDir.z), wfm), nbS);
    }
    float mf = 0.0;
    if (st.w > 0.02 && !wfOff(3)) {
        // (round 2) the stream mask is binary on the 1 m grid: its bilinear 0.3-0.9 contour cut the white
        // water off in 1 m stair steps along the banks. The mask now only has to be > 0 (its contour lies
        // under the banks, 1.2-1.9 x the half width), and the white thins out over the last ~12 cm of
        // depth at the true (smooth) water's edge instead
        float edgeD = level - terrainH(p.xz, true);
        mf = smoothstep(0.02, 0.15, st.w) * smoothstep(0.02, 0.30, level) * smoothstep(0.0, 0.12, edgeD);
        vec2 fd = normalize(st.xy + 1e-5);
        float el = 0.5;
        float wsl = (waterLevel(p.xz - fd * el) - waterLevel(p.xz + fd * el)) / (2.0 * el);
        // (round 3) the pixel footprint on the SLOPING stream surface: the cascade climbs away from a low camera
        // (film shot 4: incidence ~15-20 deg, not the ~3 deg of the horizontal lake plane, which faded every
        // stream detail into a flat white slide)
        float ddS = max(-(rd.y + wsl * dot(fd, rd.xz)), 0.02);
        float footW = t * gPixAng / ddS;
        float spd = 1.3 + 4.0 * st.z;
        float ph0 = fract(uTime * 0.4), ph1 = fract(uTime * 0.4 + 0.5);
        float wb = abs(ph0 - 0.5) * 2.0;
        vec2 q0 = p.xz - fd * spd * ph0 * 2.5, q1 = p.xz - fd * spd * ph1 * 2.5 + 3.7;
        float e = 0.06;
        vec2 g0 = vec2(texture(uNoise3, vec3((q0 + vec2(e, 0.0)) * 0.35, 0.3)).r - texture(uNoise3, vec3(q0 * 0.35, 0.3)).r,
                       texture(uNoise3, vec3((q0 + vec2(0.0, e)) * 0.35, 0.3)).r - texture(uNoise3, vec3(q0 * 0.35, 0.3)).r) / e;
        vec2 g1 = vec2(texture(uNoise3, vec3((q1 + vec2(e, 0.0)) * 0.35, 0.6)).r - texture(uNoise3, vec3(q1 * 0.35, 0.6)).r,
                       texture(uNoise3, vec3((q1 + vec2(0.0, e)) * 0.35, 0.6)).r - texture(uNoise3, vec3(q1 * 0.35, 0.6)).r) / e;
        vec2 g = mix(g0, g1, wb) * (0.10 + 0.25 * st.z);
        n = normalize(mix(n, normalize(vec3(-g.x, 1.0, -g.y) + n * 0.3), mf));
        // whitewater: streaks stretched along the flow (~2 m x 0.6 m), finer froth up close; ~80 %
        // white over the drops, ~15 % on the pools; edges no sharper than the pixel footprint
        vec2 fx = vec2(-fd.y, fd.x);
        // the streaks are long along the flow: where the stream runs toward / away from the camera only
        // the lateral footprint (not the grazing-angle one) has to resolve them
        float alV = abs(dot(fd, normalize(rd.xz + 1e-5)));
        float footS = mix(footW, max(t * gPixAng, footW * 0.27), alV * alV);
        float fineV = 1.0 - smoothstep(0.05, 0.15, footW);    // isotropic froth: full footprint
        float f0 = texture(uNoise3, vec3(dot(q0, fd) * 0.06, dot(q0, fx) * 0.22, 0.8)).r + (texture(uNoise3, vec3(q0 * 0.5, 0.8)).b - 0.5) * 0.6 * fineV;
        float f1 = texture(uNoise3, vec3(dot(q1, fd) * 0.06, dot(q1, fx) * 0.22, 0.2)).r + (texture(uNoise3, vec3(q1 * 0.5, 0.2)).b - 0.5) * 0.6 * fineV;
        float ft = wfRemap(mix(f0, f1, wb));
        float cov = mix(0.10, 0.50, smoothstep(0.04, 0.60, st.z));   // (round 2: more dark water between the white; r4 C4 / W8: 0.78 -> 0.50)
        // (session 4) where the water surface is nearly flat (the delta, the pools) the white is broken: runs of
        // dark glassy water between foam lines, not a white slide (mouth view / film shot 4)
        float wslF = (waterLevel(p.xz - fd * 0.5) - waterLevel(p.xz + fd * 0.5));
        cov *= mix(0.55, 1.0, smoothstep(0.03, 0.15, wslF));
        // (session 4) the white streaks are long along the flow (~2 m): seen along the flow at a grazing angle
        // only the lateral footprint has to resolve them (footS's 0.27 x footW faded them to a flat white
        // slide in the low mouth / film-shot-4 views); the along-view smear stays below a streak's length
        float footL = mix(footW, max(t * gPixAng * 1.5, footW * 0.05), alV * alV);
        float wd = 0.14 + 1.2 * footL;     // (round 3 i10: 0.08 -> 0.14, softer aerated edges - no white paint blotches)
        ft = mix(ft, 0.5, smoothstep(0.4, 1.5, footL));
        foamW = mf * smoothstep(1.0 - cov - wd, 1.0 - cov + wd, ft) * 0.95;
        // tumbling white water is not flat paint: grey-shaded streaks, thinner (darker) froth at its edges
        float tn = mix(0.5, texture(uNoise3, vec3(dot(q0, fd) * 0.15, dot(q0, fx) * 0.6, 0.35)).r, fineV);
        // + lane tone across the flow (~0.5-1 m lanes of brighter / greyer white water, resolved laterally)
        float lane = mix(0.5, texture(uNoise3, vec3(dot(q0, fd) * 0.02, dot(q0, fx) * 0.32, 0.55)).g, 1.0 - smoothstep(0.08, 0.3, footL));
        tn = 0.5 + (tn - 0.5) + 1.6 * (lane - 0.5);
        foamT = mix(0.42, 1.0, saturate(1.4 * (ft - (1.0 - cov)) / max(cov, 0.1) + 0.7 * (tn - 0.5) + 0.3));   // (r4 W8: grey flanks 0.55 -> 0.42)
        // steep water (level slope along the flow > ~0.35: over the lips, in the rapids) is white water
        // throughout, streaked along the flow (tone 0.7-1); along the flow only: the level's drop under
        // the banks does not count
        float stW = smoothstep(0.12, 0.30, wsl) * mf;
        // (round 2) tumbling white water has relief: mounds and standing waves (~1.5 m along x 0.7 m
        // across the flow) fixed to the bed (they do not drift with the foam), lit on their sunny side,
        // grey in the troughs, the troughs opening into dark holes on the rapids; unresolved -> the mean
        // (the same mean whiteness from 150-500 m: a white cascade)
        float visW = 1.0 - smoothstep(0.10, 0.40, footS);
        float rel = 0.5, gA = 0.0, gB = 0.0;
        if (visW > 0.0) {
            vec2 pa = vec2(dot(p.xz, fd), dot(p.xz, fx));
            const float eW = 0.18;
            rel = texture(uNoise3, vec3(pa.x * 0.10, pa.y * 0.24, 0.61)).g * 0.65 + 0.35 * texture(uNoise3, vec3(pa.x * 0.26, pa.y * 0.40, 0.13)).g;
            gA = (texture(uNoise3, vec3((pa.x + eW) * 0.10, pa.y * 0.24, 0.61)).g * 0.65 + 0.35 * texture(uNoise3, vec3((pa.x + eW) * 0.26, pa.y * 0.40, 0.13)).g - rel) / eW;
            gB = (texture(uNoise3, vec3(pa.x * 0.10, (pa.y + eW) * 0.24, 0.61)).g * 0.65 + 0.35 * texture(uNoise3, vec3(pa.x * 0.26, (pa.y + eW) * 0.40, 0.13)).g - rel) / eW;
            rel = mix(0.5, wfRemap(rel), visW);
        }
        vec3 fd3 = vec3(fd.x, 0.0, fd.y), fx3 = vec3(fx.x, 0.0, fx.y);
        nW = normalize(vec3(0.0, 1.0, 0.0) - (fd3 * gA + fx3 * gB) * 2.0 * visW);
        wW = mf * visW;
        foamT *= mix(1.0, mix(0.45, 1.08, smoothstep(0.15, 0.75, rel)), visW);
        foamW *= mix(1.0, mix(0.55, 1.0, smoothstep(0.12, 0.4, rel)), visW);
        if (stW > 0.0) {
            // (session 4) steep white water in lanes (resolved across the flow): dense white tongues and thinner
            // grey-green runs between them, not one white slide; unresolved -> ~0.82 x
            float laneV = 1.0 - smoothstep(0.08, 0.3, footL);
            // (review r2, fix 10: unresolved 0.82 / 0.92 -> 0.95 / 1.0: near-white #e9ecea from 150-500 m)
            // (r4 C4: unresolved 0.95 -> 0.6, the cascade is a white thread with grey-green water beside it)
            float lw = mix(0.6, mix(0.42, 1.0, smoothstep(0.32, 0.68, lane + 0.35 * (ft - 0.5))), laneV);
            foamW = max(foamW, stW * lw * mix(1.0, mix(0.45, 0.97, smoothstep(0.18, 0.5, rel)), visW));
            foamT = mix(foamT, mix(0.62, 1.04, saturate(ft + 0.7 * (tn - 0.5))) * mix(1.0, mix(0.6, 1.08, smoothstep(0.15, 0.75, rel)), visW), stW);
        }
        // (session 3) a tumbling cascade, not a white slide (film shot 4 / mouth view): on the steep
        // reaches the water drops over a lip every ~2.5-5 m (a ragged line across the channel); below each
        // lip a band of churned white water, then a short run of darker glassy water streaked with foam
        // lines before the next lip. Wet boulder heads break the surface, white water piled on their
        // upstream side, a dark wake below. Unresolved (footprint > ~0.5 m) -> the previous mean whiteness
        // (the 150-500 m views: a white cascade)
        float steepS = smoothstep(0.05, 0.22, wsl) * mf;
        gDbgA = vec2(stW, steepS);
        float visC = 1.0 - smoothstep(0.12, 0.5, mix(t * gPixAng, footW, alV * alV));   // (review r2: along-flow footprint)
        if (mf * visC > 0.0) {
            vec2 pa = vec2(dot(p.xz, fd), dot(p.xz, fx));
            // (review r2, fix 11) the churn follows the bake's real level drops (step pools 4.5-11 m: the old
            // synthetic 3.4 m phase did not line up with them): white water just below each drop, decaying
            // over ~2.5 m into a darker glassy run; a glassy tongue right at the lip. Filtered with the
            // footprint ALONG the flow (footF; footS is the lateral one, up to 3.7x too small here)
            float footF = mix(t * gPixAng, footW, alV * alV);
            // (review rv1) the probe distances vary across the channel (+-0.8 m): the white below each drop starts along a
            // ragged line, not a ruled one (the terraces of film shot 4)
            float jg = mix(0.0, (texture(uNoise3, vec3(pa.y * 0.35 + 0.7, pa.x * 0.02, 0.41)).r - 0.475) * 7.0, 1.0 - smoothstep(0.15, 0.5, footS));
            float o1 = max(1.2 + 0.8 * jg, 0.35), o2 = max(2.6 + 1.2 * jg, 0.8);
            float dr1 = (waterLevel(p.xz - fd * o1) - level) / o1, dr2 = (waterLevel(p.xz - fd * o2) - level) / o2;
            float dLip = (level - waterLevel(p.xz + fd * 0.6)) / 0.6;
            float aw = 0.05 + 0.6 * footF;
            float churnR = max(smoothstep(0.10 - aw * 0.2, 0.32 + aw * 0.2, dr1), 0.75 * smoothstep(0.12, 0.35, dr2));
            float churn = saturate(churnR + 0.35 * (ft - 0.5) * (1.0 - smoothstep(0.1, 0.4, footF)));
            // (round 3 i10) the churn below each drop comes in tongues along the flow (~1-2 m wide, many metres
            // long) with darker glassy lanes between them: the white water tumbles in tongues between the
            // boulders, not in bands straight across the channel (the terraces of film shot 4)
            float tg = mix(0.5, texture(uNoise3, vec3(pa.y * 0.45 + 0.3, pa.x * 0.045, 0.66)).r, 1.0 - smoothstep(0.12, 0.45, footS));
            churn *= mix(1.0, smoothstep(0.30, 0.62, tg) * 1.35, 0.85);
            float lipD = smoothstep(0.15, 0.45, dLip) * (1.0 - churnR) * (1.0 - smoothstep(0.06, 0.2, footF));   // glassy tongue at the lip
            float runF = smoothstep(0.60, 0.82, ft) * 0.7 * (1.0 - lipD);
            float wC = max(churn * mix(0.8, 1.0, ft), runF);
            // (review rv1) on the steep reaches the runs between the drops stay white in tongues (one continuous
            // tumbling ribbon between the boulders, not white bands with glassy water between them)
            wC = max(wC, stW * (0.6) * smoothstep(0.25, 0.6, tg) * mix(0.8, 1.0, ft));
            float k = steepS * visC;
            foamW = mix(foamW, wC, k);
            gDbgB = vec3(lipD, churn, wC);
            foamT = mix(foamT, mix(0.62, 1.05, churn) * mix(0.85, 1.05, ft), k * 0.7);
            // boulder heads (~2 m cells, ~1 in 4): dark wet granite domes 0.3-0.6 m across
            vec2 bs = vec2(0.42, 0.55);
            vec4 bc = wfVor(pa * bs + 11.0);
            vec2 bd = bc.xy / bs;                                 // (along, across) from the head's centre (m)
            float bR = 0.28 + 0.30 * fract(bc.w * 5.31);
            float rH = length(bd);
            // (r4 C4 / R9: an irregular lobed outline (3 + 5 lobes, own phase), not a smooth ellipse - the manhole covers)
            float angH = atan(bd.y, bd.x);
            bR *= 1.0 + 0.22 * (0.6 * sin(3.0 * angH + bc.w * 40.0) + 0.4 * sin(5.0 * angH + bc.w * 71.0));
            headE = rH / bR;
            float hasH = step(fract(bc.w * 13.7), 0.26);
            float aH = 0.03 + max(footF, footS);
            k = mf * visC;                                       // (session 4: heads on every reach, not only the steep)
            float head = hasH * (1.0 - smoothstep(bR - aH, bR + aH, rH)) * k;
            // white water piled against the upstream side, a dark trough / wake downstream
            float side = bd.x / max(rH, 1e-3);                    // -1 upstream .. +1 downstream
            float collar = hasH * (1.0 - smoothstep(bR + 0.15, bR + 0.6 + aH, rH)) * smoothstep(0.2, -0.5, side) * k;   // (r4: wider pillow 0.45 -> 0.6)
            float wake = hasH * (1.0 - smoothstep(bR * 0.8, bR * 1.3 + aH, abs(bd.y))) * smoothstep(0.0, bR, bd.x) * (1.0 - smoothstep(bR, bR + 2.2, bd.x)) * k;
            // (r4 R9) a thin white contact ring where the water wraps the head (~4 cm; unresolved -> faint)
            float ring = hasH * (1.0 - smoothstep(0.015 + 0.5 * aH, 0.04 + aH, abs(rH - bR - 0.02))) * k * mix(0.85, 0.3, smoothstep(0.03, 0.12, aH));
            foamW = max(max(foamW * (1.0 - 0.75 * wake), collar), ring);
            foamW *= 1.0 - head;
            gStreamHead = head;
            vec3 fd3h = vec3(fd.x, 0.0, fd.y), fx3h = vec3(fx.x, 0.0, fx.y);
            gStreamHeadN = normalize(vec3(0.0, 1.0, 0.0) + (fd3h * bd.x + fx3h * bd.y) / max(bR, 0.1) * 1.3);
        }
        // (round 3) the white water in 3D where it is resolved (lateral footprint < ~0.15 m: film shot 4, the
        // close stream views): heaps of aerated water standing above the water level (falls.glsl wwField),
        // up to ~0.5 m on the steep reaches, found by marching back along the ray from where it enters the
        // layer (parallax occlusion: heaps hide the water behind them, lit tops / shaded flanks). A hit is
        // shaded as opaque white water at the end; the 2D foam between the heaps thins out
        // (review rv5) the white water is a ribbon about half the channel wide winding between the boulders (the
        // channel was white from bank to bank - a painted staircase in film shot 4): dark glassy water beside it.
        // Resolved laterally only (footS); unresolved -> unchanged (the far views keep their white thread)
        vec2 sdR = streamDist(p.xz);
        gWWHw = 0.20 * sdR.y;     // (r4 C4 / W8: 0.45 -> 0.20, a ribbon ~1/4 of the bed)
        // (rv6) only on the steep reaches (the cascade): at the flat mouth the white fans out over the whole width
        // - the ribbon there left a salt-and-pepper of heap tops
        // (c1: also on the gently sloping reaches where the 3D heaps are drawn (bed slope > ~5 %): there the heaps covered the
        // channel bank to bank - film shot 4's near stream, debug 63)
        float ribV = (1.0 - smoothstep(0.2, 0.6, footS / max(gWWHw, 0.5))) * max(max(stW, steepS), smoothstep(0.03, 0.08, gStreamDir.z));
        float rib = mix(1.0, wwRibbon(vec2(-gStreamS, gStreamD), gWWHw, footS), ribV);
        gWWRib = ribV;
        foamW *= mix(1.0, 0.25 + 0.75 * rib, ribV);
        // (review rv7) the 3D heaps / boulders only on the sloping reaches (bed slope > ~5 %): on the flat mouth the grazing
        // march hit the low heap tops at random - a salt-and-pepper of white dots (film shot 4); the 2D foam stays there
        float visP = (1.0 - smoothstep(0.07, 0.22, footS)) * mf * smoothstep(0.03, 0.08, gStreamDir.z);
        float Hm = mix(0.24, 0.72, max(stW, steepS)) * saturate(foamW * 1.3);   // (i12: x1.2, more volume)
        gStreamHead *= 1.0 - visP * smoothstep(0.10, 0.22, max(ddS, 0.03));   // (the 3D boulders take over from the flat heads)
        if (visP > 0.0) {
            // curvilinear stream coordinates of this pixel (continuous across segments, same for every pixel),
            // stepped along the ray in the segment's frame
            vec2 fdS = gStreamDir.xy, fxS = vec2(-fdS.y, fdS.x);
            vec2 paP = vec2(-gStreamS, gStreamD);
            // the ray's descent toward the (locally planar, sloping) water surface per unit of t
            float dd = max(ddS, 0.03);
            // (review rv3) the 3D boulders only where the ray meets the (locally planar) surface steeply enough for the short
            // parallax march to reach their tops: at grazing views over the pools they were cut into flat horizontal
            // 'chrome pucks' along every lip (film shot 4, debug 63); there the flat 2D heads stay
            gWWBo = smoothstep(0.10, 0.22, dd);
            // (r4 R9: the march started at most 8 % of t before the surface - at film shot 4's ~10 m it saw only the lowest
            // ~0.12 m of the boulders, cut flat: dark discs / pucks lying in the white water; now up to 25 %)
            float tE = max(t - max(Hm, 1.45) / dd, t * 0.75);     // (i10/11: 0.62 -> 1.1, the boulders stand up to ~1 m; rv1 1.45: up to 1.3 m)
            // (review rv8) the heaps' fine bubbles / normals filtered by the footprint ALONG the view on the surface too (footW):
            // at grazing views their 4-30 cm detail is sub-pixel vertically - it was shaded into a grey salt-and-pepper
            float footG = max(footS, 0.35 * footW);
            gWWFoot = footG;
            float visB = 1.0 - smoothstep(0.03, 0.08, footG);
            const int NW = 14;
            float ds = (t - tE) / float(NW);
            float ti = tE + ds * fract(gRand * 7.13 + 0.37), a0 = tE, hit = -1.0, rk;
            // (review rv7) closest approach of the ray to the field: a near miss (within the march's vertical resolution)
            // counts as a partial hit there - soft heap outlines instead of random white / dark dots
            float minD = 1e9, tMin = tE;
            for (int i = 0; i < NW; i++) {
                vec2 dq = rd.xz * (ti - t);
                float dH = (t - ti) * dd - wwField(paP + vec2(dot(dq, fdS), dot(dq, fxS)), Hm, visB, rk);
                if (dH < 0.0) { hit = ti; break; }
                if (dH < minD) { minD = dH; tMin = ti; }
                a0 = ti;
                ti += ds;
            }
            float covW = 1.0;
            bool nearM = false;
            if (hit < 0.0) {
                covW = 1.0 - smoothstep(0.0, max(0.03, 1.6 * ds * dd), minD);   // (rv9: 0.8 -> 1.6 x the march resolution)
                if (covW > 0.02) { hit = tMin; nearM = true; }
            }
            if (hit > 0.0) {
                float ba = nearM ? hit : a0, bb = hit;
                for (int k = 0; k < (nearM ? 0 : 4); k++) {
                    float m = 0.5 * (ba + bb);
                    vec2 dq = rd.xz * (m - t);
                    if ((t - m) * dd < wwField(paP + vec2(dot(dq, fdS), dot(dq, fxS)), Hm, visB, rk)) bb = m; else ba = m;
                }
                vec2 dq = rd.xz * (bb - t);
                vec2 pq = paP + vec2(dot(dq, fdS), dot(dq, fxS));
                gWWP = ro + rd * bb;
                float eH = max(0.03, max(bb * gPixAng, 0.5 * footG));
                float rkx, rkz;
                float h0 = wwField(pq, Hm, visB, gWWRock);
                gWWH = h0;
                float hx = wwField(pq + vec2(eH, 0.0), Hm, visB, rkx), hz = wwField(pq + vec2(0.0, eH), Hm, visB, rkz);
                // gradient (along, across) -> world xz
                vec2 gA = vec2(hx - h0, hz - h0) / eH;
                vec2 gW = fdS * gA.x + fxS * gA.y;
                vec3 ns = normalize(vec3(wsl * fd.x, 1.0, wsl * fd.y));
                gWWN = normalize(ns - vec3(gW.x, 0.0, gW.y));
                gWWOcc = mix(0.28, 1.0, saturate(h0 / max(Hm, 0.3)));   // (review rv5: 0.45 -> 0.28, darker troughs)
                // heaps / boulders shade each other (3 steps toward the sun over the field, relative to the surface)
                float rise = uSunDir.y + wsl * dot(fd, uSunDir.xz);
                vec2 Lp = vec2(dot(uSunDir.xz, fdS), dot(uSunDir.xz, fxS));
                float shW = 0.0;
                for (int k = 1; k <= 3; k++) {
                    float sd = 0.1 * float(k * k);
                    shW = max(shW, smoothstep(0.0, 0.06, wwField(pq + Lp * sd, Hm, visB, rkx) - h0 - sd * rise));
                }
                gWWSh = 1.0 - shW;
                gWWHit = 1.0;
            }
            gWWVis = visP * covW;
            // (review rv9) the water between the heaps stays aerated (0.4 -> 0.75): the troughs the march slips through showed as
            // a grey salt-and-pepper over the white cascade (film shot 4)
            foamW *= mix(1.0, 0.4, visP);     // (r4 C4: 0.75 -> 0.4, dark glassy water between the tongues)
        }
        foamT *= 0.8;
    }
    // falls lab debug 63: stream layers (r = 3D boulder hit, g = 3D white-water hit, b = 2D foam cover)
    if (uDebug == 63) return vec3(gWWHit * gWWRock * gWWVis, gWWHit * (1.0 - gWWRock) * gWWVis, foamW);
    if (uDebug == 66) return vec3(mf, gDbgA.x, gDbgA.y);
    if (uDebug == 67) return gDbgB;
    // the stream's outflow into the lake: a few soft foam streaks drifting out, gone after ~15 m
    // (the lake stays clear turquoise right up to the cascade)
    if (dot(uStreamMouth.zw, uStreamMouth.zw) > 0.5) {
        vec2 dm = p.xz - uStreamMouth.xy;
        float along = dot(dm, uStreamMouth.zw), lat = dot(dm, vec2(-uStreamMouth.w, uStreamMouth.z));
        if (along > -8.0 && along < 70.0 && abs(lat) < 30.0) {
            float wdt = 2.0 + 0.2 * max(along, 0.0);
            float pl = exp(-max(along, 0.0) / 6.0) * exp(-lat * lat / (wdt * wdt)) * smoothstep(-8.0, 0.0, along) * (1.0 - mf);
            vec2 qa = vec2(along - uTime * 0.45, lat + 0.5 * gnoise(dm * 0.08));
            float nn = wfRemap(texture(uNoise3, vec3(qa.x * 0.05, qa.y * 0.14, 0.45 + uTime * 0.01)).r) * 0.65
                     + 0.35 * wfRemap(texture(uNoise3, vec3(qa * 0.12, 0.2)).a);
            nn = mix(nn, 0.5, smoothstep(0.4, 1.5, foot));
            float wdp = 0.1 + 1.0 * foot;
            foamW = max(foamW, smoothstep(0.82 - 0.35 * pl - wdp, 0.82 - 0.35 * pl + wdp, nn) * smoothstep(0.05, 0.5, pl) * 0.6);
            n = normalize(n + vec3(gnoise(qa * 0.7), 0.0, gnoise(qa * 0.7 + 3.1)) * 0.12 * pl);
        }
    }
    // waterfall plunge pool: foam and ring waves
    vec2 tilt;
    float pf = poolFoamF(p.xz, foot, tilt);
    n = normalize(n + vec3(tilt.x, 0.0, tilt.y));
    if (pf > foamW) foamT = mix(foamT, gPoolTone, saturate((pf - foamW) / max(pf, 1e-3)));
    foamW = max(foamW, pf);
    if (uDebug == 9) return vec3(pf, gPoolAer, foamW);
    // unresolved wave slopes (sv): their spread smears the reflection vertically (uTune[6] x sigma).
    // (r2c) Two parts of unit variance: (1) a world-space ripple-group pattern (wind-aligned, crests ~3x longer
    // than wide, drifting and evolving with uTime) at the finest scales the along-view footprint resolves
    // (cells >= 2-5 px): broken horizontal streaks, coherent in space and time (no grain, moves with
    // the water); (2) the stochastic per-sample part for what is still below the pixel (converges to a soft
    // vertical blur with the film's spp). uTune[7] = share of (1) where it is resolved. (The r2 masking tilt
    // toward the viewer, the old slot 7, is gone: it brought the pale far-water band back.)
    {
        float sg = sqrt(sv);
        int fr = int(floor(uTime * 30.0 + 0.5));
        float h1 = hash12i(ivec2(gl_FragCoord.xy) * 5 + ivec2(fr * 389 + 3, fr * 7741 + 11));
        float h2 = hash12i(ivec2(gl_FragCoord.xy) * 3 + ivec2(fr * 1543 + 17, fr * 97 + 5));
        float gs = (fract(h1 + float(uSample) * 0.6180340) + fract(h2 + float(uSample) * 0.7548777) - 1.0) * 2.45;
        float gc = 0.0, gq = 0.0, rem = 1.0;
        float tu7 = uTune[7];
        if (tu7 > 0.0) {
            vec2 wd = normalize(uWaterWind + vec2(1e-5, 0.0));
            vec3 nTs = vec3(textureSize(uNoise3, 0));
            // (r4 C6 / W2: crests 1.5x (was 3x) longer than wide and cells only up to 1.8 m (was 16 m): the long parallel
            // world-space strips converged on a vanishing point (the speed-line fan, shot 1) and shifted whole strips of
            // the far-shore reflection out into the lake (the venetian blinds, shot 0))
            vec2 q0 = vec2(dot(p.xz, wd), dot(p.xz, vec2(-wd.y, wd.x)) / 1.5) - vec2(uTime * 0.35, 0.0);
            float cs = 0.2;                                   // cell size (m) of the finest octave
            for (int j = 0; j < 3; j++) {
                float wj = smoothstep(1.5, 4.0, cs / fL) * rem;
                if (wj > 0.0) {
                    // (water5) C1-smooth (cubic) interpolation between the noise texels: the trilinear cells drew blocky
                    // facets into the far-shore reflection (shot 0's mid water)
                    vec3 nq = vec3(q0 / (16.0 * cs) + float(j) * 0.37, uTime * 0.05 + float(j) * 0.29) * nTs - 0.5;
                    vec3 ni = floor(nq), nf = nq - ni;
                    nf.xy = nf.xy * nf.xy * (3.0 - 2.0 * nf.xy);
                    float nz = texture(uNoise3, (ni + nf + 0.5) / nTs).g;
                    gc += wj * (nz - 0.5) * 5.0;              // value noise std ~0.2 -> ~unit
                    gq += wj * wj;
                    rem -= wj;
                }
                cs *= 3.0;
            }
        }
        float sh = (1.0 - rem) * tu7;                         // coherent share of the variance
        // (the per-sample part is temporal noise in the film (shot 0: 13.8 % of the water pixels changed > 20
        // levels per frame at 2 spp); with the ripple groups on it is kept at 35 %: the far, unresolved water
        // keeps a soft smear, the rest of the variance is carried by the coherent pattern or dropped)
        // (r2d) uTune[22] = per-sample share at steep views (|rd.y| 0.08 -> 0.3): looking down on the lake (flyover,
        // high overlooks) the unresolved smear only scattered the reflected sky / cloud boundary into pixel speckle
        // (shot 6: reflection high-pass std 0.24 vs the water body 0.26, ~10 % sparkle at 2 spp); at grazing views
        // it still draws the vertical streaks
        float gsS = mix(1.0, uTune[22], smoothstep(0.08, 0.3, -rd.y));
        float g = (gq > 1e-4 ? gc * inversesqrt(gq) : 0.0) * sqrt(sh) + gs * sqrt(1.0 - sh) * mix(1.0, 0.35, tu7) * gsS;
        float tilt = sg * uTune[6] * g;
        // (r4 W2: a tilt away from the viewer flattens the reflected ray onto the far beach: clamp 0.6 -> 0.35 |rd.y|)
        tilt = max(tilt, -0.35 * abs(rd.y));
        n = normalize(n - vec3(fwdH.x, 0.0, fwdH.y) * tilt);
    }
    // back-facing facet left after the filtering: project the normal just far enough to face the viewer
    // (the old fix added 0.2 x the view vector, which tilted it ~10 deg and sent the reflection ~20 deg up)
    // (r2b: always the minimal fix; uTune[9] is now the sun warmth, probe.frag)
    float nv = dot(n, -rd);
    if (nv < 0.02) n = normalize(n + (-rd) * (0.02 - nv));
    float cosi = saturate(dot(-rd, n));
    // (r4 W1: roughness-aware Fresnel - at grazing views the visible ripple facets lean toward the viewer (masking), by about the
    // slope spread: the filtered-out ripples (sv) plus a floor for the resolved ones; the effective incidence is |rd.y| + that tilt,
    // which lets the teal body through (r4 mid F 0.64 -> ~0.4; far band <= ~0.55) instead of a grey / khaki mirror)
    // (the floor for the resolved ripples fades out toward the far band, |rd.y| < ~0.01: there all the ripples are in sv and the far
    // shore's reflection keeps dominating)
    // (c1: both terms fade out below |rd.y| ~0.03: the unresolved sv alone had killed the far shore's forest reflection in the shore
    // views - flat green paint, mid-water hp 0.08-0.10 vs a 0.42 target (shots 0 / 1))
    // (the lean is weaker in the midday shore views (k 0.6: their reflections carry the ripple texture - k 1.5 left shots 0 / 1 flat
    // green paint, hp 0.08-0.10 vs 0.42) than at a low sun (k 1.5: turquoise water, not a khaki mirror of the lit wall))
    float lowSunF = 1.0 - smoothstep(0.35, 0.6, uSunDir.y);
    // (water5: midday k 0.6 -> 0.78 and the lean kept down to the horizon (it faded out below |rd.y| 0.03): the far band was
    // a yellow-green copy of the hazed far-shore forest (shots 0 / 1 h158-163, F .85-.9) - it should read teal-grey
    // (h183-200), the water body through the ripple facets; the low sun keeps the fade: there the lake mirrors the lit far forest)
    float cosF = max(cosi, abs(rd.y) + mix(0.78, 1.5, lowSunF) * (sqrt(sv) + mix(0.03, 0.06, lowSunF) * uWaveStrength) * mix(1.0, smoothstep(0.006, 0.03, abs(rd.y)), lowSunF));
    float F = 0.02 + 0.98 * pow(1.0 - cosF, 5.0);

    // reflection
    vec3 rr = reflect(rd, n);
    rr.y = max(rr.y, 0.003);
    rr = normalize(rr);
    vec3 ro2 = p + vec3(0.0, 0.05, 0.0);
#ifdef NO_REFL
    Hit hr; hr.kind = K_SKY; hr.t = 1e9;
#else
    gTraceT0 = t;
    Hit hr = trace(ro2, rr, 0.1, 25000.0, false, true, t < 300.0, QUALITY == 1 ? 320 : 160, 2.5);
    gTraceT0 = 0.0;
#endif
    if (uDebug == 60) return vec3(hr.kind == K_SKY ? 1.0 : 0.0, rr.y * 10.0, t / 4000.0);
    if (uDebug == 61) return vec3(hr.t / 4000.0, float(hr.kind) / 6.0, F);
    if (uDebug == 3) {
        vec3 kc[7] = vec3[7](vec3(0.2, 0.3, 1.0), vec3(0.6, 0.5, 0.4), vec3(0.0, 0.6, 0.0), vec3(0.5, 0.5, 0.5), vec3(0.0, 0.8, 0.9), vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 0.0));
        return kc[hr.kind] * (hr.kind == K_TERRAIN ? clamp(hr.t / 3000.0, 0.2, 1.0) : 1.0);
    }
    vec3 refl;
    if (hr.kind == K_SKY) refl = skyWithClouds(ro2, rr, 10, QUALITY == 1 ? 40 : 20);
    else refl = applyAerial(shadeHit(hr, ro2, rr, true), ro2, rr, hr.t, 8);

    // sun glint: GGX with Smith masking-shadowing (grazing views must not glitter everywhere)
    vec3 hv = normalize(uSunDir - rd);
    float a = mix(0.03, 0.08, gust) + min(foot * 0.02, 0.05);
    float nh = max(dot(n, hv), 0.0);
    float dd = nh * nh * (a * a - 1.0) + 1.0;
    float D = a * a / (PI * dd * dd);
    float nl = max(dot(n, uSunDir), 0.0);
    float cv2 = cosi * cosi, cl2 = nl * nl;
    float G1v = 2.0 / (1.0 + sqrt(1.0 + a * a * (1.0 - cv2) / max(cv2, 1e-6)));
    float G1l = 2.0 / (1.0 + sqrt(1.0 + a * a * (1.0 - cl2) / max(cl2, 1e-6)));
    float sunVis = terrainShadow(p + vec3(0, 0.5, 0), 50) * cloudShadow(p);
    float Fh = 0.02 + 0.98 * pow(1.0 - max(dot(hv, -rd), 0.0), 5.0);
    vec3 glint = uSunCol * sunVis * D * Fh * G1v * G1l / (4.0 * max(cosi, 1e-3)) * nl;
    // (r4 W14) sub-pixel sun glints: the facets of the unresolved ripples (slope spread sqrt(sv)) catch the sun in tiny
    // points - the mean glint over that wider lobe is drawn as rare per-sample sparkles of a fixed peak radiance (the
    // same mean over the samples: 1-3 px clipped white points near the sun azimuth, no flat pale dashes / glare path)
    {
        float aS = sqrt(a * a + 0.3 * sv) + 0.03;     // (0.3 sv: the sparkles stay near the glitter path, none off it)
        float ddS = nh * nh * (aS * aS - 1.0) + 1.0;
        float DS = aS * aS / (PI * ddS * ddS);
        float G1vS = 2.0 / (1.0 + sqrt(1.0 + aS * aS * (1.0 - cv2) / max(cv2, 1e-6)));
        float G1lS = 2.0 / (1.0 + sqrt(1.0 + aS * aS * (1.0 - cl2) / max(cl2, 1e-6)));
        vec3 gM = uSunCol * sunVis * DS * Fh * G1vS * G1lS / (4.0 * max(cosi, 1e-3)) * nl;
        float lS = dot(uSunCol, vec3(0.2126, 0.7152, 0.0722)) * 120.0;
        float P = dot(gM, vec3(0.2126, 0.7152, 0.0722)) / max(lS, 1e-4);
        int frG = int(floor(uTime * 24.0 + 0.5));
        float uG = hash12i(ivec2(gl_FragCoord.xy) * 7 + ivec2(uSample * 7919 + frG * 131, uSample * 3511 + frG * 977));
        glint = P >= 1.0 ? gM : (uG < P ? gM / max(P, 1e-6) : vec3(0.0));
    }

    // refraction into the lake
    vec3 tdir = refract(rd, n, 1.0 / 1.333);
    if (dot(tdir, tdir) < 0.5) tdir = normalize(rd + vec3(0.0, -0.5, 0.0));
    Hit hb = trace(p - vec3(0.0, 0.01, 0.0), tdir, 0.0, 90.0, false, false, t < 300.0, 120, 3.0);
    float path = hb.kind == K_SKY ? 90.0 : hb.t;
    vec3 bp = p + tdir * path;
    float bedDepth = max(level - bp.y, 0.0);
    vec3 bedCol = vec3(0.0);
    // (look3 r3) clearer water: 0.45 x the round-2 scattering (stones on the shelf visible 2-4 m down),
    // the in-scatter scale x1.45 keeps the deep turquoise
    // (water5: 0.45 -> 0.225 with the clearer W_ABS: the deep in-scatter ceiling wSca / (2 sext) stays, its build-up per metre
    // halves - clear shallows, the bed visible through twice the depth)
    vec3 wSca = W_SCA * 0.225;
    vec3 sext = W_ABS + wSca;
    // downwelling light at depth z (sun refracted, sky diffuse)
    float sunCosW = sqrt(1.0 - (1.0 - uSunDir.y * uSunDir.y) / (1.333 * 1.333));
    vec3 Tsun = exp(-sext * bedDepth / max(sunCosW, 0.1));
    vec3 Tsky = exp(-sext * bedDepth * 1.25);
    if (hb.kind != K_SKY) {
        // simplified bed shading (stones, silt, algae) with caustics
        vec3 bn = hb.kind == K_ROCK ? vec3(0.0, 1.0, 0.0) : terrainNormal(bp.xz, max(0.05, t * gPixAng * 2.0), detailWeight(t));
        vec3 alb;
        float n3 = gnoise(bp.xz * 0.45);
        if (hb.kind == K_ROCK) alb = rockBedAlbedo(bp, t, hb.cell, bn);
        else {
            vec3 bb;
            alb = pebbles(bp.xz, bb) * 0.95 * vec3(1.0, 0.95, 0.75);   // (r4 W6: golden-olive bed)
            bn = normalize(bn + bb * (1.0 - smoothstep(6.0, 20.0, t)));
            vec3 silt = mix(vec3(0.20, 0.21, 0.17), vec3(0.28, 0.28, 0.23), saturate(0.5 + n3));
            silt *= vec3(1.08, 1.0, 0.62) * 1.6;    // (water5: paler, golden-olive rock-flour silt, not grey: the luminous band over the shelf)
            alb = mix(alb, silt, smoothstep(1.5, 6.0, bedDepth) * 0.8);
            // (water5) the deeper bed in broad patches (~15-40 m) of pale rock-flour sand and darker cobble / rock: the turquoise
            // over it is luminous over the pale patches, deeper jade over the dark ones, not one flat sheet
            float bpN = texture(uNoise3, vec3(bp.xz * 0.021, 0.37)).r * 0.65 + texture(uNoise3, vec3(bp.xz * 0.055, 0.81)).r * 0.35;
            alb *= mix(1.0, clamp(1.0 + 2.4 * (bpN - 0.475), 0.4, 1.6), smoothstep(1.0, 4.0, bedDepth));
            alb *= mix(vec3(1.0), vec3(0.95, 1.0, 0.85), smoothstep(0.5, 4.0, bedDepth));   // algae film (r4: from 0.5 m; water5: (.85, 1, .8) -> (.95, 1, .85), less green)
            // (r4 W3) steep submerged faces (the cliff foot) are dark wet rock, not pale pebbles
            alb = mix(alb, vec3(0.018, 0.019, 0.016), smoothstep(0.6, 0.45, bn.y));
        }
        // (r4 C4 / W8) the stream's bed is wet, algae-dark stone (dark glassy water between the white tongues), not the
        // lake shelf's pale pebbles - through the few cm of stream water they made it a pale plaster sheet (film shot 4)
        alb *= mix(vec3(1.0), vec3(0.36, 0.40, 0.33), mf);
        // (water5) submerged stones are wet and carry a golden-brown periphyton film (target shallows #6e644a /
        // #6d684e / #68685e: the bed through the water was green-grey, h150 vs h90-130); not at the waterline, not in the stream
        alb = mix(alb, alb * vec3(1.0, 0.78, 0.60) * 0.85, smoothstep(0.03, 0.35, bedDepth) * (1.0 - mf));
        float sh = terrainShadow(bp + vec3(0.0, level - bp.y + 0.3, 0.0), 40);
        float ca = caustics(bp.xz - uSunDir.xz / max(uSunDir.y, 0.2) * bedDepth * 0.75, bedDepth, (t + path) * gPixAng / max(-tdir.y, 0.25));
        float ndl = max(dot(bn, uSunDir), 0.0);
        // (water5: caustic gain 0.9 -> 1.8: a bright net over the near stones)
        vec3 E = uSunCol * (1.0 - 0.03) * ndl * sh * Tsun * (0.85 + 1.8 * ca) + uSkyAmb * 0.8 * Tsky * terrainAO(bp.xz);
        // (r4 W3) contact occlusion where the bed meets the surface (no bright kerb along the cliff foot / boulder skirts)
        bedCol = alb * E * mix(0.35, 1.0, smoothstep(0.0, 0.4, bedDepth));
    }
    vec3 Tv = exp(-sext * path);
    // single scattering in the water column (approx. downwelling at mid depth)
    // single scattering: light entering at the surface decays on its way down and back up along
    // the view path (combined extinction ~ 2 x sext); deep water saturates at W_SCA / (2 sext)
    // (the water column under a terrain / cloud shadow only gets skylight: golden-hour shaded lake)
    vec3 E0 = uSunCol * max(uSunDir.y, 0.0) * 0.97 * mix(1.0, sunVis, uTune[10]) + uSkyAmb;
    // (r4 L5) low sun: the sky-lit water column keeps the shaded lake turquoise (near water #2c5b62, mid #40908e;
    // the gold look's low ambient left a khaki mirror of the lit wall)
    // (warm-grey: the low sun's light; the blue sky alone turned the shaded water ink-blue)
    // (water5: 1.5 -> 2.4 with the clearer, less scattering column: target mid #40908e C.078, ours C.052-.056)
    E0 += vec3(1.0, 1.0, 0.65) * dot(uSkyAmb, vec3(0.2126, 0.7152, 0.0722)) * 2.4 * (1.0 - smoothstep(0.35, 0.6, uSunDir.y));
    vec3 ins = E0 * uWaterIns * 1.45 * (wSca / (2.0 * sext)) * (1.0 - exp(-2.0 * sext * path));
    vec3 refr = bedCol * Tv + ins;

    if (uDebug == 4) return refl;
    if (uDebug == 5) return refr;
    if (uDebug == 68) return bedCol * Tv;
    if (uDebug == 69) return ins;
    if (uDebug == 6) return vec3(F);
    if (uDebug == 7) return glint;
    vec3 col = mix(refr, refl, F) + glint;
    if (foamW > 0.0) {
        vec3 nf = normalize(mix(normalize(n + vec3(0.0, 1.5, 0.0)), nW, wW));
        vec3 foamC = vec3(0.86, 0.89, 0.90) * (uSunCol * max(dot(nf, uSunDir), 0.0) * sunVis + skyLight(nf, 1.0) * 1.1);
        col = mix(col, foamC * foamT, foamW);
    }
    // stream boulder heads breaking the surface (stream block above): dark wet granite, a wet sheen
    if (gStreamHead > 0.0) {
        vec3 nr = gStreamHeadN;
        // (r4 C4 / W8 / R9: the same granite ramp as the 3D boulders - wet and near-black at the waterline, pale dry granite
        // on the top (was flat 0.075: black holes punched in the white water); grain + lichen up close)
        float fpH = t * gPixAng;
        float grH = mix(0.475, texture(uNoise3, vec3(p.xz * 0.9, 0.3)).r, 1.0 - smoothstep(0.03, 0.1, fpH)) * 0.6
                  + 0.4 * mix(0.475, texture(uNoise3, vec3(p.xz * 0.3, 0.7)).r, 1.0 - smoothstep(0.1, 0.3, fpH)) + 0.025;
        vec3 raH = mix(vec3(0.075, 0.074, 0.069), vec3(0.26, 0.252, 0.235), smoothstep(0.85, 0.35, headE)) * mix(0.75, 1.25, grH);
        vec3 rc = raH * (uSunCol * max(dot(nr, uSunDir), 0.0) * sunVis + skyLight(nr, 1.0))
                + uSunCol * sunVis * mix(0.3, 0.12, smoothstep(0.85, 0.35, headE)) * pow(max(dot(reflect(rd, nr), uSunDir), 0.0), 24.0);
        col = mix(col, rc, gStreamHead);
    }
    // (round 3) heaped 3D white water of the stream (stream block): opaque aerated water - bright diffuse, light
    // bleeding through (the shaded flanks are grey-blue, not black), the hollows between the heaps occluded
    if (gWWVis > 0.0 && gWWHit > 0.0) {
        vec3 nw = gWWN;
        float ndl = max(dot(nw, uSunDir), 0.0);
        // (i10) lit by a near-grey sky (aerated water scatters many times: the saturated blue ambient alone turned the
        // shaded flanks of the heaps blue-speckled)
        vec3 skW = skyLight(nw, 1.0);
        skW = mix(vec3(dot(skW, vec3(0.2126, 0.7152, 0.0722))), skW, 0.5);
        // (review rv5) tone: aerated lumps bright, thinner glassy water between them greyer / greener (the 3D white water
        // was an even white, p25-p75 224-234 in film shot 4 - flat paint; the cascade shades grey on its flanks)
        float fpW = max(length(gWWP - ro) * gPixAng, gWWFoot);
        float wt1 = mix(0.5, texture(uNoise3, vec3(gWWP.xz * vec2(0.55, 0.55), 0.13 + uTime * 0.05).xzy).b, 1.0 - smoothstep(0.08, 0.25, fpW));
        float wt2 = mix(0.5, texture(uNoise3, vec3(gWWP.xz * 0.45, 0.71 + uTime * 0.11).xzy).a, 1.0 - smoothstep(0.05, 0.15, fpW));   // (rv6: 1.6 -> 0.45, ~14 cm bubbles - the 4 cm ones were salt-and-pepper)
        float wtn = saturate(0.5 + 1.8 * (wt1 - 0.525) + 1.2 * (0.5 - wt2));
        vec3 wAlb = mix(vec3(0.40, 0.45, 0.44), vec3(0.68, 0.69, 0.70), smoothstep(0.15, 0.75, wtn));
        vec3 wc = wAlb * (uSunCol * sunVis * (ndl * gWWSh * mix(0.75, 1.0, gWWOcc) * 0.92 + 0.06) + skW * 0.95 * gWWOcc);
        // a boulder breaking the surface: dark wet granite, a wet sheen
        if (gWWRock > 0.5) {
            // (i12: textured at the hit point - the water plane's p smeared it into horizontal streaks at grazing views)
            float fpR = length(gWWP - ro) * gPixAng;
            float gr = mix(0.475, texture(uNoise3, gWWP * 0.9 + 0.3).r, 1.0 - smoothstep(0.03, 0.1, fpR)) * 0.6
                     + 0.4 * mix(0.475, texture(uNoise3, gWWP * 0.3 + 0.7).r, 1.0 - smoothstep(0.1, 0.3, fpR)) + 0.025;
            // (wet and near-black at the waterline, its top above the splash dry pale granite)
            // (r4 R9: near-black only in the splash band (< ~6 cm), grey wet granite 0.20-0.26 above - not black holes)
            vec3 ra = mix(vec3(0.075, 0.074, 0.069), vec3(0.26, 0.252, 0.235), smoothstep(0.02, 0.14, gWWH)) * mix(0.75, 1.25, gr);
            // (review rv5) granite, not a smooth egg: rough facets (normal noise ~0.5 / 0.15 m), grey-green lichen
            // patches on the dry top, dark streaks; resolved only (footprint)
            float vRk = 1.0 - smoothstep(0.03, 0.10, fpR);
            nw = normalize(nw + ((texture(uNoise3, gWWP * 0.75 + 0.4).rgb - 0.5) * 0.9 + (texture(uNoise3, gWWP * 2.2 + 0.1).rgb - 0.5) * 0.45) * vRk);
            ndl = max(dot(nw, uSunDir), 0.0);
            float lchR = smoothstep(0.56, 0.66, texture(uNoise3, gWWP * 0.35 + 2.3).r) * smoothstep(0.2, 0.5, gWWH) * vRk;
            ra = mix(ra, vec3(0.15, 0.16, 0.12), lchR * 0.6);
            ra *= mix(1.0, mix(0.55, 1.1, smoothstep(0.35, 0.65, texture(uNoise3, vec3(gWWP.x * 0.8, gWWP.y * 0.25, gWWP.z * 0.8) + 0.9).g)), vRk);
            wc = ra * (uSunCol * sunVis * ndl * gWWSh + skyLight(nw, 1.0) * gWWOcc)
               + uSunCol * sunVis * gWWSh * 0.3 * pow(max(dot(reflect(rd, nw), uSunDir), 0.0), 24.0);
        }
        col = mix(col, wc, gWWVis);
    }
    // aerated, milky turquoise water around the plunge pool (falls.glsl poolFoam)
    if (gPoolAer > 0.0) {
        vec3 aerC = vec3(0.32, 0.66, 0.62) * (uSunCol * max(uSunDir.y, 0.0) * sunVis * 0.55 + skyLight(vec3(0.0, 1.0, 0.0), 1.0));
        col = mix(col, aerC, gPoolAer * 0.5 * (1.0 - foamW));
    }
    return col;
}

