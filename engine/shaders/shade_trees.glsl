// ---------------------------------------------------------------------------
// shade_trees.glsl : conifer shading (needs trees.glsl + lighting.glsl)
// foliage: layered self-shadowing - near model: the tree's own (coarse) branches are marched toward the sun
// and toward the sky (treeOwnTrans: every spray layer passed lets ~55 % (sun) / 65 % (sky) through -> dark
// interior, light mostly on the outer sun-side twig tips), a clump-scale normal (sun side / shaded side of each
// spray clump, 4-20 m), mid model: soft shadow of the whorl skirts above (treeOwnTransMid) and dark tier
// undersides, plus the analytic path through the crown cone; neighbours' shadows, olive-green lit sprays
// with yellow-green fresh tips, translucency when backlit; bark: small grey-brown scales with lichen
// (spruce/fir) or red-brown plates (larch). Look terms use the 1080p reference footprint (dc * TREE_REF_PIXANG), so the
// look does not change with the render size; only anti-aliasing fades use the true footprint.
// ---------------------------------------------------------------------------

// sample pass index for trees.glsl (declared there before the frame uniforms exist)
int treeSampleIdx() { return uSample; }
int treeDebugMode() { return uDebug; }

// transmittance of this tree's own branches and sprays from p along L (unit), through the coarse near
// model (nearest branch per whorl, no side shoots): every spray layer the ray passes lets a fraction
// 'pass' through (porous needle mats), the trunk / branch wood a little less. Layered self-shadowing:
// the sprays of the whorls above shade the ones below, the outer sun-side tips stay lit, the interior
// behind several layers is near-black
float treeOwnTrans(vec3 p, vec3 L, Tree tr, float dc, int steps, float pass) {
    float T = 1.0;
    float t = 0.03 + 0.05 * gRand;
    vec4 inf;
    float reach = treeReach(tr.R * treeLowWiden(tr), tr.H, tr.seed) + 0.3;
    for (int i = treeZero(); i < steps; i++) {
        vec3 q = treeLocal(p + L * t, tr);
        if (q.y > tr.H + 0.4 || q.y < -0.5 || dot(q.xz, q.xz) > reach * reach) break;
        float d = sdTreeNear(q, tr, 0.03, dc, 0.0, false, inf);
        float w = 0.02 + 0.012 * t;
        if (d < w) { T *= mix(pass, 1.0, max(d, 0.0) / w); t += w; }
        else t += min(d, 0.6);
        if (T < 0.02) break;
    }
    return T;
}
// mid model: soft shadow of the whorl skirts above (each tier shades the one below it, the outer
// sun-side edge of a tier stays lit); the surface is the average spray surface, so 'res' only darkens
// to a floor (porous sprays)
float treeOwnTransMid(vec3 p, vec3 L, Tree tr, float fp) {
    float res = 1.0;
    float t = 0.10 + 0.12 * gRand;
    vec4 inf;
    float reach = treeReach(tr.R, tr.H, tr.seed) + 0.3;
    for (int i = treeZero(); i < 12; i++) {
        vec3 q = treeLocal(p + L * t, tr);
        if (q.y > tr.H + 0.4 || dot(q.xz, q.xz) > reach * reach) break;
        float d = sdTreeMid(q, tr, 1.0, fp, false, inf);
        res = min(res, saturate(3.0 * d / t));
        if (res < 0.01) break;
        t += clamp(d, 0.08, 0.8);
    }
    return res;
}

// fissured bark: returns albedo, perturbs the normal (cylindrical frame around the trunk)
vec3 barkAlbedo(vec3 q, Tree tr, float t, inout vec3 n) {
    float ang = atan(q.z, q.x);
    float rr = max(length(q.xz), 0.05);
    float fpb = t * gPixAng;                           // pixel footprint (m): detail fades once sub-pixel
    float aF = 1.0 - smoothstep(0.004, 0.02, fpb);
    // unrolled bark coordinates: one noise tile around the trunk (seamless at the wrap), ~0.95 tile per metre
    // up: spruce / fir small roundish flaking scales (worley, ~4-7 cm on a mature trunk, finer up the
    // tapering stem), larch larger plates between deep furrows
    vec3 bc = vec3(ang / TAU, q.y * mix(0.95, 0.50, tr.larch), tr.seed * 3.0);
    float plates = texture(uNoise3, bc).a;
    float fis = texture(uNoise3, bc * vec3(1.0, 0.6, 1.0) + 0.37).b;
    float fine = texture(uNoise3, bc * vec3(4.0, 4.0, 1.0) + 0.71).r;
    float edge = 1.0 - smoothstep(0.25, 0.55, plates);           // gaps between the scales
    float crack = saturate(smoothstep(0.45, 0.65, fis) * mix(0.5, 1.0, tr.larch) + 0.6 * edge * (1.0 - 0.5 * tr.larch));
    crack = mix(0.30, crack, aF);
    // spruce / fir: grey-brown scales (trunks #282621 in shade, grey on the lit side),
    // larch: red-brown plates
    // (r5 T9: plate contrast x1.5 - scaly bark reads at 1:1)
    vec3 spruceB = mix(vec3(0.033, 0.031, 0.028), vec3(0.111, 0.107, 0.097), plates);
    vec3 larchB = mix(vec3(0.056, 0.044, 0.034), vec3(0.110, 0.088, 0.068), plates);
    vec3 alb = mix(spruceB, larchB, tr.larch);
    alb *= mix(1.0, 0.30, crack) * mix(1.0, 0.55 + 0.9 * fine, aF);
    // crustose lichen: pale grey-green patches (3-15 cm), more on the shaded north side and low on the trunk;
    // dark wet / soiled foot
    float ln = texture(uNoise3, vec3(ang / TAU * 2.0, q.y * 1.4, tr.seed * 9.0)).r;
    float lich = smoothstep(0.56, 0.68, ln) * saturate(0.55 - q.z / rr * 0.5) * (1.0 - smoothstep(2.0, 0.45 * tr.H, q.y));
    alb = mix(alb, vec3(0.17, 0.18, 0.145) * mix(0.8, 1.1, fine), lich * mix(0.15, 0.40, aF) * (1.0 - 0.7 * tr.larch));
    alb *= mix(0.65, 1.0, smoothstep(0.0, 0.7, q.y));
    if (fpb < 0.03) {
        // scale relief: tilt the normal by the plate height field (around / up the stem)
        vec3 radial = vec3(q.x, 0.0, q.z) / rr;
        vec3 tang = vec3(-radial.z, 0.0, radial.x);
        const float e = 0.004;
        float hx = texture(uNoise3, bc + vec3(e, 0.0, 0.0)).a - texture(uNoise3, bc - vec3(e, 0.0, 0.0)).a;
        float hy = texture(uNoise3, bc + vec3(0.0, e, 0.0)).a - texture(uNoise3, bc - vec3(0.0, e, 0.0)).a;
        float k = 2.5 * (1.0 - smoothstep(0.008, 0.03, fpb));
        n = normalize(n - (tang * hx + vec3(0.0, hy, 0.0)) * k);
    }
    return alb;
}

vec3 shadeTree(vec3 p, vec3 rd, float t, ivec2 cell, bool cheap) {
    Tree tr;
    getTree(cell, tr);
    float lod = treeLod(t), lod2 = treeLod2(t);
    float e = max(0.006, t * gPixAng * 0.8);
    float env;
    // tetrahedral normal (4 taps, (+-1,+-1,+-1) with an odd number of +1); a loop that is not unrolled, so
    // the tree distance function is compiled once here instead of four times
    vec3 n = vec3(1e-6);
    for (int i = treeZero(); i < 4; i++) {
        vec3 kk = 2.0 * vec3(((i + 3) >> 1) & 1, (i >> 1) & 1, i & 1) - 1.0;
        n += kk * sdTree(p + kk * e, tr, lod, lod2, env);
    }
    n = normalize(n);
    sdTreeInfo(p, tr, lod, lod2, env);
    vec4 inf = gTreeInfo;
    vec3 q = treeLocal(p, tr);
    float fir = treeFir(tr);
    float th = saturate(q.y / tr.H);
    float rid = hash12i(cell + ivec2(3, 9));
    float dc = treeCamDist(tr);
    float fp = dc * uPixAng;              // true footprint: anti-aliasing fades only
    float fpL = dc * TREE_REF_PIXANG;     // look footprint (1080p reference)
    bool near = inf.w < 1.5 && treeIsNear(tr, dc);
    // crowns about a pixel wide or less: per-tree contrast (and tip glitter) fades out
    float sub = 1.0 - smoothstep(0.5, 2.0, 2.0 * tr.R / max(fp, 1e-4));
    // several crowns per pixel: their lit / shaded sides, gaps and shadows average out (no sparkle)
    // (r5 T3 / T12: from fpL 2-6, was 0.8-3: distant crowns keep their lit side / shaded side and lit tops)
    float far2 = near ? 0.0 : max(smoothstep(2.0, 6.0, fpL), sub);
    float op = treeOpen(tr);
    float r = length(q.xz);
    float renv = max(crownR(tr, q.y, fir), 0.05);
    // (trees6) per-tree variety of the mid / distant forest (sun exposure from the height above the
    // neighbours, value / hue, lit brightness, grey old trees): per-tree constants (no shimmer), ramped in by
    // camera distance from 60 to 140 m (the same on both models across the near / mid switch; the close-up
    // hero trees unchanged) and faded with the crowns' per-tree contrast once they are ~1 px (sub)
    float kV = smoothstep(60.0, 140.0, dc) * (1.0 - sub);
    float relH = treeRelH(cell, tr.dens);
    if (uDebug == 11) return vec3(inf.x / 3.0, inf.y, inf.z * 0.5 + 0.5) * (near ? 1.0 : 0.3);
    if (uDebug == 10) return vec3(tr.H / 40.0, tr.R / 5.0, tr.larch * 0.5 + fir * 0.25 + (near ? 0.2 : 0.0));
    if (uDebug == 12) return (dc > 30.0 && dc < 300.0) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 13) return (dc > 40.0 && dc < 120.0) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 14) return dc > 600.0 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 15) return dc < 60.0 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    if (uDebug == 16) return (dc >= 60.0 && dc <= 600.0) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);

    // ---- sun visibility: terrain, clouds, neighbouring crowns, this crown (analytic) ----
    float sun = terrainShadow(p, cheap ? 30 : 70) * cloudShadow(p);
    float fd = saturate(forestAt(tr.base.xz).r * 1.3 - 0.2);           // stand density
    // the side of the crown we see faces the camera: at a forest edge (open ground toward the camera)
    // that side gets the open sky, inside the stand it does not
    vec2 toCam = normalize(uCamPos.xz - tr.base.xz + 1e-3);
    float fdC = saturate(forestAt(tr.base.xz + toCam * 12.0).r * 1.3 - 0.2);
    float fdV = min(fd, mix(fdC, fd, 0.25));
    // statistical shade from the neighbours: lower crowns inside dense stands get little sun
    // (low sun: the neighbours shade all but the crown tops - golden hour)
    float sunLow = 1.0 - smoothstep(0.15, 0.60, uSunDir.y);
    // (trees6: by the height relative to the stand canopy - the dominant trees' crowns are sunlit far down,
    // the young / middle cohort stays in the canopy's shade; deeper shade low in the stand: dark gaps)
    float thE = th * mix(1.0, clamp(relH, 0.35, 1.45), 1.3 * kV);
    float lowS = 0.35 * (1.0 - 0.8 * sunLow) * mix(1.0, 0.63, kV);
    float tstat = mix(1.0, mix(lowS, 1.0, smoothstep(0.15 + 0.55 * sunLow, 0.80 + 0.15 * sunLow, thE)), mix(fd, 1.0, 0.8 * sunLow));
    float tsh = tstat;
    if (!cheap && t < 600.0 && sun > 0.0) {
        gTreeSkip = cell;
        float ts = treeShadow(p + n * 0.12);
        gTreeSkip = ivec2(-1000000);
        tsh = mix(ts, tstat, smoothstep(350.0, 600.0, t));
    }
    // sun flecks: the neighbours' crowns are porous, light falls through their gaps in patches
    // (30 cm patches: replaced by their mean 0.19 before they are sub-pixel)
    float dap = mix(smoothstep(0.50, 0.78, texture(uNoise3, p * 0.21 + 0.3).g), 0.19, smoothstep(0.08, 0.25, fp));
    tsh = max(tsh, dap * mix(0.75, 0.35, fd) * smoothstep(0.1, 0.6, th + 0.3));
    // (r5 T6: larch .30, was .45 - thin crowns, lit through)
    float sig = mix(mix(1.00, 0.70, op), 0.30, tr.larch);              // foliage extinction (1/m)
    // (the mid model's surface is the average spray surface, ~15 % inside the reach of the outer
    // sprays: their sun path is taken at the spray tips like the near model's)
    vec3 qS = q;
    if (inf.w > 1.5) qS.xz *= 1.0 / (0.85 * treeMidSlim(tr, dc));
    float Tself = exp(-sig * crownSunPath(qS, tr, fir));
    // near model: march the tree's own (coarse) branches toward the sun and toward the sky above
    // (layered self-shadowing / sky occlusion of the spray layers); the analytic cone path keeps a
    // small share for the unresolved needles and the branches the coarse model leaves out
    float Tsky = 1.0;
    bool marchK = near && !cheap;
    if (marchK) {
        // (r4 b1: .38 / .48 left more than half the crown pixels without sun - a 3-5 cm spray is crossed
        // in 2-3 march steps, so the effective layer transmittance was ~.38^2)
        // (r5 T3: .65 - more lit sprays inside the near crowns, target p95 ~190)
        // (r5 b4: .55 - with the higher near sun gain the r6 / r2 crowns keep more dark interior, pct 20/42/74/121/187)
        float pass = 0.55;
        float Ts = sun * tsh > 0.0 ? treeOwnTrans(p + n * 0.01, uSunDir, tr, dc, 18, pass) : 0.0;
        Tself = Ts * exp(-sig * 0.30 * crownSunPath(qS, tr, fir));
        vec3 rad = normalize(vec3(q.x, 0.0, q.z) + 1e-4);
        Tsky = treeOwnTrans(p + n * 0.01, normalize(vec3(rad.x * 0.6, 1.0, rad.z * 0.6)), tr, dc, 12, 0.65);
    }

    if (!near && !cheap && inf.w > 1.5) {
        float mK = 1.0 - smoothstep(0.8, 1.6, fpL);
        if (mK > 0.0) {
            float Tm = treeOwnTransMid(p + n * 0.02, uSunDir, tr, fp);
            Tself *= mix(1.0, mix(0.22, 1.0, Tm), mK);
        }
    }

    // twigs: bare inner part of the branchlets (close up only, like the geometry)
    float twig = 0.0;
    if (near && inf.x < 0.5) {
        float bare = 0.25 * (1.0 - smoothstep(0.006, 0.02, fp)) * (1.0 - 0.6 * inf.y);
        twig = 1.0 - smoothstep(bare - 0.04, bare + 0.06, inf.z);
    }
    if ((inf.x > 0.5 && inf.x < 2.5) || twig > 0.5) {
        // ---- trunk / branch wood ----
        vec3 alb;
        // (r4: x1.5 and warmer - shot 3 trunks measured 29,28,24 / 66,64,45 vs target 45,39,25 / 81,72,53)
        // (r5 T9: spruce / fir grey-brown x(1.45,1.35,1.10), r/g ~1.12 lit (target 81,72,53); the r4 x(1.85,1.55)
        // read as orange pine poles; larch keeps its red-brown)
        // (r5 b3: x(1.40,1.26,1.04) and less lichen - the b2 r6 trunks read as flat neutral-grey poles (80,80,72),
        // target trunk cores (34,25,14)-(53,42,33), lit edges (114,104,89))
        if (inf.x > 0.5 && inf.x < 1.5) alb = barkAlbedo(q, tr, t, n) * mix(vec3(1.40, 1.26, 1.04), vec3(1.85, 1.55, 1.10), tr.larch);
        else if (inf.x > 1.5 && inf.z > 0.5) {
            // dead lower branches: silver-grey wood with grey-green beard / crust lichen
            // (r5 T2: >= .12 grey, was .085 - black whips in shade)
            float lc = texture(uNoise3, q * 2.3 + tr.seed * 4.0).g;
            alb = mix(vec3(0.12, 0.12, 0.108), vec3(0.17, 0.172, 0.15), lc) * mix(0.85, 1.1, rid);
        }
        else alb = mix(vec3(0.060, 0.058, 0.052), vec3(0.090, 0.086, 0.076), rid);  // grey-brown twigs (r5 T11: was .028-.05)
        float ao = mix(0.30, 0.8, smoothstep(0.0, 0.5, th)) * terrainAO(tr.base.xz) * mix(0.75, 1.0, saturate(q.y * 0.5));
        ao *= mix(0.55, 1.0, 1.0 - tr.dens * 0.5) * mix(0.70, 1.0, saturate(r / renv)) * 1.6;
        // (r5 T9: near trunks keep readable bark in shade, ~(40,40,35) - not a featureless black cylinder)
        if (near) ao = max(ao, 0.42);
        return litSurface(alb, n, rd, sun * tsh * Tself, ao, 0.0, 0.8);
    }

    // ---- foliage ----
    // species colour (linear albedo; green, red:green ~0.65), slight per-tree variation
    vec3 spruce = mix(vec3(0.072, 0.092, 0.034), vec3(0.088, 0.110, 0.038), rid);
    vec3 firC = mix(vec3(0.062, 0.080, 0.034), vec3(0.074, 0.095, 0.040), rid);
    // (larch: muted yellow-green, lit #8c965b, never lime)
    // (r5 T6: ~1.5x the spruce albedo, h ~117 - the r4 larch was the spruce colour and never read as larch)
    vec3 larchC = mix(vec3(0.115, 0.140, 0.045), vec3(0.130, 0.155, 0.050), rid);
    vec3 alb = mix(mix(spruce, firC, fir), larchC, tr.larch);
    // (r4 b4: near crowns were ~.6 stop too dark in the mid-tones; x1.3 within 30 m, fading
    // out by 80 m so the mid model's level is unchanged)
    alb *= mix(1.3, 1.0, smoothstep(30.0, 80.0, t));
    // (close up spruce foliage is olive: red ~ green in sRGB)
    alb.r *= mix(1.10, 1.0, max(smoothstep(20.0, 45.0, t), tr.larch));
    float rv = hash12i(cell + ivec2(17, 5));
    alb *= mix(1.0, mix(0.70, 1.20, rv), (1.0 - sub) * (1.0 - 0.35 * smoothstep(0.8, 3.0, fpL)));
    // (r5 T5: per-crown hue - yellow-olive and blue-green crowns side by side, forest G-R std >= 4.3)
    float rh = hash12i(cell + ivec2(29, 61));
    alb *= mix(vec3(1.0), mix(vec3(1.07, 1.0, 0.86), vec3(0.93, 1.0, 1.10), rh), (1.0 - sub));
    if (kV > 0.0) {
        // (trees6) dark blue-green old spruce / fir beside lighter yellow-olive crowns (value and hue go
        // together), a few grey old / dying trees (thin needles, lichen)
        float v2 = hash12i(cell + ivec2(67, 23));
        float bl = saturate(1.25 * (1.0 - rv) + 0.5 * (v2 - 0.5)) * (1.0 - tr.larch);
        alb *= mix(vec3(1.0), mix(vec3(1.05, 1.02, 0.84), vec3(0.86, 0.96, 1.24), bl) * mix(0.82, 1.14, v2), kV * 1.4);
        float gr = step(hash12i(cell + ivec2(41, 77)), 0.035);
        alb = mix(alb, vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722))) * vec3(1.06, 1.0, 0.90) * 1.05, 0.55 * gr * kV);
    }
    // stand-scale patches (age, vigour, exposure): lighter olive and darker blue-green parts of the forest
    float pv = smoothstep(0.30, 0.70, textureLod(uNoise3, vec3(tr.base.xz * (1.0 / 420.0), 0.61), 0.0).g);
    alb *= mix(vec3(0.80, 0.80, 0.84), vec3(1.20, 1.15, 1.02), pv);
    vec3 radial = normalize(vec3(q.x, 0.0, q.z) + 1e-4);
    vec3 nc = normalize(radial + vec3(0.0, mix(0.30, 0.5, fir), 0.0));    // crown (cone) normal
    float depth = max(renv - r, 0.0);                                    // metres inside the envelope
    float aoC, lit;
    float farF = 0.0;
    float far = 0.0;
    // clump-scale normal: gradient of the tree distance over ~0.15-0.45 m (the normal of the foliage mass
    // of a branch / spray clump, not of a single shoot): each clump is lit on its sun side
    // and dark on the other, the crown reads as 3D masses instead of an evenly lit fuzz
    vec3 nCl = nc;
    float wCl = near && inf.x < 0.5 ? 0.9 * smoothstep(4.0, 20.0, t) : 0.0;
    if (wCl > 0.0 && !cheap) {
        float e2 = clamp(0.18 + 0.006 * t, 0.18, 0.75);
        vec3 m = vec3(0.0);
        for (int i = treeZero(); i < 4; i++) {
            vec3 kk = 2.0 * vec3(((i + 3) >> 1) & 1, (i >> 1) & 1, i & 1) - 1.0;
            m += kk * sdTree(p + kk * e2, tr, lod, lod2, env);
        }
        nCl = normalize(m + nc * 1e-3);
    }
    if (near) {
        // inf.y along the branch (0 inner .. 1 tip), inf.z along the branchlet, inf.w side
        float fx = inf.y, fs = inf.z;
        float up = inf.w;
        float bt = saturate(abs(up) * 2.0 - 1.0);     // per-branchlet tone
        up = up >= 0.0 ? 1.0 : -1.0;
        bool core = inf.x > 2.5;
        // fresh light growth at the branchlet tips (more on the outer part of the branch)
        // in patches (this year's shoots on the vigorous outer sprays), mostly on the upper side
        float tn = texture(uNoise3, q * 0.8 + tr.seed * 5.0).g;
        // (r5 T3: wider noise gate .30-.60, was .55-.80 - more of the outer sprays carry fresh tips)
        float tip = smoothstep(0.35, 1.0, fs) * smoothstep(0.35, 0.95, fx) * smoothstep(0.30, 0.60, tn) * mix(0.35, 1.0, step(0.0, up));
        // (r4: saturated yellow-green fresh tips, the old x(1.45,1.55,1.35) read as pale cream "moss")
        // (r5 T3: a little warmer, x(1.55,1.48,0.95): lit r/g ~0.95, still saturated)
        // (r5 b5: (1.50,1.44,1.08) - the (1.55,1.48,.95) tips read as lime streaks in shot03; b/g still < sprays)
        alb = mix(alb, alb * vec3(1.50, 1.44, 1.12), tip * (1.0 - 0.7 * tr.larch));
        // needle-scale self shadowing (faded out once it is sub-pixel)
        float ns = texture(uNoise3, q * 6.0 + tr.seed * 3.0).b;
        float nA = (1.0 - smoothstep(0.006, 0.02, fp)) * 0.5;
        float speck = mix(1.0, mix(0.35, 1.25, smoothstep(0.25, 0.75, ns)), nA);
        // sky occlusion: deep inside the crown, under a frond, close to the axis of a branchlet
        // the underside of a spray gets no direct sun (only what leaks through the needles)
        // (the crowns are mostly mid-tones: sky reaches well into the porous sprays; x0.92 beyond
        // 40 m matches the mid model's brightness at the switch)
        aoC = exp(-depth * (marchK ? 0.06 : 0.35)) * mix(0.43, 1.0, Tsky) * mix(0.70, 1.0, step(0.0, up)) * mix(0.80, 1.0, fs) * mix(0.68, 1.0, fx) * speck;
        lit = mix(0.22, 1.0, step(0.0, up)) * mix(0.63, 1.0, fx) * speck * mix(1.0, 0.92, smoothstep(35.0, 50.0, dc));
        if (core) {
            // dense inner crown: old needles in deep shade, a little light through the gaps
            float cn = texture(uNoise3, q * 1.3 + tr.seed * 2.0).r;
            aoC = mix(0.35, 0.60, cn) * exp(-depth * 0.4) * 1.5;
            lit = 0.15 * cn;
            n = nc;
        } else {
            // shoots are fuzzy needle brushes: their shading follows the spray (top lit, underside
            // dark) more than the round branchlet cylinder
            n = normalize(n * 0.55 + nc * 0.25 + vec3(0.0, 0.35 * up, 0.0));
            n = normalize(mix(n, normalize(nCl + nc * 0.25 + vec3(0.0, 0.15, 0.0)), wCl));
        }
        // (the deep interior seen through the gaps stays near-black; sprays keep 0.45)
        if (!core) { lit *= mix(0.62, 1.30, bt); aoC *= mix(0.80, 1.16, bt); alb *= mix(0.90, 1.10, bt); }
        if (!core) {
            // needle tufts: the lit spray tops break into small lit tufts and shaded needles between
            // them (~4 cm), replaced by their mean before they are sub-pixel
            float tf = texture(uNoise3, q * 1.6 + tr.seed * 13.0).b;
            float tfA = (1.0 - smoothstep(0.004, 0.014, fp)) * 0.5;
            lit *= mix(1.0, mix(0.35, 1.45, smoothstep(0.30, 0.70, tf)), tfA);
            // shoot clusters (~20 cm): the hanging branchlets are not evenly lit ropes; kept until the
            // clusters are ~4 px
            float tf2 = texture(uNoise3, q * 0.55 + tr.seed * 7.0).b;
            float tf2A = (1.0 - smoothstep(0.025, 0.06, fp)) * 0.4;
            lit *= mix(1.0, mix(0.40, 1.30, smoothstep(0.30, 0.70, tf2)), tf2A);
        }
        if (!core) {
            // shadows of the sprays above: the lit spray tops break into lit and shaded patches
            // (0.5-0.8 m), faded to their mean before they are ~6 px
            float sm = texture(uNoise3, q * vec3(0.10, 0.14, 0.10) + tr.seed * 6.0).r;
            float smA = (1.0 - smoothstep(0.05, 0.14, fp)) * (1.0 - 0.5 * tr.larch) * (marchK ? 0.0 : 1.0);
            // (mean ~0.93: the near model's lit level stays that of the mid model at the switch)
            lit *= mix(1.0, mix(0.42, 1.40, smoothstep(0.30, 0.70, sm)), smA);
        }
        aoC = max(aoC, marchK ? 0.10 : (core ? 0.14 : 0.38));
        lit *= mix(0.86, 1.0, smoothstep(20.0, 45.0, dc));
        // larch: open, thin-needled crown - sky reaches far into it
        aoC = mix(aoC, max(aoC, 0.65), 0.6 * tr.larch);
    } else {
        // mid / far: inf.y radial r/renv, inf.z position in the skirt (+1 top .. -1 lower edge)
        far = smoothstep(0.25, 1.2, fpL);
        float outer = smoothstep(0.35, 0.95, inf.y);
        float under = saturate(saturate(-inf.z) * (1.0 - far) * 2.5);
        // branch-scale clumps: lit spray tops / dark holes (faded out once they are sub-pixel)
        float cl = texture(uNoise3, q * 0.30 + tr.seed * 7.0).b;
        float clA = (1.0 - smoothstep(0.45, 0.90, fpL)) * (1.0 - smoothstep(0.25, 0.6, fp));
        float clump = mix(0.6, smoothstep(0.25, 0.75, cl), clA);
        // spray clumps vs gaps between them (mean ~1, like the near model's lit tips and dark holes)
        float cK = mix(1.0, mix(0.25, 1.50, smoothstep(0.25, 0.75, cl)), clA);
        // coarser clumps (1.5-3 m) once the fine ones are gone: the lit flank of a distant crown is broken
        // into lit branch clumps and dark holes, not a smooth cone (faded before they are ~2 px)
        float clB = (1.0 - clA) * (1.0 - smoothstep(0.45, 0.95, fpL)) * (1.0 - smoothstep(0.45, 0.95, fp));
        if (clB > 0.0) {
            float c2 = texture(uNoise3, q * 0.11 + tr.seed * 3.0).b;
            cK *= mix(1.0, mix(0.55, 1.30, smoothstep(0.3, 0.7, c2)), clB);
        }
        alb = mix(alb, alb * vec3(1.32, 1.30, 1.26), clump * outer * (1.0 - far) * 0.3 * (1.0 - sub));
        // (x0.86 / x0.92 near the near/mid switch: the near model shows its dark interior through the
        // gaps between the sprays, so its average sky access and lit fraction are lower)
        // (r4: darker undersides of the whorl skirts, 2x: tiers read at 60-300 m)
        const float uK = 2.0;
        aoC = mix(0.8, 1.0, outer) * mix(1.0, saturate(1.0 - 0.2 * uK), under) * mix(1.0, cK, 0.7) * 0.93 * mix(0.85, 1.0, far) * mix(0.86, 1.0, far);
        // porous crown seen from afar: part of the lit side is shadowed gaps (needle clumping)
        lit = mix(0.9, 1.0, outer) * mix(1.0, saturate(1.0 - 0.45 * uK), under) * cK * 0.93 * mix(0.85, 0.9, far) * mix(0.92, 1.0, far);
        // (r4: the crown interior seen between the outer clumps is dark, spires at 40-300 m)
        aoC *= mix(0.55, 1.0, outer * (1.0 - far));
        lit *= mix(0.775, 1.0, outer * (1.0 - far));
        // (r4 b3: the shaded flank of a 60-300 m spire is near-black: less sky fill)
        aoC *= mix(0.6, 1.0, far);
        // (r5 b4: lit mid / far crown flanks, slope forest p95 142 / 149)
        lit *= 1.55;
        aoC *= 1.25;
        if (uDebug == 36) return vec3(outer, under, clump);
        // the visible surfaces are mostly spray tops (like the near model's): normals tilt up
        n = normalize(n * mix(0.65, 0.15, far) + nc * mix(0.35, 0.85, far) + vec3(0.0, 0.30 * (1.0 - far), 0.0));
        // distant crowns: shade dominates the mix, the skylit shade is blue-green
        farF = far;
    }
    // distant forest (by distance, not footprint): unresolved skylit shade and needle sheen make it
    // grey / blue-green, not yellow-green
    float distF = smoothstep(200.0, 2000.0, t);
    // beyond a few tens of metres a pixel mixes lit sprays with skylit gaps: lit flanks read green /
    // blue-green rather than yellow-olive, and the shade picks up more sky (same for both models)
    float midF = smoothstep(15.0, 150.0, t);
    alb *= mix(vec3(1.0), vec3(0.97, 1.0, 1.04), midF);
    alb *= mix(1.0, 0.85, smoothstep(30.0, 120.0, t) * (1.0 - smoothstep(600.0, 1500.0, t)));
    // (larch at range: soft light yellow-green, never lime)
    // (trees6: x(1.10,1.15,1.00) beyond ~140 m, was (1.15,1.15,.95) - distant larch read lime, lit larch #8c965b)
    alb *= mix(vec3(1.0), mix(vec3(1.15, 1.15, 0.95), vec3(1.10, 1.15, 1.00), kV), tr.larch * smoothstep(20.0, 120.0, t));
    // (the grey / blue cast of distant forest is its unresolved sky-lit shade and sheen: the sunlit needles
    // keep their own colour - golden hour: saturated yellow-green lit flanks, not cream)
    vec3 albS = alb;
    if (kV > 0.0) {
        // (trees6) aerial perspective on the sunlit flanks too (not at golden hour): the distant lit crowns
        // greyer / bluer, not the 200 m yellow-green
        float dsK = 0.15 * distF * (1.0 - smoothstep(0.4, 0.6, sunLow)) * kV;
        albS = mix(albS, vec3(dot(albS, vec3(0.2126, 0.7152, 0.0722))) * vec3(0.96, 1.0, 1.10), dsK);
    }
    alb = mix(alb, vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722))), 0.22 * distF);
    // (the canopy seen from afar shows mostly its sunlit, sky-facing upper crowns: lighter)
    // (r5 T13: (1,1,1.04), was (0.97,1,1.15): distant shade green, b <= g - 3, not blue-grey)
    alb *= mix(vec3(1.0), vec3(1.0, 1.0, 1.04), distF);
    // stand-level sky occlusion (lower crowns inside dense stands)
    // (low sun: the low sky behind the camera is dim, the far stands' lower crowns get little sky light)
    aoC *= mix(1.0, mix(mix(0.50, 0.36, kV), 1.0, smoothstep(0.05, 0.75, thE)), max(fdV, 0.6 * sunLow * smoothstep(150.0, 500.0, t)));
    // (crowns at 20-300 m are mostly mid-tones: more sky light into the shade)
    aoC *= mix(1.0, mix(1.30, 1.0, smoothstep(150.0, 400.0, t)), smoothstep(10.0, 40.0, t));
    { float mr = smoothstep(20.0, 60.0, t) * (1.0 - smoothstep(300.0, 600.0, t)); aoC *= 1.0 + 0.18 * mr; lit *= 1.0 + 0.12 * mr; }
    aoC *= mix(1.0, 0.6, sunLow * smoothstep(300.0, 1000.0, t));
    float aoT = mix(terrainAO(tr.base.xz), 1.0, th * 0.6);
    float sunV = sun * mix(tsh, tstat, far2) * mix(Tself, mix(0.6, 0.35, sunLow), far2);
    // (r5 T6: larch x1.0, was .86 - light larch crowns)
    sunV *= mix(1.0, mix(0.76, 1.0, tr.larch), smoothstep(25.0, 180.0, t));
    // (trees6) per-tree lit brightness (needle density, crown orientation): not the same lit tip on every tree
    sunV *= 1.0 + 0.30 * (2.0 * hash12i(cell + ivec2(53, 13)) - 1.0) * kV;
    // (r5 T12: low sun lights the crown tops - the far shore's gold-topped trees over dark lower crowns)
    float gold = smoothstep(0.4, 0.6, sunLow);
    sunV *= mix(1.0, 1.4, smoothstep(0.70, 0.95, th) * gold);
    // forest seen from afar follows the terrain relief: slopes facing the sun are lit more
    if (far > 0.0) {
        vec3 nT = terrainNormal(tr.base.xz, 6.0, 0.0);
        float relief = saturate(0.30 + 0.85 * dot(nT, uSunDir)) / max(0.30 + 0.85 * uSunDir.y, 0.3);
        sunV *= mix(1.0, 0.45 + 0.55 * relief, far);
    }
    if (uDebug == 7) return vec3(sun, tsh, Tself);
    if (uDebug == 9) return vec3(crownSunPath(q, tr, fir) / 3.0, r / renv * 0.5, q.y / tr.H);
    if (uDebug == 8) return vec3(aoC, lit, near ? 1.0 : 0.0);
    float ndl = dot(n, uSunDir);
    // (mid model: the shaded side of a crown is mostly gaps into the dark interior -> less wrap)
    float wrapD = near ? saturate(ndl * 0.85 + 0.15) : saturate(ndl * mix(0.85, 0.65, far) + mix(0.22, 0.15, far));
    // mid model near the switch: its smooth skirt faces the camera and the sun together, the near model's
    // sprays point every way -> flatten its lit / shaded contrast to the near model's
    // (r4 b3: only 40 % of that flattening - the mid spires need their dark shaded flank)
    if (!near) wrapD = mix(mix(wrapD, 0.10 + 0.75 * wrapD, 0.4), wrapD, farF);
    if (!near) wrapD = mix(wrapD, saturate(ndl * 0.95 + 0.05), 0.6 * smoothstep(100.0, 250.0, t) * (1.0 - far2));
    wrapD = mix(wrapD, 0.65, 0.6 * far2);
    lit = mix(lit, 0.8, 0.5 * far2);
    aoC = mix(aoC, 0.9, 0.5 * far2);
    // sky light inside the crown has passed through needles: green, little blue
    // (r5 T13: distF (0.95,1,1.06), was (0.90,1,1.20): green, not blue-grey distant shade)
    vec3 skyT = mix(vec3(0.98, 1.0, 0.95), vec3(0.95, 1.0, 1.06), distF) * mix(vec3(1.0), vec3(1.0, 1.0, 0.92), (1.0 - aoC) * (1.0 - distF));
    skyT *= mix(vec3(1.0), vec3(1.12, 1.12, 1.04), midF);
    // (r4: the shaded interior of near crowns is olive-brown, r >= g: old needles, twigs, bark)
    // (r5 b3 / b4: (1.22,1.05,1.0), was (1.30,1.05,.70) - b2 near darks (16,18,6) vs target (11,16,13) / (16,14,7))
    skyT *= mix(vec3(1.0), vec3(1.22, 1.05, 1.0), saturate(1.0 - aoC) * (1.0 - midF));
    // sunlit needles: diffuse (olive) plus a broad, nearly white waxy sheen that greys the lit flanks
    // (r4: x1 close up and at low sun - x2 creamed the near tips and the golden-hour forest; x2 at range at
    // midday: paler lit flanks)
    // (r5 T3: near sun gain 1.45 fading to 1.20 by 80 m - near crowns p95 ~190)
    // (b4: near 1.90, was 1.45 - shot03 lit sprays p95 148 vs target 189, lit band 7 % vs 17 %)
    vec3 col = albS * uSunCol * sunV * wrapD * lit * mix(1.90, 1.20, smoothstep(20.0, 80.0, t));
    // (r5 b3: near sunlit sprays less yellow - target lit10 (180,178,135) / (168,176,136), b2 (167,177,122):
    // r/g ~1.0, b/g ~.76; the needles' gloss greys their sunlit side)
    // (b4: mid range too - target lit flanks b/g ~.79-.83, b3 r4 / shot02 ~.62; faded out by 250-600 m)
    // (trees6: beyond 600 m keep a green / blue-green lit flank (midday), not the yellow-lime of the bare albedo:
    // aerial perspective goes the other way, lit spruce r/g ~0.89, b/g ~0.70)
    vec3 farTint = mix(vec3(1.0), vec3(0.91, 1.0, 1.32), kV * (1.0 - smoothstep(0.4, 0.6, sunLow)));
    col *= mix(mix(vec3(1.08, 1.0, 1.70), vec3(1.03, 1.0, 1.25), smoothstep(30.0, 90.0, t)), farTint, smoothstep(250.0, 600.0, t));
    vec3 colA = alb * skyLightFoliage(n, aoT) * aoC * skyT;
    // (r5 T13: sheen (.010,.011,.009) x <= 1, was (.016,.017,.018) x 1.4: frosted blue-grey crowns at 50-150 m)
    float sheenK = mix(1.0, saturate(1.4 * saturate(1.0 - 1.6 * sunLow)), smoothstep(40.0, 150.0, t));
    col += uSunCol * sunV * wrapD * lit * vec3(0.010, 0.011, 0.009) * sheenK * mix(0.35, 1.0, smoothstep(15.0, 90.0, t));
    if (uDebug == 30 || uDebug == 33) return vec3(sunV, wrapD, lit);
    if (uDebug == 37) return vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722)) * 10.0, dot(skyLight(n, aoT), vec3(0.2126, 0.7152, 0.0722)), wrapD);
    if (uDebug == 38) return vec3(sunV, aoC, lit);
    if (uDebug == 31 || uDebug == 34) return vec3(aoC, Tself, tsh);
    if (uDebug == 32 || uDebug == 35) return vec3(dot(alb, vec3(0.2126, 0.7152, 0.0722)) * 10.0, dot(skyLight(n, aoT), vec3(0.2126, 0.7152, 0.0722)), n.y * 0.5 + 0.5);
    // diffuse transmission through sunlit needles (larch is thin and glows more)
    col += albS * uSunCol * sunV * saturate(-ndl * 0.6 + 0.2) * mix(0.22, 0.45, tr.larch) * 1.3;
    // needle translucency when backlit: thin outer shoots / silhouette rims glow
    float bl = pow(saturate(dot(rd, uSunDir)), 2.5);
    float rim = pow(1.0 - abs(dot(n, rd)), 2.0);
    // (r5 T3 / T12: near x1.6, was 1.3; x2 at low sun - gold rim light on the sun-side sprays)
    // (r5 b5: (.95,1.12,.58), was (.9,1.15,.45) - backlit sprays glowed lime in shot03)
    col += uSunCol * sunV * bl * albS * vec3(0.95, 1.12, 0.58) * (0.15 + 0.85 * rim) * mix(0.8, 0.2, farF) * mix(1.3, 1.6, near ? 1.0 : 0.0) * mix(1.0, 2.0, gold);
    // waxy needle sheen
    vec3 hv = normalize(uSunDir - rd);
    col += uSunCol * sunV * pow(max(dot(n, hv), 0.0), 16.0) * 0.008 * lit;
    // distant forest: the forested slopes are dark (forest / water luminance 0.25-0.35)
    // (r5 T3: on the sky-lit part only - the sunlit crowns keep their lit side, slope forest p95 ~140)
    float dD = mix(1.0, 0.72, smoothstep(120.0, 500.0, t)) * mix(1.0, 0.80, smoothstep(600.0, 1800.0, t));
    return col + colA * dD;
}
