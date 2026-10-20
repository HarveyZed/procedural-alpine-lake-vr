// ---------------------------------------------------------------------------
// water.glsl : lake surface waves (deep/capillary dispersion), gust patches,
// optical constants of a clear glacial lake.
// ---------------------------------------------------------------------------
uniform vec2 uWaterWind;      // wind direction over the lake (unit)
uniform float uWaveStrength;  // overall roughness multiplier

// absorption + scattering (1/m) and an effective single-scatter radiance factor
// (r4 W1: blue absorbed faster than green - the bed seen through 1-3 m turns green (drop-off #5f8061), not cyan-grey)
// (water5: clearer: (0.30, 0.055, 0.085) -> (0.225, 0.030, 0.038) - the camera's broad R / G bands see the low-absorption
// end of pure water's spectrum (a(570) .07, a(600) .24); with half the column scattering (water_shade wSca) the shallows
// are clear (the bed golden-olive, not green-grey), the deep in-scatter ceiling keeps its level, deep water a little bluer)
const vec3 W_ABS = vec3(0.225, 0.030, 0.038);
const vec3 W_SCA = vec3(0.095, 0.115, 0.118);
uniform vec3 uWaterIns;       // in-scatter tint scale (tunable look)
// look2 tunables (render.py Look.tune -> uTune[16]); see render.py for the meaning of each slot
uniform float uTune[24];

float waterGust(vec2 p) {
    vec2 w = uWaterWind;
    vec2 gp = p - w * uTime * 2.2;
    // (r4 C6: + cat's-paw patches ~30-80 m (0.018 / m): the near water is not one flat teal)
    float g = texture(uNoise3, vec3(gp * 0.0016, 0.21)).r * 0.55 + texture(uNoise3, vec3(gp * 0.0051, 0.63)).r * 0.27
            + texture(uNoise3, vec3(gp * 0.018 - uTime * 0.01, 0.47)).r * 0.18;
    return smoothstep(0.45, 0.75, g);     // (r4 W5: 0.38 / 0.72 -> fewer, smaller gust patches)
}

vec3 waterNormal(vec2 p, float foot, out float gust) {
    vec2 w = uWaterWind;
    gust = waterGust(p);
    vec2 grad = vec2(0.0);
    float amp = mix(0.18, 1.0, gust) * uWaveStrength;
    for (int i = 0; i < 22; i++) {
        float fi = float(i);
        float lambda = 5.0 * pow(0.80, fi);           // 5 m .. ~4 cm
        float k = TAU / lambda;
        float omega = sqrt(9.81 * k + 7.28e-5 * k * k * k);
        float ang = (hash11(fi * 7.31 + 0.5) - 0.5) * 1.6;
        vec2 d = rot2(ang) * w;
        // long waves are gentle (mirror-like far water), short ripples steeper
        float A = lambda * mix(0.0035, 0.0075, smoothstep(1.5, 0.15, lambda)) * (0.55 + 0.9 * hash11(fi * 3.17 + 1.0));
        float filt = smoothstep(1.2, 3.5, lambda / foot);
        if (filt <= 0.0) break;
        float ph = k * dot(d, p) - omega * uTime + hash11(fi * 1.7) * TAU;
        // slightly trochoidal (sharper crests)
        float s = sin(ph), c = cos(ph);
        grad += d * (A * k * (c + 0.25 * sin(2.0 * ph))) * filt;
    }
    // irregular small-scale chop (breaks up the periodic sum): analytic gradient noise
    // (finite differences of the trilinear noise texture gave blocky square cells)
    vec2 cp = p * 1.3 - w * uTime * 0.7;
    vec3 g1 = gnoised(cp * 0.9 + vec2(0.0, uTime * 0.06));
    vec3 g2 = gnoised(cp * 2.3 + vec2(uTime * 0.09, 3.7));
    float f1 = smoothstep(1.2, 3.5, 0.85 / foot), f2 = smoothstep(1.2, 3.5, 0.33 / foot);
    grad += (g1.yz * (1.17 * 0.010) * f1 + g2.yz * (2.99 * 0.004) * f2) * mix(0.3, 1.0, gust);
    grad *= amp;
    return normalize(vec3(-grad.x, 1.0, -grad.y));
}


// wave slopes filtered by the ANISOTROPIC pixel footprint: across the view it is t * pixAng, along the
// view t * pixAng / |rd.y| (up to ~300x longer at grazing angles). The old isotropic footprint (with
// |rd.y| clamped to 0.02) kept 0.3-3 m ripples unresolved-but-unfiltered on the far water: their
// back-facing facets were then flipped toward the viewer and the reflection jumped to ~20 deg up
// (the pale band along the far shore, film shots 0 / 1).
// fwd = horizontal view direction (unit), fa / fl = footprint across / along it (m).
// sv = slope variance of the filtered-out waves projected on fwd (the unresolved roughness that
// smears the reflection vertically), gust = wind patch strength.
vec3 waterNormalA(vec2 p, vec2 fwd, float fa, float fl, out float gust, out float sv) {
    vec2 w = uWaterWind;
    gust = waterGust(p);
    vec2 side = vec2(-fwd.y, fwd.x);
    vec2 grad = vec2(0.0);
    float amp = mix(0.12, 0.85, gust) * uWaveStrength;     // (r4 W5: gust floor 0.18 / 1.0 -> 0.12 / 0.85)
    sv = 0.0;
    for (int i = 0; i < 22; i++) {
        float fi = float(i);
        // (r4 W10: wavelength jitter 0.9-1.1 - the exact geometric ladder of near-aligned sines made a grating)
        float lambda = 5.0 * pow(0.80, fi) * (0.9 + 0.2 * hash11(fi * 5.1));   // 5 m .. ~4 cm
        float k = TAU / lambda;
        // (r4 W10: +-75 deg spread below 1.5 m (was +-46): no regular diagonal comb of sky streaks)
        float ang = (hash11(fi * 7.31 + 0.5) - 0.5) * mix(1.6, 2.6, smoothstep(1.6, 1.4, lambda));
        vec2 d = rot2(ang) * w;
        float A = lambda * mix(0.0035, 0.0075, smoothstep(1.5, 0.15, lambda)) * (0.55 + 0.9 * hash11(fi * 3.17 + 1.0));
        // (r4 W5: calmer shore views - waves over 1.5 m at 60 % (the marbled swell from above; 35 % left the mid water glassy, hp 0.19 vs a 0.34-0.44 target), 0.15-0.6 m ripples x0.7
        // (the crumpled-foil chop near the shore))
        A *= mix(0.6, 1.0, smoothstep(1.5, 0.6, lambda)) * mix(1.0, 0.7, smoothstep(0.7, 0.55, lambda) * smoothstep(0.12, 0.17, lambda));
        float ca = dot(d, fwd), cs = dot(d, side);
        float footD = sqrt(cs * cs * fa * fa + ca * ca * fl * fl);
        // (r2b: filter from 2 footprints, full at 6: the 1.2..3.5 ramp kept waves down to ~1 px, which
        // flickered frame to frame (shot 8: 10.5 % of the water pixels changed > 20 levels per frame))
        float filt = smoothstep(2.0, 6.0, lambda / footD);
        float sl = A * k;
        sv += 0.5 * sl * sl * (1.0 - filt * filt) * ca * ca;
        if (filt <= 0.0) continue;
        float omega = sqrt(9.81 * k + 7.28e-5 * k * k * k);
        float ph = k * dot(d, p) - omega * uTime + hash11(fi * 1.7) * TAU;
        float s = sin(ph), c = cos(ph);
        grad += d * (A * k * (c + 0.25 * sin(2.0 * ph))) * filt;
    }
    // irregular small-scale chop (isotropic noise: filtered by the larger, along-view footprint)
    float fm = max(fa, min(fl, fa * 40.0));
    vec2 cp = p * 1.3 - w * uTime * 0.7;
    float f1 = smoothstep(2.0, 6.0, 0.85 / fm), f2 = smoothstep(2.0, 6.0, 0.33 / fm);
    if (f1 > 0.0) {
        vec3 g1 = gnoised(cp * 0.9 + vec2(0.0, uTime * 0.06));
        vec3 g2 = f2 > 0.0 ? gnoised(cp * 2.3 + vec2(uTime * 0.09, 3.7)) : vec3(0.0);
        grad += (g1.yz * (1.17 * 0.010) * f1 + g2.yz * (2.99 * 0.004 * 0.7) * f2) * mix(0.3, 1.0, gust);
    }
    sv += (0.5 * (0.0117 * 0.0117) * (1.0 - f1 * f1) + 0.5 * (0.012 * 0.012 * 0.7 * 0.7) * (1.0 - f2 * f2)) * 0.5 * mix(0.09, 1.0, gust);
    grad *= amp;
    sv *= amp * amp;
    return normalize(vec3(-grad.x, 1.0, -grad.y));
}

// animated caustic pattern on the lake bed
// (r4 W6 / R8 / W3: a fine, soft net (~0.2-0.35 m cells) instead of ~1 m pool-floor cells; none at zero
// depth (the white kerb along the cliff foot), fading with depth; fp = pixel footprint (m): unresolved -> the mean)
float caustics(vec2 p, float depth, float fp) {
    float vis = 1.0 - smoothstep(0.025, 0.05, fp);
    float c = 2.0 * 0.095;                              // (the mean of the two octaves)
    if (vis > 0.0) {
        vec2 q = p * 3.0;
        float cc = 0.0;
        for (int i = 0; i < 2; i++) {
            vec2 o = vec2(uTime * 0.33, uTime * 0.21) * (i == 0 ? 1.0 : -1.3);
            vec3 v = voronoi2(q * (i == 0 ? 1.0 : 1.6) + o);
            cc += pow(saturate(1.0 - (v.y - v.x) * 2.2), 10.0);
        }
        c = mix(c, cc, vis);
    }
    return c * 0.3 * smoothstep(0.05, 0.5, depth) * exp(-depth * 0.4);
}
