// ---------------------------------------------------------------------------
// sky.glsl : single-scattering atmosphere (Rayleigh + Mie + ozone), aerial
// perspective and volumetric cumulus clouds.
// ---------------------------------------------------------------------------
#define R_E 6360000.0
#define R_A 6420000.0
#define LAKE_ALT 1800.0
const vec3 BETA_R = vec3(5.802e-6, 13.558e-6, 33.1e-6);
const vec3 BETA_O = vec3(0.650e-6, 1.881e-6, 0.085e-6);
#define H_R 8000.0
#define H_M 1200.0

uniform vec3 uSunDir;
uniform float uSunIntensity;   // radiometric scale of the sun
uniform float uHaze;           // Mie density multiplier (1 = clear alpine air)
uniform float uCloudCover;     // 0..1
uniform vec2 uWind;            // cloud drift m/s
uniform vec3 uSunCol;          // sun radiance reaching the valley (from probe)
uniform vec3 uSkyAmb;          // sky irradiance on an up-facing surface (from probe)
uniform vec3 uSkyHorizon;      // average horizon radiance (from probe)
uniform float uSkyBoost;       // approx. multiple scattering / camera sky handling
uniform float uSkyDesat;

vec3 betaM() { return vec3(3.996e-6) * uHaze; }

vec2 raySphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd), c = dot(ro, ro) - r * r;
    float d = b * b - c;
    if (d < 0.0) return vec2(1e30, -1e30);
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}
vec3 toPlanet(vec3 p) { return p + vec3(0.0, R_E + LAKE_ALT, 0.0); }

vec3 densAt(vec3 pp) {
    float h = length(pp) - R_E;
    float oz = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
    return vec3(exp(-h / H_R), exp(-h / H_M), oz);
}

vec3 transToSun(vec3 pp) {
    float h = length(pp) - R_E;
    vec3 up = pp / (R_E + h);
    float mu = dot(up, uSunDir);
    float e = degrees(asin(clamp(mu, -1.0, 1.0)));
    if (e < -1.0) return vec3(0.0);
    float am = 1.0 / (max(mu, 0.0) + 0.50572 * pow(max(e, -0.5) + 6.07995, -1.6364));
    vec3 od = BETA_R * H_R * exp(-h / H_R) + betaM() * 1.11 * H_M * exp(-h / H_M) + BETA_O * 15000.0;
    return exp(-od * am);
}

float phaseR(float mu) { return 3.0 / (16.0 * PI) * (1.0 + mu * mu); }
float phaseHG(float mu, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}
float phaseM(float mu) {
    float g = 0.76, g2 = g * g;
    return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

// in-scattered radiance along [0,tmax] and transmittance
vec3 atmoScatterRaw(vec3 ro, vec3 rd, float tmax, int n, out vec3 trans) {
    vec3 pp = toPlanet(ro);
    vec2 s = raySphere(pp, rd, R_A);
    vec2 g = raySphere(pp, rd, R_E);
    float tEnd = s.y;
    if (g.x > 0.0) tEnd = min(tEnd, g.x);
    tEnd = min(tEnd, tmax);
    float mu = dot(rd, uSunDir);
    float pr = phaseR(mu), pm = phaseM(mu);
    vec3 bm = betaM();
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    vec3 od = vec3(0.0);
    // non-uniform steps: denser near the camera
    float prevT = 0.0;
    for (int i = 0; i < n; i++) {
        float f = (float(i) + 1.0) / float(n);
        float t1 = tEnd * f * f;
        float dt = t1 - prevT;
        float tm = 0.5 * (t1 + prevT);
        prevT = t1;
        vec3 q = pp + rd * tm;
        vec3 d = densAt(q) * dt;
        od += d;
        vec3 tv = exp(-(BETA_R * od.x + bm * 1.11 * od.y + BETA_O * od.z));
        vec3 ts = transToSun(q);
        sumR += tv * ts * d.x;
        sumM += tv * ts * d.y;
    }
    trans = exp(-(BETA_R * od.x + bm * 1.11 * od.y + BETA_O * od.z));
    return uSunIntensity * (sumR * BETA_R * pr + sumM * bm * pm);
}

// Aerial perspective (finite paths) must use the same radiance calibration as the visible sky
// (skyRadiance scales the single-scattering result by uSkyBoost); without it distant ridges only
// lose blue to extinction and turn beige instead of fading into a sky-blue haze.
// Infinite paths (sky dome, lighting probe) keep the raw value: their callers apply their own gain.
#define AERIAL_GAIN 0.6
vec3 atmoScatter(vec3 ro, vec3 rd, float tmax, int n, out vec3 trans) {
    vec3 s = atmoScatterRaw(ro, rd, tmax, n, trans);
    // finite paths: the distant ranges should be a greyer blue than the raw single scattering
    // (far range #7c93b4, OKLCh h 257 / C 0.056)
    // (shaded faces 2-6 km out are already sky-blue, lit / shade only ~1.7x: the in-scatter
    // over the middle distance is lifted, the 15+ km ranges keep their calibrated value)
    if (tmax < 1e8) {
        float mid = smoothstep(700.0, 3000.0, tmax) * (1.0 - smoothstep(5000.0, 15000.0, tmax));
        // (session 4: 0.55 -> 0.25: the lift filled the shadows of the 1-6 km faces - rock p10 0.14 in shot 7
        // against a target of 0.05; the "melted wax" look of the head wall)
        // (session 5: 0.25 -> 0.12: the in-scatter alone was ~0.07 of the darkest rock's 0.12 in shot 7)
        s *= uSkyBoost * AERIAL_GAIN * (1.0 + 0.12 * mid * smoothstep(0.25, 0.5, uSunDir.y));
        // (session 6: less desaturation over the middle distance: shaded faces 1-6 km out read sky-blue,
        // target #5c729d, instead of grey-teal; the 15+ km ranges keep the calibrated 0.25)
        s = mix(s, vec3(luma(s)), 0.25 * (1.0 - 0.7 * mid));
    }
    return s;
}

vec3 skyRadiance(vec3 ro, vec3 rd, int n) {
    vec3 tr;
    vec3 c = atmoScatter(ro, rd, 1e9, n, tr) * uSkyBoost;
    c = mix(c, vec3(luma(c)), uSkyDesat);
    float mu = dot(rd, uSunDir);
    float sd = smoothstep(0.999966, 0.999985, mu);
    c += uSunCol * sd * 600.0;
    return c;
}

// ---------------------------------------------------------------------------
// clouds : fair-weather cumulus. flat bases, cauliflower domes whose height
// follows the cell strength, billowy worley erosion, multi-octave scattering.
// ---------------------------------------------------------------------------
#define CL_BOT 2350.0
#define CL_TOP 4300.0

float remap(float v, float a, float b, float c, float d) { return c + (v - a) / (b - a) * (d - c); }

// large-scale weather: where clouds can exist (0..1)
float cloudCoverage(vec2 xz) {
    vec2 q = xz + uWind * uTime;
    float a = texture(uNoise3, vec3(q * 0.000028, 0.37)).r;
    float b = texture(uNoise3, vec3(q * 0.00009 + 0.3, 0.71)).r;
    float w = a * 0.6 + b * 0.4;
    return saturate(uCloudCover + (w - 0.5) * 0.9);
}

float cloudDensity(vec3 p, bool detail) {
    float hf = (p.y - CL_BOT) / (CL_TOP - CL_BOT);
    if (hf < 0.0 || hf > 1.0) return 0.0;
    float wc = cloudCoverage(p.xz);
    if (wc <= 0.01) return 0.0;
    vec3 q = p; q.xz += uWind * uTime;
    // low-frequency perlin-worley body (cells ~1.5 km)
    vec4 lo = texture(uNoise3, q * vec3(0.00011, 0.00016, 0.00011));
    float base = lo.b * 0.65 + lo.r * 0.35;
    // cumulus height gradient: flat base, rounded top
    float hg = smoothstep(0.0, 0.06, hf) * smoothstep(1.0, 0.3, hf);
    float bc = saturate(remap(base * hg, 1.0 - wc, 1.0, 0.0, 1.0)) * wc;
    if (bc <= 0.0) return 0.0;
    if (detail) {
        q.y -= uTime * 0.8;
        // (r4 L10) a 4th octave (~40 m) and stronger erosion (m x 0.32 -> 0.36): crisp cauliflower rims instead of
        // soft cotton blobs (target: 1-2 px rims). (0.5 shrank the r5 cumulus to a sixth of their area, 0.4 to a third)
        float hfb = (texture(uNoise3, q * 0.0009).b * 0.62 + texture(uNoise3, q * 0.0029 + 0.4).b * 0.26
                  + texture(uNoise3, q * 0.0085 + 0.7).r * 0.12 + texture(uNoise3, q * 0.024 + 0.2).r * 0.08) * (1.0 / 1.08);
        float m = mix(hfb, 1.0 - hfb, saturate(hf * 6.0));   // wispy base, billowy top
        bc = saturate(remap(bc, m * 0.36, 1.0, 0.0, 1.0));
    }
    return bc;
}

float lightOD(vec3 p) {
    float od = 0.0;
    // (r4 L10) first step 45 -> 25 m (self-shadowed flanks and bases), x1.9 keeps the ~1.3 km reach; detail on the first
    // 2 steps (was 3) pays for the 4th detail octave
    float ls = 25.0, t = 0.0;
    for (int j = 0; j < 6; j++) {
        t += ls;
        od += cloudDensity(p + uSunDir * t, j < 2) * ls;
        ls *= 1.9;
    }
    return od;
}

// returns premultiplied cloud radiance (rgb) and transmittance (a)
vec4 renderClouds(vec3 ro, vec3 rd, float tmax, float jitter, int nsteps) {
#ifdef NO_CLOUDS
    return vec4(0.0, 0.0, 0.0, 1.0);
#endif
    if (uCloudCover <= 0.001 || abs(rd.y) < 1e-4) return vec4(0.0, 0.0, 0.0, 1.0);
    float ta = (CL_BOT - ro.y) / rd.y, tb = (CL_TOP - ro.y) / rd.y;
    float t0 = max(min(ta, tb), 0.0), t1 = min(max(ta, tb), tmax);
    t1 = min(t1, t0 + 40000.0);
    if (t1 <= t0) return vec4(0.0, 0.0, 0.0, 1.0);
    float dt = (t1 - t0) / float(nsteps);
    float t = t0 + dt * jitter;
    vec3 col = vec3(0.0);
    float T = 1.0;
    float mu = dot(rd, uSunDir);
    const float SIG = 0.05;
    for (int i = 0; i < nsteps; i++) {
        vec3 p = ro + rd * t;
        float d = cloudDensity(p, true);
        if (d > 0.003) {
            float od = lightOD(p) * SIG;
            float hf = saturate((p.y - CL_BOT) / (CL_TOP - CL_BOT));
            // multiple scattering octaves (Wrenninge)
            vec3 S = vec3(0.0);
            float a = 1.0, b = 1.0, cph = 1.0;
            for (int k = 0; k < 3; k++) {
                float ph = mix(phaseHG(mu, 0.65 * cph), phaseHG(mu, -0.2 * cph), 0.25) * 4.0 * PI;
                S += uSunCol * a * ph * exp(-od * b);
                a *= 0.55; b *= 0.3; cph *= 0.5;
            }
            float powder = 1.0 - 0.55 * exp(-d * 12.0);
            S *= powder;
            // (r4 L10) (0.9 + 1.6 hf) x 1.4 -> (0.8 + 2.2 hf) x 1.45: greyer bases under bright tops (target: base / top
            // 0.70-0.85); (0.45 + 1.9 hf) x 1.2 turned the whole cloud grey (r5 / r10 cloud / sky 1.5 vs 2.0-2.8)
            S += uSkyAmb * (0.8 + 2.2 * hf) * 1.45 + uSunCol * 0.035;
            float sig = SIG * d;
            float Ts = exp(-sig * dt);
            col += T * S * (1.0 - Ts);
            T *= Ts;
            if (T < 0.01) break;
        }
        t += dt;
    }
    return vec4(col, T);
}
