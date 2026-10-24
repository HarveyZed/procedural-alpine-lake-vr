// ---------------------------------------------------------------------------
// hero.glsl : a few hand-placed granite blocks for specific shots (e.g. the
// rock the camera rises from behind in the opening reveal)
// (rocks3) the same construction as the scattered boulders (rocks.glsl) instead of a rounded box blended 38 % into
// an ellipsoid (a soft potato): a jointed block of 5 near-axis faces + 4 random cuts across edges / corners with
// crisp, unevenly weathered edges, gentle bulges, crumbly worley relief that is deeper along the edges, a fracture
// step on one top edge and a cm-scale octave near the camera
// ---------------------------------------------------------------------------
uniform vec4 uHeroA[6];   // center xyz, size (m)
uniform vec4 uHeroB[6];   // stretch xyz, yaw
uniform int uHeroN;

float sdHero(vec3 p, int i) {
    vec3 q = p - uHeroA[i].xyz;
    q.xz = rot2(uHeroB[i].w) * q.xz;
    vec3 r = uHeroA[i].w * uHeroB[i].xyz;
    vec3 b = r * 0.78;                                        // half extents (the old rounded box: r * 0.74)
    float R = max(b.x, max(b.y, b.z));
    float sd = float(i) * 7.31 + 1.7;
    vec4 h1 = vec4(hash11(sd), hash11(sd + 1.13), hash11(sd + 2.31), hash11(sd + 3.77));
    vec4 h2 = vec4(hash11(sd + 4.19), hash11(sd + 5.53), hash11(sd + 6.71), hash11(sd + 7.97));
    // faces: tilted top, four sides leaning in / out a little, buried base
    float p0 = dot(q, normalize(vec3((h1.x - 0.5) * 0.5, 1.0, (h1.y - 0.5) * 0.5))) - b.y * mix(0.9, 0.98, h1.z);
    float p1 = dot(q, normalize(vec3(1.0, mix(-0.1, 0.35, h1.w), (h2.x - 0.5) * 0.4))) - b.x;
    float p2 = dot(q, normalize(vec3(-1.0, mix(-0.1, 0.35, h2.y), (h2.z - 0.5) * 0.4))) - b.x * mix(0.85, 1.0, h2.w);
    float p3 = dot(q, normalize(vec3((h1.z - 0.5) * 0.4, mix(-0.1, 0.35, h2.x), 1.0))) - b.z;
    float p4 = dot(q, normalize(vec3((h2.w - 0.5) * 0.4, mix(-0.1, 0.35, h1.x), -1.0))) - b.z * mix(0.85, 1.0, h1.w);
    // cuts across edges / corners (74-90 % of the support distance of the b ellipsoid)
    vec3 n5 = rkDir(h1.x, h1.y, 0.35, 1.0), n6 = rkDir(fract(h1.x + 0.4), h2.z, -0.2, 0.4);
    vec3 n7 = rkDir(fract(h1.x + 0.65), h2.w, 0.5, 1.2), n8 = rkDir(fract(h2.x + 0.13), h1.w, 0.1, 0.7);
    float p5 = dot(q, n5) - length(b * n5) * mix(0.74, 0.86, h2.x);
    float p6 = dot(q, n6) - length(b * n6) * mix(0.76, 0.9, h2.y);
    float p7 = dot(q, n7) - length(b * n7) * mix(0.78, 0.9, h1.z);
    float p8 = dot(q, n8) - length(b * n8) * mix(0.74, 0.88, h1.y);
    float kf = R * 0.025;                                     // crisp, unevenly weathered edges
    float d = smax(p0, p1, kf * 1.2);
    d = smax(d, p2, kf * 0.5);
    d = smax(d, p3, kf * 1.0);
    d = smax(d, p4, kf * 0.4);
    d = smax(d, p5, kf * 0.6);
    d = smax(d, p6, kf * 0.8);
    d = smax(d, p7, kf * 0.4);
    d = smax(d, p8, kf * 0.7);
    d = smax(d, -q.y - b.y, kf);                              // base (buried)
    // fracture step along the top / side-1 edge
    d = smax(d, min(p0 + b.y * mix(0.2, 0.4, h2.z), p1 + b.x * mix(0.12, 0.3, h2.w)), kf * 0.4);
    d = smax(d, rkEllipsoid(q, b * 1.6), 0.05 * R);          // trims the far corners
    // edge proximity (largest vs second-largest face distance)
    float m1 = max(max(max(p0, p1), max(p2, p3)), max(max(p4, p5), max(p6, max(p7, p8))));
    float m2 = -1e9;
    m2 = max(m2, p0 < m1 ? p0 : -1e9); m2 = max(m2, p1 < m1 ? p1 : -1e9); m2 = max(m2, p2 < m1 ? p2 : -1e9);
    m2 = max(m2, p3 < m1 ? p3 : -1e9); m2 = max(m2, p4 < m1 ? p4 : -1e9); m2 = max(m2, p5 < m1 ? p5 : -1e9);
    m2 = max(m2, p6 < m1 ? p6 : -1e9); m2 = max(m2, p7 < m1 ? p7 : -1e9); m2 = max(m2, p8 < m1 ? p8 : -1e9);
    float edge = 1.0 - smoothstep(0.0, 0.12 * R, m1 - m2);
    // relief: gentle bulges, crumbly worley pits with crisp crests (deeper on the edges), a through-crack
    vec3 s = q / R + float(i) * 3.1;
    float n1 = texture(uNoise3, s * 0.11).g - 0.487;
    float n2 = texture(uNoise3, s * 0.23 + 0.31).b - 0.525;
    float n3 = texture(uNoise3, s * 0.6 + 0.57).b - 0.525;
    d += R * (0.14 * n1 + 0.035 * n2 * (0.55 + 1.15 * edge) + 0.006 * n3);
    float ck = abs(dot(q, normalize(vec3(h2.x - 0.5, 0.15, 1.0))) - (h1.y - 0.5) * 0.6 * b.z) - R * 0.006 * saturate(0.4 + q.y / b.y);
    d = smax(d, -ck, R * 0.008);
    // cm octave near the camera (silhouettes / contact lines)
    float tc = length(p - uCamPos);
    if (tc < 14.0) {
        float m = texture(uNoise3, q * 1.45 + 0.29 + float(i)).b;
        float pz = smoothstep(0.38, 0.62, texture(uNoise3, q * 0.37 + 0.71 + float(i)).r);
        d -= (1.0 - smoothstep(8.0, 14.0, tc)) * 0.011 * mix(0.2, 1.0, max(pz, edge)) * (0.7 + edge) * (smoothstep(0.25, 0.9, m) - 0.55);
    }
    return d * 0.75;
}

float heroSDF(vec3 p, out int idx) {
    float dm = 1e9; idx = -1;
    for (int i = 0; i < 6; i++) {
        if (i >= uHeroN) break;
        float b = length(p - uHeroA[i].xyz) - uHeroA[i].w * max(uHeroB[i].x, max(uHeroB[i].y, uHeroB[i].z)) * 1.3;
        if (b > dm) continue;
        float d = b > 1.0 ? b : sdHero(p, i);
        if (d < dm) { dm = d; idx = i; }
    }
    return dm;
}
