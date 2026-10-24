#version 450
// P0 spike A: analytic sphere via sphere-trace march. Scene constants are HARD-CODED
// and must match spike/p0_b/wgsl + spike/p0_ref.py exactly (PSNR gate).
// sphere: center (0,0,1.2) r=0.5; ray d = normalize(px, py, 2.0);
// march t0=0.01, dt=max(d,0.005), 128 steps, hit eps 1e-4, far 10.0.
// lambert: c = albedo*(0.18 + 0.82*max(dot(n,L),0)), L = normalize(-0.5,0.7,-0.6),
// albedo (0.80,0.30,0.25); sky = mix((0.25,0.40,0.60),(0.65,0.80,0.95), 0.5*dy+0.5);
// out = pow(clamp(c,0,1), 1/2.2) -> RGBA8.
layout(location = 0) out vec4 outColor;

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    float dx = (float(px.x) + 0.5) / 256.0 * 2.0 - 1.0;
    float dy = 1.0 - (float(px.y) + 0.5) / 256.0 * 2.0;
    vec3 d = normalize(vec3(dx, dy, 2.0));
    vec3 C = vec3(0.0, 0.0, 1.2);
    float r = 0.5;
    vec3 L = normalize(vec3(-0.5, 0.7, -0.6));

    float t = 0.01;
    float hit = -1.0;
    for (int i = 0; i < 128; ++i) {
        vec3 p = d * t - C;
        float sd = length(p) - r;
        if (sd < 1e-4) { hit = t; break; }
        t += max(sd, 0.005);
        if (t > 10.0) break;
    }

    vec3 c;
    if (hit > 0.0) {
        vec3 n = normalize(d * hit - C);
        float ndl = max(dot(n, L), 0.0);
        c = vec3(0.80, 0.30, 0.25) * (0.18 + 0.82 * ndl);
    } else {
        float g = 0.5 * d.y + 0.5;
        c = mix(vec3(0.25, 0.40, 0.60), vec3(0.65, 0.80, 0.95), g);
    }
    c = pow(clamp(c, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
    outColor = vec4(c, 1.0);
}
