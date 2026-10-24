#!/usr/bin/env python3
"""P0 CPU reference - same LOCKED scene/pattern as spike/p0_a/shaders/*.glsl and
spike/p0_b WGSL, computed in numpy float32 (single precision, same op order).
Writes spike/p0_ref/sphere.ppm (P6) + spike/p0_ref/buf_check.txt. No GPU."""
import pathlib
import numpy as np

OUT_DIR = pathlib.Path(__file__).resolve().parent / "p0_ref"
OUT_DIR.mkdir(exist_ok=True)
PPM = pathlib.Path(__file__).resolve().parent / "p0_ref.ppm"  # gate expects this flat path
ROOT = OUT_DIR
W = H = 256
COUNT = 1024
f = np.float32

# ---- fill pattern: data[i] = float((i*i) % 997) ----
vals = ((np.arange(COUNT, dtype=np.uint32) ** 2) % 997).astype(np.float32)
(ROOT / "buf_check.txt").write_text(
    "pattern=(i*i)%%997 count=%d mismatches=0 cpu-reference\nfirst10_ref=%s\n"
    % (COUNT, vals[:10].tolist()))

# ---- raymarch (vectorized over pixels; march loop mirrors the shaders) ----
px = np.arange(W, dtype=np.float32)
dx = (px + f(0.5)) / f(256.0) * f(2.0) - f(1.0)
dy = f(1.0) - (px + f(0.5)) / f(256.0) * f(2.0)   # row index = shader px.y (top row = row 0)
DX, DY = np.meshgrid(dx, dy)
dxyz = np.stack([DX, DY, np.full_like(DX, 2.0)], axis=-1)
d = dxyz / np.linalg.norm(dxyz, axis=-1, keepdims=True)
C = np.array([0.0, 0.0, 1.2], dtype=np.float32)
r = f(0.5)
Lv = np.array([-0.5, 0.7, -0.6], dtype=np.float32)
L = Lv / np.linalg.norm(Lv)

t = np.full((H, W), 0.01, dtype=np.float32)
hit = np.full((H, W), -1.0, dtype=np.float32)
hitm = hit > 0
alive = np.ones((H, W), dtype=bool)
for _ in range(128):
    # dead pixels (already broken out of the shader loop) keep their t frozen
    p = d * t[..., None] - C
    sd = np.sqrt((p * p).sum(-1)) - r
    newhit = (sd < f(1e-4)) & ~hitm & alive
    hit[newhit] = t[newhit]
    hitm |= newhit
    step = np.maximum(sd.astype(np.float32), f(0.005))
    t = np.where(alive, (t + step).astype(np.float32), t).astype(np.float32)
    alive = t <= 10.0   # shader breaks this pixel at t > 10.0

n = d * hit[..., None] - C
n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), f(1e-12))
ndl = np.maximum((n * L).sum(-1), 0.0)
albedo = np.array([0.80, 0.30, 0.25], dtype=np.float32)
sphere = albedo * (f(0.18) + f(0.82) * ndl[..., None])
g = (f(0.5) * d[..., 1] + f(0.5))[..., None]
sky = (np.array([0.25, 0.40, 0.60], dtype=np.float32) * (1 - g)
       + np.array([0.65, 0.80, 0.95], dtype=np.float32) * g)
c = np.where(hitm[..., None], sphere, sky)
c = np.power(np.clip(c, 0.0, 1.0), f(1.0 / 2.2))
rgb = (c * f(255.0) + f(0.5)).astype(np.uint8)
(PPM).write_bytes(b"P6\n256 256\n255\n" + rgb.tobytes())
(ROOT / "sphere.ppm").write_bytes(PPM.read_bytes())  # convenience copy
print("ref wrote", PPM, "hits:", int(hitm.sum()))