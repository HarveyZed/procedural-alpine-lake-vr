"""Annotated top-down map of the world (hillshade + water + forest + grid labels) for shot planning.
usage: python tools/worldmap.py [half_extent_m] [center_x] [center_z]"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
import bake

half = float(sys.argv[1]) if len(sys.argv) > 1 else 2400.0
cx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
cz = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
N = 1024
hf = np.load(bake.CACHE / "h_far.npy"); hn = np.load(bake.CACHE / "h_near.npy")
lv = np.load(bake.CACHE / "water_level.npy"); fo = np.load(bake.CACHE / "forest.npy")
xs = cx + (np.arange(N) + 0.5) / N * 2 * half - half
zs = cz + (np.arange(N) + 0.5) / N * 2 * half - half
X, Z = np.meshgrid(xs, zs)
def samp(a, h, X, Z):
    r = a.shape[0]
    i = np.clip(((Z + h) / (2 * h) * r).astype(int), 0, r - 1); j = np.clip(((X + h) / (2 * h) * r).astype(int), 0, r - 1)
    return a[i, j]
inner = (np.abs(X) < bake.NEAR_HALF * 0.95) & (np.abs(Z) < bake.NEAR_HALF * 0.95)
H = np.where(inner, samp(hn, bake.NEAR_HALF, X, Z), samp(hf, bake.FAR_HALF, X, Z))
L = samp(lv, bake.FAR_HALF, X, Z); F = samp(fo, bake.FAR_HALF, X, Z).astype(np.float32) / 255
cell = 2 * half / N
gz, gx = np.gradient(H, cell)
n = np.stack([-gx, np.ones_like(H), -gz], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
Ld = np.array([-0.4, 0.8, -0.45]); Ld /= np.linalg.norm(Ld)
sh = np.clip((n * Ld).sum(-1), 0, 1)
col = np.ones(H.shape + (3,), np.float32) * np.array([0.62, 0.60, 0.57])
col = col * (1 - F[..., 2:3] * 0.0)
col = col * (1 - F[..., 0:1]) + np.array([0.16, 0.32, 0.12]) * F[..., 0:1]
col = col * (0.35 + 0.8 * sh[..., None])
water = H < L
col[water] = np.array([0.1, 0.55, 0.6]) * (0.5 + 0.5 * np.clip(1 + (H[water] - L[water]) / 40, 0, 1))[:, None]
cl = (np.abs(((H + 1000) % 100) - 50) > 48.8) & ~water
col[cl] *= 0.7
img = Image.fromarray((np.clip(col, 0, 1) ** (1 / 1.3) * 255).astype(np.uint8)[::-1])
d = ImageDraw.Draw(img)
step = 250 if half <= 1500 else (500 if half <= 4000 else 1000)
for v in np.arange(np.ceil((cx - half) / step) * step, cx + half, step):
    px = (v - (cx - half)) / (2 * half) * N
    d.line([(px, 0), (px, N)], fill=(255, 255, 255), width=1)
    d.text((px + 2, 2), f"x{int(v)}", fill=(255, 255, 0))
for v in np.arange(np.ceil((cz - half) / step) * step, cz + half, step):
    py = N - (v - (cz - half)) / (2 * half) * N
    d.line([(0, py), (N, py)], fill=(255, 255, 255), width=1)
    d.text((2, py + 2), f"z{int(v)}", fill=(255, 255, 0))
out = ROOT / "out" / f"worldmap_{int(half)}_{int(cx)}_{int(cz)}.png"
out.parent.mkdir(exist_ok=True)
img.save(out); print(out)
