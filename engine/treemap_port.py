"""CPU port of trees.glsl treeFinish() (stand structure: culling, final height H, crown radius R, crown base yb)
for shots_film.TreeMap, which scores framing / camera clearance from the raw bake and so counts trees that
the renderer culls and misses the taller cohorts. Vectorised over cells.

    from treemap_port import tree_final
    keep, H, R, yb, larch = tree_final(cx, cz, d)     # cx, cz int arrays of tree cells, d = trees.npy rows

usage (stats): python tools/treemap_port.py [x0 x1 z0 z1]
Noise layout: render.py uploads noise3d.npy as texture3d(arr.transpose(2,1,0,3)), so texture coordinate
(u, v, w) samples arr[u, v, w] (x = axis 0)."""
import sys
from pathlib import Path
import numpy as np

import os
CACHE = Path(os.environ.get("ALPINE_CACHE", str(Path(__file__).resolve().parent.parent / "cache_final")))
M32 = 0xFFFFFFFF
_N = None


def _noise():
    global _N
    if _N is None:
        _N = np.load(CACHE / "noise3d.npy").astype(np.float32) / 255.0
    return _N


def _hu(x):
    x = x & M32
    x ^= x >> 16; x = (x * 0x7feb352d) & M32
    x ^= x >> 15; x = (x * 0x846ca68b) & M32
    x ^= x >> 16
    return x


def _hash2u(cx, cz):
    cx = np.asarray(cx, np.int64) & M32; cz = np.asarray(cz, np.int64) & M32
    return _hu((cx * 0x9E3779B1 + _hu((cz + 0x632BE5AB) & M32)) & M32)


def _u2f(h):
    return (h >> 8).astype(np.float64) / 16777216.0


def hash42(cx, cz):
    a = _hash2u(cx, cz); b = _hu(a); d = _hu(b); e = _hu(d)
    return _u2f(a), _u2f(b), _u2f(d), _u2f(e)


def hash12(cx, cz):
    return _u2f(_hash2u(cx, cz))


def tex3(u, v, w):
    """trilinear, repeat (GL_LINEAR, texel centres at (i + 0.5) / N), lod 0"""
    N = _noise(); R = N.shape[0]
    x = np.asarray(u) * R - 0.5; y = np.asarray(v) * R - 0.5; z = np.full_like(x, w * R - 0.5)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64); z0 = np.floor(z).astype(np.int64)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]; fz = (z - z0)[:, None]
    out = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                wgt = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                out = out + wgt * N[(x0 + dx) % R, (y0 + dy) % R, (z0 + dz) % R]
    return out


def _sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fr(x):
    return x - np.floor(x)


# Drop-in replacement for engine/treemap_port.py tree_final() matching trees2 round-2 trees.glsl treeFinish()
# (copy over the old function; helpers _fr, _sm, hash42, hash12, tex3 are unchanged)
def tree_final(cx, cz, d):
    """cx, cz: cell indices (int arrays); d: (n, 4) baked rows (baseY, signed H, R, yb) with d[:,1] != 0.
    returns keep (bool), H, R, yb, larch (bool), dens exactly as trees.glsl treeFinish()."""
    cx = np.asarray(cx, np.int64); cz = np.asarray(cz, np.int64)
    hs = hash42(cx, cz)
    pos = np.stack([cx * 4.0 - 16384.0 + 4.0 * (0.5 + (hs[1] - 0.5) * 0.6),
                    cz * 4.0 - 16384.0 + 4.0 * (0.5 + (hs[2] - 0.5) * 0.6)], 1)
    H0 = np.abs(d[:, 1])
    hn = hash12(cx + 911, cz + 37)
    dens = np.clip(d[:, 3] / (0.18 * H0 * np.maximum(hn, 0.02)), 0.0, 1.0)
    dk = _sm(0.45, 0.85, dens)
    p1 = np.stack([0.8 * pos[:, 0] + 0.6 * pos[:, 1], -0.6 * pos[:, 0] + 0.8 * pos[:, 1]], 1) / 1500.0
    st = tex3(p1[:, 0], p1[:, 1], 0.43)
    hx = _fr(hs[0] * 7.73 + hs[3] * 3.17)
    keep = ~(st[:, 0] < 0.25 + 0.09 * hx - 0.07 * dk)
    p2 = np.stack([0.6 * pos[:, 0] - 0.8 * pos[:, 1], 0.8 * pos[:, 0] + 0.6 * pos[:, 1]], 1) / 260.0
    s2 = tex3(p2[:, 0], p2[:, 1], 0.17)
    refS = dens > 0.995                                                  # (r5 T1: closed stand keeps all trees)
    keep &= refS | ~(s2[:, 0] < 0.30 + 0.06 * _sm(60.0, 160.0, d[:, 0]) + 0.08 * (hx - 0.5) - 0.08 * dk)
    keep &= refS | ~(s2[:, 2] > 0.73 - 0.07 * _sm(60.0, 100.0, d[:, 0]) + 0.05 * dk + 0.03 * (hx - 0.5))   # clustered 1-5 tree openings (r5: .66 on slopes)
    grp = 0.55 * s2[:, 3] + 0.45 * st[:, 3]
    keep &= ~((dens < 0.6) & (grp < 0.52 * (1.0 - _sm(0.12, 0.60, dens)) + 0.08 * (hx - 0.5)))
    larch = d[:, 1] < 0.0
    hv = _fr(hs[0] * 13.7 + hs[3] * 5.1)
    hsc = (0.55 + 0.75 * hv) * (0.70 + 0.42 * _sm(0.28, 0.72, st[:, 1])) * (0.72 + 0.40 * _sm(0.25, 0.75, s2[:, 1]))
    hsc *= (0.82 + 0.28 * grp) * (0.55 + 0.45 * _sm(0.12, 0.50, dens))
    yg = _fr(hs[3] * 31.7 + hs[1] * 2.3)
    ygF = 0.40 + (0.16 - 0.40) * _sm(0.20, 0.40, dens)
    dom = (_fr(hs[1] * 23.7 + hs[2] * 5.3) >= 0.89).astype(np.float64)
    hsc *= np.where(yg < ygF, 0.30 + 0.32 * yg / ygF, 1.0 + 0.30 * dom)
    mc = _fr(hs[0] * 41.3 + hs[2] * 7.9)                                  # middle cohort (25 %, x0.65-0.85)
    hsc *= np.where((yg >= ygF) & (dom < 0.5) & (mc < 0.25), 0.65 + 0.20 * mc / 0.25, 1.0)
    hsc = np.minimum(hsc, 1.6)
    H = np.minimum(H0 * hsc, 34.0)
    yb = H * (0.02 + 0.23 * dens ** 1.3) * np.where(larch, 0.7, 1.0) * _sm(0.30, 0.45, dens)   # (baked species: before the conversion; r5: 0 in open stands)
    R = d[:, 2] * hsc ** 0.8 * np.where(larch, 1.25, 0.92) * (0.76 + 0.39 * _fr(hs[1] * 9.3))
    # more larch near the lake (H, yb stay the baked species'; the crown gets the larch radius)
    lp = 0.26 * (1.0 - _sm(60.0, 350.0, d[:, 0])) * (1.4 - 0.7 * dens) + 0.05
    conv = (~larch) & (H > 4.0) & (_fr(hs[2] * 17.3 + hs[0] * 4.1) < lp)
    R = np.where(conv, R * (1.25 / 0.92), R)
    larch = larch | conv
    R = np.minimum(R, 3.6)
    return keep, H, R, yb, larch, dens


if __name__ == "__main__":
    T = np.load(CACHE / "trees.npy", mmap_mode="r")
    x0, x1, z0, z1 = [float(v) for v in sys.argv[1:5]] if len(sys.argv) > 4 else (-3000.0, 3000.0, -3000.0, 3000.0)
    cxs = np.arange(int((x0 + 16384) / 4), int((x1 + 16384) / 4))
    czs = np.arange(int((z0 + 16384) / 4), int((z1 + 16384) / 4))
    CX, CZ = np.meshgrid(cxs, czs)
    d = np.asarray(T[czs[0] - 2048:czs[-1] - 2048 + 1, cxs[0] - 2048:cxs[-1] - 2048 + 1]).reshape(-1, 4)
    CX = CX.ravel(); CZ = CZ.ravel()
    ok = d[:, 1] != 0
    keep, H, R, yb, larch, dens = tree_final(CX[ok], CZ[ok], d[ok])
    print(f"baked {ok.sum()}  kept {keep.mean():.3f}")
    for lo, hi in [(0, .3), (.3, .6), (.6, .9), (.9, 1.01)]:
        m = (dens >= lo) & (dens < hi)
        if m.sum():
            print(f"  dens {lo:.1f}-{hi:.1f}: n {m.sum():7d} kept {keep[m].mean():.3f}  H mean {H[m & keep].mean():.1f}  "
                  f"H/H0 {np.mean(H[m & keep] / np.abs(d[ok][:, 1][m & keep])):.2f}  H:W {np.median(H[m & keep] / (2 * R[m & keep])):.2f}")
