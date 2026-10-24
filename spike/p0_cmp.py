#!/usr/bin/env python3
"""P0 comparator - PSNR between p0_a, p0_b and p0_ref images + buffer check summary.
Gate requires PSNR >= 26 dB pairwise (see port/PHASE_0.md). Exit 0 iff all pass."""
import pathlib
import math
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent


def load_ppm(p):
    raw = p.read_bytes()
    magic, rest = raw.split(b"\n", 1)
    assert magic == b"P6", p
    wh, rest = rest.split(b"\n", 1)
    w, h = map(int, wh.split())
    maxv, body = rest.split(b"\n", 1)
    assert maxv == b"255" and len(body) == w * h * 3, (str(p), len(body))
    return w, h, np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3).astype(np.float64)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)


imgs = {}
for name in ["p0_a/sphere.ppm", "p0_b/sphere.ppm", "p0_ref.ppm"]:
    imgs[name] = load_ppm(ROOT / name)
    print(name, "size", imgs[name][0], "x", imgs[name][1])

pairs = [("p0_a", "p0_ref"), ("p0_b", "p0_ref"), ("p0_a", "p0_b")]
ok = True
for x, y in pairs:
    def key(n):
        return n + ".ppm" if n == "p0_ref" else n + "/sphere.ppm"
    a = imgs[key(x)][2]
    b = imgs[key(y)][2]
    db = psnr(a, b)
    mx = int(np.abs(a - b).max())
    frac = float((np.abs(a - b) <= 1).mean())
    print("PSNR", x, "vs", y, "=", ("%.2f dB" % db), " max|diff|=", mx, "/255  within1/255: %.4f%%" % (frac * 100))
    if not (db >= 26.0):
        ok = False

bufo = []
for d in ["p0_a", "p0_b"]:
    txt = (ROOT / d / "buf_check.txt").read_text()
    bufo.append(d + ":" + ("mismatches=0" if "mismatches=0" in txt else "FAIL"))
print("BUF", " ".join(bufo))
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)