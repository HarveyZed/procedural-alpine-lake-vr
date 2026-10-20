"""Per-pass GPU time of film-shot mid frames (which shading pass costs what).
usage: python tools/passprof.py [--shots 0,1,...] [--w 1376 --h 576] [--spp 2]   -> out/passprof.txt"""
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
import render, shots_film
from camera_path import shot_camera
ap = argparse.ArgumentParser(); ap.add_argument("--shots", default="0,1,2,3,4,5,6,7,8")
ap.add_argument("--w", type=int, default=1376); ap.add_argument("--h", type=int, default=576); ap.add_argument("--spp", type=int, default=2)
a = ap.parse_args()
world = render.World(); S = shots_film.build(world); R = render.Renderer(a.w, a.h, world=world)
starts = np.cumsum([0.0] + [s["dur"] for s in S])
tot = {}
lines = []
for i in (int(v) for v in a.shots.split(",")):
    sh = S[i]; tm = sh["dur"] * 0.5
    cams = []
    for s in range(a.spp):
        c = shot_camera(sh, tm); c.fov *= 0.86; cams.append(c)
    R.heroes = sh.get("heroes", [])
    R.prof = {}
    t0 = time.time()
    R.render(cams, sh["look"], [float(starts[i] + tm)] * a.spp)
    dt = time.time() - t0
    p = R.prof; R.prof = None
    s = sum(p.values())
    for k, v in p.items():
        tot[k] = tot.get(k, 0.0) + v
    lines.append(f"shot {i} {sh['name']:9s} {dt:6.1f}s  " + "  ".join(f"{k} {100*v/s:4.1f}%" for k, v in sorted(p.items(), key=lambda kv: -kv[1])))
    print(lines[-1], flush=True)
s = sum(tot.values())
lines.append("ALL: " + "  ".join(f"{k} {100*v/s:4.1f}%" for k, v in sorted(tot.items(), key=lambda kv: -kv[1])))
print(lines[-1])
(ROOT / "out").mkdir(exist_ok=True)
(ROOT / "out" / "passprof.txt").write_text("\n".join(lines) + "\n")
