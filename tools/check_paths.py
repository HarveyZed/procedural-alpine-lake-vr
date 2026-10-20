"""Validate every film camera path frame by frame: ground clearance, tree clearance,
hero-rock clearance. usage: python tools/check_paths.py [--fps 30]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
import bake  # noqa: E402
import render  # noqa: E402
import shots_film  # noqa: E402
from camera_path import shot_camera  # noqa: E402

fps = 30
world = render.World()
S = shots_film.build(world)
T = shots_film.TreeMap(bake.CACHE)
bad = 0
for i, s in enumerate(S):
    n = int(s["dur"] * fps)
    worst_g, worst_t, issues = 1e9, 1e9, []
    for f in range(0, n, 2):
        c = shot_camera(s, f / fps)
        x, y, z = c.pos
        g = max(world.height(x, z), world.water(x, z))
        cg = y - g
        ct = T.clearance(x, y, z) if cg < 45 else 99.0
        ch = 99.0
        for hr in s.get("heroes", []):
            r = hr[3] * max(hr[4:7])
            ch = min(ch, np.linalg.norm(np.subtract(c.pos, hr[:3])) - r * 1.05)
        worst_g, worst_t = min(worst_g, cg), min(worst_t, ct)
        if cg < 0.6 or ct < 0.8 or ch < 0.3:
            issues.append((round(f / fps, 2), round(cg, 2), round(ct, 2), round(ch, 2)))
    flag = "OK " if not issues else "BAD"
    bad += bool(issues)
    print(f"{flag} shot {i} {s['name']:10s} min ground {worst_g:7.2f} m  min tree {worst_t:6.2f} m  issues {issues[:6]}")
sys.exit(1 if bad else 0)
