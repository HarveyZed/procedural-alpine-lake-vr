"""Bake the final world into cache_final/: v2 analytic terrain -> GPU erosion (precomputed
by engine/erosion.py into cache_erosion/v2_*.npy, or run now) -> features (waterfall
cliff, stream, trail) -> near map -> AO -> water/forest maps.
usage: python tools/bake_final.py [--run-erosion] [--fresh-raw]
(--fresh-raw re-bakes the analytic heights from heightfn.glsl; then use --run-erosion too)"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FINAL = Path(os.environ.get("ALPINE_CACHE", str(ROOT / "cache_final")))
FINAL.mkdir(exist_ok=True)
os.environ["ALPINE_CACHE"] = str(FINAL)
sys.path.insert(0, str(ROOT / "engine"))
import bake  # noqa: E402

t0 = time.time()
bake.bake_noise3d()                             # tileable 3D noise volume (cached)
if "--fresh-raw" in sys.argv and (FINAL / "h_far_raw.npy").exists():
    (FINAL / "h_far_raw.npy").unlink()          # re-evaluate the analytic heightfn (after heightfn edits)
for f in ("h_far.npy", "h_near.npy", "ao_far.npy", "ao_near.npy", "water_level.npy", "forest.npy", "trees.npy",
          "flow.npy", "talus.npy", "sediment.npy"):
    p = FINAL / f
    if p.exists():
        p.unlink()

EV = ROOT / "cache_erosion"


def erode(h, cell):
    # 1) this world cache's own erosion result (written by bake.bake_heights), 2) cache_erosion/v2_*
    if "--run-erosion" not in sys.argv and (FINAL / "erosion_height.npy").exists():
        out = {"height": np.load(FINAL / "erosion_height.npy")}
        for k in ("flow", "talus", "sediment"):
            p = FINAL / f"erosion_{k}.npy"
            if p.exists():
                out[k] = np.load(p)
        print("  reusing", FINAL / "erosion_height.npy")
        return out
    if "--run-erosion" in sys.argv or not (EV / "v2_height.npy").exists():
        import erosion
        return erosion.erode(h, cell)
    out = {"height": np.load(EV / "v2_height.npy")}
    for k in ("flow", "talus", "sediment"):
        p = EV / f"v2_{k}.npy"
        if p.exists():
            out[k] = np.load(p)
    return out


hfar, hnear = bake.bake_heights(force=False, erode=erode)
bake.bake_ao(hfar, hnear, force=True)
bake.bake_world_maps(hfar, force=True)
print(f"final bake done in {time.time()-t0:.1f}s ->", FINAL)
