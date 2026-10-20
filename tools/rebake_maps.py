"""Re-bake only the water/forest/material maps (and force tree cells to re-bake) from the current h_far.
usage: ALPINE_CACHE=... python tools/rebake_maps.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
import numpy as np
import bake
h = np.load(bake.CACHE / "h_far.npy")
bake.bake_world_maps(h, force=True)
t = bake.CACHE / "trees.npy"
if t.exists():
    t.unlink()
print("maps re-baked ->", bake.CACHE)
