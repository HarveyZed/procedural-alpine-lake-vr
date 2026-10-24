"""cache_schema - the .npy/.npz/.json cache contract (PHASE_3 T3).

RUNBOOK section 4: the world is baked once into cache_final/*.npy and Phase 4
renders exactly those files. The engine spells that contract out inline at
every np.save / np.load. This module records it in one place:

  * file name, dtype, component count, resolution and the world half-extent
    the texel grid spans (so a reader can rebuild the world <-> texel
    transform without re-deriving it),
  * which engine line writes it - provenance for the "no drift" review,
  * the resolution ladder + dispatch discipline constants the bakes obey.

Cache root follows the engine rule (engine/bake.py:15, and
tools/bake_final.py / tools/render_final.py which export ALPINE_CACHE before
importing): $ALPINE_CACHE if set, else <repo>/cache_final. CACHE is resolved
at import like the engine does; cache_dir() re-reads the environment for
callers that set it later.

No GPU and no filesystem access at import (directory creation lives in
ensure_dir(), called by the run_* entry points).
"""

import os
from pathlib import Path

import numpy as np

# copied from engine/bake.py:14-16 (ROOT + CACHE rule; the engine's module
# level CACHE.mkdir() is deliberately NOT copied - P3 modules must be
# import-safe, ensure_dir() does it at run time instead)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "cache_final"

# copied from engine/bake.py:18-23. Single source on purpose: bake_vr.py,
# features_vr.py and erosion_vr.py import these instead of restating them, so
# a UBO value cannot drift per call site.
FAR_HALF = 16384.0
FAR_RES = 8192
NEAR_HALF = 2048.0
NEAR_RES = 4096
TARN_C = (-1680.0, -420.0)
TARN_LEVEL = 520.0

# Erosion ladder, copied from engine/erosion.py:119-148 (STAGES res= values):
# 512^2 @ 64 m/cell -> 1024 -> 2048 -> 4096, then the 8192 hillslope pass
# inside the +-6 km window.
EROSION_RES_LADDER = (512, 1024, 2048, 4096, 8192)
# 16x16 shared-memory tiles: engine/erosion.py:242-254 dispatches
# g = res // 16 + 1 workgroups per axis and engine_vr/shaders_gen/
# erosion_*.comp declare "#define TS 16" + local_size_x = local_size_y = TS,
# which gl_vr.Host.local_size() reads back out of the shader source.
EROSION_LOCAL_SIZE = 16
# engine/erosion.py:201 BAND_TILES = 128 tile rows per dispatch = the
# "<= 2048 rows per dispatch" TDR rule (128 * 16 = 2048).
EROSION_BAND_TILES = 128
EROSION_MAX_DISPATCH_ROWS = EROSION_BAND_TILES * EROSION_LOCAL_SIZE  # 2048
# engine/erosion.py:242-254 flush cadence (ctx.finish every N dispatches),
# keyed by resolution band; gl_vr equivalent is Host.wait().
EROSION_FLUSH_EVERY = ((2048, 32), (4096, 8), (10 ** 9, 2))


def cache_dir() -> Path:
    """$ALPINE_CACHE else <repo>/cache_final (engine/bake.py:15)."""
    return Path(os.environ.get("ALPINE_CACHE", str(DEFAULT_CACHE)))


CACHE = cache_dir()


def set_cache_dir(path) -> Path:
    """Point every consumer at path and export ALPINE_CACHE so engine-side
    helpers agree. Returns the new directory."""
    global CACHE
    CACHE = Path(path)
    os.environ["ALPINE_CACHE"] = str(CACHE)
    return CACHE


def ensure_dir() -> Path:
    """mkdir -p the cache dir (runtime side effect, never at import)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE


def _f(dtype, res, comps, half, writer, note="", shape=None):
    return {"dtype": np.dtype(dtype), "res": res, "comps": comps,
            "half": half, "writer": writer, "note": note, "shape": shape}


# The contract. Default product = the final world maps; writer lines point at
# the original so a reviewer can diff GL output vs VR output file by file.
FILES = {
    # ---- height maps ----------------------------------------------------
    "h_far_raw.npy": _f(np.float32, FAR_RES, 1, FAR_HALF, "engine/bake.py:58",
                        "analytic heights, pre erosion/shore/features"),
    "h_eroded.npy": _f(np.float32, FAR_RES, 1, FAR_HALF, "engine/bake.py:83",
                       "post erosion + shore, pre relief/features"),
    "h_far.npy": _f(np.float32, FAR_RES, 1, FAR_HALF, "engine/bake.py:93",
                    "the far map Phase 4 raymarches"),
    "h_near.npy": _f(np.float32, NEAR_RES, 1, NEAR_HALF,
                     "engine/bake.py:224", "the 1 m near map"),
    # ---- erosion products -----------------------------------------------
    "erosion_height.npy": _f(np.float32, FAR_RES, 1, FAR_HALF,
                             "engine/bake.py:64"),
    "flow.npy": _f(np.float32, FAR_RES, 1, FAR_HALF, "engine/bake.py:67",
                   "unit-basin flow accumulation (m^2)"),
    "talus.npy": _f(np.float32, FAR_RES, 1, FAR_HALF, "engine/bake.py:579",
                    "max(erosion talus, cone deposits)"),
    "sediment.npy": _f(np.float32, FAR_RES, 1, FAR_HALF, "engine/bake.py:67"),
    "erosion_flow.npy": _f(np.float32, FAR_RES, 1, FAR_HALF,
                           "engine/bake.py:68"),
    "erosion_talus.npy": _f(np.float32, FAR_RES, 1, FAR_HALF,
                            "engine/bake.py:68"),
    "erosion_sediment.npy": _f(np.float32, FAR_RES, 1, FAR_HALF,
                               "engine/bake.py:68"),
    "cones.npy": _f(np.float16, FAR_RES, 1, FAR_HALF, "engine/bake.py:581",
                    "talus-cone deposit mask only"),
    # ---- water ----------------------------------------------------------
    "stream_level_far.npy": _f(np.float32, FAR_RES, 1, FAR_HALF,
                               "engine/bake.py:92", "-1e4 = dry"),
    "stream_level_near.npy": _f(np.float32, NEAR_RES, 1, NEAR_HALF,
                                "engine/bake.py:222"),
    "stream_near.npy": _f(np.float16, NEAR_RES, 4, NEAR_HALF,
                          "engine/bake.py:223", "RGBA cos, sin, foam, is"),
    "water_level.npy": _f(np.float32, FAR_RES, 1, FAR_HALF,
                          "engine/bake.py:720", "-10000 = no water"),
    "water_level_near.npy": _f(np.float32, NEAR_RES, 1, NEAR_HALF,
                               "engine/bake.py:686"),
    "river.npz": _f(np.float32, None, None, None, "engine/bake.py:459",
                    "keys z, x, w, hb (thalweg rows)"),
    # ---- AO + cover -----------------------------------------------------
    "ao_far.npy": _f(np.float16, FAR_RES // 2, 1, FAR_HALF,
                     "engine/bake.py:601", "half resolution"),
    "ao_near.npy": _f(np.float16, NEAR_RES, 1, NEAR_HALF,
                      "engine/bake.py:601"),
    "forest.npy": _f(np.uint8, FAR_RES, 4, FAR_HALF, "engine/bake.py:720",
                     "clip(0,1) * 255 + 0.5"),
    # ---- waterfall / stream features ------------------------------------
    "features.json": _f(str, None, None, None, "engine/features.py:391",
                        "waterfall, stream, trail, falls, corr box"),
    "stream_corr.npy": _f(np.float16, None, 4, NEAR_HALF,
                          "engine/features.py:303",
                          "crop box stored in features.json"),
    # ---- noise + Phase-4 products (contract only) ------------------------
    "noise3d.npy": _f(np.uint8, None, 4, None, "engine/bake.py:765",
                      "tileable RGBA 3D noise",
                      shape=(128, 128, 128, 4)),
    "trees.npy": _f(np.float32, 4096, 4, NEAR_HALF, "engine/render.py:178",
                    "written by the Phase-4 tree map, not by P3"),
}

# What engine/render.py refuses to run without (engine/render.py:123): the
# minimum a first run must produce for Phase 4.
RENDER_REQUIRED = ("h_far.npy", "h_near.npy", "water_level.npy", "forest.npy")


def path(name) -> Path:
    """Cache-dir-relative path for a contract name (or a bare file name)."""
    return CACHE / str(name)


def spec(name: str) -> dict:
    """Registry entry for name.

    Accepts the contract file name ("h_far.npy") or its bare stem ("h_far"),
    because one stem feeds several products (h_far.npy plus stream_level_far).
    Raises KeyError listing the known names so a typo cannot pass silently.
    """
    key = str(name)
    if key not in FILES and f"{key}.npy" in FILES:
        key = f"{key}.npy"
    if key in FILES:
        return FILES[key]
    raise KeyError(f"{name!r} is not in the cache contract; known: "
                   f"{', '.join(sorted(FILES))}")


def cell(name: str) -> float:
    """Texel size in metres for a registered map (2 * half / res)."""
    s = spec(name)
    return 2.0 * s["half"] / s["res"]


def shape_of(name: str) -> tuple:
    """Expected numpy shape (h, w) or (h, w, c)."""
    s = spec(name)
    if s["shape"] is not None:
        return tuple(s["shape"])
    if s["res"] is None:
        raise KeyError(f"{name}: variable shape ({s['note']})")
    if s["comps"] and s["comps"] > 1:
        return (s["res"], s["res"], s["comps"])
    return (s["res"], s["res"])


def missing(names=RENDER_REQUIRED) -> list:
    """Contract files not on disk yet (Phase 4's readiness probe)."""
    return [n for n in names if not path(n).exists()]


def checkpoint_name(res: int) -> str:
    """Erosion per-level checkpoint file (engine/erosion.py:1486)."""
    return f"ck_{int(res)}.npz"
