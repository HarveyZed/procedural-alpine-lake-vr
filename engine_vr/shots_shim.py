"""PHASE_11 import-over-copy shim over engine.shots_film (shots_shim).

RUNBOOK section 1 "Reuse over rewrite": the animated shot flow (vr.py
--animate) must build its shot list with the ENGINE modules
(engine/shots_film.build + engine/camera_path.shot_camera). Neither imports
here without two host-side facts being arranged, and this shim is the only
place that arranges them - it reimplements NO keyframes, path search,
easing or handheld noise (all live in engine/camera_path.py +
engine/shots_film.py, imported, never copied):

1. moderngl is NOT installed in .venv (RUNBOOK section 3) and
   engine/render.py:8 does "import moderngl" at module level (engine/gl.py:5
   likewise). Every moderngl ATTRIBUTE in engine/render.py is only touched
   inside method bodies (render.py:137/148/197/299/302/453/472/488/504),
   never at import time - so an empty placeholder module in sys.modules
   makes render/gl/bake/shots_film/camera_path importable and would fail
   LOUDLY (AttributeError) if any GL call were ever attempted here.
   FIXME(p11) A real moderngl install (find_spec hit) is preferred over the
   stub.

2. The VR world shim (SHIM_SURFACE below): shots_film.build(world) reads
   exactly four attributes off the world object; vr.PanoWorld already
   provides all four from the VR bake cache (engine_vr/cache_schema
   accessors - the same cache files vr.py _vp_skim/_vp_waterfall/_vp_stream
   read), so no adapter class is needed and build closures
   (ground/P/shore search/head_peak) run against the baked VR cache
   untouched.

Cache root: engine/bake.py:15 resolves CACHE from ALPINE_CACHE at import;
cache_schema.set_cache_dir() exports that variable, so callers must set the
cache dir BEFORE calling build_shots() (vr.py main() does). TreeMap reads
<cache>/trees.npy, written by the P3 bake (cache_schema FILE_SCHEMAS).
"""

import importlib.util
import sys
import types   # FIXME(p11)

# The exact attribute surface shots_film.build(world) consumes, with the
# build() line that touches each (engine/shots_film.py):
#   world.height(x, z)  - build():22 ground(), via shore_south():33,
#                         shore_x():39 and every P() keyframe
#   world.water(x, z)   - build():22 ground()
#   world.hfar          - build():129 head_peak block (far heightmap ndarray)
#   world.features      - build():167/:174 waterfall, :195 trail, :214
#                         stream (dict from features.json)
SHIM_SURFACE = ("height", "water", "hfar", "features")


def install_engine_imports():
    """Make engine.shots_film/camera_path importable without moderngl.

    Returns "stub" or "real" - which moderngl the engine got. Inserts the
    engine dir on sys.path the same way vr.py:39-40 does. Importing render
    creates NO GPU context: engine/gl.py only calls moderngl inside ctx(),
    which is reached solely from bake/render run functions this module never
    calls.
    """
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eng = os.path.join(root, "engine")
    if eng not in sys.path:
        sys.path.insert(0, eng)
    cur = sys.modules.get("moderngl")
    if cur is not None and getattr(cur, "__file__", None) is None:
        return "stub"          # already stubbed here - find_spec would raise
    real = importlib.util.find_spec("moderngl") is not None
    if not real:
        stub = sys.modules.get("moderngl")
        if stub is None or getattr(stub, "__file__", None) is None:
            stub = types.ModuleType("moderngl")
            stub.__doc__ = ("PHASE_11 import-only stub (engine_vr/"
                            "shots_shim.install_engine_imports); no GL call "
                            "may ever touch this object")
            sys.modules["moderngl"] = stub
        return "stub"
    return "real"


def build_shots(world, verbose=False):
    """engine.shots_film.build(world) verbatim - the film 9-shot list in
    film order. The world must expose SHIM_SURFACE (vr.PanoWorld does).
    CPU-only: numpy/scipy + cache reads, zero device calls.
    """
    install_engine_imports()
    for attr in SHIM_SURFACE:
        if not hasattr(world, attr):
            raise AttributeError("shot-world shim missing " + attr)
    import shots_film   # noqa: E402  engine module (import-over-copy)
    return shots_film.build(world, verbose=verbose)


def shot_camera():
    """Return engine.camera_path.shot_camera(shot, t) ->
    Camera(pos, target, fov, roll, aperture, focus) - the film ONLY camera
    evaluator, as the imported function (import-over-copy).
    """
    install_engine_imports()
    from camera_path import shot_camera as sc   # noqa: E402
    return sc
