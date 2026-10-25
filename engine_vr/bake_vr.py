"""bake_vr - engine/bake.py ported to engine_vr (Phase 3, T1/T2).

The bake chain is one tiled fragment pass per stage (analytic heights, macro
relief, granite relief, AO, near upsample, forest/land cover) plus a large
amount of pure numpy post-processing. Only the passes change backend; every
numpy stage is copied verbatim below the generated marker, so the numbers stay
bit-identical to the moderngl build.

Generated region (engine_vr/build_p3.py): the copied blocks carry
"copied from engine/bake.py:<lines>" headers. The substitutions applied there
(documented in port/reports/phase3.md) are only:

  import features                       -> from engine_vr import features_vr
  import features as _feat              ->   (same module, VR driver)
  src = CACHE / "_bake_macro.frag"      -> src = "bake_macro"
  src.write_text(MACRO_FRAG)                (no cache GLSL write: gl_vr
  src = CACHE / "_bake_relief.frag"     ->     compiles engine_vr/shaders_gen)
  src.write_text(RELIEF_FRAG)           -> src = "bake_relief"
  _bake_pass("bake_ao.frag", hfar...)   -> _bake_pass("bake_ao_far", hfar...)
  _bake_pass("bake_ao.frag", hnear...)  -> _bake_pass("bake_ao_near", hnear...)
                                           (one GLSL source, two manifest
                                            programs - see the report)

RELIEF_FRAG / MACRO_FRAG are deliberately NOT copied: the SPIR-V comes from
engine_vr/shaders_gen via Host.build() (Phase 1), so the cache-dir GLSL dump
has no purpose here and writing it would be a second source of truth.

Import-safe: constants come from cache_schema; nothing opens a device until a
run_* helper (or an explicit passes_vr.open_host()) installs a Host.
"""

import os  # noqa: F401  (used by the copied relief_post NO_CONES check)
import time  # noqa: F401  (used by the copied timing prints)

import numpy as np
from scipy import ndimage  # noqa: F401  (used by the copied numpy stages)

from engine_vr import cache_schema as cs
from engine_vr import passes_vr as pv
from engine_vr import features_vr as features  # noqa: F401  (copied call sites)

# Single source for the bake constants (copied from engine/bake.py:15-23 in
# cache_schema); the copied code below refers to these module globals exactly
# as the original referred to its own.
CACHE = cs.CACHE
FAR_HALF = cs.FAR_HALF
FAR_RES = cs.FAR_RES
NEAR_HALF = cs.NEAR_HALF
NEAR_RES = cs.NEAR_RES
TARN_C = cs.TARN_C
TARN_LEVEL = cs.TARN_LEVEL

# _bake_pass keeps the engine name and the engine positional/keyword shape
# (frag, res, uniforms, textures=(), tile=512, comps=1) so every copied call
# site is untouched; the body is passes_vr.bake_pass.
_bake_pass = pv.bake_pass


class _Gl:
    """The engine/gl.py entry points the copied stages use, VR-backed.

    engine/bake.py only ever calls gl.tex2d(arr, filt=..., repeat=...) inside
    the copied regions, which is passes_vr.upload with the same dtype rules
    (uint8 -> f1, float16 -> f2, else float32) and the same NEAREST/LINEAR and
    CLAMP/REPEAT choices.
    """

    @staticmethod
    def tex2d(arr, filt="linear", repeat=False, dtype=None):
        return pv.upload(arr, filt=filt, repeat=repeat, dtype=dtype)


gl = _Gl()


def build_plan():
    """Pure-data description of every bake pass this module drives (T4).

    Uniform keys are exactly the manifest uniform_block members of each
    program (validated by port/gates/gate_p3.py); values are the defaults the
    copied drivers pass on a full far-map run. "dispatch" mirrors the tiled
    draw loop: the engine walks the tile grid and calls c.finish() per tile,
    which in engine_vr is one submit + wait per tile.
    """
    def tiles(res, tile):
        return {"kind": "tiles", "tile": tile, "tiles": (res // tile) ** 2,
                "flush": "per tile"}

    stream = np.zeros((96, 4), np.float32)
    trail = np.zeros((32, 4), np.float32)
    ao_res = FAR_RES // 2
    return {
        "height_far": {
            "program": "bake_height",
            "uniforms": {"uCenter": (0.0, 0.0), "uHalf": FAR_HALF,
                         "uRes": float(FAR_RES), "uFine": 0},
            "storage_images": [], "res": FAR_RES,
            "dispatch": tiles(FAR_RES, 256),
        },
        "near_from_far": {
            "program": "bake_near",
            "uniforms": {"uFarHalf": FAR_HALF, "uFarRes": float(FAR_RES),
                         "uCenter": (0.0, 0.0), "uHalf": NEAR_HALF,
                         "uRes": float(NEAR_RES)},
            "storage_images": [], "res": NEAR_RES,
            "dispatch": tiles(NEAR_RES, 512),
        },
        "relief_macro": {
            "program": "bake_macro",
            "uniforms": {"uHalf": FAR_HALF, "uRes": float(FAR_RES),
                         "uDbg": 0},
            "storage_images": [], "res": FAR_RES,
            "dispatch": tiles(FAR_RES, 512),
        },
        "relief_granite": {
            "program": "bake_relief",
            "uniforms": {"uHalf": FAR_HALF, "uRes": float(FAR_RES),
                         "uStream": stream, "uStreamN": 0, "uTarn": TARN_C,
                         "uDbg": 0},
            "storage_images": [], "res": FAR_RES,
            "dispatch": tiles(FAR_RES, 512),
        },
        "ao_far": {
            "program": "bake_ao_far",
            "uniforms": {"uCenter": (0.0, 0.0), "uHalf": FAR_HALF,
                         "uRes": float(ao_res), "uFarHalf": FAR_HALF,
                         "uStep0": 8.0},
            "storage_images": [], "res": ao_res,
            "dispatch": tiles(ao_res, 512),
        },
        "ao_near": {
            "program": "bake_ao_near",
            "uniforms": {"uCenter": (0.0, 0.0), "uHalf": NEAR_HALF,
                         "uRes": float(NEAR_RES), "uFarHalf": FAR_HALF,
                         "uStep0": 1.5},
            "storage_images": [], "res": NEAR_RES,
            "dispatch": tiles(NEAR_RES, 512),
        },
        "world_forest": {
            "program": "bake_forest",
            "uniforms": {"uHasErosion": 0, "uHalf": FAR_HALF,
                         "uRes": float(FAR_RES), "uTarn": TARN_C,
                         "uTarnLevel": TARN_LEVEL, "uStream": stream,
                         "uStreamN": 0, "uTrail": trail, "uTrailN": 0},
            "storage_images": [], "res": FAR_RES,
            "dispatch": tiles(FAR_RES, 1024),
        },
    }


def run_all(force=False, erode=None, host=None, log=print):
    """Full bake chain (the bake.bake_all equivalent) on one Host.

    Opens a Host only when the caller did not supply one, and releases the GPU
    slot on the way out (gpuslot_vr was acquired before Host.open()).
    """
    own = host is None
    if own:
        host = pv.open_host(cache_dir=str(cs.cache_dir()), log=log)
    try:
        return bake_all(force=force)
    finally:
        if own:
            pv.close_host()


def run_eroded_bake(host=None, erode=None, force=False, log=print):
    """bake_heights() wired to the VR erosion driver (the bake_final path).

    engine/bake.py takes erosion as a callback: erode(hfar, cell_m) -> dict
    with height/flow/talus/sediment, which is exactly what erosion_vr.erode
    returns.
    """
    if erode is None:
        from engine_vr import erosion_vr

        def erode(hfar, cell_m, **kw):
            return erosion_vr.erode(hfar, cell_m, log=log, **kw)
    own = host is None
    if own:
        host = pv.open_host(cache_dir=str(cs.cache_dir()), log=log)
    try:
        return bake_heights(force=force, erode=erode)
    finally:
        if own:
            pv.close_host()



# ==== BEGIN ENGINE COPIES (generated by engine_vr/build_p3.py; do not edit) ====

# --------------------------------------------------------------------------------
# Copied from the moderngl build by engine_vr/build_p3.py: verbatim
# except the substitutions named in each block header; no constant,
# threshold or ordering was retyped. Per-line noqa comments may be
# added by build_p3.py for ruff (ast-dump verified, comments only).
# --------------------------------------------------------------------------------

# ---- copied from engine/bake.py:46-95 ('import features')
def bake_heights(far_res=FAR_RES, near_res=NEAR_RES, force=False, erode=None):
    """far map: analytic (+ optional erosion); near map: upsampled far map + fine
    detail."""
    fp, np_ = CACHE / "h_far.npy", CACHE / "h_near.npy"
    if fp.exists() and np_.exists() and not force:
        return np.load(fp), np.load(np_)
    raw = CACHE / "h_far_raw.npy"
    if raw.exists() and not force:
        hfar = np.load(raw)
    else:
        print("baking analytic far heights ...", flush=True)
        hfar = _bake_pass("bake_height.frag", far_res,
                          {"uCenter": (0.0, 0.0), "uHalf": FAR_HALF, "uRes": float(far_res), "uFine": 0}, tile=256)  # noqa: E501
        np.save(raw, hfar.astype(np.float32))
    if erode is not None:
        print("eroding ...", flush=True)
        out = erode(hfar, 2 * FAR_HALF / far_res)
        hfar = out["height"].astype(np.float32)
        # keep the raw erosion result per world cache (bake_final reuses it unless --run-erosion)  # noqa: E501
        np.save(CACHE / "erosion_height.npy", hfar)
        for k in ("flow", "talus", "sediment"):
            if k in out and out[k] is not None:
                np.save(CACHE / f"{k}.npy", np.asarray(out[k], np.float32))
                np.save(CACHE / f"erosion_{k}.npy", np.asarray(out[k], np.float32))
    # glacial lakes: a clear shallow shelf along the shore (< ~0.5 m for the first ~10 m: the stones on it  # noqa: E501
    # visible), then a drop-off into the teal deep (~2 m at 20 m, ~8 m at 30 m)
    # (shore4: was dd * (1 + 1.5 exp(-dd / 2)) - 0.7 m deep 5 m out, 1.4 m at 10 m)
    wet = hfar < 0.0
    dd = -hfar[wet]
    t_ = np.clip((dd - 0.5) / 1.5, 0.0, 1.0)
    hfar[wet] = -(0.8 * dd + 2.0 * t_ * t_ * (3.0 - 2.0 * t_) * np.minimum(dd, 2.0))   # deep: ~0.8 dd + 4  # noqa: E501
    # (r4 W6 / W11 / W4) the main lake drops off sooner: 1.5-2 m by 6-8 m from the shore, >= 5 m by 20 m, 2 m within ~3 m  # noqa: E501
    # under steep (forested) banks, a ragged shelf (+-1.2 m, 4-12 m), no flats within +-0.3 m of the level near the shore  # noqa: E501
    hfar = shore_bathymetry(hfar)
    # (valleyA) a few gentle pebble beaches
    hfar = beaches(hfar)
    # (session 6) keep the eroded map before the relief / features: tools can re-run relief_post and the  # noqa: E501
    # later steps without the ~6 min erosion (out/s6/rerelief.py)
    np.save(CACHE / "h_eroded.npy", hfar.astype(np.float32))
    # hand-placed features (waterfall cliff, inflow stream) carved after erosion
    from engine_vr import features_vr as features  # noqa: F811
    feat = features.place(hfar)
    features.save_meta(feat)
    # granite relief (ribs, couloirs, cliff bands) that the hillslope erosion planed away  # noqa: E501
    hfar = relief_post(hfar, feat)
    hfar = river_post(hfar)
    hfar, slv, _, _ = features.carve(hfar, FAR_HALF, feat)
    np.save(CACHE / "stream_level_far.npy", slv.astype(np.float32))
    np.save(fp, hfar.astype(np.float32))
    hnear = bake_near_from_far(hfar, feat)
    return hfar, hnear


# ---- copied from engine/bake.py:98-225 ('import features')
def _lake_window(h, cell):
    """main-lake mask (flood from the axis seed) in a +-2600 m window of a map centred
    on the origin."""
    c0 = h.shape[0] // 2
    win = min(int(2600 / cell), c0)
    sl = (slice(c0 - win, c0 + win), slice(c0 - win, c0 + win))
    lab, _ = ndimage.label(h[sl] < 0.0)
    ax0 = 150.0 * np.sin(0.5) + 45 * np.sin(2.1)
    main = lab == lab[win, int(ax0 / cell) + win]
    return sl, main


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shore_bathymetry(hfar):
    """(r4 W6 / W11 / W4, Track E / B requests) main-lake bed: >= ~0.7 m at the first
    wet cell, 1.5-2 m by 6-8 m, >= 5 m by
    20 m from the shore (only deepened, never raised: the outline stays), compressed
    ~2.5x under steep banks (forested
    shores: 2 m within ~3 m), a ragged +-1.2 m shelf (~10 m noise) from 2 m out; land
    within 8 m of the water >= 0.1 m +
    0.11 m per m (no flats within +-0.3 m of the level along the shore)."""
    t0 = time.time()
    cell = 2 * FAR_HALF / hfar.shape[0]
    sl, main = _lake_window(hfar, cell)
    h = hfar[sl].copy()
    din = ndimage.distance_transform_edt(main).astype(np.float32) * cell
    dout = ndimage.distance_transform_edt(~main).astype(np.float32) * cell
    x = np.maximum(din - 0.5 * cell, 0.0)
    land = np.where(main, -1e4, h).astype(np.float32)
    bank = ndimage.maximum_filter(land, size=int(24 / cell) | 1)            # highest land within ~12 m  # noqa: E501
    steep = _ss(2.0, 8.0, bank)
    xs = x * (1.0 + 1.5 * steep)
    prof = 0.35 + 1.65 * _ss(0.0, 7.0, xs) + 3.2 * _ss(8.0, 20.0, xs)
    nz = fbm_np(h.shape, 10.0 / cell, 23, octaves=2)
    nz = np.clip(nz / max(float(nz.std()), 1e-6), -1.5, 1.5) / 1.5
    prof = prof + 1.2 * nz * _ss(2.0, 8.0, x)
    hn = np.where(main, -np.maximum(-h, prof), h)
    xo = np.maximum(dout - 0.5 * cell, 0.0)
    hn = np.where((~main) & (xo < 8.0), np.maximum(hn, 0.1 + 0.11 * xo), hn)
    out = hfar.copy()
    out[sl] = hn
    print(f"  shore bathymetry: deepened {int((hn < h - 0.05).sum())} cells (max {float((h - hn).max()):.1f} m), "  # noqa: E501
          f"bank cells raised {int((hn > h + 0.05).sum())} ({time.time()-t0:.1f}s)", flush=True)  # noqa: E501
    return out


def shore_near(hnear):
    """(r4 W4) the same no-flat rule on the 1 m near map after its fine detail: land
    within 8 m of the main lake >= 0.1 m +
    0.11 m per m, the lake within 6 m <= -(0.4 + 0.15 m per m)."""
    t0 = time.time()
    cell = 2 * NEAR_HALF / hnear.shape[0]
    sl, main = _lake_window(hnear, cell)
    h = hnear[sl]
    din = ndimage.distance_transform_edt(main).astype(np.float32) * cell
    dout = ndimage.distance_transform_edt(~main).astype(np.float32) * cell
    xo = np.maximum(dout - 0.5 * cell, 0.0); xi = np.maximum(din - 0.5 * cell, 0.0)  # noqa: E702
    hn = np.where((~main) & (xo < 8.0), np.maximum(h, 0.1 + 0.11 * xo), h)
    hn = np.where(main & (xi < 6.0), np.minimum(hn, -(0.4 + 0.15 * xi)), hn)
    print(f"  shore (near): {int((hn != h).sum())} cells changed ({time.time()-t0:.1f}s)", flush=True)  # noqa: E501
    out = hnear.copy()
    out[sl] = hn
    return out


BEACH_SLOPE = 6.0          # deg, the beach face above the waterline
BEACH_W = (12.0, 30.0)     # m from the waterline: full beach / blended back into the bank  # noqa: E501
# keep-outs (x, z, r): the waterfall cliff (+ the boulder shore next to it), the stream mouth; the trail shore  # noqa: E501
BEACH_KEEP = [(515.0, -550.0, 260.0), (195.0, 1545.0, 160.0)]


def beaches(hfar):
    """(valleyA) gentle pebble beaches on ~25-30 % of the main lake shore, in ~150-400 m
    stretches: land within
    BEACH_W[0] m of the waterline is lowered onto a BEACH_SLOPE deg face 0.15 m above
    the water (only lowered,
    never below +0.15 m, so the lake outline is unchanged), blended back into the
    original bank by BEACH_W[1] m.
    Not near the waterfall, the stream mouth or the east-shore trail (z 250..650)."""
    t0 = time.time()
    res = hfar.shape[0]
    cell = 2 * FAR_HALF / res
    win = int(2600 / cell)
    c0 = res // 2
    sl = (slice(c0 - win, c0 + win), slice(c0 - win, c0 + win))
    h = hfar[sl]
    lab, _ = ndimage.label(h < 0.0)
    ax0 = 150.0 * np.sin(0.5) + 45 * np.sin(2.1)
    seed = lab[win, int(ax0 / cell) + win]
    main = lab == seed
    del lab
    dl = ndimage.distance_transform_edt(~main).astype(np.float32) * cell     # m from the lake (0 in it)  # noqa: E501
    n = h.shape[0]
    c = (np.arange(n) + 0.5) * cell - win * cell
    X, Z = np.meshgrid(c, c)
    nz = fbm_np(h.shape, 320.0 / cell, 11, octaves=2)
    bm = np.clip((nz - 0.12) / 0.30, 0.0, 1.0)
    bm = bm * bm * (3 - 2 * bm)
    for kx, kz, kr in BEACH_KEEP:
        bm *= np.clip((np.hypot(X - kx, Z - kz) - kr) / 60.0, 0.0, 1.0)
    bm *= 1.0 - (X > 0) * np.clip((Z - 230.0) / 40.0, 0, 1) * np.clip((670.0 - Z) / 40.0, 0, 1)  # noqa: E501
    w = np.clip((BEACH_W[1] - dl) / (BEACH_W[1] - BEACH_W[0]), 0.0, 1.0)
    w = w * w * (3 - 2 * w) * bm * (~main)
    prof = 0.15 + np.tan(np.radians(BEACH_SLOPE)) * dl
    hn = h - w * np.maximum(h - prof, 0.0)
    shore = (~main) & (dl <= cell * 1.5)
    frac = float((bm[shore] > 0.5).mean())
    print(f"  beaches: {frac*100:.0f} % of the shore cells, max lowering {float((h - hn).max()):.1f} m "  # noqa: E501
          f"({time.time()-t0:.1f}s)", flush=True)
    out = hfar.copy()
    out[sl] = hn
    return out


def bake_near_from_far(hfar, feat):
    """near map (1 m, +-2 km): upsampled far map + fine detail, features carved; saves
    the near caches."""
    from engine_vr import features_vr as features  # noqa: F811
    far_res = hfar.shape[0]
    near_res = NEAR_RES
    np_ = CACHE / "h_near.npy"
    tf = gl.tex2d(hfar, filt="nearest")
    hnear = _bake_pass("bake_near.frag", near_res,
                       {"uCenter": (0.0, 0.0), "uHalf": NEAR_HALF, "uRes": float(near_res),  # noqa: E501
                        "uFarHalf": FAR_HALF, "uFarRes": float(far_res)}, textures=[("uHFar", tf)], tile=512)  # noqa: E501
    tf.release()
    hnear = shore_near(hnear)
    hnear, slv_n, foam_n, ang_n = features.carve(hnear, NEAR_HALF, feat)
    isst = (slv_n > -1000.0).astype(np.float32)
    np.save(CACHE / "stream_level_near.npy", slv_n.astype(np.float32))
    np.save(CACHE / "stream_near.npy", np.stack([np.cos(ang_n) * isst, np.sin(ang_n) * isst, foam_n * isst, isst], -1).astype(np.float16))  # noqa: E501
    np.save(np_, hnear.astype(np.float32))
    return hnear


# ---- copied from engine/bake.py:288-312 ('src = CACHE / "_bake_macro.frag"\n    src.write_text(MACRO_FRAG)')  # noqa: E501
def _blur_ds(h, sigma_m, f):
    """gaussian blur (sigma in m) computed on an f x f block-mean downsample (mirror
    edges like the
    renderer's far-map lookup); returned at the low resolution (sampled bilinear on
    the GPU)."""
    res = h.shape[0]
    cell = 2 * FAR_HALF / res
    lo = h.reshape(res // f, f, res // f, f).mean((1, 3)) if f > 1 else h
    return ndimage.gaussian_filter(lo.astype(np.float32), sigma_m / (cell * f), mode="reflect")  # noqa: E501


def macro_post(hfar, dbg=False):
    """round 2: dominant massifs grown out of the eroded map (heightfn.glsl
    macroRelief)."""
    t0 = time.time()
    L = _blur_ds(hfar, 250.0, 8)
    B = _blur_ds(hfar, 40.0, 2)
    print(f"  macro blurs in {time.time()-t0:.1f}s", flush=True)
    src = "bake_macro"
    res = hfar.shape[0]
    ht = gl.tex2d(np.ascontiguousarray(hfar, np.float32), filt="nearest")
    lt = gl.tex2d(L); bt = gl.tex2d(B)  # noqa: E702
    out = _bake_pass(str(src), res, {"uHalf": FAR_HALF * res / FAR_RES, "uRes": float(res), "uDbg": 1 if dbg else 0},  # noqa: E501
                     textures=[("uH", ht), ("uL", lt), ("uB", bt)], comps=4 if dbg else 1, tile=512)  # noqa: E501
    for t in (ht, lt, bt):
        t.release()
    return out.astype(np.float32)


# ---- copied from engine/bake.py:315-422 (verbatim)
HERO_CONES = [(-2680.0, -320.0, 620.0, 50.0)]   # (session 3: R 620 -> 820 m, lift 38 -> 50 m; r4 M6: 1150 / 70 was a 1.5 km  # noqa: E501
                                                 # pancake on the longer, flatter hanging-valley floor (M5); b2 900 / 60  # noqa: E501
                                                 # still a pale disc -> 620 / 50, elongated, +-32..63 deg fan)  # noqa: E501


def talus_cones(hfar, feat=None, max_cones=900):
    """round 2: real talus cones / fans at the couloir exits (plus a big pale hero
    cone). Apices are picked
    on a 16 m grid where a steep couloir (high erosion flow, > ~45 deg above) opens
    onto repose-angle
    ground; each builds a slightly concave cone (~34 deg near the apex, ~26 deg at
    its toe) that only adds
    material (max with the ground, <= ~45 m thick). Never within ~450 m of the lake /
    tarn / inflow stream.
    Returns the new map and a 0..1 deposit mask (written into talus.npy, so
    bake_forest paints scree)."""
    t0 = time.time()
    res = hfar.shape[0]
    cell = 2 * FAR_HALF / res
    f = 4
    lo = hfar.reshape(res // f, f, res // f, f).mean((1, 3)).astype(np.float32)
    c16 = cell * f
    gz, gx = np.gradient(ndimage.gaussian_filter(lo, 1.0), c16)
    sl = np.hypot(gx, gz)
    # steepest slope within ~100 m uphill: max filter of the slope (any direction; the walk below checks uphill)  # noqa: E501
    smax = ndimage.maximum_filter(sl, size=13)
    fp = CACHE / "flow.npy"
    if fp.exists():
        fl = np.load(fp).astype(np.float32)
        fl = np.log1p(np.maximum(fl, 0)); fl = fl / max(np.percentile(fl, 99.9), 1e-6)  # noqa: E702
        fl = fl.reshape(res // f, f, res // f, f).max((1, 3))
        fl = ndimage.maximum_filter(fl, size=3)
    else:
        fl = np.zeros_like(lo)
    xs = (np.arange(res // f) + 0.5) * c16 - FAR_HALF
    X, Z = np.meshgrid(xs, xs)
    # guards: lake (h < 60 m within 1 km), tarn, inflow stream
    guard = np.ones_like(lo, bool)
    guard &= np.hypot(X - TARN_C[0], Z - TARN_C[1]) > 420.0
    if feat and "stream" in feat:
        st = np.asarray(feat["stream"], np.float32)
        for a, b in zip(st[:-1], st[1:]):
            ab = b[:2] - a[:2]; L2 = max(float(ab @ ab), 1e-6)  # noqa: E702
            tt = np.clip(((X - a[0]) * ab[0] + (Z - a[1]) * ab[1]) / L2, 0, 1)
            guard &= np.hypot(X - a[0] - ab[0] * tt, Z - a[1] - ab[1] * tt) > 450.0
    # (r4) deposit keep-outs (the bigger fans reach further): the inflow stream / cascade (150 m), the trail (150 m), the  # noqa: E501
    # waterfall (300 m), the tarn's water and its shore (below 530 m within 650 m)
    ko = (np.hypot(X - TARN_C[0], Z - TARN_C[1]) < 650.0) & (lo < 530.0)
    ko |= np.hypot(X - 515.0, Z + 550.0) < 300.0
    for key in ("stream", "trail"):
        if feat and key in feat:
            st = np.asarray(feat[key], np.float32)
            for a, b in zip(st[:-1], st[1:]):
                ab = b[:2] - a[:2]; L2 = max(float(ab @ ab), 1e-6)  # noqa: E702
                tt = np.clip(((X - a[0]) * ab[0] + (Z - a[1]) * ab[1]) / L2, 0, 1)
                ko |= np.hypot(X - a[0] - ab[0] * tt, Z - a[1] - ab[1] * tt) < 150.0
    koS = ndimage.gaussian_filter(ko.astype(np.float32), 2.5)      # soft (~60 m) edge: no deposit cliffs  # noqa: E501
    # (r4 M7: apices 120 m (was 450 m) from the lake, from 250 m (500) up, weaker couloirs too (fl 0.42 -> 0.35): big pale  # noqa: E501
    # fans out of the couloirs through the forest, a few down to the shore; the deposit is clipped  # noqa: E501
    # below +2 m, so no fan fills the lake)
    low = ndimage.binary_dilation(lo < 40.0, iterations=int(120 / c16))
    guard &= ~low
    cand = (sl > 0.45) & (sl < 0.80) & (smax > 1.15) & (lo > 250.0) & (lo < 2300.0) & guard & (fl > 0.35)  # noqa: E501
    score = np.where(cand, fl * (smax - 0.8), 0.0)
    # one apex per ~380 m (r4: ~650 m - with the larger fans 380 m buried ~10 % of the map in debris)  # noqa: E501
    peak = (score > 0) & (score == ndimage.maximum_filter(score, size=int(650 / c16) | 1))  # noqa: E501
    iy, ix = np.nonzero(peak)
    order = np.argsort(-score[iy, ix])[:max_cones]
    iy, ix = iy[order], ix[order]
    out = hfar.copy()
    dep = np.zeros_like(hfar)
    rng = np.random.default_rng(7)
    cones = []
    for y0, x0 in zip(iy, ix):
        # the cone's size follows the face above it (taller walls shed bigger cones)
        # (r4 M7: 180-520 m -> 220-700 m, elongated down the fall line (below): 300-900 m long tongues, b1's 250-950 m  # noqa: E501
        # round cones were 1-2 km pancakes on the cirque floors)
        R = float(np.clip(220.0 + 330.0 * (smax[y0, x0] - 1.0) + 120.0 * rng.random(), 220.0, 700.0))  # noqa: E501
        cones.append((y0, x0, R, 6.0))
    # hand-placed hero cones: a big pale talus cone below the tarn's west wall (x, z, R, apex lift)  # noqa: E501
    for hx, hz, R, lift in HERO_CONES:
        cones.append((int((hz + FAR_HALF) / c16), int((hx + FAR_HALF) / c16), R, lift))
    nh = len(cones) - len(HERO_CONES)
    for k, (y0, x0, R, lift) in enumerate(cones):
        ha = float(lo[y0, x0]) + lift
        cx, cz = X[y0, x0], Z[y0, x0]
        # fall direction at the apex: the cone is a fan opening downhill (apex half-angle ~70 deg)  # noqa: E501
        g = np.array([gx[y0, x0], gz[y0, x0]]); g = g / (np.linalg.norm(g) + 1e-6)  # noqa: E702
        n = int(R / cell) + 2
        r0, c0 = int((cz + FAR_HALF) / cell), int((cx + FAR_HALF) / cell)
        ra, rb, ca, cb = max(r0 - n, 0), min(r0 + n, res), max(c0 - n, 0), min(c0 + n, res)  # noqa: E501
        zz = (np.arange(ra, rb) + 0.5)[:, None] * cell - FAR_HALF - cz
        xx = (np.arange(ca, cb) + 0.5)[None, :] * cell - FAR_HALF - cx
        r = np.hypot(xx, zz)
        down = -(xx * g[0] + zz * g[1]) / (r + 1e-3)          # cos of the angle to the fall direction  # noqa: E501
        # (r4) elongated: the radius across the fall line counts 1.7x (a tongue down the slope, convex across)  # noqa: E501
        alongD = -(xx * g[0] + zz * g[1]); acrossD = xx * g[1] - zz * g[0]  # noqa: E702
        re = np.hypot(alongD, 1.7 * acrossD)
        u = np.clip(re / R, 0, 1)
        cone = ha - R * (0.68 * u - 0.12 * u * u)             # 34 deg at the apex, ~26 deg at the toe  # noqa: E501
        fan = np.clip((down - 0.45) / 0.4, 0, 1) * (1.0 - u) ** 0.5   # opens downhill (r4: +-32..63 deg), fades at the toe  # noqa: E501
        r = re
        sub = out[ra:rb, ca:cb]
        add = np.clip(cone - sub, 0.0, 32.0 + lift) * fan
        kow = koS[np.ix_(np.arange(ra, rb) // f, np.arange(ca, cb) // f)]
        add = np.where(r < R, add, 0.0) * _ss(2.0, 30.0, sub) * (1.0 - _ss(0.15, 0.55, kow))  # noqa: E501
        out[ra:rb, ca:cb] = sub + add
        dm = np.clip(add / 6.0, 0, 1) * (fan > 0.05)
        if k >= nh:   # hero cone: its whole fan is bare debris down to the toe, also where it adds < 6 m  # noqa: E501
            dm = np.maximum(dm, np.clip((fan - 0.05) / 0.25, 0, 1) * (r < 0.95 * R))
        dep[ra:rb, ca:cb] = np.maximum(dep[ra:rb, ca:cb], dm)
    print(f"  talus cones: {len(cones)} in {time.time()-t0:.1f}s", flush=True)
    return out, dep


# ---- copied from engine/bake.py:425-461 (verbatim)
def river_post(hfar):
    """(r4 M9) a meandering 16-26 m river down the lower valley's floodplain (z -2800 ..
    -16200): the thalweg of the 40 m
    blurred floor (smoothed over ~200 m) plus 260 / 610 m meanders, a 2.2 m deep
    parabolic channel below the floodplain
    level. Its level (floodplain - 0.5 m) goes into the water map (bake_world_maps
    reads cache/river.npz)."""
    t0 = time.time()
    res = hfar.shape[0]
    cell = 2 * FAR_HALF / res
    r0, r1 = int((-16200.0 + FAR_HALF) / cell), int((-2800.0 + FAR_HALF) / cell)
    c0, c1 = int((-1500.0 + FAR_HALF) / cell), int((1500.0 + FAR_HALF) / cell)
    blk = ndimage.gaussian_filter(hfar[r0:r1, c0:c1].astype(np.float32), 10.0)
    zs = (np.arange(r0, r1) + 0.5) * cell - FAR_HALF
    xs = (np.arange(c0, c1) + 0.5) * cell - FAR_HALF
    ax = 150.0 * np.sin(zs * 0.00105 + 0.5) + 45.0 * np.sin(zs * 0.0029 + 2.1)
    pen = 0.02 * np.abs(xs[None, :] - ax[:, None])             # stay near the valley axis on flat stretches  # noqa: E501
    k = np.argmin(blk + pen, 1)
    xt = ndimage.gaussian_filter1d(xs[k].astype(np.float32), 50.0)
    hb = ndimage.gaussian_filter1d(blk[np.arange(len(k)), np.clip(((xt - xs[0]) / cell).astype(int), 0, len(xs) - 1)], 10.0)  # noqa: E501
    xr = xt + 34.0 * np.sin(zs * (2 * np.pi / 260.0)) + 22.0 * np.sin(zs * (2 * np.pi / 610.0) + 1.3)  # noqa: E501
    w = 16.0 + 10.0 * (0.5 + 0.5 * np.sin(zs * (2 * np.pi / 1300.0) + 0.7))
    # the river flows south and ends where the valley closes: rows contiguous from the north end whose floor is no more  # noqa: E501
    # than 4 m above every floor upstream of them
    upmin = np.minimum.accumulate(hb[::-1])[::-1]
    ok = hb <= np.concatenate([upmin[1:], upmin[-1:]]) + 4.0
    bad = np.nonzero(~ok)[0]
    i0 = int(bad.max()) + 1 if len(bad) else 0
    zs, xr, w, hb, r0 = zs[i0:], xr[i0:], w[i0:], hb[i0:], r0 + i0
    if len(zs) < 10:
        print("  river: no valid floodplain", flush=True)
        return hfar
    out = hfar.copy()
    sub = out[r0:r1, c0:c1]
    d = np.abs(xs[None, :] - xr[:, None])
    depth = 2.2 * np.clip(1.0 - (d / (0.5 * w[:, None])) ** 2, 0.0, 1.0)
    sub[:] = np.where(depth > 0.0, np.minimum(sub, hb[:, None] - depth), sub)
    np.savez(CACHE / "river.npz", z=zs.astype(np.float32), x=xr.astype(np.float32), w=w.astype(np.float32), hb=hb.astype(np.float32))  # noqa: E501
    print(f"  river: {len(zs)} rows, floodplain {hb.max():.0f} .. {hb.min():.0f} m ({time.time()-t0:.1f}s)", flush=True)  # noqa: E501
    return out


# ---- copied from engine/bake.py:464-524 (verbatim)
DEFIN_HALF = 9000.0     # m: the de-fin / de-spike window (everything the film sees in detail)  # noqa: E501


def defin_post(hfar, feat=None):
    """(r4 M2 b / M15) organ pipes and needle fins: (1) on steep (> ~42 deg) faces above
    600 m the fine relief (h minus a
    40 m blur) is blurred ACROSS the fall line (5 taps over +-28 m along strike,
    weight 0.7) - fall-line fins / flutes go,
    ledges and steps across the slope stay; (2) de-spike above 300 m: h <= median9x9
    + 15 m (+25 m on crests = within
    30 m of the highest point 200 m around). Not around the tarn (500 m), the inflow
    stream (300 m), the waterfall."""
    t0 = time.time()
    res = hfar.shape[0]
    cell = 2 * FAR_HALF / res
    c0 = res // 2; win = int(DEFIN_HALF / cell)  # noqa: E702
    sl = (slice(c0 - win, c0 + win), slice(c0 - win, c0 + win))
    h = hfar[sl].astype(np.float32)
    n = h.shape[0]
    xs = (np.arange(n) + 0.5) * cell - win * cell
    X, Z = np.meshgrid(xs, xs)
    prot = np.clip((np.hypot(X - TARN_C[0], Z - TARN_C[1]) - 500.0) / 300.0, 0, 1)
    prot *= np.clip((np.hypot(X - 515.0, Z + 550.0) - 250.0) / 150.0, 0, 1)
    if feat and "stream" in feat:
        st = np.asarray(feat["stream"], np.float32)
        ds = np.full(h.shape, 1e9, np.float32)
        for a, b in zip(st[:-1], st[1:]):
            ab = b[:2] - a[:2]; L2 = max(float(ab @ ab), 1e-6)  # noqa: E702
            tt = np.clip(((X - a[0]) * ab[0] + (Z - a[1]) * ab[1]) / L2, 0, 1)
            ds = np.minimum(ds, np.hypot(X - a[0] - ab[0] * tt, Z - a[1] - ab[1] * tt))
        prot *= np.clip((ds - 300.0) / 200.0, 0, 1)
        del ds
    del X, Z
    # (1) across-fall-line blur of the fine relief
    B = ndimage.gaussian_filter(h, 40.0 / cell / 2.0)              # ~40 m blur (sigma 20 m)  # noqa: E501
    fr = h - B
    gS = ndimage.gaussian_filter(h, 60.0 / cell / 2.0)
    gz, gx = np.gradient(gS, cell)
    slope = np.hypot(gx, gz)
    tx, tz = -gz / (slope + 1e-6), gx / (slope + 1e-6)               # strike direction
    # (b2: weight 0.7 left the fine relief contour-terraced - across/along 0.63, a wood-grain hatch on the shot 7 walls)  # noqa: E501
    w1 = 0.45 * _ss(0.8, 1.0, slope) * _ss(550.0, 650.0, h) * prot
    rr, cc = np.mgrid[0:n, 0:n].astype(np.float32)
    acc = fr.copy()
    for o in (-28.0, -14.0, 14.0, 28.0):
        acc += ndimage.map_coordinates(fr, [rr + tz * (o / cell), cc + tx * (o / cell)], order=1, mode="nearest")  # noqa: E501
    del rr, cc
    acc /= 5.0
    h1 = h - w1 * (fr - acc)
    d1 = h - h1
    del acc, fr, B, gS
    # (2) de-spike
    med = ndimage.median_filter(h1, size=9)
    hmx = ndimage.maximum_filter(h1, size=int(400.0 / cell) | 1)
    crest = (hmx - h1) < 30.0
    lim = med + 15.0 + 25.0 * crest
    w2 = _ss(250.0, 350.0, h1) * prot
    h2 = h1 - w2 * np.maximum(h1 - lim, 0.0)
    d2 = h1 - h2
    out = hfar.copy()
    out[sl] = h2
    print(f"  de-fin: |dh| mean {float(np.abs(d1[w1 > 0.3]).mean()) if (w1 > 0.3).any() else 0:.2f} m on {int((w1 > 0.3).sum())} cells; "  # noqa: E501
          f"de-spike: {int((d2 > 1).sum())} cells > 1 m, {int((d2 > 10).sum())} > 10 m (max {float(d2.max()):.0f} m) "  # noqa: E501
          f"({time.time()-t0:.1f}s)", flush=True)
    return out


# ---- copied from engine/bake.py:527-582 ('src = CACHE / "_bake_relief.frag"\n    src.write_text(RELIEF_FRAG)')  # noqa: E501
def relief_post(hfar, feat=None, dbg=False):
    """post-erosion granite relief on the far map (heightfn.glsl reliefPost); zero below
    ~300 m,
    around the tarn and along the inflow stream, so the shores / film paths are
    untouched.
    (round 2: preceded by the macro pass - dominant massifs, see macro_post)"""
    if not dbg:
        hfar = macro_post(hfar)
    src = "bake_relief"
    res = hfar.shape[0]
    ht = gl.tex2d(np.ascontiguousarray(hfar, np.float32))
    st = np.zeros((96, 4), np.float32); n = 0  # noqa: E702
    if feat and "stream" in feat:
        a = np.asarray(feat["stream"], np.float32)[:96]; st[: len(a)] = a; n = len(a)  # noqa: E702
    out = _bake_pass(str(src), res, {"uHalf": FAR_HALF * res / FAR_RES, "uRes": float(res), "uStream": st,  # noqa: E501
                                     "uStreamN": n, "uTarn": TARN_C, "uDbg": 1 if dbg else 0},  # noqa: E501
                     textures=[("uH", ht)], comps=4 if dbg else 1, tile=512)
    ht.release()
    out = out.astype(np.float32)
    if not dbg:
        out = defin_post(out, feat)
    if not dbg:
        # (session 4) thin needle spikes on the high crests (shot 7 at 1720 px): above ~1800 m nothing may  # noqa: E501
        # stand more than 8 m above the opening of the map. The opening is the max of four line openings  # noqa: E501
        # (7 cells ~ 28 m, 0 / 90 / 45 / 135 deg): a needle (narrow in every direction) is cut down, an arete  # noqa: E501
        # survives under the line along it (a 7x7 square opening blunted every arete by slope x 14 m), planar  # noqa: E501
        # faces and concave ground are unchanged
        t0 = time.time()
        c = 3
        op = None
        for d in range(4):
            fpL = np.zeros((7, 7), bool)
            for k in range(-c, c + 1):
                r, q = [(c, c + k), (c + k, c), (c + k, c + k), (c + k, c - k)][d]
                fpL[r, q] = True
            o = ndimage.grey_opening(out, footprint=fpL, mode="mirror")
            op = o if op is None else np.maximum(op, o)
            del o
        w = np.clip((out - 1700.0) / 200.0, 0.0, 1.0)
        cut = w * np.maximum(out - (op + 8.0), 0.0)
        out -= cut
        print(f"  crest spikes: {int((cut > 1.0).sum())} cells lowered > 1 m, {int((cut > 10.0).sum())} > 10 m "  # noqa: E501
              f"(max {float(cut.max()):.0f} m) in {time.time()-t0:.1f}s", flush=True)
        del op, w, cut
    if not dbg and os.environ.get("NO_CONES") != "1":
        out, dep = talus_cones(out, feat)
        # the deposits are scree for bake_forest (it reads the erosion talus map): talus = max(erosion talus,  # noqa: E501
        # cone deposits); the erosion's own map is kept in erosion_talus.npy
        tp = CACHE / "erosion_talus.npy"
        if not tp.exists():
            tp = CACHE / "talus.npy"
        if tp.exists():
            tal = np.load(tp).astype(np.float32)
            np.save(CACHE / "talus.npy", np.maximum(tal, dep).astype(np.float32))
        # (r4 M7) the cone deposits alone (bake_forest clears the forest off the fan cores: uCones)  # noqa: E501
        np.save(CACHE / "cones.npy", dep.astype(np.float16))
    return out


# ---- copied from engine/bake.py:585-602 ('_bake_pass("bake_ao.frag", hfar.shape[0]' + '_bake_pass("bake_ao.frag", hnear.shape[0]')  # noqa: E501
def bake_ao(hfar, hnear, force=False):
    fp, np_ = CACHE / "ao_far.npy", CACHE / "ao_near.npy"
    if fp.exists() and np_.exists() and not force:
        return np.load(fp), np.load(np_)
    print("baking AO ...", flush=True)
    tf = gl.tex2d(hfar)
    tn = gl.tex2d(hnear)
    aof = _bake_pass("bake_ao_far", hfar.shape[0] // 2,
                     {"uCenter": (0.0, 0.0), "uHalf": FAR_HALF, "uRes": float(hfar.shape[0] // 2),  # noqa: E501
                      "uFarHalf": FAR_HALF, "uStep0": 8.0},
                     textures=[("uH", tf), ("uHFar", tf)])
    aon = _bake_pass("bake_ao_near", hnear.shape[0],
                     {"uCenter": (0.0, 0.0), "uHalf": NEAR_HALF, "uRes": float(hnear.shape[0]),  # noqa: E501
                      "uFarHalf": FAR_HALF, "uStep0": 1.5},
                     textures=[("uH", tn), ("uHFar", tf)])
    tf.release(); tn.release()  # noqa: E702
    np.save(fp, aof.astype(np.float16)); np.save(np_, aon.astype(np.float16))  # noqa: E702
    return aof, aon


# ---- copied from engine/bake.py:605-722 ('import features as _feat')
def world_xy(res, half):
    c = (np.arange(res) + 0.5) / res * 2 - 1
    return c * half  # same for x (cols) and z (rows)


def smooth_noise(shape, cell, seed):
    """Smooth random field ~[-1,1] with feature size `cell` texels (cubic upsampled
    lattice)."""
    rng = np.random.default_rng(seed)
    gs = (max(2, int(np.ceil(shape[0] / cell)) + 3), max(2, int(np.ceil(shape[1] / cell)) + 3))  # noqa: E501
    g = rng.standard_normal(gs).astype(np.float32)
    z = ndimage.zoom(g, (shape[0] / (gs[0] - 3), shape[1] / (gs[1] - 3)), order=3)
    return np.clip(z[: shape[0], : shape[1]] / 1.2, -1.5, 1.5)


def fbm_np(shape, cell, seed, octaves=4):
    s = np.zeros(shape, np.float32); a = 0.5; norm = 0  # noqa: E702
    for i in range(octaves):
        s += a * smooth_noise(shape, max(cell / 2 ** i, 2), seed + i * 101)
        norm += a; a *= 0.5  # noqa: E702
    return s / norm


def bake_world_maps(hfar, force=False):
    """water level map (flood fills) + forest/material map (GPU) on the far grid (4 m
    cells)."""
    wp, fp = CACHE / "water_level.npy", CACHE / "forest.npy"
    if wp.exists() and fp.exists() and not force:
        return np.load(wp), np.load(fp)
    print("baking water + forest maps ...", flush=True)
    t0 = time.time()
    res = hfar.shape[0]
    cell = 2 * FAR_HALF / res

    def idx(x, z):
        return int((z + FAR_HALF) / cell), int((x + FAR_HALF) / cell)

    level = np.full(hfar.shape, -10000.0, np.float32)
    lab, n = ndimage.label(hfar < 0.0)
    r, c = idx(150.0 * np.sin(0.5) + 45 * np.sin(2.1), 0.0)
    main = lab == lab[r, c]
    if main.sum() > 0.25 * hfar.size:
        raise RuntimeError("main lake flood leaked to the map border")
    level[ndimage.binary_dilation(main, iterations=3)] = 0.0
    del lab
    # tarn: flood inside a window around it
    r0, c0 = idx(TARN_C[0] - 800, TARN_C[1] - 800)
    r1, c1 = idx(TARN_C[0] + 800, TARN_C[1] + 800)
    win = hfar[r0:r1, c0:c1] < TARN_LEVEL
    lab2, _ = ndimage.label(win)
    rr, cc = idx(*TARN_C)
    tarn = lab2 == lab2[rr - r0, cc - c0]
    if tarn.sum() * cell * cell > 40 * 1e4:
        print("  WARNING: tarn too big / leaking:", tarn.sum() * cell * cell / 1e4, "ha")  # noqa: E501
    sub = level[r0:r1, c0:c1]
    sub[ndimage.binary_dilation(tarn, iterations=3)] = TARN_LEVEL
    print(f"  lake area {main.sum()*cell*cell/1e6:.2f} km2, tarn area {tarn.sum()*cell*cell/1e4:.1f} ha")  # noqa: E501
    sp = CACHE / "stream_level_far.npy"
    if sp.exists():
        sl = np.load(sp)
        level = np.maximum(level, sl)
    rp = CACHE / "river.npz"
    if rp.exists():   # (r4 M9) the lower valley's river: floodplain level - 0.5 m inside its channel  # noqa: E501
        rv = np.load(rp)
        r0 = int((float(rv["z"][0]) + FAR_HALF) / cell)
        xs_ = (np.arange(res) + 0.5) * cell - FAR_HALF
        c0 = int((float(rv["x"].min()) - 100.0 + FAR_HALF) / cell); c1 = int((float(rv["x"].max()) + 100.0 + FAR_HALF) / cell)  # noqa: E702,E501
        d = np.abs(xs_[None, c0:c1] - rv["x"][:, None])
        inn = d < 0.5 * rv["w"][:, None] - 1.5
        blkl = level[r0:r0 + len(rv["z"]), c0:c1]
        blkl[:] = np.where(inn & (blkl < -1000.0), (rv["hb"] - 0.5)[:, None], blkl)
        print(f"  river water cells: {int(inn.sum())}")
    # near-resolution level map (1 m): lakes from the far map (nearest) + the stream channel  # noqa: E501
    nres = NEAR_RES
    idx_n = ((np.arange(nres) + 0.5) / nres * 2 * NEAR_HALF - NEAR_HALF + FAR_HALF) / cell  # noqa: E501
    idx_n = np.clip(idx_n.astype(int), 0, res - 1)
    lnear = level[np.ix_(idx_n, idx_n)].copy()
    lnear[lnear > TARN_LEVEL - 1] = np.where(lnear[lnear > TARN_LEVEL - 1] > TARN_LEVEL - 1, TARN_LEVEL, lnear[lnear > TARN_LEVEL - 1])  # noqa: E501
    spn = CACHE / "stream_level_near.npy"
    if spn.exists():
        lnear = np.where(lnear > 0.5 + 1e-3, lnear, lnear)  # keep lakes
        sn = np.load(spn)
        lnear = np.where(sn > -1000.0, np.maximum(np.where(lnear < -1000, -1e4, lnear), sn), lnear)  # noqa: E501
    np.save(CACHE / "water_level_near.npy", lnear.astype(np.float32))

    ht = gl.tex2d(hfar)
    lt = gl.tex2d(level, filt="nearest")
    has = (CACHE / "talus.npy").exists() and (CACHE / "flow.npy").exists()
    if has:
        tal = gl.tex2d(np.load(CACHE / "talus.npy").astype(np.float32))
        fl = np.load(CACHE / "flow.npy").astype(np.float32)
        fl = np.log1p(np.maximum(fl, 0)); fl = fl / max(np.percentile(fl, 99.9), 1e-6)  # noqa: E702
        flt = gl.tex2d(np.clip(fl, 0, 1).astype(np.float32))
    else:
        tal = flt = gl.tex2d(np.zeros((4, 4), np.float32))
    cp_ = CACHE / "cones.npy"
    cnt = gl.tex2d(np.load(cp_).astype(np.float32) if cp_.exists() else np.zeros((4, 4), np.float32))  # noqa: E501
    from engine_vr import features_vr as _feat
    meta = _feat.load_meta()
    st = np.zeros((96, 4), np.float32)
    if meta:
        a = np.asarray(meta["stream"], np.float32); st[: len(a)] = a  # noqa: E702
    tr = np.zeros((32, 4), np.float32)
    if meta and "trail" in meta:
        b = np.asarray(meta["trail"], np.float32)[:32]; tr[: len(b), : b.shape[1]] = b  # noqa: E702
    extra = {"uStream": st, "uStreamN": int(len(meta["stream"])) if meta else 0,
             "uTrail": tr, "uTrailN": int(min(32, len(meta.get("trail", [])))) if meta else 0}  # noqa: E501
    forest = _bake_pass("bake_forest.frag", res,
                        {"uHalf": FAR_HALF, "uRes": float(res), "uTarn": TARN_C, "uTarnLevel": TARN_LEVEL,  # noqa: E501
                         "uHasErosion": 1 if has else 0, **extra},
                        textures=[("uH", ht), ("uLvl", lt), ("uTalus", tal), ("uFlow", flt), ("uCones", cnt)], comps=4, tile=1024)  # noqa: E501
    forest = (np.clip(forest, 0, 1) * 255 + 0.5).astype(np.uint8)
    for t in (ht, lt, tal, flt, cnt):
        try:
            t.release()
        except Exception:
            pass
    np.save(wp, level); np.save(fp, forest)  # noqa: E702
    print(f"  maps in {time.time()-t0:.1f}s")
    return level, forest


# ---- copied from engine/bake.py:725-766 (verbatim)
def bake_noise3d(force=False):
    fp = CACHE / "noise3d.npy"
    if fp.exists() and not force:
        return np.load(fp)
    print("baking 3D noise ...", flush=True)
    N = 128
    rng = np.random.default_rng(5)  # noqa: F841

    def periodic_value(period, seed):
        g = np.random.default_rng(seed).random((period,) * 3).astype(np.float32)
        z = ndimage.zoom(g, N / period, order=3, mode="grid-wrap")
        return z

    def worley(period, seed):
        r = np.random.default_rng(seed)
        pts = r.random((period, period, period, 3)).astype(np.float32)
        coords = (np.arange(N) + 0.5) / N * period
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        ci = np.floor(X).astype(int), np.floor(Y).astype(int), np.floor(Z).astype(int)
        best = np.full((N, N, N), 9.0, np.float32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cx = (ci[0] + dx); cy = (ci[1] + dy); cz = (ci[2] + dz)  # noqa: E702
                    p = pts[cx % period, cy % period, cz % period]
                    d = (cx + p[..., 0] - X) ** 2 + (cy + p[..., 1] - Y) ** 2 + (cz + p[..., 2] - Z) ** 2  # noqa: E501
                    best = np.minimum(best, d)
        return np.sqrt(best)

    def norm(a):
        a = a - a.min(); return a / a.max()  # noqa: E702

    fb = sum(0.5 ** i * periodic_value(8 * 2 ** i, 10 + i) for i in range(4))
    ch0 = norm(fb)
    ch1 = norm(periodic_value(16, 20))
    w1 = 1 - norm(worley(8, 30)); w2 = 1 - norm(worley(16, 31)); w3 = 1 - norm(worley(32, 32))  # noqa: E702,E501
    ch2 = norm(w1 * 0.625 + w2 * 0.25 + w3 * 0.125)
    ch3 = norm(1 - norm(worley(16, 40)) * 0.6 + norm(periodic_value(32, 41)) * 0.4)
    a = np.stack([ch0, ch1, ch2, ch3], -1)
    a = (a * 255 + 0.5).astype(np.uint8)
    np.save(fp, a)
    return a


# ---- copied from engine/bake.py:769-774 (verbatim)
def bake_all(force=False):
    hfar, hnear = bake_heights(force=force)
    aof, aon = bake_ao(hfar, hnear, force=force)
    level, forest = bake_world_maps(hfar, force=force)
    n3 = bake_noise3d()
    return dict(hfar=hfar, hnear=hnear, aof=aof, aon=aon, level=level, forest=forest, noise3=n3)  # noqa: E501

# ==== END ENGINE COPIES ====

# ================================================================ Phase-4 T1
#
# trees.npy and shadow-map drivers for the pano renderer (vr.py). The engine
# produces both lazily around World.__init__:
#
#   trees.npy     engine/render.py:166-180 World._bake_trees - the writer
#                 cache_schema.py documents for this Phase-4 product
#   shadow maps   engine/render.py:182-205 World.ensure_shadows - runtime
#                 textures only, never saved to disk; a pano render has one
#                 sun, so the engine's _sun_key reuse cache is not needed.
#
# Both run through passes_vr.bake_pass (same tile + per-tile finish rhythm
# as the engine tile loops) and need an installed Host
# (passes_vr.open_host first).


def _trees_textures(textures):
    """12-sampler binding list for bake_trees from the world texture dict.

    The engine binds 10 textures at that point (its dict has no uTrees /
    uShNear / uShFar yet - trees and shadows bake after it); the P1
    manifest reflects the final shader and the SPIR-V keeps all 12 sampler
    slots, so the descriptor set demands them. vr.py passes a shared 4x4
    zero f4 dummy Tex under the key '_zero4' for the three the engine never
    supplied at this stage.
    """
    want = ["uHNear", "uHFar", "uAONear", "uAOFar", "uForest", "uWaterLvl",
            "uWaterLvlNear", "uStreamNear", "uNoise3", "uTrees",
            "uShNear", "uShFar"]
    dummy = textures.get("_zero4")
    out = []
    for n in want:
        t = textures.get(n, dummy)
        if t is None:
            raise pv.GlvrError(f"bake_trees: missing texture {n}")
        out.append((n, t))
    return out


def ensure_trees(textures, tile=1024, log=print):
    """trees.npy driver - copied semantics from engine/render.py:166-180.

    One bake_trees.frag full 4096^2 RGBA32F, tiles of 1024 each flushed
    (the engine's tile loop + c.finish()), read back and np.save'd to the
    cache_schema path (f4, 4096^2, 4 comps, NEAR_HALF - the writer at
    engine/render.py:178). Returns the float32 (4096, 4096, 4) array.
    """
    path = cs.path("trees.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    a = pv.bake_pass("bake_trees", cs.NEAR_RES, {},
                     textures=_trees_textures(textures), tile=tile,
                     comps=4, log=log).astype(np.float32)
    np.save(path, a)
    n = float((a[..., 1] != 0).sum()) / 1e6
    if log:
        log(f"  trees: {n:.2f} M")   # engine count print, render.py:180
    return a


def bake_shadows(textures, sun_dir, tile=512, log=print):
    """Shadow-map pair - copied semantics from engine/render.py:182-205.

    Two bake_shadow.frag passes (near then far), each 4096^2 RGBA32F,
    tiles of 512 (both the engine tile size), the same uniforms the engine
    p.set()s (uCenter (0,0), uHalf, uRes 4096, uFarHalf, uSunDir). Outputs
    upload LINEAR-filtered, clamped (the engine sets .filter=LINEAR with
    repeat off). Returns {"uShNear": Tex, "uShFar": Tex}."""
    out = {}
    for name, src, half in (("uShNear", "uHNear", NEAR_HALF),
                            ("uShFar", "uHFar", FAR_HALF)):
        a = pv.bake_pass(
            "bake_shadow", 4096,
            {"uCenter": (0.0, 0.0), "uHalf": float(half), "uRes": 4096.0,
             "uFarHalf": FAR_HALF,
             "uSunDir": tuple(float(v) for v in sun_dir)},
            textures=[("uH", textures[src]), ("uHFar", textures["uHFar"])],
            tile=tile, comps=4, log=log)
        out[name] = pv.upload(a, filt="linear")
    return out
