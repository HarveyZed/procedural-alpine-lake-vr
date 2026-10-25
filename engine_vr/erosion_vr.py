"""erosion_vr - engine/erosion.py ported to engine_vr (Phase 3, T1/T5).

The simulation itself lives in four compute programs (erosion_fill,
erosion_accum, its FLOW_ONLY variant, erosion_hill) and ~1100 lines of pure
numpy stage logic (basin repair, protection maps, checkpoints, metrics). Only
the GL plumbing is re-implemented here; the numpy stages and erode() itself are
copied verbatim below the generated marker, so the ladder, the stage
parameters and the process order are exactly the accepted reference build.

VR plumbing contract (port/reports/phase2.md addendum + review_p2.md 7):
  * one submit per band: uOffset changes per band and a host-mapped UBO write
    is only legal with the queue idle, so a band is
    set_uniforms -> write_descriptors -> dispatch -> wait.
    That is strictly more flushing than the engine's cadence finish
    (every 32 / 8 / 2 dispatches at 2048 / 4096 / 8192), never less.
  * write_descriptors freezes each image's layout, so sampler-bound images are
    transitioned to SHADER_READ_ONLY and storage-bound images to GENERAL
    immediately before the write - that is this module's substitute for the
    descriptor-layout assert the host deliberately does not do.
  * memory_barrier() == COMPUTE -> COMPUTE, SHADER_WRITE -> SHADER_READ over
    the images the dispatch touched, which rest in GENERAL.
  * the <=2048-row dispatch rule and the 16x16 tile geometry are unchanged:
    BAND_TILES=128 tile rows and g = res // 16 + 1 groups, and _prog() asserts
    the compiled local_size is really 16.

Import-safe: no device, no vulkan call at import; erode() needs a Host
installed by passes_vr.open_host() (or the run_erosion helper below).
"""

import heapq
import math  # noqa: F401  (used by the copied axis_x)
import os  # noqa: F401  (used by the copied stages)
import time
from pathlib import Path  # noqa: F401  (used by the copied checkpoint code)

import numpy as np
from scipy import ndimage  # noqa: F401  (used by the copied stages)
from scipy.sparse import csr_matrix  # noqa: F401  (copied basin repair)
from scipy.sparse.csgraph import dijkstra  # noqa: F401  (copied basin repair)

from engine_vr import cache_schema as cs
from engine_vr import passes_vr as pv
from engine_vr.gl_vr import GlvrError

# Copied constant from engine/erosion.py:201 (tile rows per dispatch: 128
# tiles x 16 cells = 2048 rows, "far below the 2 s TDR limit").
BAND_TILES = 128

_PROGS = {}
_BINDINGS = {}


def _program_name(name, defines=()):
    """Map (shader file, defines) to the manifest program that bakes it.

    engine/erosion.py compiled the defines itself (_prog(name, defines));
    Phase 1 gave each variant its own manifest program instead, so the only
    variant this port has is erosion_accum.comp + FLOW_ONLY.
    """
    stem = pv.program_for(name)
    if not defines:
        if stem not in ("erosion_fill", "erosion_accum", "erosion_hill"):
            raise GlvrError(f"no manifest program for erosion shader "
                            f"{name!r}")
        return stem
    if stem == "erosion_accum" and tuple(defines) == ("FLOW_ONLY",):
        return "erosion_accum_flow_only"
    raise GlvrError(f"no manifest program for {stem} with defines "
                    f"{tuple(defines)}")


def _prog(name, defines=()):
    """Host-built program, cached like engine/erosion.py:154-181 _PROGS."""
    key = (name, tuple(defines))
    prog = _PROGS.get(key)
    if prog is None:
        pname = _program_name(name, defines)
        prog = pv.host().build(pname)
        lx, ly, _ = pv.host().local_size(pname)
        if (lx, ly) != (16, 16):
            raise GlvrError(
                f"{pname}: local_size {(lx, ly)} is not the (16, 16) tile the "
                f"copied band math (res // 16 + 1, BAND_TILES) assumes")
        _PROGS[key] = prog
    return prog


def _binding_names(prog):
    """(sampler binding names, storage image names) in manifest binding order.

    The engine bound textures positionally (t.use(location=i) /
    bind_to_image(i, ...)); the manifest carries the same order in its
    "binding" index, so positional lists from the copied code map onto the
    declared names and nothing else can be bound.
    """
    names = _BINDINGS.get(prog.name)
    if names is None:
        sam = [b for b in prog.bindings if b["kind"] == "combined_sampler"]
        img = [b for b in prog.bindings if b["kind"] == "storage_image"]
        names = ([b["name"] for b in sorted(sam, key=lambda b: int(b["binding"]))],
                 [b["name"] for b in sorted(img, key=lambda b: int(b["binding"]))])
        _BINDINGS[prog.name] = names
    return names


def _tex(res, comps, data=None, dtype="f4"):
    """Compute image (GENERAL, storage + sampled + transfer), NEAREST reads.

    Same call shape as engine/erosion.py:184-187 - the engine set
    filter = (NEAREST, NEAREST), which is the "nearest" sampler this port
    requests at bind time for every erosion image.
    """
    return pv.storage_image(res, comps=comps, dtype=dtype, data=data)


def _upload(arr, dtype="f4"):
    """numpy -> compute image (engine/erosion.py:190-193)."""
    a = np.ascontiguousarray(arr, dtype=np.float16 if dtype == "f2"
                             else np.float32)
    comps = 1 if a.ndim == 2 else a.shape[2]
    return _tex(a.shape[0], comps, a, dtype)


def _read(t, comps=None):
    """Compute image -> writable numpy (engine/erosion.py:196-198).

    gl_vr's fetch gives (h, w) for a single-channel image and (h, w, c)
    otherwise, as a view onto the cached staging buffer; the engine always
    reshaped to (h, w, components) and returned a copy, which is what this
    reproduces (the copy matters: erosion writes into its readbacks).
    """
    a = pv.read_image(t)
    if a.ndim == 2:
        a = a[..., None]
    n = comps or int(t.channel)
    out = a[..., :n]
    return out[..., 0].copy() if comps == 1 else out.copy()


class _Ctx:
    """The two context calls the copied code still makes: finish()."""

    @staticmethod
    def finish():
        pv.host().wait()

    @staticmethod
    def memory_barrier():
        raise GlvrError(
            "memory_barrier() needs the images a dispatch touched; the VR Sim "
            "issues it inside _run() over exactly those images")


class Sim:
    """GPU state of one resolution level (engine/erosion.py:204-331 on gl_vr).

    Names, signatures and ping-pong bookkeeping (cur/parity/ndisp) are the
    engine's, so the copied erode() drives this class unchanged.
    """

    def __init__(self, hb, hs, mask, track, cell, fluvial=True):
        self.res = hb.shape[0]
        self.cell = float(cell)
        self.c = _Ctx()
        self.H = [_upload(np.stack([hb, hs], -1))]
        if track is not None:
            self.H.append(_tex(self.res, 2))
        self.cur = 0
        self.M = _upload(mask)
        self.T = _upload(track) if track is not None else None
        self.W = self.acc = self.net = None
        if fluvial:
            self.alloc_flow(acc_comps=2)
        self.ndisp = 0
        self.parity = 0

    def alloc_flow(self, acc_comps=2, W=None, A=None):
        self.W = _upload(W) if W is not None else _tex(self.res, 1)
        if A is not None:
            self.acc = _upload(A)
        else:
            self.acc = _upload(np.zeros((self.res, self.res, acc_comps),
                                        np.float32) if acc_comps > 1
                               else np.zeros((self.res, self.res), np.float32))
        if acc_comps > 1:
            self.net = _upload(np.zeros((self.res, self.res), np.float32))

    def _run(self, name, samplers, images, defines=(), **u):
        """One dispatch series over the band grid (engine/erosion.py:234-254).

        Positional sampler/storage lists become manifest binding names here;
        a count mismatch against the manifest is an error, not a silent skip.
        """
        prog = _prog(name, defines)
        sam_names, img_names = _binding_names(prog)
        if len(samplers) != len(sam_names):
            raise GlvrError(f"{prog.name}: {len(samplers)} sampler(s) bound "
                            f"but the manifest declares {len(sam_names)} "
                            f"({sam_names})")
        if len(images) != len(img_names):
            raise GlvrError(f"{prog.name}: {len(images)} storage image(s) "
                            f"bound but the manifest declares "
                            f"{len(img_names)} ({img_names})")
        smp = {}
        for bn, t in zip(sam_names, samplers):
            if t is None:
                raise GlvrError(f"{prog.name}: sampler binding {bn} has no "
                                f"image (engine bound None here too)")
            smp[bn] = (t, "nearest", "clamp")
        sto = {}
        for bn, (t, rd, wr) in zip(img_names, images):
            if t is None:
                raise GlvrError(f"{prog.name}: storage binding {bn} has no "
                                f"image")
            sto[bn] = t
        self.parity ^= 1
        s = 8 * self.parity
        u.update(uRes=(self.res, self.res), uCell=self.cell, uShift=(s, s))
        # write_descriptors freezes layouts, so transition to the roles this
        # dispatch actually uses before writing them (no overlap: a program
        # never samples and stores the same image in one dispatch).
        pv.to_sampler_layout(list(samplers))
        pv.to_general_layout(list(sto.values()))
        g = self.res // 16 + 1
        pv.compute_bands(prog.name, u, g, BAND_TILES, samplers=smp,
                         storage_images=sto)
        # The engine issued one memory_barrier() after its band loop. Only
        # the images this dispatch WROTE need the write -> read barrier: the
        # sampler-bound images rest in SHADER_READ_ONLY, so naming them with
        # oldLayout=GENERAL would be a layout mismatch, not a stronger
        # barrier. Band-to-band visibility is already given - every band is
        # its own submission on the same queue with a wait in between.
        pv.barrier_compute_to_read(list(sto.values()))
        self.ndisp += 1
        if self.ndisp % (32 if self.res <= 2048
                         else (8 if self.res <= 4096 else 2)) == 0:
            self.c.finish()      # every band already waited; kept for parity

    # ---- routing ----------------------------------------------------------
    def fill(self, passes, eps, inner=8, init=False):
        smp = [self.H[self.cur], self.M]
        img = [(self.W, True, True)]
        if init:
            self._run("erosion_fill.comp", smp, img, uEps=eps, uInit=1,
                      uInner=inner)
        for _ in range(passes):
            self._run("erosion_fill.comp", smp, img, uEps=eps, uInit=0,
                      uInner=inner)

    def fill_converge(self, eps, chunk, max_passes, inner=16):
        """Re-initialise the fill surface from +inf and relax until every
        cell drains (engine/erosion.py:265-277)."""
        self.fill(0, eps, inner, init=True)
        done = 1
        unreached = -1
        while done < max_passes:
            self.fill(chunk, eps, inner)
            done += chunk
            self.c.finish()
            unreached = int((_read(self.W, 1) >= 1e8).sum())
            if unreached == 0:
                break
        return done, unreached

    # ---- processes --------------------------------------------------------
    def accumulate(self, P, passes):
        for _ in range(passes):
            self._run("erosion_accum.comp", [self.H[self.cur], self.W, self.M],
                      [(self.acc, True, True), (self.net, False, True)],
                      uP=P["p"], uK=P["K"], uMexp=P["m"], uNexp=P["n"],
                      uG=P["G"], uKloose=P["Kloose"], uMaxDep=P["maxDep"],
                      uMaxEro=P["maxEro"], uInner=P["acc_inner"],
                      uHardK=P["HardK"], uLakeTrap=P.get("LakeTrap", 1.0),
                      uPitTrap=P.get("PitTrap", 0.0))

    def hill(self, P, apply_fluvial):
        u = dict(uTb=P["Tb"], uTbLow=P["TbLow"], uCb=P["Cb"], uTl=P["Tl"],
                 uCl=P["Cl"], uT1=P["T1"], uR1=P["R1"], uCover=P["Cover"],
                 uDebris=P["Debris"], uApply=int(apply_fluvial),
                 uHardExp=P["HardExp"], uHardExpLow=P["HardExpLow"],
                 uTreeA=P["TreeA"], uTreeB=P["TreeB"], uRockA=P["RockA"],
                 uRockB=P["RockB"], uSpikeK=P["SpikeK"], uWexp=P["Wexp"],
                 uLooseMax=P["LooseMax"], uJitter=P.get("Jitter", 0.0),
                 uSeed=int(P.get("seed", 1)), uCreep=P.get("Creep", 0.0),
                 uCreepRock=P.get("CreepRock", 0.15))
        net = self.net if (apply_fluvial and self.net is not None) else self.M
        self._run("erosion_hill.comp", [self.H[self.cur], self.M, net],
                  [(self.H[1 - self.cur], False, True), (self.T, True, True)],
                  **u)
        self.cur ^= 1

    def step(self, P):
        """One fluvial + hillslope step."""
        self.fill(P["fill_passes"], P["eps"], P["fill_inner"])
        self.accumulate(P, P["acc_passes"])
        self.hill(P, True)
        for _ in range(P.get("thermal_per_step", 1) - 1):
            self.hill(P, False)

    def flow_only(self, p, passes, inner):
        for _ in range(passes):
            self._run("erosion_accum.comp", [self.H[self.cur], self.W, self.M],
                      [(self.acc, True, True)], defines=("FLOW_ONLY",),
                      uP=p, uInner=inner)

    # ---- io ---------------------------------------------------------------
    def read_h(self):
        hh = _read(self.H[self.cur])
        return hh[..., 0], hh[..., 1]

    def write_h(self, hb, hs):
        """Replace the current height image (the engine rewrites the texture
        in place; a mapped upload is the VR equivalent, and descriptors are
        rebuilt per dispatch anyway)."""
        a = np.ascontiguousarray(np.stack([hb, hs], -1), dtype=np.float32)
        old = self.H[self.cur]
        self.H[self.cur] = _upload(a)
        old.destroy()

    def read_state(self):
        hh = _read(self.H[self.cur])
        return hh[..., 0], hh[..., 1], _read(self.T)

    def release(self, *names):
        for n in names or ("H", "M", "T", "W", "acc", "net"):
            v = getattr(self, n)
            if v is None:
                continue
            for t in (v if isinstance(v, list) else [v]):
                t.destroy()
            setattr(self, n, None)


def build_plan():
    """Pure-data description of the erosion dispatch series (T4/T5).

    The uniform dicts below repeat the per-pass kwargs of Sim.fill /
    Sim.accumulate / Sim.hill / Sim.flow_only; Sim._run writes exactly those
    keys through gl_vr's packer, which raises on any key that is not a
    manifest member, and port/gates/gate_p3.py checks this plan against the
    same manifest. Values are the reference parameters of the 512^2 stage
    (engine/erosion.py STAGES[0] merged over _COMMON).

    Every entry carries the dispatch discipline the original keeps: 16x16
    tiles, g = res // 16 + 1 groups walked in bands of BAND_TILES tile rows,
    so no dispatch covers more than 2048 rows. The resolution ladder is
    512, 1024, 2048, 4096, 8192 (cache_schema.EROSION_RES_LADDER).
    """
    P = dict(_COMMON, **STAGES[0])
    res = int(STAGES[0]["res"])
    # erode() derives the per-level cell size from the ladder (engine/erosion.py
    # :1267-1268, "f = res // r; cell = cell_m * f"), so the 512^2 stage runs at
    # 64 m for the 4 m far map bake_heights hands it (engine/bake.py:61).
    cell = (2 * cs.FAR_HALF / cs.FAR_RES) * (cs.FAR_RES // res)
    bands = {"kind": "bands", "local_size": 16, "groups": res // 16 + 1,
             "band_tiles": BAND_TILES, "max_rows": BAND_TILES * 16,
             "ladder": list(cs.EROSION_RES_LADDER)}
    routing = {"uRes": (res, res), "uOffset": (0, 0), "uShift": (0, 0),
               "uCell": float(cell)}
    return {
        "fill_init": {
            "program": "erosion_fill",
            "uniforms": dict(routing, uEps=float(P["eps"]), uInit=1,
                             uInner=int(P["fill_inner"])),
            "storage_images": ["imgW"], "res": res, "dispatch": bands,
        },
        "fill": {
            "program": "erosion_fill",
            "uniforms": dict(routing, uEps=float(P["eps"]), uInit=0,
                             uInner=int(P["fill_inner"])),
            "storage_images": ["imgW"], "res": res, "dispatch": bands,
        },
        "accum": {
            "program": "erosion_accum",
            "uniforms": dict(routing, uP=P["p"], uK=P["K"], uMexp=P["m"],
                             uNexp=P["n"], uG=P["G"], uKloose=P["Kloose"],
                             uMaxDep=P["maxDep"], uMaxEro=P["maxEro"],
                             uInner=P["acc_inner"], uHardK=P["HardK"],
                             uLakeTrap=P.get("LakeTrap", 1.0),
                             uPitTrap=P.get("PitTrap", 0.0)),
            "storage_images": ["imgAcc", "imgNet"], "res": res,
            "dispatch": bands,
        },
        "hill_fluvial": {
            "program": "erosion_hill",
            "uniforms": dict(routing, uTb=P["Tb"], uTbLow=P["TbLow"],
                             uCb=P["Cb"], uTl=P["Tl"], uCl=P["Cl"],
                             uT1=P["T1"], uR1=P["R1"], uCover=P["Cover"],
                             uDebris=P["Debris"], uApply=1,
                             uHardExp=P["HardExp"],
                             uHardExpLow=P["HardExpLow"], uTreeA=P["TreeA"],
                             uTreeB=P["TreeB"], uRockA=P["RockA"],
                             uRockB=P["RockB"], uSpikeK=P["SpikeK"],
                             uWexp=P["Wexp"], uLooseMax=P["LooseMax"],
                             uJitter=P.get("Jitter", 0.0),
                             uSeed=int(P.get("seed", 1)),
                             uCreep=P.get("Creep", 0.0),
                             uCreepRock=P.get("CreepRock", 0.15)),
            "storage_images": ["imgH", "imgTrack"], "res": res,
            "dispatch": bands,
        },
        "hill_thermal": {
            "program": "erosion_hill",
            "uniforms": dict(routing, uTb=P["Tb"], uTbLow=P["TbLow"],
                             uCb=P["Cb"], uTl=P["Tl"], uCl=P["Cl"],
                             uT1=P["T1"], uR1=P["R1"], uCover=P["Cover"],
                             uDebris=P["Debris"], uApply=0,
                             uHardExp=P["HardExp"],
                             uHardExpLow=P["HardExpLow"], uTreeA=P["TreeA"],
                             uTreeB=P["TreeB"], uRockA=P["RockA"],
                             uRockB=P["RockB"], uSpikeK=P["SpikeK"],
                             uWexp=P["Wexp"], uLooseMax=P["LooseMax"],
                             uJitter=P.get("Jitter", 0.0),
                             uSeed=int(P.get("seed", 1)),
                             uCreep=P.get("Creep", 0.0),
                             uCreepRock=P.get("CreepRock", 0.15)),
            "storage_images": ["imgH", "imgTrack"], "res": res,
            "dispatch": bands,
        },
        "flow_only": {
            "program": "erosion_accum_flow_only",
            "uniforms": dict(routing, uP=P["p"], uInner=16),
            "storage_images": ["imgAcc"], "res": res, "dispatch": bands,
        },
    }


def run_erosion(hfar=None, cell_m=None, host=None, save=True, log=print,
                **kw):
    """Erode the far map on the VR host and (by default) write the products.

    hfar defaults to the cached raw far map (h_far_raw.npy, else h_far.npy);
    cell_m defaults to the cache ladder's far cell size (2 * FAR_HALF /
    FAR_RES = 4 m), matching engine/bake.py:61 (erode(hfar, 2 * FAR_HALF /
    far_res)). Returns erode()'s dict: height, flow, talus, sediment.
    """
    if hfar is None:
        raw = cs.path("h_far_raw")
        hfar = np.load(raw) if raw.exists() else np.load(cs.path("h_far"))
    if cell_m is None:
        cell_m = 2 * cs.FAR_HALF / float(hfar.shape[0])
    own = host is None
    if own:
        host = pv.open_host(cache_dir=str(cs.cache_dir()), log=log)
    try:
        out = erode(hfar, cell_m, log=log, **kw)
    finally:
        if own:
            pv.close_host()
    if save:
        cs.ensure_dir()
        np.save(cs.path("erosion_height"),
                np.asarray(out["height"], np.float32))
        for k in ("flow", "talus", "sediment"):
            if out.get(k) is not None:
                np.save(cs.path(k), np.asarray(out[k], np.float32))
                np.save(cs.CACHE / f"erosion_{k}.npy",
                        np.asarray(out[k], np.float32))
    return out



# ==== BEGIN ENGINE COPIES (generated by engine_vr/build_p3.py; do not edit) ====

# --------------------------------------------------------------------------------
# Copied from the moderngl build by engine_vr/build_p3.py: verbatim
# except the substitutions named in each block header; no constant,
# threshold or ordering was retyped. Per-line noqa comments may be
# added by build_p3.py for ruff (ast-dump verified, comments only).
# --------------------------------------------------------------------------------

# ---- copied from engine/erosion.py:94-148 (verbatim)
def axis_x(z):
    """Lake axis (same curve as features.axis_x / the analytic terrain)."""
    return 150.0 * np.sin(0.00105 * np.asarray(z, np.float64) + 0.5) + 45.0 * np.sin(0.0029 * np.asarray(z, np.float64) + 2.1)  # noqa: E501


# seeds and water levels of the protected water bodies (overridable through erode())
LAKE_C = (float(axis_x(0.0)), 0.0)      # (110.8, 0)
LAKE_LEVEL = 0.0
TARN_C = (-1640.0, -430.0)              # bake.TARN_C = (-1680, -420) lies in the same tarn (checked)  # noqa: E501
TARN_LEVEL = 520.0
WF_Z = -550.0                           # waterfall cliff site (features.WF_Z), east shore  # noqa: E501
BIG = 60000.0
MAX_LOWER_KNEE, MAX_LOWER = 260.0, 330.0     # soft limit of the bedrock lowering (m)


def _deg(a):
    return float(np.tan(np.radians(a)))


# ------------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------------
_COMMON = dict(HardExp=0.6, HardExpLow=0.2, Jitter=0.15, Cb=0.5, Tl=_deg(34), Cl=0.5,
               TreeA=800.0, TreeB=1350.0, RockA=650.0, RockB=900.0,
               SpikeK=1.3, Wexp=1.1, LooseMax=40.0, PitTrap=0.0)
STAGES = [
    # 64 m: long-range drainage organisation (basins breached, dendritic valley network)
    dict(_COMMON, res=512, pre=100, pre_Tb=_deg(62), pre_TbLow=_deg(54), steps=700, thermal_per_step=1,  # noqa: E501
         K=1.8e-3, m=0.4, n=1.0, G=1.0, Kloose=4.0, HardK=2.0, p=4.0, eps=1e-4, maxDep=4.0, maxEro=4.0, LakeTrap=0.25,  # noqa: E501
         fill_passes=1, fill_inner=8, acc_passes=1, acc_inner=8, reinit=300, uplift_every=100, uplift_sigma=1500.0,  # noqa: E501
         Tb=_deg(58), TbLow=_deg(50), seed=5, Debris=0.0, T1=_deg(38), R1=0.0, Cover=1.0,  # noqa: E501
         breach=True, breach_max_cut=350.0, breach_side=_deg(55), cleanup_max_cut=60.0, corridor=True, detail_wet=0.5),  # noqa: E501
    # 32 m: valleys, ridges, cirque heads
    dict(_COMMON, res=1024, pre=80, pre_Tb=_deg(60), pre_TbLow=_deg(52), steps=600, thermal_per_step=1,  # noqa: E501
         K=1.4e-3, m=0.4, n=1.0, G=1.0, Kloose=3.0, HardK=2.0, p=4.0, eps=1e-4, maxDep=3.0, maxEro=3.0, LakeTrap=0.25,  # noqa: E501
         fill_passes=1, fill_inner=8, acc_passes=1, acc_inner=8, reinit=250, uplift_every=100, uplift_sigma=1500.0,  # noqa: E501
         Creep=0.0, Tb=_deg(56), TbLow=_deg(47), seed=1, Debris=0.0, T1=_deg(40), R1=0.004, Cover=1.0,  # noqa: E501
         corridor=True, detail_wet=0.6, pit_fill=20.0, cleanup_max_cut=50.0),
    # 16 m: side valleys, couloirs, first talus
    dict(_COMMON, res=2048, pre=30, pre_Tb=_deg(62), pre_TbLow=_deg(54), steps=800, thermal_per_step=1,  # noqa: E501
         K=2.0e-3, m=0.4, n=1.0, G=1.0, Kloose=3.0, HardK=2.0, p=4.0, eps=3e-4, maxDep=2.0, maxEro=2.0, LakeTrap=0.4,  # noqa: E501
         fill_passes=1, fill_inner=8, acc_passes=1, acc_inner=8, reinit=250, uplift_every=100, uplift_sigma=1500.0,  # noqa: E501
         Creep=0.002, Tb=_deg(52), TbLow=_deg(42), seed=2, Debris=0.25, T1=_deg(40), R1=0.02, Cover=1.5,  # noqa: E501
         corridor=True, detail_wet=0.6, pit_fill=20.0, cleanup_max_cut=50.0),
    # 8 m: gullies, debris cones, scree aprons
    dict(_COMMON, res=4096, pre=30, pre_Tb=_deg(66), pre_TbLow=_deg(56), steps=400, thermal_per_step=1,  # noqa: E501
         K=1.5e-3, m=0.4, n=1.0, G=1.0, Kloose=3.0, HardK=2.0, p=4.0, eps=5e-4, maxDep=1.0, maxEro=1.0, LakeTrap=0.5,  # noqa: E501
         fill_passes=1, fill_inner=8, acc_passes=1, acc_inner=8, reinit=250, uplift_every=100, uplift_sigma=1500.0,  # noqa: E501
         Creep=0.006, Tb=_deg(54), TbLow=_deg(42), seed=3, Debris=0.25, T1=_deg(45), R1=0.04, Cover=1.5, detail_wet=0.6, pit_fill=20.0, cleanup_max_cut=50.0),  # noqa: E501
    # 4 m: hillslope only (faces, scree at the angle of repose) inside the +-window_m window  # noqa: E501
    # (outside: 8 m result + the input's 4 m detail) + full-resolution flow map
    dict(_COMMON, res=8192, pre=20, pre_Tb=_deg(68), pre_TbLow=_deg(58), steps=0, post=30,  # noqa: E501
         Creep=0.03, Tb=_deg(60), TbLow=_deg(44), seed=4, Debris=0.2, T1=_deg(50), R1=0.03, Cover=1.0,  # noqa: E501
         flow_fill=48, flow_acc=64, p=4.0, eps=5e-4, detail_wet=0.5),
]


# ---- copied from engine/erosion.py:337-1234 (verbatim)
def _pool(a, f, op="mean"):
    if f == 1:
        return np.array(a, dtype=np.float32, copy=True)
    n0, n1 = a.shape[0] // f, a.shape[1] // f
    b = a.reshape(n0, f, n1, f, *a.shape[2:])
    return getattr(b, op)((1, 3)).astype(np.float32)


def _up(a, g, order):
    """Cell-centred upsampling by an integer factor (order 1 = bilinear, 3 = bicubic)."""  # noqa: E501
    a = np.asarray(a, np.float32)
    if g == 1:
        return a.copy()
    if a.ndim == 3:
        return np.stack([_up(a[..., i], g, order) for i in range(a.shape[2])], -1)
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(a), mode="F")
    im = im.resize((a.shape[1] * g, a.shape[0] * g), Image.BICUBIC if order == 3 else Image.BILINEAR)  # noqa: E501
    return np.asarray(im, dtype=np.float32).copy()


def _up_exact(d, g):
    """Bilinear upsampling by g whose g x g block means reproduce d exactly (bilinear +
    the
    nearest-neighbour upsampled residual): changes computed on a pooled grid can be
    applied
    to the fine grid without leaving residual closed depressions at the pooled scale."""
    u = _up(d, g, 1)
    return u + np.repeat(np.repeat(d - _pool(u, g), g, 0), g, 1)


def _resample(a, n):
    """Pool or upsample a square map to n x n."""
    m = a.shape[0]
    if m == n:
        return np.asarray(a, np.float32)
    return _pool(a, m // n) if m > n else _up(a, n // m, 1)


def _smooth01(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _noise(n, feat_cells, rng, order=3):
    """Zero mean, unit std smooth noise on an n x n grid with features of feat_cells."""
    k = max(4, int(round(n / feat_cells)))
    g = rng.standard_normal((k + 1, k + 1)).astype(np.float32)
    z = ndimage.zoom(g, n / k, order=order, mode="nearest", grid_mode=True)[:n, :n]
    return ((z - z.mean()) / (z.std() + 1e-6)).astype(np.float32)


# ------------------------------------------------------------------------------------
# protection of lake and tarn
# ------------------------------------------------------------------------------------
def _component(mask, rc):
    lab, _ = ndimage.label(mask)
    if lab[rc] == 0:
        return np.zeros_like(mask)
    return lab == lab[rc]


def _touches_border(m):
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def _find_spill(h, rc, level, cell):
    """Lowest level at which the basin around rc overflows; returns (level, saddle cell,
    basin component just below the spill level)."""
    base = _component(h < level, rc).sum()
    lim = max(5 * base, 1e6 / cell ** 2)

    def comp(lv):
        m = _component(h < lv, rc)
        return m, (_touches_border(m) or m.sum() > lim)

    lo, hi = level, level + 600.0
    m_lo, _ = comp(lo)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        m, leak = comp(mid)
        if leak:
            hi = mid
        else:
            lo, m_lo = mid, m
    m_hi, _ = comp(hi)
    # the saddle joins the basin to the big outside region (stray shore cells in the tiny  # noqa: E501
    # [lo, hi) band form their own small components and are ignored)
    lab, n = ndimage.label(m_hi & ~m_lo)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    cand = (lab == big) & ndimage.binary_dilation(m_lo)
    rr, cc = np.nonzero(cand)
    k = np.argmin(h[rr, cc])
    return lo, (int(rr[k]), int(cc[k])), m_lo


def _breach_path(h, src, water, basin, level, spill_level, rad=250):
    """Cheapest 8-connected path from the spill saddle into the water body and down the
    outside to terrain below the water level (Dijkstra, cost grows above the spill
    level)."""
    r0, c0 = src
    ra, rb = max(0, r0 - rad), min(h.shape[0], r0 + rad + 1)
    ca, cb = max(0, c0 - rad), min(h.shape[1], c0 + rad + 1)
    hw = h[ra:rb, ca:cb]
    n0, n1 = hw.shape
    cost = 1.0 + 1000.0 * np.maximum(hw - spill_level, 0.0)
    idx = np.arange(n0 * n1).reshape(n0, n1)
    rows, cols, wts = [], [], []
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # noqa: E501
        s_sl = (slice(max(0, -dr), n0 - max(0, dr)), slice(max(0, -dc), n1 - max(0, dc)))  # noqa: E501
        d_sl = (slice(max(0, dr), n0 - max(0, -dr)), slice(max(0, dc), n1 - max(0, -dc)))  # noqa: E501
        rows.append(idx[s_sl].ravel())
        cols.append(idx[d_sl].ravel())
        wts.append((cost[d_sl] * np.hypot(dr, dc)).ravel())
    G = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n0 * n1, n0 * n1))  # noqa: E501
    s = (r0 - ra) * n1 + (c0 - ca)
    dist, pred = dijkstra(G, directed=True, indices=s, return_predecessors=True)
    dist = dist.reshape(n0, n1)
    targets = [water[ra:rb, ca:cb], (hw < level - 1.0) & ~basin[ra:rb, ca:cb]]
    path = [s]
    for tgt in targets:
        d = np.where(tgt, dist, np.inf)
        k = int(np.argmin(d))
        if not np.isfinite(d.flat[k]):
            raise RuntimeError("breach path: no target reachable")
        while k != s and k >= 0:
            path.append(k)
            k = int(pred[k])
    rr, cc = np.unravel_index(np.array(path), (n0, n1))
    return rr + ra, cc + ca


def make_bodies(lake_seed=LAKE_C, tarn_seed=TARN_C, lake_level=LAKE_LEVEL, tarn_level=TARN_LEVEL):  # noqa: E501
    return [dict(name="lake", seed=tuple(map(float, lake_seed)), level=float(lake_level), axis=True,  # noqa: E501
                 k_corr=(150.0, 400.0), p_corr=(100.0, 300.0)),
            dict(name="tarn", seed=tuple(map(float, tarn_seed)), level=float(tarn_level), axis=False,  # noqa: E501
                 k_corr=(80.0, 220.0), p_corr=(60.0, 200.0))]


class _Bilin:
    def __init__(self, a, cell):
        self.a, self.cell, self.half = a, cell, a.shape[0] * cell / 2

    def __call__(self, x, z):
        fx = (x + self.half) / self.cell - 0.5
        fz = (z + self.half) / self.cell - 0.5
        return float(ndimage.map_coordinates(self.a, [[fz], [fx]], order=1, mode="nearest")[0])  # noqa: E501


def stream_polyline(h0, cell, level=LAKE_LEVEL, length=1500.0):
    """Inflow stream line: lake axis north of the flood-filled north tip (as
    features.place)."""
    H = _Bilin(h0, cell)
    zt = 1200.0
    while H(float(axis_x(zt)), zt) < level + 0.3 and zt < 3500.0:
        zt += 5.0
    zs = np.arange(zt - 40.0, zt + length, 22.0)
    xs = axis_x(zs) + 28.0 * np.sin((zs - zt) * 0.006) + 12.0 * np.sin((zs - zt) * 0.017 + 1.0)  # noqa: E501
    return np.stack([xs, zs], -1), zt


def _polyline_field(pts, n, cell, half):
    """Distance (m) to a polyline and arc length (m) of the nearest point, on an n x n
    grid."""
    seg = np.diff(pts, axis=0)
    segl = np.hypot(seg[:, 0], seg[:, 1])
    s_acc = np.concatenate([[0.0], np.cumsum(segl)])
    dense, sv = [], []
    for i in range(len(seg)):
        k = max(2, int(segl[i] / (cell * 0.25)))
        t = np.linspace(0.0, 1.0, k, endpoint=False)
        dense.append(pts[i] + seg[i] * t[:, None])
        sv.append(s_acc[i] + segl[i] * t)
    dense = np.concatenate(dense)
    sv = np.concatenate(sv)
    cols = np.clip(((dense[:, 0] + half) / cell).astype(int), 0, n - 1)
    rows = np.clip(((dense[:, 1] + half) / cell).astype(int), 0, n - 1)
    mark = np.full((n, n), -1.0, np.float32)
    mark[rows, cols] = sv
    dist, (ir, ic) = ndimage.distance_transform_edt(mark < 0, return_indices=True)
    return (dist * cell).astype(np.float32), mark[ir, ic], float(s_acc[-1])


def build_protection(h0, cell, bodies=None, log=print, waterfall_z=WF_Z, corridor=True):
    bodies = bodies or make_bodies()
    res = h0.shape[0]
    half = res * cell / 2

    def rc(x, z):
        return int((z + half) / cell), int((x + half) / cell)

    win = min(res // 2, int(4096 / cell))
    c0 = res // 2
    off = c0 - win
    sl = (slice(off, c0 + win), slice(off, c0 + win))
    hw = h0[sl]
    info = {}
    for b in bodies:
        name, (x, z), level = b["name"], b["seed"], b["level"]
        r, c = rc(x, z)
        if not h0[r, c] < level:
            raise RuntimeError(f"{name}: seed ({x:.0f}, {z:.0f}) is dry in the input (h={h0[r, c]:.1f} m >= {level})")  # noqa: E501
        water = _component(hw < level, (r - off, c - off))
        if _touches_border(water) or water.sum() == 0:
            raise RuntimeError(f"{name}: water body not closed in the input")
        lv, sp, basin = _find_spill(hw, (r - off, c - off), level, cell)
        path = None
        for rad in (250, 500, 1000):
            try:
                path = _breach_path(hw, sp, water, basin, level, lv, rad)
                break
            except RuntimeError:
                log(f"  {name}: no breach path within {rad} cells, retrying wider")
        if path is None:
            raise RuntimeError(f"{name}: no breach path found")
        pr, pc = path
        d = dict(water=water, basin=basin, level=level, spill_level=lv, spill=sp, path=(pr, pc),  # noqa: E501
                 area=float(water.sum() * cell * cell), body=b)
        if b.get("axis"):
            zs = np.arange(-1400.0, 1641.0, 50.0)
            wet, inside, out_z = 0, 0, []
            for zz in zs:
                rr_, cc_ = rc(float(axis_x(zz)), zz)
                if h0[rr_, cc_] < level:
                    wet += 1
                    if water[rr_ - off, cc_ - off]:
                        inside += 1
                    else:
                        out_z.append(float(zz))
            d["axis_check"] = dict(wet_samples=wet, in_lake=inside, outside_z=out_z)
            log(f"  {name}: {inside}/{wet} wet axis samples inside the flood-filled lake"  # noqa: E501
                + (f" (separate pools at z={out_z})" if out_z else ""))
        info[name] = d
        log(f"  {name}: area {water.sum()*cell*cell/1e4:.1f} ha, spill {lv:.2f} m (freeboard {lv-level:.1f} m) at "  # noqa: E501
            f"x={(sp[1]+off)*cell-half+cell/2:.0f} z={(sp[0]+off)*cell-half+cell/2:.0f}, breach path {len(pr)} cells")  # noqa: E501

    # erodibility: off in a thin shore band and along the spill sill / breach path corridor  # noqa: E501
    K = np.ones_like(hw, dtype=np.float32)
    Pw = np.ones_like(hw, dtype=np.float32)
    Rw = np.full(hw.shape, BIG, np.float32)
    for name, d in info.items():
        b = d["body"]
        dw = ndimage.distance_transform_edt(~d["water"]).astype(np.float32) * cell
        db = ndimage.distance_transform_edt(~d["basin"]).astype(np.float32) * cell
        pm = np.ones(hw.shape, bool)
        pm[d["path"]] = False
        dp = ndimage.distance_transform_edt(pm).astype(np.float32) * cell
        del pm
        K *= np.maximum(_smooth01(5.0, 60.0, dw), _smooth01(0.0, 40.0, db)) * _smooth01(*b["k_corr"], dp)  # noqa: E501
        K[d["path"]] = 0.0
        # erosion-only preserve mask (no lowering): thin shore band + breach corridor
        Pw *= _smooth01(10.0, 60.0, dw) * _smooth01(*b["p_corr"], dp)
        # continuous deposition cap: natural shores (no "bathtub ring"), fans / deltas build  # noqa: E501
        # further out, no step anywhere; beds handled by the level masks (below the water line)  # noqa: E501
        rcap = np.where(d["water"], 3.0, 3.0 + 25.0 * _smooth01(20.0, 450.0, dw) ** 1.5 + 400.0 * _smooth01(450.0, 1500.0, dw))  # noqa: E501
        Rw = np.minimum(Rw, rcap.astype(np.float32))
        del dw, db, dp, rcap
    yy, xx = np.mgrid[0:hw.shape[0], 0:hw.shape[1]].astype(np.float32)
    # waterfall cliff site (east shore): low erodibility, hard rock, no deposits
    wf = None
    if waterfall_z is not None and "lake" in info:
        lake = info["lake"]
        zz = float(waterfall_z)
        xw = float(axis_x(zz))
        H = _Bilin(h0, cell)
        while H(xw, zz) < lake["level"] and xw < 3000.0:
            xw += 1.0
        dwf = np.hypot(xx - ((xw + half) / cell - off), yy - ((zz + half) / cell - off)) * cell  # noqa: E501
        wzm = 1.0 - _smooth01(150.0, 250.0, dwf)
        K *= 1.0 - 0.8 * wzm
        Rw = Rw * (1.0 - wzm) + np.minimum(Rw, 0.5) * wzm
        wf = dict(x=xw, z=zz, mask=wzm)
        log(f"  waterfall site: x={xw:.0f} z={zz:.0f} (protected r=150..250 m)")
        del dwf
    Pf = np.ones((res, res), np.float32)
    Pf[sl] = Pw
    Rf = np.full((res, res), BIG, np.float32)
    Rf[sl] = Rw
    Kf = np.ones((res, res), np.float32)
    Kf[sl] = K
    del Pw, Rw, K
    edge = np.minimum(np.arange(res), np.arange(res)[::-1]).astype(np.float32) * cell
    Kf *= _smooth01(0.0, 800.0, edge)[:, None] * _smooth01(0.0, 800.0, edge)[None, :]
    # broad low valley floors (gentle and low lying in the input) are depositional:
    # much lower erodibility keeps them flat (alluvial fill, one river) instead of badlands  # noqa: E501
    fq = max(1, res // 2048)
    h16 = _pool(h0, fq)
    hsm = ndimage.gaussian_filter(h16, 120.0 / (cell * fq))
    gz, gx = np.gradient(hsm, cell * fq)
    flat = (1.0 - _smooth01(0.10, 0.25, np.hypot(gx, gz))) * (1.0 - _smooth01(500.0, 900.0, hsm))  # noqa: E501
    Kf *= _up(1.0 - 0.85 * flat, fq, 1)
    del hsm, gz, gx, flat
    # crests / summits (skyline): harder, less erodible
    crest16 = _smooth01(20.0, 80.0, h16 - ndimage.gaussian_filter(h16, 150.0 / (cell * fq))) * _smooth01(1500.0, 2200.0, h16)  # noqa: E501
    crest = _up(crest16, fq, 1)
    Kf *= 1.0 - 0.8 * crest
    del h16, crest16

    route = np.full((res, res), BIG, np.float32)
    water_full, basin_full = {}, {}
    for name, d in info.items():
        pr, pc = d["path"]
        route[pr + off, pc + off] = d["level"] - 0.05
        wm = np.zeros((res, res), bool)
        wm[sl] = d["water"]
        bm = np.zeros((res, res), bool)
        bm[sl] = d["basin"]
        water_full[name], basin_full[name] = wm, bm
    log("  rock hardness field (noise + joint sets) ...")
    # glacially polished trough walls: harder rock (steeper faces) near the valley floor
    lake_c = info["lake"]["body"]["seed"] if "lake" in info else (0.0, 0.0)
    floor = ndimage.gaussian_filter(hw, 50.0 / cell) < 250.0
    floor &= np.hypot(yy - (lake_c[1] + half) / cell + off, xx - (lake_c[0] + half) / cell + off) * cell < 4000.0  # noqa: E501
    trough = np.zeros((res, res), np.float32)
    trough[sl] = 1.0 - _smooth01(300.0, 1600.0, ndimage.distance_transform_edt(~floor).astype(np.float32) * cell)  # noqa: E501
    del floor, yy, xx
    hard = hardness_field(res, cell, h0)
    hard *= (1.0 + 0.5 * trough) * (1.0 + 2.0 * crest)
    del trough
    if wf is not None:
        hard[sl] *= 1.0 + 0.6 * wf["mask"]
    np.minimum(hard, 8.0, out=hard)
    # stream corridor (16 m grid): pre-cut + softer rock so the eroded valley floor follows  # noqa: E501
    # the line features.py carves the inflow stream along
    corr = None
    if corridor and "lake" in info:
        pts, zt = stream_polyline(h0, cell, info["lake"]["level"])
        n16 = res // fq
        dist, sarc, L = _polyline_field(pts, n16, cell * fq, half)
        taper = _smooth01(0.0, 150.0, sarc) * (1.0 - _smooth01(L - 300.0, L, sarc))
        soft = (1.0 - _smooth01(50.0, 150.0, dist)) * (1.0 - _smooth01(L - 200.0, L + 100.0, sarc))  # noqa: E501
        cut = 15.0 * (1.0 - _smooth01(0.0, 90.0, dist)) * taper
        corr = dict(soft=soft.astype(np.float32), cut=cut.astype(np.float32), pts=pts, z_tip=zt, length=L)  # noqa: E501
        log(f"  stream corridor: north tip z={zt:.0f}, {L:.0f} m, pre-cut 15 m / 150 m")
        del dist, sarc
    dep_noise = 0.5 + 0.5 * np.tanh(_noise(512, 250.0 / (res * cell / 512), np.random.default_rng(29)))  # noqa: E501
    return dict(K=Kf, P=Pf, R=Rf, route=route, water=water_full, basin=basin_full, hard=hard, crest=crest,  # noqa: E501
                info=info, off=off, bodies=bodies, wf=wf, corr=corr, dep_noise=dep_noise.astype(np.float32))  # noqa: E501


def hardness_field(res, cell, h0=None, seed=11):
    """Multi-scale rock hardness (~0.3..2.7): drives differential erosion (soft bands
    become gullies/couloirs, hard bands stand as ribs and steeper faces) and breaks
    the
    8-direction symmetry of the grid processes. High summits are made of harder rock
    (steeper critical angles), which keeps horn peaks and aretes near their height.
    Two warped granite joint sets (strike ~N25E and ~N115E, spacing ~25..60 m) are
    softer:
    couloirs and ribs follow the joints."""
    rng = np.random.default_rng(seed)
    f = max(1, res // 2048)              # synthesise at <= 2048^2, then upsample bilinearly  # noqa: E501
    r, c = res // f, cell * f
    acc = np.zeros((r, r), np.float32)
    for feat, amp in ((900.0, 0.5), (260.0, 0.32), (70.0, 0.18)):
        n = max(4, int(round(r * c / feat)))
        g = rng.standard_normal((n + 1, n + 1)).astype(np.float32)
        acc += amp * ndimage.zoom(g, r / n, order=3, mode="nearest", grid_mode=True)[:r, :r]  # noqa: E501
    acc /= acc.std()
    hard = np.clip(1.0 + 0.38 * acc, 0.45, 1.8)
    if h0 is not None:
        hs = ndimage.gaussian_filter(_pool(h0, f), 150.0 / c)
        hard *= 1.0 + 0.9 * _smooth01(1500.0, 2900.0, hs)
    hard = _up(hard, f, 1)
    hard *= joint_field(res, cell, seed + 7)
    return hard


def joint_field(res, cell, seed=18):
    """Multiplicative hardness of two warped joint sets (1 = intact rock, ~0.65 in
    joints)."""
    rng = np.random.default_rng(seed)
    half = res * cell / 2
    fq = max(1, res // 1024)
    nq = res // fq
    out = np.ones((res, res), np.float32)
    xs = ((np.arange(res) + 0.5) * cell - half).astype(np.float32)
    for strike, lam, amp in ((25.0, 42.0, 0.35), (115.0, 33.0, 0.30)):
        a = math.radians(strike)
        warp = _up(_noise(nq, 400.0 / (cell * fq), rng), fq, 1)          # spacing +-25 %, bends  # noqa: E501
        wgt = _up(_smooth01(-0.6, 0.9, _noise(nq, 1800.0 / (cell * fq), rng)), fq, 1)   # regional dominance  # noqa: E501
        for r0 in range(0, res, 1024):
            zr = xs[r0:r0 + 1024, None]
            u = xs[None, :] * math.cos(a) - zr * math.sin(a)
            ph = 2.0 * math.pi * (u / lam + 1.2 * warp[r0:r0 + 1024])
            j = _smooth01(0.55, 0.95, np.cos(ph))
            out[r0:r0 + 1024] *= 1.0 - amp * j * (0.35 + 0.65 * wgt[r0:r0 + 1024])
        del warp, wgt
    return out


def level_mask(prot, f, h0l, sl=None, corridor=False):
    """(erodibility, route cap, deposition cap, hardness) at pooling factor f, float32.
    sl: optional window (slices in level coordinates)."""
    n = prot["K"].shape[0] // f

    def P(a, op="mean"):
        p = _pool(a, f, op)
        return p[sl] if sl is not None else p

    m = np.empty(h0l.shape + (4,), np.float32)
    m[..., 0] = P(prot["K"])
    m[..., 1] = P(prot["route"], "min")
    dep = np.full(h0l.shape, BIG, np.float32)
    noise = _resample(prot["dep_noise"], n)
    if sl is not None:
        noise = noise[sl]
    for b in prot["bodies"]:
        w = P(prot["water"][b["name"]].astype(np.float32)) > 0.5
        # beds: up to 3 m of fill, but stays 1.2..2 m below the water line (drop-off, no shelf)  # noqa: E501
        dep[w] = np.minimum(h0l[w] + 3.0, b["level"] - 1.2 - 0.8 * noise[w])
    m[..., 2] = dep
    hard = P(prot["hard"])
    if corridor and prot.get("corr") is not None:
        soft = _resample(prot["corr"]["soft"], n)
        hard *= 1.0 - 0.5 * (soft[sl] if sl is not None else soft)
    m[..., 3] = hard
    return m


# ------------------------------------------------------------------------------------
# coarse depression breaching (priority-flood carving)
# ------------------------------------------------------------------------------------
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def breach_depressions(z, keep_prio, eps=0.01):
    """Priority-flood with carving ("complete breaching", Lindsay 2016 / Barnes 2014):
    every closed depression gets an outlet carved along the flood path through its
    lowest
    saddle, so the whole grid drains to the border. Cells with a finite keep_prio
    (the
    protected lake / tarn basins and their rims) are seeds at that priority (their
    spill
    level) and are never modified. Returns the breached heights."""
    n0, n1 = z.shape
    N = n0 * n1
    out = np.asarray(z, np.float64).ravel().tolist()
    kp = np.asarray(keep_prio, np.float64).ravel()
    keepm = np.isfinite(kp)
    kb = bytearray(keepm.astype(np.uint8).tobytes())
    done = bytearray(kb)
    parent = [-1] * N
    border = np.zeros((n0, n1), bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    heap = []
    for i in np.flatnonzero(border.ravel() & ~keepm):
        i = int(i)
        heap.append((out[i], i))
        done[i] = 1
    for i in np.flatnonzero(keepm):
        i = int(i)
        heap.append((float(max(kp[i], out[i])), i))
    heapq.heapify(heap)  # noqa: F821
    pop, push = heapq.heappop, heapq.heappush  # noqa: F821
    while heap:
        _, c = pop(heap)
        r, q = divmod(c, n1)
        oc_keep = kb[c]
        for dr, dc in _NB8:
            rr, cc = r + dr, q + dc
            if rr < 0 or rr >= n0 or cc < 0 or cc >= n1:
                continue
            nb = rr * n1 + cc
            if done[nb]:
                continue
            done[nb] = 1
            parent[nb] = c
            zn = out[nb]
            if not oc_keep and zn < out[c]:
                v = zn
                p = c
                while p >= 0 and not kb[p]:
                    v -= eps
                    if out[p] <= v:
                        break
                    out[p] = v
                    p = parent[p]
            push(heap, (zn, nb))
    return np.array(out, np.float64).reshape(n0, n1)


def _cone_envelope(z, seeds, cell, tan_side, top):
    """Lower envelope of inverted cones (V-notch walls) with slope tan_side standing on
    the
    seed cells: g(x) = min_p z(p) + |x - p| * tan_side, each cone only up to the
    height
    top(p) (the uncut surface at the seed), so it never reaches beyond the notch
    walls.
    Iterated 8-neighbour chamfer relaxation; +inf where no cone applies."""
    g = np.where(seeds, z, np.inf).astype(np.float32)
    cap = np.where(seeds, top, -np.inf).astype(np.float32)
    if not seeds.any():
        return g
    rise = float((top - z)[seeds].max())
    n_it = int(min(400, rise / (tan_side * cell) + 2))
    # 16-neighbour chamfer (knight moves): near-isotropic cones, no octagonal facets
    nb16 = _NB8 + [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]  # noqa: E501
    steps = [(dr, dc, math.hypot(dr, dc) * cell * tan_side) for dr, dc in nb16]
    for _ in range(n_it):
        g0, c0 = g.copy(), cap.copy()
        for dr, dc, w in steps:
            ss = (slice(max(0, -dr), g.shape[0] - max(0, dr)), slice(max(0, -dc), g.shape[1] - max(0, dc)))  # noqa: E501
            ds = (slice(max(0, dr), g.shape[0] - max(0, -dr)), slice(max(0, dc), g.shape[1] - max(0, -dc)))  # noqa: E501
            cand = g0[ss] + w
            ok = (cand < c0[ss]) & (cand < g[ds])
            g[ds] = np.where(ok, cand, g[ds])
            cap[ds] = np.where(ok, c0[ss], cap[ds])
        if np.array_equal(g, g0):
            break
    return g


def _keep_prio(prot, f, shape):
    keep = np.full(shape, np.nan)
    for b in prot["bodies"]:
        bm = _pool(prot["basin"][b["name"]].astype(np.float32), f, "max") > 0
        bm = ndimage.binary_dilation(bm, iterations=2)
        keep[bm] = prot["info"][b["name"]]["spill_level"]
    return keep


def _harmonic_fill(z, rg):
    """Solve Laplace's equation on the cells rg (4-neighbour, Dirichlet = z on the cells
    around rg, Neumann at the map border). Returns the solution on the full grid (z
    elsewhere)."""
    from scipy.sparse.linalg import spsolve
    n0, n1 = z.shape
    nu = int(rg.sum())
    out = np.array(z, np.float64)
    if nu == 0:
        return out
    idx = np.full((n0, n1), -1, np.int64)
    idx[rg] = np.arange(nu)
    rr, cc = np.nonzero(rg)
    rows, cols, vals = [], [], []
    diag = np.zeros(nu)
    b = np.zeros(nu)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < n0) & (c2 >= 0) & (c2 < n1)
        i = idx[rr[ok], cc[ok]]
        j = idx[r2[ok], c2[ok]]
        diag[i] += 1.0
        inner = j >= 0
        rows.append(i[inner]); cols.append(j[inner]); vals.append(-np.ones(int(inner.sum())))  # noqa: E702,E501
        np.add.at(b, i[~inner], np.asarray(z, np.float64)[r2[ok][~inner], c2[ok][~inner]])  # noqa: E501
    rows.append(np.arange(nu)); cols.append(np.arange(nu)); vals.append(diag)  # noqa: E702
    A = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nu, nu))  # noqa: E501
    out[rg] = spsolve(A.tocsc(), b)
    return out


def drain_map(ht, prot, f, cell, max_cut, tan_side, fill_slope=0.0, membrane_km2=0.0, membrane_margin=0,  # noqa: E501
              flat_below=900.0):
    """Hybrid fill + breach (least change) of all closed depressions except the
    protected
    lake / tarn basins: deep basins are filled to within max_cut of their spill level
    (valley / cirque floors), the rest is carved through the lowest saddle (outlet
    notch,
    widened into a V-shaped col with side slopes tan_side so hillslopes do not refill
    it),
    so the map drains without kilometre-deep slots or giant flat plains."""
    keep = _keep_prio(prot, f, ht.shape)
    km = np.isfinite(keep)
    before = depression_stats(ht, cell, km)
    F = fill_depressions(ht, keep, eps=0.02)
    # carve depth per depression: small basins (cirque bowls) are carved (up to max_cut),  # noqa: E501
    # large ones (valley floors, > ~4 km2) mostly filled (flat alluvial floors are natural  # noqa: E501
    # there, a canyon to the map border is not)
    lab, nl = ndimage.label((F - ht > 0.5) & ~km, structure=np.ones((3, 3)))
    if nl:
        area = ndimage.sum(np.ones(ht.shape, np.float32), lab, np.arange(1, nl + 1)) * cell * cell / 1e6  # noqa: E501
        spill = ndimage.maximum(F, lab, np.arange(1, nl + 1))
        lowv = (spill < flat_below) & (area > 8.0)          # big low valley floors only
        cdep = np.concatenate([[0.0], np.where(lowv, 0.25 * max_cut, max_cut)])
        cmap = cdep[lab]
    else:
        cmap = np.zeros(ht.shape)
    z1 = np.where(km, ht, np.maximum(ht, F - cmap))
    hbr = breach_depressions(z1, keep)
    notch = (z1 - hbr) > 0.01
    zc = np.where(notch, hbr, ht)
    del z1, hbr
    if membrane_km2 and nl:
        # mountain basins: the floor behind the notch becomes a smooth harmonic membrane
        # spanned by the surrounding walls and the carved drain line (a valley head sloping  # noqa: E501
        # to the notch, no flat lake plain); low valley floors (spill < flat_below) and huge  # noqa: E501
        # basins are filled flat (alluvial plains)
        flat = (area > membrane_km2) | lowv
        small = np.concatenate([[False], ~flat])[lab]
        big = ndimage.binary_dilation(np.concatenate([[False], flat])[lab], iterations=2)  # noqa: E501
        rg = (ndimage.binary_dilation(small, iterations=membrane_margin) if membrane_margin else small) & ~km & ~notch & ~big  # noqa: E501
        u = _harmonic_fill(zc, rg)
        zc = np.where(rg, np.maximum(zc, u), zc)
        del small, big, rg, u
    del lab
    if fill_slope:
        # the floors behind the notches are filled with an inclined debris / alluvial surface  # noqa: E501
        # rising away from the notch (fill_slope), never above the original spill level
        zs = fill_sloped(zc, keep, F, fill_slope * cell)
    else:
        zs = fill_depressions(zc, keep, eps=0.02)
    del F
    if tan_side:
        zs = np.where(km, zs, np.minimum(zs, _cone_envelope(zs, notch, cell, tan_side, ht)))  # noqa: E501
    cutb = np.maximum(ht - zs, 0.0)
    fillb = np.maximum(zs - ht, 0.0)
    hbr = zs
    st = dict(before=before, cells=int((cutb > 0.005).sum()), max_cut=float(cutb.max()),
              mean_cut=float(cutb[cutb > 0.005].mean()) if (cutb > 0.005).any() else 0.0,  # noqa: E501
              fill_cells=int((fillb > 0.01).sum()), max_fill=float(fillb.max()))
    return hbr.astype(np.float32), st


def fill_sloped(z, keep_prio, cap, step):
    """Priority-flood filling (outlets: border + keep seeds) where the filled surface
    rises
    by `step` per cell away from the outlet (an inclined fan / debris floor instead
    of a flat
    lake plain), capped at `cap` (the plain spill-level fill); cells above stay
    untouched."""
    n0, n1 = z.shape
    zf = np.asarray(z, np.float64).ravel()
    f = zf.tolist()
    cp = np.asarray(cap, np.float64).ravel().tolist()
    kp = np.asarray(keep_prio, np.float64).ravel()
    keepm = np.isfinite(kp)
    done = bytearray(keepm.astype(np.uint8).tobytes())
    border = np.zeros((n0, n1), bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    heap = []
    for i in np.flatnonzero(border.ravel() & ~keepm):
        i = int(i)
        heap.append((f[i], i))
        done[i] = 1
    for i in np.flatnonzero(keepm):
        i = int(i)
        heap.append((float(max(kp[i], f[i])), i))
    heapq.heapify(heap)  # noqa: F821
    pop, push = heapq.heappop, heapq.heappush  # noqa: F821
    nbs = [(dr, dc, step * math.hypot(dr, dc)) for dr, dc in _NB8]
    while heap:
        v, c = pop(heap)
        r, q = divmod(c, n1)
        for dr, dc, s_ in nbs:
            rr, cc = r + dr, q + dc
            if rr < 0 or rr >= n0 or cc < 0 or cc >= n1:
                continue
            nb = rr * n1 + cc
            if done[nb]:
                continue
            done[nb] = 1
            lim = v + s_
            cnb = cp[nb]
            if cnb < lim:
                lim = cnb if cnb > v + 0.01 else v + 0.01
            if f[nb] < lim:
                f[nb] = lim
            push(heap, (f[nb], nb))
    out = np.array(f, np.float64).reshape(n0, n1)
    out[keepm.reshape(n0, n1)] = zf.reshape(n0, n1)[keepm.reshape(n0, n1)]
    return out


def fill_depressions(z, keep_prio, eps=0.01):
    """Priority-flood depression filling (Barnes 2014) with an eps gradient towards the
    outlets. Cells with a finite keep_prio (protected lake / tarn basins) are seeds
    at that
    priority and are never modified. Returns the filled heights."""
    n0, n1 = z.shape
    f = np.asarray(z, np.float64).ravel().tolist()
    kp = np.asarray(keep_prio, np.float64).ravel()
    keepm = np.isfinite(kp)
    done = bytearray(keepm.astype(np.uint8).tobytes())
    border = np.zeros((n0, n1), bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    heap = []
    for i in np.flatnonzero(border.ravel() & ~keepm):
        i = int(i)
        heap.append((f[i], i))
        done[i] = 1
    for i in np.flatnonzero(keepm):
        i = int(i)
        heap.append((float(max(kp[i], f[i])), i))
    heapq.heapify(heap)  # noqa: F821
    pop, push = heapq.heappop, heapq.heappush  # noqa: F821
    while heap:
        v, c = pop(heap)
        r, q = divmod(c, n1)
        for dr, dc in _NB8:
            rr, cc = r + dr, q + dc
            if rr < 0 or rr >= n0 or cc < 0 or cc >= n1:
                continue
            nb = rr * n1 + cc
            if done[nb]:
                continue
            done[nb] = 1
            if f[nb] < v + eps:
                f[nb] = v + eps
            push(heap, (f[nb], nb))
    out = np.array(f, np.float64).reshape(n0, n1)
    out[keepm.reshape(n0, n1)] = np.asarray(z, np.float64)[keepm.reshape(n0, n1)]
    return out


def _pflood(z):
    """Depression filling from the border (priority flood); returns filled heights."""
    n0, n1 = z.shape
    f = np.asarray(z, np.float64).ravel().tolist()
    done = bytearray(n0 * n1)
    heap = []
    for r in range(n0):
        for c in range(n1):
            if r in (0, n0 - 1) or c in (0, n1 - 1):
                i = r * n1 + c
                heap.append((f[i], i))
                done[i] = 1
    heapq.heapify(heap)  # noqa: F821
    while heap:
        v, c = heapq.heappop(heap)  # noqa: F821
        r, q = divmod(c, n1)
        for dr, dc in _NB8:
            rr, cc = r + dr, q + dc
            if 0 <= rr < n0 and 0 <= cc < n1:
                nb = rr * n1 + cc
                if not done[nb]:
                    done[nb] = 1
                    if f[nb] < v:
                        f[nb] = v
                    heapq.heappush(heap, (f[nb], nb))  # noqa: F821
    return np.array(f).reshape(n0, n1)


def depression_stats(z, cell, exclude=None, min_area=0.1e6, min_depth=2.0):
    """Closed depressions of z (number > min_area with depth > min_depth), excluding the
    components that touch `exclude` (lake / tarn)."""
    d = _pflood(z) - z
    lab, n = ndimage.label(d > min_depth, structure=np.ones((3, 3)))
    if n == 0:
        return dict(n_big=0, max_depth=0.0, area_frac=0.0)
    area = ndimage.sum(np.ones_like(d), lab, np.arange(1, n + 1)) * cell * cell
    depth = ndimage.maximum(d, lab, np.arange(1, n + 1))
    ok = np.ones(n, bool)
    if exclude is not None:
        ex = np.unique(lab[exclude & (lab > 0)])
        ok[ex[ex > 0] - 1] = False
    big = ok & (area > min_area)
    return dict(n_big=int(big.sum()), max_depth=float(depth[ok].max()) if ok.any() else 0.0,  # noqa: E501
                area_frac=float(area[ok].sum() / d.size / cell / cell))


# ------------------------------------------------------------------------------------
# hard guarantees and checks
# ------------------------------------------------------------------------------------
def enforce_basins(h, h0, prot):
    """Hard guarantee: lake/tarn cells stay under water (cells deeper than 1 m in the
    input
    stay at least 1 m deep), their rims stay above it."""
    stats = {}
    st3 = np.ones((3, 3), bool)
    for b in prot["bodies"]:
        name, level = b["name"], b["level"]
        w = prot["water"][name]
        hw, h0w = h[w], h0[w]
        # cells deeper than 1 m stay >= 1 m deep; shallower ones never rise above the input  # noqa: E501
        # (no flat shelf just under the water line)
        top = np.where(h0w < level - 1.0, level - 1.0, h0w)
        new = np.clip(hw, h0w - 4.0, np.minimum(top, h0w + 3.0))
        n_w = int((new != hw).sum())
        h[w] = new
        rim = ndimage.binary_dilation(w, structure=st3, iterations=2) & ~w
        hr = h[rim]
        floor = np.maximum(np.minimum(h0[rim], level + 0.3), level + 0.02)
        new = np.maximum(hr, floor)
        stats[name] = dict(water_cells_clamped=n_w, rim_cells_raised=int((new != hr).sum()),  # noqa: E501
                           max_rim_raise=float((new - hr).max()) if hr.size else 0.0)
        h[rim] = new
    return stats


def freeboard(h, cell, seed, level, win_m):
    """Spill level of the water body around seed (window +-win_m around the seed)."""
    res = h.shape[0]
    half = res * cell / 2
    r, c = int((seed[1] + half) / cell), int((seed[0] + half) / cell)
    w = int(win_m / cell)
    ra, rb, ca, cb = max(0, r - w), min(res, r + w), max(0, c - w), min(res, c + w)
    lv, sp, _ = _find_spill(h[ra:rb, ca:cb], (r - ra, c - ca), level, cell)
    return float(lv), (float((sp[1] + ca) * cell - half + cell / 2), float((sp[0] + ra) * cell - half + cell / 2))  # noqa: E501


def check_basins(h, cell, lake_seed=LAKE_C, tarn_seed=TARN_C, lake_level=LAKE_LEVEL, tarn_level=TARN_LEVEL):  # noqa: E501
    """The checks used by bake.py: flood fill h<0 from the lake seed and h<520 from the
    tarn seed (4-connected)."""
    res = h.shape[0]
    half = res * cell / 2

    def rc(x, z):
        return int((z + half) / cell), int((x + half) / cell)

    out = {}
    lab, _ = ndimage.label(h < lake_level)
    m = lab == lab[rc(*lake_seed)]
    out["lake_area_km2"] = float(m.sum() * cell * cell / 1e6)
    out["lake_touches_border"] = _touches_border(m)
    out["lake_closed"] = bool(m.sum() < 0.25 * h.size and not out["lake_touches_border"] and h[rc(*lake_seed)] < lake_level)  # noqa: E501
    rr, cc = np.nonzero(m)
    out["lake_bbox_xz"] = [float(cc.min() * cell - half), float(cc.max() * cell - half),
                           float(rr.min() * cell - half), float(rr.max() * cell - half)]
    del lab
    lab, _ = ndimage.label(h < tarn_level)
    t = lab == lab[rc(*tarn_seed)]
    out["tarn_area_ha"] = float(t.sum() * cell * cell / 1e4)
    out["tarn_closed"] = bool(h[rc(*tarn_seed)] < tarn_level and t.sum() * cell * cell < 40e4)  # noqa: E501
    out["bake_tarn_seed_in_tarn"] = bool(t[rc(-1680.0, -420.0)])
    return out, m, t


def peak_changes(h0, h, cell, within=6000.0, top=10):
    """Summits of the input (local maxima over ~600 m, > 1500 m) within +-within m of
    the
    origin; height change of the output's maximum within 150 m of each summit."""
    half = h0.shape[0] * cell / 2
    a, b = int((-within + half) / cell), int((within + half) / cell)
    a0 = h0[a:b, a:b]
    mx = ndimage.maximum_filter(a0, size=int(600 / cell) | 1)
    pk = np.argwhere((a0 == mx) & (a0 > 1500.0))
    a1 = ndimage.maximum_filter(h[a:b, a:b], size=int(300 / cell) | 1)
    rel = sorted(((float(a0[r, q]), float(a1[r, q])) for r, q in pk), key=lambda t: -t[0])  # noqa: E501
    pct = np.array([100.0 * (y - x) / x for x, y in rel])
    tp = pct[:top]
    return dict(top=[(round(x), round(y), round(float(p), 1)) for (x, y), p in zip(rel[:top], tp)],  # noqa: E501
                top_median_pct=round(float(np.median(tp)), 2), top_worst_pct=round(float(tp.min()), 2),  # noqa: E501
                all_n=len(rel), all_median_pct=round(float(np.median(pct)), 2), all_worst_pct=round(float(pct.min()), 2))  # noqa: E501


def slope_stats(h, cell, f=4, win=4096.0):
    """Slope (at f*cell) statistics in the central +-win m: share > 43 deg by elevation
    band,
    median above 1500 m and the 8-fold harmonic of the slope direction (grid facets)."""
    half = h.shape[0] * cell / 2
    a, b = int((-win + half) / cell), int((win + half) / cell)
    p = _pool(h[a:b, a:b], f)
    gz, gx = np.gradient(p, cell * f)
    sl = np.hypot(gx, gz)
    deg = np.degrees(np.arctan(sl))
    out = {}
    for lo, hi in ((0.5, 700), (300, 700), (700, 1500), (1500, 5000)):
        m = (p >= lo) & (p < hi)
        out[f"{lo:g}-{hi:g}"] = dict(gt43=round(float((deg[m] > 43).mean()), 4), median=round(float(np.median(deg[m])), 1))  # noqa: E501
    st = sl > 0.5
    ang = np.arctan2(gz, gx)[st]
    w = sl[st]
    out["harm8"] = round(float(np.abs((w * np.exp(8j * ang)).sum()) / w.sum()), 4)
    return out


def flat_share(h, cell, above=1200.0, deg=4.0):
    """Share of the terrain above `above` m flatter than `deg` at 64 m (filled-basin
    plains)."""
    f = max(1, int(round(64.0 / cell)))
    p = _pool(h, f) if f > 1 else h
    gz, gx = np.gradient(p, cell * f)
    m = p > above
    return round(float((np.degrees(np.arctan(np.hypot(gx, gz)))[m] < deg).mean()), 4)


def acceptance_metrics(h0, out, loose, cell, prot=None, stats=None, runtime=None):
    """The review's acceptance metrics (review_11393 fixes 1-6, 9)."""
    h = out["height"]
    m = {}
    m["finite"] = bool(all(np.isfinite(v).all() for v in out.values()))
    m["loose_max_m"] = round(float(loose.max()), 2)
    bed = h - loose
    low = h0 - bed
    m["bedrock_lowering_max_m"] = round(float(low.max()), 1)
    m["bedrock_lowering_p999_m"] = round(float(np.percentile(low[::4, ::4], 99.9)), 1)
    del bed, low
    # closed depressions at 64 m other than the lake / tarn
    f = max(1, int(round(64.0 / cell)))
    _, lm0, tm0 = check_basins(h0, cell)
    ex = _pool((lm0 | tm0).astype(np.float32), f, "max") > 0
    m["depressions_64m_input"] = depression_stats(_pool(h0, f), cell * f, ex)
    m["depressions_64m_output"] = depression_stats(_pool(h, f), cell * f, ex)
    s0, s1 = slope_stats(h0, cell), slope_stats(h, cell)
    m["slope16_input"], m["slope16_output"] = s0, s1
    m["peaks"] = peak_changes(h0, h, cell)
    m["flat_share_above1200"] = dict(input=flat_share(h0, cell), output=flat_share(h, cell))  # noqa: E501
    chk, lm1, tm1 = check_basins(h, cell)
    m["basins"] = dict(chk, lake_mask_identical=bool((lm0 == lm1).all()), tarn_mask_identical=bool((tm0 == tm1).all()))  # noqa: E501
    # lake bed: share of lake cells in the top 0.5 m (flat shelves) and change vs shore distance  # noqa: E501
    m["lake_top05m_share"] = dict(input=round(float(((h0[lm0] >= -0.5)).mean()), 4), output=round(float(((h[lm1] >= -0.5)).mean()), 4))  # noqa: E501
    half = h.shape[0] * cell / 2
    a, b = int((-4096 + half) / cell), int((4096 + half) / cell)
    dw = ndimage.distance_transform_edt(~(lm0 | tm0)[a:b, a:b]).astype(np.float32) * cell  # noqa: E501
    d = (h - h0)[a:b, a:b]
    curve = []
    for lo in (0, 50, 100, 200, 300, 400, 500, 550, 600, 650, 700, 800, 1000, 1500):
        sel = (dw > lo) & (dw <= lo + 50)
        curve.append((lo, round(float(np.abs(d[sel]).mean()), 1), round(float(np.percentile(d[sel], 95)), 1)))  # noqa: E501
    m["change_vs_shore_dist"] = curve
    del dw, d
    if stats is not None:
        m["freeboard"] = stats.get("freeboard")
    if runtime is not None:
        m["runtime_s"] = round(float(runtime), 1)
    dep = m["depressions_64m_output"]["n_big"]
    pk = m["peaks"]
    m["pass"] = dict(
        loose_lt60=m["loose_max_m"] < 60.0,
        bedrock_lowering_lt400=m["bedrock_lowering_max_m"] < 400.0,
        depressions_le5=dep <= 5,
        slope43_0_700_le_input_plus5=s1["0.5-700"]["gt43"] <= s0["0.5-700"]["gt43"] + 0.05,  # noqa: E501
        median_above1500_45_55=45.0 <= s1["1500-5000"]["median"] <= 55.0,
        harm8_lt003=s1["harm8"] < 0.03,
        peaks_median_ge_m3=pk["top_median_pct"] >= -3.0, peaks_worst_ge_m10=pk["top_worst_pct"] >= -10.0,  # noqa: E501
        lake_tarn_closed=bool(chk["lake_closed"] and chk["tarn_closed"]),
        masks_identical=m["basins"]["lake_mask_identical"] and m["basins"]["tarn_mask_identical"],  # noqa: E501
        freeboard_ge1=all(v["freeboard_m"] >= 1.0 for v in (m.get("freeboard") or {}).values()),  # noqa: E501
        finite=m["finite"])
    return m


# ---- copied from engine/erosion.py:1240-1544 (verbatim)
def erode(hfar, cell_m, stages=None, log=print, debug=None, prot=None, stop_after=None,
          lake_seed=LAKE_C, tarn_seed=TARN_C, lake_level=LAKE_LEVEL, tarn_level=TARN_LEVEL,  # noqa: E501
          waterfall_z=WF_Z, stream_corridor=True, window_m=6144.0, checkpoint_dir=None, resume_after=None,  # noqa: E501
          final_cleanup=40.0):
    """Erode the far height map hfar (square, cell_m metres per texel, centred on the
    origin). Lake / tarn are flood filled from their seeds (x, z) below their levels
    and
    stay closed with identical water masks. Returns dict(height, flow, talus,
    sediment)."""
    t_all = time.time()
    h0 = np.ascontiguousarray(hfar, dtype=np.float32)
    if not np.isfinite(h0).all():
        raise ValueError("erode: input height map has non-finite values")
    res = h0.shape[0]
    stages = stages or STAGES
    stats = dict(levels={})
    if prot is None:
        log("erosion: building lake/tarn protection ...")
        t0 = time.time()
        prot = build_protection(h0, cell_m, make_bodies(lake_seed, tarn_seed, lake_level, tarn_level), log,  # noqa: E501
                                waterfall_z=waterfall_z, corridor=stream_corridor)
        log(f"  protection {time.time()-t0:.1f}s")
    hb = hs = trk = W = A = None
    h0prev = None
    prev_res = None
    ckdir = Path(checkpoint_dir) if checkpoint_dir else None
    for st in stages:
        t0 = time.time()
        r = st["res"]
        f = res // r
        cell = cell_m * f
        final = r == res
        if resume_after is not None and r <= resume_after:
            if r == resume_after:       # experiments: continue from a saved level state
                z = np.load(ckdir / f"ck_{r}.npz")
                hb, hs, trk = z["hb"], z["hs"], z["trk"]
                W = z["W"] if z["W"].size > 1 else None
                A = z["A"] if z["A"].size > 1 else None
                h0prev, prev_res = _pool(h0, f), r
                log(f"  level {r}^2: resumed from {ckdir / f'ck_{r}.npz'}")
            continue
        h0l = _pool(h0, f)
        if hb is None:
            hb, hs = h0l.copy(), np.zeros_like(h0l)
            trk = np.zeros(h0l.shape + (2,), np.float32)
        else:
            # coarse eroded surface + the input's own finer detail (half of it where water  # noqa: E501
            # collects: the simulated channels replace the input's built-in gullies)
            g = r // prev_res
            detail = h0l - _up(h0prev, g, 3)
            if A is not None and st.get("detail_wet", 0.0) > 0.0:
                # channels = large specific catchment area a = A / cell (m): the MFD area alone  # noqa: E501
                # grows along every long hillslope and would strip the input's fine relief  # noqa: E501
                # (the feather gullies) from all lower slopes
                detail *= 1.0 - st["detail_wet"] * _smooth01(3e3, 3e4, _up(A, g, 1) / (cell * g))  # noqa: E501
            htot = _up(hb + hs, g, 3) + detail
            del detail
            hs = np.maximum(_up(hs, g, 1), 0.0)
            hb = htot - hs
            del htot
            trk = _up(trk, g, 1)
        h0prev = h0l
        wsl = None
        if final and window_m:
            wc = min(res // 2, int(round(window_m / cell_m / 16.0)) * 16)
            wsl = (slice(res // 2 - wc, res // 2 + wc), slice(res // 2 - wc, res // 2 + wc))  # noqa: E501
        fluvial = st.get("steps", 0) > 0
        mask = level_mask(prot, f, h0l if wsl is None else h0l[wsl], sl=wsl, corridor=st.get("corridor", False))  # noqa: E501
        if wsl is None:
            sim = Sim(hb, hs, mask, trk, cell, fluvial=fluvial)
        else:
            sim = Sim(hb[wsl], hs[wsl], mask, trk[wsl], cell, fluvial=False)
        log(f"  level {r}^2 ({cell:.0f} m{', window %d^2' % sim.res if wsl is not None else ''}): upload {time.time()-t0:.1f}s")  # noqa: E501
        t1 = time.time()
        pre = dict(st, Tb=st.get("pre_Tb", st["Tb"]), TbLow=st.get("pre_TbLow", st["TbLow"]), Debris=0.0, R1=0.0)  # noqa: E501
        for _ in range(st.get("pre", 0)):
            sim.hill(pre, False)
        for _ in range(st.get("post", 0)):
            sim.hill(st, False)
        sim.c.finish()
        log(f"    hillslope pre/post-pass {st.get('pre', 0)}+{st.get('post', 0)} steps {time.time()-t1:.1f}s")  # noqa: E501
        do_cut = r == stages[0]["res"] and st.get("corridor") and prot.get("corr") is not None  # noqa: E501
        if st.get("breach") or do_cut:
            hb_, hs_ = sim.read_h()
            ht = hb_ + hs_
            if do_cut:
                # inflow stream corridor: pre-cut (never below the lake level + 1.5 m)
                cut = _resample(prot["corr"]["cut"], r)
                ht = ht - np.minimum(cut, np.maximum(ht - (lake_level + 1.5), 0.0))
            if st.get("breach"):
                t1 = time.time()
                ht, stats["breach"] = drain_map(ht, prot, f, cell, st.get("breach_max_cut", 150.0),  # noqa: E501
                                                st.get("breach_side", _deg(50)), st.get("fill_slope", 0.0))  # noqa: E501
                bs = stats["breach"]
                log(f"    breach: {bs['before']['n_big']} depressions >0.1 km2 (max depth {bs['before']['max_depth']:.0f} m); "  # noqa: E501
                    f"filled {bs['fill_cells']} cells (max {bs['max_fill']:.0f} m), "
                    f"carved {bs['cells']} cells (max {bs['max_cut']:.0f} m)  {time.time()-t1:.1f}s")  # noqa: E501
            sim.write_h(ht - hs_, hs_)
            del hb_, hs_, ht
        if fluvial:
            t1 = time.time()
            npass, unr = sim.fill_converge(st["eps"], 64, r)
            log(f"    fill init {npass} passes ({unr} undrained) {time.time()-t1:.1f}s")
            t1 = time.time()
            up_every = st.get("uplift_every", 0)
            if up_every:
                # large-scale anchoring ("tectonic" uplift, never subsidence): every uplift_every  # noqa: E501
                # steps the low-pass (gaussian, uplift_sigma) of the terrain is pulled back up to  # noqa: E501
                # that of the level's start state, so the trough, massifs and mean heights keep  # noqa: E501
                # their shape while valleys, ridges, gullies and fans below that scale reorganise  # noqa: E501
                lp_f = max(1, int(round(64.0 / cell)))
                sig = st["uplift_sigma"] / (cell * lp_f)
                hb_, hs_ = sim.read_h()
                lp0 = ndimage.gaussian_filter(_pool(hb_ + hs_, lp_f), sig, mode="nearest")  # noqa: E501
                del hb_, hs_
                # spatially smooth uplift (fades out only at the map border and over the
                # protected water bodies): a K-modulated uplift would sink the low-erodibility  # noqa: E501
                # valley floors relative to their surroundings and create closed basins
                edge = np.minimum(np.arange(r), np.arange(r)[::-1]).astype(np.float32) * cell  # noqa: E501
                upm = _smooth01(0.0, 800.0, edge)[:, None] * _smooth01(0.0, 800.0, edge)[None, :]  # noqa: E501
                wet = np.zeros((r, r), np.float32)
                for b in prot["bodies"]:
                    wet = np.maximum(wet, _pool(prot["water"][b["name"]].astype(np.float32), f))  # noqa: E501
                upm *= 1.0 - np.clip(ndimage.gaussian_filter(wet, max(60.0 / cell, 0.5)) * 2.0, 0.0, 1.0)  # noqa: E501
                del edge, wet
            for s in range(st["steps"]):
                if s > 0 and s % st["reinit"] == 0:
                    sim.fill(npass, st["eps"], 16, init=True)
                sim.step(st)
                if up_every and s % up_every == up_every - 1:
                    hb_, hs_ = sim.read_h()
                    lp = ndimage.gaussian_filter(_pool(hb_ + hs_, lp_f), sig, mode="nearest")  # noqa: E501
                    U = _up(np.maximum(lp0 - lp, 0.0), lp_f, 1) * (upm * st.get("uplift_frac", 1.0))  # noqa: E501
                    sim.write_h(hb_ + U, hs_)
                    sim.fill(npass, st["eps"], 16, init=True)
                    del hb_, hs_, U
                if s % 100 == 99:
                    sim.c.finish()
                    log(f"    step {s+1}/{st['steps']}  {time.time()-t1:.1f}s")
            sim.c.finish()
            if st.get("pit_fill"):
                sim.fill_converge(st["eps"], 64, r)      # exact fill surface of the final state  # noqa: E501
            W = _read(sim.W, 1)
            A = _read(sim.acc)[..., 0]
        hbw, hsw, trkw = sim.read_state()
        if fluvial and st.get("pit_fill") and wsl is None:
            # pits left at the end of the level (input detail re-added across channels, fresh  # noqa: E501
            # hillslope dams): shallow ones (< pit_fill m) are filled with sediment up to the  # noqa: E501
            # fill surface W (the protected lake / tarn basins excepted)
            pf = np.clip(W - (hbw + hsw), 0.0, None)
            prot_b = np.zeros(pf.shape, bool)
            for b in prot["bodies"]:
                prot_b |= _pool(prot["basin"][b["name"]].astype(np.float32), f, "max") > 0  # noqa: E501
            prot_b = ndimage.binary_dilation(prot_b, iterations=2)
            lab, nl = ndimage.label(pf > 0.05)
            if nl:
                dmax = ndimage.maximum(pf, lab, np.arange(1, nl + 1))
                ok = np.concatenate([[False], dmax <= st["pit_fill"]])
                pf = np.where(ok[lab] & ~prot_b, pf, 0.0)
                hsw += pf
                trkw[..., 1] += pf
                xs = np.maximum(hsw - st["LooseMax"], 0.0)       # thick fills consolidate  # noqa: E501
                hsw -= xs
                hbw += xs
                del xs
                stats.setdefault("pit_fill", {})[r] = dict(n=int(ok.sum()), cells=int((pf > 0.05).sum()), max=float(pf.max()),  # noqa: E501
                                                           n_deeper=int(nl - ok.sum()))
                log(f"    pit fill: {int(ok.sum())} pits filled (max {pf.max():.1f} m), {nl - int(ok.sum())} deeper left")  # noqa: E501
            del pf, prot_b, lab
        sim.release()
        del mask
        if wsl is None:
            hb, hs, trk = hbw, hsw, trkw
        else:
            # feather the 4 m window into the surroundings (8 m result + input 4 m detail)  # noqa: E501
            n = hbw.shape[0]
            e = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float32)
            ramp = np.minimum.outer(e, e)
            ramp = _smooth01(0.0, 64.0, ramp)
            hb[wsl] = hb[wsl] + (hbw - hb[wsl]) * ramp
            hs[wsl] = hs[wsl] + (hsw - hs[wsl]) * ramp
            trk[wsl] = trk[wsl] + (trkw - trk[wsl]) * ramp[..., None]
            del ramp, e
        del hbw, hsw, trkw
        # no lowering in the shore band and on the sills, continuous deposition cap
        d = hb + hs - h0l
        cap = _pool(prot["R"], f, "min")
        capped = d > cap
        d = np.where(d < 0.0, d * _pool(prot["P"], f), np.minimum(d, cap))
        # soft limit of the total lowering (knee at MAX_LOWER_KNEE, asymptote MAX_LOWER): only  # noqa: E501
        # the deepest notches / retreating faces are affected
        lk, lm = MAX_LOWER_KNEE, MAX_LOWER
        d = np.where(d < -lk, -lk - (lm - lk) * np.tanh((-d - lk) / (lm - lk)), d)
        hs = np.where(capped, np.minimum(hs, np.maximum(d, 0.0)), hs)       # deposits removed by the cap  # noqa: E501
        hb = h0l + d - hs
        del d, cap, capped
        if st.get("cleanup_max_cut"):
            # drain what the simulation left closed (collapsed notches, fresh pits)
            t1 = time.time()
            ht = hb + hs
            fq = max(1, r // 512)            # on the 64 m grid; finer levels get the upsampled change  # noqa: E501
            hp = _pool(ht, fq) if fq > 1 else ht
            hn, cs = drain_map(hp, prot, f * fq, cell * fq, st["cleanup_max_cut"], st.get("breach_side", _deg(50)),  # noqa: E501
                               st.get("fill_slope", 0.0))
            dd = _up_exact(hn - hp, fq) if fq > 1 else hn - ht
            del hp, hn
            hs = np.maximum(hs + np.minimum(dd, 0.0), 0.0)
            hb = ht + dd - hs
            stats.setdefault("cleanup", {})[r] = cs
            log(f"    cleanup: {cs['before']['n_big']} depressions >0.1 km2 (max depth {cs['before']['max_depth']:.0f} m); "  # noqa: E501
                f"filled {cs['fill_cells']} cells (max {cs['max_fill']:.0f} m), carved {cs['cells']} (max {cs['max_cut']:.0f} m)"  # noqa: E501
                f"  {time.time()-t1:.1f}s")
            del ht, dd
        if final:
            # full resolution flow map on the final terrain
            t1 = time.time()
            h = hb + hs
            g = r // prev_res
            mroute = np.zeros(h.shape + (2,), np.float32)
            mroute[..., 1] = prot["route"]
            fsim = Sim(hb, hs, mroute, None, cell, fluvial=False)
            del mroute
            Wi = np.maximum(np.minimum(h, prot["route"]), _up(W, g, 1)) if W is not None else None  # noqa: E501
            Ai = _up(A, g, 1) if A is not None else None
            fsim.alloc_flow(acc_comps=1, W=Wi, A=Ai)
            del Wi, Ai, h
            if W is None:
                fsim.fill(0, st["eps"], 16, init=True)
            fsim.fill(st["flow_fill"], st["eps"], 16)
            fsim.flow_only(st["p"], st["flow_acc"], 16)
            fsim.c.finish()
            A = _read(fsim.acc, 1)
            fsim.release()
            log(f"    flow map {time.time()-t1:.1f}s")
        if st.get("breach"):
            keep = np.zeros(hb.shape, bool)
            for bb in prot["bodies"]:
                keep |= _pool(prot["basin"][bb["name"]].astype(np.float32), f, "max") > 0  # noqa: E501
            stats["depressions_after_first_level"] = depression_stats(hb + hs, cell, keep)  # noqa: E501
            log(f"    depressions >0.1 km2 after this level: {stats['depressions_after_first_level']}")  # noqa: E501
        lv = dict(time_s=round(time.time() - t0, 1), loose_max=float(hs.max()),
                  delta_min=float((hb + hs - h0l).min()), delta_max=float((hb + hs - h0l).max()))  # noqa: E501
        stats["levels"][r] = lv
        log(f"  level {r}^2 done in {time.time()-t0:.1f}s  (loose max {lv['loose_max']:.1f} m, "  # noqa: E501
            f"change {lv['delta_min']:.0f}..{lv['delta_max']:.0f} m)")
        prev_res = r
        if ckdir is not None and not final:
            ckdir.mkdir(parents=True, exist_ok=True)
            np.savez(ckdir / f"ck_{r}.npz", hb=hb, hs=hs, trk=trk,
                     W=W if W is not None else np.zeros(1, np.float32), A=A if A is not None else np.zeros(1, np.float32))  # noqa: E501
        if stop_after is not None and r == stop_after:   # experiments: intermediate level  # noqa: E501
            return dict(height=hb + hs, loose=hs, track=trk, flow=A, h0=h0l, res=r, cell=cell, prot=prot, stats=stats)  # noqa: E501
    h = hb + hs
    # skyline: soft restore of crests lowered by more than 15 m (not on drainage paths: the  # noqa: E501
    # outlet notches cut through ridges stay open)
    fq = max(1, int(round(16.0 / cell_m)))
    chan = ndimage.maximum_filter(_pool(A, fq, "max"), size=2 * int(round(100.0 / (cell_m * fq))) + 1)  # noqa: E501
    chan = _up(_smooth01(1e4, 5e4, chan), fq, 1)          # within ~100 m of a stream / notch  # noqa: E501
    restore = 0.8 * prot["crest"] * np.maximum(h0 - h - 15.0, 0.0) * (1.0 - chan)
    del chan
    h += restore
    stats["crest_restore_max"] = float(restore.max())
    del restore
    if final_cleanup:
        # last drainage check on the 64 m grid (what the restore / 4 m detail may have closed)  # noqa: E501
        t1 = time.time()
        fq = max(1, int(round(64.0 / cell_m)))
        for it in range(3):          # the bilinear upsampled change leaves small residues: repeat  # noqa: E501
            hp = _pool(h, fq)
            hn, cs = drain_map(hp, prot, fq, cell_m * fq, final_cleanup, _deg(50))
            dd = _up_exact(hn - hp, fq)
            del hp, hn
            hs = np.maximum(hs + np.minimum(dd, 0.0), 0.0)
            h += dd
            del dd
            stats.setdefault("final_cleanup", []).append(cs)
            log(f"  final cleanup {it + 1}: {cs['before']['n_big']} depressions >0.1 km2 (max depth "  # noqa: E501
                f"{cs['before']['max_depth']:.0f} m); filled {cs['fill_cells']} cells (max {cs['max_fill']:.0f} m), "  # noqa: E501
                f"carved {cs['cells']} (max {cs['max_cut']:.0f} m)  {time.time()-t1:.1f}s")  # noqa: E501
            if cs["before"]["n_big"] <= 2:
                break
    clamp = enforce_basins(h, h0, prot)
    hb = h - hs
    talus_m = np.clip(trk[..., 0], 0.0, hs)
    # talus output = 0..1 scree weight (bake_forest uses it as such): ~0 below 0.15 m of
    # hillslope debris, 1 above ~2 m (aprons, cones below the faces)
    talus = _smooth01(0.15, 2.0, talus_m)
    # exposed scree only: colluvium below the forest line is vegetated (faded out 350..650 m)  # noqa: E501
    talus *= _smooth01(350.0, 650.0, h)
    sediment = np.clip(trk[..., 1], 0.0, hs)
    if not (np.isfinite(h).all() and np.isfinite(A).all()):
        raise RuntimeError("erode: non-finite output")
    fb = {}
    for b in prot["bodies"]:
        win = 4096.0 if b["name"] == "lake" else 1500.0
        sp, xz = freeboard(h, cell_m, b["seed"], b["level"], win)
        fb[b["name"]] = dict(spill_level=round(sp, 2), freeboard_m=round(sp - b["level"], 2),  # noqa: E501
                             spill_xz=[round(xz[0]), round(xz[1])],
                             input_spill_level=round(prot["info"][b["name"]]["spill_level"], 2))  # noqa: E501
        if sp < b["level"] + 1.0:
            log(f"  WARNING: {b['name']} freeboard only {sp - b['level']:.2f} m after erosion")  # noqa: E501
    stats.update(clamp=clamp, freeboard=fb, total_s=round(time.time() - t_all, 1))
    if debug is not None:
        debug.update(prot=prot, clamp=clamp, loose=hs, talus_m=talus_m, stats=stats)
    log(f"erosion total {time.time()-t_all:.1f}s; clamp {clamp}; freeboard {fb}")
    return dict(height=h.astype(np.float32), flow=A.astype(np.float32),
                talus=talus.astype(np.float32), sediment=sediment.astype(np.float32))

# ==== END ENGINE COPIES ====
