#!/usr/bin/env python3
"""vr.py - MoltenVK equirectangular panorama renderer (Phase 4).

port/RUNBOOK.md section 6 BINDING: one pixel per (lon, lat) ray; lon in
[-pi, pi) across W, lat in [pi/2, -pi/2] top-down; pixel centre (x + 0.5);
d = (cos lat sin lon, sin lat, cos lat cos lon); Y-up engine frame.
Mono = one WxH equirect frame. SBS = two eye frames side by side, parallel
projection: SHARED direction grid, eye origins at head +/- 0.032 m along
camera-right (pano.stereo_pair), identical per-pixel seed.

RUN MODES
  --dry-run  prints one JSON render plan and exits. It never imports the
             Vulkan layer: engine_vr.passes_vr / gl_vr / bake_vr are
             imported only inside _render(), so Host.open() is
             unreachable in a dry run.
  (no flag)  first-run tool: Host via passes_vr.open_host (gpuslot
             first), world textures + shadow/tree bakes, tiled raymarch
             (scene_pano_cone -> scene_pano_trace -> scene_pano_kind*),
             engine/post grade (IMPORTED from engine/post.py, not copied)
             and one PNG (Pillow).

Y convention: every GPU image is GL row order (row 0 = first array row);
the equirect grid has lat +90 at row 0, so one display flip at readback
maps it to the top-down picture the post module and the PNG expect. The
flip is copied from engine/render.py:505 (read_tex(acc, 3)[::-1] / n).
"""

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))

from engine_vr import cache_schema as cs   # noqa: E402
from engine_vr import pano                 # noqa: E402
import post                                # noqa: E402  engine/post.py (IMPORT)

MANIFEST = ROOT / "engine_vr" / "manifest.json"

KIND_NAMES = ["sky", "terrain", "tree", "rock", "water", "cover", "hero",
              "flora"]
PANO_CONE = "scene_pano_cone"
PANO_TRACE = "scene_pano_trace"


def pano_program_names():
    return [PANO_CONE, PANO_TRACE] + [
        "scene_pano_kind" + str(k) + "_" + KIND_NAMES[k] for k in range(8)]


# ---------------------------------------------------------------------------
# copied engine snippets (every header names its exact source lines)
# ---------------------------------------------------------------------------
# copied from engine/render.py:31-38 (Camera.basis)
def basis(pos, target, roll=0.0):
    f = np.subtract(target, pos)
    f = f / np.linalg.norm(f)
    r = np.cross([0.0, 1.0, 0.0], f)
    r = r / np.linalg.norm(r)
    u = np.cross(f, r)
    if roll:
        a = math.radians(roll)
        r, u = r * math.cos(a) + u * math.sin(a),  u * math.cos(a) - r * math.sin(a)
    return np.stack([r, u, f])       # rows = GLSL mat3 columns


# copied from engine/render.py:265-269 (halton)
def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


# copied from engine/render.py:233-262 (tree_tops, docstring trimmed)
def tree_tops(a, blk=4, cell=4.0, reach=12.0):
    n = a.shape[0] // blk
    occ = a[..., 1] != 0.0
    top = np.where(occ, a[..., 0] + np.minimum(1.6 * np.abs(a[..., 1]), 34.0)
                   + 2.4, -1e4).astype(np.float32)
    top = top.reshape(n, blk, n, blk).max(axis=(1, 3))
    ob = occ.reshape(n, blk, n, blk).any(axis=(1, 3))

    def dil(x, fill):
        p = np.pad(x, 1, constant_values=fill)
        r = x.copy()
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                r = np.maximum(r, p[dy:dy + n, dx:dx + n])
        return r
    top3 = dil(top, -1e4)
    k = np.where(ob, 0, 64).astype(np.int32)
    cur = ob.copy()
    for i in range(1, 64):
        cur = dil(cur, False)
        k = np.where((k == 64) & cur, i, k)
        if cur.all():
            break
    clr = (np.maximum(k, 2) - 1) * blk * cell - reach
    return np.stack([top3, clr.astype(np.float32)], -1)


# copied from engine/render.py:42-124 (Look dataclass, defaults, sun_dir,
# shadow_dir; docstrings trimmed)
@dataclasses.dataclass(frozen=True)
class Look:
    sun_azimuth: float = 115.0
    sun_elevation: float = 44.0
    sun_intensity: float = 2.6
    haze: float = 1.0
    cloud_cover: float = 0.48
    wind: tuple = (6.0, 1.5)
    water_wind: tuple = (0.6, -0.8)
    wave_strength: float = 1.0
    water_ins: tuple = (0.20, 0.50, 0.64)
    sky_boost: float = 22.0
    amb_boost: float = 2.5
    sky_desat: float = 0.06
    exposure: float = 1.0
    # copied from engine/render.py:98-99
    tune: tuple = (1.5, 0.35, 0.35, 20.0, 1.0, 0.35, 0.25, 0.5, 0.003, 0.4,
                   1.0, 1.5, 1.0, 0.35, 0.35, 50.0, 0.0, 150.0, 1500.0, 0.4,
                   0.6, 0.7, 0.6, 0.0)
    shadow_elevation: float = 0.0
    shadow_azimuth: float = -1.0

    # copied from engine/render.py:109-111
    def sun_dir(self):
        az = math.radians(self.sun_azimuth)
        el = math.radians(self.sun_elevation)
        return np.array([math.sin(az) * math.cos(el), math.sin(el),
                         math.cos(az) * math.cos(el)])

    # copied from engine/render.py:113-119
    def shadow_dir(self):
        if self.shadow_elevation <= 0:
            return self.sun_dir()
        az = math.radians(self.shadow_azimuth if self.shadow_azimuth >= 0
                          else self.sun_azimuth)
        el = math.radians(self.shadow_elevation)
        return np.array([math.sin(az) * math.cos(el), math.sin(el),
                         math.cos(az) * math.cos(el)])


# copied from engine/shots.py:9 (axis_x)
def axis_x(z):
    return 150.0 * math.sin(z * 0.00105 + 0.5) + 45.0 * math.sin(z * 0.0029 + 2.1)


# copied from engine/shots_film.py:27-29 (toward)
def toward(p, az, pitch, dist=1500.0):
    a, e = math.radians(az), math.radians(pitch)
    return (p[0] + math.sin(a) * math.cos(e) * dist,
            p[1] + math.sin(e) * dist,
            p[2] + math.cos(a) * math.cos(e) * dist)


class PanoWorld:
    """CPU world for the VIEWPOINT recipes (heights, water, features).

    height / water copied from engine/render.py:211-230 (World.height,
    World.water); ground / P / shore_south / shore_x from
    engine/shots_film.py:21-41 (build() closures); head_peak from
    engine/shots_film.py:129-139. bake.CACHE -> cs.path and
    bake.NEAR_HALF / bake.FAR_HALF -> cache_schema constants (equal by the
    P2 constant pinning).
    """

    def __init__(self):
        self.hfar = np.load(cs.path("h_far.npy"))
        self.hnear = np.load(cs.path("h_near.npy"))
        self.level = np.load(cs.path("water_level.npy"))
        self.features = None
        fp = cs.path("features.json")
        if fp.exists():
            self.features = json.loads(fp.read_text())

    # copied from engine/render.py:211-223
    def height(self, x, z):
        def bil(a, half, x, z):
            res = a.shape[0]
            fx = (x + half) / (2 * half) * res - 0.5
            fz = (z + half) / (2 * half) * res - 0.5
            ix, iz = int(np.floor(fx)), int(np.floor(fz))
            tx, tz = fx - ix, fz - iz
            ix = np.clip(ix, 0, res - 2)
            iz = np.clip(iz, 0, res - 2)
            return ((a[iz, ix] * (1 - tx) + a[iz, ix + 1] * tx) * (1 - tz)
                    + (a[iz + 1, ix] * (1 - tx) + a[iz + 1, ix + 1] * tx) * tz)
        if max(abs(x), abs(z)) < cs.NEAR_HALF * 0.95:
            return float(bil(self.hnear, cs.NEAR_HALF, x, z))
        return float(bil(self.hfar, cs.FAR_HALF, x, z))

    # copied from engine/render.py:225-230
    def water(self, x, z):
        res = self.level.shape[0]
        ix = int((x + cs.FAR_HALF) / (2 * cs.FAR_HALF) * res)
        iz = int((z + cs.FAR_HALF) / (2 * cs.FAR_HALF) * res)
        return float(self.level[np.clip(iz, 0, res - 1),
                                np.clip(ix, 0, res - 1)])

    # copied from engine/shots_film.py:21-22 (ground)
    def ground(self, x, z):
        return max(self.height(x, z), self.water(x, z))

    # copied from engine/shots_film.py:24-25 (P)
    def P(self, x, z, agl):
        return (float(x), float(self.ground(x, z) + agl), float(z))

    # copied from engine/shots_film.py:31-35 (shore_south)
    def shore_south(self, x):
        z = -1000.0
        while self.height(x, z) < 0.0 and z > -2500:
            z -= 0.5
        return z

    # copied from engine/shots_film.py:37-41 (shore_x)
    def shore_x(self, z, side):
        x = axis_x(z)
        while self.height(x, z) < 0.0 and abs(x) < 3000:
            x += side * 0.5
        return x

    # copied from engine/shots_film.py:129-139 (head_peak)
    def head_peak(self):
        hf = self.hfar
        res = hf.shape[0]
        cell = 2 * cs.FAR_HALF / res
        i0, i1 = int((2800 + cs.FAR_HALF) / cell), int((4800 + cs.FAR_HALF) / cell)
        j0, j1 = int((-1200 + cs.FAR_HALF) / cell), int((1400 + cs.FAR_HALF) / cell)
        sub = hf[i0:i1, j0:j1]
        zz = (np.arange(i0, i1) + 0.5)[:, None] * cell - cs.FAR_HALF
        xx = (np.arange(j0, j1) + 0.5)[None, :] * cell - cs.FAR_HALF
        score = sub - 0.35 * np.hypot(xx - 100.0, zz - 3500.0)
        k = np.unravel_index(np.argmax(score), sub.shape)
        return (float((j0 + k[1] + 0.5) * cell - cs.FAR_HALF),
                float(sub[k]),
                float((i0 + k[0] + 0.5) * cell - cs.FAR_HALF))


# ---------------------------------------------------------------------------
# looks + grades, copied from engine/shots_film.py (base :119-120,
# vista :125, high :129, forest :205 + forest_grade :210, meadow :245,
# waterfall look :177, gold look :296-299 with _gt :295, gold_grade :305)
# ---------------------------------------------------------------------------
_BASE_LOOK = Look()
# copied from engine/shots_film.py:295 (final _gt) + :298 (tune=... expression)
_GOLD_TUNE = tuple({0: 0.3, 9: 1.4, 13: 0.45, 21: 1.0}.get(i, v)
                   for i, v in enumerate(_BASE_LOOK.tune))
_LOOKS = {
    "base": _BASE_LOOK,
    # copied from engine/shots_film.py:177 (wf_look)
    "waterfall": dataclasses.replace(_BASE_LOOK, sun_azimuth=228.0,
                                     sun_elevation=42.0),
    # copied from engine/shots_film.py:205 (forest look, contre-jour)
    "forest": dataclasses.replace(_BASE_LOOK, sun_azimuth=200.0,
                                  sun_elevation=55.0),
    # copied from engine/shots_film.py:245 (meadow look, side light)
    "meadow": dataclasses.replace(_BASE_LOOK, sun_azimuth=220.0,
                                  sun_elevation=58.0),
    # copied from engine/shots_film.py:296-299 (gold look)
    "gold": dataclasses.replace(_BASE_LOOK, sun_azimuth=298.0,
                                sun_elevation=26.0, cloud_cover=0.22,
                                haze=1.2, amb_boost=2.5, sky_boost=14.0,
                                tune=_GOLD_TUNE, wave_strength=1.5),
}


def _grade(name):
    # copied from engine/shots_film.py:120 (base), :125 (vista), :129 (high),
    # :210 (forest_grade), :305 (gold_grade)
    base = post.Grade()
    if name == "vista":
        return dataclasses.replace(base, black_max=0.12,
                                   contrast=base.contrast * 1.06)
    if name == "high":
        return dataclasses.replace(base, struct_gain=0.85)
    if name == "forest":
        return dataclasses.replace(base, base_compress=0.7, shadow_lift=0.6,
                                   saturation=1.0, contrast=1.2)
    if name == "gold":
        return dataclasses.replace(base, base_compress=0.65, shadow_lift=0.4,
                                   saturation=0.92, chroma_boost=0.08)
    return base


# ---------------------------------------------------------------------------
# VIEWPOINTS - annotated copies of the 9 film keyframe-0 camera setups
# (engine/shots_film.py build()). The film places most cameras through a
# neighbourhood SEARCH (search() + framing score); the panorama tool uses
# the NOMINAL search offset (0, 0), i.e. keyframe 0 of each shot.
# FIXME(port): the film's search is NOT reproduced - for a re-baked world
# the film may end up at different (searched) positions than these nominal
# keyframes; the pano therefore annotates the nominal recipe.
# ---------------------------------------------------------------------------
def _vp_forest(w):
    # copied from engine/shots_film.py:195 (trail), :199-203 (make_forest
    # at dx=dz=0, keyframe K(0.0, p0, toward(p0, 258, 1.5), 62))
    trail = np.asarray(w.features.get("trail", []), np.float64)
    i0 = int(np.clip(10, 1, len(trail) - 3))
    a = trail[i0]
    p0 = w.P(a[0], a[1], 1.9)
    return p0, toward(p0, 258, 1.5), []


def _vp_reveal(w):
    # copied from engine/shots_film.py:146-156 (make_reveal at dx=dz=0:
    # xr=-120, hero, a = P(xr-0.4, zb-4.8, 1.0), target (xr, a1+0.55, zb+2))
    xr = -120.0
    zs = w.shore_south(xr)
    zb = zs - 5.0
    gb = w.height(xr, zb)
    hero = (xr, gb + 0.6, zb, 2.0, 1.35, 0.95, 1.0, 0.25)
    a = w.P(xr - 0.4, zb - 4.8, 1.0)
    return a, (xr, a[1] + 0.55, zb + 2.0), [hero]


def _vp_skim(w):
    # copied from engine/shots_film.py:166-170 (skim K(0.0, ..., 64,
    # roll=-1.5)); roll carried in the VIEWPOINTS table
    z1 = -1150.0
    x1 = axis_x(z1) - 40.0
    return (x1, 1.7, z1), (x1 + 60.0, 22.0, z1 + 900.0), []


def _vp_waterfall(w):
    # copied from engine/shots_film.py:174-191 (foot/d/al, K0 y=2.2 ->
    # tgt y=13, wf_heroes table)
    wf = w.features["waterfall"]
    d = np.array(wf["dir"])
    al = np.array([-d[1], d[0]])
    foot = np.array([wf["x"], wf["z"]])
    p0 = foot + d * 95.0 + al * 42.0
    tgt = (foot[0] - d[0] * 2.0, 15.0, foot[1] - d[1] * 2.0)
    heroes = []
    for (v, u, yy, size, sx, sy, sz, yaw) in [
            (-24.0, 1.5, 2.0, 4.2, 1.4, 1.0, 1.1, 0.3),
            (-16.0, 2.5, 13.0, 3.4, 1.3, 0.8, 1.0, 1.1),
            (19.0, 1.0, 2.5, 5.0, 1.5, 0.9, 1.2, 2.0),
            (29.0, 3.0, 15.0, 4.0, 1.2, 1.0, 1.0, 0.7),
            (-0.5, -1.0, 0.0, 2.2, 1.3, 0.8, 1.0, 1.9),
            (13.0, 3.0, 24.0, 3.2, 1.4, 0.7, 1.1, 2.6)]:
        c = foot - d * u + al * v
        heroes.append((float(c[0]), float(yy), float(c[1]), size, sx, sy,
                       sz, yaw))
    return (float(p0[0]), 2.2, float(p0[1])), (float(tgt[0]), 13.0,
                                               float(tgt[2])), heroes


def _vp_stream(w):
    # copied from engine/shots_film.py:214-227 (make_stream at dx=dz=0:
    # km/mouth/up/d/side, c0 = mouth - d*48 + side*(-6), p0 y +1.5)
    st = w.features["stream"]
    km = next((k for k, q in enumerate(st) if q[2] > -0.5), 6)
    mouth = np.array(st[km][:2])
    up = np.array(st[min(km + 9, len(st) - 1)])
    d = mouth - np.array(st[max(km - 3, 0)][:2])
    d /= np.linalg.norm(d) + 1e-9
    side = np.array([-d[1], d[0]])
    c0 = mouth - d * 48.0 + side * (-6.0)
    p0 = (float(c0[0]), max(w.ground(float(c0[0]), float(c0[1])), 0.0) + 1.5,
          float(c0[1]))
    return p0, (float(up[0]), float(up[2]) + 18.0, float(up[1])), []


def _vp_meadow(w):
    # copied from engine/shots_film.py:241-243 (make_meadow dx=dz=0 K0)
    p0 = w.P(-2060.0, -40.0, 1.6)
    return p0, toward(p0, 133, -12.0), []


def _vp_flyover(w):
    # copied from engine/shots_film.py:250-253 (flyover K0)
    z7 = 1950.0
    p0 = (axis_x(z7) + 320.0, 620.0, z7)
    return p0, (axis_x(-1700.0), -60.0, -1700.0), []


def _vp_peaks(w):
    # copied from engine/shots_film.py:261-270 (make_peaks dx=dz=0 K0:
    # z0=-350, c0 x = axis_x(z0)+60, y0=700, t0 = hp - (350, -1750, 0))
    hp = np.array(w.head_peak())
    z0 = -350.0
    c0 = np.array([axis_x(z0) + 60.0, z0])
    return (float(c0[0]), 700.0, float(c0[1])), (float(hp[0]) - 350.0,
                                                 1750.0, float(hp[2])), []


def _vp_finale(w):
    # copied from engine/shots_film.py:310-312 (make_finale dx=dz=0 K0:
    # z9=300, x9 = shore_x(300, -1) - 2, P(x9, 300, 1.7), toward(p0, 35, 7))
    z9 = 300.0
    x9 = w.shore_x(z9, -1) - 2.0
    p0 = w.P(x9, z9, 1.7)
    return p0, toward(p0, 35, 7), []


# seeds/grade/look/target_luma per engine/shots_film.py (forest seed 4 tl 88
# :202-210; reveal 1 tl 118 :155-160 + :317; skim 2 tl 118 :168 + :317;
# waterfall 3 tl 112 (film.py:110 default) :186-191; stream 5 tl 112
# :228-230; meadow 6 tl 122 :242-244; flyover 7 tl 112 :251 + :317;
# peaks 8 tl 122 :269 + :317; finale 9 tl 106 :311 + :317).
# VIEWPOINTS order = film shot order (reveal first is index 1: the film's
# own S-list order).
VIEWPOINTS = [
    {"name": "forest", "resolve": _vp_forest, "look": "forest",
     "grade": "forest", "roll": 0.0, "seed": 4, "target_luma": 88.0},
    {"name": "reveal", "resolve": _vp_reveal, "look": "base",
     "grade": "vista", "roll": 0.0, "seed": 1, "target_luma": 118.0},
    {"name": "skim", "resolve": _vp_skim, "look": "base",
     "grade": "vista", "roll": -1.5, "seed": 2, "target_luma": 118.0},
    {"name": "waterfall", "resolve": _vp_waterfall, "look": "waterfall",
     "grade": "base", "roll": 0.0, "seed": 3, "target_luma": 112.0},
    {"name": "stream", "resolve": _vp_stream, "look": "base",
     "grade": "base", "roll": 0.0, "seed": 5, "target_luma": 112.0},
    {"name": "meadow", "resolve": _vp_meadow, "look": "meadow",
     "grade": "base", "roll": 0.0, "seed": 6, "target_luma": 122.0},
    {"name": "flyover", "resolve": _vp_flyover, "look": "base",
     "grade": "high", "roll": 0.0, "seed": 7, "target_luma": 112.0},
    {"name": "peaks", "resolve": _vp_peaks, "look": "base",
     "grade": "high", "roll": 0.0, "seed": 8, "target_luma": 122.0},
    {"name": "finale", "resolve": _vp_finale, "look": "gold",
     "grade": "gold", "roll": 0.0, "seed": 9, "target_luma": 106.0},
]


def resolve_viewpoint(world, idx):
    vp = VIEWPOINTS[idx]
    pos, tgt, heroes = vp["resolve"](world)
    return (np.asarray(pos, np.float64), np.asarray(tgt, np.float64),
            heroes)


# ---------------------------------------------------------------------------
# env knobs (copied from engine/render.py:400-412, ALPINE_LAB / ALPINE_W2K)
# ---------------------------------------------------------------------------
def _env_vec(name, n):
    v = np.zeros(n, np.float32)
    for kv in filter(None, os.environ.get(name, "").split(",")):
        i, val = kv.split("=")
        v[int(i)] = float(val)
    return v


# ---------------------------------------------------------------------------
# per-frame uniform fields (copied from engine/render.py:311-418,
# _look_uniforms; the p.set(...) calls become dict entries)
# ---------------------------------------------------------------------------
def _look_fields(look, t, features, heroes):
    out = {}
    out["uSunDir"] = tuple(look.sun_dir())
    out["uShadowDir"] = tuple(map(float, look.shadow_dir()))
    out["uSunIntensity"] = look.sun_intensity
    out["uHaze"] = look.haze
    out["uCloudCover"] = look.cloud_cover
    out["uWind"] = tuple(look.wind)
    out["uTime"] = float(t)
    ww = np.asarray(look.water_wind, np.float64)
    out["uWaterWind"] = tuple(map(float, ww / np.linalg.norm(ww)))
    out["uWaveStrength"] = look.wave_strength
    out["uWaterIns"] = tuple(look.water_ins)
    f = features
    if f:
        wf = f["waterfall"]
        out["uWF"] = (wf["x"], wf["z"], wf["height"], wf["half"])
        out["uWFDir"] = tuple(wf["dir"])
        out["uWFNotch"] = tuple(wf["notch"])
        fk = np.zeros((48, 4), np.float32)
        fi = np.zeros((2, 4), np.float32)
        fb = np.zeros((4, 4), np.float32)
        for k, fl in enumerate(f.get("falls", [])[:2]):
            kn = np.asarray(fl["knots"], np.float32)[:24]
            fk[k * 24:k * 24 + len(kn)] = kn
            fi[k] = (fl["top"], fl["vol"], 0.0, 0.0)
            wv = kn[:, 2] * 1.9 + 0.8
            b0 = kn[-1]
            MK = 1.4
            mr = 2.0 * np.sqrt(4.0 + 3.0 * b0[2]) * MK
            MXO = 20.5
            fb[2 * k] = (min(kn[:, 0].min() - 3.6, b0[0] - MXO), -0.8,
                         min((kn[:, 1] - wv).min(), b0[1] - mr), 0.0)
            fb[2 * k + 1] = (kn[:, 0].max() + 0.6,
                             max(fl["top"] + 0.7,
                                 (7.0 if k == 0 else 5.0) * MK),
                             max((kn[:, 1] + wv).max(), b0[1] + mr), 0.0)
        out["uFallK"] = fk.ravel()
        out["uFallInfo"] = fi.ravel()
        out["uFallBox"] = fb.ravel()
        out["uFallsOn"] = 1 if f.get("falls") else 0
        trl = np.zeros((32, 4), np.float32)
        tl = np.asarray(f.get("trail", []), np.float32)[:32]
        if len(tl):
            trl[:len(tl), :tl.shape[1]] = tl
        out["uTrail"] = trl.ravel()
        out["uTrailN"] = int(len(tl))
        sa = np.asarray(f.get("stream", []), np.float32).reshape(-1, 4)[:96]
        stu = np.zeros((96, 4), np.float32)
        stu[:len(sa)] = sa
        out["uStream"] = stu.ravel()
        out["uStreamN"] = int(len(sa))
        ss = np.zeros(96, np.float32)
        if len(sa) > 1:
            ss[1:len(sa)] = np.cumsum(
                np.linalg.norm(np.diff(sa[:, :2], axis=0), axis=1))
        out["uStreamS"] = ss
        if len(sa):
            m = float(sa[:, 3].max()) * 2.6 + 4.0
            out["uStreamBox"] = (sa[:, 0].min() - m, sa[:, 1].min() - m,
                                 sa[:, 0].max() + m, sa[:, 1].max() + m)
            mo = (0.0, 0.0, 0.0, 0.0)
            for k in range(1, len(sa)):
                if sa[k, 2] > 0.0 >= sa[k - 1, 2]:
                    s_ = float(-sa[k - 1, 2] / (sa[k, 2] - sa[k - 1, 2]))
                    mp = sa[k - 1, :2] + (sa[k, :2] - sa[k - 1, :2]) * s_
                    dn = sa[k - 1, :2] - sa[k, :2]
                    dn = dn / (np.linalg.norm(dn) + 1e-9)
                    mo = (float(mp[0]), float(mp[1]), float(dn[0]),
                          float(dn[1]))
                    break
        else:
            out["uStreamBox"] = (1e9, 1e9, -1e9, -1e9)
            mo = (0.0, 0.0, 0.0, 0.0)
        out["uStreamMouth"] = mo
        cb = f.get("stream_corr_box")
        out["uStreamCorrBox"] = ((float(cb[0]), float(cb[1]),
                                  1.0 / float(cb[2]), 1.0 / float(cb[3]))
                                 if cb else (0.0, 0.0, 0.0, 0.0))
    else:
        out["uFallsOn"] = 0
    # heroes: engine render.py:432-437 sets uHeroA/uHeroB/uHeroN from
    # R.heroes (same (x,y,z,s | sx,sy,sz,yaw) rows the shots build)
    ha = np.zeros((6, 4), np.float32)
    hb = np.zeros((6, 4), np.float32)
    for i, hr in enumerate(heroes[:6]):
        ha[i] = hr[:4]
        hb[i] = hr[4:8]
    out["uHeroA"] = ha.ravel()
    out["uHeroB"] = hb.ravel()
    out["uHeroN"] = len(heroes[:6])
    out["uSkyBoost"] = look.sky_boost
    out["uAmbBoost"] = look.amb_boost
    out["uSkyDesat"] = look.sky_desat
    out["uLab"] = _env_vec("ALPINE_LAB", 16)
    out["uW2K"] = _env_vec("ALPINE_W2K", 32)
    tu = np.zeros(24, np.float32)
    tu[:len(look.tune)] = look.tune
    out["uTune"] = tu
    return out


def scene_extra_fields(w, h, sun, sky, hor, eye, camrot, jitter, sample):
    """Per-sample scene fields outside _look_uniforms.

    uSunCol/uSkyAmb/uSkyHorizon copied from engine/render.py:439-441,
    uRes/uDebug from render.py:442-443, uJitter/uSample/uCamPos/uCamRot
    from render.py:453-466. FIXME(port): uTanHalfFov / uPixAng /
    uAperture / uFocusDist are camera-model members of the shared pass
    UBO. In panorama mode the PANO shader branches never read them (the
    per-texel footprint comes from panoFoot and the ray is the pixel
    direction; there is no lens). They are still written because the
    members exist in the committed UBO layout; values are inert (1.0 /
    0.0), which is NOT a numeric deviation - the shader does not touch
    them. The probe program (no PANO branch) reads NEITHER uPixAng nor
    uRes - probe.frag contains zero references to both (review_p4.md N2)
    - so the written member is dead in every program vr.py dispatches.
    """
    return {
        "uSunCol": tuple(map(float, sun)),
        "uSkyAmb": tuple(map(float, sky)),
        "uSkyHorizon": tuple(map(float, hor)),
        "uRes": (float(w), float(h)),
        "uDebug": 0,
        "uJitter": tuple(jitter),
        "uSample": int(sample),
        "uCamPos": tuple(map(float, eye)),
        "uCamRot": camrot,
        "uTanHalfFov": 1.0,
        "uPixAng": 1.0,
        "uAperture": 0.0,
        "uFocusDist": 0.0,
    }


# ---------------------------------------------------------------------------
# manifest views
# ---------------------------------------------------------------------------
def load_manifest():
    doc = json.loads(MANIFEST.read_text())
    return {p["name"]: p for p in doc["programs"]}


def ubo_member_names(prog_entry):
    ub = prog_entry.get("uniform_block") or {}
    return [m["name"] for m in ub.get("members", [])]


def sampler_binding_names(prog_entry):
    return [b["name"] for b in prog_entry.get("bindings", [])
            if "sampler" in str(b.get("kind", ""))]


# ---------------------------------------------------------------------------
# dry-run plan
#
# gate_p4.sh() captures only stdout+stderr[:900] and parses the text between
# the first '{' and the last '}' inside that window. The machine-readable
# plan is therefore ONE compact JSON document that must close well before
# char 900, followed by a human-readable detail section that contains no
# curly braces at all (bracket-free too, to stay honest about the shape).
# ---------------------------------------------------------------------------
def _scene_field_universe(feats, heroes):
    names = set(_look_fields(_LOOKS["base"], 0.0, feats, heroes))
    names.update(scene_extra_fields(1, 1, (0, 0, 0), (0, 0, 0), (0, 0, 0),
                                    (0, 0, 0), None, (0.0, 0.0), 0))
    return names


def _try_resolve(args):
    """Resolve the viewpoint position, or explain why the plan cannot."""
    missing = cs.missing()
    if missing:
        return None, {"cache_missing": missing, "features_available": False,
                      "resolve_note": "cache incomplete: "
                                      + ",".join(missing)}
    try:
        world = PanoWorld()
        feats = world.features
        pos, tgt, heroes = resolve_viewpoint(world, args.viewpoint)
        note = None if feats is not None else "features.json absent"
        return (pos, tgt, heroes, feats), {
            "cache_missing": [], "features_available": feats is not None,
            "resolve_note": note}
    except Exception as exc:
        return None, {"cache_missing": [], "features_available": False,
                      "resolve_note": type(exc).__name__ + ": " + str(exc)}


def _plan(args, w, h):
    """(summary_json, detail_lines). summary is one compact JSON object."""
    rv, cache = _try_resolve(args)
    pos = tgt = None
    heroes, feats = [], None
    if rv:
        pos, tgt, heroes, feats = rv
    universe = _scene_field_universe(feats, heroes)
    progs = load_manifest()
    names = pano_program_names() + ["probe"]
    scene_ubo = None
    probe_row = None
    sams = []
    skipped_scene = None
    detail = []
    for name in names:
        p = progs[name]
        ub = p.get("uniform_block") or {}
        members = ubo_member_names(p)
        uni = universe | ({"uProbePos"} if name == "probe" else set())
        fset = sorted(m for m in members if m in uni)
        fskip = sorted(m for m in members if m not in uni)
        snames = sampler_binding_names(p)
        row = [ub.get("set"), ub.get("binding"), ub.get("size_bytes"),
               len(members), len(snames)]
        if name == "probe":
            probe_row = ["probe"] + row
        else:
            sams.append(row[4])
            if scene_ubo is None:
                scene_ubo = row[:4]
                skipped_scene = fskip
        detail.append("program " + name + " ubo set=" + str(row[0])
                      + " binding=" + str(row[1]) + " bytes="
                      + str(row[2]) + " members=" + str(row[3])
                      + " samplers=" + str(row[4]))
        detail.append("  fields_set " + str(len(fset)) + ": "
                      + ",".join(fset))
        detail.append("  fields_skipped " + str(len(fskip)) + ": "
                      + ",".join(fskip))
        detail.append("  samplers: " + ",".join(snames))
    eyes = []
    if args.stereo == "mono":
        eyes = [["mono", None if pos is None
                 else [round(float(v), 2) for v in pos]]]
    elif pos is not None:
        fwd = tgt - pos
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross([0.0, 1.0, 0.0], fwd)
        right = right / np.linalg.norm(right)
        o_l, o_r, _dirs = pano.stereo_pair(pos, np.zeros((1, 3), np.float32),
                                           right)
        eyes = [["left", [round(float(v), 3) for v in o_l]],
                ["right", [round(float(v), 3) for v in o_r]]]
    else:
        eyes = [["left", None], ["right", None]]
    fp = pano.texel_footprint(w, h)
    vp = VIEWPOINTS[args.viewpoint]
    out_path = str(Path(args.out)) if args.out else str(
        (ROOT / "engine_vr" / "out" / ("pano_" + str(args.viewpoint)
                                       + "_" + vp["name"] + "_"
                                       + args.stereo + "_"
                                       + str(w) + "x" + str(h)
                                       + ".png")).relative_to(ROOT))
    summary = {
        "mode": "dry-run",
        "cache": [str(cs.cache_dir().relative_to(ROOT))
                  if cs.cache_dir().is_relative_to(ROOT)
                  else str(cs.cache_dir()),
                  cache["cache_missing"], cache["features_available"]],
        "viewpoint": [args.viewpoint, vp["name"], vp["look"], vp["grade"],
                      vp["seed"], vp["target_luma"], vp["roll"],
                      None if pos is None
                      else [round(float(v), 2) for v in pos],
                      None if tgt is None
                      else [round(float(v), 2) for v in tgt],
                      len(heroes)],
        "stereo": args.stereo,
        "eyes": eyes,
        "contract": "shared dirs; origins = head +/- 0.032 m along "
                    "camera-right; identical per-pixel seed",
        "res": [w, h, (w + 3) // 4, (h + 3) // 4],
        "tile": args.tile, "spp": args.spp,
        "scene_ubo": scene_ubo, "scene_samplers": sams,
        "probe": probe_row,
        "fields_skipped_scene": skipped_scene,
        "footprint_rad": [round(float(fp[0]), 6),
                          round(float(fp[h // 2]), 6),
                          round(float(fp[-1]), 6)],
    }
    detail.insert(0, "cache " + str(cs.cache_dir()) + " missing "
                  + (",".join(cache["cache_missing"]) or "none")
                  + " note " + str(cache["resolve_note"]))
    detail.append("out " + out_path)
    detail.append("post engine/post.py: e = suggest_exposure(stitched, "
                  "target_luma=viewpoint, g=grade); img = finish(stitched, "
                  "replace(grade, exposure=e), seed=viewpoint seed)")
    detail.append("same_seed mono and both SBS eyes share host, world "
                  "textures, bakes, light probe, tile order, Halton jitter "
                  "sequence and post seed; only uCamPos differs")
    detail.append("vulkan Host.open() unreachable in dry-run: passes_vr / "
                  "gl_vr / bake_vr are imported only inside _render()")
    return json.dumps(summary, separators=(",", ":")), detail


# ---------------------------------------------------------------------------
# GPU render path. The three imports below are the ONLY entry points to the
# Vulkan layer in this file; --dry-run never calls _render().
# ---------------------------------------------------------------------------
def _upload_noise3(host, pv, n3):
    """(128,128,128,4) uint8 (cache order) -> 3D sampled Tex, LINEAR repeat.

    Copied from engine/render.py:134-138 (n3.transpose(2,1,0,3), LINEAR,
    repeat x/y/z) executed on the gl_vr.Host.upload_2d staging/barrier
    pattern (gl_vr.py:838-899). FIXME(port): upload_2d's depth>1 creates a
    2D-ARRAY image; a texture3d needs VK_IMAGE_TYPE_3D, so this copy keeps
    layerCount=1 and extent depth=128 via make_image(dims=3).
    """
    from engine_vr import gl_vr as _gl_vr
    vkv = _gl_vr.vk()
    a = np.ascontiguousarray(n3.transpose(2, 1, 0, 3), np.uint8)
    d, hh, ww, c = a.shape
    img = host.make_image(ww, hh, "VK_FORMAT_R8G8B8A8_UNORM",
                          vkv.VK_IMAGE_USAGE_SAMPLED_BIT
                          | vkv.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                          vkv.VK_IMAGE_LAYOUT_UNDEFINED, depth=d, dims=3)
    data = a.tobytes()
    stg = host.staging(len(data))
    stg.write(data)
    cmd = host.begin()
    host.barrier(cmd, vkv.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 vkv.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                 vkv.VK_ACCESS_TRANSFER_WRITE_BIT,
                 [(img, vkv.VK_IMAGE_LAYOUT_UNDEFINED,
                   vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)])
    img.layout = vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    vkv.vkCmdCopyBufferToImage(cmd, stg.buf, img.image, img.layout, 1, [
        vkv.VkBufferImageCopy(
            imageSubresource=vkv.VkImageSubresourceLayers(
                aspectMask=vkv.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1,
                mipLevel=0),
            imageExtent=vkv.VkExtent3D(width=ww, height=hh, depth=d))])
    host.barrier(cmd, vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
                 (vkv.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                  | vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
                 vkv.VK_ACCESS_TRANSFER_WRITE_BIT,
                 vkv.VK_ACCESS_SHADER_READ_BIT,
                 [(img, img.layout,
                   vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)])
    img.layout = vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    host.end()
    return pv.Tex(img, "linear", "repeat")


def upload_footstrip(pv, w, h, foot_src=None):
    """Per-row footprint strip (h, 1) -> R32_SFLOAT sampled Tex.

    review_p4.md C1: pano.texel_footprint() returns (h,) - the gate-checked
    T2 contract - but passes_vr.upload expects (h, w[, c]) and computes its
    component count as "1 if a.ndim == 2 else int(a.shape[2])"
    (engine_vr/passes_vr.py:165), so a bare 1-D array died with IndexError
    before the first vkCmd. reshape(-1, 1) yields a width-1 x height-H
    image; the world.glsl VR_PANO block reads it with
    texelFetch(panoFoot, ivec2(0, row), 0), which is exactly this layout.
    Kept as a named helper so engine_vr/tmp/test_p4_fixes.py can exercise
    the real upload path on CPU (fake host, no GPU).

    foot_src=(W,H) - PHASE_6 debug hook (--debug-foot-res): take the footprint
    VALUES of a WxH render while actually rendering wxh, i.e. decouple the
    per-texel epsilon (gPixAng) from the image size. At 2:1 the strip is
    constant pi/H, so this is exactly "march as if the image were H rows
    tall". Debug-only: default None keeps the production behaviour byte-exact.
    """
    sw, sh = foot_src if foot_src else (w, h)
    strip = pano.texel_footprint(sw, sh)
    if strip.shape[0] != h:          # synthetic fine strip: replicate to h rows
        strip = np.repeat(strip[:1], h) if strip.shape[0] > h \
            else np.resize(strip, h)
    return pv.upload(strip.reshape(-1, 1), filt="nearest")


def _clear_images(host, imgs):
    """One-shot full-image zero clear (UNDEFINED/CURRENT -> COLOR_ATTACHMENT).

    The cleared render-pass variant clears only the draw rect (P3 lesson),
    so accumulation starts from an explicit vkCmdClearColorImage over the
    whole image - the equivalent of the engine clearing its accumulation
    fbo once per frame (engine/render.py:451-452).
    """
    from engine_vr import gl_vr as _gl_vr
    vkv = _gl_vr.vk()
    cmd = host.begin()
    host.barrier(cmd, vkv.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 vkv.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                 vkv.VK_ACCESS_TRANSFER_WRITE_BIT,
                 [(im, im.layout, vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                  for im in imgs])
    for im in imgs:
        im.layout = vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    # vkCmdClearColorImage takes VkClearColorValue*, NOT the VkClearValue
    # wrapper that render passes use (first-run fix: cffi TypeError otherwise)
    clr = vkv.ffi.new("VkClearColorValue[1]")
    clr[0].float32 = [0.0, 0.0, 0.0, 0.0]
    rng = [vkv.VkImageSubresourceRange(
        aspectMask=vkv.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1,
        layerCount=1)]
    for im in imgs:
        vkv.vkCmdClearColorImage(cmd, im.image,
                                 vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 clr, 1, rng)
    host.barrier(cmd, vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
                 vkv.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                 vkv.VK_ACCESS_TRANSFER_WRITE_BIT,
                 vkv.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                 [(im, im.layout,
                   vkv.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                  for im in imgs])
    for im in imgs:
        im.layout = vkv.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    host.end()


def _render(args, w, h, eye_tags):
    """Run the render chain. Returns {tag: (h, w, 3) float32 hdr, top row
    = lat +90 (display orientation, after the copied render.py:505 flip).

    Mono and SBS both call this ONCE per eye over shared host / world
    textures / bakes / light probe / sample order: the only per-eye input
    is uCamPos. That is the same-seed contract (RUNBOOK 6).
    """
    from engine_vr import passes_vr as pv
    from engine_vr import bake_vr

    missing = cs.missing()
    if missing:
        raise RuntimeError("cache incomplete: " + ",".join(missing)
                           + " (set --cache or run the P2 bake first)")
    vp = VIEWPOINTS[args.viewpoint]
    world = PanoWorld()
    if world.features is None:
        raise RuntimeError("features.json missing in " + str(cs.cache_dir()))
    pos, tgt, heroes = resolve_viewpoint(world, args.viewpoint)
    look = _LOOKS[vp["look"]]
    host = pv.open_host(cache_dir=None, res=w, tile=args.tile, spp=args.spp)
    try:
        # ---- world textures (copied from engine/render.py:122-159) ----
        tex = {}
        tex["uHFar"] = pv.upload(np.load(cs.path("h_far.npy")))
        tex["uHNear"] = pv.upload(np.load(cs.path("h_near.npy")))
        tex["uAOFar"] = pv.upload(np.load(cs.path("ao_far.npy")).astype(
            np.float32))
        tex["uAONear"] = pv.upload(np.load(cs.path("ao_near.npy")).astype(
            np.float32))
        tex["uForest"] = pv.upload(np.load(cs.path("forest.npy")))
        tex["uWaterLvl"] = pv.upload(np.load(cs.path("water_level.npy")),
                                     filt="linear")
        tex["uNoise3"] = _upload_noise3(host, pv,
                                        np.load(cs.path("noise3d.npy")))
        wln = cs.path("water_level_near.npy")
        tex["uWaterLvlNear"] = pv.upload(
            np.load(wln) if wln.exists()
            else np.full((4, 4), -1e4, np.float32))
        stn = cs.path("stream_near.npy")
        tex["uStreamNear"] = pv.upload(
            np.load(stn).astype(np.float32) if stn.exists()
            else np.zeros((4, 4, 4), np.float32))
        stc = cs.path("stream_corr.npy")
        tex["uStreamCorr"] = pv.upload(
            np.load(stc).astype(np.float32) if stc.exists()
            else np.full((4, 4, 4), 99.0, np.float32))
        tex["_zero4"] = pv.upload(np.zeros((4, 4, 4), np.float32))
        # shadows then trees, engine order (bakes run BEFORE render in the
        # engine too; the tree bake binds the real uShNear/uShFar)
        tex.update(bake_vr.bake_shadows(tex, look.shadow_dir()))
        tp = cs.path("trees.npy")
        trees = (np.load(tp) if tp.exists()
                 else bake_vr.ensure_trees(tex))
        tex["uTrees"] = pv.upload(trees, filt="nearest")
        tex["uTreeTop"] = pv.upload(tree_tops(trees), filt="nearest")

        # ---- scene programs + PANO samplers ----
        progs = load_manifest()
        names = pano_program_names()
        built = {n: host.build(n) for n in names}
        built["probe"] = host.build("probe")
        # --debug-foot-res (PHASE_6): march with another res's epsilon
        fp_tex = upload_footstrip(pv, w, h,
                                  getattr(args, "debug_foot_src", None))
        tex["panoFoot"] = fp_tex

        # ---- targets ----
        # gbuf finalLayout TRANSFER_SRC (phase2.md risk note: fallback is
        # COLOR_ATTACHMENT + an explicit barrier before readback)
        # FIXME(p7): BUG A fix - the GL-era 2-attachment G-buffer is
        # unrepresentable on MoltenVK 1.4.2 (only the highest-location written
        # fragment output lands, in framebuffer slot 0; lower locations are
        # discarded - probes engine_vr/tmp/p7probe2/3/5/7.py; GL-era truth:
        # engine/shaders/scene.frag:280-281 + engine/render.py:484 reading
        # attachment 0 channel 1 for the kind). gt0 = kind payload (the trace
        # program's single fragColor output), gt1 = h.info written by the
        # trace's uGInfo image store (engine_vr/shaders/scene.frag VR_PANO
        # block). gt1 rests in GENERAL for the whole frame - a valid
        # imageLayout for both the store and the later uGBuf1 sampler
        # (write_descriptors keeps GENERAL, gl_vr.py:1199-1214) - and is
        # never a color attachment, never read back, never transitioned.
        gt0 = pv.make_target(None, width=w, height=h)
        from engine_vr import gl_vr as _gl_vr   # FIXME(p7)
        vku = _gl_vr.vk()   # FIXME(p7)
        gt1 = host.make_image(w, h, pv.TARGET_FORMAT,   # FIXME(p7)
                              vku.VK_IMAGE_USAGE_SAMPLED_BIT
                              | vku.VK_IMAGE_USAGE_STORAGE_BIT,
                              vku.VK_IMAGE_LAYOUT_GENERAL)
        acc = pv.make_target(None, width=w, height=h)
        _clear_images(host, [acc.images[0]])
        ct = pv.make_target(None, fmt_name="VK_FORMAT_R32_SFLOAT",
                            final_layout="VK_IMAGE_LAYOUT_SHADER_READ_ONLY_"
                                         "OPTIMAL",
                            width=(w + 3) // 4, height=(h + 3) // 4)
        _clear_images(host, [ct.images[0]])
        pv.to_sampler_layout([ct.images[0]])
        pv.to_sampler_layout([gt0.images[0]])   # FIXME(p7): gt1 stays GENERAL

        # ---- descriptors (freeze layouts now; write_descriptors freezes
        # the CURRENT layout of every bound image - phase-2 contract) ----
        base = {k: v.as_sampler() for k, v in tex.items()
                if k != "_zero4"}

        def bind(name, extra=None):
            want = set(sampler_binding_names(progs[name]))
            d = {k: v for k, v in base.items() if k in want}
            if extra:
                d.update(extra)
            return d

        host.write_descriptors(built[PANO_CONE], samplers=bind(PANO_CONE))
        # FIXME(p7): the trace program writes h.info to gt1 via uGInfo
        host.write_descriptors(built[PANO_TRACE], samplers=bind(
            PANO_TRACE, {"uCone": (ct.images[0], "nearest", "clamp")}),
            storage_images={"uGInfo": gt1})
        for k in range(8):
            host.write_descriptors(built[names[2 + k]], samplers=bind(
                names[2 + k],
                # FIXME(p7): all 8 kind programs read the kind payload from
                # gt0 and the info payload from gt1 (writer/reader table in
                # port/reports/phase7.md); uGBuf1 keeps imageLayout GENERAL -
                # gt1 is never transitioned, and the queue-FIFO order
                # trace-write submission -> kind-read submission is the
                # execution order (vr.py tile loop, gl_vr.draw submits).
                {"uGBuf0": (gt0.images[0], "nearest", "clamp"),
                 "uGBuf1": (gt1, "nearest", "clamp")}))
        host.write_descriptors(built["probe"], samplers=bind("probe"))

        # ---- light probe ONCE (copied from engine/render.py:420-427;
        # the engine probes a single camera position per frame - the film
        # passes cams[0].pos. FIXME(port): the pano takes that head-centre
        # probe for BOTH eyes; the 0.032 m eye parallax of the sky probe is
        lookf = _look_fields(look, 0.0, world.features, heroes)
        pm = set(ubo_member_names(progs["probe"]))
        # NOTE(first-run): NO host.begin() here - host.draw() below is a
        # one-shot session (draw(cmd=None) owns begin/end, gl_vr.py:1281);
        # wrapping it in an external session raised "command session
        # already active". Uniform writes need no command buffer.
        pf = dict(lookf)
        pf["uProbePos"] = tuple(map(float, pos))
        pv.write_uniforms(built["probe"], {k: v for k, v in pf.items()
                                           if k in pm})
        ptar = host.make_target(3, 1, 1, pv.TARGET_FORMAT,
                                "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL")
        host.draw(built["probe"], ptar, rect=(0, 0, 3, 1), cleared=True)
        pa = pv.read_image(ptar.images[0])[0]
        sun, sky, hor = pa[0][:3], pa[1][:3], pa[2][:3]

        # ---- eyes: shared everything, uCamPos differs (pano.stereo_pair) ----
        fwd = tgt - pos
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross([0.0, 1.0, 0.0], fwd)
        right = right / np.linalg.norm(right)
        dirs_shared = np.zeros((1, 3), np.float32)   # SHARED dirs (value
        # inert here: the GPU builds dirs from the PANO branches; the
        # contract still routes origins through pano.stereo_pair)
        origins = {"mono": pos}
        if "sbs" in eye_tags or len(eye_tags) == 2:
            o_l, o_r, _d = pano.stereo_pair(pos, dirs_shared, right)
            origins["left"], origins["right"] = o_l, o_r
        camrot = basis(pos, tgt, vp["roll"]).astype(np.float32)

        n = max(1, int(args.spp))
        acc_img = acc.images[0]
        sample_fields = {}
        for s in range(n):
            # copied from engine/render.py:453-457 (halton jitter)
            jx, jy = ((halton(s + 1, 2) - 0.5, halton(s + 1, 3) - 0.5)
                      if n > 1 else (0.0, 0.0))
            f = dict(lookf)
            f.update(scene_extra_fields(w, h, sun, sky, hor, pos, camrot,
                                        (jx, jy), s))
            sample_fields[s] = f

        eye_hdr = {}
        for tag in eye_tags:
            for s in range(n):
                f = dict(sample_fields[s])
                f["uCamPos"] = tuple(map(float, origins[tag]))
                for pname in names:
                    pv.write_uniforms(
                        built[pname],
                        {k: v for k, v in f.items()
                         if k in ubo_member_names(progs[pname])})
                for y in range(0, h, args.tile):
                    for x in range(0, w, args.tile):
                        tw = min(args.tile, w - x)
                        th = min(args.tile, h - y)
                        cmd = host.begin()
                        # cone at quarter res, copied from
                        # engine/render.py:472-479
                        host.draw(built[PANO_CONE], ct,
                                  rect=(x // 4, y // 4, (tw + 3) // 4,
                                        (th + 3) // 4), cmd=cmd)
                        # FIXME(p7): the trace's single fragColor output
                        # (kind payload) goes to gt0 slot 0; uGInfo stores
                        # h.info into gt1 inside the same draw.
                        host.draw(built[PANO_TRACE], gt0, rect=(x, y, tw,
                                                                th),
                                  cmd=cmd)
                        host.end()
                        # kind bits: gbuf attachment 0 component 1
                        # (engine/render.py:484 reads components=2 then
                        # takes [1::2])
                        # FIXME(p7): kind bits from gt0 (was gbuf.images[0])
                        kb = pv.read_image(gt0.images[0],
                                           rect=(x, y, tw, th))
                        kb = kb[..., 1]
                        hit = np.unique(np.clip(kb.astype(np.int32), 0, 7))
                        if hit.size:
                            # FIXME(p7): only gt0 transitions; gt1 GENERAL
                            pv.to_sampler_layout([gt0.images[0]])
                            cmd = host.begin()
                            for k in hit:
                                # blend ONE,ONE accumulate, copied from
                                # engine/render.py:498-503
                                host.draw(built[names[2 + int(k)]], acc,
                                          rect=(x, y, tw, th), blend=True,
                                          cmd=cmd)
                            host.end()
                pv.to_sampler_layout([gt0.images[0]])   # FIXME(p7)
            a = pv.read_image(acc_img)[..., :3] / float(n)
            # display flip, copied from engine/render.py:505 ([::-1]):
            # GL row 0 (lat +90 readback bottom) -> image row 0 (top)
            eye_hdr[tag] = np.ascontiguousarray(a[::-1])
        return eye_hdr
    finally:
        pv.close_host()


def _post_save(stitched, vp, out_path):
    """Grade + tone-map + PNG.

    Copied from film.py:110 + 159-160 (suggest_exposure on the shot frame,
    finish with replace(grade, exposure=...)). FIXME(port): the engine
    stills grain with the frame index as the post seed (film.py:134,
    seed=f); a panorama has no frame index, so the shot's own seed field
    is used - identical for mono and for both SBS eyes (the seed half of
    the stereo contract). One pass over the STITCHED sbs frame keeps the
    grain pattern consistent across the seam.
    """
    g = _grade(vp["grade"])
    e = post.suggest_exposure(stitched, target_luma=vp["target_luma"], g=g)
    img = post.finish(stitched, dataclasses.replace(g, exposure=e),
                      seed=vp["seed"])
    from PIL import Image
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(p)
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="vr.py",
        description="MoltenVK equirectangular panorama renderer (Phase 4)")
    ap.add_argument("--viewpoint", type=int, default=0, metavar="N",
                    help="VIEWPOINTS index 0..8 (default 0)")
    ap.add_argument("--stereo", choices=["mono", "sbs"], default="mono")
    ap.add_argument("--res", default=None, metavar="WxH",
                    help="per-frame resolution; default 8192x4096 mono, "
                         "8192x2048 per-eye sbs; ratio must be 2:1 or "
                         "wider (W >= 2H; see review_p4.md M1)")
    ap.add_argument("--spp", type=int, default=4)
    ap.add_argument("--tile", type=int, default=192)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None,
                    help="cache directory (sets ALPINE_CACHE for the "
                         "whole process)")
    ap.add_argument("--debug-foot-res", default=None, metavar="WxH",
                    help="PHASE_6 debug only: sample the panoFoot footprint "
                         "strip of a WxH render while rendering --res, i.e. "
                         "march with another resolution's per-texel epsilon "
                         "(gPixAng). Lets a small render reproduce / prove a "
                         "step-budget (runaway-march) failure of a large one "
                         "inside the GPU time budget. Default: off "
                         "(production behaviour unchanged).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the render plan as JSON and exit "
                         "(never opens the GPU)")
    args = ap.parse_args(argv)
    if args.cache:
        cs.set_cache_dir(args.cache)
    if not 0 <= args.viewpoint <= 8:
        ap.error("--viewpoint must be 0..8")
    if args.res:
        w, h = (int(v) for v in args.res.lower().split("x"))
        if w < 2 * h:
            # review_p4.md M1: below 2:1 the RUNBOOK 6 footprint
            # max(dLat, dLon*cos lat) varies with latitude, and the
            # quarter-res cone pass can only look the strip up in 4-row
            # bands (its uPixAng is row-scaled by 4.0 in the VR_PANO
            # branch), so cone/trace LOD terms are only band-constant
            # there. Until the device probes cover non-spec ratios the
            # supported set is 2:1 or wider (e.g. 8192x4096 mono,
            # 8192x2048 per-eye sbs).
            ap.error("--res WxH needs W >= 2H (2:1 or wider); ratios with "
                     "W < 2H are unsupported (port/reports/review_p4.md "
                     "M1), e.g. 8192x4096 mono or 8192x2048 per-eye sbs")
    elif args.stereo == "sbs":
        w, h = 8192, 2048
    else:
        w, h = 8192, 4096
    foot_src = None
    if getattr(args, "debug_foot_res", None):
        fw, fh = (int(v) for v in args.debug_foot_res.lower().split("x"))
        if fw < 2 * fh:
            ap.error("--debug-foot-res WxH needs W >= 2H (same rule as --res)")
        foot_src = (fw, fh)
        args.debug_foot_src = foot_src
    vp = VIEWPOINTS[args.viewpoint]
    if args.dry_run:
        summary, detail = _plan(args, w, h)
        print(summary)
        print("PLAN DETAILS")
        for line in detail:
            print(line)
        return 0
    eye_hdr = _render(args, w, h, ["mono"] if args.stereo == "mono"
                      else ["left", "right"])
    if args.stereo == "sbs":
        stitched = np.concatenate([eye_hdr["left"], eye_hdr["right"]],
                                  axis=1)
    else:
        stitched = eye_hdr["mono"]
    out_path = args.out or str(
        ROOT / "engine_vr" / "out" / ("pano_" + str(args.viewpoint)
                                      + "_" + vp["name"] + "_"
                                      + args.stereo + "_"
                                      + str(w) + "x" + str(h) + ".png"))
    e = _post_save(stitched, vp, out_path)
    print(json.dumps({"out": out_path, "stereo": args.stereo,
                      "exposure": e, "viewpoint": vp["name"]}))
    return 0


if __name__ == "__main__":
    main()
