"""Smooth keyframed camera paths (non-uniform Catmull-Rom / Hermite) with easing and gentle drone drift."""
import math

import numpy as np

from render import Camera


def _ease(x, kind):
    x = min(max(x, 0.0), 1.0)
    if kind == "linear":
        return x
    s = x * x * x * (x * (x * 6 - 15) + 10)  # smootherstep
    if kind == "inout":
        return s
    if kind == "soft":       # mostly constant speed with gentle ramps
        return 0.65 * x + 0.35 * s
    if kind == "in":          # accelerate
        return x * x * (2 - x) if False else x ** 1.6
    if kind == "out":         # decelerate
        return 1 - (1 - x) ** 1.8
    return s


def _hermite(us, vals, u):
    vals = np.asarray(vals, np.float64)
    n = len(us)
    if n == 1:
        return vals[0]
    k = int(np.clip(np.searchsorted(us, u, side="right") - 1, 0, n - 2))
    u0, u1 = us[k], us[k + 1]
    p0, p1 = vals[k], vals[k + 1]

    def tangent(j):
        if j == 0:
            return (vals[1] - vals[0]) / (us[1] - us[0]) * 0.5
        if j == n - 1:
            return (vals[-1] - vals[-2]) / (us[-1] - us[-2]) * 0.5
        a = (vals[j] - vals[j - 1]) / (us[j] - us[j - 1])
        b = (vals[j + 1] - vals[j]) / (us[j + 1] - us[j])
        return (a * (us[j + 1] - us[j]) + b * (us[j] - us[j - 1])) / (us[j + 1] - us[j - 1])

    m0, m1 = tangent(k), tangent(k + 1)
    h = u1 - u0
    s = (u - u0) / h
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1


def _noise1(x, seed):
    # smooth 1D value noise
    i = math.floor(x)
    f = x - i

    def r(k):
        return (math.sin((k + seed * 17.13) * 127.1) * 43758.5453) % 1.0 * 2 - 1

    s = f * f * (3 - 2 * f)
    return r(i) * (1 - s) + r(i + 1) * s


def shot_camera(shot, t):
    keys = shot["keys"]
    us = np.array([k["u"] for k in keys], np.float64)
    u = _ease(t / shot["dur"], shot.get("ease", "soft"))
    u = us[0] + u * (us[-1] - us[0])
    pos = _hermite(us, [k["pos"] for k in keys], u)
    # interpolate the viewing DIRECTION (not the target point): a near target followed by a far one
    # would otherwise make the camera dip toward the ground between the keys
    dirs = []
    for k in keys:
        d = np.subtract(k["target"], k["pos"]).astype(np.float64)
        dirs.append(d / (np.linalg.norm(d) + 1e-9))
    dv = _hermite(us, dirs, u)
    dv = dv / (np.linalg.norm(dv) + 1e-9)
    tgt = np.asarray(pos) + dv * 1000.0
    fov = float(_hermite(us, [k.get("fov", 55.0) for k in keys], u))
    roll = float(_hermite(us, [k.get("roll", 0.0) for k in keys], u))
    focus = float(_hermite(us, [k.get("focus", 200.0) for k in keys], u))
    ap = float(_hermite(us, [k.get("aperture", 0.0) for k in keys], u))
    sh = shot.get("shake", 0.0)
    if sh > 0:
        d = np.asarray(tgt) - np.asarray(pos)
        dist = np.linalg.norm(d)
        f = d / dist
        r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r)
        up = np.cross(f, r)
        a = shot.get("seed", 1)
        yaw = _noise1(t * 0.35, a) * 0.6 + _noise1(t * 1.3, a + 3) * 0.15
        pit = _noise1(t * 0.31, a + 7) * 0.5 + _noise1(t * 1.1, a + 9) * 0.12
        ang = math.radians(sh)
        tgt = np.asarray(pos) + (f + r * yaw * ang + up * pit * ang) * dist
        roll += _noise1(t * 0.27, a + 11) * sh * 0.8
    return Camera(tuple(map(float, pos)), tuple(map(float, tgt)), fov=fov, roll=roll, aperture=ap, focus=focus)
