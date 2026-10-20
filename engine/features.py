"""Hand-placed world features carved after erosion: a twin waterfall cascading down a dark rock
step in the east-shore hillside into the lake, and a boulder-choked inflow stream at the
north end. Placement is derived from the (eroded) height map so it survives terrain changes.
After the near map is carved, the path of each fall is simulated on it (falls_paths) and stored in
features.json as "falls" (knots passed to falls.glsl as uFallK / uFallInfo by render.py)."""
import json
import math

import numpy as np

import bake
import gl

WF_Z = -550.0         # waterfall position along the east shore (steep hillside behind)
WF_HEIGHT = 29.0      # cliff height above the lake
WF_HALF = 22.0        # along-shore half width of the cliff block
NOTCH = (-8.0, 12.0, 3.2, 1.6)  # lip offsets (vA, vB), notch width, notch depth
# keep in sync with falls.glsl (WF_LIPB, WF_DRIFT_A, wfFaceW, wfChuteV, wfChuteHW)
LIPB = 0.50
DRIFT_A = 11.0      # over the whole face, nearly linear (session 3; was smoothstep(0, 0.62, s))
FALL_VOL = (1.0, 0.85)
FAN = ((1.4, 4.0), (0.9, 4.2))     # water half width at the lip / at the pool (session 3: wider fans, fall A ~6 m; review r2: B a lacy fan ~3x wider at the pool than at its lip)
# ledges where the water lands and bursts (fractions of the lip height; falls.glsl wfLedges carves
# matching benches): at each the sheet kicks out LEDGE_OUT m just above, and widens by LEDGE_WIDEN m
# (review r2) + a burst ~15 % below fall A's lip (the sheet breaks into white water there; no carved bench,
# a small kick-out only: LEDGE_OUT_K = per-ledge share of LEDGE_OUT)
LEDGES = ((0.85, 0.62, 0.30), (0.45,))
LEDGE_OUT_K = ((0.35, 1.0, 1.0), (1.0,))
LEDGE_OUT, LEDGE_WIDEN = 0.6, 0.25   # (session 3: 1.1 -> 0.6, the bulge read as a draped skirt)
# ... and the water steps down the rock: at each ledge the path jogs LEDGE_JOG m along the shore
# (alternating direction), the sheet is pinched (x LEDGE_PINCH) on the knot above the ledge and spread
# (x LEDGE_SPREAD) on the two knots below it
LEDGE_JOG, LEDGE_PINCH, LEDGE_SPREAD = 0.45, 0.88, 1.08   # (session 3: 0.7 / 1.25 -> bell-shaped skirts)
VMAX, FRICTION, AIRDRAG = 5.0, 2.5, 1.6
NK = 24


def axis_x(z):
    return 150.0 * math.sin(z * 0.00105 + 0.5) + 45.0 * math.sin(z * 0.0029 + 2.1)


class Sampler:
    def __init__(self, a, half):
        self.a, self.half, self.res = a, half, a.shape[0]

    def __call__(self, x, z):
        fx = (x + self.half) / (2 * self.half) * self.res - 0.5
        fz = (z + self.half) / (2 * self.half) * self.res - 0.5
        ix, iz = int(np.floor(fx)), int(np.floor(fz))
        tx, tz = fx - ix, fz - iz
        ix = min(max(ix, 0), self.res - 2); iz = min(max(iz, 0), self.res - 2)
        a = self.a
        return float((a[iz, ix] * (1 - tx) + a[iz, ix + 1] * tx) * (1 - tz) + (a[iz + 1, ix] * (1 - tx) + a[iz + 1, ix + 1] * tx) * tz)


def place(hfar):
    H = Sampler(hfar, bake.FAR_HALF)
    # waterfall: walk east from the lake axis to the waterline
    z = WF_Z
    x = axis_x(z)
    while H(x, z) < 0.0 and x < 3000:
        x += 1.0
    e = 6.0
    g = np.array([H(x + e, z) - H(x - e, z), H(x, z + e) - H(x, z - e)]) / (2 * e)
    out = -g / (np.linalg.norm(g) + 1e-6)       # downhill = toward the lake
    if out[0] > -0.3:                             # keep it facing roughly west
        out = np.array([-0.94, out[1] * 0.3]); out /= np.linalg.norm(out)
    wf = dict(x=x, z=z, height=WF_HEIGHT, half=WF_HALF, dir=out.tolist(), notch=list(NOTCH))
    # stream: from inside the lake's north tip upstream into the cirque
    zt = 1200.0
    while H(axis_x(zt), zt) < 0.3 and zt < 3500:
        zt += 5.0
    pts = []
    # track the valley floor upstream (greedy, smooth), then carve a bed that only ever cuts down
    zs = np.arange(zt - 40.0, zt + 1300.0, 22.0)
    xs, ters = [], []
    xp = axis_x(zt)
    for zz in zs:
        best = None
        for dx in np.arange(-33.0, 33.1, 3.0):
            xc = xp + dx
            hmin = min(H(xc + o, zz) for o in (-6.0, 0.0, 6.0))
            cost = hmin + 0.04 * abs(dx)
            if best is None or cost < best[0]:
                best = (cost, xc, hmin)
        xp = best[1]
        xs.append(xp); ters.append(best[2])
    xs = np.convolve(np.pad(xs, 2, mode="edge"), np.ones(5) / 5, mode="valid")   # smooth meanders
    bed = np.array([min(H(x + o, z) for o in (-4.0, 0.0, 4.0)) for x, z in zip(xs, zs)]) - 0.2
    bed[:2] = np.minimum(bed[:2], -0.6)
    for k in range(len(bed) - 2, -1, -1):          # downstream is always lower (carve, never perch)
        bed[k] = min(bed[k], bed[k + 1] - 0.08)
    stop = len(bed)
    for k in range(len(bed)):
        if bed[k] > 260.0:
            stop = k; break
    for k in range(stop):
        zz = zs[k]
        # half width of the open, boulder-strewn floodplain (bake_forest clears trees ~2-4x this);
        # the wet channel itself is STREAM_CHAN x narrower (bake_features.frag)
        hw = 9.0 if k < 4 else max(3.4, 7.5 - (zz - zt) / 240.0)
        pts.append((float(xs[k]), float(zz), float(bed[k]), hw))
    # fade the channel out at the top
    for k in range(1, 5):
        x_, z_, b_, w_ = pts[-k]
        pts[-k] = (x_, z_, b_, w_ * (0.25 + 0.15 * k))
    # forest trail along the east shore (lake seen between trunks from a path)
    trail = []
    for zz in np.arange(300.0, 600.0, 10.0):
        xx = axis_x(zz)
        while H(xx, zz) < 0.0 and xx < 3000:
            xx += 1.0
        xx += 20.0 + 6.0 * math.sin(zz * 0.031) + 3.0 * math.sin(zz * 0.11 + 1.0)
        trail.append([float(xx), float(zz), 2.2])
    return dict(waterfall=wf, stream=[list(map(float, p)) for p in pts[:96]], trail=trail)


def _uniforms(p, f, center, half, res):
    wf = f["waterfall"]
    p.set("uCenter", center); p.set("uHalf", half); p.set("uRes", float(res))
    p.set("uWF", (wf["x"], wf["z"], wf["height"], wf["half"]))
    p.set("uWFDir", tuple(wf["dir"]))
    p.set("uWFNotch", tuple(wf["notch"]))
    st = np.zeros((96, 4), np.float32)
    s = np.asarray(f["stream"], np.float32)
    st[: len(s)] = s
    p.set("uStream", st)
    p.set("uStreamN", int(len(s)))


def _face_w(hs):
    return 2.0 + 0.30 * hs


def chute_v(k, u, wf):
    vA, vB = wf["notch"][0], wf["notch"][1]
    if k == 1:
        return vB
    W = _face_w(wf["height"])
    s = np.clip((1.5 + W - u) / W, 0.0, 1.0)
    return vA + DRIFT_A * (0.75 * s + 0.25 * s * s * (3 - 2 * s))   # (session 3: straight diagonal)


def chute_hw(k, u, wf):
    W = _face_w(wf["height"])
    s = float(np.clip((1.5 + W - u) / W, 0.0, 1.0))
    return (2.4 + (5.8 - 2.4) * s) if k == 0 else (2.6 + (6.0 - 2.6) * s)   # (review r2: B's fan is wider)


def falls_paths(hnear, f):
    """simulate a water parcel down each fall's chute on the carved near map: it slides on the rock
    (with friction), leaves convex edges ballistically and records the impact where it lands again.
    Returns per fall: lip height, volume and NK knots (u, v, half width, impact) evenly spaced in height."""
    wf = f["waterfall"]
    Hn = Sampler(hnear, bake.NEAR_HALF)
    d = np.array(wf["dir"]); al = np.array([-d[1], d[0]]); foot = np.array([wf["x"], wf["z"]])

    def hl(u, v):
        x, z = foot - d * u + al * v
        return Hn(x, z)

    W = _face_w(wf["height"])
    out = []
    for k in range(2):
        u = 1.5 + (W + 9.0 if k == 0 else 0.45 * W + 3.0)
        y = hl(u, chute_v(k, u, wf)) + 0.12
        vu, vy = -1.4, 0.0
        dt = 0.004
        pts = []
        imp_acc = 0.0
        for _ in range(60000):
            vy -= 9.81 * dt
            u += vu * dt; y += vy * dt
            v = chute_v(k, u, wf)
            hs = hl(u, v)
            imp = 0.0
            if y < hs + 0.12:
                e = 0.25
                s = (hl(u + e, chute_v(k, u + e, wf)) - hl(u - e, chute_v(k, u - e, wf))) / (2 * e)
                T = np.array([-1.0, -s]); T /= np.linalg.norm(T)
                vel = np.array([vu, vy])
                vt = float(vel @ T)
                imp = float(np.linalg.norm(vel - vt * T))
                # rough rock: the thin sheet loses speed fast and stays on the face
                vt = min(max(vt, 0.4), VMAX) * (1.0 - FRICTION * dt)
                vu, vy = T * vt
                y = hs + 0.12
            else:
                vu *= 1.0 - AIRDRAG * dt          # a torn, aerated sheet: strong drag across the fall
            imp_acc = max(imp_acc * (1.0 - 6.0 * dt), imp)
            pts.append((u, v, y, imp_acc, -vy / max(abs(vu), 1e-3)))
            if y < -0.6:
                break
        P = np.array(pts)
        # the lip = where the parcel first drops steeply (leaves the bench / cleft floor)
        steep = np.nonzero((P[:, 4] > 1.2) & (P[:, 2] > 2.0))[0]
        i0 = int(steep[0]) if len(steep) else 0
        P = P[i0:]
        top = float(P[0, 2])
        ys = top - (top + 0.6) * np.arange(NK) / (NK - 1)
        yy = np.minimum.accumulate(P[:, 2])
        knots = []
        for yk in ys:
            j = int(np.clip(np.searchsorted(-yy, -yk), 0, len(P) - 1))
            uu, vv = P[j, 0], P[j, 1]
            # impact: strongest landing within half a knot spacing
            sel = (yy <= yk + (top + 0.6) / (NK - 1) * 0.6) & (yy >= yk - (top + 0.6) / (NK - 1) * 0.6)
            im = float(P[sel, 3].max()) if sel.any() else 0.0
            fr = float(np.clip((top - yk) / max(top, 1.0), 0.0, 1.0)) ** 0.8
            hw = FAN[k][0] + (FAN[k][1] - FAN[k][0]) * fr
            knots.append([float(uu), float(vv), float(hw), float(np.clip((im - 1.0) / 5.0, 0.0, 1.0))])
        K = np.array(knots)
        # no kinks: the sampled path jumps where the parcel leaves / meets the rock -> smooth u, v
        for _ in range(2):
            K[1:-1, :2] = 0.25 * K[:-2, :2] + 0.5 * K[1:-1, :2] + 0.25 * K[2:, :2]
        # deliberate ledges (benches carved by bake_features.frag at the same heights): the water lands,
        # bursts and bounces off (the sheet bulges LEDGE_OUT m outward and re-attaches below), and widens
        dy = (top + 0.6) / (NK - 1)
        for li, fl_ in enumerate(LEDGES[k]):
            yl = fl_ * (top - 0.12)
            j = int(np.argmin(np.abs(ys - yl)))
            # burst: smooth over +-1 knot (no 0 -> 0.9 -> 0 flicker along the sheet)
            for o, im_ in ((-1, 0.4), (0, 0.9), (1, 0.75), (2, 0.4)):
                if 0 <= j + o < NK - 1:
                    K[j + o, 3] = max(K[j + o, 3], im_)
            for o, a_ in ((-1, 0.45), (0, 1.0), (1, 0.75), (2, 0.4), (3, 0.15)):
                if 0 <= j + o < NK - 1:
                    K[j + o, 0] -= LEDGE_OUT * LEDGE_OUT_K[k][li] * a_
            K[j:, 2] += LEDGE_WIDEN * np.clip((ys[j:] - (yl - dy)) / -dy + 1.0, 0.0, 1.0)
            # step down the rock: pinched above the ledge, spread below it, jogging sideways
            if j > 0:
                K[j - 1, 2] *= LEDGE_PINCH
            for o in (1, 2):
                if j + o < NK:
                    K[j + o, 2] *= LEDGE_SPREAD
            sg = 1.0 if (li + k) % 2 == 0 else -1.0
            K[j, 1] += 0.5 * sg * LEDGE_JOG
            K[j + 1:, 1] += sg * LEDGE_JOG
        knots = K.tolist()
        knots[-1][3] = max(knots[-1][3], 1.0)        # pool impact
        out.append(dict(top=top, vol=FALL_VOL[k], knots=knots))
        print(f"  fall {k}: lip {top:.1f} m, base u {knots[-1][0]:.1f}, impacts {[round(kk[3], 2) for kk in knots]}", flush=True)
    return out


def carve(h, half, f, tile=512):
    """run the feature pass on a height map; returns height, stream level, foam, flow angle."""
    c = gl.ctx()
    res = h.shape[0]
    src = gl.tex2d(h)
    out = c.texture((res, res), 4, dtype="f4")
    fbo = c.framebuffer([out])
    p = gl.Pass("bake_features.frag")
    p.tex("uH", src, 0)
    _uniforms(p, f, (0.0, 0.0), half, res)
    for y in range(0, res, tile):
        for x in range(0, res, tile):
            p.draw(fbo, (x, y, tile, tile)); c.finish()
    a = gl.read_tex(out)
    fbo.release(); out.release(); src.release()
    slv = a[..., 1].copy()
    if abs(half - bake.FAR_HALF) < 1.0:
        _damp_flow_banks(f, half, res)
        # (round 2) the 4 m far stream level only where the 1 m near map does not cover the stream:
        # bake.py builds the near level map from the far one (nearest, 4 m blocks) maxed with the near
        # stream level, and the far blocks stood above the near banks as square water patches
        cc = (np.arange(res) + 0.5) / res * 2 * half - half
        inner = np.abs(cc) < 0.86 * bake.NEAR_HALF
        slv[np.ix_(inner, inner)] = -1e4
    h0 = a[..., 0].copy()
    if abs(half - bake.NEAR_HALF) < 1.0:
        _limit_cliff(h0, f, half, res)
        _check_near(h0, f, half, res)
        f["falls"] = falls_paths(h0, f)
        _stream_corridor(h0, slv, f, half, res)
        save_meta(f)
    return h0, slv, a[..., 2].copy(), a[..., 3].copy()


def _stream_corridor(h, slv, f, half, res):
    """(review r2, rockfix v3) signed distance (m) to the inflow stream's wet channel on the near grid,
    with its gradient, cropped to the stream: cache/stream_corr.npy (RGBA16F: d, dd/dx, dd/dz, 0) and
    f["stream_corr_box"] = [x0, z0, width, height] (m). rocks.glsl fetches it once per 5 m rock cell and
    keeps every block out of the channel (+ its own radius): the white water stays open."""
    from scipy import ndimage
    st = np.asarray(f.get("stream", []), np.float64)
    if len(st) < 2:
        return
    cell = 2 * half / res
    m = 40.0
    x0, x1 = st[:, 0].min() - m, st[:, 0].max() + m
    z0, z1 = max(st[:, 1].min() - m, -half), min(st[:, 1].max() + m, half)
    j0, j1 = int((x0 + half) / cell), int((x1 + half) / cell)
    i0, i1 = int((z0 + half) / cell), int((z1 + half) / cell)
    lv = slv[i0:i1, j0:j1]; hh = h[i0:i1, j0:j1]
    wet = (lv > -1000.0) & (lv > hh + 0.03) & (lv > 0.04)
    if not wet.any():
        return
    d = ndimage.distance_transform_edt(~wet) * cell - ndimage.distance_transform_edt(wet) * cell
    d = np.clip(d, -12.0, 30.0)
    gz, gx = np.gradient(d, cell)
    out = np.stack([d, gx, gz, np.zeros_like(d)], -1).astype(np.float16)
    np.save(bake.CACHE / "stream_corr.npy", out)
    f["stream_corr_box"] = [float(j0 * cell - half), float(i0 * cell - half), float((j1 - j0) * cell), float((i1 - i0) * cell)]
    print(f"  stream corridor map {out.shape[1]}x{out.shape[0]} ({int(wet.sum())} wet texels)", flush=True)


def _falls_window(f, half, res, r=150.0):
    wf = f["waterfall"]
    cell = 2 * half / res
    i0 = max(int((wf["z"] - r + half) / cell), 1); i1 = min(int((wf["z"] + r + half) / cell), res - 1)
    j0 = max(int((wf["x"] - r + half) / cell), 1); j1 = min(int((wf["x"] + r + half) / cell), res - 1)
    return i0, i1, j0, j1


def _nb8(h, i0, i1, j0, j1):
    return np.stack([h[i0 + di:i1 + di, j0 + dj:j1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj])


def _limit_cliff(h, f, half, res, jmax=5.0, iters=12):
    """Near map, around the falls: limit the height jump between neighbouring texels to jmax m (axis) /
    1.41 jmax (diagonal) by moving both sides half way (a relaxation). The cliff's block relief makes
    near-vertical texel steps that the tracer's 0.42 x height steps can overshoot."""
    i0, i1, j0, j1 = _falls_window(f, half, res)
    s = h[i0 - 1:i1 + 1, j0 - 1:j1 + 1].astype(np.float64)
    offs = [(0, 1, jmax), (1, 0, jmax), (1, 1, jmax * 1.41), (1, -1, jmax * 1.41)]
    for _ in range(iters):
        moved = 0.0
        for di, dj, lim in offs:
            a = s[1:-1, 1:-1]
            b = s[1 + di:s.shape[0] - 1 + di, 1 + dj:s.shape[1] - 1 + dj]
            ex = np.abs(a - b) - lim
            m = ex > 0
            if not m.any():
                continue
            corr = np.where(m, 0.5 * ex * np.sign(a - b), 0.0) * 0.5
            a -= corr; b += corr
            moved = max(moved, float(np.abs(corr).max()))
        if moved < 0.02:
            break
    n = int((np.abs(s[1:-1, 1:-1] - h[i0:i1, j0:j1]) > 0.05).sum())
    h[i0:i1, j0:j1] = s[1:-1, 1:-1].astype(h.dtype)
    print(f"  near map: cliff slope limit {jmax} m/texel moved {n} texels", flush=True)


def _check_near(h, f, half, res):
    """fail the bake on broken carves: non-finite heights, or single-texel needles (a texel > 3 m above
    all 8 neighbours) within 150 m of the falls"""
    if not np.isfinite(h).all():
        raise RuntimeError(f"near map: {int((~np.isfinite(h)).sum())} non-finite heights after the feature carve")
    i0, i1, j0, j1 = _falls_window(f, half, res)
    c = h[i0:i1, j0:j1]
    spike = c - _nb8(h, i0, i1, j0, j1).max(0)
    bad = np.argwhere(spike > 3.0)
    if len(bad):
        cell = 2 * half / res
        pts = [((j0 + b + 0.5) * cell - half, (i0 + a + 0.5) * cell - half, float(c[a, b])) for a, b in bad[:8]]
        raise RuntimeError(f"near map: {len(bad)} height spikes > 3 m near the falls (x, z, h): {pts}")
    print(f"  near map check OK (max texel above all neighbours {float(spike.max()):.2f} m)", flush=True)


def _damp_flow_banks(f, half, res):
    """The carved step re-routes drainage: the erosion flow map (cache/flow.npy, used only by the forest
    bake to keep trees out of gullies) still shows the old hillside's shore runoff across the new steep
    banks beside the falls, which would leave them bare. Clamp it there (not in the two fall chutes)."""
    fp = bake.CACHE / "flow.npy"
    if not fp.exists():
        return
    fl = np.load(fp)
    if fl.shape[0] != res:
        return
    wf = f["waterfall"]
    d = np.array(wf["dir"]); al = np.array([-d[1], d[0]]); foot = np.array([wf["x"], wf["z"]])
    cell = 2 * half / res
    i0 = int((foot[1] - 220 + half) / cell); i1 = int((foot[1] + 220 + half) / cell)
    j0 = int((foot[0] - 220 + half) / cell); j1 = int((foot[0] + 220 + half) / cell)
    zz, xx = np.mgrid[i0:i1, j0:j1]
    x = (xx + 0.5) * cell - half - foot[0]; z = (zz + 0.5) * cell - half - foot[1]
    u = -(x * d[0] + z * d[1]); v = x * al[0] + z * al[1]
    box = np.clip((u + 25.0) / 15.0, 0, 1) * np.clip((170.0 - u) / 40.0, 0, 1) \
        * np.clip((v + 110.0) / 30.0, 0, 1) * np.clip((140.0 - v) / 30.0, 0, 1)
    vA, vB = wf["notch"][0], wf["notch"][1]
    chute = np.maximum(np.clip(1.0 - (np.abs(v - vA) - 5.0) / 3.0, 0, 1), np.clip(1.0 - (np.abs(v - vB) - 4.0) / 3.0, 0, 1))
    w = box * (1.0 - chute)
    sub = fl[i0:i1, j0:j1]
    fl[i0:i1, j0:j1] = sub * (1.0 - w) + np.minimum(sub, 250.0) * w
    np.save(fp, fl.astype(np.float32))


def save_meta(f):
    (bake.CACHE / "features.json").write_text(json.dumps(f, indent=1))


def load_meta():
    p = bake.CACHE / "features.json"
    return json.loads(p.read_text()) if p.exists() else None
