"""The film's shot list: 60 s, 9 shots. Camera keys are placed relative to the
baked terrain / features, and every ground-level path searches its neighbourhood
for a variant that stays clear of terrain, trees and hero rocks on every frame,
so the paths stay valid when the world is re-baked."""
import math
from pathlib import Path
from dataclasses import replace

import numpy as np

import bake
import post
from camera_path import shot_camera
from render import Look
from shots import axis_x


def build(world, verbose=False):
    T = TreeMap(bake.CACHE)

    def ground(x, z):
        return max(world.height(x, z), world.water(x, z))

    def P(x, z, agl):
        return (float(x), float(ground(x, z) + agl), float(z))

    def toward(p, az, pitch, dist=1500.0):
        a, e = math.radians(az), math.radians(pitch)
        return (p[0] + math.sin(a) * math.cos(e) * dist, p[1] + math.sin(e) * dist, p[2] + math.cos(a) * math.cos(e) * dist)

    def shore_south(x):
        z = -1000.0
        while world.height(x, z) < 0.0 and z > -2500:
            z -= 0.5
        return z

    def shore_x(z, side):
        x = axis_x(z)
        while world.height(x, z) < 0.0 and abs(x) < 3000:
            x += side * 0.5
        return x

    def K(u, pos, target, fov=55.0, **kw):
        d = dict(u=u, pos=tuple(map(float, pos)), target=tuple(map(float, target)), fov=fov)
        d.update(kw)
        return d

    def path_ok(shot, min_ground=0.6, min_tree=1.0, fps=15):
        n = int(shot["dur"] * fps)
        worst = 1e9
        for f in range(n + 1):
            c = shot_camera(shot, min(f / fps, shot["dur"] - 1e-3))
            x, y, z = c.pos
            cg = y - ground(x, z)
            if cg < min_ground:
                return False, -1.0
            ct = T.clearance(x, y, z) if cg < 45 else 99.0
            if ct < min_tree:
                return False, -1.0
            for hr in shot.get("heroes", []):
                if np.linalg.norm(np.subtract(c.pos, hr[:3])) < hr[3] * max(hr[4:7]) * 1.05 + 0.3:
                    return False, -1.0
            worst = min(worst, ct)
        return True, worst

    def framing(shot, near=2.5, far=18.0):
        """how many tree trunks pass through the view cone at close range (parallax framing)."""
        cnt = 0
        for k in range(5):
            c = shot_camera(shot, shot["dur"] * (k + 0.5) / 5)
            f = np.subtract(c.target, c.pos); f[1] = 0; f /= np.linalg.norm(f) + 1e-9
            x, y, z = c.pos
            for dz in np.arange(-far, far + 1, 4.0):
                for dx in np.arange(-far, far + 1, 4.0):
                    px, pz = x + dx, z + dz
                    v = np.array([dx, 0, dz]); dist = np.linalg.norm(v)
                    if dist < near or dist > far:
                        continue
                    if np.dot(v / dist, f) < math.cos(math.radians(32)):
                        continue
                    if T.clearance(px, y, pz, rad=0) < 0.5:
                        cnt += 1
        return cnt

    def search(name, make, offsets, score=None, **kw):
        """try make(dx, dz) for each offset; first valid (or best scored) wins."""
        best, best_s = None, -1e9
        for dx, dz in offsets:
            shot = make(dx, dz)
            ok, worst = path_ok(shot, **kw)
            if not ok:
                continue
            s = score(shot) if score else 0.0
            if score is None:
                if verbose:
                    print(f"  {name}: offset ({dx},{dz}) ok, min tree {worst:.1f}")
                return shot
            if s > best_s:
                best, best_s = shot, s
        if best is None:
            print(f"  WARNING {name}: no clear path found, using nominal")
            return make(*offsets[0])
        if verbose:
            print(f"  {name}: best framing score {best_s}")
        return best

    def ring(r_max, step):
        out = [(0.0, 0.0)]
        r = step
        while r <= r_max:
            n = max(6, int(2 * math.pi * r / step))
            out += [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]
            r += step
        return out

    # midday sun high (44 deg) from the east-southeast, side-lighting the north-looking shots (reveal, skim,
    # stream, peaks): ribs and gullies lit, their west flanks in blue sky-lit shade (look2 r2b; az 70 left
    # the camera-facing south faces in shade, a flat blue massif)
    base_look = replace(Look(), sun_azimuth=115.0, sun_elevation=44.0)
    base_grade = post.Grade()
    # (look2 r2c) the open lake vistas (reveal, skim) have no deep darks in frame (water, hazy walls): a firmer
    # black point and a little more contrast (p2 43 -> ~28, global log2 Y std 1.2 -> ~1.45; target p2 8-31)
    vista_grade = replace(base_grade, black_max=0.12, contrast=base_grade.contrast * 1.06)
    # (r4 L12) high views (flyover, peaks): the 2-5 km massif's light / shade masses compressed (shot 7 rock p85/p15
    # 6.0 -> ~4.5, target for the peaks 4.2) without the crunchy ribs
    # (r5 light5) struct .5 / hi .5 flattened the massif to a grey plaster (lit / shade ~2x, target 3.4-6.5x): milder
    high_grade = replace(base_grade, struct_gain=0.85)
    # the most prominent summit above the head of the valley (re-found after every re-bake)
    hf = world.hfar
    res = hf.shape[0]
    cell = 2 * bake.FAR_HALF / res
    i0, i1 = int((2800 + bake.FAR_HALF) / cell), int((4800 + bake.FAR_HALF) / cell)
    j0, j1 = int((-1200 + bake.FAR_HALF) / cell), int((1400 + bake.FAR_HALF) / cell)
    sub = hf[i0:i1, j0:j1]
    zz = (np.arange(i0, i1) + 0.5)[:, None] * cell - bake.FAR_HALF
    xx = (np.arange(j0, j1) + 0.5)[None, :] * cell - bake.FAR_HALF
    score = sub - 0.35 * np.hypot(xx - 100.0, zz - 3500.0)      # prominent AND near the valley head
    k = np.unravel_index(np.argmax(score), sub.shape)
    head_peak = (float((j0 + k[1] + 0.5) * cell - bake.FAR_HALF), float(sub[k]), float((i0 + k[0] + 0.5) * cell - bake.FAR_HALF))
    if verbose:
        print("  head peak", np.round(head_peak, 1))
    S = []

    # 1 REVEAL ---------------------------------------------------------------
    # low behind a granite block on the south shore; rise over it to reveal the lake and the head peaks
    def make_reveal(dx, dz):
        xr = -120.0 + dx
        zs = shore_south(xr)
        zb = zs - 5.0 + dz * 0.2
        gb = world.height(xr, zb)
        hero = (xr, gb + 0.6, zb, 2.0, 1.35, 0.95, 1.0, 0.25)
        a = P(xr - 0.4, zb - 4.8, 1.0)
        b = (xr - 0.2, max(gb + 4.4, ground(xr, zb - 2.0) + 2.0), zb - 2.0)
        c = (xr + 1.0, ground(xr, zb + 9.0) + 11.0, zb + 9.0)
        look_peak = (head_peak[0] + 150.0, head_peak[1] - 350.0, head_peak[2])
        return dict(name="reveal", dur=7.5, env="reveal", ease="inout", shake=0.10, seed=1, heroes=[hero],
                    keys=[K(0.0, a, (xr, a[1] + 0.55, zb + 2.0), 52),
                          K(0.42, b, (xr + 4.0, b[1] + 26.0, zb + 400.0), 54),
                          K(1.0, c, look_peak, 56)],
                    look=base_look, grade=vista_grade)
    S.append(search("reveal", make_reveal, [(0, 0)] + [(sg * k * 15.0, 0) for k in range(1, 14) for sg in (-1, 1)], min_ground=0.5))

    # 2 SKIM -----------------------------------------------------------------
    # fast and low over the turquoise water heading north, turning right toward the waterfall shore
    z1 = -1150.0
    x1 = axis_x(z1) - 40.0
    wfp = world.features["waterfall"]
    S.append(dict(name="skim", dur=6.5, env="lakeshore", ease="linear", shake=0.16, seed=2,
                  keys=[K(0.0, (x1, 1.7, z1), (x1 + 60.0, 22.0, z1 + 900.0), 64, roll=-1.5),
                        K(1.0, (x1 + 60.0, 2.6, z1 + 260.0), (wfp["x"], 30.0, wfp["z"] + 120.0), 60, roll=2.5)],
                  look=base_look, grade=vista_grade))

    # 3 WATERFALL ---------------------------------------------------------------
    wf = world.features["waterfall"]
    d = np.array(wf["dir"]); al = np.array([-d[1], d[0]])
    foot = np.array([wf["x"], wf["z"]])
    p0 = foot + d * 95.0 + al * 42.0
    p1 = foot + d * 62.0 + al * 22.0
    tgt = (foot[0] - d[0] * 2.0, 15.0, foot[1] - d[1] * 2.0)
    wf_look = replace(base_look, sun_azimuth=228.0, sun_elevation=42.0)
    # big granite blocks set into the cliff base and ledges (real 3D relief on the face)
    wf_heroes = []
    for (v, u, yy, size, sx, sy, sz, yaw) in [(-24.0, 1.5, 2.0, 4.2, 1.4, 1.0, 1.1, 0.3), (-16.0, 2.5, 13.0, 3.4, 1.3, 0.8, 1.0, 1.1),
                                              (19.0, 1.0, 2.5, 5.0, 1.5, 0.9, 1.2, 2.0), (29.0, 3.0, 15.0, 4.0, 1.2, 1.0, 1.0, 0.7),
                                              (-0.5, -1.0, 0.0, 2.2, 1.3, 0.8, 1.0, 1.9), (13.0, 3.0, 24.0, 3.2, 1.4, 0.7, 1.1, 2.6)]:
        c = foot - d * u + al * v
        wf_heroes.append((float(c[0]), float(yy), float(c[1]), size, sx, sy, sz, yaw))
    S.append(dict(name="waterfall", dur=6.5, env="waterfall", ease="soft", shake=0.10, seed=3,
                  keys=[K(0.0, (p0[0], 2.2, p0[1]), (tgt[0], 13.0, tgt[2]), 50),
                        K(1.0, (p1[0], 3.4, p1[1]), (tgt[0], 15.0, tgt[2]), 48)],
                  look=wf_look, grade=base_grade))

    # 4 FOREST ------------------------------------------------------------------
    # slow lateral dolly inside the east-shore forest, trunks sliding past in front of the lake
    trail = np.asarray(world.features.get("trail", []), np.float64)

    def make_forest(dx, dz):
        # follow the trail northward, looking west through the trunks to the lake
        i0 = int(np.clip(10 + dz / 10, 1, len(trail) - 3))
        a, b = trail[i0], trail[i0 + 2]
        p0 = P(a[0] + dx * 0.1, a[1], 1.9)
        p1 = P(b[0] + dx * 0.1, b[1], 2.1)
        return dict(name="forest", dur=8.0, env="forest", ease="linear", shake=0.08, seed=4, target_luma=88.0,
                    keys=[K(0.0, p0, toward(p0, 258, 1.5), 62), K(1.0, p1, toward(p1, 248, 2.5), 62)],
                    look=replace(base_look, sun_azimuth=200.0, sun_elevation=55.0), grade=base_grade)   # contre-jour, front-left
    offs = [(dx, dz) for dz in range(-80, 121, 10) for dx in (0, -5, 5)]
    S.append(search("forest", make_forest, offs, score=framing, min_tree=0.7))
    # (r5 light5) contre-jour forest frame: crushed darks (p2 3 / p10 10 vs target 8-9 / 24) and over-saturated floor
    # (sat .51 vs .38) under the midday grade -> lighter base, more shadow lift, softer contrast
    forest_grade = replace(base_grade, base_compress=0.7, shadow_lift=0.6, saturation=1.0, contrast=1.2)
    S[-1]["grade"] = forest_grade

    # 5 STREAM ------------------------------------------------------------------
    st = world.features["stream"]

    def make_stream(dx, dz):
        # low over the lake at the stream mouth: boulders, then the cascading stream
        # climbing toward the cirque peaks; slow push in
        km = next((k for k, q in enumerate(st) if q[2] > -0.5), 6)          # mouth
        mouth = np.array(st[km][:2])
        up = np.array(st[min(km + 9, len(st) - 1)])
        d = mouth - np.array(st[max(km - 3, 0)][:2]); d /= np.linalg.norm(d) + 1e-9
        side = np.array([-d[1], d[0]])
        c0 = mouth - d * (48.0 - dz * 0.3) + side * (dx * 0.5 - 6.0)
        c1 = c0 + d * 8.0          # (r4 C7: slower push - the near boulders smeared over 50-90 px)
        p0 = (c0[0], max(ground(c0[0], c0[1]), 0.0) + 1.5, c0[1])
        p1 = (c1[0], max(ground(c1[0], c1[1]), 0.0) + 1.9, c1[1])
        tgt = (up[0], up[2] + 18.0, up[1])
        return dict(name="stream", dur=6.5, env="stream", ease="soft", shake=0.07, seed=5,
                    keys=[K(0.0, p0, tgt, 56), K(1.0, p1, (up[0], up[2] + 24.0, up[1]), 54)],
                    look=base_look, grade=base_grade)
    S.append(search("stream", make_stream, ring(30, 4)))

    # 6 MEADOW + TARN -------------------------------------------------------------
    tarn = (-1690.0, 515.0, -440.0)

    def make_meadow(dx, dz):
        # (r4 M5, world4 re-aim: the tarn from its north-west shoulder ~180 m above it, crest + sky on top;
        # slow rise and drift)
        p0 = P(-2060.0 + dx, -40.0 + dz, 1.6)
        p1 = P(-2052.0 + dx, -48.0 + dz, 2.6)
        return dict(name="meadow", dur=6.5, env="meadow", ease="soft", shake=0.08, seed=6, target_luma=122.0,
                    keys=[K(0.0, p0, toward(p0, 133, -12.0), 58), K(1.0, p1, toward(p1, 131, -10.5), 56)],
                    look=replace(base_look, sun_azimuth=220.0, sun_elevation=58.0), grade=base_grade)   # (r4 L7: side light, not front-lit)
    S.append(search("meadow", make_meadow, ring(40, 5), min_tree=4.0))   # (r5 qa5 B06: >= 4 m from trees)

    # 7 HIGH FLYOVER ----------------------------------------------------------------
    z7 = 1950.0
    p0 = (axis_x(z7) + 320.0, 620.0, z7)
    p1 = (axis_x(z7 - 300.0) + 260.0, 560.0, z7 - 300.0)
    S.append(dict(name="flyover", dur=6.5, env="high_view", ease="linear", shake=0.06, seed=7, target_luma=100.0,
                  keys=[K(0.0, p0, (axis_x(-1700.0), -60.0, -1700.0), 64), K(1.0, p1, (axis_x(-2400.0), -100.0, -2400.0), 66)],
                  look=base_look, grade=high_grade))

    # 8 PEAKS -------------------------------------------------------------------------
    # a slow lateral pass ~1.2 km in front of the summit ridge, sky behind the crest
    hp = np.array(head_peak)

    def make_peaks(dx, dz):
        # high above the middle of the lake looking north to the cirque head and its summits
        # (mirror of the flyover, which looks south); slow forward drift with a gentle drift sideways
        z0 = -350.0 + dz
        c0 = np.array([axis_x(z0) + 60.0 + dx, z0])
        c1 = c0 + np.array([-70.0, 280.0])
        # (r4 C2: lower and further back - the forested valley and the lake give the peaks scale instead of a
        # frame-filling wall of rock)
        y0, y1 = 700.0, 740.0
        t0 = (hp[0] - 350.0, 1750.0, hp[2])     # (r4 world4: a few degrees left - the east summit stood at the right edge)
        t1 = (hp[0] - 260.0, 1850.0, hp[2])
        return dict(name="peaks", dur=6.0, env="high_view", ease="soft", shake=0.05, seed=8,
                    keys=[K(0.0, (c0[0], y0, c0[1]), t0, 50), K(1.0, (c1[0], y1, c1[1]), t1, 50)],
                    look=base_look, grade=high_grade)
    S.append(search("peaks", make_peaks, [(0, 0), (200, 0), (-200, 0), (0, -300), (0, 300)], min_ground=100.0))

    # 9 FINALE (golden) ------------------------------------------------------------------
    # the near shore and the near water lie in shadow (cool), the far shore forest and the peaks take the warm
    # light; clean evening air: less Mie haze, thinner near air (tune 13); a weaker horizon sky term
    # (tune 0) keeps the shaded tree flanks dark (target lit / shade 3.0 vs 1.7 at midday)
    # (look2 r2c) a 10 deg sun from behind (205) front-lit the far forest flat (off-axis 167 deg, pale speckle)
    # and at 255-305 / 10-20 the west ridge shades the whole valley. 295 / 28 lights the far shore and the peaks
    # from the side (off-axis 97 deg, 90 % of the far forest lit) while the near shore stays in shade; the sun
    # gets the colour of a ~10 deg sun (tune 9 = 1.8: transmittance of a 2.8x longer air path, probe.frag);
    # sky_boost 12 (22): the low-sun sky is no brighter than the lit peaks (target: lit granite Y 0.47, sky 0.37)
    # (r2d) 298 / 26: at 300 / 24 the water turned from a khaki mirror of the lit wall to turquoise but the
    # shoreline forest fell into shade; in-scatter keeps its computed (warm) colour (21 = 1, midday 0.5)
    # (r4 L3) Look.shadow_elevation (terrain / tree shadows from a lower sun) tested at 15-28 deg / az 240-298: it never
    # shades this near shore (no occluder to the W / WNW close enough), only the far wall and far-shore forest that
    # should stay lit -> not used here (0 = the 26 deg sun); the near-shore shade needs a camera / world change
    # (r4 L4) amb_boost 2.6 -> 3.3, sky_boost 12 -> 9: the shade read too dark under a too-bright sky (r9 sky / land 5.7,
    # finale 8.3, target 3.0; finale p50 71 vs 113: its exposure is sky-capped); shade x1.27 ~ the verifier's
    # (91, 99, 108) -> target (103, 108, 120)
    _gt = {0: 0.3, 9: 1.8, 13: 0.45, 21: 1.0}
    # (r5 light5) under the darker directly-seen sky (lighting.glsl SKY_VIEW_GAIN) sky_boost 9 -> 14 (target sky top #82a4e2),
    # amb_boost 3.3 -> 2.5 (the midday value: the near shore / water in shade read too bright), sun warmth 1.8 -> 1.4 (lit wall
    # tan, not orange)
    _gt = {0: 0.3, 9: 1.4, 13: 0.45, 21: 1.0}
    gold = replace(base_look, sun_azimuth=298.0, sun_elevation=26.0, cloud_cover=0.22, haze=1.2, amb_boost=2.5,
                   sky_boost=14.0, tune=tuple(_gt.get(i, v) for i, v in enumerate(base_look.tune)),
                   wave_strength=1.5)     # (r4 L5: the wall's mirror image breaks into streaks, ripples lower the effective Fresnel)
    gold_grade = base_grade        # (r2b: neutral white balance, the shade stays cool; the warmth is the sun's)
    # (r4 L4) the finale's exposure sits on the sky cap (2.08), its shade fell to land 62 / near shore (29, 27, 17)
    # (target land 102, near shore (43, 47, 35)): a lighter base lift for this shot only -> land 71, near shore (38, 35, 23)
    # (r5 light5) + saturation .92 / chroma_boost .08: graded sat .51 / s90 .86 vs target .31 / .57 (yellow foliage, cyan water)
    gold_grade = replace(base_grade, base_compress=0.65, shadow_lift=0.4, saturation=0.92, chroma_boost=0.08)

    def make_finale(dx, dz):
        z9 = 300.0 + dz
        x9 = shore_x(z9, -1) - 2.0 + dx
        p0 = P(x9, z9, 1.7)
        p1 = P(x9 - 3.0, z9 - 9.0, 3.5)        # (r4 C7: shorter rise - the foreground rocks smeared)
        return dict(name="finale", dur=6.0, env="golden", ease="inout", shake=0.06, seed=9, target_luma=100.0,
                    keys=[K(0.0, p0, toward(p0, 35, 7), 60), K(1.0, p1, toward(p1, 38, 9), 64)],   # (r5 qa5 B07: +2 deg)
                    look=gold, grade=gold_grade)
    S.append(search("finale", make_finale, [(0, dz) for dz in (0, -20, 20, -40, 40, -60, 60, -80, 80)] + [(dx, 0) for dx in (2, 4)]))
    # (r5 light5) exposure targets: the half-water vistas and the peaks sat ~10 codes under the target sky / land (sky
    # luma 117-130 vs 140-165 under the darker directly seen sky); the flyover / finale read darker than their targets (p50 90 / 92
    # vs 124 / 113)
    for k, tl in ((0, 118.0), (1, 118.0), (6, 112.0), (7, 122.0), (8, 106.0)):
        S[k]["target_luma"] = tl
    # (r5, user: open with the strongest scene - the forest dolly through the spruces onto the lake (was at ~25 s);
    # then the rising reveal from behind the rock and the rest in order. Durations: forest 8.0, reveal 7.5, total 60 s)
    S = [S[3]] + S[0:3] + S[4:]
    return S


TREE_JIT = 0.6     # cell jitter of the tree position (trees.glsl tree placement) - keep in sync


class TreeMap:
    """CPU copy of the baked per-cell trees (same placement hash as trees.glsl) for camera clearance."""
    def __init__(self, cache):
        """per-cell trees exactly as rendered: the baked rows passed through trees.glsl treeFinish()
        (stand culling, final height / crown radius / crown base), via treemap_port."""
        raw = np.load(cache / "trees.npy")
        import treemap_port as tp
        tp.CACHE = Path(cache)
        nz = np.nonzero(raw[..., 1])
        tz, tx = nz
        keep, H, R, yb, larch, _ = tp.tree_final(tx + 2048, tz + 2048, raw[tz, tx])
        t = np.zeros_like(raw)
        t[tz, tx, 0] = raw[tz, tx, 0]
        t[tz, tx, 1] = np.where(keep, H, 0.0)
        t[tz, tx, 2] = R * 1.25                      # ~branch reach beyond the crown radius
        t[tz, tx, 3] = yb
        self.t = t

    @staticmethod
    def _hash(cx, cz):
        def hu(x):
            x &= 0xFFFFFFFF
            x ^= x >> 16; x = (x * 0x7feb352d) & 0xFFFFFFFF
            x ^= x >> 15; x = (x * 0x846ca68b) & 0xFFFFFFFF
            x ^= x >> 16
            return x
        h = hu(((cx & 0xFFFFFFFF) * 0x9E3779B1 + hu((cz + 0x632BE5AB) & 0xFFFFFFFF)) & 0xFFFFFFFF)
        a = h; b = hu(a); d = hu(b); e = hu(d)
        return [(v >> 8) / 16777216.0 for v in (a, b, d, e)]

    def trees_near(self, x, z, r):
        """(px, pz, H, R) of every rendered tree within r metres (horizontal) of (x, z)"""
        out = []
        cx0, cz0 = int(math.floor((x + 16384) / 4)), int(math.floor((z + 16384) / 4))
        n = int(r / 4) + 1
        for dz in range(-n, n + 1):
            for dx in range(-n, n + 1):
                cx, cz = cx0 + dx, cz0 + dz
                tx, tz = cx - 2048, cz - 2048
                if not (0 <= tx < 4096 and 0 <= tz < 4096):
                    continue
                by, H, R, yb = self.t[tz, tx]
                if H == 0:
                    continue
                hs = self._hash(cx, cz)
                px = cx * 4.0 - 16384 + 4.0 * (0.5 + (hs[1] - 0.5) * TREE_JIT)
                pz = cz * 4.0 - 16384 + 4.0 * (0.5 + (hs[2] - 0.5) * TREE_JIT)
                if math.hypot(px - x, pz - z) <= r:
                    out.append((px, pz, abs(float(H)), float(R)))
        return out

    def in_view(self, cam, a0, a1, d0, d1):
        """trees whose trunk lies in the horizontal view wedge [a0, a1] deg (negative = left of the view
        axis) at horizontal distance d0..d1 m from the camera"""
        p, tgt = np.asarray(cam.pos, float), np.asarray(cam.target, float)
        az = math.atan2(tgt[0] - p[0], tgt[2] - p[2])
        n = 0
        for px, pz, H, R in self.trees_near(p[0], p[2], d1):
            d = math.hypot(px - p[0], pz - p[2])
            if d < d0:
                continue
            a = math.degrees(math.atan2(px - p[0], pz - p[2]) - az)
            a = (a + 180.0) % 360.0 - 180.0
            if a0 <= a <= a1:
                n += 1
        return n

    def clearance(self, x, y, z, rad=3):
        """min horizontal distance (m) from point to any crown/trunk surface at height y."""
        cx0, cz0 = int(math.floor((x + 16384) / 4)), int(math.floor((z + 16384) / 4))
        best = 1e9
        for dz in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                cx, cz = cx0 + dx, cz0 + dz
                tx, tz = cx - 2048, cz - 2048
                if not (0 <= tx < 4096 and 0 <= tz < 4096):
                    continue
                by, H, R, yb = self.t[tz, tx]
                if H == 0:
                    continue
                H = abs(H)
                hs = self._hash(cx, cz)
                px = cx * 4.0 - 16384 + 4.0 * (0.5 + (hs[1] - 0.5) * TREE_JIT)
                pz = cz * 4.0 - 16384 + 4.0 * (0.5 + (hs[2] - 0.5) * TREE_JIT)
                q = y - by
                if q > H + 0.5 or q < -1:
                    r = 0.0 if q > H + 0.5 else 0.4
                    dist = math.hypot(x - px, z - pz) - r
                else:
                    tt = min(max((q - yb) / max(H - yb, 0.1), 0.0), 1.0)
                    r = R * (1.0 - tt) ** 0.95 * 1.1 if q >= yb else 0.5
                    dist = math.hypot(x - px, z - pz) - r
                best = min(best, dist)
        return best
