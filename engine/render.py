"""GPU renderer: uploads the baked world, renders tiled multi-sample HDR frames."""
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import moderngl
import numpy as np
import time as _t

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gl  # noqa: E402
import bake  # noqa: E402


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@dataclass
class Camera:
    pos: tuple
    target: tuple
    fov: float = 55.0          # vertical degrees
    roll: float = 0.0          # degrees
    aperture: float = 0.0      # lens radius in m (0 = pinhole)
    focus: float = 100.0       # focus distance in m

    def basis(self):
        f = normalize(np.subtract(self.target, self.pos))
        r = normalize(np.cross([0.0, 1.0, 0.0], f))
        u = np.cross(f, r)
        if self.roll:
            a = math.radians(self.roll)
            r, u = r * math.cos(a) + u * math.sin(a), -r * math.sin(a) + u * math.cos(a)
        return np.stack([r, u, f])  # rows = columns of the GLSL mat3


@dataclass
class Look:
    """Lighting / atmosphere / grading parameters for a shot."""
    # (look2 r2b: midday sun high (44 deg), side-lit from the east-southeast for the lake's north-looking
    # views: az 70 put their camera-facing south faces in shade (shot 7 lit/shade spread 5.6, target 8.0);
    # az 100-125 gives the crisp ribs and gullies)
    sun_azimuth: float = 115.0     # degrees, 0 = north(+z), 90 = east(+x)
    sun_elevation: float = 44.0
    sun_intensity: float = 2.6
    # (r5 light5) haze 2.8 -> 1.0: the 0.8-9 km terrain was 40-70 % in-scatter (a milky veil, graded mountain lit / shade
    # 1.9-2.2x vs the target 3.4-6.5x); the sky is Rayleigh-dominated (< 2 % change); the 9+ km ranges keep their
    # sky-hued veil (rFar in lighting.glsl applyAerial: target far range 0.91x the sky, in-scatter share .81)
    haze: float = 1.0
    cloud_cover: float = 0.48
    wind: tuple = (6.0, 1.5)
    water_wind: tuple = (0.6, -0.8)
    wave_strength: float = 1.0
    # (look2 r2: greener teal, target water h 194-200 not 211-230; r2d: 0.83 / 0.87 / 0.92 x: in the meadow view the
    #  lake body was 1.3-1.6x too bright and green from above (#578074 vs #3a626c); not lower: the shore water
    #  must stay vivid turquoise)
    # (r4 W1: x1.3 brighter and greener-cyan in-scatter with the bluer W_ABS - the shore-view water was a dull slate teal,
    #  ~0.05-0.10 L and 0.015-0.025 C below target; blue x1.3: shots 0 / 1 mid water h179 -> ~194, target h189)
    # (water5: with the clearer column (water.glsl W_ABS, water_shade wSca) red .36 -> .20, G .56 -> .50, B .65 -> .64: shot 0 mid
    #  water #457c79 C.059 -> #3d7b7b-ish C.064 (target #367e7d C.071), deep water bluer (target h216-223); .12 over-cut
    #  the red of the steeper views (r2 / r4 / r5 water R 15-35 below target), .34 brought the shore-view grey back)
    water_ins: tuple = (0.20, 0.50, 0.64)
    sky_boost: float = 22.0
    amb_boost: float = 2.5          # (r2b: 3.0 -> 4.5 with tune 0 3.0: distant shade HDR lit/shade 3.35 -> 2.9, haze share lit 16 % / shade 47 %)
                                    # (r5 light5: 4.5 -> 2.5 with tune 0 5 -> 1.5: shadows deeper and less blue, lit / shade at 1-5 km 2.1 -> 2.6-3.5)
    sky_desat: float = 0.06
    exposure: float = 1.0
    # look2 lighting / water knobs (uTune[16] in water.glsl / lighting.glsl):
    #  0 sky: horizon-band term on steep faces   1 sky: dome floor (n.y = -1)   2 valley-wall fill (1-ao)
    #  3 tree shadow sharpness (k in k d / t)    4 aerial: terrain-shadowed air (0 off .. 1)
    #  5 aerial: in-scatter kept in shadowed air 6 water: reflection smear by the unresolved wave slopes (x sigma)
    #  7 water: share of the smear drawn from the world-space ripple-group pattern where resolved (r2c; was the
    #    masking tilt)   8 water: min |rd.y| of the along-view footprint
    #  9 sun warmth (r2b; probe.frag: sun colour of a (1+w)x longer air path, same luminance)   10 water column: terrain shadow on in-scatter
    # 11 sky: ground bounce scale                12 tree shadow: crown gap amplitude (m)
    # 13 aerial in-scatter scale up to 3 km, ramping to 1 (sky.glsl calibration) at 15 km
    # 14 tree shadow: fine (0.15-0.5 m) gap amplitude (m), near the camera   15 tree shadow: near sharpness k
    # (r2b: 0 sky band 0.9 -> 3.0 (probe now samples 20-50 deg elevation), 2 valley-wall fill 0.7 -> 0.35,
    #  6 reflection smear 0.5, 13 near air 1 -> 0.4, 14 = 0.35 m, 15 = 50)
    # (r2c: 6 smear 0.5 -> 0.7 with 7 = 1: coherent horizontal ripple streaks instead of per-pixel grain)
    # 16 sky: band share near the camera   17 / 18 near -> far ramp (m from the camera)   19 sky: dome share near
    # 20 sky: extra warm ground bounce on steep faces near the camera (r2d, lighting.glsl nearSkyRamp)
    # 21 aerial: in-scatter chroma below ~8 km (1 = as computed, 0 = neutral grey veil)
    # 22 water: per-sample reflection smear share at steep views (1 = as at grazing views)
    # (r2d: near sky fill off (16 = 0, dome 19 = 0.4 within 150-1500 m, warm ground bounce 20 = 0.6): r1 boulder lit /
    #  shade 2.6 -> 4.2 HDR, shade neutral instead of blue; the near haze back to 1.0 (13) but near-white (21 = 0.5)
    #  and the far band 0 = 5: graded 3-5 km lit / shade 1.5-3.0 (was 1.9-4.3), lit (0.89, 1, 1.2), shade (0.68, 1, 1.6))
    # (r2d water: smear 6 = 0.7 -> 0.35 and 22 = 0.3: the far shore / peaks show in the lake as broken streaks
    #  instead of a uniform cyan; looking down (shot 6) soft cloud reflections instead of speckle)
    # (r4 W2 / W5 water: smear 6 = 0.35 -> 0.25, coherent share 7 = 1 -> 0.5, steep-view per-sample share 22 = 0.3 -> 0.6)
    # (r4 L14: 21 0.5 -> 0.3, lit rock not lavender)
    # (r5 light5: 0 sky band 5 -> 1.5, 9 sun warmth 0 -> 0.4 (lit rock / beach / grass warm, the sky-filled
    #  lit surfaces read cool-neutral), 13 near in-scatter 1 -> 0.35)
    tune: tuple = (1.5, 0.35, 0.35, 20.0, 1.0, 0.35, 0.25, 0.5, 0.003, 0.4, 1.0, 1.5, 1.0, 0.35, 0.35, 50.0,
                   0.0, 150.0, 1500.0, 0.4, 0.6, 0.7, 0.6, 0.0)     # (r5 light5: 21 0.3 -> 0.7: with the thinner haze the remaining
                                                                       #  < 8 km veil keeps more of its sky blue instead of a milky white)

    # (r4 L3) > 0: sun elevation (deg) of the baked terrain shadow maps only (World.ensure_shadows: cast terrain
    # shadow, shadowed air, water column); shading, sun colour and the near-camera occluder march keep
    # sun_elevation. Golden hour: the valley shade line of a lower sun on the near shore while the lit faces
    # keep the warm 26 deg modelling. 0 = sun_elevation
    shadow_elevation: float = 0.0
    shadow_azimuth: float = -1.0        # (r4 L3) >= 0: azimuth of that shadow sun (< 0 = sun_azimuth)

    def sun_dir(self):
        az, el = math.radians(self.sun_azimuth), math.radians(self.sun_elevation)
        return np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])

    def shadow_dir(self):
        if self.shadow_elevation <= 0.0:
            return self.sun_dir()
        az = math.radians(self.shadow_azimuth if self.shadow_azimuth >= 0.0 else self.sun_azimuth)
        el = math.radians(self.shadow_elevation)
        return np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])


class World:
    def __init__(self):
        if not all((bake.CACHE / f).exists() for f in ("h_far.npy", "h_near.npy", "water_level.npy", "forest.npy")):
            raise RuntimeError(f"world not baked in {bake.CACHE}: run python tools/bake_final.py first")
        m = bake.bake_all()
        self.t = {
            "uHFar": gl.tex2d(m["hfar"]),
            "uHNear": gl.tex2d(m["hnear"]),
            "uAOFar": gl.tex2d(m["aof"].astype(np.float32)),
            "uAONear": gl.tex2d(m["aon"].astype(np.float32)),
            "uForest": gl.tex2d(m["forest"]),
            "uWaterLvl": gl.tex2d(m["level"], filt="nearest"),
        }
        n3 = m["noise3"]
        c = gl.ctx()
        t3 = c.texture3d(n3.shape[:3][::-1], 4, np.ascontiguousarray(n3.transpose(2, 1, 0, 3)).tobytes(), dtype="f1")
        t3.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t3.repeat_x = t3.repeat_y = t3.repeat_z = True
        self.t["uNoise3"] = t3
        wln = bake.CACHE / "water_level_near.npy"
        self.t["uWaterLvlNear"] = gl.tex2d(np.load(wln) if wln.exists() else np.full((4, 4), -1e4, np.float32))
        stn = bake.CACHE / "stream_near.npy"
        self.t["uStreamNear"] = gl.tex2d(np.load(stn).astype(np.float32) if stn.exists() else np.zeros((4, 4, 4), np.float32))
        # (falls lab, review r2) signed distance to the inflow stream's wet channel (features._stream_corridor):
        # rocks.glsl keeps its blocks out of the channel (rockfix v3); uStreamCorrBox set in the stream block
        stc = bake.CACHE / "stream_corr.npy"
        self.t["uStreamCorr"] = gl.tex2d(np.load(stc).astype(np.float32) if stc.exists() else np.full((4, 4, 4), 99.0, np.float32))
        self.t["uWaterLvl"].filter = (moderngl.LINEAR, moderngl.LINEAR)
        import features as _feat
        self.features = _feat.load_meta()
        trees_p = bake.CACHE / "trees.npy"
        if not trees_p.exists():
            self._bake_trees(trees_p)
        trees_a = np.load(trees_p)
        self.t["uTrees"] = gl.tex2d(trees_a, filt="nearest")
        # (perf4) tree-free space for the march / the tree shadows (trees.glsl treeTopAt): per 16 m block (4 x 4 tree cells)
        # R = upper bound of the tree tops over the block and its 8 neighbours, G = clearance (m) to the crowns of the
        # trees further out
        self.t["uTreeTop"] = gl.tex2d(tree_tops(trees_a), filt="nearest")
        del trees_a
        self._sun_key = None
        self.hfar = m["hfar"]
        self.hnear = m["hnear"]
        self.level = m["level"]

    def _bake_trees(self, path):
        print("baking tree cells ...", flush=True)
        p = gl.Pass("bake_trees.frag")
        for i, (k, v) in enumerate(self.t.items()):
            p.tex(k, v, i)
        c = gl.ctx()
        tex = c.texture((4096, 4096), 4, dtype="f4")
        fbo = c.framebuffer([tex])
        for y in range(0, 4096, 1024):
            for x in range(0, 4096, 1024):
                p.draw(fbo, (x, y, 1024, 1024)); c.finish()
        a = gl.read_tex(tex)
        np.save(path, a.astype(np.float32))
        print(f"  trees: {(a[...,1] != 0).sum()/1e6:.2f} M", flush=True)
        fbo.release(); tex.release()

    def ensure_shadows(self, sun_dir):
        key = tuple(np.round(np.asarray(sun_dir, np.float64), 4))
        if key == self._sun_key:
            return
        t0 = time.time()
        c = gl.ctx()
        p = gl.Pass("bake_shadow.frag")
        out = {}
        for name, src, half, res, tile in (("uShNear", "uHNear", bake.NEAR_HALF, 4096, 512), ("uShFar", "uHFar", bake.FAR_HALF, 4096, 512)):
            p.tex("uH", self.t[src], 0)
            p.tex("uHFar", self.t["uHFar"], 1)
            p.set("uCenter", (0.0, 0.0)); p.set("uHalf", half); p.set("uRes", float(res))
            p.set("uFarHalf", bake.FAR_HALF); p.set("uSunDir", tuple(map(float, sun_dir)))
            old = self.t.get(name)
            tex = old if old is not None else c.texture((res, res), 4, dtype="f4")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR); tex.repeat_x = tex.repeat_y = False
            fbo = c.framebuffer([tex])
            for y in range(0, res, tile):
                for x in range(0, res, tile):
                    p.draw(fbo, (x, y, tile, tile)); c.finish()
            fbo.release()
            self.t[name] = tex
        self._sun_key = key
        print(f"  shadow maps baked in {time.time()-t0:.1f}s", flush=True)

    def bind(self, p):
        for i, (k, v) in enumerate(self.t.items()):
            p.tex(k, v, i)

    def height(self, x, z):
        """CPU bilinear terrain height (near map if inside)."""
        def bil(a, half, x, z):
            res = a.shape[0]
            fx = (x + half) / (2 * half) * res - 0.5
            fz = (z + half) / (2 * half) * res - 0.5
            ix, iz = int(np.floor(fx)), int(np.floor(fz))
            tx, tz = fx - ix, fz - iz
            ix = np.clip(ix, 0, res - 2); iz = np.clip(iz, 0, res - 2)
            return ((a[iz, ix] * (1 - tx) + a[iz, ix + 1] * tx) * (1 - tz)
                    + (a[iz + 1, ix] * (1 - tx) + a[iz + 1, ix + 1] * tx) * tz)
        if max(abs(x), abs(z)) < bake.NEAR_HALF * 0.95:
            return float(bil(self.hnear, bake.NEAR_HALF, x, z))
        return float(bil(self.hfar, bake.FAR_HALF, x, z))

    def water(self, x, z):
        res = self.level.shape[0]
        ix = int((x + bake.FAR_HALF) / (2 * bake.FAR_HALF) * res)
        iz = int((z + bake.FAR_HALF) / (2 * bake.FAR_HALF) * res)
        return float(self.level[np.clip(iz, 0, res - 1), np.clip(ix, 0, res - 1)])


def tree_tops(a, blk=4, cell=4.0, reach=12.0):   # (r4 merge: trees4 skirt x1.15 + treeLowWiden x1.2 -> treeReach up to ~11.6 m)
    """(perf4) tree-free space per block of blk x blk tree cells (uTrees texel layout: [row = z cell, col = x cell],
    (baseY, signed H, crown R, crown base)): R = max over the block and its 8 neighbours of the highest possible
    point of any tree (trees.glsl treeFinish: H <= min(1.6 |H0|, 34), a broken top's snag <= 2.4 m above it; -1e4 = none),
    G = horizontal clearance (m) from any point of the block to the crowns (reach <= 12 m, treeReach) of the trees
    outside those 3 x 3 blocks (>= 7.4 m)."""
    n = a.shape[0] // blk
    occ = a[..., 1] != 0.0
    top = np.where(occ, a[..., 0] + np.minimum(1.6 * np.abs(a[..., 1]), 34.0) + 2.4, -1e4).astype(np.float32)
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
    # Chebyshev block distance k to the nearest occupied block (capped): clearance (max(k, 2) - 1) x block - reach
    k = np.where(ob, 0, 64).astype(np.int32)
    cur = ob.copy()
    for i in range(1, 64):
        cur = dil(cur, False)
        k = np.where((k == 64) & cur, i, k)
        if cur.all():
            break
    clr = (np.maximum(k, 2) - 1) * blk * cell - reach
    return np.stack([top3, clr.astype(np.float32)], -1)


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r


SCENE_KINDS = (0, 1, 2, 3, 4, 5, 6, 7)    # hit kinds K_SKY .. K_FALLSV (lighting.glsl, falls.glsl): one shading program each
GBUF_UNIT = 28                            # texture units of the G-buffer (above the world's maps)
KIND_NAMES = {0: "sky", 1: "terrain", 2: "tree", 3: "rock", 4: "water", 5: "cover", 6: "hero", 7: "flora"}
# per-pass GPU time: set Renderer.prof = {} before render(); it accumulates seconds per pass (tools/passprof.py)


class Renderer:
    def __init__(self, width, height, quality=1, world=None, tile=192):
        self.w, self.h = width, height
        self.world = world or World()
        self.tile = tile
        t0 = time.time()
        # split renderer (scene.frag): the primary trace writes a G-buffer, then one program per hit kind
        # shades its pixels into the accumulator (one program cost ~45-50 GB / 13-36 min to compile)
        self.trace = gl.Pass("scene.frag", {"QUALITY": quality, "SCENE_PASS": 99}, compile_gb=6, targets=2)
        self.kpass = [gl.Pass("scene.frag", {"QUALITY": quality, "SCENE_PASS": k}, compile_gb=6) for k in SCENE_KINDS]
        # (perf4) cone pre-pass (scene.frag SCENE_PASS 98): per 4 x 4 pixel block the distance the primary rays start at
        self.cone = gl.Pass("scene.frag", {"QUALITY": quality, "SCENE_PASS": 98}, compile_gb=6)
        self.scene = self.trace
        self.probe = gl.Pass("probe.frag")
        print(f"shaders compiled in {time.time()-t0:.1f}s", flush=True)
        self._first = True
        c = gl.ctx()
        self.acc = c.texture((width, height), 4, dtype="f4")
        self.fbo = c.framebuffer([self.acc])
        self.gtex = [c.texture((width, height), 4, dtype="f4") for _ in range(2)]
        for g in self.gtex:
            g.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.gfbo = c.framebuffer(self.gtex)
        self.ctex = c.texture(((width + 3) // 4, (height + 3) // 4), 1, dtype="f4")
        self.ctex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.cfbo = c.framebuffer([self.ctex])
        self.ptex = c.texture((3, 1), 4, dtype="f4")
        self.pfbo = c.framebuffer([self.ptex])
        self.world.bind(self.probe)

    def _progs(self):
        return [self.cone, self.trace] + self.kpass

    def _look_uniforms(self, p, look, t):
        p.set("uSunDir", tuple(look.sun_dir()))
        p.set("uShadowDir", tuple(map(float, look.shadow_dir())))     # (r4 L3) lighting.glsl terrain / tree shadow march
        p.set("uSunIntensity", look.sun_intensity)
        p.set("uHaze", look.haze)
        p.set("uCloudCover", look.cloud_cover)
        p.set("uWind", tuple(look.wind))
        p.set("uTime", float(t))
        ww = np.asarray(look.water_wind, np.float64)
        p.set("uWaterWind", tuple(map(float, ww / np.linalg.norm(ww))))
        p.set("uWaveStrength", look.wave_strength)
        p.set("uWaterIns", tuple(look.water_ins))
        f = getattr(self.world, "features", None)
        if f:
            wf = f["waterfall"]
            p.set("uWF", (wf["x"], wf["z"], wf["height"], wf["half"]))
            p.set("uWFDir", tuple(wf["dir"]))
            p.set("uWFNotch", tuple(wf["notch"]))
            # falls water paths (features.falls_paths): uFallK[48] = 2 x 24 knots (u, v, half width,
            # impact), uFallInfo[2] = (lip height, volume, 0, 0)
            # uFallBox[4] = per fall k: [2k] = (u, y, v) min, [2k+1] = (u, y, v) max of the local-frame
            # box (u inland, v along shore) holding its water sheet, ledge spray and base mist
            # (falls.glsl fallsVolume marches each fall inside its own box)
            fk = np.zeros((48, 4), np.float32); fi = np.zeros((2, 4), np.float32)
            fb = np.zeros((4, 4), np.float32)
            for k, fl in enumerate(f.get("falls", [])[:2]):
                kn = np.asarray(fl["knots"], np.float32)[:24]
                fk[k * 24: k * 24 + len(kn)] = kn
                fi[k] = (fl["top"], fl["vol"], 0.0, 0.0)
                wv = kn[:, 2] * 1.9 + 0.8          # (falls.glsl fallDensity: spray veil out to 1.9 half widths)
                b0 = kn[-1]
                MK = 1.4                                                # (round 3) mist x1.3 (falls.glsl mistDensity kMs) + margin
                mr = 2.0 * np.sqrt(4.0 + 3.0 * b0[2]) * MK             # mist radius along the shore (review r2: narrower)
                # (round 3 i9) outward reach of the base mist spread over the pool: centre 3.5 m out + 2 x sqrt(40) x 1.3
                # (falls.glsl mistDensity: outward scale 40, kMs 1.3, cut-off r2 = 4)
                MXO = 20.5
                fb[2 * k] = (min(kn[:, 0].min() - 3.6, b0[0] - MXO), -0.8, min((kn[:, 1] - wv).min(), b0[1] - mr), 0.0)
                # (mist reaches 2 x 3.8 m (A) / 2 x 2.8 m (B) up: falls.glsl mistDensity)
                fb[2 * k + 1] = (kn[:, 0].max() + 0.6, max(fl["top"] + 0.7, (7.0 if k == 0 else 5.0) * MK), max((kn[:, 1] + wv).max(), b0[1] + mr), 0.0)
            p.set("uFallK", fk); p.set("uFallInfo", fi); p.set("uFallBox", fb)
            p.set("uFallsOn", 1 if f.get("falls") else 0)
            trl = np.zeros((32, 4), np.float32)
            tl = np.asarray(f.get("trail", []), np.float32)[:32]
            if len(tl):
                trl[: len(tl), : tl.shape[1]] = tl
            p.set("uTrail", trl); p.set("uTrailN", int(len(tl)))
            # inflow stream polyline (features.place: x, z, bed, floodplain half width) for the
            # boulder shading of the stream bed / banks (falls.glsl streamDist): uStream[96],
            # uStreamN, uStreamBox = (xmin, zmin, xmax, zmax) of the boulder-strewn floodplain
            sa = np.asarray(f.get("stream", []), np.float32).reshape(-1, 4)[:96]
            stu = np.zeros((96, 4), np.float32); stu[: len(sa)] = sa
            p.set("uStream", stu); p.set("uStreamN", int(len(sa)))
            # (round 3) uStreamS[96] = arc length (m) of each polyline point from the first: curvilinear stream
            # coordinates for the 3D white water (falls.glsl streamDist -> gStreamS, wwField)
            ss = np.zeros(96, np.float32)
            if len(sa) > 1:
                ss[1: len(sa)] = np.cumsum(np.linalg.norm(np.diff(sa[:, :2], axis=0), axis=1))
            if "uStreamS" in p.prog:
                u_ = p.prog["uStreamS"]
                u_.write(ss[: max(int(getattr(u_, "array_length", 96)), 1)].tobytes())
            # uStreamMouth = (x, z, downstream unit dir x, z) where the stream bed crosses the lake
            # level (water_shade.glsl: outflow foam plume); zeros = none
            mo = (0.0, 0.0, 0.0, 0.0)
            if len(sa):
                m = float(sa[:, 3].max()) * 2.6 + 4.0
                p.set("uStreamBox", (float(sa[:, 0].min()) - m, float(sa[:, 1].min()) - m, float(sa[:, 0].max()) + m, float(sa[:, 1].max()) + m))
                for k in range(1, len(sa)):
                    if sa[k, 2] > 0.0 >= sa[k - 1, 2]:
                        s_ = float(-sa[k - 1, 2] / (sa[k, 2] - sa[k - 1, 2]))
                        mp = sa[k - 1, :2] + (sa[k, :2] - sa[k - 1, :2]) * s_
                        dn = sa[k - 1, :2] - sa[k, :2]; dn = dn / (np.linalg.norm(dn) + 1e-9)
                        mo = (float(mp[0]), float(mp[1]), float(dn[0]), float(dn[1]))
                        break
            else:
                p.set("uStreamBox", (1e9, 1e9, -1e9, -1e9))
            p.set("uStreamMouth", mo)
            cb = f.get("stream_corr_box")
            p.set("uStreamCorrBox", (float(cb[0]), float(cb[1]), 1.0 / float(cb[2]), 1.0 / float(cb[3])) if cb else (0.0, 0.0, 0.0, 0.0))
        else:
            p.set("uFallsOn", 0)
        heroes = getattr(self, "heroes", [])
        ha = np.zeros((6, 4), np.float32); hb = np.zeros((6, 4), np.float32)
        for i, hr in enumerate(heroes[:6]):
            ha[i] = hr[:4]; hb[i] = hr[4:8]
        p.set("uHeroA", ha); p.set("uHeroB", hb); p.set("uHeroN", len(heroes[:6]))
        p.set("uSkyBoost", look.sky_boost)
        p.set("uAmbBoost", look.amb_boost)
        p.set("uSkyDesat", look.sky_desat)
        if "uLab" in p.prog:                        # lab-only knobs (lighting.glsl uLab): ALPINE_LAB="i=v,i=v"
            import os
            la = np.zeros(16, np.float32)
            for kv in filter(None, os.environ.get("ALPINE_LAB", "").split(",")):
                k, v = kv.split("=")
                la[int(k)] = float(v)
            u = p.prog["uLab"]
            u.write(la[: max(int(getattr(u, "array_length", 16)), 1)].tobytes())
        if "uW2K" in p.prog:                        # (world5 lab only) terrain_shade.glsl W2K knobs: ALPINE_W2K="i=v,i=v"
            import os
            wk = np.zeros(32, np.float32)
            for kv in filter(None, os.environ.get("ALPINE_W2K", "").split(",")):
                k, v = kv.split("=")
                wk[int(k)] = float(v)
            u = p.prog["uW2K"]
            u.write(wk[: 4 * max(int(getattr(u, "array_length", 8)), 1)].tobytes())
        if "uTune" in p.prog:                       # (the compiler may trim unused trailing slots)
            tu = np.zeros(24, np.float32); tu[: len(look.tune)] = look.tune
            u = p.prog["uTune"]
            u.write(tu[: max(int(getattr(u, "array_length", 24)), 1)].tobytes())

    def probe_light(self, look, pos, t=0.0):
        p = self.probe
        self._look_uniforms(p, look, t)
        p.set("uProbePos", tuple(map(float, pos)))
        p.draw(self.pfbo)
        a = gl.read_tex(self.ptex, 3)[0]
        return a[0], a[1], a[2]

    debug = 0

    def render(self, cams, look, times, spp=None, verbose=False):
        """cams/times: one per sample (for motion blur), or a single Camera/float."""
        if isinstance(cams, Camera):
            cams = [cams] * (spp or 1)
        if not isinstance(times, (list, tuple, np.ndarray)):
            times = [times] * len(cams)
        n = len(cams)
        c = gl.ctx()
        self.world.ensure_shadows(look.shadow_dir())
        progs = self._progs()
        sun_col, sky_amb, horizon = self.probe_light(look, cams[0].pos, times[0])
        for p in progs:
            self.world.bind(p)
            p.tex("uGBuf0", self.gtex[0], GBUF_UNIT)
            p.tex("uGBuf1", self.gtex[1], GBUF_UNIT + 1)
            p.tex("uCone", self.ctex, GBUF_UNIT + 2)
            p.set("uSunCol", tuple(map(float, sun_col)))
            p.set("uSkyAmb", tuple(map(float, sky_amb)))
            p.set("uSkyHorizon", tuple(map(float, horizon)))
            p.set("uRes", (float(self.w), float(self.h)))
            p.set("uDebug", int(self.debug))
        self.fbo.use()
        self.fbo.clear(0, 0, 0, 0)
        c.blend_func = moderngl.ONE, moderngl.ONE
        t0 = time.time()
        for s in range(n):
            cam = cams[s]
            jx, jy = (halton(s + 1, 2) - 0.5, halton(s + 1, 3) - 0.5) if n > 1 else (0.0, 0.0)
            for p in progs:
                self._look_uniforms(p, look, times[s])
                p.set("uJitter", (jx, jy))
                p.set("uSample", s)
                p.set("uCamPos", tuple(map(float, cam.pos)))
                p.set("uCamRot", cam.basis().astype("f4"))
                p.set("uTanHalfFov", math.tan(math.radians(cam.fov) / 2))
                p.set("uPixAng", 2.0 * math.tan(math.radians(cam.fov) / 2) / self.h)
                p.set("uAperture", float(cam.aperture))
                p.set("uFocusDist", float(cam.focus))
            prof = self.prof if getattr(self, "prof", None) is not None else None
            for y in range(0, self.h, self.tile):
                for x in range(0, self.w, self.tile):
                    rect = (x, y, min(self.tile, self.w - x), min(self.tile, self.h - y))
                    c.disable(moderngl.BLEND)
                    tp = time.time()
                    self.cone.draw(self.cfbo, (x // 4, y // 4, (rect[2] + 3) // 4, (rect[3] + 3) // 4))
                    if prof is not None:
                        c.finish()
                        prof["cone"] = prof.get("cone", 0.0) + time.time() - tp
                        tp = time.time()
                    self.trace.draw(self.gfbo, rect)
                    # (perf4) the hit kinds present in this tile (G-buffer readback = the tile's one sync point): a kind
                    # pass with no pixel here is skipped (it would only discard every pixel), the others are queued
                    # back to back without a c.finish() each (under GPU sharing every finish waited for other
                    # processes' work: ~7 % of the frame per empty pass)
                    kb = np.frombuffer(self.gfbo.read(viewport=rect, components=2, attachment=0, dtype="f4"), np.float32)[1::2]
                    kin = np.bincount(np.clip(kb.astype(np.int32), 0, 15), minlength=16)
                    if prof is not None:
                        prof["trace"] = prof.get("trace", 0.0) + time.time() - tp
                    c.enable(moderngl.BLEND)
                    for k, p in zip(SCENE_KINDS, self.kpass):
                        if kin[k] == 0:
                            continue
                        tp = time.time()
                        p.draw(self.fbo, rect)
                        if prof is not None:
                            c.finish()
                        if prof is not None:
                            nm = KIND_NAMES[k]
                            prof[nm] = prof.get(nm, 0.0) + time.time() - tp
            if verbose:
                print(f"  sample {s+1}/{n} {time.time()-t0:.1f}s", flush=True)
            if getattr(self, "_first", False):
                self._first = False
                print(f"  first sample (incl. any lazy driver compile) {time.time()-t0:.1f}s", flush=True)
        c.disable(moderngl.BLEND)
        img = gl.read_tex(self.acc, 3)[::-1] / n
        return img, dict(sun=sun_col, sky=sky_amb, horizon=horizon, seconds=time.time() - t0)
