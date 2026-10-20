"""Photographic finishing modelled on a phone camera's Smart-HDR look:
local tone mapping (compressed base, boosted detail), filmic S-curve with lifted chromatic toe,
mid-tone chroma boost with greens pushed toward yellow-olive, subtle bloom, vignette, grain, sRGB."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@dataclass
class Grade:
    exposure: float = 1.0
    # (look2 r2b: refit (tools/l2_fit.py on out/l2/it8, 10 targets + 9 shots: loss 2.52 -> 1.99), then fine_gain
    #  1.69 -> 1.3 and struct_gain 1.0 -> 0.85 by hand (film-safe fine detail, compressed distant lit / shade): 2.12)
    # (look2 r2c: detail 1.85 -> 1.6, struct 0.85 -> 0.7, shadow lift 0.45, black point 1.0 / 0.08 with chroma
    #  kept, then base / contrast / hi / mid grey / saturation refit (tools/l2_fit.py on out/l2/s4b, 14 midday
    #  views, loss 1.84 -> 1.27): the distant lit / shade spread (graded 3-5 km, shot 7) 5.2 -> ~4.0, shade
    #  (0.38, 1, 2.3) #213855 -> (0.48, 1, 2.1) #2b405c, no crunchy ribs)
    # (look2 r2d: refit (tools/l2_fit.py on out/l2/s5q/K = near sky fill off + whiter near haze, 19 views, loss 3.38 ->
    #  2.42) with the fading black point; then shadow lift 0 -> 0.3, black level 21 -> 12, contrast 1.16 -> 1.24, black pivot 0.18 by hand
    #  (the fit alone re-expanded the distant lit / shade of shot 7 to 4.0; now 3.0, shots 0 / 1 2.1-2.2, target 1.5))
    # (r5 light5) the r4 grade (base .8 / hi .6 / struct .6 / contrast 1.24 / sat 1.07) crushed the top of the tone range
    # (graded p90 147-160, p98 183-205 vs the target 180-188 / 215-230) and the colour (sat mean .21-.29 vs .30-.35): the
    # "too grey" frames. The directly seen sky is now darkened in the HDR (lighting.glsl SKY_VIEW_GAIN) instead of by a
    # hard base compression above the pivot, so lit rock / beach / snow keep their brightness
    base_compress: float = 0.90     # (r5: 0.8 -> 0.9) local tone mapping: scale of the large-scale log-luminance (r4: 0.5 -> 0.8 with the edge-aware base: dark masses kept dark, p2 ~12)
    detail_boost: float = 1.35      # local contrast, texture band (fine_sigma .. struct_sigma) (r4 L1: 1.6 -> 1.35)
    local_sigma: float = 0.04       # base blur, fraction of image width
    contrast: float = 1.30          # (r5: 1.24 -> 1.3)
    saturation: float = 1.12        # global chroma scale (r5: 1.07 -> 1.12)
    chroma_boost: float = 0.15      # extra chroma in mid-tones (r5: 0.05 -> 0.15)
    green_shift: float = 3.0        # degrees: push greens (h 125-160) toward yellow-olive (r5: 7 -> 3: dark foliage h 113 vs the target 119-129)
    white_balance: tuple = (1.0, 1.0, 1.0)
    toe_lift: float = 0.0
    bloom: float = 0.03
    vignette: float = 0.18
    grain: float = 0.005
    mid_grey: float = 0.107         # where the base compression pivots (linear Y after exposure)
    hi_compress: float = 0.85       # (r5: 0.6 -> 0.85) base compression above the pivot (sky, cloud, snow masses); <0 = base_compress (r4 L2: 0.9 -> 0.6)
    knee: float = 0.90              # soft shoulder of the brightest channel (linear), asymptote 1.0 (r4 L2 / L4: 0.8 -> 0.9)
    edge_eps: float = 0.25          # (r4 L1: 1.0 -> 0.25: a 2-stop tree / sky step is an edge) >0: edge-aware base (self-guided filter, eps in log2 stops^2): no bright
                                    # halos in the sky around dark ridges / tree walls; 0 = plain Gaussian base
    edge_mix: float = 1.0           # share of the edge-aware base (rest Gaussian) (r4 L1: 0.5 -> 1)
    # (r2b) the detail layer is split: a fine band (below fine_sigma, fraction of the width: ~1.5 px at 640,
    # ~8 px at 3440 -> the 640 previews and the 21:9 final get the same look) with its own, lower gain:
    # detail_boost x contrast on 1-2 px detail frosted the distant forest, grained the water, risked shimmer
    fine_sigma: float = 0.0023
    fine_gain: float = 1.4          # (r4: 1.3 -> 1.4, linear (fine_amp 0): ~2-4 px phone-style edge crispness)
    # (r2b) structure band (struct_sigma .. local_sigma: lit / shaded faces of ribs and gullies, a dark stand
    # against a meadow) gets its own gain: the look compresses big light/shadow masses (distant
    # lit / shade only 1.7x) while the micro-detail stays crisp; detail_boost x contrast on this band
    # had pushed the graded lit / shade at 3-5 km to 7-10x (shade #0d2540 vs the target #5c729d)
    struct_sigma: float = 0.010
    struct_gain: float = 1.0        # (r4 L1 / L12: 0.7 -> 0.6) (r5: 0.6 -> 1.0: the lit / shaded faces of the 1-5 km massif were flattened to 1.9-2.2x vs the target 3.4-6.5x)
    knee_top: float = 0.99          # (r4 L13: 0.93 -> 0.99) asymptote of the max-channel shoulder (0.975 left 1/3 of a bright sky at B >= 250)
    # (r2b) black point: the darkest black_pct % of the frame (luminance after the curve) is pulled toward
    # black_level (sRGB code value) by a linear offset (flare / veiling-haze removal), at most black_max
    black_auto: float = 1.0
    black_pct: float = 1.0
    black_level: float = 8.0        # (r4: 12 -> 8)
    black_max: float = 0.10         # (r4: 0.08 -> 0.1)
    # (r2c) the black point scales each pixel's RGB by (Y - b) / Y (chroma ratios kept) instead of subtracting b
    # per channel, which over-saturated the dark blue shade (graded distant shade (0.38, 1, 2.3) vs the
    # target (0.64, 1, 2.0)); 0 = old per-channel offset
    black_chroma_keep: float = 1.0
    # (r2d) the offset fades out toward black_pivot (display-linear Y): sub = b (1 - Y / pivot)^2. A plain
    # linear offset of up to 0.08 also took 0.08 off the sky-lit distant shade (Y ~0.11 -> 0.04, -1.5 stops)
    # in frames without deep darks (shot 7: graded 3-5 km lit / shade 2.27 after the curve -> 4.3 after the
    # black point; target 1.7). 0 = linear offset (r2c)
    black_pivot: float = 0.18
    # (r2c) Smart-HDR shadow lift: + shadow_lift stops (Gaussian bump in log2 Y after the local tone mapping,
    # centred at shadow_center, width shadow_width stops; monotonic for lift < ~1.1 x width). Raises the sky-lit
    # shade (distant shaded faces, Y ~0.03-0.08 -> target #5c729d, luma ~110, only 0.75 stop under the lit
    # rock) while the deepest tones (spruce interiors, crevices, p2) and the lit masses stay put
    shadow_lift: float = 0.35       # (r5: 0.3 -> 0.35)
    shadow_center: float = -4.5
    shadow_width: float = 1.3
    # (r4 L1) halo-free local tone mapping: edge-aware (guided, eps in stops^2) structure / fine blurs
    # (0 = Gaussian) and amplitude-limited band gains (amp_limit stops; 0 = linear gains)
    struct_eps: float = 0.10
    fine_eps: float = 0.0
    amp_limit: float = 0.5
    fine_amp: float = 0.18         # amplitude limit of the fine band (< 0: amp_limit; 0: linear) - (r5 qa5 B09: 0 overshot ~+20 levels at trunk / rock edges against water)
    # (r4 L4) highlight chroma roll-off from OKLab L chroma_roll0 over chroma_roll_w (was 0.78 / 0.18)
    chroma_roll0: float = 0.86
    chroma_roll_w: float = 0.12
    # (r4 L13) fine highlights ride above the max-channel shoulder (scale; 0 = shoulder on everything)
    hi_detail: float = 1.0
    hi_boost: float = 1.0           # (r4 L13) highlight expansion strength above display Y hi_t (0 = off)
    hi_t: float = 0.38
    # (r4 L13) >= 0: integral base mapping with the slope easing from hi_compress to hi2_compress between hi2_lo and
    # hi2_hi stops above the pivot (cloud tops, snow fields); < 0 = (base - pivot) x bc as before
    hi2_compress: float = 1.0
    hi2_lo: float = 2.8
    hi2_hi: float = 3.5
    # (r5) hue-selective chroma (OKLCh): the turquoise water band (h ~165-220) keeps less of the global boost (graded water C
    # .07-.11 vs the target .054-.072: a flat cyan instead of the intended teal)
    cyan_sat: float = 0.80
    cyan_h: float = 192.0
    cyan_w: float = 32.0


def _blur_small(a, sigma, scale=4):
    h, w = a.shape[:2]
    small = a[::scale, ::scale]
    sg = (sigma / scale, sigma / scale) + (0,) * (a.ndim - 2)
    b = ndimage.gaussian_filter(small, sg, mode="nearest")
    zf = (h / b.shape[0], w / b.shape[1]) + (1,) * (a.ndim - 2)
    return ndimage.zoom(b, zf, order=1)[:h, :w]


def _guided_base(logl, sigma, eps, scale=4):
    """edge-preserving blur of log2 luminance: self-guided filter (He et al.; fast variant: coefficients on a
    1/scale area-averaged image, applied to the full-res guide). (r4 L1) Gaussian windows of std sigma (px)
    instead of box means: the box windows (~244 px at 1720) left rectangular halo footprints."""
    h, w = logl.shape
    hs, ws = h // scale, w // scale
    I = logl[: hs * scale, : ws * scale].reshape(hs, scale, ws, scale).mean((1, 3)).astype(np.float32) if scale > 1 \
        else logl.astype(np.float32)
    s = max(sigma / scale, 0.5)
    gf = lambda a: ndimage.gaussian_filter(a, s, mode="nearest", truncate=3.0)
    mI = gf(I)
    var = np.maximum(gf(I * I) - mI * mI, 0.0)
    a = var / (var + eps)
    b = mI - a * mI
    a, b = gf(a), gf(b)
    if scale > 1:
        zf = (h / hs, w / ws)
        a = ndimage.zoom(a, zf, order=1, mode="nearest", grid_mode=True)[:h, :w]
        b = ndimage.zoom(b, zf, order=1, mode="nearest", grid_mode=True)[:h, :w]
    return a * logl + b


def _amp_gain(d, gain, amp):
    """(r4 L1) amplitude-limited band gain: small (texture) amplitudes get `gain`, steps over ~2 amp (stops)
    pass at gain 1, so a tree / sky or trunk / water edge is never over- or under-shot (no halos, no dark rims)."""
    if amp <= 0.0 or gain == 1.0:
        return d * gain
    return d * (1.0 + (gain - 1.0) * np.exp(-np.square(d / amp)))


def filmic(x):
    # smooth toe + long shoulder (Hable), white point ~11
    A, B, C, D, E, F = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30

    def f(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F
    return f(x) / f(11.2)


# --- OKLab ---------------------------------------------------------------
M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
M1i = np.linalg.inv(M1).astype(np.float32)
M2i = np.linalg.inv(M2).astype(np.float32)


def to_oklab(rgb):
    lms = np.cbrt(np.maximum(rgb @ M1.T, 0.0))
    return lms @ M2.T


def from_oklab(lab):
    lms = lab @ M2i.T
    return (lms ** 3) @ M1i.T


def finish(hdr, g: Grade, seed=0):
    img = np.maximum(hdr.astype(np.float32), 0.0) * g.exposure
    img *= np.asarray(g.white_balance, np.float32)
    h, w = img.shape[:2]
    if g.bloom > 0:
        bright = np.maximum(img - 1.5, 0.0)
        bl = _blur_small(bright, w * 0.006, 2) * 0.6 + _blur_small(bright, w * 0.03, 4) * 0.4
        img = img + bl * g.bloom * 4.0
    # local tone mapping in log2 luminance
    lum = img @ LUMA
    logl = np.log2(lum + 1e-5)
    if g.edge_eps > 0.0 and g.edge_mix >= 1.0:
        base = _guided_base(logl, w * g.local_sigma, g.edge_eps)
    else:
        base = _blur_small(logl, w * g.local_sigma, 4)
        if g.edge_eps > 0.0:
            base = base + (_guided_base(logl, w * g.local_sigma, g.edge_eps) - base) * g.edge_mix
    pivot = np.log2(g.mid_grey)
    bc = g.base_compress
    if g.hi_compress >= 0.0:
        # Smart-HDR: bright masses (sky, cloud, snow fields) are pulled down harder than the shadows are
        # lifted; smooth over +-0.75 stop around the pivot so the curve stays monotonic and kink-free
        k = np.clip((base - pivot) / 1.5 + 0.5, 0.0, 1.0)
        k = k * k * (3.0 - 2.0 * k)
        bc = g.base_compress + (g.hi_compress - g.base_compress) * k
    lp = ls = logl
    if g.fine_gain != g.detail_boost and g.fine_sigma > 0.0:
        sf = max(w * g.fine_sigma, 0.6)
        if g.fine_eps > 0.0:
            lp = _guided_base(logl, sf, g.fine_eps, 1 if sf < 3.0 else 2)
        else:
            lp = ndimage.gaussian_filter(logl, sf, mode="nearest") if sf < 3.0 else _blur_small(logl, sf, 2)
    ls = lp
    if g.struct_gain != g.detail_boost and g.struct_sigma > 0.0:
        ss = max(w * g.struct_sigma, 1.0)
        if g.struct_eps > 0.0:
            # (r4 L1) edge-aware structure blur (same guide as the base): a tree / sky step stays out of the
            # detail band instead of being boosted into a 15-40 px glow
            ls = _guided_base(logl, ss, g.struct_eps, 4 if ss >= 12.0 else (2 if ss >= 6.0 else 1))
        else:
            ls = ndimage.gaussian_filter(lp, ss, mode="nearest") if ss < 4.0 else _blur_small(lp, ss, 2)
    am = g.amp_limit
    if g.hi2_compress >= 0.0:
        # (r4 L13) base mapping as the integral of the local slope bc(t): base_compress below the pivot, hi_compress
        # over the sky band, easing to hi2_compress between hi2_lo and hi2_hi stops above the pivot, so cloud tops
        # and snow fields (~1.4 stops over the sky in the HDR) keep the target cloud / sky ~2 (monotonic, kink-free)
        tt = np.linspace(-16.0, 16.0, 3201, dtype=np.float32)
        s1 = np.clip((tt + 0.75) / 1.5, 0.0, 1.0); s1 = s1 * s1 * (3.0 - 2.0 * s1)
        s2 = np.clip((tt - g.hi2_lo) / max(g.hi2_hi - g.hi2_lo, 1e-3), 0.0, 1.0); s2 = s2 * s2 * (3.0 - 2.0 * s2)
        sl = g.base_compress + (g.hi_compress - g.base_compress) * s1 + (g.hi2_compress - g.hi_compress) * s2
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (sl[1:] + sl[:-1]) * (tt[1] - tt[0]))]).astype(np.float32)
        cum -= np.interp(0.0, tt, cum)
        bterm = np.interp(base - pivot, tt, cum).astype(np.float32)
    else:
        bterm = (base - pivot) * bc
    new = (pivot + bterm + _amp_gain(ls - base, g.struct_gain, am) + _amp_gain(lp - ls, g.detail_boost, am)
           + _amp_gain(logl - lp, g.fine_gain, am if g.fine_amp < 0.0 else g.fine_amp))
    if g.contrast != 1.0:
        new = pivot + (new - pivot) * g.contrast
    if g.shadow_lift:
        new = new + g.shadow_lift * np.exp(-np.square((new - g.shadow_center) / g.shadow_width))
    img = img * (np.exp2(new) / (lum + 1e-5))[..., None]
    # hue-preserving filmic curve with a partial per-channel shoulder (natural highlight roll-off)
    lum = img @ LUMA
    lt = filmic(lum * 2.0)
    hp = img * (lt / (lum + 1e-6))[..., None]
    pc = filmic(img * 2.0)
    k = np.clip(lt * 0.5, 0, 1)[..., None]
    out = hp * (1 - k) + pc * k
    out = np.maximum(out, 0.0)
    if g.hi_boost > 0.0:
        # (r4 L13) highlight expansion above display Y hi_t: u -> u + s u^2 (1 - u) (smooth at hi_t, monotonic for s < 1):
        # clouds / snow / white water back to the target 205-245 after the sky compression (hi_compress 0.6 left
        # the cloud tops at ~205, cloud / sky 1.5-1.7 vs the target 1.8-2.0); the sky (Y ~0.34) is untouched
        yo = out @ LUMA
        u = np.clip((yo - g.hi_t) / (1.0 - g.hi_t), 0.0, None)
        uc = np.minimum(u, 1.0)
        ynew = np.where(u > 0.0, yo + (1.0 - g.hi_t) * g.hi_boost * uc * uc * (1.0 - uc), yo)
        out = out * (ynew / np.maximum(yo, 1e-6))[..., None]
    # bright saturated colours (deep-blue sky under a lifted exposure): a channel running past ~0.9 is
    # brought back by desaturating toward the pixel's luminance, not clipped (the sky stays a
    # saturated mid-blue at B ~ 235-245)
    # (r4 L13) the shoulder acts on the base layer only: fine highlights standing > 0.3 stop above their
    # surroundings in bright (exposed Y > 1) areas (glints, waterfall cores, the brightest cloud rims) ride
    # on top of it and may clip (target: 0.05-0.3 % clipped, p99.9 248-253); 0 = shoulder on the whole image
    hd = None
    if g.hi_detail > 0.0:
        hd = np.clip(logl - lp - 0.3, 0.0, 2.0) * np.clip(logl, 0.0, 1.0) * g.hi_detail
        if np.any(hd > 0.0):
            hd = np.exp2(hd)[..., None]
            out = out / hd
        else:
            hd = None
    lo = out @ LUMA
    mx = out.max(-1)
    knee = g.knee
    top = max(g.knee_top, knee + 0.02)
    over = mx > knee
    if np.any(over):
        lo_c = np.minimum(lo, knee - 1e-3)
        tgt = knee + (top - knee) * np.tanh((mx - knee) / (2.0 * (top - knee)))   # soft shoulder of the max channel
        f = np.where(over, (tgt - lo_c) / np.maximum(mx - lo_c, 1e-6), 1.0)
        out = np.where(over[..., None], lo_c[..., None] + (out - lo_c[..., None]) * f[..., None], out)
    if hd is not None:
        out = out * hd
    if g.black_auto > 0.0:
        # black point from the frame's darkest tones (stable under camera motion: a low percentile of the
        # whole frame); linear offset, so lifted veiled shadows regain depth and colour
        yb = out[::2, ::2] @ LUMA
        p = float(np.percentile(yb, g.black_pct))
        bl = ((g.black_level / 255.0 + 0.055) / 1.055) ** 2.4
        if g.black_pivot > 0.0:
            # fading offset: p -> bl, sub(Y) = b (1 - Y / pivot)^2 (monotonic, 0 above the pivot); chroma kept
            fp = max(1.0 - p / g.black_pivot, 0.05) ** 2
            b = float(np.clip((p - bl) / fp, 0.0, g.black_max)) * g.black_auto
            if b > 0.0:
                yo = out @ LUMA
                sub = b * np.square(np.clip(1.0 - yo / g.black_pivot, 0.0, 1.0))
                out = out * (np.maximum(yo - sub, 0.0) / np.maximum(yo, 1e-6))[..., None]
            b = 0.0
        else:
            b = float(np.clip((p - bl) / max(1.0 - bl, 1e-3), 0.0, g.black_max)) * g.black_auto
        if b > 0.0:
            off = np.maximum(out - b, 0.0) / (1.0 - b)
            if g.black_chroma_keep > 0.0:
                yo = out @ LUMA
                sc = np.maximum(yo - b, 0.0) / ((1.0 - b) * np.maximum(yo, 1e-6))
                keep = out * sc[..., None]
                off = off + (keep - off) * g.black_chroma_keep
            out = off
    out = out + g.toe_lift * (1.0 - out) * np.exp(-out * 25.0)
    # chroma in OKLab: mid-tone boost, roll-off in highlights, greens -> yellow-olive
    lab = to_oklab(out)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    wmid = np.clip((L - 0.12) / 0.25, 0, 1) * (1 - np.clip((L - g.chroma_roll0) / g.chroma_roll_w, 0, 1))
    C = C * g.saturation * (1.0 + g.chroma_boost * wmid)
    if g.cyan_sat != 1.0:
        dh = np.abs((np.degrees(hue) - g.cyan_h + 180.0) % 360.0 - 180.0)
        C = C * (1.0 + (g.cyan_sat - 1.0) * np.clip(1.0 - dh / g.cyan_w, 0.0, 1.0))
    if g.green_shift:
        hd = np.degrees(hue)
        wg = np.clip(1 - np.abs(hd - 142.0) / 28.0, 0, 1)
        hue = hue - np.radians(g.green_shift) * wg
    lab[..., 1] = C * np.cos(hue)
    lab[..., 2] = C * np.sin(hue)
    boosted = from_oklab(lab)
    # keep the boost inside the gamut: where a channel would clip, take only the part of the chroma
    # change that fits (no flat 255 patches in saturated skies / water)
    # (r5 light5) and where a channel would go negative (saturated dark yellow-greens: forest floor B 6-11, neon), likewise
    hi = boosted > 0.97
    lo = boosted < 0.0
    if np.any(hi) or np.any(lo):
        d = boosted - out
        tt = np.where(hi & (d > 1e-6), (0.97 - out) / np.maximum(d, 1e-6), 1.0)
        tt = np.minimum(tt, np.where(lo & (d < -1e-6), out / np.maximum(-d, 1e-6), 1.0))
        tt = np.clip(tt.min(-1), 0.0, 1.0)[..., None]
        boosted = out + (boosted - out) * tt
    out = np.clip(boosted, 0.0, 1.0)
    if g.vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 * 0.7 + ((yy - h / 2) / (h / 2)) ** 2 * 0.7
        out *= (1.0 - g.vignette * np.clip(r2 - 0.25, 0, 2) ** 1.5 * 0.6)[..., None]
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * np.power(np.maximum(out, 0), 1 / 2.4) - 0.055)
    rng = np.random.default_rng(seed)
    if g.grain > 0:
        n = rng.standard_normal((h, w, 1)).astype(np.float32)
        srgb = srgb + n * g.grain * (0.4 + 0.6 * np.sqrt(np.clip(srgb, 0, 1) * (1 - np.clip(srgb, 0, 1)) * 4))
    srgb = srgb + (rng.random((h, w, 3)).astype(np.float32) - 0.5) / 255.0
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def _sky_median(img):
    """median luma of the clear blue sky in a finished sRGB frame (upper 60 %, blue-dominant), None if < 3 %."""
    a = img[: int(img.shape[0] * 0.6)].astype(np.float32)
    m = (a[..., 2] > a[..., 0] + 30.0) & (a[..., 2] >= a[..., 1]) & (a[..., 2] > 140.0)
    if m.sum() < 0.03 * img.shape[0] * img.shape[1]:
        return None
    return float(np.median(a[m] @ np.array([0.2126, 0.7152, 0.0722])))


def suggest_exposure(hdr, target_luma=112.0, g=None, lo_cap=0.25, hi_cap=2.6, sky_max=158.0, sky_cut=0.8):
    """exposure that puts the median luma of the finished frame near target (bisection); (r2b) capped so the
    clear sky's median stays at or below sky_max (target: zenith ~164, sky median luma ~158), by at most a
    factor sky_cut (forest / finale frames with a bright sky patch keep their exposure)."""
    g = g or Grade()
    small = hdr[::4, ::4]
    L = np.array([0.2126, 0.7152, 0.0722])

    def fin(e):
        gg = Grade(**{**g.__dict__, "exposure": e, "grain": 0.0, "vignette": 0.0, "bloom": 0.0})
        return finish(small, gg)

    lo, hi = 0.1, 10.0
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        m = np.median(fin(mid).astype(np.float32) @ L)
        if m < target_luma:
            lo = mid
        else:
            hi = mid
    e = float(np.clip(np.sqrt(lo * hi), lo_cap, hi_cap))
    if sky_max:
        s = _sky_median(fin(e))
        if s is not None and s > sky_max:
            lo, hi = max(lo_cap, e * sky_cut), e
            for _ in range(8):
                mid = np.sqrt(lo * hi)
                s = _sky_median(fin(mid))
                if s is None or s <= sky_max:
                    lo = mid
                else:
                    hi = mid
            e = float(np.sqrt(lo * hi))
    return e
