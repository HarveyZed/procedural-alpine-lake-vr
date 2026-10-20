"""analysis.py - objective checks for the procedural soundtrack.

* BS.1770-4 K-weighted loudness (integrated, short-term, momentary, per-segment)
* sample peak, 4x-oversampled true peak, DC offset, clipped-sample count
* click scan (isolated sample-to-sample jumps relative to the local diff RMS)
* looping scan (autocorrelation of band envelopes; strong peaks = repetition)
* log-frequency spectrogram PNGs (pillow only) with shot boundaries and a
  loudness strip

Usable as a module (soundtrack.py --analyze) or standalone:
    python audio/analysis.py out/audio/soundtrack.wav [--shots audio/film_shots.json]
"""
import argparse
import json
import math
import os
import wave

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont

SR = 48000

# ITU-R BS.1770-4 K-weighting, coefficients as published for fs = 48 kHz
_K_SOS = np.array([
    [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
])


# ------------------------------------------------------------------ loudness
def k_weight(x):
    return signal.sosfilt(_K_SOS, np.atleast_2d(x), axis=-1)


def _block_power(xk, block_s, hop_s):
    """Channel-summed mean square of K-weighted signal per block (G=1 for L/R)."""
    xk = np.atleast_2d(xk)
    n = xk.shape[-1]
    B, H = int(round(block_s * SR)), int(round(hop_s * SR))
    if n < B:
        return np.array([np.sum(np.mean(xk ** 2, axis=-1))]), np.array([n / SR / 2])
    c = np.concatenate([np.zeros((xk.shape[0], 1)), np.cumsum(xk ** 2, axis=-1)], axis=-1)
    st = np.arange(0, n - B + 1, H)
    p = ((c[:, st + B] - c[:, st]) / B).sum(axis=0)
    return p, (st + B / 2) / SR


def _lufs(p):
    return -0.691 + 10.0 * np.log10(np.maximum(p, 1e-20))


def integrated_lufs(x=None, xk=None):
    xk = k_weight(x) if xk is None else xk
    p, _ = _block_power(xk, 0.4, 0.1)
    p = p[_lufs(p) > -70.0]
    if len(p) == 0:
        return -math.inf
    rel = _lufs(p.mean()) - 10.0
    p = p[_lufs(p) > rel]
    return float(_lufs(p.mean())) if len(p) else -math.inf


def short_term(xk, hop=0.1):
    p, t = _block_power(xk, 3.0, hop)
    return t, _lufs(p)


def momentary(xk, hop=0.05):
    p, t = _block_power(xk, 0.4, hop)
    return t, _lufs(p)


def segment_lufs(xk, t0, t1):
    """Ungated mean K-weighted level of a time segment (LUFS-like)."""
    i0, i1 = int(max(0, t0) * SR), int(min(xk.shape[-1] / SR, t1) * SR)
    if i1 <= i0:
        return -math.inf
    return float(_lufs(np.sum(np.mean(xk[:, i0:i1] ** 2, axis=-1))))


# ------------------------------------------------------------------ peaks etc
def sample_peak_db(x):
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-20))


def true_peak_db(x):
    up = signal.resample_poly(np.atleast_2d(x), 4, 1, axis=-1)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-20))


def click_scan(x, win=241, ratio_thr=9.0, min_jump=2e-3):
    """Isolated jumps: |x[n]-x[n-1]| relative to the RMS of the diff in a +-2.5 ms
    window. A lone discontinuity scores ~sqrt(win)=15.5; noise/transients ~<6."""
    x = np.atleast_2d(x)
    d = np.abs(np.diff(x, axis=-1))
    loc = np.sqrt(np.maximum(uniform_filter1d(d ** 2, win, axis=-1), 0.0) + 1e-20)
    r = d / loc
    r[d < min_jump] = 0.0
    hits = np.argwhere(r > ratio_thr)
    times = sorted(set(round(float(i) / SR, 3) for _, i in hits))
    # second detector: isolated energy above 15 kHz relative to its 50 ms RMS. Clicks are
    # broadband and stand out there; designed micro-transients (grasshopper pulses, needle
    # grains) also register, so hits must be checked against the stems.
    hp = signal.sosfilt(signal.butter(8, 15000, "highpass", fs=SR, output="sos"), x, axis=-1)
    loc = np.sqrt(np.maximum(uniform_filter1d(hp ** 2, 2401, axis=-1), 0.0) + 1e-14)
    rh = np.abs(hp) / loc
    rh[np.abs(hp) < 1e-3] = 0.0
    hh = np.argwhere(rh > 8.0)
    ht = sorted(set(round(float(i) / SR, 2) for _, i in hh))
    return dict(max_ratio=float(r.max()), max_jump=float(d.max()), n_hits=len(times), hit_times=times[:20],
                hf_max_ratio=round(float(rh.max()), 2), hf_hits=len(ht), hf_hit_times=ht[:40])


def loop_scan(x, lag_min=0.75, lag_max=20.0):
    """Max normalised autocorrelation of detrended log band envelopes over lags."""
    mono = np.atleast_2d(x).mean(axis=0)
    out = {}
    for name, lo, hi in (("low", 60, 300), ("mid", 300, 2500), ("high", 2500, 12000)):
        sos = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
        b = signal.sosfilt(sos, mono)
        hop = 480  # 100 Hz envelope
        m = len(b) // hop
        e = np.log(np.mean(b[: m * hop].reshape(m, hop) ** 2, axis=1) + 1e-14)
        e = e - gaussian_filter1d(e, 300)  # remove > ~3 s trends
        e = e - e.mean()
        f = np.fft.rfft(e, 2 * m)
        ac = np.fft.irfft(np.abs(f) ** 2)[:m]
        ac /= ac[0] + 1e-20
        l0, l1 = int(lag_min * 100), min(m - 1, int(lag_max * 100))
        k = l0 + int(np.argmax(ac[l0:l1]))
        out[name] = dict(peak=round(float(ac[k]), 3), lag_s=round(k / 100, 2))
    return out


def lr_correlation(x, t0=0, t1=None):
    x = np.atleast_2d(x)
    if x.shape[0] < 2:
        return 1.0
    i0 = int(t0 * SR)
    i1 = x.shape[-1] if t1 is None else int(t1 * SR)
    L, R = x[0, i0:i1], x[1, i0:i1]
    return float(np.sum(L * R) / (np.sqrt(np.sum(L * L) * np.sum(R * R)) + 1e-20))


# ------------------------------------------------------------------ images
_MAGMA = [(0.0, (0, 0, 4)), (0.13, (28, 16, 68)), (0.25, (79, 18, 123)), (0.38, (129, 37, 129)),
          (0.5, (181, 54, 122)), (0.63, (229, 80, 100)), (0.75, (251, 135, 97)),
          (0.88, (254, 194, 135)), (1.0, (252, 253, 191))]


def _lut():
    xs = np.array([a for a, _ in _MAGMA])
    cs = np.array([c for _, c in _MAGMA], float)
    u = np.linspace(0, 1, 256)
    return np.stack([np.interp(u, xs, cs[:, i]) for i in range(3)], axis=1).astype(np.uint8)


_LUT = _lut()


def _font(size=13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


_SPEC_CACHE = {}


def spectro_array(x, width, height, fmin=20.0, fmax=20000.0, db_lo=-135.0, db_hi=-35.0, nfft=4096):
    key = (id(x), np.shape(x), width, nfft, float(np.sum(np.atleast_2d(x)[..., ::997])))
    x = np.atleast_2d(x)
    n = x.shape[-1]
    if key in _SPEC_CACHE:
        P = _SPEC_CACHE[key]
    else:
        hop = max(128, int(math.ceil(n / width)))
        nov = max(0, nfft - hop)
        _, _, P = signal.spectrogram(x, SR, window="hann", nperseg=nfft, noverlap=nov,
                                     scaling="spectrum", mode="psd", axis=-1)
        P = P.mean(axis=0)  # (bins, frames)
        _SPEC_CACHE[key] = P
    nt = P.shape[1]
    cols = np.clip(np.round(np.linspace(0, nt - 1, width)).astype(int), 0, nt - 1)
    P = P[:, cols]
    binw = SR / nfft
    c = np.vstack([np.zeros((1, width)), np.cumsum(P, axis=0)])
    edges = fmin * (fmax / fmin) ** (np.arange(height + 1) / height)
    pos = np.clip(edges / binw + 0.5, 0, P.shape[0] - 1e-6)

    def cs(p):
        i = np.floor(p).astype(int)
        fr = (p - i)[:, None]
        return c[i] + fr * (c[i + 1] - c[i])

    rows = (cs(pos[1:]) - cs(pos[:-1])) / np.maximum(pos[1:] - pos[:-1], 1e-9)[:, None]
    db = 10 * np.log10(rows + 1e-30)
    v = np.clip((db - db_lo) / (db_hi - db_lo), 0, 1)
    img = _LUT[(v * 255).astype(int)][::-1]  # low freq at bottom
    return img


def render_panel(x, title, shots=None, T=None, width=1400, spec_h=360, level_h=110, audio_cuts=None,
                 db_lo=-135.0, db_hi=-35.0, fmin=20.0, fmax=20000.0, t_offset=0.0, nfft=4096):
    x = np.atleast_2d(x)
    T = T or x.shape[-1] / SR
    ml, mr, mt, mb = 62, 12, 24, 22
    H = mt + spec_h + (level_h + 8 if level_h else 0) + mb
    W = ml + width + mr
    im = Image.new("RGB", (W, H), (18, 18, 22))
    im.paste(Image.fromarray(spectro_array(x, width, spec_h, fmin, fmax, db_lo, db_hi, nfft)), (ml, mt))
    d = ImageDraw.Draw(im)
    f = _font(13)
    d.text((ml, 4), title, fill=(235, 235, 235), font=f)

    def fy(fr):
        return mt + spec_h - 1 - (math.log(fr / fmin) / math.log(fmax / fmin)) * (spec_h - 1)

    for fr, lab in ((50, "50"), (100, "100"), (200, "200"), (500, "500"), (1000, "1k"), (2000, "2k"),
                    (5000, "5k"), (10000, "10k"), (20000, "20k")):
        y = fy(fr)
        d.line([(ml - 5, y), (ml, y)], fill=(200, 200, 200))
        d.text((4, y - 7), lab + " Hz" if fr < 1000 else lab, fill=(200, 200, 200), font=f)

    def tx(t):
        return ml + t / T * (width - 1)

    ybot = H - mb
    step = 5 if T > 20 else (1 if T > 3 else 0.25)
    for s in np.arange(0, T + 1e-6, step):
        d.line([(tx(s), ybot), (tx(s), ybot + 4)], fill=(200, 200, 200))
        d.text((tx(s) - 6, ybot + 5), f"{s + t_offset:g}s", fill=(200, 200, 200), font=f)
    if shots:
        for i, s in enumerate(shots):
            x0 = tx(s["start"])
            if i > 0:
                for yy in range(mt, mt + spec_h, 6):
                    d.line([(x0, yy), (x0, yy + 3)], fill=(255, 255, 255))
            d.text((x0 + 3, mt + 2), s["env"], fill=(255, 255, 255), font=f)
    if audio_cuts:
        for a in audio_cuts:
            for yy in range(mt, mt + spec_h, 10):
                d.line([(tx(a), yy), (tx(a), yy + 2)], fill=(120, 230, 255))
    if level_h:
        y0 = mt + spec_h + 8
        d.rectangle([ml, y0, ml + width, y0 + level_h], fill=(28, 28, 34))
        lo, hi = -60.0, -5.0

        def ly(v):
            return y0 + level_h - (np.clip(v, lo, hi) - lo) / (hi - lo) * level_h

        for g in (-50, -40, -30, -23, -16, -10):
            d.line([(ml, ly(g)), (ml + width, ly(g))], fill=(60, 60, 70) if g != -16 else (90, 140, 90))
            d.text((ml - 34, ly(g) - 7), f"{g}", fill=(160, 160, 160), font=f)
        xk = k_weight(x)
        tm, lm = momentary(xk)
        pts = [(tx(t), ly(v)) for t, v in zip(tm, lm)]
        d.line(pts, fill=(110, 110, 200), width=1)
        ts, ls = short_term(xk)
        pts = [(tx(t), ly(v)) for t, v in zip(ts, ls)]
        d.line(pts, fill=(255, 210, 90), width=2)
        d.text((ml + 4, y0 + 2), "loudness LUFS: short-term (yellow), momentary (blue), -16 line green",
               fill=(200, 200, 200), font=f)
    return im


def zoom_png(path, x, t0, t1, title, **kw):
    """Detail spectrogram of x[t0:t1] (seconds) for close inspection."""
    seg = np.atleast_2d(x)[:, int(t0 * SR):int(t1 * SR)]
    render_panel(seg, f"{title}  [{t0:g}-{t1:g} s]", None, t1 - t0, t_offset=t0, **kw).save(path)


def overview_png(path, panels, shots, T, width=1400, spec_h=120, audio_cuts=None):
    ims = [render_panel(x, name, shots, T, width, spec_h, level_h=0, audio_cuts=audio_cuts) for name, x in panels]
    H = sum(i.height for i in ims)
    out = Image.new("RGB", (ims[0].width, H), (18, 18, 22))
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    out.save(path)


# ------------------------------------------------------------------ io
def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float64) / (1 << 23)
    elif sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError("unsupported sample width")
    return x.reshape(-1, ch).T.copy(), sr


def read_any_wav(path):
    """Also handles IEEE float WAV (format 3) via scipy."""
    try:
        return read_wav(path)
    except Exception:
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        return np.asarray(x, np.float64).T, sr


# ------------------------------------------------------------------ checks
NATURE_STEMS = ["wind", "foliage", "lake", "birds", "waterfall", "stream", "meadow"]
# which nature stem must be the loudest in a shot of each env (None = no single owner)
EXPECT = {"lakeshore": ("lake",), "waterfall": ("waterfall",), "stream": ("stream",), "high_view": ("wind",),
          "forest": ("foliage", "birds"), "meadow": ("meadow", "foliage"), "golden": ("lake", "birds"),
          "reveal": None}


def _db(v):
    return 20.0 * np.log10(np.maximum(v, 1e-12))


def _interior(s, i, windows, T=None, tail=2.5):
    """Part of shot i where its own parameters are at full weight (audio window minus the
    crossfades); falls back to the visual shot trimmed by the J-cut lead. With T, the last
    `tail` seconds of the film (end fade) are excluded."""
    lo, hi = s["start"] + 0.3, s["end"] - 0.75
    if windows and i < len(windows):
        a, b, h0, h1 = windows[i]
        if (b - h1) - (a + h0) > 1.0:
            lo, hi = a + h0, b - h1
    if T is not None:
        hi = min(hi, T - tail)
        lo = min(lo, hi - 1.0)
    return lo, hi


def _frame_db(x, t0, t1, hop=480):
    m = np.atleast_2d(x)[:, int(t0 * SR):int(t1 * SR)].mean(axis=0)
    k = len(m) // hop
    return _db(np.sqrt(np.mean(m[:k * hop].reshape(k, hop) ** 2, axis=1)))


def _band(x, lo, hi, order=4):
    sos = signal.butter(order, [lo, hi], "bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, np.atleast_2d(x), axis=-1)


def _band_rel(x, t0, t1, lo, hi):
    """Band power (lo..hi Hz) relative to the total, dB (Welch on the mono sum)."""
    seg = np.atleast_2d(x)[:, int(t0 * SR):int(t1 * SR)].mean(axis=0)
    f, P = signal.welch(seg, SR, nperseg=8192)
    m = (f >= lo) & (f < hi)
    return float(10 * np.log10(P[m].sum() / (P.sum() + 1e-30) + 1e-30))


def checks(mix, stems, shots, windows=None, extra=None):
    """Objective acceptance checks; returns a list of dicts (name, value, target, ok)."""
    extra = extra or {}
    out = []

    def add(name, value, target, ok, detail=""):
        out.append(dict(name=name, value=value, target=target, ok=None if ok is None else bool(ok), detail=detail))

    T = mix.shape[-1] / SR
    xk = k_weight(mix)
    il = integrated_lufs(xk=xk)
    add("integrated loudness", round(il, 2), "-18 .. -16.5 LUFS", -18.05 <= il <= -16.45)
    tp = true_peak_db(mix)
    add("true peak", round(tp, 2), "<= -1.2 dBTP", tp <= -1.2)
    sk = {k: k_weight(stems[k]) for k in NATURE_STEMS if k in stems}
    # 1. sync: the env's own stem is the loudest nature stem in the shot interior
    for i, s in enumerate(shots or []):
        exp = EXPECT.get(s["env"])
        if not exp or not sk:
            continue
        lo, hi = _interior(s, i, windows, T, 2.5)
        lv = {k: segment_lufs(v, lo, hi) for k, v in sk.items()}
        top = max(lv, key=lv.get)
        ok = top in exp
        others = sorted(((v, k) for k, v in lv.items() if k not in exp), reverse=True)
        margin = max(lv[e] for e in exp) - (others[0][0] if others else -120.0)
        add(f"sync {s.get('name', s['env'])} ({s['env']})", f"{top} {lv[top]:.1f}", "/".join(exp), ok,
            f"margin {margin:+.1f} LU over {others[0][1] if others else '-'}; window {lo:.2f}-{hi:.2f}")
        if not ok:
            print(f"  SYNC MISMATCH: shot '{s.get('name', s['env'])}' ({s['env']}) {s['start']}-{s['end']} s: "
                  f"loudest nature stem is '{top}', expected {exp}")
    # 2. limiter activity in the high_view shots
    g = extra.get("limiter_gain")
    if g is not None:
        for s in shots or []:
            if s["env"] == "high_view":
                seg = g[int(s["start"] * SR):int(s["end"] * SR)]
                gr = float(-20 * np.log10(max(seg.min(), 1e-9))) if len(seg) else 0.0
                add(f"limiter GR in {s.get('name', s['env'])}", round(gr, 2), "< 0.1 dB", gr < 0.1)
    # 3. high_view nature short-term level (music carries the aerials)
    if "nature" in stems:
        tn, ln = short_term(k_weight(stems["nature"]), 0.1)
        for s in shots or []:
            if s["env"] == "high_view":
                m = (tn >= s["start"]) & (tn <= s["end"])
                add(f"nature short-term in {s.get('name')}", round(float(np.median(ln[m])), 1), "about -20 LUFS",
                    abs(float(np.median(ln[m])) + 20.0) <= 2.5)
    # 4. waterfall: swell, fast level variation, hiss, sub-bass
    if "waterfall" in stems:
        w = stems["waterfall"]
        others_sub = []
        sub = _band(mix, 20, 50)
        for s in shots or []:
            lvl = float(_db(np.sqrt(np.mean(sub[:, int(s["start"] * SR):int(s["end"] * SR)] ** 2))))
            (others_sub if s["env"] != "waterfall" else []).append(lvl)
        for i, s in enumerate(shots or []):
            if s["env"] != "waterfall":
                continue
            e = _frame_db(w, s["start"], s["end"], 4800)            # 0.1 s frames
            e1 = gaussian_filter1d(e, 5)                             # ~1 s smoothing
            swell = float(e1[5:].max() - e1[5])
            add("waterfall swell in shot", round(swell, 1), ">= +3 dB", swell >= 3.0)
            lo, hi = _interior(s, i, windows)
            ef = _frame_db(w, lo, hi)                                # 10 ms frames
            fast = float(np.std(ef - gaussian_filter1d(ef, 100)))
            add("waterfall fast level SD", round(fast, 2), ">= 2 dB", fast >= 2.0)
            hf = _band_rel(mix, lo, hi, 12000, 20000)
            add("waterfall shot 12-20 kHz rel. total", round(hf, 1), "about -24 dB", -28.0 <= hf <= -20.0)
            ws = float(_db(np.sqrt(np.mean(sub[:, int(s["start"] * SR):int(s["end"] * SR)] ** 2))))
            ref = max(others_sub) if others_sub else -120.0
            add("waterfall shot 20-50 Hz vs loudest other shot", round(ws - ref, 1), "<= +8 dB", ws - ref <= 8.0,
                f"{ws:.1f} dBFS vs {ref:.1f}")
            # stream should sit 5-6 LU under the waterfall
            if "stream" in stems:
                st_shot = [j for j, q in enumerate(shots) if q["env"] == "stream"]
                if st_shot:
                    j = st_shot[0]
                    l2, h2 = _interior(shots[j], j, windows)
                    d = segment_lufs(k_weight(stems["waterfall"]), lo, hi) - segment_lufs(k_weight(stems["stream"]), l2, h2)
                    add("stream under waterfall", round(d, 1), "5-6 LU", 4.0 <= d <= 7.5)
    # 5. birds vs ambience (momentary max of birds vs median momentary of the rest)
    if "birds" in stems and "nature" in stems:
        amb = stems["nature"] - stems["birds"]
        worst = None
        for s in shots or []:
            # reviewer metric: bird momentary max vs median ambience momentary in the shot
            # (excluding the nature fade-out, where the ambience median is meaningless)
            a, b = int(s["start"] * SR), int(min(s["end"], T - 4.5) * SR)
            if b - a < int(1.0 * SR):
                continue
            _, lb = momentary(k_weight(stems["birds"][:, a:b]))
            _, la = momentary(k_weight(amb[:, a:b]))
            if lb.max() > -60:
                d = float(lb.max() - np.median(la))
                worst = d if worst is None else max(worst, d)
                add(f"bird peak over ambience in {s.get('name')}", round(d, 1), "<= +6 dB", d <= 6.0)
        # instantaneous version over the whole film (robust to fades): bird momentary vs
        # ambience short-term loudness at the same moment
        tb, lb = momentary(k_weight(stems["birds"]), 0.05)
        ta, la = short_term(k_weight(amb), 0.05)
        lai = np.interp(tb, ta, la)
        m = (lb > -60) & (lai > -60)
        if m.any():
            d = float(np.max(lb[m] - lai[m]))
            add("bird momentary over ambience short-term (any time)", round(d, 1), "<= +6 dB", d <= 6.0,
                f"at {tb[m][np.argmax(lb[m] - lai[m])]:.1f} s")
    # 6. stereo coherence
    c_lo = lr_correlation(_band(mix, 20, 150))
    add("mix L/R correlation 20-150 Hz", round(c_lo, 2), ">= 0.5", c_lo >= 0.5)
    c_all = lr_correlation(mix)
    add("mix L/R correlation", round(c_all, 2), "0.2 .. 0.35 (about)", 0.15 <= c_all <= 0.45)
    # 7. lake lapping spread / meadow / wind floor / music wobble
    for k, envs, tgt in (("lake", ("lakeshore", "golden"), 12.0),):
        if k in stems:
            for i, s in enumerate(shots or []):
                if s["env"] in envs and s.get("name") != "skim":
                    lo, hi = _interior(s, i, windows, T, 4.5)
                    e = _frame_db(stems[k], lo, hi)
                    spread = float(np.percentile(e, 95) - np.percentile(e, 5))
                    add(f"lake level spread p5-p95 in {s.get('name')}", round(spread, 1), f"<= {tgt:g} dB", spread <= tgt)
    if "lake" in stems and "nature" in stems:
        # a single lap must not jump out: lake momentary vs short-term of the whole nature bed
        tl_, ll_ = momentary(k_weight(stems["lake"]), 0.05)
        tn_, ln_ = short_term(k_weight(stems["nature"]), 0.05)
        lni = np.interp(tl_, tn_, ln_)
        m = (ll_ > -60) & (lni > -60) & (tl_ > 2.5) & (tl_ < T - 4.5)
        if m.any():
            d = float(np.max(ll_[m] - lni[m]))
            add("lake lap momentary over nature short-term", round(d, 1), "<= +6 dB", d <= 6.0,
                f"at {tl_[m][np.argmax(ll_[m] - lni[m])]:.1f} s")
    if "meadow" in stems:
        for s in shots or []:
            if s["env"] == "meadow":
                d = segment_lufs(xk, s["start"], s["end"]) - segment_lufs(k_weight(stems["meadow"]), s["start"], s["end"])
                add("meadow stem under mix", round(d, 1), "<= 8 dB", d <= 8.0)
    if "wind" in stems:
        for i, s in enumerate(shots or []):
            if s["env"] in ("lakeshore", "golden", "meadow", "reveal", "high_view"):
                lo, hi = _interior(s, i, windows, T, 4.5)
                lo = max(lo, min(2.5, hi - 1.0))   # exclude the 2.5 s fade-in
                e = _frame_db(stems["wind"], lo, hi)
                add(f"wind spread p5-p95 in {s.get('name')}", round(float(np.percentile(e, 95) - np.percentile(e, 5)), 1),
                    "about 14 dB", None)
    if "music" in stems:
        m = stems["music"].mean(axis=0)
        hop = 240
        k = len(m) // hop
        ed = _db(np.sqrt(np.mean(m[:k * hop].reshape(k, hop) ** 2, axis=1)))
        r = ed - gaussian_filter1d(ed, 100)
        sd = float(r[400:-400].std())
        # 5 ms frames are shorter than the D2 period (13.6 ms): this number is dominated by
        # waveform-level interference between chord notes (~2.6 dB even with zero detune)
        add("music fast envelope SD (5 ms frames)", round(sd, 2), "info (floor ~2.6 dB w/o detune)", None)
        hop = 2400
        k = len(m) // hop
        ed = _db(np.sqrt(np.mean(m[:k * hop].reshape(k, hop) ** 2, axis=1)))
        r = ed - gaussian_filter1d(ed, 10)
        sd50 = float(r[40:-40].std())
        add("music fast envelope SD (50 ms frames, beating)", round(sd50, 2), "<= 1 dB", sd50 <= 1.0)
    # 8. music level relative to nature per shot (music_rel_db table, short-term follower)
    mt = extra.get("music_rel_target")
    if mt and "music" in stems and "nature" in stems:
        km, kn = k_weight(stems["music"]), k_weight(stems["nature"])
        for s, tgt in zip(shots or [], mt):
            lo, hi = s["start"], min(s["end"], T - 4.5)      # before the split end fades
            if hi - lo < 1.0:
                continue
            d = segment_lufs(km, lo, hi) - segment_lufs(kn, lo, hi)
            add(f"music rel. nature in {s.get('name')}", round(d, 1), f"{tgt:+.1f} +-3 LU", abs(d - tgt) <= 3.0)
    # 9. clicks, exact length, nothing cut off at the end
    cs = click_scan(mix)
    add("clicks (isolated sample jumps)", cs["n_hits"], "0", cs["n_hits"] == 0, f"max ratio {cs['max_ratio']:.1f}")
    if shots:
        Ts = max(s["end"] for s in shots)
        add("length", f"{mix.shape[-1]} smp", f"{int(round(Ts * SR))} smp ({Ts:g} s)", mix.shape[-1] == int(round(Ts * SR)))
    tail = float(_db(np.sqrt(np.mean(mix[:, -int(0.1 * SR):] ** 2))))
    add("mix level in the last 100 ms", round(tail, 1), "<= -50 dBFS (faded, nothing cut off)", tail <= -50.0)
    ll = extra.get("last_lap_end")
    if ll is not None:
        add("last lake lap ends", round(ll, 2), f"<= T-1.5 = {T - 1.5:.2f} s", ll <= T - 1.5)
    be = extra.get("bell_events")
    if be:
        last = max(t for t, _ in be)
        add("last bell onset", round(last, 2), f"<= T-4.5 = {T - 4.5:.2f} s", last <= T - 4.5 + 1e-6)
    bs = extra.get("bird_songs")
    if bs:
        add("last bird song onset", max(t for t, *_ in bs), f"<= T-3.5 = {T - 3.5:.2f} s", max(t for t, *_ in bs) <= T - 3.5)
    return out


def print_checks(res):
    print("  checks:")
    for r in res:
        flag = "INFO" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
        print(f"   [{flag}] {r['name']:<44} {str(r['value']):>16}   target {r['target']}"
              + (f"   ({r['detail']})" if r["detail"] else ""))


# ------------------------------------------------------------------ report
def analyze(mix, stems, shots, out_dir, audio_cuts=None, tag="", verbose=True, extra=None):
    """Metrics + checks + PNGs. stems: dict name -> (2, n) array at final scaling."""
    os.makedirs(out_dir, exist_ok=True)
    extra = extra or {}
    T = mix.shape[-1] / SR
    xk = k_weight(mix)
    rep = dict(
        duration_s=T,
        samples=int(mix.shape[-1]),
        integrated_lufs=round(integrated_lufs(xk=xk), 2),
        sample_peak_dbfs=round(sample_peak_db(mix), 2),
        true_peak_dbtp=round(true_peak_db(mix), 2),
        dc_offset=[float(f"{v:.2e}") for v in mix.mean(axis=-1)],
        clipped_samples=int(np.sum(np.abs(mix) >= 0.9999)),
        clicks=click_scan(mix),
        looping=loop_scan(mix),
        lr_correlation=round(lr_correlation(mix), 3),
        max_short_term=round(float(np.max(short_term(xk)[1])), 2),
        max_momentary=round(float(np.max(momentary(xk)[1])), 2),
        first_sample_abs=float(np.max(np.abs(mix[:, 0]))),
        last_sample_abs=float(np.max(np.abs(mix[:, -1]))),
    )
    if "shots_file" in extra:
        rep["shots_file"] = extra["shots_file"]
    if "info" in extra:
        rep["master"] = {k: round(float(v), 3) for k, v in extra["info"].items()}
    windows = extra.get("audio_windows")
    if windows is None and audio_cuts is not None and shots:
        a = [0.0] + list(audio_cuts)
        b = list(audio_cuts) + [T]
        windows = [(a[i], b[i], 0.6 if i else 0.0, 0.6 if i < len(b) - 1 else 0.0) for i in range(len(a))]
    rep["checks"] = checks(mix, stems, shots, windows, extra)
    if extra.get("bell_events") is not None:
        rep["bell_events"] = extra["bell_events"]
    if extra.get("bird_songs") is not None:
        rep["bird_songs"] = [list(s) for s in extra["bird_songs"]]
    segs = []
    sk = {k: k_weight(v) for k, v in stems.items()}
    for s in (shots or []):
        row = dict(name=s.get("name", s["env"]), env=s["env"], start=s["start"], end=s["end"],
                   mix=round(segment_lufs(xk, s["start"], s["end"]), 1),
                   peak=round(sample_peak_db(mix[:, int(s["start"] * SR):int(s["end"] * SR)]), 1),
                   lr_corr=round(lr_correlation(mix, s["start"], s["end"]), 2))
        for k, v in sk.items():
            row[k] = round(segment_lufs(v, s["start"], s["end"]), 1)
        segs.append(row)
    rep["segments"] = segs
    rep["stems"] = {}
    for k, v in stems.items():
        cs = click_scan(v)
        rep["stems"][k] = dict(integrated_lufs=round(integrated_lufs(xk=sk[k]), 2),
                               peak_dbfs=round(sample_peak_db(v), 2),
                               clicks=cs["n_hits"], click_max_ratio=round(cs["max_ratio"], 2))
    with open(os.path.join(out_dir, f"analysis{tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    # images
    small = []
    for k, v in [("mix", mix)] + list(stems.items()):
        title = f"MIX  {rep['integrated_lufs']} LUFS  TP {rep['true_peak_dbtp']} dBTP" if k == "mix" else f"stem: {k}"
        render_panel(v, title, shots, T, audio_cuts=audio_cuts).save(os.path.join(out_dir, f"spec_{k}{tag}.png"))
        small.append(render_panel(v, k, shots, T, spec_h=120, level_h=0, audio_cuts=audio_cuts))
        _SPEC_CACHE.clear()  # keep memory low: one spectrogram at a time
    ov = Image.new("RGB", (small[0].width, sum(i.height for i in small)), (18, 18, 22))
    y = 0
    for i in small:
        ov.paste(i, (0, y))
        y += i.height
    ov.save(os.path.join(out_dir, f"spec_overview{tag}.png"))
    if verbose:
        print_report(rep, list(stems.keys()))
        print_checks(rep["checks"])
    return rep


def print_report(rep, stem_names):
    print(f"  integrated {rep['integrated_lufs']} LUFS | sample peak {rep['sample_peak_dbfs']} dBFS | "
          f"true peak {rep['true_peak_dbtp']} dBTP | max ST {rep['max_short_term']} | max M {rep['max_momentary']}")
    print(f"  DC {rep['dc_offset']} | clipped {rep['clipped_samples']} | clicks {rep['clicks']['n_hits']} "
          f"(max ratio {rep['clicks']['max_ratio']:.1f}, max jump {rep['clicks']['max_jump']:.4f}) | "
          f"L/R corr {rep['lr_correlation']}")
    print(f"  looping {rep['looping']}")
    hdr = "  {:<10}{:>6}{:>6}{:>6}".format("segment", "mix", "peak", "corr") + "".join(f"{k[:8]:>9}" for k in stem_names)
    print(hdr)
    for r in rep["segments"]:
        print("  {:<10}{:>6}{:>6}{:>6}".format(r.get("name", r["env"])[:10], r["mix"], r["peak"], r["lr_corr"])
              + "".join(f"{r[k]:>9}" for k in stem_names))
    for k, v in rep["stems"].items():
        print(f"  stem {k:<10} I {v['integrated_lufs']:>7} LUFS  peak {v['peak_dbfs']:>6}  clicks {v['clicks']}"
              f" (max ratio {v['click_max_ratio']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--shots", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stems-dir", default=None, help="folder of stem WAVs written by soundtrack.py --stems")
    ap.add_argument("--tag", default="_file")
    a = ap.parse_args()
    x, sr = read_any_wav(a.wav)
    assert sr == SR, "expects 48 kHz"
    shots = json.load(open(a.shots)) if a.shots else None
    out = a.out or os.path.dirname(os.path.abspath(a.wav))
    stems = {}
    if a.stems_dir:
        for fn in sorted(os.listdir(a.stems_dir)):
            if fn.endswith(".wav"):
                stems[fn[:-4]] = read_any_wav(os.path.join(a.stems_dir, fn))[0]
    analyze(x, stems, shots, out, tag=a.tag)


if __name__ == "__main__":
    main()
