#!/usr/bin/env python
"""score_b.py - score candidate B: modern cinematic hybrid (Interstellar / Inception / Dune feel).

Everything is synthesised with numpy / scipy: no samples, recordings, soundfonts or downloads.

Concept
    One four-note cell carries the whole film: x = A - E - F - E (1, 5, b6, 5 in A minor). It is the
    piano motif of the forest, the shape of the 16th ostinato (root - 5th - 6th - 5th on every chord),
    the brass theme of the reveal, it turns major at the meadow lift (C - G - A - G = 1 5 6 5) and is
    "fulfilled" at the peaks, where the answer phrase climbs to C6 instead of falling back, and the
    final A-major chord puts C#6 on top (Picardy third) while the piano plays the cell in major.

Theme (quarter-note beats)
    phrase A  A4 E5 | F5. E5 | E5 G5 | D5 --          over Am  F  C  G
    phrase B  A4 E5 | F5 A5 | G5. E5 | D5 B4           over Am  F  C  G
    meadow    C5 G5 | A5. G5 | E5 C6 | B5 --           over C  G/B  Am  G      (C major, the lift)
    flyover   A4 C5 | G4. E4 | F4 A4 | A4 G#4          over F  C/E  Dm  Esus4-E (stepwise bass C B A G F E D E)
    peaks     A4 E5 | F5 A5 | G5 C6 | B5 --            over Am  F  C  G        -> A major (bVII - I)

Tempo map: 2 bars per shot so every cut is a downbeat. 60 BPM forest, 64 reveal (both without an
audible pulse), 73.85 BPM from skim to flyover (the ostinato), 80 BPM peaks / finale (after a silence).

Usage
    python audio/score_b.py --out out/audio/candidates/B_hybrid.wav
        --nature out/audio/nature/stems/nature.wav --work out/audio/work_b [--analyze] [--seed 5] [--quick]
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as an  # noqa: E402  (loudness, true peak, clicks, spectrogram PNGs)
import soundtrack as st  # noqa: E402  (bus_compressor, limiter, write_wav)

SR = 48000
DUR = 60.0
N = int(round(DUR * SR))
CR = 1000.0  # control rate for pitch / envelope curves
T0 = time.time()
VSCALE = 1.0  # voice-count scale (--quick halves it)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# =============================================================================================
# time, notes, chords
# =============================================================================================
BAR_T = np.array([0.0, 4.0, 8.0, 11.75] + [15.5 + 3.25 * k for k in range(10)] + [48.0, 51.0, 54.0, 57.0, 60.0])
SHOTS = [("forest", 0.0, 8.0), ("reveal", 8.0, 15.5), ("skim", 15.5, 22.0), ("waterfall", 22.0, 28.5),
         ("stream", 28.5, 35.0), ("meadow", 35.0, 41.5), ("flyover", 41.5, 48.0), ("peaks", 48.0, 54.0),
         ("finale", 54.0, 60.0)]


def tp(bar, beat=0.0):
    """Seconds of `beat` (0-based, may run past 4) in `bar` (1-based)."""
    return float(np.interp((bar - 1) * 4 + beat, 4 * np.arange(len(BAR_T)), BAR_T))


def gpos(t):
    """Global beat position of time t."""
    return float(np.interp(t, BAR_T, 4 * np.arange(len(BAR_T))))


_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
       "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def nm(s):
    return 12 * (int(s[-1]) + 1) + _PC[s[:-1]]


def mname(m):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
    return f"{names[int(m) % 12]}{int(m) // 12 - 1}"


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


# chord: (bass midi, upper voicing G3..C5 region, voice-led by hand, chart label)
VOX = {
    "Am": (45, [57, 60, 64, 69], "Am"),
    "F": (41, [57, 60, 65, 69], "F"),
    "C": (48, [55, 60, 64, 67], "C"),
    "G": (43, [55, 59, 62, 67], "G"),
    "Dm7": (38, [57, 60, 65, 69], "Dm7"),
    "Gsus4": (43, [55, 60, 62, 67], "Gsus4"),
    "C_hi": (48, [60, 64, 67, 72], "C"),
    "G/B": (47, [59, 62, 67, 71], "G/B"),
    "Am_m": (45, [57, 64, 69, 72], "Am"),
    "G_m": (43, [55, 62, 67, 71], "G"),
    "F_f": (41, [57, 65, 69, 72], "F"),
    "C/E": (40, [55, 64, 67, 72], "C/E"),
    "Dm": (38, [57, 62, 65, 69], "Dm"),
    "Esus4": (40, [57, 59, 64, 69], "Esus4"),
    "E": (40, [56, 59, 64, 68], "E"),
    "A": (45, [57, 61, 64, 69], "A(add9)"),
}
# (bar, beat0, beat1, chord)
CHART = []
for _b in (1, 3, 5, 7):
    CHART += [(_b, 0, 2, "Am"), (_b, 2, 4, "F"), (_b + 1, 0, 2, "C"), (_b + 1, 2, 4, "G")]
CHART += [(9, 0, 2, "Am"), (9, 2, 4, "F"), (10, 0, 2, "Dm7"), (10, 2, 3, "Gsus4"), (10, 3, 4, "G"),
          (11, 0, 2, "C_hi"), (11, 2, 4, "G/B"), (12, 0, 2, "Am_m"), (12, 2, 4, "G_m"),
          (13, 0, 2, "F_f"), (13, 2, 4, "C/E"), (14, 0, 2, "Dm"), (14, 2, 3, "Esus4"), (14, 3, 4, "E"),
          (15, 0, 2, "Am"), (15, 2, 4, "F"), (16, 0, 2, "C"), (16, 2, 4, "G"), (17, 0, 8, "A")]
CHORDS = [(tp(b, a), tp(b, e), c) for b, a, e, c in CHART]

PH_A = [(0, 1, "A4"), (1, 1, "E5"), (2, 1.5, "F5"), (3.5, .5, "E5"), (4, 1, "E5"), (5, 1, "G5"), (6, 2, "D5")]
PH_B = [(0, 1, "A4"), (1, 1, "E5"), (2, 1, "F5"), (3, 1, "A5"), (4, 1.5, "G5"), (5.5, .5, "E5"), (6, 1, "D5"),
        (7, 1, "B4")]
PH_B2 = [(0, 1, "A4"), (1, 1, "E5"), (2, 1, "F5"), (3, 1, "A5"), (4, 1.5, "A5"), (5.5, .5, "F5"), (6, 1, "C5"),
         (7, 1, "B4")]
PH_M = [(0, 1, "C5"), (1, 1, "G5"), (2, 1.5, "A5"), (3.5, .5, "G5"), (4, 1, "E5"), (5, 1, "C6"), (6, 2, "B5")]
PH_F = [(0, 1, "A4"), (1, 1, "C5"), (2, 1.5, "G4"), (3.5, .5, "E4"), (4, 1, "F4"), (5, 1, "A4"), (6, 1, "A4"),
        (7, 1, "G#4")]
PH_P = [(0, 1, "A4"), (1, 1, "E5"), (2, 1, "F5"), (3, 1, "A5"), (4, 1, "G5"), (5, 1, "C6"), (6, 2, "B5")]
DESCANT = [(0, 2, "E5"), (2, 2, "F5"), (4, 2, "E5"), (6, 2, "D5")]


def phrase(bar, ph, octave=0, vel=0.7, legato=1.0, accent=None):
    """Phrase -> [(ts, te, midi, vel)] with times from the tempo map."""
    out = []
    for i, (b, d, n) in enumerate(ph):
        ts, te = tp(bar, b), tp(bar, b + d * legato)
        v = vel * (accent[i] if accent else 1.0)
        out.append((ts, te, nm(n) + 12 * octave, v))
    return out


def chord_at(t):
    for a, b, c in CHORDS:
        if a - 1e-6 <= t < b - 1e-6:
            return c
    return CHORDS[-1][2]


def voice_lines(t0, t1, part, vel, oct_shift=0, key_override=None):
    """Sustained chord lines between t0 and t1 for voicing index `part` ('bass' or 0..3);
    repeated notes are tied, so common tones do not re-attack."""
    lines = []
    for a, b, c in CHORDS:
        a2, b2 = max(a, t0), min(b, t1)
        if b2 <= a2 + 1e-6:
            continue
        bass, vx, _ = VOX[key_override or c]
        m = bass if part == "bass" else vx[part]
        m += 12 * oct_shift
        v = vel(a2) if callable(vel) else vel
        if lines and abs(lines[-1][1] - a2) < 1e-6 and lines[-1][2] == m:
            lines[-1] = (lines[-1][0], b2, m, lines[-1][3])
        else:
            lines.append((a2, b2, m, v))
    return lines


# =============================================================================================
# DSP helpers
# =============================================================================================
def bw(kind, f, order=2):
    return signal.butter(order, f, kind, fs=SR, output="sos")


def filt(sos, x):
    return signal.sosfilt(sos, x, axis=-1)


def peq(fc, gain_db, q):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / SR
    al = math.sin(w0) / (2 * q)
    b = [1 + al * A, -2 * math.cos(w0), 1 - al * A]
    a = [1 + al / A, -2 * math.cos(w0), 1 - al / A]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def shelf(fc, gain_db, kind="high", s=0.8):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * fc / SR
    al = math.sin(w0) / 2 * math.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    c = math.cos(w0)
    sq = 2 * math.sqrt(A) * al
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
        a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
    else:
        b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
        a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def chain(*soss):
    return np.vstack(soss)


def pan2(y, p):
    th = (np.clip(p, -1, 1) + 1) * math.pi / 4
    return np.vstack([y * math.cos(th), y * math.sin(th)])


def add(bus, x, i0, gain=1.0):
    x = np.atleast_2d(x)
    a, b = max(0, i0), min(bus.shape[1], i0 + x.shape[1])
    if b > a:
        bus[:, a:b] += (gain * x[:, a - i0:b - i0]).astype(bus.dtype)


def blep_saw(f, ph0=0.0):
    dt = np.clip(np.asarray(f, float) / SR, 1e-7, 0.45)
    ph = (ph0 + np.cumsum(dt)) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] -= t + t - t * t - 1.0
    m = ph > 1.0 - dt
    t = (ph[m] - 1.0) / dt[m]
    y[m] -= t * t + t + t + 1.0
    return y


def slow_noise(rng, n, rate, fs=CR):
    """Smooth random curve, ~unit std, n samples at fs, wandering at ~rate Hz."""
    k = max(4, int(n * rate / fs) + 4)
    pts = rng.standard_normal(k)
    y = np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
    s = fs / max(rate, 1e-3) / 4
    y = gaussian_filter1d(y, s) if n > 8 * s else y
    return y / (np.std(y) + 1e-9)


def asym_env(g, att, rel, fs=CR):
    a, r = math.exp(-1.0 / (att * fs)), math.exp(-1.0 / (rel * fs))
    out = np.empty(len(g))
    y = 0.0
    for i in range(len(g)):
        x = g[i]
        c = a if x > y else r
        y = c * y + (1.0 - c) * x
        out[i] = y
    return out


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def dyn_lp(x, ctrl, f_lo, f_hi, order=2):
    """Brightness-controlled low-pass: crossfade of three static low-passes (smooth, cheap)."""
    fm = math.sqrt(f_lo * f_hi)
    ys = [filt(bw("lowpass", min(f, 20000.0), order), x) for f in (f_lo, fm, f_hi)]
    c = np.clip(ctrl, 0.0, 1.0)
    w0 = np.clip(1 - 2 * c, 0, 1)
    w2 = np.clip(2 * c - 1, 0, 1)
    return ys[0] * w0 + ys[1] * (1 - w0 - w2) + ys[2] * w2


def fade_edges(y, a_ms=2.0, b_ms=8.0):
    y = np.array(y, copy=True)
    na, nb = int(a_ms * 1e-3 * SR), int(b_ms * 1e-3 * SR)
    if na:
        y[..., :na] *= np.linspace(0, 1, na)
    if nb:
        y[..., -nb:] *= np.cos(np.linspace(0, math.pi / 2, nb)) ** 2
    return y


def curve(points, n=N, smooth_s=0.05):
    """Piecewise-linear dB automation -> smoothed linear gain over the film."""
    ts = np.array([p[0] for p in points])
    ds = np.array([p[1] for p in points])
    tt = np.arange(0, n, 48) / SR
    db = np.interp(tt, ts, ds)
    db = gaussian_filter1d(db, smooth_s * SR / 48) if smooth_s else db
    return np.interp(np.arange(n), np.arange(0, n, 48), 10 ** (db / 20.0))


# =============================================================================================
# ensemble line synthesis (strings, brass, synth pads, choir sources)
# =============================================================================================
def ensemble(notes, nv, cfg, rng):
    """A legato line played by nv detuned voices. notes: [(ts, te, midi, vel)].
    cfg: att, rel, glide, det (cents spread), drift (cents), vib (semitones), vrate, vdel, jit (s),
    gamma (vel->amp), pan, width, scoop (cents), dip, env_pow.
    Returns stereo (2, L), start sample i0, ctrl (L,) = env * vel (for brightness), onsets."""
    nv = max(1, int(round(nv * VSCALE)))
    notes = sorted(notes)
    pre = cfg.get("scoop_t", 0.05) + 0.08
    t0 = notes[0][0] - pre
    t1 = max(n[1] for n in notes) + cfg["rel"] * 6 + 0.15
    tc = np.arange(t0, t1, 1.0 / CR)
    nc = len(tc)
    m = np.full(nc, float(notes[0][2]))
    v = np.zeros(nc)
    gate = np.zeros(nc)
    since = np.zeros(nc)
    slur = np.zeros(nc)
    onsets = []
    prev_end = -1e9
    for (ts, te, mi, ve) in notes:
        s = tc >= ts
        m[s] = mi
        v[s] = ve
        since[s] = tc[s] - ts
        slur[s] = 1.0 if ts - prev_end < 0.03 else 0.0
        gate[(tc >= ts) & (tc < te)] = 1.0
        onsets.append((ts, ve, slur[s][0] if s.any() else 0.0))
        prev_end = te
    gl = max(2, int(cfg.get("glide", 0.07) * CR))
    m = uniform_filter1d(m, gl, mode="nearest")
    m = np.concatenate([np.full(gl // 2, m[0]), m[:-(gl // 2)]])  # glide after the onset, not before
    v = uniform_filter1d(v, int(0.10 * CR), mode="nearest")
    env = asym_env(gate, cfg["att"], cfg["rel"]) ** cfg.get("env_pow", 1.0)
    if cfg.get("dip", 0.0) > 0:
        d = np.zeros(nc)
        for ts, ve, sl in onsets[1:]:
            if sl:
                d += np.exp(-0.5 * ((tc - ts) / 0.035) ** 2)
        env *= 1.0 - cfg["dip"] * np.clip(d, 0, 1)
    vd = cfg.get("vib", 0.0) * smoothstep((since - cfg.get("vdel", 0.3)) / 0.5)
    sc = np.zeros(nc)
    if cfg.get("scoop", 0.0):
        sc = -cfg["scoop"] / 100.0 * np.exp(-since / cfg.get("scoop_t", 0.045)) * (1.0 - 0.6 * slur)
    L = int((t1 - t0) * SR)
    ta = t0 + np.arange(L) / SR
    out = np.zeros((2, L))
    gamma = cfg.get("gamma", 1.6)
    amp_c = env * np.maximum(v, 0) ** gamma
    for k in range(nv):
        jit = int(rng.uniform(-1, 1) * cfg.get("jit", 0.012) * CR)
        sh = (lambda a: np.concatenate([np.full(jit, a[0]), a[:-jit]])) if jit > 0 else \
            ((lambda a: np.concatenate([a[-jit:], np.full(-jit, a[-1])])) if jit < 0 else (lambda a: a))
        spread = ((k - (nv - 1) / 2) / max(1.0, (nv - 1) / 2)) if nv > 1 else 0.0
        det = spread * cfg.get("det", 8.0) + rng.normal(0, 0.25 * cfg.get("det", 8.0))
        drift = slow_noise(rng, nc, 0.35) * cfg.get("drift", 3.0)
        vr = cfg.get("vrate", 5.5) * rng.uniform(0.92, 1.08)
        vib = sh(vd) * np.sin(2 * math.pi * vr * (tc - t0) + rng.uniform(0, 6.28) + 0.5 * slow_noise(rng, nc, 0.7))
        mm = sh(m) + (det + drift) / 100.0 + vib + sh(sc)
        ak = sh(amp_c) * (1.0 + cfg.get("press", 0.05) * slow_noise(rng, nc, 1.1))
        f = mtof(np.interp(ta, tc, mm))
        a = np.interp(ta, tc, ak)
        y = blep_saw(f, rng.random()) * a
        if cfg.get("pulse", 0.0):
            y = y - cfg["pulse"] * blep_saw(f, rng.random()) * a
        p = cfg.get("pan", 0.0) + spread * cfg.get("width", 0.3) * 0.5
        out += pan2(y, p)
    out /= math.sqrt(nv)
    out = fade_edges(out, 0.0, 60.0)  # the release tail ends at -52 dB; fade the last 60 ms to zero
    ctrl = np.interp(ta, tc, env * v)
    return out, int(round(t0 * SR)), ctrl, onsets


def onset_bursts(onsets, i0, L, tau, only_new=False, scale_slur=0.5):
    b = np.zeros(L)
    tt = np.arange(int(tau * 8 * SR)) / SR
    shape = (1 - np.exp(-tt / 0.004)) * np.exp(-tt / tau)
    for ts, ve, sl in onsets:
        k = int(ts * SR) - i0
        w = ve * (scale_slur if sl else 1.0)
        if only_new and sl:
            continue
        if 0 <= k < L:
            e = min(L, k + len(shape))
            b[k:e] += w * shape[:e - k]
    return b


def strings(notes, nv, rng, pan=0.0, width=0.5, f_lo=1800, f_hi=7000, bow=0.05, att=0.28, rel=0.55,
            vib=0.17, vdel=0.25, glide=0.085, hp=120.0, det=9.0, body="violin"):
    cfg = dict(att=att, rel=rel, glide=glide, det=det, drift=3.5, vib=vib, vrate=5.6, vdel=vdel, jit=0.018,
               gamma=1.5, pan=pan, width=width, dip=0.10, env_pow=1.4, press=0.06)
    y, i0, c, ons = ensemble(notes, nv, cfg, rng)
    L = y.shape[1]
    y = dyn_lp(y, c, f_lo, f_hi)
    nz = filt(bw("bandpass", [2200, 8000]), rng.standard_normal((2, L)))
    bursts = onset_bursts(ons, i0, L, 0.05)
    y += nz * (0.5 * c + 1.2 * bursts * np.sqrt(np.maximum(c, 0))) * bow
    eq = {"violin": chain(peq(480, 1.5, 1.1), peq(1100, -1.5, 1.6), peq(2900, 1.2, 1.2), peq(330, -1.5, 1.4)),
          "viola": chain(peq(380, 1.5, 1.1), peq(1000, -1.0, 1.5), peq(260, -1.0, 1.4)),
          "cello": chain(peq(210, 1.5, 1.0), peq(320, -2.0, 1.2), peq(1500, 1.0, 1.0)),
          "bass": chain(peq(90, 1.5, 1.0), peq(300, -2.5, 1.0))}[body]
    y = filt(chain(eq, bw("highpass", hp)), y)
    return y, i0


def brass(notes, nv, rng, kind="horn", pan=0.0, width=0.3, growl=0.0, drive=1.2, breath=0.06, att=None):
    P = {"horn": dict(f_lo=320, f_hi=3200, att=0.09, rel=0.35, det=6.0, hp=70, vib=0.03),
         "trumpet": dict(f_lo=700, f_hi=6500, att=0.05, rel=0.25, det=5.0, hp=180, vib=0.04),
         "trombone": dict(f_lo=380, f_hi=4200, att=0.07, rel=0.3, det=7.0, hp=50, vib=0.0),
         "tuba": dict(f_lo=200, f_hi=1500, att=0.09, rel=0.35, det=6.0, hp=28, vib=0.0)}[kind]
    cfg = dict(att=att or P["att"], rel=P["rel"], glide=0.05, det=P["det"], drift=3.0, vib=P["vib"], vrate=4.8,
               vdel=0.45, jit=0.012, gamma=1.3, pan=pan, width=width, scoop=45.0, scoop_t=0.04, dip=0.05,
               env_pow=1.0, press=0.04)
    y, i0, c, ons = ensemble(notes, nv, cfg, rng)
    L = y.shape[1]
    cb = np.clip(c, 0, 1) ** 1.25  # brightness rises with loudness
    y = dyn_lp(y, cb, P["f_lo"], P["f_hi"])
    if growl > 0:
        gn = filt(bw("bandpass", [22, 70]), rng.standard_normal(L))
        gn /= np.std(gn) + 1e-9
        g = growl * np.clip((cb - 0.45) / 0.55, 0, 1)
        y *= 1.0 + 0.45 * g * np.tanh(gn)  # flutter AM, never changes the polarity
    pk = np.percentile(np.abs(y), 99.8) + 1e-9
    d = 1.0 + drive * cb
    y = np.tanh(d * y / pk) / np.tanh(d) * pk
    nz = filt(bw("bandpass", [450, 2600]), rng.standard_normal((2, L)))
    b = onset_bursts(ons, i0, L, 0.06, scale_slur=0.35)
    y += nz * (b * 0.9 + 0.12 * c) * breath * pk
    eq = chain(peq(300, -2.0, 1.1), bw("highpass", P["hp"]), bw("lowpass", 9000))
    if kind == "horn":
        eq = chain(eq, peq(520, 1.5, 1.0), peq(2800, -1.5, 1.2))
    return filt(eq, y), i0


def synth_pad(notes, nv, rng, pan=0.0, width=0.8, f_lo=900, f_hi=3200, att=0.9, rel=1.2, hp=160.0, det=14.0):
    cfg = dict(att=att, rel=rel, glide=0.12, det=det, drift=4.0, vib=0.0, jit=0.03, gamma=1.3, pan=pan,
               width=width, env_pow=1.6, press=0.03)
    y, i0, c, _ = ensemble(notes, nv, cfg, rng)
    y = dyn_lp(y, c, f_lo, f_hi)
    return filt(chain(bw("highpass", hp), peq(300, -2.0, 1.0)), y), i0


FORM = {  # (freq, dB, bandwidth) for F1..F5, classic choir formant tables
    "S": {"a": [(800, 0, 80), (1150, -6, 90), (2900, -32, 120), (3900, -20, 130), (4950, -50, 140)],
          "o": [(450, 0, 70), (800, -11, 80), (2830, -22, 100), (3800, -22, 130), (4950, -50, 135)]},
    "A": {"a": [(800, 0, 80), (1150, -4, 90), (2800, -20, 120), (3500, -36, 130), (4950, -60, 140)],
          "o": [(450, 0, 70), (800, -9, 80), (2830, -16, 100), (3500, -28, 130), (4950, -55, 135)]},
    "T": {"a": [(650, 0, 80), (1080, -6, 90), (2650, -9, 120), (2900, -10, 130), (3250, -24, 140)],
          "o": [(400, 0, 40), (800, -10, 80), (2600, -14, 100), (2800, -14, 120), (3000, -28, 120)]},
    "B": {"a": [(600, 0, 60), (1040, -7, 70), (2250, -10, 110), (2450, -10, 120), (2750, -22, 130)],
          "o": [(400, 0, 40), (750, -11, 80), (2400, -21, 100), (2600, -20, 120), (2900, -40, 120)]},
}


def formant(x, table):
    y = 0.0
    for f, db, b in table:
        y = y + filt(bw("bandpass", [max(30, f - b / 2), f + b / 2], 1), x) * 10 ** (db / 20.0)
    return y


def choir(notes, nv, part, rng, vowel=lambda t: 0.0, pan=0.0, width=0.5, att=0.4, rel=0.7):
    cfg = dict(att=att, rel=rel, glide=0.09, det=11.0, drift=5.0, vib=0.22, vrate=5.0, vdel=0.35, jit=0.03,
               gamma=1.4, pan=pan, width=width, dip=0.06, env_pow=1.3, press=0.07, scoop=20, scoop_t=0.06)
    y, i0, c, ons = ensemble(notes, nv, cfg, rng)
    L = y.shape[1]
    tilt = {"S": 700.0, "A": 550.0, "T": 380.0, "B": 260.0}[part]
    p = math.exp(-2 * math.pi * tilt / SR)
    y = signal.lfilter([1 - p], [1, -p], y, axis=-1) * 3.0  # glottal-like -12 dB/oct source
    y += filt(bw("bandpass", [600, 7000]), rng.standard_normal((2, L))) * 0.05 * c
    ya, yo = formant(y, FORM[part]["a"]), formant(y, FORM[part]["o"])
    w = np.clip(np.array([vowel(i0 / SR + k / SR) for k in range(0, L, 480)]), 0, 1)
    w = np.interp(np.arange(L), np.arange(0, L, 480), w)
    y = ya * (1 - w) + yo * w
    return filt(chain(bw("highpass", 110 if part != "B" else 70), peq(3000, -2.5, 1.0), peq(300, -1.5, 1.0)), y), i0


# =============================================================================================
# one-shot instruments
# =============================================================================================
def piano_note(rng, midi, vel, dur, rel=0.35):
    f0 = float(mtof(midi))
    B = 0.00011 * 2 ** ((midi - 48) / 12 * 1.25)
    tau1 = float(np.clip(9.0 * (f0 / 130.0) ** -0.55, 0.9, 16.0))
    Ls = min(dur + rel + 0.1, max(dur, 0.2) + 7.0)
    L = int(Ls * SR)
    t = np.arange(L) / SR
    y = np.zeros(L)
    nmax = int(min(40, 11000 / f0))
    x0 = 0.12 + 0.01 * rng.standard_normal()
    nc = 2.0 + 9.0 * vel ** 1.5
    for n in range(1, nmax + 1):
        fn = n * f0 * math.sqrt(1 + B * n * n)
        if fn > 16000:
            break
        a = abs(math.sin(math.pi * n * x0)) / n ** 0.7 * math.exp(-(n - 1) / nc)
        tau = tau1 / (1 + 0.012 * n * n * (f0 / 200) ** 0.5)
        for s, (dc, tf) in enumerate(((+1, 0.32), (-1, 1.0))):
            det = dc * (0.35 + 0.25 * rng.random()) / 1200.0 * math.log(2)
            y += 0.5 * a * np.sin(2 * math.pi * fn * (1 + det) * t + rng.uniform(0, 6.28)) * np.exp(-t / (tau * tf))
    hn = filt(bw("lowpass", 1500 + 5000 * vel), rng.standard_normal(L)) * np.exp(-t / 0.004) * 0.25 * vel
    thump = np.sin(2 * math.pi * 95 * t) * np.exp(-t / 0.02) * 0.12
    y += hn + thump
    y *= 1 - np.exp(-t / 0.0012)
    if midi < 89:
        k = int(dur * SR)
        if k < L:
            y[k:] *= np.exp(-(t[k:] - t[k]) / 0.16)
    y = fade_edges(y, 0.5, 30)
    return y * vel ** 1.3


def organ_note(rng, midi, dur, vel, reg, att=0.09, rel=0.45):
    REG = {"flute": [(1, 1.0), (2, 0.18), (3, 0.05), (4, 0.025)],
           "flute4": [(1, 1.0), (2, 0.55), (3, 0.06), (4, 0.12), (6, 0.02)],
           "principal": [(0.5, 0.38), (1, 1.0), (2, 0.6), (3, 0.24), (4, 0.33), (5, 0.08), (6, 0.11), (8, 0.08),
                         (10, 0.03), (12, 0.04)],
           "full": [(0.25, 0.25), (0.5, 0.5), (1, 1.0), (2, 0.7), (3, 0.3), (4, 0.42), (5, 0.1), (6, 0.16),
                    (8, 0.12), (10, 0.05), (12, 0.06), (16, 0.035)]}[reg]
    f0 = float(mtof(midi))
    L = int((dur + rel + 0.05) * SR)
    t = np.arange(L) / SR
    yl, yr = np.zeros(L), np.zeros(L)
    for r, a in REG:
        f = r * f0
        if f > 15000 or f < 31:  # no infrasonic 32' partials (headroom, speaker excursion)
            continue
        a *= min(1.0, 4000 / f) ** 0.5
        for ch, dc in ((0, 1.0), (1, -1.0)):
            det = 2 ** (dc * (0.9 + 0.6 * rng.random()) / 1200)
            ar = 1 - np.exp(-t / (att * (1 + 0.6 / max(r, 0.5))))
            s = a * np.sin(2 * math.pi * f * det * t + rng.uniform(0, 6.28)) * ar
            if ch == 0:
                yl += s
            else:
                yr += s
    chiff = filt(bw("bandpass", [min(3 * f0, 9000), min(6 * f0, 14000)]), rng.standard_normal((2, L)))
    chiff *= np.exp(-t / 0.03) * 0.08
    y = np.vstack([yl, yr]) + chiff
    k = int(dur * SR)
    env = np.ones(L)
    if k < L:
        env[k:] = np.cos(np.clip((t[k:] - t[k]) / rel, 0, 1) * math.pi / 2) ** 2
    return fade_edges(y * env, 1, 5) * vel ** 1.4 * 0.35


ORGAN_EQ = chain(bw("highpass", 40), peq(300, -2.5, 0.9))


def taiko(rng, f=58.0, vel=1.0, decay=0.5):
    L = int(1.4 * SR)
    t = np.arange(L) / SR
    fsw = f * (1 + 0.55 * np.exp(-t / 0.022))
    ph = 2 * math.pi * np.cumsum(fsw) / SR
    y = np.zeros(L)
    for r, a, dmul in ((1.0, 1.0, 1.0), (1.59, 0.45, 0.45), (2.14, 0.3, 0.32), (2.30, 0.22, 0.25),
                       (2.65, 0.15, 0.2), (3.16, 0.08, 0.15)):
        y += a * np.sin(r * ph + rng.uniform(0, 6.28)) * np.exp(-t / (decay * dmul))
    y += filt(bw("bandpass", [350, 2800]), rng.standard_normal(L)) * np.exp(-t / 0.011) * 0.9
    y += np.sin(2 * math.pi * f * 0.68 * t) * np.exp(-t / 0.09) * 0.25
    y *= 1 - np.exp(-t / 0.0007)
    return fade_edges(y, 0, 40) * vel ** 1.2


def timpani(rng, midi, vel=1.0, decay=2.0):
    f = float(mtof(midi))
    L = int(min(4.0, decay * 2.2) * SR)
    t = np.arange(L) / SR
    y = np.zeros(L)
    for r, a, dm in ((1.0, 1.0, 1.0), (1.504, 0.55, 0.7), (1.742, 0.35, 0.5), (2.0, 0.28, 0.45),
                     (2.245, 0.16, 0.35), (2.494, 0.1, 0.3), (0.55, 0.35, 0.12)):
        fr = f * r * (1 + 0.02 * np.exp(-t / 0.05))
        y += a * np.sin(2 * math.pi * np.cumsum(fr) / SR + rng.uniform(0, 6.28)) * np.exp(-t / (decay * dm))
    y += filt(bw("bandpass", [250, 2200]), rng.standard_normal(L)) * np.exp(-t / 0.008) * 0.5
    y *= 1 - np.exp(-t / 0.0008)
    return fade_edges(y, 0, 60) * vel ** 1.3


def sub_boom(rng, vel=1.0, f_end=31.0, decay=1.5):
    L = int(min(5.0, decay * 3.2) * SR)
    t = np.arange(L) / SR
    f = f_end + 60 * np.exp(-t / 0.10)
    y = np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-t / decay) * (1 - np.exp(-t / 0.003))
    y += filt(bw("lowpass", 2500), rng.standard_normal(L)) * np.exp(-t / 0.006) * 0.35
    y = np.tanh(1.6 * y) / np.tanh(1.6)
    return fade_edges(y, 0, 200) * vel


def crash(rng, vel=1.0, decay=2.4, rev=False, L_s=None):
    L = int((L_s or decay * 2.6) * SR)
    t = np.arange(L) / SR
    nz = filt(bw("highpass", 2800, 4), rng.standard_normal((2, L)))
    sh = 0
    for _ in range(6):
        fc = rng.uniform(3500, 11000)
        sh = sh + filt(bw("bandpass", [fc * 0.97, fc * 1.03]), rng.standard_normal((2, L))) * 2.5
    y = (nz + sh) * 0.3
    if rev:
        env = (t / t[-1]) ** 3.0
        y = fade_edges(y * env, 30, 12)
    else:
        env = np.exp(-t / decay) * (1 - np.exp(-t / 0.002))
        y = filt(bw("lowpass", 13000), y * env)
        y = fade_edges(y, 0, 200)
    return y * vel


def tick(rng, vel=0.5):
    L = int(0.05 * SR)
    t = np.arange(L) / SR
    f = rng.uniform(3600, 4600)
    y = np.sin(2 * math.pi * f * t) * np.exp(-t / 0.0035)
    y += filt(bw("highpass", 6000), rng.standard_normal(L)) * np.exp(-t / 0.0012) * 0.8
    y += np.sin(2 * math.pi * 1250 * t) * np.exp(-t / 0.006) * 0.25
    return fade_edges(y, 0.1, 5) * vel


def riser(rng, Ls, vel=1.0, m0=45, m1=81):
    L = int(Ls * SR)
    t = np.arange(L) / SR
    x = t / Ls
    m = m0 + (m1 - m0) * x ** 1.6
    y = 0
    for d in (-12, 0, 7, 12):
        y = y + blep_saw(mtof(m + d + 0.08 * rng.standard_normal()), rng.random())
    y = dyn_lp(y, x ** 1.4, 400, 6000) * 0.25
    nz = rng.standard_normal(L)
    nz = dyn_lp(filt(bw("highpass", 500), nz), x ** 1.2, 1500, 14000)
    out = pan2(y * 0.6, -0.2) + pan2(nz * 0.5, 0.2)
    out *= (x ** 2.6)
    return fade_edges(out, 20, 25) * vel


def pluck(rng, midi, vel, dur, bright):
    L = int((dur + 0.4) * SR)
    t = np.arange(L) / SR
    f = float(mtof(midi))
    x = blep_saw(np.full(L, f * 2 ** (7 / 1200)), rng.random()) + blep_saw(np.full(L, f * 2 ** (-7 / 1200)),
                                                                          rng.random())
    x += 0.4 * np.sin(2 * math.pi * f / 2 * t)
    env = (1 - np.exp(-t / 0.0015)) * np.exp(-t / (0.10 + 0.18 * bright))
    k = int(dur * SR)
    env[k:] *= np.exp(-(t[k:] - t[k]) / 0.05)
    yb = filt(bw("lowpass", 900 + 6500 * bright), x)
    yd = filt(bw("lowpass", 500 + 1300 * bright), x)
    w = np.exp(-t / 0.045)
    return fade_edges((yb * w + yd * (1 - w)) * env, 0.3, 10) * vel ** 1.4 * 0.5


def bass_note(rng, midi, vel, dur):
    L = int((dur + 0.08) * SR)
    t = np.arange(L) / SR
    f = float(mtof(midi))
    x = blep_saw(np.full(L, f * 1.002), rng.random()) + blep_saw(np.full(L, f * 0.998), rng.random())
    x = filt(bw("lowpass", 260 + 200 * vel, 4), x) * 0.5 + np.sin(2 * math.pi * f * t) * 0.9
    env = (1 - np.exp(-t / 0.004)) * (0.55 + 0.45 * np.exp(-t / 0.12))
    k = int(dur * SR)
    env[k:] *= np.exp(-(t[k:] - t[k]) / 0.025)
    return fade_edges(x * env, 0.5, 10) * vel ** 1.3


def spicc_note(rng, midi, vel, dur=0.15, nv=4):
    nv = max(2, int(round(nv * VSCALE)))
    L = int((dur + 0.3) * SR)
    t = np.arange(L) / SR
    y = np.zeros(L)
    for k in range(nv):
        f = float(mtof(midi + rng.normal(0, 0.07)))
        fr = f * (1 + 0.004 * np.exp(-t / 0.015))
        sh = int(rng.uniform(0, 0.009) * SR)
        s = blep_saw(fr, rng.random())
        y[sh:] += s[:L - sh]
    y /= math.sqrt(nv)
    env = (1 - np.exp(-t / 0.0025)) * (0.25 + 0.75 * np.exp(-t / 0.05))
    k = int(dur * SR)
    env[k:] *= np.exp(-(t[k:] - t[k]) / 0.045)
    y = filt(bw("lowpass", 1500 + 2200 * vel), y) * env
    y += filt(bw("bandpass", [1400, 5000]), rng.standard_normal(L)) * np.exp(-t / 0.007) * 0.35
    return fade_edges(y, 0.2, 10) * vel ** 1.5


def braam(rng, chord_midis, Ls, vel=1.0, nv=5):
    """Low brass braam: trombones + tubas in octaves / fifths, swell to a growling peak, synth sub."""
    out = None
    for m in chord_midis:
        kind = "tuba" if m < 45 else "trombone"
        notes = [(0.0, Ls, m, vel)]
        y, i0 = brass(notes, nv, rng, kind=kind, pan=rng.uniform(-0.25, 0.25), width=0.9, growl=1.0,
                      drive=2.0, breath=0.12, att=0.10)
        out = y if out is None else _sum_aligned(out, y)
    L = out.shape[1]
    t = (np.arange(L) + i0) / SR + 0.05
    env = np.clip(t / 0.2, 0, 1) ** 2 * (0.6 + 0.4 * np.exp(-np.maximum(t - 0.2, 0) / 1.0))
    env *= np.cos(np.clip((t - 0.05 - Ls) / 0.7, 0, 1) * math.pi / 2) ** 2  # release after the note
    m0 = min(chord_midis)
    sub = blep_saw(np.full(L, float(mtof(m0))), rng.random()) - blep_saw(np.full(L, float(mtof(m0)) * 1.004), 0.3)
    sub = filt(bw("lowpass", 140, 4), sub) * 0.6
    out = filt(bw("lowpass", 6500), out * env + pan2(sub * env, 0.0))
    return fade_edges(out, 0.0, 30.0), i0


def _sum_aligned(a, b):
    L = max(a.shape[1], b.shape[1])
    o = np.zeros((2, L))
    o[:, :a.shape[1]] += a
    o[:, :b.shape[1]] += b
    return o


# =============================================================================================
# reverb
# =============================================================================================
def hall_ir(rng, length=5.2, rt=(2.3, 3.1, 3.3, 2.4, 1.3), edges=(200, 750, 2600, 6500), pre=0.024,
            onset=0.07, n_er=28, er_span=0.085):
    n = int(length * SR)
    t = np.arange(n) / SR
    bands = [bw("lowpass", edges[0])] + [bw("bandpass", [edges[i], edges[i + 1]]) for i in range(len(edges) - 1)] \
        + [bw("highpass", edges[-1])]
    shared = rng.standard_normal(n)
    irs = []
    for ch in range(2):
        acc = np.zeros(n)
        for bi, (sos, T60) in enumerate(zip(bands, rt)):
            src = shared if bi == 0 else rng.standard_normal(n)  # coherent low end (mono-safe)
            acc += filt(sos, src) * np.exp(-6.908 * t / T60)
        acc *= np.clip((t - pre) / onset, 0, 1) ** 1.5
        er = np.zeros(n)
        for _ in range(n_er):
            pos = pre * 0.5 + rng.uniform(0, er_span)
            k = int(pos * SR)
            er[k:k + 32] += rng.choice([-1, 1]) * rng.uniform(0.5, 1.6) * math.exp(-pos / 0.05) * np.hanning(32)
        er = filt(bw("lowpass", 7000), er)
        acc = acc / np.sqrt(np.sum(acc ** 2)) + er / np.sqrt(np.sum(er ** 2) + 1e-12) * 0.45
        acc[-int(0.08 * SR):] *= np.linspace(1, 0, int(0.08 * SR))
        irs.append(acc / np.sqrt(np.sum(acc ** 2)))
    return np.array(irs)


def reverb(x, ir, cross=0.3):
    y = np.zeros_like(x)
    for ch in (0, 1):
        y[ch] = signal.oaconvolve(x[ch], ir[ch])[:x.shape[1]]
        y[1 - ch] += cross * signal.oaconvolve(x[ch], np.roll(ir[1 - ch], 97))[:x.shape[1]]
    return y / math.sqrt(1 + cross ** 2)


# =============================================================================================
# arrangement
# =============================================================================================
GROUPS = ["piano", "organ", "pad", "strings", "lowstr", "spicc", "arp", "bass", "brass", "braam", "choir",
          "drums", "fx"]
# group fader (dB) and reverb sends (hall, room)
GROUP_DB = {"piano": -1.0, "organ": -6.0, "pad": -11.0, "strings": -3.0, "lowstr": -4.5, "spicc": -2.5,
            "arp": -2.0, "bass": -13.5, "brass": -4.5, "braam": -7.0, "choir": 0.5, "drums": -4.0, "fx": -9.0}
SENDS = {"piano": (0.30, 0.0), "organ": (0.36, 0.0), "pad": (0.22, 0.0), "strings": (0.26, 0.05),
         "lowstr": (0.16, 0.06), "spicc": (0.10, 0.16), "arp": (0.16, 0.0), "bass": (0.0, 0.0),
         "brass": (0.22, 0.08), "braam": (0.16, 0.06), "choir": (0.32, 0.0), "drums": (0.07, 0.28), "fx": (0.2, 0.0)}
HITS = [8.0, 35.0, 48.0]          # silence before these hits
HIT_GAP = {8.0: 0.20, 35.0: 0.26, 48.0: 0.24}
MEL = {}                           # dry melody lines for the pitch check: name -> (mono buffer, ref notes)


def mel_add(name, y, i0, notes):
    if name not in MEL:
        MEL[name] = (np.zeros(N), [])
    buf, ref = MEL[name]
    m = y.mean(axis=0) if y.ndim == 2 else y
    a, b = max(0, i0), min(N, i0 + len(m))
    if b > a:
        buf[a:b] += m[a - i0:b - i0]
    ref += [(ts, te, mi) for ts, te, mi, _ in notes]


def arrange(seed):
    rng = np.random.default_rng(seed)
    B = {g: np.zeros((2, N), np.float32) for g in GROUPS}

    def R(name):
        return np.random.default_rng([seed, zlib.crc32(name.encode())])

    # ---------------- piano (forest phrase A, stream phrase B2, finale cell in A major)
    log("piano")
    pn = phrase(1, PH_A, 0, 0.5)
    pn = [(ts + (0.25 if i == 0 else 0.0), te, m, v * (0.85 if i in (3, 6) else 1.0)) for i, (ts, te, m, v) in
          enumerate(pn)]
    hi8 = [(ts, te, m + 12, v * 0.45) for ts, te, m, v in phrase(9, PH_B2, 0, 0.55)]  # stream, octave doubling
    pn += phrase(9, PH_B2, 0, 0.42)
    fin = [(tp(17, 1), tp(17, 2), nm("A4"), 0.42), (tp(17, 2), tp(17, 3), nm("E5"), 0.40),
           (tp(17, 3), tp(17, 4), nm("F#5"), 0.44), (tp(18, 0), tp(18, 3.5), nm("E5"), 0.36)]
    pn += fin
    lh = []
    for a, b, c in CHORDS:
        if b <= 8.0 + 1e-6:
            bass = VOX[c][0]
            lh += [(a + (0.25 if a == 0 else 0.0), b, bass - 12, 0.30), (a + 0.5 * (b - a), b, bass - 5, 0.20)]
        elif 28.5 - 1e-6 <= a < 35.0:
            bass = VOX[c][0]
            lh += [(a, b, bass - 12, 0.32), (a + 0.25 * (b - a), b, bass - 5, 0.22), (a + 0.5 * (b - a), b, bass, 0.2)]
    lh += [(54.0, 59.0, nm("A2"), 0.32), (tp(17, 0.5), 59.0, nm("E3"), 0.22)]
    rp = R("piano")
    pmel = np.zeros(N)
    for k, (ts, te, m, v) in enumerate(pn + hi8 + lh):
        y = piano_note(rp, m, v, te - ts)
        pan = np.clip((m - 64) / 40.0, -0.5, 0.5)
        add(B["piano"], pan2(y, pan), int(ts * SR))
        if k < len(pn):
            a = int(ts * SR)
            e = min(N, a + len(y))
            pmel[a:e] += y[:e - a]
    MEL["piano"] = (pmel, [(ts, te, m) for ts, te, m, _ in pn])

    # ---------------- organ: forest flutes, reveal principal, stream flutes, meadow 8'+4', flyover / peaks full
    log("organ")
    ro = R("organ")
    plan = [(0.0, 8.0, "flute", 0.42, 2.0), (8.0, 15.5, "principal", 0.62, 0.15), (28.5, 35.0, "flute", 0.40, 0.8),
            (35.0, 41.5, "flute4", 0.52, 0.12), (41.5, 48.0, "principal", 0.55, 0.12), (48.0, 54.0, "full", 0.70, 0.08),
            (54.0, 59.3, "full", 0.62, 0.10)]
    for t0, t1, reg, vel, att in plan:
        oct_up = 1 if t0 < 8.0 else 0  # forest: flutes an octave up (airy, no low-mid build-up)
        parts = [0, 1, 2, 3] + (["bass"] if reg in ("principal", "full") else [])
        for p in parts:
            o = -1 if p == "bass" else oct_up
            for ts, te, m, v in voice_lines(t0, t1, p, vel, o):
                te2 = te - (HIT_GAP[te] if te in HIT_GAP else 0.0)
                y = organ_note(ro, m, max(0.2, te2 - ts), v, reg, att=att if ts == t0 else 0.08,
                               rel=1.4 if t1 >= 59 else 0.35)
                add(B["organ"], y, int(ts * SR))
        if reg == "full":  # pedal 32'+16' doubling of the bass
            for ts, te, m, v in voice_lines(t0, t1, "bass", vel * 0.9, -1):
                add(B["organ"], organ_note(ro, m, te - ts, v, "flute", rel=1.2 if t1 >= 59 else 0.3), int(ts * SR))

    B["organ"] = filt(ORGAN_EQ, B["organ"].astype(np.float64)).astype(np.float32)

    # ---------------- synth pad (whole film, under everything; HP'd)
    log("pad")
    rpd = R("pad")
    pad_vel = lambda t: float(np.interp(t, [0, 8, 15.5, 28.5, 35, 41.5, 48, 54, 60],
                                        [0.32, 0.5, 0.45, 0.5, 0.55, 0.6, 0.7, 0.62, 0.6]))
    for seg in ((0.0, 7.8), (8.0, 34.74), (35.0, 47.76), (48.0, 59.0)):
        for p in (0, 1, 2, 3):
            ln = voice_lines(seg[0], seg[1], p, pad_vel, 0 if p < 3 else 0)
            y, i0 = synth_pad(ln, 5, rpd, pan=(-0.5, -0.15, 0.15, 0.5)[p], width=0.7)
            add(B["pad"], y, i0)

    # ---------------- shimmer: chord tones two octaves up, very soft, mostly reverb (air over the big shots)
    rsh = R("shimmer")
    for seg, vel in (((8.0, 15.4), 0.42), ((35.0, 47.76), 0.45), ((48.0, 59.0), 0.5)):
        for p, oc in ((2, 2), (3, 1)):
            y, i0 = synth_pad(voice_lines(*seg, p, vel, oc), 4, rsh, pan=(-0.7 if p == 2 else 0.7), width=0.5,
                              f_lo=3500, f_hi=9000, att=1.2, rel=1.5, hp=1200, det=7.0)
            add(B["pad"], y, i0, 0.8)

    # ---------------- strings (violins / violas legato), low strings sustained
    log("strings")
    rs = R("strings")

    def vln(notes, nv=10, pan=-0.45, name=None, **kw):
        y, i0 = strings(notes, nv, rs, pan=pan, width=0.5, **kw)
        add(B["strings"], y, i0)
        if name:
            mel_add(name, y, i0, notes)

    # reveal: violins sustain the top chord tone softly (the theme is in brass + celli)
    vln(voice_lines(8.0, 15.5, 3, 0.45, 1), nv=8, pan=-0.4, f_hi=8500, bow=0.07)
    vln(voice_lines(8.0, 15.5, 2, 0.42, 0), nv=6, pan=0.15, body="viola", f_hi=4000)
    # skim: descant E5 F5 E5 D5 over the ostinato
    vln(phrase(5, DESCANT, 0, 0.62), name="violins")
    # waterfall: phrase A (with the horns an octave below)
    vln(phrase(7, PH_A, 0, 0.72), nv=12, name="violins")
    # stream: violas / 2nd violins hold the harmony softly under the piano
    vln(voice_lines(28.5, 34.74, 2, 0.4, 0), nv=6, pan=0.15, body="viola", f_hi=3500)
    vln(voice_lines(28.5, 34.74, 3, 0.36, 0), nv=6, pan=-0.3, f_hi=4000)
    # meadow: the lift melody
    vln(phrase(11, PH_M, 0, 0.78), nv=14, name="violins", vib=0.19)
    # flyover: melody in octaves
    vln(phrase(13, PH_F, 0, 0.84), nv=12, name="violins")
    vln(phrase(13, PH_F, 1, 0.66), nv=8, pan=-0.55, f_hi=6000)
    # peaks + finale: the climax phrase, then the long C#6 (B5 -> C#6, Picardy third on top)
    pk = phrase(15, PH_P, 0, 0.95)
    pk[-1] = (pk[-1][0], 54.0, pk[-1][2], 0.95)
    vln(pk + [(54.0, 59.2, nm("C#6"), 0.66)], nv=14, name="violins", rel=1.2)
    vln([(a, b, m - 12, v * 0.85) for a, b, m, v in pk] + [(54.0, 59.2, nm("A4"), 0.6)], nv=10, pan=-0.15,
        f_hi=5000, rel=1.3)
    # violas / 2nd violins: inner chord voices from the meadow to the end
    for p, pan in ((1, 0.2), (2, 0.05)):
        vln(voice_lines(35.0, 47.76, p, 0.55, 0), nv=6, pan=pan, body="viola", f_hi=4200)
        vln(voice_lines(48.0, 59.2, p, lambda t: 0.75 if t < 54 else 0.6, 0), nv=6, pan=pan, body="viola",
            f_hi=4500, rel=1.2)
    # low strings: celli phrase B in the reveal, sustained bass lines from the meadow on
    rl = R("lowstr")

    def low(notes, nv, pan, body, name=None, **kw):
        y, i0 = strings(notes, nv, rl, pan=pan, width=0.35, body=body, **kw)
        add(B["lowstr"], y, i0)
        if name:
            mel_add(name, y, i0, notes)

    low(phrase(3, PH_B, -1, 0.75), 8, 0.35, "cello", f_lo=900, f_hi=3800, hp=60)
    low(voice_lines(8.0, 15.5, "bass", 0.65, -1), 6, 0.45, "bass", f_lo=500, f_hi=1600, hp=30, vib=0.08)
    for seg, vel in (((35.0, 47.76), 0.62), ((48.0, 59.2), 0.8)):
        low(voice_lines(*seg, "bass", vel, 0), 8, 0.35, "cello", f_lo=800, f_hi=3200, hp=50, rel=1.0)
        low(voice_lines(*seg, "bass", vel, -1), 6, 0.45, "bass", f_lo=450, f_hi=1500, hp=28, vib=0.08, rel=1.0)

    # ---------------- spiccato ostinato: the cell (root 5 6 5) in 16ths, accents 3+3+2
    log("spiccato")
    rsp = R("spicc")
    SIXTH = {"Am": 5, "Am_m": 5, "F": 9, "F_f": 9, "C": 9, "C_hi": 9, "C/E": 9, "G": 9, "G_m": 9, "G/B": 9,
             "Dm": 9, "Dm7": 9, "Gsus4": 9, "E": 8, "Esus4": 8}
    ROOT = {"G/B": 43, "C/E": 48}
    sp_dyn = [(5, 7, 0.78), (7, 9, 0.82), (9, 11, 0.5), (11, 13, 0.58), (13, 15, 0.9), (15, 17, 1.0)]
    for b0, b1, vel in sp_dyn:
        for bar in range(b0, b1):
            for s16 in range(16):
                beat = s16 / 4
                t = tp(bar, beat)
                if any(h - HIT_GAP[h] - 0.02 <= t < h for h in HITS):
                    continue
                if bar in (11, 12) and s16 % 2:
                    continue  # meadow: 8ths only (breath)
                c = chord_at(t + 1e-4)
                root = ROOT.get(c, VOX[c][0])
                while root > 47:
                    root -= 12
                cell = [0, 7, SIXTH.get(c, 9), 7][s16 % 4]
                acc = 1.0 if (s16 % 8) in (0, 3, 6) else 0.72
                cresc = 1.0 + 0.12 * (beat / 4)
                vv = vel * acc * cresc
                dur = 0.16 if bar < 15 else 0.14
                y = spicc_note(rsp, root + cell, min(1.0, vv), dur)
                add(B["spicc"], pan2(y, 0.3 + 0.1 * rsp.standard_normal()), int(t * SR))
                if bar >= 7 and (s16 % 2 == 0):  # basses double an octave below on 8ths
                    y = spicc_note(rsp, root + cell - 12, min(1.0, vv * 0.9), dur + 0.03, nv=3)
                    add(B["spicc"], pan2(y, 0.45), int(t * SR))

    # ---------------- synth arp (16ths, chord tones, ping-pong) + pulse bass (8ths)
    log("arp / bass")
    ra = R("arp")
    arp_plan = [(5, 7, 0.55, (0.15, 0.55)), (7, 9, 0.6, (0.45, 0.65)), (9, 11, 0.45, (0.35, 0.15)),
                (11, 13, 0.62, (0.6, 0.8)), (13, 15, 0.7, (0.55, 0.9)), (15, 17, 0.75, (0.8, 1.0))]
    arp_idx = [0, 1, 2, 3, 2, 1, 2, 3]
    for b0, b1, vel, (br0, br1) in arp_plan:
        for bar in range(b0, b1):
            for s16 in range(16):
                t = tp(bar, s16 / 4)
                if any(h - HIT_GAP[h] - 0.02 <= t < h for h in HITS):
                    continue
                c = chord_at(t + 1e-4)
                vx = sorted(VOX[c][1])
                m = vx[arp_idx[s16 % 8]] + 12
                prog = ((bar - b0) * 16 + s16) / ((b1 - b0) * 16)
                br = br0 + (br1 - br0) * prog
                acc = 1.0 if s16 % 4 == 0 else 0.75
                y = pluck(ra, m, vel * acc, 0.12, br)
                add(B["arp"], pan2(y, -0.25 if s16 % 2 else 0.25), int(t * SR))
    # ping-pong delay (dotted 8th at 73.85 BPM), inside the arp bus
    arp = B["arp"].astype(np.float64)
    dl = int(0.75 * 60 / 73.846 * SR)
    wet = np.zeros_like(arp)
    src = filt(chain(bw("highpass", 400), bw("lowpass", 5000)), arp.mean(axis=0))
    for k, g in enumerate((0.42, 0.28, 0.18, 0.11)):
        ch = k % 2
        d = dl * (k + 1)
        wet[1 - ch, d:] += g * src[:-d]
    B["arp"] = (arp + wet).astype(np.float32)

    rb = R("bass")
    bass_plan = [(5, 7, 0.75), (7, 9, 0.78), (9, 11, 0.5), (11, 13, 0.65), (13, 15, 0.85), (15, 17, 0.95)]
    for b0, b1, vel in bass_plan:
        for bar in range(b0, b1):
            for e in range(8):
                t, t2 = tp(bar, e / 2), tp(bar, e / 2 + 0.5)
                if any(h - HIT_GAP[h] - 0.02 <= t < h for h in HITS):
                    continue
                c = chord_at(t + 1e-4)
                root = VOX[c][0]
                while root > 40:
                    root -= 12
                acc = 1.0 if e % 2 == 0 else 0.7
                y = bass_note(rb, root, vel * acc, (t2 - t) * 0.82)
                add(B["bass"], pan2(y, 0.0), int(t * SR))
    # finale / forest drone (A1 sine + octave, very soft)
    tt = np.arange(int(8.0 * SR)) / SR
    dr = (np.sin(2 * math.pi * 55 * tt) + 0.3 * np.sin(2 * math.pi * 110 * tt + 1)) * smoothstep(tt / 3.0) * \
        np.cos(np.clip((tt - 7.2) / 0.6, 0, 1) * math.pi / 2) ** 2 * 0.18
    add(B["bass"], pan2(dr, 0), 0)

    # ---------------- brass: horns theme, trumpets at the climax, trombone chords, braams
    log("brass")
    rbr = R("brass")

    def hrn(notes, nv=4, kind="horn", pan=-0.25, name=None, **kw):
        y, i0 = brass(notes, nv, rbr, kind=kind, pan=pan, **kw)
        add(B["brass"], y, i0)
        if name:
            mel_add(name, y, i0, notes)

    hrn(phrase(3, PH_B, -1, 0.78), 6, name="horns", growl=0.2)
    hrn(phrase(7, PH_A, -1, 0.80), 6, name="horns", growl=0.2)
    hrn(phrase(13, PH_F, 0, 0.88), 6, name="horns", growl=0.35)
    pkh = phrase(15, PH_P, -1, 1.0)
    pkh[-1] = (pkh[-1][0], 54.0, pkh[-1][2], 1.0)
    hrn(pkh + [(54.0, 58.6, nm("A4"), 0.55)], 8, name="horns", growl=0.5)
    hrn(phrase(15, PH_P, 0, 0.74), 4, kind="trumpet", pan=0.1, growl=0.2, drive=1.6)
    # trombones: warm chord at the meadow lift, chords in flyover / peaks / finale
    for seg, vel in (((35.0, 41.5), 0.46), ((41.5, 47.76), 0.7), ((48.0, 54.0), 0.8), ((54.0, 58.6), 0.45)):
        for p, oc in (("bass", 0), (0, 0)):
            hrn(voice_lines(*seg, p, vel, oc), 4, kind="trombone", pan=0.2 if p == "bass" else 0.3, growl=0.3)
        hrn(voice_lines(*seg, "bass", vel * 0.9, -1), 3, kind="tuba", pan=0.25)
    # braams: reveal (Am), flyover arrival (F), peaks on every chord
    rbm = R("braam")
    for t0, key, Ls, vel in ((8.0, "Am", 3.4, 0.86), (41.5, "F_f", 2.8, 0.78), (48.0, "Am", 1.5, 0.92),
                             (49.5, "F", 1.5, 0.84), (51.0, "C", 1.5, 0.9), (52.5, "G", 1.5, 0.88)):
        bass = VOX[key][0]
        while bass > 45:
            bass -= 12
        mids = [bass - 12, bass, bass + 7, bass + 12] if bass - 12 >= 28 else [bass, bass + 7, bass + 12]
        y, i0 = braam(rbm, mids, Ls, vel, nv=5)
        add(B["braam"], y, int(t0 * SR) + i0)

    # ---------------- choir: meadow (ah), flyover (oh), peaks (ah, sopranos melody), finale (oh)
    log("choir")
    rc = R("choir")
    vow = lambda t: float(np.interp(t, [35, 41.0, 42.0, 47.5, 48.0, 53.5, 54.5, 60], [0, 0, 1, 1, 0, 0, 1, 1]))
    seg_vel = lambda t: float(np.interp(t, [35, 41.5, 48, 54, 59], [0.62, 0.62, 0.85, 0.7, 0.62]))
    for seg in ((35.0, 47.76), (48.0, 59.2)):
        for part, p, oc, pan in (("B", "bass", 0, 0.4), ("T", 0, 0, 0.15), ("A", 1, 0, -0.1), ("A", 2, 0, -0.2)):
            ln = voice_lines(*seg, p, seg_vel, oc)
            y, i0 = choir(ln, 6, part, rc, vow, pan=pan, width=0.5, rel=1.2 if seg[1] > 59 else 0.6)
            add(B["choir"], y, i0)
    sop = phrase(11, PH_M, 0, 0.62) + phrase(13, PH_F, 1, 0.6, legato=0.98)
    y, i0 = choir(sop, 8, "S", rc, vow, pan=-0.4, width=0.5)
    add(B["choir"], y, i0)
    sp = phrase(15, PH_P, 0, 0.85)
    sp[-1] = (sp[-1][0], 54.0, sp[-1][2], 0.85)
    y, i0 = choir(sp + [(54.0, 59.2, nm("A5"), 0.55)], 8, "S", rc, vow, pan=-0.4, width=0.5, rel=1.2)
    add(B["choir"], y, i0)
    mel_add("sopranos", y, i0, sp)

    # ---------------- percussion
    log("drums / fx")
    rd = R("drums")
    D = B["drums"]

    def taiko_ens(t, vel, f=58.0, n=3, spread=0.35, decay=0.5):
        for k in range(n):
            dt = rd.uniform(0, 0.014) if k else 0.0
            y = taiko(rd, f * rd.uniform(0.93, 1.07), vel * rd.uniform(0.85, 1.0), decay)
            add(D, pan2(y, (k - (n - 1) / 2) * spread), int((t + dt) * SR))

    # ticks (the Interstellar clock): beats in forest / reveal, 8ths later, 16ths at the build
    for bar in range(1, 17):
        div = 1 if bar <= 4 else (2 if bar in (9, 11, 12) else 4)
        for k in range(4 * div):
            t = tp(bar, k / div)
            if any(h - HIT_GAP[h] - 0.02 <= t < h for h in HITS):
                continue
            if t < 0.5:
                continue
            v = (0.28 if bar <= 2 else 0.34) * (1.0 if k % div == 0 else 0.6)
            add(D, pan2(tick(rd, v), 0.35 if k % 2 else -0.35), int(t * SR))
    # taiko groove (3+3+2 | 3+3+2 per bar), toms fills, half-time in the meadow
    groove = [0, 3, 6, 8, 11, 14]
    gv = [1.0, 0.7, 0.82, 0.92, 0.7, 0.78]
    for bar, vel in [(5, 0.8), (6, 0.85), (7, 0.82), (8, 0.88), (13, 0.95), (14, 1.0), (15, 1.0), (16, 1.0)]:
        for s, g in zip(groove, gv):
            t = tp(bar, s / 4)
            if any(h - HIT_GAP[h] - 0.25 <= t < h for h in HITS):
                continue
            taiko_ens(t, vel * g)
        if bar in (6, 8, 14, 16):
            for s in (12, 13, 14, 15):
                t = tp(bar, s / 4)
                if any(h - HIT_GAP[h] - 0.02 <= t < h for h in HITS):
                    continue
                y = taiko(rd, 120 + 15 * (s - 12), vel * (0.5 + 0.12 * (s - 12)), 0.3)
                add(D, pan2(y, -0.4 + 0.25 * (s - 12)), int(t * SR))
        if bar >= 15:  # climax: extra 8th pulse on the low drums
            for e in range(8):
                if e * 2 not in groove:
                    taiko_ens(tp(bar, e / 2), 0.55, f=50.0, n=2, spread=0.6)
    for bar in (11, 12):  # meadow half-time
        for b in (0, 2):
            taiko_ens(tp(bar, b), 0.72 if b == 0 else 0.6, f=54.0)
    # stream: soft single taiko on the downbeats, then the build (toms 8ths -> 16ths, timpani roll)
    taiko_ens(tp(9, 0), 0.5)
    for k in range(16):
        t = tp(10, k / 4)
        if t >= 35.0 - HIT_GAP[35.0] - 0.02:
            break
        if k < 8 and k % 2:
            continue
        y = taiko(rd, 110 + 4 * k, 0.25 + 0.035 * k, 0.28)
        add(D, pan2(y, 0.4 if k % 2 else -0.4), int(t * SR))
    # timpani: reveal bar 4, stream roll, meadow, peaks downbeats, finale
    for t, m, v in ((tp(4, 0), "C2", 0.7), (35.0, "C2", 0.95), (tp(12, 0), "A1", 0.7), (41.5, "F2", 0.85),
                    (48.0, "A1", 1.0), (51.0, "C2", 1.0), (54.0, "A1", 0.6)):
        add(D, pan2(timpani(rd, nm(m), v), -0.15), int(t * SR))
    t = tp(4, 2.4)  # reveal -> skim: G1 roll crescendo into the ostinato
    while t < 15.45:
        x = (t - tp(4, 2.4)) / (15.45 - tp(4, 2.4))
        add(D, pan2(timpani(rd, nm("G1"), 0.15 + 0.45 * x ** 1.6, 1.0), -0.15 + 0.1 * rd.standard_normal()),
            int(t * SR))
        t += 0.065 + 0.006 * rd.standard_normal()
    t = tp(10, 0)
    while t < 35.0 - HIT_GAP[35.0] - 0.06:
        x = (t - tp(10, 0)) / (35.0 - tp(10, 0))
        add(D, pan2(timpani(rd, nm("G1"), 0.18 + 0.5 * x ** 1.8, 1.2), -0.15 + 0.1 * rd.standard_normal()),
            int(t * SR))
        t += 0.062 + 0.006 * rd.standard_normal()
    # big hits: sub boom + taiko ensemble (+ crash)
    F = B["fx"]
    for t, v, big in ((8.0, 1.0, True), (15.5, 0.7, False), (35.0, 0.9, True), (41.5, 0.85, False),
                      (48.0, 1.0, True), (51.0, 0.85, False), (54.0, 0.55, False)):
        add(F, pan2(sub_boom(rd, v, decay=1.8 if big else 1.2), 0), int(t * SR))
        taiko_ens(t, v, f=52.0, n=4, spread=0.3, decay=0.7)
        if t in (35.0, 41.5, 48.0, 15.5):
            add(F, crash(rd, v * (0.55 if t == 15.5 else 0.8)), int(t * SR))
    # reverse cymbals into the hits, risers into 35 and 48 (and a soft one into 8)
    for t_hit, Ls, v in ((8.0, 2.2, 0.5), (15.5, 1.6, 0.5), (35.0, 2.4, 0.8), (41.5, 1.6, 0.6), (48.0, 2.6, 0.9),
                         (54.0, 2.0, 0.55)):
        y = crash(rd, v, rev=True, L_s=Ls)
        end = t_hit - HIT_GAP.get(t_hit, 0.02)
        add(F, y, int(end * SR) - y.shape[1])
    for t_hit, Ls, v in ((8.0, 3.0, 0.35), (35.0, 3.4, 0.75), (48.0, 3.0, 0.9)):
        y = riser(rd, Ls, v, m0=45 if t_hit != 8.0 else 50, m1=81 if t_hit != 8.0 else 74)
        end = t_hit - HIT_GAP[t_hit]
        add(F, y, int(end * SR) - y.shape[1])
    return B


def reverse_swells(B, ir, seed):
    """Zimmer-style reverse reverb swells of the arriving chord, ending on the hits."""
    rng = np.random.default_rng([seed, 77])
    for t_hit, key, Ls, v in ((8.0, "Am", 2.0, 0.6), (35.0, "C_hi", 2.2, 0.7), (48.0, "Am", 2.2, 0.8),
                              (54.0, "A", 2.4, 0.7), (15.5, "Am", 1.4, 0.4)):
        bass, vx, _ = VOX[key]
        L = int(1.0 * SR)
        src = np.zeros((2, L))
        for m in vx + [bass]:
            src += organ_note(rng, m, 0.6, 0.8, "principal")[:, :L]
        wet = np.vstack([signal.oaconvolve(src[c], ir[c])[:int(Ls * SR)] for c in (0, 1)])
        wet = wet[:, ::-1] * (np.linspace(0, 1, wet.shape[1]) ** 2)
        wet = fade_edges(wet, 50, 15)
        wet *= v / (np.max(np.abs(wet)) + 1e-9) * 0.5
        end = t_hit - HIT_GAP.get(t_hit, 0.03)
        add(B["fx"], wet, int(end * SR) - wet.shape[1])


def automation(g):
    """Per-group dB over time (shot arc)."""
    A = {
        "piano": [(0, 0), (28.4, 0), (28.5, 1), (53.9, 1), (54.2, 6.5), (60, 6.5)],
        "organ": [(0, 0), (60, 0)],
        "pad": [(0, -2), (8, 0), (15.5, -1), (28.5, 0), (60, 0)],
        "strings": [(0, 0), (60, 0)],
        "drums": [(0, 0), (22, 0), (22.2, -1.5), (28.5, -1.5), (35, 0), (41.0, 0), (41.5, 1.5), (47.5, 1.5),
                  (48, 2.5), (54, 2.5), (54.3, 0), (60, 0)],
        "spicc": [(0, 0), (15.5, 3), (22, 3), (22.3, 1), (60, 1)],
        "brass": [(0, 0), (47.8, 0), (48, -1.0), (54, -1.0), (54.3, -2.5), (60, -2.5)],
    }
    return A.get(g)


def duck_curve():
    """Silence before the hits: sustained layers drop out just before 8, 35 and 48 s."""
    tt = np.arange(0, N, 48) / SR
    db = np.zeros(len(tt))
    for h in HITS:
        gap = HIT_GAP[h]
        a, b = h - gap - 0.06, h - gap
        db = np.minimum(db, np.interp(tt, [a, b, h - 0.004, h], [0, -40, -40, 0], left=0, right=0))
    return np.interp(np.arange(N), np.arange(0, N, 48), 10 ** (db / 20))


# =============================================================================================
# mix + master
# =============================================================================================
# section ride (dB on every group, ahead of the reverb sends): the dramatic arc per shot
SECTION_TRIM = {"forest": 5.5, "reveal": 1.0, "skim": 3.5, "waterfall": 2.0, "stream": 3.5, "meadow": 0.5,
                "flyover": 0.0, "peaks": -1.5, "finale": 0.5}


def section_trim():
    pts = []
    for i, (name, a, b) in enumerate(SHOTS):
        g = SECTION_TRIM[name]
        pts += [(a, g), (b - 0.5, g)] if i < len(SHOTS) - 1 else [(a, g), (b, g)]
    return curve(pts, smooth_s=0.08)


NATURE_REL = {"forest": -10.5, "reveal": -14.0, "skim": -13.0, "waterfall": -6.5, "stream": -7.5,
              "meadow": -12.0, "flyover": -14.0, "peaks": -15.0, "finale": -11.0}
TARGET = -14.0
TP_MAX = -1.05


def mix_music(B, seed):
    rng = np.random.default_rng([seed, 5])
    log("reverb IRs")
    hall = hall_ir(rng)
    room = hall_ir(rng, length=1.6, rt=(0.8, 1.0, 1.1, 0.8, 0.5), pre=0.008, onset=0.02, n_er=18, er_span=0.03)
    reverse_swells(B, hall, seed)
    duck = duck_curve()
    trim = section_trim()
    dry = np.zeros((2, N))
    hall_in = np.zeros((2, N))
    room_in = np.zeros((2, N))
    stems = {}
    for g in GROUPS:
        x = B[g].astype(np.float64) * 10 ** (GROUP_DB[g] / 20.0)
        au = automation(g)
        if au:
            x *= curve(au)
        if g == "drums":  # tape-like transient shaving of the taiko peaks (keeps the master limiter light)
            thr = 0.8 * np.max(np.abs(x))
            x = thr * np.tanh(x / thr)
        if g not in ("drums", "fx"):
            x *= duck
        x *= trim
        stems[g] = x.astype(np.float32)
        dry += x
        hs, rs = SENDS[g]
        if hs:
            hall_in += x * hs
        if rs:
            room_in += x * rs
    log("convolution")
    hall_in = filt(chain(bw("highpass", 220), peq(400, -2, 0.9), bw("lowpass", 9500)), hall_in)
    room_in = filt(chain(bw("highpass", 160), bw("lowpass", 8000)), room_in)
    wet_h = reverb(hall_in, hall, 0.35)
    wet_r = reverb(room_in, room, 0.25)
    stems["hall"] = wet_h.astype(np.float32)
    stems["room"] = wet_r.astype(np.float32)
    mus = dry + wet_h + wet_r
    # music bus EQ: clean sub, de-mud, soften 3 kHz, a little air
    mus = filt(chain(bw("highpass", 26, 4), peq(285, -3.0, 0.9), peq(3200, -0.5, 1.1), shelf(7000, 2.5, "high")),
               mus)
    mus *= 10 ** ((-16.0 - an.integrated_lufs(mus)) / 20)  # glue comp works on a known level
    mus, gr = st.bus_compressor(mus, thr_db=-17.0, ratio=1.6, knee_db=8.0, att=0.03, rel=0.35)
    log(f"music glue comp max GR {gr:.1f} dB")
    return mus, stems


def master(mix, target=TARGET):
    """Normalise to the target, then a look-ahead limiter until the 4x true peak is <= TP_MAX."""
    L0 = an.integrated_lufs(mix)
    ceil, off, gr = -1.35, 0.0, 0.0
    for it in range(8):
        x = mix * 10 ** ((target + off - L0) / 20)
        y, gr = st.limiter(x, ceil, look_ms=2.5, rel_ms=90.0)
        tpk = an.true_peak_db(y)
        L2 = an.integrated_lufs(y)
        if tpk <= TP_MAX and abs(L2 - target) < 0.06:
            break
        if tpk > TP_MAX:
            ceil -= (tpk - TP_MAX) + 0.03
        off += target - L2
    g = st.limiter.last_gain
    return y, dict(lufs=round(L2, 2), tp=round(tpk, 2), limiter_gr=round(gr, 2), ceiling=round(ceil, 2),
                   pre_gain_db=round(target + off - L0, 2), iters=it + 1,
                   gr_max_at_s=round(st.limiter.last_gr_time, 2),
                   ms_over_1db=round(float(np.sum(g < 10 ** (-1 / 20)) / SR * 1000), 1),
                   ms_over_2db=round(float(np.sum(g < 10 ** (-2 / 20)) / SR * 1000), 1))


def nature_mix(music, nature):
    km, kn = an.k_weight(music), an.k_weight(nature)
    tt = np.arange(0, N, 480) / SR
    db = np.zeros(len(tt))
    pts = []
    for name, a, b in SHOTS:
        lm = an.segment_lufs(km, a + 0.3, b - 0.3)
        ln = an.segment_lufs(kn, a + 0.3, b - 0.3)
        g = (lm + NATURE_REL[name]) - ln
        pts.append((a, b, float(np.clip(g, -30, 8))))
    for i, (a, b, g) in enumerate(pts):
        db[(tt >= a - 0.3) & (tt < b - 0.3)] = g
    db[tt >= pts[-1][0] - 0.3] = pts[-1][2]
    db = gaussian_filter1d(db, 0.35 * SR / 480)
    g = np.interp(np.arange(N), np.arange(0, N, 480), 10 ** (db / 20))
    return nature * g, pts


def final_fade(x):
    t = np.arange(N) / SR
    g = np.cos(np.clip((t - 56.6) / (59.9 - 56.6), 0, 1) * math.pi / 2) ** 2
    g[t >= 59.9] = 0.0
    g *= np.clip(t / 0.02, 0, 1)
    return x * g


# =============================================================================================
# analysis
# =============================================================================================
def pitch_track(x, hop=0.01, win=0.085, fmin=90.0, fmax=1400.0):
    W, H, nfft = int(win * SR), int(hop * SR), 16384
    nfr = (len(x) - W) // H
    w = np.hanning(W)
    f0 = np.full(nfr, np.nan)
    en = np.zeros(nfr)
    kb0, kb1 = int(fmin * nfft / SR), int(fmax * nfft / SR)
    for c0 in range(0, nfr, 400):
        idx = (np.arange(c0, min(nfr, c0 + 400)) * H)[:, None] + np.arange(W)
        fr = x[idx] * w
        en[c0:c0 + len(idx)] = np.sum(fr ** 2, axis=1)
        P = np.abs(np.fft.rfft(fr, nfft, axis=1)) + 1e-12
        m = P.shape[1] // 3
        hps = np.log(P[:, :m]) + np.log(P[:, ::2][:, :m]) + np.log(P[:, ::3][:, :m])
        k = kb0 + np.argmax(hps[:, kb0:kb1], axis=1)
        f0[c0:c0 + len(idx)] = k * SR / nfft
    edb = 10 * np.log10(en + 1e-20)
    f0[edb < edb.max() - 35] = np.nan
    t = (np.arange(nfr) * H + W / 2) / SR
    return t, 69 + 12 * np.log2(f0 / 440.0)


def note_accuracy(t, p, ref):
    ok = tot = octv = 0
    for (a, b, m) in ref:
        sel = (t > a + 0.07) & (t < b - 0.03) & np.isfinite(p)
        if not sel.any():
            continue
        e = p[sel] - m
        tot += sel.sum()
        ok += np.sum(np.abs(e) < 0.5)
        octv += np.sum(np.abs(np.abs(e) - 12) < 0.5)
    return (ok / tot if tot else float("nan")), (octv / tot if tot else float("nan")), int(tot)


def chord_check(x):
    """Pitch-class energy per chord window of the harmonic stems: share inside the chord tones."""
    mono = x.mean(axis=0) if x.ndim == 2 else x
    res = []
    for a, b, c in CHORDS:
        i0, i1 = int((a + 0.12) * SR), int(min(b, 59.0) * SR)
        if i1 - i0 < 4096:
            continue
        seg = mono[i0:i1] * np.hanning(i1 - i0)
        P = np.abs(np.fft.rfft(seg)) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        sel = (f > 60) & (f < 1800)
        pcs = np.round(12 * np.log2(f[sel] / 440.0) + 69).astype(int) % 12
        ch = np.bincount(pcs, weights=P[sel], minlength=12)
        bass, vx, lab = VOX[c]
        tones = set([bass % 12] + [m % 12 for m in vx])
        if c == "A":
            tones.add(nm("B4") % 12)
        share = float(sum(ch[k] for k in tones) / (ch.sum() + 1e-20))
        res.append(dict(t=round(a, 2), chord=lab, share=round(share, 3)))
    return res


def analyze(out_dir, mix, music, nature, stems, nat_pts, minfo):
    os.makedirs(out_dir, exist_ok=True)
    rep = {}
    xk = an.k_weight(mix)
    mk, nk = an.k_weight(music), an.k_weight(nature)
    rep["integrated_lufs"] = round(an.integrated_lufs(xk=xk), 2)
    rep["true_peak_dbtp"] = round(an.true_peak_db(mix), 2)
    rep["sample_peak_dbfs"] = round(an.sample_peak_db(mix), 2)
    rep["frames"] = int(mix.shape[1])
    rep["clipped"] = int(np.sum(np.abs(mix) >= 0.9999))
    rep["dc"] = [float(f"{v:.2e}") for v in mix.mean(axis=1)]
    rep["last_100ms_dbfs"] = round(20 * np.log10(np.max(np.abs(mix[:, -4800:])) + 1e-20), 1)
    rep["clicks"] = an.click_scan(mix)
    rep["clicks_music"] = an.click_scan(music)
    rep["lr_corr"] = round(float(np.corrcoef(mix[0], mix[1])[0, 1]), 3)
    mono = mix.mean(axis=0, keepdims=True)
    rep["mono_loss_db"] = round(an.integrated_lufs(np.vstack([mono, mono])) - rep["integrated_lufs"], 2)
    ts, ls = an.short_term(xk)
    rep["max_short_term"] = round(float(np.max(ls)), 2)
    rep["master"] = minfo
    shots = []
    bands = [(20, 60), (60, 200), (200, 400), (400, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
    for name, a, b in SHOTS:
        e = dict(shot=name, mix=round(an.segment_lufs(xk, a, b), 1), music=round(an.segment_lufs(mk, a, b), 1),
                 nature=round(an.segment_lufs(nk, a, b), 1))
        seg = mix[:, int(a * SR):int(b * SR)].mean(axis=0)
        P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        tot = P[(f > 20) & (f < 16000)].sum()
        e["bands_db"] = {f"{lo}-{hi}": round(10 * np.log10(P[(f >= lo) & (f < hi)].sum() / tot + 1e-20), 1)
                         for lo, hi in bands}
        x = mix[:, int(a * SR):int(b * SR)]
        e["crest_db"] = round(20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12)), 1)
        e["corr"] = round(float(np.corrcoef(x[0], x[1])[0, 1]), 2)
        e["groups"] = {g: round(an.segment_lufs(an.k_weight(s[:, int(a * SR):int(b * SR)]), 0, b - a), 1)
                       for g, s in stems.items()}
        shots.append(e)
    rep["shots"] = shots
    tm, lm = an.momentary(xk)
    rep["cuts_momentary"] = {f"{h:g}": dict(before=round(float(np.max(lm[(tm > h - 0.35) & (tm < h - 0.02)])), 1),
                                             after=round(float(np.max(lm[(tm > h) & (tm < h + 0.6)])), 1))
                             for h in (8.0, 15.5, 22.0, 28.5, 35.0, 41.5, 48.0, 54.0)}
    rep["nature_gain_db"] = {SHOTS[i][0]: round(g, 1) for i, (_, _, g) in enumerate(nat_pts)}
    # composer checks
    acc = {}
    for name, (buf, ref) in MEL.items():
        if not ref:
            continue
        lo = min(m for _, _, m in ref)
        hi = max(m for _, _, m in ref)
        t, p = pitch_track(buf, fmin=float(mtof(lo - 3)), fmax=float(mtof(hi + 3)))
        r, o, n = note_accuracy(t, p, ref)
        acc[name] = dict(within_half_semitone=round(r, 3), octave_err=round(o, 3), frames=n)
    rep["melody_accuracy"] = acc
    harm = stems["organ"] + stems["pad"] + stems["choir"] + stems["strings"] + stems["lowstr"]
    cc = chord_check(harm)
    rep["chord_share_min"] = min(c["share"] for c in cc)
    rep["chord_share_mean"] = round(float(np.mean([c["share"] for c in cc])), 3)
    rep["chords"] = cc
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(rep, f, indent=1)
    shots_d = [dict(start=a, end=b, env=n) for n, a, b in SHOTS]
    an.render_panel(mix, "B hybrid: mix", shots_d, DUR, width=1600, spec_h=420, db_lo=-120.0, db_hi=-22.0).save(
        os.path.join(out_dir, "spec_mix.png"))
    an.render_panel(music, "B hybrid: music only", shots_d, DUR, width=1600, spec_h=420, db_lo=-120.0,
                    db_hi=-22.0).save(
        os.path.join(out_dir, "spec_music.png"))
    panels = [("mix", mix), ("music", music), ("nature", nature)] + [(g, s) for g, s in stems.items()]
    an.overview_png(os.path.join(out_dir, "spec_overview.png"), panels, shots_d, DUR, width=1400, spec_h=90)
    for i, (name, a, b) in enumerate(SHOTS):
        an.zoom_png(os.path.join(out_dir, f"spec_{i}_{name}.png"), mix, max(0, a - 0.5), min(DUR, b + 0.5),
                    f"B hybrid {name}", width=1200, spec_h=360, db_lo=-120.0, db_hi=-22.0)
    return rep


def print_report(rep):
    log(f"integrated {rep['integrated_lufs']} LUFS, TP {rep['true_peak_dbtp']} dBTP, sample peak "
        f"{rep['sample_peak_dbfs']}, frames {rep['frames']}, clipped {rep['clipped']}, DC {rep['dc']}, "
        f"last100ms {rep['last_100ms_dbfs']} dBFS")
    c = rep["clicks"]
    log(f"clicks: max ratio {c['max_ratio']:.1f} hits {c['n_hits']} {c['hit_times'][:8]} | HF {c['hf_hits']} "
        f"{c['hf_hit_times'][:10]}")
    log(f"LR corr {rep['lr_corr']}, mono loss {rep['mono_loss_db']} dB, max short-term {rep['max_short_term']}")
    log(f"master {rep['master']}")
    gs = list(rep["shots"][0]["groups"].keys())
    print("shot       mix   music nature crest corr | " + " ".join(f"{g[:6]:>6}" for g in gs))
    for e in rep["shots"]:
        print(f"{e['shot']:<10}{e['mix']:6.1f}{e['music']:7.1f}{e['nature']:7.1f}{e['crest_db']:6.1f}"
              f"{e['corr']:5.2f} | " + " ".join(f"{e['groups'][g]:6.1f}" for g in gs))
    print("bands (dB rel. total):")
    for e in rep["shots"]:
        print(f"  {e['shot']:<10} " + " ".join(f"{k}:{v:6.1f}" for k, v in e["bands_db"].items()))
    print("cuts momentary:", rep["cuts_momentary"])
    print("nature gain:", rep["nature_gain_db"])
    print("melody:", rep["melody_accuracy"])
    print(f"chord share min {rep['chord_share_min']} mean {rep['chord_share_mean']}")
    bad = [c for c in rep["chords"] if c["share"] < 0.75]
    if bad:
        print("  weak chords:", bad)


def main():
    global VSCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "audio", "candidates", "B_hybrid.wav"))
    ap.add_argument("--nature", default=os.path.join(HERE, "..", "out", "audio", "nature", "stems", "nature.wav"))
    ap.add_argument("--work", default=os.path.join(HERE, "..", "out", "audio", "work_b"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--quick", action="store_true", help="half the ensemble voices (faster preview)")
    ap.add_argument("--stems", action="store_true")
    a = ap.parse_args()
    if a.quick:
        VSCALE = 0.5
    B = arrange(a.seed)
    music, stems = mix_music(B, a.seed)
    del B
    nat, sr = an.read_any_wav(a.nature)
    assert sr == SR, sr
    nat = np.asarray(nat, np.float64)[:, :N]
    if nat.shape[1] < N:
        nat = np.pad(nat, ((0, 0), (0, N - nat.shape[1])))
    nat_g, pts = nature_mix(music, nat)
    mix = filt(bw("highpass", 18, 2), music + nat_g)  # DC / infrasonic guard
    mix = final_fade(mix)
    y, minfo = master(mix)
    y[:, int(59.9 * SR):] = 0.0
    gain = float(np.sqrt(np.sum(y ** 2) / (np.sum(mix ** 2) + 1e-20)))
    log(f"master: {minfo}")
    st.write_wav(a.out, y, seed=a.seed)
    log(f"wrote {a.out}")
    os.makedirs(a.work, exist_ok=True)
    mus_m = final_fade(music) * gain
    nat_m = final_fade(nat_g) * gain
    st.write_wav(os.path.join(a.work, "music_only.wav"), np.clip(mus_m, -0.99, 0.99), seed=a.seed)
    if a.stems:
        sd = os.path.join(a.work, "stems")
        os.makedirs(sd, exist_ok=True)
        for g, s in stems.items():
            st.write_wav(os.path.join(sd, f"{g}.wav"), (s * gain).astype(np.float32), float32=True)
    if a.analyze:
        rep = analyze(os.path.join(a.work, "analysis"), y, mus_m, nat_m, {g: s * gain for g, s in stems.items()},
                      pts, minfo)
        print_report(rep)
    log("done")


if __name__ == "__main__":
    main()
