"""music.py - epic cinematic score for the alpine-lake film, fully synthesised.

Every sound is generated here with numpy/scipy: band-limited (polyBLEP) saw ensembles through
body / formant filters, modal piano and celesta, membrane-mode drums, filtered noise, and a
generated hall impulse response. No samples, recordings, soundfonts or downloads.

The score is locked to the shot list (2 bars per shot, so every cut is a downbeat):
  forest   0-8     D minor, felt piano states the motif over soft low strings (D pedal)
  reveal   8-15.5  HIT (braam, taikos, sub boom, crash); horns restate the motif, choir enters
  skim     15.5-22 spiccato ostinato (3+3+2) and low taikos start
  waterfall 22-28.5 F | C, ostinato + drums grow (the falls' roar comes up in the mix)
  stream   28.5-35 Dm | Bb C, choir returns, toms + timpani roll + riser into the lift
  meadow   35-41.5 lift to F major: violins (+celesta, piano) sing the motif in major, lighter drums
  flyover  41.5-48 HIT: full orchestra, horns + trombones motif, choir forte, taiko ensemble
  peaks    48-54   climax in F: soaring motif to Bb, choir "ah"->"oh", fill into the finale
  finale   54-60   warm F(add9) resolution, piano/celesta echo of the motif, fade to silence

Motif (beats): A D E F~ | E D C D  (D minor), C F G A~ | G F E F (F major lift),
climax C F G A | Bb~ A G | F.

Usage (from the repository root):
  python audio/music.py --shots audio/film_shots.json
        [--nature out/audio/nature/stems/nature.wav] [--out out/audio/soundtrack.wav]
        [--music-out out/audio/work/music_only.wav] [--analyze out/audio/work/analysis] [--seed 11]
The nature stem comes from: audio/soundtrack.py --shots audio/film_shots.json
        --out out/audio/nature/bed.wav --stems
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
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analysis as an  # noqa: E402
import soundtrack as st  # noqa: E402  (make_ir, convolve_stereo, bus_compressor, limiter, write_wav)

SR = 48000
TARGET_LUFS = -14.0
CEILING_DBTP = -1.0
# nature level relative to the music, per shot (dB, short-term): ~10-14 dB under the score,
# but the waterfall and stream roar come up in their own shots
NATURE_REL = {"forest": -10.0, "reveal": -13.0, "skim": -12.0, "waterfall": -4.0, "stream": -6.5,
              "meadow": -12.0, "flyover": -14.0, "peaks": -15.0, "finale": -11.0}
MUSIC_FADE = (56.2, 59.9)   # music fade-out window (cos^2), relative to T=60 shifted by T-60
_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def rng_for(seed, name):
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


# =============================================================================
# basic DSP
# =============================================================================
def mhz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def dbg(v):
    return 10.0 ** (np.asarray(v, float) / 20.0)


def pan_gains(p):
    a = (float(np.clip(p, -1, 1)) + 1.0) * math.pi / 4.0
    return math.cos(a), math.sin(a)


def add_mono(bus, y, i0, pan=0.0, gain=1.0):
    gl, gr = pan_gains(pan)
    n = bus.shape[-1]
    i1 = i0 + len(y)
    if i0 < 0:
        y, i0 = y[-i0:], 0
    if i1 > n:
        y, i1 = y[:n - i0], n
    if i1 <= i0:
        return
    bus[0, i0:i1] += gl * gain * y
    bus[1, i0:i1] += gr * gain * y


def add_stereo(bus, y, i0, gain=1.0):
    n = bus.shape[-1]
    i1 = i0 + y.shape[-1]
    if i0 < 0:
        y, i0 = y[:, -i0:], 0
    if i1 > n:
        y, i1 = y[:, :n - i0], n
    if i1 > i0:
        bus[:, i0:i1] += gain * y


def saw(ph, dt):
    """polyBLEP band-limited sawtooth from phase (0..1) and phase increment arrays."""
    y = 2.0 * ph - 1.0
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] -= t + t - t * t - 1.0
    m = ph > 1.0 - dt
    t = (ph[m] - 1.0) / dt[m]
    y[m] -= t * t + t + t + 1.0
    return y


def osc_saw(f, rng):
    dt = np.asarray(f, float) / SR
    ph = (rng.random() + np.cumsum(dt)) % 1.0
    return saw(ph, dt)


def slow_noise(rng, L, rate_hz=2.0):
    """Smooth unit-SD random curve with bandwidth ~rate_hz (computed at 200 Hz control rate)."""
    hop = 240
    m = L // hop + 8
    v = gaussian_filter1d(rng.standard_normal(m), max(0.5, SR / hop / (2 * np.pi * rate_hz)), mode="wrap")
    v /= v.std() + 1e-12
    return np.interp(np.arange(L) / hop, np.arange(m), v)


def sos(kind, f, order=2):
    return signal.butter(order, f, kind, fs=SR, output="sos")


def filt(s, x):
    return signal.sosfilt(s, x, axis=-1)


def peq_sos(fc, gain_db, q):
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return np.hstack([b / a[0], a / a[0]])[None, :]


def shelf_sos(fc, gain_db, kind="low"):
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / SR
    cw, sw = np.cos(w), np.sin(w)
    al = sw / 2 * np.sqrt(2)
    sA = 2 * np.sqrt(A) * al
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * cw + sA), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sA)]
        a = [(A + 1) + (A - 1) * cw + sA, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sA]
    else:
        b = [A * ((A + 1) + (A - 1) * cw + sA), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sA)]
        a = [(A + 1) - (A - 1) * cw + sA, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sA]
    b, a = np.array(b), np.array(a)
    return np.hstack([b / a[0], a / a[0]])[None, :]


def bpf_ba(fc, bw):
    """RBJ band-pass, constant 0 dB peak gain."""
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / (2 * fc / bw)
    a0 = 1 + al
    return np.array([al, 0.0, -al]) / a0, np.array([1.0, -2 * np.cos(w) / a0, (1 - al) / a0])


def tv_filter(x, fc, kind="lp", q=0.707, block=64, stages=1):
    """Time-varying RBJ biquad (low-pass or band-pass), coefficients updated every `block`
    samples with the filter state carried over (fc changes slowly, so no zipper)."""
    y = np.asarray(x, float)
    fc = np.clip(fc, 20.0, 0.45 * SR)
    for _ in range(stages):
        out = np.empty_like(y)
        zi = np.zeros(2)
        for s in range(0, len(y), block):
            f = fc[min(s + block // 2, len(fc) - 1)]
            w = 2 * np.pi * f / SR
            cw, al = np.cos(w), np.sin(w) / (2 * q)
            a0 = 1 + al
            if kind == "lp":
                b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2]) / a0
            else:
                b = np.array([al, 0.0, -al]) / a0
            a = np.array([1.0, -2 * cw / a0, (1 - al) / a0])
            out[s:s + block], zi = signal.lfilter(b, a, y[s:s + block], zi=zi)
        y = out
    return y


def env_ar(L, att, t_off, rel, t0=0.0):
    """Equal-power attack / release envelope over L samples (times in s from the buffer start)."""
    t = np.arange(L) / SR - t0
    a = np.sin(0.5 * np.pi * np.clip(t / max(att, 1e-3), 0, 1))
    r = np.cos(0.5 * np.pi * np.clip((t - t_off) / max(rel, 1e-3), 0, 1))
    return a * r


def soft_clip(x, ceil_rel=1.25, knee=0.6):
    """Drum-bus peak shaving: unity below knee*c, smooth tanh shoulder up to c = ceil_rel * the
    99.9th percentile of the active samples, so only the first cycles of the loudest (overlapping)
    hits are rounded off and the master limiter has little left to do."""
    a = np.abs(x)
    act = a[a > 1e-4 * (a.max() + 1e-12)]
    c = ceil_rel * float(np.percentile(act, 99.9) if act.size else a.max() + 1e-12)
    k = knee * c
    a = np.abs(x)
    over = a > k
    y = x.copy()
    y[over] = np.sign(x[over]) * (k + (c - k) * np.tanh((a[over] - k) / (c - k)))
    return y


def tail_fade(y, frac=0.3):
    """cos^2 fade over the last `frac` of a one-shot buffer, so decays end in silence."""
    k = max(2, int(y.shape[-1] * frac))
    y[..., -k:] *= np.cos(0.5 * np.pi * np.linspace(0, 1, k)) ** 2
    return y


def edge_fade(y, ms=4.0):
    k = min(len(y) // 2, int(ms * 1e-3 * SR))
    if k > 1:
        w = np.sin(0.5 * np.pi * np.linspace(0, 1, k)) ** 2
        y[..., :k] *= w
        y[..., -k:] *= w[::-1]
    return y


def auto_curve(points, n, smooth_ms=40.0):
    """Gain automation (linear) from (time s, dB) breakpoints; interpolated in dB and smoothed."""
    hop = 48
    m = n // hop + 2
    tc = np.arange(m) * hop / SR
    ts = np.array([p[0] for p in points], float)
    vs = np.array([p[1] for p in points], float)
    d = np.interp(tc, ts, vs)
    if smooth_ms > 0:
        d = gaussian_filter1d(d, smooth_ms * 1e-3 * SR / hop, mode="nearest")
    return dbg(np.interp(np.arange(n) / hop, np.arange(m), d)), (tc, d)


def body_ir(rng, fmin, fmax, nmodes, length=0.06, qr=(6, 25), body=0.8):
    """Short resonant 'instrument body' IR: direct impulse + a random set of damped modes
    (log-spaced, roughly flat average gain), which gives the irregular formant structure of a
    wooden body instead of a synthetic smooth spectrum."""
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = np.zeros(n)
    for f in np.exp(rng.uniform(np.log(fmin), np.log(fmax), nmodes)):
        Q = rng.uniform(*qr)
        tau = Q / (np.pi * f)
        ir += rng.standard_normal() * (np.pi * f / Q) * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.283))
    ir *= np.cos(0.5 * np.pi * np.clip((t - 0.7 * length) / (0.3 * length), 0, 1))
    ir *= body / (np.sqrt(np.sum(ir ** 2)) + 1e-12)
    ir[0] += 1.0
    return ir / np.sqrt(np.sum(ir ** 2))


def conv_body(x, irs):
    n = x.shape[-1]
    return np.vstack([signal.oaconvolve(x[c], irs[c])[:n] for c in range(2)])


# =============================================================================
# timeline: 2 bars per shot, every cut is a downbeat
# =============================================================================
class Score:
    def __init__(self, shots):
        self.shots = sorted(shots, key=lambda s: s["start"])
        if len(self.shots) != 9:
            print(f"WARNING: the arrangement is written for 9 shots, got {len(self.shots)}")
        self.T = float(max(s["end"] for s in self.shots))
        b = []
        for s in self.shots:
            d = s["end"] - s["start"]
            b += [s["start"], s["start"] + d / 2]
        b.append(self.T)
        self.bars = np.array(b, float)
        self.S = [s["start"] for s in self.shots] + [self.T]
        self.n = int(round(self.T * SR))

    def t(self, beat):
        beats = 4.0 * np.arange(len(self.bars))
        B = float(beat)
        if B <= beats[-1]:
            return float(np.interp(B, beats, self.bars))
        return float(self.bars[-1] + (B - beats[-1]) * (self.bars[-1] - self.bars[-2]) / 4.0)

    def beat_len(self, beat):
        return self.t(beat + 0.5) - self.t(beat - 0.5) if beat >= 0.5 else self.t(1.0) - self.t(0.0)


# chord voicings (MIDI): contrabass, celli, violas, violins, choir men / women, low brass
CH = {
    "Dm": dict(cb=[38], vc=[50, 57], va=[62, 65], vn=[69, 74, 77], chm=[50, 57], chf=[62, 65, 69], br=[38, 45, 50], root=50, third=3),
    "C": dict(cb=[36], vc=[48, 55], va=[60, 64], vn=[67, 72, 76], chm=[48, 55], chf=[60, 64, 67], br=[36, 43, 48], root=48, third=4),
    "Bb": dict(cb=[34], vc=[46, 53], va=[62, 65], vn=[70, 74, 77], chm=[46, 53], chf=[62, 65, 70], br=[34, 41, 46], root=46, third=4),
    "F": dict(cb=[41], vc=[48, 53], va=[60, 65], vn=[69, 72, 77], chm=[48, 53], chf=[60, 65, 69], br=[41, 48, 53], root=53, third=4),
}
# (beat0, beat1, chord): i - bVII - bVI in D minor, the lift to F major at the meadow,
# plagal Bb -> F into the climax, IV - V - I resolution into the finale
CHORD_PLAN = [
    (0, 4, "Dm"), (4, 6, "C"), (6, 8, "Bb"),            # forest (D pedal)
    (8, 12, "Dm"), (12, 14, "C"), (14, 16, "Bb"),       # reveal
    (16, 20, "Dm"), (20, 24, "Bb"),                     # skim
    (24, 28, "F"), (28, 32, "C"),                       # waterfall
    (32, 36, "Dm"), (36, 38, "Bb"), (38, 40, "C"),      # stream
    (40, 44, "F"), (44, 48, "C"),                       # meadow (lift)
    (48, 52, "Dm"), (52, 54, "C"), (54, 56, "Bb"),      # flyover
    (56, 60, "F"), (60, 62, "Bb"), (62, 64, "C"),       # peaks (climax)
    (64, 72, "F"),                                      # finale
]
# motif (beat offset, midi in the horn octave, duration in beats)
M_D = [(0, 57, 1.0), (1, 62, 1.0), (2, 64, 1.0), (3, 65, 1.5), (4.5, 64, 1.0), (5.5, 62, 0.5), (6, 60, 1.0), (7, 62, 1.0)]
M_F = [(0, 60, 1.0), (1, 65, 1.0), (2, 67, 1.0), (3, 69, 1.5), (4.5, 67, 1.0), (5.5, 65, 0.5), (6, 64, 1.0), (7, 65, 1.0)]
M_C = [(0, 60, 1.0), (1, 65, 1.0), (2, 67, 1.0), (3, 69, 1.0), (4, 70, 1.5), (5.5, 69, 0.5), (6, 67, 2.0)]
MOTIF_VEL = [0.80, 0.86, 0.90, 1.0, 0.92, 0.84, 0.86, 0.90]


def chord_at(beat):
    for b0, b1, c in CHORD_PLAN:
        if b0 <= beat < b1:
            return c
    return CHORD_PLAN[-1][2]


# =============================================================================
# instruments
# =============================================================================
STR_CFG = {
    "vn": dict(nv=8, pan=-0.45, spread=0.35, hp=180, dark=1900, bright=6000, vib=(8, 15), det=7.0, fmin=250, fmax=6000),
    "va": dict(nv=6, pan=-0.12, spread=0.30, hp=110, dark=1400, bright=4500, vib=(7, 13), det=6.0, fmin=200, fmax=5000),
    "vc": dict(nv=6, pan=0.30, spread=0.30, hp=50, dark=1000, bright=3500, vib=(6, 12), det=6.0, fmin=120, fmax=4000),
    "cb": dict(nv=4, pan=0.50, spread=0.20, hp=28, dark=500, bright=1600, vib=(4, 8), det=5.0, fmin=70, fmax=2500),
}


def string_voices(n, events, cfg, rng):
    """Legato ensemble: per note `nv` detuned polyBLEP saws with own vibrato (delayed), slow
    pitch drift, an onset scoop and humanised entry. events: (t0, t1, midi, amp, att, rel)."""
    out = np.zeros((2, n))
    envsum = np.zeros(n)
    nv = cfg["nv"]
    for (t0, t1, m, amp, att, rel) in events:
        f0 = float(mhz(m))
        for _ in range(nv):
            s0 = t0 + rng.uniform(0.0, 0.045) - 0.01
            i0 = int(round(max(0.0, s0) * SR))
            i1 = min(n, int(round((t1 + rel) * SR)) + 1)
            L = i1 - i0
            if L <= 16:
                continue
            t = np.arange(L) / SR + (i0 / SR - s0)
            vr = rng.uniform(4.7, 6.3)
            vd = rng.uniform(*cfg["vib"])
            vib = vd * np.sin(2 * np.pi * vr * t + rng.uniform(0, 6.283) + 0.15 * slow_noise(rng, L, 0.8))
            vib *= np.clip((t - 0.3) / 0.7, 0, 1)
            cents = rng.normal(0, cfg["det"]) + vib + 2.5 * slow_noise(rng, L, 0.6) - 14.0 * np.exp(-np.maximum(t, 0) / 0.08)
            y = osc_saw(f0 * 2.0 ** (cents / 1200.0), rng)
            a = att * rng.uniform(0.8, 1.25)
            env = np.sin(0.5 * np.pi * np.clip(t / a, 0, 1)) * np.cos(0.5 * np.pi * np.clip((t - (t1 - s0)) / rel, 0, 1))
            env *= 1.0 + 0.07 * slow_noise(rng, L, 1.2)
            env[-min(L, 96):] *= np.linspace(1, 0, min(L, 96))
            y *= env * amp / math.sqrt(nv)
            add_mono(out, y, i0, cfg["pan"] + rng.uniform(-cfg["spread"], cfg["spread"]))
            envsum[i0:i1] += env * amp / nv
    return out, envsum


def string_finish(x, envsum, cfg, gain, bright, rng):
    """Dynamic brightness (cross-fade of a dark and a bright low-pass), wooden body IR per
    channel, rosin / bow noise following the playing level, high-pass, level automation."""
    yd = filt(sos("lowpass", cfg["dark"], 4), x)
    yb = filt(sos("lowpass", cfg["bright"], 4), x)
    y = yd + bright * (yb - yd)
    y = conv_body(y, [body_ir(rng, cfg["fmin"], cfg["fmax"], 80, 0.06, (4, 12), 0.35) for _ in range(2)])
    nz = filt(sos("bandpass", [1800, 7500]), rng.standard_normal((2, x.shape[-1])))
    y += nz * envsum * (0.010 + 0.02 * bright)
    y = filt(sos("highpass", cfg["hp"]), y)
    return y * gain


def spiccato(n, notes, cfg, rng, pan=0.0, spread=0.25):
    """Short bowed notes: 3 detuned saws, pitch settling from slightly sharp, 3 ms attack,
    velocity-dependent decay and brightness, rosin 'bite' noise. notes: (t, midi, vel)."""
    out = np.zeros((2, n))
    L = int(0.42 * SR)
    tt = np.arange(L) / SR
    bp_bite = sos("bandpass", [1500, 6000])
    for (t, m, v) in notes:
        f0 = float(mhz(m))
        y = np.zeros(L)
        for _ in range(3):
            cents = rng.normal(0, 7) + 14.0 * np.exp(-tt / 0.015)
            y += osc_saw(np.full(L, f0) * 2.0 ** (cents / 1200.0), rng)
        dec = (0.055 + 0.06 * v) * rng.uniform(0.8, 1.2)          # no two strokes alike
        att = rng.uniform(0.005, 0.009)
        env = np.sin(0.5 * np.pi * np.clip(tt / att, 0, 1)) * (0.12 * np.exp(-tt / 0.02) + 0.88 * np.exp(-tt / dec))
        env *= np.cos(0.5 * np.pi * np.clip((tt - 0.30) / 0.12, 0, 1))
        y = y * env / 3.0
        y += filt(bp_bite, rng.standard_normal(L)) * np.exp(-tt / 0.007) * (0.12 + 0.18 * v) * np.clip(tt / 0.002, 0, 1)
        y = filt(sos("lowpass", 1500 + 5000 * v ** 1.5), y)
        y[-64:] *= np.linspace(1, 0, 64)
        add_mono(out, y * v, int(round(t * SR)), pan + rng.uniform(-spread, spread))
    return out


def brass_line(n, notes, rng, nuni=3, det=5.0, bright=1.0, att=0.10, rel=0.45, drive=2.0, vib=5.0,
               growl=0.0, fmax=9000.0, pan=0.0, width=0.25):
    """Monophonic brass line (horn / trombone / tuba) -> stereo. Notes that touch are slurred
    (portamento, no re-attack). Brightness follows the level (lip/bore non-linearity): a
    time-varying low-pass at f0*(1 + 7*bright*amp^1.5) and a tanh stage driven by the level."""
    T0 = max(0.0, notes[0][0] - 0.03)
    T1 = min(n / SR, max(b for _, b, _, _ in notes) + rel + 0.05)
    i0 = int(T0 * SR)
    L = int(T1 * SR) - i0
    t = (i0 + np.arange(L)) / SR
    midi = np.full(L, float(notes[0][2]))
    since = t - notes[0][0]
    amp = np.zeros(L)
    for k, (a, b, m, v) in enumerate(notes):
        sel = t >= a
        midi[sel] = m
        since[sel] = t[sel] - a
        slur_in = k > 0 and notes[k - 1][1] >= a - 0.02
        slur_out = k < len(notes) - 1 and notes[k + 1][0] <= b + 0.02
        at = 0.04 if slur_in else att * rng.uniform(0.85, 1.2)
        r = 0.09 if slur_out else rel
        e = np.sin(0.5 * np.pi * np.clip((t - a) / at, 0, 1)) * np.cos(0.5 * np.pi * np.clip((t - b) / r, 0, 1))
        if not slur_in:   # small 'blat' accent on a fresh attack
            e *= 1.0 + 0.18 * v * np.exp(-np.maximum(t - a - at, 0) / 0.12) * (t >= a)
        amp = np.maximum(amp, v * e)
    hop = 48
    mc = gaussian_filter1d(midi[::hop], 0.022 * SR / hop, mode="nearest")
    midi = np.interp(np.arange(L) / hop, np.arange(len(mc)), mc)
    vr = rng.uniform(4.8, 5.6)
    vamt = vib * np.clip((since - 0.35) / 0.5, 0, 1)
    common = vamt * np.sin(2 * np.pi * vr * t) + 3.0 * slow_noise(rng, L, 1.0)
    src = np.zeros(L)
    for _ in range(nuni):
        c = common + rng.normal(0, det) + 2.0 * slow_noise(rng, L, 0.7)
        src += osc_saw(mhz(midi + c / 100.0), rng)
    src /= nuni
    fc = np.clip(mhz(midi) * (1.0 + 7.0 * bright * amp ** 1.5), 120.0, fmax)
    y = tv_filter(src, fc, "lp", q=0.8, stages=2)
    bp_br = sos("bandpass", [700, 3200])
    for k, (a, b, m, v) in enumerate(notes):           # breath / lip noise on fresh attacks
        if k > 0 and notes[k - 1][1] >= a - 0.02:
            continue
        j0 = int((a - T0) * SR)
        if 0 <= j0 < L:
            bl = min(L - j0, int(0.25 * SR))
            tt = np.arange(bl) / SR
            y[j0:j0 + bl] += filt(bp_br, rng.standard_normal(bl)) * np.exp(-tt / 0.035) * np.clip(tt / 0.004, 0, 1) * 0.35 * v
    if growl > 0:
        y *= 1.0 + growl * amp ** 2 * (0.6 * np.sin(2 * np.pi * 31.0 * t) + 0.4 * slow_noise(rng, L, 25.0))
    y = np.tanh(drive * (0.35 + amp) * y * 1.6) / np.tanh(drive)
    y = filt(sos("lowpass", fmax), y) * amp
    y = edge_fade(y, 3.0)
    out = np.zeros((2, L))
    gl, gr = pan_gains(pan - width)
    out[0] += gl * y * 0.5
    out[1] += gr * y * 0.5
    d = int(0.011 * SR)          # second desk slightly later / other side: width
    y2 = np.concatenate([np.zeros(d), y[:-d]])
    gl, gr = pan_gains(pan + width)
    out[0] += gl * y2 * 0.5
    out[1] += gr * y2 * 0.5
    return out, i0


def braam(rng, midis, L=5.0, vel=1.0):
    """Trailer 'braam': detuned low brass/saw cluster, filter that snaps open and slowly
    closes, heavy level-driven saturation and a 33 Hz growl."""
    n = int(L * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for ch in range(2):
        src = np.zeros(n)
        for m in midis:
            for _ in range(4):
                c = rng.normal(0, 9) + 4.0 * slow_noise(rng, n, 0.5) - 25.0 * np.clip((t - (L - 2.0)) / 2.0, 0, 1)
                src += osc_saw(mhz(m + c / 100.0), rng)
        src /= 4 * len(midis)
        env = np.sin(0.5 * np.pi * np.clip(t / 0.045, 0, 1)) * (0.5 + 0.5 * np.exp(-t / 0.7))
        env *= np.cos(0.5 * np.pi * np.clip((t - (L - 2.4)) / 2.4, 0, 1))
        fc = 150 + 2300 * np.sin(0.5 * np.pi * np.clip(t / 0.13, 0, 1)) ** 2 * np.exp(-np.maximum(t - 0.13, 0) / 0.85)
        y = tv_filter(src, fc, "lp", q=1.3, stages=2)
        y *= 1.0 + 0.35 * np.sin(2 * np.pi * 33.0 * t + 0.6 * np.sin(2 * np.pi * 3.1 * t)) * np.exp(-t / 1.1)
        y = np.tanh(3.2 * y * env) / np.tanh(3.2)
        out[ch] = filt(sos("highpass", 30), y)
    return edge_fade(out, 3.0) * vel


FORMANTS = {   # (Hz, bandwidth Hz, dB) - female (alto/soprano) and male (tenor/bass) blends
    ("f", "a"): [(800, 80, 0), (1150, 90, -5), (2900, 120, -24), (3900, 130, -24), (4950, 140, -45)],
    ("f", "o"): [(450, 70, 0), (800, 80, -9), (2830, 100, -22), (3800, 130, -26), (4950, 135, -50)],
    ("m", "a"): [(630, 80, 0), (1060, 90, -6), (2450, 120, -14), (2800, 130, -16), (3250, 140, -26)],
    ("m", "o"): [(400, 50, 0), (780, 80, -10), (2500, 100, -18), (2800, 120, -20), (3000, 120, -30)],
}


def formant_bank(x, part, vowel):
    y = np.zeros_like(x)
    for fc, bw, g in FORMANTS[(part, vowel)]:
        b, a = bpf_ba(fc, bw)
        y += signal.lfilter(b, a, x, axis=-1) * dbg(g)
    return y


def choir_part(n, events, part, rng, nv=5):
    """Ensemble of singers: per note `nv` saw voices with singer vibrato (5-6 Hz, 12-22 cents),
    jitter, onset scoop, slow attack; then a vowel formant bank (ah / oh morph) per part.
    events: (t0, t1, midis, amp, att, rel). Returns the raw source and the envelope sum."""
    src = np.zeros((2, n))
    envsum = np.zeros(n)
    for (t0, t1, midis, amp, att, rel) in events:
        for m in midis:
            f0 = float(mhz(m))
            for _ in range(nv):
                s0 = t0 + rng.uniform(-0.03, 0.09)
                i0 = int(round(max(0.0, s0) * SR))
                i1 = min(n, int(round((t1 + rel) * SR)) + 1)
                L = i1 - i0
                if L <= 16:
                    continue
                t = np.arange(L) / SR + (i0 / SR - s0)
                vd = rng.uniform(12, 22)
                vib = vd * np.sin(2 * np.pi * rng.uniform(5.0, 6.0) * t + rng.uniform(0, 6.283)) * np.clip((t - 0.4) / 0.8, 0, 1)
                cents = rng.normal(0, 7) + vib + 4.0 * slow_noise(rng, L, 0.5) + 3.0 * slow_noise(rng, L, 12.0) \
                    - 30.0 * np.exp(-np.maximum(t, 0) / 0.12)
                y = osc_saw(f0 * 2.0 ** (cents / 1200.0), rng)
                a = att * rng.uniform(0.8, 1.3)
                env = np.sin(0.5 * np.pi * np.clip(t / a, 0, 1)) * np.cos(0.5 * np.pi * np.clip((t - (t1 - s0)) / rel, 0, 1))
                env *= 1.0 + 0.1 * slow_noise(rng, L, 2.0)
                env[-min(L, 96):] *= np.linspace(1, 0, min(L, 96))
                add_mono(src, y * env * amp / math.sqrt(nv), i0, rng.uniform(-0.75, 0.75))
                envsum[i0:i1] += env * amp / nv
    return src, envsum


def choir_finish(src, envsum, part, vowel_curve, rng):
    src = filt(sos("lowpass", 5000), src)
    ya = formant_bank(src, part, "a")
    yo = formant_bank(src, part, "o")
    y = ya + vowel_curve * (yo - ya)
    nz = rng.standard_normal(src.shape) * envsum * 0.05
    na, no = formant_bank(nz, part, "a"), formant_bank(nz, part, "o")
    y += na + vowel_curve * (no - na)
    y += filt(sos("bandpass", [3500, 9000]), rng.standard_normal(src.shape)) * envsum * 0.004   # breath air
    return filt(sos("highpass", 110 if part == "f" else 70), y)


def piano_note(rng, midi, vel, dur, rel=0.25, hammer_pos=0.12):
    """Modal piano: stiff-string partials f_k = k f0 sqrt(1 + B k^2), 2 detuned strings per note
    (beating), two-stage decay, high partials decay faster, velocity-dependent hammer spectrum,
    hammer-position comb, felt-hammer noise and key thump, damper release."""
    f0 = float(mhz(midi))
    natural = 6.0 * 2.0 ** (-(midi - 60) / 18.0)
    Ls = min(dur + rel + 0.05, natural * 2.2 + 0.3, 9.0)
    L = int(Ls * SR)
    t = np.arange(L) / SR
    B = 0.00012 * 2.0 ** ((midi - 48) / 14.0)
    fh = 700 + 3800 * vel ** 1.6
    y = np.zeros(L)
    nstr = 2 if midi > 40 else 1
    for k in range(1, 64):
        fk = k * f0 * math.sqrt(1 + B * k * k)
        if fk > 15000:
            break
        a = (1.0 / k ** 0.7) * math.exp(-(fk / fh) ** 1.3) * (abs(math.sin(math.pi * k * hammer_pos)) + 0.06)
        if a < 1e-4:
            continue
        tau = natural / (1.0 + fk / 900.0)
        e = 0.72 * np.exp(-t / (0.22 * tau)) + 0.28 * np.exp(-t / tau)
        for s in range(nstr):
            d = 1.0 + (rng.normal(0, 0.9) + (0.8 if s else 0.0)) / 1731.0
            y += a / nstr * e * np.sin(2 * np.pi * fk * d * t + rng.uniform(0, 6.283))
    nz = rng.standard_normal(L)
    y += filt(sos("lowpass", min(fh * 1.5, 9000)), nz) * np.exp(-t / 0.004) * 0.06 * vel
    y += np.sin(2 * np.pi * 70.0 * t) * np.exp(-t / 0.025) * 0.04 * vel
    y *= np.cos(0.5 * np.pi * np.clip((t - dur) / rel, 0, 1)) * np.clip(t / 0.0012, 0, 1)
    y[-64:] *= np.linspace(1, 0, 64)
    return y * vel


def celesta_note(rng, midi, vel, Ls=2.6):
    f0 = float(mhz(midi))
    L = int(max(Ls, 5.0) * SR)
    t = np.arange(L) / SR
    y = np.sin(2 * np.pi * f0 * t) * np.exp(-t / 1.0)
    y += 0.22 * np.sin(2 * np.pi * 4.0 * f0 * t + 1.0) * np.exp(-t / 0.22)
    y += 0.12 * np.sin(2 * np.pi * 2.76 * f0 * t + 2.0) * np.exp(-t / 0.07)
    y += 0.05 * np.sin(2 * np.pi * 5.40 * f0 * t) * np.exp(-t / 0.03)
    y += filt(sos("bandpass", [2500, 9000]), rng.standard_normal(L)) * np.exp(-t / 0.003) * 0.08
    y *= np.clip(t / 0.001, 0, 1)
    return tail_fade(y, 0.35) * vel


def taiko(rng, f_end=55.0, vel=1.0, size=1.0):
    """Big drum: pitch-dropping fundamental + inharmonic membrane modes, skin slap, low thud,
    soft saturation. size scales the decays."""
    L = int((1.3 * size + 0.3) * SR)
    t = np.arange(L) / SR
    f = f_end * (1.0 + 1.3 * np.exp(-t / 0.016))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t / (0.24 * size))
    for r, a, tau in ((1.59, 0.42, 0.13), (2.14, 0.28, 0.09), (2.65, 0.18, 0.07), (3.16, 0.12, 0.05), (3.65, 0.08, 0.04)):
        y += a * np.sin(r * ph + rng.uniform(0, 6.283)) * np.exp(-t / (tau * size))
    nz = rng.standard_normal(L)
    y += filt(sos("bandpass", [250, 2000]), nz) * np.exp(-t / 0.011) * np.clip(t / 0.0015, 0, 1) * (0.35 + 0.5 * vel)
    y += filt(sos("lowpass", 320), nz) * np.exp(-t / 0.04) * 0.8
    y += filt(sos("bandpass", [140, 520]), nz) * np.exp(-t / 0.03) * 0.9          # skin punch
    y *= np.clip(t / 0.0007, 0, 1)
    y = np.tanh(2.0 * y) / np.tanh(2.0)
    return tail_fade(y, 0.3) * vel


def timpani(rng, midi, vel=1.0, decay=2.2):
    f0 = float(mhz(midi))
    L = int((decay * 4.5 + 0.2) * SR)
    t = np.arange(L) / SR
    ph = 2 * np.pi * np.cumsum(f0 * (1.0 + 0.018 * np.exp(-t / 0.05))) / SR
    y = np.zeros(L)
    for r, a, tf in ((1.0, 1.0, 1.0), (1.504, 0.55, 0.6), (1.742, 0.38, 0.45), (2.0, 0.3, 0.4), (2.245, 0.18, 0.3),
                     (2.494, 0.12, 0.25), (0.62, 0.5, 0.06)):
        y += a * np.sin(r * ph + rng.uniform(0, 6.283)) * np.exp(-t / (decay * tf))
    y += filt(sos("lowpass", 1400), rng.standard_normal(L)) * np.exp(-t / 0.006) * 0.6 * vel
    y *= np.clip(t / 0.0008, 0, 1)
    return tail_fade(y, 0.35) * vel * 0.6


def sub_boom(f=36.7, vel=1.0, decay=1.5, L=6.5):
    n = int(L * SR)
    t = np.arange(n) / SR
    fi = f * (1.0 + 1.6 * np.exp(-t / 0.045))
    y = np.sin(2 * np.pi * np.cumsum(fi) / SR) * np.exp(-t / decay) * np.clip(t / 0.003, 0, 1)
    y = np.tanh(1.8 * y) / np.tanh(1.8)
    y = filt(sos("lowpass", 180), y)
    return tail_fade(y, 0.4) * vel


def crash(rng, vel=1.0, decay=1.8, L=7.5):
    n = int(L * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for c in range(2):
        nz = rng.standard_normal(n)
        y = filt(sos("highpass", 2800, 4), nz) * (0.6 * np.exp(-t / decay) + 0.4 * np.exp(-t / 0.08))
        y += filt(sos("bandpass", [5500, 11000]), rng.standard_normal(n)) * np.exp(-t / (decay * 1.3)) * \
            (1 + 0.3 * slow_noise(rng, n, 3.0))
        y = filt(peq_sos(4200, 4.0, 1.2), y)
        y *= np.clip(t / 0.001, 0, 1)
        out[c] = y
    return tail_fade(edge_fade(out, 5), 0.35) * vel * 0.5


def reverse_cymbal(rng, L=2.4, vel=1.0):
    y = crash(rng, 1.0, decay=1.2, L=L + 0.2)[:, :int(L * SR)][:, ::-1].copy()
    k = int(0.012 * SR)
    y[:, -k:] *= np.linspace(1, 0, k) ** 2           # ends on the cut
    y[:, :k * 4] *= np.linspace(0, 1, k * 4)
    return y * vel


def riser(rng, L=3.0, vel=1.0, midi=50):
    """Noise sweep (band-pass 300 Hz -> 7 kHz) + a rising saw stack, accelerating into the cut."""
    n = int(L * SR)
    t = np.arange(n) / SR
    x = t / L
    out = np.zeros((2, n))
    for c in range(2):
        fc = 300.0 * (7000.0 / 300.0) ** (x ** 1.6)
        y = tv_filter(rng.standard_normal(n), fc, "bp", q=2.0, stages=1) * 1.3
        tone = sum(osc_saw(mhz(midi + 12.0 * x ** 1.8 + rng.normal(0, 0.08)), rng) for _ in range(3)) / 3.0
        tone = tv_filter(tone, 400 + 5000 * x ** 2, "lp", q=0.9)
        y = y + 0.35 * tone
        y *= x ** 2.2
        out[c] = y
    k = int(0.012 * SR)
    out[:, -k:] *= np.linspace(1, 0, k) ** 2
    return out * vel


# =============================================================================
# arrangement
# =============================================================================
def arrange(sc, seed):
    rng = rng_for(seed, "arrange")
    tb, S = sc.t, sc.S
    E = {k: [] for k in ("cb", "vc", "va", "vn", "vn_mel", "spicc_vc", "spicc_va", "spicc_cb", "piano", "celesta",
                         "horn", "trb", "lowbr", "chm", "chf", "taiko", "timp", "boom", "crash", "revcym", "riser",
                         "braam", "revswell")}
    hum = lambda s=0.008: rng.normal(0, s)  # noqa: E731

    # ---------------- sustained strings (legato pads), per chord
    for b0, b1, name in CHORD_PLAN:
        c = CH[name]
        t0, t1 = tb(b0), tb(b1)
        intro = b1 <= 8
        hit = b0 in (8, 48, 56, 64)
        att = 1.4 if b0 == 0 else (0.18 if hit else (0.6 if intro else 0.4))
        rel = 0.7 if b1 < 72 else 1.5
        t1x = t1 + (0.15 if b1 < 72 else 0.0)
        if not intro:
            E["cb"] += [(t0, t1x, m, 1.0, att, rel) for m in c["cb"]]
        for sec in ("vc", "va"):
            E[sec] += [(t0, t1x, m, 1.0, att, rel) for m in c[sec]]
        if b0 >= 8:
            vn = list(c["vn"])
            amp = 0.55 if 40 <= b0 < 48 else 1.0          # meadow: the violins sing the melody instead
            if b0 >= 64:
                vn = vn + [79]                             # finale F(add9)
            E["vn"] += [(t0, t1x, m, amp, att, rel) for m in vn]
    E["cb"].insert(0, (0.0, tb(8) + 0.1, 38, 1.0, 2.0, 0.5))       # D pedal under the intro

    # ---------------- violin melody (meadow lift, climax)
    def melody(dst, start_beat, motif, octave, vels, last_dur=None, amp=1.0):
        for k, (ob, m, d) in enumerate(motif):
            a = tb(start_beat + ob) + hum(0.006)
            dd = d if (last_dur is None or k < len(motif) - 1) else last_dur
            b = tb(start_beat + ob + dd)
            dst.append((a, b + 0.06, m + octave, amp * vels[k % len(vels)], 0.12, 0.35))

    melody(E["vn_mel"], 40, M_F, 12, MOTIF_VEL, amp=1.4)
    melody(E["vn_mel"], 56, M_C + [(8, 65, 4.0)], 12, MOTIF_VEL[:7] + [0.55], amp=1.5)

    # ---------------- spiccato ostinato (16ths, 3+3+2), skim .. peaks
    acc = {0, 3, 6, 8, 11, 14}
    for bar in range(4, 16):
        for i in range(16):
            B = 4 * bar + i / 4.0
            if (bar == 15 and i >= 8) or (bar in (11, 13) and i == 15):
                continue                                    # rest before the finale hit: drum fill only
            name = chord_at(B)
            c = CH[name]
            meadow = bar in (10, 11)
            if meadow and i % 2:
                continue
            r = c["root"]
            v = (0.95 if i in acc else 0.6) + rng.normal(0, 0.05)
            if meadow:
                v *= 0.75
            t = tb(B) + hum(0.006)
            E["spicc_vc"].append((t, r + [0, 12, 7][(i % 8) % 3] if (i % 8) < 6 else r + [0, 12][(i % 8) - 6], float(np.clip(v, 0.2, 1))))
            if bar >= 6 and not meadow:
                rv = r + 12
                cell = [c["third"], 7, 12]
                E["spicc_va"].append((t + hum(0.004), rv + cell[(i % 8) % 3] if (i % 8) < 6 else rv + [c["third"], 7][(i % 8) - 6],
                                      float(np.clip(v * 0.9, 0.2, 1))))
            if bar >= 12 and i in acc:
                E["spicc_cb"].append((t, CH[name]["cb"][0], float(np.clip(v, 0.2, 1))))

    # ---------------- piano: intro motif (felt, soft), meadow doubling, finale echo
    for ob, m, d in M_D:
        b = 0.5 + ob
        E["piano"].append((tb(b) + 0.35 + hum(0.012), m + 12, 0.36 + 0.06 * rng.random() + (0.05 if m + 12 >= 76 else 0), tb(b + d) - tb(b) + 0.25, "rh"))
    for b, ms in ((0, (38, 50)), (2, (57,)), (4, (48, 55)), (6, (46, 53))):
        for m in ms:
            E["piano"].append((tb(b) + 0.35 + hum(0.01), m, (0.19 if m < 45 else 0.22) + 0.04 * rng.random(), tb(b + 2) - tb(b) + 0.6, "lh"))
    for ob, m, d in M_F:                                   # meadow: piano an octave over the violins, soft
        E["piano"].append((tb(40 + ob) + hum(0.008), m + 24, 0.26, tb(40 + ob + d) - tb(40 + ob) + 0.15, "mel"))
    f0 = S[8]
    for dt, m, v in ((0.0, 41, 0.38), (0.0, 53, 0.34), (0.55, 72, 0.50), (1.35, 77, 0.48), (2.15, 79, 0.46), (2.95, 81, 0.50)):
        E["piano"].append((f0 + dt + hum(0.01), m, v, 6.0 - dt, "fin" if m >= 72 else "lh"))
    for ob, m, d in M_F:
        E["celesta"].append((tb(40 + ob) + hum(0.006), m + 24, 0.5))
    for dt, m in ((0.55, 84), (1.35, 89), (2.15, 91), (2.95, 93)):
        E["celesta"].append((f0 + dt + hum(0.006), m, 0.5))

    # ---------------- horns / trombones: motif statements, low brass chord swells
    def line(start_beat, motif, octave, vel, extra=(), last_vel=1.0):
        out = []
        for k, (ob, m, d) in enumerate(list(motif) + list(extra)):
            a = tb(start_beat + ob) + hum(0.01)
            b = tb(start_beat + ob + d)
            out.append((a, b, m + octave, vel * (MOTIF_VEL[k % 8] if k < len(motif) else last_vel)))
        return out

    E["horn"].append(line(8, M_D, 0, 0.75))
    E["horn"].append(line(48, M_D, 0, 0.85))
    E["trb"].append(line(48, M_D, -12, 0.78))
    E["horn"].append(line(56, M_C, 0, 1.0, extra=[(8, 65, 2.5)], last_vel=0.5))
    E["trb"].append(line(56, M_C, -12, 0.9, extra=[(8, 53, 2.5)], last_vel=0.45))
    lowbr_v = {8: 0.75, 12: 0.55, 14: 0.55, 16: 0.35, 20: 0.35, 24: 0.4, 28: 0.4, 32: 0.5, 36: 0.6, 38: 0.7, 40: 0.4, 44: 0.42,
               48: 0.8, 52: 0.75, 54: 0.78, 56: 1.0, 60: 0.95, 62: 1.0, 64: 0.35}
    for b0, b1, name in CHORD_PLAN:
        if b0 not in lowbr_v:
            continue
        t0, t1 = tb(b0), tb(b1)
        if b0 == 64:
            t1 = S[8] + 3.8
        for m in CH[name]["br"]:
            E["lowbr"].append([(t0 + hum(0.012), t1 + 0.05, m, lowbr_v[b0])])

    # ---------------- choir
    def ch_events(b0, b1, amp, att, rel=1.2, hi=False):
        name = chord_at(b0)
        f = list(CH[name]["chf"]) + ([CH[name]["vn"][-1]] if hi else [])
        E["chf"].append((tb(b0), tb(b1) + 0.15, f, amp, att, rel))
        E["chm"].append((tb(b0), tb(b1) + 0.15, list(CH[name]["chm"]), amp, att, rel))

    for b0, b1, a in ((8, 12, 0.6), (12, 14, 0.65), (14, 16, 0.7)):
        ch_events(b0, b1, a, 1.2 if b0 == 8 else 0.5, rel=1.0)
    for b0, b1, a in ((32, 36, 0.55), (36, 38, 0.7), (38, 40, 0.85)):
        ch_events(b0, b1, a, 0.9 if b0 == 32 else 0.4, rel=0.6)
    for b0, b1, a in ((40, 44, 0.4), (44, 48, 0.42)):
        ch_events(b0, b1, a, 0.6, rel=0.6)
    for b0, b1, a in ((48, 52, 0.85), (52, 54, 0.88), (54, 56, 0.92), (56, 60, 1.1), (60, 62, 1.1), (62, 64, 1.12)):
        ch_events(b0, b1, a, 0.25 if b0 in (48, 56) else 0.3, rel=0.5, hi=True)
    E["chf"].append((S[8], S[8] + 4.6, [60, 65, 69, 72], 0.42, 0.35, 1.6))
    E["chm"].append((S[8], S[8] + 4.6, [41, 48, 53], 0.42, 0.35, 1.6))

    # ---------------- drums
    def taikos(bar, idx, f, vels, size, pan_sd=0.35, jitter=0.007):
        for i, v in zip(idx, vels):
            E["taiko"].append((tb(4 * bar + i / 4.0) + hum(jitter), f * rng.uniform(0.97, 1.03), float(np.clip(v + rng.normal(0, 0.04), 0.1, 1.1)),
                               size, float(np.clip(rng.normal(0, pan_sd), -0.8, 0.8))))

    LOW, MID, HI = 57.0, 95.0, 150.0
    # reveal: the hit, a heartbeat
    taikos(2, [0], LOW, [1.0], 1.4, 0.1)
    taikos(2, [0], MID, [0.8], 0.9, 0.5)
    taikos(3, [0, 10], LOW, [0.55, 0.5], 1.3)
    for bar in (4, 5):           # skim
        taikos(bar, [0, 6, 10], LOW, [0.85, 0.58, 0.62], 1.2)
    taikos(5, [4, 12], MID, [0.35, 0.4], 0.8)
    for bar in (6, 7):           # waterfall
        taikos(bar, [0, 3, 6, 10], LOW, [0.85, 0.5, 0.6, 0.6], 1.2)
        taikos(bar, [0, 3, 6, 8, 11, 14], MID, [0.5, 0.35, 0.4, 0.45, 0.35, 0.4], 0.8)
    taikos(8, [0, 3, 6, 10, 12], LOW, [0.85, 0.55, 0.65, 0.6, 0.65], 1.2)   # stream
    taikos(8, [0, 3, 6, 8, 11, 14], MID, [0.55, 0.4, 0.45, 0.5, 0.4, 0.45], 0.8)
    taikos(9, [0, 3, 6, 8, 10, 12, 14], LOW, [0.85, 0.6, 0.7, 0.7, 0.75, 0.8, 0.85], 1.2)
    taikos(9, list(range(8, 16)), MID, list(np.linspace(0.45, 0.95, 8)), 0.8)
    taikos(9, [12, 13, 14, 15], HI, [0.5, 0.6, 0.7, 0.85], 0.55)
    taikos(10, [0], LOW, [0.8], 1.4, 0.1)  # meadow: the lift downbeat, then breath
    taikos(11, [0, 8], LOW, [0.45, 0.5], 1.2)
    taikos(11, list(range(8, 15)), HI, list(np.linspace(0.3, 0.8, 7)), 0.55)     # 16th rest before the hit
    taikos(11, [12, 13, 14], MID, [0.5, 0.6, 0.72], 0.8)
    for bar in (12, 13, 14, 15):   # flyover + peaks: full ensemble
        last = bar == 15
        lo_idx = [0, 3, 6, 8, 11, 14] if bar < 14 else [0, 3, 6, 8, 10, 11, 14]
        if last:
            lo_idx = [0, 3, 6]
        taikos(bar, lo_idx, LOW, [1.0 if i == 0 else 0.78 for i in lo_idx], 1.3)
        mid_idx = [i for i in range(16) if not (last and i >= 8) and not (bar == 13 and i == 15)]
        mv = (0.85, 0.42) if bar >= 14 else (0.75, 0.37)
        taikos(bar, mid_idx, MID, [mv[0] if i in acc else mv[1] for i in mid_idx], 0.8, 0.45, 0.005)
    taikos(13, [12, 13, 14], HI, [0.58, 0.68, 0.8], 0.55)
    taikos(15, list(range(8, 15)), LOW, list(np.linspace(0.6, 1.0, 7)), 1.2)
    taikos(15, list(range(8, 15)), HI, list(np.linspace(0.45, 0.95, 7)), 0.55)
    E["taiko"].append((S[8] + hum(0.004), LOW, 0.55, 1.5, 0.0))         # finale arrival, soft
    # timpani hits and rolls (midi)
    for t, m, v in ((S[1], 38, 1.0), (S[1], 33, 0.7), (S[5], 41, 0.6), (S[6], 38, 1.0), (S[7], 41, 1.0), (S[7], 36, 0.7), (S[8], 41, 0.5)):
        E["timp"].append((t + hum(0.004), m, v, 2.4))
    for a, b, m, v0, v1 in ((tb(6), tb(8), 38, 0.05, 0.55), (tb(13.5), tb(16), 33, 0.08, 0.5), (tb(37), tb(40), 36, 0.1, 0.75),
                            (tb(46), tb(48), 33, 0.1, 0.8), (tb(54), tb(56), 36, 0.15, 0.85), (tb(62), tb(64), 36, 0.2, 0.95)):
        k, t = 0, a
        while t < b - 0.03:
            x = (t - a) / (b - a)
            E["timp"].append((t + hum(0.004), m, (v0 + (v1 - v0) * x ** 1.6) * rng.uniform(0.85, 1.1) * 0.55, 1.2))
            t += 1.0 / (13.0 + 3.0 * x) * rng.uniform(0.92, 1.08)
            k += 1
    # booms, crashes, reverse swells, risers, braams
    E["boom"] += [(S[1], 36.7, 1.2, 1.6), (S[2], 36.7, 0.35, 0.9), (S[5], 43.7, 0.55, 0.9), (S[6], 36.7, 1.4, 1.6), (S[7], 43.7, 1.4, 1.8),
                  (S[8], 43.7, 0.45, 1.4)]
    E["crash"] += [(S[1], 1.0), (S[5], 0.45), (S[6], 1.25), (S[7], 1.25), (S[8], 0.35)]
    E["revcym"] += [(S[1], 2.6, 0.7), (S[2], 2.0, 0.45), (S[5], 2.4, 0.6), (S[6], 2.6, 0.8), (S[7], 2.0, 0.8), (S[8], 1.8, 0.45)]
    E["riser"] += [(S[1], 3.0, 0.35, 50), (S[5], 3.0, 0.32, 48), (S[6], 4.0, 0.42, 50), (S[7], 3.0, 0.38, 48)]
    E["braam"] += [(S[1], (26, 38, 45), 1.2), (S[6], (26, 38, 45), 1.4), (S[7], (29, 41, 48), 1.4)]
    E["revswell"] += [(S[1], (50, 57, 62, 65, 69), 2.4, 0.5), (S[5], (53, 60, 65, 69, 72), 2.4, 0.45),
                      (S[6], (50, 57, 62, 65, 69), 2.2, 0.5), (S[7], (53, 60, 65, 69, 72), 2.0, 0.5)]
    return E


# "suck" before the big hits: the sustained layers dip just before these cuts (shot index: dB),
# while the fill, riser and reverse cymbal drive into the hit; restored on the cut
DUCKS = {5: -5.0, 6: -9.0, 7: -7.0}


def pre_hit_duck(sc, pre=0.55, hold=0.07):
    hop = 48
    m = sc.n // hop + 2
    tc = np.arange(m) * hop / SR
    d = np.zeros(m)
    for i, ddb in DUCKS.items():
        c = sc.S[i]
        x = np.clip((tc - (c - pre)) / (pre - hold), 0, 1)
        shape = np.sin(0.5 * np.pi * x) ** 2 * ddb
        shape[tc >= c] = 0.0
        d += shape
    d = gaussian_filter1d(d, 0.012 * SR / hop)
    return dbg(np.interp(np.arange(sc.n) / hop, np.arange(m), d))


# per-group automation (time s, dB), written against the cut list S
def automation(S, T):
    s = S
    return {
        "strings": [(0, -18), (3.0, -14), (s[1] - 1.2, -11), (s[1] - 0.2, -10), (s[1] + 0.3, -8), (s[1] + 2.0, -6), (s[2] - 0.3, -6),
                    (s[2] + 0.4, -10), (s[3], -11), (s[4], -10), (s[5] - 0.3, -5), (s[5] + 0.4, -6), (s[6] - 0.4, -4),
                    (s[6] + 0.4, -4), (s[7] - 0.3, -3.5), (s[7] + 0.6, 0), (s[8] - 0.2, 0), (s[8] + 0.8, -4), (T - 3, -7), (T, -10)],
        "spicc": [(s[2], 0), (s[3], -1.5), (s[4], -2), (s[5] - 0.2, -1), (s[5] + 0.2, -6), (s[6] - 1, -3), (s[6], -2.5), (s[7] - 0.3, -2.5), (s[7], 0), (s[8], 0)],
    }   # choir, brass, piano and drums are shaped by their note velocities / amplitudes in arrange()


# =============================================================================
# render
# =============================================================================
GROUP_DB = {   # static group faders (dB), calibrated from the per-shot stem table
    "strings": -5.0, "vn_mel": -4.5, "spicc": 17.0, "piano": 7.0, "celesta": -7.0, "brass": 8.0, "lowbr": -5.0,
    "braam": 0.0, "choir": 10.0, "taiko": 7.0, "timp": -5.0, "boom": -1.0, "crash": -14.0, "revcym": -14.0,
    "riser": -19.0, "revswell": -8.0,
}
SEND = {"strings": 0.40, "vn_mel": 0.45, "spicc": 0.30, "piano": 0.42, "celesta": 0.55, "brass": 0.35, "lowbr": 0.32, "braam": 0.18,
        "choir": 0.55, "taiko": 0.20, "timp": 0.28, "boom": 0.0, "crash": 0.30, "revcym": 0.25, "riser": 0.30, "revswell": 0.0}


def render_music(sc, seed=11):
    n, T, S = sc.n, sc.T, sc.S
    E = arrange(sc, seed)
    A = automation(S, T)
    duck = pre_hit_duck(sc)
    G = {}
    mel = {}

    log("strings")
    g_str, (tc, dstr) = auto_curve(A["strings"], n, 60)
    bright = np.interp(np.arange(n) / SR, tc, np.clip((dstr + 14.0) / 14.0, 0, 1) ** 1.2)
    acc = np.zeros((2, n))
    for sec in ("cb", "vc", "va", "vn"):
        r = rng_for(seed, "str_" + sec)
        x, es = string_voices(n, E[sec], STR_CFG[sec], r)
        acc += string_finish(x, es, STR_CFG[sec], g_str * duck, bright, r) * dbg({"cb": 0.0, "vc": -0.5, "va": -1.5, "vn": -2.0}[sec])
    G["strings"] = acc
    r = rng_for(seed, "vn_mel")
    x, es = string_voices(n, E["vn_mel"], dict(STR_CFG["vn"], nv=7, pan=-0.2, spread=0.4), r)
    mel["violins"] = x.mean(axis=0)
    G["vn_mel"] = string_finish(x, es, dict(STR_CFG["vn"], dark=3500, bright=7500), duck, np.full(n, 0.7), r)

    log("spiccato")
    g_sp, _ = auto_curve(A["spicc"], n, 30)
    r = rng_for(seed, "spicc")
    sp = spiccato(n, E["spicc_vc"], STR_CFG["vc"], r, pan=0.25) + 0.75 * spiccato(n, E["spicc_va"], STR_CFG["va"], r, pan=-0.25) \
        + 0.8 * spiccato(n, [(t, m, v) for t, m, v in E["spicc_cb"]], STR_CFG["cb"], r, pan=0.45, spread=0.1)
    sp = conv_body(sp, [body_ir(r, 140, 4500, 80, 0.06, (4, 12), 0.35) for _ in range(2)])
    G["spicc"] = filt(sos("highpass", 60), sp) * g_sp * duck

    log("piano / celesta")
    r = rng_for(seed, "piano")
    pb = np.zeros((2, n))
    mel["piano"] = np.zeros(n)
    for (t, m, v, d, kind) in E["piano"]:
        y = piano_note(r, m, v, d)
        pan = float(np.clip((m - 60) / 40.0, -0.4, 0.4))
        add_mono(pb, y, int(round(t * SR)), pan)
        if kind in ("rh", "fin"):
            i0 = int(round(t * SR))
            yy = y[:max(0, min(len(y), n - i0))]
            mel["piano"][i0:i0 + len(yy)] += yy
    G["piano"] = conv_body(pb, [body_ir(r, 80, 3500, 90, 0.12, (6, 25), 0.45) for _ in range(2)])
    cb_ = np.zeros((2, n))
    for (t, m, v) in E["celesta"]:
        add_mono(cb_, celesta_note(r, m, v), int(round(t * SR)), r.uniform(-0.3, 0.3))
    G["celesta"] = cb_

    log("brass")
    r = rng_for(seed, "brass")
    br = np.zeros((2, n))
    mel["horn"] = np.zeros(n)
    for ln in E["horn"]:
        y, i0 = brass_line(n, ln, r, nuni=4, bright=0.75, drive=1.6, vib=4.0, growl=0.05, fmax=6000, pan=-0.3, width=0.2)
        add_stereo(br, y, i0)
        mel["horn"][i0:i0 + y.shape[-1]] += y.mean(axis=0)
    for ln in E["trb"]:
        y, i0 = brass_line(n, ln, r, nuni=3, bright=1.0, drive=2.2, vib=2.0, growl=0.12, fmax=7000, pan=0.3, width=0.2)
        add_stereo(br, y * 0.8, i0)
    G["brass"] = filt(sos("highpass", 45), br)
    lb = np.zeros((2, n))
    for ln in E["lowbr"]:
        y, i0 = brass_line(n, ln, r, nuni=3, bright=0.55, att=0.35, rel=0.8, drive=1.8, vib=0.0, growl=0.08, fmax=4000,
                           pan=float(np.clip((ln[0][2] - 44) / 20.0, -0.4, 0.4)), width=0.25)
        add_stereo(lb, y, i0)
    G["lowbr"] = filt(sos("highpass", 32), lb) * duck
    bb = np.zeros((2, n))
    for (t, ms, v) in E["braam"]:
        add_stereo(bb, braam(r, ms, 5.0, v), int(round(t * SR)) - int(0.004 * SR))
    G["braam"] = bb

    log("choir")
    r = rng_for(seed, "choir")
    vowel = np.interp(np.arange(n) / SR, [0, S[5] - 0.5, S[5] + 0.5, S[6] - 0.3, S[6] + 0.3, S[7] + 2.0, S[7] + 4.5, T],
                      [0.15, 0.15, 0.8, 0.8, 0.1, 0.1, 0.85, 0.9])
    chb = np.zeros((2, n))
    for part, key, nv in (("f", "chf", 5), ("m", "chm", 5)):
        src, es = choir_part(n, E[key], part, r, nv)
        chb += choir_finish(src, es, part, vowel, r) * (1.0 if part == "f" else 0.8)
    G["choir"] = chb * duck

    log("drums / fx")
    r = rng_for(seed, "drums")
    tk = np.zeros((2, n))
    for (t, f, v, size, pan) in E["taiko"]:
        add_mono(tk, taiko(r, f, v, size), int(round(t * SR)), pan)
    G["taiko"] = tk
    tp = np.zeros((2, n))
    for (t, m, v, dec) in E["timp"]:
        add_mono(tp, timpani(r, m, v, dec), int(round(t * SR)), 0.15)
    G["timp"] = tp
    bm = np.zeros((2, n))
    for (t, f, v, dec) in E["boom"]:
        add_mono(bm, sub_boom(f, v, dec), int(round(t * SR)), 0.0)
    G["boom"] = bm
    cr = np.zeros((2, n))
    for (t, v) in E["crash"]:
        add_stereo(cr, crash(r, v), int(round(t * SR)))
    G["crash"] = cr
    rc = np.zeros((2, n))
    for (t, L, v) in E["revcym"]:
        y = reverse_cymbal(r, L, v)
        add_stereo(rc, y, int(round(t * SR)) - y.shape[-1])
    G["revcym"] = rc
    rs = np.zeros((2, n))
    for (t, L, v, m) in E["riser"]:
        y = riser(r, L, v, m)
        add_stereo(rs, y, int(round(t * SR)) - y.shape[-1])
    G["riser"] = filt(sos("highpass", 120), rs)

    log("reverb")
    hall = st.make_ir(rng_for(seed, "hall"), rt60=[2.2, 2.4, 2.6, 2.6, 2.4, 2.1, 1.6, 1.1, 0.6],
                      band_db=[-12, -6, -1, 0, 0, -1, -3, -7, -14], length=4.8, predelay=0.028, onset=0.06, er=12, er_span=0.07)
    # reverse swells: the next chord through the hall, reversed, ending on the cut
    sw = np.zeros((2, n))
    for (t, ms, L, v) in E["revswell"]:
        k = int(L * SR)
        dry = np.zeros(k)
        for m in ms:
            y = piano_note(r, m, 0.7, L, rel=0.3)[:k]
            dry[:len(y)] += y
        wet = np.vstack([signal.oaconvolve(dry, hall[c])[:k] for c in range(2)])
        y = wet[:, ::-1].copy()
        y *= np.linspace(0, 1, k) ** 1.5
        kk = int(0.015 * SR)
        y[:, -kk:] *= np.linspace(1, 0, kk) ** 2
        add_stereo(sw, y * v, int(round(t * SR)) - k)
    G["revswell"] = filt(sos("highpass", 150), sw)

    for k in G:
        G[k] = G[k] * dbg(GROUP_DB[k])
    G["taiko"] = soft_clip(G["taiko"])
    G["spicc"] = soft_clip(G["spicc"], 1.2, 0.6)
    send = sum(G[k] * SEND[k] for k in G if k not in ("taiko", "timp"))
    send = send + filt(sos("highpass", 160, 2), (G["taiko"] * SEND["taiko"] + G["timp"] * SEND["timp"]))
    wet = st.convolve_stereo(send, hall) * dbg(-1.0)
    G["reverb"] = filt(sos("highpass", 60), wet)
    for k in G:
        G[k] = edge_fade(G[k], 2.0)
    return G, E, mel


def music_bus(G, n, T):
    mus = sum(G.values())
    mus = filt(sos("highpass", 26, 4), mus)
    mus = filt(peq_sos(280, -1.5, 0.9), mus)         # de-mud
    mus = filt(shelf_sos(9000, 1.0, "high"), mus)     # air
    fs, fe = MUSIC_FADE[0] + T - 60.0, MUSIC_FADE[1] + T - 60.0
    t = np.arange(n) / SR
    fade = np.cos(0.5 * np.pi * np.clip((t - fs) / (fe - fs), 0, 1)) ** 2
    fin = np.clip(t / 0.02, 0, 1)
    return mus * fade * fin, fade * fin


def nature_gain(sc, nature, music, lead=0.45, xf=1.6):
    """Per-shot static nature gain so that nature sits NATURE_REL dB under the music (ungated
    K-weighted shot means; nature measured over its J-cut audio window), with raised-cosine
    transitions centred on the shifted cuts."""
    nk, mk = an.k_weight(nature), an.k_weight(music)
    T, S = sc.T, sc.S
    gd = []
    for i, s in enumerate(sc.shots):
        a, b = S[i], S[i + 1]
        lm = an.segment_lufs(mk, a + 0.3, b - 0.3)
        ln = an.segment_lufs(nk, max(0.6, a - lead + 0.6), min(T - 4.5 if i == len(sc.shots) - 1 else T, b - lead - 0.6))
        rel = NATURE_REL.get(s["name"], -12.0)
        gd.append(float(np.clip(lm + rel - ln, -30, 12)))
    t = np.arange(sc.n) / SR
    g = np.full(sc.n, gd[0])
    for i in range(1, len(gd)):
        c = S[i] - lead
        w = np.sin(0.5 * np.pi * np.clip((t - (c - xf / 2)) / xf, 0, 1)) ** 2
        g = g + w * (gd[i] - gd[i - 1])
    return dbg(g), gd


def master(mix, target=TARGET_LUFS, ceiling=CEILING_DBTP):
    info = {}
    x = filt(sos("highpass", 22, 2), mix)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x * dbg(target - an.integrated_lufs(x))
    x, info["comp_max_gr_db"] = st.bus_compressor(x, thr_db=-19.0, ratio=1.8, knee_db=8.0, att=0.03, rel=0.35)
    x = x * dbg(target - an.integrated_lufs(x))
    ceil = ceiling - 0.25
    tp = 0.0
    for _ in range(8):
        y, info["limiter_max_gr_db"] = st.limiter(x, ceil, look_ms=4.0, rel_ms=150.0)
        yl = an.integrated_lufs(y)
        if abs(yl - target) > 0.1:
            x = x * dbg(target - yl)
            y, info["limiter_max_gr_db"] = st.limiter(x, ceil, look_ms=4.0, rel_ms=150.0)
        tp = an.true_peak_db(y)
        if tp <= ceiling - 0.05:
            break
        ceil -= (tp - ceiling) + 0.08
    info["true_peak_db"] = tp
    info["ceiling_used_db"] = ceil
    info["limiter_gr_at_s"] = st.limiter.last_gr_time
    info["integrated_lufs"] = an.integrated_lufs(y)
    return y, info


# =============================================================================
# analysis (listen-proxy checks)
# =============================================================================
def pitch_track(x, hop=0.01, win=0.085, fmin=90.0, fmax=1400.0):
    """Harmonic-product-spectrum f0 per frame (3 harmonics), voiced where the frame is within
    35 dB of the loudest frame."""
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


def motif_png(path, tracks, shots, T):
    from PIL import Image, ImageDraw
    W, H, ml, mt = 1400, 520, 60, 30
    im = Image.new("RGB", (ml + W + 12, mt + H + 40), (18, 18, 22))
    d = ImageDraw.Draw(im)
    f = an._font(13)
    lo, hi = 52, 96
    tx = lambda t: ml + t / T * (W - 1)  # noqa: E731
    ty = lambda m: mt + H - (m - lo) / (hi - lo) * H  # noqa: E731
    for m in range(lo, hi + 1):
        if m % 12 in (2, 5, 9):   # D, F, A lines
            d.line([(ml, ty(m)), (ml + W, ty(m))], fill=(45, 45, 55))
            d.text((4, ty(m) - 7), ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][m % 12] + str(m // 12 - 1),
                   fill=(170, 170, 170), font=f)
    for s in shots:
        d.line([(tx(s["start"]), mt), (tx(s["start"]), mt + H)], fill=(110, 110, 110))
        d.text((tx(s["start"]) + 3, mt + 2), s["name"], fill=(220, 220, 220), font=f)
    cols = {"piano": (120, 200, 255), "horn": (255, 190, 80), "violins": (140, 255, 140)}
    for name, (t, p, ref) in tracks.items():
        for (a, b, m) in ref:
            d.rectangle([tx(a), ty(m) - 3, tx(b), ty(m) + 3], outline=(90, 90, 90))
        for tt, pp in zip(t, p):
            if np.isfinite(pp) and lo <= pp <= hi:
                d.point((tx(tt), ty(pp)), fill=cols[name])
    d.text((ml, 6), "motif pitch track (HPS f0 of the dry melody lines): piano blue, horns orange, violins green; boxes = written notes",
           fill=(235, 235, 235), font=f)
    for s in range(0, int(T) + 1, 5):
        d.text((tx(s) - 6, mt + H + 6), f"{s}s", fill=(200, 200, 200), font=f)
    im.save(path)


def motif_accuracy(t, p, ref):
    """Fraction of voiced frames inside a written note (excluding 60 ms after the onset) whose
    pitch is within 0.5 semitone of it (octave errors counted separately)."""
    ok = tot = octv = 0
    for (a, b, m) in ref:
        sel = (t > a + 0.06) & (t < b - 0.03) & np.isfinite(p)
        if not sel.any():
            continue
        e = p[sel] - m
        tot += sel.sum()
        ok += np.sum(np.abs(e) < 0.5)
        octv += np.sum(np.abs(np.abs(e) - 12) < 0.5)
    return (ok / tot if tot else float("nan")), (octv / tot if tot else float("nan")), int(tot)


def analyze(out_dir, sc, mix, music, nature, G, E, mel, info, gd):
    os.makedirs(out_dir, exist_ok=True)
    T, S, shots = sc.T, sc.S, sc.shots
    xk, mk, nk = an.k_weight(mix), an.k_weight(music), an.k_weight(nature)
    rep = dict(info)
    rep["integrated_lufs"] = an.integrated_lufs(xk=xk)
    rep["true_peak_dbtp"] = an.true_peak_db(mix)
    rep["sample_peak_dbfs"] = an.sample_peak_db(mix)
    rep["dc"] = [float(v) for v in mix.mean(axis=-1)]
    rep["clipped"] = int(np.sum(np.abs(mix) >= 0.9999))
    rep["frames"] = int(mix.shape[-1])
    rep["clicks_mix"] = an.click_scan(mix)
    rep["clicks_music"] = an.click_scan(music)
    rep["lr_corr"] = an.lr_correlation(mix)
    mono = np.vstack([mix.mean(axis=0)] * 2)
    rep["mono_foldown_loss_db"] = rep_lufs_st = float(an.integrated_lufs(mix) - an.integrated_lufs(mono))  # noqa: F841
    rep["last_100ms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(mix[:, -4800:] ** 2)) + 1e-12))
    ts, ls = an.short_term(xk)
    tm, lm = an.momentary(xk)
    rep["max_short_term"] = float(ls.max())
    rep["max_momentary"] = float(lm.max())
    rep["nature_gain_db"] = gd
    rows = []
    gk = {k: an.k_weight(g) for k, g in G.items()}
    for i, s in enumerate(shots):
        a, b = S[i] + 0.25, S[i + 1] - 0.25
        row = dict(shot=s["name"], mix=an.segment_lufs(xk, a, b), music=an.segment_lufs(mk, a, b), nature=an.segment_lufs(nk, a, b))
        row["nature_minus_music"] = row["nature"] - row["music"]
        row["st_max"] = float(ls[(ts >= S[i]) & (ts <= S[i + 1])].max()) if np.any((ts >= S[i]) & (ts <= S[i + 1])) else None
        for k, g in gk.items():
            row[k] = an.segment_lufs(g, a, b)
        rows.append(row)
    del gk
    rep["shots"] = rows
    # spectral balance per shot: octave-band power (dB re total) and slope 125 Hz - 8 kHz (dB/oct)
    fr, P = signal.welch(mix.mean(axis=0), SR, nperseg=8192)
    oct_c = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    for i, row in enumerate(rows):
        a, b = int((S[i] + 0.25) * SR), int((S[i + 1] - 0.25) * SR)
        fr, P = signal.welch(mix[:, a:b].mean(axis=0), SR, nperseg=8192)
        bp = np.array([P[(fr >= c / 1.414) & (fr < c * 1.414)].sum() for c in oct_c]) + 1e-20
        lv = 10 * np.log10(bp / bp.sum())
        row["oct_db"] = [round(float(v), 1) for v in lv]
        sel = slice(2, 9)
        row["slope_db_oct"] = float(np.polyfit(np.log2(oct_c[sel]), lv[sel], 1)[0])
    cuts = []
    for c in S[1:-1]:
        pre = float(np.max(lm[(tm > c - 0.8) & (tm < c - 0.05)]))
        post = float(np.max(lm[(tm > c) & (tm < c + 0.6)]))
        cuts.append(dict(cut=c, momentary_pre=pre, momentary_post=post, jump=post - pre))
    rep["cuts"] = cuts
    rep["short_term_curve"] = [(round(float(a), 1), round(float(b), 2)) for a, b in zip(ts[::5], ls[::5])]
    # motif pitch tracks
    tracks, acc = {}, {}
    refs = {
        "piano": [(t, t + d - 0.25, m) for (t, m, v, d, k) in E["piano"] if k in ("rh", "fin")],
        "horn": [(a, b, m) for ln in E["horn"] for (a, b, m, v) in ln],
        "violins": [(a, b - 0.06, m) for (a, b, m, amp, att, rel) in E["vn_mel"]],
    }
    for name, x in mel.items():
        t, p = pitch_track(x)
        ref = refs[name]
        if name == "piano":   # the ringing piano notes overlap: judge each note until the next one starts
            ref = [(a, min(b, ref[i + 1][0]) if i + 1 < len(ref) else b, m) for i, (a, b, m) in enumerate(ref)]
        tracks[name] = (t, p, ref)
        acc[name] = motif_accuracy(t, p, ref)
    rep["motif_pitch_accuracy"] = {k: dict(within_half_semitone=round(v[0], 3), octave_errors=round(v[1], 3), frames=v[2])
                                   for k, v in acc.items()}
    motif_png(os.path.join(out_dir, "motif_pitch.png"), tracks, shots, T)
    an.render_panel(mix, "score + nature mix (r5 cut): spectrogram and loudness vs the cut list", shots, T,
                    spec_h=320, level_h=150, db_lo=-120.0, db_hi=-25.0).save(os.path.join(out_dir, "spec_mix.png"))
    panels = [("mix", mix), ("music", music), ("nature (at mix gain)", nature)]
    grp = {"strings+melody": ("strings", "vn_mel", "spicc"), "brass+braam": ("brass", "lowbr", "braam"), "choir": ("choir",),
           "piano+celesta": ("piano", "celesta"), "drums": ("taiko", "timp", "boom", "crash"), "fx": ("revcym", "riser", "revswell"),
           "hall reverb": ("reverb",)}
    for name, ks in grp.items():
        panels.append((name, sum(G[k] for k in ks if k in G)))
    an.overview_png(os.path.join(out_dir, "spec_overview.png"), panels, shots, T, spec_h=110)
    for i, s in enumerate(shots):
        an.zoom_png(os.path.join(out_dir, f"spec_{i}_{s['name']}.png"), mix, max(0.0, S[i] - 0.6), min(T, S[i + 1] + 0.4),
                    f"mix: {s['name']}", spec_h=300, level_h=110, width=900, db_lo=-120.0, db_hi=-25.0)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=float)
    return rep


def print_report(rep):
    print(f"  integrated {rep['integrated_lufs']:.2f} LUFS | true peak {rep['true_peak_dbtp']:.2f} dBTP | sample peak "
          f"{rep['sample_peak_dbfs']:.2f} | max ST {rep['max_short_term']:.2f} | max M {rep['max_momentary']:.2f}")
    print(f"  DC {rep['dc'][0]:.2e} {rep['dc'][1]:.2e} | clipped {rep['clipped']} | frames {rep['frames']} | "
          f"last 100 ms {rep['last_100ms_dbfs']:.1f} dBFS | L/R corr {rep['lr_corr']:.3f} | mono fold-down loss {rep['mono_foldown_loss_db']:.2f} dB")
    c, cm = rep["clicks_mix"], rep["clicks_music"]
    print(f"  clicks mix {c['n_hits']} (max ratio {c['max_ratio']:.1f}) hf {c['hf_hits']} | clicks music {cm['n_hits']} "
          f"(max ratio {cm['max_ratio']:.1f}) hf {cm['hf_hits']} {cm['hf_hit_times'][:12]}")
    keys = ["strings", "vn_mel", "spicc", "piano", "celesta", "brass", "lowbr", "braam", "choir", "taiko", "timp", "boom", "reverb"]
    print("  shot        mix  music nature  n-m  stmax | " + " ".join(f"{k[:6]:>6}" for k in keys))
    for r in rep["shots"]:
        print(f"  {r['shot']:<9} {r['mix']:6.1f} {r['music']:6.1f} {r['nature']:6.1f} {r['nature_minus_music']:5.1f} {r['st_max']:6.1f} | "
              + " ".join(f"{max(r.get(k, -99), -99):6.1f}" for k in keys))
    print("  octave bands (dB re total) 31 63 125 250 500 1k 2k 4k 8k 16k | slope 125-8k dB/oct")
    for r in rep["shots"]:
        print(f"  {r['shot']:<9} " + " ".join(f"{v:5.1f}" for v in r["oct_db"]) + f" | {r['slope_db_oct']:5.2f}")
    print("  cuts: " + "  ".join(f"{c['cut']:.1f}:{c['momentary_pre']:.1f}->{c['momentary_post']:.1f}" for c in rep["cuts"]))
    print("  motif: " + "  ".join(f"{k} {v['within_half_semitone']:.2f} (oct {v['octave_errors']:.2f}, {v['frames']} fr)"
                                  for k, v in rep["motif_pitch_accuracy"].items()))
    print(f"  master: comp {rep.get('comp_max_gr_db', 0):.2f} dB, limiter {rep.get('limiter_max_gr_db', 0):.2f} dB at "
          f"{rep.get('limiter_gr_at_s', 0):.2f} s, ceiling {rep.get('ceiling_used_db', 0):.2f}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", default=os.path.join(ROOT, "audio", "film_shots.json"))
    ap.add_argument("--nature", default=os.path.join(ROOT, "out", "audio", "nature", "stems", "nature.wav"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "audio", "soundtrack.wav"))
    ap.add_argument("--music-out", default=None, help="also write the mastered music without nature (24-bit)")
    ap.add_argument("--stems", default=None, help="write float32 group stems (at mix gain) into this folder")
    ap.add_argument("--analyze", default=None, help="write analysis.json + spectrogram / pitch PNGs into this folder")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--target-lufs", type=float, default=TARGET_LUFS)
    a = ap.parse_args()

    with open(a.shots, "r", encoding="utf-8") as fh:
        shots = json.load(fh)
    shots = shots.get("shots", shots) if isinstance(shots, dict) else shots
    sc = Score(shots)
    log(f"shots {a.shots}: T={sc.T:.3f} s, bars {np.round(sc.bars, 3).tolist()}")
    G, E, mel = render_music(sc, a.seed)
    music, fade = music_bus(G, sc.n, sc.T)
    # glue: normalise the score to about -15 LUFS, gentle RMS compression
    music = music * dbg(-15.0 - an.integrated_lufs(music))
    music, mcomp = st.bus_compressor(music, thr_db=-17.0, ratio=1.6, knee_db=8.0, att=0.04, rel=0.4)
    log(f"music bus glue compressor max GR {mcomp:.2f} dB")

    nat, sr = an.read_any_wav(a.nature)
    assert sr == SR, sr
    nat = np.asarray(nat, float)[:, :sc.n]
    if nat.shape[-1] < sc.n:
        nat = np.pad(nat, ((0, 0), (0, sc.n - nat.shape[-1])))
    ng, gd = nature_gain(sc, nat, music)
    nature = nat * ng
    log("nature gain per shot (dB): " + ", ".join(f"{s['name']} {g:+.1f}" for s, g in zip(sc.shots, gd)))
    mix = music + nature
    out, info = master(mix, a.target_lufs)
    sg = float(np.sqrt(np.sum(out ** 2) / (np.sum(mix ** 2) + 1e-20)))   # approximate static gain for the stems
    info["mix_to_master_gain_db"] = 20 * math.log10(sg)
    log("master: " + ", ".join(f"{k}={v:.2f}" for k, v in info.items()))
    st.write_wav(a.out, out, False, a.seed)
    log(f"wrote {a.out} ({out.shape[-1]} frames, {out.shape[-1] / SR:.3f} s)")
    if a.music_out:
        mo, minfo = master(music * fade, a.target_lufs)
        st.write_wav(a.music_out, mo, False, a.seed)
        log(f"wrote {a.music_out} (music only, TP {minfo['true_peak_db']:.2f})")
    if a.stems or a.analyze:
        scale = sg
        Gs = {k: v * scale * (np.mean(np.abs(music)) / (np.mean(np.abs(sum(G.values()))) + 1e-20)) for k, v in G.items()}
        if a.stems:
            os.makedirs(a.stems, exist_ok=True)
            from scipy.io import wavfile
            for k, v in Gs.items():
                wavfile.write(os.path.join(a.stems, f"{k}.wav"), SR, v.T.astype(np.float32))
            wavfile.write(os.path.join(a.stems, "nature.wav"), SR, (nature * scale).T.astype(np.float32))
            log(f"wrote stems to {a.stems}")
        if a.analyze:
            x, _ = an.read_wav(a.out)
            rep = analyze(a.analyze, sc, x, music * scale, nature * scale, Gs, E, mel, info, gd)
            print_report(rep)
    log("done")


if __name__ == "__main__":
    main()
