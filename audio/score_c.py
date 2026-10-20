"""score_c.py - candidate C: emotional nature-documentary epic score for the alpine-lake film.

Everything is synthesised here with numpy / scipy: no samples, recordings, soundfonts or
downloads. Strings (ensembles of detuned saw voices with portamento, delayed vibrato, bow
noise, body EQ, spiccato), French horns / trombones / tuba (scoop, breath, brightness that
rises with loudness, growl), SATB choir (glottal source + formant banks, ah/oh morph),
modal piano / celesta / harp, timpani (rolls), toms, gran cassa, suspended cymbal swells and
crashes, sub hits, riser, a synthesised stereo hall IR, bus EQ / compression, nature-bed mix,
-14 LUFS master with a true-peak limiter.

    python audio/score_c.py [--shots audio/film_shots.json]
        [--nature out/audio/nature/stems/nature.wav]
        [--out out/audio/soundtrack.wav] [--work out/audio/work_c] [--analyze]

Form: 2 bars of 4/4 per shot, every cut is a downbeat (60 / 64 / 73.85 / 80 BPM).
Theme (D major): F#4 D5 | C#5 B4 A4 - the rising sixth that falls back by step.
"""
import argparse
import json
import math
import os
import sys
import time
import wave
import zlib

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d, uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as AN  # noqa: E402

SR = 48000
DUR = 60.0
N = int(round(DUR * SR))
_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def rng_for(seed, name):
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


# =============================================================================
# pitch / time helpers
# =============================================================================
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nm(s):
    """'F#4' -> 66, 'Bb1' -> 34"""
    acc, i = 0, 1
    while s[i] in "#b":
        acc += 1 if s[i] == "#" else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + _PC[s[0]] + acc


def mhz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def dbg(v):
    return 10.0 ** (np.asarray(v, float) / 20.0)


class Grid:
    """2 bars (8 beats) per shot; beat positions map linearly inside each shot."""

    def __init__(self, shots):
        self.shots = shots
        self.st = [s["start"] for s in shots] + [shots[-1]["end"]]

    def tb(self, b):  # absolute beat (0-based) -> seconds
        i = int(min(max(b // 8, 0), len(self.shots) - 1))
        fr = (b - 8 * i) / 8.0
        return self.st[i] + fr * (self.st[i + 1] - self.st[i])

    def t(self, bar, beat=1.0):
        return self.tb((bar - 1) * 4 + (beat - 1))

    def beat_len(self, bar):
        i = (bar - 1) // 2
        return (self.st[i + 1] - self.st[i]) / 8.0

    def span(self, bar, beat, dur):
        b0 = (bar - 1) * 4 + (beat - 1)
        return self.tb(b0), self.tb(b0 + dur)


# =============================================================================
# the score (composer's sheet)
# =============================================================================
# chord chart: (bar, beat, beats, symbol, "B T A S") - S is the top inner voice under the tune
CHART = [
    (1, 1, 4, "Dadd9", "D3 A3 E4 F#4"),
    (2, 1, 2, "Gmaj7/B", "B2 G3 D4 F#4"),
    (2, 3, 1, "Asus4", "A2 A3 D4 E4"),
    (2, 4, 1, "A", "A2 A3 C#4 E4"),
    (3, 1, 2, "D", "D2 F#3 A3 D4"),
    (3, 3, 1, "D(add9)", "D2 F#3 A3 E4"),     # inner voice climbs D4 E4 F#4 under the tune
    (3, 4, 1, "D", "D2 F#3 A3 F#4"),
    (4, 1, 2, "A/C#", "C#3 E3 A3 E4"),
    (4, 3, 1, "Bm7(sus4)", "B2 F#3 A3 E4"),
    (4, 4, 1, "Bm7", "B2 F#3 A3 D4"),
    (5, 1, 2, "G", "G2 D3 B3 D4"),
    (5, 3, 2, "Em7", "E2 D3 G3 B3"),
    (6, 1, 2, "D/F#", "F#2 D3 A3 D4"),
    (6, 3, 1, "G", "G2 D3 B3 D4"),
    (6, 4, 1, "A", "A2 E3 A3 C#4"),
    (7, 1, 2, "Bm", "B2 F#3 B3 D4"),
    (7, 3, 2, "Bm/A", "A2 F#3 B3 D4"),
    (8, 1, 2, "Gmaj7", "G2 D3 B3 D4"),
    (8, 3, 1, "Asus4", "A2 E3 A3 D4"),
    (8, 4, 1, "A", "A2 E3 A3 C#4"),
    (9, 1, 2, "D", "D2 F#3 A3 D4"),
    (9, 3, 2, "G/B", "B1 G3 B3 D4"),
    (10, 1, 2, "Bb", "Bb1 F3 Bb3 D4"),       # pivot: bVI of D = IV of F
    (10, 3, 2, "C", "C2 E3 G3 C4"),          # bVII of D = V of F
    (11, 1, 2, "F", "F1 C3 A3 C4"),          # the lift: F major
    (11, 3, 1, "Fsus4", "F1 C3 A3 Bb3"),     # 4-3 in the inner voice
    (11, 4, 1, "F", "F1 C3 F3 A3"),
    (12, 1, 2, "C/E", "E2 C3 G3 C4"),
    (12, 3, 2, "Dm7", "D2 C3 F3 A3"),
    (13, 1, 2, "Bb", "Bb1 F3 Bb3 D4"),
    (13, 3, 2, "Gm7", "G1 F3 Bb3 D4"),
    (14, 1, 2, "F/A", "A1 F3 A3 C4"),
    (14, 3, 1, "Bb", "Bb1 F3 Bb3 D4"),
    (14, 4, 0.5, "Csus4", "C2 G3 C4 F4"),
    (14, 4.5, 0.5, "C", "C2 G3 C4 E4"),
    (15, 1, 2, "Bb", "Bb1 F3 Bb3 D4"),       # climax
    (15, 3, 1, "Eb/Bb", "Bb1 G3 Bb3 Eb4"),
    (15, 4, 1, "Bb", "Bb1 F3 Bb3 D4"),
    (16, 1, 2, "Db", "Db2 Ab3 Db4 F4"),      # bVI
    (16, 3, 2, "Eb", "Eb2 G3 Bb3 Eb4"),      # bVII
    (17, 1, 2, "F", "F1 A3 C4 F4"),          # I - golden arrival
    (17, 3, 2, "Bb/F", "F1 Bb3 D4 F4"),      # plagal
    (18, 1, 4, "F(add9)", "F2 A3 C4 F4"),
]

# the theme, written at its home register: (bar, beat, beats, note)
TUNE = [
    # reveal - P1
    (3, 1, 2, "F#4"), (3, 3, 2, "D5"), (4, 1, 1.5, "C#5"), (4, 2.5, 0.5, "B4"), (4, 3, 2, "A4"),
    # skim - P2 (sequence a step up)
    (5, 1, 2, "G4"), (5, 3, 2, "E5"), (6, 1, 1.5, "D5"), (6, 2.5, 0.5, "C#5"), (6, 3, 1, "B4"), (6, 4, 1, "A4"),
    # waterfall - P3 (the climb)
    (7, 1, 1, "B4"), (7, 2, 1, "C#5"), (7, 3, 1, "D5"), (7, 4, 1, "E5"), (8, 1, 2, "F#5"), (8, 3, 1, "E5"),
    (8, 4, 0.5, "D5"), (8, 4.5, 0.5, "C#5"),
    # stream - the turn towards F
    (9, 1, 1, "A4"), (9, 2, 0.5, "F#4"), (9, 2.5, 0.5, "A4"), (9, 3, 2, "B4"), (10, 1, 2, "D5"), (10, 3, 1.75, "E5"),
    # meadow - P1 in F
    (11, 1, 2, "A4"), (11, 3, 2, "F5"), (12, 1, 1.5, "E5"), (12, 2.5, 0.5, "D5"), (12, 3, 2, "C5"),
    # flyover - P2 in F
    (13, 1, 2, "Bb4"), (13, 3, 2, "G5"), (14, 1, 1.5, "F5"), (14, 2.5, 0.5, "E5"), (14, 3, 1, "D5"),
    (14, 4, 0.75, "C5"),
    # peaks - P1 transformed: minor sixth up, over IV - bVI - bVII
    (15, 1, 2, "D5"), (15, 3, 2, "Bb5"), (16, 1, 1.5, "Ab5"), (16, 2.5, 0.5, "F5"), (16, 3, 1.75, "G5"),
    # finale - golden arrival on the third
    (17, 1, 3.6, "A5"),
]
FOREST_PIANO = [(1, 2, 1, "F#4"), (1, 3, 2, "D5"), (2, 1, 1.5, "C#5"), (2, 2.5, 0.5, "B4"), (2, 3, 2, "A4")]
FINALE_PIANO = [(17, 3, 1, "A4"), (17, 4, 1, "F5"), (18, 1, 1.5, "E5"), (18, 2.5, 0.5, "D5"), (18, 3, 2.0, "C5")]

# dynamics per bar for the melody (0..1)
MEL_VEL = {3: .82, 4: .78, 5: .84, 6: .82, 7: .80, 8: .90, 9: .55, 10: .70, 11: .70, 12: .66,
           13: .93, 14: .95, 15: 1.0, 16: 1.0, 17: .78, 18: .5}


# =============================================================================
# DSP basics
# =============================================================================
def sos(kind, f, order=2):
    return signal.butter(order, f, kind, fs=SR, output="sos")


def filt(s, x):
    return signal.sosfilt(s, x, axis=-1)


def peq(fc, gain_db, q):
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return np.hstack([b / a[0], a / a[0]])[None, :]


def shelf(fc, gain_db, kind="low"):
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
        a = [(A + 1) - (A - 1) * cw + sA, 2 * ((A - 1) + (A + 1) * cw), (A + 1) - (A - 1) * cw - sA]
    b, a = np.array(b), np.array(a)
    return np.hstack([b / a[0], a / a[0]])[None, :]


def chain(*soss):
    return np.vstack(soss)


def bp_sos(fc, bw):
    """RBJ band-pass with 0 dB peak."""
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / (2 * fc / bw)
    a0 = 1 + al
    return np.array([[al / a0, 0.0, -al / a0, 1.0, -2 * np.cos(w) / a0, (1 - al) / a0]])


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
    bus[0, i0:i1] += (gl * gain) * y
    bus[1, i0:i1] += (gr * gain) * y


def add_st(bus, y, i0, gain=1.0):
    n = bus.shape[-1]
    i1 = i0 + y.shape[-1]
    if i0 < 0:
        y, i0 = y[:, -i0:], 0
    if i1 > n:
        y, i1 = y[:, :n - i0], n
    if i1 > i0:
        bus[:, i0:i1] += gain * y


def saw_blep(ph, dt):
    y = 2.0 * ph - 1.0
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] -= t + t - t * t - 1.0
    m = ph > 1.0 - dt
    t = (ph[m] - 1.0) / dt[m]
    y[m] -= t * t + t + t + 1.0
    return y


def osc_saw(f, rng):
    dt = np.minimum(np.asarray(f, float) / SR, 0.45)
    ph = (rng.random() + np.cumsum(dt)) % 1.0
    return saw_blep(ph, dt)


def slow_noise(rng, L, rate_hz=2.0, hop=240):
    m = L // hop + 8
    v = gaussian_filter1d(rng.standard_normal(m), max(0.5, SR / hop / (2 * np.pi * rate_hz)), mode="wrap")
    v /= v.std() + 1e-12
    return np.interp(np.arange(L) / hop, np.arange(m), v)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade_tail(y, k):
    k = min(k, y.shape[-1])
    if k > 1:
        y[..., -k:] *= np.cos(0.5 * np.pi * np.linspace(0, 1, k)) ** 2
    return y


def body_ir(rng, fmin, fmax, nmodes, length=0.05, qr=(5, 18), mix=0.7):
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = np.zeros(n)
    for f in np.exp(rng.uniform(np.log(fmin), np.log(fmax), nmodes)):
        Q = rng.uniform(*qr)
        tau = Q / (np.pi * f)
        ir += rng.standard_normal() * (np.pi * f / Q) * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.283))
    ir *= np.cos(0.5 * np.pi * np.clip((t - 0.6 * length) / (0.4 * length), 0, 1))
    ir *= mix / (np.sqrt(np.sum(ir ** 2)) + 1e-12)
    ir[0] += 1.0
    return ir / np.sqrt(np.sum(ir ** 2))


# =============================================================================
# expressive legato player: pitch / amplitude curves for one performer
# =============================================================================
def player(notes, rng, att=0.25, rel=0.4, glide=0.07, vib_rate=5.4, vib_c=14.0, vib_delay=0.3,
           drift_c=4.0, detune_c=0.0, jitter=0.012, scoop_c=0.0, scoop_t=0.05, join=0.04, slide_p=0.4,
           dip=0.12, amp_wobble=0.03, fast=(), fast_att=0.05):
    """notes: [(t0, t1, midi, v0, v1)]. Returns (i0, f_hz, amp) over the active region."""
    nts = []
    for (t0, t1, mi, v0, v1) in sorted(notes, key=lambda n: n[0]):
        j = rng.normal(0, jitter)
        nts.append([t0 + j, t1 + j + rng.normal(0, 0.5 * jitter), float(mi), max(v0, 1e-4), max(v1, 1e-4)])
    for a, b in zip(nts[:-1], nts[1:]):          # legato lines must not overlap
        if b[0] - a[1] < join:
            a[1] = b[0]
    ts = nts[0][0] - 0.05
    te = max(n[1] for n in nts) + rel + 0.1
    i0 = max(0, int(ts * SR))
    i1 = min(N, int(te * SR))
    L = i1 - i0
    if L <= 16:
        return i0, None, None
    H = 48
    nc = L // H + 2
    tc = (i0 + np.arange(nc) * H) / SR
    pitch = np.full(nc, nts[0][2])
    lev = np.zeros(nc)
    phrases = []
    cur = None
    for k, (t0, t1, mi, v0, v1) in enumerate(nts):
        m = (tc >= t0) & (tc < t1)
        pitch[tc >= t0] = mi
        if np.any(m):
            fr = (tc[m] - t0) / max(t1 - t0, 1e-3)
            lev[m] = np.exp(np.log(v0) + (np.log(v1) - np.log(v0)) * fr)
        legato = k > 0 and t0 - nts[k - 1][1] < join
        if legato:
            cur[1] = t1
            cur[2] = v1
            # portamento / slur
            pa, pb = nts[k - 1][2], mi
            d = abs(pb - pa)
            g = glide * rng.uniform(0.75, 1.3) * (1 + 0.06 * min(d, 12))
            if d >= 5 and rng.random() < slide_p:
                g *= 2.2
            g = min(g, 0.3, 0.45 * (t1 - t0) + 0.02)
            s0 = t0 - 0.35 * g
            m2 = (tc >= s0) & (tc < s0 + g)
            pitch[m2] = pa + (pb - pa) * smoothstep((tc[m2] - s0) / g)
        else:
            cur = [t0, t1, v1, mi]
            phrases.append(cur)
    lev = gaussian_filter1d(lev, 0.03 * SR / H, mode="nearest")
    env = np.zeros(nc)
    for (p0, p1, vend, _) in phrases:
        a = att * rng.uniform(0.8, 1.25)
        if any(-0.03 <= p0 - h <= 0.08 for h in fast):
            a = fast_att * rng.uniform(0.8, 1.2)
        up = np.sin(0.5 * np.pi * np.clip((tc - p0) / a, 0, 1)) ** 2
        down = np.cos(0.5 * np.pi * np.clip((tc - p1) / rel, 0, 1)) ** 2
        w = up * down
        w[tc < p0] = 0
        env = np.maximum(env, w)
        m = (tc >= p1) & (tc < p1 + rel)
        lev[m] = np.maximum(lev[m], vend)
    amp = lev * env
    # bow change / re-articulation dips and vibrato envelope
    vibenv = np.zeros(nc)
    for (p0, p1, _, _) in phrases:
        vibenv = np.maximum(vibenv, np.clip((tc - p0 - vib_delay) / 0.45, 0, 1) * (tc < p1 + rel))
    for k in range(1, len(nts)):
        tb = nts[k][0]
        if tb - nts[k - 1][1] < join:
            g = np.exp(-((tc - tb) / 0.035) ** 2)
            amp *= 1 - dip * g
            vibenv *= 1 - 0.55 * np.exp(-((tc - tb) / 0.09) ** 2)
    # scoops on articulated note starts
    cents = np.full(nc, detune_c)
    if scoop_c:
        for (p0, _, _, _) in phrases:
            m = tc >= p0
            cents[m] -= scoop_c * np.exp(-(tc[m] - p0) / scoop_t)
    rate = vib_rate * rng.uniform(0.92, 1.08) * (1 + 0.04 * slow_noise(rng, nc * H, 0.3)[::H][:nc])
    vph = 2 * np.pi * np.cumsum(rate) * H / SR + rng.uniform(0, 6.28)
    depth = vib_c * rng.uniform(0.7, 1.3) * (0.6 + 0.4 * np.clip(amp / (amp.max() + 1e-9), 0, 1))
    cents += depth * vibenv * np.sin(vph)
    cents += drift_c * slow_noise(rng, nc * H, 0.4)[::H][:nc]
    midi_c = pitch + cents / 100.0
    ta = np.arange(L) / H
    f = mhz(np.interp(ta, np.arange(nc), midi_c))
    amp_a = np.interp(ta, np.arange(nc), amp)
    if amp_wobble:
        amp_a *= 1 + amp_wobble * slow_noise(rng, L, 6.0)
    return i0, f, amp_a


def to_notes(G, items, vel_fn, oct_shift=0, hair=None):
    """(bar, beat, beats, note) -> (t0, t1, midi, v0, v1)."""
    out = []
    for (bar, beat, dur, nn) in items:
        t0, t1 = G.span(bar, beat, dur)
        v = vel_fn(bar, beat)
        v0, v1 = (v, v) if hair is None else hair(bar, beat, dur, v)
        out.append((t0, t1, nm(nn) + oct_shift if isinstance(nn, str) else nn + oct_shift, v0, v1))
    return out


# =============================================================================
# instruments
# =============================================================================
SEC = {  # string sections: voices, pan centre/spread, LP dark/bright, body EQ, bow-noise band
    "vln1": dict(nv=14, pan=-0.55, spread=0.28, lo=2400, hi=7500, eq=[(480, 2.5, 1.2), (1250, -1.5, 1.4), (2900, 1.0, 1.6), (4000, -2.5, 1.2)],
                 bow=(2500, 7000), hp=180, body=(260, 6000)),
    "vln2": dict(nv=12, pan=-0.22, spread=0.22, lo=2100, hi=6500, eq=[(450, 2.0, 1.2), (1200, -1.5, 1.4), (2800, 0.8, 1.6), (4000, -2.5, 1.2)],
                 bow=(2200, 6500), hp=160, body=(250, 6000)),
    "vla": dict(nv=10, pan=0.22, spread=0.22, lo=1500, hi=5000, eq=[(350, 1.0, 1.0), (280, -1.5, 1.2), (900, 1.5, 1.2), (2600, -2.0, 1.2)],
                bow=(1800, 5000), hp=110, body=(180, 4500)),
    "vc": dict(nv=10, pan=0.46, spread=0.2, lo=1000, hi=3800, eq=[(220, 1.0, 1.2), (320, -2.0, 1.2), (650, 1.5, 1.2), (2400, -2.5, 1.2)],
               bow=(1200, 4000), hp=55, body=(90, 3500)),
    "cb": dict(nv=8, pan=0.66, spread=0.14, lo=600, hi=2000, eq=[(90, 1.5, 1.0), (300, -2.5, 1.0), (1200, -2.0, 1.0)],
               bow=(800, 2500), hp=32, body=(45, 1800)),
}


_JIT_SOS = signal.butter(2, 260, "lowpass", fs=SR, output="sos")


def string_section(name, lines, rng, att=0.3, rel=0.45, vib=1.0, glide=0.075, bow=1.0, fast=()):
    """lines: list of legato note lists; voices are distributed over the lines (divisi)."""
    c = SEC[name]
    desks = [np.zeros((2, N)), np.zeros((2, N))]
    envm = np.zeros(N)
    nl = len(lines)
    for v in range(c["nv"]):
        notes = lines[v % nl]
        if not notes:
            continue
        rv = np.random.default_rng(rng.integers(1 << 31))
        i0, f, amp = player(notes, rv, att=att, rel=rel, glide=glide, vib_rate=5.5, vib_c=13.0 * vib,
                            vib_delay=rv.uniform(0.18, 0.45), drift_c=3.5, detune_c=rv.normal(0, 5.0),
                            jitter=0.014, slide_p=0.35, dip=0.14, fast=fast, fast_att=0.06)
        if f is None:
            continue
        jit = signal.sosfilt(_JIT_SOS, rv.standard_normal(len(f)))
        f = f * (1 + 0.012 * jit)
        y = osc_saw(f, rv)
        y = 0.82 * y + 0.18 * osc_saw(f * (1 + rv.normal(0, 0.0007)), rv)  # bow-pressure "double"
        y *= amp
        p = c["pan"] + rv.uniform(-1, 1) * c["spread"]
        add_mono(desks[v % 2], y, i0, p)
        envm[i0:i0 + len(amp)] += amp
    if not np.any(envm):
        return desks[0]
    fb = c["body"]
    bus = np.zeros((2, N))
    for d in desks:   # wooden body: dense irregular resonances, one body per desk
        ir = body_ir(rng, fb[0], fb[1], 72, 0.04, (4, 11), 0.4)
        bus += np.vstack([signal.oaconvolve(d[k], ir)[:N] for k in range(2)])
    nv = c["nv"]
    w = np.clip(envm / nv / 0.9, 0, 1) ** 0.8
    lo = filt(sos("lowpass", c["lo"], 2), bus)
    hi = filt(sos("lowpass", c["hi"], 2), bus)
    y = lo + (hi - lo) * w
    eq = chain(sos("highpass", c["hp"], 2), *[peq(fq, g, q) for fq, g, q in c["eq"]], sos("lowpass", 11000, 2))
    y = filt(eq, y)
    # bow noise follows the bowing envelope, accentuated on attacks
    if bow > 0:
        e = envm / nv
        de = np.maximum(np.diff(e, prepend=0), 0) * SR * 0.05
        nz = rng.standard_normal((2, N)) * (0.5 * e + np.clip(de, 0, 1.5))
        nz = filt(chain(sos("bandpass", list(c["bow"]), 2)), nz)
        y += 0.022 * bow * nz
    return y / math.sqrt(nv)


def spiccato(name, hits, rng, nv=6, dec=0.055, bright=1.0):
    """hits: [(t, midi, vel)] - short bow strokes: sharp chiff, quick decay, body."""
    c = SEC[name]
    bus = np.zeros((2, N))
    Ls = int(0.34 * SR)
    t = np.arange(Ls) / SR
    for (t0, mi, vel) in hits:
        for v in range(nv):
            dt = rng.normal(0, 0.006)
            f0 = mhz(mi + rng.normal(0, 0.06))
            f = f0 * (1 + 0.012 * np.exp(-t / 0.018))
            y = osc_saw(f, rng)
            d = dec * rng.uniform(0.85, 1.2) * (0.8 + 0.4 * vel)
            env = (1 - np.exp(-t / 0.0022)) * (0.72 * np.exp(-t / d) + 0.28 * np.exp(-t / (2.6 * d)))
            env *= np.cos(0.5 * np.pi * np.clip((t - 0.22) / 0.12, 0, 1)) ** 2
            ch = rng.standard_normal(Ls) * np.exp(-t / 0.006) * 0.6
            y = y * env + ch * (t < 0.04)
            fc = min(9000, f0 * (5 + 9 * vel * bright) + 1200)
            y = signal.sosfilt(sos("lowpass", fc, 2), y)
            add_mono(bus, y * vel, int((t0 + dt) * SR), c["pan"] + rng.uniform(-1, 1) * c["spread"])
    eq = chain(sos("highpass", c["hp"], 2), *[peq(fq, g, q) for fq, g, q in c["eq"]])
    return filt(eq, bus) / math.sqrt(nv)


BRASS = {
    "horn": dict(cut=(550, 1300, 3000), eq=[(420, 2.5, 1.0), (250, -1.0, 1.0), (2800, -3.5, 1.0)], lp=5500, hp=70,
                 drive=1.6, breath=(700, 2200), scoop=35, vib=4.0, att=0.10),
    "tbn": dict(cut=(700, 1700, 4200), eq=[(550, 1.5, 1.0), (300, -1.5, 1.0), (3000, -2.5, 1.0)], lp=7000, hp=55,
                drive=2.2, breath=(900, 3000), scoop=28, vib=2.5, att=0.07),
    "tuba": dict(cut=(300, 700, 1500), eq=[(180, 1.5, 1.0), (350, -2.0, 1.0)], lp=3000, hp=28,
                 drive=1.8, breath=(400, 1400), scoop=20, vib=2.0, att=0.12),
}


def brass_player(kind, notes, rng, pan, gain=1.0, growl=0.0, fast=()):
    c = BRASS[kind]
    bus = np.zeros((2, N))
    i0, f, amp = player(notes, rng, att=c["att"], rel=0.35, glide=0.045, vib_rate=5.0, vib_c=c["vib"],
                        vib_delay=0.45, drift_c=2.5, detune_c=rng.normal(0, 3.0), jitter=0.01,
                        scoop_c=c["scoop"], scoop_t=0.045, slide_p=0.1, dip=0.25, amp_wobble=0.02, fast=fast, fast_att=0.035)
    if f is None:
        return bus
    y = osc_saw(f, rng)
    a = amp / (amp.max() + 1e-9)
    w = np.clip(a, 0, 1) ** 1.3
    l0 = signal.sosfilt(sos("lowpass", c["cut"][0], 2), y)
    l1 = signal.sosfilt(sos("lowpass", c["cut"][1], 2), y)
    l2 = signal.sosfilt(sos("lowpass", c["cut"][2], 2), y)
    w0 = np.clip(1 - 2 * w, 0, 1)
    w2 = np.clip(2 * w - 1, 0, 1)
    y = l0 * w0 + l1 * (1 - w0 - w2) + l2 * w2
    if growl > 0:   # loud: more drive and a little lip flutter
        dr = 1 + c["drive"] * growl * w ** 2
        fl = 1 + 0.07 * growl * w ** 3 * np.sin(2 * np.pi * 27 * np.arange(len(y)) / SR + rng.uniform(0, 6))
        y = np.tanh(dr * y * fl) / np.tanh(dr)
    y *= amp
    # breath: burst on every articulation + a little constant air
    L = len(y)
    da = np.maximum(np.diff(amp, prepend=0), 0) * SR * 0.04
    nz = rng.standard_normal(L) * (0.05 * amp + np.clip(da, 0, 1.0) * 0.6)
    nz = signal.sosfilt(sos("bandpass", list(c["breath"]), 2), nz)
    y = y + 0.35 * nz
    add_mono(bus, y * gain, i0, pan)
    return bus


def brass_finish(kind, bus):
    c = BRASS[kind]
    eq = chain(sos("highpass", c["hp"], 2), *[peq(fq, g, q) for fq, g, q in c["eq"]], sos("lowpass", c["lp"], 2))
    return filt(eq, bus)


# --- choir ------------------------------------------------------------------
FORMANTS = {  # (freqs, dB, bandwidths)
    ("S", "a"): ([800, 1150, 2900, 3900, 4950], [0, -6, -32, -20, -50], [80, 90, 120, 130, 140]),
    ("S", "o"): ([450, 800, 2830, 3800, 4950], [0, -11, -22, -22, -50], [70, 80, 100, 130, 135]),
    ("A", "a"): ([800, 1150, 2800, 3500, 4950], [0, -4, -20, -36, -60], [80, 90, 120, 130, 140]),
    ("A", "o"): ([450, 800, 2830, 3500, 4950], [0, -9, -16, -28, -55], [70, 80, 100, 130, 135]),
    ("T", "a"): ([650, 1080, 2650, 2900, 3250], [0, -6, -7, -8, -22], [80, 90, 120, 130, 140]),
    ("T", "o"): ([400, 800, 2600, 2800, 3000], [0, -10, -12, -12, -26], [70, 80, 100, 130, 135]),
    ("B", "a"): ([600, 1040, 2250, 2450, 2750], [0, -7, -9, -9, -20], [60, 70, 110, 120, 130]),
    ("B", "o"): ([400, 750, 2400, 2600, 2900], [0, -11, -21, -20, -40], [40, 80, 100, 120, 120]),
}
CHOIR_PAN = {"S": -0.3, "A": 0.3, "T": -0.15, "B": 0.15}


def formant_bank(x, part, vowel):
    fr, gd, bw = FORMANTS[(part, vowel)]
    y = np.zeros_like(x)
    for f, g, b in zip(fr, gd, bw):
        y += dbg(g) * filt(bp_sos(f, b * 1.6), x)   # ensemble = wider formants
    return y


def choir_part(part, notes, rng, vowel_o, nv=9, fast=()):
    """vowel_o: function t(sec array) -> 0..1 amount of 'o' (0 = 'ah')."""
    src = np.zeros((2, N))
    envm = np.zeros(N)
    for v in range(nv):
        rv = np.random.default_rng(rng.integers(1 << 31))
        i0, f, amp = player(notes, rv, att=0.45, rel=0.6, glide=0.09, vib_rate=5.2, vib_c=28.0,
                            vib_delay=rv.uniform(0.2, 0.5), drift_c=6.0, detune_c=rv.normal(0, 7.0),
                            jitter=0.025, slide_p=0.2, dip=0.1, amp_wobble=0.06, fast=fast, fast_att=0.09)
        if f is None:
            continue
        # glottal-ish source: saw + jitter, extra spectral tilt applied after the sum
        y = osc_saw(f * (1 + 0.002 * rv.standard_normal(len(f)).cumsum() / np.sqrt(np.arange(1, len(f) + 1))), rv)
        y *= amp
        add_mono(src, y, i0, CHOIR_PAN[part] + rv.uniform(-0.35, 0.35))
        envm[i0:i0 + len(amp)] += amp
    if not np.any(envm):
        return src
    src = filt(sos("lowpass", 900 if part in "SA" else 600, 1), src) * 3.0
    br = rng.standard_normal((2, N)) * (envm / nv) * 0.05       # aspiration
    src = src + br
    a = formant_bank(src, part, "a")
    o = formant_bank(src, part, "o")
    tt = np.arange(N) / SR
    wo = vowel_o(tt)
    y = a * (1 - wo) + o * wo * 1.25
    y = filt(chain(sos("highpass", 110 if part in "SA" else 75, 2), peq(3000, -3, 1.0)), y)
    return y / math.sqrt(nv)


# --- keyboards / plucked / mallets -----------------------------------------
def piano_note(rng, midi, vel, dur, pedal=0.0):
    f0 = mhz(midi)
    B = 2.4e-4 * 2 ** ((midi - 60) / 14.0)
    tau1 = 7.0 * 2 ** (-(midi - 48) / 16.0)
    Ls = min(dur + 0.4 + pedal, 6.5)
    L = int(Ls * SR)
    t = np.arange(L) / SR
    y = np.zeros(L)
    nmax = int(min(30, 11000 / f0))
    x0 = 0.12
    for n in range(1, nmax + 1):
        fn = n * f0 * math.sqrt(1 + B * n * n)
        if fn > 16000:
            break
        a = abs(math.sin(math.pi * n * x0)) / n ** (1.55 - 0.55 * vel)
        a *= 1.0 / (1 + (fn / (f0 * (4 + 22 * vel) + 600)) ** 2)
        tau = tau1 / (1 + 0.18 * (n - 1) ** 1.2)
        dec = 0.65 * np.exp(-t / (0.22 * tau)) + 0.35 * np.exp(-t / tau)
        for s, dc in ((0, 0.0), (1, rng.normal(0, 0.9)), (2, rng.normal(0, 0.9))):   # 3 strings, beating
            if (midi < 40 or n > 14) and s == 2:
                continue
            fs = fn * 2 ** (dc / 1200.0)
            ph = rng.uniform(0, 6.283) if s else 0.0
            y += a * dec * np.sin(2 * np.pi * fs * t + ph) * (0.5 if s else 0.7)
    # damper
    toff = dur + pedal
    y *= np.where(t < toff, 1.0, np.exp(-(t - toff) / 0.11))
    # hammer: short felt thump + noise
    hm = rng.standard_normal(L) * np.exp(-t / 0.004) * (0.04 + 0.08 * vel)
    hm = signal.sosfilt(sos("lowpass", 1200 + 3500 * vel, 2), hm)
    y = (y + hm) * (1 - np.exp(-t / 0.0012))
    fade_tail(y, int(0.15 * SR))
    return y * vel


def celesta_note(rng, midi, vel, Ls=2.8):
    f0 = mhz(midi)
    L = int(Ls * SR)
    t = np.arange(L) / SR
    y = np.sin(2 * np.pi * f0 * t) * np.exp(-t / 1.4)
    y += 0.22 * np.sin(2 * np.pi * 2 * f0 * t + 0.5) * np.exp(-t / 0.5)
    y += 0.12 * np.sin(2 * np.pi * 2.76 * f0 * t + 1.0) * np.exp(-t / 0.12)
    y += 0.06 * np.sin(2 * np.pi * 5.4 * f0 * t + 2.0) * np.exp(-t / 0.05)
    y += signal.sosfilt(sos("lowpass", 3500, 2), rng.standard_normal(L) * np.exp(-t / 0.002) * 0.05)
    y *= 1 - np.exp(-t / 0.0008)
    fade_tail(y, int(0.4 * SR))
    return y * vel


def harp_note(rng, midi, vel, Ls=None):
    f0 = mhz(midi)
    tau1 = 4.5 * 2 ** (-(midi - 48) / 15.0)
    Ls = Ls or min(5.0, 0.6 + 2.2 * tau1)
    L = int(Ls * SR)
    t = np.arange(L) / SR
    y = np.zeros(L)
    p = 0.3 + rng.uniform(-0.04, 0.04)
    for n in range(1, int(min(24, 9000 / f0)) + 1):
        a = abs(math.sin(math.pi * n * p)) / n ** 1.25
        tau = tau1 / (1 + 0.35 * (n - 1))
        y += a * np.sin(2 * np.pi * n * f0 * (1 + 1e-5 * n * n) * t + rng.uniform(0, 0.3)) * np.exp(-t / tau)
    pl = rng.standard_normal(L) * np.exp(-t / 0.003) * 0.06
    y = (y + signal.sosfilt(sos("bandpass", [500, 4000], 1), pl)) * (1 - np.exp(-t / 0.0015))
    fade_tail(y, int(0.3 * SR))
    return y * vel


# --- percussion -------------------------------------------------------------
def timp_hit(rng, midi, vel, decay=1.0, L=3.5):
    f = mhz(midi)
    n = int(L * SR)
    t = np.arange(n) / SR
    glide = 1 + 0.03 * vel * np.exp(-t / 0.07)
    y = np.zeros(n)
    for r, a, d in ((1.0, 1.0, 1.9), (1.504, 0.55, 1.1), (1.742, 0.25, 0.7), (2.0, 0.32, 0.8),
                    (2.245, 0.15, 0.5), (2.494, 0.14, 0.45), (2.8, 0.08, 0.3), (2.98, 0.08, 0.3)):
        ph = 2 * np.pi * np.cumsum(f * r * glide) / SR
        y += a * (0.4 + 0.6 * vel) ** (r - 1) * np.sin(ph + rng.uniform(0, 6)) * np.exp(-t / (d * decay))
    y += 0.7 * np.sin(2 * np.pi * np.cumsum(0.55 * f * glide) / SR) * np.exp(-t / 0.12)   # thud
    m = rng.standard_normal(n) * np.exp(-t / 0.005) * (0.3 + 0.5 * vel)
    y += signal.sosfilt(sos("lowpass", 1500 + 3000 * vel, 2), m)
    y *= 1 - np.exp(-t / 0.0015)
    fade_tail(y, int(0.5 * SR))
    return y * vel


def timp_roll(rng, bus, midi, t0, t1, v0, v1, pan=0.0, rate=15.0, stop_before=0.0):
    t = t0
    k = 0
    while t < t1 - stop_before:
        fr = (t - t0) / max(t1 - t0, 1e-3)
        v = v0 * (v1 / v0) ** (fr ** 1.6)
        v *= rng.uniform(0.82, 1.0) * (1.0 if k % 2 else 0.93)
        add_mono(bus, timp_hit(rng, midi, v, decay=0.8, L=2.2), int(t * SR), pan + rng.uniform(-0.04, 0.04))
        t += (1.0 / (rate * (0.8 + 0.4 * fr))) * rng.uniform(0.9, 1.1)
        k += 1


def drum_hit(rng, f0, vel, decay=0.6, L=2.0, modes=((1.0, 1.0, 1.0), (1.59, 0.45, 0.55), (2.14, 0.3, 0.4),
                                                   (2.65, 0.2, 0.3), (3.16, 0.12, 0.25)),
             sweep=0.5, slap=0.4, slap_band=(800, 3500)):
    n = int(L * SR)
    t = np.arange(n) / SR
    g = 1 + sweep * vel * np.exp(-t / 0.035)
    y = np.zeros(n)
    for r, a, d in modes:
        y += a * np.sin(2 * np.pi * np.cumsum(f0 * r * g) / SR + rng.uniform(0, 6)) * np.exp(-t / (d * decay))
    s = rng.standard_normal(n) * np.exp(-t / 0.008)
    y += slap * vel * signal.sosfilt(sos("bandpass", list(slap_band), 2), s)
    y *= 1 - np.exp(-t / 0.001)
    fade_tail(y, int(0.4 * SR))
    return y * vel


def bass_drum(rng, vel, L=4.0):
    return drum_hit(rng, 46.0, vel, decay=1.9, L=L, modes=((1.0, 1.0, 1.0), (1.47, 0.4, 0.6), (1.94, 0.25, 0.45),
                                                           (2.4, 0.15, 0.35)), sweep=0.45, slap=0.15, slap_band=(300, 1800))


def tom(rng, which, vel):
    f = {"lo": 74.0, "mid": 98.0, "hi": 131.0}[which] * rng.uniform(0.985, 1.015)
    return drum_hit(rng, f, vel, decay=0.55, L=1.6, sweep=0.35, slap=0.35)


def cymbal(rng, L, t_peak, vel, decay=2.6, kind="swell"):
    """Suspended cymbal. 'swell' = mallet roll crescendo peaking at t_peak then ringing;
    'crash' = hit at t_peak. Noise + a cloud of inharmonic shimmer partials."""
    n = int(L * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    nz = rng.standard_normal((2, n))
    lo = filt(sos("bandpass", [900, 5000], 2), nz)
    hi = filt(sos("bandpass", [4500, 14000], 2), nz)
    sh = np.zeros((2, n))
    for _ in range(70):
        f = math.exp(rng.uniform(math.log(500), math.log(11000)))
        am = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 4) * t + rng.uniform(0, 6))
        c = rng.integers(2)
        sh[c] += np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * am / math.sqrt(f / 500)
    sh /= 30.0
    tp = t_peak
    if kind == "swell":
        up = np.clip(t / tp, 0, 1) ** 3.0
        after = np.exp(-np.maximum(t - tp, 0) / decay)
        e_lo = np.where(t < tp, up, after)
        e_hi = np.where(t < tp, up ** 1.3, np.exp(-np.maximum(t - tp, 0) / (0.6 * decay)))
        rollmod = 1 + 0.25 * np.sin(2 * np.pi * 9.0 * t) * (t < tp)
        e_lo, e_hi = e_lo * rollmod, e_hi * rollmod
    else:
        att = np.clip((t - tp) / 0.002, 0, 1)
        e_lo = att * np.exp(-np.maximum(t - tp, 0) / decay)
        e_hi = att * (0.6 * np.exp(-np.maximum(t - tp, 0) / (0.25 * decay)) + 0.4 * np.exp(-np.maximum(t - tp, 0) / (0.7 * decay)))
    out = lo * e_lo * 0.5 + hi * e_hi * 0.45 + sh * e_lo
    out = filt(sos("highpass", 350, 2), out)
    fade_tail(out, int(0.6 * SR))
    return out * vel


def sub_hit(vel, f0=50.0, f1=33.0, decay=1.6, L=5.0):
    n = int(L * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.25)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / decay) * (1 - np.exp(-t / 0.004))
    y += 0.15 * np.sin(4 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.5 * decay))
    fade_tail(y, int(1.0 * SR))
    return y * vel


def riser(rng, L, vel, f_lo=300.0, f_hi=6000.0):
    """Noise riser: a band of noise sweeping up (crossfaded fixed bands, no zipper / clicks)."""
    n = int(L * SR)
    t = np.arange(n) / SR
    fr = t / L
    nz = rng.standard_normal((2, n))
    lfc = np.log(f_lo) + (np.log(f_hi) - np.log(f_lo)) * fr ** 1.5
    out = np.zeros((2, n))
    cents = np.exp(np.linspace(np.log(f_lo), np.log(f_hi), 12))
    for fc in cents:
        b = filt(sos("bandpass", [fc / 1.35, min(fc * 1.35, 20000)], 2), nz)
        out += b * np.exp(-((lfc - np.log(fc)) / 0.3) ** 2)
    out *= (fr ** 2.2) * vel
    out[:, -int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return out


# --- hall reverb ------------------------------------------------------------
def make_ir(rng, rt=(2.6, 3.1, 2.4, 1.5), pre=0.018, L=4.8, er=1.0, width=1.0):
    """Stereo hall IR: early reflections + band-wise exponential diffuse tail.
    rt = RT60 for bands <250 Hz, 250 Hz-1.5 kHz, 1.5-5 kHz, >5 kHz."""
    n = int(L * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n))
    edges = [(None, 250), (250, 1500), (1500, 5000), (5000, None)]
    for (lo, hi), r in zip(edges, rt):
        nz = rng.standard_normal((2, n))
        if lo is None:
            s = sos("lowpass", hi, 4)
        elif hi is None:
            s = sos("highpass", lo, 4)
        else:
            s = sos("bandpass", [lo, hi], 2)
        nz = filt(s, nz)
        ir += nz * np.exp(-6.9078 * t / r)
    # density build-up after the pre-delay
    ramp = np.clip((t - pre) / 0.045, 0, 1) ** 1.5
    ir *= ramp
    ir *= 1.0 / (np.sqrt(np.sum(ir ** 2) / 2) + 1e-12)
    # early reflections
    erb = np.zeros((2, n))
    for k in range(26):
        d = pre * 0.4 + rng.uniform(0.004, 0.085) ** 1.1
        g = er * 0.55 * (0.012 / max(d, 0.012)) ** 0.9 * rng.uniform(0.5, 1.0)
        for c in range(2):
            i = int((d + rng.uniform(0, 0.006)) * SR)
            if i < n:
                erb[c, i] += g * rng.choice([-1, 1]) * rng.uniform(0.6, 1.0)
    erb = filt(sos("lowpass", 6000, 1), erb)
    ir = ir + erb * 1.0
    # width: blend L/R
    m = 0.5 * (ir[0] + ir[1])
    s = 0.5 * (ir[0] - ir[1]) * width
    ir = np.vstack([m + s, m - s])
    fade_tail(ir, int(0.4 * SR))
    return ir / (np.sqrt(np.sum(ir ** 2) / 2) + 1e-12)


def convolve_st(x, ir):
    return np.vstack([signal.oaconvolve(x[c], ir[c])[:N] for c in range(2)])


# =============================================================================
# arrangement
# =============================================================================
def chart_lines(G, bars, which, shift=0, vel=0.6, merge=True, vel_fn=None, fit=None):
    """Legato line of one chart voice (0=B 1=T 2=A 3=S) over a bar range; equal consecutive
    pitches are tied into one note (held common tones)."""
    out = []
    for (bar, beat, dur, sym, v) in CHART:
        if bar not in bars:
            continue
        mi = nm(v.split()[which]) + shift
        if fit:
            lo, hi = fit
            while mi < lo:
                mi += 12
            while mi > hi:
                mi -= 12
        t0, t1 = G.span(bar, beat, dur)
        vv = vel_fn(bar, beat) if vel_fn else vel
        if merge and out and out[-1][2] == mi and abs(out[-1][1] - t0) < 1e-6:
            out[-1] = (out[-1][0], t1, mi, out[-1][3], vv)
        else:
            out.append((t0, t1, mi, vv, vv))
    return out


def chord_at(bar, beat):
    best = None
    for c in CHART:
        if (c[0], c[1]) <= (bar, beat):
            best = c
    return best


def tune_part(G, bars, shift, vel_scale=1.0, hair=True):
    items = [n for n in TUNE if n[0] in bars]

    def vf(bar, beat):
        return MEL_VEL[bar] * vel_scale

    def hp(bar, beat, dur, v):   # long notes swell a little (messa di voce)
        if dur >= 2:
            return v * 0.85, v * 1.05
        return v, v
    return to_notes(G, items, vf, shift, hp if hair else None)


def cut_at(notes, t_cut, gap=0.0):
    """Shorten notes so they end `gap` s before t_cut (silence before a hit)."""
    out = []
    for (t0, t1, mi, v0, v1) in notes:
        if t0 < t_cut - gap < t1 <= t_cut + 0.02:
            t1 = t_cut - gap
        out.append((t0, t1, mi, v0, v1))
    return out


def arrange_and_render(G, seed):
    R = lambda name: rng_for(seed, name)  # noqa: E731
    sh = {s["name"]: s for s in G.shots}
    buses = {}
    HITS = [G.t(11), G.t(13), G.t(15), G.t(17)]       # 35, 41.5, 48, 54 s
    GAP = 0.16
    FAST = (G.t(13), G.t(15))                         # sforzando tutti entries: flyover, peaks

    def gap_all(notes):
        for h in HITS:
            notes = cut_at(notes, h, GAP)
        return notes

    # ---------------------------------------------------------------- strings
    log("strings")
    # vln1: forest crescendo note, tune 8va (reveal/skim/meadow/flyover), at pitch elsewhere
    v1 = [(G.t(2, 3), G.t(3), nm("E5"), 0.12, 0.55)]
    v1 += tune_part(G, range(3, 7), 12)
    v1 += tune_part(G, [7, 8], 0)
    v1 += tune_part(G, [9, 10], 0, 0.9)
    v1 += tune_part(G, [11, 12], 12, 0.75)
    v1 += tune_part(G, [13, 14], 12)
    v1 += tune_part(G, [15, 16], 0)
    t0, t1 = G.span(17, 1, 3.6)
    v1 += [(t0, t1, nm("A5"), 0.8, 0.22)]
    v1 = gap_all(v1)
    # vln1 divisi upper octave at the climax for brilliance
    v1b = gap_all(tune_part(G, [15, 16], 12, 0.55))
    buses["vln1"] = string_section("vln1", [v1, v1, v1, v1b], R("vln1"), att=0.22, rel=0.5, vib=1.1, fast=FAST)

    # vln2: forest S pad, tune at pitch (reveal/skim/meadow/flyover), 8vb (waterfall/peaks), S line (stream/finale)
    def pad_vel(lo, hi):
        return lambda bar, beat: lo if bar < 2 else hi
    v2 = chart_lines(G, [1, 2], 3, 0, vel_fn=lambda b, k: 0.16 if b == 1 else 0.24)
    v2 += tune_part(G, range(3, 7), 0, 0.92)
    v2 += tune_part(G, [7, 8], 0, 0.8)
    v2 += chart_lines(G, [9, 10], 3, 0, vel_fn=lambda b, k: 0.32 if b == 9 else 0.5)
    v2 += tune_part(G, [11, 12], 0, 0.8)
    v2 += tune_part(G, [13, 14], 0, 0.95)
    v2 += tune_part(G, [15, 16], -12, 0.95)
    v2 += chart_lines(G, [17, 18], 3, 0, vel_fn=lambda b, k: 0.5 if b == 17 else 0.3)
    v2 = gap_all(v2)
    buses["vln2"] = string_section("vln2", [v2], R("vln2"), att=0.3, rel=0.5, fast=FAST)

    # violas: A + S lines sustained (spiccato ostinato in skim / flyover / peaks is separate)
    sus_bars = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 17, 18]

    def vla_vel(bar, beat):
        return {1: .18, 2: .26, 3: .55, 4: .52, 7: .55, 8: .62, 9: .34, 10: .5, 11: .42, 12: .42, 17: .5, 18: .3}.get(bar, .5)
    va_a = gap_all(chart_lines(G, sus_bars, 2, 0, vel_fn=vla_vel))
    va_s = gap_all(chart_lines(G, [b for b in sus_bars if b not in (1, 2, 9, 10, 17, 18)], 3, 0, vel_fn=vla_vel))
    va_lines = [va_a, va_s] if va_s else [va_a]
    # moving inner voice in the reveal (viola line under the tune): D4 E4 F#4 | E4 ... D4
    buses["vla"] = string_section("vla", va_lines, R("vla"), att=0.35, rel=0.5, fast=FAST)

    # celli: bass (8va where below C2) + tenor; tune 8vb in the waterfall
    def vc_vel(bar, beat):
        return {1: .2, 2: .28, 3: .6, 4: .58, 5: .55, 6: .58, 9: .38, 10: .55, 11: .45, 12: .45, 13: .7, 14: .72,
                15: .85, 16: .88, 17: .55, 18: .32}.get(bar, .55)
    vc_b = chart_lines(G, [b for b in range(1, 19) if b not in (7, 8)], 0, 0, vel_fn=vc_vel, fit=(36, 60))
    vc_t = chart_lines(G, [b for b in range(3, 19) if b not in (5, 6, 7, 8, 15, 16)], 1, 0, vel_fn=vc_vel, fit=(43, 64))
    vc_m = tune_part(G, [7, 8], -12, 0.95)
    # flyover: celli counter-line climbing against the falling tune (contrary motion)
    vc_c = to_notes(G, [(13, 1, 1.5, "F3"), (13, 2.5, 0.5, "G3"), (13, 3, 1, "A3"), (13, 4, 1, "Bb3"),
                        (14, 1, 2, "C4"), (14, 3, 1, "D4"), (14, 4, 0.84, "E4")], lambda b, k: 0.8)
    vc_b, vc_t = gap_all(vc_b), gap_all(vc_t)
    vc_t = [n for n in vc_t if not (G.t(13) - 0.01 <= n[0] < G.t(15) - 0.01)] + vc_c
    buses["vc"] = string_section("vc", [vc_b + vc_m, sorted(vc_t + vc_m)], R("vc"), att=0.3, rel=0.55, fast=FAST)

    def cb_vel(bar, beat):
        return {3: .55, 4: .5, 5: .55, 6: .55, 7: .6, 8: .65, 9: .35, 10: .55, 11: .4, 12: .4, 13: .75, 14: .78,
                15: .66, 16: .66, 17: .55, 18: .3}.get(bar, .5)
    cb = gap_all(chart_lines(G, range(3, 19), 0, 0, vel_fn=cb_vel, fit=(28, 42)))
    buses["cb"] = string_section("cb", [cb], R("cb"), att=0.3, rel=0.6, vib=0.5, fast=FAST)

    # spiccato ostinato: 16ths, accents 3+3+2, in skim / flyover / peaks; celli 8ths
    log("spiccato")
    rs = R("spic")
    hits_va, hits_vc = [], []
    for bar, base in ((5, .5), (6, .55), (13, .56), (14, .6), (15, .6), (16, .64)):
        bl = G.beat_len(bar)
        for k in range(16):
            beat = 1 + k * 0.25
            c = chord_at(bar, beat)
            tones = sorted({(nm(x) - 55) % 12 + 55 for x in c[4].split()[1:]})   # viola range G3..F#4
            tones = tones + [tones[0] + 12]
            pat = [0, 2, 1, 0, 2, 1, 0, 3]
            mi = tones[pat[k % 8] % len(tones)]
            acc = k % 8 in (0, 3, 6)
            v = base * (1.0 if acc else 0.55)
            t = G.t(bar, beat)
            if any(abs(t - (h - bl * 0.25)) < 1e-3 for h in HITS + [G.t(7)]):
                continue   # 16th rest before the hits
            hits_va.append((t, mi, v))
            if k % 2 == 0:
                bmi = nm(c[4].split()[0])
                while bmi < 36:
                    bmi += 12
                hits_vc.append((t, bmi + (12 if k % 8 == 6 else 0), v * 0.9))
    buses["spic"] = spiccato("vla", hits_va, rs, nv=6) + 0.8 * spiccato("vc", hits_vc, rs, nv=5, dec=0.07)

    # ---------------------------------------------------------------- brass
    log("brass")
    hb = np.zeros((2, N))
    h_mel = tune_part(G, [3, 4], 0, 0.9) + tune_part(G, [5, 6], -12, 0.92) + tune_part(G, [7, 8], -12, 0.78) \
        + tune_part(G, [10], -12, 0.75) \
        + tune_part(G, [13, 14], -12, 1.0) + tune_part(G, [15, 16], -12, 1.0)
    h_mel = gap_all(h_mel)
    for k, pan in enumerate((-0.42, -0.34)):
        hb += brass_player("horn", h_mel, R(f"horn{k}"), pan, 1.0, growl=0.7, fast=FAST)

    def hn_vel(bar, beat):
        return {3: .5, 4: .5, 5: .45, 6: .5, 7: .55, 8: .62, 10: .55, 11: .3, 12: .32, 13: .7, 14: .75, 15: .9, 16: .95,
                17: .5, 18: .28}.get(bar, .4)
    hn_bars = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    h3 = gap_all(chart_lines(G, hn_bars, 2, 0, vel_fn=hn_vel, fit=(52, 70)))
    h4 = gap_all(chart_lines(G, hn_bars, 3, 0, vel_fn=hn_vel, fit=(55, 74)))
    hb += brass_player("horn", h3, R("horn3"), -0.26, 0.8, growl=0.5, fast=FAST)
    hb += brass_player("horn", h4, R("horn4"), -0.5, 0.8, growl=0.5, fast=FAST)
    # finale horn: the tune's last note at pitch, warm
    t0, t1 = G.span(17, 1, 3.6)
    hb += brass_player("horn", [(t0, t1, nm("A4"), 0.7, 0.2)], R("hornF"), -0.38, 0.9)
    buses["horns"] = brass_finish("horn", hb)

    tb = np.zeros((2, N))

    def tb_vel(bar, beat):
        return {5: .32, 6: .36, 7: .45, 8: .55, 10: .5, 13: .62, 14: .7, 15: 1.0, 16: 1.0, 17: .5, 18: .25}.get(bar, .3)
    tb_bars = [5, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18]
    for k, (vi, rng_, pan) in enumerate(((1, (45, 60), 0.28), (2, (50, 64), 0.38), (0, (40, 55), 0.46))):
        notes = gap_all(chart_lines(G, tb_bars, vi, 0, vel_fn=tb_vel, fit=rng_))
        tb += brass_player("tbn", notes, R(f"tbn{k}"), pan, 0.8, growl=0.9, fast=FAST)
    buses["tbn"] = brass_finish("tbn", tb)
    tu = gap_all(chart_lines(G, [13, 14, 15, 16, 17], 0, 0, vel_fn=tb_vel, fit=(29, 46)))
    buses["tuba"] = brass_finish("tuba", brass_player("tuba", tu, R("tuba"), 0.5, 0.9, growl=0.6, fast=FAST))

    # ---------------------------------------------------------------- choir
    log("choir")

    def ch_vel(bar, beat):
        return {11: .42, 12: .45, 13: .0, 14: .55, 15: 1.0, 16: 1.0, 17: .62, 18: .34}.get(bar, 0.0)
    ch_bars = [11, 12, 14, 15, 16, 17, 18]
    ranges = {"B": (43, 60), "T": (48, 66), "A": (55, 71), "S": (62, 79)}
    tt_o = lambda tt: np.clip((tt - G.t(17, 2)) / 3.0, 0, 1) * 0.9 + 0.35 * np.clip(1 - np.abs(tt - G.t(11, 3)) / 3.0, 0, 1)  # noqa: E731
    ch = np.zeros((2, N))
    for part, vi in (("B", 0), ("T", 1), ("A", 2), ("S", 3)):
        notes = chart_lines(G, ch_bars, vi, 0, vel_fn=ch_vel, fit=ranges[part])
        notes = [n for n in notes if n[3] > 0.01]
        if part == "S":   # sopranos sing the tune at the climax
            notes = [n for n in notes if n[0] < G.t(15) - 0.01 or n[0] >= G.t(17) - 0.01]
            notes += tune_part(G, [15, 16], 0, 1.0)
            t0, t1 = G.span(17, 1, 3.6)
            notes = [n for n in notes if not (t0 - 0.01 <= n[0] < t1)] + [(t0, t1, nm("A5") - 12, 0.6, 0.2)]
        notes = gap_all(sorted(notes))
        ch += choir_part(part, notes, R("choir" + part), tt_o, fast=FAST)
    buses["choir"] = ch

    # ---------------------------------------------------------------- piano / celesta / harp
    log("piano / celesta / harp")
    rp = R("piano")
    pb = np.zeros((2, N))

    def pn(t, mi, vel, dur, pedal=0.0):
        y = piano_note(rp, mi, vel, dur, pedal)
        add_mono(pb, y, int(t * SR), -0.05 + (mi - 64) / 60.0)
    # forest: open fifth, flowing 8th arpeggios, the theme's head a beat late
    pn(0.35, nm("D2"), 0.32, 3.6, 0.3)
    pn(0.37, nm("A2"), 0.26, 3.6, 0.3)
    for bar in (1, 2):
        for k in range(8):
            beat = 1 + 0.5 * k
            if bar == 1 and k == 0:
                continue
            c = chord_at(bar, beat)
            vs = [nm(x) for x in c[4].split()]
            b0 = vs[0] + (12 if vs[0] < 45 else 0)
            pat = [b0, vs[1], vs[2], vs[1], b0 + 12, vs[1], vs[2], vs[3]]
            mi = pat[k]
            v = 0.2 + 0.05 * (k % 4 == 0) + (0.06 if bar == 2 else 0)
            pn(G.t(bar, beat) + rp.normal(0, 0.006), mi, v, G.beat_len(bar) * 2.5, 0.4)
    for (bar, beat, dur, n) in FOREST_PIANO:
        t0, t1 = G.span(bar, beat, dur)
        pn(t0 + 0.01, nm(n), 0.46, t1 - t0 + 0.3, 0.2)
        pn(t0 + 0.02, nm(n) - 12, 0.16, t1 - t0 + 0.3, 0.2)
    # stream: piano doubles the tune two octaves up, very soft (music box glint)
    for (t0, t1, mi, a, b) in tune_part(G, [9], 12, 0.42, hair=False):
        pn(t0, mi, a, t1 - t0, 0.4)
    # finale: the theme's head returns on the piano
    pn(G.t(17, 3) - 0.02, nm("F2"), 0.3, 4.5, 0.0)
    pn(G.t(17, 3), nm("C3"), 0.22, 4.5, 0.0)
    for (bar, beat, dur, n) in FINALE_PIANO:
        t0, t1 = G.span(bar, beat, dur)
        pn(t0, nm(n), 0.44, max(t1 - t0, 0.6) + 0.5, 0.6)
    pn(G.t(18, 1), nm("A3"), 0.2, 2.6)
    pn(G.t(18, 1) + 0.01, nm("F3"), 0.18, 2.6)
    buses["piano"] = filt(chain(sos("highpass", 45, 2), peq(250, -2.0, 1.0), peq(2800, -1.5, 1.2)),
                          np.vstack([signal.oaconvolve(pb[c], body_ir(rp, 90, 3000, 28, 0.06))[:N] for c in range(2)]))

    rc = R("celesta")
    cel = np.zeros((2, N))
    for (bar, beat, n, v) in ((1, 3.5, "A5", .25), (1, 4.5, "F#6", .2), (2, 1.5, "D6", .24), (2, 2.5, "C#6", .2),
                              (2, 3.5, "E6", .22), (2, 4.5, "A5", .2)):
        add_mono(cel, celesta_note(rc, nm(n), v), int(G.t(bar, beat) * SR), 0.32)
    for (t0, t1, mi, a, b) in tune_part(G, [11, 12], 12, 0.36, hair=False):
        add_mono(cel, celesta_note(rc, mi, a), int(t0 * SR), 0.32)
    for (bar, beat, n, v) in ((17, 3, "C6", .2), (18, 1, "G5", .22), (18, 2.5, "F6", .16), (18, 3, "A5", .14)):
        add_mono(cel, celesta_note(rc, nm(n), v), int(G.t(bar, beat) * SR), 0.32)
    buses["celesta"] = filt(sos("highpass", 300, 2), cel)

    rh = R("harp")
    hp = np.zeros((2, N))

    def harp_arp(bar, beat, notes, step, vel):
        t = G.t(bar, beat)
        for k, mi in enumerate(notes):
            add_mono(hp, harp_note(rh, mi, vel * rh.uniform(0.85, 1.05)), int((t + k * step) * SR),
                     -0.48 + (mi - 60) / 120)
    # stream: flowing 8ths, chord tones climbing
    for bar in (9, 10):
        for k in range(8):
            beat = 1 + 0.5 * k
            c = chord_at(bar, beat)
            pcs = sorted({nm(x) % 12 for x in c[4].split()})
            pool = sorted(p + 12 * o for o in range(4, 7) for p in pcs if 55 <= p + 12 * o <= 84)
            idx = [0, 2, 4, 1, 3, 5, 2, 4][k] % len(pool)
            harp_arp(bar, beat, [pool[idx]], 0.0, 0.36 + 0.05 * (bar == 10))
    # meadow: gliss-like arpeggio up the F chord on the lift, then soft 8ths
    fpool = sorted(p + 12 * o for o in range(2, 7) for p in (5, 9, 0) if 41 <= p + 12 * o <= 89)
    harp_arp(11, 1, fpool, 0.055, 0.42)
    for bar in (11, 12):
        for k in range(1 if bar == 11 else 0, 8):
            if bar == 11 and k < 3:
                continue
            beat = 1 + 0.5 * k
            c = chord_at(bar, beat)
            pcs = sorted({nm(x) % 12 for x in c[4].split()})
            pool = sorted(p + 12 * o for o in range(4, 7) for p in pcs if 57 <= p + 12 * o <= 81)
            harp_arp(bar, beat, [pool[[0, 2, 1, 3, 2, 4, 3, 5][k] % len(pool)]], 0.0, 0.3)
    # finale: rolled F chord
    harp_arp(18, 1, [nm(x) for x in ("F2", "C3", "F3", "A3", "C4", "F4", "G4", "A4", "C5")], 0.07, 0.34)
    buses["harp"] = filt(chain(sos("highpass", 60, 2), peq(300, -1.5, 1.0)), hp)

    # ---------------------------------------------------------------- percussion
    log("percussion")
    rt = R("timp")
    tp = np.zeros((2, N))
    dr = np.zeros((2, N))
    cy = np.zeros((2, N))
    fx = np.zeros((2, N))

    def timp(t, n, v, dec=1.0):
        add_mono(tp, timp_hit(rt, nm(n), v, dec), int(t * SR), rt.uniform(-0.08, 0.08))

    def bd(t, v):
        add_mono(dr, bass_drum(rt, v), int(t * SR), 0.12)

    def swell(t_peak, L_pre, v, dec=2.6, pan_w=1.0):
        y = cymbal(rt, L_pre + dec * 3.0, L_pre, v, dec, "swell")
        add_st(cy, y, int((t_peak - L_pre) * SR))

    def crash(t, v, dec=2.4):
        y = cymbal(rt, dec * 3.2 + 0.2, 0.05, v, dec, "crash")
        add_st(cy, y, int((t - 0.05) * SR))

    def sub(t, v, f0=50, f1=33):
        add_mono(fx, sub_hit(v, f0, f1), int(t * SR), 0.0)

    # forest -> reveal: timpani roll on A, cymbal swell, suck-out, hit on D
    timp_roll(rt, tp, nm("A2"), G.t(2, 2.5), G.t(3) - 0.13, 0.04, 0.6, rate=13)
    swell(G.t(3), 2.6, 0.36, 1.7)
    timp(G.t(3), "D2", 0.95)
    bd(G.t(3), 0.45)
    sub(G.t(3), 0.45)
    timp(G.t(4, 1), "A2", 0.42)
    timp(G.t(4, 3), "D2", 0.3)
    # skim: low toms 3+3+2, timp downbeats, fill into the waterfall
    swell(G.t(5), 1.8, 0.22, 1.3)
    for bar in (5, 6):
        timp(G.t(bar), "G2" if bar == 5 else "D2", 0.65)
        for k in range(8):
            beat = 1 + 0.5 * k
            acc = k in (0, 3, 6)
            if bar == 6 and k >= 6:
                break
            add_mono(dr, tom(rt, "lo" if acc else "mid", 0.72 if acc else 0.27), int(G.t(bar, beat) * SR),
                     -0.15 if acc else 0.2)
    for k in range(7):   # 16th fill b6 beats 3.5-4.75, rest on the last 16th
        beat = 3.5 + 0.25 * k
        add_mono(dr, tom(rt, ["hi", "hi", "mid", "mid", "lo", "lo", "lo"][k], 0.3 + 0.08 * k), int(G.t(6, beat) * SR),
                 [0.3, 0.25, 0.0, -0.05, -0.2, -0.25, -0.2][k])
    # waterfall: hit, half-time toms, cymbal swell onto the Gmaj7 peak
    timp(G.t(7), "B2", 0.85)
    bd(G.t(7), 0.6)
    crash(G.t(7), 0.4, 1.6)
    sub(G.t(7), 0.4, 55, 36)
    for bar in (7, 8):
        for beat in (1, 3):
            if (bar, beat) != (7, 1):
                add_mono(dr, tom(rt, "lo", 0.55), int(G.t(bar, beat) * SR), -0.1)
            add_mono(dr, tom(rt, "mid", 0.2), int(G.t(bar, beat + 1.5) * SR), 0.15)
    swell(G.t(8), 1.8, 0.3, 1.5)
    timp(G.t(8), "G2", 0.75)
    timp(G.t(8, 3), "A2", 0.32)
    timp(G.t(9), "D2", 0.3)
    # stream -> meadow: timp roll on C, riser, reverse cymbal, suck-out, the lift
    timp_roll(rt, tp, nm("C2"), G.t(10, 1), G.t(11) - GAP - 0.02, 0.05, 0.42, rate=14)
    swell(G.t(11), 3.0, 0.5, 2.0)
    crash(G.t(11), 0.3, 1.8)
    add_st(fx, riser(rt, 2.6, 0.05), int((G.t(11) - GAP - 2.6) * SR))
    timp(G.t(11), "F2", 0.6)
    sub(G.t(11), 0.38, 48, 32)
    # meadow -> flyover: soft toms building, swell
    for k in range(10):
        beat = 3 + 0.25 * k
        if G.t(12, beat) > G.t(13) - GAP - 0.01:
            break
        add_mono(dr, tom(rt, "lo" if k % 3 == 0 else "mid", 0.12 + 0.06 * k), int(G.t(12, beat) * SR), 0.0)
    swell(G.t(13), 2.2, 0.4, 1.5)
    # flyover: big hit, driving toms, timp accents, fill, reverse cymbal into the peaks
    timp(G.t(13), "F2", 1.0)
    bd(G.t(13), 0.85)
    crash(G.t(13), 0.62, 1.8)
    sub(G.t(13), 0.65)
    for bar in (13, 14):
        for k in range(8):
            beat = 1 + 0.5 * k
            if bar == 14 and k >= 5:
                break
            acc = k in (0, 3, 6)
            if (bar, k) != (13, 0):
                add_mono(dr, tom(rt, "lo" if acc else "mid", 0.72 if acc else 0.3), int(G.t(bar, beat) * SR),
                         -0.15 if acc else 0.2)
        timp(G.t(bar, 3), "C2" if bar == 13 else "Bb2", 0.6)
    for k in range(6):
        beat = 3.5 + 0.25 * k
        if G.t(14, beat) > G.t(15) - GAP - 0.01:
            break
        add_mono(dr, tom(rt, ["mid", "mid", "lo", "lo", "lo", "lo"][k], 0.28 + 0.07 * k), int(G.t(14, beat) * SR), 0.0)
    swell(G.t(15), 2.5, 0.38, 1.6)
    add_st(fx, riser(rt, 2.2, 0.04, 500, 9000), int((G.t(15) - GAP - 2.2) * SR))
    # peaks: the climax
    timp(G.t(15), "Bb2", 1.0, 1.3)
    bd(G.t(15), 1.0)
    crash(G.t(15), 1.0, 2.2)
    sub(G.t(15), 0.9, 52, 30)
    for bar in (15, 16):
        for beat in (1, 2, 3, 4, 1.5, 2.5, 3.5, 4.5):
            if (bar, beat) == (15, 1):
                continue
            if bar == 16 and beat > 2.5:
                continue
            big = beat in (1, 3)
            add_mono(dr, tom(rt, "lo" if big else "mid", 0.8 if big else (0.38 if beat % 1 == 0 else 0.22)),
                     int(G.t(bar, beat) * SR), -0.12 if big else 0.18)
    timp(G.t(15, 3), "F2", 0.7)
    timp(G.t(16), "Db2", 0.85)
    bd(G.t(16), 0.6)
    timp_roll(rt, tp, nm("Eb2"), G.t(16, 3), G.t(17) - GAP - 0.02, 0.35, 0.95, rate=16)
    swell(G.t(17), 1.6, 0.42, 2.3)
    # finale: warm arrival, then it breathes out
    timp(G.t(17), "F2", 0.6, 1.4)
    bd(G.t(17), 0.35)
    sub(G.t(17), 0.4, 44, 32)
    timp(G.t(18), "F2", 0.18, 1.5)
    buses["timp"] = filt(chain(sos("highpass", 32, 2), peq(320, -2.0, 1.0)), tp)
    buses["drums"] = filt(chain(sos("highpass", 35, 2), peq(380, -2.5, 1.0), peq(2500, -2.0, 1.0)), dr)
    buses["cym"] = filt(chain(peq(3500, -2.0, 1.0), shelf(10000, 1.0, "high")), cy)
    buses["fx"] = filt(sos("highpass", 24, 2), fx)
    return buses


# =============================================================================
# mix
# =============================================================================
GROUP_DB = {"vln1": -8.5, "vln2": -5.0, "vla": -5.0, "vc": -4.0, "cb": -4.0, "spic": 0.5,
            "horns": -11.5, "tbn": -11.0, "tuba": -11.0, "choir": -6.5, "piano": -6.0, "celesta": -11.0,
            "harp": -10.0, "timp": -9.0, "drums": -7.5, "cym": -12.0, "fx": -9.0}
# the conductor: macro dynamics on the whole score (time s, dB)
CONDUCTOR = [(0.0, -3.5), (6.0, -3.5), (7.85, 0.0), (15.3, 0.0), (15.6, 0.7), (21.8, 0.7), (22.3, -1.0), (26.6, -1.0), (28.45, -2.5),
             (28.6, -1.0), (31.75, -1.0), (34.8, -0.5), (35.0, 0.0), (46.0, 0.0), (47.8, -1.0), (47.95, 1.5),
             (51.0, 1.5), (53.9, 1.0), (54.1, 0.0), (60.0, 0.0)]


def conductor_curve(points, n=N, hop=48):
    m = n // hop + 2
    tc = np.arange(m) * hop / SR
    d = np.interp(tc, [p[0] for p in points], [p[1] for p in points])
    d = gaussian_filter1d(d, 0.05 * SR / hop, mode="nearest")
    return dbg(np.interp(np.arange(n) / hop, np.arange(m), d))


# reverb sends (dB) to [hall, far hall, drum room]
SENDS = {"vln1": (-6, None, None), "vln2": (-6, None, None), "vla": (-6, None, None), "vc": (-7, None, None),
         "cb": (-10, None, None), "spic": (-9, None, None), "horns": (None, -3, None), "tbn": (None, -5, None),
         "tuba": (None, -8, None), "choir": (None, -2, None), "piano": (-9, None, None), "celesta": (-4, None, None),
         "harp": (-6, None, None), "timp": (None, None, -6), "drums": (None, None, -6), "cym": (-6, None, None),
         "fx": (None, None, None)}
SUSTAIN = ("vln1", "vln2", "vla", "vc", "cb", "horns", "tbn", "tuba", "choir")


def mix_music(G, buses, seed):
    log("mix + hall")
    r = rng_for(seed, "hall")
    irs = [make_ir(r, (2.6, 3.0, 2.3, 1.5), 0.020, 4.6, er=1.0),
           make_ir(r, (2.9, 3.4, 2.6, 1.6), 0.032, 5.0, er=1.3),
           make_ir(r, (1.6, 2.4, 1.9, 1.2), 0.012, 3.2, er=1.2)]
    dry = np.zeros((2, N))
    sends = [np.zeros((2, N)) for _ in irs]
    stems = {}
    for k, y in buses.items():
        g = dbg(GROUP_DB[k])
        y = y * g
        stems[k] = y
        dry += y
        for i, s in enumerate(SENDS[k]):
            if s is not None:
                sends[i] += y * dbg(s)
    sends[2] = filt(sos("highpass", 140, 2), sends[2])
    sends[0] = filt(sos("highpass", 90, 2), sends[0])
    sends[1] = filt(sos("highpass", 100, 2), sends[1])
    wet = np.zeros((2, N))
    for s, ir in zip(sends, irs):
        wet += convolve_st(s, ir)
    wet = filt(chain(peq(300, -2.5, 0.9), peq(3200, -2.0, 1.0)), wet)
    music = (dry + wet * 0.8) * conductor_curve(CONDUCTOR)
    # bus EQ: keep 200-400 Hz and 2-4 kHz clean, a little air
    music = filt(chain(sos("highpass", 24, 2), peq(290, -1.5, 0.9), peq(3100, -1.2, 1.1), shelf(11000, 1.0, "high")), music)
    music = bus_comp(music, thr_db=-20.0, ratio=1.8, att=0.03, rel=0.35)
    return music, stems, wet


def bus_comp(x, thr_db=-20.0, ratio=2.0, att=0.03, rel=0.3, hop=240):
    """RMS feed-forward compressor at a 5 ms control rate, soft knee."""
    m = (x[0] ** 2 + x[1] ** 2) * 0.5
    nb = len(m) // hop
    p = m[:nb * hop].reshape(nb, hop).mean(axis=1)
    lv = 10 * np.log10(p + 1e-12)
    a_a, a_r = math.exp(-hop / (att * SR)), math.exp(-hop / (rel * SR))
    s = np.empty(nb)
    acc = lv[0]
    for i in range(nb):
        v = lv[i]
        a = a_a if v > acc else a_r
        acc = a * acc + (1 - a) * v
        s[i] = acc
    knee = 6.0
    over = s - thr_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio), (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    g = dbg(-gr)
    gs = np.interp(np.arange(len(m)) / hop, np.arange(nb) + 0.5, g)
    return x * gs


# nature level relative to the music, per shot (dB, ungated K-weighted)
NATURE_REL = {"forest": -9.0, "reveal": -13.0, "skim": -12.0, "waterfall": -6.0, "stream": -7.0,
              "meadow": -12.0, "flyover": -14.0, "peaks": -15.0, "finale": -12.0}


def nature_gain(G, nature, music, lead=0.4, xf=1.2):
    nk, mk = AN.k_weight(nature), AN.k_weight(music)
    pts = []
    for s in G.shots:
        a, b = s["start"] + 0.3, s["end"] - 0.3
        ln = AN.segment_lufs(nk, a, b)
        lm = AN.segment_lufs(mk, a, b)
        g = lm + NATURE_REL[s["name"]] - ln
        pts.append((s["start"], s["end"], float(np.clip(g, -30, 12))))
    t = np.arange(N) / SR
    d = np.full(N, pts[0][2])
    for i in range(1, len(pts)):
        c = pts[i][0] - lead
        w = smoothstep((t - (c - xf / 2)) / xf)
        d = d + (pts[i][2] - pts[i - 1][2]) * w
    return dbg(d), pts


def limiter(x, ceiling_db=-1.15, look=0.003, rel=0.12):
    up = signal.resample_poly(x, 4, 1, axis=-1)
    pk = np.abs(up).max(axis=0)
    pk = pk[:x.shape[-1] * 4].reshape(-1, 4).max(axis=1)
    c = dbg(ceiling_db)
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    w = int(look * SR)
    g = minimum_filter1d(need, 2 * w + 1)
    # release smoothing (one-pole at 1 ms control rate, attack instant)
    hop = 48
    nb = len(g) // hop + 1
    gb = np.array([g[i * hop:(i + 1) * hop].min() for i in range(nb)]) if False else \
        np.minimum.reduceat(g, np.arange(0, len(g), hop))
    ar = math.exp(-hop / (rel * SR))
    out = np.empty_like(gb)
    acc = 1.0
    for i in range(len(gb)):
        v = gb[i]
        acc = v if v < acc else ar * acc + (1 - ar) * v
        out[i] = acc
    out = minimum_filter1d(out, 3)
    gs = np.interp(np.arange(len(g)) / hop, np.arange(len(out)), out)
    gs = np.minimum(gs, g)
    gs = gaussian_filter1d(gs, 0.0006 * SR)
    gs = np.minimum(gs, minimum_filter1d(need, 2 * w + 1) * 1.0005)
    return x * gs, float(20 * np.log10(gs.min()))


def master(mix, target=-14.0, ceiling=-1.15):
    g = target - AN.integrated_lufs(mix)
    y, gr = mix, 0.0
    for it in range(6):
        y, gr = limiter(mix * dbg(g), ceiling)
        lu = AN.integrated_lufs(y)
        if abs(lu - target) < 0.03:
            break
        g += target - lu
    return y, g, gr


def end_fade(y, t0=56.8, t1=59.9):
    t = np.arange(y.shape[-1]) / SR
    w = np.cos(0.5 * np.pi * np.clip((t - t0) / (t1 - t0), 0, 1)) ** 2
    w[t >= t1] = 0.0
    return y * w


def write_wav24(path, x, rng):
    d = (rng.random(x.shape) - rng.random(x.shape)) / (1 << 23)
    d[:, int(59.9 * SR):] = 0.0          # digital silence in the last 100 ms
    x = np.clip(x + d, -1, 1 - 1 / (1 << 23))
    v = np.round(x.T * (1 << 23)).astype(np.int32).reshape(-1)
    v[v > (1 << 23) - 1] = (1 << 23) - 1
    b = np.empty((len(v), 3), np.uint8)
    b[:, 0] = v & 255
    b[:, 1] = (v >> 8) & 255
    b[:, 2] = (v >> 16) & 255
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(b.tobytes())


def write_f32(path, x):
    from scipy.io import wavfile
    wavfile.write(path, SR, np.asarray(x.T, np.float32))


# =============================================================================
# analysis
# =============================================================================
def octave_bands(x, t0, t1):
    seg = np.mean(np.atleast_2d(x)[:, int(t0 * SR):int(t1 * SR)], axis=0)
    f, P = signal.welch(seg, SR, nperseg=8192)
    out = {}
    for fc in (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000):
        m = (f >= fc / 1.414) & (f < fc * 1.414)
        out[fc] = float(10 * np.log10(np.sum(P[m]) + 1e-20))
    return out


def chroma_check(G, x):
    """Per chart segment: correlation of the measured pitch-class profile with the chord's
    pitch classes, and the rank of the written chord among all 24 triads + itself."""
    mono = np.mean(x, axis=0)
    res = []
    names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    tmpl = {}
    for r in range(12):
        tmpl[names[r]] = {r, (r + 4) % 12, (r + 7) % 12}
        tmpl[names[r] + "m"] = {r, (r + 3) % 12, (r + 7) % 12}
    for (bar, beat, dur, sym, v) in CHART:
        t0, t1 = G.span(bar, beat, dur)
        a, b = int((t0 + 0.08) * SR), int((t1 - 0.02) * SR)
        if b - a < 4096:
            continue
        f, P = signal.welch(mono[a:b], SR, nperseg=min(16384, (b - a) // 2 * 2))
        m = (f > 40) & (f < 2000)
        midi = 69 + 12 * np.log2(f[m] / 440.0)
        pc = np.round(midi).astype(int) % 12
        ch = np.bincount(pc, weights=np.sqrt(P[m]), minlength=12)
        ch = ch / (ch.max() + 1e-12)
        want = {nm(x) % 12 for x in v.split()}
        for (tb_, tk, td, tn) in TUNE:
            u0, u1 = G.span(tb_, tk, td)
            if u0 < t1 - 0.05 and u1 > t0 + 0.05:
                want.add(nm(tn) % 12)
        t = np.array([1.0 if i in want else 0.0 for i in range(12)])
        corr = float(np.corrcoef(ch, t)[0, 1])
        scores = {k: float(np.corrcoef(ch, [1.0 if i in s else 0.0 for i in range(12)])[0, 1]) for k, s in tmpl.items()}
        scores["(written)"] = corr
        rank = sorted(scores.values(), reverse=True).index(corr) + 1
        top = [names[i] for i in np.argsort(-ch)[:4]]
        res.append(dict(bar=bar, beat=beat, chord=sym, corr=round(corr, 2), rank=rank, top_pcs=top))
    return res


def melody_check(G, x):
    """Median f0 of the dry vln2 bus over the middle of each tune note (vln2 plays the tune
    at a known octave in bars 3-6 and 11-14)."""
    mono = np.mean(x, axis=0)
    out = []
    for (bar, beat, dur, n) in TUNE:
        if bar not in (3, 4, 5, 6, 11, 12, 13, 14):
            continue
        t0, t1 = G.span(bar, beat, dur)
        a = int((t0 + 0.3 * (t1 - t0)) * SR)
        b = int((t1 - 0.2 * (t1 - t0)) * SR)
        seg = mono[a:b] * np.hanning(b - a)
        F = np.abs(np.fft.rfft(seg, 1 << 17))
        f = np.fft.rfftfreq(1 << 17, 1 / SR)
        want = mhz(nm(n))
        m = (f > want / 1.5) & (f < want * 1.5)
        # harmonic sum over 4 partials for each candidate
        cand = f[m]
        hs = np.zeros(len(cand))
        for h in (1, 2, 3, 4):
            hs += np.interp(cand * h, f, F)
        fe = cand[np.argmax(hs)]
        cents = 1200 * math.log2(fe / want)
        out.append(dict(bar=bar, beat=beat, note=n, cents=round(cents, 1)))
    return out


def analyze(G, out_dir, final, music, nature_g, stems, info):
    os.makedirs(out_dir, exist_ok=True)
    rep = {}
    xk = AN.k_weight(final)
    rep["frames"] = int(final.shape[-1])
    rep["integrated_lufs"] = round(AN.integrated_lufs(xk=xk), 2)
    rep["true_peak_dbtp"] = round(AN.true_peak_db(final), 2)
    rep["sample_peak_dbfs"] = round(AN.sample_peak_db(final), 2)
    rep["clipped"] = int(np.sum(np.abs(final) >= 0.9999))
    rep["dc"] = [float(f"{v:.2e}") for v in final.mean(axis=1)]
    rep["last100ms_dbfs"] = round(float(20 * np.log10(np.abs(final[:, -4800:]).max() + 1e-12)), 1)
    ts, ls = AN.short_term(xk)
    rep["max_short_term"] = round(float(ls.max()), 2)
    rep["max_short_term_t"] = round(float(ts[np.argmax(ls)]), 2)
    rep["clicks"] = AN.click_scan(final)
    rep["clicks_music_only"] = AN.click_scan(music)
    mono = final.mean(axis=0, keepdims=True)
    rep["mono_loss_db"] = round(rep["integrated_lufs"] - AN.integrated_lufs(np.vstack([mono[0], mono[0]])), 2)
    rep["lr_corr"] = round(float(np.corrcoef(final[0], final[1])[0, 1]), 3)
    mk, nk = AN.k_weight(music), AN.k_weight(nature_g)
    sh = []
    for s in G.shots:
        a, b = s["start"], s["end"]
        row = dict(shot=s["name"], mix=round(AN.segment_lufs(xk, a, b), 1), music=round(AN.segment_lufs(mk, a, b), 1),
                   nature=round(AN.segment_lufs(nk, a, b), 1))
        m = (ts >= a + 1.5) & (ts <= b)
        row["st_max"] = round(float(ls[m].max()), 1) if np.any(m) else None
        ob = octave_bands(music, a, b)
        ref = np.mean([ob[f] for f in (125, 500, 1000)])
        row["b250_rel"] = round(ob[250] - 0.5 * (ob[125] + ob[500]), 1)
        row["b2k4k_rel"] = round(0.5 * (ob[2000] + ob[4000]) - 0.5 * (ob[1000] + ob[8000]), 1)
        row["bands"] = {str(k): round(v - ref, 1) for k, v in ob.items()}
        sh.append(row)
    rep["shots"] = sh
    # momentary level around the cuts
    tm, lm = AN.momentary(xk)
    cuts = []
    for s in G.shots[1:]:
        c = s["start"]
        pre = lm[(tm > c - 0.6) & (tm < c - 0.1)]
        post = lm[(tm > c + 0.1) & (tm < c + 0.6)]
        cuts.append(dict(t=c, pre=round(float(pre.mean()), 1), post=round(float(post.mean()), 1)))
    rep["cuts"] = cuts
    rep["chroma"] = chroma_check(G, music)
    rep["chroma_mean_corr"] = round(float(np.mean([c["corr"] for c in rep["chroma"]])), 3)
    rep["chroma_rank1"] = int(sum(1 for c in rep["chroma"] if c["rank"] == 1))
    rep["chroma_n"] = len(rep["chroma"])
    if "vln2" in stems:
        mel = melody_check(G, stems["vln2"])
        rep["melody"] = mel
        rep["melody_within_25c"] = int(sum(1 for m in mel if abs(m["cents"]) <= 25))
        rep["melody_n"] = len(mel)
    st_tab = {}
    for k, y in stems.items():
        yk = AN.k_weight(y)
        st_tab[k] = [round(AN.segment_lufs(yk, s_["start"], s_["end"]), 1) for s_ in G.shots]
    rep["stem_lufs"] = st_tab
    rep.update(info)
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(rep, f, indent=1)
    # pictures
    AN.render_panel(final, "C_documentary master: spectrogram + loudness", G.shots, DUR).save(os.path.join(out_dir, "spec_mix.png"))
    panels = [("mix", final), ("music", music), ("nature (as mixed)", nature_g)]
    groups = {"strings": ["vln1", "vln2", "vla", "vc", "cb"], "spiccato": ["spic"], "brass": ["horns", "tbn", "tuba"],
              "choir": ["choir"], "piano/celesta/harp": ["piano", "celesta", "harp"], "timp/drums": ["timp", "drums", "fx"],
              "cymbals": ["cym"]}
    for g, ks in groups.items():
        panels.append((g, sum(stems[k] for k in ks if k in stems)))
    AN.overview_png(os.path.join(out_dir, "spec_overview.png"), panels, G.shots, DUR)
    for i, s in enumerate(G.shots):
        AN.zoom_png(os.path.join(out_dir, f"spec_{i}_{s['name']}.png"), final, max(0, s["start"] - 0.5),
                    min(DUR, s["end"] + 0.5), f"{s['name']}", width=1000, spec_h=300)
    return rep


def print_report(rep):
    print(f"frames {rep['frames']}  LUFS {rep['integrated_lufs']}  TP {rep['true_peak_dbtp']} dBTP  "
          f"SP {rep['sample_peak_dbfs']}  clipped {rep['clipped']}  DC {rep['dc']}  last100ms {rep['last100ms_dbfs']} dBFS")
    print(f"max short-term {rep['max_short_term']} at {rep['max_short_term_t']} s   mono loss {rep['mono_loss_db']} dB  "
          f"LR corr {rep['lr_corr']}  limiter max GR {rep.get('limiter_gr_db')} dB")
    for k in ("clicks", "clicks_music_only"):
        c = rep[k]
        print(f"{k}: max ratio {c['max_ratio']:.1f} hits {c['n_hits']} {c['hit_times'][:8]}  hf hits {c['hf_hits']} {c['hf_hit_times'][:8]}")
    print("shot        mix   music nature st_max  250rel 2-4k_rel   bands(31..16k rel)")
    for r in rep["shots"]:
        print(f"{r['shot']:<10} {r['mix']:6.1f} {r['music']:6.1f} {r['nature']:6.1f} {r['st_max']!s:>6}  {r['b250_rel']:5.1f} "
              f"{r['b2k4k_rel']:6.1f}   " + " ".join(f"{v:5.1f}" for v in r["bands"].values()))
    print("cuts (momentary pre -> post): " + "  ".join(f"{c['t']:g}: {c['pre']}->{c['post']}" for c in rep["cuts"]))
    print(f"chroma: mean corr {rep['chroma_mean_corr']}  written chord ranked 1st in {rep['chroma_rank1']}/{rep['chroma_n']}")
    for c in rep["chroma"]:
        if c["rank"] > 1 or c["corr"] < 0.5:
            print(f"   bar {c['bar']}.{c['beat']} {c['chord']}: corr {c['corr']} rank {c['rank']} top {c['top_pcs']}")
    print("stem LUFS per shot: " + " ".join(s_["shot"][:6] for s_ in rep["shots"]))
    for k, v in rep.get("stem_lufs", {}).items():
        print(f"   {k:<8}" + " ".join(f"{x:6.1f}" if x > -90 else "     -" for x in v))
    if "melody" in rep:
        print(f"melody (vln2 stem): {rep['melody_within_25c']}/{rep['melody_n']} notes within 25 cents; worst: " +
              ", ".join(f"{m['note']}@{m['bar']}.{m['beat']}:{m['cents']}" for m in sorted(rep["melody"], key=lambda m: -abs(m["cents"]))[:4]))


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="audio/film_shots.json")
    ap.add_argument("--nature", default="out/audio/nature/stems/nature.wav")
    ap.add_argument("--out", default="out/audio/soundtrack.wav")
    ap.add_argument("--work", default="out/audio/work_c")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--stems", action="store_true")
    a = ap.parse_args()
    shots = json.load(open(a.shots))
    G = Grid(shots)
    log(f"grid: {[round(G.t(b), 3) for b in range(1, 19)]}")
    buses = arrange_and_render(G, a.seed)
    music, stems, wet = mix_music(G, buses, a.seed)
    nat, sr = AN.read_any_wav(a.nature)
    assert sr == SR
    nat = nat[:, :N] if nat.shape[1] >= N else np.pad(nat, ((0, 0), (0, N - nat.shape[1])))
    ng, pts = nature_gain(G, nat, music)
    nat_g = nat * ng
    mix = filt(sos("highpass", 10, 1), end_fade(music + nat_g, 57.6, 59.9))   # DC guard
    final, g, gr = master(mix)
    final = end_fade(final, 59.0, 59.9)
    log(f"master gain {g:+.2f} dB, limiter max GR {gr:.2f} dB")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_wav24(a.out, final, rng_for(a.seed, "dither"))
    log(f"wrote {a.out}")
    os.makedirs(a.work, exist_ok=True)
    gl = dbg(g)
    music_m = end_fade(music, 57.6, 59.9) * gl
    if a.stems:
        sd = os.path.join(a.work, "stems")
        os.makedirs(sd, exist_ok=True)
        write_f32(os.path.join(sd, "music.wav"), music_m)
        write_f32(os.path.join(sd, "nature.wav"), nat_g * gl)
        for k, y in stems.items():
            write_f32(os.path.join(sd, f"{k}.wav"), y * gl)
    if a.analyze:
        log("analysis")
        info = dict(master_gain_db=round(g, 2), limiter_gr_db=round(gr, 2),
                    nature_gain_db={s["name"]: round(p[2], 1) for s, p in zip(shots, pts)})
        rep = analyze(G, os.path.join(a.work, "analysis"), final, music_m, nat_g * gl,
                      {k: v * gl for k, v in stems.items()}, info)
        print_report(rep)
    log("done")


if __name__ == "__main__":
    main()
