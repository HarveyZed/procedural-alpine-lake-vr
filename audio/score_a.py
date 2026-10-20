"""score_a.py - candidate A: refined epic orchestral score for the alpine-lake film.

Fully synthesised with numpy / scipy (no samples, recordings, soundfonts or downloads).
Strings, brass and choir are rendered player by player as legato lines (per-player pitch curve
with portamento, delayed vibrato, drift, onset scoop, bow-change dips, timing spread) and then
shaped by dynamics-driven brightness, high-Q body resonances, bow / breath noise and formants.
Basic DSP, the modal piano, celesta, drums and FX one-shots are shared with audio/music.py.

Tempo: a constant 120 BPM (beat = 0.5 s); every cut of the r5 list falls on a beat
(shot lengths 16 15 13 13 13 13 13 12 12 beats), so the 13-beat shots are 4+4+4+1 with the
extra beat used as a breath / fill before the next section.

Theme (D minor, 16 beats):  D4 A4 D5~ | C5. Bb4 A4~ | F4 C5 F5~ | E5. D5 E5~
Chord loop i-VI-III-VII (Dm Bb F C) = vi-IV-I-V in F, so the lift to F major at the meadow
re-reads the same loop; the climax closes with the heroic bVI-bVII-I (Db Eb F).

Usage (from the repository root):
  python audio/score_a.py [--shots audio/film_shots.json]
        [--nature out/audio/nature/stems/nature.wav] [--out out/audio/candidates/A_orchestral.wav]
        [--music-out out/audio/work_a/music_only.wav] [--stems out/audio/work_a/stems] [--analyze out/audio/work_a/an1]
        [--seed 5] [--fast]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analysis as an  # noqa: E402
import soundtrack as st  # noqa: E402
import music as mu  # noqa: E402  (DSP helpers, piano / celesta / drum one-shots, master)

SR = 48000
HOP = 48                       # control rate 1 kHz
BEAT = 0.5                     # 120 BPM
SHOT_BEATS = [16, 15, 13, 13, 13, 13, 13, 12, 12]
TARGET_LUFS = -14.0
CEILING_DBTP = -1.0
NATURE_REL = {"forest": -9.0, "reveal": -13.0, "skim": -12.0, "waterfall": -4.5, "stream": -6.5,
              "meadow": -11.0, "flyover": -14.0, "peaks": -15.0, "finale": -10.5}
MUSIC_FADE = (56.6, 59.85)
VS = 1.0                       # player-count scale (--fast: 0.5)
_T0 = time.time()

dbg, sos, filt, rng_for = mu.dbg, mu.sos, mu.filt, mu.rng_for


def log(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def m2f(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def ctrl_noise(rng, Lc, rate_hz):
    """unit-SD smooth noise at the 1 kHz control rate"""
    sig = max(0.5, 1000.0 / (2 * np.pi * rate_hz))
    pad = int(4 * sig) + 8
    v = gaussian_filter1d(rng.standard_normal(Lc + 2 * pad), sig)[pad:pad + Lc]
    return v / (v.std() + 1e-12)


def to_audio(c, L):
    return np.interp(np.arange(L) / HOP, np.arange(len(c)), c)


# =============================================================================
# timeline
# =============================================================================
class TL:
    def __init__(self, shots):
        self.shots = sorted(shots, key=lambda s: s["start"])
        self.T = float(max(s["end"] for s in self.shots))
        self.S = [float(s["start"]) for s in self.shots] + [self.T]
        nb = [int(round((s["end"] - s["start"]) / BEAT)) for s in self.shots]
        if nb != SHOT_BEATS:
            print(f"WARNING: score written for shot beats {SHOT_BEATS}, cut list gives {nb}")
        self.B = np.concatenate([[0], np.cumsum(nb)]).astype(float)
        self.n = int(round(self.T * SR))

    def t(self, b):
        b = float(b)
        if b <= self.B[-1]:
            return float(np.interp(b, self.B, self.S))
        return self.T + (b - self.B[-1]) * BEAT


# =============================================================================
# harmony: voicings (bass, mid 4 voices, high 3 voices) and the chord chart (beats)
# =============================================================================
VOI = {
    "Dm": (38, [50, 57, 62, 65], [69, 74, 77]),
    "Bb/D": (38, [50, 58, 62, 65], [70, 74, 77]),
    "Bb": (34, [50, 58, 62, 65], [70, 74, 77]),
    "F/C": (36, [48, 57, 60, 65], [69, 72, 77]),
    "F": (41, [48, 57, 60, 65], [69, 72, 77]),
    "C": (36, [52, 55, 60, 64], [67, 72, 76]),
    "Gm": (43, [50, 58, 62, 67], [70, 74, 79]),
    "A": (45, [52, 57, 61, 64], [69, 73, 76]),
    "Csus": (36, [48, 55, 60, 65], [67, 72, 77]),
    "C/E": (40, [52, 55, 60, 64], [67, 72, 76]),
    "Gm7": (43, [50, 58, 62, 65], [70, 74, 77]),
    "F/A": (45, [48, 57, 60, 65], [69, 72, 77]),
    "Db": (37, [49, 56, 61, 65], [68, 73, 77]),
    "Eb": (39, [51, 58, 63, 67], [70, 75, 79]),
    "Fadd9": (41, [48, 57, 60, 65], [67, 72, 77]),
    "Bb/F": (41, [50, 58, 62, 65], [70, 74, 77]),
}
CHORDS = [
    (0, 4, "Dm"), (4, 8, "Bb/D"), (8, 12, "F/C"), (12, 16, "C"),                       # forest
    (16, 20, "Dm"), (20, 24, "Bb"), (24, 28, "F"), (28, 31, "C"),                      # reveal
    (31, 35, "Dm"), (35, 39, "Bb"), (39, 41, "Gm"), (41, 44, "A"),                     # skim
    (44, 48, "Dm"), (48, 52, "F"), (52, 55, "C"), (55, 57, "Bb"),                      # waterfall
    (57, 61, "Gm"), (61, 65, "Bb"), (65, 67, "Csus"), (67, 70, "C"),                   # stream
    (70, 74, "F"), (74, 78, "C/E"), (78, 80, "Dm"), (80, 82, "Bb"), (82, 83, "C"),     # meadow (F major)
    (83, 87, "Dm"), (87, 91, "Bb"), (91, 93, "F"), (93, 96, "C"),                      # flyover
    (96, 100, "Bb"), (100, 102, "F/A"), (102, 104, "Gm7"), (104, 106, "Db"), (106, 108, "Eb"),  # peaks
    (108, 112, "Fadd9"), (112, 114, "Bb/F"), (114, 120, "Fadd9"),                      # finale
]
CHORD_PCS = {k: sorted({m % 12 for m in [v[0]] + v[1] + v[2]}) for k, v in VOI.items()}


def chord_at(b):
    for b0, b1, c in CHORDS:
        if b0 <= b < b1:
            return c
    return CHORDS[-1][2]


# melodies: (beat offset, midi, beats)
THEME_D = [(0, 62, 1), (1, 69, 1), (2, 74, 2), (4, 72, 1.5), (5.5, 70, .5), (6, 69, 2),
           (8, 65, 1), (9, 72, 1), (10, 77, 2), (12, 76, 1.5), (13.5, 74, .5), (14, 76, 2)]
THEME_F = [(0, 65, 1), (1, 72, 1), (2, 77, 2), (4, 76, 1.5), (5.5, 74, .5), (6, 72, 2),
           (8, 69, 1), (9, 76, 1), (10, 81, 2), (12, 79, .6)]
SKIM = [(0, 62, 1), (1, 69, 1), (2, 74, 2), (4, 72, 1.5), (5.5, 70, .5), (6, 69, 2), (8, 70, 1), (9, 69, 1), (10, 73, 3)]
WFALL = [(0, 62, 1), (1, 69, 1), (2, 74, 2), (4, 65, 1), (5, 72, 1), (6, 77, 2), (8, 76, 1.5), (9.5, 74, .5), (10, 72, 1), (11, 74, 2)]
DESCANT = [(0, 81, 4), (4, 84, 4), (8, 79, 3), (11, 77, 2)]
STREAM = [(0, 67, 1), (1, 74, 1), (2, 79, 2), (4, 70, 1), (5, 77, 1), (6, 82, 2), (8, 72, 1), (9, 79, 1), (10, 84, 2.3)]
FLY = [(0, 62, 1), (1, 69, 1), (2, 74, 2), (4, 72, 1.5), (5.5, 70, .5), (6, 69, 2), (8, 65, 1), (9, 72, 1), (10, 77, 1), (11, 76, 1.2)]
PEAK = [(0, 74, 1), (1, 77, 1), (2, 82, 2), (4, 81, 1.5), (5.5, 79, .5), (6, 77, 4), (10, 79, 2), (12, 81, 4)]
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def nname(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


# =============================================================================
# player-level line renderer (strings, brass, choir)
# =============================================================================
def player_ctrl(notes, tc, P, rng):
    Lc, K, tj = len(tc), len(notes), P["tjit"]
    on = []
    for k, nt in enumerate(notes):
        legin = k > 0 and notes[k - 1][1] >= nt[0] - P["lgap"]
        on.append((nt[0] + rng.normal(0, tj * (0.35 if legin else 1.0)), legin))
    m = np.full(Lc, float(notes[0][2]))
    env = np.zeros(Lc)
    since = np.zeros(Lc)
    scoop = np.zeros(Lc)
    fresh = []
    for k, nt in enumerate(notes):
        a, legin = on[k]
        legout = k < K - 1 and on[k + 1][1]
        b = on[k + 1][0] if legout else nt[1] + rng.normal(0, tj * 0.4)
        b = max(b, a + 0.03)
        v0, v1 = nt[3], nt[4]
        att = nt[5] if nt[5] is not None else P["att"]
        sel = tc >= a
        m[sel] = nt[2]
        since[sel] = tc[sel] - a
        x = np.clip((tc - a) / (b - a), 0, 1)
        lev = v0 + (v1 - v0) * x
        inside = (tc >= a) & (tc < b)
        if legin:
            w = 1.0 - P["dip"] * np.exp(-((tc - a) / 0.045) ** 2)
        else:
            at = max(0.004, att * rng.uniform(0.8, 1.25))
            w = np.sin(0.5 * np.pi * np.clip((tc - a) / at, 0, 1)) ** 2
            fresh.append((a, v0, at))
            if P["scoop"]:
                dt = tc - a
                scoop += np.where(dt >= 0, -P["scoop"] * rng.uniform(0.6, 1.3) * np.exp(-np.maximum(dt, 0) / P["stau"]), 0.0)
        e = np.where(inside, lev * w, 0.0)
        if not legout:
            r = P["rel"] * rng.uniform(0.85, 1.2)
            ib = int(np.clip(np.searchsorted(tc, b) - 1, 0, Lc - 1))
            lb = float(e[ib]) if inside[ib] else float(v1)
            tail = (tc >= b) & (tc < b + r)
            e[tail] = lb * np.cos(0.5 * np.pi * (tc[tail] - b) / r)
        env = np.maximum(env, e)
    return m, env, since, fresh, scoop


def render_line(n, notes, P, rng, players=None, want_dry=False):
    """notes: [(t0, t1, midi, v0, v1, att|None)], monophonic, sorted. Returns stereo (2, L), i0,
    mean player envelope (control rate), control times, smoothed midi of player 0, fresh onsets."""
    notes = sorted(notes, key=lambda z: z[0])
    T0 = max(0.0, notes[0][0] - 0.15)
    T1 = min(n / SR, max(z[1] for z in notes) + P["rel"] * 1.3 + 0.12)
    i0, i1 = int(T0 * SR), int(T1 * SR)
    L = i1 - i0
    Lc = L // HOP + 2
    tc = (i0 + np.arange(Lc) * HOP) / SR
    npl = max(1, int(round((players or P["players"]) * VS)))
    out = np.zeros((2, L))
    envs = np.zeros(Lc)
    dry = np.zeros(L) if want_dry else None
    mc0, fresh0 = None, []
    trem = P.get("trem")
    for p in range(npl):
        m, env, since, fresh, scoop = player_ctrl(notes, tc, P, rng)
        g = P["glide"] * rng.uniform(0.7, 1.35)
        ms = gaussian_filter1d(m, max(1.0, g * 1000.0 / 2.0), mode="nearest")
        if p == 0:
            mc0, fresh0 = ms, fresh
        env = gaussian_filter1d(env, P.get("esm", 10.0), mode="nearest")
        if trem:
            ta, tb_, rate = trem
            fr = rate * rng.uniform(0.85, 1.15) * (1 + 0.08 * ctrl_noise(rng, Lc, 1.5))
            ph = 2 * np.pi * np.cumsum(fr) / 1000.0 + rng.uniform(0, 6.283)
            win = np.clip((tc - ta) / 0.4, 0, 1) * np.clip((tb_ - tc) / 0.2, 0, 1)
            env = env * (1.0 - win * 0.5 * (1.0 - np.abs(np.sin(0.5 * ph)) ** 0.7))
        vd = rng.uniform(*P["vib"])
        vr = rng.uniform(*P["vrate"]) * (1 + 0.05 * ctrl_noise(rng, Lc, 0.4))
        vph = 2 * np.pi * np.cumsum(vr) / 1000.0 + rng.uniform(0, 6.283)
        vamt = vd * np.clip((since - P["vdel"]) / P["vramp"], 0, 1) * (0.55 + 0.45 * np.clip(env / 0.8, 0, 1.25))
        cents = rng.normal(0, P["det"]) + vamt * np.sin(vph) + P["drift"] * ctrl_noise(rng, Lc, 0.35) \
            + P["jit"] * ctrl_noise(rng, Lc, 9.0) + scoop
        ma = to_audio(ms + cents / 100.0, L)
        ea = to_audio(env * (1 + P["anoise"] * ctrl_noise(rng, Lc, 2.5)), L)
        y = mu.osc_saw(m2f(ma), rng) * ea
        d = int(rng.uniform(0, P["dly"]) * 1e-3 * SR)
        if d:
            y = np.concatenate([np.zeros(d), y[:-d]])
        gl, gr = mu.pan_gains(P["pan"] + rng.uniform(-P["spread"], P["spread"]))
        out[0] += gl * y
        out[1] += gr * y
        envs += env
        if want_dry:
            dry += y
    out /= math.sqrt(npl)
    envs /= npl
    return out, i0, envs, tc, mc0, fresh0, dry


def body_ir(rng, fmin, fmax, nm, length, qr, mix):
    """direct path + a dense set of high-Q damped modes (instrument body / soundboard)"""
    nn = int(length * SR)
    t = np.arange(nn) / SR
    ir = np.zeros(nn)
    for f in np.exp(rng.uniform(np.log(fmin), np.log(fmax), nm)):
        Q = rng.uniform(*qr)
        tau = Q / (np.pi * f)
        ir += rng.standard_normal() / tau * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.283))
    ir *= np.cos(0.5 * np.pi * np.clip((t - 0.75 * length) / (0.25 * length), 0, 1))
    ir *= mix / (np.sqrt(np.sum(ir ** 2)) + 1e-12)
    ir[0] += 1.0
    return ir / np.sqrt(np.sum(ir ** 2))


def conv2(x, irs):
    nn = x.shape[-1]
    return np.vstack([signal.oaconvolve(x[c], irs[c])[:nn] for c in range(2)])


class Bus:
    """full-length stereo bus + control-rate envelope + fresh onsets"""
    def __init__(self, n):
        self.y = np.zeros((2, n))
        self.e = np.zeros(n // HOP + 2)
        self.fresh = []

    def add(self, y, i0, envs=None, fresh=()):
        mu.add_stereo(self.y, y, i0)
        if envs is not None:
            j0 = int(round(i0 / HOP))
            j1 = min(len(self.e), j0 + len(envs))
            self.e[j0:j1] = np.maximum(self.e[j0:j1], envs[:j1 - j0])
        self.fresh += list(fresh)


# =============================================================================
# instruments
# =============================================================================
STR = {   # orchestral seating: vn1 left, vn2 left-centre, va centre-right, vc right, cb right
    "vn1": dict(players=10, pan=-0.55, spread=0.25, dly=5, det=4.0, tjit=0.020, vib=(10, 19), vrate=(5.2, 6.4), vdel=0.22,
                vramp=0.5, drift=2.5, jit=1.3, anoise=0.07, glide=0.075, scoop=12, stau=0.05, att=0.35, rel=0.5, lgap=0.03,
                dip=0.14, dark=2300, bright=7800, hp=190, fmin=260, fmax=7000, bow=0.018),
    "vn2": dict(players=8, pan=-0.22, spread=0.22, dly=5, det=4.5, tjit=0.022, vib=(9, 17), vrate=(5.1, 6.3), vdel=0.25,
                vramp=0.5, drift=2.5, jit=1.3, anoise=0.07, glide=0.075, scoop=12, stau=0.05, att=0.35, rel=0.5, lgap=0.03,
                dip=0.14, dark=2100, bright=7000, hp=170, fmin=250, fmax=6500, bow=0.018),
    "va": dict(players=6, pan=0.12, spread=0.2, dly=5, det=4.5, tjit=0.022, vib=(8, 15), vrate=(5.0, 6.1), vdel=0.25,
               vramp=0.5, drift=2.5, jit=1.3, anoise=0.07, glide=0.08, scoop=10, stau=0.05, att=0.4, rel=0.55, lgap=0.03,
               dip=0.14, dark=1700, bright=5500, hp=115, fmin=190, fmax=5000, bow=0.016),
    "vc": dict(players=6, pan=0.4, spread=0.2, dly=5, det=4.0, tjit=0.022, vib=(8, 14), vrate=(4.9, 5.9), vdel=0.25,
               vramp=0.5, drift=2.5, jit=1.2, anoise=0.07, glide=0.085, scoop=10, stau=0.05, att=0.4, rel=0.6, lgap=0.03,
               dip=0.14, dark=1300, bright=4200, hp=55, fmin=90, fmax=4000, bow=0.014),
    "cb": dict(players=4, pan=0.45, spread=0.12, dly=5, det=4.0, tjit=0.025, vib=(5, 10), vrate=(4.6, 5.5), vdel=0.3,
               vramp=0.6, drift=2.0, jit=1.0, anoise=0.06, glide=0.09, scoop=8, stau=0.06, att=0.45, rel=0.7, lgap=0.03,
               dip=0.12, dark=650, bright=2200, hp=30, fmin=45, fmax=2500, bow=0.01),
}
BRASS = {
    "hn": dict(players=4, pan=-0.28, spread=0.16, dly=10, det=3.5, tjit=0.012, vib=(0, 2.5), vrate=(4.5, 5.5), vdel=0.4,
               vramp=0.6, drift=2.0, jit=1.5, anoise=0.04, glide=0.035, scoop=35, stau=0.035, att=0.07, rel=0.32, lgap=0.03,
               dip=0.10, bright=0.6, drive=1.3, fmax=4800, growl=0.0, hp=60, breath=0.15,
               form=[(420, 3.0, 0.9), (1050, 1.5, 1.4), (2900, -4.0, 1.0)]),
    "tb": dict(players=3, pan=0.28, spread=0.14, dly=10, det=3.5, tjit=0.012, vib=(0, 2.0), vrate=(4.5, 5.5), vdel=0.4,
               vramp=0.6, drift=2.0, jit=1.5, anoise=0.04, glide=0.04, scoop=30, stau=0.035, att=0.08, rel=0.35, lgap=0.03,
               dip=0.10, bright=0.85, drive=1.9, fmax=7000, growl=0.12, hp=45, breath=0.18,
               form=[(520, 2.0, 1.0), (1250, 2.5, 1.4), (3300, -2.5, 1.0)]),
    "tu": dict(players=2, pan=0.12, spread=0.08, dly=6, det=3.0, tjit=0.015, vib=(0, 1.5), vrate=(4.5, 5.5), vdel=0.4,
               vramp=0.6, drift=1.5, jit=1.0, anoise=0.04, glide=0.05, scoop=20, stau=0.04, att=0.12, rel=0.45, lgap=0.03,
               dip=0.08, bright=0.5, drive=1.5, fmax=2400, growl=0.07, hp=24, breath=0.08,
               form=[(240, 2.0, 1.0)]),
}
CHOIR = dict(players=6, pan=0.0, spread=0.55, dly=8, det=6.0, tjit=0.035, vib=(16, 30), vrate=(4.6, 5.6), vdel=0.3,
             vramp=0.8, drift=5.0, jit=3.0, anoise=0.08, glide=0.09, scoop=25, stau=0.08, att=0.55, rel=0.9, lgap=0.03, dip=0.05)
FORM = dict(mu.FORMANTS)
FORM[("f", "u")] = [(325, 50, 0), (700, 60, -12), (2530, 170, -30), (3500, 180, -40), (4950, 200, -50)]
FORM[("m", "u")] = [(350, 40, 0), (600, 60, -20), (2400, 100, -32), (2675, 120, -28), (2950, 120, -36)]


def strings_section(n, lines, cfg, rng, want_dry=False):
    """lines: list of (notes, players, P-overrides). Section bus with dynamic brightness, body
    resonance, bow noise, bow-attack scratches and high-pass."""
    bus = Bus(n)
    dry = np.zeros(n) if want_dry else None
    for notes, pl, over in lines:
        if not notes:
            continue
        P = dict(cfg, **(over or {}))
        y, i0, envs, tc, mc, fresh, d = render_line(n, notes, P, rng, pl, want_dry)
        bus.add(y, i0, envs, fresh)
        if want_dry:
            dry[i0:i0 + len(d)] += d[:n - i0]
    x = bus.y
    b = np.clip((to_audio(bus.e, n) - 0.25) / 0.7, 0, 1) ** 1.2      # timbre follows the dynamics
    yd = filt(sos("lowpass", cfg["dark"], 4), x)
    yb = filt(sos("lowpass", cfg["bright"], 2), x)
    y = filt(sos("lowpass", 9500, 2), yd + b * (yb - yd))             # no saw buzz above the bridge hill
    y = conv2(y, [body_ir(rng, cfg["fmin"], cfg["fmax"], 90, 0.16, (14, 42), 1.1) for _ in range(2)])
    ea = to_audio(bus.e, n)
    nz = filt(sos("bandpass", [2200, 9000]), rng.standard_normal((2, n)))
    y += nz * ea * cfg["bow"] * (0.4 + 0.6 * b)
    bp = sos("bandpass", [1200, 6000])
    for (a, v, at) in bus.fresh:            # bow-attack scratch on quick attacks only
        if at > 0.2:
            continue
        L = int(0.05 * SR)
        tt = np.arange(L) / SR
        s = filt(bp, rng.standard_normal(L)) * np.exp(-tt / 0.012) * np.clip(tt / 0.002, 0, 1) * v * 0.05
        mu.add_mono(y, s, int(a * SR), cfg["pan"])
    y = filt(sos("highpass", cfg["hp"]), y)
    return y, bus.e, dry


def spicc_section(n, notes, cfg, rng, players=4):
    """short bowed strokes: detuned players, 2-5 ms attack, pitch settling, velocity-dependent
    decay and brightness, rosin bite; notes: (t, midi, vel)"""
    out = np.zeros((2, n))
    L = int(0.36 * SR)
    tt = np.arange(L) / SR
    bite = sos("bandpass", [1500, 7000])
    npl = max(2, int(round(players * VS)))
    lps = {}
    for (t, m, v) in notes:
        f0 = float(m2f(m))
        bn = filt(bite, rng.standard_normal(L)) * np.exp(-tt / 0.005) * np.clip(tt / 0.0015, 0, 1)
        acc = np.zeros(L + int(0.03 * SR))
        for p in range(npl):
            cents = rng.normal(0, 6) + 14.0 * np.exp(-tt / 0.012)
            y = mu.osc_saw(np.full(L, f0) * 2.0 ** (cents / 1200.0), rng)
            dec = (0.035 + 0.058 * v) * rng.uniform(0.8, 1.25)
            att = rng.uniform(0.002, 0.005)
            env = np.sin(0.5 * np.pi * np.clip(tt / att, 0, 1)) * (0.25 * np.exp(-tt / 0.012) + 0.75 * np.exp(-tt / dec))
            env *= np.cos(0.5 * np.pi * np.clip((tt - 0.26) / 0.1, 0, 1))
            y = y * env + bn * (0.10 + 0.22 * v) * rng.uniform(0.6, 1.2)
            d = min(1400, int(abs(rng.normal(0, 0.006)) * SR))
            acc[d:d + L] += y
        key = int(round((1400 + 6500 * v ** 1.4) / 250.0))
        if key not in lps:
            lps[key] = sos("lowpass", key * 250.0)
        acc = filt(lps[key], acc) * v / math.sqrt(npl)
        acc[-64:] *= np.linspace(1, 0, 64)
        mu.add_mono(out, acc, int(round(t * SR)), cfg["pan"] + rng.uniform(-cfg["spread"], cfg["spread"]))
    out = conv2(out, [body_ir(rng, cfg["fmin"], cfg["fmax"], 80, 0.12, (12, 36), 0.9) for _ in range(2)])
    return filt(sos("highpass", cfg["hp"] * 0.8), out)


def brass_section(n, lines, cfg, rng, want_dry=False):
    """lines: list of (notes, players). Per line: level-driven low-pass (brightness rises with
    loudness), level-driven saturation (rasp), growl, breath / lip noise on fresh attacks,
    bore / bell formant EQ."""
    out = np.zeros((2, n))
    dry = np.zeros(n) if want_dry else None
    bbr = sos("bandpass", [600, 3500])
    bbr2 = sos("bandpass", [700, 3000], 4)
    for notes, pl in lines:
        if not notes:
            continue
        y, i0, envs, tc, mc, fresh, d = render_line(n, notes, cfg, rng, pl, want_dry)
        L = y.shape[-1]
        a = np.clip(envs, 0, 1.2)
        fcc = np.clip(m2f(mc) * (1.0 + 8.0 * cfg["bright"] * a ** 1.6), 150.0, cfg["fmax"])
        fca = to_audio(fcc, L)
        y = np.vstack([mu.tv_filter(y[c], fca, "lp", q=0.75, stages=2) for c in range(2)])
        k = cfg["drive"]
        y = np.tanh(k * y * 1.3) / (k * 1.3) * (1 + 0.3 * k)
        t = (i0 + np.arange(L)) / SR
        aa = to_audio(a, L)
        if cfg["growl"] > 0:
            y *= 1.0 + cfg["growl"] * aa ** 3 * (0.6 * np.sin(2 * np.pi * 29.0 * t) + 0.4 * mu.slow_noise(rng, L, 25.0))
        y += filt(bbr, rng.standard_normal((2, L))) * aa * 0.012
        for (a0, v, at) in fresh:
            j0 = int((a0 - i0 / SR) * SR)
            if 0 <= j0 < L:
                bl = min(L - j0, int(0.2 * SR))
                tt = np.arange(bl) / SR
                y[:, j0:j0 + bl] += filt(bbr2, rng.standard_normal(bl)) * np.exp(-tt / 0.03) * np.clip(tt / 0.003, 0, 1) * cfg["breath"] * v
        y = filt(sos("lowpass", cfg["fmax"], 2), y)
        mu.add_stereo(out, mu.edge_fade(y, 3.0), i0)
        if want_dry:
            dry[i0:i0 + L] += d[:n - i0]
    for fc, g, q in cfg["form"]:
        out = filt(mu.peq_sos(fc, g, q), out)
    return filt(sos("highpass", cfg["hp"]), out), dry


def formant(x, part, vowel):
    y = np.zeros_like(x)
    for fc, bw, g in FORM[(part, vowel)]:
        b, a = mu.bpf_ba(fc, bw)
        y += signal.lfilter(b, a, x, axis=-1) * dbg(g)
    return y


def choir_section(n, lines, part, vowels, rng):
    """lines: list of (notes, players); vowels: {vowel: weight curve (audio rate)}"""
    bus = Bus(n)
    P = dict(CHOIR)
    for notes, pl in lines:
        if notes:
            y, i0, envs, tc, mc, fresh, _ = render_line(n, notes, P, rng, pl)
            bus.add(y, i0, envs)
    src = filt(sos("lowpass", 1000, 1), bus.y)          # glottal source tilt (-12 dB/oct overall)
    ea = to_audio(bus.e, n)
    nz = rng.standard_normal((2, n)) * ea * 0.05
    y = np.zeros((2, n))
    for v, w in vowels.items():
        y += w * (formant(src, part, v) + formant(nz, part, v))
    y += filt(sos("bandpass", [4000, 10000]), rng.standard_normal((2, n))) * ea * 0.003
    return filt(sos("highpass", 120 if part == "f" else 75), y)


def harp_note(rng, midi, vel, Ls=4.5):
    f0 = float(m2f(midi))
    L = int(Ls * SR)
    t = np.arange(L) / SR
    y = np.zeros(L)
    pp = rng.uniform(0.12, 0.2)
    for k in range(1, 30):
        fk = k * f0 * (1 + 0.00003 * k * k)
        if fk > 12000:
            break
        a = (abs(math.sin(math.pi * k * pp)) + 0.03) / k ** 1.1 * math.exp(-(fk / (2200 + 3500 * vel)) ** 1.2)
        tau = 3.2 / (1 + fk / 700.0)
        y += a * np.exp(-t / tau) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.283))
    y += filt(sos("lowpass", 3500), rng.standard_normal(L)) * np.exp(-t / 0.003) * 0.04
    y *= np.clip(t / 0.0015, 0, 1)
    return mu.tail_fade(y, 0.3) * vel


# =============================================================================
# arrangement
# =============================================================================
def arrange(tl, seed):
    rng = rng_for(seed, "arrange")
    tb = tl.t
    E = {}

    def N(b0, b1, m, v0, v1=None, att=None):
        return (tb(b0), tb(b1), int(m), float(v0), float(v0 if v1 is None else v1), att)

    def dyn(points):
        bs = np.array([p[0] for p in points], float)
        vs = np.array([p[1] for p in points], float)
        return lambda b: float(np.interp(b, bs, vs))

    def pad(sel, spans, D, att_first=0.35, off=0):
        """legato chord-tone line: sel(voicing) -> midi, per chord inside each span"""
        out = []
        for s0, s1, *sa in spans:
            first = True
            for b0, b1, c in CHORDS:
                a, b = max(b0, s0), min(b1, s1)
                if b - a <= 0.01:
                    continue
                v0, v1 = D(a), D(b)
                if max(v0, v1) < 0.02:
                    first = True
                    continue
                out.append(N(a, b, sel(VOI[c]) + off, v0, v1, (sa[0] if sa else att_first) if first else None))
                first = False
        return out

    def melody(start, mel, octave, vels, art=None, cut=None, att=0.12):
        """legato melodic line; art: set of note indices that are re-articulated (tongued /
        new bow) instead of slurred; cut: absolute beat where the last note stops"""
        out = []
        for k, (ob, m, d) in enumerate(mel):
            b0, b1 = start + ob, start + ob + d
            if cut is not None:
                b1 = min(b1, cut)
            t0, t1 = tb(b0), tb(b1)
            if art is not None and k + 1 < len(mel) and (k + 1) in art:
                t1 -= 0.045
            v = vels[k] if isinstance(vels, (list, tuple)) else vels
            v0, v1 = (v if isinstance(v, tuple) else (v, v))
            out.append((t0, t1, m + octave, v0, v1, att))
        return out

    SP = [(0, 15.4, 1.6), (16, 69.3, 0.09), (70, 82.6, 0.5), (83, 95.2, 0.08), (96, 119.6, 0.08)]   # (span, attack of its first note)
    LOW = dyn([(0, .34), (11, .38), (15.4, .6), (16, .95), (18, .72), (30, .8), (31, .55), (44, .55), (57, .55),
               (69.3, .9), (70, .25), (80, .3), (82.6, .42), (83, 1.0), (85, .85), (95.2, .92), (96, 1.0), (107, 1.0),
               (108, .85), (112, .62), (119.6, .3)])
    MID = dyn([(0, .3), (11, .34), (15.4, .56), (16, .92), (18, .7), (30, .78), (31, .5), (44, .55), (57, .45),
               (69.3, .92), (70, .24), (80, .3), (82.6, .42), (83, .95), (95.2, .9), (96, 1.0), (107, 1.0),
               (108, .85), (112, .62), (119.6, .3)])
    HIGH = dyn([(0, 0), (15.9, 0), (16, .85), (18, .66), (30, .75), (31, .42), (43.9, .45), (44, 0), (95.9, 0),
                (96, .9), (107, .95), (108, .8), (112, .6), (119.6, .28)])
    VN1P = dyn([(0, 0), (15.9, 0), (16, .82), (18, .64), (30, .74), (31, .4), (43.9, .42), (44, 0), (107.9, 0),
                (108, .72), (112, .55), (119.6, .25)])
    E["cb"] = [(pad(lambda v: v[0], SP, LOW, 1.6), 4, None)]
    E["vc"] = [(pad(lambda v: v[1][0], SP, LOW, 1.4), 6, None)]
    E["va"] = [(pad(lambda v: v[1][1], SP, MID, 1.4), 4, None), (pad(lambda v: v[1][2], SP, MID, 1.4), 4, None)]
    E["vn2"] = [(pad(lambda v: v[1][3], SP[1:], MID, 0.3), 5, None), (pad(lambda v: v[2][0], SP[1:], HIGH, 0.3), 5, None)]
    E["vn1"] = [(pad(lambda v: v[2][1], SP[1:], VN1P, 0.3), 5, None), (pad(lambda v: v[2][2], SP[1:], VN1P, 0.3), 5, None)]
    # forest: high violins flautando A5 shimmer (common tone of Dm, Bbmaj7, F, C6)
    E["vn1"].append(([N(1.5, 15.4, 81, .16, .3, 2.5)], 6, dict(vib=(4, 8), dark=1800, bow=0.03)))
    # stream: tremolo in the inner strings while the build grows
    trem = dict(trem=(tb(57), tb(69.3), 15.0))
    E["va"].append((pad(lambda v: v[1][2] + 12, [(57, 69.3, 0.6)], dyn([(57, .3), (69.3, .9)])), 4, trem))
    E["vn2"].append((pad(lambda v: v[2][1], [(57, 69.3, 0.6)], dyn([(57, .25), (69.3, .9)])), 5, trem))

    # ---- string melodies
    vm = []
    vm += melody(44, DESCANT, 0, [(.45, .55), (.55, .62), (.6, .5), (.5, .62)])
    vm += melody(57, STREAM, 0, [.5, .55, .62, .6, .66, .74, .74, .82, (.86, 1.0)], art={1, 2, 4, 5, 7, 8}, cut=69.3)
    vm += melody(70, THEME_F, 0, [.64, .68, .78, .74, .68, .7, .66, .74, (.82, .9), .78], cut=82.6, att=0.18)
    vm += melody(83, FLY, 12, [.8, .82, .9, .86, .8, .82, .8, .85, .92, .88], art={1, 2, 3, 6, 7, 8}, cut=95.2)
    vm += melody(96, PEAK, 0, [.95, .98, 1.0, 1.0, .95, (1.0, .92), (.95, 1.0), (1.0, .5)], art={1, 2, 3, 6})
    E["vn1"].append((vm, 10, dict(glide=0.09)))
    E["vn1_mel_ref"] = vm
    v2 = []
    v2 += melody(70, THEME_F, 0, [.48, .52, .6, .56, .52, .54, .5, .56, .64, .58], cut=82.6, att=0.2)
    v2 += melody(83, FLY, 0, [.8, .82, .9, .85, .8, .82, .8, .85, .9, .85], art={1, 2, 3, 6, 7, 8}, cut=95.2)
    v2 += melody(96, PEAK[:7], -12, [.85, .88, .92, .9, .85, (.92, .88), (.9, .95)], art={1, 2, 3, 6})
    E["vn2"].append((v2, 7, None))
    E["vc"].append((melody(44, WFALL, -12, [.62, .66, .72, .66, .7, .76, .74, .7, .7, (.72, .6)], art={1, 2, 4, 5, 6}), 5, None))

    # ---- spiccato ostinato (16ths, 3+3+2, cycle restarts on every bar line and every cut)
    acc = {0, 3, 6}
    SPD = dyn([(31, .72), (44, .8), (57, .7), (69.3, 1.0), (83, .8), (95.2, .84), (96, .88), (108, .9)])
    spans = [(31, 69.3), (83, 95.2), (96, 107.75)]
    bars = []
    for i in range(len(tl.B) - 1):
        b = tl.B[i]
        while b < tl.B[i + 1] - 1e-6:
            bars.append(b)
            b += 4
    E["sp_vc"], E["sp_cb"], E["sp_va"], E["sp_vn"] = [], [], [], []
    for s0, s1 in spans:
        for bb in bars:
            for k in range(16):
                B = bb + k / 4.0
                if B < s0 or B >= s1 or B >= tl.B[np.searchsorted(tl.B, bb, side="right")]:
                    continue
                c = VOI[chord_at(B)]
                i = k % 8
                v = (1.0 if i in acc else 0.62) * SPD(B) * rng.uniform(0.92, 1.05)
                t = tb(B) + rng.normal(0, 0.004)
                E["sp_vc"].append((t, c[0] + 12 + [0, 0, 12, 0, 0, 12, 0, 12][i], v))
                if B >= 44 and i in acc:
                    E["sp_cb"].append((t, c[0], v))
                if B >= 35:
                    E["sp_va"].append((t + rng.normal(0, 0.003), [c[1][1], c[1][2], c[1][3]][i % 3] + (12 if c[1][1] < 55 else 0), v * 0.9))
                if 44 <= B < 69.3 or B >= 96:
                    E["sp_vn"].append((t + rng.normal(0, 0.003), [c[2][0], c[2][1], c[2][2]][(i + 1) % 3], v * 0.8))

    # ---- brass
    hn = []
    hn.append((melody(16, THEME_D, 0, [.66, .7, .78, .74, .7, .72, .68, .74, .84, .8, .76, (.8, .7)], art={1, 2, 3, 6, 7, 8, 9},
                      cut=31), 4))
    hn.append((melody(31, SKIM, 0, [.58, .6, .66, .62, .58, .6, .6, .58, (.62, .72)], art={1, 2, 3, 6, 7, 8}), 4))
    hn.append((melody(44, WFALL, 0, [.62, .66, .72, .66, .7, .76, .74, .7, .7, (.72, .6)], art={1, 2, 3, 4, 5, 6, 9}), 4))
    hn.append((melody(57, STREAM, -12, [.5, .55, .6, .62, .66, .72, .76, .82, (.86, 1.0)], art={1, 2, 3, 4, 5, 6, 7, 8},
                      cut=69.3), 4))
    hn.append((pad(lambda v: v[1][2], [(70, 82.6, 0.6)], dyn([(70, .15), (80, .2), (82.6, .26)])), 2))
    hn.append((pad(lambda v: v[1][3], [(70, 82.6, 0.6)], dyn([(70, .14), (80, .19), (82.6, .25)])), 2))
    hn.append((melody(83, FLY, 0, [.86, .88, .92, .9, .86, .88, .86, .9, .94, .9], art={1, 2, 3, 6, 7, 8}, cut=95.2), 6))
    hn.append((melody(96, PEAK, -12, [.97, 1.0, 1.0, 1.0, .97, (1.0, .95), (.97, 1.0), (.95, .45)], art={1, 2, 3, 6}), 6))
    hn.append((pad(lambda v: v[1][2], [(110, 119.4, 0.8)], dyn([(110, .27), (116, .22), (119.4, .08)])), 2))
    hn.append((pad(lambda v: v[1][3], [(110, 119.4, 0.8)], dyn([(110, .25), (116, .2), (119.4, .08)])), 2))
    E["hn"] = hn
    E["hn_ref"] = [z for ln, _ in hn[:4] + hn[6:8] for z in ln]
    tb_ = []
    TBD = dyn([(16, .75), (17, .5), (30, .62), (31, .62), (57, .45), (69.3, .9), (96, 1.0), (107, 1.0), (108, .55), (119, .12)])
    for j in range(3):
        tb_.append((pad(lambda v, j=j: v[1][j], [(16, 31, 0.08), (96, 108, 0.06), (108, 119, 0.5)], TBD), 1))
    tb_.append((melody(57, STREAM, -12, [.45, .5, .55, .58, .62, .68, .72, .78, (.82, .95)], art={1, 2, 3, 4, 5, 6, 7, 8},
                       cut=69.3), 3))
    tb_.append((melody(83, FLY, -12, [.92, .95, 1.0, .96, .92, .94, .92, .95, 1.0, .95], art={1, 2, 3, 6, 7, 8}, cut=95.2), 3))
    E["tb"] = tb_
    TUD = dyn([(16, .9), (17, .65), (31, .6), (61, .5), (69.3, .9), (83, 1.0), (95.2, .9), (96, 1.0), (108, .7), (117, .2)])
    E["tu"] = [(pad(lambda v: v[0] + (12 if v[0] < 36 else 0), [(16, 31, 0.08), (61, 69.3, 0.4), (83, 95.2, 0.06),
                                                              (96, 108, 0.06), (108, 117, 0.4)], TUD), 2)]

    # ---- choir (S A T B from the voicings; the sopranos take the melody at the climax)
    CD = dyn([(20, .4), (31, .72), (60.9, 0), (61, .3), (69.3, .85), (70, .17), (82.6, .22), (83, .8), (95.2, .85),
              (96, 1.0), (107.5, 1.0), (108, .7), (119.4, .2)])
    csp = [(20, 31, 1.2), (61, 69.3, 0.8), (70, 82.6, 0.8), (83, 95.2, 0.15), (96, 108, 0.12), (108, 119.4, 0.3)]
    E["ch_S"] = [(pad(lambda v: v[2][1], [s for s in csp if s[0] != 96], CD), 6)]
    E["ch_S"].append((melody(96, PEAK[:7], 0, [.95, .98, 1.0, 1.0, .95, (1.0, .92), (.95, 1.0)], art={1, 2, 3, 6}, att=0.1), 6))
    E["ch_A"] = [(pad(lambda v: v[1][3], csp, CD), 6)]
    E["ch_T"] = [(pad(lambda v: v[1][1], csp, CD), 6)]
    E["ch_B"] = [(pad(lambda v: v[1][0], csp, CD), 6)]

    # ---- piano: theme in the forest, head-motif echo in the finale (bookends)
    pv = [.40, .43, .50, .46, .40, .42, .39, .44, .52, .48, .41, .46]
    E["piano"], E["piano_ref"] = [], []
    for k, (ob, m, d) in enumerate(THEME_D):
        t0 = tb(ob) + 0.12 + rng.normal(0, 0.01)
        t1 = min(tb(ob + d), tb(15.4))
        E["piano"].append((t0, m, pv[k], t1 - t0 + 0.15))
        E["piano_ref"].append((t0, t1, m))
    for b, ms, v in ((0, (38, 45), .22), (2, (57,), .16), (4, (38, 46), .22), (6, (58,), .16), (8, (36, 45), .22), (10, (57,), .16),
                     (12, (36, 43), .22), (14, (55,), .16)):
        for m in ms:
            E["piano"].append((tb(b) + 0.12 + rng.normal(0, 0.008), m, v + 0.03 * rng.random(), tb(min(b + 2, 15.4)) - tb(b) + 0.3))
    for b, m, v, d in ((112, 65, .6, 8), (113, 72, .63, 7), (114, 77, .68, 6), (112, 41, .26, 8), (112, 46, .22, 2), (114, 48, .26, 6)):
        E["piano"].append((tb(b) + rng.normal(0, 0.01), m, v, tb(b + d) - tb(b)))
        if m >= 60:
            E["piano_ref"].append((tb(b), tb(b + 1), m))
    # celesta: the F-major theme an octave up in the meadow, a two-note sparkle at the end
    E["cel"] = [(tb(70 + ob) + rng.normal(0, 0.006), m + 12, 0.32) for ob, m, d in THEME_F]
    E["cel"] += [(tb(115), 84, 0.26), (tb(116), 89, 0.24)]
    # harp: rising/falling 8th arpeggios in the meadow, a golden roll at the finale arrival
    E["harp"] = []
    for k in range(int((82.5 - 70) * 2)):
        B = 70 + k * 0.5
        pcs = CHORD_PCS[chord_at(B)]
        pool = [m for m in range(55, 88) if m % 12 in pcs]
        idx = k % 12
        j = idx if idx <= 6 else 12 - idx
        E["harp"].append((tb(B) + rng.normal(0, 0.006), pool[min(len(pool) - 1, 1 + j)], 0.34 + 0.08 * (k % 4 == 0)))
    for k, m in enumerate([53, 57, 60, 65, 69, 72, 77, 81, 84, 89]):
        E["harp"].append((tb(108) + 0.045 * k, m, 0.4 + 0.02 * k))

    # ---- drums
    E["taiko"] = []
    LO, MI, HI = 52.0, 92.0, 148.0

    def tk(b, f, v, size, pan_sd=0.3, jit=0.006):
        E["taiko"].append((tb(b) + rng.normal(0, jit), f * rng.uniform(0.97, 1.03), float(np.clip(v * rng.uniform(0.93, 1.05), 0.05, 1.2)),
                           size, float(np.clip(rng.normal(0, pan_sd), -0.8, 0.8))))

    def bar(b0, lo, mi, hi, vlo=.8, vmi=.45, vhi=.5, nb=4):
        for k in lo:
            if k < nb * 4:
                tk(b0 + k / 4, LO, vlo * (1.0 if k % 8 == 0 else 0.8), 1.25)
        for k in mi:
            if k < nb * 4:
                tk(b0 + k / 4, MI, vmi * (1.0 if k in (4, 12) else 0.75), 0.8, 0.45, 0.005)
        for k in hi:
            if k < nb * 4:
                tk(b0 + k / 4, HI, vhi, 0.55, 0.5, 0.004)

    def hitpoint(b, big=1.0):
        tk(b, LO, 1.0 * big, 1.6, 0.05)
        tk(b, LO * 0.94, 0.9 * big, 1.5, 0.05)
        tk(b, MI, 0.75 * big, 1.0, 0.3)

    hitpoint(16)
    for b in (20, 24, 28):
        tk(b, LO, 0.55, 1.4, 0.1)
    tk(26.5, LO, 0.4, 1.3, 0.1)
    for k, v in zip(range(4), (.3, .4, .52, .66)):
        tk(30 + k / 4, MI, v, 0.8, 0.4)
    G1 = [0, 6, 10]
    G2 = [0, 3, 6, 10]
    G3 = [0, 3, 6, 8, 11, 14]
    bar(31, G1, [4, 12], [], .78, .42)
    bar(35, G2, [4, 12], [], .8, .44)
    bar(39, G2, [4, 12, 14], [], .82, .46)
    bar(43, [], [0, 1, 2, 3], [], vmi=.55, nb=1)
    for b0 in (44, 48, 52):
        bar(b0, G3, [4, 12, 7, 15], [], .85, .5)
    bar(56, [0], [1, 2, 3], [], .8, .55, nb=1)
    bar(57, G3, [4, 12], [], .82, .5)
    bar(61, G3, [4, 12, 14, 15], [], .86, .55)
    for k in range(16):
        tk(65 + k / 4, MI, 0.4 + 0.5 * k / 15, 0.8, 0.4, 0.004)
        if k % 2 == 0:
            tk(65 + k / 4, LO, 0.55 + 0.4 * k / 15, 1.2, 0.2)
    tk(69, LO, 1.0, 1.5, 0.05)
    tk(69, HI, 0.8, 0.6, 0.3)
    tk(70, LO, 0.32, 1.7, 0.05)                       # the lift: one soft, deep stroke
    for b, v in ((78, .3), (80, .36), (81, .4), (81.5, .3)):
        tk(b, LO, v, 1.4, 0.1)
    for k in range(8):
        B = 81 + k / 4 if k < 4 else 82 + (k - 4) / 4
        tk(B, MI if k < 6 else HI, 0.3 + 0.6 * k / 7, 0.8, 0.4, 0.004)
    hitpoint(83)
    for b0, mi in ((83, list(range(16))), (87, list(range(16))), (91, list(range(16)))):
        bar(b0, G3, mi, [12, 13, 14, 15] if b0 == 91 else [], .86, .46, .6)
    tk(95, LO, 1.0, 1.5, 0.05)
    tk(95, MI, 0.8, 1.0, 0.3)
    hitpoint(96, 1.25)
    bar(96, G3, list(range(16)), [], 1.0, .55)
    bar(100, G3, list(range(16)), [14, 15], 1.0, .58, .65)
    for k in range(16):
        tk(104 + k / 4, MI, 0.5 + 0.45 * k / 15, 0.85, 0.4, 0.004)
        if k in (0, 3, 6, 8, 11, 14):
            tk(104 + k / 4, LO, 0.85 + 0.15 * (k >= 8), 1.3, 0.2)
        if k >= 12:
            tk(104 + k / 4, HI, 0.6 + 0.1 * (k - 12), 0.55, 0.4, 0.003)
    tk(108, LO, 0.5, 1.8, 0.05)                        # finale: a soft, round arrival

    # timpani (t, midi, vel, decay) incl. rolls
    E["timp"] = [(tb(b) + rng.normal(0, 0.004), m, v, 2.4) for b, m, v in
                 ((16, 38, 1.0), (31, 38, .45), (44, 38, .6), (57, 43, .55), (83, 38, 1.0), (96, 46, 1.0), (100, 45, .7),
                  (102, 43, .7), (104, 37, .95), (106, 39, 1.0), (108, 41, .55))]
    for a, b, m, v0, v1 in ((12, 15.3, 38, .05, .5), (65, 69, 48, .1, .8), (91, 95, 41, .1, .7), (108.3, 113, 41, .3, .04)):
        t, ta, tbb = tb(a), tb(a), tb(b)
        while t < tbb - 0.02:
            x = (t - ta) / (tbb - ta)
            E["timp"].append((t + rng.normal(0, 0.003), m, (v0 + (v1 - v0) * x ** 1.5) * rng.uniform(0.85, 1.1) * 0.55, 1.2))
            t += 1.0 / (13.0 + 3.0 * x) * rng.uniform(0.92, 1.08)
    # fx
    S = lambda b: tb(b)  # noqa: E731
    E["boom"] = [(S(16), 36.7, 1.2, 1.7), (S(44), 36.7, 0.35, 0.9), (S(70), 43.7, 0.4, 2.0), (S(83), 36.7, 1.3, 1.7),
                 (S(96), 29.1 * 1.25, 1.65, 2.0), (S(108), 43.7, 0.4, 1.6)]
    E["crash"] = [(S(16), 1.0), (S(44), 0.35), (S(70), 0.22), (S(83), 1.15), (S(96), 1.3), (S(108), 0.4)]
    E["revcym"] = [(S(83), 2.2, 0.7), (S(108), 2.6, 0.75), (S(44), 1.6, 0.35)]
    E["riser"] = [(S(15.4), 3.0, 0.35, 50), (S(69.3), 3.5, 0.42, 48), (S(95.2), 3.0, 0.42, 46)]
    E["braam"] = [(S(16), (26, 38, 45), 1.1), (S(83), (26, 38, 45), 1.25), (S(96), (22, 34, 41), 1.35)]
    E["revswell"] = [(S(15.4), (50, 57, 62, 65, 69), 2.6, 0.5), (S(69.3), (53, 60, 65, 69, 72), 2.8, 0.5)]
    return E


# =============================================================================
# render
# =============================================================================
GDB = {"strings": -0.5, "str_mel": -3.5, "spicc": 11.5, "horns": 10.0, "lowbrass": 4.0, "choir": 16.0, "piano": 12.0,
       "harp_cel": 4.5, "taiko": 6.5, "timp": -5.0, "boom_braam": -1.5, "fx": -13.0}
# global arc (time s, dB): lighter-but-lifted meadow, flyover held back a little so the
# climax after the 0.4 s silence is clearly the biggest moment
ARC = [(0, 0.0), (15.42, 0.0), (15.56, -2.2), (22.0, -1.8), (28.5, -1.5), (34.6, 0.6), (34.9, 0.6), (35.0, 1.5), (41.4, 1.5), (41.5, -0.6), (47.9, -0.9), (48.0, 1.8), (53.9, 1.8), (54.0, 0.8),
       (60.0, 0.8)]
SEND = {"strings": 0.42, "str_mel": 0.45, "spicc": 0.28, "horns": 0.55, "lowbrass": 0.32, "choir": 0.58, "piano": 0.42,
        "harp_cel": 0.55, "taiko": 0.22, "timp": 0.3, "boom_braam": 0.10, "fx": 0.25}


def render(tl, seed):
    n = tl.n
    E = arrange(tl, seed)
    G, mel = {}, {}

    log("strings (legato)")
    pads = np.zeros((2, n))
    melb = np.zeros((2, n))
    mel["violins"] = np.zeros(n)
    for sec in ("cb", "vc", "va", "vn2", "vn1"):
        r = rng_for(seed, "str_" + sec)
        y, e, d = strings_section(n, E[sec], STR[sec], r, want_dry=(sec == "vn1"))
        gain = {"cb": 0.0, "vc": -0.5, "va": -1.5, "vn2": -1.5, "vn1": -1.0}[sec]
        pads += y * dbg(gain)
        if sec == "vn1":
            # dry reference of the melody line only (re-render cheaply with 1 player)
            P = dict(STR["vn1"], glide=0.09)
            yy, i0, *_ = render_line(n, E["vn1_mel_ref"], P, rng_for(seed, "ref_vn"), 1)
            mel["violins"][i0:i0 + yy.shape[-1]] += yy.mean(axis=0)[:n - i0]
        log(f"  {sec} done")
    G["strings"] = pads
    log("spiccato")
    r = rng_for(seed, "spicc")
    sp = spicc_section(n, E["sp_vc"], STR["vc"], r, 4) + 0.8 * spicc_section(n, E["sp_cb"], STR["cb"], r, 3) \
        + 0.7 * spicc_section(n, E["sp_va"], STR["va"], r, 4) + 0.55 * spicc_section(n, E["sp_vn"], STR["vn2"], r, 4)
    G["spicc"] = sp

    log("brass")
    r = rng_for(seed, "brass")
    hn, hdry = brass_section(n, E["hn"], BRASS["hn"], r, want_dry=True)
    G["horns"] = hn
    yy = np.zeros(n)
    P = dict(BRASS["hn"])
    for ln, _ in E["hn"][:4] + E["hn"][6:8]:
        y1, i0, *_ = render_line(n, ln, P, rng_for(seed, "ref_hn"), 1)
        yy[i0:i0 + y1.shape[-1]] += y1.mean(axis=0)[:n - i0]
    mel["horn"] = yy
    tbb, _ = brass_section(n, E["tb"], BRASS["tb"], r)
    tub, _ = brass_section(n, E["tu"], BRASS["tu"], r)
    G["lowbrass"] = tbb + tub * dbg(-2.0)

    log("choir")
    r = rng_for(seed, "choir")
    t = np.arange(n) / SR
    wu = np.clip((t - (tl.S[5] - 0.6)) / 0.6, 0, 1) * np.clip(((tl.S[6] - 0.2) - t) / 0.4, 0, 1)
    wo = np.clip((t - (tl.S[8] + 0.5)) / 2.0, 0, 1) * 0.85
    wa = np.clip(1.0 - wu - wo, 0, 1)
    vow = {"a": wa, "o": np.clip(wo + 0.8 * wu, 0, 1), "u": 0.2 * wu}
    chf = choir_section(n, E["ch_S"] + E["ch_A"], "f", vow, r)
    chm = choir_section(n, E["ch_T"] + E["ch_B"], "m", vow, r)
    G["choir"] = chf + 0.85 * chm

    log("piano / harp / celesta")
    r = rng_for(seed, "piano")
    pb = np.zeros((2, n))
    mel["piano"] = np.zeros(n)
    for (t0, m, v, d) in E["piano"]:
        y = mu.piano_note(r, m, v, d)
        i0 = int(round(t0 * SR))
        mu.add_mono(pb, y, i0, float(np.clip((m - 60) / 40.0, -0.45, 0.45)))
        if m >= 60:
            yy = y[:max(0, n - i0)]
            mel["piano"][i0:i0 + len(yy)] += yy
    G["piano"] = conv2(pb, [body_ir(r, 70, 4000, 110, 0.14, (10, 40), 0.7) for _ in range(2)])
    hc = np.zeros((2, n))
    for (t0, m, v) in E["harp"]:
        mu.add_mono(hc, harp_note(r, m, v), int(round(t0 * SR)), float(np.clip(-0.5 + (m - 55) / 40.0, -0.6, 0.2)))
    hc = conv2(hc, [body_ir(r, 90, 3000, 60, 0.1, (8, 25), 0.6) for _ in range(2)])
    for (t0, m, v) in E["cel"]:
        mu.add_mono(hc, mu.celesta_note(r, m, v) * 0.8, int(round(t0 * SR)), r.uniform(0.1, 0.4))
    G["harp_cel"] = filt(sos("highpass", 90), hc)

    log("drums / fx")
    r = rng_for(seed, "drums")
    tkb = np.zeros((2, n))
    for (t0, f, v, size, pan) in E["taiko"]:
        mu.add_mono(tkb, mu.taiko(r, f, v, size), int(round(t0 * SR)), pan)
    tpb = np.zeros((2, n))
    for (t0, m, v, dec) in E["timp"]:
        mu.add_mono(tpb, mu.timpani(r, m, v, dec), int(round(t0 * SR)), 0.18)
    room = st.make_ir(rng_for(seed, "room"), rt60=[0.9, 0.9, 0.85, 0.8, 0.75, 0.65, 0.5, 0.4, 0.3],
                      band_db=[-4, -1, 0, 0, 0, -1, -3, -6, -10], length=1.2, predelay=0.006, onset=0.012, er=30, er_span=0.045)
    G["taiko"] = mu.soft_clip(tkb + 0.32 * st.convolve_stereo(tkb, room))
    G["timp"] = tpb + 0.25 * st.convolve_stereo(tpb, room)
    bb = np.zeros((2, n))
    for (t0, f, v, dec) in E["boom"]:
        mu.add_mono(bb, mu.sub_boom(f, v, dec), int(round(t0 * SR)), 0.0)
    for (t0, ms, v) in E["braam"]:
        mu.add_stereo(bb, mu.braam(r, ms, 5.0, v) * 0.8, int(round(t0 * SR)) - int(0.004 * SR))
    G["boom_braam"] = bb
    fx = np.zeros((2, n))
    for (t0, v) in E["crash"]:
        mu.add_stereo(fx, mu.crash(r, v), int(round(t0 * SR)))
    for (t0, L, v) in E["revcym"]:
        y = mu.reverse_cymbal(r, L, v)
        mu.add_stereo(fx, y, int(round(t0 * SR)) - y.shape[-1])
    for (t0, L, v, m) in E["riser"]:
        y = filt(sos("highpass", 150), mu.riser(r, L, v, m))
        mu.add_stereo(fx, y, int(round(t0 * SR)) - y.shape[-1])

    log("hall reverb")
    hall = st.make_ir(rng_for(seed, "hall"), rt60=[2.6, 2.9, 3.1, 3.0, 2.8, 2.4, 1.8, 1.2, 0.7],
                      band_db=[-10, -4, -1, 0, 0, -1, -3, -7, -13], length=6.0, predelay=0.024, onset=0.07, er=26, er_span=0.09)
    for (t0, ms, L, v) in E["revswell"]:
        k = int(L * SR)
        dry = np.zeros(k)
        for m in ms:
            y = mu.piano_note(r, m, 0.7, L, rel=0.3)[:k]
            dry[:len(y)] += y
        wet = np.vstack([signal.oaconvolve(dry, hall[c])[:k] for c in range(2)])
        y = wet[:, ::-1].copy() * np.linspace(0, 1, k) ** 1.5
        kk = int(0.015 * SR)
        y[:, -kk:] *= np.linspace(1, 0, kk) ** 2
        mu.add_stereo(fx, filt(sos("highpass", 150), y) * v * 6.0, int(round(t0 * SR)) - k)
    G["fx"] = fx
    G["str_mel"] = np.zeros((2, n))       # melodies are inside the string sections (shared players' bus)

    arc, _ = mu.auto_curve(ARC, n, 15.0)       # dramatic arc on the dry groups (the hall follows)
    for k in G:
        G[k] = G[k] * dbg(GDB[k]) * (arc if k not in ("fx",) else 1.0)
    send = sum(G[k] * SEND[k] for k in G if k not in ("taiko", "timp"))
    send = send + filt(sos("highpass", 150, 2), G["taiko"] * SEND["taiko"] + G["timp"] * SEND["timp"])
    wet = st.convolve_stereo(send, hall) * dbg(-1.0)
    wet = filt(mu.peq_sos(300, -3.0, 0.8), filt(sos("lowpass", 10500, 2), filt(sos("highpass", 70), wet)))
    G["reverb"] = wet
    for k in G:
        G[k] = mu.edge_fade(G[k], 2.0)
    del G["str_mel"]
    return G, E, mel


def music_bus(G, n, T):
    x = sum(G.values())
    x = filt(sos("highpass", 28, 4), x)
    x = filt(mu.peq_sos(290, -3.0, 0.8), x)          # de-mud
    x = filt(mu.peq_sos(3100, -1.5, 1.1), x)         # tame harshness
    x = filt(mu.shelf_sos(6500, 2.0, "high"), x)     # air / sheen
    x = filt(mu.shelf_sos(60, -1.0, "low"), x)       # keep the sub from booming
    mid, side = 0.5 * (x[0] + x[1]), 0.5 * (x[0] - x[1])
    side = filt(sos("highpass", 110, 2), side)       # mono bass below ~110 Hz
    x = np.vstack([mid + side, mid - side])
    fs, fe = MUSIC_FADE[0] + T - 60.0, MUSIC_FADE[1] + T - 60.0
    t = np.arange(n) / SR
    fade = np.cos(0.5 * np.pi * np.clip((t - fs) / (fe - fs), 0, 1)) ** 2 * np.clip(t / 0.02, 0, 1)
    return x * fade, fade


# =============================================================================
# analysis
# =============================================================================
def chord_check(x, tl):
    """chroma (60-2000 Hz) per chord span vs triad templates: rank of the written chord"""
    mono = x.mean(axis=0)
    f, tt, Z = signal.stft(mono, SR, nperseg=8192, noverlap=8192 - 2400)
    P = np.abs(Z) ** 2
    sel = (f > 60) & (f < 2000)
    pc = np.round(12 * np.log2(f[sel] / 440.0) + 69).astype(int) % 12
    C = np.zeros((12, P.shape[1]))
    for k in range(12):
        C[k] = P[sel][pc == k].sum(axis=0)
    res = []
    tmpl = {}
    for r in range(12):
        for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            v = np.zeros(12)
            v[[(r + i) % 12 for i in iv]] = 1
            tmpl[NOTE_NAMES[r] + q] = v
    for b0, b1, name in CHORDS:
        a, b = tl.t(b0) + 0.15, tl.t(b1) - 0.05
        s = (tt >= a) & (tt < b)
        if not s.any():
            continue
        ch = np.sqrt(C[:, s].mean(axis=1))
        ch /= np.linalg.norm(ch) + 1e-12
        w = np.zeros(12)
        w[CHORD_PCS[name]] = 1
        w /= np.linalg.norm(w)
        sc = {k: float(ch @ (v / np.linalg.norm(v))) for k, v in tmpl.items()}
        best = max(sc, key=sc.get)
        written = float(ch @ w)
        res.append(dict(beat=b0, chord=name, written_corr=round(written, 3), best_triad=best, best_corr=round(sc[best], 3),
                        top3=[nm for nm in sorted(CHORD_PCS[name], key=lambda p: -ch[p])][:3]))
    return res


def chord_root_ok(r):
    nm = r["chord"].split("/")[0].replace("add9", "").replace("sus", "").replace("7", "")
    return r["best_triad"].rstrip("m") == nm.rstrip("m")


def analyze(out_dir, tl, mix, music, nature, G, E, mel, info, gd):
    os.makedirs(out_dir, exist_ok=True)
    T, S, shots = tl.T, tl.S, tl.shots
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
    rep["lr_corr_low"] = an.lr_correlation(filt(sos("lowpass", 150, 4), mix))
    mono = np.vstack([mix.mean(axis=0)] * 2)
    rep["mono_foldown_loss_db"] = float(an.integrated_lufs(mix) - an.integrated_lufs(mono))
    rep["last_100ms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(mix[:, -4800:] ** 2)) + 1e-12))
    ts, ls = an.short_term(xk)
    tm, lm = an.momentary(xk)
    rep["max_short_term"] = float(ls.max())
    rep["max_momentary"] = float(lm.max())
    rep["nature_gain_db"] = gd
    rows = []
    gk = {k: an.k_weight(g) for k, g in G.items()}
    oct_c = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    for i, s in enumerate(shots):
        a, b = S[i] + 0.25, S[i + 1] - 0.25
        row = dict(shot=s["name"], mix=an.segment_lufs(xk, a, b), music=an.segment_lufs(mk, a, b), nature=an.segment_lufs(nk, a, b))
        row["n_m"] = row["nature"] - row["music"]
        selt = (ts >= S[i]) & (ts <= S[i + 1])
        row["st_max"] = float(ls[selt].max())
        for k, g in gk.items():
            row[k] = an.segment_lufs(g, a, b)
        ia, ib = int(a * SR), int(b * SR)
        fr, Pw = signal.welch(mix[:, ia:ib].mean(axis=0), SR, nperseg=8192)
        bp = np.array([Pw[(fr >= c / 1.414) & (fr < c * 1.414)].sum() for c in oct_c]) + 1e-20
        lv = 10 * np.log10(bp / bp.sum())
        row["oct_db"] = [round(float(v), 1) for v in lv]
        rows.append(row)
    rep["shots"] = rows
    cuts = []
    for c in S[1:-1]:
        pre = float(np.max(lm[(tm > c - 0.8) & (tm < c - 0.05)]))
        gap = float(np.min(lm[(tm > c - 0.45) & (tm < c - 0.02)]))
        post = float(np.max(lm[(tm > c) & (tm < c + 0.6)]))
        cuts.append(dict(cut=c, pre=pre, gap_min=gap, post=post))
    rep["cuts"] = cuts
    rep["short_term_curve"] = [(round(float(a), 1), round(float(b), 2)) for a, b in zip(ts[::5], ls[::5])]
    tracks, accs = {}, {}
    refs = {"horn": [(a, b, m) for (a, b, m, *_r) in E["hn_ref"]],
            "violins": [(a, b, m) for (a, b, m, *_r) in E["vn1_mel_ref"]],
            "piano": [(a, b, m) for (a, b, m) in E["piano_ref"]]}
    for name, x in mel.items():
        t, p = mu.pitch_track(x)
        ref = refs[name]
        if name == "piano":
            ref = [(a, min(b, ref[i + 1][0]) if i + 1 < len(ref) else b, m) for i, (a, b, m) in enumerate(ref)]
        tracks[name] = (t, p, ref)
        accs[name] = mu.motif_accuracy(t, p, ref)
    rep["motif_pitch_accuracy"] = {k: dict(within_half_semitone=round(v[0], 3), octave_errors=round(v[1], 3), frames=v[2])
                                   for k, v in accs.items()}
    mu.motif_png(os.path.join(out_dir, "motif_pitch.png"), tracks, shots, T)
    rep["chords"] = chord_check(music, tl)
    rep["chord_root_match"] = float(np.mean([chord_root_ok(r) for r in rep["chords"]]))
    an.render_panel(mix, "candidate A (orchestral) + nature: spectrogram and loudness vs the cut list", shots, T,
                    spec_h=320, level_h=150, db_lo=-120.0, db_hi=-25.0).save(os.path.join(out_dir, "spec_mix.png"))
    panels = [("mix", mix), ("music", music), ("nature (at mix gain)", nature)]
    grp = {"strings": ("strings",), "spiccato": ("spicc",), "horns": ("horns",), "trombones+tuba": ("lowbrass",), "choir": ("choir",),
           "piano+harp+celesta": ("piano", "harp_cel"), "drums": ("taiko", "timp", "boom_braam"), "fx": ("fx",), "hall": ("reverb",)}
    for name, ks in grp.items():
        panels.append((name, sum(G[k] for k in ks if k in G)))
    an.overview_png(os.path.join(out_dir, "spec_overview.png"), panels, shots, T, spec_h=100)
    for i, s in enumerate(shots):
        an.zoom_png(os.path.join(out_dir, f"spec_{i}_{s['name']}.png"), mix, max(0.0, S[i] - 0.6), min(T, S[i + 1] + 0.4),
                    f"mix: {s['name']}", spec_h=300, level_h=110, width=900, db_lo=-120.0, db_hi=-25.0)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=float)
    return rep


def print_report(rep):
    print(f"  integrated {rep['integrated_lufs']:.2f} LUFS | TP {rep['true_peak_dbtp']:.2f} dBTP | sample peak {rep['sample_peak_dbfs']:.2f}"
          f" | max ST {rep['max_short_term']:.2f} | max M {rep['max_momentary']:.2f}")
    print(f"  DC {rep['dc'][0]:.1e} {rep['dc'][1]:.1e} | clipped {rep['clipped']} | frames {rep['frames']} | last 100 ms "
          f"{rep['last_100ms_dbfs']:.1f} dBFS | L/R corr {rep['lr_corr']:.2f} (<150 Hz {rep['lr_corr_low']:.2f}) | mono loss "
          f"{rep['mono_foldown_loss_db']:.2f} dB")
    c, cm = rep["clicks_mix"], rep["clicks_music"]
    print(f"  clicks mix {c['n_hits']} (max ratio {c['max_ratio']:.1f}) | music {cm['n_hits']} (max ratio {cm['max_ratio']:.1f}) hf {cm['hf_hits']}")
    keys = ["strings", "spicc", "horns", "lowbrass", "choir", "piano", "harp_cel", "taiko", "timp", "boom_braam", "fx", "reverb"]
    print("  shot        mix  music nature  n-m  stmax | " + " ".join(f"{k[:6]:>6}" for k in keys))
    for r in rep["shots"]:
        print(f"  {r['shot']:<9} {r['mix']:6.1f} {r['music']:6.1f} {r['nature']:6.1f} {r['n_m']:5.1f} {r['st_max']:6.1f} | "
              + " ".join(f"{max(r.get(k, -99), -99):6.1f}" for k in keys))
    print("  octave bands (dB re total) 31 63 125 250 500 1k 2k 4k 8k 16k")
    for r in rep["shots"]:
        print(f"  {r['shot']:<9} " + " ".join(f"{v:5.1f}" for v in r["oct_db"]))
    print("  cuts (momentary pre / min in last 0.45 s / post): " + "  ".join(
        f"{c['cut']:.1f}:{c['pre']:.1f}/{c['gap_min']:.1f}/{c['post']:.1f}" for c in rep["cuts"]))
    print("  motif: " + "  ".join(f"{k} {v['within_half_semitone']:.2f} (oct {v['octave_errors']:.2f}, {v['frames']} fr)"
                                  for k, v in rep["motif_pitch_accuracy"].items()))
    print(f"  chords: root match {rep['chord_root_match']:.2f} | " + " ".join(
        f"{r['chord']}>{r['best_triad']}({r['written_corr']:.2f})" for r in rep["chords"]))
    print(f"  master: comp {rep.get('comp_max_gr_db', 0):.2f} dB, limiter {rep.get('limiter_max_gr_db', 0):.2f} dB at "
          f"{rep.get('limiter_gr_at_s', 0):.2f} s, ceiling {rep.get('ceiling_used_db', 0):.2f}")


# =============================================================================
def main():
    global VS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", default=os.path.join(ROOT, "audio", "film_shots.json"))
    ap.add_argument("--nature", default=os.path.join(ROOT, "out", "audio", "nature", "stems", "nature.wav"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "audio", "candidates", "A_orchestral.wav"))
    ap.add_argument("--music-out", default=None)
    ap.add_argument("--stems", default=None)
    ap.add_argument("--analyze", default=None)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="half the players per section (quick test renders)")
    a = ap.parse_args()
    if a.fast:
        VS = 0.5
    with open(a.shots, "r", encoding="utf-8") as fh:
        shots = json.load(fh)
    shots = shots.get("shots", shots) if isinstance(shots, dict) else shots
    tl = TL(shots)
    log(f"shots {a.shots}: T={tl.T:.3f} s, shot beats {np.diff(tl.B).astype(int).tolist()}")
    G, E, mel = render(tl, a.seed)
    music, fade = music_bus(G, tl.n, tl.T)
    music = music * dbg(-15.0 - an.integrated_lufs(music))
    music, mcomp = st.bus_compressor(music, thr_db=-17.0, ratio=1.7, knee_db=8.0, att=0.03, rel=0.35)
    log(f"music bus glue compressor max GR {mcomp:.2f} dB")
    nat, sr = an.read_any_wav(a.nature)
    assert sr == SR, sr
    nat = np.asarray(nat, float)[:, :tl.n]
    if nat.shape[-1] < tl.n:
        nat = np.pad(nat, ((0, 0), (0, tl.n - nat.shape[-1])))
    mu.NATURE_REL = NATURE_REL
    ng, gd = mu.nature_gain(tl, nat, music)
    nature = nat * ng
    log("nature gain per shot (dB): " + ", ".join(f"{s['name']} {g:+.1f}" for s, g in zip(tl.shots, gd)))
    mix = music + nature
    out, info = mu.master(mix, TARGET_LUFS, CEILING_DBTP)
    tail = int(0.1 * SR)
    out[:, -tail:] = 0.0                                   # the last 100 ms are digital silence
    k = int(0.05 * SR)
    out[:, -tail - k:-tail] *= np.cos(0.5 * np.pi * np.linspace(0, 1, k)) ** 2
    sg = float(np.sqrt(np.sum(out ** 2) / (np.sum(mix ** 2) + 1e-20)))
    info["mix_to_master_gain_db"] = 20 * math.log10(sg)
    log("master: " + ", ".join(f"{kk}={v:.2f}" for kk, v in info.items()))
    st.write_wav(a.out, out, False, a.seed)
    log(f"wrote {a.out} ({out.shape[-1]} frames, {out.shape[-1] / SR:.3f} s)")
    if a.music_out:
        mo, minfo = mu.master(music * fade, TARGET_LUFS, CEILING_DBTP)
        mo[:, -tail:] = 0.0
        st.write_wav(a.music_out, mo, False, a.seed)
        log(f"wrote {a.music_out} (music only, TP {minfo['true_peak_db']:.2f})")
    if a.stems or a.analyze:
        scale = sg * float(np.mean(np.abs(music)) / (np.mean(np.abs(sum(G.values()))) + 1e-20))
        Gs = {kk: v * scale for kk, v in G.items()}
        if a.stems:
            os.makedirs(a.stems, exist_ok=True)
            from scipy.io import wavfile
            for kk, v in Gs.items():
                wavfile.write(os.path.join(a.stems, f"{kk}.wav"), SR, v.T.astype(np.float32))
            wavfile.write(os.path.join(a.stems, "nature.wav"), SR, (nature * sg).T.astype(np.float32))
            log(f"wrote stems to {a.stems}")
        if a.analyze:
            x, _ = an.read_wav(a.out)
            rep = analyze(a.analyze, tl, x, music * sg, nature * sg, Gs, E, mel, info, gd)
            print_report(rep)
    log("done")


if __name__ == "__main__":
    main()
