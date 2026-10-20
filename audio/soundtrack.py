#!/usr/bin/env python
"""soundtrack.py - 100 % procedural soundtrack for the alpine lake film.

Everything is synthesised with numpy/scipy from random numbers and oscillators:
no samples, no recordings, no external assets.

    python audio/soundtrack.py [--shots out/film_shots.json] [--out out/audio/soundtrack.wav]
                               [--seed N] [--stems] [--analyze]

The shot list drives everything (default: the film renderer's out/film_shots.json,
fallback audio/film_shots.json). Each shot names an environment ("reveal", "lakeshore",
"forest", "waterfall", "stream", "meadow", "high_view", "golden"; aliases such as
"skim", "flyover", "finale" are mapped via ENV_ALIASES) whose parameters live in
ENV_TABLE below; NAME_TABLE adds per-shot overrides by shot name. Ambience changes
lead the picture (J-cut, LEAD seconds before each visual cut) with an equal-sum
raised-cosine crossfade. See README.md for the layer list and how to tune it.
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
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, minimum_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analysis as an  # noqa: E402

SR = 48000
OFF = -120.0          # "layer off" in dB tables
CTRL_HOP = 240        # control-rate step for gain curves (5 ms)
LEAD = 0.45           # J-cut: sound changes this many seconds before the picture
XFADE = 1.2           # ambience crossfade length around each (lead-shifted) cut
TARGET_LUFS = -17.0   # integrated loudness target of the master
CEILING_DBTP = -1.2   # limiter ceiling (true-peak, 4x oversampled check)
MUSIC_WIN_S = 3.0     # music level follows nature short-term loudness (3 s window) ...
MUSIC_SMOOTH_S = 2.0  # ... Gaussian-smoothed (sigma, s) so it does not pump with single gusts ...
MUSIC_CLAMP_DB = 8.0  # ... and the gain may move at most +-8 dB around its median
FADE_IN = 2.5
FADE_OUT = 1.5         # master fade (music + its reverb tail)
FADE_OUT_NATURE = 4.5  # nature bus fades over the last 4.5 s
EVENT_STOP = 3.5       # no new lake laps / brook events / bird songs after T - 3.5 s
BELL_GAIN = 10 ** (-3.0 / 20.0)
# per-parameter crossfade multiplier: a key listed here gets an extended crossfade where the
# outgoing shot's value is higher than the incoming one (a long release, e.g. the waterfall
# roar decaying into the forest), stretched after the cut point.
XF_MULT = {"falls_db": 2.5}
WF_HP = 90.0          # waterfall high-pass corner close to the fall (prox >= 0.8; Hz, 3rd order) ...
WF_HP_FAR = 70.0      # ... and far from it (prox -> 0): keeps the close roar from booming in 20-50 Hz
                      # while the receding tail stays a soft low rumble
WF_SURGE = (0.25, 0.18, 0.10)  # waterfall level surges (relative) at 0.25 Hz, 1.5 Hz and 4 Hz
WF_THUMP = 0.16       # waterfall 'mass impact' thump level (roar is ~0.1 RMS)
WF_THUMP_HP = 0.0     # extra high-pass (Hz, 3rd order) on the thumps; 0 = none
COH_D = 0.8           # virtual capsule spacing (m) for the diffuse-field L/R coherence of noise beds
WIND_FLOOR = 0.22     # wind level between gusts relative to the gust term (1.5 * gust envelope): ~14 dB under a typical gust
WIND_NORM = 1.8       # keeps the gusty (high_view) wind level where the tables expect it

# =============================================================================
# Environment table. Values are constants or lists [v0, v1, ...] which are
# interpolated (smoothstep) across the shot's audio window. *_db are levels in
# dB relative to each layer's unit level (about -20 dBFS RMS); OFF = silent.
# Bird entries b_<species> are expected numbers of singing individuals.
# =============================================================================
DEFAULTS = dict(
    # wind bed
    wind_db=-10.0, wind_fc=400.0, wind_bw=1.1, wind_lp=6500.0, wind_tilt=0.6,
    gust_rate=0.15, gust_depth=0.9, gust_len=1.0, whistle=0.0,
    # foliage / grass rustle (driven by the same gusts); rustle_bw = band width in octaves
    rustle_db=OFF, rustle_fc=3500.0, rustle_bw=1.0, grain_rate=60.0,
    # lake lapping (lake_period = mean interval between laps, s); open-water ripple
    lake_db=OFF, lake_period=1.3, lake_lp=10000.0, ripple_db=OFF,
    # songbirds
    birds_db=0.0, b_chaffinch=0.0, b_robin=0.0, b_thrush=0.0, b_tit=0.0, b_tsit=0.0,
    bird_dmin=20.0, bird_dmax=90.0, woodpecker=0.0,
    # waterfall
    falls_db=OFF, falls_prox=0.1, falls_pan=0.35,
    # stream
    stream_db=OFF, stream_lp=14000.0,
    # meadow insects
    insects_db=OFF, bee=0.0,
    # music: pad_db shapes the pad's brightness envelope, music_rel_db is the music
    # loudness relative to the nature short-term loudness (LU), bell = bell figure
    pad_db=-6.0, bell=0.0, music_rel_db=-12.0,
    # reverb (send return level and forest/valley impulse response mix)
    rev_db=-8.0, rev_forest=0.0,
)

ENV_TABLE = {
    "reveal": dict(
        wind_db=[-13.0, -8.0], wind_fc=380.0, wind_lp=[2200.0, 6500.0], gust_rate=0.13,
        lake_db=[-24.0, -18.0, -12.0], lake_lp=[1400.0, 9000.0], lake_period=1.35,
        b_chaffinch=1.0, b_thrush=1.0, b_robin=0.5, bird_dmin=25.0, bird_dmax=110.0,
        falls_db=[OFF, -36.0, -30.0], falls_prox=0.08, falls_pan=0.45,
        pad_db=[-13.0, -6.0, 0.0], bell=1.0, rev_db=[-11.0, -7.0], music_rel_db=[-12.0, -8.0, -6.0],
    ),
    "lakeshore": dict(
        wind_db=-11.0, wind_fc=350.0, lake_db=-11.0, lake_period=1.2,
        b_chaffinch=1.0, b_thrush=1.0, b_robin=0.7, b_tit=0.4, bird_dmin=20.0, bird_dmax=100.0,
        falls_db=-31.0, falls_prox=0.1, falls_pan=0.45, pad_db=-6.7, bell=0.35, birds_db=2.0,
        music_rel_db=-12.0,
    ),
    "forest": dict(
        wind_db=-17.0, wind_fc=300.0, wind_lp=1500.0, gust_rate=0.14,
        rustle_db=-15.0, rustle_fc=2600.0, grain_rate=80.0,
        lake_db=-26.0, lake_lp=1500.0,
        b_chaffinch=1.5, b_robin=1.0, b_tit=1.0, b_tsit=2.0, b_thrush=0.5, bird_dmin=15.0, bird_dmax=60.0,
        birds_db=-3.0, woodpecker=1.0, falls_db=-40.0, falls_prox=0.05,
        pad_db=-8.4, rev_db=-8.0, rev_forest=1.0, music_rel_db=-10.0,
    ),
    "high_view": dict(
        wind_db=[-15.0, -13.0], wind_fc=480.0, wind_bw=1.4, wind_lp=6000.0, gust_rate=0.32, gust_depth=1.2,
        gust_len=0.65,
        whistle=0.35, b_thrush=0.4, b_chaffinch=0.3, bird_dmin=150.0, bird_dmax=300.0,
        falls_db=-38.0, falls_prox=0.03, falls_pan=-0.3, pad_db=-5.0, bell=0.0, rev_db=-10.0,
        music_rel_db=-6.0,
    ),
    "waterfall": dict(
        wind_db=OFF, falls_db=[-8.0, -4.0, -2.5], falls_prox=[0.6, 0.8], falls_pan=0.0,
        lake_db=-15.0, lake_period=0.95,
        pad_db=-4.5, rev_db=-11.0, music_rel_db=-16.0,
    ),
    "stream": dict(
        wind_db=-20.0, stream_db=-13.0, rustle_db=-21.0, falls_db=-25.0, falls_prox=0.2, falls_pan=-0.4,
        b_robin=1.0, b_chaffinch=1.0, b_tit=0.5, bird_dmin=20.0, bird_dmax=80.0, birds_db=-6.0, pad_db=-5.0,
        music_rel_db=-14.0,
    ),
    "meadow": dict(
        wind_db=-12.0, wind_fc=500.0, wind_lp=5000.0, rustle_db=-22.0, rustle_fc=3200.0, rustle_bw=1.6,
        grain_rate=40.0, insects_db=6.0, bee=1.0, stream_db=-24.0, stream_lp=3000.0, lake_db=-24.0,
        lake_lp=2500.0, b_chaffinch=0.6, b_thrush=0.5, b_tit=0.3, bird_dmin=40.0, bird_dmax=150.0, birds_db=-2.0,
        falls_db=OFF, pad_db=-10.5, bell=0.35, music_rel_db=-9.0,
    ),
    "golden": dict(
        wind_db=[-13.0, -17.0], wind_fc=330.0, gust_rate=0.08, gust_depth=0.5,
        lake_db=-10.0, lake_period=1.65,
        b_thrush=1.5, b_robin=1.0, b_chaffinch=0.2, bird_dmin=18.0, bird_dmax=70.0, birds_db=2.0,
        falls_db=-34.0, falls_prox=0.07, pad_db=[-8.0, -5.0, -3.5], bell=1.0, rev_db=-7.0,
        music_rel_db=[-8.0, -5.0],
    ),
}

# shot env/name spellings that map onto an ENV_TABLE entry
ENV_ALIASES = {
    "skim": "lakeshore", "lake": "lakeshore", "shore": "lakeshore",
    "flyover": "high_view", "aerial": "high_view", "peaks": "high_view", "ridge": "high_view",
    "highview": "high_view", "high-view": "high_view", "high view": "high_view",
    "finale": "golden", "sunset": "golden", "golden_hour": "golden",
    "falls": "waterfall", "creek": "stream", "brook": "stream", "woods": "forest",
}

# per-shot overrides by shot *name* (merged onto ENV_TABLE[env]). The film shot list has no
# intensity field, so the intensity (0..1) lives here as well.
NAME_TABLE = {
    "reveal": dict(intensity=0.4),
    # fast, 3 m above open water, heading for the waterfall shore: no near shore, far laps
    # only, open-water ripple, the fall pre-laps from the right
    "skim": dict(intensity=0.45, lake_db=-20.0, lake_lp=2500.0, ripple_db=-14.0, wind_db=-14.0,
                 falls_db=[-30.0, -24.0, -15.0], falls_pan=[0.6, 0.35],
                 bird_dmin=60.0, bird_dmax=220.0, birds_db=0.0, bell=0.0),
    "waterfall": dict(intensity=0.75),
    "forest": dict(intensity=0.3),
    "stream": dict(intensity=0.55),
    "meadow": dict(intensity=0.35),
    # ~600 m above the valley: the music carries the aerials, wind stays modest, no birds
    "flyover": dict(intensity=0.5, wind_db=-13.0, gust_rate=0.2, gust_len=1.0, b_thrush=0.0, b_chaffinch=0.0),
    # passing the summits: fewer, shorter gusts (ENV gust_len 0.65) that rise out of a low floor
    "peaks": dict(intensity=0.6, wind_db=[-14.0, -17.0], gust_rate=0.24, whistle=0.5, b_thrush=0.0,
                  b_chaffinch=0.0),
    "finale": dict(intensity=0.2),
}


def norm_env(s):
    """Robust env lookup: env field (or the name when env is missing), aliases, fallback."""
    raw = str(s.get("env") or s.get("name") or "lakeshore").strip().lower().replace("-", "_")
    env = ENV_ALIASES.get(raw, raw)
    if env not in ENV_TABLE:
        nm = str(s.get("name") or "").strip().lower()
        env = ENV_ALIASES.get(nm, nm)
    if env not in ENV_TABLE:
        print(f"  warning: unknown env '{raw}', using 'lakeshore'")
        env = "lakeshore"
    return env


def resolve_params(shot):
    """Merge DEFAULTS, ENV_TABLE[env], NAME_TABLE[name] and the shot's intensity (0..1)."""
    P = dict(DEFAULTS)
    P.update(ENV_TABLE[shot["env"]])
    P.update({k: v for k, v in NAME_TABLE.get(shot.get("name", ""), {}).items() if k != "intensity"})
    I = shot["intensity"]

    def fmap(key, fn):
        v = P[key]
        P[key] = [fn(x) for x in v] if isinstance(v, (list, tuple)) else fn(v)

    def add_db(key, d):
        fmap(key, lambda x: x if x <= -90 else x + d)

    add_db("wind_db", 6.0 * (I - 0.5))
    fmap("gust_rate", lambda x: x * (0.6 + 0.8 * I))
    fmap("gust_depth", lambda x: x * (0.7 + 0.6 * I))
    add_db("falls_db", 3.0 * (I - 0.5))
    add_db("stream_db", 2.0 * (I - 0.5))
    fmap("lake_period", lambda x: x * (1.15 - 0.3 * I))
    return P


# =============================================================================
# helpers
# =============================================================================
def rng_for(seed, *names):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF] + [zlib.crc32(n.encode()) for n in names])


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def db_lin(v):
    v = np.asarray(v, float)
    return np.where(v <= -90.0, 0.0, 10.0 ** (v / 20.0))


def smooth_noise(rng, t, rate):
    """Smooth zero-mean unit-std random curve, correlation time ~1/rate s, sampled at t."""
    t = np.asarray(t, float)
    lo, hi = float(t.min()) - 2.0 / rate, float(t.max()) + 2.0 / rate
    step = 0.125 / rate
    m = int((hi - lo) / step) + 3
    v = gaussian_filter1d(rng.standard_normal(m), 4.0, mode="reflect")
    v = (v - v.mean()) / (v.std() + 1e-12)
    return np.interp(t, lo + np.arange(m) * step, v)


def pan_gains(p):
    th = (np.clip(p, -1, 1) + 1.0) * np.pi / 4.0
    return np.cos(th), np.sin(th)


def add_at(buf, x, i0):
    """Add mono (n,) or stereo (2, n) x into stereo buf at sample i0 (bounds-safe)."""
    x = np.atleast_2d(x)
    n = buf.shape[-1]
    L = x.shape[-1]
    a, b = max(0, i0), min(n, i0 + L)
    if b <= a:
        return
    buf[:, a:b] += x[:, a - i0:b - i0]


def add_panned(buf, mono, i0, pan, gain=1.0):
    gl, gr = pan_gains(pan)
    add_at(buf, np.vstack([mono * gl * gain, mono * gr * gain]), i0)


_SOS_CACHE = {}


def butter_sos_q(order, f, kind):
    """butter_sos with frequencies quantised to 1/24 octave and cached (for many grains)."""
    q = lambda v: float(2.0 ** (round(math.log2(max(v, 10.0)) * 24) / 24.0))  # noqa: E731
    fq = tuple(q(v) for v in f) if isinstance(f, (list, tuple)) else q(f)
    key = (order, fq, kind)
    if key not in _SOS_CACHE:
        _SOS_CACHE[key] = butter_sos(order, list(fq) if isinstance(fq, tuple) else fq, kind)
    return _SOS_CACHE[key]


def butter_sos(order, f, kind):
    if kind == "bandpass":
        f = [max(10.0, f[0]), min(f[1], 0.47 * SR)]
        if f[1] <= f[0] * 1.05:
            f[1] = min(f[0] * 1.5, 0.48 * SR)
    else:
        f = min(max(10.0, f), 0.47 * SR)
    return signal.butter(order, f, kind, fs=SR, output="sos")


def tail_fade(x, ms=8.0):
    k = min(x.shape[-1], int(ms * 1e-3 * SR))
    if k > 1:
        x[..., -k:] *= np.linspace(1.0, 0.0, k)
    return x


class Grid:
    """STFT grid for time-varying, zero-phase spectral filtering (WOLA, Hann, 75 % overlap).
    Filtering white noise through Grid.filter with a per-frame magnitude = noise through a
    slowly modulated filter (band-pass, tilt, low-pass ...)."""

    def __init__(self, n, nfft=4096, hop=1024):
        self.n, self.nfft, self.hop = n, nfft, hop
        self.pad = nfft
        self.L = n + 2 * nfft + hop
        self.nfr = (self.L - nfft) // hop + 1
        self.times = (np.arange(self.nfr) * hop + nfft / 2 - self.pad) / SR
        self.freqs = np.fft.rfftfreq(nfft, 1.0 / SR)
        self.fs = np.maximum(self.freqs, 10.0)
        self.win = signal.windows.hann(nfft, sym=False)
        self.norm = np.sum(self.win ** 2) / hop

    def stft(self, x):
        xp = np.zeros(self.L)
        xp[self.pad:self.pad + self.n] = x
        fr = np.lib.stride_tricks.sliding_window_view(xp, self.nfft)[::self.hop][:self.nfr] * self.win
        return np.fft.rfft(fr, axis=1)

    def istft(self, X):
        y = np.fft.irfft(X, n=self.nfft, axis=1) * self.win
        R = self.nfft // self.hop
        y = y.reshape(self.nfr, R, self.hop)
        out = np.zeros((self.nfr + R - 1, self.hop))
        for m in range(R):
            out[m:m + self.nfr] += y[:, m, :]
        return out.reshape(-1)[self.pad:self.pad + self.n] / self.norm

    def filter(self, x, mag):
        X = self.stft(x)
        X *= mag
        return self.istft(X)

    def coherence(self, d=0.4):
        """Magnitude-squared coherence of a diffuse sound field between two omni capsules d
        metres apart: sinc^2(2 f d / c). 1 at low frequencies, ~0 above c/(2d)."""
        return np.sinc(2.0 * self.freqs * d / 343.0) ** 2

    def filter_pair(self, S, rng, d=COH_D, S2=None, gain=1.0):
        """Natural-coherence stereo noise texture: L = C*S*sqrt(g) + NL*S*sqrt(1-g),
        R = C*S2*sqrt(g) + NR*S2*sqrt(1-g) with shared noise C and independent NL/NR.
        S2 (right-channel magnitude) defaults to S."""
        g = self.coherence(d)
        a, b = np.sqrt(g)[None, :], np.sqrt(1.0 - g)[None, :]
        XC = self.stft(rng.standard_normal(self.n) * gain) * a
        out = np.zeros((2, self.n))
        for ch, M in ((0, S), (1, S if S2 is None else S2)):
            X = self.stft(rng.standard_normal(self.n) * gain)
            X *= b
            X += XC
            X *= M
            out[ch] = self.istft(X)
        return out

    def seethe(self, rng, sigma_frames, amount, n_bands=40, fmin=40.0, fmax=20000.0):
        """Random, smooth time-frequency gain field exp(amount*N) - the 'living' texture of water/wind."""
        F = rng.standard_normal((self.nfr, n_bands))
        F = gaussian_filter(F, sigma=(sigma_frames, 1.0), mode="reflect")
        F /= F.std() + 1e-12
        lf = np.log(np.clip(self.fs, fmin, fmax) / fmin) / np.log(fmax / fmin) * (n_bands - 1)
        i0 = np.floor(lf).astype(int)
        i1 = np.minimum(i0 + 1, n_bands - 1)
        fr = lf - i0
        return np.exp(amount * (F[:, i0] * (1 - fr) + F[:, i1] * fr))

    @staticmethod
    def unit(S):
        """Normalise each frame's magnitude to unit mean-square (level-independent of shape)."""
        return S / np.sqrt(np.mean(S ** 2, axis=1, keepdims=True) + 1e-20)


def tv_lowpass(x, fc_t, t_ctrl):
    """Time-varying 2nd-order-like low-pass (distance / occlusion) for a stereo signal."""
    if np.min(fc_t) > 15000:
        return x
    g = Grid(x.shape[-1], 2048, 512)
    fc = np.exp(np.interp(g.times, t_ctrl, np.log(fc_t)))
    mag = 1.0 / np.sqrt(1.0 + (g.fs[None, :] / fc[:, None]) ** 4)
    return np.vstack([g.filter(x[c], mag) for c in range(x.shape[0])])


def tame_peaks(x, gain, crest_db=9.0, ratio=3.0):
    """Fast transient compressor for event textures (lake laps, brook splashes): reduces
    peaks more than crest_db above the active RMS so that single events cannot dominate
    the master limiter."""
    m = gain > 0.5 * gain.max()
    if not m.any():
        return x
    thr = 10 * np.log10(np.mean(x[:, m] ** 2) + 1e-20) + crest_db
    return bus_compressor(x, thr_db=thr, ratio=ratio, knee_db=6.0, att=0.004, rel=0.18, block=48)[0]


def norm_active(x, gain, target=0.1, thresh=0.5):
    """Scale texture x so its RMS where gain is high equals target."""
    m = gain > thresh * gain.max()
    if not m.any():
        return x
    r = np.sqrt(np.mean(x[:, m] ** 2) + 1e-20)
    return x * (target / r)


# =============================================================================
# timeline: shots -> per-shot weights (J-cut crossfades) -> parameter curves
# =============================================================================
class Timeline:
    def __init__(self, shots, duration=None, lead=LEAD, xfade=XFADE):
        clean = []
        cursor = 0.0
        for idx, s in enumerate(shots):
            env = norm_env(s)
            name = str(s.get("name") or env).strip().lower()
            # timing: start/end, or t0/t1, or durations summed from the previous end
            st = s.get("start", s.get("t0"))
            en = s.get("end", s.get("t1"))
            if st is None:
                st = cursor
            if en is None:
                if s.get("dur") is None and s.get("duration") is None:
                    raise SystemExit(f"shot {idx} has no end/t1/dur")
                en = float(st) + float(s.get("dur", s.get("duration")))
            st, en = float(st), float(en)
            cursor = en
            inten = s.get("intensity", NAME_TABLE.get(name, {}).get("intensity", 0.5))
            clean.append(dict(start=st, end=en, env=env, name=name,
                              intensity=float(np.clip(inten, 0.0, 1.0)), notes=s.get("notes", "")))
        if not clean:
            raise SystemExit("empty shot list")
        clean.sort(key=lambda s: s["start"])
        self.shots = clean
        self.T = float(duration) if duration else max(s["end"] for s in clean)
        self.n = int(round(self.T * SR))
        k = len(clean)
        a = [0.0]
        for i in range(1, k):
            ai = clean[i]["start"] - lead
            ai = max(ai, a[-1] + 0.1)
            ai = min(ai, self.T - 0.1 * (k - i))
            a.append(ai)
        self.a = np.array(a)
        self.b = np.append(self.a[1:], self.T)
        self.h = np.zeros(k + 1)
        for j in range(1, k):
            la, lb = self.b[j - 1] - self.a[j - 1], self.b[j] - self.a[j]
            self.h[j] = max(0.01, min(xfade / 2.0, 0.45 * la, 0.45 * lb))
        self.params = [resolve_params(s) for s in clean]
        self.ctrl_t = np.arange(0, self.n + 2 * CTRL_HOP, CTRL_HOP) / SR
        self._cache = {}

    def ramp(self, j, t, hb=None):
        """Crossfade ramp of boundary j: starts h before the (lead-shifted) cut and ends hb
        after it (hb = h: symmetric raised-cosine)."""
        c, h = self.a[j], self.h[j]
        hb = h if hb is None else hb
        x = np.clip((t - (c - h)) / (h + hb), 0.0, 1.0)
        return np.sin(0.5 * np.pi * x) ** 2

    def h_after(self, key=None):
        """Per-boundary crossfade length after the cut for parameter `key`. Keys in XF_MULT get
        a release XF_MULT times longer wherever the outgoing shot's value is the higher one."""
        m = XF_MULT.get(key) if key else None
        if not m:
            return self.h
        if ("hb", key) in self._cache:
            return self._cache[("hb", key)]
        hb = self.h.copy()
        for j in range(1, len(self.shots)):
            v_out = float(self._curve(self.params[j - 1][key], np.array([1.0]))[0])
            v_in = float(self._curve(self.params[j][key], np.array([0.0]))[0])
            if v_out > v_in + 1e-9:
                room = (self.b[j] - self.a[j]) - (self.h[j + 1] if j + 1 < len(self.h) else 0.0) - 0.05
                hb[j] = max(self.h[j], min(self.h[j] * (2.0 * m - 1.0), room))
        self._cache[("hb", key)] = hb
        return hb

    def weights(self, t, key=None):
        t = np.atleast_1d(np.asarray(t, float))
        k = len(self.shots)
        hb = self.h_after(key)
        R = [np.ones_like(t)] + [self.ramp(j, t, hb[j]) for j in range(1, k)] + [np.zeros_like(t)]
        return np.stack([R[i] * (1.0 - R[i + 1]) for i in range(k)])

    def u(self, i, t):
        return np.clip((t - self.a[i]) / max(1e-6, self.b[i] - self.a[i]), 0.0, 1.0)

    @staticmethod
    def _curve(v, u):
        if isinstance(v, (list, tuple)):
            v = np.asarray(v, float)
            if len(v) == 1:
                return np.full_like(u, v[0])
            x = u * (len(v) - 1)
            i = np.clip(np.floor(x).astype(int), 0, len(v) - 2)
            f = smoothstep(x - i)
            return v[i] * (1 - f) + v[i + 1] * f
        return np.full_like(u, float(v))

    def param(self, key, t, mode="lin"):
        """mode: 'lin' (weighted mean), 'log' (geometric), 'db' (dB -> linear gain, then weighted)."""
        t = np.atleast_1d(np.asarray(t, float))
        W = self.weights(t, key)
        out = np.zeros_like(t)
        for i, P in enumerate(self.params):
            if not np.any(W[i] > 0):
                continue
            v = self._curve(P[key], self.u(i, t))
            if mode == "db":
                v = db_lin(v)
            elif mode == "log":
                v = np.log(np.maximum(v, 1e-9))
            out += W[i] * v
        return np.exp(out) if mode == "log" else out

    def ctrl_curve(self, key, mode="lin", smooth=0.0):
        ck = (key, mode, smooth)
        if ck not in self._cache:
            v = self.param(key, self.ctrl_t, mode)
            if smooth > 0:
                v = gaussian_filter1d(v, smooth * SR / CTRL_HOP, mode="nearest")
            self._cache[ck] = v
        return self._cache[ck]

    def ctrl(self, key, mode="lin", smooth=0.0):
        """Parameter curve at audio rate."""
        v = self.ctrl_curve(key, mode, smooth)
        return np.interp(np.arange(self.n) / SR, self.ctrl_t, v)

    def active_span(self, key, thresh=1e-3, pad=0.6):
        g = self.param(key, self.ctrl_t, "db")
        idx = np.nonzero(g > thresh)[0]
        if len(idx) == 0:
            return None
        return max(0.0, self.ctrl_t[idx[0]] - pad), min(self.T, self.ctrl_t[idx[-1]] + pad)

    def at(self, key, t, mode="lin"):
        return float(self.param(key, np.array([t]), mode)[0])


# =============================================================================
# WIND
# =============================================================================
class WindField:
    """Shared gust process: discrete gusts (natural attack/decay envelopes) that move
    across the stereo field. Drives the wind bed, canopy soughing and grass rustle."""

    def __init__(self, tl, rng):
        # renewal process in warped time: integrate the (time-varying) gust rate and start
        # a gust whenever the integral passes a random threshold U(0.5, 1.5). Irregular,
        # but with far less shot-to-shot variance than a Poisson process at these low rates.
        self.events = []
        dt = 0.05
        tg = np.arange(-6.0, tl.T + 1.0, dt)
        rate = tl.param("gust_rate", np.clip(tg, 0, tl.T))
        phase, thr = rng.uniform(0.0, 1.0), rng.uniform(0.5, 1.5)
        for t, r in zip(tg, rate):
            phase += r * dt
            if phase < thr:
                continue
            phase, thr = 0.0, rng.uniform(0.5, 1.5)
            tc = float(np.clip(t, 0, tl.T))
            A = tl.at("gust_depth", tc) * rng.uniform(0.45, 1.0)
            glen = tl.at("gust_len", tc)       # time scale of the gust shape (shorter = gustier)
            ta, td = rng.uniform(1.0, 3.2) * glen, rng.uniform(1.8, 5.5) * glen
            p0 = rng.uniform(-1, 1)
            p1 = float(np.clip(p0 + rng.uniform(-1.3, 1.3), -1, 1))
            self.events.append((t, A, ta, td, p0, p1))

    def env(self, t, ch=None):
        t = np.asarray(t, float)
        g = np.zeros_like(t)
        for (t0, A, ta, td, p0, p1) in self.events:
            x = t - t0
            m = (x > 0) & (x < ta + 7 * td)
            if not m.any():
                continue
            xx = x[m]
            e = np.where(xx < ta, np.sin(0.5 * np.pi * xx / ta) ** 2,
                         np.exp(-(np.maximum(xx - ta, 0) / td) ** 1.3))  # rounded shoulder, natural tail
            if ch is not None:
                p = p0 + (p1 - p0) * np.clip(xx / (ta + td), 0, 1)
                e = e * np.sqrt(1.0 + (0.65 * p if ch == 1 else -0.65 * p))
            g[m] += A * e
        return g


def layer_wind(tl, rng, wf):
    """Wind bed: noise through slowly modulated band-pass (pink/brown tilt), gusts raise
    level and centre frequency; L/R independent noise + gusts panned = decorrelated stereo.
    Optional faint 'ridge whistle' resonance riding strong gusts (high_view)."""
    n = tl.n
    gain = tl.ctrl("wind_db", "db", smooth=0.35)
    g = Grid(n)
    t, fs = g.times, g.fs
    lf = np.log2(fs)
    fc = tl.param("wind_fc", t, "log")
    bw = tl.param("wind_bw", t)
    lp = tl.param("wind_lp", t, "log")
    tilt = tl.param("wind_tilt", t)
    whistle = tl.param("whistle", t)
    hp = 1.0 / np.sqrt(1.0 + (70.0 / fs) ** 6)
    slow = smooth_noise(rng, t, 0.04)
    gus = [wf.env(t, 0), wf.env(t, 1)]
    gm = 0.5 * (gus[0] + gus[1])
    # spectral shape is shared by both channels (one wind field), level/whistle follow each
    # channel's gust envelope; filter_pair gives a diffuse-field L/R coherence
    fcc = fc * 2.0 ** (0.30 * smooth_noise(rng, t, 0.07) + 0.55 * np.minimum(gm, 1.5))
    S0 = (fs[None, :] / 500.0) ** (-tilt[:, None]) * hp[None, :]
    bp = np.exp(-0.5 * ((lf[None, :] - np.log2(fcc)[:, None]) / bw[:, None]) ** 2)
    S0 = Grid.unit(S0 * (0.08 + bp) / np.sqrt(1.0 + (fs[None, :] / lp[:, None]) ** 4))
    fl_common = smooth_noise(rng, t, 1.3)
    base = 0.4 + 0.16 * slow + 0.10 * smooth_noise(rng, t, 0.15)
    fw = 640.0 * 2.0 ** (0.35 * smooth_noise(rng, t, 0.05)) * (1.0 + 0.22 * np.minimum(gm, 1.5))
    Sc = []
    for ch in (0, 1):
        gu = gus[ch]
        fl = 0.8 * fl_common + 0.6 * smooth_noise(rng, t, 1.3)
        flutter = 1.0 + 0.22 * fl * np.clip(0.25 + gu, 0, 1.5)
        lvl = np.maximum(base * WIND_NORM * (WIND_FLOOR + 1.5 * gu) * np.maximum(flutter, 0.3), 0.02)
        S = S0 * lvl[:, None]
        if whistle.max() > 0:
            wamp = 3.0 * whistle * np.minimum(gu, 1.5) ** 2
            S = S + wamp[:, None] * np.exp(-0.5 * ((lf[None, :] - np.log2(fw)[:, None]) / 0.03) ** 2)
        Sc.append(S)
    out = g.filter_pair(Sc[0], rng, COH_D, Sc[1], gain=0.1)
    dry = out * gain
    return dry, dry * 0.3


def noise_grain(rng, dur, f_lo, f_hi, att=0.0006):
    L = max(32, int(dur * SR))
    x = signal.sosfilt(butter_sos_q(2, [f_lo, f_hi], "bandpass"), rng.standard_normal(L + 96))[96:]
    tt = np.arange(L) / SR
    env = (1.0 - np.exp(-tt / att)) * np.exp(-tt / (dur / 3.5))
    return tail_fade(x * env / (np.std(x) + 1e-9), 1.5)


def layer_foliage(tl, rng, wf):
    """Canopy soughing (gust-driven high-mid noise with a living spectral texture) plus
    needle/leaf rustle grains whose density follows the gusts. Grass in the meadow."""
    n = tl.n
    gain = tl.ctrl("rustle_db", "db", smooth=0.3)
    out = np.zeros((2, n))
    span = tl.active_span("rustle_db")
    if span is None:
        return out, out.copy()
    g = Grid(n, 2048, 512)
    t, fs = g.times, g.fs
    fc = tl.param("rustle_fc", t, "log")
    bw = tl.param("rustle_bw", t)
    gm = 0.5 * (wf.env(t, 0) + wf.env(t, 1))
    S = np.exp(-0.5 * ((np.log2(fs)[None, :] - np.log2(fc)[:, None]) / bw[:, None]) ** 2)
    S = Grid.unit(S / np.sqrt(1.0 + (180.0 / fs[None, :]) ** 4))
    Sc = []
    for ch in (0, 1):  # canopy texture differs per side (different trees), shared gusts
        lvl = (0.35 + 1.0 * gm) * np.maximum(0.2, 1.0 + 0.3 * smooth_noise(rng, t, 2.0))
        Sc.append(S * g.seethe(rng, 2.0, 0.5) * lvl[:, None])
    out = g.filter_pair(Sc[0], rng, COH_D, Sc[1], gain=0.1)
    out = norm_active(out, gain, 0.1)
    # rustle grains (density follows gusts)
    grains = np.zeros((2, n))
    ct = tl.ctrl_t
    rate_c = tl.param("grain_rate", ct) * (0.3 + 1.2 * np.interp(ct, t, gm))
    rate_c *= tl.param("rustle_db", ct, "db") > 1e-3
    rmax = float(rate_c.max()) + 1e-6
    fcc = tl.ctrl_curve("rustle_fc", "log")
    tt = span[0]
    while True:
        tt += rng.exponential(1.0 / rmax)
        if tt >= span[1]:
            break
        if rng.random() > np.interp(tt, ct, rate_c) / rmax:
            continue
        f0 = float(np.interp(tt, ct, fcc)) * 2 ** rng.uniform(-0.6, 1.0)
        gr = noise_grain(rng, rng.uniform(0.004, 0.02), f0 / 1.5, min(f0 * 1.6, 12000), att=0.0015)
        add_panned(grains, gr * float(np.exp(rng.normal(0, 0.6))), int(tt * SR), rng.uniform(-0.9, 0.9))
    grains = norm_active(grains, gain, 0.06)
    dry = (out + grains) * gain
    return dry, dry * 0.4


# =============================================================================
# WATER
# =============================================================================
def bubble(f0, amp, xi=0.1, dmul=1.0):
    """Minnaert-style bubble: exponentially decaying sine whose pitch rises as it decays
    (van den Doel). f(t) = f0 (1 + xi*d*t)."""
    d = (0.043 * f0 + 0.0014 * f0 ** 1.5) * dmul
    L = int(min(0.4, 6.0 / d) * SR)
    tt = np.arange(L) / SR
    f = f0 * (1.0 + xi * d * tt)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.exp(-d * tt) * (1.0 - np.exp(-tt / 0.0006))
    return tail_fade(amp * np.sin(ph) * env, 3.0)


def lake_lap(rng, amp, pan):
    """One small wavelet reaching the stones of a small lake (not sea surf): a 2-6 ms slap
    transient, a quick low slosh, a soft short wash, 2-4 hollow 'clucks' (air pockets under
    stones), a light pebble drain with tiny bubbles, sometimes a hollow clop on a boulder."""
    m = int(1.2 * SR)
    tt = np.arange(m) / SR

    def ad(t0, att, tau):
        x = tt - t0
        e = np.where(x < att, np.sin(0.5 * np.pi * np.clip(x, 0, None) / att) ** 2, np.exp(-(x - att) / tau))
        return np.where(x < 0, 0.0, e)

    def band(lo, hi, L=m, order=2):
        y = signal.sosfilt(butter_sos_q(order, [lo, hi], "bandpass"), rng.standard_normal(L + 256))[256:]
        return y / (np.std(y) + 1e-12)

    out = np.zeros((2, m))
    Ls = int(rng.uniform(0.002, 0.006) * SR)
    slap = np.zeros(m)
    slap[:Ls] = band(800, 3000, Ls) * np.hanning(Ls) * rng.uniform(0.3, 0.6)
    fc1 = rng.uniform(220, 420)
    slosh = band(fc1 / 2.0, fc1 * 2.2) * ad(0.0, rng.uniform(0.02, 0.05), rng.uniform(0.08, 0.18))
    d2 = rng.uniform(0.04, 0.1)
    wlo = rng.uniform(700, 1100)
    wash = band(wlo, wlo * rng.uniform(2.5, 3.5)) * ad(d2, rng.uniform(0.04, 0.09), rng.uniform(0.12, 0.25))
    wash *= rng.uniform(0.08, 0.15)
    add_panned(out, slap + slosh + wash, 0, pan)
    for _ in range(int(rng.integers(2, 5))):  # clucks
        b = bubble(rng.uniform(250, 900), rng.uniform(0.5, 1.1), rng.uniform(0.3, 0.6), 2.0)
        add_panned(out, b, int(rng.uniform(0.01, 0.32) * SR), float(np.clip(pan + rng.normal(0, 0.15), -1, 1)))
    # drain through pebbles: high trickle with granular AM
    d3 = d2 + rng.uniform(0.1, 0.25)
    gam = gaussian_filter1d(np.abs(rng.standard_normal(m)), 90) ** 2
    gam /= gam.mean() + 1e-12
    drain = band(1500, 5000, m, 3) * ad(d3, rng.uniform(0.05, 0.1), rng.uniform(0.08, 0.16)) * gam
    drain *= 0.6 * rng.uniform(0.04, 0.08)
    add_panned(out, drain, 0, float(np.clip(pan + rng.uniform(-0.3, 0.3), -1, 1)))
    for _ in range(rng.poisson(6)):  # tiny bubbles
        f0 = float(np.exp(rng.uniform(np.log(900), np.log(4500))))
        b = bubble(f0, rng.uniform(0.05, 0.16), rng.uniform(0.05, 0.25))
        add_panned(out, b, int((d2 + rng.uniform(0.0, 0.45)) * SR), float(np.clip(pan + rng.normal(0, 0.25), -1, 1)))
    if rng.random() < 0.2:  # hollow clop against a boulder
        L = int(0.2 * SR)
        x = np.zeros(L)
        x[:144] = rng.standard_normal(144) * np.hanning(144)
        b, a = signal.iirpeak(rng.uniform(170, 340), rng.uniform(5, 9), fs=SR)
        y = signal.lfilter(b, a, x)
        y = tail_fade(y / (np.max(np.abs(y)) + 1e-12) * rng.uniform(0.15, 0.3), 20)
        add_panned(out, y, int(rng.uniform(0.02, 0.08) * SR), pan)
    k = int(0.15 * SR)
    out[:, -k:] *= np.linspace(1, 0, k) ** 2
    return out * amp


def open_water(tl, rng, span, gain):
    """Open-water texture far from any shore (camera low over the lake): a soft diffuse chop
    bed that breathes with wavelet groups, plus a sparse ripple of tiny bubbles/plips
    (1-4 kHz, 20-40 per second, bursty)."""
    n = tl.n
    g = Grid(n, 2048, 512)
    S = np.exp(-0.5 * (np.log2(g.fs / 1400.0) / 1.4) ** 2) / np.sqrt(1.0 + (250.0 / g.fs) ** 4)
    groups = np.exp(0.5 * smooth_noise(rng, g.times, 0.9) + 0.3 * smooth_noise(rng, g.times, 3.0))
    S = Grid.unit(S[None, :] * np.ones((g.nfr, 1))) * g.seethe(rng, 1.0, 0.5) * groups[:, None]
    bed = norm_active(g.filter_pair(S, rng), gain, 0.05)
    rip = np.zeros((2, n))
    t0, t1 = span[0], min(span[1], tl.T - EVENT_STOP)
    ct = np.arange(t0, tl.T + 1.0, 0.01)
    rate = rng.uniform(20, 40)
    burst = np.exp(0.7 * smooth_noise(rng, ct, 1.5))
    burst /= burst.mean()
    rmax = rate * float(burst.max())
    tt = t0
    while True:
        tt += rng.exponential(1.0 / rmax)
        if tt >= t1:
            break
        if rng.random() > rate * float(np.interp(tt, ct, burst)) / rmax:
            continue
        f0 = float(np.exp(rng.uniform(np.log(1000), np.log(4000))))
        b = bubble(f0, float(np.exp(rng.normal(0, 0.5))), rng.uniform(0.05, 0.3))
        add_panned(rip, b, int(tt * SR), rng.uniform(-0.9, 0.9))
    rip = norm_active(rip, gain, 0.05)
    return bed + rip


def layer_lake(tl, rng, wf):
    """Small-lake lapping at the stones (irregular 0.7-1.8 s, overlapping, gently compressed)
    over a faint low water bed; plus the open-water texture (ripple_db) used when the camera
    is out over the lake."""
    n = tl.n
    gain = tl.ctrl("lake_db", "db", smooth=0.2)
    rgain = tl.ctrl("ripple_db", "db", smooth=0.3)
    dry = np.zeros((2, n))
    if gain.max() >= 1e-4:
        out = np.zeros((2, n))
        gc = tl.ctrl_curve("lake_db", "db")
        t = rng.uniform(0.0, 0.8)
        look = int(1.2 * SR / CTRL_HOP)
        layer_lake.last_lap_end = None
        while t < tl.T - EVENT_STOP:
            j = int(t * SR / CTRL_HOP)
            if max(gc[min(j, len(gc) - 1)], gc[min(j + look, len(gc) - 1)]) > 1e-3:
                amp = float(np.exp(rng.normal(0.0, 0.5))) * (0.45 if rng.random() < 0.3 else 1.0)
                lap = lake_lap(rng, amp, rng.uniform(-0.7, 0.7))
                add_at(out, lap, int(t * SR))
                layer_lake.last_lap_end = t + lap.shape[-1] / SR
            t += tl.at("lake_period", t) * rng.uniform(0.6, 1.5)
        out = norm_active(out, gain, 0.1)
        g = Grid(n)
        S = np.exp(-0.5 * (np.log2(g.fs / 300.0) / 0.9) ** 2)[None, :] * np.ones((g.nfr, 1))
        bed = norm_active(g.filter_pair(Grid.unit(S) * g.seethe(rng, 6.0, 0.45), rng), gain, 0.035)
        tex = tv_lowpass(tame_peaks(out, gain, 6.0) + bed, tl.ctrl_curve("lake_lp", "log"), tl.ctrl_t)
        dry += tex * gain
    span = tl.active_span("ripple_db")
    if span is not None:
        dry += open_water(tl, rng, span, rgain) * rgain
    return dry, dry * 0.2


def layer_waterfall(tl, rng, wf):
    """Waterfall roar: pink noise with an air-absorption low-pass, a 90 Hz rumble bump, a
    proximity-dependent sub high-pass (WF_HP_FAR .. WF_HP), a living time-frequency texture and
    surges (+-2 dB at 0.25 Hz plus faster 1.5 Hz / 4 Hz flicker), low 'mass impact' thumps of
    falling water, bursty splash grains. Proximity sets bandwidth (700 Hz .. 9.5 kHz), rumble,
    thumps, splashes and stereo width; far away it is a narrow, dull rumble at falls_pan."""
    n = tl.n
    gain = tl.ctrl("falls_db", "db", smooth=0.25)
    out = np.zeros((2, n))
    if gain.max() < 1e-4:
        return out, out.copy()
    g = Grid(n)
    t, fs = g.times, g.fs
    prox = tl.param("falls_prox", t)
    fc = 700.0 * 2.0 ** (3.7 * prox)
    S = (fs[None, :] / 1000.0) ** -0.5 / np.sqrt(1.0 + (fs[None, :] / fc[:, None]) ** 2.5)
    S = S * (1.0 + (0.6 * prox)[:, None] * np.exp(-0.5 * (np.log2(fs / 90.0) / 0.8) ** 2)[None, :])
    hpf = WF_HP_FAR * (WF_HP / WF_HP_FAR) ** np.clip(prox / 0.8, 0.0, 1.0)
    S = S / np.sqrt(1.0 + (hpf[:, None] / fs[None, :]) ** 6)
    surge = (1.0 + WF_SURGE[0] * smooth_noise(rng, t, 0.25) + WF_SURGE[1] * smooth_noise(rng, t, 1.5)
             + WF_SURGE[2] * smooth_noise(rng, t, 4.0))
    S = Grid.unit(S) * np.maximum(surge, 0.3)[:, None] * g.seethe(rng, 2.0, 0.28)
    ts = np.arange(n) / SR
    width = np.interp(ts, t, np.clip(0.2 + 0.6 * prox, 0, 1))
    th = (np.interp(ts, t, tl.param("falls_pan", t)) + 1.0) * np.pi / 4.0
    diff = g.filter_pair(S, rng, gain=0.1)          # diffuse part (natural coherence)
    pt = g.filter(rng.standard_normal(n) * 0.1, S)       # point-source part at falls_pan
    a, b = np.sqrt(width), np.sqrt(1.0 - width)
    out[0] = a * diff[0] + b * np.sqrt(2.0) * np.cos(th) * pt
    out[1] = a * diff[1] + b * np.sqrt(2.0) * np.sin(th) * pt
    del diff, pt
    ct = tl.ctrl_t
    on = tl.ctrl_curve("falls_db", "db") > 1e-3
    pc = tl.param("falls_prox", ct)
    sv = np.interp(ct, t, surge)
    pan_c = tl.param("falls_pan", ct)
    # mass impacts: 55-140 Hz noise bursts, 80-200 ms, 1-3 per second, random level
    sos_t = butter_sos(2, [55.0, 140.0], "bandpass")
    if WF_THUMP_HP > 0:
        sos_t = np.vstack([sos_t, butter_sos(3, WF_THUMP_HP, "highpass")])
    tt = 0.0
    while True:
        tt += rng.exponential(1.0 / 3.0)
        if tt >= tl.T:
            break
        p = float(np.interp(tt, ct, pc * on))
        if rng.random() > np.clip((p - 0.25) / 0.5, 0, 1) * 0.7:
            continue
        d = rng.uniform(0.08, 0.2)
        L = int(d * SR)
        x = signal.sosfilt(sos_t, rng.standard_normal(L + 2400))[2400:]
        e = np.arange(L) / SR
        env = (1.0 - np.exp(-e / 0.008)) * np.exp(-e / (d / 3.0))
        y = tail_fade(x / (np.std(x) + 1e-12) * env, 10) * WF_THUMP * float(np.exp(rng.normal(0, 0.6)))
        y *= float(np.interp(tt, ct, sv))
        add_panned(out, y, int(tt * SR), float(np.clip(np.interp(tt, ct, pan_c) + rng.normal(0, 0.2), -1, 1)))
    # splash grains near the fall, bursty (clusters when a surge hits the pool)
    burst = np.exp(0.9 * smooth_noise(rng, ct, 1.2))
    burst /= np.percentile(burst, 90)
    pg = pc ** 2 * on * np.minimum(burst, 1.0)
    rmax = 80.0
    tt = 0.0
    while True:
        tt += rng.exponential(1.0 / rmax)
        if tt >= tl.T:
            break
        if rng.random() > float(np.interp(tt, ct, pg)):
            continue
        gr = noise_grain(rng, rng.uniform(0.015, 0.07), rng.uniform(300, 900), rng.uniform(2500, 7000), att=0.002)
        add_panned(out, gr * 0.025 * float(np.exp(rng.normal(0, 0.5))), int(tt * SR), rng.uniform(-0.8, 0.8))
    dry = out * gain
    return dry, dry * 0.45


def layer_stream(tl, rng, wf):
    """Babbling brook: many short resonant bubbles (decaying, upward-sweeping sines) from
    a few 'spots' (water over particular rocks) with bursty rates, low gloops, small
    splashes, over a gurgling filtered-noise bed."""
    n = tl.n
    gain = tl.ctrl("stream_db", "db", smooth=0.2)
    out = np.zeros((2, n))
    span = tl.active_span("stream_db")
    if span is None:
        return out, out.copy()
    t0, t1 = span[0], min(span[1], tl.T - EVENT_STOP)
    g = Grid(n, 2048, 512)
    S = (np.exp(-0.5 * (np.log2(g.fs / 1600.0) / 1.3) ** 2) + 0.25 * (g.fs / 400.0) ** -0.5)
    S = S / np.sqrt(1.0 + (180.0 / g.fs) ** 4)
    S = Grid.unit(S[None, :] * np.ones((g.nfr, 1)))
    bed = norm_active(g.filter_pair(S * g.seethe(rng, 1.5, 0.55), rng), gain, 0.07)
    bub = np.zeros((2, n))
    ct = np.arange(t0, max(t1, t0 + 0.02), 0.01)
    for _ in range(6):
        pan = rng.uniform(-0.85, 0.85)
        fc = float(np.exp(rng.uniform(np.log(450), np.log(2600))))
        rate = rng.uniform(15, 45)
        spread = rng.uniform(0.25, 0.5)
        burst = np.exp(1.0 * smooth_noise(rng, ct, 2.5))
        burst /= burst.mean()
        rmax = rate * float(burst.max())
        tt = t0
        while True:
            tt += rng.exponential(1.0 / rmax)
            if tt >= t1:
                break
            if rng.random() > rate * float(np.interp(tt, ct, burst)) / rmax:
                continue
            f0 = fc * 2.0 ** rng.normal(0, spread)
            amp = float(np.exp(rng.normal(0, 0.45))) * (f0 / 1000.0) ** -0.4
            add_panned(bub, bubble(f0, amp, rng.uniform(0.05, 0.2)), int(tt * SR),
                       float(np.clip(pan + rng.normal(0, 0.1), -1, 1)))
    tt = t0
    while True:  # low gloops
        tt += rng.exponential(1.0 / 3.0)
        if tt >= t1:
            break
        add_panned(bub, bubble(rng.uniform(180, 450), 0.7 * float(np.exp(rng.normal(0, 0.4))), rng.uniform(0.2, 0.5), 2.0),
                   int(tt * SR), rng.uniform(-0.8, 0.8))
    bub = norm_active(bub, gain, 0.09)
    spl = np.zeros((2, n))
    tt = t0
    while True:  # small splashes over boulders
        tt += rng.exponential(1.0 / 9.0)
        if tt >= t1:
            break
        gr = noise_grain(rng, rng.uniform(0.02, 0.08), rng.uniform(500, 1200), rng.uniform(3000, 7000), att=0.0015)
        add_panned(spl, gr * float(np.exp(rng.normal(0, 0.5))), int(tt * SR), rng.uniform(-0.85, 0.85))
    spl = norm_active(spl, gain, 0.03)
    tex = tv_lowpass(tame_peaks(bed + bub + spl, gain, 10.0, 2.5), tl.ctrl_curve("stream_lp", "log"), tl.ctrl_t)
    dry = tex * gain
    return dry, dry * 0.3




# =============================================================================
# BIRDS
# =============================================================================
def contour(pts, n):
    u = np.linspace(0.0, 1.0, n)
    pu = np.array([p[0] for p in pts], float)
    pf = np.log(np.array([p[1] for p in pts], float))
    if len(pts) <= 2:
        return np.exp(np.interp(u, pu, pf))
    return np.exp(PchipInterpolator(pu, pf)(u))


def bird_note(rng, dur, pts, amp=1.0, harm=(1.0,), vib=None, shape=0.8, att=0.25):
    """Tonal note with a pitch contour (log-PCHIP through pts=[(u, Hz), ...]), fast syrinx
    pitch jitter (0.4 %, ~150 Hz) and amplitude flutter (15 %, ~60 Hz), optional vibrato/FM,
    a few harmonics, 3-5 % breath noise riding on the pitch contour, and an asymmetric
    sin^shape body envelope with raised-cosine edges (>= 2 ms: no clicky note edges)."""
    n = max(48, int(dur * SR))
    f = contour(pts, n)
    tt = np.arange(n) / SR
    f = f * (1.0 + 0.004 * smooth_noise(rng, tt, 150.0))
    if vib:
        f = f * (1.0 + vib[1] * np.sin(2 * np.pi * vib[0] * tt + rng.uniform(0, 6.28)))
    ph = 2 * np.pi * np.cumsum(f) / SR + rng.uniform(0, 6.28)
    y = np.zeros(n)
    for k, h in enumerate(harm, 1):
        if h > 0:
            y += h * np.clip((19000.0 - k * f) / 3000.0, 0, 1) * np.sin(k * ph)
    # breath: narrow-band noise (+-350 Hz) modulated onto the carrier phase
    nz = signal.sosfilt(butter_sos_q(2, 350.0, "lowpass"), rng.standard_normal(n + 256))[256:]
    y += rng.uniform(0.03, 0.05) * nz / (np.std(nz) + 1e-12) * np.sin(ph + 1.1) * np.clip((19000.0 - f) / 3000.0, 0, 1)
    y *= np.maximum(1.0 + 0.15 * smooth_noise(rng, tt, 60.0), 0.2)
    u = (np.arange(n) + 0.5) / n
    uw = np.where(u < att, 0.5 * u / att, 0.5 + 0.5 * (u - att) / (1 - att))
    env = np.sin(np.pi * uw) ** shape
    r = min(n // 3, max(int(0.002 * SR), int(min(0.004, 0.12 * dur) * SR)))
    edge = 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    env[:r] *= edge
    env[-r:] *= edge[::-1]
    return amp * y * env


def assemble(parts, tail=0.05):
    end = max(t + len(a) / SR for t, a in parts) + tail
    y = np.zeros(int(end * SR) + 2)
    for t, a in parts:
        i = int(round(t * SR))
        y[i:i + len(a)] += a
    return y


# -- chaffinch: accelerating descending trill(s) + terminal flourish ---------
def chaffinch_type(rng):
    return dict(f1=rng.uniform(5600, 6800), n1=int(rng.integers(4, 7)), p1=rng.uniform(0.11, 0.14),
                f2=rng.uniform(4300, 5100), n2=int(rng.integers(3, 6)), p2=rng.uniform(0.065, 0.085),
                f3=rng.uniform(3000, 3700), n3=int(rng.integers(0, 4)), fl=rng.uniform(2700, 3300))


def song_chaffinch(rng, ty):
    parts, t = [], 0.0
    p = ty["p1"] * rng.uniform(0.97, 1.03)
    for i in range(ty["n1"]):
        fs = ty["f1"] * (1 - 0.03 * i) * rng.uniform(0.99, 1.01)
        parts.append((t, bird_note(rng, p * 0.6, [(0, fs), (0.3, fs * 0.9), (1, fs * rng.uniform(0.55, 0.65))],
                                   amp=0.55 + 0.45 * min(1.0, (i + 1) / 3.0), harm=(1, 0.05), shape=0.8, att=0.2)))
        t += p
        p *= 0.92
    t += 0.02
    for i in range(ty["n2"]):
        fs = ty["f2"] * (1 - 0.025 * i)
        parts.append((t, bird_note(rng, ty["p2"] * 0.7, [(0, fs), (1, fs * 0.68)], amp=0.9, harm=(1, 0.07), att=0.25)))
        t += ty["p2"]
    for i in range(ty["n3"]):
        f = ty["f3"] * (1 - 0.02 * i)
        parts.append((t, bird_note(rng, 0.06, [(0, f * 0.85), (0.4, f * 1.1), (1, f * 0.8)], amp=0.85,
                                   harm=(1, 0.1), att=0.35)))
        t += 0.09
    t += 0.03
    fl = ty["fl"] * rng.uniform(0.98, 1.02)
    parts.append((t, bird_note(rng, 0.24, [(0, fl * 0.75), (0.2, fl * 1.35), (0.45, fl * 1.6), (0.7, fl * 1.1),
                                           (1, fl * 0.7)], amp=1.0, harm=(1, 0.15, 0.04), vib=(55, 0.04),
                               att=0.45, shape=0.6)))
    return assemble(parts)


# -- robin: varied, wistful warbles from a personal repertoire ---------------
def robin_type(rng):
    kinds = ["seee", "up", "down", "trill", "warble", "low", "pair"]
    rep = []
    for _ in range(12):
        k = kinds[int(rng.choice(len(kinds), p=[0.14, 0.16, 0.16, 0.14, 0.16, 0.12, 0.12]))]
        rep.append(dict(kind=k, f=rng.uniform(0.8, 1.25), d=rng.uniform(0.8, 1.25), vr=rng.uniform(25, 45),
                        vd=rng.uniform(0.06, 0.14), n=int(rng.integers(4, 9)), r=rng.uniform(0.85, 1.0)))
    return dict(rep=rep)


def robin_element(rng, e):
    f, d = e["f"], e["d"]
    k = e["kind"]
    if k == "seee":
        F = 7600 * f
        gl = rng.uniform(0.03, 0.05) * (1 if rng.random() < 0.5 else -1)   # 3-5 % glide
        return bird_note(rng, 0.22 * d, [(0, F * (1 - gl / 2)), (0.6, F * (1 + gl / 4)), (1, F * e["r"] * (1 + gl / 2))],
                         amp=0.45, shape=1.0, att=0.35, vib=(25.0, 0.015))
    if k == "up":
        return bird_note(rng, 0.09 * d, [(0, 2600 * f), (1, 6000 * f)], amp=0.8, att=0.6, shape=0.7, harm=(1, 0.05))
    if k == "down":
        return bird_note(rng, 0.09 * d, [(0, 7000 * f), (0.5, 4800 * f), (1, 3000 * f)], amp=0.8, att=0.3)
    if k == "trill":
        parts, t = [], 0.0
        for i in range(e["n"]):
            F = 4800 * f * (1.12 if i % 2 else 0.92)
            parts.append((t, bird_note(rng, 0.022, [(0, F * 1.1), (1, F * 0.85)], amp=0.6, att=0.3)))
            t += 1.0 / 30.0
        return assemble(parts, 0.005)
    if k == "warble":
        F = 4200 * f
        return bird_note(rng, 0.18 * d, [(0, F * 0.9), (0.5, F * 1.1), (1, F * 0.95)], amp=0.7,
                         vib=(e["vr"], e["vd"]), att=0.3, harm=(1, 0.06))
    if k == "low":
        F = 2600 * f
        return bird_note(rng, 0.11 * d, [(0, F), (0.5, F * 1.15), (1, F * 0.9)], amp=0.6, harm=(1, 0.18, 0.05), att=0.4)
    F = 4500 * f
    a = bird_note(rng, 0.05, [(0, F * 0.9), (1, F * 1.2)], amp=0.7, att=0.5)
    b = bird_note(rng, 0.06, [(0, F * 1.25), (1, F * 0.85)], amp=0.7, att=0.3)
    return assemble([(0, a), (0.07, b)], 0.005)


def song_robin(rng, ty):
    rep = ty["rep"]
    parts, t = [], 0.0
    seq = [rep[int(i)] for i in rng.choice(len(rep), int(rng.integers(5, 10)))]
    if rng.random() < 0.5:
        seq.insert(0, dict(kind="seee", f=rng.uniform(0.95, 1.1), d=1.0, vr=0, vd=0, n=0, r=rng.uniform(0.9, 1.0)))
    for e in seq:
        a = robin_element(rng, e) * rng.uniform(0.6, 1.0)
        parts.append((t, a))
        t += len(a) / SR + rng.uniform(0.012, 0.07)
    return assemble(parts)


# -- thrush (blackbird / ring-ouzel like): mellow fluted phrase + soft twitter --
def thrush_type(rng):
    base = rng.uniform(1600, 2200)
    motifs = []
    for _ in range(5):
        notes = []
        for _ in range(int(rng.integers(3, 6))):
            f0 = base * 2 ** rng.uniform(-0.45, 0.55)
            shp = int(rng.integers(4))
            pts = [[(0, f0), (1, f0 * 1.02)], [(0, f0 * 0.85), (1, f0 * 1.12)],
                   [(0, f0 * 1.12), (1, f0 * 0.86)], [(0, f0 * 0.9), (0.45, f0 * 1.12), (1, f0 * 0.92)]][shp]
            notes.append((rng.uniform(0.11, 0.32), pts, rng.uniform(0.03, 0.11), rng.uniform(0.6, 1.0)))
        motifs.append(dict(notes=notes, twitter=rng.random() < 0.65, tw_f=rng.uniform(4200, 6200)))
    return dict(motifs=motifs)


def song_thrush(rng, ty):
    m = ty["motifs"][int(rng.integers(len(ty["motifs"])))]
    parts, t = [], 0.0
    for d, pts, gap, a in m["notes"]:
        parts.append((t, bird_note(rng, d, pts, amp=a, harm=(1, 0.22, 0.07), vib=(rng.uniform(5, 8), 0.006),
                                   shape=0.9, att=0.3)))
        t += d + gap
    if m["twitter"]:
        t += 0.05
        for _ in range(int(rng.integers(3, 7))):
            f = m["tw_f"] * rng.uniform(0.85, 1.15)
            d = rng.uniform(0.025, 0.05)
            parts.append((t, bird_note(rng, d, [(0, f), (1, f * rng.uniform(0.7, 1.3))], amp=0.22, harm=(1, 0.05), att=0.3)))
            t += d + rng.uniform(0.015, 0.04)
    return assemble(parts)


# -- great tit "tee-cha tee-cha" ---------------------------------------------
def tit_type(rng):
    return dict(fh=rng.uniform(6000, 7000), fl=rng.uniform(3700, 4400), per=rng.uniform(0.26, 0.34),
                dh=rng.uniform(0.06, 0.085), dl=rng.uniform(0.07, 0.1))


def song_tit(rng, ty):
    parts, t = [], 0.0
    fh, fl = ty["fh"], ty["fl"]
    for k in range(int(rng.integers(3, 7))):
        a = 0.8 * (1 - 0.06 * k)
        parts.append((t, bird_note(rng, ty["dh"], [(0, fh * 1.02), (1, fh * 0.95)], amp=a, harm=(1, 0.04), att=0.25)))
        parts.append((t + ty["dh"] + 0.03, bird_note(rng, ty["dl"], [(0, fl * 1.06), (0.4, fl), (1, fl * 0.9)],
                                                     amp=a * 0.9, harm=(1, 0.1), att=0.3)))
        t += ty["per"] * rng.uniform(0.97, 1.03)
    return assemble(parts)


# -- thin 'tsee' contact calls (goldcrest / coal tit in the conifers) ---------
def tsit_type(rng):
    return dict(f=rng.uniform(7200, 8800))


def song_tsit(rng, ty):
    parts, t = [], 0.0
    for _ in range(int(rng.integers(1, 4))):
        f = ty["f"] * rng.uniform(0.97, 1.03)
        d = rng.uniform(0.03, 0.06)
        parts.append((t, bird_note(rng, d, [(0, f * 1.03), (1, f * 0.9)], amp=0.5, att=0.3, shape=1.0)))
        t += d + rng.uniform(0.08, 0.25)
    return assemble(parts)


def woodpecker_drum(rng):
    """Distant drumming: 10-16 separate, slightly decelerating knocks with a decrescendo.
    Each knock = a few very short damped wood modes (+-4 % jitter per knock) and a click,
    low-passed at 3 kHz (distance)."""
    parts, t = [], 0.0
    p = rng.uniform(0.05, 0.06)
    nk = int(rng.integers(10, 17))
    modes = [(rng.uniform(380, 520), 0.008, 1.0), (rng.uniform(900, 1200), 0.005, 0.6), (rng.uniform(1900, 2500), 0.003, 0.35)]
    L = int(0.045 * SR)
    tt = np.arange(L) / SR
    hp = butter_sos(2, 700, "highpass")
    for k in range(nk):
        y = sum(a * np.sin(2 * np.pi * f * rng.uniform(0.96, 1.04) * tt + rng.uniform(0, 6.28)) * np.exp(-tt / tau)
                for f, tau, a in modes)
        y = y + 0.4 * signal.sosfilt(hp, rng.standard_normal(L)) * np.exp(-tt / 0.0015)
        y *= 1.0 - np.exp(-tt / 0.0006)
        dyn = (1.0 - 0.6 * k / nk) * (0.75 + 0.25 * min(1.0, k / 2.0))   # quick swell, then decrescendo
        parts.append((t, tail_fade(y * dyn * rng.uniform(0.85, 1.0), 5)))
        t += p
        p *= 1.03
    return signal.sosfilt(butter_sos(2, 3000, "lowpass"), assemble(parts))


SPECIES = {  # interval between songs of one individual (s), relative level, makers
    "chaffinch": dict(interval=(7.0, 13.0), level=1.0, typ=chaffinch_type, song=song_chaffinch),
    "robin": dict(interval=(3.5, 7.0), level=0.75, typ=robin_type, song=song_robin),
    "thrush": dict(interval=(4.0, 9.0), level=0.9, typ=thrush_type, song=song_thrush),
    "tit": dict(interval=(6.0, 15.0), level=0.8, typ=tit_type, song=song_tit),
    "tsit": dict(interval=(2.5, 6.0), level=0.35, typ=tsit_type, song=song_tsit),
}
MAX_BIRD_OVERLAP = 2
BIRD_CAP_DB = 5.0    # a song's momentary max may exceed the local ambience short-term level by this


def momentary_max(y):
    """Max BS.1770 momentary loudness (0.4 s blocks) of a mono signal (an equal-power
    panned stereo copy has the same loudness)."""
    B = int(0.4 * SR)
    yk = an.k_weight(np.pad(y, (0, max(0, B - len(y)))))
    p, _ = an._block_power(yk, 0.4, 0.02)
    return float(-0.691 + 10.0 * np.log10(np.max(p) + 1e-20))


def place(dry, send, mono, t0, pan, dist, amp, cap_lufs=None):
    """Spatialise a mono event: air-absorption low-pass, 1/r level, equal-power pan with a
    small interaural delay, and a reverb send that falls off slower than the direct sound.
    cap_lufs: optional ceiling for the direct sound's momentary loudness. Returns the
    placed direct sound's momentary max (LUFS, pre-master scale)."""
    fc = float(np.clip(16000.0 * (10.0 / dist) ** 0.5, 2200.0, 17000.0))
    y = signal.sosfilt(butter_sos(2, fc, "lowpass"), mono) if fc < 16500 else mono
    y = tail_fade(y.copy(), 3)
    gdir = amp * min(1.0, 10.0 / dist)
    lm = momentary_max(y) + 20.0 * math.log10(max(gdir, 1e-12))
    if cap_lufs is not None and lm > cap_lufs:
        red = 10.0 ** ((cap_lufs - lm) / 20.0)
        amp, gdir, lm = amp * red, gdir * red, cap_lufs
    gl, gr = pan_gains(pan)
    dly = int(round(abs(pan) * 0.00035 * SR))
    L, R = y * gl * gdir, y * gr * gdir
    if dly:
        if pan > 0:
            L = np.concatenate([np.zeros(dly), L])
        else:
            R = np.concatenate([np.zeros(dly), R])
    i0 = int(round(t0 * SR))
    _add_ch(dry, 0, L, i0)
    _add_ch(dry, 1, R, i0)
    s = float(np.clip(0.08 + dist / 250.0, 0.08, 0.6)) * amp * min(1.0, 10.0 / dist) ** 0.5
    _add_ch(send, 0, y * gl * s, i0)
    _add_ch(send, 1, y * gr * s, i0)
    return lm


def _add_ch(buf, ch, x, i0):
    n = buf.shape[-1]
    a, b = max(0, i0), min(n, i0 + len(x))
    if b > a:
        buf[ch, a:b] += x[a - i0:b - i0]


def layer_birds(tl, rng, wf, amb=None):
    """Per-shot bird populations (each shot is a different place): individuals with a
    fixed song type, position and distance sing at natural intervals; sparse (at most
    MAX_BIRD_OVERLAP songs at once), J-cut aware via the shot weights. amb = (t, LUFS)
    short-term loudness of the rest of the nature bed: each song is capped at
    amb + BIRD_CAP_DB so a close bird never jumps out of the ambience. No song starts
    after T - EVENT_STOP or ends after T - 1.5 s."""
    n = tl.n
    dry, send = np.zeros((2, n)), np.zeros((2, n))
    cands = []
    for i, P in enumerate(tl.params):
        lo, hi = tl.a[i] - tl.h[i], tl.b[i] + tl.h[i + 1]
        for sp, S in SPECIES.items():
            dens = float(P["b_" + sp])
            cnt = int(dens) + int(rng.random() < dens - int(dens))
            for _ in range(cnt):
                ind = dict(sp=sp, ty=S["typ"](rng), pan=rng.uniform(-0.85, 0.85),
                           dist=float(np.exp(rng.uniform(np.log(P["bird_dmin"]), np.log(P["bird_dmax"])))))
                t = lo + rng.uniform(0.1, 0.8) * rng.uniform(*S["interval"])
                while t < hi:
                    cands.append((t, i, ind, rng.random()))
                    t += rng.uniform(*S["interval"])
        if P["woodpecker"] >= 0.5:
            t = tl.a[i] + rng.uniform(0.15, 0.5) * (tl.b[i] - tl.a[i])
            cands.append((t, i, dict(sp="woodpecker", pan=rng.uniform(-0.8, 0.8), dist=rng.uniform(70, 120)), 0.0))
    cands.sort(key=lambda c: c[0])
    busy = []
    self_stats = []
    for t, i, ind, _ in cands:
        if ind["sp"] == "woodpecker":
            mono = woodpecker_drum(rng) * 0.6
        else:
            mono = SPECIES[ind["sp"]]["song"](rng, ind["ty"]) * SPECIES[ind["sp"]]["level"]
        dur = len(mono) / SR
        w = float(tl.weights(np.array([t, t + dur]))[i].min())
        if w < 0.2 or t < 0.3 or t > tl.T - EVENT_STOP or t + dur > tl.T - 1.5:
            continue
        if ind["sp"] != "woodpecker" and sum(1 for s, e in busy if s < t + dur and e > t) >= MAX_BIRD_OVERLAP:
            continue
        busy.append((t, t + dur))
        amp = w * tl.at("birds_db", t + dur / 2, "db") * rng.uniform(0.7, 1.0) * 0.5
        pan = float(np.clip(ind["pan"] + rng.normal(0, 0.05), -0.95, 0.95))
        cap = None
        if amb is not None:
            cap = float(np.mean(np.interp(np.linspace(t, t + dur, 5), amb[0], amb[1]))) + BIRD_CAP_DB
        lm = place(dry, send, mono, t, pan, ind["dist"], amp, cap)
        self_stats.append((round(t, 2), ind["sp"], round(lm, 1), None if cap is None else round(cap, 1)))
    layer_birds.stats = self_stats
    return dry, send




# =============================================================================
# MEADOW insects
# =============================================================================
def grasshopper_track(rng, t0, t1):
    """Field-grasshopper-like stridulation: bouts of chirps, each chirp a train of
    leg-stroke pulses (short noise ticks), band-passed high."""
    m = int((t1 - t0) * SR) + 1
    y = np.zeros(m)
    fcen = rng.uniform(7000, 11000)
    prate = rng.uniform(30, 48)
    clen = rng.uniform(0.15, 0.32)
    cgap = rng.uniform(0.25, 0.6)
    PL = int(0.004 * SR)
    pt = np.arange(PL) / SR
    penv = np.exp(-pt / 0.0012) * (1 - np.exp(-pt / 0.0002))
    t = rng.uniform(0, 3.0)
    while t < t1 - t0:
        for _ in range(int(rng.integers(4, 11))):
            npl = max(3, int(clen * prate))
            for p in range(npl):
                tp = t + p / prate * (1 + 0.05 * rng.normal())
                i = int(tp * SR)
                if 0 <= i < m - PL:
                    y[i:i + PL] += rng.standard_normal(PL) * penv * np.sin(np.pi * (p + 0.5) / npl) ** 0.7 * rng.uniform(0.7, 1)
            t += clen + cgap * rng.uniform(0.85, 1.15)
        t += rng.uniform(1.0, 4.0)
    return signal.sosfilt(butter_sos(2, [fcen / 1.7, min(fcen * 1.6, 20000)], "bandpass"), y)


def bee_flyby(rng, dur):
    """A bumblebee passing by: harmonic buzz at the wingbeat rate with Doppler, 1/r level
    and a moving pan."""
    n = int(dur * SR)
    tt = np.arange(n) / SR
    v, d0 = rng.uniform(2.0, 3.5), rng.uniform(1.2, 3.0)
    side = 1 if rng.random() < 0.5 else -1
    x = side * v * (tt - dur / 2) + 0.3 * smooth_noise(rng, tt, 0.8)          # meandering path
    yv = d0 + 0.35 * smooth_noise(rng, tt, 0.6)
    r = np.sqrt(x * x + np.maximum(yv, 0.5) ** 2)
    vr = np.gradient(r) * SR
    # wingbeat wavers with manoeuvres (+-3 %), Doppler from the radial velocity
    f0 = rng.uniform(150, 200) * (1 + 0.03 * smooth_noise(rng, tt, 2.5)) * 343.0 / (343.0 + vr)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    w = np.zeros(n)
    for k in range(1, 25):
        w += k ** -1.3 * np.clip((7000 - k * f0) / 1500, 0, 1) * np.sin(k * ph + 0.3 * k)
    w += 0.15 * rng.standard_normal(n) * (0.5 + 0.5 * np.sin(ph))
    w = signal.sosfilt(butter_sos(2, 4000, "lowpass"), w)
    fade = np.sin(np.pi * tt / dur) ** 1.5        # gradual approach and departure
    a = fade * (1.0 / r) * (1 + 0.25 * smooth_noise(rng, tt, 4.0)).clip(0.3, None)
    p = np.clip(x / r, -1, 1) * 0.9
    gl, gr = pan_gains(p)
    return np.vstack([w * a * gl, w * a * gr]) * 0.012


def layer_meadow(tl, rng, wf):
    """Very subtle insect life: a few grasshoppers, a faint high chorus, rare bee fly-bys."""
    n = tl.n
    gain = tl.ctrl("insects_db", "db", smooth=0.3)
    out = np.zeros((2, n))
    span = tl.active_span("insects_db")
    if span is None:
        return out, out.copy()
    t0, t1 = span
    i0 = int(t0 * SR)
    for _ in range(5):
        tr = grasshopper_track(rng, t0, t1)
        tr /= np.max(np.abs(tr)) + 1e-12  # sparse pulses: normalise by peak, not RMS
        dist = rng.uniform(2.0, 18.0)
        tr = signal.sosfilt(butter_sos(2, float(np.clip(18000 * (3 / dist) ** 0.4, 6000, 18000)), "lowpass"), tr)
        add_panned(out, tr * 0.25 * (2.0 / dist) ** 0.7, i0, rng.uniform(-0.9, 0.9))
    g = Grid(n, 2048, 512)
    S = np.exp(-0.5 * (np.log2(g.fs / 7500.0) / 0.5) ** 2)
    chorus = np.vstack([g.filter(rng.standard_normal(n), Grid.unit(S[None, :] * np.ones((g.nfr, 1)))
                                 * (0.6 + 0.4 * np.tanh(smooth_noise(rng, g.times, 0.4)))[:, None]) for _ in (0, 1)])
    out = out + norm_active(chorus, gain, 0.004)
    for i, P in enumerate(tl.params):
        nb = int(P["bee"] >= 0.5) + int(P["bee"] >= 1.5)
        for k in range(nb):
            dur = rng.uniform(2.2, 3.2)
            t = tl.a[i] + rng.uniform(0.15, 0.6) * max(0.1, tl.b[i] - tl.a[i] - dur)
            add_at(out, bee_flyby(rng, dur), int(t * SR))
    dry = out * gain
    return dry, dry * 0.25


# =============================================================================
# MUSIC: warm slow pad over a D/A open fifth + sparse soft bell notes
# =============================================================================
PEDAL = [38, 45]                             # D2 A2 open fifth under every chord
CHORD_TONIC = PEDAL + [54, 57, 64]           # Dadd9          | F#3 A3 E4
CHORD_END = PEDAL + [50, 54, 57, 64]         # Dadd9 (open)   | D3 F#3 A3 E4
CHORD_DOM = PEDAL + [52, 57, 59, 64]         # Asus2/D        | E3 A3 B3 E4
PROG = [
    PEDAL + [54, 59, 62, 64],   # Bm7(add11)/D  | F#3 B3 D4 E4
    PEDAL + [55, 59, 62, 69],   # Gadd9/D       | G3 B3 D4 A4
    PEDAL + [52, 55, 59, 62],   # Em7/D         | E3 G3 B3 D4
    PEDAL + [54, 57, 62, 64],   # Dadd9 (high)  | F#3 A3 D4 E4
]
SHOTS_PER_CHORD = 2
PAD_DETUNE_CENTS = 2.0   # SD of the Gaussian voice detune


def chord_plan(k):
    """One chord per SHOTS_PER_CHORD shots: tonic, PROG..., dominant suspension, tonic."""
    m = max(1, int(math.ceil(k / SHOTS_PER_CHORD)))
    if m == 1:
        return [CHORD_END]
    if m == 2:
        return [CHORD_TONIC, CHORD_END]
    if m == 3:
        return [CHORD_TONIC, CHORD_DOM, CHORD_END]
    return [CHORD_TONIC] + [PROG[i % len(PROG)] for i in range(m - 3)] + [CHORD_DOM, CHORD_END]


def midi_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def chord_windows(tl):
    """(start, end, first shot, last shot) of each chord: shots grouped in pairs."""
    k = len(tl.shots)
    out = []
    for c in range(int(math.ceil(k / SHOTS_PER_CHORD))):
        i0, i1 = c * SHOTS_PER_CHORD, min(k - 1, c * SHOTS_PER_CHORD + SHOTS_PER_CHORD - 1)
        out.append((tl.a[i0], tl.b[i1], i0, i1))
    return out


def chord_env(win, c, t):
    a, b = win[c][0], win[c][1]
    rise = smoothstep(t / 3.5) if c == 0 else smoothstep((t - (a - 1.6)) / 3.6)
    fall = np.ones_like(t) if c == len(win) - 1 else 1.0 - smoothstep((t - (b - 1.0)) / 4.5)
    return rise * fall


def bell_tone(rng, f, dur=7.0):
    """Soft mallet bar: inharmonic free-bar partials (1, 2.76, 5.40, 8.93) with independent
    decays; no near-unison partial pairs, so no beating/tremolo."""
    n = int(dur * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    for r, a, tau in ((1.0, 1.0, 4.0), (2.76, 0.25, 1.2), (5.40, 0.08, 0.5), (8.93, 0.03, 0.25)):
        if f * r < 15000:
            y += a * np.sin(2 * np.pi * f * r * tt + rng.uniform(0, 6.28)) * np.exp(-tt / tau)
    y *= 1.0 - np.exp(-tt / 0.005)
    y = signal.sosfilt(butter_sos(2, 6000, "lowpass"), y)
    return tail_fade(y, 300)


def layer_music(tl, rng):
    """Pad: 5-6 voices per note (Gaussian detune, SD 2 cents; own random wavetable per
    voice), chords changing every second shot over a sustained D/A open fifth, common tones
    held, level-tracking brightness (low-pass 900 Hz + 1800 Hz * envelope). Bells: very sparse
    soft mallet notes from the current chord. Returns (pad_dry, pad_send, bell_dry, bell_send)."""
    n = tl.n
    ct = tl.ctrl_t
    chords = chord_plan(len(tl.shots))
    win = chord_windows(tl)
    shot_chord = {i: c for c, w in enumerate(win) for i in range(w[2], w[3] + 1)}
    notes = {}
    for c, ch in enumerate(chords):
        ce = chord_env(win, c, ct)
        lowest, top = min(ch), max(ch)
        for m in ch:
            va = 0.55 if m == lowest else (0.3 if m == top else 0.42)
            notes[m] = np.maximum(notes.get(m, np.zeros_like(ct)), ce * va)
    # common tones are held through a chord change (no dip in the crossfade)
    for c in range(len(chords) - 1):
        b = win[c][1]
        msk = (ct >= b - 1.6) & (ct <= b + 3.5)
        for m in set(chords[c]) & set(chords[c + 1]):
            lv = min(0.55 if m == min(chords[c]) else (0.3 if m == max(chords[c]) else 0.42),
                     0.55 if m == min(chords[c + 1]) else (0.3 if m == max(chords[c + 1]) else 0.42))
            notes[m][msk] = np.maximum(notes[m][msk], lv)
    DS = 4                      # pad is band-limited (< 3.2 kHz): synthesise at SR/4, upsample
    SRL = SR // DS
    nl = n // DS + 2
    pad = np.zeros((2, nl))
    TBL = 4096
    xs = np.arange(TBL) / TBL
    for m, env in sorted(notes.items()):
        on = env > 1e-4
        if not on.any():
            continue
        edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
        runs = list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1))
        f = midi_hz(m)
        K = int(max(1, min(8, 3200 // f)))
        nv = 5 if f < 100 else 6
        vg = math.sqrt(3.0 / nv)
        bass_w = 0.6 if f < 100 else 1.0
        pans = np.zeros(nv) if f < 150 else rng.permutation(np.linspace(-0.85, 0.85, nv)) + rng.normal(0, 0.05, nv)
        voices = []
        for v in range(nv):  # each voice: its own detune and its own random partial weights
            w = np.arange(1, K + 1, dtype=float) ** -2.0 * np.exp(0.3 * rng.standard_normal(K))
            tab = sum(w[k - 1] * np.sin(2 * np.pi * k * xs + (0.0 if k == 1 else rng.uniform(0, 6.28)))
                      for k in range(1, K + 1))
            voices.append((np.append(tab, tab[0]), rng.normal(0.0, PAD_DETUNE_CENTS), float(np.clip(pans[v], -1, 1))))
        for r0, r1 in runs:
            s, e = int(ct[r0] * SRL), min(nl, int(ct[r1] * SRL) + 1)
            if e - s < 64:
                continue
            tt = np.arange(s, e) / SRL
            envs = np.interp(tt, ct, env) * (1.0 + 0.10 * smooth_noise(rng, tt, 0.12))
            for table, cents, pan in voices:
                fv = f * 2 ** (cents / 1200.0) * (1.0 + 0.0005 * smooth_noise(rng, tt, 0.07))
                pos = ((np.cumsum(fv) / SRL + rng.random()) % 1.0) * TBL
                i0 = pos.astype(int)
                fr = pos - i0
                w = (table[i0] * (1 - fr) + table[i0 + 1] * fr) * envs * (bass_w * vg)
                gl, gr = pan_gains(pan)
                pad[0, s:e] += w * gl
                pad[1, s:e] += w * gr
    pad = signal.resample_poly(pad, DS, 1, axis=-1)[:, :n]
    # brightness follows the musical level (pad_db swells): duller when soft
    pl = tl.ctrl_curve("pad_db", "db", smooth=1.2) * np.max([chord_env(win, c, ct) for c in range(len(win))], axis=0)
    benv = np.clip(pl / (pl.max() + 1e-12), 0.0, 1.0)
    pad = tv_lowpass(pad, 900.0 + 1800.0 * benv, ct)
    pad *= tl.ctrl("pad_db", "db", smooth=1.2)
    # bells
    bells = np.zeros((2, n))
    placed = []
    latest = tl.T - FADE_OUT_NATURE
    for i, P in enumerate(tl.params):
        a, b = tl.a[i], tl.b[i]
        pcs = sorted(set(m % 12 for m in chords[shot_chord[i]]))
        pool = [m for m in range(74, 89) if m % 12 in pcs]
        L = b - a
        lvl = float(np.max(db_lin(P["pad_db"] if not isinstance(P["pad_db"], list) else max(P["pad_db"]))))
        evs = []
        if P["bell"] >= 1.0 and i == len(tl.params) - 1:
            # closing figure: starts at a + 5 % of the window (inside the finale, after the J-cut),
            # notes 1.1-1.4 s apart, as many (2-3) as fit by T - 4.5 s; descends onto the tonic D
            start = a + 0.05 * L
            gaps = list(rng.uniform(1.1, 1.4, size=2))
            while len(gaps) > 1 and start + sum(gaps) > latest:
                gaps.pop()
            if start + sum(gaps) > latest:          # very short last shot
                gaps = [1.1]
                start = max(a, latest - 1.1)
            tonic = [m for m in pool if m % 12 == 2]
            low = min(tonic) if tonic else min(pool)
            upper = [m for m in pool if m > low]
            ms = sorted((int(m) for m in rng.choice(upper, size=min(len(gaps), len(upper)), replace=False)),
                        reverse=True) + [low]
            times = start + np.concatenate([[0.0], np.cumsum(gaps)])
            evs = list(zip(times[-len(ms):], ms))
        elif P["bell"] >= 1.0:
            t = a + 0.62 * L
            m1 = int(rng.choice(pool))
            evs = [(t, m1)]
            lower = [m for m in pool if m < m1]
            if lower:
                evs.append((t + rng.uniform(0.8, 1.2), int(rng.choice(lower))))
        elif rng.random() < P["bell"]:
            evs = [(a + rng.uniform(0.3, 0.7) * L, int(rng.choice(pool)))]
        for t, m in evs:
            if t > latest + 1e-6:
                continue
            placed.append((round(float(t), 2), int(m)))
            add_panned(bells, bell_tone(rng, midi_hz(m)) * 0.28 * BELL_GAIN * lvl * rng.uniform(0.8, 1.0),
                       int(t * SR), rng.uniform(-0.45, 0.45))
    layer_music.bell_events = placed
    return pad * 0.75, pad * 0.65, bells * 0.55, bells * 0.9




# =============================================================================
# REVERB (synthesised impulse responses)
# =============================================================================
BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def make_ir(rng, rt60, band_db, length, predelay, onset, er=0, er_span=0.05, echoes=()):
    """Stereo IR: per-octave-band decaying noise (frequency-dependent RT60), smooth
    onset build-up after a pre-delay, sparse early reflections, optional discrete
    smeared echoes (valley walls). Normalised to unit energy per channel."""
    n = int(length * SR)
    tt = np.arange(n) / SR
    ir = np.zeros((2, n))
    # 63/125/250 Hz bands use the same noise in both channels: a real room/valley tail is
    # coherent at low frequencies (wavelength >> ear/mic spacing)
    shared = {fc: rng.standard_normal(n) for fc in BANDS[:3]}
    for ch in (0, 1):
        acc = np.zeros(n)
        for fc, T60, gdb in zip(BANDS, rt60, band_db):
            if fc == BANDS[0]:
                sos = butter_sos(2, fc * 1.414, "lowpass")
            elif fc == BANDS[-1]:
                sos = signal.butter(2, fc / 1.414, "highpass", fs=SR, output="sos")
            else:
                sos = butter_sos(2, [fc / 1.414, fc * 1.414], "bandpass")
            # independent noise per band: band powers add, no coherent crossover ripple
            src = shared[fc] if fc in shared else rng.standard_normal(n)
            acc += signal.sosfilt(sos, src) * 10 ** (gdb / 20.0) * np.exp(-6.908 * tt / T60)
        env = np.clip((tt - predelay) / onset, 0.0, 1.0) ** 1.5
        acc *= env
        for _ in range(er):
            pos = predelay + rng.uniform(0.0, er_span)
            k = int(pos * SR)
            acc[k:k + 48] += rng.choice([-1, 1]) * rng.uniform(0.6, 2.0) * np.exp(-pos / 0.06) * np.hanning(48)
        ir[ch] = acc
    E = np.sum(ir ** 2, axis=1, keepdims=True)
    for delay, gdb, lp in echoes:
        for ch in (0, 1):
            L = int(0.12 * SR)
            bt = np.arange(L) / SR
            burst = signal.sosfilt(butter_sos(2, lp, "lowpass"), rng.standard_normal(L)) * np.exp(-bt / 0.025)
            burst *= 1 - np.exp(-bt / 0.004)
            burst *= np.sqrt(E[ch, 0] * 10 ** (gdb / 10.0) / np.sum(burst ** 2))
            k = int((delay + rng.uniform(-0.01, 0.01)) * SR)
            ir[ch, k:k + L] += burst[: max(0, min(L, n - k))]
    ir[:, -int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    return ir / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))


def ir_valley(rng):
    return make_ir(rng, rt60=[2.3, 2.6, 2.8, 2.6, 2.3, 1.6, 1.0, 0.6, 0.35],
                   band_db=[-8, -3, 0, 0, 0, -2, -5, -9, -15], length=3.4, predelay=0.03, onset=0.06,
                   er=6, er_span=0.08, echoes=((0.31, -15, 2500), (0.58, -19, 1800), (0.93, -24, 1400)))


def ir_forest(rng):
    return make_ir(rng, rt60=[0.8, 0.9, 1.0, 1.1, 1.1, 1.0, 0.8, 0.55, 0.35],
                   band_db=[-8, -4, -1, 0, 0, -1, -2, -5, -10], length=1.8, predelay=0.008, onset=0.02,
                   er=18, er_span=0.06)


def convolve_stereo(x, ir):
    n = x.shape[-1]
    inL = 0.75 * x[0] + 0.25 * x[1]
    inR = 0.25 * x[0] + 0.75 * x[1]
    return np.vstack([signal.fftconvolve(inL, ir[0])[:n], signal.fftconvolve(inR, ir[1])[:n]])


# =============================================================================
# MASTER
# =============================================================================
def bus_compressor(x, thr_db=-21.0, ratio=1.5, knee_db=10.0, att=0.05, rel=0.6, block=240):
    n = x.shape[-1]
    nb = int(math.ceil(n / block))
    p = np.pad(x, ((0, 0), (0, nb * block - n))) ** 2
    lvl = 10 * np.log10(p.reshape(2, nb, block).mean(axis=2).mean(axis=0) + 1e-12)
    aa, ar = math.exp(-block / (att * SR)), math.exp(-block / (rel * SR))
    sm = np.empty(nb)
    s = lvl[0]
    for i in range(nb):
        v = lvl[i]
        c = aa if v > s else ar
        s = c * s + (1 - c) * v
        sm[i] = s
    over = sm - thr_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    g = 10 ** (-gr / 20.0)
    return x * np.interp(np.arange(n), (np.arange(nb) + 0.5) * block, g), float(gr.max())


def limiter(x, ceiling_db, look_ms=3.0, rel_ms=120.0):
    """Look-ahead peak limiter: required gain -> min over +-lookahead -> moving average
    (guarantees the ceiling) -> block-wise release smoothing."""
    c = 10 ** (ceiling_db / 20.0)
    pk = np.max(np.abs(x), axis=0)
    req = np.minimum(1.0, c / np.maximum(pk, 1e-12))
    L = int(look_ms * 1e-3 * SR)
    gmin = minimum_filter1d(req, 2 * L + 1)
    gs = uniform_filter1d(gmin, L + 1)
    B = 32
    nb = int(math.ceil(len(gs) / B))
    gb = np.pad(gs, (0, nb * B - len(gs)), constant_values=1.0).reshape(nb, B).min(axis=1)
    a = math.exp(-B / (rel_ms * 1e-3 * SR))
    r = np.empty(nb)
    v = 1.0
    for i in range(nb):
        v = gb[i] if gb[i] < v else a * v + (1 - a) * gb[i]
        r[i] = v
    g = np.minimum(gs, np.interp(np.arange(len(gs)), np.arange(nb) * B + B / 2, r))
    y = x * g
    limiter.last_gr_time = float(np.argmin(g) / SR)
    limiter.last_gain = g
    return np.clip(y, -c, c), float(20 * np.log10(g.min()))


def fades(n, fin=FADE_IN, fout_nature=FADE_OUT_NATURE, fout_master=FADE_OUT):
    """Fade curves (nature_bus_gain, music_bus_gain): both share the fade-in and the short
    master fade-out; the nature bus additionally fades over the last fout_nature seconds, so
    the ambience recedes first and the music (and its reverb tail) closes the film."""
    t = np.arange(n) / SR
    T = n / SR
    gi = smoothstep(t / fin) ** 1.3

    def out(fo):
        return np.cos(0.5 * np.pi * np.clip((t - (T - fo)) / fo, 0, 1)) ** 2

    gm = gi * out(fout_master)
    return gm * out(fout_nature), gm


def lufs(x):
    return an.integrated_lufs(x)


def master(mix, target=TARGET_LUFS, ceiling=CEILING_DBTP):
    info = {}
    x = signal.sosfilt(signal.butter(4, 25.0, "highpass", fs=SR, output="sos"), mix, axis=-1)
    x = x - x.mean(axis=-1, keepdims=True)
    g0 = 10 ** ((target - lufs(x)) / 20.0)
    x = x * g0
    x, info["comp_max_gr_db"] = bus_compressor(x)
    g1 = 10 ** ((target - lufs(x)) / 20.0)
    x = x * g1
    info["static_gain_db"] = 20 * math.log10(g0 * g1)
    ceil = ceiling - 0.2
    for _ in range(6):
        y, info["limiter_max_gr_db"] = limiter(x, ceil)
        yl = lufs(y)
        if abs(yl - target) > 0.25:  # limiter shaved loudness: small makeup, re-limit
            x = x * 10 ** ((target - yl) / 20.0)
            y, info["limiter_max_gr_db"] = limiter(x, ceil)
        tp = an.true_peak_db(y)       # inter-sample peaks: lower the sample ceiling until TP fits
        if tp <= ceiling - 0.05:
            break
        ceil -= (tp - ceiling) + 0.1
    info["true_peak_db"] = tp
    info["ceiling_used_db"] = ceil
    info["limiter_max_gr_at_s"] = limiter.last_gr_time
    return y, info


# =============================================================================
# output
# =============================================================================
def write_wav(path, x, float32=False, seed=0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    x = np.asarray(x, np.float64)
    if float32:
        from scipy.io import wavfile
        wavfile.write(path, SR, x.T.astype(np.float32))
        return
    rng = np.random.default_rng([seed, 24])
    q = float(2 ** 23 - 1)
    d = rng.random(x.shape) - rng.random(x.shape)  # TPDF dither, +-1 LSB
    xi = np.clip(np.round(x * q + d), -(2 ** 23), 2 ** 23 - 1).astype("<i4")
    inter = xi.T.reshape(-1)
    b = inter.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(b)


# =============================================================================
# render
# =============================================================================
NATURE_LAYERS = [   # birds last: each song is capped against the rest of the nature bed
    ("wind", layer_wind),
    ("foliage", layer_foliage),
    ("lake", layer_lake),
    ("waterfall", layer_waterfall),
    ("stream", layer_stream),
    ("meadow", layer_meadow),
    ("birds", layer_birds),
]
STEM_ORDER = ["wind", "foliage", "lake", "birds", "waterfall", "stream", "meadow", "reverb", "music"]


def music_gain(tl, nature, pad, hop=0.05):
    """Music level that follows the story: gain(t) = L_nat(t) + music_rel_db(t) - L_pad(t)
    (short-term loudness, MUSIC_WIN_S window, Gaussian-smoothed), clamped to +-MUSIC_CLAMP_DB
    around its median. Measured on the pad only, so bell strikes are not ducked."""
    tn, Ln = an._block_power(an.k_weight(nature), MUSIC_WIN_S, hop)[::-1]
    tm, Lm = an._block_power(an.k_weight(pad), MUSIC_WIN_S, hop)[::-1]
    Ln, Lm = an._lufs(Ln), an._lufs(np.maximum(Lm, 1e-12))
    sg = MUSIC_SMOOTH_S / hop
    Ln = gaussian_filter1d(Ln, sg, mode="nearest")
    Lm = gaussian_filter1d(Lm, sg, mode="nearest")
    gdb = Ln + tl.param("music_rel_db", tn) - Lm
    core = (tn > 3.0) & (tn < tl.T - 3.0)
    med = float(np.median(gdb[core] if core.any() else gdb))
    gdb = gaussian_filter1d(np.clip(gdb, med - MUSIC_CLAMP_DB, med + MUSIC_CLAMP_DB), 0.25 / hop, mode="nearest")
    music_gain.curve = (tn, gdb)
    return 10.0 ** (np.interp(np.arange(tl.n) / SR, tn, gdb) / 20.0)


def render(shots, seed=7, duration=None, lead=LEAD, xfade=XFADE, target=TARGET_LUFS, verbose=True):
    t_all = time.time()
    tl = Timeline(shots, duration, lead, xfade)
    if verbose:
        print(f"film {tl.T:.3f} s = {tl.n} samples, {len(tl.shots)} shots, seed {seed}")
        for i, s in enumerate(tl.shots):
            print(f"  {s['start']:6.2f}-{s['end']:6.2f}  {s['name']:<10} env {s['env']:<10} I={s['intensity']:.2f}  "
                  f"audio {tl.a[i]:6.2f}-{tl.b[i]:6.2f}")
    wf = WindField(tl, rng_for(seed, "gusts"))
    stems, send = {}, np.zeros((2, tl.n))
    for name, fn in NATURE_LAYERS:
        t0 = time.time()
        if name == "birds":
            bed = sum(stems.values())
            pb, tb = an._block_power(an.k_weight(bed), 3.0, 0.1)
            dry, snd = fn(tl, rng_for(seed, name), wf, amb=(tb, an._lufs(pb)))
            del bed
        else:
            dry, snd = fn(tl, rng_for(seed, name), wf)
        stems[name] = dry
        send += snd
        if verbose:
            print(f"  layer {name:<10} {time.time() - t0:5.1f} s")
    t0 = time.time()
    irv, irf = ir_valley(rng_for(seed, "ir_valley")), ir_forest(rng_for(seed, "ir_forest"))
    wfor = tl.ctrl("rev_forest", smooth=0.3)
    rg = tl.ctrl("rev_db", "db", smooth=0.3)
    stems["reverb"] = convolve_stereo(send * rg * (1 - wfor), irv) + convolve_stereo(send * rg * wfor, irf)
    del send
    nature = sum(stems.values())
    pd, ps, bd, bs = layer_music(tl, rng_for(seed, "music"))
    pad = pd + convolve_stereo(ps, irv) * 0.8
    bell = bd + convolve_stereo(bs, irv) * 0.8
    mg = music_gain(tl, nature, pad)
    music = (pad + bell) * mg
    del pad, bell, pd, ps, bd, bs
    # split fades: nature recedes over the last FADE_OUT_NATURE s, music closes with FADE_OUT
    fn_nat, fn_mus = fades(tl.n)
    for k in list(stems):
        stems[k] = stems[k] * fn_nat
    nature = nature * fn_nat
    stems["music"] = music * fn_mus
    if verbose:
        gc = music_gain.curve[1]
        print(f"  reverb+music {time.time() - t0:5.1f} s (music gain {gc.min():+.1f}..{gc.max():+.1f} dB)")
    mix = nature + stems["music"]
    out, info = master(mix, target)
    # stems at the master's static gain (for analysis only)
    sg = 10 ** (info["static_gain_db"] / 20.0)
    stems = {k: stems[k] * sg for k in STEM_ORDER if k in stems}
    stems["nature"] = nature * sg
    info["render_s"] = time.time() - t_all
    if verbose:
        print("  master: " + ", ".join(f"{k}={v:.2f}" for k, v in info.items()))
    # music_rel_db target per shot, averaged over the part analysis.py measures (before the end fades)
    mrel = [float(np.mean(tl.param("music_rel_db", np.linspace(s["start"], max(s["start"] + 1.0, min(s["end"], tl.T - 4.5)), 64))))
            for s in tl.shots]
    extra = dict(limiter_gain=limiter.last_gain, bell_events=getattr(layer_music, "bell_events", []),
                 bird_songs=getattr(layer_birds, "stats", []), music_gain_db=music_gain.curve,
                 music_rel_target=mrel, last_lap_end=getattr(layer_lake, "last_lap_end", None),
                 audio_windows=[(float(a), float(b), float(tl.h[i]), float(tl.h[i + 1]))
                                for i, (a, b) in enumerate(zip(tl.a, tl.b))])
    return out, stems, tl, info, extra


def default_shots_path():
    film = os.path.join(ROOT, "out", "film_shots.json")
    return film if os.path.exists(film) else os.path.join(HERE, "film_shots.json")


def load_shots(path):
    with open(path, "r", encoding="utf-8") as fh:
        shots = json.load(fh)
    if isinstance(shots, dict):
        shots = shots.get("shots", [])
    return shots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", default=None,
                    help="shot list JSON (default: out/film_shots.json, fallback audio/film_shots.json)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "audio", "soundtrack.wav"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=None, help="override film length (s); default = last shot end")
    ap.add_argument("--lead", type=float, default=LEAD, help="J-cut lead in seconds")
    ap.add_argument("--xfade", type=float, default=XFADE, help="ambience crossfade length in seconds")
    ap.add_argument("--target-lufs", type=float, default=TARGET_LUFS)
    ap.add_argument("--float", action="store_true", help="write float32 WAV instead of 24-bit PCM")
    ap.add_argument("--stems", action="store_true", help="also write stems (float32) to <out dir>/stems/")
    ap.add_argument("--analyze", action="store_true", help="write spectrogram PNGs + analysis.json next to the output")
    a = ap.parse_args()
    path = a.shots or default_shots_path()
    print(f"shot list: {path}")
    out, stems, tl, info, extra = render(load_shots(path), a.seed, a.duration, a.lead, a.xfade, a.target_lufs)
    write_wav(a.out, out, a.float, a.seed)
    print(f"wrote {a.out}  ({out.shape[-1]} samples, {out.shape[-1] / SR:.3f} s)")
    od = os.path.dirname(os.path.abspath(a.out))
    if a.stems:
        sd = os.path.join(od, "stems")
        for k, v in stems.items():
            write_wav(os.path.join(sd, f"{k}.wav"), v, float32=True)
        print(f"wrote stems to {sd}")
    if a.analyze:
        x, _ = an.read_wav(a.out) if not a.float else an.read_any_wav(a.out)
        an.analyze(x, stems, tl.shots, od, audio_cuts=list(tl.a[1:]), extra=dict(extra, shots_file=path, info=info))


if __name__ == "__main__":
    main()
