"""Render the film.  usage:
  python film.py preview            -> 480x270, 1 spp, every frame (animatic) + mp4
  python film.py final [--spp 12]   -> 1920x1080 frames into out/frames (resumable), then encode
  python film.py stills             -> one mid-shot still per shot (contact sheet)
  python film.py encode [--upscale 3440x1440] -> mp4 from out/frames + out/audio/soundtrack.wav
Options: --shots 1,3  (only those shots)  --w --h --spp --fps --every N (render every Nth frame)
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))
import post  # noqa: E402
import render  # noqa: E402
from camera_path import shot_camera  # noqa: E402
import shots_film  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["preview", "final", "stills", "encode"])
ap.add_argument("--w", type=int)
ap.add_argument("--h", type=int)
ap.add_argument("--spp", type=int)
ap.add_argument("--q", type=int)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--shots", default="")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--out", default="")
ap.add_argument("--shutter", type=float, default=0.5)   # fraction of frame time (180 deg)
# ultra-wide output (e.g. --w 3440 --h 1440): the shots are framed for 16:9; keep the vertical view a bit
# tighter so the extra width shows more of the shores/forest at the sides instead of a fisheye
ap.add_argument("--vfov-scale", type=float, default=None)
# encode: scale the frames to this size (e.g. 3440x1440: the final is rendered at 2752x1152, 24 fps, 8 spp and
# upscaled 1.25x)
ap.add_argument("--upscale", default="")
args = ap.parse_args()

defaults = {"preview": (480, 270, 1, 0), "final": (1920, 1080, 12, 1), "stills": (960, 540, 6, 1), "encode": (1920, 1080, 1, 1)}
W, H, SPP, Q = defaults[args.mode]
W = args.w or W; H = args.h or H; SPP = args.spp or SPP; Q = args.q if args.q is not None else Q
FPS = args.fps
# default: 1.0 at 16:9, ~0.86 at 21:9 (horizontal view grows from ~95 to ~115 deg at a 60 deg shot)
VFOV_SCALE = args.vfov_scale if args.vfov_scale else (1.0 if W / H < 1.9 else 0.86)

world = render.World()
SHOTS = shots_film.build(world)
total = sum(s["dur"] for s in SHOTS)
starts = np.cumsum([0.0] + [s["dur"] for s in SHOTS])
sel = set(int(v) for v in args.shots.split(",") if v.strip()) if args.shots else None

outdir = Path(args.out) if args.out else ROOT / "out" / ("frames" if args.mode in ("final", "encode") else args.mode)
outdir.mkdir(parents=True, exist_ok=True)

# export exact shot timings for the soundtrack
(ROOT / "out").mkdir(exist_ok=True)
(ROOT / "out" / "film_shots.json").write_text(json.dumps(
    [{"index": i, "name": s["name"], "start": float(starts[i]), "end": float(starts[i + 1]), "env": s.get("env", "lakeshore")}
     for i, s in enumerate(SHOTS)], indent=1))


def frame_samples(f, spp):
    """camera/look/time for each sample of frame f (motion blur within the shot)."""
    T = f / FPS
    i = int(np.searchsorted(starts, T, side="right") - 1)
    i = min(max(i, 0), len(SHOTS) - 1)
    shot = SHOTS[i]
    t0 = T - starts[i]
    cams, times = [], []
    shut = args.shutter / FPS
    for s in range(spp):
        off = ((s + 0.5) / spp - 0.5) * shut if spp > 1 else 0.0
        tl = float(np.clip(t0 + off, 0.0, shot["dur"] - 1e-4))
        c = shot_camera(shot, tl)
        c.fov = c.fov * VFOV_SCALE
        cams.append(c)
        times.append(float(starts[i] + tl + shot.get("time_offset", 0.0)))
    return i, shot, cams, times


def fade(T):
    a = 1.0
    if T < 1.6:
        a = min(a, (T / 1.6) ** 1.5)
    if T > total - 2.2:
        a = min(a, max(0.0, (total - T) / 2.2) ** 1.3)
    return a


EXPO = ROOT / "out" / "exposures.json"


def expo_for(i):
    """per-shot exposure measured once on the shot's middle frame (cached in out/exposures.json, keyed by the
    render settings so a preview never sets the exposure of the final)."""
    table = json.loads(EXPO.read_text()) if EXPO.exists() else {}
    key = f"{i}:{SHOTS[i]['name']}:{W}x{H}:q{Q}:v{VFOV_SCALE:.3f}:fps{FPS}"
    if key in table:
        return table[key]
    f = int((starts[i] + SHOTS[i]["dur"] * 0.5) * FPS)
    _, shot, cams, times = frame_samples(f, 2)
    R.heroes = shot.get("heroes", [])
    hdr, _ = R.render(cams, shot["look"], times)
    e = post.suggest_exposure(hdr, target_luma=shot.get("target_luma", 112.0), g=shot["grade"])
    table[key] = e
    EXPO.write_text(json.dumps(table, indent=1))
    return e


if args.mode == "encode":
    pass
else:
    R = render.Renderer(W, H, quality=Q, world=world)
    nframes = int(round(total * FPS))
    if args.mode == "stills":
        frames = [int((starts[i] + SHOTS[i]["dur"] * 0.5) * FPS) for i in range(len(SHOTS))]
    else:
        frames = list(range(0, nframes, args.every))
    t_start = time.time()
    done = 0
    from dataclasses import replace as _rp
    from concurrent.futures import ThreadPoolExecutor

    def finish_save(hdr, g, f, fn, fade_a):
        # grade + save; in final mode this runs on a worker thread while the GPU renders the next frame
        # (post.finish ~11 s at 2752x1152 - numpy releases the GIL); written to a temp file and renamed so a
        # killed render never leaves a half-written frame that the resume would skip
        img = post.finish(hdr, g, seed=f)
        if fade_a < 1.0:
            img = (img.astype(np.float32) * fade_a + 0.5).astype(np.uint8)
        tmp = fn.with_name(fn.stem + ".tmp.png")
        Image.fromarray(img).save(tmp)
        tmp.replace(fn)

    pool = ThreadPoolExecutor(max_workers=1) if args.mode == "final" else None
    pending = None
    for f in frames:
        i, shot, cams, times = frame_samples(f, SPP)
        if sel and i not in sel:
            continue
        fn = outdir / (f"{f:05d}.png" if args.mode != "stills" else f"shot{i:02d}_{shot['name']}.png")
        if fn.exists() and args.mode == "final":
            continue
        look = shot["look"]
        R.heroes = shot.get("heroes", [])
        hdr, info = R.render(cams, look, times)
        g = shot["grade"]
        if "exposure" not in shot:
            shot["exposure"] = post.suggest_exposure(hdr, target_luma=shot.get("target_luma", 112.0), g=g) if args.mode == "stills" else expo_for(i)
        gx = _rp(g, exposure=g.exposure * shot["exposure"])
        fade_a = fade(f / FPS) if args.mode != "stills" else 1.0
        if pool is not None:
            if pending is not None:
                pending.result()          # at most one frame in flight (bounded memory, errors surface)
            pending = pool.submit(finish_save, hdr, gx, f, fn, fade_a)
        else:
            finish_save(hdr, gx, f, fn, fade_a)
        done += 1
        el = time.time() - t_start
        if done % 10 == 1 or args.mode == "stills":
            print(f"frame {f} shot {i} {shot['name']}: {info['seconds']:.2f}s  (avg {el/done:.2f}s/frame)", flush=True)
    if pending is not None:
        pending.result()

if args.mode in ("preview", "final", "encode"):
    src = outdir
    name = "preview.mp4" if args.mode == "preview" else "alpine_lake_film.mp4"
    vid = ROOT / "out" / name
    audio = ROOT / "out" / "audio" / "soundtrack.wav"
    # concat list (works for sparse frame numbering too)
    frames_list = sorted(src.glob("[0-9][0-9][0-9][0-9][0-9].png"))
    lst = src / "frames.txt"
    dur = args.every / FPS
    lst.write_text("".join(f"file '{f.resolve().as_posix()}'\nduration {dur:.6f}\n" for f in frames_list))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio.exists():
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "320k", "-shortest"]
    else:
        print("note: out/audio/soundtrack.wav not found, encoding without sound", flush=True)
    if args.upscale:
        uw, uh = args.upscale.lower().split("x")
        cmd += ["-vf", f"scale={uw}:{uh}:flags=lanczos"]     # (r5 qa5 B09: no extra unsharp - post already sharpens)
    cmd += ["-c:v", "libx264", "-preset", "slow" if args.mode != "preview" else "fast",
            "-crf", "14" if args.mode != "preview" else "20", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart", str(vid)]
    if frames_list:
        cmd = cmd[:-1] + ["-r", str(FPS), cmd[-1]]
        subprocess.run(cmd, check=False)
        print("->", vid)
