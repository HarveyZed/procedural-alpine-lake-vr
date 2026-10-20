"""Final film render driver: runs film.py final in a restart loop (resumable: finished frames are
skipped), then encodes with the soundtrack.  Defaults: 24 fps,
rendered at 2752x1152 with 8 spp (motion blur), upscaled to 3440x1440 (Lanczos) at encode.
  python tools/render_final.py [--cache cache_final] [--spp 8] [--w 2752 --h 1152] [--fps 24] [--upscale 3440x1440]
Takes ~14 h on a Radeon 780M iGPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache_final")
ap.add_argument("--spp", type=int, default=8)
# per-shot sample counts (perf4, round 4): 6 spp where 6 vs 8 is invisible (calm moves), 8 spp where near
# crowns / rocks move fast through the frame
ap.add_argument("--spp-groups", default="1,2,6,7:6;0,3,4,5,8:8")   # (r5 order: 0 forest, 1 reveal, 2 skim, 3 waterfall, 4 stream, 5 meadow, 6 flyover, 7 peaks, 8 finale)
ap.add_argument("--w", type=int, default=2752)
ap.add_argument("--h", type=int, default=1152)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--upscale", default="3440x1440")
ap.add_argument("--out", default="out/frames")
args = ap.parse_args()

env = dict(os.environ, ALPINE_CACHE=str(ROOT / args.cache))
env.setdefault("ALPINE_MAX_PROC_GB", "30")       # full-res frames peak at ~17 GB in the driver (gpuslot watchdog)
outdir = ROOT / args.out
outdir.mkdir(parents=True, exist_ok=True)
(ROOT / "out").mkdir(exist_ok=True)
log = open(ROOT / "out" / "render_final.log", "a", encoding="utf-8")
common = ["--fps", str(args.fps), "--out", str(outdir)]


def expected():
    sj = ROOT / "out" / "film_shots.json"          # written by film.py (shot timings)
    return int(round(json.loads(sj.read_text())[-1]["end"] * args.fps)) if sj.exists() else 10 ** 9


for attempt in range(40):
    done = len(list(outdir.glob("[0-9]" * 5 + ".png")))
    log.write(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: {done}/{expected()} frames present\n"); log.flush()
    if done >= expected():
        break
    groups = [g.split(":") for g in args.spp_groups.split(";") if g] or [("", str(args.spp))]
    ok = True
    for shots, spp in groups:
        r = subprocess.run([PY, str(ROOT / "film.py"), "final", "--w", str(args.w), "--h", str(args.h),
                            "--spp", spp] + (["--shots", shots] if shots else []) + common, cwd=ROOT, env=env, stdout=log, stderr=log)
        ok = ok and r.returncode == 0
    if ok and len(list(outdir.glob("[0-9]" * 5 + ".png"))) >= expected():
        break
    time.sleep(5)
have = len(list(outdir.glob("[0-9]" * 5 + ".png")))
if have < expected():
    msg = f"only {have}/{expected()} frames rendered, not encoding (see out/render_final.log)"
    log.write(msg + "\n"); log.close()
    sys.exit(msg)
log.write(f"[{time.strftime('%H:%M:%S')}] rendering finished, encoding\n"); log.flush()
r = subprocess.run([PY, str(ROOT / "film.py"), "encode", "--upscale", args.upscale] + common, cwd=ROOT, env=env, stdout=log, stderr=log)
log.write(f"[{time.strftime('%H:%M:%S')}] done (encode exit {r.returncode})\n"); log.close()
print("encode ok -> out/alpine_lake_film.mp4" if r.returncode == 0 else f"encode failed ({r.returncode}), see out/render_final.log")
sys.exit(r.returncode)
