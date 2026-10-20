"""Render a named still. usage: python still.py NAME [--w 960] [--h 540] [--spp 4] [--q 1]"""
import argparse
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))
import numpy as np  # noqa: E402
import render  # noqa: E402
import post  # noqa: E402
import shots  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--w", type=int, default=960)
ap.add_argument("--h", type=int, default=540)
ap.add_argument("--spp", type=int, default=4)
ap.add_argument("--q", type=int, default=1)
ap.add_argument("--tag", default="")
args = ap.parse_args()

world = render.World()
R = render.Renderer(args.w, args.h, quality=args.q, world=world)
S = shots.stills(world)
out = ROOT / "out" / "stills"
out.mkdir(parents=True, exist_ok=True)
names = list(S) if args.names == ["all"] else args.names
for name in names:
    cam, look, t = S[name]
    hdr, info = R.render(cam, look, t, spp=args.spp)
    np.save(out / f"hdr_{name}.npy", hdr.astype(np.float16))
    img = post.finish(hdr, post.Grade(exposure=look.exposure))
    fn = out / f"{name}{args.tag}.png"
    Image.fromarray(img).save(fn)
    print(f"{name}: {info['seconds']:.1f}s  sun={np.round(info['sun'],3)} sky={np.round(info['sky'],3)} -> {fn}", flush=True)
