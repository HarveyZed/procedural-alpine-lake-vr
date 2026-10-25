#!/usr/bin/env python3
"""package_vr.py - VR delivery packaging for the MoltenVK pano port (Phase 5).

WHAT THIS IS
    A packaging CLI and an ffmpeg *argument builder*. It inspects the PNGs
    vr.py wrote, classifies their equirect layout, writes a delivery manifest,
    and prints the exact ffmpeg command line that turns the frames into an 8K
    SBS turntable carrying spherical metadata.

WHAT IT IS NOT
    It NEVER spawns ffmpeg: no subprocess, no os.system, no execvp, no --run
    flag. Phase 5 ran under a zero-execution discipline (port/RUNBOOK.md
    section 1) and the encode is the operator first real spend, so the command
    line is printed for a human to read and then run.

LAYOUT CONTRACT (port/RUNBOOK.md section 6; vr.py CLI)
    vr.py writes ONE PNG per pano; with --stereo sbs that single PNG is the
    left|right pair concatenated on x, so an 8192x4096-per-eye pano is a
    16384x4096 file. vr.py enforces W >= 2H per eye (vr.py:1104-1117, the
    review_p4.md M1 fix), therefore
        mono -> total aspect exactly 2:1     one full equirect
        sbs  -> total aspect exactly 4:1     two 2:1 equirects side by side
        sbs  -> total aspect  >4:1           two >2:1 latitude-cropped bands
    Anything between 2:1 and 4:1 cannot be two >=2:1 eyes, and is rejected here
    for the same reason vr.py refuses to render it.

PACKING
    Headsets and ingest services want HALF side-by-side: the packed frame keeps
    the 2:1 equirect shape and each eye is squeezed to half width. vr.py writes
    FULL side-by-side (unsqueezed eyes), so the video path squashes each eye by
    two with one scale filter - scaling the whole packed frame horizontally by
    1/2 scales both adjacent halves by 1/2, so it is the same operation.
    --packing full keeps the frame exactly as vr.py wrote it.

METADATA HONESTY
    The tag set below is the commonly used projection/stereo family
    (projection=equirectangular + projection_range/bounds + a stereo mode + an
    initial viewport + a minimum-tile hint). It is written but NOT verified:
    nothing has been decoded by a headset or an ingest service in this
    workspace. --tag KEY=VALUE is the escape hatch for whatever the first
    player actually reads; port/reports/phase5.md records this as unknown.
"""
import argparse
import glob
import hashlib
import json
import re
import shutil
import struct
import sys
from pathlib import Path

# vr.py default output name: pano_{idx}_{name}_{stereo}_{w}x{h}.png
VP_NAME = re.compile(r"pano_(\d+)_(?P<name>[a-z0-9]+)_(mono|sbs)_(\d+)x(\d+)")
PNG_SIG = bytes([0x89]) + b"PNG\r\n\x1a\n"
FENCE = chr(96) * 3

DEFAULT_TAGS = [
    ("projection", "equirectangular"),
    ("projection_range", "full"),
    ("projection_bounds", "0;0;1;1"),
    ("minimum_tile_width", "512"),
    ("minimum_tile_height", "512"),
    ("initial_view", "0;0;0"),
    ("source", "alpine-lake vr.py (MoltenVK port, Phase 5)"),
]


def png_size(path):
    """(w, h) straight from the PNG IHDR - no image library, no decode."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != PNG_SIG or head[12:16] != b"IHDR":
        raise ValueError("not a PNG (bad signature or IHDR)")
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def classify(w, h):
    """(stereo, per_eye_w, note) for one pano frame, None if not equirect.

    stereo is "mono" or "sbs". A frame is equirect-shaped when the per-eye
    aspect is 2:1 or wider, which is exactly the guard vr.py applies at
    vr.py:1104-1117.
    """
    if h <= 0 or w % 2:
        return None
    if w == 2 * h:
        return ("mono", w, "full equirect, 2:1")
    if w % 4:
        return None
    per = w // 2
    if per < 2 * h:
        return None
    # Under the section 6 mapping every row set spans lat +pi/2..-pi/2, so a
    # 2:1 eye has a square angular footprint at the equator (dLat = dLon) and a
    # wider eye only re-samples latitude: texel footprint becomes dLat flat.
    over = per / (2.0 * h)                   # 1.0 at 2:1 per eye
    note = ("two 2:1 equirects side by side, square footprint at the equator"
            if abs(over - 1.0) < 1e-9 else
            "two " + format(per / h, ".2f") + ":1 bands side by side; full "
            "latitude covered, x" + format(over, ".2f") + " redundant "
            "latitude sampling (footprint = dLat everywhere)")
    return ("sbs", per, note)


def tags(stereo, packing, extra):
    """The metadata pairs the encode command writes, in argv order."""
    mode = {"mono": "mono",
            "sbs": "sbs_half" if packing == "half" else "sbs_full"}[stereo]
    out = list(DEFAULT_TAGS) + [("stereo_mode", mode), ("stereo_format", mode)]
    if stereo == "sbs" and packing == "half":
        out.append(("cropped_stereo_format", "SIDE_BY_SIDE_HALF"))
    return out + extra


def packed_dims(stereo, packing, ew, eh):
    """(w, h) of one encoded frame: what a viewer sees as one equirect."""
    if stereo == "mono":
        return ew, eh
    return (ew if packing == "half" else 2 * ew), eh


def build_ffmpeg(target, n_frames, args, stereo, packing, ew, eh):
    """The ffmpeg argv for the 8K SBS turntable (ew/eh = per-eye resolution).

    Built and returned only - nothing in this module executes it.
    """
    vf = []
    if args.crop_lat is not None:
        # Keep a central slice of latitude, rows centred on the equator; the
        # result is no longer a full sphere, so projection_range changes below.
        crop_h = min(max(2, (int(round(eh * args.crop_lat)) // 2) * 2),
                     (eh // 2) * 2)
        # y goes on the chroma grid too: eh=32 with --crop-lat 0.001 gives
        # crop_h=2 and (eh - crop_h) // 2 = 15, an ODD offset that misaligns
        # 4:2:0 chroma in the encoded file. Floor to even (centred within a
        # row) - review_p5 N7.
        vf.append("crop=iw:" + str(crop_h) + ":0:"
                  + str((eh - crop_h) // 2 // 2 * 2))
    if stereo == "sbs" and packing == "half":
        vf.append("scale=trunc(iw/4)*2:ih:flags=" + args.scaler)
    else:
        vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=" + args.scaler)
    vf.append("format=" + args.pix_fmt)
    tg = tags(stereo, packing, args.tag_pairs)
    if args.crop_lat is not None:
        tg = [(k, "partial" if k == "projection_range" else v) for k, v in tg]
    argv = ["ffmpeg", "-y", "-loglevel", "info"]
    if n_frames <= 1 and not args.pattern:
        argv += ["-loop", "1", "-t", format(args.still_seconds, "g")]
    if args.pattern:                        # explicit printf pattern / still
        argv += ["-framerate", str(args.fps), "-start_number",
                 str(args.start_number), "-i", args.pattern]
    elif n_frames <= 1:
        argv += ["-i", target]
    else:                                   # concat demuxer over the frame list
        argv += ["-f", "concat", "-safe", "0", "-i", target]
        vf.insert(0, "fps=" + str(args.fps))
    argv += ["-an", "-r", str(args.fps), "-vf", ",".join(vf)]
    argv += ["-c:v", args.codec, "-pix_fmt", args.pix_fmt, "-preset", args.preset,
             "-crf", str(args.crf)]
    # Closed GOP for 360 playback: fixed interval, scene-cut keyframes off. A
    # slow turntable has almost no scene change, so scene detection would place
    # keyframes arbitrarily while leaving seek points sparse.
    argv += ["-g", str(args.gop), "-keyint_min", str(args.gop),
             "-sc_threshold", "0"]
    if args.hevc_params:
        argv += ["-x265-params", args.hevc_params]
    argv += ["-movflags", "+faststart+use_metadata_tags"]
    for k, v in tg:
        argv += ["-metadata:s:v:0", k + "=" + v]
    if args.out:
        out_path = args.out
    else:
        pw, ph = packed_dims(stereo, packing, ew, eh)
        suffix = "sbs_" + packing if stereo == "sbs" else "mono"
        name = "turntable_" + str(pw) + "x" + str(ph) + "_" + suffix + ".mp4"
        out_path = str(Path(args.out_dir) / name)
    return argv + [out_path]


def frames_list_text(entries, dwell):
    """Contents of the ffmpeg concat-demuxer list for these frames.

    vr.py writes one PNG per pano and its names are NOT a frame sequence
    (pano_0_forest_sbs_..., pano_1_reveal_sbs_...), so a printf percent pattern
    cannot address them. The concat demuxer takes the list verbatim, which also
    makes the turntable order explicit and reviewable. Paths are absolute, so
    the list works from any cwd (with -safe 0).
    """
    def line(p):
        # single quotes are quoted as '\'' - the concat demuxer reads a
        # shell-ish single-quoted path, so that is the one escape it needs.
        q = str(Path(p).resolve()).replace("'", "'\\''")
        return "file '" + q + "'"

    out = ["# ffmpeg concat list - package_vr.py; one pano per line"]
    for e in entries:
        out.append(line(e["file"]))
        out.append("duration " + format(dwell, ".4f"))
    if entries:
        # The concat demuxer ends the stream at the last entry, so the final
        # file is repeated to hold its duration on screen.
        out.append(line(entries[-1]["file"]))
    return "\n".join(out) + "\n"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shell(s):
    """Quote one argv token so the printed line is pasteable into bash."""
    safe = "-._/:=%+,[]"
    if s and all(c.isalnum() or c in safe for c in s):
        return s
    return json.dumps(s)          # double quotes, which bash honours


def collect(globs):
    """Expand the --in arguments: explicit .png paths and/or glob patterns.

    A quoted glob never reaches us expanded (the gate passes it verbatim), so
    metacharacters decide the route - a pattern also ends in .png.

    Patterns go to glob.glob, NOT Path().glob: Path().glob() is anchored to
    the cwd and raises NotImplementedError("Non-relative patterns are
    unsupported") on an absolute pattern, while a delivery directory is
    absolute by definition and this tool's own help advertises globs and
    paths (review_p5 M1). glob.glob takes either root, and recursive=True
    keeps the "**" behaviour Path().glob had. Relative patterns still match
    relative (the gate runs "engine_vr/out/*.png", port/gates/gate_p5.py:39).
    """
    found = []
    for g in globs:
        if any(c in g for c in "*?["):
            found += [Path(p) for p in glob.glob(g, recursive=True)]
        else:
            found.append(Path(g))
    return sorted({f for f in found if f.is_file()})


def scan(files):
    """-> (entries, set of stereo kinds, first (w, h)); skips unreadable input."""
    entries, st, first = [], set(), None
    for f in files:
        try:
            w, h = png_size(f)
        except (ValueError, OSError) as exc:
            print("SKIP   " + str(f) + ": " + str(exc))
            continue
        c = classify(w, h)
        if c is None:
            print("REJECT " + str(f) + ": " + str(w) + "x" + str(h)
                  + " is not equirect-shaped - the per-eye aspect must be 2:1 "
                  + "or wider, and vr.py refuses to render the same regime "
                  + "(review_p4 M1)")
            continue
        stereo, per, note = c
        m = VP_NAME.match(f.name)
        if m and m.group(3) != stereo:
            print("WARN   " + f.name + ": the filename says " + m.group(3)
                  + " but " + str(w) + "x" + str(h) + " measures " + stereo
                  + " - packaging by measurement. A mono-shaped file whose name"
                  + " claims sbs is usually a single eye, not a packed pair.")
        st.add(stereo)
        if first is None:
            first = (w, h)
        entries.append({"file": str(f), "bytes": f.stat().st_size, "w": w, "h": h,
                        "stereo": stereo, "per_eye": [per, h],
                        "aspect": round(w / h, 4), "layout": note,
                        "viewpoint_idx": int(m.group(1)) if m else None,
                        "viewpoint": m.group("name") if m else None})
    return entries, st, first


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="package_vr.py",
        description="package vr.py panos for delivery: a manifest plus an ffmpeg "
                    "argv builder. This tool never runs ffmpeg.",
    )
    ap.add_argument("--in", dest="glob", nargs="+", required=True,
                    help="pano PNG(s): quoted glob(s) and/or explicit paths")
    ap.add_argument("--out-dir", required=True, help="delivery directory")
    ap.add_argument("--video", action="store_true",
                    help="print the ffmpeg 8K SBS turntable argv (never runs it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and write nothing (never runs ffmpeg)")
    ap.add_argument("--packing", choices=("half", "full"), default="half",
                    help="SBS packing for --video; half = 2:1 packed frame")
    ap.add_argument("--fps", type=int, default=12, help="turntable frame rate")
    ap.add_argument("--start-number", type=int, default=0, dest="start_number")
    ap.add_argument("--pattern",
                    help="override the input pattern (frame_00001 style)")
    ap.add_argument("--out", help="video path (default out-dir/turntable_8k_sbs.mp4)")
    ap.add_argument("--codec", default="libx265")
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--crf", type=float, default=22)
    ap.add_argument("--pix-fmt", default="yuv420p", dest="pix_fmt")
    ap.add_argument("--scaler", default="spline")
    ap.add_argument("--gop", type=int, default=120, help="keyframe interval in frames")
    ap.add_argument("--hevc-params", default="", help="extra x265-params string")
    ap.add_argument("--still-seconds", type=float, default=8.0, dest="still_seconds",
                    help="duration when --in matches exactly one frame")
    ap.add_argument("--crop-lat", type=float, default=None, dest="crop_lat",
                    help="keep only this fraction of latitude, in (0, 1], "
                         "equator-centred (0.75 = the middle 75%%)")
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                    help="extra or overriding spherical metadata (repeatable)")
    ap.add_argument("--note", default="", help="free-text line for the manifest")
    args = ap.parse_args(argv)

    args.tag_pairs = []
    for t in args.tag:
        if "=" not in t:
            ap.error("--tag " + t + ": expected KEY=VALUE")
        k, v = t.split("=", 1)
        args.tag_pairs.append((k.strip(), v.strip()))

    # crop_lat is a FRACTION of the frame height that build_ffmpeg turns into
    # crop=iw:<h>:0:<y>. Unvalidated it reaches argv literally: --crop-lat 5
    # printed crop=iw:1280:0:-512 against a 256-row pano (a crop taller than
    # the input at a negative y), so ffmpeg - not argparse - failed at encode
    # time, and a negative fraction silently meant "keep 2 rows" (review_p5
    # N7). (0, 1] is the only range with crop_h <= ih and y >= 0.
    if args.crop_lat is not None and not 0.0 < args.crop_lat <= 1.0:
        ap.error("--crop-lat " + format(args.crop_lat, "g") + ": expected the "
                 "fraction of latitude to KEEP, in (0, 1] (e.g. 0.75 keeps "
                 "the middle 75% around the equator); anything else asks "
                 "ffmpeg for a crop taller than the frame or for a negative "
                 "crop y")

    entries, st, first = scan(collect(args.glob))
    if not entries:
        print("NOTE   no usable pano PNG under " + str(args.glob) + ": nothing "
              + "matched. Render first (port/RUN_FIRST.md ladder), then re-run.")
    if len(st) > 1:
        print("ERROR  mixing mono and SBS panos in one package - the encoder "
              + "has one layout. Package them separately.")
        return 2
    stereo = st.pop() if st else "sbs"
    packing = "full" if stereo == "mono" else args.packing
    if first is None:
        ew, eh = 8192, 4096              # template mode (nothing matched yet)
    elif stereo == "mono":
        ew, eh = first
    else:
        ew, eh = first[0] // 2, first[1]
    out_dir = Path(args.out_dir)
    print("package_vr: " + str(len(entries)) + " pano(s), stereo=" + stereo
          + ", packing=" + packing + ", per-eye " + str(ew) + "x" + str(eh)
          + ", out-dir " + str(out_dir))
    for e in entries:
        vp = ""
        if e["viewpoint_idx"] is not None:
            vp = "  vp" + str(e["viewpoint_idx"]) + " " + str(e["viewpoint"])
        print("  " + e["file"] + "  " + str(e["w"]) + "x" + str(e["h"])
              + "  " + e["stereo"] + ": " + e["layout"] + vp)

    manifest = {
        "tool": "package_vr.py (Phase 5) - argument builder only; this tool "
                "never spawns ffmpeg",
        "frames": len(entries), "stereo": stereo, "packing": packing,
        "per_eye": [ew, eh],
        "equirect": "lon in [-pi,pi) east, lat +pi/2 at row 0 down to -pi/2 at "
                    "row H-1, pixel centres at (x+0.5, y+0.5); port/RUNBOOK.md "
                    "section 6",
        "caveat": "pano pixels are NOT comparable with film.py frames: an "
                  "equirect texel footprint is not a perspective frustum, post.py "
                  "vignette and sky-weighted exposure are tuned for a rectangular "
                  "frame, and the light probe is one sample at head centre shared "
                  "by both eyes.",
        "note": args.note, "files": entries,
    }
    argv_out = None
    if args.video:
        if not entries:
            print("WARN   --video with no input frames: printing the argv "
                  + "template against the --pattern placeholder only.")
        list_text = frames_list_text(entries, 1.0 / max(1, args.fps))
        manifest["frames_list"] = list_text
        if args.pattern:
            target = args.pattern
        elif len(entries) == 1:
            target = entries[0]["file"]        # -loop 1 -t reads the image
        elif entries:
            target = str(out_dir / "frames.txt")
        else:
            target = "engine_vr/out/pano_*_sbs_*.png"
        argv_out = build_ffmpeg(target, len(entries), args, stereo, packing, ew, eh)
        manifest["encode"] = {"argv": argv_out, "fps": args.fps,
                              "codec": args.codec, "crf": args.crf,
                              "pix_fmt": args.pix_fmt, "gop": args.gop,
                              "packing": packing, "executed": False,
                              "metadata_verified": False}
        print("")
        packed_w, _packed_h = packed_dims(stereo, packing, ew, eh)
        if packed_w != 8192:
            print("NOTE   the encoded frame is " + str(packed_w)
                  + " wide after packing, not 8K. vr.py defaults to 8192 per eye"
                  + " for sbs, which packs to exactly 8192 with --packing half; a"
                  + " smaller --res ships a smaller turntable (fine for a test,"
                  + " not for delivery).")
        print("ENCODE COMMAND (printed, NOT executed - Phase 5 never ran ffmpeg):")
        print(" ".join(shell(a) for a in argv_out))
        print("")
        print("equirect / spherical metadata that command writes:")
        for k, v in tags(stereo, packing, args.tag_pairs):
            print("  -metadata:s:v:0 " + k + "=" + v)
        print("NOTE: those tags are UNVERIFIED against any headset or ingest "
              + "service; override with --tag KEY=VALUE after the first playback "
              + "test (port/reports/phase5.md open unknowns).")

    if args.dry_run:
        if argv_out and entries:
            head = manifest["frames_list"].splitlines()[:5]
            print("frames.txt (would be written to " + str(out_dir)
                  + ", then read by -f concat):")
            for ln in head:
                print("  " + ln)
            print("  ... " + str(len(entries)) + " frame(s), "
                  + format(1.0 / max(1, args.fps), ".4f") + " s each")
        print("DRY-RUN  wrote nothing (no manifest, no copies, no ffmpeg). "
              + "Target directory would be " + str(out_dir))
        return 0
    dest = out_dir / "pano"
    dest.mkdir(parents=True, exist_ok=True)
    for e in entries:
        src = Path(e["file"])
        shutil.copy2(src, dest / src.name)
        e["packaged_as"] = "pano/" + src.name
        e["sha256"] = sha256(src)
    if argv_out and entries:
        # Re-point the list at the packaged copies so the delivery directory is
        # self-contained, and make the paths absolute so the concat demuxer
        # works from any cwd. Regenerate it if the panos move.
        dwell = 1.0 / max(1, args.fps)
        packed = [{**e, "file": str(dest / Path(e["packaged_as"]).name)}
                  for e in entries]
        manifest.pop("frames_list")
        (out_dir / "frames.txt").write_text(frames_list_text(packed, dwell))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    md = ["# Delivery manifest", "",
          "- frames: " + str(len(entries)) + "   stereo: " + stereo
          + "   packing: " + packing + "   per-eye: " + str(ew) + "x" + str(eh),
          "- equirect contract: " + manifest["equirect"],
          "- note: " + (args.note or "-"), ""]
    if argv_out:
        md += ["## Encode command (printed by package_vr.py, not executed)", "",
               FENCE + "sh", " ".join(shell(a) for a in argv_out), FENCE, "",
               "Spherical metadata is written but unverified.", ""]
    md += ["## Caveat", "", manifest["caveat"], "", "## Files", ""]
    for e in entries:
        md.append("- " + e.get("packaged_as", e["file"]) + "  " + str(e["w"])
                  + "x" + str(e["h"]) + "  " + e["stereo"] + "  sha256 "
                  + e.get("sha256", "-"))
    (out_dir / "MANIFEST.md").write_text("\n".join(md) + "\n")
    print("wrote " + str(out_dir / "manifest.json") + " + MANIFEST.md; copied "
          + str(len(entries)) + " pano(s) into " + str(dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
