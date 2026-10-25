#!/usr/bin/env python3
"""check_viewpoints.py - CPU buried-camera guard for vr.py's 9 VIEWPOINTS.

port/PHASE_5.md T2. Why this exists: vr.py's camera recipes are annotated
copies of the film's NOMINAL keyframe-0 setups - the film's neighbourhood
SEARCH that framed (and lifted) the real cameras was NOT reproduced
(vr.py VIEWPOINTS FIXME(port) :302-304; port/reports/phase4.md risk 5). On a
re-baked MoltenVK world a nominal spot can land inside the new terrain, and
the pano renderer will happily render from inside a rock without raising
anything. This guard answers, per viewpoint, how far the head sits above the
baked ground/water of THIS cache before any GPU pano probe
(port/RUN_FIRST.md section 5) spends a render on it.

PROVENANCE. Round 1 of Phase 5 left this file; round 2 (this version) kept
the interface the brief and the gate depend on (--cache / --min-clearance /
--dry-run, exit 0 = clear, 1 = flagged, 2 = cache unusable) and the clearance
rule, and fixed the two things that made it unrunnable as ladder step 0:

1. T2 asks for "flag < 5 m as buried", but six of the nine recipes STAND at
   agl 1.0-2.2 m by design (vr.py:312/324/378/404 place through
   world.P(x, z, agl); :333/:356 sit at an absolute y of 1.7 / 2.2). With the
   mandated 5 m default a HEALTHY cache therefore flags every standing camera
   and step 0 aborts the ladder on a correct world. This tool keeps the 5 m
   default and the non-zero exit, but it now separates BELOW-SURFACE (head
   under the surface - always fatal) from BELOW-THRESH (under the threshold -
   fatal as specified, but annotated with the recipe's own nominal stance),
   and --nominal-ok exempts a placed recipe that reports exactly the agl its
   source line passes to P(). Read the classification, not just the exit code.
2. It reports which cache map each sample came from and whether that map's
   shape/dtype still matches engine_vr/cache_schema.py. A wrong-resolution
   bake buries or floats every camera at once, and without this line that
   failure looks like nine unrelated bad viewpoints.

CPU ONLY. It imports vr.py, and vr.py imports only numpy, engine/post.py,
engine_vr.cache_schema and engine_vr.pano at module level; the Vulkan layer
(engine_vr.passes_vr / gl_vr / bake_vr) is imported only inside vr._render()
(vr.py module docstring, "RUN MODES"). So this tool runs with no GPU, no
vulkan environment sourced, and no cache present (--dry-run).

Clearance convention - vr.py's OWN world: PanoWorld.ground(x, z) =
max(height, water) copied from engine/shots_film.py:21-22; PanoWorld.height
(vr.py:189-202) samples h_near.npy inside 0.95 * NEAR_HALF on both axes and
h_far.npy everywhere else, which is why the per-sample map name is printed.

Exit codes: 0 = nothing flagged; 1 = BELOW-SURFACE / BELOW-THRESH / UNKNOWN /
map spec mismatch; 2 = cache missing, unloadable, or import failed.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MIN_CLEARANCE_M = 5.0

# How each recipe derives its head height, read off vr.py. This is
# documentation, not a live binding: if a recipe changes, this table goes
# stale and the --nominal-ok exemption is simply wrong for that row.
#   "agl"  placed through world.P(x, z, agl)  -> clearance == agl exactly
#   "agl+" world.P-like with max(ground, 0)   -> clearance >= agl
#   "abs"  an absolute y in the source        -> clearance is a real question
NOMINAL = {
    "forest": ("agl", 1.9, "vr.py:312 w.P(trail[i0], 1.9)"),
    "reveal": ("agl", 1.0, "vr.py:324 w.P(xr - 0.4, zb - 4.8, 1.0)"),
    "skim": ("abs", 1.7, "vr.py:333 absolute y = 1.7, over the lake"),
    "waterfall": ("abs", 2.2, "vr.py:356 absolute y = 2.2"),
    "stream": ("agl+", 1.5, "vr.py:371 max(ground, 0) + 1.5"),
    "meadow": ("agl", 1.6, "vr.py:378 w.P(-2060, -40, 1.6)"),
    "flyover": ("abs", 620.0, "vr.py:385 absolute y = 620 (airborne)"),
    "peaks": ("abs", 700.0, "vr.py:395 absolute y = 700 (airborne)"),
    "finale": ("agl", 1.7, "vr.py:404 w.P(x9, 300, 1.7)"),
}

# First move when a recipe is flagged, keyed by viewpoint name. Each line
# cites the port/RUN_FIRST.md section-6 failure-signature matrix row and the
# report that recorded the mechanism, so a failed check routes to a probe
# instead of to a re-bake guess.
REMEDIES = {
    "forest": "trail point comes from features.json; a shifted spot means the "
              "carve ran after the height bake - RUN_FIRST matrix 4/22",
    "reveal": "hero boulder + shore_south(xr) come from this cache: check "
              "features.json and h_near shape/dtype first - matrix 4/20",
    "skim": "absolute y over the lake: water_level.npy is the surface, so a "
            "buried skim camera means the flood fill ran on a different "
            "height map - matrix 23",
    "waterfall": "absolute y from features waterfall x/z/dir: re-check the "
                 "carve before the height map - matrix 4/22",
    "stream": "uses max(ground, 0) + 1.5, so BELOW-SURFACE here means the "
              "stream features and the baked ground disagree - matrix 4/22",
    "meadow": "near-map sample (h_near): compare shape against "
              "cache_schema.shape_of - matrix 2/20",
    "flyover": "airborne at y = 620: a flagged flyover cannot be a camera "
               "bug - the cache is a different world or the wrong resolution; "
               "re-run step 0 with --cache at the cache you just baked",
    "peaks": "airborne at y = 700: same reading as flyover",
    "finale": "far-map sample at shore_x(300, -1) - 2: check h_far "
              "shape/cell and that erosion did not plane the ridge away "
              "(finite heights + flow sum) - matrix 12/20",
}
DEFAULT_REMEDY = ("check the map shapes against cache_schema first, then the "
                  "erosion finite/flow result - matrix 12/20")
MAPS = ("h_far.npy", "h_near.npy", "water_level.npy")


def _map_for(cs, x, z):
    """Which height map PanoWorld.height() will sample at this position.

    Mirrors vr.py:200-202 (near map wins inside 0.95 * NEAR_HALF on both
    axes), so the printed map name is the map that produced the number.
    """
    if max(abs(x), abs(z)) < cs.NEAR_HALF * 0.95:
        return "h_near.npy"
    return "h_far.npy"


def _map_lines(world, cs):
    """Cache provenance: shape/dtype of each loaded map vs the contract."""
    out = []
    for name, arr in (("h_far.npy", world.hfar), ("h_near.npy", world.hnear),
                      ("water_level.npy", world.level)):
        stem = name[:-4]
        try:
            want = cs.shape_of(stem)
            cell = cs.cell(stem)
            ok = tuple(arr.shape) == tuple(want)
            out.append(f"    {name:<16} shape={str(tuple(arr.shape)):<14} "
                       f"dtype={arr.dtype}  contract {tuple(want)} "
                       f"cell {cell:.2f} m  {'ok' if ok else 'SPEC-MISMATCH'}")
        except KeyError as exc:
            out.append(f"    {name:<16} not in the cache contract: {exc}")
    return out


def _classify(name, clear, args, nominal):
    """(flag, note) for one measured clearance."""
    mode, want, cite = nominal
    if clear < 0.0:
        return "BELOW-SURFACE", f"head is {abs(clear):.1f} m under the surface"
    if clear < args.min_clearance:
        if (args.nominal_ok and mode in ("agl", "agl+")
                and abs(clear - want) < 1e-6):
            return "OK", f"nominal stance {want} m by construction ({cite})"
        return ("BELOW-THRESH",
                f"under the {args.min_clearance:g} m threshold; recipe "
                f"stance is {mode} {want} m ({cite})")
    return "OK", f"{mode} {want} m ({cite})"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="check_viewpoints.py",
        description="CPU-only buried-camera guard for vr.py VIEWPOINTS "
                    "(port/PHASE_5.md T2; never opens a device)")
    ap.add_argument("--cache", default=None,
                    help="cache directory (sets ALPINE_CACHE for the "
                         "process, same rule as vr.py --cache)")
    ap.add_argument("--min-clearance", type=float, metavar="M",
                    default=DEFAULT_MIN_CLEARANCE_M,
                    help="flag a camera below this head clearance in metres "
                         "(default 5 m as specified by port/PHASE_5.md T2; "
                         "the recipes stand at agl 1.0-2.2 m on purpose, so "
                         "see --nominal-ok)")
    ap.add_argument("--nominal-ok", action="store_true",
                    help="do not flag a placed recipe whose clearance is "
                         "exactly the agl its source line passes to world.P "
                         "- use this for the pass/fail form of ladder step 0")
    ap.add_argument("--no-spec-check", action="store_true",
                    help="do not fail on cache_schema shape mismatches "
                         "(synthetic test caches only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what WOULD be checked; reads no cache, "
                         "touches no device (pre-bake smoke mode)")
    args = ap.parse_args(argv)

    try:
        import vr
        from engine_vr import cache_schema as cs
    except Exception as exc:
        print(f"import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.cache:
        cs.set_cache_dir(args.cache)

    if args.dry_run:
        print(f"WOULD CHECK {len(vr.VIEWPOINTS)} vr.py viewpoints, cache "
              f"{cs.cache_dir()}")
        print(f"  rule: clearance = pos.y - PanoWorld.ground(pos.x, pos.z); "
              f"flag below {args.min_clearance:g} m, and anything under the "
              f"surface is BELOW-SURFACE")
        for i, vp in enumerate(vr.VIEWPOINTS):
            mode, want, cite = NOMINAL.get(
                vp["name"], ("?", float("nan"), "no nominal row"))
            print(f"  WOULD CHECK vp{i} {vp['name']:<9} roll={vp['roll']} "
                  f"seed={vp['seed']} stance={mode} {want} m  ({cite})")
        print("  cache files it would load: " + ", ".join(MAPS)
              + ", features.json (optional)")
        missing = cs.missing()
        print("  cache files missing right now: "
              + (", ".join(missing) or "none"))
        print("  dry-run resolved nothing - this is expected pre-bake")
        print("  real runs (port/RUN_FIRST.md section 3):")
        print(f"    .venv/bin/python engine_vr/check_viewpoints.py "
              f"--cache \"{cs.cache_dir()}\"            # brief form: 5 m")
        print(f"    .venv/bin/python engine_vr/check_viewpoints.py "
              f"--cache \"{cs.cache_dir()}\" --nominal-ok "
              f"--min-clearance 0.5   # pass/fail form")
        return 0

    missing = cs.missing()
    if missing:
        print("cache incomplete - run the bake ladder in port/RUN_FIRST.md "
              "first; missing: " + ", ".join(missing), file=sys.stderr)
        return 2
    try:
        world = vr.PanoWorld()
    except Exception as exc:
        print(f"PanoWorld load failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    bad = 0
    print(f"clearance check on cache {cs.cache_dir()} "
          f"(flag below {args.min_clearance:g} m; nominal-ok="
          f"{args.nominal_ok})")
    spec_bad = False
    for line in _map_lines(world, cs):
        print(line)
        if "SPEC-MISMATCH" in line:
            spec_bad = True
    if spec_bad and not args.no_spec_check:
        bad += 1
        print("  map shapes do not match cache_schema: every clearance below "
              "is measured against the wrong world (matrix 2/20)")
    feats = "present" if world.features else "MISSING"
    print(f"    features.json  {feats}")
    for i, vp in enumerate(vr.VIEWPOINTS):
        name = vp["name"]
        nominal = NOMINAL.get(name, ("?", float("nan"), "no nominal row"))
        pos = (float("nan"),) * 3
        terr = clear = float("nan")
        try:
            pos, _tgt, _heroes = vr.resolve_viewpoint(world, i)
            terr = world.ground(float(pos[0]), float(pos[2]))
            clear = float(pos[1]) - terr
            flag, note = _classify(name, clear, args, nominal)
        except Exception as exc:
            bad += 1
            print(f"  vp{i} {name:<9} pos=(nan)  clearance=   n/a  UNKNOWN")
            print(f"      resolve failed: {type(exc).__name__}: {exc}")
            print(f"      remedy: {REMEDIES.get(name, DEFAULT_REMEDY)}")
            continue
        if flag != "OK":
            bad += 1
        print(f"  vp{i} {name:<9} "
              f"pos=({pos[0]:9.1f},{pos[1]:8.1f},{pos[2]:9.1f})  "
              f"surface={terr:8.1f} [{_map_for(cs, pos[0], pos[2])}]  "
              f"clearance={clear:8.1f} m  {flag}")
        print(f"      {note}")
        if flag != "OK":
            print(f"      remedy: {REMEDIES.get(name, DEFAULT_REMEDY)}")
    print("RESULT: " + ("ALL CLEAR" if bad == 0 else
                        f"{bad} flagged - do not start the pano probe ladder "
                        "until BELOW-SURFACE / UNKNOWN / SPEC-MISMATCH are "
                        "gone (port/RUN_FIRST.md section 3 and 6)"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
