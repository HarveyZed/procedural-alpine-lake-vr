#!/usr/bin/env python3
"""build_p4.py - Phase-4 T3 builder: SPIR-V parity proof + scene_pano_* set.

Three steps, in this order:

1. PARITY (PHASE_4.md T3 invariant a): snapshot sha256 of every committed
   engine_vr/spirv/*.spv, then re-run the FULL Phase-1/2 build of every base
   program (build_p1.PROGRAMS + build_p1.EXTRA + the shared vertex stage)
   WITHOUT the VR_PANO define, and require every .spv to come back
   byte-identical. The VR_PANO branches added to engine_vr/shaders/
   {world.glsl, scene.frag} are resolved away by glstrip_vr before
   vulkanize when the define is absent, so the generated sources are the
   same bytes; glslc is deterministic (proven in P2, same flags via
   build_p1.compile_source). Any mismatch exits 1 without touching the
   manifest.

2. PANO VARIANTS: scene_pano_cone / scene_pano_trace / scene_pano_kind0..7
   - same scene.frag, defines {QUALITY 1, SCENE_PASS k, VR_PANO 1}. Every
   added shader line is FIXME(port)-annotated (T3 invariant b/c).

3. MANIFEST: append the new entries to engine_vr/manifest.json. Existing
   entries are kept byte-identical (the file is re-dumped with the same
   json.dumps(indent=1) formatting build_p1 uses); rerunning this script
   replaces only the scene_pano_* entries.

Run: .venv/bin/python engine_vr/build_p4.py   (takes ~as long as a P1 build)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_p1 as B                                  # noqa: E402
from build_p1 import KIND_NAMES                       # noqa: E402

SPIRV = HERE / "spirv"
MANIFEST = HERE / "manifest.json"


def _spec(name, pass_id, note):
    return {"name": name, "frag": "scene.frag",
            # STRING define values, like every build_p1 spec:
            # compile_source's truthiness filter (v not in (None, "", True))
            # drops the "=1" for an int 1 (1 == True), and glslc then sees a
            # bare -DQUALITY conflicting with the generated header's
            # #define. build_p1's own specs all use strings for this reason.
            "defines": {"QUALITY": "1", "SCENE_PASS": str(pass_id),
                        "VR_PANO": "1"},
            "call_site": "vr.py (Phase-4 pano renderer, T3)", "note": note,
            "strip": True}


PANO_SPECS = [_spec("scene_pano_cone", 98,
                    "equirect cone pre-pass: block-centre lon/lat ray, "
                    "per-texel footprint strip for the cone half-angle"),
              _spec("scene_pano_trace", 99,
                    "equirect primary raygen: texel-centre lon/lat ray "
                    "(RUNBOOK §6), gPixAng = per-texel footprint, no DOF")]
for _k in range(8):
    PANO_SPECS.append(
        _spec(f"scene_pano_kind{_k}_{KIND_NAMES[_k]}", _k,
              f"equirect kind-pass {_k} ({KIND_NAMES[_k]}); directions "
              "identical to scene_pano_trace"))


def hash_spv():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SPIRV.glob("*.spv"))}


def main():
    t0 = time.time()
    before = hash_spv()
    print(f"parity: {len(before)} committed .spv hashed")

    fails = []
    for spec in B.PROGRAMS + B.EXTRA:
        e, _res = B.build_one(spec)
        ok = e["compiled"] and e["validated"]
        print(("OK   " if ok else "FAIL ") + "base " + e["name"])
        if not ok:
            fails.append(e["name"])
            print("     " + "\n     ".join(
                (e.get("error") or "?").splitlines()[:8]))
    vs = B.build_vertex()
    print(("OK   " if vs["compiled"] and vs["validated"] else "FAIL ")
          + "base _fullscreen.vert")
    if not (vs["compiled"] and vs["validated"]):
        fails.append("_fullscreen.vert")

    after = hash_spv()
    same = [n for n in before if before[n] == after.get(n)]
    diff = [n for n in before if before[n] != after.get(n)]
    print(f"SPIR-V parity: {len(same)}/{len(before)} hashes match"
          + ("" if not diff else "  MISMATCH: " + ", ".join(diff)))
    if diff or fails:
        print("PARITY FAILED - manifest left untouched")
        return 1

    entries = []
    for spec in PANO_SPECS:
        e, _res = B.build_one(spec)
        ok = e["compiled"] and e["validated"]
        print(("OK   " if ok else "FAIL ") + "pano " + e["name"])
        if not ok:
            print("     " + "\n     ".join(
                (e.get("error") or "?").splitlines()[:8]))
            return 1
        entries.append(e)

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = {e["name"] for e in entries}
    progs = [p for p in man["programs"] if p["name"] not in names]
    progs.extend(entries)
    man["programs"] = progs
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n",
                        encoding="utf-8")
    print(f"manifest: {len(progs)} programs "
          f"(+{len(entries)} scene_pano_*) in {time.time() - t0:.0f}s")
    for e in entries:
        mem = [m["name"] for m in e["uniform_block"]["members"]]
        print(f"  {e['name']}: UBO {e['uniform_block']['size_bytes']} B, "
              f"{len(mem)} members, samplers "
              f"{[b['name'] for b in e['bindings'] if 'sampler' in b['kind']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
