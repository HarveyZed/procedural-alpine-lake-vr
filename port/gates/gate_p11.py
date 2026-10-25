#!/usr/bin/env python3
"""Gate P11 - animated shot flow (static + evidence lint)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REP = REPO / "port/reports/phase11.md"


def main() -> int:
    bad = []

    def req(cond, msg):
        if not cond:
            bad.append(msg)

    r = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=REPO,
                       capture_output=True, text=True)
    mod = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    req(bool(mod), "nothing modified vs HEAD (review-time gate)")
    for f in mod:
        p = REPO / f
        if p.exists() and f.endswith(".py"):
            req("FIXME(p11)" in p.read_text(encoding="utf-8",
                                           errors="replace"),
                f"{f} modified without FIXME(p11)")
    req(not any(f.startswith("engine/") for f in mod),
        "engine/ files modified in P11 - should be import-over-copy")

    def je(name, chk):
        p = REPO / "engine_vr/tmp" / name
        if not p.exists():
            req(False, f"missing engine_vr/tmp/{name}")
            return
        try:
            chk(json.loads(p.read_text()))
        except Exception as e:
            req(False, f"{name}: {e}")

    def shot(d):
        assert d.get("shot") == "skim", d.get("shot")
        assert d.get("frame_count", 0) > 0
        assert d["t1"] > d["t0"]
    je("p11_shot_probe.json", shot)

    def fp(d):
        assert d.get("camera_moved") is True
        assert d.get("water_animated") is True
        assert d.get("determinism_bit_identical") is True
    je("p11_frame_probe.json", fp)

    if not REP.exists():
        bad.append("phase11.md missing")
    else:
        txt = REP.read_text(encoding="utf-8", errors="replace").lower()
        for s in ("shim", "time model", "resume", "encode", "residual"):
            req(s in txt, f"phase11.md missing: {s}")
    for f in ("vr.py",):
        rc = subprocess.run([str(REPO / ".venv/bin/python"), "-m",
                             "py_compile", str(REPO / f)],
                            capture_output=True, text=True)
        req(rc.returncode == 0, f"{f} compile")

    if bad:
        for b in bad:
            print("FAIL:", b)
        return 1
    print("GATE P11: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
