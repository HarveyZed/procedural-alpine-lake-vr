#!/usr/bin/env python3
"""Gate P8 - eyes photometry + mountain tile holes. Static + evidence lint."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REP = REPO / "port/reports/phase8.md"


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
        if p.exists() and f.endswith((".py", ".glsl", ".frag")):
            req("FIXME(p8)" in p.read_text(encoding="utf-8",
                                           errors="replace"),
                f"{f} modified without FIXME(p8)")

    def je(name, chk):
        p = REPO / "engine_vr/tmp" / name
        if not p.exists():
            req(False, f"missing engine_vr/tmp/{name}")
            return
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            req(False, f"{name}: {e}")
            return
        req(chk(d), f"{name} failed check: {d}")

    je("p8_eye_photometry.json",
       lambda d: isinstance(d.get("lr_mean_ratio"), (int, float))
       and abs(d["lr_mean_ratio"] - 1.0) < 0.01)
    je("p8_cap_test.json",
       lambda d: d.get("foot8192_repro_exit") == 0)
    par = REPO / "engine_vr/tmp/p8_parity.json"
    shaders = [f for f in mod if "shaders/" in f or f.endswith(".frag")]
    if shaders or not par.exists():
        if shaders:
            je("p8_parity.json", lambda d: d.get("film_identical") is True)

    if not REP.exists():
        bad.append("phase8.md missing")
    else:
        txt = REP.read_text(encoding="utf-8", errors="replace").lower()
        for s in ("bisect", "cap", "acceptance", "residual"):
            req(s in txt, f"phase8.md missing: {s}")

    rc = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "py_compile",
                         str(REPO / "vr.py")], capture_output=True, text=True)
    req(rc.returncode == 0, "vr.py compile")

    if bad:
        for b in bad:
            print("FAIL:", b)
        return 1
    print("GATE P8: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
