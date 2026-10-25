#!/usr/bin/env python3
"""Gate P7 - G-buffer wiring + determinism. Static checks + report lint.
Usage: .venv/bin/python port/gates/gate_p7.py"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REP = REPO / "port/reports/phase7.md"


def main() -> int:
    bad = []

    def req(cond, msg):
        if not cond:
            bad.append(msg)

    r = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=REPO,
                       capture_output=True, text=True)
    mod = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    req(mod, "nothing modified vs HEAD")
    for f in mod:
        p = REPO / f
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        if f.endswith((".py", ".glsl", ".frag")):
            req("FIXME(p7)" in txt, f"{f} modified without FIXME(p7)")

    # SPIR-V parity: film programs byte-identical claim must be evidenced
    par = REPO / "engine_vr/tmp/p7_parity.json"
    req(par.exists(), "engine_vr/tmp/p7_parity.json missing (parity run)")
    if par.exists():
        try:
            d = json.loads(par.read_text())
        except Exception as e:
            req(False, f"parity json: {e}")
            d = {}
        if d:
            req(d.get("film_identical") is True,
                "parity json film_identical != true")
            req(isinstance(d.get("film_programs"), int)
                and d["film_programs"] >= 28,
                "parity json must cover >=28 film programs")

    # determinism evidence
    det = REPO / "engine_vr/tmp/p7_determinism.json"
    req(det.exists(), "engine_vr/tmp/p7_determinism.json missing")
    if det.exists():
        try:
            dd = json.loads(det.read_text())
            req(dd.get("max_drgb") == 0 or dd.get("residual_reason"),
                "determinism json needs max_drgb==0 or residual_reason")
        except Exception as e:
            req(False, f"determinism json: {e}")

    if not REP.exists():
        bad.append("port/reports/phase7.md missing")
    else:
        txt = REP.read_text(encoding="utf-8", errors="replace").lower()
        for s in ("gl-era truth", "kind", "parity", "determin"):
            req(s in txt, f"phase7.md missing topic: {s}")

    rc = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "py_compile",
                         str(REPO / "vr.py")], capture_output=True, text=True)
    req(rc.returncode == 0, "vr.py no longer compiles")

    if bad:
        for b in bad:
            print("FAIL:", b)
        return 1
    print("GATE P7: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
