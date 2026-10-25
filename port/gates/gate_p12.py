#!/usr/bin/env python3
"""Gate P12 - motion blur parity (static + evidence)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    bad = []

    def req(c, m):
        if not c:
            bad.append(m)

    r = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=REPO,
                       capture_output=True, text=True)
    mod = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    req(any(f == "vr.py" for f in mod), "vr.py not modified")
    req(not any(f.startswith("engine/") for f in mod), "engine/ modified")
    src = (REPO / "vr.py").read_text(encoding="utf-8")
    req("FIXME(p12)" in src, "no FIXME(p12) markers")
    req("--shutter" in src, "no --shutter flag")
    p = REPO / "engine_vr/tmp/p12_evidence.json"
    if not p.exists():
        bad.append("missing p12_evidence.json")
    else:
        d = json.loads(p.read_text())
        req(float(d.get("edge_blur_ratio", 1.0)) < 0.9,
            "edge_blur_ratio not < 0.9")
        req(d.get("shutter0_parity") is True, "shutter0 parity not true")
        req(d.get("determinism") is True, "determinism not true")
    rep = REPO / "port/reports/phase12.md"
    req(rep.exists(), "phase12.md missing")
    if rep.exists():
        t = rep.read_text(encoding="utf-8", errors="replace").lower()
        for s in ("shutter", "parity", "accum", "residual"):
            req(s in t, "phase12.md missing section: " + s)
    rc = subprocess.run([str(REPO / ".venv/bin/python"), "-m",
                         "py_compile", str(REPO / "vr.py")],
                        capture_output=True, text=True)
    req(rc.returncode == 0, "vr.py compile")
    if bad:
        for b in bad:
            print("FAIL:", b)
        return 1
    print("GATE P12: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
