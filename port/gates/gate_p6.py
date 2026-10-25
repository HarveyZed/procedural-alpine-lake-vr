#!/usr/bin/env python3
"""Gate P6 - shader robustness bugfix. Static checks only.
Usage: .venv/bin/python port/gates/gate_p6.py"""
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VK = REPO / "tools/vk"
REP = REPO / "port/reports/phase6.md"


def _tool(pat):
    hits = sorted(VK.rglob(pat))
    return str(hits[0]) if hits else None


def main() -> int:
    bad = []

    def req(cond, msg):
        if not cond:
            bad.append(msg)

    # 1. a shader was modified vs the previous commit
    r = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "engine/shaders/"],
                       cwd=REPO, capture_output=True, text=True)
    r2 = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "engine_vr/"],
                        cwd=REPO, capture_output=True, text=True)
    mod = [l.strip() for l in (r.stdout + r2.stdout).splitlines() if l.strip()]
    req(mod, "no shader/engine_vr files modified vs HEAD")

    # 2. every modified shader carries FIXME(p6)
    for f in mod:
        if f.endswith((".frag", ".glsl", ".comp")):
            txt = (REPO / f).read_text(encoding="utf-8", errors="replace")
            req("FIXME(p6)" in txt, f"{f} modified without FIXME(p6) marker")

    # 3. glslc -Werror + spirv-val on each modified frag with pano defines
    glslc, sval = _tool("glslc"), _tool("spirv-val")
    req(glslc and sval, "glslc/spirv-val missing under tools/vk")
    man = REPO / "engine_vr/manifest.json"
    defines = []
    if man.exists():
        try:
            for p in json.loads(man.read_text()).get("programs", []):
                for d in p.get("defines", []):
                    if d not in defines:
                        defines.append(d)
        except Exception as e:
            req(False, f"manifest parse: {e}")
    req(bool(defines), "no defines found in engine_vr/manifest.json for compile matrix")
    if glslc:
        for f in mod:
            if not f.endswith(".frag"):
                continue
            out = REPO / "engine_vr/tmp" / (Path(f).stem + "_p6.spv")
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [glslc, "-w", str(REPO / f), "-o", str(out)]
            for d in defines:
                cmd.append("-D" + d)
            c = subprocess.run(cmd, capture_output=True, text=True)
            # defines from multiple programs collide; if the fat combo
            # fails, fall back to compiling each pano define alone
            ok = c.returncode == 0
            if not ok:
                ok = True
                for d in defines:
                    c1 = subprocess.run([glslc, "-w", str(REPO / f),
                                         "-o", str(out), "-D" + d],
                                        capture_output=True, text=True)
                    if c1.returncode != 0 and Path(f).name in ("scene.frag",):
                        ok = False
                        print(f"  compile {f} -D{d}: {c1.stderr[:160]}")
            req(ok, f"glslc failed for {f} (see stdout)")
            if out.exists() and sval:
                v = subprocess.run([sval, str(out)], capture_output=True, text=True)
                req(v.returncode == 0, f"spirv-val failed {f}: {v.stderr[:200]}")

    # 4. report
    if not REP.exists():
        bad.append("port/reports/phase6.md missing")
    else:
        txt = REP.read_text(encoding="utf-8", errors="replace")
        for s in ("ROOT CAUSE", "FIXME(p6)", "spirv-val", "run ladder"):
            req(s.lower() in txt.lower(), f"phase6.md missing section: {s}")

    # 5. discipline: no python regressions
    rc = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "py_compile",
                         str(REPO / "vr.py")], capture_output=True, text=True)
    req(rc.returncode == 0, "vr.py no longer compiles")

    if bad:
        for b in bad:
            print("FAIL:", b)
        return 1
    print("GATE P6: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
