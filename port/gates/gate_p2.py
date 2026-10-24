#!/usr/bin/env python3
"""Gate P2 — static checks on the Vulkan host layer. Run: .venv/bin/python port/gates/gate_p2.py"""
import ast, py_compile, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VR = ROOT / "engine_vr"
FAILS = []

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)

def sh(args, timeout=120):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[:800]

hosts = [p for p in sorted(VR.glob("*.py")) if not re.match(r"(build_p1|selfcheck_p1|glstrip_vr)\.py$", p.name)]
check("host modules exist (gl_vr.py, gpuslot_vr.py)", (VR / "gl_vr.py").exists() and (VR / "gpuslot_vr.py").exists(),
      str([p.name for p in hosts]))
for p in hosts:
    try:
        py_compile.compile(str(p), doraise=True); ok, det = True, ""
    except Exception as e:
        ok, det = False, str(e)[:200]
    check(f"py_compile {p.name}", ok, det)

rc, out = sh([".venv/bin/python", "-m", "ruff", "check", "--select", "E,F", "--quiet", "engine_vr/"])
check("ruff E,F clean on engine_vr/", rc == 0, out.strip()[-400:])

BAD_MODS = {"vulkan", "moderngl", "wgpu"}
for p in hosts:
    tree = ast.parse(p.read_text())
    bad = []
    for st in tree.body:  # module level only
        for node in ast.walk(st):
            if isinstance(node, ast.Call):
                f = node.func
                root = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                base = None
                n = f
                while isinstance(n, ast.Attribute):
                    n = n.value
                if isinstance(n, ast.Name):
                    base = n.id
                if base in BAD_MODS or root in {"create_standalone_context", "create_device"}:
                    bad.append(f"line {node.lineno}: {ast.unparse(node)[:60]}")
    check(f"no module-level device/context calls in {p.name}", not bad, "; ".join(bad[:3]))

allsrc = "".join(p.read_text(errors="ignore") for p in VR.glob("*.py"))
check("no moderngl anywhere in engine_vr", "moderngl" not in allsrc)
check("manifest is the binding source (gl_vr reads manifest.json)", "manifest.json" in (VR / "gl_vr.py").read_text() if (VR / "gl_vr.py").exists() else False)
lit = re.findall(r"binding\s*=\s*\d+", (VR / "gl_vr.py").read_text() if (VR / "gl_vr.py").exists() else "")
check("no hardcoded numeric bindings in gl_vr.py", not lit, str(lit[:5]))

rc, out = sh(["git", "status", "--porcelain"])
dirty_ro = [l for l in out.splitlines() if l[3:].split(" -> ")[0].strip().split("/")[0] in ("engine", "audio", "docs")
            or l[3:].strip() in ("film.py", "still.py")]
check("read-only trees untouched", not dirty_ro, str(dirty_ro))

print(f"\nGATE P2: {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'}")
sys.exit(0 if not FAILS else 1)
