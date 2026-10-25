#!/usr/bin/env python3
"""Gate P5 — static validation of first-run protocol + packaging. Run: .venv/bin/python port/gates/gate_p5.py"""
import ast, py_compile, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FAILS = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" - " + detail) if detail else ""))
    if not ok: FAILS.append(name)
def sh(args, t=120):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=t)
    return p.returncode, (p.stdout + p.stderr)[:2000]
docs = ROOT / "port" / "RUN_FIRST.md"
check("RUN_FIRST.md exists", docs.exists())
if docs.exists():
    t = docs.read_text()
    check("probes #1..#13 all present", all(("#%d" % i) in t or ("#%d" % i).replace("#", "# ") in t for i in range(1, 14)), "probe refs")
    check("env table has ALPINE_ vars", "ALPINE_MAX_PROC_GB" in t and "ALPINE_CACHE" in t and "env.sh" in t)
    check("budget table present", bool(re.search(r"\bMB\b|\bGB\b", t)))
    check("failure-signature section", "failure" in t.lower() and "signature" in t.lower())
    check("step-0 clearance check referenced", "check_viewpoints" in t)
    check("acceptance + pano-vs-film caveat", "NOT comparable" in t or "not comparable" in t.lower())
for f in [ROOT / "package_vr.py", ROOT / "engine_vr" / "check_viewpoints.py"]:
    check("%s exists" % f.name, f.exists())
    if f.exists():
        try: py_compile.compile(str(f), doraise=True); check("py_compile %s" % f.name, True)
        except Exception as e: check("py_compile %s" % f.name, False, str(e)[:150])
def imports_of(path):
    s = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import): s |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module: s.add(n.module.split(".")[0])
    return s
watch = [ROOT / "package_vr.py", ROOT / "engine_vr" / "check_viewpoints.py", ROOT / "vr.py", *sorted((ROOT / "engine_vr").glob("*.py"))]
check("no moderngl/wgpu imports", not any({"moderngl", "wgpu"} & (imports_of(p) if p.exists() else set()) for p in watch if p.is_file()))
rc, out = sh([".venv/bin/python", "-m", "ruff", "check", "--select", "E,F", "--quiet", "engine_vr/", "vr.py", "package_vr.py"])
check("ruff E,F", rc == 0, out.strip()[-300:])
rc, out = sh([".venv/bin/python", "package_vr.py", "--dry-run", "--in", "engine_vr/out/*.png", "--out-dir", "engine_vr/out/pkg"])
check("package_vr.py --dry-run exit0", rc == 0, out[:200])
rc, out = sh([".venv/bin/python", "package_vr.py", "--dry-run", "--video", "--in", "engine_vr/out/*.png", "--out-dir", "engine_vr/out/pkg"])
okv = rc == 0 and ("ffmpeg" in out and "equirect" in out)
check("--video dry-run prints ffmpeg argv w/ equirect metadata, no spawn", okv, out[:220])
rc, out = sh([".venv/bin/python", "engine_vr/check_viewpoints.py", "--dry-run"])
check("check_viewpoints --dry-run exit0 pre-bake", rc == 0, out[:200])
rc, out = sh(["git", "status", "--porcelain"])
dirty = [l for l in out.splitlines() if l[3:].strip().split("/")[0] in ("engine", "audio", "docs") or l[3:].strip() in ("film.py", "still.py")]
check("read-only trees untouched", not dirty, str(dirty[:2]))
print("")
print("GATE P5: " + ("PASS" if not FAILS else "FAIL (" + str(len(FAILS)) + ")"))
sys.exit(0 if not FAILS else 1)
