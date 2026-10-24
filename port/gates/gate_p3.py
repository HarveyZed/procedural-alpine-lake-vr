#!/usr/bin/env python3
"""Gate P3 — static cross-validation of bake/erosion/features VR drivers vs manifest. Run: .venv/bin/python port/gates/gate_p3.py"""
import ast, importlib, json, py_compile, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
VR = ROOT / "engine_vr"
sys.path.insert(0, str(ROOT))
FAILS = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" — " + detail) if detail else "")); 
    if not ok: FAILS.append(name)
def sh(args, t=120):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=t)
    return p.returncode, (p.stdout + p.stderr)[:900]
mods = ["bake_vr", "erosion_vr", "features_vr", "cache_schema"]
for m in mods:
    check(f"{m}.py exists", (VR / f"{m}.py").exists())
for p in sorted(VR.glob("*.py")):
    try: py_compile.compile(str(p), doraise=True); ok, d = True, ""
    except Exception as e: ok, d = False, str(e)[:180]
    check(f"py_compile {p.name}", ok, d)
rc, out = sh([".venv/bin/python", "-m", "ruff", "check", "--select", "E,F", "--quiet", "engine_vr/"])
check("ruff E,F engine_vr/", rc == 0, out.strip()[-300:])
def imports_of(path):
    s = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import): s |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module: s.add(n.module.split(".")[0])
    return s
check("no moderngl/wgpu imports in engine_vr", not any({"moderngl", "wgpu"} & imports_of(p) for p in VR.glob("*.py")))
BAD = {"vulkan", "moderngl", "wgpu"}
for m in mods:
    f = VR / f"{m}.py"
    if not f.exists(): continue
    bad = []
    for st in ast.parse(f.read_text()).body:
        for node in ast.walk(st):
            if isinstance(node, ast.Call):
                n = node.func
                while isinstance(n, ast.Attribute): n = n.value
                if isinstance(n, ast.Name) and n.id in BAD: bad.append(f"line {node.lineno}")
    check(f"no module-level {m} device calls", not bad, ";".join(bad[:3]))
man = json.loads((VR / "manifest.json").read_text())
progs = {p["name"]: p for p in (man["programs"] if isinstance(man, dict) else man)}
for m in mods[:3]:
    f = VR / f"{m}.py"
    if not f.exists(): continue
    try:
        mod = importlib.import_module(f"engine_vr.{m}")
        plan = mod.build_plan()
        errs = []
        for name, ent in plan.items():
            pr = progs.get(ent.get("program"))
            if pr is None: errs.append(f"{name}: program {ent.get('program')!r} not in manifest"); continue
            members = {mm["name"] for mm in (pr.get("uniform_block") or {}).get("members", [])}
            for u in ent.get("uniforms", {}) or {}:
                if u not in members: errs.append(f"{name}: uniform {u!r} not a manifest member")
            simg = {b["name"] for b in pr.get("bindings", []) if "image" in str(b.get("kind", "")).lower()}
            for si in ent.get("storage_images", []) or []:
                if si not in simg: errs.append(f"{name}: storage image {si!r} not in manifest bindings")
        check(f"build_plan cross-validates {m}", not errs and len(plan) > 0, "; ".join(errs[:4]) or f"{len(plan)} passes")
    except Exception as e:
        check(f"build_plan cross-validates {m}", False, f"{type(e).__name__}: {str(e)[:160]}")
rc, out = sh(["git", "status", "--porcelain"])
dirty = [l for l in out.splitlines() if l[3:].strip().split("/")[0] in ("engine", "audio", "docs") or l[3:].strip() in ("film.py", "still.py")]
check("read-only trees untouched", not dirty, str(dirty[:2]))
print(f"\nGATE P3: {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'}")
sys.exit(0 if not FAILS else 1)
