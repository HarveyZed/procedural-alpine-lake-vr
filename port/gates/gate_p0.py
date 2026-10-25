#!/usr/bin/env python3
"""Gate P0 — static checks only (RUNBOOK §2). Run: .venv/bin/python port/gates/gate_p0.py"""
import re, subprocess, sys, py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAILS, WARNS = [], []

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)

def sh(args, timeout=120):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[:400]

# 1. toolchain
glslc, sval = ROOT / "tools/vk/bin/glslc", ROOT / "tools/vk/bin/spirv-val"
for tool in (glslc, sval):
    ok = tool.exists() and sh([str(tool), "--version"])[0] == 0
    check(f"toolchain {tool.name} --version", ok)

# 2. compile probe (write .spv inside workspace only)
probe = ROOT / "spike/.gate_p0_probe.comp"
probe.write_text("#version 450\nlayout(local_size_x=16) in;\nlayout(std430,binding=0) buffer B{float d[];};\nvoid main(){d[gl_GlobalInvocationID.x]=1.0;}\n")
spv = ROOT / "spike/.gate_p0_probe.spv"
rc, out = sh([str(glslc), "--target-env=vulkan1.2", str(probe), "-o", str(spv)])
check("glslc compiles 450 compute probe", rc == 0, out.strip()[-200:] if rc else "")
if spv.exists():
    rc, out = sh([str(sval), str(spv)])
    check("spirv-val accepts probe", rc == 0, out.strip()[-200:] if rc else "")
    spv.unlink(missing_ok=True)
probe.unlink(missing_ok=True)

# 3. spike outputs
def ppm_ok(p):
    try:
        b = p.read_bytes()
        m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+255\s", b)
        return bool(m) and int(m.group(1)) >= 256 and int(m.group(2)) >= 256 and len(b) > 20
    except Exception:
        return False
for f in ("spike/p0_a/sphere.ppm", "spike/p0_b/sphere.ppm", "spike/p0_ref.ppm"):
    p = ROOT / f
    check(f"ppm {f}", ppm_ok(p))
for f in ("spike/p0_a/buf_check.txt", "spike/p0_b/buf_check.txt"):
    check(f"buf check exists {f}", (ROOT / f).exists())

# 4. report
rep = ROOT / "port/reports/phase0.md"
if not rep.exists():
    check("report exists", False)
else:
    t = rep.read_text()
    for sec in ("Status:", "Files written", "Deviations", "API contracts", "Risks"):
        check(f"report section '{sec}'", sec in t)
    psnr = [float(x) for x in re.findall(r"PSNR[^\d]{0,20}([\d.]+)", t)]
    check("PSNR values present (>=2 per backend)", len(psnr) >= 2, str(psnr))
    if psnr:
        check("PSNR >= 26 dB", all(v >= 26 for v in psnr), str(psnr))
    check("recommendation present", re.search(r"(?i)recommend", t) is not None)
    check("status DONE", "Status: DONE" in t or "Status:DONE" in t)

# 5. python hygiene: spike scripts compile; no moderngl in new code
for py in list((ROOT / "spike").rglob("*.py")):
    try:
        py_compile.compile(str(py), doraise=True)
        ok, det = True, ""
    except Exception as e:
        ok, det = False, str(e)[:200]
    check(f"py_compile {py.relative_to(ROOT)}", ok, det)
def _imports_moderngl(path):  # AST import check; substring hits are contract comments (gate_p2 lesson)
    import ast as _ast
    try:
        tree = _ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return True
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Import) and any(a.name.split(".")[0] == "moderngl" for a in n.names):
            return True
        if isinstance(n, _ast.ImportFrom) and (n.module or "").split(".")[0] == "moderngl":
            return True
    return False

hits = [str(p.relative_to(ROOT)) for p in list((ROOT / "spike").rglob("*.py")) + list((ROOT / "engine_vr").rglob("*.py")) if _imports_moderngl(p)]
check("no moderngl in new code", not hits, str(hits))

# 6. diff discipline (GC-run git; agents never git)
rc, out = sh(["git", "status", "--porcelain"])
dirty_ro = [l for l in out.splitlines() if l[3:].split(" -> ")[0].strip()
            .split("/")[0] in ("engine", "audio", "docs") or l[3:].strip() in ("film.py", "still.py")]
check("read-only trees untouched", not dirty_ro, str(dirty_ro))

print(f"\nGATE P0: {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'}")
sys.exit(0 if not FAILS else 1)
