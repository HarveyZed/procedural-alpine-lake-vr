#!/usr/bin/env python3
"""Gate P4 — static + pure-CPU numeric validation of the pano renderer. Run: .venv/bin/python port/gates/gate_p4.py"""
import ast, importlib, json, math, py_compile, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
VR = ROOT / "engine_vr"
sys.path.insert(0, str(ROOT))
FAILS = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" - " + detail) if detail else ""))
    if not ok: FAILS.append(name)
def sh(args, t=180):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=t)
    return p.returncode, (p.stdout + p.stderr)[:900]
for f in [ROOT / "vr.py", VR / "pano.py"]:
    check(f.name + " exists", f.exists())
for p in [ROOT / "vr.py", *sorted(VR.glob("*.py"))]:
    try:
        py_compile.compile(str(p), doraise=True); ok, d = True, ""
    except Exception as ex:
        ok, d = False, str(ex)[:180]
    check("py_compile " + p.name, ok, d)
rc, out = sh([".venv/bin/python", "-m", "ruff", "check", "--select", "E,F", "--quiet", "engine_vr/", "vr.py"])
check("ruff E,F", rc == 0, out.strip()[-300:])
def imports_of(path):
    s = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import): s |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module: s.add(n.module.split(".")[0])
    return s
check("no moderngl/wgpu imports (vr.py+engine_vr)", not any({"moderngl", "wgpu"} & (imports_of(p) if p.exists() else set()) for p in [ROOT / "vr.py", *VR.glob("*.py")]))
try:
    pano = importlib.import_module("engine_vr.pano")
    W, H = 360, 180
    d = pano.equirect_directions(W, H, 0.0, 0.0, 0.0)
    check("dirs shape/dtype/finiteness", d.shape == (H, W, 3) and d.dtype == np.float32 and np.isfinite(d).all(), str(d.shape))
    n = np.linalg.norm(d, axis=-1)
    check("dirs unit norm", np.abs(n - 1.0).max() < 2e-6, "%.2e" % np.abs(n - 1).max())
    lon = np.arctan2(d[..., 0], d[..., 2]); lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    check("lat monotone top-down +-pi/2", lat[0].mean() > 1.56 and lat[-1].mean() < -1.56 and bool(np.all(np.diff(lat[:, W // 2]) < 0)))
    check("lon wraps [-pi,pi) left->right", lon[0, 0] < -3.13 and lon[0, -1] > 3.12 and bool(np.all(np.diff(lon[0, 1:W // 4]) > 0)))
    fp = pano.texel_footprint(8192, 4096)
    dLon = 2 * math.pi / 8192; dLat = math.pi / 4096
    phi = np.linspace(math.pi / 2, -math.pi / 2, 4096, endpoint=False) + dLat / 2
    expect = np.maximum(dLat, dLon * np.cos(phi))
    check("footprint == max(dLat, dLon*cos lat)", fp.shape == (4096,) and np.abs(fp - expect).max() < 1e-9, "%.2e" % np.abs(fp - expect).max())
    eye = pano.equirect_directions(16, 8, 0.0, 0.0, 0.0)
    if hasattr(pano, "stereo_pair"):
        cam = np.array([10.0, 5.0, -2.0], "float32"); right = np.array([1.0, 0.0, 0.0], "float32")
        ol, orr, dl = pano.stereo_pair(cam, eye, right)
        ol = np.asarray(ol, "float32"); orr = np.asarray(orr, "float32")
        sep = float(np.linalg.norm(orr - ol))
        mid_ok = bool(np.all(np.abs((ol + orr) / 2 - cam) < 1e-6))
        check("stereo: +-0.032 m along right, parallel (shared dirs), midpoint", abs(sep - 0.064) < 1e-6 and np.allclose(dl, eye) and mid_ok, "sep %.4f midpoint_ok %s" % (sep, mid_ok))
    else:
        check("stereo: +-0.032 m along right, parallel (shared dirs)", False, "no stereo_pair")
except Exception as ex:
    check("pano pure-CPU numerics", False, "%s: %s" % (type(ex).__name__, str(ex)[:200]))
rc, out = sh([".venv/bin/python", "vr.py", "--dry-run", "--viewpoint", "0", "--res", "256x128"])
okdry = rc == 0; plan = None
if okdry:
    try:
        plan = json.loads(out[out.index("{"): out.rindex("}") + 1])
    except Exception as ex:
        okdry, out = False, "no JSON plan: " + str(ex)[:120] + " " + out[:200]
check("vr.py --dry-run exit0 + JSON plan", okdry, out[:200])
rc, out = sh([".venv/bin/python", "vr.py", "--dry-run", "--viewpoint", "3", "--stereo", "sbs", "--res", "512x256"])
check("vr.py --dry-run sbs exit0", rc == 0, out[:150])
rc, out = sh(["git", "status", "--porcelain"])
dirty = [l for l in out.splitlines() if l[3:].strip().split("/")[0] in ("engine", "audio", "docs") or l[3:].strip() in ("film.py", "still.py")]
check("read-only trees untouched", not dirty, str(dirty[:2]))
print("")
print("GATE P4: " + ("PASS" if not FAILS else "FAIL (" + str(len(FAILS)) + ")"))
sys.exit(0 if not FAILS else 1)
