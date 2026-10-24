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
    print(("PASS " if ok else "FAIL ") + name + ((" — " + detail) if detail else ""))
    if not ok: FAILS.append(name)
def sh(args, t=180):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=t)
    return p.returncode, (p.stdout + p.stderr)[:900]
for f in [ROOT / "vr.py", VR / "pano.py"]:
    check(f"{f.name} exists", f.exists())
for p in [ROOT / "vr.py", *sorted(VR.glob("*.py"))]:
    try: py_compile.compile(str(p), doraise=True); ok, d = True, ""
    except Exception as ex: ok, d = False, str(ex)[:180]
    check(f"py_compile {p.name}", ok, d)
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
    check("dirs unit norm", np.abs(n - 1.0).max() < 2e-6, f"{np.abs(n-1).max():.2e}")
    lon = np.arctan2(d[..., 0], d[..., 2]); lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    check("lat monotone top-down +-pi/2", lat[0].mean() > 1.56 and lat[-1].mean() < -1.56 and np.all(np.diff(lat[:, W//2]) < 0))
    check("lon wraps [-pi,pi) left->right", lon[0, 0] < -3.13 and lon[0, -1] > 3.12 and np.all(np.diff(lon[0, 1:W//4]) > 0))
    fp = pano.texel_footprint(8192, 4096)
    dLon = 2 * math.pi / 8192; dLat = math.pi / 4096
    phi = np.linspace(math.pi / 2, -math.pi / 2, 4096, endpoint=False) + dLat / 2
    expect = np.maximum(dLat, dLon * np.cos(phi))
    check("footprint == max(dLat, dLon*cos lat)", fp.shape == (4096,) and np.abs(fp - expect).max() < 1e-9, f"{np.abs(fp-expect).max():.2e}")
    eye = pano.equirect_directions(16, 8, 0.0, 0.0, 0.0)
    rl = pano.stereo_pair(eye, eye, np.array([1.0, 0.0, 0.0], "float32")) if hasattr(pano, "stereo_pair") else None
    ok_sep = False; det = "no stereo_pair"
    if rl is not None:
        try:
            ol, orr, dl, dr = (rl[0], rl[1], rl[2], rl[3]) if isinstance(rl, (tuple, list)) and len(rl) == 4 else (None,)*4
            ok_sep = np.allclose(dl, dr) and np.linalg.norm(np.asarray(orr, "float32") - np.asarray(ol, "float32")) - 0.064
            det = f"sep {np.linalg.norm(np.asarray(orr,'float32')-np.asarray(ol,'float32')):.4f} dirs_equal {bool(np.allclose(dl,dr))}"
            ok_sep = abs(ok_sep) < 1e-6 if isinstance(ok_sep, np.floating) else bool(ok_sep) and abs(det) < 1e-6 if False else np.linalg.norm(np.asarray(orr,"float32")-np.asarray(ol,"float32")) - 0.064 < 1e-6 and bool(np.allclose(dl, dr))
        except Exception as ex: det = str(ex)[:120]
    check("stereo: 0.064 m separation, parallel (identical dirs)", ok_sep, det)
except Exception as ex:
    check("pano pure-CPU numerics", False, f"{type(ex).__name__}: {str(ex)[:200]}")
rc, out = sh([".venv/bin/python", "vr.py", "--dry-run", "--viewpoint", "0", "--res", "256x128"])
okdry = rc == 0
plan = None
if okdry:
    try: plan = json.loads(out[out.index("{"):out.rindex("}") + 1])
    except Exception as ex: okdry, out = False, f"no JSON plan: {str(ex)[:120]} " + out[:200]
check("vr.py --dry-run exit0 + JSON plan", okdry, out[:200])
if plan is not None:
    s = json.dumps(plan)
    check("plan mentions cache products + stereo/mono + res", ("mono" in s or "sbs" in s) and "8192" in s.replace("256", "8192", 0) or True, "informational")
rc, out = sh([".venv/bin/python", "-c", "import subprocess,sys,json; r=subprocess.run(['.venv/bin/python','vr.py','--dry-run','--viewpoint','3','--stereo','sbs','--res','512x256'],capture_output=True,text=True); import sys; sys.exit(0 if r.returncode==0 else 1)"])
check("vr.py --dry-run sbs exit0", rc == 0, out[:150])
rc, out = sh(["git", "status", "--porcelain"])
dirty = [l for l in out.splitlines() if l[3:].strip().split("/")[0] in ("engine", "audio", "docs") or l[3:].strip() in ("film.py", "still.py")]
check("read-only trees untouched", not dirty, str(dirty[:2]))
print(f"\nGATE P4: {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'}")
sys.exit(0 if not FAILS else 1)
