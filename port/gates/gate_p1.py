#!/usr/bin/env python3
"""Gate P1 — every real shader variant compiles glslc->spirv-val; manifest<->SPIR-V reflection parity.
Run: .venv/bin/python port/gates/gate_p1.py"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "tools/vk/bin"
FAILS = []

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)

def sh(args, timeout=300):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[:600]

mf = ROOT / "engine_vr/manifest.json"
check("manifest.json exists", mf.exists())
if mf.exists():
    man = json.loads(mf.read_text())
    progs = man["programs"] if isinstance(man, dict) else man
    check("manifest has >= 12 programs (9 split + vs + bake/erosion/post floor)", len(progs) >= 12, str(len(progs)))
    seen_names = set()
    for pr in progs:
        name = pr["name"]
        src = ROOT / pr["source"]
        ok = src.exists() and name not in seen_names
        seen_names.add(name)
        if not ok:
            check(f"program {name}: source present+unique", False, str(src))
            continue
        defines = pr.get("defines", {}) or {}
        cmd = [str(BIN / "glslc"), "--target-env=vulkan1.2"]
        for k, v in defines.items():
            cmd.append(f"-D{k}={v}" if v not in (None, "", True) else f"-D{k}")
        stage = {"frag": ".frag", "vert": ".vert", "comp": ".comp"}.get(pr.get("stage", ""), "")
        out = ROOT / f"spike/.gate_p1_{re.sub(r'\W', '_', name)}.spv"
        cmd += [str(src), "-o", str(out)]
        rc, o = sh(cmd)
        if rc != 0:
            check(f"glslc {name}", False, o.strip()[-300:]); out.unlink(missing_ok=True); continue
        rc, o = sh([str(BIN / "spirv-val"), str(out)])
        if rc != 0:
            check(f"spirv-val {name}", False, o.strip()[-300:]); out.unlink(missing_ok=True); continue
        rc, dis = sh([str(BIN / "spirv-dis"), str(out)])
        # robust scan: pair DescriptorSet/Binding decorations per resource
        sets = re.findall(r"Decorate %(\S+) DescriptorSet (\d+)", dis)
        binds = re.findall(r"Decorate %(\S+) Binding (\d+)", dis)
        by_res = {}
        for res, s in sets: by_res.setdefault(res, [None, None])[0] = int(s)
        for res, b in binds: by_res.setdefault(res, [None, None])[1] = int(b)
        actual = {(v[0], v[1]) for v in by_res.values() if v[0] is not None and v[1] is not None}
        declared = {(int(d["set"]), int(d["binding"])) for d in pr.get("bindings", [])}
        check(f"compile+val+reflection parity {name}", actual == declared and len(actual) > 0,
              f"actual={sorted(actual)} declared={sorted(declared)}" if actual != declared else "")
        out.unlink(missing_ok=True)
    check("glslc/spirv-val/spirv-dis all used on >=12 programs", len(seen_names) >= 12, str(len(seen_names)))
print(f"\nGATE P1: {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'}")
sys.exit(0 if not FAILS else 1)
