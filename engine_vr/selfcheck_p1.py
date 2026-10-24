#!/usr/bin/env python3
"""Port of gate_p1.py's checks WITHOUT its 600-byte output truncation (see port/reports/phase1.md
"gate bug"): same commands, same regexes, same parity rule, plus two checks the gate cannot make —
that (set, binding) is unique inside a program and that every declared pair names exactly one record.
Run: .venv/bin/python engine_vr/selfcheck_p1.py     (read-only; writes only spike/.selfcheck_*.spv)
"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "tools/vk/bin"
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" — " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def sh(args, timeout=300):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr            # <- the gate truncates this to [:600]


man = json.loads((ROOT / "engine_vr/manifest.json").read_text())
progs = man["programs"]
check("manifest has >= 12 programs", len(progs) >= 12, str(len(progs)))
seen, total = set(), 0
for pr in progs:
    name = pr["name"]
    src, defines = ROOT / pr["source"], pr.get("defines", {}) or {}
    if not src.exists() or name in seen:
        check(f"program {name}: source present+unique", False, str(src))
        continue
    seen.add(name)
    cmd = [str(BIN / "glslc"), "--target-env=vulkan1.2"]
    for k, v in defines.items():
        cmd.append(f"-D{k}={v}" if v not in (None, "", True) else f"-D{k}")
    out = ROOT / f"spike/.selfcheck_{re.sub(r'\W', '_', name)}.spv"
    cmd += [str(src), "-o", str(out)]
    rc, o = sh(cmd)
    if rc != 0:
        check(f"glslc {name}", False, o.strip()[-300:]); continue
    rc, o = sh([str(BIN / "spirv-val"), str(out)])
    if rc != 0:
        check(f"spirv-val {name}", False, o.strip()[-300:]); continue
    rc, dis = sh([str(BIN / "spirv-dis"), str(out)])
    sets = re.findall(r"Decorate %(\S+) DescriptorSet (\d+)", dis)
    binds = re.findall(r"Decorate %(\S+) Binding (\d+)", dis)
    by_res = {}
    for res, s in sets:
        by_res.setdefault(res, [None, None])[0] = int(s)
    for res, b in binds:
        by_res.setdefault(res, [None, None])[1] = int(b)
    actual = {(v[0], v[1]) for v in by_res.values() if v[0] is not None and v[1] is not None}
    declared = [(int(d["set"]), int(d["binding"])) for d in pr.get("bindings", [])]
    dup = sorted({p for p in declared if declared.count(p) > 1})
    check(f"unique (set,binding) {name}", not dup, str(dup))
    check(f"compile+val+reflection parity {name}", actual == set(declared) and len(actual) > 0,
          "" if actual == set(declared) else f"actual={sorted(actual)} declared={sorted(set(declared))}")
    total += len(actual)
    out.unlink(missing_ok=True)
check("every program compiled, validated and reflected", len(seen) == len(progs), f"{len(seen)}/{len(progs)}")
print(f"\nSELFCHECK: {'PASS' if not FAILS else 'FAIL (' + str(len(FAILS)) + ')'} — "
      f"{len(progs)} programs, {total} reflected (set,binding) pairs, 0 GPU calls")
sys.exit(0 if not FAILS else 1)
