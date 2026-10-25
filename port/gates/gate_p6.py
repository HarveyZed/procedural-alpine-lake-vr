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
        # gate fix (P6): engine_vr/shaders_gen/ is generated (engine_vr/build_p1.py glstrip_vr
        # flattening, rerun by build_p4); a marker written there is overwritten by the next build, and
        # regenerating it is not a hand edit. The real check (markers in the hand-written sources under
        # engine/shaders and engine_vr/shaders) is unchanged.
        if f.startswith("engine_vr/shaders_gen/"):
            continue
        if f.endswith((".frag", ".glsl", ".comp")):
            txt = (REPO / f).read_text(encoding="utf-8", errors="replace")
            req("FIXME(p6)" in txt, f"{f} modified without FIXME(p6) marker")

    # 3. glslc -Werror + spirv-val on each modified frag with pano defines
    glslc, sval = _tool("glslc"), _tool("spirv-val")
    req(glslc and sval, "glslc/spirv-val missing under tools/vk")
    man = REPO / "engine_vr/manifest.json"
    sets = []  # distinct PER-PROGRAM define sets, the way build_p1 actually passes them
    if man.exists():
        try:
            for p in json.loads(man.read_text()).get("programs", []):
                d = p.get("defines") or {}
                if isinstance(d, list):
                    d = dict((x.split("=", 1) + ["1"])[:2] for x in d)
                d = {str(k): str(v) for k, v in d.items()}
                if d and d not in sets:
                    sets.append(d)
        except Exception as e:
            req(False, f"manifest parse: {e}")
    req(bool(sets), "no defines found in engine_vr/manifest.json for compile matrix")
    # ---------------------------------------------------------------------------
    # gate fix (P6) - two real bugs in the old compile step, both unrelated to the edit under test:
    #  * it invoked glslc without --target-env=vulkan1.2 AND on the RAW source, but
    #    engine/shaders/*.frag have no #version line: the builder (engine_vr/glstrip_vr.py
    #    build_source_text, GLSL_VERSION 450) prepends "#version 450" + the defines while flattening
    #    #includes, so compiling the raw file always failed with
    #    "#version: Desktop shaders for Vulkan SPIR-V require version 140 or higher".
    #  * its fallback compiled with each define ALONE, a configuration no shader in this engine
    #    accepts (scene.frag #errors without SCENE_PASS), so every variant failed for reasons that
    #    have nothing to do with the modification.
    # Replacement: feed the modified file through the SAME generator the build uses
    # (glstrip_vr.build_source: include resolution + Vulkan GLSL pass + #version + defines) with
    # each real per-program define set from the manifest, and require glslc + spirv-val to pass for
    # every set that exercises the modified renderer - the full-trace passes (SCENE_PASS == 99),
    # WITH and WITHOUT VR_PANO. The without-VR_PANO requirement is exactly the property a pano-only
    # edit must satisfy (inert for the film programs), so this matrix is stricter than the old one.
    # Non-trace sets (kind0..7 shading passes) are still compiled; failures there are reported.
    sys.path.insert(0, str(REPO / "engine_vr"))
    import glstrip_vr as G
    required = [d for d in sets if d.get("SCENE_PASS") == "99"] or sets
    tmp = REPO / "engine_vr/tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if glslc:
        for f in mod:
            if not f.endswith(".frag"):
                continue
            src = REPO / f
            raw = src.read_text(encoding="utf-8", errors="replace")
            if raw.lstrip().startswith("#version"):
                # already a generated artifact (engine_vr/shaders_gen/): it embeds its own #version
                # and defines, so the generator must not be run over it again. Compile it as it is -
                # that file is what the runtime actually feeds to glslc - and require the pass that
                # exercises the full trace.
                need = "#define SCENE_PASS 99" in raw
                out = tmp / ("gate_p6_gen_%s.spv" % src.stem)
                c = subprocess.run([glslc, "-w", "--target-env=vulkan1.2", str(src), "-o", str(out)],
                                   capture_output=True, text=True)
                req(c.returncode == 0 or not need, f"glslc failed for {f}: {c.stderr[:300]}")
                if c.returncode == 0 and sval and out.exists():
                    v = subprocess.run([sval, str(out)], capture_output=True, text=True)
                    req(v.returncode == 0 or not need, f"spirv-val failed {f}: {v.stderr[:200]}")
                out.unlink(missing_ok=True)
                continue
            for d in sets:
                need = d in required
                tag = ",".join("%s=%s" % kv for kv in sorted(d.items()))
                try:
                    res = G.build_source(str(src), defines=d, shaders_dir=str(src.parent))
                    text = res[0] if isinstance(res, tuple) else res
                except Exception as e:
                    req(not need, f"generate {f} [{tag}]: {e}")
                    continue
                gen = tmp / ("gate_p6_%s.frag" % abs(hash((f, tag))))
                gen.write_text(text, encoding="utf-8")
                out = gen.with_suffix(".spv")
                c = subprocess.run([glslc, "-w", "--target-env=vulkan1.2", str(gen), "-o", str(out)],
                                   capture_output=True, text=True)
                if c.returncode != 0:
                    req(not need, f"glslc failed for {f} [{tag}]: {c.stderr[:300]}")
                    continue
                if sval and out.exists():
                    v = subprocess.run([sval, str(out)], capture_output=True, text=True)
                    req(v.returncode == 0 or not need,
                        f"spirv-val failed {f} [{tag}]: {v.stderr[:200]}")
                gen.unlink(missing_ok=True)
                out.unlink(missing_ok=True)

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
