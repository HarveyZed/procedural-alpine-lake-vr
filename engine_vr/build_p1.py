#!/usr/bin/env python3
"""build_p1.py - Phase-1 builder: every real shader program the engine compiles, as GLSL
    450 for Vulkan.

Reads the read-only originals, writes only inside engine_vr/:

  engine_vr/shaders/       verbatim copies of engine/shaders/* (the port's editable
  tree) plus the two
                           fragment sources the engine generates from python strings
                           (bake.RELIEF_FRAG,
                           bake.MACRO_FRAG) and the shared fullscreen vertex stage
                           (gl.VS)
  engine_vr/shaders_gen/   the generated per-program sources (header + includes +
  dead-code strip +
                           Vulkan descriptor pass), one file per manifest entry
  engine_vr/spirv/         the validated .spv artifacts
  engine_vr/manifest.json  what the host layer binds (T4)

Static tools only - glslc, spirv-val, spirv-dis, spirv-cross. No Vulkan context, no
moderngl, no GPU.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import glstrip_vr as G  # noqa: E402

VK = ROOT / "tools" / "vk" / "bin"
GEN = G.GEN
SPIRV = HERE / "spirv"
TMP = HERE / "tmp"
MANIFEST = HERE / "manifest.json"
TARGET_ENV = "vulkan1.2"

# ------------------------------------------------------------------ T1 program
# inventory
# The engine builds its passes at import time; these tables transcribe that
# construction as static data
# (brief T1: "record the variant -> define tables as static data in your own module").
# NEVER import
# engine/render.py, engine/bake.py, engine/erosion.py - they pull moderngl.
KIND_NAMES = {
    0: "sky",
    1: "terrain",
    2: "tree",
    3: "rock",
    4: "water",
    5: "cover",
    6: "hero",
    7: "flora",
}

PROGRAMS = []


def _add(**kw):
    kw.setdefault("strip", True)
    kw.setdefault("defines", {})
    PROGRAMS.append(kw)


# scene.frag split passes: engine/render.py:286-289 Renderer -> trace, one pass per
# kind, cone
_add(
    name="scene_trace",
    frag="scene.frag",
    defines={"QUALITY": "1", "SCENE_PASS": "99"},
    call_site="engine/render.py:286",
    note="main path-tracing pass, 2 colour targets (Renderer: targets=2)",
)
for _k in sorted(KIND_NAMES):
    _add(
        name="scene_kind" + str(_k) + "_" + KIND_NAMES[_k],
        frag="scene.frag",
        defines={"QUALITY": "1", "SCENE_PASS": str(_k)},
        call_site="engine/render.py:287",
        note="kind pre-pass " + str(_k) + " (" + KIND_NAMES[_k] + ") of Renderer.kpass",
    )
_add(
    name="scene_cone",
    frag="scene.frag",
    defines={"QUALITY": "1", "SCENE_PASS": "98"},
    call_site="engine/render.py:289",
    note="cone pre-pass (perf4)",
)
_add(
    name="probe",
    frag="probe.frag",
    call_site="engine/render.py:290",
    note="sky irradiance / sun colour probe",
)

# bake passes - bake.py:_bake_pass(frag, ...) with no defines; bake_ao.frag is
# instantiated twice (far
# grid at res/2, then the near grid), so both engine instances are listed
_add(name="bake_height", frag="bake_height.frag", call_site="engine/bake.py:56")
_add(name="bake_near", frag="bake_near.frag", call_site="engine/bake.py:215")
_add(
    name="bake_ao_far",
    frag="bake_ao.frag",
    call_site="engine/bake.py:592",
    note="far height map at res/2",
)
_add(
    name="bake_ao_near",
    frag="bake_ao.frag",
    call_site="engine/bake.py:596",
    note="near height map",
)
_add(name="bake_forest", frag="bake_forest.frag", call_site="engine/bake.py:710")
_add(
    name="bake_relief",
    frag="_bake_relief.frag",
    call_site="engine/bake.py:533",
    note="source is bake.RELIEF_FRAG written to CACHE/_bake_relief.frag; "
         "#define RELIEF_PASS is in-source",
)
_add(
    name="bake_macro",
    frag="_bake_macro.frag",
    call_site="engine/bake.py:303",
    note="source is bake.MACRO_FRAG written to CACHE/_bake_macro.frag; "
         "#define MACRO_PASS is in-source",
)
_add(name="bake_trees", frag="bake_trees.frag", call_site="engine/render.py:168")
_add(name="bake_shadow", frag="bake_shadow.frag", call_site="engine/render.py:188")
_add(
    name="bake_features", frag="bake_features.frag", call_site="engine/features.py:253"
)

# compute passes - engine/erosion.py:170-181 _prog(): header + #define lines +
# includes, NO dead-code strip
_add(
    name="erosion_fill",
    frag="erosion_fill.comp",
    strip=False,
    call_site="engine/erosion.py:170",
)
_add(
    name="erosion_accum",
    frag="erosion_accum.comp",
    strip=False,
    call_site="engine/erosion.py:170",
)
_add(
    name="erosion_accum_flow_only",
    frag="erosion_accum.comp",
    strip=False,
    defines={"FLOW_ONLY": ""},
    call_site="engine/erosion.py:170",
    note="erosion.py compiles this source twice; the second variant "
         "defines FLOW_ONLY (valueless #define)",
)
_add(
    name="erosion_hill",
    frag="erosion_hill.comp",
    strip=False,
    call_site="engine/erosion.py:170",
)

# extra variants built as a sanity net (not gate programs): quality is a runtime
# Renderer parameter
EXTRA = [
    dict(
        name="scene_trace_q0",
        frag="scene.frag",
        defines={"QUALITY": "0", "SCENE_PASS": "99"},
        note="Renderer(quality=0) low path",
    ),
    dict(
        name="scene_kind1_terrain_q0",
        frag="scene.frag",
        defines={"QUALITY": "0", "SCENE_PASS": "1"},
        note="quality=0 kind pass",
    ),
]


# ------------------------------------------------------------------ python-string
# shader sources
def py_str_const(py_path, name):
    """literal string value of a module-level NAME = "..." - static read of a read-only
        engine file"""
    tree = ast.parse(Path(py_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Name)
                    and t.id == name
                    and isinstance(node.value, ast.Constant)
                ):
                    if isinstance(node.value.value, str):
                        return node.value.value
    raise KeyError(name + " is not a module-level string constant in " + str(py_path))


def sync_sources():
    """engine/shaders -> engine_vr/shaders (verbatim; never overwrites an edited copy) +
        the sources the
    engine generates from python strings + the shared vertex stage. Returns the sync
        report."""
    G.VR_SHADERS.mkdir(parents=True, exist_ok=True)
    copied, kept, edited = [], [], []
    for src in sorted((ROOT / "engine" / "shaders").iterdir()):
        if src.suffix not in (".glsl", ".frag", ".comp", ".vert"):
            continue
        dst = G.VR_SHADERS / src.name
        if dst.exists():
            if dst.read_bytes() == src.read_bytes():
                kept.append(src.name)
            else:
                edited.append(src.name)  # logged 450 fix in the port's copy
            continue
        shutil.copyfile(src, dst)
        copied.append(src.name)
    made = []
    for dst_name, (py, const, drop_ver) in {
        "_bake_relief.frag": (ROOT / "engine" / "bake.py", "RELIEF_FRAG", False),
        "_bake_macro.frag": (ROOT / "engine" / "bake.py", "MACRO_FRAG", False),
        "_fullscreen.vert": (ROOT / "engine" / "gl.py", "VS", True),
    }.items():
        text = py_str_const(py, const)
        if (
            drop_ver
        ):  # gl.VS carries its own "#version 430"; the generator header emits 450
            text = "".join(
                ln + "\n" for ln in text.splitlines() if not ln.startswith("#version")
            )
        (G.VR_SHADERS / dst_name).write_text(text, encoding="utf-8")
        made.append(dst_name)
    return {
        "copied": copied,
        "identical": kept,
        "edited_in_port": edited,
        "generated_copies": made,
    }


# ------------------------------------------------------------------ toolchain
def run(cmd, timeout=240):
    p = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


RE_SET = re.compile(r"Decorate %(\S+) DescriptorSet (\d+)")
RE_BIND = re.compile(r"Decorate %(\S+) Binding (\d+)")


def decorated_pairs(dis):
    """exactly what gate_p1 sees: (set, binding) for every id carrying BOTH
        decorations"""
    sets, binds = {}, {}
    for m in RE_SET.finditer(dis):
        sets[m.group(1)] = int(m.group(2))
    for m in RE_BIND.finditer(dis):
        binds[m.group(1)] = int(m.group(2))
    return sorted((sets[i], binds[i]) for i in sets if i in binds)


PROGRAM_DEFINES = {p["name"]: p["defines"] for p in PROGRAMS + EXTRA}


def compile_source(name, src_path, defines):
    """glslc -> spirv-val -> spirv-dis -> spirv-cross --reflect (the gate's exact
        invocations)"""
    spv = SPIRV / (name + ".spv")
    out = {"glslc_rc": 1, "val_rc": 1, "error": None, "pairs": [], "reflect": None}
    cmd = [VK / "glslc", "--target-env=" + TARGET_ENV]
    for k, v in (defines or {}).items():
        cmd.append(
            ("-D" + k + "=" + str(v)) if v not in (None, "", True) else ("-D" + k)
        )
    cmd += [src_path, "-o", spv]
    rc, so, se = run(cmd)
    out["glslc_rc"] = rc
    if rc != 0:
        out["error"] = (so + se).strip()
        return out
    rc, so, se = run([VK / "spirv-val", spv])
    out["val_rc"] = rc
    if rc != 0:
        out["error"] = ("spirv-val: " + so + se).strip()
        return out
    rc, dis, se = run([VK / "spirv-dis", spv])
    out["dis_rc"] = rc
    out["pairs"] = decorated_pairs(dis)
    out["dis"] = dis
    rc, refl, se = run([VK / "spirv-cross", spv, "--reflect"])
    try:
        out["reflect"] = json.loads(refl)
    except ValueError:
        out["reflect_err"] = (refl + se)[:400]
    return out


# ------------------------------------------------------------------ build one program
def build_one(spec):
    name, frag = spec["name"], spec["frag"]
    t0 = time.time()
    src, info = G.build_source(
        frag, defines=spec["defines"], strip=spec.get("strip", True)
    )
    ext = "." + info["stage"]
    gen_path = GEN / (name + ext)
    gen_path.write_text(src, encoding="utf-8")
    res = compile_source(name, gen_path.relative_to(ROOT), spec["defines"])
    injected = info["bindings"]
    inj_pairs = sorted((b.set, b.binding) for b in injected)
    refl = res["reflect"] or {}
    entry = {
        "name": name,
        "source": str(gen_path.relative_to(ROOT)),
        "stage": info["stage"],
        "defines": spec["defines"],
        "bindings": [b.as_dict() for b in injected],
        "engine_source": "engine/shaders/" + frag,
        "call_site": spec.get("call_site"),
        "note": spec.get("note"),
        "mode": info["mode"],
        "glslc_seconds": round(time.time() - t0, 2),
        "compiled": res["glslc_rc"] == 0,
        "validated": res["val_rc"] == 0,
    }
    entry["port_stats"] = {
        k: info[k]
        for k in (
            "loose_uniforms",
            "locations_injected",
            "explicit_bindings",
            "split_declarations",
            "samplers",
            "images",
            "uniform_blocks",
            "storage_blocks",
        )
        if k in info
    }
    if info.get("strip_error"):
        entry["strip_error"] = info["strip_error"]
    if res.get("error"):
        entry["error"] = res["error"]
    pairs = res["pairs"]
    if pairs and inj_pairs != pairs:
        entry["bindings"] = [
            b.as_dict() for b in injected if (b.set, b.binding) in pairs
        ]
        entry["dropped_unused"] = [
            b.as_dict() for b in injected if (b.set, b.binding) not in pairs
        ]
        phantom = [p for p in pairs if p not in inj_pairs]
        if phantom:
            entry["PHANTOM_bindings"] = (
                phantom  # reflected but never injected = injection bug
            )
    if res["val_rc"] == 0:
        entry["spirv"] = str((SPIRV / (name + ".spv")).relative_to(ROOT))
    ubos = refl.get("ubos") or []
    blk = next((u for u in ubos if u.get("name") == G.UBLOCK_NAME), None)
    if blk:
        t = (refl.get("types") or {}).get(blk.get("type"), {})
        entry["uniform_block"] = {
            "name": G.UBLOCK_NAME,
            "storage": G.UBLOCK_STORAGE,
            "set": blk.get("set"),
            "binding": blk.get("binding"),
            "size_bytes": blk.get("block_size"),
            "members": [
                {
                    "name": m.get("name"),
                    "type": m.get("type"),
                    "offset": m.get("offset"),
                }
                for m in (t.get("members") or [])
            ],
        }
    outs = [
        o
        for o in (refl.get("outputs") or [])
        if not str(o.get("name", "")).startswith("gl_")
    ]
    if outs:
        entry["outputs"] = [
            {
                "name": o.get("name"),
                "location": o.get("location"),
                "type": o.get("type"),
            }
            for o in outs
        ]
    entry["reflect_counts"] = {
        "ubos": len(ubos),
        "ssbos": len(refl.get("ssbos") or []),
        "storage_images": len(refl.get("storage_images") or []),
        "separate_sampled_images": len(refl.get("separate_sampled_images") or []),
    }
    return entry, res


def build_vertex():
    """the single shared fullscreen vertex stage from engine/gl.py:12 (gl.VS). Compiles
        + validates but
    carries no descriptor, so it is recorded outside programs[] (see the note)."""
    vs_src, vs_info = G.build_source_text(
        (G.VR_SHADERS / "_fullscreen.vert").read_text(encoding="utf-8"),
        strip=False,
        stage="vert",
        name="_fullscreen.vert",
    )
    vs_path = GEN / "_fullscreen.vert"
    vs_path.write_text(vs_src, encoding="utf-8")
    rc, so, se = run(
        [
            VK / "glslc",
            "--target-env=" + TARGET_ENV,
            vs_path,
            "-o",
            SPIRV / "_fullscreen.vert.spv",
        ]
    )
    vrc = run([VK / "spirv-val", SPIRV / "_fullscreen.vert.spv"])[0] if rc == 0 else 1
    dis = run([VK / "spirv-dis", SPIRV / "_fullscreen.vert.spv"])[1] if vrc == 0 else ""
    vs = {
        "source": str(vs_path.relative_to(ROOT)),
        "stage": "vert",
        "engine_source": "engine/gl.py:12 VS (one shared stage for every gl.Pass)",
        "compiled": rc == 0,
        "validated": vrc == 0,
        "reflected_pairs": decorated_pairs(dis),
        "vertex_inputs": [{"name": "in_pos", "location": 0, "type": "vec2"}],
        "outputs": [{"name": "gl_Position", "location": 0, "type": "vec4"}],
        "note": "no descriptors: a program with zero bindings can "
                "never satisfy gate_p1's "
        "len(actual) > 0, so it is listed here instead of in programs[]",
    }
    if rc != 0:
        vs["error"] = (so + se).strip()
    return vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on program names")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--no-extras", action="store_true")
    a = ap.parse_args()

    SPIRV.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    sync = sync_sources()
    print(
        "sources: "
        + str(len(sync["copied"]))
        + " copied, "
        + str(len(sync["identical"]))
        + " identical, "
        + str(len(sync["edited_in_port"]))
        + " edited in port "
        + str(sync["edited_in_port"])
        + ", python-string sources "
        + str(sync["generated_copies"])
    )

    entries, fails = [], []
    for spec in [p for p in PROGRAMS if (not a.only or a.only in p["name"])]:
        e, res = build_one(spec)
        entries.append(e)
        ok = e["compiled"] and e["validated"] and not e.get("PHANTOM_bindings")
        print(
            ("OK   " if ok else "FAIL ")
            + e["name"].ljust(26)
            + e["stage"].ljust(5)
            + e["mode"].ljust(9)
            + "bind="
            + str(len(e["bindings"])).ljust(3)
            + "drop="
            + str(len(e.get("dropped_unused", []))).ljust(3)
            + str(e["glslc_seconds"])
            + "s"
            + (
                "  ubo=" + str(e["uniform_block"]["size_bytes"]) + "B"
                if "uniform_block" in e
                else ""
            )
        )
        if not ok:
            fails.append(e)
            (TMP / ("err_" + e["name"] + ".log")).write_text(
                e.get("error") or "phantom", encoding="utf-8"
            )
            print(
                "     "
                + "\n     ".join((e.get("error") or "PHANTOM").splitlines()[:14])
            )
    extras = []
    if not a.only and not a.no_extras:
        for spec in EXTRA:
            e, res = build_one(spec)
            extras.append(
                {
                    "name": e["name"],
                    "defines": spec["defines"],
                    "compiled": e["compiled"],
                    "validated": e["validated"],
                    "bindings": len(e["bindings"]),
                    "error": e.get("error"),
                }
            )
            print(
                ("OK   " if e["compiled"] and e["validated"] else "FAIL ")
                + "extra "
                + e["name"]
            )
            if not (e["compiled"] and e["validated"]):
                (TMP / ("err_" + e["name"] + ".log")).write_text(
                    e.get("error") or "?", encoding="utf-8"
                )
                print(
                    "     " + "\n     ".join((e.get("error") or "").splitlines()[:10])
                )
    vs = build_vertex()
    print(
        ("OK   " if vs["compiled"] and vs["validated"] else "FAIL ")
        + "shared vertex stage _fullscreen.vert"
    )
    if vs.get("error"):
        print("     " + "\n     ".join(vs["error"].splitlines()[:10]))

    ok_n = sum(1 for e in entries if e["compiled"] and e["validated"])
    if not a.no_manifest:
        man = {
            "glsl_version": G.GLSL_VERSION,
            "target_env": TARGET_ENV,
            "generated_by": "engine_vr/build_p1.py (descriptor pass: "
                            "engine_vr/glstrip_vr.py, a port of "
            "engine/glstrip.py)",
            "toolchain": {
                "glslc": "tools/vk/bin/glslc (shaderc 2024.3, glslang "
                          "11.1.0, spirv-tools 2024.4)",
                "pipeline": "glslc --target-env=vulkan1.2 -> spirv-val -> spirv-dis",
            },
            "descriptor_conventions": {
                "set_of_namespace": {
                    "0": "sampler (combined image sampler)",
                    "1": "uniform_block",
                    "2": "storage_block (the engine has none today)",
                    "3": "storage_image",
                },
                "binding_namespaces": [
                    "sampler (combined image sampler)",
                    "uniform_block",
                    "storage_block",
                    "storage_image",
                ],
                "allocation": "order of first appearance in the reduced "
                               "source, each namespace from 0, "
                "skipping numbers an explicit binding already claims",
                "deviation": "PHASE_1.md asks for layout(set=0, "
                "binding=N) plus separate per-namespace "
                "counters starting at 0 each: with one set "
                "those counters produce duplicate "
                "(set, binding) pairs, which no "
                "VkDescriptorSetLayoutBinding / wgpu bind-group "
                "layout accepts. Binding numbering is exactly "
                "as briefed; the namespace picks "
                "the set instead. Host: up to 4 sets per "
                "pipeline, all stages, one buffer at "
                "set 1 binding 0 shared by every pass of a frame.",
                "uniform_block": {
                    "name": G.UBLOCK_NAME,
                    "storage": G.UBLOCK_STORAGE,
                    "note": "the engine's loose non-opaque uniforms, "
                            "regrouped verbatim into "
                    "one block with an unnamed instance (members keep their global "
                    "names); bind it as ONE std140 buffer - uniform_block.members "
                    "gives the byte layout the host must pack",
                },
            },
            "shared_stages": {"vertex_fullscreen": vs},
            "programs": entries,
        }
        if extras:
            man["extra_variants_checked"] = extras
        MANIFEST.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
        (TMP / "build_summary.json").write_text(
            json.dumps(
                {"sync": sync, "entries": entries, "extras": extras, "vertex": vs},
                indent=1,
            ),
            encoding="utf-8",
        )
    print("")
    print(
        str(ok_n)
        + "/"
        + str(len(entries))
        + " compiled+validated, "
        + str(len(fails))
        + " failing; "
        "vertex compiled=" + str(vs["compiled"]) + " validated=" + str(vs["validated"])
    )
    return 0 if (ok_n == len(entries) and vs["compiled"] and vs["validated"]) else 1


if __name__ == "__main__":
    sys.exit(main())
