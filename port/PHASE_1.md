# PHASE 1 — Make every REAL shader compile to valid SPIR-V (zero GPU)

You are the P1 coder. Read port/RUNBOOK.md (§1 constraints, §4 architecture, §7 D6/D7), then
port/reports/phase0.md §risks (binding-injection hint: only 14/22 samplers explicitly bound
today), then skim spike/p0_a/run.py (hardware-proven pattern for later phases — not your work).

## Mission

Produce a static, fully machine-checked shader pipeline: **every program the real renderer and
bake actually compile today** (9 scene.frag split-pass variants + vertex + post shaders +
bake_*.frag + erosion_*.comp + any others engine/ references) must compile
`glslc → SPIR-V` and pass `spirv-val`, with GLSL bumped 430→450 and **all descriptors given
explicit `layout(set=0, binding=N)`** via a ported glstrip, emitting a machine-readable
**manifest.json** that Phase 2's host layer will consume to build descriptor set layouts.
No GPU execution whatsoever this phase (not even micro) — glslc/spirv-val/spirv-dis/spirv-cross
in `tools/vk/bin/` are the only executables you run.

## Tasks

T1 **Inventory** (read-only): engine/gl.py (#include loader), engine/glstrip.py (reduce()
semantics), engine/render.py (pass list + defines per split-pass variant + VS), engine/post.py
(its shader programs + defines), engine/bake.py (bake_* programs + defines), engine/erosion.py
(erosion_*.comp + defines/dispatch metadata). Record exact variant→define tables as static data
in your own module — **do NOT import** engine modules at runtime (render.py imports gl.py →
moderngl). Reading sources with the file tools is the way.

T2 **engine_vr/glstrip_vr.py**: port glstrip (include resolution, #if resolution per defines,
dead-function stripping) keeping its semantics (same resolution results; you may import
engine.glstrip if that avoids copying — check it has no moderngl dependency). Add the
**binding injection pass**: for every top-level `uniform` texture/sampler/combined-image/
uniform buffer/storage buffer/storage image lacking an explicit binding, inject
`layout(set=0, binding=N)` deterministically: namespaces (separate counters) for
{sampler-ish combined}, {uniform buffers}, {storage buffers}, {storage images}, counter order =
first appearance in preprocessed source, start at 0 each. Explicit bindings already present:
keep, only verify uniqueness (collision → hard error with message).

T3 **430→450 bump** in copies under engine_vr/shaders/ (originals read-only): change only
`#version` lines first; then fix ONLY what glslc 450 actually rejects. Zero numerical or
visual-semantics changes; every non-version edit goes in the report's required §450 diff log.

T4 **engine_vr/manifest.json**: one entry per generated program: name, injected source path
(under engine_vr/shaders_gen/ or similar), stage, defines (map), and bindings list
[{name, kind, set, binding, glsl_type}]. Gate contract: for each program gate_p1.py compiles it
(glslc --target-env=vulkan1.2, includes resolvable), runs spirv-val, runs spirv-dis and checks
the set of (set,binding) decorated pairs EQUALS exactly the manifest's declared pairs
(no undeclared resources, no phantoms).

T5 Run `port/gates/gate_p1.py` (GC-owned; do not edit — report suspected gate bugs instead) to
PASS, then write port/reports/phase1.md per RUNBOOK §5 + §450 diff log + §glslc strictness list
(categories of errors that appeared purely because of 450 strictness — this table informs P2).

## Boundaries

Writes: engine_vr/shaders/**, engine_vr/shaders_gen/** (or your generated-dir choice recorded in
manifest), engine_vr/glstrip_vr.py, engine_vr/manifest.json, port/reports/phase1.md. No git.
No GPU/device contexts. No imports of engine modules that pull moderngl. Proxy env only if you
need network (you shouldn't). If one program cannot be made to compile after honest iteration,
leave it failing, document precisely, and report — partial PASS with rigorous notes beats a hack.
