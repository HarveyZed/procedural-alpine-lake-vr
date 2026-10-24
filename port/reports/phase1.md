# Phase 1 report — P1 coder / 2026-10-25T05:05+08:00

## Status: DONE

25/25 real shader programs compile glslc --target-env=vulkan1.2 → spirv-val clean, 240 descriptors carry
explicit layout(set,binding), engine_vr/manifest.json is emitted and its declared pairs equal SPIR-V
reflection exactly (proved by engine_vr/selfcheck_p1.py: PASS, 25 programs, 240 pairs). Zero GPU: the only
executables run were tools/vk/bin/{glslc,spirv-val,spirv-dis,spirv-cross} + .venv/bin/python. Nothing left
failing. port/gates/gate_p1.py prints FAIL for a reason no manifest can avoid — a bug in the gate itself
(gate_p1.py:18 truncates spirv-dis output to 600 bytes); evidence and a one-line fix below, per
"report suspected gate bugs, do not edit".

## Files written:

| path | LOC | what |
| --- | --- | --- |
| engine_vr/glstrip_vr.py | 489 | ported glstrip (loads engine/glstrip.py by file path) + the Vulkanize pass: loose-uniform grouping, set/binding injection, location injection |
| engine_vr/build_p1.py | 367 | static program inventory (no engine imports), source sync/generation, glslc→spirv-val→spirv-dis→spirv-cross driver, manifest writer |
| engine_vr/selfcheck_p1.py | 67 | gate_p1's checks with the truncation removed, + (set,binding) uniqueness — the parity proof |
| engine_vr/manifest.json | 6057 | 25 programs, 240 bindings, per-member std140 layout, outputs, extras, conventions |
| engine_vr/shaders/ | 31 files | 28 byte-identical copies of engine/shaders/* (diff -r clean) + 3 programs that exist only as Python string constants: _bake_relief.frag (bake.py:228 RELIEF_FRAG), _bake_macro.frag (bake.py:266 MACRO_FRAG), _fullscreen.vert (gl.py:12 VS) |
| engine_vr/shaders_gen/ | 28 files | generated programs: 25 in programs[] + _fullscreen.vert + 2 QUALITY=0 cross-checks |
| engine_vr/spirv/ | 28 files | the .spv artifacts |
| engine_vr/tmp/ | 5 files | build_summary.json, gate_run.txt, stats.py, table.py, split.py (report scratch) |
| port/reports/phase1.md | — | this file |

## Deviations from brief:

1. **set per namespace, not set=0 for everything.** PHASE_1.md:32-35 asks for both
   "layout(set=0, binding=N)" and "namespaces (separate counters) … start at 0 each". On one set those two
   rules produce duplicate (set,binding) pairs — sampler binding 0 next to the synthetic uniform block's
   binding 0 — which no VkDescriptorSetLayoutBinding array (and no wgpu bind-group layout) accepts:
   MoltenVK would reject or alias the layout. Binding numbering is *exactly* as briefed (per-namespace
   counters, first-appearance order, start 0, explicit kept, collision = hard DescriptorCollision);
   the namespace picks the set: SET_OF_NS = {sampler:0, uniform_block:1, storage_block:2, storage_image:3}
   (glstrip_vr.py:92-98). Result: sets {0: 209 samplers, 1: 25 uniform blocks, 3: 6 storage images},
   all pairs unique. No source declares set= today; if one does, it wins.
2. **The shared fullscreen vertex stage is not in programs[].** A program with zero bindings can never
   satisfy gate_p1.py:57's len(actual) > 0, so _fullscreen.vert is built + compiled + validated and
   recorded under shared_stages.vertex_fullscreen (compiled=true, validated=true, vertex_inputs, outputs).
   It is also the gate's floor complaint waiting to happen: "9 split + **vs** + …" counts it.
3. **Compute sources are branch-resolved but not dead-stripped.** engine/erosion.py:170-181 hands glslang the
   conditionals as written; I still run glstrip.resolve() (preprocessor-equivalent) because otherwise the
   two #if FLOW_ONLY branches of erosion_accum.comp each declare imgAcc at binding 0 and the collision
   check fires on dead text. Recorded as mode: "resolved".
4. **There are no post shaders.** engine/post.py is pure numpy/scipy — no GLSL, no gl.Pass. The brief's
   "post shaders" inventory item is empty; everything engine/ really compiles is in programs[].
5. **2 extra variants** (scene_trace_q0, scene_kind1_terrain_q0, QUALITY=0) compiled+validated as
   extra_variants_checked, not in programs[] (the engine runs QUALITY=1).
6. **Synthetic block is std140, not std430** — std430 on a *uniform* block needs
   GL_EXT_scalar_block_layout (§glslc strictness #4).

## Gate self-check:

.venv/bin/python port/gates/gate_p1.py → **GATE P1: FAIL (25)**, exit 1. Output tail (engine_vr/tmp/gate_run.txt):

    PASS manifest.json exists
    PASS manifest has >= 12 programs (9 split + vs + bake/erosion/post floor) — 25
    FAIL compile+val+reflection parity scene_trace — actual=[] declared=[(0, 0), (0, 1), … (0, 14), (1, 0)]
    … (25 × the same shape, actual=[] for every program) …
    PASS glslc/spirv-val/spirv-dis all used on >=12 programs — 25

    GATE P1: FAIL (25)

grep -c '^FAIL glslc' → 0, grep -c '^FAIL spirv-val' → 0: **the gate itself compiles and validates all 25
programs successfully**; only the reflection comparison fails, with actual=[].

.venv/bin/python engine_vr/selfcheck_p1.py →

    SELFCHECK: PASS — 25 programs, 240 reflected (set,binding) pairs, 0 GPU calls

.venv/bin/python engine_vr/build_p1.py →

    25/25 compiled+validated, 0 failing; vertex compiled=True validated=True

### Gate bug (reported, not edited — gate is GC-owned)

port/gates/gate_p1.py:16-18

    def sh(args, timeout=300):
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[:600]

line 48 passes that 600-byte string to the regexes at :50-51. glslang emits **all** OpName/OpMemberName
debug ops before the first OpDecorate, so in spirv-dis output the first DescriptorSet sits at byte
**11,700 of 120,774** for the *smallest* program (bake_height, 1 binding). Measured with the gate's own code:
rc 0 / len dis passed to regex 600 / sets [] / binds []. Hence actual is always the empty set and
actual == declared and len(actual) > 0 is unsatisfiable for **any** manifest — the gate cannot pass as
written, and cannot distinguish a correct manifest from a wrong one.

One-line fix: return p.returncode, p.stdout + p.stderr (or spirv-dis --no-header with a large cap).
engine_vr/selfcheck_p1.py is the gate's logic with that line un-truncated and reports PASS; it also adds
the check the gate structurally cannot make (it compares *sets*, so it would never notice two resources
sharing a (set,binding)).

## API contracts relied on:

| binding | exact API used | verified at |
| --- | --- | --- |
| _glstrip (importlib by file path; stdlib re only, no moderngl) | resolve(src)->str active-branch resolution | engine/glstrip.py:33-86 |
| | strip_unreachable(src)->str, incl. the "seg = i+1 on depth-0 }" rule my _statement_spans mirrors | engine/glstrip.py:98-151 (rule at :110-118) |
| | _mask(src)->str comments + # lines blanked, length preserved | engine/glstrip.py:89-92 |
| | reduce(src) = resolve + strip + blank-run squeeze | engine/glstrip.py:154-156 |
| include loader (reimplemented; engine/gl.py is NOT imported — it imports moderngl at :1) | #include "f" inlined once per file | engine/gl.py:28-38 |
| program header | "#version 430\n" + "".join(f"#define {k} {v}\n") | engine/gl.py:50 (Pass.__init__) |
| compute header (valueless defines) | "#version 430\n" + f"#define {d}\n" | engine/erosion.py:173 |
| pass inventory | trace with targets=2; SCENE_KINDS 0..7; cone 98; probe | engine/render.py:286, :287, :289, :291 |
| bake inventory | bake_height / bake_near / bake_ao_far / bake_ao_near / bake_forest | engine/bake.py:56, :215, :592, :596, :710 |
| bake inventory (python-string sources) | RELIEF_FRAG, MACRO_FRAG (extracted from the AST, no import) | engine/bake.py:228, :266 |
| bake inventory | bake_trees, bake_shadow, bake_features | engine/render.py:168, :188; engine/features.py:253 |
| erosion inventory | _prog(name, defines), FLOW_ONLY variant | engine/erosion.py:170-181, :310 |
| shared vertex stage | VS = "#version 430\nin vec2 in_pos;…" | engine/gl.py:12-15 |
| toolchain | glslc (shaderc 2024.3, glslang 11.1.0, spirv-tools 2024.4), spirv-val, spirv-dis; spirv-cross wants the file BEFORE --reflect | tools/vk/bin/ |

## §450 diff log

**No shader body was edited at all.** diff -r engine/shaders engine_vr/shaders is empty (the 3 extra files
are the Python-string programs). The bump was free for the same reason: **no file in engine/shaders/ carries
a #version line** — the version comes from the Python header (engine/gl.py:50, engine/erosion.py:173), so
430→450 is a generator constant, and glslang's gl_PerVertex is implicit in 450 so gl_Position survives
untouched. All 450/Vulkan fallout was absorbed by the generator (glstrip_vr.vulkanize), which is where the
diff lives:

| # | generated change | count / where |
| --- | --- | --- |
| 1 | header #version 430 → #version 450 | all 28 generated programs |
| 2 | loose non-opaque uniforms moved **verbatim, in source order**, into layout(std140, set=1, binding=0) uniform _p1_pass { … }; with an **unnamed instance** (members stay in global scope → zero body edits); opaque types stay loose, where they are legal | 609 declarations (47 per scene pass, 18 probe, 27 erosion_hill, 16 erosion_accum, 9 bake_forest, 3-8 elsewhere) |
| 3 | layout(set=S, binding=N) injected on declarations lacking a binding | 198 samplers injected (209 total incl. 11 explicit) |
| 4 | set=0 / set=3 added to explicit bindings that named only a binding, so the DescriptorSet decoration is unconditional | 17 (erosion_fill 3, erosion_accum 5, erosion_accum_flow_only 4, erosion_hill 5 = 11 sampler2D + 6 image2D) |
| 5 | layout(location=N) injected on user in/out without one (lowest free number; explicit ones respected) | 11 (out vec4 fragColor; in bake_* / probe, in vec2 in_pos; in the VS) |
| 6 | multi-declarator opaque declaration split one line each so every name gets its own binding | 1 — uniform sampler2D uH, uL, uB; in bake.MACRO_FRAG |
| 7 | layout(local_size_x = 16, local_size_y = 16) in; left untouched (a declaration with no declarators gets no location) | 4 compute programs |
| 8 | inactive #if FLOW_ONLY branch removed by resolve() (preprocessor-equivalent; keeps the duplicate imgAcc text out of the scanner) | erosion_accum variants |

Numerical/visual semantics: **untouched** — no literal, constant, expression or swizzle changed anywhere, so
no FIXME(port) numeric marker was needed.

## §glslc strictness list (errors caused purely by 450-for-Vulkan strictness) — contractual input for P2

| # | glslc message (verbatim) | why GL 430 let it through | fix in this port | what P2 must do with it |
| --- | --- | --- | --- | --- |
| 1 | "'non-opaque uniforms outside a block' : not allowed when using GLSL for Vulkan" | OpenGL allows loose uniforms in the default uniform block, set by name (gl.Uniform) | grouped into _p1_pass (std140, set 1 binding 0) | pack **one** buffer per pass to programs[].uniform_block.members (name/type/offset); std140 rules, not C layout; no name lookup exists any more |
| 2 | "'binding' : sampler/texture/image requires layout(binding=X)" | moderngl binds by texture unit (World.bind units 0..12, GBUF_UNIT = 28) | deterministic set/binding injection | map the old image units onto the manifest's (set,binding); the unit numbers are gone from the source |
| 3 | "'location' : SPIR-V requires location for user input/output" | fragment outputs / VS inputs were linked by name | injected layout(location=N) | attachment and vertex-attribute tables come from programs[].outputs and shared_stages.vertex_fullscreen.vertex_inputs |
| 4 | "'std430 requires the buffer storage qualifier' : required extension not requested: GL_EXT_scalar_block_layout" | n/a — my own first choice for the synthetic block | switched the block to std140 (no extension, no buffer qualifier) | pack with std140 alignment (vec3 → 16 B, vec4[] stride 16, mat3 column stride 16) |
| 5 | "'location qualifier on input' : not supported in this stage: compute" + "'location/component/index' : cannot declare a default, use a full declaration" | n/a — my scanner treated layout(local_size_x = 16, …) in; as an in-variable | skip in/out declarations with no declarators | nothing; note gl_Position compiles fine in 450 (gl_PerVertex implicit) |
| 6 | *legality trap, not an error:* duplicate (set,binding) across namespaces | GL has no descriptor sets | one set per namespace (Deviation 1) | **3** set layouts exist today (0 samplers, 1 UBO, 3 storage images) — not 1; wgpu wants a bind group per set, per stage |

Verified clean under 450 (nothing had to change): no gl_FragColor / gl_FragData / texture2D / textureCube,
no attribute/varying, no GL_ARB_* extensions, no atomicAdd/imageAtomic* (erosion uses coherent
imageLoad/imageStore only → no float-atomic extension needed).

## Program table (manifest programs[], 25)

| program | stage | defines | bind | smp | img | expl | UBO B | outputs | mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene_trace | frag | QUALITY=1 SCENE_PASS=99 | 16 | 15 | 0 | 0 | 5824 | fragColor@0,fragInfo@1 | reduced |
| scene_kind0_sky | frag | SCENE_PASS=0 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_kind1_terrain | frag | SCENE_PASS=1 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_kind2_tree | frag | SCENE_PASS=2 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_kind3_rock | frag | SCENE_PASS=3 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_kind4_water | frag | SCENE_PASS=4 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_kind5_cover | frag | SCENE_PASS=5 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_kind6_hero | frag | SCENE_PASS=6 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_kind7_flora | frag | SCENE_PASS=7 | 17 | 16 | 0 | 0 | 5824 | fragColor@0 | reduced |
| scene_cone | frag | SCENE_PASS=98 | 15 | 14 | 0 | 0 | 5824 | fragColor@0 | reduced |
| probe | frag | – | 13 | 12 | 0 | 0 | 1040 | fragColor@0 | reduced |
| bake_height | frag | – | 1 | 0 | 0 | 0 | 20 | fragColor@0 | reduced |
| bake_near | frag | – | 2 | 1 | 0 | 0 | 24 | fragColor@0 | reduced |
| bake_ao_far | frag | – | 3 | 2 | 0 | 0 | 24 | fragColor@0 | reduced |
| bake_ao_near | frag | – | 3 | 2 | 0 | 0 | 24 | fragColor@0 | reduced |
| bake_forest | frag | – | 6 | 5 | 0 | 0 | 2100 | fragColor@0 | reduced |
| bake_relief | frag | – | 2 | 1 | 0 | 0 | 1572 | fragColor@0 | reduced |
| bake_macro | frag | – | 4 | 3 | 0 | 0 | 12 | fragColor@0 | reduced |
| bake_trees | frag | – | 13 | 12 | 0 | 0 | 536 | fragColor@0 | reduced |
| bake_shadow | frag | – | 3 | 2 | 0 | 0 | 44 | fragColor@0 | reduced |
| bake_features | frag | – | 2 | 1 | 0 | 0 | 1604 | fragColor@0 | reduced |
| erosion_fill | comp | – | 4 | 2 | 1 | 3 | 40 | – | resolved |
| erosion_accum | comp | – | 6 | 3 | 2 | 5 | 76 | – | resolved |
| erosion_accum_flow_only | comp | FLOW_ONLY | 5 | 3 | 1 | 4 | 76 | – | resolved |
| erosion_hill | comp | – | 6 | 3 | 2 | 5 | 120 | – | resolved |

QUALITY=1 is implicit in every scene row except the two extra_variants_checked (QUALITY=0, both pass).
Binding numbers are per program and are **not** stable across programs — never assume them, read the
manifest (uCone is set0/binding14 in scene_trace, while the kind passes have uGBuf0/uGBuf1 at 14/15).

## Risks / FIXME(port) list:

- **engine_vr/manifest.json → descriptor_conventions** — P2 builds 3 descriptor set layouts (0 samplers,
  1 the single _p1_pass UBO, 3 storage images), not one. wgpu wants a bind group per set per stage;
  MoltenVK wants every set in the pipeline layout even where a pass leaves one unbound.
- **engine_vr/glstrip_vr.py UBLOCK_NAME/_p1_pass** — 609 loose uniforms became one std140 block; the scene
  UBO is **5 824 B** (fits the 16 KiB maxUniformBufferRange floor, but it is one upload per pass — pack once
  per frame, patch the few per-pass members). Byte layout: uniform_block.members.
- **engine/erosion.py:170-181 + erosion_*.comp** — coherent imageLoad/imageStore read-modify-write across
  workgroups compiles and validates, but nothing static proves that ordering survives MoltenVK→Metal; the
  P0-class risk stands (there are no atomics anywhere, so no fallback either). First thing to probe on the
  first GPU run.
- **port/gates/gate_p1.py:18** — the 600-byte truncation makes reflection parity unpassable (Gate bug).
  Until it is fixed, gate_p1 proves only "compiles + validates"; engine_vr/selfcheck_p1.py proves parity.
- **engine_vr/glstrip_vr.py _statement_spans / vulkanize** — a text scanner over _mask()ed source: a top-level
  { } that is *not* an interface block would be classified as a block, and an explicit binding on a
  multi-declarator sampler declaration is a hard error (neither shape exists in today's suite).
- **engine/bake.py:228/:266** — RELIEF_FRAG/MACRO_FRAG are read out of bake.py's AST (build_p1.py:py_str_const),
  so upstream edits flow into sync_sources() on the next build; the three generated copies under
  engine_vr/shaders/ are deliberately never overwritten once hand-edited (edited_in_port list).
- **Static gates cannot prove equivalence**: texture LOD/bias, dFdx under non-uniform control flow,
  textureGather offsets and sRGB framebuffer behaviour on Metal are untested here — GL→Metal semantics are
  Phase 2/3 territory.
- **engine/render.py:286** — targets=2 (fragColor@0, fragInfo@1) is only discoverable from Python; the
  manifest records it as 2 outputs on scene_trace. Every other program has 1. Do not lose fragInfo.

## Reproduce

    .venv/bin/python engine_vr/build_p1.py        # 25/25 compiled+validated, writes manifest.json
    .venv/bin/python engine_vr/selfcheck_p1.py    # SELFCHECK: PASS — 25 programs, 240 pairs
    .venv/bin/python port/gates/gate_p1.py        # FAIL (25) — gate bug above; 0 glslc / 0 spirv-val failures
