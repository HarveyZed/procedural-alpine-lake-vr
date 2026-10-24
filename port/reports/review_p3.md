# Review P3 — bake/erosion/features VR drivers (independent contract review)

Reviewer: independent contract reviewer (read-only; no GPU code executed; no edits
outside this report). Scope: engine_vr/passes_vr.py (512 L), engine_vr/cache_schema.py
(199 L), engine_vr/bake_vr.py (912 L), engine_vr/erosion_vr.py (1765 L),
engine_vr/features_vr.py (146 L), engine_vr/build_p3.py (382 L), manifest additions,
glstrip_vr/shaders_gen flow_only variant, gates, phase3 report.
Reference baseline: engine/{bake,erosion,features,render,gl}.py, manifest schema,
port/reports/phase3.md. Vulkan ground truth: vulkan 1.3.275.1 in .venv
(imported with `tools/vk/env.sh` sourced — `import vulkan` dlopens bare
libvulkan and raises OSError otherwise, RUNBOOK §2; my round-1 note that "names
need no env" was wrong, corrected here).

## 1. API table verdict

- Sites checked in the four new modules: **43** — 2 direct `vulkan.*` function calls
  (`vkDestroyFramebuffer` passes_vr.py:283, `vkDestroyRenderPass` :284, both 3
  positional args = (device, obj, pAllocator) ✓), **37 constants**, **4 `VK_…` name
  strings** resolved through `getattr(vkv, name)` for fmt/final_layout
  (gl_vr.py:405, :964, :968), **0 raw-cffi sites**. Ground-truth check against the imported module (`hasattr` + signature
  arity, /tmp/p3rev/{ast3,tri3}.py): **43/43 confirmed, 0 discrepancies**.
- P3-called gl_vr wrapper surface (`upload_2d`, `make_image` incl. `samples=1`
  default and `usage=` override, `make_target(name,res,res,depth,fmt,final_layout)`,
  `write_descriptors(samplers=)`, `draw(rect, cleared, cmd)`, `fetch`, `to_staging`,
  `write_uniforms`, `staging/read`, `Host.wait/end`, `program_for/build`) —
  signatures re-read against every call site in the new files: all match.
- gl_vr.py unchanged since P2 ACCEPT (1480 lines, wc matches P2 artifact); the P2
  full sweep (73 funcs / 73 ctors / 184 consts, 0 discrepancies) still covers it.
- Manifest: 15 P3 programs cross-validated by my own std140 recompute (earlier
  round): every uniform/texture/storage binding + UBO byte layout matches shader
  `layout(...) uniform/set/binding` and member offsets; no name collisions;
  `erosion_accum_flow_only` source =
  `engine_vr/shaders_gen/erosion_accum_flow_only.comp` (a genuine preprocessed
  variant with `#define FLOW_ONLY` baked at line 2, dead branches stripped), rebuilt
  by ensure_spirv define-hash + mtime (gl_vr.py:1010-1012). GpuImage sets `samples=VK_SAMPLE_COUNT_1_BIT`
  explicitly (gl_vr.py:409-419, field :417) — the multisample image-creation
  trap does not apply.

## 2. Copy fidelity

- 19 `# ---- copied from engine/<file>:<s>-<e>` blocks (11 bake / 3 erosion / 5
  features) in the stated regions; AST round-trip via `ast.unparse(ast.parse(...))`
  (whitespace-insensitive by construction): **19/19 byte-identical to the engine
  block** (/tmp/p3rev/copyintegrity.py). The one intentional exception is visible
  and marked: erosion `_read` (erosion_vr.py:123-136) adapts gl_vr's (h,w)/(h,w,c)
  fetch shape; header constants are `cs.*` aliases (bake_vr.py:45-51,
  features_vr.py:35-40) — single source, cannot drift.
- The AST method's blind spot is comments/docstrings only; I additionally verified
  every engine constant the copied code depends on resolves to the same value via
  cache_schema. No numeric literal is retyped anywhere outside copied blocks.
- Substitutions all reduce to module aliasing (`import features` → features_vr,
  `gl.tex2d` → `pv.upload` through `_Gl` with identical dtype rules, cache-GLSL
  write → program-name string) plus the documented `bake_ao.frag` → `bake_ao_far`/
  `bake_ao_near` manifest split (same source, two programs — manifest confirms both
  entries bind uH/uHFar identically).

## 3. Findings

CRITICAL: none. MAJOR: none.

MINOR:
- **N1 (must fix, report text)** phase3.md:266-268 describes risk #1 as
  "imgAcc (rg32f, atomic Add)". There are **zero atomic operations in every shader
  of this port** (`grep -rn atomic engine/shaders/ engine_vr/shaders_gen/` → no
  matches). The flow pass is neighbor-gather + per-thread store:
  `imageLoad(imgAcc, q)` for neighbors (erosion_accum_flow_only.comp:97) and
  `imageStore(imgAcc, p, vec4(A))` per owning thread (:144), on `coherent image2D`
  (:62). The residual hardware question is coherent load/store visibility +
  compute→sample re-read + MVK R32G32_SFLOAT storage support — NOT atomic format
  feature bits. Fix the wording so the P4 first-run failure matrix doesn't chase
  VK_FORMAT_FEATURE_*_ATOMIC_BIT.
- **N2** build_p3.py:349-350 advertises `--check` as "report what would change,
  write nothing", but :353-359 splice/rewrap/annotate *first* and :365 then raises
  `SystemExit("--check is not supported after splicing")` — the flag has already
  mutated the files when it aborts, and only after the first target. Either
  implement pre-splice diffing or delete the flag (one-line fix).
- **N3** passes_vr.bake_pass:307-338 creates the target at :328 and frees it at
  :337 with no try/finally; an exception inside the tile loop or readback leaks a
  target up to 8192²×16 B = 1.07 GiB until `Host.close` (slot is still released —
  run_* wrappers own close in finally, bake_vr.py:162-164/:184-186 — so it is a
  transient-footprint issue, and MVK reclaims everything at device destroy).
  One-line try/finally around :331-336.
- **N4** to_sampler_layout/to_general_layout compute `todo` before mutating
  layouts (passes_vr.py:428-446, 455-474); the hill stage falls back to binding
  `net = … else self.M` (erosion_vr.py:286-287 == engine/erosion.py:294-295), so
  M can appear twice in the sampler list and receive a second, legal
  self-transition barrier (SRO→SRO with srcStage BOTTOM_OF_PIPE/0, valid no-op).
  Also noted: the None-sampler/arity guards (erosion_vr.py:195-213) raise where
  the engine would too (`t.use()` on None), so parity holds. Harmless; note only.
- **N5** build_plan (bake_vr.py:140-144) documents `uHasErosion: 0, uStreamN: 0`
  as "the defaults … actually pass" while the copied runtime path derives
  uHasErosion from cache presence (`has = talus.npy and flow.npy exist`, copied at
  bake_vr.py:841 == engine/bake.py:690,712) and streams from features meta
  (bake.py:708-709 copied at bake_vr.py:838-839). Runtime is correct in both the
  fresh-cache and eroded-cache cases; the plan docstring overstates. Doc-only.
- **N6** erosion `_upload`/`_tex` require square 2-D images (`a.shape[0]`,
  erosion_vr.py:120); engine `texture()` was likewise square. Non-square inputs
  would mis-size silently in either backend — pre-existing parity, not a VR defect.
- **N7** engine/erosion.py `shade_rgb` (engine/erosion.py:1558) really does carry
  undefined `nx`/`nz` (F821; author's non-copy decision verified correct).
- Carried from P2 (unchanged, out of P3 scope): N9 `_pad_rows` `shape[-2]`
  wrong dim for depth>1 (latent, unused by P3); N10 always-fire sampled-read
  barrier; gpuslot rusage-zero headroom.

## 4. Author's named risks — verdicts

1. **rg32f/r32f storage writes/atomics/coherent RMW (phase3.md:266-289)** —
   **RISK REAL, PROBE DESIGN AGREED, WORDING WRONG.** The 512² synthetic-cone
   probe is minimal, needs no cache, and exercises fill→accum→hill→read_state
   exactly like erode() (STAGES[0] + `erode(hfar, cell_m, stages=…)` signature
   parity engine/erosion.py:1240 == erosion_vr.py:1457 confirmed). But see N1:
   nothing in this port is an atomic; it is coherent imageLoad/imageStore + a
   compute-written r32f re-sampled by the next program. Rank #1 stays correct.
2. **Layout/validation chain across compute↔sample (phase3.md:290-296)** —
   **AGREE, and stronger than stated.** The freeze is real: write_descriptors
   bakes `img.layout` into each WriteDescriptorSetImage (gl_vr.py:1190-1191,
   :1205), so any post-descriptor layout change is a validation-class bug; P3's
   helpers transition BEFORE descriptor writes in `_run` (erosion_vr.py Sim) and
   `barrier_compute_to_read` names only images the dispatch wrote
   (erosion_vr.py:232, the fix the report highlights). gl_vr only ever leaves
   images in {UNDEFINED, TRANSFER_DST, GENERAL, SRO, TRANSFER_SRC,
   COLOR_ATTACHMENT} (upload_2d mbar M4 :893-898, fetch :913-929, draw
   :1282-1300, renderpass final layouts), which the to_* helpers cover. The
   unrecognised-layout fallback (BOTTOM_OF_PIPE/0) is correctly self-flagged.
3. **1.07 GiB target lifetime + readback row order (phase3.md:297-304)** —
   **AGREE, and readback fidelity is verified statically.** No double flip
   anywhere in the P3 chain: engine read_tex flips nothing
   (engine/gl.py:131-135), VR to_staging/fetch is row-order-preserving
   (gl_vr.py:905-949), and every engine flip is display/PNG-only
   (engine/render.py:505, engine/erosion.py:1576). Targets are destroyed per
   pass (passes_vr.py:271-289, 337), Host.close ordering was P2-verified, and
   the gpuslot watchdog measures this process which holds everything between
   begin/end. The proposed h_far_raw patch-vs-CPU probe is the right test; the
   N3 finally gap is the only lifetime blemish.

Secondary risks 4-6 (double descriptor uH==uHFar — legal in Vulkan, two
descriptors over one view, read-only; uStream/uTrail array-UBO member offsets —
my std140 recompute confirms uStreamN@1600 / uTrailN@2096; per-band submit cost —
correctness-neutral) are all correctly characterized.

## 5. Barrier coverage vs the P2 docstring table

No new gaps. Table rows re-checked against P3 flows: upload carries the M4
src-access barrier; draw load-reentry names the last toucher derived from the
image's rest layout (gl_vr.py:1282-1298) and the first draw out of UNDEFINED
names COLOR/WRITE — a harmless conservative transition; to_staging is a WAR
read after the last write stage; the bake/features targets rest in
TRANSFER_SRC so the sampled post-barrier (gl_vr.py:1339-1355) does not fire;
erosion's compute→sample round trip is explicitly bridged
(barrier_compute_to_read → to_sampler_layout before descriptor writes).
Per-band one-submission-per-band (passes_vr.py:365-391) is strictly *more*
flushing than the engine cadence — performance risk only, as the report says.

## 6. Verdict

**ACCEPT-WITH-FIXES** — no CRITICAL/MAJOR findings; API sweep 43/43 clean; copy
fidelity machine-proven (19/19 AST-equal); gates independently re-run green
(py_compile, ruff E,F, gate_p3 PASS).

Mandatory fix (trivial, report-only):
1. N1 — correct phase3.md:266-268 "atomic Add" wording; P4's probe stays but must
   debug visibility/format-support, not atomics.

Recommended (do before or during P4):
2. N2 — honest `--check` or drop the flag (build_p3.py:349-365).
3. N3 — try/finally around bake_pass draw/read (passes_vr.py:331-336).

Nothing here blocks the P4 first-run ladder; risk #1's probe remains the correct
gate before the full bake.
