# Phase 3 report — p3 / 2026-10-25 07:10 +08:00

## Status: DONE

Bake / erosion / feature drivers now drive `engine_vr/gl_vr.py` instead of
moderngl, with zero algorithm or constant drift, plus the cache contract
Phase 4 renders against. Nothing in this phase opened a device: no
`Host.open()`, no `vk()`, no queue submit was ever executed here (see
"Gate self-check" and `engine_vr/tmp/probe_p3.py`).

## Files written (paths + LOC)

| file | LOC | role |
| --- | --- | --- |
| `engine_vr/passes_vr.py` | 475 | host/session glue: `open_host`/`close_host`, `Tex`, `upload`, `storage_image`, `write_uniforms`, `program_for`, `bake_pass`, `compute_pass`, `compute_bands`, `barrier_compute_to_read`, `to_sampler_layout`, `to_general_layout`, `make_target`/`destroy_target`, `read_image` |
| `engine_vr/cache_schema.py` | 203 | T3 cache contract: 25 registered products (name -> dtype/shape/res/half/writer), ALPINE_CACHE-aware `cache_dir()/path()/spec()/cell()/shape_of()/missing()/checkpoint_name()` |
| `engine_vr/bake_vr.py` | 907 | bake driver: `build_plan` (7 passes), `run_all`, `run_eroded_bake`, 11 copied blocks from engine/bake.py |
| `engine_vr/erosion_vr.py` | 1765 | erosion driver: VR `_tex/_upload/_read`, `Sim`, `build_plan` (6 passes), `run_erosion`, 3 copied blocks from engine/erosion.py (incl. `erode` verbatim) |
| `engine_vr/features_vr.py` | 535 | feature driver: `_UniformSink`, `_uniform_fields`, `carve`, `build_plan` (2 passes), `run_carve`, 5 copied blocks from engine/features.py |
| `engine_vr/build_p3.py` | 381 | the copy assembler (P1 precedent `build_p1.py`): splices engine ranges into the three drivers, applies declared substitutions with asserted counts, rewraps copied docstrings, adds lint pragmas - all AST-verified |
| `engine_vr/tmp/probe_p3.py` | - | no-GPU probe: blocks `Host.open`/`Host.build`/`vk`, imports the modules, runs the plans, checks band/tile geometry |
| `engine_vr/tmp/audit_p3.py` | - | this report's numbers: LOC, copy blocks, constants, UBO coverage |
| `port/reports/phase3.md` | this file | - |

gl_vr.py (1480) and gpuslot_vr.py (458) were **not** touched, reformatted or
renamed - imported exactly as reviewed and accepted.

## Deviations from brief

None of substance; five implementation choices worth naming:

1. **No GLSL is written to the cache any more.** `macro_post`/`relief_post`
   in the original do `src = CACHE/"_bake_macro.frag"; src.write_text(...)`
   and then pass that Path as the program name. The VR host compiles one
   SPIR-V per *manifest program* from `engine_vr/shaders_gen`, so the copies
   pass the names `"bake_macro"` / `"bake_relief"` instead (P1 mirrored
   `MACRO_FRAG`/`RELIEF_FRAG` into those sources; the manifest sources for
   these two programs are exactly those strings). No shader text is
   duplicated or retyped.
2. **`bake_ao.frag` became two program names.** One GLSL source, two manifest
   entries (`bake_ao_far` at res 4096 with `uStep0=8.0`, `bake_ao_near` at
   res 4096 with `uHalf=NEAR_HALF`, `uStep0=1.5`), so `bake_ao` names each
   explicitly rather than routing through `program_for()`.
3. **Copied text is byte-identical except whitespace inside copied docstrings
   and trailing `# noqa` comments.** `build_p3.py` rewraps over-long copied
   docstrings to keep ruff E501 clean and appends per-line lint pragmas; both
   steps are accepted only when `ast.dump()` (with docstrings
   whitespace-normalised) is unchanged - i.e. comments/whitespace only. No
   constant, threshold, ordering or expression was retyped: 19 copy blocks
   carry `# ---- copied from engine/<file>:<s>-<e> (<subs or verbatim>)`
   headers and 40 module-level engine constants travel with those headers.
4. **Erosion flush cadence became per-band submits.** The engine rewrote
   `uOffset` per band inside one command stream and called
   `ctx.finish()` every 32/8/2 dispatches. A UBO write is only legal at queue
   idle, so each band is now write-uniforms, write-descriptors, dispatch,
   wait. Fewer dispatches are batched, nothing else changed; the cadence
   `finish()` is kept as `host.wait()` for parity.
5. **`_uniforms()` is driven through `_UniformSink`.** The copied
   `features._uniforms(p, f, center, half, res)` calls `p.set(name, value)`;
   the sink is a 12-line object whose `.set()` records the fields, so the
   original function stays verbatim and its 8 keys are exactly the manifest
   members (checked: 8/8).

## Gate self-check

```
$ .venv/bin/python port/gates/gate_p3.py
PASS bake_vr.py exists
PASS erosion_vr.py exists
PASS features_vr.py exists
PASS cache_schema.py exists
PASS py_compile bake_vr.py
PASS py_compile build_p1.py
PASS py_compile build_p3.py
PASS py_compile cache_schema.py
PASS py_compile erosion_vr.py
PASS py_compile features_vr.py
PASS py_compile gl_vr.py
PASS py_compile glstrip_vr.py
PASS py_compile gpuslot_vr.py
PASS py_compile passes_vr.py
PASS py_compile selfcheck_p1.py
PASS ruff E,F engine_vr/
PASS no moderngl/wgpu imports in engine_vr
PASS no module-level bake_vr device calls
PASS no module-level erosion_vr device calls
PASS no module-level features_vr device calls
PASS no module-level cache_schema device calls
PASS build_plan cross-validates bake_vr - 7 passes
PASS build_plan cross-validates erosion_vr - 6 passes
PASS build_plan cross-validates features_vr - 2 passes
PASS read-only trees untouched - []

GATE P3: PASS

$ .venv/bin/python -m ruff check --select E,F engine_vr/
All checks passed!
```

Extra self-checks run locally (all CPU-only, no device):

```
$ .venv/bin/python engine_vr/tmp/probe_p3.py
plans are pure data: bake 5428 B, erosion 3004 B, features 3775 B json
no vulkan module imported by any driver import      (assert 'vulkan' not in sys.modules)
ladder  512^2: 33 groups, 1 band,  528 rows | 1024^2: 65 groups, 1 band, 1040 rows
       2048^2: 129 groups, 2 bands, 2048 rows | 4096^2: 257/3/2048 | 8192^2: 513/5/2048
       (never above 2048 rows; every group covered exactly once)
tile 256 at 8192^2: 1024 draws, exact tiling | 512 at 4096^2: 64 | 512 at 8192^2: 256
tile 1024 at 4096^2: 16 | 1024 at 8192^2: 64

$ .venv/bin/python - <<'PY'   # STD140 array members, CPU only
bake_features size 1604: uStream[0].x @64 = 0.0, uStream[3].y @116 = 6.0, uStreamN @1600 = 96
bake_forest   size 2100: uStream[1].z @56 = 3.0, uStreamN @1568 = 96,
                         uTrail[2].w @1628 = 4.0, uTrailN @2096 = 32
erosion_hill  size  120: uRes.x @0 = 512, uShift.x @16 = 8, uApply @28 = 1, uSeed @116 = 32761
erosion_accum_flow_only size 76: uP @28 = 0.35, uK @32 = 0.0 (skipped member), uInner @64 = 16
unknown uniform key -> KeyError('uNope') from Std140Packer.pack
```

## API contracts relied on

| binding | exact API | verified at |
| --- | --- | --- |
| session order | `gpuslot_vr.acquire()` **before** `Host.open()`, release after `close()` | `engine_vr/passes_vr.py:75-116`; port/reports/phase2.md addendum (P3 contract notes) |
| UBO writes only at queue idle | `Host.set_uniforms(prog, fields)` -> `Std140Packer.pack` | `engine_vr/gl_vr.py` (`set_uniforms`, `Std140Packer`), phase2 addendum; port/reports/review_p2.md 7 |
| descriptors freeze the layouts the images rest in *right now* (GENERAL stays GENERAL, anything else binds as SHADER_READ_ONLY) | `Host.write_descriptors(prog, samplers=, storage_images=, ubo_now=True)` | `engine_vr/gl_vr.py:1177-1226`, layout choice at :1190; review_p2 7 ("bind only what you just wrote/waited") |
| image layout bookkeeping | `GpuImage.layout` maintained at :436 (init), :881/:899 (`upload_2d` ends at rest_layout), :929 (`to_staging` -> TRANSFER_SRC), :1300/:1357 (`draw` -> color attachment -> `final_layout`) | `engine_vr/passes_vr.py:417-475` mirrors exactly those transitions |
| `memory_barrier()` == `barrier(COMPUTE->COMPUTE, SHADER_WRITE->SHADER_READ)` over storage images resting GENERAL | `Host.barrier(cmd, src_stage, dst_stage, src_access, dst_access, images)` | port/reports/phase2.md addendum; `passes_vr.py:394-414` |
| readback keeps engine row order (no vertical flip anywhere) | `Host.to_staging`/`fetch`/`read_2d`, tile rect verbatim to `VkRect2D`/`VkViewport` | phase2 addendum ("Y-ORIENTATION CLAIM", `gl_vr.py:853-856`) |
| `draw()` raises when `len(prog.outputs) != target.n` (M6) | `Host.draw` | review_p2 7 (M6), re-checked in `passes_vr.py:321-323` |
| program names/bindings | manifest `programs[]`: `uniform_block.members`, `bindings[].{name,kind,set,binding}`, kinds exactly `combined_sampler`/`uniform_block`/`storage_image`; `descriptor_conventions` set0 sampler, set1 UBO, set3 storage_image | `engine_vr/manifest.json`, `gl_vr.py:1087-1095`, `1146-1150`; cross-validated by `port/gates/gate_p3.py` |
| local size / band math | `Host.local_size(name)` -> (16,16,1) from `#define TS 16`; `tileOrigin = (gl_WorkGroupID + uOffset) * TS - uShift - halo` so `uOffset` is in TILES, `uShift` in cells | `engine_vr/shaders_gen/erosion_*.comp`; `erosion_vr.py:71-85` asserts `local_size == (16, 16, 1)` |
| target formats | `FMT_BY_COMPS_DTYPE` {(1,'f4'): R32_SFLOAT, (2,'f4'): R32G32_SFLOAT, (4,'f4'): RGBA32_SFLOAT, (4,'u1'): R8G8B8A8_UNORM, ...} | `gl_vr.py` module constants; `passes_vr.py:154-225` |
| unknown uniform key behaviour | `Std140Packer.pack` raises `KeyError('uNope')` (not `GlvrError`) - the drivers never rely on catching it, the gate proves no unknown key exists | probe above |

## Per-pass mapping (engine line range -> vr call)

| engine site | what it did in GL | engine_vr call |
| --- | --- | --- |
| `engine/bake.py:26-43` `_bake_pass` | texture(res,res,4,'f4') -> fbo -> `gl.Pass(frag)` -> `p.set`/`p.tex` -> tile loops with `c.finish()` -> `gl.read_tex` -> releases | `passes_vr.bake_pass()` `passes_vr.py:307-340`: `host.build` -> `write_uniforms` -> `write_descriptors(samplers)` -> `make_target(res,res,1,RGBA32_SFLOAT,TRANSFER_SRC)` -> per tile `begin/draw(rect)/end(wait)` -> `read_image` -> `destroy_target` |
| `engine/gl.py` `tex2d(arr, filt, repeat, dtype)` | comps from trailing axis, f1=uint8/f2=f16/else f32, NEAREST or LINEAR | `passes_vr.upload()` `passes_vr.py:154-186` -> `Host.upload_2d` resting SHADER_READ_ONLY; sampler state carried by `Tex` and handed to `write_descriptors` as `{binding name: (img, filt, addr)}` |
| `engine/gl.py` `read_tex(t, comps)` | `fbo.read` -> reshape -> slice | `passes_vr.read_image()` `passes_vr.py:291-304` (`to_staging`+`fetch` in one session, plus the mandatory copy - the engine mutates its readback in place at `bake.py:75`) |
| `bake.py:46-95` `bake_heights` | analytic far bake + erosion callback + shore/beach numpy + features + relief + carve | `bake_vr.py:200-252` copied; only substitution: `import features` -> `from engine_vr import features_vr as features` |
| `bake.py:56-57` far pass | `_bake_pass('bake_height.frag', 8192, {uCenter,uHalf,uRes,uFine:0}, tile=256)` | plan `height_far` -> program `bake_height`, 4/4 members, tile 256 |
| `bake.py:61` erosion hook | `erode(hfar, 2*FAR_HALF/far_res)` | `bake_vr.run_eroded_bake()` `bake_vr.py:167-186` wires `erosion_vr.erode` (same signature/return dict) |
| `bake.py:98-225` | `_lake_window`, `_ss`, `shore_bathymetry`, `shore_near`, `BEACH_*`, `beaches`, `bake_near_from_far` | `bake_vr.py:254-391` copied (`bake_near_from_far` -> program `bake_near`, sampler `uHFar` nearest) |
| `bake.py:288-312` `_blur_ds`/`macro_post` | numpy blur pair + `_bake_pass(_bake_macro.frag, res, {uHalf,uRes,uDbg}, [uH,uL,uB], comps)` | `bake_vr.py:394-420`; program `bake_macro`, 3/3 members |
| `bake.py:315-422` `talus_cones` | pure numpy (reads CACHE/flow.npy) | `bake_vr.py:429-537` verbatim |
| `bake.py:425-461` `river_post` | pure numpy -> river.npz | `bake_vr.py:540-580` verbatim |
| `bake.py:464-524` `DEFIN_HALF`/`defin_post` | pure numpy | `bake_vr.py:586-648` verbatim |
| `bake.py:527-582` `relief_post` | macro + `_bake_pass(_bake_relief.frag, res, {uHalf,uRes,uStream,uStreamN,uTarn,uDbg}, [uH])` + numpy post + talus/cones saves | `bake_vr.py:651-708`; program `bake_relief`, 6/6 members |
| `bake.py:585-602` `bake_ao` | two `_bake_pass('bake_ao.frag', ...)` at 4096, same texture bound twice for the far call | `bake_vr.py:711-728`; programs `bake_ao_far` / `bake_ao_near`, samplers uH+uHFar, 5/5 members each, float16 saves |
| `bake.py:605-722` `world_maps` | numpy water/forest + `_bake_pass('bake_forest.frag', res, 9 uniforms, [uH,uLvl,uTalus,uFlow,uCones], comps=4, tile=1024)` | `bake_vr.py:732-852`; program `bake_forest`, 9/9 members, tile 1024, uint8 quantise kept |
| `bake.py:725-774` `bake_noise3d`/`bake_all` | pure numpy, 128^3x4 uint8 | `bake_vr.py:855-906` verbatim |
| `engine/features.py:118-128` `_uniforms` | `p.set` of 8 members incl. `uStream` 96xvec4 | copied verbatim `features_vr.py:272-283` onto `_UniformSink` (`features_vr.py:43-66`) |
| `engine/features.py:246-277` `carve` | `gl.ctx()`, `tex2d(h)`, texture+fbo, tile loop with `c.finish()`, `read_tex`, far/near numpy branches | `features_vr.carve()` `features_vr.py:69-108`: `upload` -> `write_uniforms` -> `write_descriptors({'uH': ...})` -> `make_target` -> per-tile `begin/draw/end` -> `read_image` -> `destroy_target`; all numpy branches copied unchanged |
| `engine/features.py:280-396` | `_stream_corridor`, `_falls_window`, `_nb8`, `_limit_cliff`, `_check_near`, `_damp_flow_banks`, `save_meta`/`load_meta` | copied, `bake.CACHE` -> `cs.CACHE` (4 sites) |
| `engine/erosion.py:154-198` | `_src`/`_prog` GLSL include inlining, `_tex` (NEAREST), `_upload`, `_read` | `erosion_vr.py:52-137`: `_program_name` (maps `erosion_accum.comp` + `('FLOW_ONLY',)` -> `erosion_accum_flow_only`, raises on any other define), `_prog` (asserts `local_size == (16,16,1)`), `_binding_names` (positional order = manifest `binding` index), `_tex`/`_upload`/`_read` over `storage_image`/`read_image` |
| `engine/erosion.py:201` `BAND_TILES=128` | 128 tile rows = 2048 cells per dispatch | `erosion_vr.py:46` (annotated) |
| `engine/erosion.py:242-254` `Sim._run` band loop | `g=res//16+1`; per band `uOffset=(0,y0)`; `prog.run(g, min(128, g-y0), 1)`; one `memory_barrier()` after the loop; finish every 32/8/2 | `erosion_vr.py:187-236` + `passes_vr.compute_bands()` `passes_vr.py:365-391`: same grid, same band height, per band uniforms+descriptors+dispatch+wait, then `barrier_compute_to_read(storage images)` |
| `engine/erosion.py:204-331` `Sim` | H ping-pong parity + `uShift=8*parity`, W/acc/net allocation, fill/fill_converge/accumulate/hill/step/flow_only/read_h/write_h/read_state/release | `erosion_vr.py:153-327` (same method names, same parity, same kwargs; `release()` -> `GpuImage.destroy()`) |
| `engine/erosion.py:94-148`, `337-1234`, `1240-1544` | constants, numpy/scipy helpers, `erode()` | `erosion_vr.py:466-1765` verbatim (`erode` untouched, including the `Sim` call sites and the final-level `fsim`/`alloc_flow(acc_comps=1)` hand-off) |

## UBO coverage (T1: every member set, or skipped with rationale)

| program | members | set by the port | bindings used |
| --- | --- | --- | --- |
| `bake_height` | 4 | 4/4 `uCenter uHalf uRes uFine` | - |
| `bake_near` | 5 | 5/5 `uFarHalf uFarRes uCenter uHalf uRes` | sampler `uHFar` |
| `bake_ao_far` | 5 | 5/5 `uCenter uHalf uRes uFarHalf uStep0` | `uH`, `uHFar` |
| `bake_ao_near` | 5 | 5/5 same | `uH`, `uHFar` |
| `bake_macro` | 3 | 3/3 `uHalf uRes uDbg` | `uH`, `uL`, `uB` |
| `bake_relief` | 6 | 6/6 `uHalf uRes uStream uStreamN uTarn uDbg` | `uH` |
| `bake_forest` | 9 | 9/9 `uHasErosion uHalf uRes uTarn uTarnLevel uStream uStreamN uTrail uTrailN` | `uH`, `uLvl`, `uTalus`, `uFlow`, `uCones` |
| `bake_features` | 8 | 8/8 `uWF uWFDir uWFNotch uCenter uHalf uRes uStream uStreamN` | `uH` |
| `erosion_fill` | 7 | 7/7 `uRes uOffset uShift uCell uEps uInit uInner` | samplers `texH`, `texM` / storage `imgW` |
| `erosion_accum` | 16 | 16/16 `uRes uOffset uShift uCell uP uK uMexp uNexp uG uKloose uMaxDep uMaxEro uHardK uInner uLakeTrap uPitTrap` | `texH`, `texW`, `texM` / `imgAcc`, `imgNet` |
| `erosion_hill` | 27 | 27/27 `uRes uOffset uShift uCell uApply uTb uTbLow uCb uTl uCl uT1 uR1 uCover uDebris uHardExp uHardExpLow uTreeA uTreeB uRockA uRockB uSpikeK uWexp uLooseMax uJitter uSeed uCreep uCreepRock` | `texH`, `texM`, `texNet` / `imgH`, `imgTrack` |
| `erosion_accum_flow_only` | 16 | **6/16** `uRes uOffset uShift uCell uP uInner` | `texH`, `texW`, `texM` / `imgAcc` |

The single skip list, `erosion_accum_flow_only`: `uK uMexp uNexp uG uKloose
uMaxDep uMaxEro uHardK uLakeTrap uPitTrap`. Rationale: the original calls
`erosion_accum.comp` with `defines=("FLOW_ONLY",)` and passes only `uP`,
`uInner` (+ routing); that program object's other UBO words were never
written by anything, so they read zero in the GL build forever. `pack()`
zero-fills unwritten members, so the VR build writes the same bytes -
verified on CPU: `flow_only size 76: uP @28 = 0.35, uK @32 = 0.0, uInner @64
= 16, uPitTrap @72 = 0`. `bake_trees` and `bake_shadow` are Phase 4's
(`engine/render.py:151-205`) and are deliberately not driven here.

Dispatch/geometry discipline (T5) is machine-checked, not asserted in prose:
ladder 512->4096 for the fluvial stages plus the 8192 hillslope stage, 16x16
tiles, bands of `BAND_TILES=128` tile rows, and every dispatch <= 2048 rows
with full group coverage (512^2: 1 band/528 rows ... 8192^2: 5 bands, 2048
rows max). Pass ordering inside a level is the copied `Sim.step`: fill ->
accumulate -> hill(fluvial) -> extra hill(thermal) - and the FLOW_ONLY variant
is used only where the original used it, in the final-level `fsim` block.

## Cache contract (T3)

`engine_vr/cache_schema.py` registers 25 products with dtype/shape/res/half
and the exact writer, honouring `ALPINE_CACHE` exactly like
`tools/render_final.py` / `tools/bake_final.py` (`Path(os.environ.get(
"ALPINE_CACHE", ROOT/"cache_final"))`, resolved at import, never created at
import - `ensure_dir()` is explicit). Names Phase 4 reads: `h_far.npy`,
`h_near.npy`, `water_level.npy`, `forest.npy` (`RENDER_REQUIRED`,
`cache_schema.py:152-154` = the four `engine/render.py:123` insists on),
plus `h_far_raw`, `h_eroded`, `erosion_height`, `erosion_{flow,talus,
sediment}`, `flow`, `talus`, `sediment`, `stream_level_far`,
`stream_level_near`, `stream_near` (f16 RGBA), `water_level_near`,
`ao_far`/`ao_near` (f16), `cones` (f16), `river.npz` (z,x,w,hb f4),
`noise3d.npy` (uint8 128^3x4), `stream_corr` (f16 RGBA), `features.json`,
`ck_{res}.npz` checkpoints, and `trees.npy` marked as a Phase-4 product
(`engine/render.py:178`) so P3 never claims it. `spec()` accepts
`"h_far.npy"` or the bare stem `"h_far"`; `cell()`/`shape_of()` derive from
the same table (far cell 4.0 m, near 1.0 m). `ALPINE_CACHE=/tmp/p3_probe`
verified to redirect `cache_dir()`/`path()` and to create nothing.

## Risks / FIXME(port) list

No `FIXME(port)` marker was needed - nothing numeric was touched. Residual
risks, file:line, one line each:

- `engine_vr/erosion_vr.py:232` - `barrier_compute_to_read` now names only
  the storage images the dispatch wrote; sampler-bound images rest in
  SHADER_READ_ONLY and must never be named with `oldLayout=GENERAL`. First
  run must confirm no validation complaint survives (probe 2).
- `engine_vr/passes_vr.py:417-447` - `to_sampler_layout` maps a layout it
  does not recognise to `BOTTOM_OF_PIPE/0`; if `gl_vr` ever leaves an image
  in a layout outside {GENERAL, SHADER_READ_ONLY, TRANSFER_SRC,
  TRANSFER_DST} that barrier is under-specified (gl_vr today only uses those
  four: :436, :881/:899, :929, :1300/:1357).
- `engine_vr/passes_vr.py:365-391` - one submission per band: correctness is
  exact (same queue, wait between bands), but the 8192^2 stage now does 5
  submits per pass instead of one command stream; if the first run is
  submission-bound the fix is a per-band UBO/push-constant path, not a
  layout change.
- `engine_vr/erosion_vr.py:310-317` - `write_h` re-uploads the height image
  (`gl_vr` has no in-place image write), so each level switch round-trips
  2 x res^2 x 8 B through the staging buffer. Bandwidth, not correctness.
- `engine_vr/passes_vr.py:271-289` - `destroy_target` deliberately does what
  the P2 host does not (frees attachments/framebuffers/render passes),
  because bake targets reach ~1 GiB; if `Host` ever caches a target it will
  see a freed handle. Today it does not cache targets (only staging buffers).
- `engine_vr/bake_vr.py:711-728` - `bake_ao_far` binds the *same* image as
  both `uH` and `uHFar` (exactly as the original did with one texture object
  on two units); first run should confirm MoltenVK is happy with two
  descriptors over one image.
- `engine_vr/features_vr.py:111-143` - `build_plan()` uses an empty feature
  dict (`uStreamN=0`, zero `uStream`) so the plan stays pure data; the real
  values come from `place()`/`_uniform_fields` at run time.
- `engine_vr/build_p3.py` (whole file) - re-running it regenerates the copy
  regions; hand edits inside `# ==== BEGIN/END ENGINE COPIES ====` are lost
  by design.
- `engine_vr/erosion_vr.py:1558 (original `shade_rgb`)` - deliberately not
  copied: the original has a real bug there (`nx`/`nz` undefined, ruff F821
  in `engine/erosion.py:1558`) and it is CLI preview code, not part of the
  bake chain. `save_previews`/`_print_metrics`/`_main` are likewise out of
  scope for the drivers.

## Open risks: first-run probes, in order

Ranked by what can silently produce a wrong answer rather than an error.

1. **rg32f/r32f storage-image writes + coherent RMW through
   MoltenVK** (RUNBOOK 7 D6 - explicitly UNVERIFIED). CORRECTED per
   review_p3.md N1: there are NO atomic ops in any shader - imgAcc
   (rg32f) is neighbor-gather imageLoad + per-thread coherent imageStore
   (flow_only comp :97/:144). imgTrack (rg32f, coherent load/store), imgW/imgNet
   (r32f) plus the *sampled* re-read of a compute-written r32f in the next
   program. A wrong result here is not an exception - it is a silent flat
   flow field or a NaN height. Probe: the smallest stage in isolation
   (512^2, synthetic cone far map so nothing needs the cache), which runs
   fill -> accumulate -> hill -> read_state exactly as `erode()` does:

   ```
   cd /path/to/procedural-alpine-lake && ALPINE_CACHE=/tmp/p3_probe .venv/bin/python -c "
   import numpy as np
   from engine_vr import erosion_vr as ev
   g = np.mgrid[-16384.0:16384.0:4.0, -16384.0:16384.0:4.0]
   hfar = np.maximum(0.0, 300.0 - 0.06 * np.hypot(g[0], g[1])).astype('float32')
   out = ev.run_erosion(hfar=hfar, cell_m=4.0, stages=[ev.STAGES[0]], save=False)
   h, fl = out['height'], out['flow']
   print('finite', bool(np.isfinite(h).all()), 'h', float(h.min()), float(h.max()),
         'flow_abs_sum', float(np.abs(fl).sum()))
   "
   ```

   Pass = finite heights, the cone still recognisably eroded, and a flow field
   whose absolute sum is > 0 and finite. Anything else stops the port.
2. **Layout/validation chain across the compute<->sample round trip** (the
   `write_descriptors` freeze + review_p2 N10 sampled-read barrier). Probe:
   one `erosion_fill` at 512^2 on uploaded data, then read the storage image
   back (TRANSFER_SRC) and re-bind it as a sampler for the next program.
   Watch `VK_LAYER_KHRONOS_validation` output for layout mismatch and the
   "binding an image whose layout differs from the write-time layout" class
   of error.
3. **Bake target lifetime + readback fidelity at the big resolutions**: an
   8192^2 RGBA32_SFLOAT target is 1.07 GiB; `destroy_target` frees it per
   pass, and `read_image` must return the engine's row order (no flip).
   Probe: the analytic far bake alone (`bake_height`, tile 256) into
   `h_far_raw.npy`, then compare a 64x64 patch against a CPU evaluation of
   the analytic height field at the same world coordinates, plus a
   `cache_final/h_far_raw.npy` shape/dtype check against
   `cache_schema.spec("h_far_raw")`.
4. **Double descriptor for one image** in `bake_ao_far` (`uH` == `uHFar`),
   and the float16 downconvert on those two saves.
5. **Array UBO members on a real device**: `uStream` (96xvec4, stride 16,
   `uStreamN` at 1600) and `uTrail` (32xvec4, `uTrailN` at 2096). The
   offsets are verified on CPU above; the device probe is a `bake_forest`
   pass with 96 stream points and a visual/numeric diff of the forest mask
   against the GL build's output.
6. **Per-band submit cost at 8192^2** (submission-bound regression, not a
   correctness issue): time the hillslope stage and compare against the GL
   cadence before deciding whether bands need to be batched per submission.

## Notes for Phase 4

- Import-safe: `import engine_vr.bake_vr` /`erosion_vr`/`features_vr` /  `cache_schema` never touch Vulkan; all heavy work is behind `run_all` /
  `run_eroded_bake` / `run_erosion` / `run_carve`, each of which takes an
  optional `host=` so one `Host` (and one gpu slot) can serve a whole chain.
- `run_eroded_bake()` is the `tools/bake_final.py` equivalent: bake_heights
  with the VR `erode` callback, then `bake_ao` and `bake_world_maps` as in
  `bake_all()`.
- `cache_schema.missing()` is the readiness probe Phase 4 wants; it defaults
  to the four files `engine/render.py:123` requires.
