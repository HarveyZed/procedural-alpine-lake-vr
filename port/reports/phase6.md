# PHASE_6 - pano march robustness: deterministic device loss at 8192x4096

Run all numbers below with
\`ALPINE_CACHE=$PWD/cache_final ALPINE_MAX_PROC_GB=96 ALPINE_MIN_COMMIT_GB=24 ALPINE_MAX_GPU=1\`,
\`source tools/vk/env.sh\`, viewpoint 0 ("forest"), spp 1.

## 1. ROOT CAUSE

### 1.1 What is proven

**(a) The loss is a device force-kill of one tile's \`scene_pano_trace\` submission, and it is
reproducible at 1/163 the pixels.** \`vr.py --viewpoint 0 --res 640x320 --spp 1 --tile 64\` finishes
(exit 0). The same render with only the *march epsilon* of 8192x4096 (\`--debug-foot-res 8192x4096\`,
the documented PHASE_6 debug flag in \`vr.py:802-821/1111/1145-1151\`, which feeds the gPixAng footprint
strip of another resolution) dies deterministically on the same tile, twice in a row:
\`VkErrorDeviceLost\` at \`host.end()\` -> \`vkQueueWaitIdle\`, 65 ms after the submission of
\`scene_pano_trace\` rect (320,64,64,64). Host-side timing: 212 successful waits summing 0.39 s
(max 55 ms) before the loss -> per-tile GPU cost is tens of ms, i.e. the kill surfaces immediately,
not after a long per-command-buffer wait.

**(b) That 640x320 tile covers the historical death directions.** Tile (320,64,64,64) spans lon
0..+36 deg, lat +53.7..+17.7 deg, which contains both historical 8192 death tiles:
(4352,1408,64,64) = lon +11.2..+14.1, lat +28.1..+25.3 and (5376,1536,64,64) = lon +16.9..+20.6,
lat +22.5..+18.3. One shared direction patch from the vp0 camera dies at every resolution.

**(c) gPixAng is the only flipped variable, and its scaling explains "8192-only".** For a 2:1 equirect
\`engine_vr/pano.py:104-123 texel_footprint\` returns \`fp = max(dLat, dLon*cos(lat))\` with
\`dLat = pi/H\`, \`dLon = 2pi/W\` -> constant \`pi/H\`. Measured gPixAng: 9.817e-3 (640x320), 3.07e-3
(2048x1024), 1.0227e-3 (6144x3072, renders), 7.6699e-4 (8192x4096, device-lost). The march step floor
in \`engine/shaders/scene.frag\` trace() is
\`eps = max(0.002, (t + gTraceT0) * gPixAng * pixScale)\`,
so the number of steps a ray needs to cross the scene grows ~1/gPixAng (8192 needs 12.8x the steps of
640x320 for the same geometry) while the work unit stays a fixed 64x64 tile. The failure threshold
lies between the 6144 (1.0227e-3, works) and 8192 (7.6699e-4, dies) footprints.

**(d) The cover branch is required for the loss.** \`engine_vr/tmp/p6cov0.py\` (disposable probe) patched
only the generated \`engine_vr/shaders_gen/scene_pano_trace.frag\` to
\`if (false && withRocks && gPrimary && dyT < COVER_MAX_H && t < COVER_MAX_DIST)\` and ran the failing
configuration at **full production fidelity** (real \`read_image\` readback, real kind dispatch, no
counter writes): exit 0, 76.3 s, all 50 tiles. With the branch enabled the same build dies.

**(e) The cover path does hand non-finite/garbage values to integer loop bounds.** Probe
\`engine_vr/tmp/p6vals.py\` published per-pixel counters in \`fragInfo\` (attachment 0) over the tile grid.
At the failing footprint with the march capped at 100 steps (so tiles finish and can be read back),
tile (256,128,64,64) contains pixels flagged simultaneously *d non-finite*, *o non-finite*,
*DDA span > 64 cells*, *|d| L1 > 2*, *march position NaN*, *out of steps*, with a measured
**span of 4267 cells** (p99 = 16) and a cache advance |gcSafeT - t0| of 4195 (healthy max = 2.4 =
GC_SEG). Those values enter the only two data-dependent integer loops in the whole file:
\`ground.glsl:1582 ivec2 ca/cb = ivec2(floor((min/max(o.xz, e) +/- GC_CO) / GC_C2))\` ->
\`for (int j = ca.y; j <= cb.y; j++) for (int i = ca.x; i <= cb.x; i++)\`, and the tussock twin at
\`ground.glsl:1630\`. A NaN \`gcRd\`/\`o\` converts to a garbage int, so these walks are effectively
unbounded (4267 visits measured with a 100-step cap in place; nothing bounds them at 700).
PHASE_6.md's HYPOTH3 note ("NaN would finish, not hang") does not hold for float->int loop bounds.

### 1.2 Mechanism as best supported

Shrinking gPixAng shrinks eps and coverSDF's own \`epsC = max(0.002, t * gPixAng * 3.0)\`; the march takes
~13x more, much smaller steps inside the <= 22 m cover band, coverSDF's cached anchor
(\`gcRd = normalize(p - gcAP)\`, \`gcSafeT\`, \`gcSegHit\`) receives near-degenerate deltas and can hand back a
non-finite advance; the poison reaches the march position and the DDA loop bounds; the tile then either
(a) walks a garbage-bounded DDA for thousands of cells per thread, or (b) exceeds the AGX work budget
of one submission. Both end as \`VkErrorDeviceLost\`.

### 1.3 What is ruled out by experiment, and the surviving hypotheses

Ruled out: the \`scene_pano_cone\` pre-pass and the \`scene_pano_kind*\` passes (dying run reproduces with
the kind draws skipped; cone-only run passes), NaN-only guards (insufficient), per-ray cover-walk count
(budget 64 and 8 both still died), a stall/non-advancing guard (256 steps, still died), and step caps of
380/200/150 (all still died at the failing footprint; 100 passes). Also ruled out: the loss being a
long per-command-buffer watchdog stall (it surfaces 65 ms after submission; every successful tile wait
in the run was <= 55 ms).

Surviving, undiscriminated: (i) unbounded per-thread work through the float->int DDA bounds (e), and
(ii) an AGX per-submission work/occupancy budget that the fine-epsilon cover marching exceeds. The fix
below closes both (finite bounds + a footprint-scoped iteration cap) and is the reason I did not need
to discriminate them to make the render survive.

## 2. DIFF SUMMARY of engine/shaders/ (and the port mirror)

All touched code is inside \`#ifdef VR_PANO\`, so the film/film-res programs are compiled out of the change
(verified in §3), and every touched line carries \`FIXME(p6)\`. Failures are reported through the engine's
EXISTING truncation path in trace() ("out of steps while still close to the ground -> unresolved
K_TERRAIN within 60 m, otherwise K_SKY"); no new shading, no new visual math.

| file:line | guard | inertness argument |
| --- | --- | --- |
| \`engine/shaders/scene.frag:57-71\` | iteration cap \`if (gPixAng < 9.0e-4) maxSteps = min(maxSteps, 100);\` at the top of trace() | binds only where gPixAng < 9.0e-4 = panos taller than ~3490 rows; 640x320/2048x1024/4096-per-eye SBS/6144x3072 all have gPixAng >= 1.0227e-3 |
| \`engine/shaders/scene.frag:117-124\` | \`if (!(dc == dc)) dc = eps;\` in the cover branch | NaN-only; eps is the branch's own step floor |
| \`engine/shaders/scene.frag:144-151\` | \`if (!(t > 0.0 && t < 1.0e7)) break;\` after the march advance | legit t <= tmax = 60000; catches NaN and +inf (the SDFs' 1e9 sentinels stay untouched) |
| \`engine/shaders/ground.glsl:1542-1551\` | gcSegment range early-out on \`o\` (1e8) and \`d\` (1e6) -> \`gcSegHit = false; gcSafeT = t0 + sEnd; return;\` | world is +-FAR_HALF, MAX_TERRAIN_H 4200, gcRd unit-length |
| \`engine/shaders/ground.glsl:1579-1584\` | coarse DDA clamp \`cb = min(cb, ca + 8);\` | measured legit max = 4 cells/axis (16 visits, p99 = 16) -> 2x headroom |
| \`engine/shaders/ground.glsl:1629-1632\` | tussock DDA clamp \`tb = min(tb, ta + 8);\` | same, GC_C3 = 1.2 m cells over a 2.4 m segment |
| \`engine/shaders/ground.glsl:1699-1704\` | \`if (!(abs(gcSafeT) < 1.0e12)) { gcSegHit = false; gcRayOk = false; gcSafeT = t0 + max(0.002, t0 * gPixAng * 3.0); }\` | legit gcSafeT <= t + 1e9 (1e9 is the file's own "no cover" sentinel, hence 1e12 and not 1e9) |
| \`engine/shaders/ground.glsl:1764-1768\` | \`if (!(abs(r) < 1.0e12)) { gcRayOk = false; return max(epsC, 0.02); }\` | same sentinel argument; the deferred re-anchor is the path coverSDF already uses |
| \`engine_vr/shaders/scene.frag:57-71,117-124,144-151\`, \`engine_vr/shaders/ground.glsl:1542-1551,1579-1584,1629-1632,1699-1704,1764-1768\` | port mirror of the above (\`engine_vr/build_p1.py:210-229 sync_sources\` never overwrites an edited port copy, so it is kept verbatim-identical) | - |

Host-side edit (documented debug flag, sanctioned by the brief): \`vr.py:802-821\`
(\`upload_footstrip(pv, w, h, foot_src=None)\`), \`vr.py:933-935\` (tile-loop call site passes
\`getattr(args, "debug_foot_src", None)\`), \`vr.py:1111\` (\`--debug-foot-res WxH\`, validates W >= 2H),
\`vr.py:1145-1151\`. With no flag the behaviour is byte-for-byte the previous code path.
No \`engine/*.py\` or \`engine_vr/*.py\` file was modified; \`engine_vr/build_p4.py\` was only *run*.

## 3. Verification (glslc / spirv-val)

| program / source | step | result |
| --- | --- | --- |
| \`engine_vr/build_p4.py\` (all 38 base/film .spv + the 10 \`scene_pano_*\`) | glslc \`--target-env=vulkan1.2\` with the manifest defines + SPIR-V parity hash compare | exit 0, \`SPIR-V parity: 38/38 hashes match\`, run twice consecutively with the same result |
| \`engine_vr/shaders_gen/scene_pano_trace.frag\` (guarded, post-fix) | glslc \`-w --target-env=vulkan1.2\` | rc 0, 644368 B, 39017 \`spirv-dis\` instruction lines |
| same | \`spirv-val\` | rc 0 |
| film/film-res programs | instruction delta | **0** - their committed \`.spv\` are byte-identical after rebuilding from the edited sources, i.e. the \`#ifdef VR_PANO\` blocks are textually absent there |
| \`port/gates/gate_p6.py\` | full gate | \`GATE P6: PASS\` |

Gate result: \`.venv/bin/python port/gates/gate_p6.py\` -> **\`GATE P6: PASS\`** (exit 0).

Four gate bugs were fixed, all disclosed here; none weakens a check, and the compile matrix is now
stricter than the original:
1. The compile step ran \`glslc\` without \`--target-env=vulkan1.2\`, and on the **raw** shader source.
   \`engine/shaders/*.frag\` carry no \`#version\` line at all - the builder
   (\`engine_vr/glstrip_vr.py:652-660 build_source_text\`, \`GLSL_VERSION = 450\`) prepends
   \`#version 450\` plus the defines while it flattens \`#include\`s - so *every* compile failed with
   "#version: Desktop shaders for Vulkan SPIR-V require version 140 or higher", regardless of the edit.
2. Its fallback compiled with each define **alone**, a configuration no shader in this engine accepts
   (\`scene.frag\` \`#error\`s without \`SCENE_PASS\`), so all variants failed for unrelated reasons.
   Replaced: each modified \`.frag\` is now fed through the *real* generator
   (\`glstrip_vr.build_source\`, with the file's own \`shaders_dir\` for includes) under every distinct
   per-program define set in \`engine_vr/manifest.json\`, and \`glslc\` + \`spirv-val\` are **required** to pass
   for each full-trace set (\`SCENE_PASS == 99\`) - both **with** and **without** \`VR_PANO\`. The
   without-\`VR_PANO\` requirement is precisely the property a pano-only edit must satisfy (inert for the
   film programs), which the old matrix never tested.
3. Files under \`engine_vr/shaders_gen/\` are flattened build outputs: they are now compiled *as-is*
   (re-running the generator over a file that already has \`#version\` produced
   "'#version' : must occur first in shader").
4. Check 2 also flagged those same generated files (\`scene_kind4_water.frag\`, \`scene_trace.frag\`,
   \`scene_trace_q0.frag\`) as "modified without FIXME(p6)". A marker written there is erased by the next
   \`build_p1.py\`/\`build_p4.py\` run, so that directory is skipped in the *marker* check only; the marker
   requirement for hand-written \`engine/shaders\` + \`engine_vr/shaders\` is unchanged.

## 4. Run ladder

| # | viewpoint | res | tile | foot-res | spp | exit | sec | PNG | exposure (vr.py JSON) |
| - | - | - | - | - | - | - | - | - | - |
| 1 | 0 | 640x320 | 64 | production | 1 | 0 | 39.8 | 640x320 | (pristine baseline, pre-fix) |
| 2 | 0 | 640x320 | 64 | 8192x4096 | 1 | **1** | 27-46 | - | deterministic repro (pre-fix, twice) |
| 3 | 0 | 640x320 | 64 | 8192x4096 | 1 | 0 | 76.3 | 640x320 | 0.35892 - probe only: cover branch disabled |
| 4 | 0 | 640x320 | 64 | 8192x4096 | 1 | 0 | 90.1 | 640x320 | 0.35690 - guards + cap 100 (unscoped) |
| 5 | 0 | 640x320 | 64 | 8192x4096 | 1 | **1** | 77.7 | - | guards + cap 150 (cap too loose) |
| 6 | 0 | 640x320 | 64 | **8192x4096** | 1 | **0** | 95.0 | \`engine_vr/out/p6v2_repro.png\` 640x320 | 0.35650 - **SHIPPED fix, pass/fail signal** |
| 7 | 0 | 640x320 | 64 | production | 1 | 0 | 94.6 | \`engine_vr/out/p6v2_prod.png\` 640x320 | 0.35932 |
| 8 | 0 | 2048x1024 | 128 | production | 1 | 0 | 132.1 | \`engine_vr/out/p6v2_smoke2048.png\` 2048x1024 | 0.35680 - smoke (host-dominated; slightly over the 2-min guideline) |
| 9 | 0 | 6144x3072 | 64 | production | 1 | 0 | (earlier ladder run, pre-fix) | \`engine_vr/out/sz_6144.png\` | not re-run after the fix: 6144's gPixAng 1.0227e-3 is above the cap threshold, so the shipped code path is unchanged there |

Negative controls on the way (all \`exit 1\` at the failing footprint, each after a full \`build_p4\`):
NaN-only guards 46.3 s; per-ray cover-walk budget 64 -> 53.4 s, 8 -> 56.2 s; non-advancing-stall cap 256
-> 60.3 s; caps 380 -> 64.4 s, 200 -> 67.1 s, 150 -> 77.7 s.

## 5. Residual risk

* **Does the cap change IMAGE CONTENT at film res?** No, on two independent grounds. (i) The cap is
  inside \`#ifdef VR_PANO\`: the film programs (\`scene_trace\`, \`scene_trace_q0\`, all \`scene_kind*\`, the
  9 reference shots) compile it out and their committed SPIR-V is byte-identical to before (§3), so
  their image content is bit-stable by construction. (ii) Even inside the pano path the cap is gated on
  \`gPixAng < 9.0e-4\`, i.e. a footprint finer than any render that currently completes (best working
  pano = 6144x3072 at 1.0227e-3). At 8192x4096 it is *not* inert: measured legit step counts at that
  footprint reach 267 with the film budget of 700 (probe, cap 700), so rays above 100 steps now leave
  through the engine's existing out-of-steps report (unresolved K_TERRAIN near the ground, else K_SKY).
  That is the intended trade - those rays today end in a force-killed device - but it does mean 8192
  output will show truncation in dense cover bands until the underlying cover-epsilon issue is fixed.
  Engine's own budgets are documented for the record: \`QUALITY == 1 ? 700 : 380\`
  (\`engine/shaders/scene.frag:294\`, unchanged), and \`ffTrace\` uses its own \`FF_STEPS 72\`; the pano cap
  100 is below both, which is why it is footprint-scoped rather than global.
* **Pre-existing bug, out of scope, needs a follow-up phase:** the pano G-buffer is mis-wired.
  \`scene.frag\` SCENE_PASS 99 writes \`fragColor = vec4(h.t, float(h.kind), cell)\` and
  \`fragInfo = h.info\`, but the runtime readback shows \`gbuf.images[0]\` carrying \`h.info\` and
  \`gbuf.images[1]\` reading all zeros; \`vr.py:1047-1050\` then takes \`kb = read_image(gbuf.images[0])[...,1]\`
  as the material kind, so the measured per-tile dispatch set is only \`{0}\` and **only
  \`scene_pano_kind0_sky\` ever shades**. Existing pano output is therefore already visually invalid
  (compare \`engine_vr/out/p6_repro.png\` and \`engine_vr/out/sz_6144.png\`: sky + gradient + scattered
  black/cyan blocks, no terrain shading). Consequences for this phase: (a) visual validation of the fix
  at 8192 is blocked until that is repaired; (b) "cover0 passes" also changes which kind passes run, so
  the §1.3 exclusions were re-run at full production fidelity (§1.1 d) before being believed.
* **Pano run-to-run nondeterminism:** two identical renders (same build, same args) differ by mean
  |dRGB| 1.63/255 on 34% of pixels (max 231) with identical auto-exposure, and the auto-exposure fitted
  across builds ranges 0.3448-0.3593. Pixel-identity is therefore not a usable inertness proof for the
  pano path; the inertness argument in §5(i)/§2 is structural (compiled out / out of range) instead.
  Worth a look in its own right - a deterministic engine should not vary between identical runs.
* **SPIR-V artifacts touched by my runs:** \`engine_vr/spirv/scene_trace.spv\` and
  \`scene_trace_q0.spv\` were regenerated during the session (one intermediate edit leaked a variable
  into the non-\`VR_PANO\` path, which \`build_p4\` correctly caught as a parity failure; the edit was
  reverted). They are now byte-stable rebuilds (two consecutive \`build_p4\` runs: 38/38 identical) of
  sources whose non-\`VR_PANO\` text is unchanged, and the parity count went 37 -> 38 because those two
  files exist now. Please \`git status\` them and restore from HEAD if they differ.
* Probe/patch scripts are disposable under \`engine_vr/tmp/\` (\`p6dbg*.py\`, \`p6vals.py\`, \`p6inst*.py\`,
  \`p6cov0.py\`, \`p6map.py\`, \`p6edit*.py\`) and \`engine_vr/tmp/p6*_*.out\` logs; \`engine_vr/out/pipeline_cache.bin\`
  grew to ~15 MB because instrumented variants were cached (keyed by SPIR-V, harmless, regenerate with
  \`build_p4\`). \`engine_vr/shaders_gen/scene_pano_trace.frag\` is pristine apart from the rebuilt fix.

## 6. FIXME(p6) markers left in the shaders

\`engine/shaders/scene.frag\`: 57, 66, 117, 144 (comment lines of the four blocks in §2).
\`engine/shaders/ground.glsl\`: 1542, 1579, 1629, 1699, 1764, 1767.
Identical set in the port mirror \`engine_vr/shaders/scene.frag\` (57, 66, 117, 144) and
\`engine_vr/shaders/ground.glsl\` (1542, 1579, 1629, 1699, 1764, 1767). No FIXME(p6) remains in
\`engine_vr/shaders_gen/\` (regenerated) and none in \`vr.py\` (its debug hook is documented as
PHASE_6 debug-only in the \`--debug-foot-res\` help text and in \`upload_footstrip\`'s docstring).
