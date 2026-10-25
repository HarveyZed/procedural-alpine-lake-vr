# P4 independent review — vr.py / engine_vr/pano.py / build_p4.py / pano shader chain

Reviewer: independent contract reviewer (round 4). Date of run: CPU-only review; no
device, no git, no edits outside this report. Env note (P3 lesson stands): bare
"import vulkan" fails ("Cannot find Vulkan SDK version", _vulkan.py:111); all name/signature
checks below were run after "source tools/vk/env.sh" with vulkan **1.3.275.1**
(.venv/lib/python3.12/site-packages/vulkan/__init__.py).

Artifacts: vr.py 1119 L (NEW), engine_vr/pano.py 134 L (NEW), engine_vr/build_p4.py 137 L,
bake_vr.py +80 (ensure_trees/bake_shadows), world.glsl/scene.frag VR_PANO edits, 10 pano
programs, gate_p4.py, port/reports/phase4.md.

## 1. API table verdict

Direct vulkan surface in P4-new files: **vr.py = 25 sites, pano.py = 0** (AST: pano.py
imports only __future__/numpy). All confirmed against the live module (hasattr +
inspect.signature), zero discrepancies:

| kind | sites | verdict |
| --- | --- | --- |
| functions | vkCmdCopyBufferToImage vr.py:783 — wrapper (commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions); vkCmdClearColorImage vr.py:825 — (commandBuffer, image, imageLayout, pColor, rangeCount, pRanges) | both 6 positional args match, order correct |
| struct ctors | VkBufferImageCopy :784, VkImageSubresourceLayers :785–787, VkExtent3D :788, VkImageSubresourceRange :821–823 | all names exist; kwargs used (aspectMask/mipLevel/layerCount; baseArrayLayer/baseMipLevel default 0) legal |
| constants | 15 unique VK_* bits/layouts at vr.py:770–772, 777–795, 812–833 | all exist (missing: []) |
| cffi | vkv.ffi.new("VkClearValue[1]") :819 + clr[0].color.float32 = [0,0,0,0] :820 | same P2-validated pattern as gl_vr.py:1310–1315 |
| host helpers | barrier(cmd, src_stage, dst_stage, src_access, dst_access, images=()) — calls pass 6 positionals; make_image(w, h, fmt_name, usage, initial_layout, depth=1, dims=2) — vr.py:768–772 (ww, hh, "VK_FORMAT_R8G8B8A8_UNORM", usage, UNDEFINED, depth=d, dims=3); staging(size) | signatures match call sites |

**N=25 checked, 25 confirmed, 0 API discrepancies.** The heavy lifting stays on the
P2-verified gl_vr/passes_vr machinery. Gate P4 re-run by me: PASS (its dry-runs are 2:1 and
never reach upload — see C1).

## 2. Findings (severity)

### CRITICAL
**C1 — every real render crashes before the first draw: 1-D footprint upload.**
vr.py:903–904 "fp = pano.texel_footprint(w, h)" returns shape (h,) float32 and is passed to
"pv.upload(fp, filt='nearest')". passes_vr.upload (engine_vr/passes_vr.py:166) computes
"comps = 1 if a.ndim == 2 else int(a.shape[2])" → for ndim==1 → IndexError: tuple index out
of range, reproduced in pure CPU (comps_of(footprint) → IndexError). Guaranteed first-run
failure of "vr.py" without --dry-run; gate P4 passing is fully consistent (it never
executes upload; dry-run returns in _plan). Fix (one line): pass "fp.reshape(-1, 1)" at
vr.py:904 — (h,1) yields a width-1 × height-H image, and the shader reads
"texelFetch(panoFoot, ivec2(0, int(y)), 0)" (world.glsl VR block), so the (h,1) layout is
exactly the indexing contract. (Alternative: teach upload that ndim==1 means (h,1); a
reshape inside pano.py would break texel_footprint's gate-checked (h,) contract — fix at
the call site.)

### MAJOR
**M1 — pano cone pass samples the footprint at quarter-res rows (LOD inconsistency at
W < 2H).** scene_pano_cone.frag keeps "#define uPixAng panoFootAt(gl_FragCoord.y)"
(panoFootAt = texelFetch(panoFoot, ivec2(0, int(clamp(y,0,uRes.y-1))), 0).r, uRes = FULL
res), but pass 98 draws into a quarter-res rect (vr.py:912–915, (w+3)//4 × (h+3)//4; engine
analogue engine/render.py:472–479). In the cone call graph — coneTrace :4528 →
terrainHD/rockRelief "float fp = max(dist*uPixAng, 0.02)" :302, treesSDF "float dcm =
fp/max(uPixAng,1e-6)" :1204 / "float fp = dc*uPixAng" :1335, rocksSDF "float px =
R/max(dc*uPixAng,1e-6)" :1707 — gl_FragCoord.y is quarter-res, so every texel indexes the
bottom-quarter latitude band (rows 0…H/4). The cone main() block-centre value
(fc = gl_FragCoord.xy*4.0, gPixAng = panoFootAt(fc.y)) is correct; the *macro* use inside
the SDF/LOD helpers is not. Engine semantics: uPixAng is one full-res constant for ALL
passes (engine/render.py:465 "p.set('uPixAng', 2*tan(fov/2)/h)"). Masked while the strip
footprint is constant, i.e. W ≥ 2H (2:1 → dLon=dLat; default SBS 4:1 masked too); bites any
--res with W < 2H (vr.py:1086–1087 has no ratio guard): e.g. 6144×4096 → dLon = 1.333×dLat,
cone-pass cull/shrink terms use the wrong latitude band while the trace pass uses its true
row footprint → cone/trace LOD decisions inconsistent → silently non-conservative cone →
drop/aliasing artifacts with zero errors. Not author-flagged. Fix options: pass-98-specific
"#define uPixAng panoFootAt(gl_FragCoord.y * 4.0)" (matches the fc-scaling precedent in
main()), or reject/warn W < 2H in --res and document 2:1/4:1 as the supported set.

### MINOR
- **N1 (latent contract bug) — pano.equirect_directions applies the rotation inverse.**
  pano.py:88–89 basis has rows r,u,f and returns "dirs @ basis.T" = B·d; the shader/engine
  apply R·d with R's *columns* r,u,f (engine/render.py:31–38 "rows = columns of the GLSL
  mat3"; vr.py:63–72 term-order-equivalent; gl_vr Std140Packer packs numpy row c into GLSL
  column c, gl_vr.py:253–254/311–314 — shader side verified correct). B·d ≠ Bᵀ·d for
  non-identity angles (proof: yaw=90°, texel(0,0): pano (0.3536, 0.9239, -0.1464) vs
  R@d (-0.3536, 0.9239, 0.1464)). Currently inert — vr.py never calls it (grid fully
  in-shader; stereo_pair dirs object is shared/inert vr.py:967–969) and gate_p4.py:36/48
  calls it only with (0,0,0) — but the pano.py:22–24 docstring "CPU and GPU reproduce
  identical rays" is false for nonzero yaw/pitch/roll, so any P5 CPU-vs-GPU ray check that
  uses this helper false-fails. Fix: "dirs @ basis", or drop the claim.
- **N2 (doc) — vr.py:566–574 probe note is wrong in the harmless direction.** It claims
  "The probe program (no PANO branch) does read uPixAng; it gets the equatorial footprint
  (see _render, FIXME there)". probe.frag reads NO uPixAng/uSample/uJitter/uRes (grep
  empty) and no such FIXME exists; vr.py:949–954 probe UBO = lookf ∩ probe-members +
  uProbePos = exact engine probe_light parity (engine/render.py:420–426). All pano variants
  macro-replace uPixAng, so the UBO write (vr.py:587 "uPixAng": 1.0) is a dead member —
  harmless. Delete/correct the sentence.
- **N3 (doc) — phase4.md §5 "glslc -O"**: this SDK's glslc exposes no -O flag (verified
  when re-deriving); the committed build uses none. Doc-only.
- **N4 (hygiene) — --res accepts any ratio** (vr.py:1086–1087) with no 2:1-or-wider note;
  this is what exposes M1. Fold into M1's fix.

## 3. Independent verification of the author's claims

- **T3 base-parity invariant: VERIFIED (stronger than the author's hash print).**
  /tmp/p4rev/parity.py re-derived all gen sources via glstrip_vr.build_source and
  recompiled the committed shaders_gen files (cwd=ROOT, glslc --target-env=vulkan1.2 + the
  manifest defines): **base 28/28 sha256 == engine_vr/spirv/*.spv, re-derived base/pano
  sources byte-identical, pano 10/10 hash-match, pano validation ok 10, no compile
  failures.** Base scene_trace.frag carries zero VR_PANO residue.
- **Shader-edit audit: clean.** world.glsl VR block (+20 L) sits right after "uniform float
  uPixAng;", every added line FIXME(port)-tagged; in the generated scene_pano_trace.frag
  the UBO member "float uPixAng;" (:127) precedes "#define uPixAng" (:197) — no macro
  mangling of the UBO declaration — and all four macro users (:302/:1204/:1335/:1707) come
  after the define. scene.frag cameraRay #ifdef keeps the engine lines byte-identical in
  #else.
- **Manifest ↔ SPIR-V: 10/10.** Via build_p1.decorated_pairs (canonical): every pano
  program's dis pairs == manifest bindings; panoFoot at (set0, bind12) in all 10; UBO
  (1,0) 5824 B / 47 members; sampler counts 15/16/17 == report "cone15/trace16/kind17"
  (nBind 16/17/18 minus UBO). (A "10/10 mismatch" scare during this review was my own
  double-path-prefix bug, corrected.)
- **pano.py ↔ shader algebra: exact.** lon=-π+(x+0.5)·2π/w, lat=π/2-(y+0.5)·π/h,
  d=(cos lat·sin lon, sin lat, cos lat·cos lon); 4 texels at 8192×4096 hand-derived equal
  the shader formulas under the readback flip (vr.py:1031–1034 a[::-1], copied from
  engine/render.py:505); orientation chain is robust to vertex-flip convention because the
  lat formula fixes memory-row→lat. panoFoot NEAREST + texelFetch = filter-independent
  ✓. stereo_pair: origins = cam ± 0.032·right, dirs shared ✓.
- **uExposureHint "dead in engine": VERIFIED** — declared engine/shaders/lighting.glsl:15
  only; no engine Python ever sets it. Skipping the write (zero-filled) is parity.
- **T0 fixes from review_p3: VERIFIED REAL.** build_p3.py:356–373 --check runs the full
  splice/rewrap/annotate on a tempfile.TemporaryDirectory scratch copy and diffs (no
  mutation); passes_vr.py:307–347 bake_pass try/finally with destroy_target in finally
  (:333 "N3 (review_p3)" comment); bake_vr.py:968–987 bake_shadows == engine/render.py:182–205
  (uCenter/uHalf/uRes 4096/uFarHalf/uSunDir, uH+uHFar, tile 512, RGBA32F, LINEAR+clamp);
  bake_vr.py:948–966 ensure_trees == render.py:166–180 (4096^2, tile 1024, np.save, count
  print). The three _zero4/shadow dummy binds in _trees_textures are never sampled in
  bake_trees' call graph (main: texelFetch(uForest) + terrainH + waterLevel; uSh reads
  live in sunShadowMap/D, world.glsl:330–346) → harmless.
- **VIEWPOINTS 3/9 spot-check: all exact** vs engine/shots_film.py — skim (vr.py:328–333 vs
  :164–169), reveal (:316–325 vs :146–157), peaks (:389–396 vs :259–270): constants,
  roll=-1.5, seeds 2/1/8, grades, targets all match keyframe-0.
- **Misc parity:** halton jitter vr.py:981 == engine/render.py:457 and shared across eyes
  ✓; per-sample fields dict correct; uTime=0.0 static deviation inherent; kind readback
  read_image(gbuf.images[0], rect=...)[...,1] == engine/render.py:484 comp-1 kind bits ✓;
  per-tile cone+trace in one command session == render.py:472–479; blend ONE,ONE kind
  accumulate == render.py:498–503; post usage vr.py:1040–1059 == film.py:110/134
  (suggest_exposure(hdr, target_luma=..., g=...), finish(img, g, seed=...); pano uses vp
  seed — deviation documented); post.py import-pure (dataclass/numpy/scipy only).

## 4. Barrier analysis (vs vr.py docstring table + P2 table)

_clear_images (vr.py:801–837): TOP_OF_PIPE→TRANSFER (access 0→TRANSFER_WRITE) for
UNDEFINED→TRANSFER_DST, vkCmdClearColorImage, then TRANSFER→COLOR_ATTACHMENT_OUTPUT
(WRITE→WRITE) — textbook; full-image clear correctly replaces the per-rect cleared
renderpass (P3 lesson applied; engine analogue render.py:451–452). _upload_noise3
(vr.py:756–798) mirrors the P2 upload pattern with dims=3/layerCount=1 FIXME'd; final
TRANSFER→(FRAGMENT|COMPUTE) SHADER_READ barrier ✓. Per-tile M5 auto-transitions and
read_image's from-any-layout barrier are P2-verified machinery; gbuf rests TRANSFER_SRC
(vr.py:910–911 comment, fallback documented), ct rests SRO via to_sampler_layout ✓.
**No new gaps** beyond carried P2 N10 (always-fire sample barrier) / N9.

## 5. Verdicts on the author's 5 open risks (phase4.md §9)

1. *First-run layout chain* — **AGREE.** Biggest real-device risk; probe #7 design sound.
   Addendum: probe #7 cannot even start until **C1** is fixed — upload precedes the first
   vkCmd, so the "loud error" tier is currently occupied by a CPU exception the risk list
   does not anticipate.
2. *Geometry convention stack* — **AGREE.** Each layer individually sourced correct (three
   flips + basis-column convention hand-verified here); composition risk is real and
   probes #8/#9/#11 are the right guards. Caveat: the CPU helper meant to pin this down
   (equirect_directions) is itself rotation-transposed (N1) — don't trust it for #9.
3. *post over equirect* — **AGREE.** Rectangular vignette/sky-weights on a sphere; probe
   #12 + P5 grade override is the right containment.
4. *First-run prerequisites/scale* — **AGREE.** No cache_final here; 512 MB acc + 2×512 MB
   gbuf + 448 tile readbacks/sample; probe #13 timing budget correct.
5. *Nominal-vs-searched viewpoints / dead members* — **AGREE**, documentation-level,
   FIXME(port) at vr.py:298–304 present as claimed.

The five are correctly characterized — but the round's two real defects (C1, M1) sit
*outside* the list; the risk section's "loud error" example would in fact currently be a
Python IndexError, not a validation message.

## 6. Overall verdict: **ACCEPT-WITH-FIXES**

Mandatory (before any first-run probe / P5):
1. **C1**: vr.py:904 upload fp.reshape(-1, 1) (one line; restores the shader (h,1)
   indexing contract).
2. **M1**: either a pass-98 "#define uPixAng panoFootAt(gl_FragCoord.y * 4.0)" in the cone
   variant (glstrip_vr edit → rebuild via build_p4), or hard-reject / --res-guard W < 2H and
   document the supported ratio set. Pick one; leaving both open makes any non-2:1 pano a
   silent-artifact generator.

Recommended (cheap, non-blocking):
3. N1 fix "dirs @ basis" (or docstring correction) in pano.py before P5 writes any
   CPU-vs-GPU ray probe.
4. N2 probe docstring correction; N3 §5 "-O" wording; N4 ratio note in --res help.

Everything else — SPIR-V parity (38/38 re-derived identical), manifest↔dis pairs, API
surface (25/25), copied-block semantics, T0 fixes from P3, viewpoint fidelity — verified
clean. With C1 fixed and M1 addressed, P4 is ready for device probes.

---

## 7. Re-verification of the P4 fix round (phase4.md §11) — reviewer round 2

Same discipline: CPU-only, vulkan checks under tools/vk/env.sh, no GPU, no edits.

### C1 — VERIFIED FIXED
vr.py:802-817 new upload_footstrip(pv, w, h) -> pv.upload(texel_footprint(w,h).reshape(-1, 1),
filt="nearest"); sole render call site vr.py:921. Interplay re-checked on the seam:
(h,1) -> ndim==2 -> comps=1 (passes_vr.py:165 - the author's line cite, not my :166, is
exact) -> (1,'f4') -> VK_FORMAT_R32_SFLOAT via the P2-verified FMT_BY_COMPS_DTYPE; image is
w=1 x h=H, matching texelFetch(panoFoot, ivec2(0, row), 0). test_p4_fixes monkeypatches
passes_vr.use_host + gl_vr._VK (no Host, no device) and asserts ndim/shape/dtype/format
plus a bare-1-D still-IndexErrors tripwire — suite executed by me: 30 PASS / 0 FAIL.
texel_footprint keeps its gate-checked (h,) contract (gate_p4 footprint formula check
untouched and passing).

### M1a — VERIFIED FIXED, no regressions
engine_vr/shaders/world.glsl:40-64: #if SCENE_PASS == 98 -> '#define uPixAng
panoFootAt(gl_FragCoord.y * 4.0)', #else keeps the byte-identical original; both branches
FIXME(port); whole block still inside #ifdef VR_PANO (world.glsl:32) so base programs
resolve it away (grep: no panoFoot/uPixAng-define in any base shaders_gen file). Manifest
defines confirm branch reachability: cone SCENE_PASS=98, trace=99, kinds 0..7 (all VR_PANO).
Generated sources: cone line 207 scaled; trace:197 + kinds:193-197 unscaled; define stays
after the UBO member declaration (no macro mangling). Scale semantics are exactly the
main() convention: helpers at row r use 4.0*(r+0.5) = 4r+2 = block centre fc.y — the fix
aligns the SDF/LOD terms with the pass's own gPixAng choice, including clamp headroom for
odd h ((h+3)//4 rows, clamp guards the tail).
Parity re-derived from engine/shaders via glstrip_vr.build_source after the edit: base
28/28 + pano 10/10 sha256 match committed engine_vr/spirv, generated sources byte-identical
(base and pano), spirv-val 10/10, zero compile failures. Change-set proof: my round-2
compiled artifacts (pre-fix world.glsl, identical glslc flags) vs committed engine_vr/spirv
— 38 compared, exactly one differs: scene_pano_cone.spv. Base 28 and the 9 other pano
programs are byte-identical across the edit; the cone .spv was rebuilt and manifest pairs
re-verified 10/10 against the regenerated manifest (UBO 5824 B / 47 members, panoFoot (0,12)
unchanged).

### M1b — VERIFIED FIXED
vr.py:1105-1118: guard sits immediately after --res parsing, before viewpoint lookup,
_plan and any Host/GPU work; argparse error (exit code 2), message names review_p4.md.
511x256 and 6144x4096 rejected (rc=2), 1024x1024 rejected, 256x128/512x256 accepted,
default mono/sbs branches untouched, dry-run compact-JSON line-1 shape intact (51-line
output). gate_p4 re-run: PASS (28 checks); ruff E,F clean over engine_vr/ + vr.py.

### N1 — VERIFIED FIXED (independent proof)
pano.py:89-100 now 'dirs = dirs @ basis'. My own reference (columns r,u,f applied as
R@d_local, derived from the shader, not from pano.py): max|diff| <= 2.98e-08 over
(0,0,0), (90,0,0), (37.5,-12,8), (-140,45,-20), (180,5,0) at 64x32; yaw=90 texel(0,0) =
(-0.04901, 0.99880, 0.00241) == R@d (the old code gave the sign-flipped inverse);
unit norms preserved; float32 output. stereo_pair is a dirs passthrough (unrotated), so the
fix cannot shift vr.py's (inert) eye dirs. The docstring claim is now true for nonzero
angles.

### N3 — REBUTTAL UPHELD (author right, reviewer wrong on the reason)
tools/vk/bin/glslc --help does list -O / -Os / -O0 (shaderc wrapper) — my review_p4 N3
reason ("no -O in this SDK") is false, retracted. The substantive finding stands via the
author's correction: build_p1.compile_source (build_p1.py:282-292, read line-by-line)
assembles only --target-env + -D defines + source -> -o; no -O is ever passed, and all 38
committed hashes (including my 28/28 parity baseline) derive from that unflagged command.
The §5 "glslc -O" wording was wrong; the build was never -O. Fixed in §5 by the author.

### New-defect hunt over the diff (vr.py, pano.py, world.glsl, manifest, §11)
Scope is exactly those 4 files + phase4.md (+test): find -newermt confirms no collateral
edits. No new defects found. Non-issues examined: '#if SCENE_PASS == 98' evaluates false
(not error) if SCENE_PASS were undefined, but every VR_PANO program defines it (and base
strips the block); upload_footstrip hardcodes nearest as the old inline call did; band
granularity below 2:1 is now explicitly a rejected regime, so the M1 residual class is
closed-by-policy until device probes say otherwise. Test suite stays tmp/ (not a gate);
acceptable — gate_p4 covers the contract-level checks, and the C1 upload seam is covered by
test_p4_fixes as designed.

### Re-verification verdict: **ACCEPT** — all P4 findings (C1, M1a/M1b, N1-N4) verified
fixed; 38/38 parity re-derived post-edit; change set exactly the cone triple; author's N3
rebuttal upheld against my original reason (finding itself survives). Clear to commit and
tag p4-pass; device probes #7-#13 in phase4.md §9 are the correct next stage, and probe #9
may now legitimately use equirect_directions as the CPU reference.
