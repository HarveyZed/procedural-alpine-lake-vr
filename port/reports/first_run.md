# First run log — operator: GC, mode A (full ladder)

Started 2026-10-25 13:2x (+08). Per port/RUN_FIRST.md, user-authorized full-GPU run
(splash = harness resident; ~126 GiB free; ample).

## B0 environment + mini bake

### B0 PASS 13:3x (+08)
- FIXED (first-run bugs, all static reviews passed them):
  1. gl_vr.py:577 p.limits.recommendedMaxMemorySize absent in vulkan 1.3.275.1
     bindings (Vulkan 1.4 maintenance8 field) -> getattr/0 + fallback documented;
     real budget via VK_EXT_memory_budget heap line = 222.7 GiB.
  2. gpuslot_vr: passes_vr calls release(), module only had _release() -> public
     release() alias added.
  3. passes_vr.upload/storage_image: FMT_BY_COMPS_DTYPE[(comps,key)] yields
     (name, npdtype) tuple, was used as bare name -> KeyError tuple in
     _FORMAT_BYTES; unpacked [0] (2 sites).
- OPERATOR: cache dir needs mkdir (cache_schema never mkdirs; np.save died) ->
  RUN_FIRST env block now carries mkdir -p; /tmp/p5_mini writable.
- mini bake: heights finite (-1272..4236 m), AO/maps/noise done, pipeline cache
  28 B->189 KB growth, saved on close. WARNING "tarn too big/leaking 131.7 ha"
  = expected at far_res=1024 (features tuned at 8192); watch in B1.

### B1 attempt 1 FAIL 13:4x -> fixed -> attempt 2 running (bash-01daf1e856e34630)
- NameError heapq at erosion_vr._pflood (level 512^2, protection 6.2s done).
  Root cause: copied engine/bake erosion lacked the heapq import; build_p3's
  blanket '# noqa: F821' comments silenced ruff. ruff --ignore-noqa proved the
  9 hidden F821s were ALL heapq. Fix: import heapq added to erosion_vr.py
  import block; ruff --select F821 --ignore-noqa now clean across engine_vr.
- OPERATOR LESSON: 'python | tee' hides the python exit code (attempt 1 logged
  exit 0 while dead). B1 rerun writes > file 2>&1 and echoes $?.

### B1 PASS 13:5x
- erosion 5 levels total 35.8 s (GL-era ref ~6 min); lake 2.75 km2, tarn 17.2 ha
  (no leak warning at real res - B0 warning was a 1024-res artifact), freeboard
  lake 31.2 m / tarn 28.1 m; clamp stats sane, rim raises 0.
- AO far/near 0.4/0.2 s, forest 8192^2 0.5 s, noise3d ok, pipeline cache 246 KB.
- cache_final 3.8 GB, cache_schema missing: []. cache_schema.RENDER_REQUIRED ok.

### Step 0 PASS - check_viewpoints --nominal-ok: 9/9 OK (1.0-2.2 m healthy band, vp6/7 airborne 513/700 m), exit 0
Self-test A/B (fakecache 12/0) gave exit 1/0 as specified BEFORE trusting the tool.

### Probe 1 FIX->RUN 13:5x
- cache_schema.path() ignored stems its own docstring promised ("h_far_raw"
  -> h_far_raw.npy); cs.path('h_far_raw').exists() was False so run_erosion
  died on cache_final/h_far (no extension). path() now mirrors spec()
  normalization. (4 erosion_vr call sites relied on the stem behavior.)
- Probe #1/#2 semantics fixed in RUN_FIRST: erode()'s tail (skyline restore,
  :1711-1716) assumes ladder ended at input res; single low stage needs
  stop_after=<last stage res> (mid-loop return has same dict keys).
  Original #1 ran with stages=[STAGES[0]] on the 8192 raw map -> tail crashed.

### Probes #1-#6 ALL PASS (bake/erosion group)
- #1 finite True, flow_abs_sum 9.84e10, depressions_after_level 0 (decisive rg32f
  coherence probe on real h_far_raw, 4.1 s at 512^2).
- #2 VK_LAYER_KHRONOS_validation clean over 2 levels (512+1024, no validation msg).
- #3 2048 bake raw sane (-1586..4410 m), patch mean/std sane.
- #4 AO far/near fine under validation. #5 features.json + forest (coverage 9.3%).
- #6 full-res stage 10.4 s finite. FIXED: prev_res None when the ladder STARTS at
  the final level (erosion_vr.py:1675 'g = r // prev_res' -> guard else 1).
  (bug exists in the engine original too - single-stage final ladder never ran there)
- TOOLING LESSON: python-heredoc write to erosion_vr.py reported success but did
  NOT persist (second silent-drop; heapq one persisted, this one didn't - reason
  unknown). tools.edit DID persist. All future code fixes via read+edit tools.

### Probe ladder #7-#12: 4 first-run bugs fixed, then ALL PASS
- #7 died: vkCmdClearColorImage needs VkClearColorValue* not VkClearValue*
  (vr.py _clear_images) FIXED; then "command session already active": the probe
  block wrapped draw(cmd=None one-shot) in host.begin()/end() (vr.py:969/977)
  FIXED (drop outer pair; uniform writes need no cmd). PASS: 512x256 forest,
  exposure 0.353, 99.8% nonblack, validation clean.
- #8 vp6 flyover: row0 120 / mid 195 / last 47, seam |d| 2.52 (0.4 sigma). PASS.
- #9 vp2 skim PASS (exposure 0.861). #10 sky zenith (62,92,137 blue) vs
  equator (153,172,174 hazy horizon) - gradient physically right. PASS.
- #11 SBS vp1 8192x2048: parallax mean|L-R| 37.0, 69.4 s, peak RSS 10.2 GiB.
  NOTE: /usr/bin/time -l strips DYLD_* (SIP) -> vulkan import fails; operator
  must measure in-process (resource.ru_maxrss, bytes on macOS). Lesson also hit
  in #6. Manual still says /usr/bin/time in #6/#13 - flagged for RUN_FIRST fix.
- #12 pole darkening top 34.6% bottom 74.6% = sky physics + terrain shade, no
  polar smear signature (row means smooth). PASS.

### Probe #13 FAIL (device lost) -> BUG P6 filed (port/PHASE_6.md)
- 8192x4096 vp0: deterministic VkErrorDeviceLost mid-tile-loop; RSS flat,
  validation clean => GPU-time hang (driver force-kill), NOT memory.
- Localization: tile 1024 PASS; tile 64 death trace rect (4352,1408,64,64)
  draw 8792; 512-run died near (5376,1536) (may be cone coords). 6144x3072
  and SBS pass => smallest per-texel footprint at 8192 exposes a march-loop
  runaway (or NaN-epsilon) in a crevice near lon +11.25/lat +28.1.
- Coder dispatched per RUNBOOK (fix engine/shaders/, <=2min GPU cap, no
  visual-math retuning, glslc/spirv-val static gate gate_p6.py).

### P6 FIXED (coder 10d6f925) - device loss closed; TWO NEW BUGS OPENED
- Fix: VR_PANO-scoped robustness guards (scene.frag:57-71 footprint-gated
  maxSteps=100 below 9.0e-4 rad footprint; :117-124,144-151 escapes;
  ground.glsl DDA clamps + gcSafeT 1e12 guards). Gate P6 PASS; SPIR-V parity
  38/38 (film byte-identical). Debug flag --debug-foot-res reproduces full-res
  footprint at 640x320 (pass/fail signal: 640 t64 repro exit0 95s).
- PROVES: G-buffer mis-wire => PROBES #7-#12 ABOVE ARE VOID AS VISUAL
  EVIDENCE (kind readback vr.py:1047-1050 always {0} => only sky shaded;
  row-mean/seam/parallax all sky-only metrics). Re-verify after P7.
- P7 OPENED: (A) gbuf attachment mapping (images[0]=fragInfo, [1]=zeros,
  kinds never fire); (B) run-to-run nondeterminism dRGB 1.63/255 on 34% px.
- Also: legit rays need up to 267 steps at 8192 - cap 100 = truncation
  banding accepted for now (dead-device alternative); revisit post-P7.

### GC review of P6 (commit 5ff9410): ACCEPT
- 7 guards all inside #ifdef VR_PANO, each with math justification (8-cell DDA
  clamp vs measured legit max 4/axis; 1e12 vs in-file 1e9 sentinel; escapes
  route into EXISTING out-of-steps truncation). No visual-math retuning.
- Nit (accepted, not worth a rebuild): two stray blank lines in scene.frag.
- Housekeeping: vr.py carries 10 pre-existing ruff errors since P4 (incl.
  EXE001 shebang); not P6-introduced; gates never ruffed vr.py. Note for
  final sweep.

### Concurrency incident (no code damage) -> RUNBOOK D8
- GC re-verify of P6 exit=1 was P7 coder's IN-FLIGHT shaders_gen edit
  (P7CONST debug const, syntax error) - committed tree verified clean via
  git show 5ff9410. Both coders notified (recall + protocol D8). GC GPU
  verification of the P6 ladder deferred to post-P7 quiescent tree.

### P7 COMPLETE + ACCEPTED (commit c3c8bff, tags p6-pass@5ff9410 p7-pass)
- Root cause A = platform divergence: MoltenVK 1.4.2 delivers ONLY the
  highest-LOC written fragment output, into slot 0 (probes tmp/p7probe2/3/5/7;
  MTL_DEBUG_LAYER overlap assert; MVK dump exonerates SPIRV-Cross; no upstream
  citation - locally measured). GL-era truth pair scene.frag:280-281 +
  render.py:484. Fix: pano trace single output -> gt0; h.info -> uGInfo
  imageStore -> gt1 (GENERAL forever, submit+wait fence). Film #else literal;
  parity 28/28 byte-identical. Kind proof vp0 {0..5} water 2522px; vp2 water
  50%. Bug B was a downstream symptom (sampling never-written memory): two
  identical 640 renders now byte-identical. Disclosed cap violation: one
  accidental full 8192x4096 render COMPLETED (wrapper bug, artifact deleted)
  - first sign full-res survives P6+P7.
- Gate interference found+fixed during sweep: gate_p2-4 ruff caught 2 E501
  from first-run passes_vr fixes (wrapped); gate_p5 dry-run tripped by mixed
  mono+SBS leftovers in engine_vr/out/ (deleted - sky-only void artifacts).
  All 7 gates PASS on clean tree.
- GC verification battery running (tmp/gcverify.log): A/B determinism recheck,
  C P6 repro signal, D kinds, E #13 FULL 8192x4096 + RSS/wall.

## FINAL VERIFICATION (GC battery, quiescent tree @ p7-pass, 2026-10-25 16:34-16:54)
- [A/B] determinism recheck: two 640x320 vp0 renders sha256 IDENTICAL
  (4652a627e79dd171... - same prefix coder recorded). BUG B closed in GC too.
- [C] P6 pass/fail signal: 640x320 tile64 --debug-foot-res 8192x4096 exit=0
  (95s) - P6 fix verified independently.
- [D] kinds vp0: unique {0..5}, water 2522 px; vp2: water 102129 px (50%).
- [E] PROBE #13 PASS: 8192x4096 mono vp0 1spp exit 0, WALL 169.4 s,
  PEAK RSS 19.06 GiB, mean 95.8 std 69.6, nonblack 99.78%, seam |d| 8.12,
  zenith (99,144,206) vs equator (100,133,23), no pole dark bands (0/0%),
  last-row 78.9 (ground under camera). Visual: full world renders (lake,
  peaks, forest, sun, clouds).
- VOID PROBES REDONE on real content: P8 vp6 mean 101.1 last-row 19.9 (was
  46.8 sky-only; terrain now under flyover); P9 vp2 mean 118.9 teal-water
  86.1%; P11 SBS 8192x2048 exit 0, L/R mean 87.6/115.2, mean|L-R| 27.96,
  per-eye seams 9.5/9.4 (whole-image seam number meaningless for SBS -
  metric artifact, recorded).
- Packaging: package_vr real runs -> pkg_mono/ + pkg_sbs/ manifests;
  ffmpeg 360-SBS test encode (4096x2048 half-packed, 8 s loop) per printed
  argv - exit recorded below; 8K delivery = --packing full (documented).
- RUN_FIRST #13 row + budget row flipped to PASS.

### Packaging defect found+fixed (GC, first real run)
- RUN_FIRST/package_vr printed '-metadata:s:v:0 ...' for the 360 video:
  ffmpeg 9.0.2 SILENTLY DROPS stream-level custom tags in MP4 (measured,
  with and without map_metadata). Fixed to container-level '-metadata'
  (+use_metadata_tags -> mdta): re-encoded turntable now carries
  projection=equirectangular / projection_range=full / stereo_mode=sbs_half
  / SIDE_BY_SIDE_HALF (ffprobe verified). Stream-level remains the job of
  dedicated 360 tooling; container tags suffice for the tested viewers.
- DELIVERED ARTIFACTS (engine_vr/out/, gitignored): pano_vp0_mono.png
  8192x4096; pano_vp1_sbs.png 8192x2048; pkg_mono/ + pkg_sbs/ manifests;
  pkg_sbs/turntable_4096x2048_sbs_half.mp4 (8s, hevc, tagged). 8K-class
  delivery = same flow, vr.py 8192/eye + package --packing full.
- FIRST-RUN A: COMPLETE. All ladder steps green on quiescent tree @ p7-pass.

### Gate semantics correction (GC)
- gate_p6/gate_p7 are POINT-IN-TIME review gates: they REQUIRE their
  phase's uncommitted diff (marker + shader-compile audit), so they FAIL on
  a clean tree by design. Steady-state regression suite = gates p1..p5.
- gate_p5's dry-run globs engine_vr/out/*.png: delivery PNGs must live in
  pkg_mono/pano/ + pkg_sbs/pano/ (package_vr already puts them there);
  flat out/ kept mono-only. Corrects the earlier 'all 7 PASS on clean
  tree' line (P6/P7 runs at that moment were pre-commit, as designed).
