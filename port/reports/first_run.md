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
