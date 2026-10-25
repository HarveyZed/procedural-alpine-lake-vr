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
