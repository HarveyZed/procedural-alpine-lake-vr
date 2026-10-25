# BUG P6 — scene_pano_trace runaway ray → VkErrorDeviceLost at 8192x4096

## SYMPTOM (deterministic, first run 2026-10-25, Apple M5 Ultra, MoltenVK)
`vr.py --viewpoint 0 --res 8192x4096 --spp 1` (default tile 512) dies at
`host.end()` -> `vkQueueWaitIdle` -> `vulkan._vulkan.VkErrorDeviceLost`.
NOT memory: peak RSS flat 2.1 GiB; heap budget 222 GiB. NOT validation
(errors zero with VK_LAYER_KHRONOS_validation). Driver force-kills a
command buffer that overruns its GPU-time budget = classic hang.

## EVIDENCE (instrumented Host.draw wrapper, logs engine_vr/tmp/p13*.out)
- tile 512: death draw ~1267 (trace), last rect (5376,1536,192,192), t+39s.
- tile 64:  DEATH draw 8792, scene_pano_trace rect (4352,1408,64,64).
  (two concurrent instrumented runs alias to that tile; 512-run's rect may
  be cone-pass coords - treat (4352,1408)±64px as the primary suspect,
  (5376,1536) as secondary suspect.)
- tile 1024: FULL PASS (wrote /tmp/p13t.png). Coarser per-tile epsilon ->
  no runaway.
- 6144x3072 (2:1) PASS. 8192x1024 PASS. SBS --res 4096x2048 PASS.
- vp1 and vp6 at full 8192x4096: UNTESTED. Test them - the fix scope
  depends on whether it is viewpoint-specific (one hidden crevice) or
  resolution-driven (footprint epsilon in general).

## GEOMETRY MATH (derive before touching loops)
8192x4096 equirect: dLon = 360/8192 = 0.04395 deg/texel, dLat = 90/2048...
(use RUNBOOK §6: dLat = 180/4096 deg). Suspect tile (4352,1408): lon =
4352/8192*360-180 = +11.25 deg; lat = 90 - 1408/4096*180 = +28.1 deg.
Secondary (5376,1536): lon +56.25, lat +22.5. gPixAng at 28 deg lat =
dLon*cos(28) -> the smallest step footprint of any shipped resolution.

## CODE MAP (READ-ONLY originals for reference; fix in engine_vr/ only)
- engine/shaders/scene.frag  - root; march loop(s): main raymarch +
  shadow-ray loop; includes listed in README.
- engine/shaders/ground.glsl 3002L, falls.glsl 2069L, rocks.glsl 792L,
  trees.glsl 958L, water*.glsl, heightfn.glsl 760L, world.glsl (panoFootAt
  cone lookup world.glsl:50-64).
- pano mode = P4 defines; per-texel footprint strip: vr.py upload_footstrip
  (vr.py:830 area, pano.texel_footprint(w,h).reshape(-1,1)) -> R32 texture;
  cone pass samples it via panoFootAt(y*4.0) (M1 fix, world.glsl:50-64).
- shader variants compile via engine_vr manifest/P4 (see port/PHASE_4.md
  §SPIR-V + manifest.json); same frag with defines -> scene_pano_kind*.

## HYPOTHESES (rank)
1. Adaptive march under step_floor(footprint): epsilon = f(gPixAng)*k;
   epsilon -> 0 in a crevice, loop guard compares against a value that
   also shrinks -> steps never exceed epsilon -> unbounded iterations.
2. Secondary-ray runaway (reflection/refraction/shadow) that inherits the
   tiny footprint; primary loop fine.
3. Deterministic NaN: NaN comparisons defeat every exit test while steps
   stay finite-count but each sample costlier... (less likely - would
   finish, not hang).

## FIX RULES
- NO re-tuning of visual math (RUNBOOK §1). A robustness guard is allowed:
  fixed max-iteration caps are legitimate; changing step rates, noise
  scales, or geometry is NOT.
- Prefer: per-loop iteration caps + a degenerate-sample escape that
  produces the SAME color the engine produces when a march truncates
  (find the engine's existing truncation/fallback path FIRST - there is
  likely already a 'max steps -> return t/hit=false' exit; make degenerate
  cases take that path, never invent new shading).
- Every touched loop: add `// FIXME(p6): ...` line.
- If you can localize the shader function without running: add a DEBUG
  define `PANO_DEBUG_ID` that writes (loopIndex, iterCount/8) to the
  trace output color (document how), so the USER can run one 64px crop to
  identify the exact loop. Do not run it yourself beyond the P6 cap.

## VERIFICATION (static gates; you NEVER run full res)
- glslc -c -Werror every modified .frag/.glsl variant config + spirv-val
  (same driver as port/gates/, e.g. engine_vr/tmp/p4spv.py; -O parity
  check per P4 decision N3).
- SPIR-V instruction-delta report: diff opcode counts before/after per
  program; justify every non-trivial delta.
- ruff/py_compile untouched (no python change unless a new DEBUG path needs it).

## RUN LADDER (GPU exemption: <=2 min wall + <=640x320 each;
   ALL runs MUST carry ALPINE_MAX_PROC_GB=96 ALPINE_MIN_COMMIT_GB=24
   ALPINE_MAX_GPU=1 - the first-run env block; NOT yet in RUN_FIRST §2,
   noted)
1. repro: --viewpoint 0 --res 640x320 --tile 64 --spp 1
2. suspect crop (if crop support exists - check vr.py flags first) else
   the 640 render IS the smoke test
3. --viewpoint 1 --res 640x320; --viewpoint 6 --res 640x320
4. --stereo sbs --res 640x320 (per-eye)
5. --viewpoint 0 --res 2048x1024 --spp 1 --tile 128
6. --viewpoint 0 --res 1536x768 --spp 2

## REPORT port/reports/phase6.md
1. ROOT CAUSE: which loop, which shader line(s), why 8192-only, with the
   math. If you cannot prove it, say which hypotheses survive and why.
2. DIFF SUMMARY of engine/shaders/ (path:line each).
3. Verification: glslc/spirv-val pass table + instruction-delta notes.
4. Run ladder results (exit, sec, PNG dims, exposure from vr.py JSON line).
5. Residual risk: does an iteration cap change IMAGE CONTENT on the
   9 reference shots at film res? Argue from the cap value (pick cap >=
   the engine's own max-steps if one exists; document the value).
6. Any FIXME(p6) left in shaders: list.

## HARD CONSTRAINTS
- Only engine/shaders/*.frag|*.glsl (+ engine_vr/ + vr.py) writable;
  port/RUN_FIRST.md + port/reports/phase6.md yours. engine/*.py other
  than the above: READ-ONLY. NO git commands.
- NEVER run 8192x4096 or full res; never more than 2 concurrent GPU procs;
  no global installs; proxy https://<your-proxy-host>:<your-proxy-port> if downloads needed.
- Python via .venv/bin/python; source tools/vk/env.sh in EVERY shell.
