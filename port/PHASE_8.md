# BUG P8 - user-visual bugs on delivered panos (A: tile-shaped sky holes in mountains; B: per-eye photometry mismatch)

## SYMPTOMS (user report 2026-25-10, both REPRODUCED by GC from delivered artifacts)
A. engine_vr/out/pkg_mono/pano/pano_vp0_mono.png (8192x4096, vp0, 1spp):
   rectangular transparent holes in the far mountain band through which the
   SKY shows; hole edges align exactly with the 192 px tile grid (see crop
   engine_vr/tmp/bug_mtn.png: crop (4096,1300)-(6656,2100) of the pano).
   => prime suspect: P6 cap engine/shaders/scene.frag:71
   'if (gPixAng < 9.0e-4) maxSteps = min(maxSteps, 100);' binding at
   8192 (footprint 7.67e-4): rays needing >100 steps leave through trace()'s
   out-of-steps path -> far-from-ground returns K_SKY. Legit worst case
   measured in P6 was 267 steps. Adjacent tiles differ because cone warm-up
   t0 varies per block => tile-shaped patchwork. (P6 chose cap=100 when the
   DDA/NaN guards were NOT yet in place; with the guards the higher caps
   380/700 may now survive - prove or disprove on GPU.)
B. engine_vr/out/pkg_sbs/pano/pano_vp1_sbs.png (4096x2048 per eye): LEFT is
   systematically ~20-35% DARKER than RIGHT in every latitude band,
   including pure sky (sky-top L 168.5 / R 204.1; horizon 77.8/107.4;
   ground 29.1/49.7). Identical composition - NOT parallax. An eye offset of
   0.032 m CANNOT change sky radiance: this is per-eye photometry (exposure,
   accumulation weight, sample-seed-driven dither bias, or cone-pass mixup).
   GC facts already established: vr.py computes ONE exposure e on the
   STITCHED image (vr.py:1192-1202) - so a per-eye grade would have to come
   from _render (acc normalization / per-eye scale on readback) or from the
   shaders' inputs (per-eye cone image, jitter seed, uCamRot basis).

## TASK
Bisection-first, fix-second.
1. BUG B root cause: render vp1 stereo 640x320 (fast) and dump PER-EYE HDR
   stats BEFORE _post_save (monkey-patch or --debug flag; tmp/ scripts are
   fine). If mean(L)/mean(R) != 1 already in raw HDR, bisect inputs: same
   seed for both eyes vs per-eye seed; cone pass per eye; eye transform
   (parallel projection means ONLY uCamPos differs - verify uCamRot and
   gPixAng are IDENTICAL between eyes; RUNBOOK §6). Fix the divergence in
   the port layer (FIXME(p8)). Acceptance: raw HDR mean ratio L/R within 0.5%
   AND stitched band means agree within 1% on a 1024x512 stereo render.
2. BUG A: with the P6 guards in place, re-test higher caps on the 8192
   footprint at REDUCED res first (--debug-foot-res 8192x4096, tile 64,
   640x320 - the P6 pass/fail signal): try removing the 100-cap entirely
   (engine's own 380/700 budget), then 280 if not. If the full-res footprint
   survives WITHOUT cap at 640 repro, also smoke a real 2048x1024. Fix =
   raise/remove the cap (keep the guards). If a kill STILL occurs at higher
   cap, the honest fallback: keep cap but make the out-of-steps fallback at
   pano distances shade as terrain-not-sky - that is a VISUAL-MATH change:
   DO NOT implement it yourself; document it as the remaining option and
   stop (GC decision).
   Acceptance: re-render vp0 2048x1024@1spp; crop the mountain band; NO
   tile-shaped sky holes (GC will view the crop).
3. Re-deliver is GC's job afterwards; your GPU budget ends at smoke renders.

## VERIFICATION (GC will re-run)
- gate: .venv/bin/python port/gates/gate_p8.py (static: FIXME(p8) markers on
  modified files; tmp/p8_eye_photometry.json {"lr_mean_ratio": x} with |x-1|
  < 0.01; tmp/p8_cap_test.json {"cap": ...|"none", "foot8192_repro_exit":
  0}; phase8.md sections; py_compile vr.py; SPIR-V parity file
  tmp/p8_parity.json film_identical true if any shader touched).
- write port/reports/phase8.md: 1 symptoms re-prod, 2 BUG B bisection table
  (each experiment: inputs -> L/R raw means), 3 BUG A cap ladder incl any
  device-lost timings, 4 diff list path:line, 5 acceptance re-renders
  (paths + band-stat tables), 6 residual risks.

## GPU RUN CAPS (first-run mode A, GC-authorized)
- 640x320 + 1024x512 unrestricted (spp<=1, tile<=64 for the foot repro);
  ONE 2048x1024@1spp smoke; NO full 8192 renders; each run <= 3 min;
  sequential; env block ALPINE_CACHE=$PWD/cache_final
  ALPINE_MAX_PROC_GB=96 ALPINE_MIN_COMMIT_GB=24 ALPINE_MAX_GPU=1;
  source tools/vk/env.sh in every shell; run_in_background for >2 min runs.

## HARD CONSTRAINTS
Writable: vr.py, engine_vr/ (glue + shaders via documented build flow only),
engine/shaders/ ONLY for the cap line (FIXME(p8), one-line change max),
port/reports/phase8.md, port/gates/gate_p8.py (bugfix only), engine_vr/tmp/.
No git writes. No global installs. Proxy https://<your-proxy-host>:<your-proxy-port>.
Batch-discipline D8: disclose shaders_gen/spirv windows (<90 s) to GC.
