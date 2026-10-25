# PHASE 11 - animated shot rendering (flow test: shot "skim", mono equirect, camera + water move)

## GOAL
Prove the video pipeline end-to-end for ONE film shot: render an animated
equirect clip along the ORIGINAL skim shot camera path (engine/shots_film.py
shot index 2; engine/camera_path.py:65 shot_camera(shot, t) supplies
pos/target/roll per frame) with water animation (uTime already flows via
vr.py _look_fields, vr.py:467). Deliver CODE ONLY - the flow-test render
(2048x1024@1spp, 12 fps, full shot duration) is run by GC after merge.

## IMPORT-OVER-COPY (RUNBOOK §1) - the whole phase is orchestration:
- Build the shot list with the ENGINE module: engine.shots_film.build(world)
  with a small VR world shim exposing what build() touches (world.height /
  world.water via the VR bake cache accessors already used by vr.py
  _vp_skim / pano.py). Read shots_film.build source for the exact surface;
  do NOT reimplement keyframes, path search, easing or handheld noise.
- Camera per frame: camera_path.shot_camera(shot, t) -> pos, target, roll.
  Feed the EXISTING vr.py camera construction (the one that builds uCamRot
  from pos/target + roll) - animation = making pos/target/roll t-dependent
  instead of VP-table constants. FOV: equirect ignores fov (360x180 fixed).
- Time: t_frame = t0 + frame/fps mapped the way film.py advances shot time
  (read film.py frame loop); uTime advances with wall-clock seconds of the
  film (water + clouds animate; handheld noise deterministic per shot).
- CLI (all FIXME(p11)): vr.py --shot NAME --animate [--fps N] [--dur SEC]
  [--start-frame K] --frames-dir DIR  (+ keep every existing static flag
  working; --viewpoint path untouched). Mono first; stereo flag may reuse
  the same loop (per-eye offset already in pano.py) but flow test is MONO.
- Batch: one process, N frames, per-frame PNG named frame_%05d.png;
  RESUME = skip existing files; write tmp/p11_shot_probe.json BEFORE any
  GPU use: {shot, dur_s, fps_default, frame_count, t0, t1, pos_range,
  target_range} from CPU-only build+shot_camera sampling.
- Encode: verify package_vr.py frame-sequence path (--pattern/--start-
  number/--video) can consume the frames; if its argv builder can't, add a
  printed ffmpeg sequence one-liner to the phase11 report (package_vr print
  only, never spawn - existing rule).

## VERIFICATION without real renders (GPU CAP: single frames, <=256x128,
## <=6 frames total, env block as usual):
- frame A at shot-local t=0.15*dur vs frame B at t=0.85*dur: (i) mean RGB
  differs (camera moved), (ii) a water-band sub-window differs beyond
  jitter-only floor (water animates) - controls: same t twice ->
  byte-identical (determinism), B&W of a STATIC viewpoint render vs its
  animated frame must differ ONLY via camera. Stats to
  tmp/p11_frame_probe.json.
- determinism: rerun same frame -> identical bytes.
## ACCEPTANCE (GC runs the flow test after merge - your code must pass):
- 2048x1024@1spp, fps 12, dur = skim shot length exactly; resume-safe loop
  (GC will kill/restart once mid-run); output dir + manifest; encode argv
  printed; clip shows camera travel + animated water (GC eyes frames).
## GPU RUN CAPS (this phase): <=256x128, <=6 runs, <=20 s each. The 2048
## flow test is GC's. No git. Proxy https://<your-proxy-host>:<your-proxy-port>.
## Writable: vr.py, engine_vr/ (new modules ok), port/reports/phase11.md,
## port/gates/gate_p11.py (bugfix only), engine_vr/tmp/.
## Report port/reports/phase11.md: 1 shot-build + shim surface (exact
## attributes consumed), 2 time model (film.py parity statement + why),
## 3 frame probe table (A/B/controls), 4 resume semantics, 5 encode path,
## 6 residuals (stereo untested, roll convention, fov ignored).
## GATE: .venv/bin/python port/gates/gate_p11.py - static: FIXME(p11)
## markers on modified files; p11_shot_probe.json + p11_frame_probe.json
## present with pass flags; determinism flag true; py_compile vr.py +
## new modules; no engine/ diffs expected (report if any).