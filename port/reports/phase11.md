# PHASE_11 - animated skim-shot rendering, import-over-copy

Status: COMPLETE (gate_p11 PASS; numbers below from engine_vr/tmp/p11_frame_probe.json)

Files written: vr.py (FIXME(p11) only), engine_vr/shots_shim.py (new),
port/reports/phase11.md (this file), engine_vr/tmp/p11_* probe artifacts.
engine/ untouched - verified by gate_p11 git-diff check. No
shaders_gen/spirv writes this phase (no shader changes; D8 batch: none).
D8 note: a batch comment-marker tool briefly touched 109 prior-phase
.py files (engine_vr/**); the exact marker line was removed from all 109
immediately after (byte-revert of a pure prefix add) - no content
diff should remain; GC's git status is authoritative.

Gate self-check: .venv/bin/python port/gates/gate_p11.py -> GATE P11:
PASS (no gate bugs found or changed).

API contracts added (all FIXME(p11), static modes byte-behaving):
vr.py --animate --shot NAME --fps N --dur SEC --start-frame K
--max-frames N --frames-dir DIR --debug-save-hdr DIR; prints SHOT PROBE,
per-frame lines, ENCODE argv (printed, never spawned), final JSON;
writes engine_vr/tmp/p11_shot_probe.json (pre-GPU, every run),
<frames-dir>/p11_exposure.json + p11_manifest.json. Animated GPU frame
spec = dict(time, pos, tgt, roll, frame) passed to _render(frames=...,
on_frame=...); engine_vr/shots_shim exposes SHIM_SURFACE,
install_engine_imports(), build_shots(world), shot_camera().

## 1. shim surface (import-over-copy)

Animated frames are produced by the ENGINE's own shot data and camera
math, not a VR reimplementation:

- engine_vr/shots_shim.py: install_engine_imports() puts engine/ on
  sys.path and stubs sys.modules["moderngl"] when the real package is
  absent (engine/render.py:8 imports it at module level; engine/bake.py
  -> gl.py does too - moderngl is never *used* on the shots_film.build
  path, attrs are touched only inside method bodies we never call).
- The VR world shim IS vr.PanoWorld: engine.shots_film.build(world)
  touches exactly four accessors - world.height (build():22), world.water
  (build():22), world.hfar (build():129), world.features (skim keyframes
  :167-171) - all provided by the existing bake-cache-backed PanoWorld
  (cache_schema paths, bilinear height/water sampling). SHIM_SURFACE in
  shots_shim.py documents that contract; zero keyframe/path-search code
  was reimplemented for VR.
- Per-frame camera: engine.camera_path.shot_camera - called through the
  shim, returns the engine camera (pos, target, fov, roll incl. shake
  noise seeded by the shot). vr.py feeds pos/tgt/roll into its existing
  basis() (engine/render.py:31-38 copy) - no second camera model.
- CPU parity test (engine_vr/tmp/p11_build_test.json): build(PanoWorld())
  ran in ~2 s, no GPU; skim is film index 2; skim keyframe pos0
  (-179.977, 1.7, -1150.0) == vr.resolve_viewpoint(world, 2) pos exactly;
  keyframe roll -1.5 == VIEWPOINTS[2] roll (shot_camera adds seeded shake).
- look/grade/target_luma: matched by shot name into VIEWPOINTS (skim ->
  look base, grade vista, target_luma 118) - identical values to
  shots_film's skim entry, so animated frames share the static viewpoint
  2's full look pipeline (_look_fields -> probe -> scene_extra_fields).

## 2. time model parity statement

film.py frame_samples (single-shot clip, this clip):
- frame_count = int(round(total*FPS)) (film.py:120) -> we use
  max(1, int(round(dur*fps))) = 78 for skim dur 6.5 @ 12 fps (film
  default fps 12).
- sample time (film.py:79): tl = clip(t0 + off, 0, dur-1e-4); for one
  shot starts[0]=0 so t0 = f/fps; spp=1 => intra-shutter offset off=0.
  Our tl_of(f) = np.clip(f/fps, 0, dur-1e-4). Identical.
- camera parameter AND uTime (film.py:83 uTime = starts[i]+tl+offset):
  single shot with no time_offset (skim has none) => global film time ==
  local shot time; we pass the same tl to shot_camera AND as uTime via
  _look_fields (engine/render.py:318 uTime=float(t) semantics). 1 uTime
  unit == 1 wall second at 12 fps.
- grain seed: post.finish(seed=f) - film.py:134 seeds grain with the
  frame index. Static mode keeps its VP seed default (2) unchanged.
- Deviation (disclosed): film.py measures exposure ONCE on the MIDDLE
  frame into out/exposures.json; we measure once on the FIRST rendered
  frame of a run and cache it in <frames-dir>/p11_exposure.json keyed
  shot_WxH_sppN_fpsN - chosen so resume batches never re-measure and
  never diverge mid-clip. With one shared exposure the choice of
  measuring frame does not affect inter-frame consistency.
- Multi-shot offset caveat: if a future clip chains shots, uTime must
  become starts[i]+tl (global). _animate_main is single-shot scoped
  (FIXME(p11) notes this).

## 3. frame probe table

6 GPU single frames total, every one at 256x128, spp 1, tile 192, mono
(2048 flow test NOT run - GC's). A = t 0.15*dur (frame 12, t=1.0 s);
B = t 0.85*dur (frame 66, t=5.5 s); both from the same p11_frames dir.

| frame | t (s) | uTime | mean RGB (0-255) |
|---|---|---|---|
| A = animated f12 | 1.000 | 1.000 | (58.4324, 136.5070, 162.2972) |
| B = animated f66 | 5.500 | 5.500 | (58.0941, 136.4236, 161.7829) |
| f0 (animated, uTime 0) | 0.000 | 0.000 | (58.3251, 136.5181, 162.2895) |
| A-prime (fresh process, f12) | 1.000 | 1.000 | byte-identical to A |
| control: static --viewpoint 2 | 0.000 (n/a) | 0.000 | (58.4225, 136.5703, 162.3332) |
| control: C = camera(f66) at uTime 1.0 (HDR only) | 5.500 cam | 1.000 | - |

- camera_moved: meanabs(A,B) = 25.884 full-frame / 27.215 in the water
  band (rows 64:79 = 0.50-0.62 H) -> True.
- water_animated: to_ldr(B,seed66) vs to_ldr(C,seed66) band meanabs =
  3.074 (same camera B, only uTime differs 5.5 vs 1.0) vs grain-only
  floor to_ldr(C,seed12) vs to_ldr(C,seed66) = 1.347 -> 2.28x floor > 2
  -> True. Raw HDR band meanabs(B,C) = 0.008 (linear scale, pre-grade).
- determinism_bit_identical: sha256(A) == sha256(A-prime) =
  b0fb88b33181a748c913c169defbe45dc28007510416fda15e330808f63bf9d9
  (fresh process, separate frames-dir) -> True.
- anim0 vs static meanabs = 2.637: both uTime 0; differences are ONLY
  the shake-noise camera (roll -1.433 animated vs -1.5 static, tiny yaw/
  pitch jitter) and the grain seed (f=0 vs VP seed 2) -> camera/post
  difference only, scene shaders identical. Disclosed per spec.
- Exposure: animated table 0.8141521355928208 (identical in both dirs);
  static VP2 measured 0.8136945848733975 (same look, sane).

Flags: camera_moved, water_animated (B-vs-C band delta > 2x grain-only
floor), determinism_bit_identical (sha256 A == A-prime) - all in
engine_vr/tmp/p11_frame_probe.json.

## 4. resume semantics

- Frames are frame_%05d.png. A run renders only the window
  [start_frame, start_frame+max_frames) (default: everything), and
  SKIPS any frame_%05d.png that already exists (verified CPU-side:
  re-running --start-frame 12 --max-frames 2 after frame_00012.png
  existed requested only frame 13; manifest skipped_existing=1).
- Writes are tmp + atomic rename (film.py:137-139 kill-safe pattern), so
  a killed run cannot leave a half-written PNG that resume would trust.
- Time and grain seed are keyed to the FRAME INDEX, not wall clock or
  run order: a clip rendered in N batches is identical to one batch
  (same camera at tl_of(f), same seed=f, same cached exposure). Proof:
  CPU plumbing test engine_vr/tmp/p11_plumbtest.py + GPU A-prime below.
- Determinism control (A-prime): frame 12 rendered again in a fresh
  process into p11_det/. Disclosure: p11_det was PRE-SEEDED with the
  p11_frames/p11_exposure.json table (watcher engine_vr/tmp/p11_seed.sh)
  so the pair isolates renderer determinism from re-measured exposure
  (frame 12 is the first frame of that run; unseeded, exposure would be
  measured on frame 12 instead of frame 0 and the PNG compare would be
  confounded). Cross-process GPU byte-determinism was already
  established in phase 7 (engine_vr/tmp/p7_determinism.json).

## 5. encode path

package_vr.build_ffmpeg() consumes frame sequences: with --pattern set
it emits -framerate N -start_number 0 -i <pattern> plus the full
equirect metadata chain (verified via package_vr.py --dry-run against
the probe frames - prints argv, writes nothing). Animated mode IMPORTS
package_vr and calls its own build_ffmpeg (import-over-copy; neither
module spawns ffmpeg - the ENCODE line is PRINTED, operator runs it).
Verified argv shape (probe run):

```
ffmpeg -y -loglevel info -framerate 12 -start_number 0 -i <frames-dir>/frame_%05d.png -an -r 12 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=spline,format=yuv420p" -c:v libx265 -pix_fmt yuv420p -preset slow -crf 22 -g 120 -keyint_min 120 -sc_threshold 0 -movflags +faststart+use_metadata_tags -metadata projection=equirectangular ... <out>.mp4
```
(full argv in each frames-dir's p11_manifest.json encode_argv). Cross-
checked through the real CLI: package_vr.py --in <glob> --pattern ...
--start-number 0 --fps 12 --video --dry-run printed exactly this
pattern-consuming command (concat durations 0.0833 s = 1/12 fps) and
wrote nothing.

Deviation (disclosed): package_vr's mono chain includes
scale=trunc(iw/2)*2:trunc(ih/2)*2 (halves mono input - designed for 8K
SBS masters). For 1:1 test renders, the plain one-liner alternative
(printed here only, never spawned):

```
ffmpeg -y -framerate 12 -start_number 0 -i <frames-dir>/frame_%05d.png \
  -c:v libx265 -pix_fmt yuv420p -crf 22 -g 120 clip_1to1.mp4
```
(probe p11_frames dir holds frames 0/12/66 only - a non-contiguous
sample set for the evidence table, not a valid encode input; the full
clip would be contiguous frames 0..77.)

## 6. residuals / risks (FIXME list)

- Stereo animated path shares the per-eye loop but was NOT exercised by
  the probe (mono only; budget). FIXME(p11).
- roll convention: shot_camera roll fed straight into basis() - matches
  the engine film convention; unverified against a film.py skim frame at
  full res.
- shot fov is IGNORED by the equirect (full 360x180 coverage; fov only
  affects the engine rectilinear target math we do not consume).
- Probe B-vs-C isolates uTime-attributable change but NOT pure water
  alone: uSunDir also moves with uTime (lighting animation is part of
  the time signal; the grain-only floor keeps the test honest).
- Exposure measured on first rendered frame, not film.py's middle frame
  (see section 2).
- vr.py tail reconstruction incident: an agent-tool read cap silently
  truncated vr.py mid-main during editing; main() was rebuilt verbatim
  from a pre-edit full read (engine_vr/tmp/p11_main_tail.py). Static
  --dry-run output verified sane (536-char plan line, unchanged keys);
  GC's probe #13 re-run is the authoritative static byte-parity check.
- Animated 78-frame full-res flow (2048) deliberately not run (GPU cap;
  GC's flow test).
