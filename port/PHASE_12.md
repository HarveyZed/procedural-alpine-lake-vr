# PHASE 12 - BUG: animated frames have NO motion blur (film.py parity gap)

## SYMPTOM (user, HMD): 'pixel级颤抖, 山也在抖, 特别是边缘' at every spp;
unaffected by playback rate, unaffected by denoise post. This is classic
per-frame sub-pixel camera stepping with a ZERO-duration shutter.
## ROOT CAUSE (proven by reading the contract owner):
film.py:75-81 distributes the spp samples ACROSS the shutter:
    shut = args.shutter / FPS          # film.py:36 default 0.5 (180 deg)
    off  = ((s+0.5)/spp - 0.5) * shut if spp > 1 else 0.0
    tl   = clip(t0 + off, 0, dur-1e-4)
    per-sample Camera shot_camera(shot, tl) AND per-sample uTime.
vr.py animate mode (P11) renders ONE camera at ONE tl per frame and
jitters only pixel-space per sample (vr.py:1061-1068) -> no shutter, no
motion blur. The film-era frames HAD blur at every final render.
## FIX - replicate film.py EXACTLY, import-over-math:
Per frame f and sample s: tl_s = clip(t0_f + ((s+0.5)/spp-0.5)*shutter/fps,
0, dur-1e-4) (film.py:77-78); per-sample Camera = camera_path.shot_camera
(shot, tl_s) - recompute per-sample uCamPos/uCamTarget/uCamRot AND uTime
(= global starts-based time like film times[]; single-shot clip => tl_s,
same value the camera consumed). Sample jitter (halton jx/jy, seed s)
stays EXACTLY as today. Cost: UNCHANGED (film.py spends the same spp
budget; the blur is free at equal spp - keep it that way, do NOT multiply
renders). Mono first: restructure so the per-sample loop feeds
host.draw accum with per-sample fields (today the eye loop owns samples;
see vr.py:1043-1073 + FIXME(p8) per-eye clear invariant - stereo needs the
same per-sample origin from pano.stereo_pair(cam.pos_s...); implement
stereo per-sample too but mono is the accepted path).
- --shutter passthrough flag (default 0.5, film.py:36). spp=1 => off=0
  (film.py:77) - static renders and the existing static path MUST stay
  byte-identical (frames=None branch untouched; frames + spp1 also
  untouched semantics: off=0 reproduces P11 behavior).
## EVIDENCE (GPU cap <=256x128, <=8 frames):
E1  moving t0=3.25 vs static same-camera frame, IDENTICAL sample jitter
    (force off=0 for both): static sharp, moving shows edge energy drop on
    a mountain-silhouette crop (report edge-gradient ratio, expect < 0.9).
E2  same-camera --shutter 0 moving frame == P11 behavior (byte-identical).
E3  determinism rerun.
## ACCEPTANCE (GC runs): 1 s window 4096x2048@12spp 24fps (film final
spp), --shutter 0.5, shake ON (parity), encode crf16; user compares in HMD
against spp24_1s.mp4. Static probe #13 unchanged (GC regression).
## GPU RUN CAPS: <=256x128, <=8 renders, <=60 s total GPU. No git. No
engine/ edits. env block + env.sh as usual; proxy <your-proxy-host>:<your-proxy-port>.
## Writable: vr.py, port/reports/phase12.md, engine_vr/tmp/, gate_p12 bugfix.
## Report: root cause echo, film.py line parity table, per-sample restructure
notes (accum invariant!), E1-E3 table, stereo per-sample status, residuals.
## GATE: .venv/bin/python port/gates/gate_p12.py - static: FIXME(p12)
markers; p12_evidence.json flags (edge_blur_ratio, shutter0_parity,
determinism); py_compile; engine/ untouched; static-path guard: frames=None
code path diff-clean vs git HEAD outside FIXME(p12) hunks.