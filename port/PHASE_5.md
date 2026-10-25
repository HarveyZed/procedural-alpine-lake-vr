# PHASE 5 — First-run protocol + VR packaging (final phase; zero GPU, static)

You are the P5 coder. Read first: port/RUNBOOK.md §1/§6/§7, every port/reports/phase{0..4}.md
+ review_p*.md (their §open-risks sections are your raw material; probe numbering is GLOBAL
and already fixed: phase3 #1-#6, phase4 #7-#13), engine_vr/cache_schema.py + gpuslot_vr.py
(ALPINE_* env + budget formulas), vr.py CLI surface, engine/gpuslot.py doc (memory lessons),
tools/render_final.py (resumable-run pattern to mirror).

## Mission

Turn four phases of accumulated runtime-deferred risk into an OPERATOR-EXECUTABLE first-run
protocol, plus the VR delivery packaging. Deliverables:

T1 port/RUN_FIRST.md — the first-run manual (this is the project's moment of truth document):
   - environment block: source tools/vk/env.sh; ALPINE_* variables table (from gpuslot_vr) with
     recommended values for THIS machine (M5 Ultra, unified memory) and what each guards;
   - ordered ladder: bake (cache_schema full chain via engine_vr build plan; resumable —
     mirror tools/render_final.py checkpoint pattern with ALPINE_CACHE) -> probe #1..#13 EXACTLY
     as numbered in the phase reports (command line, expected observation, abort criterion,
     what a failure signature means — e.g. #1 silent flat flow field = coherent RMW through
     Metal, NOT atomics (review_p3 N1); #7 validation-warning class; #9 use vr.basis NOT
     equirect_directions per review_p4 N1);
   - memory/time budget table per step (from phase4 report numbers: acc 512 MB, gbuf 2x512 MB,
     readback counts; bake estimates derived from code, marked ESTIMATE);
   - step 0: CPU-only viewpoint clearance check (see T2) before ANY pano probe;
   - failure-signature table (message/observation -> cause -> remedy), harvested verbatim-
     precise from the reports;
   - acceptance section: what 'first run succeeded' means (file outputs + checksum-free visual
     sanity vs docs/img framing note: film is perspective-fov, pano is 2:1 equirect — style
     same-source, pixels NOT comparable).
T2 engine_vr/check_viewpoints.py — pure numpy/CPU: load baked far heightmap per cache_schema,
   for each of vr.py's 9 VIEWPOINTS report ground clearance (camera_y - terrain_height at
   camera xz), flag < 5 m as buried (film used clearance SEARCH, not reproduced — review_p4
   finding); --dry-run works pre-bake by printing what it WOULD check; exit code non-zero if
   any buried (used by RUN_FIRST step 0 and gate_p5).
T3 package_vr.py at repo root — VR packaging CLI: --in glob of pano PNGs (mono + SBS from
   vr.py), --out-dir; mono equirect PNG passthrough + side/top metadata notes; SBS already
   produced by vr.py (verify aspect contract 2:1 per eye); --video: ffmpeg command builder for
   8K SBS turntable (equirect 3840x2160-per-eye or per vr.py res, -metadata projection=equirect
   minvex/viewport equivalent spherical tags, yuv420p, crf; slow yaw keyframes) — MUST support
   --dry-run printing exact ffmpeg argv WITHOUT spawning ffmpeg; no ffmpeg execution ever in
   your session.
T4 RUN_FIRST.md references package_vr.py usage + final delivery manifest (file tree: what the
   user gets: mono PNG x9, SBS x9, optional video, plus cache_final for reuse).
T5 Report port/reports/phase5.md per RUNBOOK §5 + what remains genuinely UNKNOWN until the
   user's first run (the honest limit, RUNBOOK §4).

## Boundaries

ZERO GPU / ZERO ffmpeg execution (dry-run printing only). No git, no pip. Writes ONLY:
port/RUN_FIRST.md, engine_vr/check_viewpoints.py, package_vr.py, port/reports/phase5.md,
engine_vr/tmp/ test helpers. Read-only everything else; gates untouched.
Self-check until green: .venv/bin/python port/gates/gate_p5.py + ruff E,F over engine_vr/
vr.py package_vr.py.
