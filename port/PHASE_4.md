# PHASE 4 — Pano renderer: vr.py + equirect raygen + post + PNG/SBS (zero GPU, static phase)

You are the P4 coder. Read first: port/RUNBOOK.md §1 + §6 (equirect spec — BINDING) + §7,
port/reports/review_p3.md (N2/N3 are yours), engine_vr/passes_vr.py + cache_schema.py
(your host surface), engine/render.py + engine/post.py + engine/shots_film.py (the original
renderer chain: split-pass raymarch cone/trace/kinds → G-buffer shade → post grade),
engine_vr/manifest.json (scene_* programs), port/reports/phase3.md (probe ordering context).

## Mission

Build the VR panorama renderer end to end, statically verified:

T0 Carry-overs: fix N2 (build_p3.py --check must validate BEFORE mutating) and N3
   (bake_pass try/finally releasing targets up to 1.07 GiB) per review_p3.md.
T1 bake_trees + bake_shadow drivers (P3 deferred them deliberately): extend bake_vr so
   cache_schema's trees.npy and shadow product are produced; coverage rules as before.
T2 engine_vr/pano.py — PURE CPU, no imports from vulkan/engine: equirect_directions(w,h,
   yaw,pitch,roll)->(h,w,3) f32 per RUNBOOK §6 (lon [-pi,pi) across W, lat +pi/2..-pi/2 top-down,
   d=(cos lat*sin lon, sin lat, cos lat*cos lon) rotated by camera basis), texel_footprint(w,h)->
   (h,) f32 = max(dLat, dLon*cos(lat)) (the gPixAng replacement), stereo_pair(cam_pos, dirs,
camera_right) -> (origin_l, origin_r, dirs): origins = cam_pos +/- 0.032 m along
camera_right, PARALLEL projection - both eyes share the SAME dirs array unchanged
(only origins shift); same-seed rule docstring for noise. Pure functions, numerically
   testable — gate_p4 will hammer them.
T3 VR_PANO shader hook (the ONLY sanctioned shader edit): scene raymarch needs per-texel step
   epsilon. Mechanism is your choice (e.g. #ifdef VR_PANO branch sampling an R32F footprint
   image, or band-constant uniform with quantified error), BUT hard invariants, gate-enforced:
   (a) recompiling every P1 program WITHOUT the PANO define must produce SPIR-V identical to
   the committed engine_vr/spirv/ artifacts (default-path zero drift); (b) every added line
   carries FIXME(port); (c) no visual constant changes anywhere; (d) document mechanism +
   error analysis in the report.
T4 vr.py entry at repo root: argparse CLI — --viewpoint <0..8> (VIEWPOINTS table: annotated
   copy of the 9 shot positions/targets from engine/shots_film.py, constants with source
   headers), --stereo mono|sbs (SBS = per-eye width halves, IED per §6), --res WxH (default
   8192x4096 mono / 8192x2048 per-eye sbs), --spp, --out, --cache (ALPINE_CACHE), --dry-run
   (print plan JSON, NEVER touch vulkan — gate-checked via subprocess probe), then full render:
   open host (gpuslot first), load cache via cache_schema, tiled raymarch through
   passes_vr/gl_vr, post via engine/post.py import (pure numpy), PNG out via Pillow.
   Mono and both SBS eyes share one host, one bake, one seed.
T5 Report port/reports/phase4.md per RUNBOOK §5 + §open-risks (probe order for first run,
   continuing phase3.md numbering) + LOC table.

## Boundaries

ZERO GPU (dry-run only; Host.open never called by you). No git, no pip. Writes: engine_vr/**,
vr.py, port/reports/phase4.md. engine/ audio/ docs/ READ-ONLY. Do not reformat accepted
P2/P3 files outside T0's two fixes. Self-check until green: .venv/bin/python
port/gates/gate_p4.py + ruff E,F over engine_vr/ and vr.py.
