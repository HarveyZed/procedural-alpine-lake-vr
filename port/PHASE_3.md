# PHASE 3 — Bake + erosion + features drivers on the VR host (zero GPU, static phase)

You are the P3 coder. Read first: port/RUNBOOK.md (§1/§4/§7), engine_vr/manifest.json (bake_* +
erosion_* programs, uniform_block member tables, storage-image bindings), port/reports/phase2.md
addendum + port/reports/review_p2.md §6 (host is ACCEPTed; contract notes for you: acquire()
before Host.open(); UBO writes only at queue-idle; read_2d row order keeps engine flip site),
engine_vr/gl_vr.py public API (Host.open/build/write_descriptors/set_uniforms(dispatch APIs),
gpuslot_vr.acquire/compile_guard).

## Mission

Port the THREE bake-side drivers that turn the baked cache into existence, driving them through
engine_vr/gl_vr.py instead of moderngl, with the exact same output contract as the original
pipeline: **engine/bake.py (783 L) → engine_vr/bake_vr.py**, **engine/erosion.py (1687 L) →
engine_vr/erosion_vr.py**, **engine/features.py (396 L) → engine_vr/features_vr.py**. The npy
cache files they write MUST keep the exact file names, shapes, dtypes and coordinate conventions
the original cache_final/*.npy have (P4 renders them; schema drift = silent garbage). Reuse
engine/treemap_port.py (already pure numpy) as-is by import if its deps allow.

## Tasks

T1 Map every GPU touchpoint in the three engine drivers (framebuffer programs, uniforms,
image bindings, dispatch sizes, ping-pong buffers, readbacks) to the gl_vr API; where an
original uniform is a bare name, your call-site key MUST equal the manifest uniform_block member
name for that program (they were kept identical by design — verify per program, list coverage
in the report: every manifest member either set or explicitly skipped with rationale).
T2 Faithfulness: NO algorithm/constant changes. Prefer importing pure helpers from engine/ when
a module is moderngl-free (check imports first); otherwise copy the helper into engine_vr/ with
a "copied from engine/<file>:<lines>" header. Numeric constants get a single source: mirror via
copy + source annotation, never retype silently.
T3 Cache schema module engine_vr/cache_schema.py: file names, shapes, dtypes, res ladder,
path helpers (ALPINE_CACHE env honored like tools/render_final.py) — P4 depends on this module.
T4 build_plan() convention (gate-checked, import-safe): each of the three modules exposes
build_plan() -> dict {pass_name: {"program": manifest name, "uniforms": {member names...},
"storage_images": [...], "res": int-or-ladder, "dispatch": optional}} — pure data, no GPU calls;
gate_p3 cross-validates every name against manifest.json.
T5 Erosion dispatch discipline (from engine/erosion.py docstring, keep exact): multi-res ladder
512→4096, 16x16 tiles, ≤2048-row dispatches (TDR-era rule — keep as a correctness invariant),
fill→accum→hill ordering incl. accum FLOW_ONLY variant usage exactly as original.
T6 Report port/reports/phase3.md per RUNBOOK §5 + per-pass mapping table (engine line → vr call)
+ UBO coverage list + §open-risks (what you'd probe first at first-run).

## Boundaries

ZERO GPU (gl_vr must never be opened; your modules import-safe: heavy work only inside
explicit run_* functions). Writes ONLY engine_vr/** + port/reports/phase3.md. No git, no pip.
Self-check loop until green: .venv/bin/python port/gates/gate_p3.py; ruff E,F over engine_vr/
must stay clean (do not reformat P2 files — they are review-ACCEPTed as-is).
