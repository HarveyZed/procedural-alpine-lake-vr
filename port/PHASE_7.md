# BUG P7 — pano G-buffer mis-wire (kind passes never fire) + pano nondeterminism

## CONTEXT
First run proved the pano renders only the sky: P6 coder measured gbuf
images[0] carries fragInfo and images[1] reads ALL ZEROS, so the kind
readback vr.py:1047-1050 ('kb = read_image(gbuf.images[0]); kb = kb[...,1];
hit = unique(clip(kb,0,7))') yields {0} every tile and only
scene_pano_kind0_sky draws. Every pano artifact to date is sky-only. THIS
BLOCKS THE DELIVERABLE (RUNBOOK deliverable = images of the WORLD).

## BUG A - attachment mapping (blocker)
1. Establish the GL-era truth, READ-ONLY:
   - engine/shaders/scene.frag: 'layout(location=0) out' vs 'location=1'
     outputs of the TRACE path (fragColor0 = ? fragColor1 = ?); which
     output encodes kind in an odd component (engine/render.py:484-503
     reads the G-buffer texture with components=2 and takes [1::2], then
     dispatches kinds; quote the exact lines in your report).
   - engine/render.py + engine/gl.py: framebuffer creation order of the 2
     RTs and which texture that readback binds.
2. Compare to port: engine_vr/manifest.json scene_trace/scene_pano_trace
   programs 'outputs' array order + gl_vr render pass attachment creation
   order (gl_vr.py make_target/pass creation) + vr.py readback index
   (vr.py:1033-1050). Find where order or format flipped (suspects:
   outputs[] reversed at manifest build (tools/build_p4?); or engine wrote
   kind into output0 but VR path uploads fragInfo first; or FMT mismatch
   truncates output1).
3. Fix in the PORT layer (vr.py readback / passes_vr / build glue). Only
   touch shaders if you prove the shader output ORDER differs from the
   engine (then it is a shader-port bug, same rule: FIXME(p7)).
   shaders_gen/spirv changes must be regenerated via the documented
   build_p4 flow and reported (film 28 programs must stay byte-identical).
4. PROVE with GPU (caps below): vp0 640x320 -> set of unique kinds MUST be
   a strict superset of {0} (terrain >=1); vp2 (lake skim) MUST include the
   water kind; count shaded px per kind (fragInfo counter trick from
   phase6.md or a --debug counter flag). Re-run the voided visual metrics
   (first_run.md #8/#11): rows of a landscape pano must now differ from a
   sky-only render.

## BUG B - nondeterminism (after A)
Same command twice: mean dRGB 1.63/255 on 34% px (phase6.md). The engine
chain is deterministic by construction (halton jitter s-index, fixed tile
order, no atomics). Suspects: uninitialized acc memory (cleared once via
vkCmdClearColorImage - verify clear covers the image), stale tile residue
across samples (blend ONE,ONE per tile - verify rect mapping), garbage
cells surviving P6 guards only sometimes (nonfinite -> int cast is UB), or
MVK transient-memory behavior. Fix or bound: after fix, two identical
640x320 renders must have max|dRGB| == 0 (integer-identical PNG bytes or
document why a residual exists).

## VERIFICATION
- gate: .venv/bin/python port/gates/gate_p7.py (static: FIXME(p7) markers
  on modified files, SPIR-V parity report present w/ 28 film programs
  byte-identical, report sections, py_compile vr.py).
- ruff + py_compile on touched .py. NO git. NO 8192x4096.

## GPU RUN CAPS (same as P6)
<=640x320 any tile; ONE 2048x1024@1spp smoke allowed; <=2 min each
(2048 host-dominated ~130 s acceptable); sequential; env block
ALPINE_CACHE=$PWD/cache_final ALPINE_MAX_PROC_GB=96
ALPINE_MIN_COMMIT_GB=24 ALPINE_MAX_GPU=1; source tools/vk/env.sh in every
shell.

## REPORT port/reports/phase7.md
1. GL-era truth (quote engine lines) vs port behavior - root cause A.
2. Diff summary with path:line per change; state explicitly if shaders
   moved and why.
3. Kind-set proof per viewpoint (vp0, vp2) + shaded px per kind.
4. BUG B: root cause or bounded behavior + byte-identity evidence
   (sha256 of two renders).
5. Voided metrics re-run table (old sky-only value vs new).
6. SPIR-V parity result. Any gate_p7 bug you had to fix: disclose.

## HARD CONSTRAINTS
Writable: vr.py, engine_vr/ (incl. build glue + shaders_gen via build),
engine/shaders ONLY IF proven wrong, port/reports/phase7.md,
port/gates/gate_p7.py (bugfixes only). Everything else READ-ONLY.
No global installs. No git. Downloads via https://<your-proxy-host>:<your-proxy-port>.
