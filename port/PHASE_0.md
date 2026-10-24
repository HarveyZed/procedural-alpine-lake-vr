# PHASE 0 — Backend spike: MoltenVK vs wgpu-py (GPU-exempt micro-runs)

You are the P0 coder. Read `port/RUNBOOK.md` fully first (§1 constraints, §2 why).

## Mission (one paragraph)

Produce evidence to pick ONE backend for the macOS port: **A = MoltenVK** (Python `vulkan`
bindings + GLSL→SPIR-V, GLSL suite survives ~unchanged with 430→450 headers) vs **B = wgpu-py**
(WebGPU, needs GLSL→WGSL via naga, no workspace-local naga yet). You are authorized (user
exemption, P0 only) to run TWO micro-spikes — 256×256 single frames, one compute pass + one
sphere raymarch each, seconds of GPU, ≤200 MB. Everything else stays write-only.

## Task 1 — workspace toolchain under `tools/vk/` (no GPU)

Goal: `tools/vk/bin/glslc` and `tools/vk/bin/spirv-val` execute (`--version`), plus a route to
`libMoltenVK.dylib` (+ loader if needed) recorded for later phases. Try in order, log which works:

1. **brew bottle extraction** (no `brew install`!): `brew fetch --force --retry --json=v2
   glslang spirv-tools` (proxy env!), read bottle URL from JSON, `curl` the `.tar.gz`, extract
   into `tools/vk/pkgs/`, copy binaries to `tools/vk/bin/` and dylibs they need to
   `tools/vk/lib/`. Verify with `--version` and by compiling a 5-line `.comp`
   (`glslc --target-env=vulkan1.2 -o` a probe .spv INSIDE the workspace, then `spirv-val` it).
2. **LunarG Vulkan SDK dmg** fallback: download macos-arm64 dmg from
   sdk.lunarg.com (`curl -L`), `hdiutil attach -nobrowse -readonly`, copy ONLY
   glslc/spirv-val/spirv-cross binaries + libMoltenVK + loader into tools/vk/, detach.
3. **MoltenVK GitHub release** (khronosGroup/MoltenVK `latest`): macOS dynamic framework archive
   → `tools/vk/lib/`.

Notes: `import vulkan` (PyPI bindings 1.3.275.1, already in .venv) fails until a loader/driver is
visible — that's your completion test for task 1. Preferred discovery route: write a one-file
`tools/vk/env.sh` exporting `DYLD_LIBRARY_PATH`/`VK_DRIVER_FILES` (icd json you author in
tools/vk/) sourced by scripts; fallback: `ctypes.CDLL(".../libMoltenVK.dylib")` direct-link
(MoltenVK exports Vulkan entry points) and hand it to the bindings' loader hooks. Document
exactly what worked with commands.

## Task 2 — Spike A (MoltenVK): `spike/p0_a/`

Pure-python vulkan bindings. One instance+device+queue, one compute shader (fill a storage
buffer), one fullscreen fragment pass raymarching a single lit sphere (analytic sphere, one
light, lambert+ambient, sky gradient), 256×256 RGBA8 → readback → `p0_a/sphere.ppm` (PPM P6)
+ `p0_a/buf_check.txt`. GLSL here may be `#version 450` from the start. Keep the whole host
script under ~400 lines — LOC-to-image is itself a measured metric.

## Task 3 — Spike B (wgpu-py): `spike/p0_b/`

Same scene, WGSL written by hand (hand-translation is allowed ONLY for the spike; production
conversion would be naga's job — count LOC translated + any semantic gaps you hit, that's the
conversion-risk sample). Compute pass → buffer (same fill pattern), render pass → texture →
`copy_texture_to_buffer` → `p0_b/sphere.ppm` + `buf_check.txt`. wgpu-py 0.32 API: the installed
package source in `.venv/lib/python3.12/site-packages/wgpu/` is ground truth (also
`wgpu.gui.offline` exists); do not guess APIs from memory of other versions.

## Task 4 — CPU reference + comparison

`spike/p0_ref.py` (numpy): raytrace the same sphere at 256×256 → `p0_ref.ppm`;
`spike/p0_cmp.py`: PSNR of A-vs-ref and B-vs-ref (≥26 dB expected, ≥40 = near-exact).
Sanity-check `buf_check.txt` of both spikes equal the numpy fill reference.

## Task 5 — report `port/reports/phase0.md` (§5 format) plus this REQUIRED table

For each backend: tool versions used, wall time (image out), peak RSS (`/usr/bin/time -l`),
PSNR vs CPU ref, host-code LOC, frictions hit (exact errors, how resolved), and for B a
GLSL→WGSL translation gap log (which GLSL 430 features used by the real suite look risky in
naga: image atomics? textureSize? ubo binding style? — check `engine/shaders/*.glsl` usages,
grep for `imageAtomic`, `coherent`, `layout(std140`, `gl_GlobalInvocationID`, shared mem).
Close with: your recommendation + the 3 top risks you'd hand the next phase (name the file).

## Boundaries

Writes only under `tools/vk/`, `spike/`, `port/reports/phase0.md`. No git. No `brew install`.
No installs into `.venv` beyond what's already there (numpy/pillow/wgpu/vulkan present). GPU runs
ONLY of `spike/p0_a` and `spike/p0_b` scripts at ≤256². If a task dead-ends after honest tries,
write it up — a rigorous negative result for one backend IS a passing P0 outcome. Gate
`port/gates/gate_p0.py` (static: files exist, tool `--version` exits 0, PPMs parse at 256²,
PSNR lines present, report lint) is GC-run; you may dry-check it but its re-run of glslc is the
only execution you should expect from it.
