# PHASE 2 — Vulkan host layer `engine_vr/gl_vr.py` (zero GPU, static phase)

You are the P2 coder. Read port/RUNBOOK.md (§1, §4, §7 D6/D7), port/PHASE_1.md outcomes via
engine_vr/manifest.json, port/reports/phase1.md (esp. §glslc strictness list + §risks:
coherent image RMW ordering and sRGB semantics stay UNVERIFIED until first-run — design so
those can be probed early), spike/p0_a/run.py (hardware-proven Vulkan boilerplate: env.sh,
instance/device/queue, memory alloc, readback).

## Mission

Write the host layer that Phase 3/4 will drive: manifest-driven MoltenVK context, descriptors,
pipelines, passes, tiling, readback, and a memory watchdog — all code written and statically
gated, NEVER executed on GPU in this phase. Module-level import must work with tools/vk env
sourced but create NO device unless an explicit build()/run entrypoint is called.

## Tasks

T1 `engine_vr/gl_vr.py` core: instance (portability enable), device select (prefer integrated/
unified heap, log chosen device + VK_EXT_memory_budget numbers at runtime), queue, command
pool/buffer helpers, one-shot submit+wait helper, pipeline cache with file persistence
(out/pipeline_cache.bin) — huge raymarch PSO compiles must survive restarts.
T2 Shader build: `glslc --target-env=vulkan1.2` via subprocess from tools/vk/bin over
engine_vr/shaders_gen sources listed in manifest (cache SPIR-V under engine_vr/spirv/, reuse
build_p1 outputs if present), then vkCreateShaderModule paths. No runtime spirv translation
assumptions beyond MoltenVK's.
T3 Descriptor layer: build per-program VkDescriptorSetLayouts from manifest sets (0 sampler,
1 the single std140 _p1_pass UBO, 3 storage image; binding numbers from manifest ONLY — never
hardcoded). Host API: write_descriptors(prog, samplers=..., ubo_fields=..., storage_images=...)
where ubo_fields is packed by a std140 packer driven by programs[].uniform_block.members
(name/type/offset; assert packer size == manifest bytes at build; vec3→16B rule).
T4 Pipelines: compute + graphics (color attachment count from manifest outputs — scene_trace
has 2, RGBA16F both; others 1); pipeline layout per set present.
T5 Pass orchestration mirroring engine/render.py semantics: render passes (clear/load ops per
engine usage), image layout transitions, vkCmdPipelineBarrier mappings for each GL sync point
you find in render.py/bake.py/erosion.py dispatch order (keep a GL→VK barrier table in the
module docstring — it is contractual for GC review), tiling (viewport sub-rect + world-offset
uniform fields per manifest member), multi-sample HDR accumulation (add/avg via compute pass),
readback to numpy via staging buffer (byteswapped None, row padding rules of copy_buffer_to_image
reverse).
T6 `engine_vr/gpuslot_vr.py`: single-flight GPU slot + budget watchdog per engine/gpuslot.py
semantics (same ALPINE_* env names; on macOS also cap by 0.75×recommendedMaxMemorySize or
ALPINE_MAX_PROC_GB override; no TDR here — document differences).
T7 P5-ladder flags: argparse-compatible knobs (--res, --tiles, --spp, --cache-dir) surfaced as
parameters; document (do NOT implement CLI main) — vr.py CLI arrives in P4.

## Boundaries & checks

Writes: engine_vr/** (new files; do not rewrite P1 modules — import them), port/reports/
phase2.md. No git. Zero GPU execution (no vkCreateDevice outside guarded functions); running
glslc/spirv-val is allowed. .venv/lib/python3.12/site-packages/vulkan/ is the API ground truth —
cite file:line per API in your report's contract table. Static self-check:
.venv/bin/python port/gates/gate_p2.py must PASS (py_compile, ruff E/F, module-level call
blacklist, no-moderngl, manifest-only bindings scan, read-only trees untouched).
Report per RUNBOOK §5 + §barrier-table + §unverified-risks (name the probe you would run first
at first-run time).
