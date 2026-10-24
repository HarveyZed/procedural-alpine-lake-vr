# Phase 2 report

## Status
DONE — zero GPU executed (no vkCreateDevice ran, ever; all validation was
py_compile / AST / ruff / glslc-CPU / pure-python packer tests).

## Files written
| path | LOC | role |
|---|---|---|
| engine_vr/gl_vr.py | 1334 | MoltenVK host: instance/device/queue, glslc build + SPIR-V cache, std140 packer, descriptors, pipelines, render passes, tiling, readback, pipeline-cache persistence, budget hookup |
| engine_vr/gpuslot_vr.py | 452 | ADAPTED inherit: slot locks, watchdog, compile_guard on Mach VM stats + rusage, MoltenVK memory budget cap |
| engine_vr/tmp/selftest_packer_p2.py | ~160 | packer self-test (25 UBOs / 609 members), ruff-clean |
| engine_vr/tmp/wrap_e501*.py, probe_mem*.py, probe_rusage*.py | scratch | mechanical lint wrappers + memory-layout probes (all ruff-clean, zero GPU) |
| port/reports/phase2.md | — | this |

## Inherited gpuslot_vr.py — verdict: ADAPTED (structure kept, memory layer replaced)
Kept: same ALPINE_* env semantics (ALPINE_SLOTS / ALPINE_MAX_GPU /
ALPINE_MIN_COMMIT_GB / ALPINE_NO_SLOT / ALPINE_MAX_PROC_GB / ALPINE_COMPILE_GB
/ ALPINE_SLOTS/compiled markers), O_EXCL pid-lock slots with stale reclaim,
atexit release, daemon watchdog, compile_guard (MoltenVK compiles
SPIR-V->Metal inside vkCreateGraphicsPipelines/vkCreateComputePipelines, so
the compile lock still makes sense), no-TDR difference documented.
Replaced after probing this macOS (macOS 26, 16 KiB pages, 256 GiB):
1. mach host_page_size64 symbol does not exist -> was silently dead; now
   host_page_size on the host_self() handle (gpuslot_vr.py:141).
2. free/inactive counters: host_statistics64 needs the host_self() **uint32**
   handle and writes **32-bit words** into a preloaded c_uint32[24]
   (count_in < 24 fails rc=0x5); the old c_uint64-struct read gave garbage
   (gpuslot_vr.py:170 _vm_pages). Cross-checked vs /usr/bin/vm_stat: 56.92
   vs 56.41-56.53 GiB reclaimable.
3. proc_pid_rusage: public header slots 6/7 (resident/phys_footprint)
   read **zero** on this libc; slot8-slot9 (internal-purgeable) and
   ru_maxrss tracked a 512 MB dirty alloc (+512.25/+512.52 MB). Shipped
   proc_gb = max(ru_maxrss, slot8-slot9, slot7) with sanity bounds
   (gpuslot_vr.py:223).
4. budget: configure_budget(recommendedMaxMemorySize) fed by
   Host.open() (gl_vr.py:552-557); cap = min(ALPINE_MAX_PROC_GB,
   0.75 x recommended) (gpuslot_vr.py:250-270). No TDR on macOS -
   watchdog exit(3) protects the machine, not a display-reset timeout.

## Gate self-check
```
$ .venv/bin/python port/gates/gate_p2.py
PASS host modules exist (gl_vr.py, gpuslot_vr.py) - ['gl_vr.py', 'gpuslot_vr.py']
PASS py_compile gl_vr.py
PASS py_compile gpuslot_vr.py
PASS ruff E,F clean on engine_vr/
PASS no module-level device/context calls in gl_vr.py
PASS no module-level device/context calls in gpuslot_vr.py
PASS no moderngl import anywhere in engine_vr - []
PASS manifest is the binding source (gl_vr reads manifest.json)
PASS no hardcoded numeric bindings in gl_vr.py - []
PASS read-only trees untouched - []
GATE P2: PASS

$ .venv/bin/python engine_vr/tmp/selftest_packer_p2.py   # no env.sh, no vulkan import
PACKER SELFTEST: PASS - 25 UBOs, 609 members, 0 GPU calls
```
Packer self-test: every manifest uniform_block re-derived offset-by-offset
(std140), signature values verified at every member offset over full array
spans, vec3/mat3 pad lanes zero, unknown member rejected, erosion workgroup
size parsed (TS=16 -> 16x16x1).

## Deviations
1. **P1 files lint-reformatted** (build_p1.py, glstrip_vr.py,
   selfcheck_p1.py): gate_p2 requires `ruff check --select E,F engine_vr/`
   clean; the committed P1 files had 189 pre-existing E501/E401/E702.
   Changes are mechanical only (ruff format, comment wraps, implicit string
   concatenation splits, import split). **Proof of no behavior change:**
   re-running build_p1.py reproduced all 56 shaders_gen/spirv artifacts
   byte-identical (`engine_vr/tmp/p1_sha_before/after.txt`, shas all equal).
2. **manifest.json drift**: that same build_p1 re-run rewrote the manifest's
   per-program `glslc_seconds` (volatile timing field written by unmodified
   code in build_p1.py build_one). All other manifest bytes unchanged.
   Whoever runs git may `git checkout -- engine_vr/manifest.json` for a
   pristine copy (this session runs no git per RUNBOOK section 1).
3. std140 size semantics pinned to the manifest: `size_bytes` is the upload
   length (e.g. 20 for bake_height - last ivec2 end, NOT 16-rounded);
   array members own stride spans (uTrail = 32x vec4 over [16,528)), a lone
   vec3 spans 12 when a scalar follows. Packer validates all directions.
4. Padded uploads (rowPitch != w*bpp) raise instead of silently mis-copying -
   every engine format has 4 B-aligned rows; guard at gl_vr.py:777-786.
5. `apiVersion` uses the binding's own `VK_MAKE_VERSION(1,2,0)`
   (_vulkan.py:2575) - the binding exports no VK_API_1_2 constant.
6. No CLI main (T7): driver knobs are Host(cache_dir=, res=, tile=, spp=)
   + host.ensure_spirv/build/draw/dispatch/read_2d; vr.py CLI is P4.

## API contracts relied on
All line numbers are in
`.venv/lib/python3.12/site-packages/vulkan/` (vulkan 1.3.275.1; cited file
`_vulkan.py` unless noted). Every constant is imported from the module,
never re-implemented.

Instance/device/queue - vkCreateInstance:11032,
vkEnumeratePhysicalDevices:11061, vkGetPhysicalDeviceProperties:11080,
vkGetPhysicalDeviceQueueFamilyProperties:11099,
vkGetPhysicalDeviceMemoryProperties:11114,
vkGetPhysicalDeviceMemoryProperties2:14262 (T1 budget log;
VkPhysicalDeviceMemoryBudgetPropertiesEXT ctor:8659,
VK_EXT_memory_budget sType:4214, name:4213), vkCreateDevice:11198,
vkEnumerateDeviceExtensionProperties:11354, vkGetDeviceQueue:11376,
vkQueueSubmit:11397, vkQueueWaitIdle:11408, vkDeviceWaitIdle:11416,
vkDestroyInstance:11054, vkDestroyDevice:11244. Portability enumeration:
VK_KHR_portability_enumeration name:5201 +
VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR:5202; subset name:3775.

Memory/buffer/image - vkAllocateMemory:11424, vkFreeMemory:11447,
vkMapMemory:22553 (positional, returns cdata void*),
vkUnmapMemory:11455, vkGetBufferMemoryRequirements:11502,
vkBindBufferMemory:11522, vkGetImageMemoryRequirements:11533,
vkBindImageMemory:11553, vkCreateBuffer:11828, vkDestroyBuffer:11851,
vkCreateImage:11890, vkDestroyImage:11913, vkCreateImageView:11942,
vkDestroyImageView:11965. Memory-property bits:677-680.

Shader/pipeline - vkCreateShaderModule:11973, vkDestroyShaderModule:11996,
vkCreateGraphicsPipelines:12069, vkCreateComputePipelines:12119,
vkDestroyPipeline:12193, vkCreatePipelineLayout:12201,
vkCreateDescriptorSetLayout:12263, vkDestroyDescriptorSetLayout:12286,
vkCreateDescriptorPool:12294, vkAllocateDescriptorSets:12335,
vkUpdateDescriptorSets:12368,
VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO:861.

Pipeline cache persistence - LOAD via
VkPipelineCacheCreateInfo(initialDataSize=, pInitialData=bytes)
(vkCreatePipelineCache:12004, ctor:6697). **SAVE is not wrapped**:
vkGetPipelineCacheData has zero defs in _vulkan.py; it IS in the prebuilt
cffi cdef (vulkan.cdef.h:4015; PFN:3667), so gl_vr.py:604-633 calls
`vulkan._vulkan.lib.vkGetPipelineCacheData` (lib handle bound at
_vulkan.py:106) size-then-fill. A future binding regen changes nothing.

Pass/framebuffer/commands - vkCreateFramebuffer:12378,
vkDestroyFramebuffer:12401, vkCreateRenderPass:12409,
vkDestroyRenderPass:12432, vkCreateCommandPool:12482,
vkAllocateCommandBuffers:12523, vkFreeCommandBuffers:12545,
vkBeginCommandBuffer:12554, vkResetCommandBuffer:12571,
vkCmdBindPipeline:12580, vkCmdSetViewport:12597, vkCmdSetScissor:12606,
vkCmdBindDescriptorSets:12670, vkCmdDraw:12702, vkCmdDispatch:12770,
vkCmdCopyBufferToImage:12861, vkCmdCopyImageToBuffer:12872,
vkCmdClearColorImage:12928, vkCmdPipelineBarrier:13004,
vkCmdBeginRenderPass:13095, vkCmdEndRenderPass:13110.
Layout/Stage/Access/Format constants: all imported
(IMAGE_LAYOUT group:570-578; STAGE group:816-827 + 1114-1127; ACCESS
group:663-670; DESCRIPTOR_TYPE group:243-248; USAGE 604-608/182-188;
FORMATs R8G8B8A8_UNORM:332, R16_SFLOAT:371, R16G16_SFLOAT:378,
R32_SINT:394, R16G16B16A16_SFLOAT:392, R32G32_SFLOAT:401,
R32G32B32A32_SINT:403, R32G32B32A32_SFLOAT:404; sampler/filter
1087-1103; QUEUE_FAMILY_IGNORED:2613).

## GL -> VK barrier table (contractual - full text also in gl_vr.py docstring :36-101)
| engine GL sync point | gl_vr lowering |
|---|---|
| c.finish() per tile (bake.py:39, render.py 176/201, erosion.py 251-254) | one-shot submit + vkQueueWaitIdle (Host.end(wait=True)); ordering inside a recorded session needs no extra barriers (single queue) |
| ctx.memory_barrier() erosion.py:252 (GL_SHADER_IMAGE_ACCESS) | vkCmdPipelineBarrier COMPUTE->COMPUTE, SHADER_WRITE->SHADER_READ over touched storage images; images live in GENERAL (gl_vr.py:744,1297) |
| gl.read_tex() (render.py 177/425/505, bake.py:40, erosion _read) | barrier src=writer stage/access (COLOR_OUTPUT+COLOR_WRITE after draws, COMPUTE+SHADER_WRITE after dispatches, 0/0 when resting in SHADER_READ_ONLY) -> TRANSFER, layout->TRANSFER_SRC; vkCmdCopyImageToBuffer -> host-coherent staging; submit+wait; read mapping (gl_vr.py:831-886) |
| texture upload (erosion _upload, world maps) | staging->copyBufferToImage (UNDEFINED->TRANSFER_DST), barrier TRANSFER_WRITE->FRAGMENT_READ, layout->SHADER_READ_ONLY (GENERAL when the image is also storage) (gl_vr.py:788) |
| p.set(...) uniform writes (render.py 435-447) | memcpy into the persistent-mapped UBO - safe under queue-idle discipline: the engine only sets uniforms with the queue idle; **P3 contract** |
| render.py:451-452 acc fbo.clear once/frame | render-pass variant loadOp=CLEAR (gl_vr.py:894) |
| cone.draw(cfbo)->trace samples uCone (render.py 459-481) | cone target finalLayout=SHADER_READ_ONLY - render-pass end is the implicit external dependency |
| trace.draw(gfbo)->gfbo.read(rect)->kinds sample uGBuf (render.py 484-486) | gbuf target finalLayout=TRANSFER_SRC (readback barrier-free); host.barrier re-transitions TRANSFER_SRC->SHADER_READ_ONLY (FRAGMENT access) before kind passes |
| kind blend ONE,ONE into acc (render.py:452) | "blend" graphics-pipeline variant + LOAD render-pass; layout-only barrier when target rested in SHADER_READ |
| bind_to_image erosion images | descriptors bind GENERAL for the whole life; dispatch->dispatch order = in-order queue + band barrier; **UNVERIFIED probe P2-E1** |

Row order and sync points mirror engine/render.py, engine/bake.py:39-40 and
engine/erosion.py:230-260 dispatch order exactly (contract section 7 D6 pipeline).

## MoltenVK semantics claimed explicitly (review these)
1. Portability enumeration + subset extension enablement as in the proven
   spike (instance flag bit, ext names).
2. Single queue family with GRAPHICS|COMPUTE|TRANSFER; SHARING_MODE_EXCLUSIVE;
   no external memory -> queue-family-IGNORED barriers only for self-sync.
3. Device selection prefers INTEGRATED (Apple unified memory);
   recommendedMaxMemorySize drives the watchdog cap (0.75 factor).
4. RGBA16F color attachments + filterable float32 sampling are the
   MoltenVK/Metal-supported path (phase1 compiled them; runtime unproven).
5. vkCmdClearColorImage is only used for whole-image clears; tile clears go
   through the render pass.
6. Pipeline cache blobs are MoltenVK-internal (Metal state archives);
   cache invalidation across driver updates relies on the header UUID
   inside the blob, hence we persist opaquely.
7. GENERAL-layout storage images with coherent loads/stores is the GL
   semantics translation MoltenVK does for image2D; workgroup-ordering
   across bands relies on the explicit barrier (probe P2-E1).

## Unverified risks -> exact first-run probes
| id | risk | probe |
|---|---|---|
| P2-E1 | coherent imageLoad/imageStore RMW across workgroups through MoltenVK->Metal (erosion fill/accum/hill). No atomics anywhere, but band->band ordering is load-bearing. | Run erosion 512 twice: (a) as shipped, (b) with a per-dispatch barrier variant (host.barrier COMPUTE->COMPUTE issued inside the band loop instead of after it). Compare imgW/imgAcc0/imgH0 checksums to the CPU/GL reference stats logged by erosion.py band loop. Divergence => add per-dispatch SHADER_STORAGE barriers. |
| P2-S2 | sRGB / LOD / dFdx / textureGather semantics: sampler2D uploads are L8/R16F linear; bake_ao/relief use dFdx+Gather. | Upload a synthetic checkerboard, read back NEAREST at texel centers and LINEAR at texel edges via host.read_2d; compare against hand-computed GL rules. Run bake_relief 64 tile vs numpy reference. |
| P2-C3 | pipeline-cache file survives restarts and second vkCreatePipelineCache accepts the blob (raw-lib save path). | open -> build('scene_trace') -> save -> close -> open -> build again; assert second run's build time drops and file size > 1 KiB. |

## Risks FIXME(port) (file:line, why)
- gl_vr.py:426-445 Target finalLayout TRANSFER_SRC for gbuf: if MoltenVK
  validation rejects a render-pass whose final layout is TRANSFER_SRC it
  aborts the whole scene pass first run (fallback: finalLayout
  COLOR_ATTACHMENT + explicit barrier - the barrier helper already
  supports re-transitioning before kind passes).
- gl_vr.py:285-323 mat3 pack: GLSL column c <- numpy row c (moderngl's
  upload convention). Wrong transpose => camera/world rotated or mirrored
  on first frame - the single highest-impact member (uCamRot@1088).
- gl_vr.py:1092-1100 set_uniforms: mapped-UBO writes assume queue idle.
  If P3 ever sets uniforms between begin()/end() the frame races
  (stale or torn params).
- gl_vr.py:604-633 pipeline cache SAVE depends on raw cffi symbols
  (`vulkan._vulkan.lib`); a binding regen must keep lib or move the call
  to the wrapped API when one appears.
- gl_vr.py:765-786 staging row-alignment: rowLength=0 (tight) assumed;
  engine formats qualify, any new 1- or 2-byte-per-row texture must extend
  the guard (raises, never copies garbage).
- gpuslot_vr.py:223-248 proc_gb rusage slot layout was **probed**, not
  documented by headers on this libc; an OS update can reshuffle slots -
  sanity bounds fall back to ru_maxrss (never below, may overestimate).
- gate note: manifest.json currently differs from HEAD only in 25
  `glslc_seconds` fields (see Deviations 2).

## P3 must know
- Entry: Host(cache_dir=WS/engine_vr/out, res=..., tile=192, spp=1),
  open() (only guarded device creation; logs device + budget),
  build(name) per manifest program, write_descriptors(prog, samplers=...,
  storage_images=..., ubo_now=True), set_uniforms(prog, fields),
  begin()/draw(prog,target,rect,cleared=...,blend=...)/end(wait=True),
  dispatch(prog,gx,gy), upload_2d, read_2d (returns t.read() order -
  the engine's [::-1] flip stays where engine/render.py:505 does it),
  save_pipeline_cache()/close().
- Queue-idle discipline is the UBO contract (see risk above).
- One shared UBO per program; pack once per pass (5824 B memcpy per
  set_uniforms call is fine).
- gpuslot_vr.acquire() must run before Host.open() in P4 drivers so the
  slot lock + watchdog are up before Metal compiles anything; budget
  auto-configures from recommendedMaxMemorySize at open().
- readback staging keeps row 0 = upload row 0 (GL texture order).
- engine_vr/spirv/ already holds all 28 validated .spv; ensure_spirv
  rebuilds only when the shader_gen source is newer (glslc from
  tools/vk/bin, --target-env=vulkan1.2).
---

## Addendum - Phase-2 FIX pass (review_p2.md section 5), landed post ACCEPT-WITH-FIXES

Scope: engine_vr/gl_vr.py (1334 -> 1480 LOC) and engine_vr/gpuslot_vr.py
(452 -> 458) + this addendum; nothing else written. ZERO GPU executed again
(no vkCreateDevice ever ran). Every review finding was re-verified against
the bindings ground truth (.venv/.../vulkan/_vulkan.py, vulkan 1.3.275.1)
and CPU-only probes before editing: ALL findings held, NONE rejected. The
rusage 160-B write and every cffi failure mode were re-measured empirically
on this machine (prefill 0xAA + scan; buffer probe), not taken on trust.
Self-check after fixes: GATE P2: PASS | ruff E,F clean on engine_vr/ |
PACKER SELFTEST: PASS - 25 UBOs, 609 members, 0 GPU calls.

### section 5 mandatory-fix ledger
| item | status | landed at | verification |
|---|---|---|---|
| C1 deviceName double-decode | FIXED | gl_vr.py:575, 617 | StrWrap/_cstr auto-decodes char[] (_vulkan.py:20-66); probe: StrWrap(props).deviceName is str; .decode() would raise AttributeError |
| C2 inverted UBO guard | FIXED | gl_vr.py:1124 `if prog.ublocker is not None:` | ubo starts None in Program.__init__; old guard never created any UBO - set_uniforms always raised |
| C3 int(render_pass) key | FIXED | gl_vr.py:1366 `key = (render_pass, bool(blend))` | probe: int() raises TypeError on `struct VkRenderPass_T *`; cdata IS hashable |
| C4 vkMapMemory re-wrap | FIXED | gl_vr.py:374 (store), 376-384 (write/read) | _vulkan.py:22553-22560 returns ffi.buffer(ppData[0],size); probe: re-wrap TypeError, slice-assign works |
| C5 rusage buffer overrun | FIXED | gpuslot_vr.py:81 `_RUSAGE_WORDS = 20` | own-pid probe: proc_pid_rusage(flavor 2) wrote EXACTLY 160 B (last offset 159); old 96 B buffer was a 64 B heap overflow |
| C6 sparse descriptor sets | FIXED | gl_vr.py:1229-1246 consecutive-run bind at each set id | logic simulated against manifest: bake_height {1} -> bind@1; erosion {0,1,3} -> (0,[s0,s1])+(3,[s3]); 20 scene {0,1} -> (0,[s0,s1]) |
| M1 fullscreen VBO unbound | FIXED | gl_vr.py:1333-1336 (bind), 538/725-727 (create-once/close-destroy) | chain: _fullscreen.vert:2 in_pos @loc0 -> attribute binding=_VB_ATTR(0) stride 8 (gl_vr.py:1403-1413) -> draw() binds fullscreen_vbo() (-1,-1)(3,-1)(-1,3) |
| M2 pipeline-cache ffi save/load | FIXED (static-correct) | load gl_vr.py:640-651 cast("void*", from_buffer(blob)) + explicit initialDataSize; save 673-683 char[] + size>0 guard | probes: ffi.new("void[]",n) and ctor(bytes) -> ValueError "array item of unknown size: void"; ctor on void* without explicit size -> TypeError no len(); char[]+buffer OK; raw-lib vkGetPipelineCacheData shape (0 wrapper defs, present on _vulkan.lib) confirmed |
| M3 compile-guard condition | FIXED | gpuslot_vr.py:398 | restores engine/gpuslot.py:171 `if self.need_gb is not None and self.need_gb <= 8.0:` exactly |
| M4 upload barrier | FIXED | gl_vr.py:888-899 | TRANSFER_WRITE -> FRAGMENT|COMPUTE SHADER_READ (was BOTTOM_OF_PIPE/0 = layout-only, no execution dependency); COMPUTE added because storage images rest in GENERAL |
| M5 re-entry barrier + false implicit-dep claim | FIXED | gl_vr.py:1276-1300 src=LAST TOUCHER (TRANSFER_SRC -> TRANSFER read WAR; SHADER_READ_ONLY -> FRAGMENT read; GENERAL -> COMPUTE write), 1339-1356 explicit COLOR_ATTACHMENT_OUTPUT(W) -> FRAGMENT|COMPUTE(R) barrier for passes ending at SHADER_READ_ONLY, docstring rows 58-102 rewritten | spec truth: render-pass end implicit EXTERNAL dependency orders COLOR-attachment accesses ONLY - sampled reads need the explicit barrier; MVK serializes and hides its absence today, contract now says so |
| M6 attachment-count contract | FIXED | gl_vr.py:1259-1267 draw() raises GlvrError when len(prog.outputs) != target.n | stricter than min(): spec requires blend-attachmentCount == subpass view count, so min() would silently build invalid pipelines; engine pairings (2->gbuf, 1->acc) unaffected - guard is a contract check |
| minor VK_EXT_memory_budget enable | FIXED | gl_vr.py:592-597 (enable when offered), 610-614 (query moved AFTER vkCreateDevice) | unrequested pNext is ignored -> budget log would read 0.0 GiB at runtime (P5 watchdog input) |
| minor close() wait-before-save | FIXED | gl_vr.py:701-702 | vkDeviceWaitIdle now precedes save_pipeline_cache (save became live thanks to M2) |
| minor unmap-before-free | FIXED | gl_vr.py:386-394 | vkUnmapMemory precedes vkDestroyBuffer/vkFreeMemory |
| minor vkCmdClearColorImage claim | CORRECTED (report text) | corrections item 5 below | 0 occurrences in gl_vr.py (grep) - claim was false |
| N3 queue-family StopIteration | UNTOUCHED | - | review marked N2-N8 deferrable except the two above; left for P3 |

### corrections to this report (claims the review flagged; re-verified)
1. T1 "pipeline-cache persistence DONE" was FALSE as written: save and
   load BOTH threw (ffi.new void[] / ctor len(void*) errors) and died in
   their except branches - save always returned False, load silently made
   an empty cache. M2 fixes both paths; a runtime round-trip (probe P2-C3)
   remains first-run work under the zero-GPU discipline.
2. Barrier-table row "texture upload ... TRANSFER_WRITE -> FRAGMENT_READ"
   did not describe the code (BOTTOM_OF_PIPE/0). Table and code now match.
3. Barrier-table row "the render-pass end IS the implicit EXTERNAL
   dependency (COLOR w -> r)" is false per spec - implicit end-of-pass
   covers COLOR accesses only. Rewritten; explicit barrier added (M5).
4. "compile_guard (same semantics)" overstated: condition had drifted
   (need0 <= 8.0 vs the original need_gb is not None and <= 8.0). M3
   restores the original semantics exactly.
5. MoltenVK-semantics claim #5 was false: vkCmdClearColorImage is never
   called by gl_vr.py. ALL clears (tiles and the once-per-frame acc clear)
   go through render-pass LOAD_OP_CLEAR variants.
6. Device-log/budget lines had never executed (C1 AttributeError at the
   first open() before reaching them); budget query additionally required
   the VK_EXT_memory_budget enablement to return non-zero.

### runtime-deferred (static-correct, first-run verified)
M2 cache round-trip (P2-C3), C6/M1/M4/M5 barrier effects and the new
sampled-read barrier cost on Metal, and the budget log values - all need
a device; landing this pass executed none (zero-GPU discipline).
