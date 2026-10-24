# P2 independent contract review — engine_vr/gl_vr.py + engine_vr/gpuslot_vr.py

Reviewer: independent (no GPU use, no edits to reviewed files). Ground truth =
.venv/lib/python3.12/site-packages/vulkan/ (bindings 1.3.275.1, cffi backend),
engine/ read-only, engine_vr/manifest.json, spike/p0_a/run.py. All "empirical"
claims were executed CPU-only in throwaway subprocesses with tools/vk env
(import only; no device, no loader calls beyond the import-time dlopen).

## 1. API table verdict

Checked (AST-extracted from gl_vr.py, aliases vkv/_vv/vulkan): 71 vulkan
function call sites (~60 unique functions), 73 struct-ctor sites (~34 unique
structs, all kwargs matched against the generated ctor signatures), 166 VK_*
constant references, 2 raw-cffi sites
(vulkan._vulkan.lib.vkGetPipelineCacheData) = **312 API identifiers; 312/312
names exist; 0 arity/order mismatches against the wrapper signatures; 0 unknown
struct fields; 0 constant typos; vkGetPipelineCacheData present in _vulkan.lib
(hasattr, never called), absent from the Python wrapper module (author report
claim: true)**. Gates: py_compile OK; ruff E/F clean (extended ruleset: 18
hygiene hits, no F/E9).

Every defect found is misuse of a correctly-spelled API, not a name/signature
error. Discrepancy list (gl_vr.py file:line -> ground truth -> runtime effect):

| # | Site | Ground truth | Why it breaks |
|---|------|--------------|---------------|
| D1 | gl_vr.py:540, :573 [p.deviceName.decode()] | _vulkan.py:20 (_cstr), :37 (StrWrap): char[] fields auto-decode to str (empirical) | AttributeError in Host.open() right after device pick; open() can never return |
| D2 | gl_vr.py:1048 [if prog.ubo is not None:] | ubo = None at gl_vr.py:468 (Program.__init__) | guard inverted (must test prog.ublocker): UBO never allocated -> set_uniforms raises GlvrError (:1097-98); UBO descriptor write silently skipped (:1137) -> set1 unbound at every draw |
| D3 | gl_vr.py:1220 [int(render_pass)] | handles are cdata pointers; empirical: int() not supported on cdata "struct VkRenderPass_T *" (they ARE hashable; usable as dict keys directly) | TypeError inside _gfx_pipeline (called from draw :1208) mid-command-buffer on the first graphics draw |
| D4 | gl_vr.py:352, :358 [vkv.ffi.buffer(self.mapped, ...)] | _vulkan.py:22553 vkMapMemory **returns an ffi.buffer**, not a pointer (empirical: re-wrap raises TypeError "argument 1 must be _CDataBase, not buffer") | every GpuBuffer.write/read raises TypeError: upload_2d (:809), fetch (:873), UBO write (:1099), fullscreen_vbo (:1328) - the whole upload/readback/uniform path is dead |
| D5 | gl_vr.py:620 [ffi.new("void[]", n)]; :594-597 [VkPipelineCacheCreateInfo(pInitialData=bytes)] | cffi cannot size void[]: empirical ValueError "array item of unknown size: 'void'" for both | save always dies in except :630 (logs, returns False); load ctor always dies in except :600 (pcache=None once a cache file exists). T1 pipeline-cache persistence broken in BOTH directions, silently |
| D6 | gl_vr.py:1153-1158 [_bind_sets] binds sorted allocated sets at firstSet=0 | manifest set keys: 20 programs (0,1); bake_height (1,); erosion_fill / erosion_accum / erosion_accum_flow_only / erosion_hill (0,1,3) | bake_height: UBO set bound at slot 0 (pipeline layout expects set1); erosion: set3 storage set bound at slot 2, slot3 unbound -> descriptor set/layout mismatch = UB or garbage for 5 of 25 programs |
| D7 | gpuslot_vr.py:236-240 [(c_uint64*12)()] buffer (96 B; _RUSAGE_WORDS=12 at :76) | empirical garbage-scan probe: proc_pid_rusage(pid, 2, buf) writes **160 bytes**; the ABI carries no size parameter | 64-byte out-of-bounds heap write on every proc_gb() call, including the watchdog thread every 5 s - memory corruption, not a wrong number |

Verified-correct highlights (representative, all against generated wrapper code):
vkCreateInstance + ENUMERATE_PORTABILITY flag + portability ext (:518-526)
legal and spike-matched; handle-return ([0]-style) usages match the wrapper
returns; vkGetPhysicalDeviceQueueFamilyProperties /
vkEnumerateDeviceExtensionProperties (returns a generator - materialized once,
OK) usage correct; vkCmdPipelineBarrier arity 10 (GT _vulkan.py:13004),
vkCmdBindDescriptorSets arity 8 (:12670), vkAllocateDescriptorSets array return
+ [0] (:12335) - all match call sites; VkShaderModuleCreateInfo(pCode=bytes)
empirically OK; pNext chain VkPhysicalDeviceMemoryProperties2(pNext=
VkPhysicalDeviceMemoryBudgetPropertiesEXT) constructs with auto sType
(1000237000); find_memtype (propertyFlags & need) == need precedence correct;
generated struct ctors auto-fill sType, so the module's explicit-sType style is
consistent and harmless.

Std140 (T3): independent recomputation of all 25 uniform_blocks - member tables
encode arrays as one entry whose span runs to the next offset (uTrail vec4@16 ->
528 = 32xvec4; uFallK 48xvec4 (engine fk(48,4)); uStream 96xvec4; uW2K 8xvec4 =
engine wk 32 floats); every offset is std140-aligned for its (array) type;
Std140Packer constructs 25/25, packer.size == manifest size_bytes 25/25;
empirical pack: uCamRot GLSL column c == numpy row c at 1088+16c, uTrail/uW2K
stride 16, overrun guards raise. mat3 convention == the raw moderngl upload
semantics used by engine/gl.py:92-99 (u.write of row-major tobytes; shader
column c receives input slice c) -> matches what the shipped shaders were tuned
against. Note: the ctor "offset cross-check" (gl_vr.py:246-250) is
near-tautological (cursor = the aligned next manifest offset by construction);
it validates alignment, not GLSL re-derivation - weaker than the docstring's
"both directions" claim. Acceptable given the manifest is glslc-reflected.

## 2. Findings by severity

### CRITICAL (guaranteed first-run failure / memory unsafety)
- C1 = D1 decode-on-str: Host.open() AttributeError (:540, :573). Everything else is behind this.
- C2 = D2 inverted UBO guard: first set_uniforms raises (gl_vr.py:1048, 468, 1097-98); UBO never bound even if set_uniforms were skipped (:1137).
- C3 = D3 int(cdata render_pass): TypeError at the first draw pipeline lookup (:1220).
- C4 = D4 buffer double-wrap: TypeError on the FIRST buffer write/read (gl_vr.py:352,358 vs _vulkan.py:22553). upload_2d / read_2d / set_uniforms / fullscreen_vbo all dead.
- C5 = D7 rusage 96 B buffer vs 160 B kernel write: 64-byte OOB heap write every 5 s from the watchdog (gpuslot_vr.py:76, 236). Slot mapping itself confirmed correct (slot 8 resident_size 16.7->553.9 MB and slot 9 phys_footprint +512.4 MB on a 256 MB dirty alloc, own-pid probe) - the buffer size is the bug.
- C6 = D6 _bind_sets firstSet=0: bake_height + 4 erosion programs bind sets to wrong slots (gl_vr.py:1153-1158) - spec UB; MVK garbage or fault on those 5 programs' draws/dispatches.

### MAJOR (wrong result / crash under specific passes)
- M1 vertex buffer never bound: engine_vr/shaders_gen/_fullscreen.vert:2 reads in_pos; the pipeline declares binding 0, stride 8 (gl_vr.py:1260-1267), but vkCmdBindVertexBuffers exists nowhere in gl_vr.py (grep: only the _VB_ATTR constant at :174; draw() :1160-1216 never binds). Metal substitutes a zero-filled buffer -> in_pos=(0,0) -> degenerate triangle -> ALL graphics passes render nothing while returning success. fullscreen_vbo() (:1318-1329) acknowledges the need but draw() never uses it. Not in the author's risk list. (Related latent trap: fmt_map vec3->R32G32B32A32 (:1232-1233) would be the wrong stride; manifest in_pos is vec2 only today.)
- M2 = D5: T1 pipeline-cache persistence never functions (load AND save); the report lists T1 as done. The raw-cffi two-call protocol (symbol present in _vulkan.lib, size-then-fill) is otherwise the right shape - it dies purely on void[] sizing ("char[]"+cast fixes it).
- M3 compile_guard drift: port small-compile branch gpuslot_vr.py:392 [if need0 <= 8.0:] vs ORIGINAL engine/gpuslot.py:171 [if self.need_gb is not None and self.need_gb <= 8.0:]. With need_gb=None the port lets estimate-small compiles (all bake/erosion shaders, need0=6..8) bypass the machine-wide compile.lock; the original skipped the lock ONLY for explicitly measured small compiles. Concurrency budget guard weakened exactly for multi-process builds; absent from the author's deviation list.
- M4 contract row violated in code (upload barrier): docstring :58-63 promises "barrier TRANSFER_WRITE -> FRAGMENT_READ"; code gl_vr.py:823-826 barriers TRANSFER -> **BOTTOM_OF_PIPE** with dstAccessMask=0. Spec-wise a later sampled read is unordered w.r.t. the copy (layout-transition-only). MVK's in-order single queue hides it; on any out-of-order driver: torn first-use textures.
- M5 readback re-entry race: gbuf rests in TRANSFER_SRC; the next-tile draw() LOAD path transitions TRANSFER_SRC -> COLOR_ATTACHMENT with srcStage=FRAGMENT, srcAccess=0 (gl_vr.py:1178-1181) although the last touches were COLOR_ATTACHMENT writes + a TRANSFER read -> the new color write is unprotected against the readback copy (spec). Same MVK-tolerated class as M4.
- M6 color-blend attachment count coupling: pColorBlendState pAttachments = [blend_att]*len(prog.outputs) (:1287) vs render-pass attachment count target.n (:909, :916) - a 2-output program drawn into a 1-attachment target yields a pipeline/render-pass attachment-count mismatch (invalid, silent without validation). No guard.

### MINOR (hygiene)
- N1 VK_EXT_memory_budget is used in pNext (:579-585) but never enabled in vkCreateDevice (enable list :555-557 = portability_subset only) -> per spec the unrequested pNext is ignored -> heap budget/usage log prints 0.0 GiB. The T1 "budget log" is half-fake; recommendedMaxMemorySize (:542) is a core limit and works.
- N2 close() (:635-668): save_pipeline_cache before vkDeviceWaitIdle (benign only because save is broken - wrong ordering once M2 is fixed); no vkUnmapMemory before vkFreeMemory in GpuBuffer.destroy (:361-364); Targets/framebuffers/render-passes/staging never destroyed (leak on any Host reuse).
- N3 queue family next(...) without default (:550-552): StopIteration instead of GlvrError on an ICD without a COMPUTE|GRAPHICS family. Presentless selection itself is correct (headless, single family, spike-confirmed).
- N4 vkCmdClearColorImage: cited in the phase2 report lowering table + the "whole-image clears" MoltenVK claim, but never called in gl_vr.py (clears go through render-pass LOAD_OP_CLEAR :899) - stale contract text.
- N5 pipeline cache blob (once M2 is fixed) is not scoped to MoltenVK version/build; cross-version loading relies on implementation tolerance. Informational.
- N6 ruff extended ruleset: 18 hygiene items (BLE001/S110 broad excepts etc.); the author's declared gate (E,F) passes - gate_p2's claim holds.
- N7 gpuslot_vr re-reads ALPINE_MAX_PROC_GB each watchdog loop (original reads once, engine/gpuslot.py:83) - deliberate improvement for configure_budget, undocumented (harmless).
- N8 fetch() returns a read-only numpy (np.frombuffer over bytes, gl_vr.py:873); engine/render.py:505 only slices/divides - safe today, but any P3 in-place write raises ValueError. Note for P3.

## 3. Author's top-3 risks - verdicts

- R1 (finalLayout=TRANSFER_SRC "aborts the scene pass first run", ~gl_vr.py:426-445/906): DISAGREE - overrated. The exact configuration (render-pass attachment ending in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL + vkCmdCopyImageToBuffer) is already hardware-proven on this MoltenVK in spike/p0_a/run.py (the report's own P2-S1/P2-E1 evidence). Legal per spec here. The first run in fact dies at C1/C3 long before this could matter. Not a meaningful residual risk.
- R2 (mat3 transpose mirrors the world, gl_vr.py:285-323, uCamRot@1088): AGREE the impact would be highest-class, but the code is CORRECT and residual risk is near zero. Empirical pack: column c = numpy row c at 1088+16c; that is byte-for-byte the raw-upload convention engine/gl.py:92-99 (u.write(row-major tobytes)) that every shipped shader was tuned against - shader column c receives input slice c in BOTH backends. Downgrade from "risk" to "confirm on first render" (probe P2-E2 already planned). The author was right to look here; the transpose is right.
- R3 (mapped-UBO writes assume queue idle, gl_vr.py:1092-1100): AGREE as characterized - a documented P3 contract (docstring :64-68) with zero enforcement (no assert that no command session is active / queue idle). Becomes live once C2 is fixed; a one-line guard (raise if self._active_cmd is not None) would turn a silent race into a loud error. Not a P2 defect per se.

Net: of the three named risks, one is overrated (R1), one is already correct
(R2), one is correctly characterized but moot until C2 (R3). None of the six
actual first-run blockers (C1-C6) appear on the author's risk list.

## 4. Missing-barrier analysis vs the docstring table (gl_vr.py:32-90)

Rows that hold in code:
- c.finish() per tile -> Host.end(wait=True) = submit + vkQueueWaitIdle (:720-735); mirrors bake.py:39, render.py:176-201, erosion.py:251-254.
- read_tex row -> to_staging layout->stage map (:841-857) exactly as promised (COMPUTE+SHADER_WRITE after dispatch, COLOR_OUTPUT+COLOR_WRITE for draw targets, access=0 for SHADER_READ_ONLY images), copy to host-coherent staging, wait, read mapped memory.
- erosion band barrier (COMPUTE->COMPUTE, SHADER_WRITE->SHADER_READ over GENERAL storage images) - generic mechanism present (:744-763); P3 must actually issue it.
- acc CLEAR-once row: passes[cleared] with LOAD_OP_CLEAR + initialLayout UNDEFINED (:899-906); engine clears once per frame at render.py:451-452.
- kind-pass blend ONE,ONE + LOAD variant; "layout-only barrier" row - code literally is layout-only (srcAccess=0, :1178-1181), matching its own wording (but see M5 for the wrong SRC side).
- storage images live in GENERAL for their whole descriptor life (:1128-1129).

Rows that do NOT hold:
1. Upload row: promised TRANSFER_WRITE -> FRAGMENT_READ; code is TRANSFER -> BOTTOM_OF_PIPE with dstAccess=0 (:823-826). [M4]
2. "cone.draw then trace samples uCone - the render-pass end IS the implicit EXTERNAL dependency (COLOR w -> r)" (:71-73): per spec the implicit subpass dependency covers color-attachment accesses only; a later pass *sampling* the image (trace samples uCone; kinds sample uGBuf0/1) is a FRAGMENT_SHADER read that implicit deps do NOT cover. Needs one explicit COLOR_ATTACHMENT_OUTPUT(+WRITE) -> FRAGMENT_SHADER(+SHADER_READ) barrier between those passes (host.barrier can express it today). MoltenVK lowers everything into one in-order MTLCommandBuffer, so this runs on target - but the table's claim is false as written and the sync is genuinely absent. [MAJOR-spec; runtime-safe on MVK only]
3. gbuf readback -> resample row: the re-transition (TRANSFER_SRC -> SHADER_READ_ONLY via host.barrier) is P3 duty and unenforced; worse, descriptor layout is frozen at write_descriptors time (:1114-1115: GENERAL else SHADER_READ_ONLY) - a sampler written while the image sits in TRANSFER_SRC records a wrong layout. Recommend a bind-time assert (recorded layout == img.layout).
4. draw() LOAD src-stage after readback (M5): no table row covers the TRANSFER_SRC -> COLOR_ATTACHMENT re-entry src stage/access, and the implemented choice (FRAGMENT+0) is unsound per spec.
5. dispatch->dispatch "in-order queue + band barrier" is the same in-order bet as row 2, correct only if P3 actually issues the band barrier; gl_vr cannot enforce it.

All of these are invisible on MoltenVK (single queue, serialized encoders) -
consistent with the zero-GPU state - but the table IS the P2 contract, and rows
1-2 currently misstate what the code does or assumes.

## 5. Verdict

**ACCEPT-WITH-FIXES** at architecture level - the skeleton is right (manifest-
driven descriptors, one-shot sessions, spike-aligned device bootstrap, byte-
exact std140 packer, gpuslot budget plumbing) - but as written the module cannot
execute: Host.open() raises before any render (C1); the uniform, upload,
readback and draw paths each raise independently (C2-C4); 5 of 25 programs bind
descriptor sets to wrong slots (C6); the watchdog corrupts its own heap every
5 s (C5); and graphics draws produce blank output by construction (M1).
Mandatory fixes before P3 merge (all local, no design change):

1. C1: drop the two .decode() calls (:540, :573) - deviceName is already str.
2. C2: gate on prog.ublocker and create the UBO (:1048).
3. C3: use the hashable render-pass handle (or id()) as the pipeline cache key (:1220).
4. C4: store vkMapMemory's ffi.buffer directly and slice-assign it (:346-358; empirical: slice-assign works, no re-wrap).
5. C5: rusage buffer >= 160 B (e.g. c_uint64*20; slots 7/8/9 already read the right fields) (:76, :236).
6. C6: bind each allocated set at its own set index (per-set vkCmdBindDescriptorSets, or bind 0..max padded with dummy sets) (:1153-1158).
7. M1: bind fullscreen_vbo() in draw() - or make the shared VS positionless (gl_VertexIndex) and drop the attribute; either way keep manifest + pipeline input state in sync.
8. M2: use char[] + cast for pInitialData and the save blob so T1 persistence actually works in both directions (:594-597, :620).
9. M3: restore the need_gb-is-not-None condition on the lock-free small branch (:392 vs engine/gpuslot.py:171).
10. M4/M5: fix the two barrier stage/access pairs to match the table (upload dst = FRAGMENT_SHADER+SHADER_READ; gbuf re-entry src = TRANSFER+TRANSFER_READ or split COLOR/TRANSFER) - MVK tolerant, but the table is contractual.
11. N1: enable VK_EXT_memory_budget when present in dev_exts (:553-557) so the heap-budget log reports real numbers.

Everything else (N2-N8, barrier-table wording, bind-time layout assert) can
ride along with P3.

Author claims audited TRUE: portability-enumeration pattern matches the spike;
RGBA16F default per T4; descriptor pool sizing generous (132 sets vs <= 75
allocated; sampler 434 vs 209); zero module-level device calls; manifest-only
bindings; gpuslot slot/watchdog/env parity; rusage slots 8/9 mapping (buffer
size aside); mat3 convention (R2); vkGetPipelineCacheData unwrapped-but-in-lib.
Author claims audited FALSE/MISLEADING: T1 pipeline-cache persistence "DONE"
(M2); upload-barrier table row (M4); compile_guard "same semantics" as the
original (M3, unstated deviation); the three named risks as the top three risks
(section 3).


---

## 6. Re-verification of fix round (2026-10-25)

Reviewed engine_vr/gl_vr.py (now 1481 L) and engine_vr/gpuslot_vr.py (459 L)
against the same ground truth (.venv vulkan 1.3.275.1 generated wrappers,
empirical CPU-only probes; no device, no GPU). Gates re-run: py_compile OK;
ruff E,F clean ("All checks passed"); port/gates/gate_p2.py -> GATE P2: PASS;
Std140Packer validation re-run unchanged (25/25 blocks, uCamRot@1088,
scene_trace 5824).

API sweep on the NEW file (checker corrected this round: create-style wrappers
take the output param with a default - pBuffer=None - so a 3-pos call is the
documented pattern; the round-1 checker's raw ARITY hits were that
false-positive class, verified against the generated vkCreateBuffer body):
73 function call sites (+vkCmdBindVertexBuffers, +vkUnmapMemory), 73 struct
ctors, 184 constants, 2 raw-cffi sites -> REAL BAD: NONE. vkGetPipelineCacheData
raw prototype re-confirmed (VkDevice, VkPipelineCache, size_t*, void*) ->
VkResult, matching the new two-call save.

Item verdicts (ACCEPTED unless noted):
- C1 gl_vr.py:575,617 deviceName plain str, .decode gone (grep: only a comment at :573).
- C2 gl_vr.py:1124 gate now on prog.ublocker; UBO built :1125-1128; destroyed in close :704-705.
- C3 gl_vr.py:1366 key=(render_pass, bool(blend)) - cdata handles are hashable (empirical dict probe; int() still TypeError, so the old line WAS a crash).
- C4 gl_vr.py:374,379,384 mapped buffer stored directly; write = slice-assign, read = bytes(slice) - both empirically OK on a real ffi.buffer.
- C5 gpuslot_vr.py:81 _RUSAGE_WORDS=20 (160 B = exactly the measured kernel write), buffer :241, slots (7,8,9) unchanged.
- C6 gl_vr.py:1235-1246 consecutive-run binds at their own firstSet; sparse manifest sets (1,) and (0,1,3) now bind {1} and {0,1}+{3} correctly; handle-array conversion (_cast_ptr2 Iterable path) verified in _vulkan.py.
- M1 gl_vr.py:1334-1336 lazy fullscreen_vbo + vkCmdBindVertexBuffers every draw; triangle data :1469 == engine/gl.py:15; attribute state :1407-1413 matches (stride 8, vec2); vbo destroyed on close :725-727.
- M2 ACCEPTED STATIC-CORRECT: load :645-651 ffi.from_buffer(bytes)+cast void* + explicit initialDataSize (empirically constructs; the no-size TypeError the author cites reproduces verbatim); save :668-686 size_t[1] probe + ffi.NULL + char[] fill, size<=0 guard. Runtime round-trip still unproven (zero-GPU) - correctly labelled runtime-deferred.
- M3 gpuslot_vr.py:398 exact original condition restored.
- M4 gl_vr.py:893-898 upload barrier TRANSFER->FRAGMENT|COMPUTE, TRANSFER_WRITE->SHADER_READ; docstring row :58-65 now matches code.
- M5 gl_vr.py:1282-1294 re-entry src = last toucher (incl. TRANSFER_READ WAR from readback); :1339-1355 explicit COLOR_OUTPUT+WRITE -> FRAGMENT|COMPUTE+SHADER_READ barrier after passes ending SHADER_READ_ONLY; docstring :73-102 rewritten truthfully (implicit EXTERNAL covers color accesses only; cross-pass sampled reads explicitly synced). make_target default final_layout = SHADER_READ_ONLY (:961).
- M6 gl_vr.py:1266-1269 raise-on-mismatch. DEVIATION RULING: ACCEPTED - correct, my min() suggestion was wrong. The spec requires the pipeline blend attachmentCount to equal the subpass view count; silently truncating blend state builds an INVALID pipeline object that only validation would catch. Raising a GlvrError at draw() converts a spec violation into a loud host error; engine pairings (scene_trace 2->gbuf, kinds 1->acc) satisfy equality.
- N1 :594-597 enable VK_EXT_memory_budget when offered; budget log :611-614 after vkCreateDevice, guarded :624-633.
- N2a close :701-702 vkDeviceWaitIdle BEFORE save_pipeline_cache. N2b destroy :388-392 vkUnmapMemory before free. N4 vkCmdClearColorImage references gone (grep NONE).

New findings this round (both MINOR, neither blocks):
- N9 gl_vr.py:849 _pad_rows computes row = arr.shape[-2]*bpp: correct for (h,w,c) 2D uploads but picks h instead of w for depth>1 ((d,h,w,c)) - a mis-guard for odd-width 3D maps only; all engine 3D maps are POT-sized so rows are 4B-multiple regardless; also the docstring (:845-847) "builds (and strips) the pad" overstates: the helper only raises. Suggest shape[-3]*0 + explicit 2D/3D row-length selection in P3 if any odd-width 3D upload appears.
- N10 the sampled-read barrier (:1339) fires on every draw whose target ends at SHADER_READ_ONLY, including targets nobody samples next (kind passes into acc). Harmless, near-zero cost on MVK's in-order queue; optional per-target flag in P3.
- Residue note on C5: 20 words = 160 B is EXACTLY the measured write; a future OS build writing >160 B re-introduces the overflow. Author documents the probe; 24 words would be free headroom. Non-blocking.

Verdict history check: no runtime claims were accepted on trust - every fix was
re-judged against the generated wrapper or a CPU-only probe.

## 7. FINAL VERDICT: ACCEPT

All 11 mandatory fixes ACCEPTED (M2 static-correct with runtime deferred,
consistent with the zero-GPU discipline of P2). The M6 deviation is ruled in
the author's favor (raise beats silent min()). Residue = three MINOR notes
(N9 pad-rows dim for depth>1, N10 always-fire sample barrier, C5 zero
headroom) - none blocking; fold into P3 alongside the already-listed P3
duties (descriptor-layout bind assert, target final_layout choices per table).
P2 contract (docstring barrier table :32-102) is now truthful against code.
