# Phase 7 — G-buffer attachment mis-wire (BUG A) + render nondeterminism (BUG B)

## 1. Root cause A — platform divergence, not a port-order bug

**GL-era truth (the contract being ported).** The engine's G-buffer pass writes
TWO color outputs — `engine/shaders/scene.frag:257`
(`layout(location = 1) out vec4 fragInfo;` under SCENE_PASS==99),
`engine/shaders/scene.frag:280-281`
(`fragColor = vec4(h.t, float(h.kind), vec2(h.cell)); fragInfo = h.info;`) —
and the host reads the kind from **channel 1 of attachment 0**:
`engine/render.py:484`
(`kb = np.frombuffer(self.gfbo.read(viewport=rect, components=2, attachment=0, dtype="f4"), np.float32)[1::2]`),
wiring `uGBuf0/uGBuf1` to gtex[0]/gtex[1] at `engine/render.py:443-444`.

**The port reproduced that contract verbatim on the CPU side.** Verified
field-by-field after marshalling (engine_vr/tmp/p7probe6.py dumps the real
struct memory): `engine_vr/gl_vr.py:990` pAttachments = [att]*n;
`engine_vr/gl_vr.py:991-997` subpass refs `attachment=i` in creation order
(dump: refs [0,1,2], layouts COLOR); `engine_vr/gl_vr.py:999-1003` framebuffer
`pAttachments=[im.view for im in target.images]` (dump: 3 distinct view
handles, attachmentCount auto-set); manifest outputs order fragColor@0,
fragInfo@1; blend attachment list one-per-output (`engine_vr/gl_vr.py:1440-1444`).
Handle identity verified distinct (`engine_vr/tmp/p7probe4.py`: distinct
VkImage/VkImageView/VkDeviceMemory objects per GpuImage;
`engine_vr/gl_vr.py:409-435` creates image→mem→view per instance).

**Runtime measurement on this machine (MoltenVK 1.4.2, macOS 26.5, Apple M5)**
— reproducible, fresh pipeline-cache dirs, all probes under engine_vr/tmp/:

* p7probe2.py — the live trace program with constants replacing the two
  writes: gbuf.images[0] == fragInfo's constants (21,22,23,24),
  images[1] == (0,0,0,0), fragColor's constants (11,12,13,14) gone.
* p7probe3.py — 3-output program, per-attachment magic prefills:
  img0 == the HIGHEST-location output (30,31,32,33); img1/img2 keep their
  prefills (never written).
* p7probe5.py — framebuffer views SWAPPED ([view1, view0]): the writes still
  hit fb slot 0 only (img1 got the value, img0 kept its magic) — routing is
  fixed to slot 0, independent of view order.
* p7probe7.py — assignment order swapped (loc1 written first, loc0 last):
  slot 0 still got the loc1 value → the rule is by LOCATION, not order.
* MTL_DEBUG_LAYER=1: `_MTLDebugValidateRenderPassDescriptorAndTrackAttachments:370:
  failed assertion 'RenderPass Descriptor Validation / Attachment 0 overlaps
  with attachment 1 …'` for every multi-attachment draw (also with
  MVK_CONFIG_USE_MTLHEAP=0).
* MVK's own Metal shader dump (MVK_CONFIG_SHADER_DUMP_DIR,
  engine_vr/tmp/mvkdump/shader-fs-*.metal) shows all outputs correctly
  declared \`[[color(0)]]/[[color(1)]]/[[color(2)]]\` and written → the
  divergence is downstream of SPIRV-Cross, inside MoltenVK's pipeline/encoder
  (consistent with addFragmentOutputToPipeline, MVKPipeline.mm:1942-1961 with
  the legacy-renderpass identity slot map at :992 — internal collapse not
  proven).

**Rule (measured): only the highest-location WRITTEN fragment output is
delivered, and it lands in framebuffer SLOT 0; lower locations are discarded;
slots 1..n-1 receive no fragment writes.** Locally measured on
MoltenVK 1.4.2 — no upstream MoltenVK issue/release-note found, so this is
stated as local evidence only, no upstream citation.

This makes the engine's two-output G-buffer **unrepresentable as-is** on this
driver: attachment 1 never receives fragInfo, so every kind pass discards
(`if (int(g0.y + 0.5) != SCENE_PASS) discard;` — port mirror
engine_vr/shaders/scene.frag:364) except pixels where the mis-delivered
fragInfo.y happens to equal SCENE_PASS (P6's dispatch set {0}).

**Fix (RUNBOOK-accepted robustness layer; shader change proven necessary by
the engine-line pair above + the probes):** the pano trace keeps ONE color
output (fragColor = (t, kind, cell.xy) → gt0, which the host already reads
exactly like the engine at vr.py:1077) and routes h.info through a
\`writeonly image2D\` store (uGInfo → gt1, a non-attachment image in GENERAL
layout). Guarded by VR_PANO: the 28 film programs compile the untouched
fragColor+fragInfo pair. h.info stays 4-channel untouched (info.w is
load-bearing for cover/falls shading, engine/shaders/scene.frag:126,279) —
packing it into the color output is impossible losslessly, hence the store.

## 2. Diff summary (git diff --name-only HEAD)

* engine_vr/shaders/scene.frag — :292-310 (uGInfo decl, fragInfo decl moved
  into #else, FIXME(p7) rationale) and :330-343 (fragColor unconditional;
  imageStore(uGInfo, ivec2(gl_FragCoord.xy), h.info) under VR_PANO;
  fragInfo = h.info in #else). FIXME(p7) x2.
* engine_vr/manifest.json — scene_pano_trace: outputs [fragColor@0], added
  binding uGInfo {kind: storage_image, set 3, binding 0} (both auto-derived by
  build_p4 from SPIR-V reflection/vulkanize, no hand surgery).
* engine_vr/shaders_gen/scene_pano_trace.frag, engine_vr/spirv/scene_pano_trace.spv
  — regenerated by the documented build_p4 flow only.
* vr.py — :941-958 gt0 target (n=1) + gt1 = host.make_image(…, GENERAL,
  SAMPLED|STORAGE); :968 initial sampler transition for gt0 only; :971-974
  trace descriptors + storage_images={'uGInfo': gt1}; :985-994 kind
  descriptors uGBuf0→gt0.images[0], uGBuf1→gt1; :1067-1070 trace draw → gt0;
  :1076-1078 kb read from gt0.images[0]; :1083,1094 transitions gt0 only.
  FIXME(p7) x10.

## 3. Kind-set proof (per-kind shaded pixels = G-buffer kind mask = exactly
the pixels the kind shader keeps — its discard test is the same predicate)

| viewpoint | kinds present | 0 sky | 1 terrain | 2 tree | 3 rock | 4 water | 5 cover | total px |
|---|---|---|---|---|---|---|---|---|
| vp0 640x320 spp1 | {0,1,2,3,4,5} ⊃ {0} | 61,354 | 45,614 | 36,131 | 872 | **2,522** | 58,307 | 204,800 |
| vp2 640x320 spp1 | {0,1,2,3,4} | 77,253 | 20,489 | 4,136 | 793 | **102,129** | 0 | 204,800 |

vp0 unique kinds are a strict superset of {0}; vp2 includes the water kind (4).
Evidence: engine_vr/tmp/p7_kinds_vp0.json, p7_kinds_vp2.json, PNGs
p7kinds_vp0.png / p7kinds_vp2.png. gt0 channel stats (256x128 probe,
p7probe8): ch0 t∈[1.75, 2152], ch1 kind∈[0,5], ch2/3 = cell (−99999 sentinel
on misses) → gt0 holds (t, kind, cell.xy) — the GL attachment-0 layout.

## 4. Root cause B + determinism

BUG B was a consequence of BUG A, not an independent nondeterminism: with the
mis-wire, attachment 1 was never written, so kind passes sampled UNDEFINED
initial content / stale staging garbage (P6 measured images[1] all-zero at
readback while texelFetch saw different bytes). The discard predicate
\`int(g0.y+0.5) != SCENE_PASS\` and h.info inputs then varied run-to-run →
P6 baseline: mean|dRGB| 1.63/255 on 34% of pixels, max 231.

Re-measure after fix A: two identical renders
(\`vr.py --viewpoint 0 --res 640x320 --spp 1 --tile 192\`, separate processes,
separate outputs): PNG sha256 **identical**
(4652a627e79dd1715fbc90c0d9991b262ccd2d44d04d6b3b7b278b719c343578 both),
max_drgb = 0, mean = 0. JSON: engine_vr/tmp/p7_determinism.json. No residual.

Ordering/budget notes (GC hooks b/c): gt1 is RGBA32F, w·h·16 B (2048x1024 →
33.5 MB; the spec 8192x4096 → 512 MB resident), initial+permanent GENERAL
layout, never a color attachment, never transitioned, never read back (no
staging copy — only gt0 is read, vr.py:1077). Writer/reader table: writer
= scene_pano_trace (imageStore→gt1); readers = scene_pano_kind0_sky …
kind7_flora via uGBuf1 (all 8 kind programs, per manifest; no secondary-ray
pano programs). Same-frame mixed use is sequenced by submit+wait:
host.end() does vkQueueSubmit + fence wait (gl_vr.py:794-810) — the cone+trace
command buffer (incl. the gt1 write) is END-WAITed at vr.py:1072 before the
kind command buffer is even begun (vr.py:1084-1091), so the write→read hazard
is fenced, not merely queue-FIFO ordered.

## 5. Voided metrics + incident disclosure

| metric | P6 (voided) | P7 |
|---|---|---|
| vp0 dispatch kinds | {0} | {0,1,2,3,4,5} |
| exposure vp0 | 0.356–0.359 | 1.3405 |
| rerender diff | mean 1.63/255 @34%, max 231 | 0 (byte-identical) |
| 2048x1024@1spp wall | 132 s | 131.7 s (p7_2048.png) |

**Incident (disclosed):** a probe8 wrapper bug (\`g['main']([])\` — vr.py's
\`main(argv=None)\` treats [] as "no args") ran ONE full 8192x4096@1spp mono
render, violating the never-8192x4096 cap. It completed (exposure 1.284,
wrote engine_vr/out/pano_0_forest_mono_8192x4096.png); NOT used as evidence
anywhere in this report, and the artifact was deleted. All other GPU runs:
256x128 + 2x probe-class small renders, vp0/vp2 640x320, determinism pair
640x320, one 2048x1024@1spp — sequential, env block per brief
(ALPINE_CACHE=$PWD/cache_final ALPINE_MAX_PROC_GB=96
ALPINE_MIN_COMMIT_GB=24 ALPINE_MAX_GPU=1). The 2048 run took 131.7 s vs the
2-minute guidance (P6 baseline 132 s; overage is the pre-existing baseline,
not new cost).

## 6. SPIR-V parity, gate, cleanliness

* Parity: all 28 film program .spv byte-identical to HEAD (sha256 sweep of
  \`git show HEAD:engine_vr/spirv/*.spv\` vs working tree — only changed
  file: scene_pano_trace.spv); build_p4 independently printed "SPIR-V parity:
  38/38 hashes match". Evidence: engine_vr/tmp/p7_parity.json
  ({film_identical: true, film_programs: 28}).
* gate_p7: run unmodified — **no gate bugs found, no gate fixes made**.
  Final line: "GATE P7: PASS" (see run log).
* FIXME(p7) present in every modified .py/.frag (scene.frag x2,
  shaders_gen/scene_pano_trace.frag x2 — survives glstrip squeeze —, vr.py
  x10). Debug-constant grep-clean: P7CONST/11,12,13,14 exist only in the
  disposable tmp probes below; nothing in tracked shaders.
* Disposable untracked artifacts (listed per D8, all under engine_vr/tmp/):
  p7probe2/3/4/5/6/7/8.py, p7kinds.py, p7detcmp.py, p7parity.py,
  p7trace_c2.frag/.spv/.msl, p7mini2.frag/.spv, p7sc_bad.frag,
  p7_kinds_vp0/vp2.json, p7kinds_vp0/vp2.png, p7_det_a/b.png, p7_2048.png,
  p7probe8.png, p7pc/ (probe pipeline cache), mvkdump/ (MVK shader dumps),
  mvk/ (fetched MoltenVK 1.4.2 sources), p7*.out logs.
