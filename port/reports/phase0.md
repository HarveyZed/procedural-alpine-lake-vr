# Phase 0 report — phase0-coder / 2026-10-25 04:3x local (Asia/Shanghai)

## Status: DONE

Backend recommendation: **A — MoltenVK via python `vulkan` bindings** (§Recommendation).

## Files written

| path | LOC / size | role |
|---|---|---|
| tools/vk/env.sh | 17 | PATH + DYLD_LIBRARY_PATH + VK_DRIVER_FILES bootstrap (`source tools/vk/env.sh`) |
| tools/vk/bin/ | 6 bins | glslc, spirv-cross, spirv-val, spirv-dis, spirv-opt, glslangValidator (all `--version` exit 0 WITHOUT env — rpath @loader_path/../lib) |
| tools/vk/lib/ | ~46 MB | libvulkan (loader 1.4.363) + libMoltenVK (1.4.2) + glslang/SPIRV dylibs, install-names rewritten, re-signed |
| tools/vk/icd.d/molten_icd.json | 10 | authored ICD, `is_portability_driver: true` |
| tools/vk/pkgs/ | ~465 MB | 4 brew bottle tarballs + lunarg sdk payload lists (provenance) |
| spike/p0_a/run.py | 283 | MoltenVK host (instance→device→compute+render→readback→PPM) |
| spike/p0_a/run.sh | 9 | /usr/bin/time wrapper (SIP DYLD-strip workaround) |
| spike/p0_a/shaders/ | 58 | fill.comp 11, fs.vert 6, sphere.frag 41 — LOCKED scene, shared w/ B + ref |
| spike/p0_b/run.py | 138 | wgpu-py host incl WGSL (scene mirrored) |
| spike/p0_ref.py | 61 | numpy float32 CPU reference → spike/p0_ref.ppm |
| spike/p0_cmp.py | 52 | PSNR/max-diff comparator + buf_check rollup |
| port/reports/phase0.md | this | |

GPU contact: only spike/p0_a + spike/p0_b, 256², single frame each (user P0 exemption). engine/ never executed.

## Comparison table (REQUIRED)

| metric | A — MoltenVK | B — wgpu-py |
|---|---|---|
| stack | vulkan bindings 1.3.275.1 → loader 1.4.363 → MoltenVK 1.4.2 (reports Vulkan 1.4.357, Metal) | wgpu-py 0.32.0 → wgpu-native (naga), `backend_type=Metal`, Apple M5 Ultra |
| shader path | GLSL 450 → SPIR-V via glslc (shaderc v2023.8/v2024.3, glslang 11.1.0, spirv-tools v2024.4); validated SPIRV-Tools v2026.4 | WGSL → naga inside bundled libwgpu_native (no external compiler) |
| external toolchain to install | 511 MB under tools/vk/ (bottles + LunarG payload + ICD json + env vars) | none (wheel bundles native lib) |
| wall time (image out, /usr/bin/time -l real) | 0.46 s | 0.60 s |
| peak RSS (max RSS) | 58.5 MB (58,556,416 B) | 60.2 MB (60,178,432 B) |
| PSNR vs CPU ref (256² sphere raymarch) | PSNR 98.06 dB, max \|diff\| 1/255 | PSNR 98.06 dB, max \|diff\| 1/255 |
| A vs B | bit-identical (MSE zero → infinite dB) | — |
| compute fill (i*i % 997, 1024 f32) | 0 mismatches vs numpy | 0 mismatches vs numpy |
| host LOC (incl. shaders) | 283+58+9 = 350 | 138 |

Both ≥ 26 dB gate; both buffers exact. A-vs-B bit-identity means neither backend perturbs the
reference math at 256² (single-sample, no filtering).

## Toolchain route (Task 1) — what actually ended up in tools/vk

1. PyPI has no vulkan-loader/MoltenVK wheels (confirmed by constraint) → route 1 = brew **bottles**
   (fetch only, never install): `brew fetch` itself blocked (sandbox denies /opt/homebrew/var locks) →
   curl `formulae.brew.sh/api/formula/<f>.json`, take `bottle.stable.files.arm64_*` URL, ghcr.io needs a
   repo-scoped anon token (`GET /token?scope=repository:homebrew/core/<f>:pull`), then plain curl.
   Bottles: glslang 16.6.0, spirv-tools 1.4.363.0, vulkan-loader 1.4.363.0, molten-vk 1.4.2.
   Bottle binaries carried `@@HOMEBREW_PREFIX@@` install-name placeholders → `install_name_tool -id @rpath/… -add_rpath`
   surgery in bin AND lib dylibs, then `codesign --force -s -` on everything touched (required on arm64).
2. glslc: bottle glslang's bin has no glslc, and PyPI shaderc is absent → route 2 = LunarG SDK package:
   `https://sdk.lunarg.com/sdk/download/1.3.296.0/mac/vulkan_sdk.dmg` (versioned URL needs no login;
   the "latest" alias does). It is a ZIP wrapping InstallVulkan.app; the payload is
   `Contents/Resources/installer.dat` (Qt qres) with a 7z at offset 455410 whose nested 7z at +1096009
   holds the SDK; sliced both offsets, macOS bsdtar extracts `macOS/bin/glslc` + `spirv-cross` — both
   statically linked (only /usr/lib/libc++ + libSystem), zero fixups, re-signed. The GUI installer was
   NEVER run (no writes outside workspace). 266 MB dmg + intermediates deleted after extraction;
   `pkgs/lunarg/payload2_list.txt` kept as provenance.
3. Task-1 completion test PASSED: after `source tools/vk/env.sh`, `import vulkan` works; instance created
   with `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` + `VK_KHR_portability_enumeration` →
   1 physical device: "Apple M5 Ultra", apiVersion 1.4.357, loader reports 1.4.363; 4 queue families all
   GRAPHICS|COMPUTE|TRANSFER; `VK_KHR_portability_subset` enabled on device.
4. Probe: `glslc --target-env=vulkan1.2 .probe.comp -o .probe.spv && spirv-val .probe.spv` → OK.

## Deviations from brief

1. **/usr/bin/time -l needed a wrapper** for A: `/usr/bin/time` is SIP-protected → dyld strips
   `DYLD_LIBRARY_PATH` → `import vulkan` fails with "Cannot find Vulkan SDK version". Fixed with
   `spike/p0_a/run.sh` which re-sources env.sh *inside* time's child before exec'ing python. Numbers
   above are still from `/usr/bin/time -l`. B unaffected (bundled lib via @rpath).
2. **spike/p0_b: `queue.on_submitted_work_done_sync()` is broken in wgpu 0.32.0** — see frictions;
   replaced by `map_sync` (a full fence for prior submits).
3. **CPU reference writes `spike/p0_ref.ppm` flat** (gate's expected path) plus a copy under `spike/p0_ref/`.
4. `brew fetch --json=pretty` unsupported by this brew CLI → used formula-API JSON + ghcr token instead
   (same fetch-only contract; nothing installed).
5. Engine GPU/FS untouched; `glslangValidator`, `spirv-dis`, `spirv-opt`, `spirv-cross` added alongside the
   required glslc/spirv-val (extra Phase-1 headroom, tools/vk-only write).

## Gate self-check

`.venv/bin/python port/gates/gate_p0.py` — run twice: pre-report and final (this file present).
Final tail:

```
PASS toolchain glslc --version
PASS toolchain spirv-val --version
PASS glslc compiles 450 compute probe
PASS spirv-val accepts probe
PASS ppm spike/p0_a/sphere.ppm
PASS ppm spike/p0_b/sphere.ppm
PASS ppm spike/p0_ref.ppm
PASS buf check exists spike/p0_a/buf_check.txt
PASS buf check exists spike/p0_b/buf_check.txt
PASS report exists
PASS py_compile spike/p0_ref.py
PASS py_compile spike/p0_cmp.py
PASS py_compile spike/p0_a/run.py
PASS py_compile spike/p0_b/run.py
PASS no moderngl in new code — []
PASS read-only trees untouched — []

GATE P0: PASS
```

`spike/p0_cmp.py` exit 0: PSNR A↔ref 98.06 dB, B↔ref 98.06 dB, A↔B inf; BUF both mismatches=0; RESULT: PASS.

## Frictions hit (exact errors → resolution)

**Toolchain / A**
- `brew fetch --json=pretty X` → "Error: invalid option: --json" → formula API JSON + ghcr anon-token curl.
- `brew fetch X` → sandbox denied /opt/homebrew/var cache locks → fetch fully out-of-tree under tools/vk/pkgs.
- bottle `glslangValidator` is a **symlink** to `glslang` (bottle bin) — copying by name silently shipped a
  loader-erroring stub; renamed real binary, deleted placeholder.
- dyld: "library not loaded @@HOMEBREW_PREFIX@@/opt/…/libX" → install_name_tool -id/-change/-add_rpath + re-codesign (arm64 kills invalid sigs).
- `sdk.lunarg.com/...latest?platform=macOS` → 401 login wall → versioned URL 1.3.296.0 works anonymously.
- hdiutil: "无法识别映像" — the .dmg is a ZIP; then `installer.dat`: "Qt Binary Resource file"; zero PK-zip
  members; found 7z magic `37 7A BC AF 27 1C` by byte-scan; outer 7z held only Helpers/ → second 7z magic
  inside it at +1096009 held macOS/{bin,lib}. (No 7z binary needed: macOS bsdtar = libarchive.)
- bindings: `VK_API_VERSION_1_2` absent in 1.3.275.1 → `(1<<22)|(2<<12)`.
- bindings: property strings **auto-decode to str** → `vk.ffi.string(e.extensionName)` raised
  `TypeError: string() argument 1 must be _cffi_backend._CDataBase, not str` → use the str directly.
- bindings: `VkShaderModuleCreateInfo(pCode=…)` — pre-casting to uint32_t* raised
  `TypeError: a bytes-like object is required, not '_CDataBase'` (ctor casts internally) → pass raw bytes.
- bindings: `vkBeginCommandBuffer(cmd, info, None)` → "takes 2 positional arguments" → no pAllocator param.
- `/usr/bin/time` strips DYLD env (SIP) → run.sh wrapper (Deviation 1).

**B (wgpu-py)**
- `create_compute_pipeline(module=…)` → unexpected kwarg → 0.32 wants `compute=dict(module, entry_point)`.
- `GPUComputePassEncoder` has no `dispatch` → `dispatch_workgroups` (_classes.py:2127).
- `rp.draw(0, 3)` drew **nothing** (silent black frame, 3.9-dB-equivalent garbage): wgpu-py `draw(vertex_count, …)`
  first arg (apidiff deviation from WebGPU IDL order) → `rp.draw(3)`. THE scariest bug of the phase: no error,
  just a black image. Phase-1 PSNR gate against CPU ref is what caught it.
- `queue.on_submitted_work_done_sync()` → `TypeError: initializer for ctype
  'void(*)(WGPUQueueWorkDoneStatus, WGPUStringView, void *, void *)' must be a pointer to same type, not
  cdata 'void(*)(WGPUQueueWorkDoneStatus, void *, void *)'` (wgpu_native/\_api.py:4099) — real ABI bug in
  wgpu-py 0.32.0; do not use that call.
- `GPUAdapterInfo` has no `.backend` attribute (use `.backend_type`/dict repr).

**CPU ref**
- numpy float32 march overflowed on already-broken pixels (`RuntimeWarning: overflow … in multiply`) →
  freeze t after break (`alive` mask, spike/p0_ref.py:52-56). Comparator then reported 100% pixels within 1/255.

## API contracts relied on (binding + API + verified at)

- vulkan 1.3.275.1 `.venv/lib/python3.12/site-packages/vulkan/\_vulkan.py:104-111` — dlopen(`libvulkan.dylib`)
  bare soname ⇒ DYLD_LIBRARY_PATH=tools/vk/lib is the ONLY discovery mechanism (why env.sh + run.sh exist).
- _vulkan.py struct factories (`Vk…(sType=…)` auto-sType; bytes for char*) ; `vkMapMemory` :22553 returns
  `ffi.buffer`; `vkAllocateCommandBuffers` :12523 returns array (`[0]`); `vkGetDeviceQueue` :11376 auto `pQueue`;
  portability: instance flag `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`(0x1) + ext
  `VK_KHR_portability_enumeration` + device ext `VK_KHR_portability_subset` — verified live (device "Apple M5 Ultra").
- wgpu 0.32.0 `.venv/lib/python3.12/site-packages/wgpu/_classes.py`: `request_adapter_sync` :99,
  `request_device_sync` :575, compute pipeline `compute=dict` :1037, render pipeline :1067,
  `GPUCommandEncoder.draw` arg order :1876, `dispatch_workgroups` :2127, `copy_texture_to_buffer` :2043
  (bytes_per_row %256 — 256px RGBA8 = 1024 ✓), `map_sync` :1361 / `read_mapped` :1401 / `submit` :2294;
  broken `on_submitted_work_done_sync` → `wgpu/backends/wgpu_native/_api.py:4099`.
- gate: `port/gates/gate_p0.py:50-63` — report-lint sections/regexes; PPM list uses flat `spike/p0_ref.ppm`.
- shaders: SPIR-V target 1.5 via `--target-env=vulkan1.2` (glslc targets list, verified `spirv-val --version`).

## GLSL→WGSL translation gap log (engine/shaders grep → naga risk)

| engine feature (grep hits) | naga/WGSL outlook | risk |
|---|---|---|
| `texture(` 395 · sampler3D uNoise3 (heightfn) · texelFetch 18 · textureLod 4 · textureSize 1 | textureSample*/textureSampleLevel/textureLoad/textureDimensions — mechanical, but separate `sampler` objects: every `texture(s,uv)` call rewritten | MED effort / LOW mystery |
| coherent image2D 10 + imageStore 6 / imageLoad 3 (erosion_*.comp) | WGSL storage textures fine; **coherent/aliasing semantics absent**; MoltenVK-side image atomic order untested | **HIGH** — erosion is the one pass needing rethink |
| barrier() 11 + shared float sh[..] 30 (erosion_hill.comp:41-44,74,116) | `var<workgroup>` 1:1; compute barriers exist in WGSL but engine's memoryBarrier+coherent combo ≠ WGSL memory model | HIGH (same passes) |
| layout(local_size TS macro) (erosion_common.glsl:9) | workgroup_size needs literals → expand in glstrip | LOW |
| no std140/std430 blocks (plain `uniform sampler2D`), layout(binding) only 14 | A: glslc needs explicit bindings for most of 22 samplers (shaderc auto-bind opt or inject in glstrip); B: @group/@binding rewrite | MED (both backends) |
| gl_GlobalInvocationID 0 direct · gl_FragCoord 19 · out vec4 25 (MRT) | builtins direct; multi-`@location` outputs supported | LOW |
| #include 31 via glstrip.py custom preprocessor | A: **preprocessor + GLSL both survive verbatim** (compile GLSL→SPIR-V at runtime). B: WGSL has no #include → port glstrip anyway + full GLSL→WGSL rewrite of ~all engine shader LOC | **decisive for A** |
| imageAtomic* 0 · subgroups 0 · push_constant 0 | n/a — nothing scary in the gaps we feared | — |

## Risks / FIXME(port) list (hand-off, top 3)

1. **`engine/shaders/erosion_hill.comp:20,74` (+ erosion_\* siblings)** — `coherent image2D` + `barrier()` +
   memoryBarrier have no faithful WGSL mapping; under A they map to real Vulkan memory model BUT through
   MoltenVK→Metal the RMW/atomic ordering on `rg32f` coherent images is unverified (P0 scene had no atomics).
   Phase-1 first task: 256² erosion micro-spike through backend A vs numpy ref.
2. **`engine/glstrip.py` + binding coverage** — only 14 layout(binding) across the suite; Vulkan SPIR-V needs
   deterministic slots. Fix inside glstrip (inject bindings from the moderngl slot map it already parses).
   Phase-1 second task: compile real `engine/shaders/scene.frag` (9 variants) through glslc → SPIR-V, expect
   GL→Vulkan semantic fallout (no legacy `gl_\*` frag outputs, texture() LOD semantics ok).
3. **`spike/p0_b/run.py:117`** — wgpu-py 0.32.0 `on_submitted_work_done_sync` ABI break + `draw()` arg-order
   foot-gun (`spike/p0_b/run.py:120`): if the team ever switches to B, pin wgpu, add the CPU-ref PSNR gate to
   every pass, and re-test sampler3D noise + 3-texture MRT under naga (untested here).

## Recommendation

**Choose A (MoltenVK + python vulkan bindings + GLSL 450→SPIR-V via tools/vk/bin/glslc).** Top reasons:

1. **Shader-source economics**: the engine's ~5k LOC GLSL suite keeps compiling as GLSL — glstrip.py's
   `#include`/strip preprocessor stays the single transformation, then glslc → SPIR-V. Backend B forces a
   full GLSL→WGSL rewrite (~all engine shader LOC, 395 `texture(` call sites, no include system) and the
   erosion coherent/barrier semantics have no WGSL home (§gap log rows 2,3,7).
2. **Numerics proven identical where it matters**: A hit PSNR 98.06 dB / max 1/255 vs the numpy float32
   reference and was **bit-identical** to B; compute buffer exact. Vulkan's explicit memory/layout model is
   also what the engine's split-pass ping-pong design was written against.
3. **Toolchain risk already paid**: tools/vk/ is done, verified, relocatable, gate-green; the binding's
   ceremony cost (283 vs 138 LOC) is one-time host boilerplate that Phase 1 amortizes into a reusable
   `vk_util`, whereas B's savings are cosmetic next to its shader rewrite bill.

Honest counterweight for the record: B installed nothing (wheel-only), ran 0.60 s/60 MB (≈A), and is the
right fallback if MoltenVK image-atomic ordering (risk 1) fails — naga would then force the rethink anyway,
with a much simpler host layer.

— phase0-coder, session-30804b66 child