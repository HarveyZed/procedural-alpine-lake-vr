# RUN_FIRST — operator manual for the first GPU run (Phase 5)

Written 2026-10-25 by the Phase-5 round-2 coder. **Nothing in this file has
been executed.** Phases 0-5 ran under the zero-GPU / zero-ffmpeg discipline
(port/RUNBOOK.md §1), so every expectation below is derived from source, from
the phase 0-4 reports, or from the reviews — and every number that is not
sourced is marked ESTIMATE. This document is the hedge RUNBOOK §7 D6 asks for:
the erosion coherent read-modify-write path is UNVERIFIED on MoltenVK, and it
is the first thing the ladder tests.

How to read the ladder: work top to bottom, one step at a time. Each step has
a command, the observation that means it passed, and an **abort criterion**.
Do not skip ahead: the probes are ordered by review_p3/phase3 ranking
(silently-wrong first, loud-errors later), so a later pass does not excuse an
earlier failure. Canonical probe IDs are the `#N` headings; prose
cross-references use the short form (P7, P9 …) so the `#N` strings stay unique.

---

## 1. Environment

### 1.1 Loader + SDK (do this in every shell)

```bash
cd /path/to/procedural-alpine-lake
source tools/vk/env.sh        # VKSDK_LOCAL, PATH, DYLD_LIBRARY_PATH,
                              # VK_DRIVER_FILES=tools/vk/icd.d/molten_icd.json
vulkaninfo --summary | head -20   # optional sanity: MoltenVK 1.4.x visible
```

`tools/vk/icd.d/molten_icd.json` carries `is_portability_driver: true`, so
every device create must carry `VK_DEVICE_CREATE_ENABLE_PORTABILITY_FEATURES_BIT`
— gl_vr sets it (phase0: VK_KHR_portability_subset is enabled at gl_vr.py:594-597).
If you ever see `VK_ERROR_INCOMPATIBLE_DRIVER`, the ICD env is not set in that
shell: re-source tools/vk/env.sh rather than passing flags.

### 1.2 `ALPINE_*` variables (engine_vr/gpuslot_vr.py:8-9 + engine_vr/cache_schema.py)

Recommended column is tuned for **this** machine: Mac17,15, Apple M5 Ultra,
274877906944 B = 262144 MB = 256 GiB unified memory, 16384 B pages. Defaults
come from gpuslot_vr.py (`DEFAULT_MAX_PROC_GB = 16.0`, compile 50, min-commit 6,
max-gpu 3, `WATCHDOG_HEADROOM_GB = 6.0`, `WATCHDOG_FLOOR_GB = 2.5`,
`BUDGET_FACTOR = 0.75`).

| Variable | Default | What it actually does | Recommended here | Why |
|---|---|---|---|---|
| `ALPINE_CACHE` | `<repo>/cache_final` | The world cache. Resolved at **import** (cache_schema.py:67 `CACHE = cache_dir()`), and `bake_vr` re-exports it (`bake_vr.py:45 CACHE = cs.CACHE`). | `$PWD/cache_final` for the real bake; `/tmp/p5_mini` for the smoke bake | This workspace has **no** `cache_final`, so the first `vr.py` dry-run honestly reports `cache_missing`. Set it in the shell *before* python starts. |
| `ALPINE_MAX_PROC_GB` | 16 | Per-process hard budget. Watchdog cap = `min(ALPINE_MAX_PROC_GB, 0.75 × recommendedMaxMemorySize)`; exceeding it **hard-exits with code 3**. | **64** for the bake ladder, **96** for 8192×4096 renders | The 16 GB default is sized for the GL-era frames (`tools/render_final.py` sets 30 with the comment "full-res frames peak at ~17 GB in the driver (gpuslot watchdog)"). Here one 8192² RGBA32F bake target alone is 1024 MB and a pano render holds gbuf 1024 MB + acc 512 MB plus per-tile staging; 64/96 GB leaves the pipeline-compile spike room on a 256 GiB box. |
| `ALPINE_COMPILE_GB` | 50 | `compile_guard` (gpuslot_vr.py:368) waits for this much reclaimable RAM around `vkCreateGraphicsPipelines`; MoltenVK compiles SPIR-V→Metal there. | 50 (leave the default) | Fresh compile peaks 45-50 GB per the gpuslot_vr docstring. The guard keys repeat compiles off `cache/slots/compiled`, so only the first run pays. |
| `ALPINE_MIN_COMMIT_GB` | 6 | macOS has no Windows commit limit: this is **reclaimable RAM** (free + inactive pages) the slot acquirer waits for. | 24 | With 256 GiB unified, 6 GB is far below the driver's real appetite; 24 GB makes the queue behave like a queue instead of letting two 8K jobs thrash. |
| `ALPINE_MAX_GPU` | 3 | How many processes may hold a slot at once. | **1** for the whole first session | One noisy first run beats three interleaved validation logs. |
| `ALPINE_SLOTS` | `<repo>/cache/slots` | Directory holding the `O_EXCL` slot files + the `compiled` key cache. | leave default | Keep it on disk where the pipeline cache lives so the second run is cheap. |
| `ALPINE_NO_SLOT` | unset | `1` bypasses slot acquisition (and therefore the watchdog). | unset | The watchdog is the only thing that stops the first-run overshoot becoming a jetsam event. Use `NO_SLOT=1` only for a deliberately tiny reproduction. |
| `VK_LOADER_LAYERS_ENABLE` | unset | `VK_LAYER_KHRONOS_validation` turns on the loader's debug messaging. | set it for P2, P7 | Layout-mismatch is a **warning class** bug (phase4 §9.1), invisible without it. |
| `ALPINE_LAB`, `ALPINE_W2K` | unset | Debug knobs read at vr.py:553-554 (`_env_vec`, copied from engine/render.py:400-412). | unset | Defaults match the GL build. |

Import-order trap (verified by reading `cache_schema.py:62-76` and by an
import-order experiment, not on GPU): `cs.set_cache_dir(p)` rebinds
`cs.CACHE` **and** exports `ALPINE_CACHE`, but a module imported *before* that
call keeps its own stale `CACHE` copy. `vr.py --cache DIR` is safe because it
imports `passes_vr` / `bake_vr` lazily inside `_render()` (vr.py:858). For any
`python -c` driver, prefer the environment form
(`ALPINE_CACHE=/tmp/p5_mini .venv/bin/python -c ...`) — all commands below use
that form.

### 1.3 Copy-paste env block

```bash
cd /path/to/procedural-alpine-lake
source tools/vk/env.sh
export ALPINE_CACHE=$PWD/cache_final
export ALPINE_MAX_PROC_GB=64      # 96 for 8192x4096 renders
export ALPINE_COMPILE_GB=50
export ALPINE_MIN_COMMIT_GB=24
export ALPINE_MAX_GPU=1
unset ALPINE_NO_SLOT
mkdir -p "$ALPINE_CACHE"       # cache_schema never mkdirs; bake np.save needs it
                                          # (first-run lesson 2026-10-25: B0 died
                                          # FileNotFoundError h_far_raw.npy without it)
.venv/bin/python port/gates/gate_p5.py | tail -3    # must still say GATE P5: PASS
```

---

## 2. Step B — the bake ladder (nothing renders without it)

The workspace has no `cache_final`, so before any probe there is no world. The
chain is `bake_vr.bake_all()` = `bake_heights` → `bake_ao` → `bake_world_maps`
→ `bake_noise3d` (bake_vr.py:900), and `run_eroded_bake()` (bake_vr.py:167) is
the `tools/bake_final.py` equivalent: `bake_heights` wired to the VR erosion
callback. `vr.py` needs only `cache_schema.RENDER_REQUIRED`
(`h_far.npy`, `h_near.npy`, `water_level.npy`, `forest.npy`) ≈ 832 MB, but the
full cache is what the shader set was written against.

### B0 — mini bake on a scratch cache (the real first GPU contact)

~1/64 of the pixels of the real bake, and it exercises device open, the whole
SPIR-V→Metal compile, `bake_pass` readback order, `features.json`, the AO
double-descriptor and the array UBO members. Do this **before** the 8K bake.

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=/tmp/p5_mini .venv/bin/python -c "
from engine_vr import passes_vr as pv, bake_vr as bv
h = pv.open_host(log=print)
try:
    bv.bake_heights(far_res=1024, near_res=512, force=True)
    import numpy as np
    hf = np.load(bv.CACHE / 'h_far.npy'); hn = np.load(bv.CACHE / 'h_near.npy')
    bv.bake_ao(hf, hn, force=True)
    bv.bake_world_maps(hf, force=True)
    bv.bake_noise3d(force=True)
    print('mini bake done', hf.shape, float(hf.min()), float(hf.max()))
finally:
    pv.close_host()
"
```

- Expect: one `[gl_vr] device: Apple M5 Ultra … api=0x…` line, a
  `recommendedMaxMemorySize=… -> watchdog cap … GiB` line (that number is what
  §4 still needs), `heap …: budget=… GiB usage=… GiB` lines (VK_EXT_memory_budget
  is enabled at gl_vr.py:594-597 — if they print `0.0 GiB`, review_p2 N1 is
  back), `pipeline cache: loaded … B` on the first run (0 B expected, no cache
  yet) and `saved … B` on exit, then finite heights.
- **Abort criterion:** any Vulkan validation error, a non-zero exit, a
  non-finite height, or a watchdog exit code 3. Write down the
  `recommendedMaxMemorySize` line — §4's budget table has a hole where it goes.
- failure signature → source: `0.0 GiB` heap lines = review_p2 N1 (budget
  extension not enabled in that build); blank/black output with no error =
  review_p2 M1 (fullscreen VBO never bound in that build — fixed and
  re-verified, so a recurrence is a regression).

### B1 — the real bake

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final ALPINE_MAX_PROC_GB=64 .venv/bin/python -c "
from engine_vr import bake_vr as bv, erosion_vr as ev, passes_vr as pv
h = pv.open_host(cache_dir='$PWD/cache_final', log=print)
try:
    bv.run_eroded_bake(host=h, force=True)
    bv.bake_all(force=True)
finally:
    pv.close_host()
"
.venv/bin/python -c "from engine_vr import cache_schema as cs; print('missing:', cs.missing())"
```

- Expect: `missing: []`. `run_eroded_bake` writes `h_far_raw.npy`,
  `erosion_height.npy`, `flow/talus/sediment.npy` + `erosion_*.npy`,
  `h_eroded.npy`, `features.json`, `stream_level_far.npy`, `h_far.npy`,
  `h_near.npy`; `bake_all` adds AO (`ao_far/ao_near`, float16 saves at
  bake_vr.py:727), the world maps, `noise3d.npy`. Erosion runs five levels
  (512/1024/2048/4096/8192, `STAGES` at erosion_vr.py:497) and supports
  `checkpoint_dir=` / `resume_after=` — if the 8192 level dies, resume from the
  checkpoint instead of restarting the ladder.
- The GL-era reference figure in the code is **~6 min for erosion**
  (`bake_vr.py:236` comment) and **~1 min GPU for trees** (phase4 §9.4).
  Everything else on this device is ESTIMATE: far bake = 1024 tiles of 256 with
  one draw + one readback each (bake_heights tile=256, bake_vr.py:211-212).
- **Abort criterion:** `RuntimeError: main lake flood leaked to the map border`
  — the raise is bake_vr.py:774, inside `bake_world_maps` (def at
  bake_vr.py:755) — means the height map is wrong, not the flood fill: stop
  and re-check P1/P3 results before re-baking.
- failure signature → source: `features.json missing in <dir>` from `_render`
  is the documented late symptom of a half-finished bake (vr.py cache guard).

### B2 — trees (runtime-baked, not part of `bake_all`)

`bake_vr.ensure_trees()` (bake_vr.py:948, `bake_trees` 4096² tile 1024 comps 4)
writes `trees.npy`; `bake_shadows()` (bake_vr.py:968) is runtime-only and
expects the 12-sampler list from `_trees_textures` (bake_vr.py:925, including
the `_zero4` dummy). `vr.py` calls both on first use, so the first P7 render
pays for them; a re-run should print a pipeline-cache hit instead.

---

## 3. Step 0 — `check_viewpoints.py` (CPU, zero GPU, before any probe)

`engine_vr/check_viewpoints.py` answers one question cheaply: does each of the
nine `vr.py` viewpoint recipes sit above the ground/water of the *baked*
cache, or inside an eroded-away / never-baked surface? It imports `vr` and
`cache_schema` only (both lazy about Vulkan), so it runs with no GPU, no
vulkan environment and no cache. Round 2 kept the interface and fixed two
things (see the module docstring): it now separates a real burial from the
brief's threshold flag, and it prints which cache map each sample came from
plus whether that map still matches `cache_schema`.

**Read the classification, not just the exit code.** port/PHASE_5.md T2
mandates "flag < 5 m as buried", but six of the nine recipes *stand* at agl
1.0-2.2 m by design — `forest`/`reveal`/`meadow`/`finale` are placed through
`world.P(x, z, agl)` (vr.py:312/324/378/404, so their clearance **equals** the
agl the source passes: 1.9/1.0/1.6/1.7 m, immune to any re-bake), and
`skim`/`waterfall` sit at an absolute y of 1.7 / 2.2 m (vr.py:333/356),
`stream` at `max(ground, 0) + 1.5` (vr.py:371). With the mandated 5 m default
a perfectly healthy cache therefore flags seven recipes and step 0 would abort
the ladder on a correct world. Run both forms:

```bash
# 3a. diagnostic form (the T2 5 m threshold, kept as specified)
cd /path/to/procedural-alpine-lake && .venv/bin/python engine_vr/check_viewpoints.py --cache "$PWD/cache_final"

# 3b. pass/fail form - this is the one the ladder gates on
cd /path/to/procedural-alpine-lake && .venv/bin/python engine_vr/check_viewpoints.py --cache "$PWD/cache_final" --nominal-ok --min-clearance 0.5
```

- Expect (3b): three map lines `h_far.npy shape=(8192, 8192) ... contract
  (8192, 8192) cell 4.00 m ok`, `h_near.npy` at 4096² cell 1.00 m,
  `water_level.npy` at 8192², `features.json present`, then nine lines of
  `pos / surface [which map] / clearance / flag` and `RESULT: ALL CLEAR`.
  Expect (3a): the standing recipes report `BELOW-THRESH` with their nominal
  stance quoted — that is the threshold, not a burial; `flyover`/`peaks` are
  airborne (y = 620 / 700) and pass either way.
- Exit codes: 0 = nothing flagged; 1 = `BELOW-SURFACE` (head under the
  surface), `BELOW-THRESH`, `UNKNOWN` (the recipe's own resolve raised), or a
  map `SPEC-MISMATCH`; 2 = cache incomplete or `PanoWorld` failed to load.
- **Abort criterion:** any `BELOW-SURFACE`, `UNKNOWN` or `SPEC-MISMATCH` stops
  the ladder (3b exit 1 or any exit 2). A head inside terrain renders a
  plausible-looking muddy pano that every later probe would then "pass".
  `BELOW-THRESH` in 3a on a standing recipe is *not* an abort.
- failure signature → source: `BELOW-SURFACE` on `skim` is a water-surface
  question (`water_level.npy`, matrix row 23, `bake_world_maps` flood-fill
  raise at bake_vr.py:774); on `waterfall`/`stream` it is the carve/features vs height
  ordering (matrix 4/22); `UNKNOWN` naming `features` = the bake wrote heights
  but no `features.json` (matrix 4); `SPEC-MISMATCH` on every map = B1 baked at
  a resolution other than the contract, which invalidates every number below it
  (matrix 2/20). `BELOW-SURFACE` on a *placed* recipe is impossible by
  construction, so if you ever see it, the cache changed between the resolve
  and the sample — re-run.
- Static behaviour of the tool itself is verified CPU-only against a
  synthetic cache, and these two lines ARE the verification — run them as
  written. `mk_fakecache.py` writes 64² constant maps, which are below the
  cache_schema contract, so the map-shape guard has to be told to stand down:
  `--no-spec-check` exists for exactly this (check_viewpoints.py:169-171,
  "synthetic test caches only") and is part of the documented form here, not
  of the real-cache form above.

```bash
# A. height 12 -> the two absolute-y cameras sit under the surface: exit 1
cd /path/to/procedural-alpine-lake && .venv/bin/python engine_vr/tmp/mk_fakecache.py 12 && .venv/bin/python engine_vr/check_viewpoints.py --cache "$PWD/engine_vr/tmp/fakecache" --nominal-ok --min-clearance 0.5 --no-spec-check

# B. height 0 -> nothing buried: exit 0
cd /path/to/procedural-alpine-lake && .venv/bin/python engine_vr/tmp/mk_fakecache.py 0 && .venv/bin/python engine_vr/check_viewpoints.py --cache "$PWD/engine_vr/tmp/fakecache" --nominal-ok --min-clearance 0.5 --no-spec-check
```

  Measured on this host (2026-10-25, CPU only): A prints `vp2 skim ...
  clearance= -10.3 m BELOW-SURFACE` and `vp3 waterfall ... -9.8 m`, then
  `RESULT: 2 flagged`, exit **1**; B prints `RESULT: ALL CLEAR`, exit **0**.
  Without `--no-spec-check` BOTH fixtures exit 1 on the spec line alone (three
  `SPEC-MISMATCH` map lines; A then reads `RESULT: 3 flagged`) — that is the
  map-shape guard doing its job, and it is why review_p5 N1 called the earlier
  wording unreproducible. Real burials flag with or without the flag.
- Pre-bake it is still useful: `--dry-run` prints the nine recipes with their
  stance rule and `vr.py` cite, the cache files it *would* sample, what is
  missing right now, and the two command lines it suggests for the **real**
  cache (no `--no-spec-check` there — a real cache has to pass the shape
  guard). It exits 0, which is what the static gate runs
  (port/gates/gate_p5.py:44).

---

## 4. Memory and time budget

`Source` column: **code** = arithmetic on shapes/dtypes read from source;
**report** = quoted from a phase/review report; **machine** = measured on this
host in Phase 5 without touching the GPU; **ESTIMATE** = judgement, no
measurement. There is no `cache_final` in this workspace, so bake sizes are
derived from `cache_schema.FILES`, never from an existing cache.

| Item | Size | Source |
|---|---|---|
| Unified memory | 262144 MB (256 GiB) | machine (`hw.memsize` 274877906944) |
| `recommendedMaxMemorySize` → watchdog cap | **MEASURED: field absent** in vulkan 1.3.275.1 bindings (Vulkan-1.4 maintenance8) → logged 0.0, fallback = `ALPINE_MAX_PROC_GB` (gl_vr first-run fix); real headroom via VK_EXT_memory_budget: **heap 0 budget 222.7 GiB** | first run 2026-10-25 |
| Watchdog rule | cap = min(`ALPINE_MAX_PROC_GB`, 0.75 × recommendedMax), 6 GB headroom, 2.5 GB floor, 5 s poll, hard exit 3 | code (gpuslot_vr.py) |
| Fresh shader-compile spike | 45000-50000 MB (45-50 GB) | report (gpuslot_vr docstring, ALPINE_COMPILE_GB default 50) |
| GL-era full-res frame driver peak | ~17000 MB | report (tools/render_final.py comment) |
| Erosion ~6 min, trees ~1 min GPU | **MEASURED: erosion 5 levels 35.8 s**; AO far/near 0.4/0.2 s; forest 8192² 0.5 s; noise3d <0.1 s | first run 2026-10-25 (bake group, see reports/first_run.md) |
| Full bake on disk | ~4.0 GB total | code (cache_schema FILES sum) |
| `RENDER_REQUIRED` minimum | ~832 MB (h_far 256 MB + h_near 64 MB + water_level 256 MB + forest 256 MB) | code |
| `h_far_raw/h_eroded/h_far/erosion_*/flow/talus/sediment/water_level/stream_level_far` | 256 MB each (8192² f4) | code |
| `forest.npy` | 256 MB (8192² u8 RGBA) | code |
| `trees.npy` | 256 MB (4096² f4 RGBA) | code |
| `stream_near.npy` / `cones.npy` | 128 MB each (f2) | code |
| `ao_far.npy` + `ao_near.npy` | 32 MB each (4096² f2) | code |
| `noise3d.npy` | 8 MB (128³ u8 RGBA) | code |
| pano `gbuf` at 8192×4096 (2 attachments RGBA32F) | 1024 MB (2 × 512 MB) | code + report (phase4 §9.4 says gbuf 2×512 MB) |
| pano `acc` at 8192×4096 RGBA32F | 512 MB | code + report (phase4 §9.4) |
| cone pass, quarter-res R32F (2048×1024) | 8 MB | code |
| bake target 8192² RGBA32F | 1024 MB (phase3 writes 1.07 GiB) | code + report (phase3 P3) |
| erosion `imgAcc`/`imgTrack` 8192² RG32F | 512 MB each | code |
| one tile readback 192² RGBA32F | 0.5625 MB | code |
| tiles per sample at 8192×4096, `--tile 192` | **946** = ceil(8192/192) × ceil(4096/192) | code — note phase4 §8/§9 write **448**; see §6 |
| readback traffic per sample per eye | ~532 MB (946 × 0.5625 MB) | code |
| 8192×4096 mono `--spp 1` wall time | **BLOCKED by P6** (march runaway at full res, port/PHASE_6.md). Partial measurement: 1267 draws/39.2 s before death; SBS 4096-per-eye measured 69.4 s for 16.8 Mpx total ⇒ mono 8192×4096 ≈ 2.5-3 min/sample when P6 lands. | first run; P6 pending |
| Pipeline-cache reuse | **MEASURED PASS**: run 2 logs `pipeline cache: loaded 3609489 B`; ~40 s compile avoided | first run | (M2 fixed) |
| Host RSS, SBS 4096×2048/eye spp1 | 10.2 GiB peak (run with ALPINE_MAX_PROC_GB=96: watchdog counts host RSS, driver budget is separate) | first run |
| Host RSS, 8192×4096 tile loop | 2.1 GiB and flat (no leak) - P6 hang is GPU-side | first run |

Two rules of thumb for the session: keep `ALPINE_MAX_GPU=1` while anything
8K-sized is live, and treat a watchdog exit code 3 as a *result*, not a crash —
it means the budget you set is smaller than the pass, so raise
`ALPINE_MAX_PROC_GB` deliberately rather than re-running blind.

---


> OPERATOR (first-run): do NOT wrap GPU steps in /usr/bin/time -l - SIP strips
> DYLD_* and 'import vulkan' dies ("Cannot find Vulkan SDK version"). Measure
> in-process: resource.getrusage(RUSAGE_SELF).ru_maxrss (bytes on macOS).

## 5. The probe ladder

Numbering is global and frozen: P1-P6 are `phase3.md` §"Open risks" and P7-P13
are `phase4.md` §8. Commands are copied verbatim from those reports where the
report gives one (marked *verbatim*, with the line), and derived from the real
argparse / function signatures otherwise (marked *derived*, with file:line).
No probe was run.

#### #1 — rg32f/r32f storage writes + coherent read-modify-write (the D6 hedge)

- Command (adapted per R1 from port/reports/phase3.md:278-288):

> Note (R1 fix, review_p5.md S9): probes #1/#2/#6 use the REAL h_far_raw
> loaded by run_erosion(hfar=None) (erosion_vr.py:436-438; exists after B1),
> NOT a synthetic cone. A cone (apex at the lake seed LAKE_C=(110.8,0),
> LAKE_LEVEL=0.0) stays dry at the seed -> lake protection raises
> 'lake: seed ... is dry in the input' (erosion_vr.py:730-731) before any
> level math; the tarn level (520 m) would be the next landmine (:734).

```bash
cd /path/to/procedural-alpine-lake && .venv/bin/python -c "
import numpy as np
from engine_vr import erosion_vr as ev
out = ev.run_erosion(cell_m=4.0, stages=[ev.STAGES[0]], stop_after=512,
                    save=False)
h, fl = out['height'], out['flow']
print('finite', bool(np.isfinite(h).all()), 'h', float(h.min()), float(h.max()),
      'flow_abs_sum', float(np.abs(fl).sum()))
"
```

- Expect: `finite True`, `h.max()` below the synthetic 300 m cone top but above
  `h.min() >= 0`, and `flow_abs_sum > 0` and finite. The cone must still be
  recognisably a cone, now eroded.
- **Abort criterion (author's words, phase3.md:291): "Anything else stops the
  port."** A flat flow field or a NaN height is the silent failure this probe
  exists for.
- failure signature → source: **this is not an atomics question.** review_p3 N1
  corrected phase3.md:266-268: `imgAcc` is `rg32f` neighbour-gather `imageLoad`
  (`erosion_accum_flow_only.comp:97`) plus a per-thread **coherent**
  `imageStore` (:144) on a coherent `image2D` (:62) — "there are zero atomic
  operations in every shader of this port" (phase1). The residual question is
  coherent load/store visibility, compute→sample re-read, and MoltenVK
  `R32G32_SFLOAT` storage support. Do **not** go hunting
  `VK_FORMAT_FEATURE_*_ATOMIC_BIT` (review_p3 N1 explicitly warns the P4
  failure matrix would chase the wrong thing there). Also phase1: no atomic
  fallback exists anywhere, so there is no "degraded mode" to fall back to.

#### #2 — layout/validation chain across the compute↔sample round trip

- Command (*derived* — runs the same fill→accumulate→hill→read_state chain
  twice at 512² so the level switch re-uploads a compute-written r32f and
  re-samples it, which is the round trip phase3 describes; validation on):

```bash
cd /path/to/procedural-alpine-lake && VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation .venv/bin/python -c "
import numpy as np
from engine_vr import erosion_vr as ev
out = ev.run_erosion(cell_m=4.0, stages=list(ev.STAGES[:2]), stop_after=1024,
                    save=False)
print('finite', bool(np.isfinite(out['height']).all()), 'flow_abs_sum', float(np.abs(out['flow']).sum()))
"
```

- Expect: `finite True`, flow sum > 0, and **no** validation output at all. The
  two-stage run forces the level switch, i.e. `write_h` re-upload
  (erosion_vr.py:310-317, 2 × res² × 8 B per switch) plus the sampled re-read of
  a compute-written `r32f`.
- **Abort criterion:** any message in the "binding an image whose layout
  differs from the write-time layout" class, or a layout mismatch at all.
- failure signature → source: `write_descriptors` freezes layouts once per
  program (the `to_sampler_layout` unknown→`BOTTOM_OF_PIPE` fallback,
  passes_vr.py:417-447) and `barrier_compute_to_read` names only the written
  images (erosion_vr.py:232) — review_p2 N10's always-firing sampled-read
  barrier is the harmless member of the same family. phase2 M4
  (upload barrier TRANSFER→BOTTOM_OF_PIPE) is the torn-first-use variant that
  only shows on out-of-order drivers.

#### #3 — bake target lifetime + readback fidelity at big resolutions

- Command (*derived* from `bake_vr.bake_heights(far_res, near_res, force)` at
  bake_vr.py:200, which calls `_bake_pass("bake_height.frag", far_res, …,
  tile=256)` — exactly the "analytic far bake alone" phase3 asks for, plus the
  spec check):

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=/tmp/p5_p3 .venv/bin/python -c "
import numpy as np
from engine_vr import bake_vr as bv, cache_schema as cs, passes_vr as pv
h = pv.open_host(log=print)
try:
    bv.bake_heights(far_res=2048, near_res=1024, force=True)
finally:
    pv.close_host()
raw = np.load(bv.CACHE / 'h_far_raw.npy')
print('raw', raw.shape, raw.dtype, float(raw.min()), float(raw.max()))
print('spec', cs.spec('h_far_raw'))
patch = raw[1000:1064, 1000:1064]
print('patch mean/std', float(patch.mean()), float(patch.std()))
"
```

- Expect: shape/dtype match `cs.spec("h_far_raw")` for the resolution you
  requested, finite range, and a 64×64 patch that you can compare against a
  CPU evaluation of the same analytic field at those texels' world
  coordinates. The axis helper is `bake_vr.world_xy(res, half)`
  (bake_vr.py:732-734 — `(i + 0.5) / res * 2 - 1) * half`, same for cols/x
  and rows/z); the texel size is `cs.cell("h_far_raw")`
  (cache_schema.py:178-181) and the expected shape is
  `cs.shape_of("h_far_raw")` (:184-193). `cache_schema` has NO
  world-coordinate helper of its own (review_p5 N9). Row 0 must be the
  **first uploaded row** (phase2 addendum:
  "readback row 0 = upload row 0") — no flip.
- **Abort criterion:** a flipped or transposed patch, a spec mismatch, or a
  resident 1024 MB target that is never released (watch the `heap … usage=`
  lines across passes: `destroy_target` must bring the peak back down, and
  `bake_pass` now runs under `try/finally` for that reason).
- failure signature → source: review_p3 N3 is exactly this (no `try/finally`
  leaked a 1.07 GiB target); review_p3 N6 covers the square-image-only
  `_upload`/`_tex` assumption that would show up as a shape error rather than
  a wrong image; review_p2 C5 (the 20-word `_RUSAGE_WORDS` buffer at
  gpuslot_vr.py:81, zero headroom) is the readback-adjacent accounting seam.

#### #4 — double descriptor for one image (`uH` == `uHFar`) + f16 saves

- Command: the `bake_ao` line of **B0** already is this probe (bake_vr.py:711
  binds the same `tf` as `uH` and `uHFar` at :721, saves both AO maps as
  float16 at :727). Re-run it alone with validation on if you skipped B0:

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=/tmp/p5_mini VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation .venv/bin/python -c "
import numpy as np
from engine_vr import bake_vr as bv, passes_vr as pv
h = pv.open_host(log=print)
try:
    hf = np.load(bv.CACHE / 'h_far.npy'); hn = np.load(bv.CACHE / 'h_near.npy')
    aof, aon = bv.bake_ao(hf, hn, force=True)
    print('ao', aof.shape, aof.dtype, aon.shape, aon.dtype)
finally:
    pv.close_host()
"
```

- Expect: no validation complaint about aliasing two bindings of one image,
  `ao_far.npy` / `ao_near.npy` written float16 at half the far resolution
  (`hfar_res // 2`, `uStep0 = 8.0`), values inside [0, 1] after normalisation.
- **Abort criterion:** a sampler-object/descriptor validation error, or an AO
  map that is all-zero (the second binding reading a stale view).
- failure signature → source: review_p2 C6 (sparse descriptor sets keyed by
  `firstSet=0`) is the historical form of "two descriptors, one image"; the
  current code raises on attachment-count coupling (review_p2 M6 → raise).
  Duplicate sampler self-transition is legal per review_p3 N4 — do not treat a
  self-transition message as this failure.

#### #5 — array UBO members on a real device (`uStream` 96×vec4, `uTrail` 32×vec4)

- Command (*derived*: `bake_world_maps` runs `bake_forest.frag` with
  `uStream`/`uStreamN`/`uTrail`/`uTrailN` from `features.json`, bake_vr.py:837-839;
  needs a bake first, so run it on the mini cache):

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=/tmp/p5_mini .venv/bin/python -c "
import json, numpy as np
from engine_vr import bake_vr as bv, passes_vr as pv
meta = json.loads((bv.CACHE / 'features.json').read_text())
print('stream points', len(meta.get('stream', [])), 'trail points', len(meta.get('trail', [])))
h = pv.open_host(log=print)
try:
    bv.bake_world_maps(np.load(bv.CACHE / 'h_far.npy'), force=True)
finally:
    pv.close_host()
f = np.load(bv.CACHE / 'forest.npy')
print('forest', f.shape, f.dtype, 'coverage', float((f[..., 0] > 0).mean()))
"
```

- Expect: `stream points` > 0 so `uStreamN` is non-zero, a forest coverage in
  the same ballpark as the GL build's, and no `uTrailN`/`uStreamN` misread (a
  misread shows as forest hugging the stream everywhere, or nowhere).
- **Abort criterion:** coverage ~0 or ~1, or a `write_uniforms`/offset assert.
- failure signature → source: phase3 verified the CPU-side offsets
  (`uStreamN` at 1600, `uTrailN` at 2096, stride 16) and this probe is its
  device half; review_p2 C2 (inverted UBO size guard) and C4 (`vkMapMemory`
  re-wrap) are the two historical ways an array member silently truncated.
  review_p3 N5 is the documentation-level sibling (the plan docstring
  overstates `uHasErosion`/`uStreamN` — read the code, not the docstring).

#### #6 — per-band submit cost at 8192² (submission-bound, not correctness)

- Command: time the hillslope half of the P1 probe at the 8192 level, i.e.
  keep P1's 4.0 grid (`np.mgrid[-16384.0:16384.0:4.0, ...]` → 8192² at
  4 m/px) and `cell_m=4.0` from #1 **unchanged**, replace
  `stages=[ev.STAGES[0]]` with `stages=[ev.STAGES[-1]]` and add
  `import time` bookkeeping:

```bash
cd /path/to/procedural-alpine-lake && .venv/bin/python -c "
import numpy as np, time
from engine_vr import erosion_vr as ev
t = time.time()
out = ev.run_erosion(cell_m=4.0, stages=[ev.STAGES[-1]], save=False)
print('sec', round(time.time() - t, 1), 'finite', bool(np.isfinite(out['height']).all()))
"
```

- Expect: a wall time you can divide by the number of bands. `compute_bands
  submits one submission per band (passes_vr.py:365-391 → 5 submits per pass at
  8192² with `EROSION_BAND_TILES = 128` / `EROSION_MAX_DISPATCH_ROWS = 2048`).
- The grid size is code truth here (review_p5 N6): the level index comes from
  the **input** shape — `res = h0.shape[0]` (erosion_vr.py:1470), then
  `r = st["res"]`, `f = res // r`, `cell = cell_m * f`,
  `final = r == res` (:1485-1488), with `STAGES[-1]["res"] = 8192` (:523).
  A step-8.0 grid is 4096², so `f = 0` and `_pool` divides by zero at
  erosion_vr.py:533 — the probe dies before it opens a device, and the
  `cell_m=8.0` alternative the review offered cannot rescue it (`f` is
  shape arithmetic, not a cell value). An 8192² input at `cell_m=4.0` is
  also the self-consistent declaration: at `f = 1`, 4 m is exactly the
  far-map contract (`2 * FAR_HALF / FAR_RES`, cache_schema.py:37-38) and
  `run_erosion`'s own default (erosion_vr.py:439-440).
- **Abort criterion:** none for correctness — this probe decides whether bands
  need batching. Stop only if the wall time is > 10× the GL cadence, which
  makes the bake ladder impractical.
- failure signature → source: phase3 §Risks names the fix if it is slow
  (per-band UBO/push constants, **not** a layout change) and the level-switch
  re-upload cost (erosion_vr.py:310-317) as the other submit-side suspect.

#### #7 — pano draw-chain smoke + validation layout chain

- Command (*verbatim*, port/reports/phase4.md:202):

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation .venv/bin/python vr.py --viewpoint 0 --res 512x256 --spp 1 --out engine_vr/out/p4_p7.png
```

- Expect: exit 0, a **non-black** PNG, zero validation complaints. This is the
  first run of 12 frozen descriptors + per-tile auto-transitions + per-sample
  gbuf SRO flips — the largest layout surface in the port (phase4 §9.1) — and
  it also pays for `ensure_trees` + `bake_shadows` (B2).
- **Abort criterion:** any validation error, or an all-black/all-white frame.
- failure signature → source, in the order to suspect:
  1. a Python `IndexError` before the first `vkCmd*` = review_p4 C1, the 1-D
     footprint upload at passes_vr.py:165 (`comps = 1 if a.ndim == 2 else
     a.shape[2]`), fixed by the `reshape(-1, 1)` in `upload_footstrip` — the
     "loud error" tier here is a Python exception, not validation
     (review_p4 §5 verdict on C1);
  2. a validation **warning** about a frozen layout = phase4 §9.1, documented
     fallback is gbuf final layout `COLOR_ATTACHMENT_OPTIMAL` + an explicit
     barrier before readback. The `TRANSFER_SRC` -> `SHADER_READ_ONLY`
     mapping phase2 recorded lives in `to_sampler_layout`,
     passes_vr.py:422-452 (TRANSFER_SRC branch :442-444), and the
     readback-hot `TRANSFER_SRC` final layout is `make_target`'s default
     (passes_vr.py:264-266 -> `Target.final_layout` gl_vr.py:446-462);
     gl_vr.py:426-445 is image bind + ImageView creation, not the fallback
     (review_p5 N4);
  3. `RuntimeError: cache incomplete: …` or `features.json missing in …` = B1
     did not finish (vr.py `_render` cache guard);
  4. a black frame with no diagnostics = review_p2 M1 class (fullscreen VBO
     never bound) or M4 (upload barrier) — both were fixed and re-verified in
     the P2 fix round, so a recurrence is a regression, not a new risk.

#### #8 — equirect orientation (RUNBOOK §6 signs end to end)

- Command (*verbatim*, port/reports/phase4.md:211):

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final .venv/bin/python vr.py --viewpoint 6 --res 1024x512 --spp 1 --out engine_vr/out/p4_p8.png
```

- Expect (RUNBOOK §6: lon ∈ [-π, π) increasing east, lat +π/2 at row 0 down to
  −π/2 at row H−1, pixel centres at (x+0.5, y+0.5)):
  - row 0 = zenith (sky), row H−1 = nadir (ground), horizon near the mid rows;
  - column 0 and column W−1 are continuous across the seam (no cut, no
    mirrored half);
  - a quick numeric guard:

```bash
cd /path/to/procedural-alpine-lake && .venv/bin/python -c "
import numpy as np
from PIL import Image
a = np.asarray(Image.open('engine_vr/out/p4_p8.png').convert('RGB')).astype('float32')
lum = a.mean(2)
H, W = lum.shape
print('row0', round(float(lum[0].mean()), 1), 'mid', round(float(lum[H//2].mean()), 1), 'last', round(float(lum[-1].mean()), 1))
print('seam |d|', round(float(np.abs(lum[:, 0] - lum[:, -1]).mean()), 2))
"
```

  i.e. `row0` (sky) brighter than `last` (ground), and a small seam delta.
- **Abort criterion:** ground at the top, sky at the bottom, a discontinuous
  seam, or a mirrored panorama.
- failure signature → source: three independent layers can each flip Y — GL
  row order in the shader branch, the `engine/render.py:505` readback flip, and
  the lon ordering (phase4 §8). review_p4 N2 is the misleading-documentation
  variant: the vr.py probe docstring claimed `probe.frag` reads `uPixAng`, but
  it does not, and `uPixAng` is dead in every PANO variant — do not debug
  through that member.

#### #9 — `uCamRot` mat3 convention + roll sign (highest-impact silent-wrong)

- Command (*verbatim*, port/reports/phase4.md:219):

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final .venv/bin/python vr.py --viewpoint 2 --res 1024x512 --spp 1 --out engine_vr/out/p4_p9.png
```

- Expect: vp2 is the skim with `roll = -1.5` deg (vr.py `VIEWPOINTS`, :415). Fit
  the horizon line (fit a line to the strongest horizontal luma gradient in the
  middle third of the frame); the tilt must match the direction the GL film
  skim frame leans, and its magnitude must be
  `atan2`-consistent with 1.5 deg at this resolution (1.5 deg ≈ 13 px of rise
  across 512 rows — ESTIMATE: exact px depend on the fitted band).
- CPU reference, in this order:
  1. **Primary: `vr.basis(pos, target, roll)`** (vr.py:63) — its **rows are the
     GLSL mat3 columns** (`np.stack([r, u, f])`), the moderngl convention the
     engine uses. Feed that to your own equirect ray generation.
  2. Secondary cross-check only: `engine_vr/pano.py` `equirect_directions`.
     Per review_p4 N1 it used `dirs @ basis.T`, the inverse rotation, so **any
     CPU-vs-GPU ray check built on it false-failed**. That is now fixed at
     `pano.py:100` (`dirs @ basis`, re-verified max diff 2.98e-08), and
     review_p4 §7 then allowed this probe to use it — but the round-2 brief
     instruction and this manual keep `vr.basis` as the reference so the check
     stays independent of the helper that was once wrong. See §6.
- **Abort criterion:** opposite-sign tilt, or a tilt that is right for a
  transpose of `uCamRot` (a transposed mat3 shows as a mirrored/rotated world
  that still looks plausible — this is precisely the "silently wrong" case).
- failure signature → source: phase2 §Risks `gl_vr.py:285-323` — the mat3
  branch of `Std140Packer._write`, `gl_vr.py:311-317` (GLSL column c ←
  numpy row c), called "highest impact" for `uCamRot`, which is the UBO member
  at byte **offset 1088** (manifest.json:315-317, type mat3) — not a
  `gl_vr.py` line number (review_p5 N3);
  review_p2 R2 verdict: the mat3 convention itself was reviewed correct, so a
  failure here points at vr.py's basis construction or the roll sign, not the
  pack; review_p4 N1 for the `equirect_directions` trap.

#### #10 — light-probe parity

- Command (*derived*: reads the sky patch out of the P7 render and compares
  against the probe floats the same program wrote; `vr.py` reads the probe
  target as `pa = pv.read_image(ptar.images[0])[0]`, `sun, sky, hor =
  pa[0][:3], pa[1][:3], pa[2][:3]`):

```bash
cd /path/to/procedural-alpine-lake && .venv/bin/python -c "
import numpy as np
from PIL import Image
a = np.asarray(Image.open('engine_vr/out/p4_p7.png').convert('RGB')).astype('float32')
H, W = a.shape[:2]
zen = a[:H//16].reshape(-1, 3).mean(0)
hor = a[H//2 - 4:H//2 + 4].reshape(-1, 3).mean(0)
print('zenith mean RGB', zen.round(3))
print('equator mean RGB', hor.round(3))
"
```

- Expect: zenith reads as sky and the equator band as horizon/sun-ward in the
  same base look, i.e. the same ordering (sun > sky > horizon) as the P3 probe
  floats for that look.
- **Abort criterion:** an inverted row order in the probe target (`pa[0]` is
  documented as the sun row) or a channel swap — both mean the probe and the
  frame disagree about which way is up.
- failure signature → source: phase4 §8 records the read as
  `a[0]=(3,4)`-style indexing of the probe target, so a row-order slip here is
  the same class as the P3 readback-row-order question in P3 and phase2
  addendum; a stale look is the `uSunCol`/`uSkyAmb`/`uSkyHorizon` upload.

#### #11 — stereo disparity direction and magnitude (the SBS contract on device)

- Command (*verbatim*, port/reports/phase4.md:231):

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final .venv/bin/python vr.py --viewpoint 1 --stereo sbs --res 4096x2048 --spp 1 --out engine_vr/out/p4_p11.png
```

- Expect: one PNG, 8192×2048, left|right (vr.py concatenates on x). vp1's hero
  boulder sits ~5 m out, so the shift is
  `atan(0.032/5) / (2π) × 4096 ≈ 4.2 px` **inward** (cross-correlate a strip
  through the hero between the two halves). `pano.py` constants:
  `IPD_M = 0.064`, `EYE_OFFSET_M = 0.032`, `stereo_pair()` puts the **left** eye
  at `-camera_right` and shares one `dirs` array (RUNBOOK §6: parallel axes,
  ±0.032 m along camera-right, identical seed).
- **Abort criterion:** outward disparity, zero disparity (both eyes got the same
  origin), or a disparity of the wrong *scale* (a sign/units bug in the eye
  offset, e.g. 0.064 applied as ±0.064).
- failure signature → source: phase4 §8 attributes wrong direction to
  "`stereo_pair` right-vector sign or eye→`uCamPos` wiring"; the identical-seed
  requirement is RUNBOOK §6, and vr.py keeps the seed in `VIEWPOINTS`
  (`same_seed` shows up in the `--dry-run` plan lines) — if the two eyes differ
  in noise, the seed leaked into the per-eye path.

#### #12 — post pipeline over an equirect frame

- Command (*derived*: row-mean luma profile of the P8 render, sky rows vs
  equator; `post.finish`'s vignette is rectangular and `suggest_exposure`
  weights sky by luma rows):

```bash
cd /path/to/procedural-alpine-lake && .venv/bin/python -c "
import numpy as np
from PIL import Image
a = np.asarray(Image.open('engine_vr/out/p4_p8.png').convert('RGB')).astype('float32').mean(2)
H, W = a.shape
rows = a.mean(1)
top, eq, bot = rows[:H//8].mean(), rows[H//2 - H//16:H//2 + H//16].mean(), rows[-H//8:].mean()
print('top/pole %.3f  equator %.3f  bottom/pole %.3f' % (top, eq, bot))
print('pole darkening vs equator: top %.1f%%  bottom %.1f%%' % (100*(1-top/eq), 100*(1-bot/eq)))
"
```

- Expect: a smooth profile, and pole darkening **below ~8 % relative** (the
  threshold phase4 §8 sets). On a 2:1 sphere the poles land in the vignette
  corners and the sky fraction is fixed by layout, not by view.
- **Abort criterion:** none that stops the port — above ~8 % it is a documented
  defect: record the numbers and recommend a pano `Grade` override in `vr.py`
  (the engine is read-only, but `vr.py` may override grade fields; the
  `--dry-run` plan already reports `fields_skipped_scene` and `footprint_rad`).
- failure signature → source: phase4 §9.3 (post over equirect is a perspective
  assumption) plus §7 of this file for why the pano's pixels are NOT comparable
  to film frames.

#### #13 — 8192×4096 cost (the render budget decision)

- Command (*derived* from the P7/P8 form plus `/usr/bin/time -l`, which phase4
  §8 names for peak RSS):

```bash
cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final ALPINE_MAX_PROC_GB=96 .venv/bin/python -c "
import sys, time, resource
sys.argv = ['vr.py', '--viewpoint', '0', '--res', '8192x4096', '--spp', '1', '--out', 'engine_vr/out/p4_p13.png']
t = time.time()
import vr; vr.main()
print('sec', round(time.time()-t, 1), 'peak RSS MB',
      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024*1024))
"
```

- Expect: it finishes without a watchdog exit 3, and you get wall time +
  `maximum resident set size`. Scale check: **946** tiles per sample
  (ceil(8192/192) × ceil(4096/192)), each 2 draws (cone quarter-rect + trace) in
  one session plus one `read_image(gbuf.images[0], rect)` → ~532 MB of readback
  traffic per sample per eye, and the kind-bit blend into `acc` per tile.
- **Abort criterion:** a watchdog exit 3 means `ALPINE_MAX_PROC_GB` is too small
  for this pass — raise it deliberately (64 → 96 → 128) and note the number;
  a wall time that makes `--spp 4` impractical is a result too: it decides tile
  batching (`--tile`) and whether the delivery ships `--spp 2`.
- failure signature → source: phase3 §Risks "one submission per band"
  (passes_vr.py:365-391) and phase3 P6 are the submit-cost lineage; M5
  (readback re-entry) is the race to watch when per-tile readbacks dominate.

---

## 6. Failure-signature matrix (all rows, with the report each is traced to)

Read the left column first: it is what you actually see. Rows are
ordered "cheapest to rule out" first within each group. `Source` is the review
or phase report that recorded the mechanism, so a fix round can be audited.

| # | Observed failure signature | Likely mechanism | Source |
|---|---|---|---|
| 1 | `IndexError` at passes_vr.py:165 before any validation output | 1-D footprint array uploaded where `comps = a.shape[2]` was assumed | review_p4 C1 (fixed via `upload_footstrip` `reshape(-1,1)`; §5 verdict: a Python exception, not validation) |
| 2 | `ap.error`-style exit naming W ≥ 2H | cone pass indexes the bottom latitude quarter when W < 2H | review_p4 M1 (vr.py:1104-1117 rejects it; world.glsl `SCENE_PASS == 98` uses `panoFootAt(gl_FragCoord.y * 4.0)`) |
| 3 | `RuntimeError: cache incomplete: …` | `cache_schema.missing()` non-empty; B1 unfinished | vr.py `_render` cache guard; phase4 §9.4 |
| 4 | `RuntimeError: features.json missing in <dir>` | bake wrote heights but not features | vr.py `_render` cache guard |
| 5 | Watchdog exit **code 3**, log line naming the budget | pass exceeded `min(ALPINE_MAX_PROC_GB, 0.75 × recommendedMax)` | gpuslot_vr.py watchdog (5 s poll, 6 GB headroom, 2.5 GB floor) |
| 6 | Acquire hangs, log repeats the free-RAM number | `ALPINE_MIN_COMMIT_GB` / `ALPINE_COMPILE_GB` gate never satisfied; or two 8K jobs share the box | gpuslot_vr.py:323 `acquire()`, :368 `compile_guard`; recommend `ALPINE_MAX_GPU=1` |
| 7 | Heap budget lines print `0.0 GiB` | `VK_EXT_memory_budget` not actually enabled | review_p2 N1 (fixed at gl_vr.py:594-597; recurrence = regression) |
| 8 | Validation: "binding an image whose layout differs from the write-time layout" | frozen descriptor layouts vs per-tile auto-transitions | phase4 §9.1 + `to_sampler_layout` passes_vr.py:422-452 (TRANSFER_SRC branch :442-444); fallback = gbuf `COLOR_ATTACHMENT_OPTIMAL` + explicit barrier; review_p5 N4 moved this pointer off gl_vr.py:426-445 (image bind/view creation) |
| 9 | Harmless extra `vkCmdPipelineBarrier` before every sampled read | always-firing sampled-read barrier | review_p2 N10 (harmless; do not chase) |
| 10 | Torn/garbage first use of a freshly uploaded texture | upload barrier `TRANSFER → BOTTOM_OF_PIPE` on an out-of-order driver | review_p2 M4 |
| 11 | Blank or black frame, zero errors | fullscreen VBO not bound for a graphics pass | review_p2 M1 (fixed + re-verified; recurrence = regression) |
| 12 | Flat flow field / NaN heights from the erosion probe, exit 0 | coherent `imageLoad`/`imageStore` RMW visibility, compute→sample re-read, or `R32G32_SFLOAT` storage support on MoltenVK — **not** atomics | review_p3 N1 + phase1 ("no atomics anywhere, so no fallback either") + RUNBOOK §7 D6 |
| 13 | Hunting `VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT` | chasing a feature the port never uses | review_p3 N1 (explicitly retracted wording) |
| 14 | Every CPU-vs-GPU ray check fails by ~the same small error | using `equirect_directions` as ground truth | review_p4 N1 (was `dirs @ basis.T`; fixed at pano.py:100, max diff 2.98e-08) — use `vr.basis` rows-as-columns as the independent reference |
| 15 | Debugging `uPixAng` changes nothing | `uPixAng` is dead in all PANO variants; `probe.frag` never reads it | review_p4 N2 |
| 16 | Disparity outward, or identical eyes | `stereo_pair` right-vector sign, or eye→`uCamPos` wiring | phase4 §8; pano.py `EYE_OFFSET_M = 0.032` |
| 17 | World plausible but rotated/mirrored; horizon tilt the wrong sign | `uCamRot` column order vs vr.py `basis()` rows; roll sign | phase2 §Risks gl_vr.py:285-323, mat3 pack branch gl_vr.py:311-317; `uCamRot` is the UBO member at byte offset 1088 (manifest.json:315-317), not a gl_vr.py line (review_p5 N3); review_p2 R2 (pack reviewed correct) |
| 18 | Sky at the bottom of the pano / seam discontinuity | one of three Y/lon layers flipped (shader row order, readback flip, lon order) | phase4 §8; engine/render.py:505 flip |
| 19 | Poles visibly darker than the equator band | rectangular vignette + sky-weighted exposure over a 2:1 frame | phase4 §9.3 (fix = pano `Grade` override in vr.py) |
| 20 | `read_image` returns engine rows flipped or transposed | readback row order assumption | phase2 addendum ("readback row 0 = upload row 0"); review_p3 N6 (square-image-only upload path) |
| 21 | 1024 MB target stays resident across passes | missing `destroy_target` / no `try/finally` | review_p3 N3 (fixed: `bake_pass` has `try/finally`, passes_vr.py:307) |
| 22 | Forest clings to the stream everywhere or nowhere | `uStreamN`/`uTrailN` misread from the array UBO | phase3 P5 (`uStreamN` @1600, `uTrailN` @2096); review_p2 C2/C4 |
| 23 | "main lake flood leaked to the map border" | the height field, not the flood fill | bake_vr.py:774 (raise; `bake_world_maps` def :755) |
| 24 | Frame-to-frame noise differs between the two eyes | seed not shared across eyes | RUNBOOK §6 (identical seed) |
| 25 | Shaders look un-optimised; you reach for `glslc -O` | no `-O` is ever passed and this SDK accepts it | review_p4 N3: author's "no -O" claim **retracted** (build_p1.py:282-292); the reviewer's "no -O flag in this SDK" reason was then retracted in review_p4.md:244-250 |
| 26 | A `fetch()`-returned array won't accept in-place writes | read-only numpy view of mapped memory | review_p2 N8 |
| 27 | Depth > 1 readback has scrambled rows | `_pad_rows` used the wrong dimension for depth > 1 | review_p2 N9 |
| 28 | rusage counters read back as zeros right after a pass | 20-word `_RUSAGE_WORDS` buffer with zero headroom | review_p2 C5 (gpuslot_vr.py:81) |

---

## 7. Acceptance

The port is accepted when **all** of these hold, in this order:

1. B0/B1/B2 complete, `cache_schema.missing()` prints `[]`, and the bake left
   `features.json`, `noise3d.npy` and `trees.npy` behind.
2. Step 0 prints `RESULT: ALL CLEAR`.
3. P1 and P2 pass with zero validation output — the D6 hedge is retired only
   when the coherent RMW probe is green (RUNBOOK §7 D6).
4. P7 renders non-black with zero validation complaints, and P8/P9/P11 agree on
   orientation, roll sign and disparity direction.
5. P3/P4/P5 pass at their resolutions; P12 is *measured*, not necessarily
   green (above ~8 % pole darkening is a documented defect plus a `Grade`
   recommendation).
6. P13 produces a wall-time + peak-RSS number that this file's budget table
   records in place of the ESTIMATE rows, and a `--spp` choice for delivery.
7. The delivery artefacts exist: one 8192×4096 mono PNG, one SBS PNG
   (`--stereo sbs`, one file, left|right), and a `package_vr.py` package.

**Caveat that must travel with every accepted image:** the pano pixels are NOT
comparable with film.py frames. An equirect texel footprint is
`max(dLat, dLon·cosφ)` (RUNBOOK §6) — latitude is sampled uniformly while
longitude is compressed toward the poles — whereas a film frame is a
perspective frustum; `post.finish`'s vignette and `suggest_exposure` are tuned
for a rectangular frame (§12 above); the light probe is one sample at head
centre shared by both eyes (`vr.py` draws the probe once at head centre, the
FIXME sits ~0.032 m below the probe footprint); and `VIEWPOINTS` are keyframe-0
recipes, not the film's searched camera path (phase4 §9.5, FIXME(port)
vr.py:298-304). A pano that looks "worse" than a film frame at the same
`--res` is expected, not a bug.

---

## 8. Packaging the delivery (package_vr.py — prints ffmpeg, never runs it)

`package_vr.py` (repo root) is the packaging CLI and ffmpeg **argument
builder**. It reads each PNG's IHDR, classifies the layout, writes
`manifest.json` + `MANIFEST.md` (+ `pano/` copies + `frames.txt`) and prints
the ffmpeg argv. Phase 5 never spawned ffmpeg and the tool has no `--run`, so
the encode stays an operator decision:

```bash
cd /path/to/procedural-alpine-lake
.venv/bin/python package_vr.py --dry-run --in 'engine_vr/out/pano_*_sbs_*.png' --out-dir engine_vr/out/pkg
.venv/bin/python package_vr.py --dry-run --video --in 'engine_vr/out/pano_*_sbs_*.png' --out-dir engine_vr/out/pkg --fps 12
# satisfied? then drop --dry-run to write the package, and run the printed ffmpeg line yourself
```

Two input-side rules the fix round made explicit:

- `--in` takes explicit paths and/or quoted globs, and a glob may be
  **absolute** (`'/Volumes/delivery/pano_*_sbs_*.png'`) because a delivery
  directory is absolute by definition. Patterns go through `glob.glob`
  (package_vr.py:219-239; `Path().glob` raised NotImplementedError on an
  absolute pattern - review_p5 M1). Relative patterns still resolve from the
  cwd, which is the route the static gate runs
  (`engine_vr/out/*.png`, port/gates/gate_p5.py:39-41).
- `--crop-lat` is validated to `(0, 1]` (package_vr.py:320-331, review_p5
  N7): anything else exits 2 with a usage error instead of putting a `crop=`
  taller than the frame or a negative crop y into the argv. 1.0 is legal (a
  no-op crop); a smaller value keeps an even-height band centred on the
  equator and flips `projection_range` to `partial`.

Shape rules it enforces (all derived from `vr.py` and RUNBOOK §6): mono is
2:1; SBS is two eyes at 2:1 or wider packed on x (so 4:1 or wider total);
between 2:1 and 4:1 total is rejected for the same reason vr.py refuses W < 2H
(review_p4 M1). `--packing half` is what a headset wants, and since vr.py
writes **full** side-by-side the builder squeezes with one filter
(`scale=trunc(iw/4)*2:ih`) — squashing the packed frame by half squashes both
adjacent halves by half. With vr.py's SBS default (`--res 8192x2048` per eye)
that lands on exactly an 8192-wide packed frame; render `--res 8192x4096` if
you want 8192×4096 packed (4:1 eyes, redundant latitude). Multi-frame input
goes through a concat list because vr.py's filenames are viewpoint names, not
a frame sequence. The spherical metadata set it writes is the common
projection/stereo family and is **UNVERIFIED** against any player — see
port/reports/phase5.md.

For a yaw turntable: vr.py has **no yaw flag** (vr.py:1079-1149), so N-frame
turntables need a per-frame yaw knob added to a copy of `vr.py` first, or a
per-eye tensor export (`_render(..., ["left", "right"])`, phase4 §10) stitched
by you. `package_vr.py --fps` only sequences whatever frames you hand it.

---

## 9. What to report back if something aborts

For every abort, capture: the exact command line, the last 40 lines of output,
the `[gl_vr] device:` + `recommendedMaxMemorySize` + `heap … budget=` lines,
whether validation was enabled, the matrix row you matched, and which report
that row cites. That is enough for a fix round without a second GPU session.
