# Phase 5 report — round-2 coder, 2026-10-25 (session clock 09:2x +08:00)

## Status: DONE

Zero GPU, zero ffmpeg execution, no git, no pip. Everything below was built
from source, from the phase 0-4 reports and their reviews, and from CPU-only
fixtures. No probe in port/RUN_FIRST.md has been run; the manual states that in
its first paragraph.

## Files written:

| Path | LOC | State |
|---|---|---|
| port/RUN_FIRST.md | 804 | new (T1 + T4) |
| package_vr.py | 441 | new (T3) |
| engine_vr/check_viewpoints.py | 271 | round-1 file (133 L) reviewed and rewritten (T2) |
| engine_vr/tmp/mk_pkgtest_pngs.py | 27 | new helper: stdlib PNG writer for package_vr fixtures |
| engine_vr/tmp/mk_fakecache.py | 40 | new helper: synthetic cache so step 0 can be exercised pre-bake |
| port/reports/phase5.md | — | this file (T5) |

Nothing else was created, moved or deleted outside those paths and
engine_vr/tmp/. engine_vr/tmp/pkgtest/*.png and engine_vr/tmp/fakecache/* are
fixture output. No gate, engine/, audio/ or docs/ file was touched (gate_p5
"read-only trees untouched - []"). cache_final still does not exist: this
workspace has never baked.

## Inherited round-1 code (the brief asked for an explicit verdict)

Round 1 left engine_vr/check_viewpoints.py (133 L). Verdict: **right tool, two
defects that made ladder step 0 unusable; kept the interface, rewrote the
semantics.**

- Kept: the CLI the brief and gate_p5 depend on (--cache / --min-clearance /
  --dry-run), the exit-code ladder (0 clear / 1 flagged / 2 cache unusable),
  the clearance rule (pos.y - PanoWorld.ground(x, z)), the CPU-only import
  discipline (vr.py touches Vulkan only inside _render()), and the --dry-run
  pre-bake mode the gate runs.
- Defect 1 (fatal for the ladder): port/PHASE_5.md T2 mandates "flag < 5 m as
  buried, exit non-zero if any buried", and round 1 implemented exactly that —
  but six of the nine recipes STAND at agl 1.0-2.2 m by construction
  (vr.py:312 forest agl 1.9, :324 reveal 1.0, :378 meadow 1.6, :404 finale 1.7
  all go through world.P; :333 skim y=1.7 and :356 waterfall y=2.2 are absolute;
  :371 stream is max(ground,0)+1.5). On a correct cache the 5 m default flags
  seven of nine and step 0 aborts a healthy world. Fix: keep the 5 m default and
  the non-zero exit (the brief's contract), but classify
  BELOW-SURFACE / BELOW-THRESH / OK / UNKNOWN and add --nominal-ok so the
  pass/fail form (--nominal-ok --min-clearance 0.5) gates on real burials.
  RUN_FIRST §3 documents both forms and why.
- Defect 2: it reported a bare clearance number with no provenance. Now it
  prints, per sample, which map PanoWorld.height() actually used (the
  near-map rule mirrored from vr.py:200-202) and, once per run, each loaded
  map's shape/dtype/cell against cache_schema.shape_of/cell with a
  SPEC-MISMATCH flag that counts as a failure — because a wrong-resolution bake
  moves every camera at once and otherwise looks like nine unrelated bad
  viewpoints. Added per-viewpoint remedy lines citing the RUN_FIRST §6 matrix
  rows, and the exact command lines to re-run.
- Verified statically, CPU-only (fixtures, no GPU) with the synthetic-cache
  form of the check — `--no-spec-check`, because the 64² fixtures are below
  contract by design and the shape guard must stand down (RUN_FIRST §3 carries
  the exact two commands): --dry-run rc 0 pre-bake; fakecache height 0 →
  RESULT: ALL CLEAR rc 0; height 12 → BELOW-SURFACE on skim (-10.3 m) and
  waterfall (-9.8 m), RESULT: 2 flagged, rc 1. Spec check left ON: both
  fixtures exit 1 on the three SPEC-MISMATCH map lines alone (height 12 then
  reads 3 flagged) — review_p5 N1, corrected in the fix-round addendum.

## Deviations from brief:

1. **T2 threshold semantics** as above: the mandated 5 m flagging is kept but
   no longer conflated with burial; the pass/fail ladder form is
   `--nominal-ok --min-clearance 0.5`. (Brief-vs-code contradiction — see the
   contradictions section.)
2. **T3 per-eye resolution**: the brief sketches "equirect 3840x2160-per-eye
   or per vr.py res". package_vr.py derives everything from the actual PNG
   IHDRs and never hardcodes 3840x2160; it prints a NOTE when the packed width
   is not 8192 instead of assuming it.
3. **T3 "slow yaw keyframes"**: vr.py has no yaw/pitch/heading flag
   (argparse surface vr.py:1079-1149) and no per-frame camera path, so a
   turntable cannot be generated from the CLI as it stands. package_vr.py
   sequences whatever frames it is given (--fps, concat list, --pattern) and
   RUN_FIRST §8 records the two real options (add a per-frame yaw knob to a
   copy of vr.py, or export per-eye tensors via `_render(...,
   ["left","right"])` per phase4 §10). Inventing a yaw flag would have been a
   write outside the allowed paths.
4. **Two tmp helpers added** (allowed by PHASE_5.md §Boundaries) so both new
   tools have reproducible CPU-only fixtures.
5. **Report shape**: RUNBOOK §5 headings kept verbatim; "what remains genuinely
   UNKNOWN" (T5) is the last section, plus a contradictions section because
   four report-vs-report conflicts had to be resolved to write the manual.

## Gate self-check:

```
$ .venv/bin/python port/gates/gate_p5.py | tail -6
PASS ruff E,F
PASS package_vr.py --dry-run exit0 - NOTE   no usable pano PNG under ['engine_vr/out/*.png']: nothing matched. Render first (port/RUN_FIRST.md ladder), then re-run.
PASS --video dry-run prints ffmpeg argv w/ equirect metadata, no spawn - NOTE   no usable pano PNG under ...
PASS check_viewpoints --dry-run exit0 pre-bake - WOULD CHECK 9 vr.py viewpoints, cache <ws>/cache_final
PASS read-only trees untouched - []

GATE P5: PASS
```

All 17 checks PASS, zero FAIL (RUN_FIRST exists; probes #1..#13 present; ALPINE_ env
table; budget table; failure-signature section; step-0 reference; acceptance +
pano-vs-film caveat; package_vr exists + py_compile; check_viewpoints exists +
py_compile; no moderngl/wgpu imports; ruff E,F; both package_vr dry-runs;
check_viewpoints --dry-run; read-only trees).

```
$ .venv/bin/python -m ruff check --select E,F engine_vr/ vr.py package_vr.py
All checks passed!
$ .venv/bin/python -c "import py_compile; py_compile.compile('package_vr.py', doraise=True)"   # ok
$ .venv/bin/python -c "import py_compile; py_compile.compile('engine_vr/check_viewpoints.py', doraise=True)"  # ok
```

Extra CPU checks beyond the gate:

- `package_vr.py --dry-run --in engine_vr/tmp/pkgtest/pano_*_sbs_*.png`:
  classifies 1024x256 as SBS full, prints the concat-list encode with
  `scale=trunc(iw/4)*2:ih:flags=spline` and the metadata block; writes nothing.
- Same command without --dry-run into engine_vr/tmp/pkgtest/pkg: writes
  frames.txt (absolute paths), manifest.json, MANIFEST.md and pano/ copies with
  sha256; exit 0.
- REJECT path on a 4:3 PNG (cites review_p4 M1) and SKIP on a non-PNG whose
  name ends in .png.
- `vr.py --dry-run` plan output was re-read (not re-run) only to confirm no yaw
  flag exists.
- ffmpeg is installed on this machine (/opt/homebrew/bin/ffmpeg) and was never
  executed, not even with -version or -h.

## API contracts relied on:

- **engine_vr/cache_schema.py** — `cache_dir()` reads $ALPINE_CACHE each call
  (:62-64); `CACHE = cache_dir()` at import (:67); `set_cache_dir()` rebinds
  `cs.CACHE` and exports the env var (:70-76) but cannot reach a module that
  already copied it (bake_vr.py:45 `CACHE = cs.CACHE`) → RUN_FIRST §1.2 tells
  the operator to set ALPINE_CACHE in the shell. `path()` :157, `spec()` :162,
  `cell()` :178, `shape_of()` :184, `missing()` :196, `FILES` :92-155,
  `FAR_HALF = 16384.0` :37, `NEAR_HALF = 2048.0` :39. Verified by reading those
  lines and by an import-order experiment run in this session
  (`ALPINE_CACHE=/tmp/env_a` + `set_cache_dir('/tmp/env_b')`, importing bake_vr
  before vs after → stale vs fresh `bv.CACHE`).
- **vr.py** — `PanoWorld.__init__` loads h_far/h_near/water_level + optional
  features.json (:179-186); `height()` near-map rule
  `max(abs(x), abs(z)) < cs.NEAR_HALF * 0.95` (:200-202); `water()` :205-210;
  `ground() = max(height, water)` :213-214; `P()` :217-218; `shore_south`
  :221-225; `shore_x` :228-232; `head_peak` :235-243; nine `_vp_*` stance lines
  :312/:324/:333/:356/:371/:378/:385/:395/:404; `VIEWPOINTS` :415-434;
  `resolve_viewpoint` returns (pos, tgt, heroes) :437-441. Verified by reading
  and by calling resolve_viewpoint + PanoWorld against the synthetic cache.
- **engine_vr/passes_vr.py** — `open_host(cache_dir=None, res=None, tile=192,
  spp=1, log=print, **host_kwargs)` acquires the gpuslot BEFORE Host.open and
  releases on failure (:75-95); `close_host()` saves the pipeline cache
  best-effort then closes and releases (:98-114). Used verbatim in RUN_FIRST
  B0/B1 so the documented ladder cannot strand a slot.
- **engine_vr/bake_vr.py** — `bake_heights(far_res=FAR_RES, near_res=NEAR_RES,
  force=False, erode=None)` :200 calling `_bake_pass("bake_height.frag",
  far_res, ..., tile=256)` :211; `bake_ao(hfar, hnear, force)` :711 binding one
  texture as both uH and uHFar :721 with float16 saves :727; `bake_world_maps`
  (def :755) flood-leak raise :774; `bake_forest` reading uStreamN/uTrailN from
  features.json :837-839; `bake_all` chain :900; `ensure_trees` :948;
  `bake_shadows` :968; `run_eroded_bake` :167; `run_all` :151; GL-era "~6 min
  erosion" comment :236.
- **engine_vr/erosion_vr.py** — `run_erosion(hfar=..., cell_m=..., stages=...,
  save=False)` returning a dict with 'height'/'flow' (used verbatim for probe
  #1 and derived for #2/#6); `STAGES` :497; checkpoint/resume kwargs.
- **engine_vr/gpuslot_vr.py** — ALPINE_* names and defaults, watchdog cap
  `min(ALPINE_MAX_PROC_GB, 0.75 x recommendedMaxMemorySize)`, 6 GB headroom,
  2.5 GB floor, 5 s poll, hard exit 3, `acquire()` :323, `compile_guard()`
  :368, `_RUSAGE_WORDS = 20` :81.
- **engine_vr/gl_vr.py** — portability-features bit and VK_EXT_memory_budget at
  device create :594-597; the `recommendedMaxMemorySize -> watchdog cap` log
  line in the open() block :575-633 (quoted in RUN_FIRST B0 expectations).
- **engine/gpuslot.py / tools/render_final.py** — the memory lessons and the
  "full-res frames peak at ~17 GB in the driver" comment quoted in the budget
  table.
- **stdlib only in package_vr.py**: struct+zlib IHDR parse, hashlib.sha256,
  shutil.copy2, json, argparse, glob. No PIL import there, so the packager runs
  with the repo interpreter regardless of imaging deps.
- **PIL 13.0.0 / numpy 2.5.4** in .venv — verified importable, and imageio/cv2
  are NOT installed, which is why every analysis snippet in RUN_FIRST uses PIL.
- **ffmpeg argv** — no API contract: nothing was executed and no ffmpeg manpage
  or version was consulted this session. The flag set and the spherical
  metadata family are marked UNVERIFIED in the tool output, in MANIFEST.md and
  in RUN_FIRST §8.

## Contradictions found across the reports (and how Phase 5 resolved them)

1. **Tiles per sample.** phase4 §8/§9 say 448 tiles at 8192x4096 with
   tile=192; ceil(8192/192) x ceil(4096/192) = 43 x 22 = **946** (43, not 44 —
   44 x 22 would be 968; review_p5 N8). RUN_FIRST §4
   uses 946 for the readback-traffic row (946 x 0.5625 MB ≈ 532 MB per sample
   per eye) and prints the 448 figure beside it so the operator can see the
   discrepancy; §13 (P13) asks the operator to count actual tiles on the first
   run. Neither number changes any pass/fail criterion.
2. **What probe #9 may trust as ground truth.** review_p4 N1 (round 1) found
   `pano.equirect_directions` used `dirs @ basis.T` — the inverse — so any
   CPU-vs-GPU ray check built on it false-failed; the fix landed at pano.py:100
   (`dirs @ basis`, max diff 2.98e-08) and review_p4 §7 then permitted probe
   #9 to use it. The round-2 brief says use vr.basis. Resolution: **vr.basis
   (rows-as-columns) is the primary reference; equirect_directions is a
   secondary cross-check only**, documented in RUN_FIRST §9 with the reason, so
   the check stays independent of the helper that was once wrong.
3. **"Atomic Add" in probe #1.** phase3 wrote probe #1 up as an atomicity test;
   review_p3 N1 retracted that — the erosion shaders do coherent
   imageLoad/imageStore read-modify-write with zero atomics anywhere in the
   port. RUN_FIRST P1 carries the corrected wording and an explicit
   do-not-chase line for VK_FORMAT_FEATURE_*_ATOMIC_BIT (matrix row 13).
4. **`glslc -O`.** phase4 claimed no -O was passed and that the SDK accepted
   it; review_p4 N3 retracted the claim, then review_p4.md:244-250 retracted the
   reviewer's "no -O flag in this SDK" reasoning. RUN_FIRST matrix row 25 keeps
   the surviving lesson only: do not reach for -O as an optimisation knob.
5. **T2's 5 m threshold vs vr.py's stance values** (brief-vs-code, listed under
   Deviations). Resolution: keep both semantics, separate them in the output,
   and gate the ladder on the pass/fail form.
6. Checked and NOT contradictory: phase4 §9.4's "gbuf 2x512 MB, acc 512 MB,
   cone 8 MB" matches the arithmetic on 8192x4096 RGBA32F, and phase3's "1.07
   GiB bake target" matches an 8192² RGBA32F target.

## Risks / FIXME(port) list:

- **port/RUN_FIRST.md §4 (ESTIMATE rows)** — wall time per tile, compile spike
  on this device, pipeline-cache reuse and the 946-vs-448 tile count stay
  estimates; they must be replaced with measured numbers after P13, not
  believed.
- **port/RUN_FIRST.md §5 P5/P6** — the derived bake commands call
  `bake_world_maps(hfar, force=True)` and `run_erosion(stages=[STAGES[-1]])`
  directly; if a future phase changes those signatures the commands rot. They
  cite bake_vr.py:829+/erosion_vr run_erosion for the next reader.
- **port/RUN_FIRST.md §2 B1** — the bake one-liner interpolates `$PWD` into a
  python -c string; copy-paste from a shell works, pasting into a script file
  without the shell does not. Noted inline.
- **package_vr.py DEFAULT_TAGS** — the projection/stereo metadata family is a
  best-guess common set; unverified against any player or ffprobe. Printed as
  UNVERIFIED in the tool's own output, in MANIFEST.md and in RUN_FIRST §8.
  Remedy after the first encode: ffprobe the file, then pin the tags with
  `--tag`.
- **package_vr.py frames.txt** — absolute paths into out-dir/pano, so a package
  moved to another machine needs `--out-dir` re-run before the encode. Documented
  in MANIFEST.md output.
- **package_vr.py turntable scope** — no yaw generator (see Deviations 3): a
  single-frame "`-loop 1 -t`" encode is what the tool can honestly produce from
  one PNG today.
- **engine_vr/check_viewpoints.py NOMINAL table (~line 60)** — it duplicates the
  stance values from vr.py:312-404 as documentation; nothing binds them, so a
  future recipe edit silently makes --nominal-ok wrong for that row. FIXME(port):
  a gate check could regex vr.py for `w.P(..., agl)` and compare.
- **engine_vr/check_viewpoints.py** — it measures the CPU world
  (`PanoWorld.ground`); if vr.py's render path ever samples a different map or
  adds a head-height offset for stereo, this check must follow or it starts
  lying. The map name it prints is the guard against exactly that drift.
- **engine_vr/tmp/mk_fakecache.py** — writes 64² maps far below contract on
  purpose; never point a real render at engine_vr/tmp/fakecache.
- **RUN_FIRST §3** — `BELOW-SURFACE` on a placed recipe is unreachable by
  construction; if it ever prints, the cache changed between resolve and sample.
  Kept as a tripwire rather than an assert.

## What stays unknown until the first run (the honest limit, RUNBOOK §4)

1. **The watchdog cap.** `recommendedMaxMemorySize` on this M5 Ultra is
   printed only by `gl_vr.Host.open()`. Until it prints, every ALPINE_MAX_PROC_GB
   recommendation (64 bake / 96 render) is an opinion, and an 8192² render could
   hard-exit 3 on a budget that a 256 GiB machine can actually pay.
2. **Whether the erosion read-modify-write is correct on MoltenVK.** Probe #1 is
   the D6 hedge: coherent `imageLoad`/`imageStore` neighbour gather,
   compute→sample re-read of the same r32f/rg32f images, and
   `R32G32_SFLOAT` storage support. Static review cannot settle it and there is
   no atomic fallback anywhere in the port, so a failure has no degraded mode.
3. **Cost.** Wall time for 946 tiles x (2 draws + 1 readback) per sample per eye
   at 8192x4096, whether `--spp 4` is reachable, how much the pipeline cache
   actually saves on run 2 (phase2 M2 lineage), and the real peak RSS of the
   bake chain versus the compile guard's 50 GB assumption.
4. **Layout behaviour of the biggest surface in the port.** Probe #7 is the
   first time 12 frozen descriptors, per-tile auto transitions and per-sample
   gbuf SRO flips meet a driver; phase4 §9.1's fallback (gbuf final layout
   COLOR_ATTACHMENT + explicit barrier) is a written plan, not an observation.
5. **Orientation parity on device.** `uCamRot` column order and the roll sign
   (#9), and the three independent Y/lon layers of #8, are reviewed-correct in
   source and unmeasured in pixels; a plausible-but-rotated pano is the failure
   mode the reviews ranked highest-impact.
6. **Whether the viewpoints still fit the baked world.** The film's clearance
   SEARCH was never reproduced (vr.py FIXME(port) :302-304), so step 0 tells you
   the truth only after a real bake — the synthetic cache proves the tool, not
   the world.
7. **Packaging truth.** Whether any headset/player reads the tags
   package_vr.py writes, and whether libx265 accepts the exact argv, is unknown
   until someone runs the printed command; Phase 5 deliberately never did.
8. **Post over equirect magnitude.** The vignette/exposure bias on a 2:1 frame
   (#12) is reasoned about but never measured; the ~8 % pole-darkening
   threshold is a judgement, not a number from the engine.

## Addendum — P5 fix round (review_p5.md M1 + N1-N9), 2026-10-25

Zero GPU, zero ffmpeg (package_vr.py has no exec surface, and the new
engine_vr/tmp/test_p5_fixes.py monkeypatches subprocess/os exec entry points to
raise, so a spawn would fail the run). Writes: package_vr.py,
port/RUN_FIRST.md, this report, engine_vr/tmp/test_p5_fixes.py.
engine_vr/check_viewpoints.py untouched — N1 needed no new flag,
`--no-spec-check` already exists (check_viewpoints.py:169-171, "synthetic test
caches only") and is what RUN_FIRST §3's self-test now quotes verbatim.

- **M1 (fixed)** — `package_vr.collect` :219-239 routes patterns through
  `glob.glob(..., recursive=True)`. `Path().glob` raised
  `NotImplementedError: Non-relative patterns are unsupported` on an absolute
  glob (reproduced live before the fix), which is the normal operator input for
  a delivery directory. Absolute glob, relative glob, explicit path and an
  unmatched pattern are all covered by the new test.
- **N7 (fixed)** — package_vr.py:320-331: `--crop-lat` must be in (0, 1];
  out of range argparse exits 2 with a usage error. Before: `--crop-lat 5`
  printed `crop=iw:1280:0:-512` against a 256-row pano (crop taller than the
  input, negative y) and `-0.5` quietly meant "keep 2 rows". The in-range
  case got one follow-on fix the review did not ask for: `crop_h` is clamped
  to the frame and the crop **y** is floored to an even number
  (package_vr.py:133-140), because `--crop-lat 0.001` on a 32-row frame used
  to emit `crop=iw:2:0:15` — legal for ffmpeg, but an odd y offsets luma and
  chroma by half a pixel in a 4:2:0 encode, which is the kind of defect that
  only shows up in the delivered file.
- **N1 (fixed)** — RUN_FIRST §3's synthetic self-test is reproducible as
  written: height 12 → `RESULT: 2 flagged`, exit 1 (skim -10.3 m, waterfall
  -9.8 m); height 0 → `RESULT: ALL CLEAR`, exit 0; both with
  `--no-spec-check`, and the doc now states what changes without it.
- **N6 (fixed, with a correction to the review's alternative)** — probe #6's
  grid step is 4.0 again. `res = h0.shape[0]` (erosion_vr.py:1470) and
  `f = res // r` / `cell = cell_m * f` / `final = r == res`
  (:1485-1488) mean a 4096² grid at `STAGES[-1]` (res 8192, :523) gives
  `f = 0` and `_pool` divides by zero at :533 — the probe never reached a
  device, so the timing intent did not "survive" as review_p5 §2 claimed, and
  `cell_m=8.0` could not rescue it either (f is shape arithmetic). 8192² at
  `cell_m=4.0` is also exactly the far-map contract (cache_schema.py:37-38,
  run_erosion default erosion_vr.py:439-440).
- **N2/N3/N4/N5/N8/N9 (citation polish, RUN_FIRST §3/§5/§6 + this report)** —
  `comps = a.shape[2]` is passes_vr.py:165 (was :166, twice);
  `uCamRot` @1088 is a UBO **byte offset** (manifest.json:315-317), the mat3
  pack is gl_vr.py:311-317 (gl_vr.py:1088 is descriptor-kind dispatch); the
  `TRANSFER_SRC` fallback is `to_sampler_layout`
  passes_vr.py:422-452 (branch :442-444) plus `make_target`'s
  `TRANSFER_SRC` default (passes_vr.py:264-266 →
  `Target.final_layout` gl_vr.py:446-462), not gl_vr.py:426-445; the
  flood-leak raise is bake_vr.py:774; the tile product is 43 x 22 = 946; and
  probe #5's CPU reference axis is `bake_vr.world_xy` (bake_vr.py:732-734)
  with `cs.cell`/:178 and `cs.shape_of`/:184 — `cs.world_xy` never existed.
- Gates untouched. Green: gate_p5 PASS, test_p5_fixes PASS, ruff E,F over
  engine_vr/ vr.py package_vr.py clean.
