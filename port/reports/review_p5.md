# review_p5 — independent review of the first-run protocol (P5)

Reviewer: independent contract reviewer (round: P5). Date 2026-10-25. READ-ONLY:
read + CPU-only execution of the two new CLI tools and gate_p5 (no GPU, no
ffmpeg — ffmpeg argv judged by inspection only; no git writes).
Targets: port/RUN_FIRST.md (804 L), package_vr.py (441 L, repo root),
engine_vr/check_viewpoints.py (271 L, round-2 rewrite), port/reports/phase5.md (285 L).

## 1. Probe-ladder / command-line verdict (check 1)

**52 command/API points checked, 0 non-existent names, 0 wrong flags; 5 verbatim blocks
byte-match their cited reports; 6 doc blemishes (§2).**

- Ladder command lines audited: B0, B1, missing-peek, §3 3a/3b, #1-#13 (incl. #8
  numeric guard), §8 package lines. All `vr.py` flags used exist
  (--viewpoint/--res/--spp/--out/--stereo/--cache/--dry-run; `--tile` default 192,
  vr.py:1091). All 19 python API surfaces exist with matching kwargs:
  `run_erosion(hfar,cell_m,host,save,**kw→stages)` erosion_vr.py:427, `STAGES` :497,
  `bake_heights(far_res,near_res,force)` bake_vr.py:200 (tile=256 at :211-212),
  `bake_ao(hfar,hnear,force)->(aof,aon)` :711/:728, `bake_world_maps(hfar,force)` :755,
  `bake_noise3d(force)` :855, `bake_all(force)` :900, `run_eroded_bake(host,force)` :167,
  `ensure_trees` :948, `bake_shadows` :968, `open_host(cache_dir,res,tile,spp,log)`
  (phase5 contract), `pv.read_image` (probe read verbatim-matches vr.py:976-977:
  `pa = pv.read_image(ptar.images[0])[0]` / `sun, sky, hor = pa[0][:3], pa[1][:3], pa[2][:3]`).
- Verbatim claims verified character-for-character: #1 == phase3.md:278-288;
  #7 == phase4.md:202; #8 == :211; #9 == :219; #11 == :231. Derived probes #2/#3/#5/#6/#10/#12/#13
  genuinely match the cited signatures; `EROSION_BAND_TILES=128` /
  `EROSION_MAX_DISPATCH_ROWS=2048` exist (cache_schema.py:55-56) and the 5-bands
  claim checks out (g = res//16+1 = 513, ceil(513/128)=5).
- Semantics spot-checks: #9 "vp2 skim roll=-1.5" ✓ (VIEWPOINTS vr.py:415-433, skim
  = index 2, `"roll": -1.5` at :421); #11 disparity math ✓
  (atan(0.032/5)/(2pi)*4096 = 4.19 px; pano.py:46-47 IPD_M=0.064/EYE_OFFSET_M=0.032;
  stereo_pair puts LEFT at -camera_right and shares one `dirs`; `same_seed` string
  at vr.py:745). `VK_LOADER_LAYERS_ENABLE`, `/usr/bin/time -l`, PIL 13.0.0 all real.

## 2. Findings

### MAJOR
- **M1 — package_vr.py:227 `Path().glob(g)` crashes on an ABSOLUTE glob**
  (`NotImplementedError: Non-relative patterns are unsupported` — reproduced live:
  `package_vr.collect('/abs/path/pano_1*.png')`). Deliveries move files to
  absolute locations; the tool's own help says "quoted glob(s) and/or explicit
  paths", so an absolute glob is a foreseeable operator input and the failure is an
  uncaught traceback. One-line fix (use `glob.glob` or split root). This is the
  only mandatory fix.

### MINOR (doc blemishes; none changes a pass/fail outcome)
- **N1 — RUN_FIRST.md:224-228 not reproducible as written.** The synthetic
  self-test claims `mk_fakecache.py 0` then form 3b gives `RESULT: ALL CLEAR`
  exit 0. Rerun: form 3b as written exits **1** (64² fixture trips
  SPEC-MISMATCH); ALL CLEAR/exit 0 needs `--no-spec-check` (the flag's own help
  says "synthetic test caches only"; phase5.md:57-60 states the true behaviour).
  Height-12 side of the claim verified exactly: skim −10.3 m, waterfall −9.8 m,
  exit 1 (and exit 1 with `--no-spec-check` too — real burials flag regardless).
- **N2 — RUN_FIRST.md:497 and :699 cite `passes_vr.py:166`; the `comps = a.shape[2]`
  IndexError line is :165** (verified: sed shows comps at 165). Off-by-one inherited
  from review_p4 round-1 text.
- **N3 — matrix row 17 (RUN_FIRST.md:715) "uCamRot @1088":** `gl_vr.py` contains no
  `uCamRot` string at all; :1088 is the descriptor kind dispatch (`combined_sampler`).
  The mat3 pack actually cited correctly by range is the Std140Packer mat3 branch
  gl_vr.py:311-315 ("GLSL column c = numpy row c (moderngl)").
- **N4 — §5#7 item 2 / matrix row 8 (RUN_FIRST.md:503, :706) "phase2 … TRANSFER_SRC
  fallback at gl_vr.py:426-445":** that range is `vkBindImageMemory`/ImageView
  creation. The layout-fallback mapping lives in `to_sampler_layout`
  (passes_vr.py:422-447) + the Target final-layout default. Mechanism described in
  the row is right; the pointer is wrong.
- **N5 — "main lake flood leaked to the map border" raise is bake_vr.py:774,**
  cited as :755 (the `def bake_world_maps` line) in RUN_FIRST:156/721 and
  phase5.md:156. Function-level citation is defensible; the raise line should be :774.
- **N6 — probe #6 (RUN_FIRST.md:464-467) internal parameter mismatch:** `mgrid`
  step 8.0 builds 4096² at 8 m/px but `cell_m=4.0` declares 4 m/px — the synthetic
  cone is half the declared physical size. Timing intent (8192-level stage via
  `STAGES[-1]`) survives, but flux magnitudes are off-scale; pass `cell_m=8.0` or
  keep step 4.0.
- **N7 — package_vr.py `--crop-lat` is unvalidated** (:296-297, used :129-133):
  `--crop-lat 5` prints `crop=iw:1280:0:-512` (crop larger than input, negative y)
  — ffmpeg fails at runtime instead of argparse. Clamp to (0,1].
- **N8 — phase5.md:186 wrong intermediate:** "ceil(8192/192) x ceil(4096/192) =
  **44 x 22** = 946" — it is 43 × 22 (44×22 = 968). RUN_FIRST's own 946 and 532 MB
  are correct.
- **N9 — RUN_FIRST.md:377 references a non-existent symbol** "cs.world_xy-equivalent"
  (cache_schema has no `world_xy`; real helpers are `cell`/`shape_of` + `spec`).

### Verified true (no finding) — spot-checked matrix rows 1,2,5,8,13,17,22,25,28
Row 2 guard exact (vr.py:1104-1117; `SCENE_PASS == 98` scaled `panoFootAt(y*4.0)` in
world.glsl is the implemented M1a fix). Row 5 watchdog: cap = min(MAX_PROC_GB, 0.75×
recommendedMax) at gpuslot_vr.py:274, `os._exit(3)` :309, `time.sleep(5)` :294.
**Row 13 (the do-not-chase row) is correct: the port has zero atomic shader ops**
(review_p3 N1 evidence; erosion uses coherent `imageLoad`/`imageStore` neighbour
gather, erosion_accum_flow_only.comp :62/:97/:144), so hunting
VK_FORMAT_FEATURE_*_ATOMIC_BIT is chasing a feature the port never imports. Row 22
offsets (uStreamN @1600, uTrailN @2096) match my P3 CPU recomputation. Row 25 cites
my own retraction honestly (review_p4.md:244-250 exists and says exactly that).
Row 28 `_RUSAGE_WORDS = 20` at gpuslot_vr.py:81 exact. Row 21 try/finally present.

## 3. Memory numbers (check 2)
**946 is right; 448 (phase4.md:241/:262) is an arithmetic error; RUN_FIRST resolves
it correctly and labels sources.** ceil(8192/192)=43 × ceil(4096/192)=22 = 946
(vr.py:1091 default tile 192; tiling loop vr.py:1015-1018); 946 × 0.5625 MB = 531.6
MB/sample/eye ✓. Budget-table rows independently recomputed vs cache_schema: h_far /
h_far_raw / water / forest 256 MB each, h_near 64 MB, trees 4096² f4 256 MB, ao 4096²
f2 32 MB, noise3d 128³ u8 8 MB, gbuf 2×512 MB + acc 512 MB at 8192×4096 RGBA32F, bake
target 1024 MB — all correct; the "machine" row hw.memsize 274877906944 matches
`sysctl` here; every non-derived row carries ESTIMATE or a report name, and
`recommendedMaxMemorySize → watchdog cap` is honestly **UNKNOWN until first run**
(RUN_FIRST:247) with a hole explicitly reserved in §4 (B0 abort criterion tells the
operator to record it).

## 4. check_viewpoints.py semantics (check 4)
Classification logic verified by reading + live reruns:
- BELOW-SURFACE (clear<0) / BELOW-THRESH / OK / UNKNOWN on resolve failure /
  SPEC-MISMATCH per map vs `cache_schema.shape_of`; `--nominal-ok` exempts ONLY
  `agl`/`agl+` modes whose clearance equals the source agl within 1e-6
  (check_viewpoints.py:136-148); `abs` recipes can never be exempted — correct
  conservatism (verified: at `--nominal-ok --min-clearance 5` only skim/waterfall
  stay flagged).
- Exit codes: 0 clear / 1 any flag incl. spec-mismatch / 2 cache incomplete or
  import failure — all reproduced (missing cache → 2; gate's `--dry-run` → 0).
- **The round-2 rewrite rationale is TRUE, not a face-saver:** on a healthy cache
  the mandated 5 m default flags **exactly 7 of 9** (measured on the height-0
  fixture: 7 BELOW-THRESH), because the stances are forest agl 1.9 (vr.py:312),
  reveal 1.0 (:324), skim abs 1.7 (:333), waterfall abs 2.2 (:356), stream
  max(ground,0)+1.5 (:371), meadow 1.6 (:378), finale 1.7 (:404); flyover/peaks
  (620/700, :385/:395) pass. Round-1's buried 5 m rule really would abort step 0
  on a correct world; keeping the T2 default as the diagnostic form (3a) and adding
  the pass/fail form (3b) is the right direction. Provenance lines mirror vr.py's
  own near-map rule (vr.py:189-202; `_map_for` at :107-115) and `P() = ground+agl`
  (:217-218), so the `agl` exact-match exemption is by construction, not luck.
- Caveat the author already documented: the NOMINAL table is a copy, not a binding
  (drift risk if a recipe changes; suggested gate — phase5.md:239-242).

## 5. package_vr.py (check 5) — ffmpeg by inspection only, never spawned
- No exec surface anywhere: `grep subprocess|os.system|execvp|popen` matches only
  the docstring denial (:11). Live `--dry-run` wrote nothing (target dir absent
  after); write-mode created manifest.json + MANIFEST.md + pano/ copies + frames.txt
  with absolute quoted paths. Mono|SBS mixing guard exits 2 ✓.
- argv validity: `-loop 1 -t` for a still, `-framerate/-start_number -i pattern`,
  concat-demuxer path with `fps=` filter head, `scale=trunc(iw/4)*2` half-pack math
  (16384→8192 wide ✓), `format=yuv420p` + `-pix_fmt yuv420p`, `-crf/-preset/-g/-keyint_min/-sc_threshold/-x265-params/+faststart+use_metadata_tags` — every
  option is real ffmpeg syntax; `;`- and space-bearing values are shell-quoted so
  the printed line is pasteable. Crop ordering (crop before scale) and even-size
  truncation are consistent.
- Metadata: `projection=equirectangular` + range/bounds + stereo_mode/format +
  initial_view + min-tile is a plausible common family but is **not** the mechanism
  headset players actually read (SphericalVideoBox/sv3d — ffmpeg writes keyed udta
  text with `use_metadata_tags`). This is honestly labelled UNVERIFIED in four
  places (docstring :35-41, printed NOTE :387-389, manifest `metadata_verified:
  false` :372, MANIFEST.md :428) and phase5.md:280-282 keeps it an unknown.
  Adequate honesty; first-playback `--tag` re-pin is the documented remedy.
- Defects: M1 (absolute glob) and N7 (`--crop-lat` range) above.

## 6. Fabrication audit (check 6)
~30 `file:line` citations sampled across RUN_FIRST §1-§8 and phase5.md:
cache_schema.py:67/154, gpuslot_vr.py:65-68/81/309/323/368, passes_vr.py:307/370/422,
erosion_vr.py:232/310-317/497, bake_vr.py:45/200/211/236/711/721/727/837-839/900/948/968,
gl_vr.py:594-597 (portability + VK_EXT_memory_budget — exact), review_p4.md:244-250,
vr.py:63/302/312/324/333/356/371/378/385/395/404/415/553-554/858/976-977/1015-1018/1091/1104-1117,
pano.py:46-47, phase4 §8 "12 frozen descriptors" — **all land where claimed.** The
four blemishes are N2/N3/N4/N5 (line-level pointers, one inherited from my own
review_p4); nothing fabricated wholesale.

## 7. Honest-limits section (check 7)
phase5.md:253-285 lists **eight** unknowns, covering all four the brief demands:
memory cap (recommendedMaxMemorySize/watchdog), coherent-RMW correctness (with the
"no atomic fallback exists" consequence stated), 946-tile wall-time cost, and
player metadata readability — plus layout behaviour, orientation parity, viewpoint
fit after re-bake, and pole-darkening magnitude. RUN_FIRST keeps ESTIMATE labels on
every non-derived number and refuses to invent per-tile rates (0.46 s phase0 figure
explicitly disclaimed as not a tile rate). Truthful.

## 8. Verdict
**ACCEPT-WITH-FIXES.**
Mandatory before any real first-run session:
1. **M1** — fix `package_vr.collect` (package_vr.py:227) to accept absolute globs.
Recommended in the same pass (cheap, all one-liners):
2. N1 — RUN_FIRST §3 self-test: add `--no-spec-check` to the quoted command.
3. N6 — probe #6: `cell_m=8.0` (or grid step 4.0).
4. N7 — clamp `--crop-lat` to (0,1].
5. N2/N3/N4/N5/N8/N9 — citation/wording polish (the :166→:165 fix should ride with
   the others since the matrix is the 4am lookup surface).
None of these changes any probe's pass/fail semantics. The ladder ordering
(silently-wrong first), the D6 hedge framing, the exit-code contracts, and the
memory math are sound; the manual is fit to gate a real GPU session once M1 lands.

---

## 9. Re-verification of the P5 fix round (reviewer, same-day round 2)

Scope: READ-ONLY reruns of every claimed fix + adjudication of the N6 rebuttal.
Verdict **ACCEPT-WITH-FIXES** - every claimed fix verified in place, but the
rebuttal round surfaced one NEW MAJOR (item 7) in the same probe family. One
doc-only mandatory fix, then tag.

**1) M1 collect() - VERIFIED live, no ffmpeg spawned.** package_vr.py:229-243 now
routes metacharacter patterns to glob.glob(g, recursive=True) (import glob :44)
and plain paths straight to Path. Runs: absolute glob OK; absolute '**' OK (5 pngs);
relative gate-route pattern OK; explicit path OK; unmatched -> []; glob+explicit
dedup -> 1 OK; end-to-end --in ABS_GLOB --out-dir ... --video --dry-run exits 0,
prints the ffmpeg line, writes nothing (mixed mono|SBS dir still hits the ERROR/rc!=0
guard). No spawn surface: grep subprocess/os.system/execvp/popen matches only the
docstring denial (:11).

**2) N7 crop-lat - VERIFIED.** Out of range -> argparse rc 2 for 0, 1.5, -1 (guard
package_vr.py:331-336, 0.0 < x <= 1.0); 1.0 rc 0. The UNREQUESTED extra fix is
correct and was needed: --crop-lat 0.001 on the 256-row test pano now emits
crop=iw:2:0:126 - y floored to EVEN (package_vr.py:133-140) where the old math
gives y=127, an odd offset that misaligns 4:2:0 chroma. 0.5 -> crop=iw:128:0:64.

**3) N6 rebuttal - AUTHOR RIGHT, reviewer retracts.** f = res // r
(erosion_vr.py:1486, res = h0.shape[0] :1470) is shape arithmetic; cell_m scales
into cell = cell_m * f only AFTER f is computed, so cell_m=8.0 could not rescue an
8192-level stage fed a 4096^2 grid - f = 0 regardless, and _pool divides by it at
:533. Keeping P1's 4.0 grid (8192^2, f = 1) is also the self-consistent far-map
declaration (4 m = 2*FAR_HALF/FAR_RES, cache_schema.py:37-38). My original N6
mis-stated which side of the grid/cell mismatch the probe sat on; the surviving
manual wording ('a step-8.0 grid is 4096^2') is accurate - RUN_FIRST:504-513 is
code truth for the shape math. But see item 7: neither the old nor the new #6 command
can actually reach that math with this terrain, and the claimed first death is wrong.

**4) Section-3 self-test lines - VERIFIED VERBATIM.** Both bash lines run as
printed: A (height 12) skim -10.3 m / waterfall -9.8 m BELOW-SURFACE,
'RESULT: 2 flagged', rc 1; B (height 0) 'RESULT: ALL CLEAR', rc 0. --no-spec-check
pre-existed (check_viewpoints.py:169-171) and the file is untouched this round
(mtime 09:15 < package_vr.py fix 09:34). The without-flag behaviour the doc now
states (both fixtures rc 1 on SPEC-MISMATCH) matches my round-1 measurement.

**5) Tripwires - REAL and PINNING (engine_vr/tmp/test_p5_fixes.py).** Full run: 40
checks PASS incl. the discipline assert (subprocess/os/pty/multiprocessing patched
to raise; final 'nothing spawned - []'). Grep-confirmed one pin each: N2 comps is
passes_vr.py:165 (:356; RUN_FIRST:537/:747 cite :165); N3 no uCamRot symbol exists
in gl_vr.py (:363) and matrix row 17 now cites manifest.json:315-317, uCamRot
mat3 offset 1088 (verified in manifest); N4 to_sampler_layout def at :422 and
RUN_FIRST:754 routes the fallback through it (:380-393); N5 raise bake_vr.py:774
pinned + row 23 cites it (:396-403; RUN_FIRST:157/:218); N8 43 x 22 = 946
(:411-413; phase5.md:190 fixed); N9 no cache_schema.world_xy, RUN_FIRST:399 cites
bake_vr.world_xy (:399).

**6) Collateral - clean.** GATE P5 PASS re-run; ruff --select E,F rc 0 on the three
changed/touched files (the '31 errors' figure is bare-default-rules on the tmp
test file, outside the gate's E,F scope). No writes by me outside this appendix.

**Reviewer correction to my own section 8:** mandatory-fix #3 ('cell_m=8.0') was
wrong - delete it; the shipped fix (keep the 4.0 grid) is correct.


**7) NEW MAJOR R1 - the synthetic-cone erosion probes (#1 verbatim, #2 derived, #6
rewritten THIS round) die in lake protection before any level math, on any machine,
GPU or not.** build_protection requires every named water body's seed below its
level: 'if not h0[r, c] < level: raise' (erosion_vr.py:730-731), and the water
component must be closed inside the central window (:733-734). The ladder's cones
put the apex at the lake seed: LAKE_C = (110.8, 0) with LAKE_LEVEL = 0.0 (:478-479)
-> on the 300 m cone (RUN_FIRST:315-316 #1, :493-496 #6) h(seed) = 293.5; on #2's
120 m cone h = 113.4. LIVE REPRO (erode called directly, no host): RuntimeError
'lake: seed (111, 0) is dry in the input (h=293.5 m >= 0.0)' for BOTH STAGES[0] and
STAGES[-1] - so the author's ':533 ZeroDivisionError' narrative for the step-8 case
is also not the first death (protection runs at erode :1473 before the level loop;
and run_erosion opens the device at :442-443 BEFORE erode, so 'the probe dies
before it opens a device' is wrong for the run_erosion form too). And #6's rewrite
ships this terrain unchanged - its f = 1 timing loop is unreachable. Tarn is a
second landmine: TARN_LEVEL = 520 > the cone's max 300, so h < 520 fills the whole
central window -> 'water body not closed' (:734) once lake is fixed. Phase3
lineage, but the P3 review also missed it - my error of coverage, recorded.
Mandatory pre-run fix (doc-only): drop hfar= from #1/#2/#6 so run_erosion loads the
REAL cached map (hfar=None -> h_far_raw, erosion_vr.py:436-438, which exists after
B1 and passes protection by construction - the production erode chain runs it
through the same build_protection), keep stages= as the cost knob; the 'cone still
recognisably eroded' expectation wording must become a real-world expectation. If
synthetic terrain is wanted instead, carve a closed sub-0 lake basin at (110.8, 0)
AND a tarn basin ringed above 520 within the central 2048 m window - strictly more
work, no gain.

**Reviewer correction to my own section 8:** mandatory-fix #3 ('cell_m=8.0') was
wrong - delete it; the shipped fix (keep the 4.0 grid) is correct.

**FINAL VERDICT (round 2): ACCEPT-WITH-FIXES.** M1 and N7 verified live; N1
verbatim; N6 belongs to the author on the cell_m question; N2-N9 pinned by passing
tripwires; gates green. Sole mandatory item: R1 (RUN_FIRST-only, ~6 lines) - until
it lands, the first GPU session dies at probe #1 with a protection error whose
matrix remedy points at the wrong thing (matrix 23 'different height map').
After R1: commit + tag p5-pass.