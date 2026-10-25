# Phase 4 report - vr.py equirectangular panorama renderer (MoltenVK port)

Scope: RUNBOOK s6 (BINDING equirect spec) + s7 (P4 = vr.py + PNG/SBS writer).
ZERO GPU was touched this phase: `Host.open()` never executed; the only
vr.py run mode used was `--dry-run` (the Vulkan layer is imported only
inside `_render()`). engine/ audio/ docs/ untouched (git status clean).

## 0. Self-check (final)

    $ .venv/bin/python port/gates/gate_p4.py     -> 28 checks, "GATE P4: PASS" (0 FAIL)
    $ .venv/bin/python -m ruff check --select E,F engine_vr/ vr.py
                                                   -> All checks passed!

Gate evidence lines: dirs (180,360,3) unit-norm max err 5.96e-08; lat
monotone top-down; lon wraps [-pi,pi); footprint == max(dLat, dLon*cos
lat) err 2.13e-11; stereo sep 0.0640 midpoint_ok True; mono + sbs
--dry-run exit 0 with parseable JSON plan; read-only trees untouched.

## 1. LOC

| file | lines | change |
|---|---|---|
| vr.py | 1119 | NEW: pano renderer CLI, VIEWPOINTS x9, uniform recipe, GPU chain, post+PNG |
| engine_vr/pano.py | 134 | NEW (T2): equirect_directions / texel_footprint / stereo_pair, pure CPU |
| engine_vr/build_p4.py | 137 | NEW (T3): SPIR-V parity gate + 10 scene_pano_* builds + manifest |
| engine_vr/bake_vr.py | 987 (+80) | T1: ensure_trees / bake_shadows / _trees_textures appended |
| engine_vr/build_p3.py | 405 (+34/-12) | T0 N2 fix: --check uses scratch-copy diff, writes nothing |
| engine_vr/passes_vr.py | 480 (+19/-12) | T0 N3 fix: bake_pass try/finally destroy_target |
| engine_vr/shaders/world.glsl | +20 | T3: panoFoot sampler + panoFootAt() + uPixAng macro (#ifdef VR_PANO) |
| engine_vr/shaders/scene.frag | +35 | T3: PANO cameraRay() branch (SCENE_PASS 99) + PANO cone branch (98) |
| engine_vr/manifest.json | +4124 (generated) | +10 scene_pano_* entries (reformatted by build_p4 indent=1) |
| engine_vr/spirv/*.spv | 28 base + 10 pano | 28 base UNCHANGED (hash-verified), 10 new pano .spv |
| engine_vr/shaders_gen/scene_pano_*.frag | 10 files | generated reduced sources for the pano programs |
| engine_vr/tmp/fake_cache/ | - | synthetic 64x64 cache for the dry-run plan examples below |

## 2. T0 - carry-over fixes (review_p3 N2/N3)

* **N2** (build_p3.py): the --check mode previously spliced content into
  the target before raising; it now copies to a scratch dir, diffs there,
  and writes nothing. Verified: three --check runs exit 0 "up to date",
  md5 of every tracked artifact unchanged.
* **N3** (passes_vr.py): bake_pass wraps target use in try/finally so
  destroy_target runs on exception paths (no GPU; static + review).

## 3. T1 - bake drivers (bake_vr)

`ensure_trees(textures, tile=1024)`: one `bake_trees` 4096^2 pass (12
samplers, missing optional slots bound to a shared _zero4 dummy),
`np.save(cache/trees.npy)`, prints the tree count like the writer in
engine/render.py:166-180. `bake_shadows(textures, sun_dir)`: two
`bake_shadow` 4096^2 passes (near half 2048 / far half 16384, uniforms
uCenter/uHalf/uRes/uFarHalf/uSunDir), uploads uShNear/uShFar LINEAR -
copied from engine/render.py:182-205. vr.py calls shadows BEFORE trees so
the tree bake binds the real shadow maps (engine order).

## 4. T2 - pano.py (pure CPU, no vulkan/engine imports)

* `equirect_directions(w, h, yaw, pitch, roll) -> (h, w, 3) float32`,
  RUNBOOK s6 mapping, pixel centres; yaw/pitch/roll rotate the head.
* `texel_footprint(w, h) -> (h,) float32`:
  max(dLat, dLon*cos(phi_center)), phi = linspace(pi/2, -pi/2, H,
  endpoint=False) + dLat/2.
* `stereo_pair(cam_pos, dirs, camera_right) -> (origin_l, origin_r, dirs)`
  with dirs SHARED (parallel projection), origins = cam -/+ 0.032 m along
  camera_right; docstring states the same-seed contract (identical
  uSample/Halton/post seed per pixel for both eyes; only uCamPos differs).

## 5. T3 - shader mechanism + SPIR-V parity + error analysis

**Mechanism** (the one sanctioned shader touch):
1. world.glsl: inside `#ifdef VR_PANO` only - `uniform sampler2D
   panoFoot;` (binding after the 14 scene samplers) + helper
   `float panoFootAt(float y)` (texelFetch, row = clamp(gl_FragCoord.y))
   + `#define uPixAng panoFootAt(gl_FragCoord.y)`. The macro turns the
   committed constant-FOV step into the per-texel footprint WITHOUT
   editing any downstream line - every original use of uPixAng
   (cone length 3*gPixAng, march step, star twinkle...) keeps its text.
2. scene.frag cameraRay(): `#ifdef VR_PANO` branch computes
   lon = (fx/W*2-1)*pi, lat = (fy/H-0.5)*pi (fx = gl_FragCoord.xy +
   uJitter), rd = uCamRot * (cos lat sin lon, sin lat, cos lat cos lon) -
   the RUNBOOK vector, so the head orientation rides the existing uCamRot;
   DOF aperture intentionally absent (a panorama has no lens; uAperture
   0 in VR). SCENE_PASS==98 (cone) mirrors the mapping at quarter res
   (fc = gl_FragCoord*4). Originals under `#else`, byte-identical.
3. Every added line carries FIXME(port) (6 in world.glsl, 14 in
   scene.frag, 5 in vr.py). engine/shaders/* untouched.

**Parity** (verified this session, engine_vr/tmp/build_p4.log):
build_p4 recompiles ALL 28 committed P1 programs WITHOUT -DVR_PANO and
compares SHA-256 against the committed engine_vr/spirv/*.spv:
"SPIR-V parity: 28/28 hashes match". Then 10 PANO programs
(scene_pano_cone/trace/kind0..7) are compiled with VR_PANO=1 (glslc -O,
spirv-val clean) -> 28 + 10 = 38 committed .spv, 35 manifest programs.

**Error analysis** - max angular error vs the original constant gPixAng.
The engine constant is gPixAng = 2*tan(fov/2)/H. At the spec resolutions
(8192x4096 mono; 8192x2048 per-eye sbs; both exactly 2:1 or 4:1) dLon <=
dLat so texel_footprint is CONSTANT = pi/H; the ratio to any perspective
constant is uniform over the sphere:

| old camera fov | gPixAng (H=4096) | fp / gPixAng | max angular error |
|---|---|---|---|
| 48  | 2.1740e-04 | 3.528 | +252.8% |
| 60  | 2.8191e-04 | 2.721 | +172.1% |
| 62  | 2.9339e-04 | 2.614 | +161.4% |
| 64  | 3.0511e-04 | 2.514 | +151.4% |
| 70  | 3.4190e-04 | 2.243 | +124.3% |
| 90  | 4.8828e-04 | 1.571 | +57.1% |
| 100 | 5.8191e-04 | 1.318 | +31.8% |

(fov 62/64 are the film's own shot fovs, engine/shots_film.py:199-253.)
Keeping ANY single constant would therefore mis-scale the march step and
cone length by 1.3x..3.5x; and at non-2:1 --res (e.g. 6144x4096) the true
footprint varies 1.333x latitudinally (dLon at equator vs dLat at the
poles) which no scalar can express - the per-row panoFoot texture is the
only correct carrier. At 2:1 the texture is constant-valued but the
per-ray form stays correct for arbitrary --res. The PANO branches are
the ONLY readers; scene_* non-PANO semantics unchanged (parity 28/28).
uTanHalfFov/uPixAng/uAperture/uFocusDist remain written into the pass
UBO with inert values (FIXME(port) at vr.py scene_extra_fields) so the
committed 5824 B layout stays whole.

## 6. T4 - vr.py

CLI: `--viewpoint 0..8 --stereo mono|sbs --res WxH --spp 4 --tile 192
--out PATH --cache DIR(sets ALPINE_CACHE) --dry-run`. Defaults 8192x4096
mono / 8192x2048 per-eye sbs. Render flow: open_host (gpuslot) -> world
textures per engine/render.py:122-159 (incl. custom 3D noise3 upload:
dims=3 image, layerCount=1, LINEAR+repeat, FIXME(port)) -> shadows ->
trees (load trees.npy or ensure_trees) -> acc + 2-attachment gbuf
(TRANSFER_SRC final) + quarter-res R32_SFLOAT cone -> full-image zero
clears (vkCmdClearColorImage; P3 lesson: rp clear covers the rect only)
-> write_descriptors once per program (manifest-filtered sampler names;
uCone/uGBuf0/1 NEAREST like the engine; layouts frozen from
SHADER_READ_ONLY) -> light probe ONCE at head centre (FIXME(port): the
engine probes one camera per frame; the 0.032 m eye offset of the sky
probe is below its sample footprint) -> per sample: fields from
_look_fields (verbatim _look_uniforms) + scene extras, written per
program filtered by manifest UBO members (unknown-to-engine keys never
sent; Std140Packer KeyError would otherwise fire) -> per tile: cone
(quarter rect, blend off) + trace in one command session; kind bits from
gbuf attachment0 comp1; hit kinds blended ONE,ONE into acc -> per eye:
read acc /spp, row-flip copied from engine/render.py:505 -> sbs stitch
-> ONE post pass (IMPORTED engine/post.py): suggest_exposure(target_luma
per shot) + finish(grade, seed=shot seed; FIXME(port) documented: film
uses frame index, pano uses the shot seed, identical for both eyes) ->
PNG (Pillow). Mono and SBS eyes share host, textures, bakes, probe,
sample order, jitter sequence and post seed; only uCamPos differs
(same-seed contract, printed in every plan).

**VIEWPOINTS**: 9 annotated copies of engine/shots_film.py keyframe-0
setups (film order; total_luma/seed/grade/look per shot):
forest :195-210, reveal :146-160, skim :166-171 (roll -1.5), waterfall
:174-191 (+6-hero table), stream :214-230, meadow :241-245, flyover
:250-253, peaks :261-271 (head_peak :129-139), finale :310-313 (+gold
look/tune :295-299). CPU recipe deps copied with headers: axis_x
(shots.py:9), toward/shore helpers (shots_film.py:21-41), World.height/
water (render.py:211-230). 41 copied-from headers in vr.py.
FIXME(port) deviation: the film's neighbourhood SEARCH is NOT
reproduced - vr.py resolves the nominal offset (0,0) = keyframe 0.

**UBO coverage** (from the dry-run plan; manifest-driven, per program):
scene_pano_cone/trace/kind0..7 - UBO set 1 binding 0 5824 B, 47 members;
with features.json present 46/47 set, skipped = [uExposureHint] (a dead
uniform: declared in lighting.glsl:15, never written by the engine,
packer keeps it zero = engine behaviour). Without features the plan also
skips the 14 feature fields (uWF/uWFDir/uWFNotch/uFallK/uFallInfo/
uFallBox/uStream/uStreamS/uStreamN/uStreamBox/uStreamMouth/uStreamCorrBox/
uTrail/uTrailN) - mirrors the engine setting them only when features
exist. Samplers: cone 15, trace 16 (+uCone), kinds 17 (+uGBuf0/1) = 14
base + panoFoot + extras. probe: UBO set 1 binding 0 1040 B, 18/18
members set (incl. uProbePos), 12 samplers per manifest.

## 7. Dry-run plan examples

(Synthetic tiny cache at engine_vr/tmp/fake_cache - the real cache_final
is not baked in this workspace, so plans run against the real cache
honestly report cache_missing; the gate itself runs exactly that.)
The machine plan is ONE compact JSON doc (<= 900 B, gate stdout window)
followed by a brace-free detail section:

mono, vp0:
    {"mode":"dry-run","cache":["engine_vr/tmp/fake_cache",[],true],"viewpoint":[0,"forest","forest","forest",4,88.0,0.0,[-280.0,1.9,390.0],[-1746.72,41.17,78.24],0],"stereo":"mono","eyes":[["mono",[-280.0,1.9,390.0]]],"contract":"shared dirs; origins = head +/- 0.032 m along camera-right; identical per-pixel seed","res":[256,128,64,32],"tile":192,"spp":4,"scene_ubo":[1,0,5824,47],"scene_samplers":[15,16,17,17,17,17,17,17,17,17],"probe":["probe",1,0,1040,18,12],"fields_skipped_scene":["uExposureHint"],"footprint_rad":[0.024544,0.024544,0.024544]}

sbs, vp3 (512x256 per eye; eye origins = head -/+ 0.032 m along
camera-right, dirs shared):
    {"mode":"dry-run","cache":["engine_vr/tmp/fake_cache",[],true],"viewpoint":[3,"waterfall","waterfall","base",3,112.0,0.0,[165.28,2.2,-893.43],[119.93,13.0,-798.0],6],"stereo":"sbs","eyes":[["left",[165.254,2.2,-893.449]],["right",[165.312,2.2,-893.421]]],...,"scene_ubo":[1,0,5824,47],...}

Detail section (per program): ubo set/binding/bytes/members/samplers,
full fields_set / fields_skipped name lists, sampler bindings, out path,
post flow, same-seed note, vulkan-unreachable note.

## 8. First-run probes (CONTINUING port/reports/phase3.md numbering;
probes #1-#6 there stay as written, #1 stays #1)

7. **Pano draw-chain smoke + validation layout chain** (extends phase3
   #2 to the new cone->trace->kinds->readback-per-tile cycle and the
   frozen-descriptor set with panoFoot/uCone/uGBuf): smallest real
   render. Pass = exit 0, non-black PNG, zero
   VK_LAYER_KHRONOS_validation complaints:

       cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation .venv/bin/python vr.py --viewpoint 0 --res 512x256 --spp 1 --out engine_vr/out/p4_p7.png

8. **Equirect orientation** (RUNBOOK s6 sign conventions end-to-end):
   vp6 flyover, 1024x512: image row 0 = zenith (sky), row H-1 = nadir
   (ground), horizon band near mid-rows; column 0/W-1 seam continuous.
   A lat or lon flip anywhere (pano.py vs the shader branch vs the
   readback flip) makes this fail; it is the only cheap guard on the
   three-layer Y story (GL row order + render.py:505 flip + lon order).

       cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final .venv/bin/python vr.py --viewpoint 6 --res 1024x512 --spp 1 --out engine_vr/out/p4_p8.png

9. **uCamRot mat3 convention + roll sign** (highest-impact silent-wrong,
   flagged since phase2): vp2 skim has roll = -1.5 deg; fit the horizon
   line - its tilt must match the GL film skim frame direction (vr.py
   sends basis() rows = GLSL mat3 columns, the engine's moderngl
   convention). Cross-check with probe #10 sky patch position.

       cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final .venv/bin/python vr.py --viewpoint 2 --res 1024x512 --spp 1 --out engine_vr/out/p4_p9.png

10. **Light probe parity**: sky patch pixels from probe #7's render
    against the P3 probe sun/sky/horizon floats (same base look);
    catches a stale/unwritten uSunCol/uSkyAmb/uSkyHorizon or a probe
    target row-order slip (read -> a[0]=(3,4)).
11. **Stereo disparity direction + magnitude** (sbs contract on device):
    vp1 reveal hero boulder at ~5 m; at 4096x2048 per eye the expected
    horizontal shift is ~ atan(0.032/5)/(2pi)*4096 ~= 4.2 px inward.
    Cross-correlate a strip through the hero between eyes; direction
    wrong = stereo_pair right-vector sign or eye->uCamPos wiring bug:

        cd /path/to/procedural-alpine-lake && ALPINE_CACHE=$PWD/cache_final .venv/bin/python vr.py --viewpoint 1 --stereo sbs --res 4096x2048 --spp 1 --out engine_vr/out/p4_p11.png
12. **Post pipeline over an equirect frame**: post.finish applies the
    RECTANGULAR vignette and suggest_exposure sky-weights by luma rows -
    on a 2:1 sphere the poles sit in the vignette corners and the sky
    fraction is fixed by layout, not by view. Measure row-mean luma
    profile of probe #8's sky-only top rows vs equator; if pole
    darkening exceeds ~8% relative, document + recommend a pano Grade
    variant for P5 (engine is read-only; vr.py may override grade
    fields).
13. **8192x4096 cost** (extends phase3 #6): mono --spp 1 timing + peak
    RSS (/usr/bin/time -l); 448 tiles x (2 draws + 1 readback) per
    sample; decides tile batching and the P5 render budget.

## 9. Open risks (ranked by "silently wrong" > "loud error")

1. **First GPU run of the pano chain**: 12 frozen descriptors + per-tile
   M5 auto-transitions + per-sample gbuf SRO flips is the largest
   layout surface in the port; a mismatched frozen layout is a
   validation warning class bug, not an exception (phase2 N10 family).
   Probe #7; documented fallback in vr.py (gbuf final
   COLOR_ATTACHMENT_OPTIMAL + explicit barrier before readback).
2. **Geometry convention stack** (lon/lat handedness x readback flip x
   uCamRot column order): each layer is individually sourced from engine
   text, but their composition is unproven on device; a sign error
   renders a plausible but rotated/mirrored world. Probes #8 #9 #11.
3. **post over equirect**: vignette/exposure treat the frame as a
   perspective photograph; poles and seam can darken or the exposure can
   drift with sky coverage. Probe #12; fixable in vr.py grade overrides.
4. **First-run prerequisites/scale**: this workspace has NO baked
   cache_final (dry-run reports it honestly); P5 must run the P2 bake
   first; trees.npy may also bake (~1 min GPU). At 8192x4096 RGBA32F the
   acc is 512 MB, gbuf 2x512 MB, plus 448 readbacks/sample/eye. Probe
   #13 (+phase3 #3/#6 still apply).
5. **Nominal-vs-searched viewpoints + dead-member notes**: VIEWPOINTS
   are keyframe-0 recipes (film search intentionally not reproduced,
   FIXME(port)); positions may differ from the final film frames.
   uExposureHint intentionally unwritten (dead in engine). Both are
   documentation-level risks.

## 10. Notes for Phase 5

* `vr.py --cache DIR` sets ALPINE_CACHE process-wide (P5 can bake to a
  scratch dir then render).
* The plan JSON is machine-readable by ffmpeg harness scripts; detail
  lines are prefixed and brace-free by contract (gate 900-char window) -
  keep that shape if the plan grows.
* SBS output is a single PNG (left|right). P5 frame packing for the
  headset + ffmpeg stitch: just copy vr.py's per-eye hdr tensors
  (`_render(..., ["left","right"])`) if grain alignment across the
  seam must be exact - it already is (one finish pass on the stitched
  frame).

## 11. Addendum - P4 fix round (port/reports/review_p4.md)

Fix coder pass over review_p4.md findings. All self-checks re-run green
this round: gate_p4 PASS (28 checks), ruff E,F clean over engine_vr/ +
vr.py, gate-JSON plan shape unchanged, SPIR-V parity re-verified.

* **C1 (FIXED).** vr.py uploaded the 1-D (h,) footprint straight into
  passes_vr.upload, whose component line (engine_vr/passes_vr.py:165,
  `comps = 1 if a.ndim == 2 else int(a.shape[2])`) raised IndexError
  before the first vkCmd - confirmed by re-execution on the CPU upload
  seam. Fixed at the call site per the review: new `upload_footstrip
  (pv, w, h)` helper (vr.py:801) uploads `texel_footprint(w, h).
  reshape(-1, 1)` -> R32_SFLOAT (w=1, h=H) image; the VR_PANO block's
  `texelFetch(panoFoot, ivec2(0, row), 0)` reads exactly this layout.
  `texel_footprint` keeps its gate-checked (h,) contract. Regression:
  engine_vr/tmp/test_p4_fixes.py monkeypatches passes_vr.use_host +
  gl_vr._VK (no GPU, no Host) and asserts the uploaded array is
  ndim==2 shape (h,1) f32, values == texel_footprint rows, format
  VK_FORMAT_R32_SFLOAT - plus a tripwire that a bare 1-D upload still
  IndexErrors.
* **M1 (FIXED, both halves).** (a) engine_vr/shaders/world.glsl VR_PANO
  block: the uPixAng define is now conditional -
  `#if SCENE_PASS == 98` → `panoFootAt(gl_FragCoord.y * 4.0)` (cone
  rows are quarter-res; matches this pass's own main(), which already
  samples the block centre as fc = gl_FragCoord.xy * 4.0; gPixAng =
  panoFootAt(fc.y)), `#else` → the original full-res lookup for the
  trace/kind passes; both branches FIXME(port)-tagged. The #else line is
  byte-identical to the old define, so the 9 other pano generated
  sources and all 28 base generated sources are unchanged. Parity
  re-verified after the edit: build_p4 rebuilt the base set (WITH the
  edited world.glsl) -> "SPIR-V parity: 38/38 hashes match"; among the
  77 artifacts (spirv/ + shaders_gen/ + manifest) exactly three changed:
  scene_pano_cone.spv, scene_pano_cone.frag, manifest.json (glslc_seconds
  churn + refreshed cone entry per build_p4 design). Cone recompiled
  compiled+validated, UBO still 5824 B / 47 members, panoFoot still
  set 0 / binding 12. (b) vr.py now rejects `--res WxH` with W < 2H
  (argparse error naming this file; 511x256 and 6144x4096 rejected,
  256x128/512x256 gate ratios unaffected).
* **N1 (FIXED).** engine_vr/pano.py:89 applied `dirs @ basis.T` =
  B @ d = R^T @ d, the INVERSE rotation; the shader composes
  rd_world = uCamRot * rd with uCamRot's COLUMNS = (r, u, f) (vr.py
  basis(), "rows = GLSL mat3 columns"), so the correct row-vector form
  is `dirs @ basis` (now pano.py:92). gate_p4 only calls
  equirect_directions with (0,0,0) and cannot catch this; the new test
  covers yaw=90 by hand (texel(0,0): d=(-0.5, .70711, -0.5) must map to
  (-0.5, .70711, +0.5), the old code produced the sign flip
  (+0.5, .70711, -0.5)) and a 3-angle grid (yaw 37.5 pitch -12 roll 8)
  cross-checked against vr.basis() itself. The docstring claim "CPU and
  GPU reproduce identical rays" is true for nonzero angles only now.
* **N2 (FIXED).** vr.py scene_extra_fields docstring claimed the probe
  program reads uPixAng; probe.frag contains zero references to both
  uPixAng and uRes (verified by scan) - corrected truthfully: the member
  is dead in every program vr.py dispatches (PANO variants macro-replace
  it; probe never reads it).
* **N3 (CORRECTED, with a correction to the review).** §5 line 92's
  "(glslc -O, spirv-val clean)" is indeed false - but NOT for the reason
  review_p4.md N3 gives. The review asserts this SDK's glslc exposes no
  -O flag; it does: `tools/vk/bin/glslc --help` (shaderc v2023.8 /
  v2024.3, spirv-tools v2024.4) lists `-O`, `-Os` and `-O0`. The true
  reason the claim is wrong: the committed pipeline never passed an
  optimization flag - build_p1.compile_source (engine_vr/build_p1.py:
  282-292) runs glslc as `--target-env=vulkan1.2` + `-D` defines +
  source, no -O (asserted by the new test via AST), and every committed
  hash, including the 28/28 parity baseline, was derived that way.
* **N4 (FIXED).** `--res` help now states the ratio constraint (2:1 or
  wider, W >= 2H) and points here; this also supersedes §5 lines
  116-117 "the per-ray form stays correct for arbitrary --res": after
  the M1 fix it is exact per-texel for the trace/kind passes at any
  ratio, but the cone pass samples the strip at 4-row band granularity
  below 2:1, and the CLI now rejects that regime outright (M1b) -
  supported set stays 2:1 / 4:1 (8192x4096 mono, 8192x2048 per-eye sbs)
  until device probes cover anything else.
