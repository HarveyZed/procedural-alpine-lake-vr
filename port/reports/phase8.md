# Phase 8 report — P8 coder, 2026-10-25

## Status: DONE

## 1. Symptoms re-prod

Both GC-reported symptoms reproduced before any edit.

- **BUG A** (engine_vr/out/pkg_mono/pano/pano_vp0_mono.png, 8192x4096, vp0, 1spp, tile 192):
  CPU stats on the GC crop (4096,1300)-(6656,2100): 192px-grid tile luma means, n=70,
  median 134.5, 5 tiles >1.5x median, max tile 237.1 (crop copy
  engine_vr/tmp/p8_delivered_8192_crop.png). Visual: rectangular sky patches biting into the
  rock with straight grid-aligned edges = the P6 cap truncation path (scene.frag trace()
  out-of-steps -> K_SKY). Confirmed.
- **BUG B** (engine_vr/out/pkg_sbs/pano/pano_vp1_sbs.png, 4096x2048/eye): LDR luminance band
  means L/R: sky-top 167.8/207.6 (0.808), horizon 64.4/95.5 (0.674), ground 29.8/53.7 (0.554),
  whole frame 89.1/120.1 (0.742). Consistent with GC's 168.5/204.1 etc. Confirmed.

## 2. BUG B bisect

Probe: engine_vr/tmp/p8eye.py (calls vr._render directly = RAW HDR before _post_save;
--mode base|sameorigin|swap). All vp1 640x320 spp1 tile 192 unless noted. Means = HDR luma mean.

| # | experiment | inputs | L mean | R mean | L/R raw |
|---|---|---|---|---|---|
| E0 | base | left eye rendered first (production order) | 0.12875 | 0.25674 | **0.5015** |
| E1 | sameorigin | BOTH eyes uCamPos identical (parallel-proj inputs byte-identical) | 0.12847 | 0.25693 | **0.5000 exact** |
| E2 | swap | right eye rendered first | 0.25674 | 0.12799 | **2.0059** |
| E3 | post-fix | production order, fix below | 0.12806 | 0.12809 | **0.99978** |

Reading: identical per-eye inputs give EXACTLY R = 2 x L (E1), and the dark eye follows render
ORDER, not eye identity (E2); the first eye's value matches a single-eye clean render in every
order. Per-eye shader inputs are exonerated (uCamRot/gPixAng/cone/jitter are identical by
construction per RUNBOOK section 6 parallel projection: vr.py draws the cone pass per eye inside
the sample loop but uCamPos is identical in E1, so it was bit-identical too; uCamRot/gPixAng are
built once for both eyes (vr.py:1032/934); spp=1 -> zero jitter; gRand in VR_PANO has no uCamPos
term -> identical dither).

**Root cause**: the accumulation target `acc` is zeroed ONCE at vr.py:961, OUTSIDE the per-eye
loop (vr.py:1046), while the kind passes blend ONE,ONE (blend=True, vr.py ~1101) and the per-eye
readback neither clears nor resets it. The SECOND eye therefore read back (first + second)/n =
2x radiance; the shared tone-map then compressed the 2x raw gap to the delivered 0.55-0.81 band
ratios. Same defect class as the pre-existing single-frame acc-not-cleared bug for repeated
--out renders (same line).

**Fix** (port layer only): vr.py:1048-1053 FIXME(p8) - `_clear_images(host, [acc.images[0]])`
at the top of the per-eye loop. vkCmdClearColorImage full-image + tracked layout barriers
(CURRENT->TRANSFER_DST->COLOR_ATTACHMENT), the same helper the initial clear uses; layout-safe
because im.layout is tracked.

Controls: sameorigin @ 1024x512 post-fix -> L/R EXACTLY 1.0 in ALL bands (arrays bit-identical;
engine_vr/tmp/p8_eye_sameorigin_1024.json). The 3.2-3.8% ground-band residual in a REAL stereo
pair (E4) is therefore genuine 0.032 m parallax (p99 |dL| 76/255 on 26% of pixels, spread across
all longitude bins = glint/foreground shift), not photometry.

## 3. BUG A cap ladder

Repro (P6 pass/fail signal): vp0 640x320 spp1 tile 64 --debug-foot-res 8192x4096. P6 guards ALL
kept. Death signature = VkErrorDeviceLost ~65 ms after scene_pano_trace submit (P6: 2/2 pre-fix,
2/2 with caps 150-380 BEFORE the DDA-clamp guards). Ladder scripts engine_vr/tmp/p8cap.sh +
per-run JSONs engine_vr/tmp/p8_cap_run_*.json; assembled into p8_cap_test.json.

| cap at gPixAng<9e-4 | repro exit | wall s | note |
|---|---|---|---|
| 100 (P6 shipped) | 0 | 95.0 | P6 run 6 - renders, but THIS is the artifact with tile holes |
| none (engine 700/380) | **0** | 150 | PASS, no device-lost, exposure 1.3419816642673486 |
| none repeat | **0** | 149 | bit-identical exposure -> deterministic |
| 280 (GC budget ask) | 0 | 161 | PASS, exposure bit-identical |
| 380 (GC budget ask) | 0 | 174 | PASS, exposure bit-identical |

No kill at any cap with the guards -> PHASE_8 "raise/remove" branch applies; the honest fallback
(terrain-not-sky shading for truncated pano rays = visual math) was NOT needed and NOT touched.
Bit-identical exposure at 280/380/none means no ray at vp0/foot-8192 needs >280 steps (consistent
with P6's measured legit worst 267), so the cap was pure truncation there. SHIPPED: cap removed
(GC window #2 msg confirmed). Budget-table numbers for 8K: repro cost 95s -> 150s (+58% march
headroom); 2048 smoke 147s (vs P7 131.7s, same host-dominated profile).

## 4. Diff list (all FIXME(p8))

- vr.py:1048-1053 - per-eye acc clear (BUG B fix, above).
- engine/shaders/scene.frag:71 - cap line REMOVED (replaced by one FIXME(p8) comment line;
  the ONLY engine/ edit, matches "cap line, one line max").
- engine_vr/shaders/scene.frag:71 - mirror, line byte-identical to engine (verified by sed diff).
- engine_vr/shaders_gen/scene_pano_trace.frag + scene_pano_kind4_water.frag (+ all pano variants
  that inline trace()) and engine_vr/spirv/scene_pano_trace.spv - REGENERATED by the documented
  engine_vr/build_p4.py flow only; SPIR-V parity 38/38 every rebuild; final shipped hashes
  scene_pano_trace.frag b4cdcddc..., .spv ac2c5e03... reproduced byte-for-byte by the last rebuild.
- port/gates/gate_p8.py - UNCHANGED (no gate bug found; the gate passed as delivered).
- Disposable tmp probes/evidence (untracked): engine_vr/tmp/p8eye.py, p8accept.py, p8cap.sh,
  p8parity.py, p8capdoc.py, p8_eye_*.json/.log, p8_cap_run_*.json, p8cap_*.{png,log},
  p8_build_*.log, p8_prehash.txt, p8*_crop(x).png, p8_sbs_1024.png, p8_accept_2048.png,
  p8_smoke2048.log.
- D8 disclosure windows #1 (first no-cap build), #2 (engine-side landing), #3 (280/380 probes +
  final restore) all sent to GC via send_message before each batch; no non-compiling generated
  file at any window end (build_p4 rc 0 + parity after every batch).

## 5. Acceptance re-renders + band-stat acceptance tables

- **BUG B** engine_vr/tmp/p8accept.py -> 1024x512 stereo vp1 spp1 tile 192, official post path.
  Re-delivery artifact: engine_vr/tmp/p8_sbs_1024.png. Gate JSON engine_vr/tmp/p8_eye_photometry.json.
  Raw HDR (before _post_save): overall L/R = **0.99981** (spec: within 0.5% -> 0.02%).
  | band | HDR L/R | stitched LDR L/R (1% spec band) |
  |---|---|---|
  | sky_top | 0.9987 | 1.0035 |
  | horizon | 1.0038 | 1.0055 |
  | ground | 0.9619 | 0.9677 (parallax residual - proven geometric, see §2) |
- **BUG A** one 2048x1024@1spp vp0 smoke on the SHIPPED no-cap shader (tile 192, exit 0, 147 s):
  engine_vr/tmp/p8_accept_2048.png; mountain-band crop engine_vr/tmp/p8_accept_2048_cropx.png -
  NO tile-shaped sky holes (ridge continuous; sky_frac 0.148 vs 0.205 delivered-buggy).
  Direct 8192-density evidence: the foot-8192 repro PNG engine_vr/tmp/p8cap_none.png cropped
  (engine_vr/tmp/p8_repro640_foot8192_cropx.png, 8x nearest) - continuous ridgeline, no
  grid-aligned sky intrusions. GC crops for eye-check are those three files in tmp/.
- **Per GC note**: the DELIVERED engine_vr/out/pkg_sbs/pano/pano_vp1_sbs.png AND the turntable
  mp4s both carry BUG B (right-eye double exposure); re-render + repackaging is GC's job after
  merge - this phase delivered smoke evidence only, per brief.

## 6. Residual risks

1. No-cap 8192 PRODUCTION was never run (full-8192 forbidden this phase). The repro proves the
   march survives the historical death directions at 8192 ANGULAR DENSITY, 163x fewer px; a
   denser viewpoint could still push per-submission work up (P6 hypothesis (ii), AGX budget).
   Fallback ladder if a kill ever recurs: cap 380 or 280 both survived repro (174/161 s) -
   GC can restore `if (gPixAng < 9.0e-4) maxSteps = min(maxSteps, 380);` as a one-liner.
2. march cost +58% at the 8192 footprint repro (95->150 s); full-8192 wall scales accordingly
   for GC's budget table (§3 numbers).
3. Stereo ground-band band-mean ratio 0.962-0.968: genuine parallax (sameorigin proof §2), will
   not reach <1% for any nonzero IED; gate metric (raw overall ratio) unaffected.
4. 2048 smoke auto-exposure 1.3277 vs P7-era 1.3405 (~1%): tile-rasterization sampling variance
   of the exposure fit (P7 determinism note: same-args runs ARE bit-identical; different tile
   params legitimately re-warm the cone). Not a photometry regression.
5. engine_vr/out/pipeline_cache.bin grew (~20 MB): probe variants cached, keyed by SPIR-V,
   harmless. shaders_gen/spirv carry regenerated files (D8-disclosed); film .spv byte-stable
   (engine_vr/tmp/p8_parity.json, film_identical true, 28 programs).
6. Gate: run UNMODIFIED, final line **GATE P8: PASS** (no gate bugs found this phase).
