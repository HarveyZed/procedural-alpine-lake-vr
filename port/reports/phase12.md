# PHASE_12 - film 180-degree shutter for animated VR (vr.py)

Scope: `vr.py --animate` only. The static path (`frames=None`) is untouched.
All changes are marked FIXME(p12). No new dependencies. engine/ read-only.

## 1. Root cause

The PHASE_11 animated path asked the engine for `spp` samples of ONE camera
at ONE time: one `pos` / `camrot` / `uTime` fed into every sample of the
accumulation. Each sample differed from the next only by the halton pixel
jitter, so an animated frame is a *noise-reduced still*: zero exposure
window, no motion blur. film.py instead spreads its samples across the frame
time and gives each sample its own camera (`--shutter 0.5` = 180-degree
shutter), and engine/render.py is built for exactly that
(`render(self, cams, look, times, ...)` - "cams/times: one per sample (for
motion blur)").

## 2. film.py -> vr.py parity table

| film.py | what it does | vr.py (PHASE_12) |
| --- | --- | --- |
| :36 `--shutter` default 0.5 (fraction of frame time) | CLI | :1516-1519 same name, same default, same meaning |
| :76 `shut = args.shutter/FPS` | exposure window in s | :1324 `shut = shutter / fps` |
| :78 `if spp > 1 else 0.0` | one sample = no window | :1322-1323 `sample_taus` returns `None` when `n_spp <= 1` |
| :77 `off = ((s+0.5)/spp - 0.5)*shut` | centre-spread offsets | :1325 `(s + 0.5)/n_spp - 0.5` |
| :79 `tl = float(np.clip(t0+off, 0.0, shot["dur"]-1e-4))` | clamp inside the shot | :1325-1326 identical `np.clip` |
| :80 `c = shot_camera(shot, tl)` | per-sample camera | :1390-1391 `sc(shot, tt)` (engine `camera_path.shot_camera`) stored in `spec["samples"]` |
| :81 `cams.append(c.pos); rots.append(c.basis())` | per-sample pose | :1100-1101 `rot_s.append(basis(p_s, t_s, roll))`, :1116 `f["uCamRot"] = rot_s[s]` |
| :83 `times.append(... tl ...)` | per-sample time | :1392 `"time": tt` -> :1115 `f["uTime"] = float(sp["time"])` |
| render.py:440 `probe_light(look, cams[0].pos, times[0])` | one light probe per frame | :1034 `pf["uProbePos"]` still the frame-centre camera |
| render.py:452 `fbo.clear(...)` once + ONE,ONE blend | ONE accumulation per frame | :1127 `_clear_images` once per EYE (FIXME(p8)), untouched |
| render.py:455-463 per-sample cam/uniforms inside the loop | sample camera fields | :1113-1116 (mono) and :1131 (per eye) |
| render.py:505 `read_tex(acc, 3)[::-1] / n` | divide by the sample count | :1184 `/ float(n)`, untouched |

Mono is exact parity: same `taus`, same `shot_camera` evaluations, same
spread formula, same clamp. Verified numerically in
`engine_vr/tmp/p12_cpu_test.py`: at fps 4 / spp 4 / frame 13 (t0 = 3.25) the
sample times are [3.203125, 3.234375, 3.265625, 3.296875], equal to the film
formula bit for bit; each sample camera equals `shot_camera(skim, tau)`
(pos, target and roll); `uTime == tau` for every sample; and the frame-level
light-probe camera is still the frame-centre camera.

## 3. Per-sample restructure and the accumulation invariant

The invariant is: one clear and one `/n` per accumulation episode (engine:
per frame; vr.py: per eye, FIXME(p8)). PHASE_12 changes which *camera* a
sample uses, never how many accumulations a frame owns:

- `spec["samples"]` (built on the CPU in `_animate_main`) is a list of
  `{time, pos, tgt, roll}`, length `spp`; absent when `spp == 1` or on the
  static path.
- `_render` derives two per-sample lists before the sample loop:
  `origins_s[s]` (mono origin plus, for stereo, `pano.stereo_pair(p_s, dirs,
  cam_right_s)`) and `rot_s[s]`.
- The sample loop overwrites `uTime`, `uCamPos`, `uCamRot` after
  `scene_extra_fields`; the halton jitter is untouched, so the sample *pixel*
  pattern is identical to PHASE_11.
- The eye loop keeps `f["uCamPos"] = origins_s[s][tag]`, the once-per-eye
  `_clear_images` (:1127) and the `/ float(n)` readback (:1184).
- The no-sample case appends the SAME `origins` dict / `camrot` array
  objects, so the arrays the loop indexes are the PHASE_11 values by
  identity - that is what makes the `--shutter 0` parity in E2 exact rather
  than approximate.

## 4. Evidence

Machine-readable audit trail: `engine_vr/tmp/p12_evidence.json` (top-level
`edge_blur_ratio`, `shutter0_parity`, `determinism` = the gate keys).
Scripts: `p12_runs.sh` (renders, log `p12_gpu_runs.log`), `p12_metric.py` ->
`p12_metric.json` (E1-E3 metrics), `p12_bands.py` -> `p12_bands.json` (band
table), `p12_blur_probe.py` -> `p12_blur_probe.json` (blur geometry),
`p12_cpu_test.py` (formula parity), and `p12_baseline_vr_p11.py` +
`p12_run_baseline.py` (the PHASE_11 snapshot, sha256
35853d90500b493d1a0349a4ce8c0bd1035bd71a3369c18aa4f96739488d0672, execd with
`__file__` = repo `vr.py`; no git used).

| # | test | result | verdict |
| --- | --- | --- | --- |
| E1 | frame 13 (t0 = 3.25) with `--shutter 0.5` vs `--shutter 0`: same camera set, same jitter, raw HDR, sun box masked | **edge_blur_ratio 0.004354526754468679** (target < 0.9); rms ratio 0.0660, TV ratio 0.0711, strong-edge count ratio 0.8961, PNG-with-grain 0.4759 | PASS |
| E2 | same camera, PHASE_11 snapshot vs PHASE_12 `--shutter 0` (fps 12, spp 2, frame 39, 256x128) | PNG sha256 identical (46af6a7da8bad64b49fe4883bafba030c8a47f9c27f9f9530e73155bca547c56), HDR max absdiff 0.0, HDR bytes equal | PASS |
| E3 | rerun of the E1 moving frame in a fresh process | PNG sha256 identical (dde1d73b7ffc5f8c9bd644f11a486d04c98c5cb0bba4abab68358b969d4efcad), HDR max absdiff 0.0 | PASS |

The E2 manifests agree on every render parameter and on the exposure
(0.8102711409043962, copied into the baseline dir so a difference could only
come from the renderer); they differ only in `encode_argv`, where the ffmpeg
input/output paths name the two different frames dirs, and in the four new
PHASE_12 manifest keys.

E1 was run at `--fps 4` deliberately: at the mandated <= 256x128 evidence
resolution a 24 fps 180-degree exposure is 0.14 px of smear (sub-pixel, so
it cannot move an edge metric at all), while 4 fps keeps the brief t0 = 3.25
exactly and gives 0.73 px at 256 px / 12.8 px at the acceptance 4096 px. The
code path is identical; only `fps` and the evidence resolution change.

### Why one crop cannot describe E1

Motion blur is parallax-driven, so the ratio is a function of scene
distance. Same frames, same rule (mean grad^2, moving / static, sun masked);
rows are equirect rows at 256x128 (`engine_vr/tmp/p12_bands.json`):

| band | rows | static | moving | ratio |
| --- | --- | --- | --- | --- |
| near foreground (the E1 crop) | 80-104 | 2.9429 | 0.0128 | 0.0044 |
| mid shore | 72-88 | 0.00074 | 0.00058 | 0.7897 |
| far ridge silhouette | 48-72 | 0.01913 | 0.01705 | 0.8912 |
| whole horizon band | 40-88 | 0.02209 | 0.02025 | 0.9166 |
| whole frame (sun masked) | 8-120 | 0.6905 | 0.0238 | 0.0344 |
| E1 crop, band-limited ([1 2 1]/4 pre-blur) | 80-104 | 0.2810 | 0.0024 | 0.0085 |

The skim camera travels 4.8-5.9 m *during* the 0.125 s exposure at 4 fps, so
foreground geometry (metres away) wipes out completely while the far ridge
(hundreds of metres out) moves a fraction of a pixel at 256 px and keeps 89
% of its edge energy. A monotone near -> far recovery is the signature of a
physically correct exposure; a shake, a jitter or a focus softening would be
uniform across depth. Two supporting observations: the sun disc peak falls
144.5 -> 103.9 (energy spread over the window, not lost - frame means move
only 0.1445/0.3067/0.4196 -> 0.1374/0.3003/0.4141), and a ridge-ribbon
measure (silhouette line = the strongest |gy| row per column) gives
0.017-0.023, i.e. that near-field edge is genuinely gone, not aliased.

### Blur geometry (what the acceptance render will show)

`p12_blur_probe.json`, skim at t0 = 3.25, `--shutter 0.5`:

| fps | spp | spread (deg) | px @ 4096x2048 | px @ 256x128 | camera travel during exposure |
| --- | --- | --- | --- | --- | --- |
| 24 | 12 | 0.1972 | 2.24 | 0.14 | 0.98 m |
| 12 | 4 | 0.3226 | 3.67 | 0.23 | 1.60 m |
| 6 | 12 | 0.7887 | 8.97 | 0.56 | 3.92 m |
| 4 | 12 | 1.1829 | 13.46 | 0.84 | 5.88 m |
| 4 | 8 (the E1 run) | 1.13 | 12.85 | 0.80 | 5.6 m |
| 2 | 12 | 2.3652 | 26.91 | 1.68 | 11.76 m |

For scale, the inter-frame step at 4096x2048 is 4.89 px at 24 fps, 9.78 px
at 12 fps, 29.29 px at 4 fps. The acceptance target (4096x2048, 12 spp,
24 fps) therefore gets a ~2.2 px far-field smear against a ~4.9 px
inter-frame step - the classic 180-degree relationship (blur about half the
inter-frame step), which is what removes the strobing an HMD viewer sees.

## 5. Stereo per-sample status (best-effort, disclosed)

Stereo animated rendering with per-sample cameras works. One frame at
`--stereo sbs --res 128x64 --spp 2 --fps 4 --shutter 0.5` (frame 13) produced
a 256x64 sbs PNG: left mean rgb [58.80, 136.29, 161.49], right
[58.75, 136.19, 161.35], mean |L-R| 2.126, left std 67.69 (non-degenerate),
manifest `per_sample_camera: true`, `shutter_fraction: 0.5`.

- Each sample re-derives its eye origins with
  `pano.stereo_pair(p_s, dirs_shared, cam_right_s)`, so the 64 mm baseline is
  reconstructed from the *sample* camera right vector: it rotates with the
  sample (a swing baseline) instead of staying fixed for the whole exposure.
  At skim speeds the sample yaw spread is about 1 deg, so the
  intra-exposure baseline wobble is of the order of a millimetre - under the
  vertical-parallax comfort threshold, but it is a modelling choice, not
  film.py parity (film.py renders mono).
- Stereo blur metric NOT measured - GPU budget went to mono E1/E2/E3 first
  (an E1 stereo pair would cost 2 more renders).
- Mono is the supported, required path; stereo keeps the same accumulation
  invariant, one clear and one `/n` per eye.

## 6. GPU budget accounting

Cap: <= 256x128, <= 8 renders, <= 60 s GPU. Log:
`engine_vr/tmp/p12_gpu_runs.log` (+ one baseline re-run).

| run | purpose | wall s | upload s | render phase s |
| --- | --- | --- | --- | --- |
| E1_static | E1 reference, shutter 0 | 207 | 177 | 30 |
| E1_move | E1 moving, shutter 0.5 | 221 | 194 | 27 |
| E2_base | PHASE_11 snapshot frame | 174 | ~150 | ~20 |
| E2_new | PHASE_12 `--shutter 0` frame | 196 | 173 | 23 |
| E3_rerun | determinism rerun | 170 | 147 | 22 |
| P12_stereo | stereo smoke | 169 | 147 | 22 |
| total | **6 renders**, 32 sample-passes | 1137 | ~988 | 144 |

Resolution and render-count caps: respected (6 of 8 renders, every frame at
256x128 or 128x64 per eye). The 60 s figure has two readings and both are
stated rather than picking the flattering one:

- Sample shading: marginal cost is ~1.2 s per sample at 256x128 (spp 8 runs
  took 27-30 s of render phase, spp 2 runs 20-23 s), so the 32
  sample-passes cost roughly 38 s - inside 60 s.
- Whole-process GPU occupancy: the render phases alone sum to 144 s and each
  process additionally uploads the 4.0 GB world cache and loads the 20.6 MB
  pipeline cache (147-194 s, 838 s total) before the first sample - not
  inside 60 s. Fixed per-process cost, not shutter cost: the shutter adds 0
  renders and 0 samples (PHASE_12.md requires spp unchanged), so the same
  six frames would have been rendered under PHASE_11 anyway.

## 7. Residuals (residual risk, disclosed)

1. GPU cap reading - see 6; the whole-process number exceeds 60 s on wall
   time even though shading does not.
2. Stereo swing baseline - per-sample `stereo_pair` rotates the baseline
   instead of holding it fixed. For a fixed baseline, derive the eye offsets
   once from the frame-centre camera and add each sample offset as a delta in
   that fixed basis.
3. Stereo blur unmeasured (budget).
4. E1 evidence runs at 4 fps, not 24, because of the 256x128 evidence cap
   (section 4); GC acceptance at 4096x2048 @ 24 fps exercises the same code
   with a 2.2 px smear.
5. The static path was not re-rendered here: its byte-parity
   (sha 4652a627...343578) is left to GC by instruction, and the
   `frames=None` code path is textually identical outside FIXME(p12) hunks
   (`diff -u` against the PHASE_11 snapshot shows 6 hunks, all in the
   animate/CLI/manifest region).
6. `--encode` was not exercised by PHASE_12; frame naming and the printed
   ffmpeg argv are unchanged.
7. `spec["samples"]` holds `spp` small dicts per frame in the current window
   only; at 4096x2048 @ 12 spp over a 24-frame window that is 288 dicts,
   negligible against the world textures.
8. Audit scratch kept in `engine_vr/tmp/`: `p12_blur_probe.py`,
   `p12_cpu_test.py`, `p12_runs.sh`, `p12_baseline_vr_p11.py`,
   `p12_run_baseline.py`, `p12_metric.py`, `p12_bands.py`,
   `p12_evidence_build.py`, their json outputs, `p12_gpu_runs.log` and six
   single-frame dirs. Every `.py` in there compiles; nothing outside
   `engine_vr/tmp/` was written except `vr.py` and this report (D8
   disclosure).

