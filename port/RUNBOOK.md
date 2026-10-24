# RUNBOOK — alpine-lake → macOS VR-360 port

Contract between the general contractor (GC) and coder agents. Read this fully, then read your
`port/PHASE_N.md`. Repo files not named as writable in your phase are READ-ONLY.

## 0. Deliverable & definition of done

Turn `engine/` (a moderngl/OpenGL-4.3 procedural renderer, Windows/AMD-only) into a renderer that
produces **equirectangular VR panoramas on this Apple M5 Ultra (macOS, Metal 4)**:

1. Mono equirect still, e.g. 8192×4096 PNG (head, configurable W×H).
2. SBS stereo still: two eye views (parallel projection, IED 0.064 m, ±0.032 m along camera-right),
   packed 2W×H side-by-side, same RNG seed both eyes (no temporal/noise mismatch between eyes).
3. Optional turntable video (P4): N mono/stereo frames → ffmpeg equirect 360 video.

Out of scope: audio/ (whole subsystem), film.py/film_shots choreography, encode, motion blur,
film aspect/framing. The visual quality bar is the 9 stills in docs/img/ at equivalent spp.

**First-run protocol (P4) is part of the deliverable** because of §2 — we ship code that has
passed static gates, plus an operator protocol for the user's one live run.

## 1. Hard constraints (every agent, every phase)

- **NEVER run the project.** No bake, no renderer, no `import moderngl` context, no device queue
  submit — the local LLM serving occupies unified memory. Sole historical exception: P0 spikes
  (user-granted). Only sanctioned live runs after P0: none, until the user initiates first-run.
- **No global installs, nothing written outside this workspace.** pip installs go to `.venv`
  (`uv pip install --python .venv/bin/python ...`, `export UV_CACHE_DIR=$PWD/tools/.uvcache`).
  Network needs proxy: `export HTTPS_PROXY=http://<your-proxy-host>:<your-proxy-port> HTTP_PROXY=$HTTPS_PROXY`.
  If the sandbox denies a write, stop and report — do not relocate output outside the workspace.
- **No git commands** (GC owns commits/tags). Do not touch `.git/`.
- Writable trees (default): `engine_vr/`, `vr.py`, `spike/`, `port/` (your `reports/phaseN.md`
  only, never others' reports), `tools/vk/` (tooling only). Everything else READ-ONLY, especially
  `engine/`, `audio/`, `docs/`, `film.py`, `still.py`, `tools/*.py`, `engine/shaders/`.
- **Do not re-tune visual math.** Port = driver-API translation. Any numerical/constant change
  requires a `FIXME(port)` comment + mention in your report.
- Reuse over rewrite: where original logic is GPU-independent (numpy/struct code), import from
  `engine/` instead of copying. Copy only where driver calls are interleaved.

## 2. Why static gates & first-run protocol exist

Zero-run discipline (§1) means no gate can execute shaders end-to-end. Gates are machine checks
that need no GPU: `glslc`/`spirv-val` on every shader, `py_compile`, guarded imports, `ruff`,
diff discipline. GC adds a line-by-line API-contract review against the installed bindings'
source in `.venv/lib/python3.12/site-packages/` (ground truth — consult it, not memory).
Remaining runtime risk is handled by the P4 first-run protocol (ladder + budgets + failure
signatures), executed by the user when they free GPU memory.

## 3. Workspace map

- `port/RUNBOOK.md` (this file) · `port/PHASE_N.md` briefs · `port/gates/gate_pN.py` (run with
  `.venv/bin/python`) · `port/reports/phaseN.md` (agent artifact, §5 format)
- `spike/p0_*` — P0 artifacts (kept; the only code ever GPU-run by agents)
- `tools/vk/` — workspace-local toolchain (git-excluded): `bin/` (glslc, spirv-val), `lib/`
  (libMoltenVK.dylib, libvulkan), `env.sh` (DYLD/VK_DRIVER_FILES for MoltenVK route)
- `.venv` — py3.12.15: numpy 2.5.4, scipy 1.18.1, pillow 13.0.0, wgpu 0.32.0, vulkan 1.3.275.1,
  ruff. `import vulkan` requires tools/vk env (fails without it by design).
- Originals: `engine/` (+`engine/shaders/` ~14.4k GLSL lines), `engine/shots_film.py` (9 shot
  locations → pano viewpoints in P3), `docs/img/` quality baseline.

## 4. Source architecture primer (read README.md "Architecture" too)

World is analytic (`heightfn.glsl`, 760 L) baked to `cache_final/*.npy` (8192² far heightmap, 4 m
cells, ±16 km) via GPU frag/compute passes (`bake_*.frag`, `erosion_*.comp` — the ONLY compute
shaders; MFD hydrology in `erosion.py` 1687 L). Runtime = **split-pass raymarcher**: `scene.frag`
compiled 9× with different `#define`s via `engine/glstrip.py` (own preprocessor + dead-function
stripper; motivation: one monolithic program blows driver memory), each pass writes G-buffer
(2 float RTs); shading passes read the G-buffer; `post.py` tone-maps tiled HDR accumulators.
`render.py` holds Camera/Look/pass-loop; `gl.py:23` `create_standalone_context(require=430)`;
`gpuslot.py` = single-GPU-slot + memory watchdog (env `ALPINE_*`). Marching epsilon derives from
constant `gPixAng` — see §6 before touching step logic.

## 5. Protocol per phase

Read RUNBOOK + PHASE_N → work → write `port/reports/phaseN.md` (§5 format) → GC runs
`gate_pN.py` + review → GC commits/tags on pass; on fail a FRESH agent redoes the phase with raw
gate output. Report format:

```
# Phase N report — <agent id/timestamp>
## Status: DONE | BLOCKED(<reason>)
## Files written: (paths + LOC)
## Deviations from brief: (list, or "none")
## Gate self-check: (each gate section you ran locally, with output tail)
## API contracts relied on: (binding + exact API used + where you verified it, e.g. site-packages file:line)
## Risks / FIXME(port) list: (each: file:line, one-line why)
```

## 6. VR projection spec (binding for P3/P4)

Equirect: lon ∈ [−π, π) across width W, lat ∈ [π/2, −π/2] top-down across height H, pixel center
`(x+0.5)`, direction `d = (cos(lat)·sin(lon), sin(lat), cos(lat)·cos(lon))` mapped to world axes
per P3's documented world-frame convention (Y-up as in engine). **Per-texel angular footprint**:
`dLat = π/H`, `dLon = 2π/W`, footprint(φ) = `max(dLat, dLon·cos φ)`; marching epsilon that used
the constant `gPixAng` MUST use this per-ray value in pano mode (else polar smear). Stereo:
parallel projection (no convergence), eye offsets ±0.032 m along camera-right, identical seed.

## 7. Decisions log (GC-owned; agents do not edit)

- D1 Coder agents one-per-phase, serial. D2 No agent↔agent message relay; repo files are the
  contract. D3 Deliverable = VR-360 panos, not the film (user pivot). D4 Zero project execution
  post-P0; static gates + first-run protocol. D5 Toolchain workspace-local in tools/vk/.
- D6 Backend: **PENDING P0** (MoltenVK vs wgpu-py; P0 report evidence → GC records verdict).
