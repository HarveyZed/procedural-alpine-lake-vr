"""passes_vr - the shared plumbing between the P3 drivers and gl_vr.

This is where the moderngl call shapes used by engine/bake.py,
engine/erosion.py and engine/features.py get their MoltenVK equivalents:

  moderngl (engine/gl.py)             engine_vr (this module / gl_vr)
  ---------------------------------   ----------------------------------
  gl.ctx() standalone context         open_host(): gpuslot_vr.acquire()
                                      BEFORE gl_vr.Host(...).open()
                                      (port/reports/phase2.md addendum)
  gl.tex2d(arr, filt, repeat)         upload(arr, filt, repeat) -> Tex
  c.texture((r,r),4,"f4") + fbo       make_target(res, RGBA32F,
                                      final_layout=TRANSFER_SRC)
  Pass(frag) + p.set + p.tex          build(program) + set_uniforms +
                                      write_descriptors(samplers=...)
  p.draw(fbo,(x,y,tile,tile))         begin/draw(rect)/end(wait=True)
  c.finish() in the tile loop         Host.end(wait=True) per tile
  gl.read_tex(t)                      read_image(target.images[0])
  tex.release()                       Tex.release() -> GpuImage.destroy()
  fbo.release()                       destroy_target() - gl_vr keeps targets
                                      alive for the whole session (review_p2
                                      N2); an 8192^2 RGBA32F target is 1 GiB,
                                      so P3 frees them explicitly.

Uniform keys are never filtered or renamed: a call-site key must equal the
manifest uniform_block member name, and gl_vr's Std140Packer raises on a key
it does not know - unlike moderngl's Pass.set, which silently skips, and that
silence is exactly the drift this port must not inherit.

Import-safe: nothing here opens a device. host() raises until use_host() or
open_host() has run, so an accidental import cannot touch the GPU.
"""

import time

import numpy as np

from engine_vr import gl_vr
from engine_vr.gl_vr import GlvrError

# Every bake/erosion pass writes an RGBA float32 output (moderngl
# c.texture((res, res), 4, "f4") at engine/bake.py:26-43 and
# engine/features.py:246-277) against a manifest program with one vec4
# fragColor output, so one RGBA32F color attachment is the faithful target.
TARGET_FORMAT = "VK_FORMAT_R32G32B32A32_SFLOAT"

# engine/gl.py tex2d dtype letters -> gl_vr FMT_BY_COMPS_DTYPE dtype keys.
_DTYPE_BY_LETTER = {"f1": "u1", "f2": "f2", "f4": "f4"}

_HOST = None


# --------------------------------------------------------------------- host
def use_host(host) -> None:
    """Install an already-open Host as the module default."""
    global _HOST
    _HOST = host


def host() -> gl_vr.Host:
    """The installed Host, or a clear error (never an implicit open())."""
    if _HOST is None:
        raise GlvrError(
            "no engine_vr host installed: call "
            "engine_vr.passes_vr.open_host() (or use_host()) first - P3 "
            "modules must stay import-safe, so nothing opens a device here")
    return _HOST


def current_host():
    """The installed Host or None (no raise) - for status/report code."""
    return _HOST


def open_host(cache_dir=None, res=None, tile=192, spp=1, log=print,
              **host_kwargs) -> gl_vr.Host:
    """Acquire the GPU slot FIRST, then create and open the Host.

    The ordering is the phase-2 addendum contract: gpuslot_vr.acquire()
    before Host.open(), so the memory watchdog owns the slot before any
    allocation and gl_vr._gpuslot() only ever borrows a held slot.
    """
    from engine_vr import gpuslot_vr

    slot = gpuslot_vr.acquire()
    try:
        h = gl_vr.Host(cache_dir=cache_dir, res=res, tile=tile, spp=spp,
                       log=log, **host_kwargs)
        h.open()
    except BaseException:
        gpuslot_vr.release()
        raise
    h._p3_slot = slot
    use_host(h)
    return h


def close_host() -> None:
    """Close the installed Host and release the slot (inverse of open)."""
    global _HOST
    h = _HOST
    _HOST = None
    if h is not None:
        try:
            h.save_pipeline_cache()
        except Exception:                    # cache save is best effort
            pass
        h.close()
    try:
        from engine_vr import gpuslot_vr

        gpuslot_vr.release()
    except Exception:
        pass


# ---------------------------------------------------------------- textures
class Tex:
    """A GpuImage plus the sampler state it is meant to be read with.

    The engine passes moderngl textures around; the VR equivalent has to keep
    the filter/wrap choice next to the image because gl_vr takes the sampler
    per binding at write_descriptors() time and freezes the image layout then.
    """

    __slots__ = ("img", "filt", "addr")

    def __init__(self, img, filt="linear", addr="clamp"):
        self.img = img
        self.filt = filt
        self.addr = addr

    def release(self) -> None:
        """Destroy the image (the moderngl tex.release() equivalent)."""
        if self.img is not None:
            self.img.destroy()
            self.img = None

    def as_sampler(self):
        return (self.img, self.filt, self.addr)


def sampler_value(t):
    """Normalise a texture argument to gl_vr's (image, filt, addr) tuple."""
    if isinstance(t, Tex):
        return t.as_sampler()
    if isinstance(t, gl_vr.GpuImage):
        return (t, "linear", "clamp")
    if isinstance(t, tuple) and len(t) == 3:
        return t
    raise GlvrError(f"not a texture or sampler value: {t!r}")


def upload(arr, filt="linear", repeat=False, dtype=None, extra_usage=0):
    """numpy (h, w[, c]) -> Tex resting in SHADER_READ_ONLY.

    Copied dtype rules from engine/gl.py tex2d(): component count from the
    trailing axis, "f1" means uint8, "f2" means float16, anything else is
    float32 with an explicit astype; filter and wrap follow the same two
    arguments the engine takes.
    """
    h = host()
    vkv = gl_vr.vk()
    a = np.asarray(arr)
    comps = 1 if a.ndim == 2 else int(a.shape[2])
    letter = dtype or ("f1" if a.dtype == np.uint8 else
                       ("f2" if a.dtype == np.float16 else "f4"))
    if letter not in _DTYPE_BY_LETTER:
        raise GlvrError(f"unsupported tex2d dtype letter {dtype!r}")
    key = _DTYPE_BY_LETTER[letter]
    if letter == "f1":
        a = np.ascontiguousarray(a, np.uint8)
    elif letter == "f2":
        a = np.ascontiguousarray(a, np.float16)
    else:
        a = np.ascontiguousarray(a, np.float32)
    try:
        fmt_name = gl_vr.FMT_BY_COMPS_DTYPE[(comps, key)][0]  # table -> (name, np dtype); FIXME(first-run) P3 used it as bare name
    except KeyError:
        raise GlvrError(
            f"no VK format for {comps} component(s) of {letter!r}") from None
    usage = (vkv.VK_IMAGE_USAGE_SAMPLED_BIT
             | vkv.VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra_usage)
    img = h.upload_2d(a, fmt_name, usage,
                      vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    return Tex(img, filt, "repeat" if repeat else "clamp")


def storage_image(res, comps=1, dtype="f4", data=None, depth=1, dims=2,
                  width=None, height=None):
    """An image resting in GENERAL, usable as storage image and as sampler.

    The erosion ping-pong maps are written by compute and read back as
    samplers by the next program, so they carry STORAGE | SAMPLED |
    TRANSFER_SRC | TRANSFER_DST. With data they upload through the same path
    the samplers use, which ends with the TRANSFER -> COMPUTE|FRAGMENT
    SHADER_READ barrier gl_vr's upload_2d records (its M4 fix).
    """
    h = host()
    vkv = gl_vr.vk()
    key = _DTYPE_BY_LETTER[dtype]
    try:
        fmt_name = gl_vr.FMT_BY_COMPS_DTYPE[(comps, key)][0]  # table -> (name, np dtype); FIXME(first-run) P3 used it as bare name
    except KeyError:
        raise GlvrError(
            f"no VK format for {comps} component(s) of {dtype!r}") from None
    usage = (vkv.VK_IMAGE_USAGE_STORAGE_BIT | vkv.VK_IMAGE_USAGE_SAMPLED_BIT
             | vkv.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
             | vkv.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
    w = int(width or res)
    hpx = int(height or res)
    gen = vkv.VK_IMAGE_LAYOUT_GENERAL
    if data is None:
        img = h.make_image(w, hpx, fmt_name, usage, gen, depth=depth,
                           dims=dims)
        img.layout = gen
        return img
    a = np.ascontiguousarray(data)
    if key == "f2":
        a = a.astype(np.float16)
    elif key == "u1":
        a = a.astype(np.uint8)
    else:
        a = a.astype(np.float32)
    return h.upload_2d(a, fmt_name, usage, gen, depth=depth)


# --------------------------------------------------------------- uniforms
def write_uniforms(prog, fields) -> None:
    """A UBO write - only legal with the queue idle (phase-2 addendum)."""
    host().set_uniforms(prog, fields)


# ------------------------------------------------------------ bake passes
def program_for(name) -> str:
    """Map an engine fragment argument to a manifest program name.

    engine/bake.py passes GLSL file names - and, for the two passes whose
    source is a Python string constant written into the cache dir, a Path -
    while the VR host compiles one SPIR-V per manifest program from
    engine_vr/shaders_gen. Mapping is by file stem:

      "bake_height.frag"          -> "bake_height"
      Path(...)/_bake_relief.frag -> "bake_relief"   (source is bake's
      Path(...)/_bake_macro.frag  -> "bake_macro"     RELIEF_FRAG /
                                                     MACRO_FRAG string,
                                                     mirrored by P1)

    bake_ao.frag is ambiguous: one GLSL source, two manifest programs
    (bake_ao_far / bake_ao_near), so bake_vr names those explicitly instead
    of routing them through here.
    """
    s = str(name).replace(chr(92), "/")
    stem = s.rsplit("/", 1)[-1]
    for ext in (".frag", ".comp", ".vert"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if stem.startswith("_"):
        stem = stem[1:]
    return stem


def make_target(res, fmt_name=TARGET_FORMAT,
                final_layout="VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL",
                n_attachments=1, width=None, height=None):
    return host().make_target(width or res, height or res, n_attachments,
                              fmt_name, final_layout)


def destroy_target(target) -> None:
    """Free a Target's attachments, framebuffers and render passes.

    gl_vr deliberately never destroys targets (per-session lifetime, accepted
    in review_p2), but P3 allocates up to 1 GiB per bake target, so they are
    released as soon as the pass has been read back.
    """
    if target is None:
        return
    vkv = gl_vr.vk()
    h = host()
    for rp, fb in list(getattr(target, "passes", {}).values()):
        vkv.vkDestroyFramebuffer(h.device, fb, None)
        vkv.vkDestroyRenderPass(h.device, rp, None)
    target.passes.clear()
    for img in list(target.images):
        img.destroy()
    target.images.clear()


def read_image(img, rect=None, depth=1) -> np.ndarray:
    """Image -> writable numpy (h, w[, c]) in GL row order.

    Host.fetch() returns a read-only view of the cached staging buffer. The
    engine writes into its readbacks (engine/bake.py:75 mutates the
    bake_height result in place) and that staging buffer is reused by the
    next transfer of the same size, so the copy is not optional.
    """
    h = host()
    cmd = h.begin()
    stg, n = h.to_staging(cmd, img, rect=rect, depth=depth)
    h.end()
    a = h.fetch(stg, n, img, rect=rect, depth=depth)
    return np.array(a, copy=True)


def bake_pass(program, res, uniforms, textures=(), tile=512, comps=1,
              fmt_name=TARGET_FORMAT, log=print):
    """Port of engine/bake.py:26-43 _bake_pass().

    Same shape: build the program, write every uniform, bind the named
    textures, draw the grid tile by tile flushing after each tile (the
    engine calls c.finish() inside its tile loop), read the result back, free
    the target. Returns (h, w) when comps == 1, else (h, w, comps) float32.
    """
    h = host()
    name = program_for(program)
    prog = h.build(name)
    if uniforms:
        write_uniforms(prog, uniforms)
    if prog.outputs and len(prog.outputs) != 1:
        raise GlvrError(f"{name}: expected 1 color output, manifest has "
                        f"{len(prog.outputs)}")
    samplers = {}
    for bind_name, tex in textures:
        samplers[bind_name] = sampler_value(tex)
    h.write_descriptors(prog, samplers=samplers)
    target = h.make_target(res, res, 1, fmt_name,
                           "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL")
    t0 = time.time()
    try:
        # N3 (review_p3): a failed tile draw or readback used to leak the
        # 4096^2 RGBA32F target (up to 1.07 GiB); the teardown now runs no
        # matter how the loop exits.
        for y in range(0, res, tile):
            for x in range(0, res, tile):
                cmd = h.begin()
                h.draw(prog, target, (x, y, tile, tile), cmd=cmd)
                h.end()              # submit + wait == the engine's finish()
        a = read_image(target.images[0])
    finally:
        destroy_target(target)
    if log:
        log(f"  {name} {res}^2 in {time.time() - t0:.1f}s")
    return a[..., 0] if comps == 1 else a


# ---------------------------------------------------------- compute passes
def compute_pass(program, fields, samplers=None, storage_images=None,
                 groups=None, cmd=None):
    """One compute dispatch with a fresh uniform + descriptor write.

    The order is the contract order and it is not negotiable: queue idle,
    set_uniforms (a host-mapped UBO write), write_descriptors (freezes the
    layouts the images rest in right now), dispatch.
    """
    h = host()
    prog = h.build(program)
    if fields:
        write_uniforms(prog, fields)
    h.write_descriptors(prog, samplers=samplers,
                        storage_images=storage_images)
    if isinstance(groups, tuple):
        gx, gy, gz = groups
    else:
        gx = gy = gz = groups
    return h.dispatch(prog, gx, gy, gz, cmd=cmd)


def compute_bands(program, fields, groups, band_tiles, samplers=None,
                  storage_images=None, offset_key="uOffset", flush=None):
    """Dispatch a compute pass in bands of band_tiles tile rows.

    Copied discipline from engine/erosion.py:242-254 (_run): the group grid
    is walked in bands so no dispatch covers more than band_tiles times
    local_size rows (128 * 16 = 2048, the TDR-safe rule), and uOffset - which
    the shaders add to gl_WorkGroupID before scaling by the tile size - is
    rewritten per band. Because uOffset changes per band and a UBO write is
    only legal with the queue idle, every band is its own
    write, descriptors, dispatch, wait cycle.
    """
    h = host()
    prog = h.build(program)
    per = band_tiles or groups
    for y0 in range(0, groups, per):
        rows = min(per, groups - y0)
        f = dict(fields)
        f[offset_key] = (0, y0)
        write_uniforms(prog, f)
        h.write_descriptors(prog, samplers=samplers,
                            storage_images=storage_images)
        h.dispatch(prog, groups, rows, 1)
        h.wait()
        if flush is not None:
            flush()
    return prog


def barrier_compute_to_read(images) -> None:
    """ctx.memory_barrier() for storage images resting in GENERAL.

    port/reports/phase2.md addendum: memory_barrier() ==
    barrier(COMPUTE -> COMPUTE, SHADER_WRITE -> SHADER_READ) over the storage
    images, which stay in GENERAL. Pass only the images the dispatch wrote:
    an image bound as a sampler rests in SHADER_READ_ONLY and must not be
    named with oldLayout=GENERAL.
    """
    h = host()
    vkv = gl_vr.vk()
    images = [i for i in images if i is not None]
    if not images:
        return
    gen = vkv.VK_IMAGE_LAYOUT_GENERAL
    cmd = h.begin()
    h.barrier(cmd, vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              vkv.VK_ACCESS_SHADER_WRITE_BIT, vkv.VK_ACCESS_SHADER_READ_BIT,
              [(i, gen, gen) for i in images])
    h.end()


def to_sampler_layout(images) -> None:
    """GENERAL / TRANSFER_SRC -> SHADER_READ_ONLY so an image can be sampled.

    write_descriptors() freezes whatever layout the image rests in (GENERAL
    stays GENERAL, anything else is bound as SHADER_READ_ONLY), so an image a
    compute pass just wrote - or a readback that left it in TRANSFER_SRC -
    must be transitioned before it is bound as a sampler.
    """
    h = host()
    vkv = gl_vr.vk()
    ro = vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    todo = [i for i in images if i is not None and i.layout != ro]
    if not todo:
        return
    cmd = h.begin()
    for i in todo:
        src = i.layout
        if src == vkv.VK_IMAGE_LAYOUT_GENERAL:
            stage, access = (vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             vkv.VK_ACCESS_SHADER_WRITE_BIT)
        elif src == vkv.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            stage, access = (vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
                             vkv.VK_ACCESS_TRANSFER_READ_BIT)
        else:
            stage, access = (vkv.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0)
        h.barrier(cmd, stage,
                  vkv.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                  | vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  access, vkv.VK_ACCESS_SHADER_READ_BIT, [(i, src, ro)])
        i.layout = ro
    h.end()


def to_general_layout(images) -> None:
    """SHADER_READ_ONLY / TRANSFER_SRC -> GENERAL before compute writes."""
    h = host()
    vkv = gl_vr.vk()
    gen = vkv.VK_IMAGE_LAYOUT_GENERAL
    todo = [i for i in images if i is not None and i.layout != gen]
    if not todo:
        return
    cmd = h.begin()
    for i in todo:
        src = i.layout
        if src == vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            stage, access = (vkv.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             vkv.VK_ACCESS_SHADER_READ_BIT)
        elif src == vkv.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            stage, access = (vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
                             vkv.VK_ACCESS_TRANSFER_READ_BIT)
        elif src == vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            stage, access = (vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
                             vkv.VK_ACCESS_TRANSFER_WRITE_BIT)
        else:
            stage, access = (vkv.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0)
        h.barrier(cmd, stage, vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  access, vkv.VK_ACCESS_SHADER_WRITE_BIT, [(i, src, gen)])
        i.layout = gen
    h.end()
