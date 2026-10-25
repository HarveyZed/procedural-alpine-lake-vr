"""gl_vr - manifest-driven MoltenVK host layer (Phase 2).

Port target: the ModernGL host (engine/gl.py + engine/render.py pass
patterns) on the MoltenVK Vulkan backend, driven ENTIRELY by
engine_vr/manifest.json - every descriptor set/binding, every uniform-block
byte offset, and every color-attachment table comes from the manifest, never
from hardcoded numbers (the gate forbids literal binding digits in this file
for exactly that reason).

USAGE (guarded - nothing touches the GPU at import time):
    source tools/vk/env.sh          # DYLD path to libvulkan.dylib (RUNBOOK D6)
    from engine_vr import gl_vr
    host = gl_vr.Host(cache_dir=None, tile=192, res=None, spp=1)
    info = host.open()              # instance + device + queue + pipeline cache
    prog = host.build("scene_trace")  # SPIR-V (glslc, cached) + pipelines
    host.close()                    # saves the pipeline cache file

P5-ladder parameters are plain Host kwargs (PHASE_2 T7; the vr.py CLI is P4):
    res  (w, h) frame size          tile   tiling edge length
    spp  sample budget (informational for P3 drivers)
    cache_dir  pipeline-cache directory (default engine_vr/out)

GPU DISCIPLINE: the vulkan module is imported lazily inside Host.open(); no
vkCreateDevice anywhere else. Nothing in this module touches the device
without an explicit open()/build() call, and P2 never executed it against
hardware (PHASE_2 zero-GPU rule; first run happens in P3/P5).

Import style: engine_vr modules may be imported as a package or with
engine_vr/ on sys.path (P4 decides); the gpuslot_vr import below covers both.

------------------------------------------------------------------------------
GL -> VULKAN SYNCHRONIZATION MAPPING (contractual for the GC review)

The engine is a strictly ordered single-queue program: every ModernGL call
joins the implicit GL pipeline. One MoltenVK queue preserves ORDER, so what
must be explicit in Vulkan is (a) memory visibility between stages touching
the same image/buffer, (b) image-layout transitions, (c) host<->device
handoffs. Engine call sites below are engine/ files (read-only ground truth).

| engine GL sync point          | gl_vr lowering                             |
|---------------------------------|----------------------------------------------|
| c.finish() per tile - bake.py   | one-shot submit + vkQueueWaitIdle: the     |
| 39, render.py 176/201,          | Host.end(wait=True) pattern. Ordering      |
| erosion.py 251-254 band flush   | INSIDE a recorded session needs no barrier |
|                                 | (single queue, in-order).                  |
| ctx.memory_barrier(),           | host.barrier(cmd, COMPUTE->COMPUTE,        |
| erosion.py:252 after a band of  | SHADER_WRITE->SHADER_READ) over the        |
| compute dispatches (GL_SHADER_  | storage images touched; their layout stays |
| IMAGE_ACCESS device barrier)    | GENERAL.                                     |
| gl.read_tex() after draws -     | image barrier: src stage = whoever wrote   |
| render.py 177, 425, 505, bake.  | it (COLOR_ATTACHMENT_OUTPUT + COLOR_       |
| py 40, erosion _read            | ATTACHMENT_WRITE for draw targets, COMPUTE |
|                                 | + SHADER_WRITE after dispatches, 0 access  |
|                                 | for images resting in SHADER_READ_ONLY);   |
|                                 | dst TRANSFER READ; layout -> TRANSFER_SRC; |
|                                 | vkCmdCopyImageToBuffer into host-coherent  |
|                                 | staging; submit+wait; read mapped memory.  |
| texture upload (erosion         | host bytes -> host-visible COHERENT        |
| _upload, world map uploads)     | staging -> vkCmdCopyBufferToImage (layout  |
|                                 | UNDEFINED -> TRANSFER_DST), barrier        |
|                                 | TRANSFER_WRITE -> FRAGMENT+COMPUTE_SHADER  |
|                                 | READ (explicit read dst access - NOT a    |
|                                 | BOTTOM_OF_PIPE layout-only transition),  |
|                                 | layout -> SHADER_READ_ONLY (GENERAL for  |
|                                 | images that later serve as storage).     |
| p.set(...) uniform writes,      | memcpy into the persistent mapped UBO.     |
| render.py 435-447               | Safe because every engine uniform write    |
|                                 | happens with the queue idle (next use is   |
|                                 | a later submit); queue-idle discipline is  |
|                                 | part of the P3 contract for this host.     |
| render.py:451-452 acc fbo.use() | Target render-pass variant loadOp=CLEAR    |
| + clear()                       | (acc cleared once per frame, P3 cadence).  |
| render.py 459-481 cone.draw     | cone target finalLayout=SHADER_READ_ONLY.  |
| (cfbo), then trace samples      | The render-pass end implicit EXTERNAL      |
| uCone (same queue)              | dependency covers COLOR-attachment         |
|                                 | accesses ONLY (spec) - a later pass SAMPLING|
|                                 | the image is a FRAGMENT read it does NOT    |
|                                 | cover, so draw() records an explicit        |
|                                 | COLOR_ATTACHMENT_OUTPUT(WRITE) ->           |
|                                 | FRAGMENT|COMPUTE_SHADER(SHADER_READ) barrier|
|                                 | whenever a pass ends at SHADER_READ_ONLY.   |
| trace.draw(gfbo) then gfbo.read | gbuf target built finalLayout=TRANSFER_SRC |
| (rect), render.py 484-486 - the | so the readback needs no extra barrier     |
| tile's only sync point          | (layout already TRANSFER_SRC); before      |
|                                 | kind passes sample uGBuf0/1 again,         |
|                                 | host.barrier re-transitions TRANSFER_SRC - |
|                                 | > SHADER_READ_ONLY (FRAGMENT access).      |
|                                 | Re-entry of a draw() LOAD pass into a      |
|                                 | target that rests OUT of COLOR_ATTACHMENT  |
|                                 | records a transition barrier whose src     |
|                                 | stage/access NAMES THE LAST TOUCHER:       |
|                                 | TRANSFER_SRC -> TRANSFER read (WAR guard   |
|                                 | vs the readback copy), SHADER_READ_ONLY -> |
|                                 | FRAGMENT read, GENERAL -> COMPUTE write;   |
|                                 | dst COLOR_ATTACHMENT_OUTPUT(WRITE).        |
| kind passes blend ONE,ONE into  | same target, "blend" graphics-pipeline     |
| acc (render.py:452 blend_func)  | variant; the LOAD render-pass keeps the    |
|                                 | layout in COLOR_ATTACHMENT; after a CLEAR  |
|                                 | pass the target rests at finalLayout, so   |
|                                 | the next LOAD draw takes the src=last-     |
|                                 | toucher row above (not a layout-only       |
|                                 | no-access barrier - see M5 fix note).      |
| bind_to_image (erosion images)  | descriptors carry VK_IMAGE_LAYOUT_GENERAL  |
|                                 | their whole life; dispatch->dispatch order |
|                                 | = in-order queue + the band barrier above. |
|                                 | UNVERIFIED through MoltenVK->Metal:        |
|                                 | coherent imageLoad/imageStore RMW across   |
|                                 | workgroups (phase1 risk; probe 1).         |

Y-ORIENTATION CLAIM (stated explicitly for review, not silently assumed):
every GL framebuffer in this engine is a headless texture, never a window.
ModernGL texture row r corresponds to fragment row y=r in GL (bottom-origin)
AND to Vulkan image row r (top-origin): both index systems flip together, so
gl_FragCoord-driven shader content lands at the same array index in both
backends, readback maps row r to row r with no vertical flip anywhere, and a
tile rect (x, y, w, h) transfers verbatim from fbo.viewport to
VkViewport/VkRect2D. Probe 3 (fragcoord-driven gradient vs GL reference,
port/reports/phase2.md) confirms this on first run.

MOLTENVK SEMANTIC CLAIMS used by this module (also in the report):
  * VK_KHR_portability_enumeration at instance + portability_subset device
    extension follow spike/p0_a/run.py (hardware-proven).
  * Host-coherent unified memory: HOST_VISIBLE|HOST_COHERENT buffers are CPU-
    mapped once and written between submits (no flush - coherent). The
    bindings' vkMapMemory ALREADY returns an ffi.buffer (_vulkan.py:22553-
    22560: ffi.buffer(ppData[0], size)) - store it directly; re-wrapping it
    with ffi.buffer() raises TypeError (empirical).
  * Pipeline cache data is read through the cffi lib handle because
    vkGetPipelineCacheData is NOT wrapped by the bindings (see
    save_pipeline_cache docstring). cffi cannot size "void[]"; the save blob
    is a char[] and the load pointer is ffi.cast("void*",
    ffi.from_buffer(bytes)) passed with an explicit initialDataSize (the
    ctor's len() auto-size path raises on a void* cdata; _cast_ptr passes a
    CData void* through unchanged, _vulkan.py:69-99, empirical).
  * Color-attachment default RGBA16F per PHASE_2 T4; make_target() lets P3
    override to RGBA32F if a parity diff needs it (engine's GL targets were
    RGBA32F).
"""

import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent.parent
MANIFEST_PATH = WS / "engine_vr" / "manifest.json"
SPIRV_DIR = WS / "engine_vr" / "spirv"
GLSLC = WS / "tools" / "vk" / "bin" / "glslc"
PIPELINE_CACHE_FILE = "pipeline_cache.bin"

MAX_SETS = 4          # manifest descriptor_conventions: sets 0..3 namespace

# std140 rules for the member types the manifest emits (phase1 §glslc table:
# vec3 -> 16-byte alignment, mat3 = 3 columns stride 16, no other aggregates).
STD140 = {
    "float": (4, 4, "<f"),
    "int": (4, 4, "<i"),
    "uint": (4, 4, "<I"),
    "vec2": (8, 8, "<2f"),
    "ivec2": (8, 8, "<2i"),
    "vec3": (16, 12, "<3f"),   # std140: 16-aligned, 12 B payload, 4 B tail pad
    "ivec3": (16, 12, "<3i"),
    "vec4": (16, 16, "<4f"),
    "ivec4": (16, 16, "<4i"),
    "mat3": (16, 48, None),    # 3 columns, 16-byte column stride
}

# (comps, dtype) -> (VK format name, numpy dtype) for uploads/readbacks.
FMT_BY_COMPS_DTYPE = {
    (1, "f4"): ("VK_FORMAT_R32_SFLOAT", np.float32),
    (2, "f4"): ("VK_FORMAT_R32G32_SFLOAT", np.float32),
    (4, "f4"): ("VK_FORMAT_R32G32B32A32_SFLOAT", np.float32),
    (1, "f2"): ("VK_FORMAT_R16_SFLOAT", np.float16),
    (2, "f2"): ("VK_FORMAT_R16G16_SFLOAT", np.float16),
    (4, "f2"): ("VK_FORMAT_R16G16B16A16_SFLOAT", np.float16),
    (1, "i4"): ("VK_FORMAT_R32_SINT", np.int32),
    (4, "i4"): ("VK_FORMAT_R32G32B32A32_SINT", np.int32),
    (4, "u1"): ("VK_FORMAT_R8G8B8A8_UNORM", np.uint8),
}
# Color-attachment default (PHASE_2 T4). Engine GL used RGBA32F; the manifest
# says only "vec4", P2 follows the brief's RGBA16F and exposes an override.
DEFAULT_TARGET_FORMAT = "VK_FORMAT_R16G16B16A16_SFLOAT"

_FORMAT_BYTES = {   # format name -> (texel bytes, channel bytes, numpy dtype)
    "VK_FORMAT_R32_SFLOAT": (4, 4, np.float32),
    "VK_FORMAT_R32G32_SFLOAT": (8, 4, np.float32),
    "VK_FORMAT_R32G32B32_SFLOAT": (12, 4, np.float32),
    "VK_FORMAT_R32G32B32A32_SFLOAT": (16, 4, np.float32),
    "VK_FORMAT_R16_SFLOAT": (2, 2, np.float16),
    "VK_FORMAT_R16G16_SFLOAT": (4, 2, np.float16),
    "VK_FORMAT_R16G16B16A16_SFLOAT": (8, 2, np.float16),
    "VK_FORMAT_R32G32B32A32_SINT": (16, 4, np.int32),
    "VK_FORMAT_R8G8B8A8_UNORM": (4, 1, np.uint8),
}

_VB_ATTR = 0        # vertex-buffer index for the fullscreen triangle attr
_KEEP = []          # cdata/bytes kept alive until the session's submit waits


def _gpuslot():
    """gpuslot_vr under either import style (package or path-on-sys.path)."""
    try:
        from engine_vr import gpuslot_vr
    except ImportError:
        import gpuslot_vr
    return gpuslot_vr


_VK = None


def vk():
    """The vulkan module; imported on first GPU-side call only."""
    global _VK
    if _VK is None:
        import vulkan
        _VK = vulkan
    return _VK


class GlvrError(RuntimeError):
    pass


def load_manifest(path=None):
    """The manifest contract (phase1) as a dict."""
    p = Path(path) if path else MANIFEST_PATH
    return json.loads(p.read_text())


# --------------------------------------------------------------------------
# std140 packer (manifest member tables only)
# --------------------------------------------------------------------------

_STRIDE = {"float": 4, "int": 4, "uint": 4, "vec2": 8,
              "ivec2": 8, "vec3": 16, "vec4": 16, "mat3": 48}


class Std140Packer:
    """Packs {member: value} dicts to the exact _p1_pass byte layout.

    The member table IS the contract; offsets are additionally cross-checked
    against std140 rules at construction (both directions) so a manifest/
    packer drift raises instead of corrupting frames.

    Values per member type:
      float/int/uint     scalar
      vec2..vec4         sequence (or scalar broadcast is NOT supported -
                         the engine always passes full tuples)
      mat3               (3,3) array or flat 9; GLSL COLUMN c receives NUMPY
                         ROW c (matches moderngl's mat3 uniform upload)
      long sequences     write an array across the member's slot span at the
                         type stride (uTrail: 32 x vec4 over [16, 528))
    """

    def __init__(self, uniform_block):
        self.name = uniform_block["name"]
        self.size = int(uniform_block["size_bytes"])
        self.members = {}     # name -> (type, offset, span_bytes)
        ordered = sorted(uniform_block["members"], key=lambda m: int(m["offset"]))
        cursor = 0
        for i, mb in enumerate(ordered):
            t = mb["type"]
            if t not in STD140:
                raise GlvrError(f"std140: unsupported member type {t!r}"
                                f" ({self.name}.{mb['name']})")
            align, size, _ = STD140[t]
            cursor = (cursor + align - 1) // align * align
            if cursor != int(mb["offset"]):
                raise GlvrError(
                    f"std140 offset mismatch {self.name}.{mb['name']}: "
                    f"manifest {mb['offset']} vs computed {cursor}")
            nxt = (int(ordered[i + 1]["offset"]) if i + 1 < len(ordered)
                   else self.size)
            span = nxt - cursor
            # A member may own an ARRAY span (uTrail: 32 x vec4 over
            # [16, 528) - GLSL 'vec4 uTrail[32]', manifest type 'vec4'),
            # or just its alignment-rounded single slot (a lone vec3
            # spans 16 with a 12-byte element). std140 stride per slot:
            # vec3/vec4 -> 16, mat3 -> 48, rest -> element size.
            stride = _STRIDE[t]
            # single element (a lone vec3 spans only 12 - std140 aligns
            # vec3 to 16 but a following float may sit at +12) or a
            # whole number of array slots
            if span != size and (span < stride or span % stride):
                raise GlvrError(
                    f"std140: {self.name}.{mb['name']}: span {span} is "
                    f"neither a single {t} (size {size}) nor a whole "
                    f"number of std140 slots (stride {stride})")
            self.members[mb["name"]] = (t, cursor, span)
            cursor += span
        # manifest size_bytes is the upload length (last end rounded up
        # by glslc's own rules, e.g. 20 for a ivec2-terminated block).
        # The host must never write past it, and must cover every member.
        if cursor > self.size:
            raise GlvrError(f"std140 size mismatch {self.name}: manifest "
                            f"{self.size} < computed member end {cursor}")

    def pack(self, values):
        buf = bytearray(self.size)
        for name, val in values.items():
            t, off, span = self.members[name]
            self._write(name, t, val, buf, off, span)
        return bytes(buf)

    def _write(self, name, t, val, buf, off, span):
        align, size, fmt = STD140[t]
        if t == "mat3":
            a = np.asarray(val, dtype=np.float32).reshape(-1)
            if a.size != 9:
                raise GlvrError(f"{self.name}.{name}: mat3 needs 9 floats")
            for c in range(3):     # GLSL column c = numpy row c (moderngl)
                col = struct.pack("<3f", *(float(x) for x in a[c * 3:c * 3 + 3]))
                buf[off + 16 * c:off + 16 * c + 12] = col
            return
        if fmt is None:
            raise GlvrError(f"{self.name}.{name}: type {t} unimplemented")
        n = int(fmt[1]) if len(fmt) > 2 else 1
        if n == 1 and np.isscalar(val):
            data = struct.pack(fmt, val)
            if len(data) > span:
                raise GlvrError(f"{self.name}.{name}: scalar overruns slot")
            buf[off:off + len(data)] = data
            return
        vals = list(np.asarray(val).ravel())
        if len(vals) % n:
            raise GlvrError(f"{self.name}.{name}: {t} array not a multiple"
                            f" of {n} components")
        stride = align
        count = len(vals) // n
        chunk_len = struct.calcsize(fmt)
        tail = stride * (count - 1) + chunk_len
        if tail > span:   # a lone vec3 fits its 12 B in a 12 B slot
            raise GlvrError(f"{self.name}.{name}: array of {count} x {t}"
                            f" needs {tail} B, slot span {span} B")
        cast = float if fmt[-1] == "f" else int
        for i in range(count):
            chunk = struct.pack(fmt, *(cast(x) for x in vals[i * n:(i + 1) * n]))
            buf[off + stride * i:off + stride * i + len(chunk)] = chunk


# --------------------------------------------------------------------------
# host objects
# --------------------------------------------------------------------------

class GpuBuffer:
    """Buffer + memory (+ persistent map when host-visible)."""

    def __init__(self, host, size, usage, host_visible):
        vkv = vk()
        self.host = host
        self.size = int(max(size, 4))
        dev = host.device
        self.buf = vkv.vkCreateBuffer(dev, vkv.VkBufferCreateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, size=self.size,
            usage=usage, sharingMode=vkv.VK_SHARING_MODE_EXCLUSIVE), None)
        req = vkv.vkGetBufferMemoryRequirements(dev, self.buf)
        need = ((vkv.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                 | vkv.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                if host_visible else vkv.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.mem = vkv.vkAllocateMemory(dev, vkv.VkMemoryAllocateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=req.size,
            memoryTypeIndex=host.find_memtype(need)), None)
        vkv.vkBindBufferMemory(dev, self.buf, self.mem, 0)
        self.mapped = None
        if host_visible:
            # the bindings return an ffi.buffer already (_vulkan.py:22553-
            # 22560) - store it as-is; slice-assign works on it directly
            # (re-wrapping with ffi.buffer() raises TypeError, empirical).
            self.mapped = vkv.vkMapMemory(dev, self.mem, 0, self.size, 0)

    def write(self, data, offset=0):
        if self.mapped is None:
            raise GlvrError("buffer is not host-visible")
        self.mapped[offset:offset + len(data)] = data

    def read(self, n=None, offset=0):
        if self.mapped is None:
            raise GlvrError("buffer is not host-visible")
        return bytes(self.mapped[offset:offset + (n or self.size)])

    def destroy(self):
        vkv = vk()
        if self.mapped is not None:
            # unmap before free (spec: mapped memory must not be freed with
            # an outstanding mapping; loaders validate this)
            vkv.vkUnmapMemory(self.host.device, self.mem)
            self.mapped = None
        vkv.vkDestroyBuffer(self.host.device, self.buf, None)
        vkv.vkFreeMemory(self.host.device, self.mem, None)


class GpuImage:
    """Image + view + layout bookkeeping (layout it rests in at idle)."""

    def __init__(self, host, width, height, fmt_name, usage,
                 initial_layout, depth=1, dims=2):
        vkv = vk()
        self.host = host
        self.w, self.h, self.depth = int(width), int(height), int(depth)
        self.fmt = getattr(vkv, fmt_name)
        self.fmt_name = fmt_name
        self.texel, self.channel, self.np_dtype = _FORMAT_BYTES[fmt_name]
        dev = host.device
        self.image = vkv.vkCreateImage(dev, vkv.VkImageCreateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=(vkv.VK_IMAGE_TYPE_3D if dims == 3
                       else vkv.VK_IMAGE_TYPE_2D),
            format=self.fmt,
            extent=vkv.VkExtent3D(width=self.w, height=self.h,
                                  depth=self.depth),
            mipLevels=1, arrayLayers=1, samples=vkv.VK_SAMPLE_COUNT_1_BIT,
            tiling=vkv.VK_IMAGE_TILING_OPTIMAL, usage=usage,
            initialLayout=initial_layout,
            sharingMode=vkv.VK_SHARING_MODE_EXCLUSIVE), None)
        req = vkv.vkGetImageMemoryRequirements(dev, self.image)
        self.mem = vkv.vkAllocateMemory(dev, vkv.VkMemoryAllocateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=req.size,
            memoryTypeIndex=host.find_memtype(
                vkv.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), None)
        vkv.vkBindImageMemory(dev, self.image, self.mem, 0)
        self.view = vkv.vkCreateImageView(dev, vkv.VkImageViewCreateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image=self.image,
            viewType=(vkv.VK_IMAGE_VIEW_TYPE_3D if dims == 3
                      else vkv.VK_IMAGE_VIEW_TYPE_2D),
            format=self.fmt,
            subresourceRange=vkv.VkImageSubresourceRange(
                aspectMask=vkv.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1,
                layerCount=1)), None)
        self.layout = initial_layout

    def destroy(self):
        vkv = vk()
        d = self.host.device
        vkv.vkDestroyImageView(d, self.view, None)
        vkv.vkDestroyImage(d, self.image, None)
        vkv.vkFreeMemory(d, self.mem, None)


class Target:
    """n RGBA color attachments + render-pass/framebuffer per load variant.

    passes[cleared=True]  loadOp=CLEAR,  initialLayout=UNDEFINED (discard)
    passes[cleared=False] loadOp=LOAD,   initialLayout=COLOR_ATTACHMENT_OPTIMAL
    Both store, both end at final_layout (SHADER_READ_ONLY by default;
    TRANSFER_SRC for readback-hot targets like the G-buffer).
    """

    def __init__(self, host, width, height, n_attachments, fmt_name,
                 final_layout):
        vkv = vk()
        self.host = host
        self.w, self.h = int(width), int(height)
        self.n = int(n_attachments)
        self.fmt_name = fmt_name
        self.final_layout = final_layout
        usage = (vkv.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                 | vkv.VK_IMAGE_USAGE_SAMPLED_BIT
                 | vkv.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                 | vkv.VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        self.images = [host.make_image(width, height, fmt_name, usage,
                                       vkv.VK_IMAGE_LAYOUT_UNDEFINED)
                       for _ in range(self.n)]
        self.passes = {}
        for cleared in (True, False):
            self.passes[cleared] = host._make_renderpass(self, cleared)


class Program:
    """One manifest program: SPIR-V, shader module, set layouts, pipeline(s),
    allocated descriptor sets, persistent UBO. Built by Host.build()."""

    def __init__(self, host, spec):
        self.host = host
        self.spec = spec
        self.name = spec["name"]
        self.is_compute = spec["stage"] == "comp"
        self.outputs = spec.get("outputs", [])
        self.ublocker = (Std140Packer(spec["uniform_block"])
                         if spec.get("uniform_block") else None)
        self.bindings = spec["bindings"]
        self.by_name = {b["name"]: b for b in self.bindings}
        self.sets = {}
        for b in self.bindings:
            self.sets.setdefault(int(b["set"]), []).append(b)
        self.layouts = {}
        self.sets_alloc = {}
        self.pipeline_layout = None
        self.module = None
        self.compute_pipeline = None
        self.gfx_pipelines = {}
        self.ubo = None
        self._keep = []

    def ubo_bytes(self, fields):
        if self.ublocker is None:
            raise GlvrError(f"{self.name}: program has no uniform block")
        return self.ublocker.pack(fields)


# --------------------------------------------------------------------------
# the host
# --------------------------------------------------------------------------

class Host:
    """One MoltenVK context: instance, device, queue, pools, caches.

    res / tile / spp are carried for P3/P4 drivers (T7) - the host does not
    interpret them; cache_dir holds pipeline_cache.bin (T1 persistence).
    """

    def __init__(self, cache_dir=None, res=None, tile=192, spp=1, log=print):
        self.cache_dir = Path(cache_dir) if cache_dir else (WS / "engine_vr"
                                                            / "out")
        self.res = res
        self.tile = int(tile)
        self.spp = int(spp)
        self.log = log
        self.manifest = load_manifest()
        self.programs_by_name = {p["name"]: p
                                 for p in self.manifest["programs"]}
        self.inst = self.device = self.queue = self.pd = None
        self.qfi = -1
        self.pcache = None
        self.cmd_pool = None
        self.descriptor_pool = None
        self._active_cmd = None
        self._programs = {}
        self._vert_module = None
        self._sampler_cache = {}
        self._staging = {}
        self._vbo = None       # fullscreen triangle, created on first draw()
        self.opened = False

    # ------------------------------------------------------------------ open
    def open(self, log_budget=True):
        """Instance + device + queue + pipeline cache. Guarded: ONLY place
        that creates a device (PHASE_2 T1; spike/p0_a/run.py pattern)."""
        import vulkan as vkv
        globals()["_VK"] = vkv
        gs = _gpuslot()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        exts = [vkv.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME.encode()]
        self.inst = vkv.vkCreateInstance(vkv.VkInstanceCreateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=vkv.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
            pApplicationInfo=vkv.VkApplicationInfo(
                apiVersion=vkv.VK_MAKE_VERSION(1, 2, 0),
                pApplicationName=b"alpine-vr",
                applicationVersion=1, pEngineName=b"gl_vr", engineVersion=1),
            ppEnabledExtensionNames=exts), None)
        pds = vkv.vkEnumeratePhysicalDevices(self.inst)
        if not pds:
            raise GlvrError("no Vulkan physical devices (MoltenVK ICD?)"
                            " source tools/vk/env.sh first")
        order = {vkv.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 0,
                 vkv.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 1,
                 vkv.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: 2,
                 vkv.VK_PHYSICAL_DEVICE_TYPE_CPU: 3}
        props = {i: vkv.vkGetPhysicalDeviceProperties(pd)
                 for i, pd in enumerate(pds)}
        pick = min(range(len(pds)), key=lambda i: (
            order.get(props[i].deviceType, 9), i))
        self.pd, p = pds[pick], props[pick]
        # deviceName is a char[] field behind StrWrap: the bindings auto-decode
        # it to str (_cstr/_cstr3 + StrWrap.__getattr__, _vulkan.py:20-66;
        # empirical). Calling .decode() on it raises AttributeError (str has no
        # .decode), so it must NOT be re-decoded here.
        self.log(f"[gl_vr] device: {p.deviceName} "
                 f"type={p.deviceType} api=0x{p.apiVersion:x}")
        # recommendedMaxMemorySize is absent from this binding's
        # VkPhysicalDeviceLimits (vulkan 1.3.275.1 headers predate the field
        # moving out of VkPhysicalDeviceMemoryBudgetPropertiesEXT): getattr to
        # 0 falls the watchdog back to hw.memsize * factor (gpuslot_vr.py:
        # budget_limit_gb), and the real heap budget still logs below via
        # VK_EXT_memory_budget after device creation.
        try:
            rec = int(p.limits.recommendedMaxMemorySize)
        except AttributeError:      # FIXME(first-run): binding lacks the field
            rec = 0
        gs.configure_budget(rec)
        self.log(f"[gl_vr] recommendedMaxMemorySize="
                 f"{rec / 2 ** 30:.1f} GiB -> watchdog cap "
                 f"{gs.budget_limit_gb():.1f} GiB")
        qfs = vkv.vkGetPhysicalDeviceQueueFamilyProperties(self.pd)
        self.qfi = next(i for i, q in enumerate(qfs)
                        if q.queueFlags & vkv.VK_QUEUE_COMPUTE_BIT
                        and q.queueFlags & vkv.VK_QUEUE_GRAPHICS_BIT)
        # e.extensionName is auto-decoded str (StrWrap, see deviceName note
        # above); enable every optional extension this module actually reads.
        # VK_EXT_memory_budget MUST be enabled for the heap-budget pNext query
        # to return real numbers - an unrequested pNext is silently ignored by
        # the loader and the log would print 0.0 GiB.
        dev_exts = [e.extensionName for e in
                    vkv.vkEnumerateDeviceExtensionProperties(self.pd, None)]
        enable = []
        for ext in (vkv.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
                    vkv.VK_EXT_MEMORY_BUDGET_EXTENSION_NAME):
            if ext in dev_exts:
                enable.append(ext.encode())
        self.device = vkv.vkCreateDevice(self.pd, vkv.VkDeviceCreateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=vkv.VkDeviceQueueCreateInfo(
                queueFamilyIndex=self.qfi, queueCount=1,
                pQueuePriorities=[1.0]),
            ppEnabledExtensionNames=enable), None)
        self.queue = vkv.vkGetDeviceQueue(self.device, self.qfi, 0)
        self.memprops = vkv.vkGetPhysicalDeviceMemoryProperties(self.pd)
        self.cmd_pool = vkv.vkCreateCommandPool(
            self.device, vkv.VkCommandPoolCreateInfo(
                sType=vkv.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                queueFamilyIndex=self.qfi), None)
        # budget numbers are only filled once VK_EXT_memory_budget is enabled
        # on the logical device, so this runs AFTER vkCreateDevice
        if log_budget:
            self._log_memory_budget(vkv)
        self._open_pipeline_cache(vkv)
        self.opened = True
        return {"device": p.deviceName,
                "recommendedMaxMemorySize": rec, "queue_family": self.qfi}

    def _log_memory_budget(self, vkv):
        """VK_EXT_memory_budget heap numbers at runtime (T1, informational).
        Called after vkCreateDevice with the extension enabled (open()); the
        pNext is ignored unless VK_EXT_memory_budget is a device enablement."""
        try:
            budget = vkv.VkPhysicalDeviceMemoryBudgetPropertiesEXT()
            mp2 = vkv.VkPhysicalDeviceMemoryProperties2(pNext=budget)
            vkv.vkGetPhysicalDeviceMemoryProperties2(self.pd, mp2)
            for i in range(mp2.memoryProperties.memoryHeapCount):
                self.log(f"[gl_vr] heap {i}: budget="
                         f"{budget.heapBudget[i] / 2 ** 30:.1f} GiB usage="
                         f"{budget.heapUsage[i] / 2 ** 30:.1f} GiB")
        except Exception as exc:
            self.log(f"[gl_vr] VK_EXT_memory_budget unavailable: {exc}")

    # -------------------------------------------------------- pipeline cache
    def _open_pipeline_cache(self, vkv):
        path = self.cache_dir / PIPELINE_CACHE_FILE
        initial = path.read_bytes() if path.exists() else None
        try:
            from vulkan import _vulkan as _vv
            # cffi cannot size a void[] and the ctor len() path raises on a
            # void* cdata; pass a real void* cast instead. from_buffer keeps
            # the bytes alive (empirical), and _new refs the pointer for the
            # struct lifetime (_vulkan.py:69-99, 6257-6293).
            ptr = None
            if initial:
                ptr = _vv.ffi.cast("void*", _vv.ffi.from_buffer(initial))
            self.pcache = vkv.vkCreatePipelineCache(self.device,
                vkv.VkPipelineCacheCreateInfo(
                    initialDataSize=len(initial) if initial else 0,
                    pInitialData=ptr), None)
            if initial:
                self.log(f"[gl_vr] pipeline cache: loaded {len(initial)} B")
        except Exception as exc:
            self.log(f"[gl_vr] pipeline cache create failed: {exc}")
            self.pcache = None

    def save_pipeline_cache(self):
        """Persist pipeline_cache.bin. vkGetPipelineCacheData is NOT wrapped
        by the python bindings (no _callApi def in vulkan/_vulkan.py); call
        it raw through the module's cffi lib handle (two-call size-then-fill;
        prototype present in vulkan/vulkan.cdef.h). Cache failures never
        abort a render run."""
        if self.pcache is None:
            return False
        from vulkan import _vulkan as _vv
        try:
            n = _vv.ffi.new("size_t[1]")
            rc = _vv.lib.vkGetPipelineCacheData(self.device, self.pcache,
                                                n, _vv.ffi.NULL)
            if rc != 0:
                self.log(f"[gl_vr] pipeline cache size query rc={rc}")
                return False
            size = int(n[0])
            if size <= 0:
                return False
            # void[] is unsizable for cffi (ValueError, empirical); char[]
            # is byte-addressable and casts to the void* parameter implicitly
            buf = _vv.ffi.new("char[]", size)
            rc = _vv.lib.vkGetPipelineCacheData(self.device, self.pcache,
                                                n, buf)
            if rc != 0:
                return False
            data = bytes(_vv.ffi.buffer(buf, int(n[0])))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / PIPELINE_CACHE_FILE).write_bytes(data)
            self.log(f"[gl_vr] pipeline cache: saved {len(data)} B")
            return True
        except Exception as exc:
            self.log(f"[gl_vr] pipeline cache save failed: {exc}")
            return False

    # ---------------------------------------------------------------- close
    def close(self):
        if not self.opened:
            return
        vkv = vk()
        # idle BEFORE touching the cache: vkGetPipelineCacheData while the
        # queue can still reference the cache is a host-side race, and the
        # data grabbed mid-submit can be stale/torn (review N2 ordering fix)
        vkv.vkDeviceWaitIdle(self.device)
        self.save_pipeline_cache()
        for prog in self._programs.values():
            if prog.ubo is not None:
                prog.ubo.destroy()
            if prog.module is not None:
                vkv.vkDestroyShaderModule(self.device, prog.module, None)
            if prog.compute_pipeline is not None:
                vkv.vkDestroyPipeline(self.device, prog.compute_pipeline,
                                      None)
            for pl in prog.gfx_pipelines.values():
                vkv.vkDestroyPipeline(self.device, pl, None)
            if prog.pipeline_layout is not None:
                vkv.vkDestroyPipelineLayout(self.device,
                                            prog.pipeline_layout, None)
            for lay in set(prog.layouts.values()):
                vkv.vkDestroyDescriptorSetLayout(self.device, lay, None)
        if self.descriptor_pool is not None:
            vkv.vkDestroyDescriptorPool(self.device, self.descriptor_pool,
                                        None)
        for s in self._sampler_cache.values():
            vkv.vkDestroySampler(self.device, s, None)
        if self._vert_module is not None:
            vkv.vkDestroyShaderModule(self.device, self._vert_module, None)
        if self._vbo is not None:
            self._vbo.destroy()
            self._vbo = None
        if self.pcache is not None:
            vkv.vkDestroyPipelineCache(self.device, self.pcache, None)
        vkv.vkDestroyCommandPool(self.device, self.cmd_pool, None)
        vkv.vkDestroyDevice(self.device, None)
        vkv.vkDestroyInstance(self.inst, None)
        self.opened = False

    # -------------------------------------------------------------- plumbing
    def find_memtype(self, need):
        props = self.memprops
        for i in range(props.memoryTypeCount):
            if props.memoryTypes[i].propertyFlags & need == need:
                return i
        raise GlvrError(f"no memory type for flags {need:#x}")

    def make_buffer(self, size, usage, host_visible=False):
        return GpuBuffer(self, size, usage, host_visible)

    def make_image(self, w, h, fmt_name, usage, initial_layout, depth=1,
                   dims=2):
        return GpuImage(self, w, h, fmt_name, usage, initial_layout,
                        depth=depth, dims=dims)

    def sampler(self, filt="linear", addr="clamp"):
        vkv = vk()
        key = (filt, addr)
        if key not in self._sampler_cache:
            f = vkv.VK_FILTER_LINEAR if filt == "linear" \
                else vkv.VK_FILTER_NEAREST
            a = vkv.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE if addr == "clamp" \
                else vkv.VK_SAMPLER_ADDRESS_MODE_REPEAT
            self._sampler_cache[key] = vkv.vkCreateSampler(self.device,
                vkv.VkSamplerCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    magFilter=f, minFilter=f, addressModeU=a, addressModeV=a,
                    addressModeW=a,
                    borderColor=vkv.VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK),
                None)
        return self._sampler_cache[key]

    def begin(self):
        """Start a one-shot command buffer (session API: begin ... end)."""
        vkv = vk()
        if self._active_cmd is not None:
            raise GlvrError("command session already active")
        cmd = vkv.vkAllocateCommandBuffers(self.device,
            vkv.VkCommandBufferAllocateInfo(
                sType=vkv.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.cmd_pool,
                level=vkv.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1))[0]
        vkv.vkBeginCommandBuffer(cmd, vkv.VkCommandBufferBeginInfo(
            sType=vkv.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vkv.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        self._active_cmd = cmd
        return cmd

    def end(self, submit=True, wait=True):
        vkv = vk()
        cmd = self._active_cmd
        if cmd is None:
            raise GlvrError("no active command session")
        vkv.vkEndCommandBuffer(cmd)
        self._active_cmd = None
        if submit:
            vkv.vkQueueSubmit(self.queue, 1, [vkv.VkSubmitInfo(
                sType=vkv.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pCommandBuffers=[cmd])], None)
            if wait:
                vkv.vkQueueWaitIdle(self.queue)
            vkv.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, [cmd])
            self._programs_keep_after_submit(cmd)
        return cmd

    def _programs_keep_after_submit(self, cmd):
        """Descriptor-info structs must outlive the submit that references
        them; with wait=True they can be dropped here."""
        for prog in self._programs.values():
            prog._keep.clear()
        del _KEEP[:]

    def barrier(self, cmd, src_stage, dst_stage, src_access, dst_access,
                images=()):
        """vkCmdPipelineBarrier with image barriers; images entries are
        (GpuImage|handle, old_layout, new_layout)."""
        vkv = vk()
        ibs = []
        for img, old, new in images:
            handle = img.image if isinstance(img, GpuImage) else img
            ibs.append(vkv.VkImageMemoryBarrier(
                sType=vkv.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=src_access, dstAccessMask=dst_access,
                oldLayout=old, newLayout=new,
                srcQueueFamilyIndex=vkv.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vkv.VK_QUEUE_FAMILY_IGNORED,
                image=handle,
                subresourceRange=vkv.VkImageSubresourceRange(
                    aspectMask=vkv.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1,
                    layerCount=1)))
        vkv.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, [], 0, [],
                                 len(ibs), ibs)

    def staging(self, size):
        """Host-coherent staging buffer, cached per (rounded-up) size."""
        vkv = vk()
        size = int(max(size, 4))
        if size not in self._staging:
            self._staging[size] = self.make_buffer(
                size, vkv.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | vkv.VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_visible=True)
        return self._staging[size]

    # ------------------------------------------------------- upload/readback
    @staticmethod
    def _pad_rows(arr, bpp):
        """VkBufferImageCopy row rule: with bufferRowLength=0 the texel rows
        are tightly packed, but each row must still start 4-byte aligned -
        true whenever texel bytes * width % 4 == 0. Otherwise rows are padded
        to 4 B and this helper builds (and strips) the pad (the exact inverse
        pair of rules the brief calls 'the reverse of copy_buffer_to_image').
        """
        row = arr.shape[-2] * bpp
        pad = (-row) % 4
        return pad

    def upload_2d(self, arr, fmt_name, usage, rest_layout, depth=1):
        """numpy (h,w[,c]) (or (d,h,w[,c]) for depth>1) -> GpuImage resting
        at rest_layout. Row 0 of the array -> row 0 of the image (GL order;
        see Y-ORIENTATION CLAIM)."""
        vkv = vk()
        a = np.ascontiguousarray(arr)
        if a.ndim == 2:
            a = a[..., None]
        if depth > 1:
            d, h, w = a.shape[0], a.shape[1], a.shape[2]
        else:
            d, h, w = 1, a.shape[0], a.shape[1]
        bpp = _FORMAT_BYTES[fmt_name][0]
        if self._pad_rows(a, bpp):
            raise GlvrError("padded upload unimplemented - use a texel "
                            "format whose row is 4 B aligned (all engine "
                            "formats are)")
        img = self.make_image(w, h, fmt_name, usage,
                              vkv.VK_IMAGE_LAYOUT_UNDEFINED, depth=d)
        data = a.tobytes()
        stg = self.staging(len(data))
        stg.write(data)
        cmd = self.begin()
        self.barrier(cmd, vkv.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     vkv.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                     vkv.VK_ACCESS_TRANSFER_WRITE_BIT,
                     [(img, vkv.VK_IMAGE_LAYOUT_UNDEFINED,
                       vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)])
        img.layout = vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        vkv.vkCmdCopyBufferToImage(cmd, stg.buf, img.image, img.layout, 1, [
            vkv.VkBufferImageCopy(
                imageSubresource=vkv.VkImageSubresourceLayers(
                    aspectMask=vkv.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=d,
                    mipLevel=0),
                imageExtent=vkv.VkExtent3D(width=w, height=h, depth=d))])
        # M4 fix (review): TRANSFER_WRITE -> FRAGMENT|COMPUTE SHADER_READ,
        # matching the contract-table row. BOTTOM_OF_PIPE + access 0 is a
        # layout transition only: it does NOT make the copy's writes
        # available to a later shader read (no execution dependency).
        # COMPUTE included because storage images rest in GENERAL here.
        self.barrier(
            cmd, vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
            (vkv.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
             | vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
            vkv.VK_ACCESS_TRANSFER_WRITE_BIT, vkv.VK_ACCESS_SHADER_READ_BIT,
            [(img, img.layout, rest_layout)])
        img.layout = rest_layout
        self.end()
        return img

    def to_staging(self, cmd, img, rect=None, depth=1):
        """Record: transition img -> TRANSFER_SRC (if needed) + copy to its
        staging buffer. Returns (staging, nbytes). P3 may keep the layout or
        barrier it back afterwards; see the GL->VK table."""
        vkv = vk()
        w, h = img.w, img.h
        x, y, rw, rh = (0, 0, w, h) if rect is None else [int(v) for v in rect]
        n = rw * rh * depth * img.texel
        stg = self.staging(n)
        src = img.layout
        if src != vkv.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            if src == vkv.VK_IMAGE_LAYOUT_GENERAL:
                stage, access = (vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 vkv.VK_ACCESS_SHADER_WRITE_BIT)
            elif src == vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                stage, access = (vkv.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0)
            elif src == vkv.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                stage, access = (vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 vkv.VK_ACCESS_TRANSFER_WRITE_BIT)
            else:                      # COLOR_ATTACHMENT / UNDEFINED
                stage = vkv.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                access = vkv.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            self.barrier(cmd, stage, vkv.VK_PIPELINE_STAGE_TRANSFER_BIT,
                         access, vkv.VK_ACCESS_TRANSFER_READ_BIT,
                         [(img, src,
                           vkv.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)])
            img.layout = vkv.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        vkv.vkCmdCopyImageToBuffer(cmd, img.image, img.layout, stg.buf, 1, [
            vkv.VkBufferImageCopy(
                imageSubresource=vkv.VkImageSubresourceLayers(
                    aspectMask=vkv.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=depth,
                    mipLevel=0),
                imageOffset=vkv.VkOffset3D(x=x, y=y, z=0),
                imageExtent=vkv.VkExtent3D(width=rw, height=rh, depth=depth))])
        return stg, n

    @staticmethod
    def fetch(stg, n, img, rect=None, depth=1):
        """Host-side unpack of a to_staging result (call after wait): numpy
        (h,w,c) or (h,w) for 1-channel, GL row order."""
        rw, rh = (img.w, img.h) if rect is None else (int(rect[2]),
                                                      int(rect[3]))
        a = np.frombuffer(stg.read(n), dtype=img.np_dtype)
        comps = img.texel // img.channel
        shape = ((depth, rh, rw, comps) if depth > 1 else (rh, rw, comps))
        a = a.reshape(shape)
        return a[..., 0] if comps == 1 else a

    def read_2d(self, img, rect=None, depth=1):
        """One-shot convenience: own session + wait, returns numpy."""
        cmd = self.begin()
        stg, n = self.to_staging(cmd, img, rect=rect, depth=depth)
        self.end()
        return self.fetch(stg, n, img, rect=rect, depth=depth)

    # ----------------------------------------------------------- render pass
    def make_target(self, width, height, n_attachments=1,
                    fmt_name=DEFAULT_TARGET_FORMAT,
                    final_layout="VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL"):
        vkv = vk()
        return Target(self, width, height, n_attachments, fmt_name,
                      getattr(vkv, final_layout))

    def _make_renderpass(self, target, cleared):
        vkv = vk()
        fmt = getattr(vkv, target.fmt_name)
        att = vkv.VkAttachmentDescription(
            format=fmt, samples=vkv.VK_SAMPLE_COUNT_1_BIT,
            loadOp=(vkv.VK_ATTACHMENT_LOAD_OP_CLEAR if cleared
                    else vkv.VK_ATTACHMENT_LOAD_OP_LOAD),
            storeOp=vkv.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vkv.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vkv.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=(vkv.VK_IMAGE_LAYOUT_UNDEFINED if cleared
                           else vkv.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
            finalLayout=target.final_layout)
        rp = vkv.vkCreateRenderPass(self.device, vkv.VkRenderPassCreateInfo(
            sType=vkv.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            pAttachments=[att] * target.n,
            pSubpasses=[vkv.VkSubpassDescription(
                pipelineBindPoint=vkv.VK_PIPELINE_BIND_POINT_GRAPHICS,
                pColorAttachments=[
                    vkv.VkAttachmentReference(
                        attachment=i,
                        layout=vkv.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                    for i in range(target.n)])],
            dependencyCount=0), None)
        fb = vkv.vkCreateFramebuffer(self.device,
            vkv.VkFramebufferCreateInfo(
                sType=vkv.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=rp, pAttachments=[im.view for im in target.images],
                width=target.w, height=target.h, layers=1), None)
        return rp, fb

    # ---------------------------------------------------------------- build
    def ensure_spirv(self, name):
        """SPIR-V path for a manifest program (or the shared vertex stage);
        rebuilds via glslc only when missing or older than its source.
        engine_vr/spirv/ cache doubles as the manifest-declared artifact
        location from phase1, so P1 outputs are reused as-is when current."""
        spec = (self.manifest["shared_stages"]["vertex_fullscreen"]
                if name == "_vertex" else self.programs_by_name[name])
        src = WS / spec["source"]
        if name == "_vertex":
            out, args = SPIRV_DIR / "_fullscreen.vert.spv", []
        else:
            out = WS / spec["spirv"]
            args = [f"-D{k}={v}" if v else f"-D{k}"
                    for k, v in (spec.get("defines") or {}).items()]
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(GLSLC), "--target-env=vulkan1.2"] + args + \
                [str(src), "-o", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise GlvrError(f"glslc {name} failed: {r.stderr[:400]}")
        return out

    def local_size(self, name):
        """Workgroup size for a compute program, resolved from ITS SOURCE
        (erosion comp declares layout(local_size_x = TS) with #define TS 16;
        the manifest carries no local_size key)."""
        src = (WS / self.programs_by_name[name]["source"]).read_text()
        m = re.search(r"local_size_x\s*=\s*(\w+)", src)
        if not m:
            raise GlvrError(f"{name}: no local_size in source")
        tok = m.group(1)
        if tok.isdigit():
            x = int(tok)
        else:
            d = re.search(r"#define\s+" + re.escape(tok) + r"\s+(\d+)", src)
            if not d:
                raise GlvrError(f"{name}: unresolved local_size macro {tok}")
            x = int(d.group(1))
        my = re.search(r"local_size_y\s*=\s*(\w+)", src)
        y = x
        if my and not my.group(1).isdigit():
            d = re.search(r"#define\s+" + re.escape(my.group(1)) + r"\s+(\d+)",
                          src)
            y = int(d.group(1)) if d else x
        elif my:
            y = int(my.group(1))
        return x, y, 1

    def build(self, name):
        """Build one program: SPIR-V, shader module, set layouts (from the
        manifest only), pipeline layout, pipeline(s), descriptor sets, UBO.

        MoltenVK compiles SPIR-V -> Metal inside vkCreate*Pipelines, so the
        whole block runs under gpuslot_vr.compile_guard (T6 semantics: same
        as engine/gl.py wrapping moderngl program creation).
        """
        gs = _gpuslot()
        vkv = vk()
        if name not in self.programs_by_name:
            raise GlvrError(f"{name}: not in manifest programs[]")
        if name in self._programs:
            return self._programs[name]
        prog = Program(self, self.programs_by_name[name])
        stage_src = (WS / prog.spec["source"]).read_text()
        with gs.compile_guard(stage_src):
            data = self.ensure_spirv(name).read_bytes()
            prog.module = vkv.vkCreateShaderModule(self.device,
                vkv.VkShaderModuleCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    codeSize=len(data), pCode=data), None)
            _KEEP.append(data)
            stage_flag = (vkv.VK_SHADER_STAGE_COMPUTE_BIT if prog.is_compute
                          else vkv.VK_SHADER_STAGE_FRAGMENT_BIT)
            # All four sets go into the pipeline layout; unused sets share
            # one EMPTY layout (phase1 §glslc table item 6: MoltenVK wants
            # every set present even when nothing binds there).
            empty = vkv.vkCreateDescriptorSetLayout(self.device,
                vkv.VkDescriptorSetLayoutCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO),
                None)
            for s in range(MAX_SETS):
                bs = prog.sets.get(s)
                if not bs:
                    prog.layouts[s] = empty
                    continue
                lb = []
                for b in bs:
                    bnd = int(b["binding"])
                    kind = b["kind"]
                    if kind == "combined_sampler":
                        dt = vkv.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    elif kind == "uniform_block":
                        dt = vkv.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                    elif kind == "storage_image":
                        dt = vkv.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    else:
                        raise GlvrError(f"{name}: binding kind {kind!r}"
                                        " unimplemented in P2")
                    lb.append(vkv.VkDescriptorSetLayoutBinding(
                        binding=bnd, descriptorType=dt, descriptorCount=1,
                        stageFlags=stage_flag))
                prog.layouts[s] = vkv.vkCreateDescriptorSetLayout(self.device,
                    vkv.VkDescriptorSetLayoutCreateInfo(
                        sType=vkv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        pBindings=lb), None)
            prog.pipeline_layout = vkv.vkCreatePipelineLayout(self.device,
                vkv.VkPipelineLayoutCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    pSetLayouts=[prog.layouts[s] for s in range(MAX_SETS)]),
                None)
            if prog.is_compute:
                prog.compute_pipeline = vkv.vkCreateComputePipelines(
                    self.device, self.pcache, 1,
                    vkv.VkComputePipelineCreateInfo(
                        sType=vkv.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        stage=vkv.VkPipelineShaderStageCreateInfo(
                            sType=vkv.
                                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            stage=vkv.VK_SHADER_STAGE_COMPUTE_BIT,
                            module=prog.module, pName=b"main"),
                        layout=prog.pipeline_layout), None)[0]
            # C2 fix (review): gate on ublocker, NOT ubo - ubo starts None
            # in Program.__init__, so the old "if prog.ubo is not None"
            # never created the UBO: set_uniforms raised and set1 stayed
            # unbound at every draw.
            if prog.ublocker is not None:
                prog.ubo = self.make_buffer(
                    prog.ublocker.size,
                    vkv.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    host_visible=True)
            if self.descriptor_pool is None:
                self._make_descriptor_pool()
            for s in range(MAX_SETS):
                if prog.sets.get(s):
                    prog.sets_alloc[s] = vkv.vkAllocateDescriptorSets(
                        self.device, vkv.VkDescriptorSetAllocateInfo(
                            sType=vkv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                            descriptorPool=self.descriptor_pool,
                            pSetLayouts=[prog.layouts[s]]))[0]
        self._programs[name] = prog
        return prog

    def _make_descriptor_pool(self):
        """One pool for the process; counts derived from manifest sums
        (T3), doubled for P3's set churn."""
        vkv = vk()
        n_samp = sum(1 for p in self.manifest["programs"] for b in
                     p["bindings"] if b["kind"] == "combined_sampler")
        n_ubo = sum(1 for p in self.manifest["programs"]
                    if p.get("uniform_block"))
        n_img = sum(1 for p in self.manifest["programs"] for b in
                    p["bindings"] if b["kind"] == "storage_image")
        self.descriptor_pool = vkv.vkCreateDescriptorPool(self.device,
            vkv.VkDescriptorPoolCreateInfo(
                sType=vkv.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                flags=vkv.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                maxSets=4 * len(self.manifest["programs"]) + 32,
                pPoolSizes=[
                    vkv.VkDescriptorPoolSize(
                        type=vkv.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        descriptorCount=2 * n_samp + 16),
                    vkv.VkDescriptorPoolSize(
                        type=vkv.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        descriptorCount=2 * n_ubo + 8),
                    vkv.VkDescriptorPoolSize(
                        type=vkv.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        descriptorCount=2 * n_img + 8)]), None)

    # ------------------------------------------------------------ descriptors
    def set_uniforms(self, prog, fields):
        """Pack + memcpy into the persistent mapped UBO. Queue must be idle
        for the previous submit (barrier-table row 'p.set(...)')."""
        if isinstance(prog, str):
            prog = self.build(prog)
        if prog.ubo is None:
            raise GlvrError(f"{prog.name}: no uniform block")
        prog.ubo.write(prog.ubo_bytes(fields))

    def write_descriptors(self, prog, samplers=None, storage_images=None,
                          ubo_now=True):
        """samplers: {binding name: (GpuImage, 'linear|nearest',
        'clamp|repeat')} - names come from the manifest and are stable across
        programs; storage_images: {binding name: GpuImage} resting in
        GENERAL; ubo_now binds the program's mapped UBO."""
        vkv = vk()
        if isinstance(prog, str):
            prog = self.build(prog)
        writes = []
        for name, val in (samplers or {}).items():
            img, filt, addr = val
            b = prog.by_name[name]
            layout = (img.layout if img.layout == vkv.VK_IMAGE_LAYOUT_GENERAL
                      else vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            info = vkv.VkDescriptorImageInfo(sampler=self.sampler(filt, addr),
                                             imageView=img.view,
                                             imageLayout=layout)
            prog._keep.append(info)
            writes.append(vkv.VkWriteDescriptorSet(
                sType=vkv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=prog.sets_alloc[int(b["set"])],
                dstBinding=int(b["binding"]), descriptorCount=1,
                descriptorType=vkv.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=info))
        for name, img in (storage_images or {}).items():
            b = prog.by_name[name]
            info = vkv.VkDescriptorImageInfo(
                imageView=img.view, imageLayout=vkv.VK_IMAGE_LAYOUT_GENERAL)
            prog._keep.append(info)
            writes.append(vkv.VkWriteDescriptorSet(
                sType=vkv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=prog.sets_alloc[int(b["set"])],
                dstBinding=int(b["binding"]), descriptorCount=1,
                descriptorType=vkv.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=info))
        if ubo_now and prog.ubo is not None:
            b = prog.by_name[prog.ublocker.name]
            bi = vkv.VkDescriptorBufferInfo(buffer=prog.ubo.buf, offset=0,
                                            range=prog.ublocker.size)
            prog._keep.append(bi)
            writes.append(vkv.VkWriteDescriptorSet(
                sType=vkv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=prog.sets_alloc[int(b["set"])],
                dstBinding=int(b["binding"]), descriptorCount=1,
                descriptorType=vkv.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pBufferInfo=bi))
        if writes:
            vkv.vkUpdateDescriptorSets(self.device, len(writes), writes, 0,
                                       None)

    # -------------------------------------------------------- draw / dispatch
    def _bind_sets(self, cmd, prog, bind_point):
        # C6 fix (review): manifest set ids are SPARSE - erosion programs
        # use sets 0/1/3, bake_height uses set 1 only. Binding the sorted
        # allocated sets at firstSet=0 put the UBO at slot 0 and the
        # storage set at slot 2 (layout expects 1/3). Bind each set at its
        # OWN index; runs of consecutive ids share one API call.
        vkv = vk()
        items = sorted(prog.sets_alloc.items())
        i = 0
        while i < len(items):
            start = items[i][0]
            run = [items[i][1]]
            i += 1
            while i < len(items) and items[i][0] == start + len(run):
                run.append(items[i][1])
                i += 1
            vkv.vkCmdBindDescriptorSets(cmd, bind_point, prog.pipeline_layout,
                                        start, len(run), run, 0, [])

    def draw(self, prog, target, rect=None, cleared=False, blend=False,
             clear_value=(0.0, 0.0, 0.0, 0.0), cmd=None):
        """Fullscreen-triangle draw into a Target.

        rect: GL-style tile (x, y, w, h) - verbatim to VkRect2D/VkViewport
        (see Y-ORIENTATION CLAIM). cleared picks the render-pass load
        variant; blend picks the ONE,ONE pipeline variant (kind passes).
        cmd: record into an open session instead of a one-shot draw.
        """
        vkv = vk()
        if isinstance(prog, str):
            prog = self.build(prog)
        # M6 guard (review): pipeline blend-attachment count is
        # len(prog.outputs) while the render pass has target.n; a mismatch
        # is invalid per spec and silently garbage without validation.
        # Refuse loudly (min() would only hide a contract violation; the
        # engine never pairs programs with foreign targets: scene_trace
        # 2 outputs -> gbuf 2 attachments, kinds 1 -> acc 1).
        if len(prog.outputs) != target.n:
            raise GlvrError(
                f"{prog.name}: {len(prog.outputs)} manifest outputs vs "
                f"target with {target.n} color attachment(s)")
        own = cmd is None
        if own:
            cmd = self.begin()
        if not cleared:
            color = vkv.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            if target.images[0].layout != color:
                # M5 fix (review): the LOAD re-entry barrier must name the
                # LAST TOUCHER as src, not FRAGMENT/0. A target resting in
                # TRANSFER_SRC was last read by the readback copy: the new
                # color write needs a WAR guard against that read; resting
                # in SHADER_READ_ONLY means a sampler read it; GENERAL means
                # a compute write.
                rest = target.images[0].layout
                if rest == vkv.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                    stage = vkv.VK_PIPELINE_STAGE_TRANSFER_BIT
                    access = vkv.VK_ACCESS_TRANSFER_READ_BIT
                elif rest == vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                    stage = vkv.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                    access = vkv.VK_ACCESS_SHADER_READ_BIT
                elif rest == vkv.VK_IMAGE_LAYOUT_GENERAL:
                    stage = vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                    access = vkv.VK_ACCESS_SHADER_WRITE_BIT
                else:
                    stage = vkv.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                    access = vkv.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                self.barrier(cmd, stage,
                             vkv.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             access, vkv.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                             [(im, im.layout, color) for im in target.images])
                for im in target.images:
                    im.layout = color
        rp, fb = target.passes[bool(cleared)]
        if rect:
            area = vkv.VkRect2D(
                offset=vkv.VkOffset2D(x=int(rect[0]), y=int(rect[1])),
                extent=vkv.VkExtent2D(width=int(rect[2]), height=int(rect[3])))
        else:
            area = vkv.VkRect2D(offset=vkv.VkOffset2D(x=0, y=0),
                                extent=vkv.VkExtent2D(width=target.w,
                                                      height=target.h))
        clr = None
        if cleared:
            clr = vkv.ffi.new("VkClearValue[%d]" % target.n)
            for i in range(target.n):
                clr[i].color.float32 = list(clear_value)
            _KEEP.append(clr)
        vkv.vkCmdBeginRenderPass(cmd, vkv.VkRenderPassBeginInfo(
            sType=vkv.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=rp, framebuffer=fb, renderArea=area,
            pClearValues=clr), vkv.VK_SUBPASS_CONTENTS_INLINE)
        vkv.vkCmdSetViewport(cmd, 0, 1, [vkv.VkViewport(
            x=float(area.offset.x), y=float(area.offset.y),
            width=float(area.extent.width), height=float(area.extent.height),
            minDepth=0.0, maxDepth=1.0)])
        vkv.vkCmdSetScissor(cmd, 0, 1, [area])
        pipe = self._gfx_pipeline(prog, rp, blend)
        vkv.vkCmdBindPipeline(cmd, vkv.VK_PIPELINE_BIND_POINT_GRAPHICS, pipe)
        self._bind_sets(cmd, prog, vkv.VK_PIPELINE_BIND_POINT_GRAPHICS)
        # M1 fix (review): _fullscreen.vert READS in_pos (manifest
        # shared_stages.vertex_fullscreen: in_pos location 0 vec2) and the
        # pipeline declares attribute binding 0 / stride 8, so a vertex
        # buffer MUST be bound - Metal zero-fills unbound buffers, giving a
        # degenerate triangle and blank output on every pass. Bind the
        # (-1,-1)(3,-1)(-1,3) triangle once and keep it for the host life.
        if self._vbo is None:
            self._vbo = self.fullscreen_vbo()
        vkv.vkCmdBindVertexBuffers(cmd, _VB_ATTR, 1, [self._vbo.buf], [0])
        vkv.vkCmdDraw(cmd, 3, 1, 0, 0)
        vkv.vkCmdEndRenderPass(cmd)
        if target.final_layout == vkv.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            # M5/M4 companion (review sec.4): the render-pass end implicit
            # EXTERNAL dependency covers COLOR-attachment accesses ONLY - a
            # later pass SAMPLING this image (fragment/compute read) is
            # unordered against the color writes without an explicit
            # barrier. MoltenVK serializes everything and hides the absence
            # today; the contract table must be truthful AND correct.
            # old=new layout (already at finalLayout): execution+memory
            # dependency only, no transition.
            self.barrier(
                cmd, vkv.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                (vkv.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                 | vkv.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
                vkv.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                vkv.VK_ACCESS_SHADER_READ_BIT,
                [(im, target.final_layout, target.final_layout)
                 for im in target.images])
        for im in target.images:
            im.layout = target.final_layout
        if own:
            self.end()

    def _gfx_pipeline(self, prog, render_pass, blend):
        vkv = vk()
        # C3 fix (review): handles are cdata pointers - int() raises
        # TypeError on them (empirical), but they ARE hashable, so use the
        # render-pass handle itself as the key.
        key = (render_pass, bool(blend))
        if key in prog.gfx_pipelines:
            return prog.gfx_pipelines[key]
        if self._vert_module is None:
            vdata = self.ensure_spirv("_vertex").read_bytes()
            self._vert_module = vkv.vkCreateShaderModule(self.device,
                vkv.VkShaderModuleCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    codeSize=len(vdata), pCode=vdata), None)
            _KEEP.append(vdata)
        vi = self.manifest["shared_stages"]["vertex_fullscreen"] \
            ["vertex_inputs"][0]
        fmt_map = {"vec2": vkv.VK_FORMAT_R32G32_SFLOAT,
                   "vec3": vkv.VK_FORMAT_R32G32B32A32_SFLOAT}
        stages = [
            vkv.VkPipelineShaderStageCreateInfo(
                sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vkv.VK_SHADER_STAGE_VERTEX_BIT,
                module=self._vert_module, pName=b"main"),
            vkv.VkPipelineShaderStageCreateInfo(
                sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vkv.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=prog.module, pName=b"main")]
        blend_att = vkv.VkPipelineColorBlendAttachmentState(
            blendEnable=1 if blend else 0,
            srcColorBlendFactor=vkv.VK_BLEND_FACTOR_ONE,
            dstColorBlendFactor=vkv.VK_BLEND_FACTOR_ONE,
            colorBlendOp=vkv.VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=vkv.VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=vkv.VK_BLEND_FACTOR_ONE,
            alphaBlendOp=vkv.VK_BLEND_OP_ADD,
            colorWriteMask=vkv.VK_COLOR_COMPONENT_R_BIT
            | vkv.VK_COLOR_COMPONENT_G_BIT | vkv.VK_COLOR_COMPONENT_B_BIT
            | vkv.VK_COLOR_COMPONENT_A_BIT)
        pipe = vkv.vkCreateGraphicsPipelines(self.device, self.pcache, 1,
            vkv.VkGraphicsPipelineCreateInfo(
                sType=vkv.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pStages=stages,
                pVertexInputState=vkv.VkPipelineVertexInputStateCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    pVertexBindingDescriptions=[
                        vkv.VkVertexInputBindingDescription(
                            binding=_VB_ATTR, stride=8,
                            inputRate=vkv.VK_VERTEX_INPUT_RATE_VERTEX)],
                    pVertexAttributeDescriptions=[
                        vkv.VkVertexInputAttributeDescription(
                            location=int(vi["location"]), binding=_VB_ATTR,
                            format=fmt_map[vi["type"]], offset=0)]),
                pInputAssemblyState=vkv.VkPipelineInputAssemblyStateCreateInfo(
                    sType=vkv
                        .VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    topology=vkv.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
                pViewportState=vkv.VkPipelineViewportStateCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    viewportCount=1, scissorCount=1),
                pRasterizationState=vkv.VkPipelineRasterizationStateCreateInfo(
                    sType=vkv
                        .VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    polygonMode=vkv.VK_POLYGON_MODE_FILL,
                    cullMode=vkv.VK_CULL_MODE_NONE,
                    frontFace=vkv.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    lineWidth=1.0),
                pMultisampleState=vkv.VkPipelineMultisampleStateCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    rasterizationSamples=vkv.VK_SAMPLE_COUNT_1_BIT),
                pColorBlendState=vkv.VkPipelineColorBlendStateCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    pAttachments=[blend_att] * len(prog.outputs)),
                pDynamicState=vkv.VkPipelineDynamicStateCreateInfo(
                    sType=vkv.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                    pDynamicStates=[vkv.VK_DYNAMIC_STATE_VIEWPORT,
                                    vkv.VK_DYNAMIC_STATE_SCISSOR]),
                layout=prog.pipeline_layout, renderPass=render_pass,
                subpass=0), None)[0]
        prog.gfx_pipelines[key] = pipe
        return pipe

    def dispatch(self, prog, gx, gy=1, gz=1, cmd=None):
        vkv = vk()
        if isinstance(prog, str):
            prog = self.build(prog)
        if not prog.is_compute:
            raise GlvrError(f"{prog.name}: not a compute program")
        own = cmd is None
        if own:
            cmd = self.begin()
        vkv.vkCmdBindPipeline(cmd, vkv.VK_PIPELINE_BIND_POINT_COMPUTE,
                              prog.compute_pipeline)
        self._bind_sets(cmd, prog, vkv.VK_PIPELINE_BIND_POINT_COMPUTE)
        vkv.vkCmdDispatch(cmd, int(gx), int(gy), int(gz))
        if own:
            self.end()
        return cmd

    # ---------------------------------------------------------------- misc
    def wait(self):
        vk().vkDeviceWaitIdle(self.device)

    def fullscreen_vbo(self):
        """The (-1,-1) (3,-1) (-1,3) triangle covering all pixels - same
        vertices engine/gl.py uploads (gl.py:15) - for P3 paths that bind an
        attribute buffer instead of relying on the vertex stage only."""
        vkv = vk()
        data = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32).tobytes()
        buf = self.make_buffer(len(data),
                               vkv.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                               | vkv.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               host_visible=True)
        buf.write(data)
        return buf


__all__ = ["Host", "GlvrError", "Std140Packer", "load_manifest", "Target",
           "GpuImage", "GpuBuffer", "MANIFEST_PATH", "FMT_BY_COMPS_DTYPE",
           "DEFAULT_TARGET_FORMAT"]
