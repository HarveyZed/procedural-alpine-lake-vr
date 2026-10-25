"""Machine-wide guard for the Vulkan (MoltenVK) render host: GPU slot + memory budget.

Port of engine/gpuslot.py to macOS/Apple-Silicon semantics. Same environment
names and the same public API (acquire / compile_guard / free_commit_gb /
proc_gb) so engine code and P3/P4 drivers keep working unchanged.

WHAT IS THE SAME AS engine/gpuslot.py
  * ALPINE_SLOTS (default <repo>/cache/slots), ALPINE_NO_SLOT, ALPINE_MAX_GPU,
    ALPINE_MIN_COMMIT_GB, ALPINE_MAX_PROC_GB, ALPINE_COMPILE_GB.
  * O_CREAT|O_EXCL lock files holding the pid, stale locks of dead processes
    reclaimed, atexit release, machine-wide compile lock, daemon watchdog that
    hard-exits the process (code 3) rather than letting it eat the machine.

WHAT IS DIFFERENT ON macOS / APPLE SILICON (documented, not silent)
  1. There is no "commit limit". engine/gpuslot.py read ullAvailPageFile from
     GlobalMemoryStatusEx; the closest macOS quantity is reclaimable RAM: the
     Mach VM statistics free + inactive pages (reclaimable without paging).
     No windll, no pagefile.
  2. The per-process number is the larger of ru_maxrss (resource module, bytes
     on macOS) and proc_pid_rusage footprint slots. NOTE (measured on this
     macOS, port/reports/phase2.md): the in-process rusage_info layout does
     NOT match the public header - the resident/footprint uint64 slots land
     two words later than published, and the published slots read 0. We
     therefore take the MAX of several probed slots with a sanity bound
     instead of trusting one offset, and fall back to ru_maxrss.
  3. No TDR. Windows/AMD resets the GPU after ~2 s of a hung command; on Metal
     a runaway dispatch hangs until the watchdog kills us, so the watchdog is
     the ONLY protection and its limits are conservative.
  4. Hard cap = min(ALPINE_MAX_PROC_GB, 0.75 x device recommendedMaxMemorySize)
     where the latter is MoltenVK's VkPhysicalDeviceLimits.recommendedMaxMemory
     Size (fed by gl_vr.Host.open() through configure_budget()); standalone use
     falls back to hw.memsize. ALPINE_MAX_PROC_GB still wins when smaller.
  5. Fail-open on unreadable VM stats (1e9) is deliberate - the same direction
     as the inherited behaviour: a broken stat read must not deadlock every
     job on a threshold it cannot measure. proc_gb reads are the primary
     protection and use only stdlib + probed ctypes paths.

ctypes recipes used here were verified against /usr/bin/vm_stat on this
machine (256 GiB, 16 KiB pages) - see engine_vr/tmp/probe_mem*.py.
host_statistics64 needs the Mach host_self() handle as a uint32 and WRITES
32-BIT COUNTER WORDS (a uint64-fielded struct misreads garbage); the count
argument must be preloaded >= 24 or the kernel rejects the call.
"""
import atexit
import ctypes
import ctypes.util
import os
import resource
import sys
import threading
import time
from pathlib import Path

SLOTS = Path(os.environ.get(
    "ALPINE_SLOTS",
    str(Path(__file__).resolve().parent.parent / "cache" / "slots")))
COMPILED = SLOTS / "compiled"
HELD = None
_COMPILING = [False]
_LOCK = threading.Lock()

# configured by gl_vr.Host.open() from the device memory budget (bytes); 0 = unset
_BUDGET_BYTES = [0]

DEFAULT_MAX_PROC_GB = 16.0
WATCHDOG_HEADROOM_GB = 6.0
WATCHDOG_FLOOR_GB = 2.5
BUDGET_FACTOR = 0.75

# rusage_info slots that measured sane on this libc (public header order lies:
# see module docstring item 2). probed via engine_vr/tmp/probe_rusage*.py:
# the resident/footprint pair lands at uint64 slots 8/9 here, slot 7 is the
# published phys_footprint offset and reads 0 on this build but is correct on
# stock headers, so we keep it in the max().
_RUSAGE_SLOTS = (7, 8, 9)
# C5 fix (review): proc_pid_rusage(flavor 2 = RUSAGE_INFO_V2) WRITES the
# whole rusage_info_v2 struct - measured empirically on this machine by
# prefilling a 1 KiB buffer with 0xAA and scanning for changes: exactly 160
# bytes written (last offset 159). The old (c_uint64 * 12) = 96 B buffer was
# a 64-byte heap overflow. 20 words = 160 B, exact.
_RUSAGE_WORDS = 20


def _env_gb(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _sysctl_bytes(name="hw.memsize"):
    """sysctlbyname for a size_t (hw.memsize); 0.0 when unavailable."""
    try:
        libc = ctypes.CDLL(None)
        libc.sysctlbyname.restype = ctypes.c_int
        libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.c_void_p, ctypes.c_size_t]
        n = ctypes.c_size_t(0)
        if libc.sysctlbyname(name.encode(), None, ctypes.byref(n), None, 0) != 0:
            return 0.0
        buf = ctypes.create_string_buffer(int(n.value))
        if libc.sysctlbyname(name.encode(), ctypes.cast(buf, ctypes.c_void_p),
                             ctypes.byref(n), None, 0) != 0:
            return 0.0
        return float(int.from_bytes(buf.raw[:int(n.value)], "little"))
    except Exception:
        return 0.0


def machine_gb():
    """Total physical memory (GiB); 0 when the sysctl is unavailable."""
    return _sysctl_bytes() / 2 ** 30


_LIBSYS = None
_HOST_SELF = [None]
_PAGE = [None]


def _libsys():
    global _LIBSYS
    if _LIBSYS is None:
        _LIBSYS = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.dylib")
    return _LIBSYS


def _host_self():
    """Mach host self port as uint32; None on failure.

    restype MUST be c_uint32: declaring c_void_p makes the follow-up kernel
    call segfault (observed). A raw int handle (0 / -2) returns
    MACH_SEND_INVALID_DEST (0x10000003) - this API needs the real port name.
    """
    if _HOST_SELF[0] is None:
        try:
            lib = _libsys()
            lib.host_self.restype = ctypes.c_uint32
            hs = int(lib.host_self())
            _HOST_SELF[0] = hs if hs > 0 else False
        except Exception:
            _HOST_SELF[0] = False
    return _HOST_SELF[0] or None


def _page_bytes():
    """vm_page_size via host_page_size (host_page_size64 does not exist in
    this libSystem - the inherited gpuslot_vr.py failed on it), fallback
    SC_PAGESIZE, fallback 4096 (Apple Silicon is 16384; 4096 only under-sizes
    the watchdog, which is the safe direction)."""
    if _PAGE[0] is None:
        page = 0
        try:
            hs = _host_self()
            if hs:
                lib = _libsys()
                lib.host_page_size.restype = ctypes.c_int
                lib.host_page_size.argtypes = [ctypes.c_uint32,
                                               ctypes.POINTER(ctypes.c_uint64)]
                out = ctypes.c_uint64(0)
                if lib.host_page_size(ctypes.c_uint32(hs),
                                      ctypes.byref(out)) == 0:
                    page = int(out.value)
        except Exception:
            page = 0
        if not page:
            try:
                page = int(os.sysconf("SC_PAGESIZE"))
            except Exception:
                page = 4096
        _PAGE[0] = float(page or 4096)
    return _PAGE[0]


def _vm_pages():
    """(free, inactive) page counts via host_statistics64, or None.

    The kernel fills 32-bit words (vm_statistics64 counters are natural-sized
    but this kernel writes 24 words of 4 bytes for HOST_VM_INFO64 when the
    count argument is preloaded with 24; bigger counts are rejected).
    """
    hs = _host_self()
    if hs is None:
        return None
    try:
        lib = _libsys()
        lib.host_statistics64.restype = ctypes.c_int
        lib.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_uint,
                                          ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_uint)]
        buf = (ctypes.c_uint32 * 24)()
        count = ctypes.c_uint(24)          # must be PRELOADED - in/out
        # flavor 4 = HOST_VM_INFO64
        rc = lib.host_statistics64(ctypes.c_uint(hs), ctypes.c_uint(4),
                                   ctypes.cast(buf, ctypes.c_void_p),
                                   ctypes.byref(count))
        if rc != 0:
            return None
        return int(buf[0]), int(buf[2])    # free_count, inactive_count
    except Exception:
        return None


def free_commit_gb():
    """Reclaimable system memory in GiB: the macOS reading of 'free commit'.

    free + inactive pages - reclaimable without paging (wire is pinned, purge-
    able is already counted inside inactive by the VM). Never raises: an
    unreadable VM stat returns 1e9 so a caller waiting on a threshold does not
    spin forever; the page-total sanity bound catches a mislaid struct read
    (garbage counters would otherwise open the gate machine-wide).
    """
    try:
        pages = _vm_pages()
        if pages is None:
            return 1e9
        free, inactive = pages
        page = _page_bytes()
        mem = _sysctl_bytes()
        tot = (free + inactive) * page
        if mem and not (0 <= tot <= 4.0 * mem):
            return 1e9
        return tot / 2 ** 30
    except Exception:
        return 1e9


def proc_gb():
    """Footprint of this process in GiB - the conservative max of the reads
    that are known-good on this libc (module docstring item 2)."""
    best = 0.0
    try:
        best = max(best, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)
    except Exception:
        pass
    try:
        mem = _sysctl_bytes()
        libc = ctypes.CDLL(None)
        libc.proc_pid_rusage.restype = ctypes.c_int
        libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        buf = (ctypes.c_uint64 * _RUSAGE_WORDS)()
        # flavor 2 = RUSAGE_INFO_V2; the arg is an OUT pointer (rusage_info_t*).
        # A POINTER(c_void_p)+byref wrapper gets a ZEROED struct on this libc -
        # pass the buffer address directly (probe_rusage2.py).
        if libc.proc_pid_rusage(os.getpid(), 2, ctypes.cast(buf, ctypes.c_void_p)) == 0:
            for s in _RUSAGE_SLOTS:
                v = float(buf[s])
                if 0 < v and (not mem or v < 4.0 * mem):
                    best = max(best, v / 2 ** 30)
    except Exception:
        pass
    return best


def configure_budget(recommended_max_bytes):
    """gl_vr.Host.open() feeds MoltenVK's recommendedMaxMemorySize in here."""
    _BUDGET_BYTES[0] = int(max(0, recommended_max_bytes or 0))


def budget_limit_gb():
    """Watchdog limit: min(ALPINE_MAX_PROC_GB, 0.75 x device budget).

    ALPINE_MAX_PROC_GB is an explicit override of the ORIGINAL limit; when the
    device budget is known the tighter of the two wins, because on unified
    memory exceeding the recommended working set is what makes Metal fail
    allocations and jetsam kill the process.
    """
    lim = _env_gb("ALPINE_MAX_PROC_GB", DEFAULT_MAX_PROC_GB)
    if _BUDGET_BYTES[0] <= 0:
        mem = _sysctl_bytes()          # standalone: hw.memsize stands in
        if mem > 0:
            _BUDGET_BYTES[0] = int(mem)
    if _BUDGET_BYTES[0] > 0:
        lim = min(lim, _BUDGET_BYTES[0] * BUDGET_FACTOR / 2 ** 30)
    return lim


def _alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _watchdog():
    while True:
        time.sleep(5)
        if _COMPILING[0]:
            continue
        lim = budget_limit_gb()
        mine = proc_gb()
        if (mine > lim
                or (mine > WATCHDOG_HEADROOM_GB
                    and free_commit_gb() < WATCHDOG_FLOOR_GB)): 
            print(f"[gpuslot_vr] WATCHDOG: this process uses {mine:.1f} GB "
                  f"(limit {lim:.1f} GB, machine reclaimable "
                  f"{free_commit_gb():.1f} GB) - exiting to protect the machine. "
                  f"Restart it (memory is freed on exit); long-lived processes "
                  f"that rebuild Host/pipelines leak device memory.",
                  file=sys.stderr, flush=True)
            _release()
            os._exit(3)


def _release():
    global HELD
    with _LOCK:
        if HELD is not None:
            try:
                HELD.unlink()
            except OSError:
                pass
            HELD = None


def release():
    """Public release (passes_vr.open_host error path + close_host
    call release(); the internal watchdog keeps using _release())."""
    _release()


def acquire():
    """Block until a GPU slot is free and enough memory is reclaimable."""
    global HELD
    if HELD is not None or os.environ.get("ALPINE_NO_SLOT") == "1":
        return
    n = int(os.environ.get("ALPINE_MAX_GPU", "3"))
    need = _env_gb("ALPINE_MIN_COMMIT_GB", 6)
    SLOTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last = 0.0
    while True:
        if free_commit_gb() >= need:
            for i in range(n):
                f = SLOTS / f"slot{i}.lock"
                try:
                    with _LOCK:
                        fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(fd, str(os.getpid()).encode())
                        os.close(fd)
                        HELD = f
                    atexit.register(_release)
                    threading.Thread(target=_watchdog, daemon=True).start()
                    if time.time() - t0 > 5:
                        waited = time.time() - t0
                        print(f"[gpuslot_vr] got slot {i} after {waited:.0f}s",
                              file=sys.stderr, flush=True)
                    return
                except FileExistsError:
                    try:
                        pid = int(f.read_text().strip() or "0")
                    except (OSError, ValueError):
                        pid = 0
                    if pid and not _alive(pid):
                        try:
                            f.unlink()
                        except OSError:
                            pass
        if time.time() - last > 60:
            print(f"[gpuslot_vr] waiting: {n} slots busy or reclaimable "
                  f"{free_commit_gb():.1f} GB < {need} GB",
                  file=sys.stderr, flush=True)
            last = time.time()
        time.sleep(3)


class compile_guard:
    """with compile_guard(source, need_gb): ...create pipelines...

    MoltenVK compiles SPIR-V to Metal inside vkCreateGraphicsPipelines, so this
    guard wraps pipeline creation, exactly like the original wrapped the driver
    shader compile. Sources whose key is already recorded are a no-op.
    """

    def __init__(self, source, need_gb=None):
        import hashlib
        self.key = hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()
        self.size = len(source)
        self.need_gb = need_gb
        self.lock = None

    def _need(self):
        big = _env_gb("ALPINE_COMPILE_GB", 50)
        need0 = max(6.0, big * min(1.0, (self.size / 535000.0)) ** 1.3)
        if self.need_gb is not None:
            need0 = float(self.need_gb)
        return need0

    def __enter__(self):
        if os.environ.get("ALPINE_NO_SLOT") == "1":
            return self
        COMPILED.mkdir(parents=True, exist_ok=True)
        if (COMPILED / self.key).exists() or len(self.key) < 8:
            return self
        need0 = self._need()
        # M3 fix (review): ORIGINAL condition (engine/gpuslot.py:171)
        if self.need_gb is not None and self.need_gb <= 8.0:
            t0 = time.time()
            while free_commit_gb() < need0:
                if time.time() - t0 > 60:
                    print(f"[gpuslot_vr] waiting for {need0:.0f} GB reclaimable "
                          f"to compile", file=sys.stderr, flush=True)
                    t0 = time.time()
                time.sleep(3)
            _COMPILING[0] = True
            return self
        f = SLOTS / "compile.lock"
        t0 = time.time()
        last = 0.0
        while True:
            waited = time.time() - t0
            if free_commit_gb() >= need0:
                try:
                    fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    self.lock = f
                    _COMPILING[0] = True
                    if waited > 5:
                        print(f"[gpuslot_vr] compiling after {waited:.0f}s wait",
                              file=sys.stderr, flush=True)
                    return self
                except FileExistsError:
                    try:
                        pid = int(f.read_text().strip() or "0")
                    except (OSError, ValueError):
                        pid = 0
                    if pid and not _alive(pid):
                        try:
                            f.unlink()
                        except OSError:
                            pass
                    elif pid == 0 and time.time() - f.stat().st_mtime > 30:
                        try:
                            f.unlink()
                        except OSError:
                            pass
            if time.time() - last > 60:
                print(f"[gpuslot_vr] waiting to compile: another compile running "
                      f"or reclaimable {free_commit_gb():.1f} GB < {need0:.0f} GB",
                      file=sys.stderr, flush=True)
                last = time.time()
            time.sleep(3)

    def __exit__(self, et, ev, tb):
        _COMPILING[0] = False
        if self.lock is not None:
            try:
                self.lock.unlink()
            except OSError:
                pass
        if et is None:
            try:
                (COMPILED / self.key).touch()
            except OSError:
                pass
        return False
