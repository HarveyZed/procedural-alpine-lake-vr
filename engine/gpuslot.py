"""Machine-wide guard for GPU render / bake processes of this project.

Several look-dev agents render in parallel on one iGPU that shares system RAM. Overnight the
machine ran out of commit memory (Windows "virtual memory insufficient", python.exe the largest
consumer) and the session died. Every process that opens a GPU context now:
  1. takes one of ALPINE_MAX_GPU (default 3; split renderer processes peak at ~5.6 GB) slots - lock files in <repo>/cache/slots,
     stale locks of dead processes are reclaimed;
  2. waits while the free commit memory is below ALPINE_MIN_COMMIT_GB (default 6 GB).
  3. (compile_guard) a FRESH compile of the big scene shader makes the AMD driver peak at ~30 GB of
     commit for ~1-3 min (measured with tools/memprobe.py; cached compiles cost nothing). Sources the
     driver has not compiled before are therefore compiled one at a time machine-wide, and only when
     enough commit memory is free (ALPINE_COMPILE_GB, default 50 GB; a fresh compile of the full scene peaks at 45-50 GB, often at the first draw).
  4. a watchdog thread stops a slot-holding process (exit code 3, message on stderr) whose own commit
     grows past ALPINE_MAX_PROC_GB (default 16 GB) outside of a shader compile, or past 6 GB while the
     machine has < 2.5 GB commit left (a long-lived render server that leaked GPU objects reached 35 GB
     and left the machine with 3 GB). Restarting the process frees everything.
Set ALPINE_NO_SLOT=1 to bypass (single interactive use)."""
import atexit
import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

SLOTS = Path(os.environ.get("ALPINE_SLOTS", str(Path(__file__).resolve().parent.parent / "cache" / "slots")))
_held = None
_compiling = [False]


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_commit_gb():
    try:
        m = _MEMSTAT(); m.dwLength = ctypes.sizeof(_MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPageFile / 2 ** 30
    except Exception:
        return 1e9


def _alive(pid):
    try:
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = wintypes.DWORD()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259                  # STILL_ACTIVE
    except Exception:
        return True


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def proc_gb():
    try:
        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        c = _PMC(); c.cb = ctypes.sizeof(_PMC)
        k.K32GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PrivateUsage / 2 ** 30
    except Exception:
        return 0.0


def _watchdog():
    lim = float(os.environ.get("ALPINE_MAX_PROC_GB", "16"))
    while True:
        time.sleep(5)
        if _compiling[0]:
            continue
        mine = proc_gb()
        if mine > lim or (mine > 6.0 and free_commit_gb() < 2.5):
            print(f"[gpuslot] WATCHDOG: this process uses {mine:.1f} GB (limit {lim:.0f} GB, machine free commit "
                  f"{free_commit_gb():.1f} GB) - exiting to protect the machine. Restart it (memory is freed on exit); "
                  f"long-lived processes that rebuild Renderers/shaders leak GPU memory.", file=sys.stderr, flush=True)
            _release()
            os._exit(3)


def _release():
    global _held
    if _held is not None:
        try:
            _held.unlink()
        except OSError:
            pass
        _held = None


def acquire():
    """block until a GPU slot is free and enough commit memory is available."""
    global _held
    if _held is not None or os.environ.get("ALPINE_NO_SLOT") == "1":
        return
    n = int(os.environ.get("ALPINE_MAX_GPU", "3"))
    need = float(os.environ.get("ALPINE_MIN_COMMIT_GB", "6"))
    SLOTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); last = 0.0
    while True:
        if free_commit_gb() >= need:
            for i in range(n):
                f = SLOTS / f"slot{i}.lock"
                try:
                    fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode()); os.close(fd)
                    _held = f
                    atexit.register(_release)
                    import threading
                    threading.Thread(target=_watchdog, daemon=True).start()
                    if time.time() - t0 > 5:
                        print(f"[gpuslot] got slot {i} after {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
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
            print(f"[gpuslot] waiting: {n} render slots busy or free commit {free_commit_gb():.1f} GB < {need} GB",
                  file=sys.stderr, flush=True)
            last = time.time()
        time.sleep(3)


COMPILED = SLOTS / "compiled"


class compile_guard:
    """with compile_guard(source): ...compile...   (no-op for sources compiled before on this machine)"""

    def __init__(self, source, need_gb=None):
        import hashlib
        self.key = hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()
        self.size = len(source)
        self.need_gb = need_gb       # free commit to wait for (None: estimate from the source size)
        self.lock = None

    def __enter__(self):
        if os.environ.get("ALPINE_NO_SLOT") == "1":
            return self
        COMPILED.mkdir(parents=True, exist_ok=True)
        if (COMPILED / self.key).exists() or len(self.key) < 8:
            return self
        # the cost grows with the source: ~50 GB for the full ~535 KB scene shader, a few GB for the bake passes
        big = float(os.environ.get("ALPINE_COMPILE_GB", "50"))
        need0 = max(6.0, big * min(1.0, (self.size / 535000.0)) ** 1.3)
        if self.need_gb is not None:          # (split scene passes: each < 1 GB, tools/split_probe.py)
            need0 = float(self.need_gb)
        if self.need_gb is not None and self.need_gb <= 8.0:
            # small measured compiles (split scene passes, < 1 GB each): no machine-wide lock, only wait
            # for the free commit
            t0 = time.time()
            while free_commit_gb() < need0:
                if time.time() - t0 > 60:
                    print(f"[gpuslot] waiting for {need0:.0f} GB free commit to compile", file=sys.stderr, flush=True); t0 = time.time()
                time.sleep(3)
            _compiling[0] = True
            return self
        f = SLOTS / "compile.lock"
        t0 = time.time(); last = 0.0
        while True:
            waited = time.time() - t0
            need = need0
            if free_commit_gb() >= need:
                try:
                    fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode()); os.close(fd)
                    self.lock = f
                    _compiling[0] = True
                    if waited > 5:
                        print(f"[gpuslot] compiling after {waited:.0f}s wait", file=sys.stderr, flush=True)
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
                print(f"[gpuslot] waiting to compile a new shader: another compile running or free commit "
                      f"{free_commit_gb():.1f} GB < {need:.0f} GB (a fresh compile peaks at ~30 GB)", file=sys.stderr, flush=True)
                last = time.time()
            time.sleep(3)

    def __exit__(self, et, ev, tb):
        _compiling[0] = False
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
