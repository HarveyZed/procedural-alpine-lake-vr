"""Run a python script in-process while logging this process's private memory and the machine's free
commit (1 s) to out/memwrap.log; prints the peaks at exit.
usage: python tools/memwrap.py <script.py> [args ...]"""
import atexit, runpy, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
import gpuslot
(ROOT / "out").mkdir(exist_ok=True)
log = open(ROOT / "out/memwrap.log", "a", encoding="utf-8")
peak = [0.0, 1e9]
t0 = time.time()
log.write(f"== {time.strftime('%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}\n"); log.flush()
def mon():
    last = -99.0
    while True:
        m, f = gpuslot.proc_gb(), gpuslot.free_commit_gb()
        peak[0] = max(peak[0], m); peak[1] = min(peak[1], f)
        if abs(m - last) > 0.5:
            log.write(f"{time.time()-t0:7.1f}s proc {m:5.1f} GB  free {f:5.1f} GB\n"); log.flush(); last = m
        time.sleep(1.0)
threading.Thread(target=mon, daemon=True).start()
atexit.register(lambda: (print(f"[memwrap] peak proc {peak[0]:.1f} GB, min free {peak[1]:.1f} GB, {time.time()-t0:.0f}s", flush=True),
                         log.write(f"   peak proc {peak[0]:.1f} GB, min free {peak[1]:.1f} GB\n"), log.flush()))
script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
