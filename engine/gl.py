"""OpenGL helpers: standalone GPU context, shader loading with #include, fullscreen passes."""
import re
from pathlib import Path

import moderngl
import numpy as np

SHADERS = Path(__file__).resolve().parent / "shaders"
_ctx = None
_vbo = None

VS = """#version 430
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""


def ctx():
    global _ctx, _vbo
    if _ctx is None:
        import gpuslot
        gpuslot.acquire()      # machine-wide render slot + free-memory guard (see gpuslot.py)
        _ctx = moderngl.create_standalone_context(require=430)
        _vbo = _ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    return _ctx


def _preprocess(name, seen):
    src = (SHADERS / name).read_text(encoding="utf-8")

    def inc(m):
        f = m.group(1)
        if f in seen:
            return ""
        seen.add(f)
        return _preprocess(f, seen)

    return re.sub(r'^[ \t]*#include\s+"([^"]+)"', inc, src, flags=re.M)


class Pass:
    """A fullscreen fragment-shader pass."""

    def __init__(self, frag, defines=None, compile_gb=None, targets=1):
        import os
        c = ctx()
        header = "#version 430\n" + "".join(f"#define {k} {v}\n" for k, v in (defines or {}).items())
        if os.environ.get("ALPINE_NONCE"):           # force a fresh driver compile (compile-cost tests)
            header += f"#define ALPINE_NONCE {os.environ['ALPINE_NONCE']}\n"
        full = header + _preprocess(frag, set())
        self.source = full
        if os.environ.get("ALPINE_NO_STRIP") != "1":
            # only the code this pass runs (glstrip.py): a smaller compile, and edits elsewhere keep its
            # driver cache entry valid
            try:
                import glstrip
                self.source = glstrip.reduce(full)
            except ValueError as e:
                print(f"[gl] {frag}: no source reduction ({e})", flush=True)
        import gpuslot

        def build():
            # (the monolithic scene shader peaked at ~45-50 GB per fresh compile, see gpuslot.py; compile_gb =
            # free commit to wait for, for passes measured to compile small)
            with gpuslot.compile_guard(self.source, compile_gb):
                self.prog = c.program(vertex_shader=VS, fragment_shader=self.source)
                # the AMD driver may compile lazily at the first draw: do it inside the guard, with the
                # real target layout (float RGBA x targets)
                wt = [c.texture((1, 1), 4, dtype="f4") for _ in range(targets)]
                fb = c.framebuffer(wt); fb.use()
                c.vertex_array(self.prog, [(_vbo, "2f", "in_pos")]).render(moderngl.TRIANGLES, vertices=3)
                c.finish(); fb.release()
                for t in wt:
                    t.release()
        try:
            try:
                build()
            except moderngl.Error:
                if self.source is full:
                    raise
                print(f"[gl] {frag}: the reduced source failed to compile - compiling the full source", flush=True)
                self.source = full
                build()
        except moderngl.Error as e:
            dump = Path(__file__).resolve().parent.parent / "cache" / f"failed_{Path(frag).stem}.glsl"
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_text("\n".join(f"{i+1:5d} {l}" for i, l in enumerate(self.source.split("\n"))))
            raise RuntimeError(f"shader compile failed ({frag}); numbered source in {dump}\n{e}") from None
        self.vao = c.vertex_array(self.prog, [(_vbo, "2f", "in_pos")])
        self._tex_units = {}

    def set(self, name, value):
        if name not in self.prog:
            return
        u = self.prog[name]
        if isinstance(value, np.ndarray):
            u.write(value.astype("f4").tobytes())
        else:
            u.value = value

    def tex(self, name, texture, unit):
        texture.use(location=unit)
        self.set(name, unit)

    def draw(self, fbo, rect=None):
        fbo.use()
        if rect is None:
            fbo.viewport = (0, 0, fbo.width, fbo.height)
        else:
            fbo.viewport = rect
        self.vao.render(moderngl.TRIANGLES, vertices=3)


def tex2d(arr, filt="linear", repeat=False, dtype=None):
    """Upload a HxW or HxWxC float/uint8 numpy array as a 2D texture (row 0 = v 0)."""
    c = ctx()
    a = np.ascontiguousarray(arr)
    comps = 1 if a.ndim == 2 else a.shape[2]
    if dtype is None:
        dtype = "f1" if a.dtype == np.uint8 else ("f2" if a.dtype == np.float16 else "f4")
        if dtype == "f4":
            a = a.astype(np.float32)
    t = c.texture((a.shape[1], a.shape[0]), comps, a.tobytes(), dtype=dtype)
    f = moderngl.LINEAR if filt == "linear" else moderngl.NEAREST
    t.filter = (f, f)
    t.repeat_x = repeat
    t.repeat_y = repeat
    return t


def read_tex(t, comps=None, dtype="f4"):
    comps = comps or t.components
    npdt = {"f4": np.float32, "f1": np.uint8, "f2": np.float16}[dtype]
    a = np.frombuffer(t.read(), dtype=npdt).reshape(t.height, t.width, t.components)
    return a[..., :comps]
