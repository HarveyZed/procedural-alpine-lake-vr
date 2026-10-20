"""Per-pass shader source reduction for the split renderer (scene.frag is compiled 9 times).

reduce(src) resolves the #if / #ifdef / #ifndef / #elif / #else / #endif blocks for the defines in the
source itself, then drops every function definition that main() cannot reach. Each pass's source - and so
the driver's compile-cache key - then holds only the code that pass runs: an edit to the grass code no
longer recompiles the sky or the tree-shading pass, and the compiler parses less. Anything it cannot
parse raises ValueError; gl.Pass then compiles the full source instead (and also retries with the full
source if a reduced one fails to compile).
    python engine/glstrip.py            -> sizes of the 9 scene passes, full vs reduced
"""
import re

_ID = re.compile(r"[A-Za-z_]\w*")


def _eval(expr, macros):
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)",
               lambda m: "1" if (m.group(1) or m.group(2)) in macros else "0", expr)
    for _ in range(16):                                  # expand object-like macros
        n = _ID.sub(lambda m: "(" + macros[m.group(0)] + ")" if macros.get(m.group(0)) not in (None, "") else m.group(0), e)
        if n == e:
            break
        e = n
    e = re.split(r"//", e)[0]
    e = _ID.sub("0", e)                                  # undefined identifiers are 0 (C preprocessor)
    if not re.fullmatch(r"[\d\s()+\-*/%<>=!&|.]*", e):
        raise ValueError(f"cannot evaluate #if {expr!r}")
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    return bool(eval(e, {"__builtins__": {}}, {}))


def resolve(src):
    """keep only the active branches of the conditional blocks (directives of resolved blocks removed)"""
    macros = {}
    out, stack = [], []
    active = True
    for line in src.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            d = s[1:].strip()
            kw = d.split(None, 1)[0] if d else ""
            rest = d[len(kw):].strip()
            if kw in ("if", "ifdef", "ifndef"):
                if not active:
                    c = False
                elif kw == "if":
                    c = _eval(rest, macros)
                else:
                    c = (rest.split()[0] in macros) == (kw == "ifdef")
                stack.append([active, c])
                active = active and c
                continue
            if kw == "elif":
                parent, taken = stack[-1]
                c = parent and not taken and _eval(rest, macros)
                stack[-1][1] = taken or c
                active = parent and c
                continue
            if kw == "else":
                parent, taken = stack[-1]
                active = parent and not taken
                stack[-1][1] = True
                continue
            if kw == "endif":
                active = stack.pop()[0]
                continue
            if not active:
                continue
            if kw == "define":
                m = re.match(r"(\w+)(\()?", rest)
                if not m:
                    raise ValueError(f"bad #define {line!r}")
                if m.group(2):
                    macros[m.group(1)] = None                # function-like: defined, never expanded
                else:
                    macros[m.group(1)] = re.split(r"//|/\*", rest[len(m.group(1)):])[0].strip()
            elif kw == "undef":
                macros.pop(rest.split()[0], None)
            out.append(line)
            continue
        if active:
            out.append(line)
    if stack:
        raise ValueError("unterminated #if")
    return "\n".join(out)


def _mask(src):
    """comments and preprocessor lines blanked (same length), for brace / identifier scanning"""
    m = re.sub(r"//[^\n]*|/\*.*?\*/", lambda x: re.sub(r"[^\n]", " ", x.group(0)), src, flags=re.S)
    return re.sub(r"^[ \t]*#[^\n]*", lambda x: " " * len(x.group(0)), m, flags=re.M)


_HDR = re.compile(r"(?:[A-Za-z_]\w*\s+)*([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\((?:[^()]|\([^()]*\))*\)", re.S)


def strip_unreachable(src):
    mk = _mask(src)
    funcs = []                       # (name, start, end)
    depth, seg, cur = 0, 0, None
    for i, ch in enumerate(mk):
        if ch == "{":
            if depth == 0:
                h = mk[seg:i]
                hs = h.strip()
                m = _HDR.fullmatch(hs)
                cur = (m.group(2), seg + (len(h) - len(h.lstrip()))) if m and m.group(1) != "struct" else None
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced braces")
            if depth == 0:
                if cur:
                    funcs.append((cur[0], cur[1], i + 1))
                cur = None
                seg = i + 1
        elif ch == ";" and depth == 0:
            seg = i + 1
    if depth != 0:
        raise ValueError("unbalanced braces")
    defs = {}
    for f in funcs:
        defs.setdefault(f[0], []).append(f)
    if "main" not in defs:
        raise ValueError("no main()")
    # roots: main, anything named outside function bodies (prototypes, global initialisers) and in #defines
    outside, last = [], 0
    for _, s, e in sorted(funcs, key=lambda f: f[1]):
        outside.append(mk[last:s])
        last = e
    outside.append(mk[last:])
    ids = set(_ID.findall("".join(outside)))
    ids |= set(_ID.findall("\n".join(l for l in src.split("\n") if l.lstrip().startswith("#define"))))
    reach = {"main"} | (ids & defs.keys())
    todo = list(reach)
    while todo:
        n = todo.pop()
        for _, s, e in defs[n]:
            for k in set(_ID.findall(mk[s:e])) & defs.keys():
                if k not in reach:
                    reach.add(k)
                    todo.append(k)
    drop = sorted((f for f in funcs if f[0] not in reach), key=lambda f: f[1])
    out, last = [], 0
    for _, s, e in drop:
        out.append(src[last:s])
        last = e
    out.append(src[last:])
    return "".join(out)


def reduce(src):
    r = strip_unreachable(resolve(src))
    return re.sub(r"\n[ \t]*(?:\n[ \t]*){2,}", "\n\n", r)       # squeeze blank runs


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gl
    body = gl._preprocess("scene.frag", set())
    for k in (99, 0, 1, 2, 3, 4, 5, 6, 7):
        full = "#version 430\n#define QUALITY 1\n#define SCENE_PASS %d\n" % k + body
        r = reduce(full)
        print(f"SCENE_PASS {k:2d}: {len(full)/1000:6.0f} KB -> {len(r)/1000:6.0f} KB")
