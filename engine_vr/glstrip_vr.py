"""glstrip_vr.py — Phase-1 port of engine/glstrip.py + the GLSL-450-for-Vulkan descriptor pass.

Stages (no GPU, pure text):

  1. include resolution      engine/gl.py::_preprocess semantics (include-once, '#include "x.glsl"')
  2. #if/#ifdef resolution   engine/glstrip.resolve — IMPORTED by path, not copied, so branch results are
                             identical by construction (engine/glstrip.py imports only 're', no moderngl)
  3. dead-function stripping engine/glstrip.strip_unreachable, same reachability roots
  4. wrap_loose_uniforms     Vulkan GLSL forbids "non-opaque uniforms outside a block": every loose
                             uniform <scalar/vector/matrix/array> at top level is moved verbatim, in
                             source order, into one synthetic std140 uniform block with an unnamed
                             instance (members stay in global scope -> no shader body is edited).
                             Opaque types (sampler*/image*) stay loose, where they are legal.
                             std430 on a *uniform* block would need GL_EXT_scalar_block_layout.
  5. binding injection       deterministic layout(set=S, binding=N) on every descriptor. Separate counters
                             per namespace {combined samplers, uniform blocks, storage blocks, storage
                             images}, allocated in order of first appearance in the reduced source, each
                             starting at 0 and skipping numbers already claimed by an explicit binding.
                             A namespace gets its own set (SET_OF_NS: sampler 0, uniform block 1, storage
                             block 2, storage image 3) so the pairs stay unique - see the note at SET_OF_NS.
                             A declaration that already has 'binding=' keeps it; only 'set=' is added when
                             absent (so the DescriptorSet decoration is unconditional). Two explicit
                             bindings on the same set+number = hard error DescriptorCollision.
  6. location injection      SPIR-V requires a location on user in/out, so a top-level in/out declaration
                             without one gets layout(location=N), lowest free number first.

No moderngl anywhere: engine/glstrip.py is loaded by file path (stdlib only); engine/gl.py is NOT loaded
(it imports moderngl) — its include loader is reimplemented here.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC_SHADERS = ROOT / "engine" / "shaders"      # read-only originals (reference only)
VR_SHADERS = HERE / "shaders"                  # the ported copies this build reads
GEN = HERE / "shaders_gen"

GLSL_VERSION = 450
UBLOCK_NAME = "_p1_pass"                        # the synthetic pass-parameter block
UBLOCK_STORAGE = "std140"   # std430 on a *uniform* block needs GL_EXT_scalar_block_layout; std140 needs none


# --------------------------------------------------------------- engine/glstrip reuse (stdlib-only)
def _load_engine_glstrip():
    p = ROOT / "engine" / "glstrip.py"
    spec = importlib.util.spec_from_file_location("_engine_glstrip_p1", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_glstrip = _load_engine_glstrip()
resolve = _glstrip.resolve                       # active-branch resolution (defines read from the source)
strip_unreachable = _glstrip.strip_unreachable   # drop functions main() cannot reach
_mask = _glstrip._mask                           # comments + preprocessor lines blanked, length kept


class GlslError(RuntimeError):
    pass


class DescriptorCollision(GlslError):
    pass


INCLUDE_RE = re.compile(r'^[ \t]*#include\s+"([^"]+)"', re.M)


def preprocess(name, seen=None, shaders_dir=None):
    """engine/gl.py::_preprocess — inline '#include "f"' once per file (cycle-safe)."""
    shaders_dir = Path(shaders_dir or VR_SHADERS)
    seen = set() if seen is None else seen
    src = (shaders_dir / name).read_text(encoding="utf-8")

    def inc(m):
        f = m.group(1)
        if f in seen:
            return ""
        seen.add(f)
        return preprocess(f, seen, shaders_dir)

    return INCLUDE_RE.sub(inc, src)


# --------------------------------------------------------------------- qualifier / type tables
SAMPLER_PREFIXES = ("sampler", "isampler", "usampler")
IMAGE_PREFIXES = ("image", "iimage", "uimage")
NAMESPACES = ("sampler", "uniform_block", "storage_block", "image")
# Each namespace gets its OWN descriptor set. PHASE_1.md asks for layout(set=0, binding=N) *and*
# "namespaces (separate counters) ... start at 0 each": with every resource in set 0 those two rules
# produce duplicate (set, binding) pairs (sampler binding 0 next to the synthetic uniform block's
# binding 0), which is not a legal VkDescriptorSetLayoutBinding array - MoltenVK/wgpu would reject or
# alias the layout. The binding counters keep exactly the briefed semantics; the set id separates the
# namespaces so every (set, binding) pair is unique. An explicit 'set=' in the source still wins.
SET_OF_NS = {"sampler": 0, "uniform_block": 1, "storage_block": 2, "image": 3}
_KIND_NS = {"combined_sampler": "sampler", "uniform_block": "uniform_block",
            "storage_block": "storage_block", "storage_image": "image"}
_QUAL_WORDS = {"uniform", "buffer", "in", "out", "coherent", "volatile", "restrict", "readonly",
               "writeonly", "highp", "mediump", "lowp", "flat", "smooth", "noperspective", "centroid",
               "patch", "sample", "invariant", "precise", "layout"}


def _layout_args(args):
    """(binding, set, location) parsed from a layout(...) argument list; None where absent."""
    def g(key):
        m = re.search(r"\b" + key + r"\s*=\s*(\d+)", args or "")
        return int(m.group(1)) if m else None
    return g("binding"), g("set"), g("location")


def _split_top(text, seps=","):
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in seps:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _strip_qualifiers(body):
    """peel layout(...) / storage-class / interpolation qualifiers: -> (words, [layout_args], tail)"""
    lays, words = [], []
    while True:
        m = re.match(r"\s*(layout\s*\(([^)]*)\)|[A-Za-z_]\w*)\s*", body, re.S)
        if not m:
            break
        if m.group(2) is not None:
            lays.append(m.group(2))
            words.append("layout")
        elif m.group(1) in _QUAL_WORDS:
            words.append(m.group(1))
        else:
            break
        body = body[m.end():]
    return words, lays, body


_DECL_RE = re.compile(r"([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*(//.*|/\*.*\*/)?\s*$", re.S)


def _split_decl(tail):
    """'<type> name[N], name2 /*c*/, ...' -> (type, [(name, array, comment)])"""
    m = re.match(r"\s*([A-Za-z_]\w*)\s*(.*)", tail, re.S)
    if not m:
        return None, []
    typ, rest = m.group(1), m.group(2)
    names = []
    for d in _split_top(rest):
        d = d.strip().rstrip(";").strip()
        if not d:
            continue
        mm = _DECL_RE.match(d)
        if not mm:
            return typ, names
        names.append((mm.group(1), (mm.group(2) or "").strip(), (mm.group(3) or "").strip()))
    return typ, names


# --------------------------------------------------------------------- top-level statement spans
def _statement_spans(src, mk):
    """spans of top-level statements: up to ';' at brace depth 0 (for a block declaration
    'uniform X { ... } inst;' that is the ';' after the closing brace).  The span start is advanced past
    leading blank lines, comments and (blanked) preprocessor lines so that body() starts at the
    declaration itself."""
    raw_spans, depth, seg = [], 0, 0
    for i, ch in enumerate(mk):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise GlslError("unbalanced braces")
            if depth == 0:                      # a function body or a block declaration ends here: the
                raw_spans.append((seg, i + 1))  # next top-level statement starts after it (same rule as
                seg = i + 1                     # engine/glstrip.strip_unreachable's segment tracking)
        elif ch == ";" and depth == 0:
            raw_spans.append((seg, i + 1))
            seg = i + 1
    if depth != 0:
        raise GlslError("unbalanced braces")
    spans = []
    for s, e in raw_spans:
        p = s
        while p < e:
            # skip whitespace, then a whole line if that line is blank under the mask (comment/#line)
            while p < e and mk[p] in " \t\r\n":
                p += 1
            if p >= e:
                break
            nl = mk.find("\n", p)
            nl = e if nl < 0 or nl >= e else nl + 1
            if mk[p:nl].strip() == "":
                p = nl
                continue
            break
        if p < e:
            spans.append((p, e))
    return spans


class Binding:
    def __init__(self, name, kind, binding, glsl_type, block=None, explicit=False, set_=None):
        self.name, self.kind, self.binding = name, kind, binding
        self.glsl_type, self.block, self.explicit = glsl_type, block, explicit
        self.set = SET_OF_NS[_KIND_NS[kind]] if set_ is None else set_

    def as_dict(self):
        d = {"name": self.name, "kind": self.kind, "set": self.set, "binding": self.binding,
             "glsl_type": self.glsl_type}
        if self.block:
            d["block"] = self.block
        if self.explicit:
            d["explicit_binding"] = True
        return d

    def __repr__(self):
        return (f"<{self.kind} {self.name} set={self.set} binding={self.binding}"
                + (" explicit>" if self.explicit else ">"))


def _layout_prefix(has_layout, args0, binding=None, loc=None, want_set=True, set_n=0):
    """text to insert so the declaration carries set/binding (or location).
    With an existing layout(...) this is the insertion *inside* its parentheses; otherwise a full prefix."""
    if has_layout:
        adds = []
        eb, es, el = _layout_args(args0)
        if binding is not None:
            if eb is None:
                adds.append(f"binding={binding}")
            if es is None and want_set:
                adds.append(f"set={set_n}")
        if loc is not None and el is None:
            adds.append(f"location={loc}")
        return "".join(a + ", " for a in adds)
    bits = []
    if loc is not None:
        bits.append(f"location={loc}")
    else:
        if want_set:
            bits.append(f"set={set_n}")
        if binding is not None:
            bits.append(f"binding={binding}")
    return "layout(" + ", ".join(bits) + ") "


def vulkanize(src):
    """Loose-uniform grouping + binding/location injection. Returns (source, [Binding], info)."""
    mk = _mask(src)
    spans = _statement_spans(src, mk)

    # ---- pass A: classify every top-level statement, reserve every explicit binding ----
    recs, used_pair = [], {}          # (set, binding) -> declaring name; explicit + injected alike
    explicit_locs = []
    for s, e in spans:
        body = src[s:e]
        words, lays, tail = _strip_qualifiers(body)
        if not words:
            continue
        has_layout = "layout" in words
        args0 = lays[0] if lays else ""
        eb, es, el = _layout_args(args0)
        is_io = ("in" in words or "out" in words) and "uniform" not in words and "buffer" not in words
        if is_io:
            _t, io_names = _split_decl(tail)
            if not io_names:
                continue                              # 'layout(local_size_x = 16) in;' - no variables
            if el is None:
                recs.append({"s": s, "e": e, "what": "io", "has_layout": has_layout, "args0": args0})
            else:
                explicit_locs.append(el)
            continue
        if "uniform" not in words and "buffer" not in words:
            continue
        if "{" in tail:                                  # interface block with a body
            kind = "storage_block" if "buffer" in words else "uniform_block"
            m = re.match(r"\s*([A-Za-z_]\w*)", tail)
            block = m.group(1) if m else "?"
            after = tail[tail.index("}") + 1:].strip().rstrip(";").strip()
            inst = after.split()[0] if after else block
            ns = _KIND_NS[kind]
            set_n = SET_OF_NS[ns] if es is None else es
            if eb is not None:
                if (set_n, eb) in used_pair:
                    raise DescriptorCollision(f"explicit binding {ns}/set={set_n}/binding={eb} declared "
                                              f"twice: {used_pair[(set_n, eb)]} and {block}")
                used_pair[(set_n, eb)] = block
            recs.append({"s": s, "e": e, "what": "block", "kind": kind, "name": block, "inst": inst,
                         "binding": eb, "set": set_n, "has_layout": has_layout, "args0": args0,
                         "glsl_type": ("buffer " if kind == "storage_block" else "uniform ") + block})
            continue
        typ, names = _split_decl(tail)
        if typ is None or not names:
            continue                                     # 'layout(local_size_x=..) in;' and friends
        opaque = typ.startswith(SAMPLER_PREFIXES) or typ.startswith(IMAGE_PREFIXES)
        if "buffer" in words and not opaque:
            raise GlslError("unsupported storage declaration: " + body[:80].replace("\n", " "))
        if not opaque:
            member = re.sub(r"\s+", " ", tail).strip().rstrip(";").strip()
            recs.append({"s": s, "e": e, "what": "loose", "member": member,
                         "names": [n for n, _, _ in names]})
            continue
        kind = "combined_sampler" if typ.startswith(SAMPLER_PREFIXES) else "storage_image"
        ns = _KIND_NS[kind]
        if len(names) == 1:
            nm, arr, _c = names[0]
            set_n = SET_OF_NS[ns] if es is None else es
            if eb is not None:
                if (set_n, eb) in used_pair:
                    raise DescriptorCollision(f"explicit binding {ns}/set={set_n}/binding={eb} declared "
                                              f"twice: {used_pair[(set_n, eb)]} and {nm}")
                used_pair[(set_n, eb)] = nm
            recs.append({"s": s, "e": e, "what": "decl", "kind": kind, "split": False,
                         "name": nm, "binding": eb, "set": set_n, "has_layout": has_layout,
                         "args0": args0, "glsl_type": typ + arr})
        else:                                            # 'uniform sampler2D a, b, c;' -> one line each
            if eb is not None:
                raise GlslError("an explicit binding on a multi-declarator sampler/image declaration "
                                "cannot be shared: " + body[:80].replace("\n", " "))
            recs.append({"s": s, "e": e, "what": "decl", "kind": kind, "split": True, "typ": typ,
                         "names": names, "set": SET_OF_NS[ns], "has_layout": has_layout, "args0": args0})

    # ---- pass B: allocate the missing bindings / locations, in first-appearance order ----
    counter = {ns: 0 for ns in NAMESPACES}
    bindings, edits, info = [], [], {"mode": None, "loose_uniforms": 0, "explicit_bindings": 0,
                                     "locations_injected": 0, "samplers": 0, "images": 0,
                                     "uniform_blocks": 0, "storage_blocks": 0, "split_declarations": 0}

    def alloc(ns, name, explicit_binding, set_n):
        if explicit_binding is not None:
            info["explicit_bindings"] += 1
            return explicit_binding
        while (set_n, counter[ns]) in used_pair:      # never overwrite an explicit binding
            counter[ns] += 1
        b = counter[ns]
        counter[ns] = b + 1
        used_pair[(set_n, b)] = name
        return b

    def next_loc():
        n = 0
        while n in explicit_locs:
            n += 1
        explicit_locs.append(n)
        info["locations_injected"] += 1
        return n

    loose_first = None
    for r in recs:
        if r["what"] == "io":
            pre = _layout_prefix(r["has_layout"], r["args0"], loc=next_loc())
            if r["has_layout"]:
                at = r["s"] + src[r["s"]:r["e"]].index("layout(") + len("layout(")
                edits.append((at, at, pre))
            else:
                edits.append((r["s"], r["s"], pre))
            continue
        if r["what"] == "loose":
            if loose_first is None:
                loose_first = r["s"]
            edits.append((r["s"], r["e"], ""))
            info["loose_uniforms"] += len(r["names"])
            continue
        if r["what"] == "block":
            ns = _KIND_NS[r["kind"]]
            b = alloc(ns, r["name"], r["binding"], r["set"])
            bindings.append(Binding(r["inst"], r["kind"], b, r["glsl_type"], block=r["name"],
                                    explicit=r["binding"] is not None, set_=r["set"]))
            info["storage_blocks" if r["kind"] == "storage_block" else "uniform_blocks"] += 1
            pre = _layout_prefix(r["has_layout"], r["args0"], binding=b, set_n=r["set"])
            at = (r["s"] + src[r["s"]:r["e"]].index("layout(") + len("layout(")) if r["has_layout"] else r["s"]
            if pre:
                edits.append((at, at, pre))
            continue
        # opaque resource declaration
        ns = _KIND_NS[r["kind"]]
        if not r["split"]:
            b = alloc(ns, r["name"], r["binding"], r["set"])
            bindings.append(Binding(r["name"], r["kind"], b, r["glsl_type"],
                                    explicit=r["binding"] is not None, set_=r["set"]))
            info["samplers" if ns == "sampler" else "images"] += 1
            pre = _layout_prefix(r["has_layout"], r["args0"], binding=b, set_n=r["set"])
            at = (r["s"] + src[r["s"]:r["e"]].index("layout(") + len("layout(")) if r["has_layout"] else r["s"]
            if pre:
                edits.append((at, at, pre))
            continue
        info["split_declarations"] += 1
        raw = src[r["s"]:r["e"]]
        words, lays, tail = _strip_qualifiers(raw)
        quals = " ".join(w for w in words if w != "layout")
        parts = []
        for nm, arr, com in r["names"]:
            b = alloc(ns, nm, None, r["set"])
            bindings.append(Binding(nm, r["kind"], b, r["typ"] + arr, set_=r["set"]))
            info["samplers" if ns == "sampler" else "images"] += 1
            tail_c = ("   " + com) if com else ""
            parts.append(f"layout(set={r['set']}, binding={b}) {quals} {r['typ']} {nm}{arr};{tail_c}")
        edits.append((r["s"], r["e"], "\n".join(parts)))

    # ---- splice: the synthetic uniform block at the first loose declaration's old position ----
    block_text = ""
    if loose_first is not None:
        ub_set = SET_OF_NS["uniform_block"]
        b = alloc("uniform_block", UBLOCK_NAME, None, ub_set)
        bindings.insert(0, Binding(UBLOCK_NAME, "uniform_block", b,
                                   f"{UBLOCK_STORAGE} uniform {UBLOCK_NAME}", block=UBLOCK_NAME,
                                   set_=ub_set))
        members = [r["member"] for r in recs if r["what"] == "loose"]
        info["uniform_blocks"] = info["uniform_blocks"] + 1
        block_text = ("// port(P1): the engine's loose pass uniforms, grouped into one block for Vulkan "
                      "GLSL (unnamed instance -> members stay global)\n"
                      f"layout({UBLOCK_STORAGE}, set={ub_set}, binding={b}) uniform {UBLOCK_NAME} {{\n"
                      + "\n".join("    " + m + ";" for m in members) + "\n};\n\n")
    out, last, block_done = [], 0, not block_text
    for s, e, rep in sorted(edits, key=lambda t: (t[0], t[1])):
        out.append(src[last:s])
        if not block_done and s >= loose_first:
            out.append(block_text)
            block_done = True
        out.append(rep)
        last = e
    out.append(src[last:])
    if not block_done:
        out.append(block_text)
    info["samplers"] = sum(1 for x in bindings if x.kind == "combined_sampler")
    info["images"] = sum(1 for x in bindings if x.kind == "storage_image")
    info["uniform_blocks"] = sum(1 for x in bindings if x.kind == "uniform_block")
    info["storage_blocks"] = sum(1 for x in bindings if x.kind == "storage_block")
    return "".join(out), bindings, info


# --------------------------------------------------------------------- program build
def build_source_text(body, defines=None, strip=True, stage="frag", glsl_version=None, name=None):
    """Pass header + glstrip.reduce + the Vulkan pass from an IN-MEMORY body (shared vertex stage,
    the two passes the engine writes into its cache dir)."""
    v = glsl_version or GLSL_VERSION
    header = f"#version {v}\n" + "".join(f"#define {k} {v_}\n" for k, v_ in (defines or {}).items())
    full = header + body
    # engine/gl.py::Pass -> glstrip.reduce (resolve + dead-code strip + blank squeeze).
    # engine/erosion.py::_prog does neither for compute: it hands glslang the conditionals as written.
    # resolve() is applied either way: it only keeps the branches a preprocessor would take, which is
    # what makes the two #if branches of erosion_accum.comp (each declaring imgAcc at binding 0) one
    # declaration for the scanner. Dead-code stripping stays off where the engine keeps it off.
    mode = "reduced"
    try:
        src = _glstrip.resolve(full)
    except ValueError as exc:
        mode, src, strip_err = "unresolved", full, str(exc)
    if strip and mode == "reduced":
        try:
            src = _glstrip.strip_unreachable(src)
            src = re.sub(r"\n[ \t]*(?:\n[ \t]*){2,}", "\n\n", src)     # reduce()'s squeeze
        except ValueError as exc:
            mode, src, strip_err = "full", full, str(exc)
    if not strip and mode == "reduced":
        mode = "resolved"
    src, bindings, info = vulkanize(src)
    info["mode"] = mode
    if mode == "full":
        info["strip_error"] = strip_err
    info["stage"] = stage
    info["defines"] = dict(defines or {})
    info["frag"] = str(name) if name else None
    return src, {"bindings": bindings, **info}


def build_source(frag, defines=None, strip=True, shaders_dir=None, glsl_version=None):
    """engine/gl.py::Pass pipeline for one fragment/compute program -> (source, info)."""
    return build_source_text(preprocess(frag, set(), shaders_dir), defines=defines, strip=strip,
                             stage=(Path(frag).suffix.lstrip(".") or "frag"),
                             glsl_version=glsl_version, name=frag)

if __name__ == "__main__":
    src, m = build_source("bake_height.frag")
    print(src[:1800])
    for b in m["bindings"]:
        print("  ", b)
    print({k: v for k, v in m.items() if k != "bindings"})