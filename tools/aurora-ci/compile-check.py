#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Cheap pull-request compile check for an arm64 kernel tree.

For the range BASE..HEAD this script:

* compiles, with W=1, the kbuild objects of the C/assembly files the range
  changes (plus a capped set of objects that include changed headers) at the
  merge base and at HEAD, and reports W=1 warnings present at HEAD but not at
  the merge base ("new" warnings);
* with --per-commit, also compiles the same object set at every non-merge
  commit of the range, in order, in one incremental build directory, and
  reports commits that do not compile on their own.

Objects are compiled only when the configuration selects them.  Selection is
read from kbuild itself (an injected makefile prints each visited directory's
object lists), because `make path/file.o` compiles an unselected file anyway,
with the wrong flags.

Output: GitHub workflow annotations on stdout, report.json and summary.md in
--out.  Exit status: 0 = no compile errors (warnings allowed), 1 = compile
errors introduced by the range, 2 = the check itself could not run.

Python 3.10+ standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

LISTER_MK = r"""
# Injected into kbuild sub-makes through MAKEFILES (see compile-check.py).
ifdef obj
$(obj)/: __aurora_list_objs
PHONY += __aurora_list_objs
__aurora_list_objs:
	@$(if $(filter %/scripts/Makefile.build,$(MAKEFILE_LIST)),printf '%s\n' 'AURORA-OBJS|$(obj)|$(strip $(real-obj-y))|$(strip $(real-obj-m))|$(strip $(lib-y))|$(strip $(filter %.o,$(targets)))',:)
endif
"""

SOURCE_EXTS = (".c", ".S")
HEADER_EXTS = (".h",)
BUILD_FILE_RE = re.compile(r"(^|/)(Makefile[^/]*|Kbuild[^/]*|Kconfig[^/]*)$")

QUIET_RE = re.compile(r"^  ([A-Z][A-Z0-9_]*(?: \[M\])?) +(\S+)\s*$")
DIAG_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+): "
    r"(?P<sev>warning|error|fatal error): (?P<msg>.*)$")
TOOL_DIAG_RE = re.compile(r"^(?:\S*/)?(?:clang|clang-\d+|ld\.lld|ld\.lld-\d+|ld): "
                          r"(?P<sev>error|warning): (?P<msg>.*)$")
FLAG_RE = re.compile(r"\s\[(-W[^\]]+)\]$")
# W=1 also runs scripts/kernel-doc on every compiled C file.
KDOC_RE = re.compile(r"^(?:Warning|Error): (?P<file>[^\s:][^:]*):(?P<line>\d+) (?P<msg>.*)$")
MAKE_ERR_RE = re.compile(
    r"^make(?:\[\d+\])?: \*\*\* \[(?P<mk>[^\]]*?):(?P<ln>\d+): (?P<target>[^\]]+)\] Error \d+")
NO_RULE_RE = re.compile(r"No rule to make target '(?P<t>[^']+)'(?:, needed by '(?P<by>[^']+)')?")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class SetupError(Exception):
    pass


def run(cmd, cwd=None, env=None, logfile: Path | None = None, check=False):
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    dt = time.monotonic() - t0
    if logfile is not None:
        with logfile.open("a") as f:
            f.write(f"$ {' '.join(map(str, cmd))}\n{p.stdout}\n[rc={p.returncode} {dt:.1f}s]\n")
    if check and p.returncode:
        raise SetupError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n"
                         + p.stdout[-4000:])
    return p.returncode, p.stdout, dt


class Git:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, *args, check=True) -> str:
        p = subprocess.run(["git", "-C", str(self.path), *args], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, errors="replace")
        if check and p.returncode:
            raise SetupError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
        return p.stdout

    def rev(self, r: str) -> str:
        return self("rev-parse", "--verify", "--quiet", r + "^{commit}").strip()


# --------------------------------------------------------------------------
# Change sets

def name_status(git: Git, a: str, b: str):
    """Return (changed paths on the b side, deleted paths, renames old->new)."""
    out = git("diff", "--no-ext-diff", "--name-status", "-M", "-z", a, b)
    parts = out.split("\0")
    changed, deleted, renames = [], [], {}
    i = 0
    while i < len(parts) - 1:
        st = parts[i]
        if not st:
            i += 1
            continue
        if st[0] in "RC":
            old, new = parts[i + 1], parts[i + 2]
            changed.append(new)
            if st[0] == "R":
                renames[old] = new
            i += 3
        else:
            path = parts[i + 1]
            (deleted if st[0] == "D" else changed).append(path)
            i += 2
    return changed, deleted, renames


def added_lines(git: Git, a: str, b: str) -> dict[str, set[int]]:
    out = git("diff", "--no-ext-diff", "--no-color", "-U0", "-M", a, b, "--",
              "*.c", "*.h", "*.S")
    res: dict[str, set[int]] = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
            if cur is not None:
                res.setdefault(cur, set())
        elif line.startswith("@@") and cur is not None:
            m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if m:
                start, count = int(m.group(1)), int(m.group(2) or 1)
                res[cur].update(range(start, start + count))
    return res


# --------------------------------------------------------------------------
# Header includers: a cheap include graph built from the source text.
#
# Fixed-string `git grep`s for header basenames over the checked-out tree
# (well under 1 s each for the kernel) find candidate #include lines.  Each
# spelling is resolved the way the kernel's include paths would resolve it:
# relative to the including file for "..." includes, then include/,
# arch/arm64/include/ and their uapi/ trees, asm/ falling back to
# asm-generic/, and finally a same-subsystem file with a matching path suffix
# (for directories added with ccflags -I).  Includes spelled through macros
# and headers forced with -include are not seen.

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
SYS_DIRS = ("include/", "arch/arm64/include/", "include/uapi/", "arch/arm64/include/uapi/")


class IncludeIndex:
    """Reverse include lookups, filled lazily one batch of basenames at a time.

    Each batch is one fixed-string `git grep` for the header basenames over
    the checked-out tree (about 1 s for the kernel), whose hits are parsed as
    #include lines.  Files the range changes are read at HEAD instead.
    """

    def __init__(self, git: Git, head: str, changed: list[str]):
        self.git, self.head = git, head
        self.changed = [c for c in changed if c.endswith((".c", ".h", ".S"))]
        self.headers = set(git("ls-files", "--", "*.h", "*.c", check=False).split())
        self.headers.update(self.changed)
        self.by_base: dict[str, list[tuple[str, str, bool]]] = {}
        self.done: set[str] = set()
        hb: dict[str, list[str]] = {}
        for h in self.headers:
            hb.setdefault(os.path.basename(h), []).append(h)
        self._hb = hb

    def _load(self, bases):
        todo = sorted(set(bases) - self.done)
        if not todo:
            return
        self.done.update(todo)
        pats = []
        for b in todo:
            pats += ["-e", b]
        changed = set(self.changed)

        def add(f, text):
            m = INCLUDE_RE.match(text)
            if m and os.path.basename(m.group(2)) in todo_set:
                self.by_base.setdefault(os.path.basename(m.group(2)), []).append(
                    (f, m.group(2), m.group(1) == '"'))

        todo_set = set(todo)
        out = self.git("grep", "-n", "-F", *pats, "--", "*.c", "*.h", "*.S", check=False)
        for line in out.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] not in changed:
                add(parts[0], parts[2])
        if self.changed:
            out = self.git("grep", "-n", "-F", *pats, self.head, "--", *self.changed,
                           check=False)
            for line in out.splitlines():
                parts = line.split(":", 3)
                if len(parts) == 4:
                    add(parts[1], parts[3])

    def resolve(self, includer: str, spell: str, quoted: bool) -> list[str]:
        known = self.headers
        if quoted:
            c = os.path.normpath(os.path.join(os.path.dirname(includer), spell))
            if c in known:
                return [c]
        for d in SYS_DIRS:
            if d + spell in known:
                return [d + spell]
        if spell.startswith("asm/"):
            g = "include/asm-generic/" + spell[4:]
            if g in known:
                return [g]
        tail = "/" + spell.lstrip("./")
        return [h for h in self._hb.get(os.path.basename(spell), [])
                if h.endswith(tail) and common_prefix(h, includer) >= 2]

    def includers_many(self, hs) -> dict[str, list[str]]:
        hs = list(hs)
        self._load(os.path.basename(h) for h in hs)
        res = {}
        for h in hs:
            r = set()
            for f, spell, quoted in self.by_base.get(os.path.basename(h), []):
                if f != h and h in self.resolve(f, spell, quoted):
                    r.add(f)
            res[h] = sorted(r)
        return res

    def includers(self, h: str) -> list[str]:
        return self.includers_many([h])[h]


C_WORDS = set("""auto break case char const continue default define defined do double else
elif endif enum error extern float for goto if ifdef ifndef include inline int long register
restrict return short signed sizeof static struct switch typedef undef union unsigned void
volatile while bool true false NULL u8 u16 u32 u64 s8 s16 s32 s64 size_t ssize_t dma_addr_t
__iomem __user __init __exit __must_check EXPORT_SYMBOL EXPORT_SYMBOL_GPL""".split())


def changed_identifiers(git: Git, ranges: list[tuple[str, str]], path: str) -> set[str]:
    """Identifiers on lines a range adds to or removes from `path`."""
    ids: set[str] = set()
    for a, b in ranges:
        out = git("diff", "--no-ext-diff", "--no-color", "-U0", a, b, "--", path, check=False)
        for line in out.splitlines():
            if line[:1] in "+-" and not line.startswith(("+++", "---")):
                body = re.sub(r"/\*.*?\*/|//.*$", "", line[1:])
                ids.update(w for w in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", body)
                           if w not in C_WORDS)
    return ids


def owner_dirs(h: str) -> list[str]:
    """The header's directory and the directory that likely implements it."""
    d = os.path.dirname(h)
    res = [h]
    for pat, rep in ((r"^include/(?:uapi/)?linux/(.+)$", r"drivers/\1"),
                     (r"^include/(?:uapi/)?drm(/.*)?$", r"drivers/gpu/drm\1"),
                     (r"^include/(?:uapi/)?sound(/.*)?$", r"sound\1"),
                     (r"^arch/([^/]+)/include/(?:uapi/)?asm(/.*)?$", r"arch/\1\2")):
        m = re.match(pat, d)
        if m:
            res.append(m.expand(rep) + "/" + os.path.splitext(os.path.basename(h))[0])
    return res


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        n += 1
    return n


# --------------------------------------------------------------------------
# Make output parsing

def norm_path(p: str, srcdir: str, builddir: str) -> str:
    if p.startswith("/"):
        p = os.path.normpath(p)          # e.g. drivers/x/epic/../y.h
    if p.startswith(srcdir + "/"):
        return p[len(srcdir) + 1:]
    if p.startswith(builddir + "/"):
        return "<build>/" + p[len(builddir) + 1:]
    if not p.startswith("/"):
        q = os.path.normpath(p)
        return "<build>/" + q
    return p


def parse_make_output(text: str, srcdir: str, builddir: str):
    diags, failed, misc_errors = [], [], []
    obj = None
    cur = None
    include_chain: list[str] = []
    for line in text.splitlines():
        m = QUIET_RE.match(line)
        if m and not line.startswith("  In file"):
            obj = m.group(2)
            cur = None
            include_chain = []
            continue
        if line.startswith("In file included from "):
            include_chain.append(line[len("In file included from "):].rstrip(":"))
            continue
        m = DIAG_RE.match(line)
        if m:
            msg = m.group("msg")
            flags = []
            fm = FLAG_RE.search(msg)
            if fm:
                flags = [f for f in fm.group(1).split(",") if f != "-Werror"]
                msg = msg[: fm.start()]
            sev = "error" if m.group("sev") in ("error", "fatal error") else "warning"
            cur = {
                "file": norm_path(m.group("file"), srcdir, builddir),
                "line": int(m.group("line")), "col": int(m.group("col")),
                "severity": sev, "flag": flags[0] if flags else "",
                "message": msg.strip(), "object": obj, "context": [],
            }
            diags.append(cur)
            include_chain = []
            continue
        m = KDOC_RE.match(line)
        if m:
            cur = None
            diags.append({"file": norm_path(m.group("file"), srcdir, builddir),
                          "line": int(m.group("line")), "col": 0, "severity": "warning",
                          "flag": "kernel-doc", "message": m.group("msg").strip(),
                          "object": obj, "context": []})
            continue
        m = TOOL_DIAG_RE.match(line)
        if m:
            diags.append({"file": "", "line": 0, "col": 0, "severity": m.group("sev"),
                          "flag": "", "message": m.group("msg").strip(), "object": obj,
                          "context": []})
            cur = None
            continue
        m = MAKE_ERR_RE.match(line)
        if m:
            t = m.group("target")
            if os.path.splitext(t)[1]:          # a file target, not a directory
                failed.append(t)
            cur = None
            continue
        m = NO_RULE_RE.search(line)
        if m and line.startswith("make"):
            misc_errors.append(line.strip())
            continue
        if cur is not None and len(cur["context"]) < 8 and not line.startswith("make"):
            cur["context"].append(line)
    return diags, sorted(set(failed)), misc_errors


def count_cmds(out: str) -> int:
    """Number of kbuild commands (quiet "  TAG  target" lines) in make output."""
    return sum(1 for l in out.splitlines() if QUIET_RE.match(l))


def norm_msg(msg: str) -> str:
    return re.sub(r"\b\d+\b", "N", msg)


# --------------------------------------------------------------------------
# The checker

class Checker:
    def __init__(self, a):
        self.a = a
        self.tree = Path(a.tree).resolve()
        self.out = Path(a.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.logdir = self.out / "logs"
        self.logdir.mkdir(exist_ok=True)
        self.build = self.out / "build"
        self.build.mkdir(exist_ok=True)
        self.git_main = Git(self.tree)
        self.timings: dict[str, float] = {}
        self.cpu: dict[str, float] = {}
        self.notes: list[str] = []
        self.unchecked: list[dict] = []
        self.lister = self.out / "kbuild-list-objs.mk"
        self.lister.write_text(LISTER_MK)
        self.env = dict(os.environ)
        self.env["LC_ALL"] = "C"
        self.env.pop("MAKEFILES", None)
        self.env.pop("MAKEFLAGS", None)
        self.selected: dict[str, str] = {}
        self.included_c: dict[str, list[str]] = {}
        self.obj_diags: dict[str, list[dict]] = {}
        self.selected_src: dict[str, str] = {}
        self.cur_sha = None

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def mark() -> tuple[float, float]:
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.monotonic(), ru.ru_utime + ru.ru_stime

    def tick(self, key: str, start: tuple[float, float]) -> None:
        """Add wall and child-CPU seconds since `start` to phase `key`."""
        wall, cpu = self.mark()
        self.timings[key] = round(self.timings.get(key, 0.0) + wall - start[0], 2)
        self.cpu[key] = round(self.cpu.get(key, 0.0) + cpu - start[1], 2)

    def make_cmd(self, *extra):
        a = self.a
        cmd = ["make", f"O={self.build}", f"LLVM={a.llvm}", "ARCH=arm64", "W=1",
               f"-j{a.jobs}"]
        if a.rust == "skip":
            # Keep CONFIG_RUST=y (and every C driver that depends on it) in the
            # configuration, but do not build rust/ or run its prepare step.
            cmd += ["CONFIG_RUST=", "RUSTC=/nonexistent/rustc", "BINDGEN=/nonexistent/bindgen"]
        cmd += a.make_arg
        cmd += list(extra)
        return cmd

    # -- source tree --------------------------------------------------------
    def setup_tree(self):
        g = self.git_main
        if not self.a.worktree:
            # In place (the CI checkout): states are checked out in --tree and
            # its original HEAD (branch or detached commit) is restored at exit.
            if g("status", "--porcelain", "--untracked-files=no").strip():
                raise SetupError("--tree has local changes; commit or stash them, "
                                 "or use --worktree DIR")
            self.src = self.tree
            branch = g("symbolic-ref", "-q", "--short", "HEAD", check=False).strip()
            self.orig_head = branch or g("rev-parse", "HEAD").strip()
        else:
            self.src = Path(self.a.worktree).resolve()
            if not (self.src / ".git").exists():
                t0 = self.mark()
                g("worktree", "add", "--detach", "-f", str(self.src), self.mb)
                self.tick("worktree_add", t0)
            self.orig_head = None
        self.git = Git(self.src)

    def checkout(self, sha: str):
        if self.cur_sha == sha:
            return
        t0 = self.mark()
        self.git("checkout", "-q", "--detach", "-f", sha)
        self.tick("checkout", t0)
        self.cur_sha = sha

    # -- configuration ---------------------------------------------------------
    def configure(self, label: str):
        t0 = self.mark()
        lf = self.logdir / f"config-{label}.log"
        cfg = self.build / ".config"
        shutil.copyfile(self.a.config, cfg)
        if self.a.fragment:
            frags = [str(self.src / f) for f in self.a.fragment if (self.src / f).exists()]
            if frags:
                run(["scripts/kconfig/merge_config.sh", "-m", "-O", str(self.build), str(cfg),
                     *frags], cwd=self.src, env=self.env, logfile=lf, check=True)
        sc = ["scripts/config", "--file", str(cfg), "--disable", "WERROR"]
        if not self.a.keep_debug_info:
            sc += ["--enable", "DEBUG_INFO_NONE", "--disable", "DEBUG_INFO_DWARF5",
                   "--disable", "DEBUG_INFO_DWARF4",
                   "--disable", "DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT"]
        run(sc, cwd=self.src, env=self.env, logfile=lf, check=True)
        want_rust = "CONFIG_RUST=y" in cfg.read_text().splitlines()
        run(self.make_cmd("olddefconfig"), cwd=self.src, env=self.env, logfile=lf, check=True)
        have_rust = "CONFIG_RUST=y" in cfg.read_text().splitlines()
        if want_rust and not have_rust:
            self.notes.append("CONFIG_RUST was dropped by olddefconfig: C objects that depend "
                              "on Rust drivers are not covered.")
        self.frag_ids = self.fragment_ids()
        self.tick("config", t0)

    def fragment_ids(self):
        ids = []
        for f in self.a.fragment:
            ids.append(self.git("rev-parse", f"HEAD:{f}", check=False).strip())
        return ids

    # -- object selection ------------------------------------------------------
    def list_objects(self, label: str) -> tuple[bool, str]:
        """Run prepare + the kbuild walk; fill self.selected. Returns (ok, output)."""
        t0 = self.mark()
        env = dict(self.env)
        env["MAKEFILES"] = str(self.lister)
        rc, out, _ = run(self.make_cmd("-k", "KBUILD_BUILTIN=", "KBUILD_MODULES=", "."),
                         cwd=self.src, env=env, logfile=self.logdir / f"list-{label}.log")
        self.tick("prepare_and_list", t0)
        sel: dict[str, str] = {}
        for line in out.splitlines():
            if not line.startswith("AURORA-OBJS|"):
                continue
            f = line.rstrip("\n").split("|")
            if len(f) != 6:
                continue
            # Objects linked into the kernel or a module first, then other
            # objects this directory builds (e.g. X.o behind X.pi.o).
            for kind, lst in (("y", f[2]), ("m", f[3]), ("lib", f[4]), ("target", f[5])):
                for o in lst.split():
                    if o.endswith(".o") and o not in sel:
                        sel[o] = kind
        self.selected = sel
        self.selected_src = {}
        for o, kind in sel.items():
            s = self.source_of(o)
            if s and s not in self.selected_src:
                self.selected_src[s] = o
        ok = rc == 0 and bool(sel)
        return ok, out

    def source_of(self, o: str) -> str | None:
        """Source file of object o: X.o <- X.c/X.S, or X.<tag>.o <- X.c/X.S."""
        b = os.path.normpath(o[:-2])
        cands = [b]
        stem, dot, tag = b.rpartition(".")
        if dot and "/" not in tag:
            cands.append(stem)
        for c in cands:
            for ext in SOURCE_EXTS:
                if (self.src / (c + ext)).is_file():
                    return c + ext
        return None

    # -- compile -------------------------------------------------------------------
    def compile(self, objs: list[str], label: str):
        if not objs:
            return {"rc": 0, "loose_diags": [], "failed": [], "misc_errors": [], "seconds": 0.0,
                    "recompiled": 0}
        lf = self.logdir / f"build-{label}.log"
        t0 = self.mark()
        rc, out, dt = run(self.make_cmd("-k", "-Otarget", *objs), cwd=self.src,
                          env=self.env, logfile=lf)
        self.tick("compile", t0)
        diags, failed, misc = parse_make_output(out, str(self.src), str(self.build))
        compiled = sorted({m.group(2) for l in out.splitlines()
                           if (m := QUIET_RE.match(l)) and m.group(1).split()[0] in ("CC", "AS")}
                          | set(failed))
        # make only recompiles objects whose inputs changed, so an object's
        # diagnostics are those of its most recent compile in this run.
        for t in compiled:
            self.obj_diags[t] = [d for d in diags if d.get("object") == t]
        loose = [d for d in diags if d.get("object") not in self.obj_diags]
        return {"rc": rc, "loose_diags": loose, "failed": failed, "misc_errors": misc,
                "seconds": round(dt, 2), "recompiled": len(compiled)}

    def effective_diags(self, objs: list[str], loose: list[dict]) -> list[dict]:
        """Diagnostics of each object's latest compile (it may predate this state)."""
        out = list(loose)
        for o in objs:
            keys = [o]
            s = self.source_of(o)
            if s and os.path.splitext(s)[0] + ".o" != o:
                keys.append(os.path.splitext(s)[0] + ".o")     # X.o behind X.pi.o
            for k in keys:
                out.extend(self.obj_diags.get(k, []))
        return out

    # -- main flow ----------------------------------------------------------------
    def execute(self):
        a = self.a
        g = self.git_main
        self.base = g.rev(a.base)
        self.head = g.rev(a.head)
        if not self.base or not self.head:
            raise SetupError(f"cannot resolve {a.base} / {a.head} in {self.tree}")
        self.mb = g("merge-base", self.base, self.head).strip()
        if not self.mb:
            raise SetupError("no merge base")
        commits = g("rev-list", "--reverse", "--topo-order", "--no-merges",
                    f"{self.base}..{self.head}").split()
        n_merges = len(g("rev-list", "--merges", f"{self.base}..{self.head}").split())
        per_commit = a.per_commit and len(commits) > 0
        if per_commit and len(commits) > a.max_commits:
            self.notes.append(f"Per-commit check skipped: {len(commits)} commits exceed "
                              f"--max-commits {a.max_commits}; only HEAD was compiled.")
            per_commit = False
        if n_merges:
            self.notes.append(f"{n_merges} merge commit(s) in the range are not compiled on "
                              "their own.")

        # Changed files: whole range, and per commit.
        changed, deleted, renames = name_status(g, self.mb, self.head)
        per_commit_changes = {}
        if per_commit:
            for c in commits:
                ch, _, _ = name_status(g, c + "^", c)
                per_commit_changes[c] = ch
        all_changed = set(changed)
        for ch in per_commit_changes.values():
            all_changed.update(ch)

        sources = sorted(p for p in all_changed if p.endswith(SOURCE_EXTS))
        headers = sorted(p for p in all_changed if p.endswith(HEADER_EXTS))
        rust_files = sorted(p for p in all_changed if p.endswith(".rs"))
        build_files = sorted(p for p in all_changed if BUILD_FILE_RE.search(p))

        self.setup_tree()
        states = [("base", self.mb)]
        if per_commit:
            states += [(f"commit-{i + 1}", c) for i, c in enumerate(commits)]
            if commits[-1] != self.head:
                states.append(("head", self.head))
        else:
            states.append(("head", self.head))

        # First state: configure + list, then decide the object set.
        self.checkout(self.mb)
        self.configure("base")
        ok, out = self.list_objects("base")
        base_prepare_cmds = count_cmds(out)
        if not ok:
            raise SetupError("kbuild walk failed at the merge base; see logs/list-base.log\n"
                             + out[-3000:])

        ranges = [(self.mb, self.head)] + [(c + "^", c) for c in per_commit_changes]
        object_sources, header_cov = self.pick_sources(sources, headers, all_changed, ranges)
        if len(object_sources) > a.max_objects:
            self.notes.append(f"Object set capped at {a.max_objects} of "
                              f"{len(object_sources)} candidates.")
            object_sources = object_sources[: a.max_objects]

        for p in rust_files:
            self.unchecked.append({"path": p, "reason": "Rust source (not compiled by this job)"})

        # Every object must be compiled at least once in this run so that its
        # diagnostics are known (a reused build directory may hold it already).
        for src in object_sources:
            for o in {self.selected_src.get(src), os.path.splitext(src)[0] + ".o"}:
                if o:
                    (self.build / o).unlink(missing_ok=True)

        results = []
        prev_sha = None
        for label, sha in states:
            st = {"label": label, "sha": sha}
            if label == "base":
                st["prepare_cmds"] = base_prepare_cmds
            if sha != self.cur_sha:
                prev_sha = self.cur_sha
                self.checkout(sha)
                diff_names = set(g("diff", "--name-only", prev_sha, sha).split())
                if self.a.fragment and self.fragment_ids() != self.frag_ids:
                    self.configure(label)
                    relist = True
                else:
                    relist = any(BUILD_FILE_RE.search(p) for p in diff_names)
                if relist:
                    ok, out = self.list_objects(label)
                else:
                    ok, out = self.prepare(label)
                st["prepare_cmds"] = count_cmds(out)
                st["relisted"] = relist
                if not ok:
                    st["prepare_failed"] = True
                    st["prepare_log_tail"] = out[-3000:]
                    pd, _, _ = parse_make_output(out, str(self.src), str(self.build))
                    st["diags"] = [d for d in pd if d["severity"] == "error"]
                    results.append(st)
                    continue
            objs = []
            for s in object_sources:
                o = self.selected_src.get(s)
                if o:
                    objs.append(o)
            st["objects"] = objs
            log(f"[{label}] {sha[:12]}: compiling {len(objs)} object(s)")
            r = self.compile(objs, label)
            st.update(r)
            st["diags"] = self.effective_diags(objs, r.pop("loose_diags"))
            results.append(st)

        # Sources never selected at HEAD (or at any state) are reported.
        head_sel = set(self.selected_src)
        for s, via in self.included_c.items():
            self.notes.append(f"{s} is not an object itself; it is compiled through "
                              f"{', '.join(via)}, which #include it.")
        for s in sources:
            if s not in head_sel and s not in self.included_c and (self.src / s).exists():
                if s.startswith(("tools/", "scripts/", "samples/", "Documentation/", "usr/")):
                    why = "outside kernel objects (host tool, sample or documentation)"
                else:
                    why = "not built by this configuration"
                self.unchecked.append({"path": s, "reason": why})

        return self.analyse(results, commits, renames, header_cov, build_files, per_commit)

    def prepare(self, label: str):
        t0 = self.mark()
        rc, out, _ = run(self.make_cmd("-k", "prepare"), cwd=self.src, env=self.env,
                         logfile=self.logdir / f"prepare-{label}.log")
        self.tick("prepare_and_list", t0)
        return rc == 0, out

    def pick_sources(self, sources, headers, all_changed, ranges):
        """Changed sources plus a capped, ranked set of header includers.

        Includers that use identifiers the range adds to or removes from the
        header (a renamed function, a changed macro or field) come first,
        weighted by how rare each identifier is among the includers; then
        files the range changes, then files near the header or its likely
        implementation directory.
        """
        chosen = [s for s in sources]
        header_cov = []
        selected = self.selected_src
        changed_set = set(all_changed)
        if headers or any(s not in selected for s in sources):
            t0 = self.mark()
            idx = IncludeIndex(self.git, self.head, sorted(changed_set))
            self.tick("include_index", t0)
        # A changed .c that is not an object itself may be #included by one.
        for s in sources:
            if s in selected or not s.endswith(".c"):
                continue
            via = [inc for inc in idx.includers(s) if inc in selected]
            via.sort(key=lambda c: (0 if c in changed_set else 1, -common_prefix(c, s), c))
            via = via[: self.a.header_objects]
            if via:
                self.included_c[s] = via
                for inc in via:
                    if inc not in chosen:
                        chosen.append(inc)
        for h in headers:
            if h.startswith(("tools/", "scripts/", "Documentation/", "samples/")):
                header_cov.append({"header": h, "includers_found": 0, "selected": 0,
                                   "compiled": 0, "note": "outside kernel objects"})
                continue
            seen, frontier, cands = {h}, [h], []
            depth = 0
            while frontier and depth < self.a.header_depth:
                nxt = []
                level = idx.includers_many(frontier)
                for x in frontier:
                    for inc in level[x]:
                        if inc in seen:
                            continue
                        seen.add(inc)
                        if inc.endswith(SOURCE_EXTS):
                            cands.append((depth, inc))
                        # Follow headers, and .c files that are #included
                        # by other files rather than compiled themselves.
                        if ((inc.endswith(".h") or inc not in selected)
                                and len(nxt) < self.a.header_fanout):
                            nxt.append(inc)
                frontier = nxt
                depth += 1
            sel = [(d, c) for d, c in cands if c in selected]
            ref = owner_dirs(h) + sorted(p for p in changed_set if p.endswith(SOURCE_EXTS))
            score = self.identifier_scores(h, [c for _, c in sel], ranges)

            def rank(item):
                d, c = item
                prox = max(common_prefix(c, r) for r in ref)
                return (-score.get(c, 0.0), 0 if c in changed_set else 1, -prox, d, c)

            sel.sort(key=rank)
            # Users of changed identifiers first; then the best-ranked file of
            # each directory, so the sample covers different users.
            pick, dirs = [], set()
            for _, c in sel:
                if len(pick) >= self.a.header_objects:
                    break
                if score.get(c) or os.path.dirname(c) not in dirs:
                    pick.append(c)
                    dirs.add(os.path.dirname(c))
            for _, c in sel:
                if len(pick) >= self.a.header_objects:
                    break
                if c not in pick:
                    pick.append(c)
            for c in pick:
                if c not in chosen:
                    chosen.append(c)
            header_cov.append({"header": h, "includers_found": len(cands),
                               "selected": len(sel), "compiled": len(pick),
                               "users_of_changed_identifiers": sum(1 for c in score if score[c]),
                               "objects": pick})
        return chosen, header_cov

    def identifier_scores(self, h, cands, ranges) -> dict[str, float]:
        """Score each candidate by the changed identifiers of `h` it uses."""
        ids = changed_identifiers(self.git_main, ranges, h)
        if not ids or not cands:
            return {}
        pats = []
        for i in sorted(ids):
            pats += ["-e", i]
        uses: dict[str, set[str]] = {}
        for k in range(0, len(cands), 500):
            out = self.git("grep", "-o", "-w", "-F", *pats, "--", *cands[k:k + 500],
                           check=False)
            for line in out.splitlines():
                f, _, w = line.partition(":")
                uses.setdefault(f, set()).add(w)
        df: Counter = Counter()
        for ws in uses.values():
            df.update(ws)
        return {f: sum(1.0 / df[w] for w in ws) for f, ws in uses.items()}

    # -- analysis -----------------------------------------------------------------
    def analyse(self, results, commits, renames, header_cov, build_files, per_commit):
        base = results[0]
        head = results[-1]
        added = added_lines(self.git_main, self.mb, self.head)

        def uniq(diags, sev):
            seen, out = set(), []
            for d in diags:
                if d["severity"] != sev:
                    continue
                k = (d["file"], d["line"], d["col"], d["flag"], d["message"])
                if k in seen:
                    continue
                seen.add(k)
                out.append(d)
            return out

        def nkey(d, mapping=None):
            f = d["file"]
            if mapping:
                f = mapping.get(f, f)
            return (f, d["flag"], norm_msg(d["message"]))

        base_w = Counter(nkey(d, renames) for d in uniq(base.get("diags", []), "warning"))
        head_w = uniq(head.get("diags", []), "warning")
        remaining = Counter(base_w)
        new_w, old_w = [], []
        for d in head_w:
            k = nkey(d)
            on_added = d["line"] in added.get(d["file"], set())
            d2 = dict(d, on_added_line=on_added)
            if remaining[k] > 0:
                remaining[k] -= 1
                d2["new"] = False
                old_w.append(d2)
            else:
                d2["new"] = True
                new_w.append(d2)

        base_err = Counter(nkey(d, renames) for d in uniq(base.get("diags", []), "error"))
        base_failed = set(base.get("failed", []))
        errors = []
        for st in results[1:]:
            errs = uniq(st.get("diags", []), "error")
            st_errs = []
            for d in errs:
                if base_err.get(nkey(d), 0) > 0 and d.get("object") in base_failed:
                    continue        # already broken at the merge base
                st_errs.append(d)
            failed = [o for o in st.get("failed", []) if o not in base_failed]
            if st.get("prepare_failed") or st_errs or failed or st.get("misc_errors"):
                errors.append({"label": st["label"], "sha": st["sha"],
                               "subject": self.git_main("log", "-1", "--format=%s",
                                                        st["sha"]).strip(),
                               "prepare_failed": bool(st.get("prepare_failed")),
                               "diags": st_errs, "failed_objects": failed,
                               "misc_errors": st.get("misc_errors", []),
                               "log_tail": st.get("prepare_log_tail", "")})

        report = {
            "base": self.base, "head": self.head, "merge_base": self.mb,
            "commits": commits, "per_commit": per_commit,
            "states": [{k: v for k, v in st.items() if k not in ("diags", "prepare_log_tail")}
                       | {"warnings": len(uniq(st.get("diags", []), "warning")),
                          "errors": len(uniq(st.get("diags", []), "error"))}
                       for st in results],
            "new_warnings": new_w,
            "preexisting_warnings": old_w,
            "errors": errors,
            "header_coverage": header_cov,
            "unchecked": self.unchecked,
            "build_files_changed": build_files,
            "notes": self.notes,
            "timings": self.timings,
            "cpu_seconds": self.cpu,
            "base_errors": len(uniq(base.get("diags", []), "error")),
        }
        return report


# --------------------------------------------------------------------------
# Output

def esc_data(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def esc_prop(s: str) -> str:
    return esc_data(s).replace(":", "%3A").replace(",", "%2C")


def annotate(kind: str, d: dict, title: str) -> str:
    props = []
    if d.get("file") and not d["file"].startswith("<build>"):
        props.append(f"file={esc_prop(d['file'])}")
        if d.get("line"):
            props.append(f"line={d['line']}")
        if d.get("col"):
            props.append(f"col={d['col']}")
    props.append(f"title={esc_prop(title)}")
    msg = d["message"] + (f" [{d['flag']}]" if d.get("flag") else "")
    return f"::{kind} {','.join(props)}::{esc_data(msg)}"


def emit(report: dict, out: Path, total: float) -> int:
    lines = []
    for e in report["errors"]:
        t = f"Compile error at {e['label']} {e['sha'][:12]}"
        for d in e["diags"]:
            lines.append(annotate("error", d, t))
        if e["prepare_failed"] and not e["diags"]:
            lines.append(f"::error title={esc_prop(t)}::{esc_data('make prepare failed')}")
        for o in e["failed_objects"]:
            if not any(d.get("object") == o for d in e["diags"]):
                lines.append(f"::error title={esc_prop(t)}::{esc_data('failed to build ' + o)}")
        for m in e["misc_errors"]:
            lines.append(f"::error title={esc_prop(t)}::{esc_data(m)}")
    for d in report["new_warnings"]:
        lines.append(annotate("warning", d, "New W=1 warning"))
    for u in report["unchecked"]:
        lines.append(f"::notice file={esc_prop(u['path'])},title=Not compiled::"
                     f"{esc_data(u['reason'])}")
    for n in report["notes"]:
        lines.append(f"::notice title=compile-check::{esc_data(n)}")
    seen = set()
    for l in lines:
        if l not in seen:
            seen.add(l)
            print(l)

    report["timings"]["total"] = round(total, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=list) + "\n")

    md = []
    status = "FAIL (compile errors)" if report["errors"] else "pass"
    md.append(f"## Kernel compile check: {status}\n")
    md.append(f"Range `{report['merge_base'][:12]}..{report['head'][:12]}`, "
              f"{len(report['commits'])} commit(s), per-commit "
              f"{'on' if report['per_commit'] else 'off'}.\n")
    md.append("| state | commit | objects | recompiled | prepare cmds | W=1 warnings | errors "
              "| compile s |")
    md.append("|---|---|---|---|---|---|---|---|")
    for st in report["states"]:
        pf = " (prepare failed)" if st.get("prepare_failed") else ""
        md.append(f"| {st['label']}{pf} | `{st['sha'][:12]}` | {len(st.get('objects', []))} | "
                  f"{st.get('recompiled', 0)} | {st.get('prepare_cmds', 0)} | "
                  f"{st['warnings']} | {st['errors']} | {st.get('seconds', 0)} |")
    md.append("")
    if report["errors"]:
        md.append("### Compile errors\n")
        for e in report["errors"]:
            md.append(f"- **{e['label']}** `{e['sha'][:12]}` {e['subject']}")
            for d in e["diags"][:20]:
                md.append(f"  - `{d['file']}:{d['line']}:{d['col']}` {d['message']}")
            for o in e["failed_objects"]:
                md.append(f"  - failed: `{o}`")
            if e["prepare_failed"]:
                md.append("  - `make prepare` failed")
        md.append("")
    md.append(f"### New W=1 warnings: {len(report['new_warnings'])}\n")
    for d in report["new_warnings"][:100]:
        tag = "" if d["on_added_line"] else " (line not added by this range)"
        md.append(f"- `{d['file']}:{d['line']}:{d['col']}` {d['message']} `{d['flag']}`{tag}")
    md.append(f"\nPre-existing W=1 warnings at HEAD (same as merge base): "
              f"{len(report['preexisting_warnings'])}\n")
    if report["header_coverage"]:
        md.append("### Changed headers\n")
        md.append("| header | includers found | selected by config | use changed "
                  "identifiers | compiled |")
        md.append("|---|---|---|---|---|")
        for h in report["header_coverage"]:
            md.append(f"| `{h['header']}` | {h['includers_found']} | {h['selected']} | "
                      f"{h.get('users_of_changed_identifiers', 0)} | {h['compiled']} |")
        md.append("")
    if report["unchecked"] or report["build_files_changed"]:
        md.append("### Not compiled by this job\n")
        for u in report["unchecked"]:
            md.append(f"- `{u['path']}`: {u['reason']}")
        for b in report["build_files_changed"]:
            md.append(f"- `{b}`: build/config file (its effect is only seen by the full build)")
        md.append("")
    for n in report["notes"]:
        md.append(f"> {n}\n")
    md.append("Wall time (s): " + ", ".join(f"{k} {v}" for k, v in report["timings"].items()))
    md.append("\nCPU time of child processes (s): "
              + ", ".join(f"{k} {v}" for k, v in report["cpu_seconds"].items()))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return 1 if report["errors"] else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tree", required=True, help="kernel git checkout")
    p.add_argument("--base", required=True, help="target branch tip (or PR base)")
    p.add_argument("--head", required=True, help="PR head")
    p.add_argument("--config", required=True, help="starting .config")
    p.add_argument("--out", required=True, help="output directory (build dir, logs, report)")
    p.add_argument("--fragment", action="append", default=[],
                   help="in-tree config fragment merged at each state (repeatable)")
    p.add_argument("--per-commit", action="store_true", help="compile every commit")
    p.add_argument("--max-commits", type=int, default=20,
                   help="skip the per-commit pass for longer ranges")
    p.add_argument("--max-objects", type=int, default=150,
                   help="cap on objects compiled per state")
    p.add_argument("--header-objects", type=int, default=8,
                   help="objects compiled per changed header")
    p.add_argument("--header-depth", type=int, default=3,
                   help="include levels followed from a changed header")
    p.add_argument("--header-fanout", type=int, default=100,
                   help="including headers followed per level")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    p.add_argument("--llvm", default=os.environ.get("LLVM", "1"),
                   help="kbuild LLVM= value; write --llvm=-18 for a version suffix, "
                        "or give a prefix such as /usr/lib/llvm-18/bin/")
    p.add_argument("--rust", choices=("skip", "toolchain"), default="skip",
                   help="skip (default): keep CONFIG_RUST=y in the configuration but pass "
                        "CONFIG_RUST= to make, so rust/ and its prepare step never run and "
                        "no Rust toolchain is needed; toolchain: build normally (rustc, "
                        "rust-src and bindgen must be installed)")
    p.add_argument("--keep-debug-info", action="store_true",
                   help="keep the configuration's DEBUG_INFO (default: DEBUG_INFO_NONE, "
                        "which also drops BTF and makes prepare and compiles faster)")
    p.add_argument("--worktree", metavar="DIR",
                   help="check out states in this git worktree (created if missing) "
                        "instead of in --tree itself")
    p.add_argument("--make-arg", action="append", default=[],
                   help="extra make argument (repeatable)")
    a = p.parse_args(argv)
    t0 = time.monotonic()
    out = Path(a.out).resolve()
    ck = Checker(a)
    try:
        report = ck.execute()
    except SetupError as e:
        print(f"::error title=compile-check setup failed::{esc_data(str(e)[:3000])}")
        log(str(e))
        return 2
    finally:
        if getattr(ck, "orig_head", None):
            ck.git_main("checkout", "-q", "-f", ck.orig_head, check=False)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    report["cpu_seconds"]["total"] = round(ru.ru_utime + ru.ru_stime, 1)
    return emit(report, out, time.monotonic() - t0)


if __name__ == "__main__":
    sys.exit(main())
