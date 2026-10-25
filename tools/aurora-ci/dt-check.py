#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
dt-check.py - report device tree and DT binding problems that a change adds.

Compares a base and a head commit of a kernel tree and reports only the
problems that exist at head and not at base:

  * every DTB enabled by a minimal config (default CONFIG_ARCH_APPLE=y) is
    built with W=1 at both commits; build failures and dtc warnings are
    compared for all of them (this costs seconds);
  * DTBs whose output bytes changed (or that are new) get a full dt-validate
    schema check at both commits;
  * DTBs that use a binding the change touched get a dt-validate check
    limited (-l) to the touched bindings at both commits;
  * changed binding YAML files get `make dt_binding_check DT_SCHEMA_FILES=...`
    at both commits;
  * changed .dts/.dtsi/.dtso files get scripts/dtc/dt-check-style.

Warnings are keyed without line numbers or source file names (DTB, node
path, property, schema and message), so moving code around does not create
new warnings.

Exit status: 0 = no new blocking problems, 1 = new blocking problems,
2 = the check could not run.

Python 3.10+, standard library only.  Needs git, tar, make, a host C
compiler, flex and bison, plus dtschema (dt-validate, dt-mk-schema,
dt-doc-validate, dt-extract-example) and yamllint on PATH.
"""

import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

VERSION = '0.1'
BINDINGS = 'Documentation/devicetree/bindings'

CLASSES = collections.OrderedDict([
    ('dtb-build', 'DTB fails to build (preprocessor or dtc error)'),
    ('binding-invalid', 'binding fails YAML parsing or meta-schema validation'),
    ('binding-dropped', 'dt-mk-schema left the binding out of the processed schema'),
    ('example-build', 'binding example fails to extract or compile'),
    ('schema', 'DTB node fails a binding schema (dt-validate)'),
    ('unmatched', 'DTB compatible not documented by any binding (dt-validate -m)'),
    ('dtc-warning', 'dtc warning (W=1)'),
    ('example-schema', 'binding example fails schema validation, or dtc warns on it'),
    ('binding-warning', 'dt-mk-schema warning about a binding'),
    ('yamllint', 'yamllint finding in a binding'),
    ('style', 'coding style finding (dt-check-style)'),
    ('tool-error', 'a checker crashed or could not run'),
])
DEFAULT_BLOCKING = ('dtb-build', 'binding-invalid', 'binding-dropped', 'example-build')
DTB_SCOPED = ('dtb-build', 'schema', 'unmatched', 'dtc-warning')

ARCHIVE_PATHS = [
    ':(top,glob)*',              # top-level Makefile, Kconfig, ...
    ':(top,glob)**/Kconfig*',    # Kconfig needs every Kconfig file
    'scripts',
    'include/dt-bindings',
    BINDINGS,
    'tools/lib/python',          # scripts/jobserver-exec imports from here
    ':(top,glob)arch/*/boot/dts/**',  # scripts/dtc/include-prefixes targets
]


class SetupError(Exception):
    """The check could not run (exit status 2)."""


def note(msg):
    print(f'[dt-check] {msg}', file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# small helpers

class Timings:
    def __init__(self):
        self.t = collections.OrderedDict()

    @contextlib.contextmanager
    def __call__(self, name):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.t[name] = round(self.t.get(name, 0) + time.monotonic() - t0, 2)


def run(cmd, cwd=None, env=None, logfile=None):
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode('utf-8', 'replace')
    if logfile:
        with open(logfile, 'w', encoding='utf-8') as f:
            f.write('$ ' + ' '.join(cmd) + '\n' + out +
                    f'\n[exit {p.returncode}, {time.monotonic() - t0:.1f}s]\n')
    return p.returncode, out


def git(repo, *args, check=True, binary=False):
    p = subprocess.run(['git', '-C', repo, *args], stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise SetupError(f"git {' '.join(args)}: "
                         f"{p.stderr.decode('utf-8', 'replace').strip()}")
    return p.stdout if binary else p.stdout.decode('utf-8', 'replace')


def git_ok(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


LOC_RE = re.compile(r'(?:[^\s:,()\'"]+/)?[^\s:,()\'"/]+\.(?:dtsi|dtso|dts|h|yaml|tmp)'
                    r':\d+(?:[.:]\d+)?(?:-\d+(?:[.:]\d+)?)?')
BYTES_RE = re.compile(r"b'(?:[^'\\]|\\.)*'|b\"(?:[^\"\\]|\\.)*\"")


def mask(text):
    """Remove the parts of a message that move when code moves."""
    text = LOC_RE.sub('<loc>', text)
    text = BYTES_RE.sub("b'...'", text)
    return text


def problem(cls, scope, key, message, file=None, line=None, detail=None):
    return {'class': cls, 'scope': scope, 'key': [str(k) for k in key],
            'message': message, 'file': file, 'line': line,
            'detail': detail or []}


def short_dtb(dtb, arch):
    prefix = f'arch/{arch}/boot/dts/'
    return dtb[len(prefix):] if dtb.startswith(prefix) else dtb


# --------------------------------------------------------------------------
# git: changed files and tree extraction

def changed_files(repo, base, head):
    raw = git(repo, 'diff', '--name-status', '-z', '-M', '--no-ext-diff',
              base, head, binary=True)
    parts = raw.split(b'\0')
    res, i = [], 0
    while i < len(parts) and parts[i]:
        st = parts[i].decode()
        i += 1
        if st[0] in 'RC':
            old, new = parts[i].decode(), parts[i + 1].decode()
            i += 2
            res.append({'status': st[0], 'score': st[1:], 'old': old, 'path': new})
        else:
            p = parts[i].decode()
            i += 1
            res.append({'status': st[0], 'old': None if st[0] == 'A' else p,
                        'path': None if st[0] == 'D' else p})
    return res


def dt_symlink_targets(repo, sha):
    """Files that include/dt-bindings symlinks point at (e.g. uapi headers)."""
    out = git(repo, 'ls-tree', '-r', '-z', sha, '--', 'include/dt-bindings',
              binary=True, check=False)
    targets = set()
    for ent in out.split(b'\0'):
        if not ent or b'\t' not in ent:
            continue
        meta, path = ent.split(b'\t', 1)
        mode, _typ, oid = meta.split()
        if mode == b'120000':
            tgt = git(repo, 'cat-file', 'blob', oid.decode()).strip()
            targets.add(os.path.normpath(os.path.join(
                os.path.dirname(path.decode()), tgt)))
    return targets


def classify(path, symlink_targets):
    if path.startswith(BINDINGS + '/'):
        return 'binding' if path.endswith('.yaml') else 'binding-infra'
    if (re.match(r'arch/[^/]+/boot/dts/', path) or
            path.startswith('include/dt-bindings/') or path in symlink_targets):
        return 'dts'
    if path in ('scripts/dtc/dt-check-style',) or \
            path.startswith('scripts/dtc/dt-style-selftest/'):
        return 'binding-infra'
    if path.startswith('scripts/dtc/') or path == 'scripts/Makefile.dtbs':
        return 'tooling'
    return None


def _archive(repo, sha, specs, dest):
    a = subprocess.Popen(['git', '-C', repo, 'archive', '--format=tar', sha,
                          '--', *specs], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    t = subprocess.run(['tar', '-x', '-f', '-', '-C', dest], stdin=a.stdout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    a.stdout.close()
    err = a.stderr.read().decode('utf-8', 'replace')
    a.wait()
    if a.returncode or t.returncode:
        raise SetupError(f'git archive {sha[:12]} failed: {err.strip()} '
                         f'{t.stderr.decode("utf-8", "replace").strip()}')


def _dangling(dest):
    res = set()
    for root, dirs, files in os.walk(dest):
        for n in dirs + files:
            p = os.path.join(root, n)
            if os.path.islink(p) and not os.path.exists(p):
                tgt = os.path.normpath(os.path.join(root, os.readlink(p)))
                rel = os.path.relpath(tgt, dest)
                if not rel.startswith('..'):
                    res.add(rel)
    return res


def extract_tree(repo, sha, dest, arch):
    """Write the subset of the tree that DT builds and checks need."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    specs = [s for s in ARCHIVE_PATHS + [f'arch/{arch}']
             if s.startswith(':(') or git_ok(repo, 'cat-file', '-e', f'{sha}:{s}')]
    _archive(repo, sha, specs, dest)
    for _ in range(3):
        missing = sorted(p for p in _dangling(dest)
                         if git_ok(repo, 'cat-file', '-e', f'{sha}:{p}'))
        if not missing:
            break
        _archive(repo, sha, missing, dest)


# --------------------------------------------------------------------------
# DTB build (W=1) and its log

QUIET_RE = re.compile(r'^  ([A-Z][A-Z0-9_-]*)(?: \[C\])?\s+(\S+)\s*$')
MAKE_ERR_RE = re.compile(r'^make(?:\[\d+\])?: \*\*\* \[(?:[^\]]*?): (\S+)\] '
                         r'(?:Error \d+|Interrupt)')
DTC_WARN_RE = re.compile(r'^(?:(?P<file>[^\s:]+):(?P<line>\d+)\.\d+(?:-[\d.]+)?: )?'
                         r'Warning \((?P<check>[^)]+)\): (?P<msg>.*)$')
ALSO_RE = re.compile(r'^\s+also defined at (?P<file>[^\s:]+):(?P<line>\d+)')
ERR_RES = [
    re.compile(r'^Error: (?P<file>[^\s:]+):(?P<line>\d+)\.\d+(?:-[\d.]+)? (?P<msg>.*)$'),
    re.compile(r'^(?:(?P<file>[^\s:]+):(?P<line>\d+)\.\d+(?:-[\d.]+)?: )?'
               r'ERROR \((?P<check>[^)]+)\): (?P<msg>.*)$'),
    re.compile(r'^(?P<file>[^\s:]+):(?P<line>\d+):\d+: (?:fatal )?error: (?P<msg>.*)$'),
    re.compile(r'^(?P<msg>FATAL ERROR: .*)$'),
    re.compile(r'^(?P<msg>ERROR: .*)$'),
]


class Side:
    def __init__(self, name, sha, out):
        self.name = name
        self.sha = sha
        self.src = os.path.join(out, 'src', name)
        self.obj = os.path.join(out, 'build', name)
        self.logs = os.path.join(out, 'logs')
        self.dtbs = {}        # dtb -> info dict
        self.failed = {}      # dtb -> problem
        self.problems = []    # comparable problems
        self.schema = None    # processed-schema.json path
        self.schema_cmd = None

    def rel(self, path):
        """Repo-relative path for a path inside this side's src or obj."""
        if not path:
            return path
        for root in (self.src, self.obj):
            if path.startswith(root + '/'):
                path = path[len(root) + 1:]
                break
        else:
            if os.path.isabs(path):
                rp = os.path.realpath(path)
                for root in (os.path.realpath(self.src), os.path.realpath(self.obj)):
                    if rp.startswith(root + '/'):
                        return rp[len(root) + 1:]
        if path.startswith('scripts/dtc/include-prefixes/'):
            rp = os.path.realpath(os.path.join(self.src, path))
            sr = os.path.realpath(self.src)
            if rp.startswith(sr + '/'):
                return rp[len(sr) + 1:]
        return path


def parse_dtbs_log(side, out):
    """Return {dtb: {'warnings': [...], 'errors': [...]}}, failed set, stray errors."""
    per = collections.defaultdict(lambda: {'warnings': [], 'errors': []})
    failed, stray = set(), []
    cur, last = None, None
    for line in out.splitlines():
        m = QUIET_RE.match(line)
        if m:
            cur = m.group(2) if m.group(1) == 'DTC' and m.group(2).endswith('.dtb') else None
            last = None
            continue
        m = MAKE_ERR_RE.match(line)
        if m:
            if m.group(1).endswith('.dtb'):
                failed.add(m.group(1))
            cur = None
            continue
        m = DTC_WARN_RE.match(line)
        if m and cur:
            last = {'file': side.rel(m.group('file')), 'line': m.group('line'),
                    'check': m.group('check'), 'msg': m.group('msg'), 'also': []}
            per[cur]['warnings'].append(last)
            continue
        m = ALSO_RE.match(line)
        if m and last is not None:
            last['also'].append(f"{side.rel(m.group('file'))}:{m.group('line')}")
            continue
        for rx in ERR_RES:
            m = rx.match(line)
            if m:
                d = m.groupdict()
                err = {'file': side.rel(d.get('file')), 'line': d.get('line'),
                       'msg': d['msg'], 'text': line.replace(side.src + '/', '')}
                (per[cur]['errors'] if cur else stray).append(err)
                break
    return per, failed, stray


def parse_deps(cmdfile, side):
    deps, src, in_deps = set(), None, False
    with open(cmdfile, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('source_'):
                src = line.split(':=', 1)[1].strip()
                deps.add(src)
            elif line.startswith('deps_'):
                in_deps = True
            elif in_deps:
                s = line.strip()
                if not s:
                    in_deps = False
                    continue
                s = s.rstrip('\\').strip()
                if s and not s.startswith('$('):
                    deps.add(s)
    return {side.rel(d) for d in deps}, side.rel(src)


def build_side(args, side, env):
    """Extract, configure and build all DTBs with W=1."""
    extract_tree(args.tree, side.sha, side.src, args.arch)
    if os.path.exists(side.obj):
        shutil.rmtree(side.obj)
    os.makedirs(side.obj)
    frag = os.path.join(side.obj, 'dt-check.config')
    with open(frag, 'w') as f:
        f.write('\n'.join(args.kconfig) + '\n')
    mk = ['make', '-C', side.src, f'O={side.obj}', f'ARCH={args.arch}']
    rc, out = run(mk + [f'KCONFIG_ALLCONFIG={frag}', 'allnoconfig'], env=env,
                  logfile=os.path.join(side.logs, f'{side.name}-config.log'))
    if rc != 0:
        raise SetupError(f'{side.name}: allnoconfig failed (see logs/{side.name}-config.log)')
    with open(os.path.join(side.obj, '.config')) as f:
        cfg = f.read()
    for sym in args.kconfig:
        if sym.endswith('=y') and f'\n{sym}\n' not in '\n' + cfg:
            raise SetupError(f'{side.name}: {sym} did not stick in .config')
    rc, out = run(mk + [f'-j{args.jobs}', '-k', '-Otarget', 'W=1', 'dtbs'], env=env,
                  logfile=os.path.join(side.logs, f'{side.name}-dtbs.log'))
    per, failed, stray = parse_dtbs_log(side, out)
    side.dtbs_rc = rc
    dtsroot = os.path.join(side.obj, 'arch', args.arch, 'boot', 'dts')
    for root, _dirs, files in os.walk(dtsroot):
        for n in files:
            if not n.endswith('.dtb'):
                continue
            path = os.path.join(root, n)
            dtb = os.path.relpath(path, side.obj)
            info = {'path': path, 'sha': sha256(path), 'deps': set(), 'source': None,
                    'warnings': per[dtb]['warnings'] if dtb in per else []}
            cmdfile = os.path.join(root, f'.{n}.cmd')
            if os.path.exists(cmdfile):
                info['deps'], info['source'] = parse_deps(cmdfile, side)
            side.dtbs[dtb] = info
    for dtb in failed | {d for d in per if per[d]['errors'] and d not in side.dtbs}:
        errs = per[dtb]['errors'] if dtb in per else []
        side.failed[dtb] = errs
    if rc != 0 and not side.failed:
        side.failed['<make dtbs>'] = stray or [{'file': None, 'line': None,
                                                'msg': out.strip().splitlines()[-1] if out.strip() else 'make failed',
                                                'text': ''}]


# --------------------------------------------------------------------------
# FDT parsing (compatibles), binding $ref closure

def fdt_compatibles(path):
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 40 or struct.unpack_from('>I', data, 0)[0] != 0xd00dfeed:
        return set()
    (_magic, _size, off_struct, off_strings, _rsv, _ver, _lc, _cpu,
     size_strings, size_struct) = struct.unpack_from('>10I', data, 0)
    strings = data[off_strings:off_strings + size_strings]
    i, end, out = off_struct, off_struct + size_struct, set()
    while i < end:
        tok = struct.unpack_from('>I', data, i)[0]
        i += 4
        if tok == 1:                       # FDT_BEGIN_NODE
            j = data.index(b'\0', i)
            i = (j + 1 + 3) & ~3
        elif tok == 3:                     # FDT_PROP
            ln, nameoff = struct.unpack_from('>II', data, i)
            i += 8
            name = strings[nameoff:strings.index(b'\0', nameoff)]
            if name == b'compatible':
                out.update(x.decode('ascii', 'replace')
                           for x in data[i:i + ln].split(b'\0') if x)
            i = (i + ln + 3) & ~3
        elif tok == 9:                     # FDT_END
            break
    return out


REF_RE = re.compile(r'''\$ref:\s*['"]?([^\s'"#]+)''')
TOKEN_RE = re.compile(r'[A-Za-z0-9][\w.,+-]*')


def binding_closure(src, changed):
    """Changed bindings plus every binding that $refs them (transitively)."""
    broot = os.path.join(src, BINDINGS)
    rev = collections.defaultdict(set)
    for root, _dirs, files in os.walk(broot):
        for n in files:
            if not n.endswith('.yaml'):
                continue
            path = os.path.join(root, n)
            rel = os.path.relpath(path, src)
            try:
                text = open(path, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            for ref in REF_RE.findall(text):
                if ref.startswith('http://devicetree.org/schemas/'):
                    tgt = BINDINGS + '/' + ref[len('http://devicetree.org/schemas/'):]
                elif ref.startswith('/schemas/'):
                    tgt = BINDINGS + '/' + ref[len('/schemas/'):]
                elif ref.startswith('/'):
                    continue
                else:
                    tgt = os.path.normpath(os.path.join(os.path.dirname(rel), ref))
                rev[tgt].add(rel)
    seen, todo = set(changed), list(changed)
    while todo:
        f = todo.pop()
        for r in rev.get(f, ()):
            if r not in seen:
                seen.add(r)
                todo.append(r)
    return seen


def binding_selectors(src, files):
    """(tokens that may be compatibles, applies-to-everything flag)."""
    tokens, everything = set(), False
    for rel in files:
        path = os.path.join(src, rel)
        if not os.path.exists(path):
            continue
        text = open(path, encoding='utf-8', errors='replace').read()
        tokens.update(TOKEN_RE.findall(text))
        m = re.search(r'^select:(.*(?:\n[ \t]+.*)*)', text, re.M)
        if m:
            body = m.group(1).strip()
            if body != 'false' and 'compatible' not in body:
                everything = True
        if re.search(r'compatible:\s*\n\s+pattern:', text):
            everything = True
    return tokens, everything


# --------------------------------------------------------------------------
# processed schema

def schema_cache_key(args, sha, dtschema_version):
    tree = git(args.tree, 'rev-parse', f'{sha}:{BINDINGS}').strip()
    return f'{dtschema_version}-{tree}'


def install_schema(side, json_path, cmd_template):
    d = os.path.join(side.obj, BINDINGS)
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(json_path, os.path.join(d, 'processed-schema.json'))
    if cmd_template:
        with open(os.path.join(d, '.processed-schema.json.cmd'), 'w') as f:
            f.write(cmd_template.replace('@SRCTREE@', side.src))
    side.schema = os.path.join(d, 'processed-schema.json')


def parse_schema_log(side, out):
    probs = []
    for line in out.splitlines():
        m = re.match(r'^(\S+\.yaml): (.*)$', line)
        if not m:
            continue
        f, msg = side.rel(m.group(1)), m.group(2)
        cls = 'binding-dropped' if msg.startswith('ignoring') else 'binding-warning'
        probs.append(problem(cls, f, ('mk-schema', mask(msg)), msg, file=f,
                             line=yaml_key_line(side.src, f, msg)))
    return probs


def build_schema(args, side, env, cache_dir, key):
    """Build (or fetch from cache) processed-schema.json for one side."""
    if cache_dir:
        cj = os.path.join(cache_dir, 'schema', key + '.json')
        if os.path.exists(cj):
            tmpl = open(cj[:-5] + '.cmd').read() if os.path.exists(cj[:-5] + '.cmd') else None
            install_schema(side, cj, tmpl)
            logtxt = open(cj[:-5] + '.log').read() if os.path.exists(cj[:-5] + '.log') else ''
            logtxt = logtxt.replace('@SRCTREE@', side.src)
            return parse_schema_log(side, logtxt), True
    mk = ['make', '-C', side.src, f'O={side.obj}', f'ARCH={args.arch}']
    rc, out = run(mk + ['dt_binding_schemas'], env=env,
                  logfile=os.path.join(side.logs, f'{side.name}-schema.log'))
    path = os.path.join(side.obj, BINDINGS, 'processed-schema.json')
    if rc != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise SetupError(f'{side.name}: building processed-schema.json failed '
                         f'(see logs/{side.name}-schema.log)')
    side.schema = path
    cmdf = os.path.join(side.obj, BINDINGS, '.processed-schema.json.cmd')
    tmpl = open(cmdf).read().replace(side.src, '@SRCTREE@') if os.path.exists(cmdf) else None
    side.schema_cmd = tmpl
    if cache_dir:
        os.makedirs(os.path.join(cache_dir, 'schema'), exist_ok=True)
        base = os.path.join(cache_dir, 'schema', key)
        shutil.copyfile(path, base + '.json.tmp')
        os.replace(base + '.json.tmp', base + '.json')
        if tmpl:
            open(base + '.cmd', 'w').write(tmpl)
        open(base + '.log', 'w').write(out.replace(side.src, '@SRCTREE@'))
    return parse_schema_log(side, out), False


# --------------------------------------------------------------------------
# dt-validate

UNEXPECTED_RE = re.compile(r'^(Additional|Unevaluated) properties are not allowed '
                           r'\((.*) (?:was|were) unexpected\)$')
NOMATCH_RE = re.compile(r'^(.*) do(?:es)? not match any of the regexes: .*$')
QUOTED_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")
# jsonschema messages start with the instance value ("[[45], [253]] is too
# long", "{...whole node...} is not valid under ...").  Values such as phandle
# numbers change when unrelated nodes are added, so keys drop them.
INSTANCE_RE = re.compile(
    r"^(?P<inst>.+) (?P<rest>(?:is|are) (?:too long|too short|not of type .+|"
    r"not valid under any of the given schemas|not one of .+|"
    r"greater than(?: or equal to)? .+|less than(?: or equal to)? .+|"
    r"not a multiple of .+|expected to be empty|a required property|"
    r"a dependency of .+|not allowed for .+)|should not be valid under .+|"
    r"does not match .+|does not contain .+|has non-unique elements)$", re.S)


def mask_instance(msg):
    m = INSTANCE_RE.match(msg)
    if m and not m.group('rest').startswith(('is a required property', 'is a dependency of')):
        return '<value> ' + m.group('rest')
    if msg.startswith('False schema does not allow '):
        return 'False schema does not allow <value>'
    return mask(msg)


def schema_short(url):
    if not url:
        return '?'
    url = url.rstrip('#')
    for p in ('http://devicetree.org/schemas/', 'http://devicetree.org/meta-schemas/'):
        if url.startswith(p):
            return url[len(p):]
    return url


def diag_problems(dtb, diag):
    """Problems from one dt-validate JSON diagnostic."""
    t = diag.get('type')
    node = diag.get('node') or diag.get('nodename') or '?'
    if t == 'unmatched':
        compat = diag.get('compatible') or []
        if isinstance(compat, str):
            compat = [compat]
        return [problem('unmatched', dtb, (node, ','.join(compat)),
                        f'{node}: compatible {compat} matches no binding',
                        detail=[diag.get('message', '')])]
    pp = [str(x) for x in diag.get('property_path') or []]
    schema = schema_short(diag.get('schema'))
    msg = diag.get('message', '')
    where = node + (': ' + '/'.join(pp) if pp else '')
    detail = [c.get('message', '') for c in diag.get('context') or []][:8]
    m = UNEXPECTED_RE.match(msg)
    if m:
        return [problem('schema', dtb, (node, '/'.join(pp), schema, m.group(1) + '-unexpected', n),
                        f"{where}: property '{n}' is not allowed ({m.group(1).lower()} properties) [{schema}]",
                        detail=detail, line=None, file=None) | {'prop': n, 'node': node}
                for n in QUOTED_RE.findall(m.group(2))]
    m = NOMATCH_RE.match(msg)
    if m:
        return [problem('schema', dtb, (node, '/'.join(pp), schema, 'name-not-matching', n),
                        f"{where}: '{n}' does not match the allowed names [{schema}]",
                        detail=detail) | {'prop': n, 'node': node}
                for n in QUOTED_RE.findall(m.group(1))]
    sp = diag.get('schema_path') or []
    keyword = str(sp[-1]) if sp else ''
    p = problem('schema', dtb, (node, '/'.join(pp), schema, keyword, mask_instance(msg)),
                f'{where}: {msg} [{schema}]', detail=detail)
    p['node'], p['prop'] = node, pp[0] if pp else None
    return [p]


TEXT_HDR_RE = re.compile(r'^(\S+\.dtb): (.*)$')


def text_problems(side, text, path_to_dtb):
    """Fallback parser for dt-validate text output (dtschema < 2026.9)."""
    probs, cur = [], None
    for line in text.splitlines():
        m = TEXT_HDR_RE.match(line)
        if m:
            dtb = path_to_dtb.get(os.path.abspath(os.path.join(side.obj, m.group(1))),
                                  path_to_dtb.get(m.group(1), m.group(1)))
            hdr = m.group(2)
            cls = 'unmatched' if 'failed to match any schema' in hdr else 'schema'
            cur = problem(cls, dtb, (mask(hdr),), hdr)
            probs.append(cur)
        elif line.startswith('\t') and cur is not None:
            s = line.strip()
            if s.startswith('from schema $id:'):
                cur['key'].append(schema_short(s.split(':', 1)[1].strip()))
            else:
                cur['detail'].append(s)
        else:
            cur = None
    return probs


def run_validation(args, sides, tasks, env, json_ok, cache_dir):
    """tasks: list of (side_name, limit or None, [dtb]).  Returns problems per side."""
    jobs = []
    for side_name, limit, dtbs in tasks:
        side = sides[side_name]
        dtbs = sorted(dtbs, key=lambda d: -os.path.getsize(side.dtbs[d]['path']))
        n = max(1, min(len(dtbs), args.jobs))
        for k in range(n):
            chunk = dtbs[k::n]
            if chunk:
                jobs.append((side_name, limit, chunk))
    results = {name: [] for name in sides}
    errors = []
    validated = {name: {} for name in sides}

    def one(idx, side_name, limit, chunk):
        side = sides[side_name]
        jout = os.path.join(side.logs, f'{side_name}-validate-{idx}.json')
        cmd = ['dt-validate', '-m'] + (['-l', limit] if limit else []) + \
              ['-u', os.path.join(side.src, BINDINGS), '-p', side.schema]
        if json_ok:
            cmd += ['--json-output', jout]
        if cache_dir:
            cmd += ['--cache-dir', os.path.join(cache_dir, 'validate')]
        cmd += [side.dtbs[d]['path'] for d in chunk]
        rc, out = run(cmd, cwd=side.obj, env=env,
                      logfile=os.path.join(side.logs, f'{side_name}-validate-{idx}.log'))
        return idx, side_name, limit, chunk, rc, out, jout

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, i, *j) for i, j in enumerate(jobs)]
        for fut in concurrent.futures.as_completed(futs):
            idx, side_name, limit, chunk, rc, out, jout = fut.result()
            side = sides[side_name]
            p2d = {side.dtbs[d]['path']: d for d in chunk}
            p2d.update({os.path.relpath(side.dtbs[d]['path'], side.obj): d for d in chunk})
            if rc != 0 or 'Traceback (most recent call last)' in out:
                tail = out.strip().splitlines()[-1] if out.strip() else f'exit {rc}'
                errors.append(f'{side_name}: dt-validate chunk {idx} failed: {tail}')
                continue
            for d in chunk:
                validated[side_name][d] = 'limited' if limit else 'full'
            if json_ok:
                with open(jout, encoding='utf-8') as f:
                    diags = json.load(f)
                for diag in diags:
                    fpath = diag.get('file') or ''
                    dtb = p2d.get(fpath) or p2d.get(os.path.abspath(os.path.join(side.obj, fpath)))
                    if dtb is None:
                        continue
                    for p in diag_problems(dtb, diag):
                        p['mode'] = 'limited' if limit else 'full'
                        results[side_name].append(p)
            else:
                for p in text_problems(side, out, p2d):
                    p['mode'] = 'limited' if limit else 'full'
                    results[side_name].append(p)
    return results, validated, errors


# --------------------------------------------------------------------------
# source map from dtc -T -T (for annotations of schema problems)

LOCPIECE_RE = re.compile(r'^(.*):(\d+):\d+-\d+:\d+$')


def source_map(side, dtb, arch):
    tmp = os.path.join(side.obj, os.path.dirname(dtb), '.' + os.path.basename(dtb) + '.dts.tmp')
    dtc = os.path.join(side.obj, 'scripts', 'dtc', 'dtc')
    if not (os.path.exists(tmp) and os.path.exists(dtc)):
        return {}, {}
    rc, out = run([dtc, '-q', '-I', 'dts', '-O', 'dts', '-T', '-T',
                   '-i', os.path.join(side.src, os.path.dirname(dtb)),
                   '-i', os.path.join(side.src, 'scripts', 'dtc', 'include-prefixes'),
                   '-o', '-', tmp])
    if rc != 0:
        return {}, {}
    nodes, props, stack = {}, {}, []

    def locs(comment):
        res = []
        for piece in comment.split(', '):
            m = LOCPIECE_RE.match(piece.strip())
            if m:
                res.append((side.rel(m.group(1)), int(m.group(2))))
        return res

    for raw in out.splitlines():
        s = raw.strip()
        comment = ''
        if s.endswith('*/') and '/*' in s:
            comment = s[s.rfind('/*') + 2:-2].strip()
            s = s[:s.rfind('/*')].strip()
        if s.endswith('{'):
            name = s[:-1].strip().split()[-1]
            if not stack:
                path = '/'
            else:
                path = (stack[-1].rstrip('/') + '/' + name)
            stack.append(path)
            nodes[path] = locs(comment)
        elif s.startswith('};'):
            if stack:
                stack.pop()
        elif s.endswith(';') and stack and not s.startswith('/'):
            name = re.split(r'\s*=\s*|;', s, maxsplit=1)[0].split()[-1]
            props[(stack[-1], name)] = locs(comment)
    return nodes, props


def pick_location(cands, changed):
    for f, ln in cands:
        if f in changed:
            return f, ln
    return cands[0] if cands else (None, None)


# --------------------------------------------------------------------------
# dt_binding_check log

YAMLLINT_RE = re.compile(r'^(?P<file>\S+\.yaml):(?P<line>\d+):\d+: \[(?P<lvl>error|warning)\] '
                         r'(?P<msg>.*?)(?: \((?P<rule>[\w-]+)\))?$')
STYLE_RE = re.compile(r'^(?P<file>\S+?):(?P<line>\d+):(?: example (?P<ex>\d+))? '
                      r'\[(?P<rule>[\w-]+)\] (?P<msg>.*)$')
YAMLPOS_RE = re.compile(r'^(?P<file>\S+\.yaml):(?P<line>\d+):\d+: (?P<msg>.*)$')
FILEMSG_RE = re.compile(r'^(?P<file>\S+\.(?:yaml|dtb|dts)): (?P<msg>.*)$')


def example_yaml(path):
    return re.sub(r'\.example\.dt[bs]$', '.yaml', path)


def parse_binding_log(side, out):
    probs = []
    tag, target, cur, tb = None, None, None, None
    lines = out.splitlines()
    for line in lines:
        if tb is not None:                 # inside a Python traceback
            if line.startswith(' ') or line.startswith('Traceback'):
                continue
            exc = line.strip()
            tb = None
            if tag == 'DTEX' or (tag == 'DTC' and target and 'example' in target):
                f = example_yaml(side.rel(target))
                probs.append(problem('example-build', f, ('extract', mask(exc)),
                                     f'example extraction failed: {exc}', file=f))
            elif tag == 'CHKDT':
                probs.append(problem('binding-invalid', 'dt-doc-validate', ('crash', mask(exc)),
                                     f'dt-doc-validate crashed: {exc}'))
            else:
                probs.append(problem('tool-error', tag or 'make', (mask(exc),),
                                     f'{tag or "make"} step crashed: {exc}'))
            continue
        if line.startswith('Traceback (most recent call last)'):
            tb = []
            cur = None
            continue
        m = QUIET_RE.match(line)
        if m:
            tag, target, cur = m.group(1), m.group(2), None
            continue
        m = MAKE_ERR_RE.match(line)
        if m:
            t = m.group(1)
            if '.example.' in t:
                f = example_yaml(side.rel(t))
                probs.append(problem('example-build', f, ('make',),
                                     'example does not build', file=f))
            cur = None
            continue
        if line.startswith('\t') and cur is not None:
            s = line.strip()
            if s.startswith('from schema $id:'):
                cur['key'].append(schema_short(s.split(':', 1)[1].strip()))
            else:
                cur['detail'].append(s)
            continue
        cur = None
        if tag == 'LINT':
            m = YAMLLINT_RE.match(line)
            if m:
                f = side.rel(m.group('file'))
                rule = m.group('rule') or m.group('lvl')
                cls = 'binding-invalid' if rule == 'syntax' else 'yamllint'
                probs.append(problem(cls, f, ('yamllint', m.group('lvl'), rule, mask(m.group('msg'))),
                                     f"yamllint {m.group('lvl')}: {m.group('msg')} ({rule})",
                                     file=f, line=int(m.group('line'))))
            continue
        if tag == 'STYLE':
            m = STYLE_RE.match(line)
            if m:
                f = side.rel(m.group('file'))
                probs.append(problem('style', f, (m.group('rule'), mask(m.group('msg'))),
                                     f"[{m.group('rule')}] {m.group('msg')}",
                                     file=f, line=int(m.group('line'))))
                continue
            m = FILEMSG_RE.match(line)
            if m:
                f = side.rel(m.group('file'))
                probs.append(problem('style', f, ('load', mask(m.group('msg'))),
                                     m.group('msg'), file=f))
            continue
        if tag == 'CHKDT':
            m = YAMLPOS_RE.match(line)
            if m:
                f = side.rel(m.group('file'))
                probs.append(problem('binding-invalid', f, ('parse', mask(m.group('msg'))),
                                     m.group('msg'), file=f, line=int(m.group('line'))))
                continue
            m = FILEMSG_RE.match(line)
            if m:
                f = side.rel(m.group('file'))
                cur = problem('binding-invalid', f, ('meta', mask(m.group('msg'))),
                              m.group('msg'), file=f)
                probs.append(cur)
            continue
        if tag in ('DTC', 'DTEX'):
            m = DTC_WARN_RE.match(line)
            if m and m.group('file') and '.example.' in m.group('file'):
                f = example_yaml(side.rel(m.group('file')))
                probs.append(problem('example-schema', f,
                                     ('dtc', m.group('check'), mask(m.group('msg'))),
                                     f"example dtc warning ({m.group('check')}): {m.group('msg')}",
                                     file=f))
                continue
            hit = False
            for rx in ERR_RES:
                m = rx.match(line)
                if m:
                    f = example_yaml(side.rel(target)) if target else 'example'
                    probs.append(problem('example-build', f, ('dtc', mask(m.group('msg'))),
                                         f"example does not compile: {m.group('msg')}", file=f))
                    hit = True
                    break
            if hit:
                continue
            m = FILEMSG_RE.match(line)
            if m and '.example.' in m.group('file'):
                f = example_yaml(side.rel(m.group('file')))
                cur = problem('example-schema', f, ('validate', mask(m.group('msg'))),
                              m.group('msg'), file=f)
                probs.append(cur)
            continue
        if tag == 'SCHEMA':
            m = FILEMSG_RE.match(line)
            if m:
                probs.extend(parse_schema_log(side, line))
    return probs


KEYLINE_RE = re.compile(r'^(\s*)(?:-\s+)?([^\s:#\'"][^:#]*?|\'[^\']*\'|"[^"]*"):(?:\s|$)')


def yaml_key_line(src, rel, message):
    """Line of the YAML key path a dtschema message starts with (a:b:c: ...)."""
    m = re.match(r'^ignoring, error in schema: (.*)$', message)
    if m:
        path = [x.strip() for x in m.group(1).split(': ')]
    else:
        head = message.split(': ', 1)[0]
        if ' ' in head:
            return None
        path = [x for x in head.split(':') if x]
    if not path:
        return None
    try:
        with open(os.path.join(src, rel), encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    stack, best = [], None
    for i, line in enumerate(lines, 1):
        km = KEYLINE_RE.match(line)
        if not km or line.lstrip().startswith('#'):
            continue
        ind = len(km.group(1))
        key = km.group(2).strip('\'"')
        while stack and stack[-1][0] >= ind:
            stack.pop()
        stack.append((ind, key))
        keys = [k for _, k in stack]
        n = 0
        while n < len(keys) and n < len(path) and keys[n] == path[n]:
            n += 1
        if n == len(keys) and n > 0 and (best is None or n > best[0]):
            best = (n, i)
            if n == len(path):
                break
    return best[1] if best else None


def examples_line(src, rel):
    try:
        with open(os.path.join(src, rel), encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                if line.startswith('examples:'):
                    return i
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------
# style check of DTS sources

def style_check(side, files, mode, python, env):
    checker = os.path.join(side.src, 'scripts', 'dtc', 'dt-check-style')
    files = [f for f in files if os.path.exists(os.path.join(side.src, f))]
    if not files or not os.path.exists(checker):
        return []
    rc, out = run([python, checker, f'--mode={mode}', *files], cwd=side.src, env=env,
                  logfile=os.path.join(side.logs, f'{side.name}-style.log'))
    probs = []
    if 'Traceback (most recent call last)' in out:
        exc = [l for l in out.splitlines() if l and not l.startswith(' ')][-1]
        return [problem('tool-error', 'dt-check-style', (mask(exc),),
                        f'dt-check-style crashed: {exc}')]
    for line in out.splitlines():
        m = STYLE_RE.match(line)
        if m:
            f = m.group('file')
            probs.append(problem('style', f, (m.group('rule'), mask(m.group('msg'))),
                                 f"[{m.group('rule')}] {m.group('msg')}", file=f,
                                 line=int(m.group('line'))))
    return probs


# --------------------------------------------------------------------------
# comparison and output

def compare(base_probs, head_probs, scope_map):
    def key(p, mapped):
        s = scope_map.get(p['scope'], p['scope']) if mapped else p['scope']
        return (p['class'], s, tuple(p['key']))
    bc = collections.Counter(key(p, True) for p in base_probs)
    hg = collections.defaultdict(list)
    for p in head_probs:
        hg[key(p, False)].append(p)
    new = []
    for k, plist in hg.items():
        extra = len(plist) - bc.get(k, 0)
        if extra > 0:
            new.extend(plist[len(plist) - extra:])
    hc = collections.Counter({k: len(v) for k, v in hg.items()})
    fixed = []
    bg = collections.defaultdict(list)
    for p in base_probs:
        bg[key(p, True)].append(p)
    for k, plist in bg.items():
        extra = len(plist) - hc.get(k, 0)
        if extra > 0:
            fixed.extend(plist[:extra])
    return new, fixed


def group_new(new, blocking, changed_head, arch):
    groups = collections.OrderedDict()
    for p in new:
        cls = p['class']
        gk = (cls, tuple(p['key'])) if cls in DTB_SCOPED else (cls, p['scope'], tuple(p['key']))
        g = groups.get(gk)
        if g is None:
            g = groups[gk] = {'class': cls, 'blocking': cls in blocking,
                              'message': p['message'], 'scopes': [], 'file': None,
                              'line': None, 'detail': p['detail'][:10], 'count': 0}
        g['count'] += 1
        sc = short_dtb(p['scope'], arch) if cls in DTB_SCOPED else p['scope']
        if sc not in g['scopes']:
            g['scopes'].append(sc)
        if p.get('file') and (g['file'] is None or
                              (p['file'] in changed_head and g['file'] not in changed_head)):
            g['file'], g['line'] = p['file'], p.get('line')
    order = list(CLASSES)
    for g in groups.values():
        g['scopes'].sort()
    return sorted(groups.values(), key=lambda g: (not g['blocking'], order.index(g['class'])))


def esc_data(s):
    return s.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')


def esc_prop(s):
    return esc_data(s).replace(':', '%3A').replace(',', '%2C')


def annotate(groups, limit):
    shown = 0
    for g in groups:
        if shown >= limit:
            break
        level = 'error' if g['blocking'] else 'warning'
        where = ''
        if g['file']:
            where = f"file={esc_prop(g['file'])}"
            if g['line']:
                where += f",line={g['line']}"
            where += ','
        scopes = g['scopes']
        tail = ', '.join(scopes[:5]) + (f' and {len(scopes) - 5} more' if len(scopes) > 5 else '')
        title = f"dt-check {g['class']}" + (
            f" ({len(scopes)} DTB{'' if len(scopes) == 1 else 's'})" if g['class'] in DTB_SCOPED else '')
        msg = g['message'] + (f'\nIn: {tail}' if g['class'] in DTB_SCOPED else '')
        if g['detail']:
            msg += '\n' + '\n'.join(g['detail'][:5])
        print(f"::{level} {where}title={esc_prop(title)}::{esc_data(msg)}", flush=True)
        shown += 1
    if len(groups) > shown:
        print(f'::notice title=dt-check::{len(groups) - shown} more new problem groups; '
              'see the job summary or report.json', flush=True)


def md_escape(s):
    return s.replace('|', '\\|').replace('<', '&lt;').replace('>', '&gt;')


def write_summary(path, report):
    st = report['status']
    lines = []
    head = {'skipped': 'nothing to check', 'pass': 'no new blocking problems',
            'fail': 'new blocking problems', 'error': 'check could not run'}[st]
    lines.append(f"## Device tree check: {st.upper()} ({head})\n")
    lines.append(f"base `{report['base'][:12]}` -> head `{report['head'][:12]}`"
                 + (' (merge base of the given base and head)' if report.get('merge_base_used') else '')
                 + f"; dtschema {report['versions'].get('dtschema', '?')}\n")
    if report.get('error'):
        lines.append(f"**Error:** {md_escape(report['error'])}\n")
    if st == 'skipped':
        lines.append('No device tree, DT header or binding files changed.\n')
    if report.get('counts'):
        lines.append('| Class | Blocks | Base | Head | New | Fixed |')
        lines.append('|---|---|---:|---:|---:|---:|')
        for cls, c in report['counts'].items():
            lines.append(f"| {cls} | {'yes' if cls in report['blocking_classes'] else 'no'} | "
                         f"{c['base']} | {c['head']} | {c['new']} | {c['fixed']} |")
        lines.append('')
    groups = report.get('new', [])
    blk = [g for g in groups if g['blocking']]
    adv = [g for g in groups if not g['blocking']]

    def fmt(g):
        loc = f" (`{g['file']}{':' + str(g['line']) if g['line'] else ''}`)" if g['file'] else ''
        sc = g['scopes']
        where = ', '.join(f'`{s}`' for s in sc[:6]) + (f' and {len(sc) - 6} more' if len(sc) > 6 else '')
        return f"- **{g['class']}** {md_escape(g['message'])}{loc} in {where}"
    if blk:
        lines.append(f'### New blocking problems ({len(blk)})\n')
        lines += [fmt(g) for g in blk[:100]]
        lines.append('')
    if adv:
        lines.append(f'### New advisory problems ({len(adv)} groups)\n')
        lines.append('<details><summary>show</summary>\n')
        lines += [fmt(g) for g in adv[:200]]
        if len(adv) > 200:
            lines.append(f'- ... {len(adv) - 200} more in report.json')
        lines.append('\n</details>\n')
    if st in ('pass', 'fail'):
        d = report['dtbs']
        lines.append('### Scope\n')
        lines.append(f"- DT-related files changed: {len(report['dt_changed_files'])}")
        lines.append(f"- DTBs built: base {d['base_built']}, head {d['head_built']}; "
                     f"failing at head: {len(d['head_failed'])}")
        lines.append(f"- DTBs depending on changed files: {len(d['affected'])}; "
                     f"output changed or new: {len(d['changed_output'])}")
        lines.append(f"- schema check: full {len(d['validated_full'])}, "
                     f"limited to changed bindings {len(d['validated_limited'])}")
        lines.append(f"- bindings checked: {len(report['bindings_checked'])}")
        lines.append(f"- fixed problems: {sum(c['fixed'] for c in report['counts'].values())}")
        lines.append('')
    if report.get('notes'):
        lines.append('### Notes\n')
        lines += [f'- {md_escape(n)}' for n in report['notes']]
        lines.append('')
    lines.append('### Timings (s)\n')
    lines.append(', '.join(f'{k} {v}' for k, v in report['timings_s'].items()) + '\n')
    text = '\n'.join(lines) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


# --------------------------------------------------------------------------
# main flow

def tool_versions():
    v = {'python': sys.version.split()[0], 'dt-check': VERSION}
    for name, cmd in (('dtschema', ['dt-validate', '-V']), ('yamllint', ['yamllint', '--version']),
                      ('make', ['make', '--version']), ('git', ['git', '--version'])):
        try:
            rc, out = run(cmd)
            v[name] = out.strip().splitlines()[0] if rc == 0 and out.strip() else '?'
        except OSError:
            v[name] = None
    if v.get('yamllint'):
        v['yamllint'] = v['yamllint'].replace('yamllint ', '')
    return v


def check(args, report, timings):
    out = args.out
    repo = args.tree
    env = dict(os.environ)
    env['LC_ALL'] = 'C'
    env['TMPDIR'] = os.path.join(out, 'tmp')
    for k in ('KBUILD_OUTPUT', 'KCONFIG_CONFIG', 'CROSS_COMPILE', 'LLVM'):
        env.pop(k, None)
    os.makedirs(env['TMPDIR'], exist_ok=True)
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)

    with timings('resolve'):
        base = git(repo, 'rev-parse', '--verify', args.base + '^{commit}').strip()
        head = git(repo, 'rev-parse', '--verify', args.head + '^{commit}').strip()
        report['base_requested'] = base
        if base != head and not git_ok(repo, 'merge-base', '--is-ancestor', base, head):
            mb = git(repo, 'merge-base', base, head, check=False).strip()
            if mb:
                base = mb
                report['merge_base_used'] = True
                note(f'base is not an ancestor of head; using merge base {mb[:12]}')
            else:
                report['notes'].append('no merge base found (shallow clone?); diffing the given commits')
        report['base'], report['head'] = base, head
        changed = changed_files(repo, base, head)
        targets = dt_symlink_targets(repo, head) | dt_symlink_targets(repo, base)
        dt_changed = []
        for c in changed:
            kinds = {classify(p, targets) for p in (c['path'], c['old']) if p}
            kinds.discard(None)
            if kinds:
                c['kind'] = sorted(kinds)
                dt_changed.append(c)
        report['changed_files'] = len(changed)
        report['dt_changed_files'] = dt_changed

    if not dt_changed:
        report['status'] = 'skipped'
        note('no DT-related files changed; nothing to do')
        return 0

    for tool in ('make', 'tar', 'dt-validate', 'dt-mk-schema', 'dt-doc-validate',
                 'dt-extract-example', 'flex', 'bison'):
        if not shutil.which(tool):
            raise SetupError(f'{tool} not found on PATH')
    if not shutil.which('yamllint'):
        report['notes'].append('yamllint not on PATH; binding lint skipped')
    _, help_out = run(['dt-validate', '--help'])
    json_ok = '--json-output' in help_out
    validate_cache = args.cache_dir if '--cache-dir' in help_out else None
    if not json_ok:
        report['notes'].append('dt-validate has no --json-output (dtschema < 2026.9); '
                               'using the less precise text parser')
    dtschema_version = report['versions'].get('dtschema') or '?'

    # file renames (base path -> head path), changed paths on each side
    renames = {c['old']: c['path'] for c in dt_changed if c['status'] == 'R'}
    changed_head = {c['path'] for c in dt_changed if c['path']}
    changed_base = {c['old'] for c in dt_changed if c['old']}
    kinds = set(k for c in dt_changed for k in c['kind'])
    bind_head = sorted(c['path'] for c in dt_changed if c['path'] and 'binding' in c['kind']
                       and c['path'].endswith('.yaml'))
    bind_base = sorted(c['old'] for c in dt_changed if c['old'] and 'binding' in c['kind']
                       and c['old'].endswith('.yaml'))
    dts_head = sorted(p for p in changed_head if p.endswith(('.dts', '.dtsi', '.dtso'))
                      and re.match(r'arch/[^/]+/boot/dts/', p))
    dts_base = sorted(p for p in changed_base if p.endswith(('.dts', '.dtsi', '.dtso'))
                      and re.match(r'arch/[^/]+/boot/dts/', p))

    sides = {'base': Side('base', base, out), 'head': Side('head', head, out)}

    # 1. build all DTBs with W=1 on both sides
    with timings('extract+config+dtbs'):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(build_side, args, s, env) for s in sides.values()]
            for f in futs:
                f.result()
    B, H = sides['base'], sides['head']

    # DTB renames follow .dts renames
    dtb_map = {}
    for old, new in renames.items():
        if old.endswith('.dts') and new.endswith('.dts'):
            dtb_map[old[:-4] + '.dtb'] = new[:-4] + '.dtb'
    scope_map = dict(renames)
    scope_map.update(dtb_map)

    def head_name(dtb):
        return dtb_map.get(dtb, dtb)

    base_ok = {head_name(d): d for d in B.dtbs}
    head_ok = set(H.dtbs)
    head_failed = set(H.failed)
    base_failed = {head_name(d) for d in B.failed}

    # DTBs depending on changed files (either side)
    affected = {}
    for d, info in H.dtbs.items():
        hit = sorted(info['deps'] & changed_head)
        if hit:
            affected[d] = hit
    for d, info in B.dtbs.items():
        hit = sorted(info['deps'] & changed_base)
        if hit:
            affected.setdefault(head_name(d), []).extend(renames.get(h, h) for h in hit)
    affected = {d: sorted(set(v)) for d, v in affected.items()}
    changed_output = sorted(d for d in head_ok
                            if d not in base_ok or H.dtbs[d]['sha'] != B.dtbs[base_ok[d]]['sha'])
    new_dtbs = sorted(d for d in head_ok if d not in base_ok and d not in base_failed)
    stray = sorted(set(changed_output) - set(affected))
    if stray and 'tooling' not in kinds:
        report['notes'].append(f'{len(stray)} DTB(s) changed output without a tracked '
                               f'dependency on a changed file: '
                               + ', '.join(short_dtb(d, args.arch) for d in stray[:10]))
    unused = sorted(p for p in dts_head if not any(p in i['deps'] for i in H.dtbs.values())
                    and p not in {e for errs in H.failed.values() for e in [x.get('file') for x in errs]})
    if unused:
        report['notes'].append('changed DTS files not used by any DTB built with this config: '
                               + ', '.join(unused[:20]))

    # 2. problems that cost nothing extra: build failures and dtc warnings
    comparable = {d for d in head_ok if d in base_ok or d in new_dtbs}
    for side in (B, H):
        for d, errs in side.failed.items():
            e0 = next((e for e in errs if e.get('file')), errs[0] if errs else
                      {'file': None, 'line': None, 'msg': 'build failed', 'text': ''})
            side.problems.append(problem(
                'dtb-build', d, ('build-failed',),
                f"{short_dtb(d, args.arch)} does not build: {e0['msg']}",
                file=e0.get('file'), line=int(e0['line']) if e0.get('line') else None,
                detail=[e['text'] for e in errs[:10]]))
        for d, info in side.dtbs.items():
            if (head_name(d) if side is B else d) not in comparable:
                continue
            for w in info['warnings']:
                side.problems.append(problem(
                    'dtc-warning', d, (w['check'], mask(w['msg'])),
                    f"dtc {w['check']}: {w['msg']}", file=w['file'],
                    line=int(w['line']) if w['line'] else None,
                    detail=[f'also defined at {a}' for a in w['also']]))
    # a DTB that already failed at base is not new breakage
    for d in list(head_failed):
        if d in base_failed:
            report['notes'].append(f'{short_dtb(d, args.arch)} fails to build at base and head')

    # 3. which DTBs get schema checks
    full = set() if args.no_schema else {d for d in changed_output}
    if args.full and not args.no_schema:
        full = set(head_ok)
    limited, limit = set(), None
    if (bind_head or bind_base) and not args.no_schema:
        with timings('binding-closure'):
            clos_h = binding_closure(H.src, bind_head)
            clos_b = binding_closure(B.src, bind_base)
            tok_h, all_h = binding_selectors(H.src, clos_h)
            tok_b, all_b = binding_selectors(B.src, clos_b)
            closure = sorted({renames.get(p, p) for p in clos_b} | clos_h)
            # dt-core.yaml carries the property type checks; a binding change that
            # drops a property definition shows up there, not in the binding itself
            limit = ':'.join([p[len(BINDINGS) + 1:] for p in closure] + ['dt-core.yaml'])
            tokens = tok_h | tok_b
            for d in head_ok:
                if d in full:
                    continue
                if all_h or all_b:
                    limited.add(d)
                    continue
                comp = fdt_compatibles(H.dtbs[d]['path'])
                if d in base_ok:
                    comp |= fdt_compatibles(B.dtbs[base_ok[d]]['path'])
                if comp & tokens:
                    limited.add(d)
        report['binding_closure'] = closure
    need_validate = bool(full or limited)
    need_bcheck = bool(bind_head or bind_base or 'binding-infra' in kinds)

    # 4. processed schema for each side that needs one
    if need_validate or need_bcheck:
        with timings('schema'):
            keys = {n: schema_cache_key(args, s.sha, dtschema_version) for n, s in sides.items()}
            need = {'head': True,
                    'base': bool(full & set(base_ok)) or bool(limited & set(base_ok)) or bool(bind_base)
                    or 'binding-infra' in kinds}
            shared = keys['base'] == keys['head']
            todo = ['head'] if shared else [n for n in ('base', 'head') if need[n]]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                futs = {n: ex.submit(build_schema, args, sides[n], env, args.cache_dir, keys[n])
                        for n in todo}
                for n, f in futs.items():
                    probs, hit = f.result()
                    sides[n].problems.extend(probs)
                    report['schema_cache'][n] = 'hit' if hit else 'built'
            if shared and need['base']:
                install_schema(B, H.schema, H.schema_cmd or (
                    open(os.path.join(H.obj, BINDINGS, '.processed-schema.json.cmd')).read()
                    .replace(H.src, '@SRCTREE@')
                    if os.path.exists(os.path.join(H.obj, BINDINGS, '.processed-schema.json.cmd'))
                    else None))
                B.problems.extend(p for p in H.problems if p['class'] in ('binding-dropped', 'binding-warning'))
                report['schema_cache']['base'] = 'same as head'

    # 5. dt-validate
    validated = {'base': {}, 'head': {}}
    if need_validate:
        with timings('dt-validate'):
            tasks = []
            if full:
                tasks.append(('head', None, sorted(full)))
                bfull = sorted(base_ok[d] for d in full if d in base_ok)
                if bfull:
                    tasks.append(('base', None, bfull))
            if limited:
                tasks.append(('head', limit, sorted(limited)))
                blim = sorted(base_ok[d] for d in limited if d in base_ok)
                if blim:
                    tasks.append(('base', limit, blim))
            res, validated, errors = run_validation(args, sides, tasks, env, json_ok, validate_cache)
            for e in errors:
                report['notes'].append(e)
            if errors:
                report['tool_failures'] = errors
            ok_pairs = {d for d in validated['head']
                        if d not in base_ok or base_ok[d] in validated['base']}
            for n, side in sides.items():
                for p in res[n]:
                    hd = head_name(p['scope']) if n == 'base' else p['scope']
                    if hd in ok_pairs:
                        side.problems.append(p)

    # 6. binding checks
    bfiles = {'base': bind_base, 'head': bind_head}
    if 'binding-infra' in kinds:
        for n, s in sides.items():
            extra = scope_bindings(s.src, args.binding_scope)
            bfiles[n] = sorted(set(bfiles[n]) | set(extra))
    if need_bcheck:
        with timings('dt_binding_check'):
            def bcheck(n):
                s = sides[n]
                files = [f for f in bfiles[n] if os.path.exists(os.path.join(s.src, f))]
                if not files:
                    return n, []
                rel = ':'.join(f[len(BINDINGS) + 1:] for f in files)
                rc, out_ = run(['make', '-C', s.src, f'O={s.obj}', f'ARCH={args.arch}',
                                f'-j{max(1, args.jobs // 2)}', '-k', '-Otarget',
                                'dt_binding_check', f'DT_SCHEMA_FILES={rel}'], env=env,
                               logfile=os.path.join(s.logs, f'{n}-binding-check.log'))
                probs = parse_binding_log(s, out_)
                for p in probs:
                    if p['class'] in ('example-schema', 'example-build') and p['file'] and not p['line']:
                        p['line'] = examples_line(s.src, p['file'])
                    elif p['class'] in ('binding-invalid', 'binding-dropped', 'binding-warning') \
                            and p['file'] and not p['line']:
                        p['line'] = yaml_key_line(s.src, p['file'], p['message'])
                return n, probs
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                for n, probs in ex.map(bcheck, ['base', 'head']):
                    sides[n].problems.extend(probs)
        report['bindings_checked'] = bfiles['head']

    # 7. style of changed DTS sources
    if dts_head and not args.no_style:
        with timings('dts-style'):
            py = shutil.which('python3') or sys.executable
            for n, files in (('base', dts_base), ('head', dts_head)):
                sides[n].problems.extend(style_check(sides[n], files, args.style_mode, py, env))

    # 8. compare
    with timings('compare'):
        new, fixed = compare(B.problems, H.problems, scope_map)
        # style findings in a file that no longer compiles are knock-on noise
        broken = {e.get('file') for errs in H.failed.values() for e in errs if e.get('file')}
        if broken:
            dropped = [p for p in new if p['class'] == 'style' and p['scope'] in broken]
            if dropped:
                new = [p for p in new if p not in dropped]
                report['notes'].append(f'{len(dropped)} new style finding(s) not reported for '
                                       'files that fail to compile')
        # annotate schema problems with source lines from the head DTB
        need_map = sorted({p['scope'] for p in new if p['class'] in ('schema', 'unmatched')})
        maps = {}
        for d in need_map[:60]:
            maps[d] = source_map(H, d, args.arch)
        for p in new:
            if p['class'] not in ('schema', 'unmatched') or p['scope'] not in maps:
                continue
            nodes, props = maps[p['scope']]
            node = p.get('node') or p['key'][0]
            cands = []
            prop = p.get('prop')
            if prop and (node, prop) in props:
                cands = props[(node, prop)]
            elif prop and (node.rstrip('/') + '/' + prop) in nodes:
                cands = nodes[node.rstrip('/') + '/' + prop]
            elif node in nodes:
                cands = nodes[node]
            if cands:
                p['file'], p['line'] = pick_location(cands, changed_head)
            elif H.dtbs.get(p['scope'], {}).get('source'):
                p['file'] = H.dtbs[p['scope']]['source']
        blocking = set(args.block)
        groups = group_new(new, blocking, changed_head, args.arch)
        counts = collections.OrderedDict()
        for cls in CLASSES:
            b = sum(1 for p in B.problems if p['class'] == cls)
            h = sum(1 for p in H.problems if p['class'] == cls)
            nn = sum(1 for p in new if p['class'] == cls)
            ff = sum(1 for p in fixed if p['class'] == cls)
            if b or h or nn or ff:
                counts[cls] = {'base': b, 'head': h, 'new': nn, 'fixed': ff}

    report['counts'] = counts
    report['new'] = groups
    report['fixed'] = [{'class': p['class'], 'scope': p['scope'], 'message': p['message']}
                       for p in fixed[:300]]
    report['fixed_total'] = len(fixed)
    report['dtbs'] = {
        'base_built': len(B.dtbs), 'head_built': len(H.dtbs),
        'head_failed': sorted(short_dtb(d, args.arch) for d in head_failed),
        'new': [short_dtb(d, args.arch) for d in new_dtbs],
        'affected': {short_dtb(d, args.arch): v for d, v in sorted(affected.items())},
        'changed_output': [short_dtb(d, args.arch) for d in changed_output],
        'validated_full': sorted(short_dtb(d, args.arch) for d, m in validated['head'].items() if m == 'full'),
        'validated_limited': sorted(short_dtb(d, args.arch) for d, m in validated['head'].items() if m == 'limited'),
    }
    report['bindings_checked'] = report.get('bindings_checked', [])
    if any(g['class'] == 'binding-dropped' for g in groups):
        report['notes'].append('a binding was dropped from the head schema, so DTB results '
                               'for nodes it covered (including "fixed" counts) are not meaningful')
    annotate(groups, args.max_annotations)
    if report.get('tool_failures'):
        report['status'] = 'error'
        report['error'] = '; '.join(report['tool_failures'][:3])
        return 2
    nblock = sum(1 for g in groups if g['blocking'])
    report['status'] = 'fail' if nblock else 'pass'
    return 1 if nblock else 0


def scope_bindings(src, patterns):
    res = []
    broot = os.path.join(src, BINDINGS)
    for root, _dirs, files in os.walk(broot):
        for n in files:
            if not n.endswith('.yaml') or n.startswith('processed-schema'):
                continue
            rel = os.path.relpath(os.path.join(root, n), src)
            if any(p in n for p in patterns):
                res.append(rel)
                continue
            try:
                text = open(os.path.join(root, n), encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            if any(p + ',' in text for p in patterns):
                res.append(rel)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, help='base commit (PR target)')
    ap.add_argument('--head', required=True, help='head commit (PR result)')
    ap.add_argument('--tree', default='.', help='git repository holding both commits')
    ap.add_argument('--out', required=True, help='work and report directory')
    ap.add_argument('--arch', default='arm64')
    ap.add_argument('--kconfig', action='append', default=None,
                    help='config line for allnoconfig (repeatable; default CONFIG_ARCH_APPLE=y)')
    ap.add_argument('--jobs', '-j', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--block', default=','.join(DEFAULT_BLOCKING),
                    help='comma-separated problem classes that fail the check '
                         f'(default {",".join(DEFAULT_BLOCKING)}; classes: {",".join(CLASSES)})')
    ap.add_argument('--binding-scope', default='apple',
                    help='comma-separated name patterns for bindings rechecked when binding '
                         'tooling changes (default apple)')
    ap.add_argument('--style-mode', default='strict', choices=('relaxed', 'strict'),
                    help='dt-check-style mode for changed DTS files (default strict)')
    ap.add_argument('--cache-dir', help='reuse processed schemas and dt-validate results')
    ap.add_argument('--full', action='store_true', help='schema-check every DTB fully')
    ap.add_argument('--no-schema', action='store_true', help='skip dt-validate on DTBs')
    ap.add_argument('--no-style', action='store_true', help='skip dt-check-style on DTS files')
    ap.add_argument('--max-annotations', type=int, default=40)
    ap.add_argument('--keep', action='store_true', help='keep extracted sources and build dirs')
    args = ap.parse_args(argv)
    args.tree = os.path.abspath(args.tree)
    args.out = os.path.abspath(args.out)
    args.kconfig = args.kconfig or ['CONFIG_ARCH_APPLE=y']
    args.block = [b for b in args.block.split(',') if b]
    args.binding_scope = [b for b in args.binding_scope.split(',') if b]
    args.cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else None
    for b in args.block:
        if b not in CLASSES:
            ap.error(f'unknown class {b}')
    os.makedirs(args.out, exist_ok=True)

    timings = Timings()
    report = {'tool': 'dt-check.py', 'base': args.base, 'head': args.head,
              'merge_base_used': False, 'status': None, 'blocking_classes': args.block,
              'classes': CLASSES, 'versions': tool_versions(), 'notes': [],
              'schema_cache': {}, 'timings_s': timings.t}
    t0 = time.monotonic()
    try:
        rc = check(args, report, timings)
    except SetupError as e:
        report['status'] = 'error'
        report['error'] = str(e)
        print(f'::error title=dt-check::{esc_data(str(e))}', flush=True)
        rc = 2
    timings.t['total'] = round(time.monotonic() - t0, 2)
    report['exit_code'] = rc
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, default=sorted)
        f.write('\n')
    text = write_summary(os.path.join(args.out, 'summary.md'), report)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(text)
    if not args.keep:
        for d in ('src', 'build', 'tmp'):
            shutil.rmtree(os.path.join(args.out, d), ignore_errors=True)
    note(f"status {report['status']}, exit {rc}, {timings.t['total']}s")
    return rc


if __name__ == '__main__':
    sys.exit(main())
