"""
python3 mulit_run_case.py test/test_ops.py
python3 mulit_run_case.py test/test_ops.py --crash-list skip.txt --fail-list failed.txt
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
CHILD_FLAG = "--resilient-child"
CHILD_ROOT_FLAG = "--resilient-repo-root"
ROOT_FLAG = "--repo-root"
CRASH_LIST_FLAG = "--crash-list"
CRASH_LIST_ENV = "RESILIENT_CRASH_LIST"
FAIL_LIST_FLAG = "--fail-list"
ALLOW_EMPTY_FLAG = "--allow-empty"
PLUGIN_MODULE = os.path.splitext(os.path.basename(__file__))[0]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_CWD = tempfile.gettempdir()
POLL_SECONDS = 5
OK_STATUSES = frozenset({"PASSED", "SKIPPED"})


def _parse_cli(argv):
    opts, rest = {}, []
    i = 0
    while i < len(argv):
        a = argv[i]
        name, eq, value = a.partition("=")
        if a == ALLOW_EMPTY_FLAG:
            opts[ALLOW_EMPTY_FLAG] = True
        elif name in (ROOT_FLAG, CRASH_LIST_FLAG, FAIL_LIST_FLAG):
            if not eq:
                if i + 1 >= len(argv):
                    raise ValueError(f"{a} requires a value")
                i += 1
                value = argv[i]
            opts[name] = value
        else:
            rest.append(a)
        i += 1
    return opts, rest


@dataclass(frozen=True)
class Config:

    repo_root: str
    progress_file: str
    per_test_timeout: int
    collect_timeout: int
    pytest_args: tuple = ()
    crash_list: str = ""
    fail_list: str = ""
    allow_empty: bool = False

    @classmethod
    def from_env(cls, argv, repo_root, progress_file, crash_list=None,
                 fail_list=None, allow_empty=False):
        crash_list = crash_list or os.environ.get(CRASH_LIST_ENV, "")
        return cls(
            repo_root=os.path.abspath(repo_root or os.getcwd()),
            progress_file=progress_file,
            per_test_timeout=int(os.environ.get("PER_TEST_TIMEOUT", "90")),
            collect_timeout=int(os.environ.get("COLLECT_TIMEOUT", "900")),
            pytest_args=tuple(argv),
            crash_list=os.path.abspath(crash_list) if crash_list else "",
            fail_list=os.path.abspath(fail_list) if fail_list else "",
            allow_empty=allow_empty,
        )


# ---------------------------------------------------------------------------
# 2. Progress
# ---------------------------------------------------------------------------

class ProgressLog:

    START, RESULT, TOTAL = "START", "RESULT", "TOTAL"

    def __init__(self, path: str):
        self.path = path

    def record_start(self, nodeid: str) -> None:
        self._append(self.START, nodeid)

    def record_result(self, nodeid: str, status: str) -> None:
        self._append(self.RESULT, nodeid, status)

    def record_total(self, total: int) -> None:
        self._append(self.TOTAL, str(total))

    def _append(self, *fields: str) -> None:
        with open(self.path, "a") as f:
            f.write("\t".join(fields) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return -1

    def _lines(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                yield line.rstrip("\n").split("\t")

    def done_nodeids(self) -> set:
        return {p[1] for p in self._lines()
                if p[0] == self.RESULT and len(p) >= 3}

    def stuck_nodeid(self):
        running = None
        for p in self._lines():
            if p[0] == self.START:
                running = p[1]
            elif p[0] == self.RESULT and len(p) >= 3 and running == p[1]:
                running = None
        return running

    def total_cases(self):
        total = None
        for p in self._lines():
            if p[0] == self.TOTAL and len(p) >= 2:
                total = int(p[1])
        return total

    def results(self) -> dict:
        """Map nodeid -> status; a non-OK status wins over an earlier OK one.

        A case can be reported twice (e.g. the call passes but teardown errors);
        keeping only the first record would hide the failure from the summary.
        """
        results = {}
        for p in self._lines():
            if p[0] != self.RESULT or len(p) < 3:
                continue
            nodeid, status = p[1], p[2]
            previous = results.get(nodeid)
            if previous is None or (previous in OK_STATUSES
                                    and status not in OK_STATUSES):
                results[nodeid] = status
        return results

    def status_counts(self) -> dict:
        counts = {}
        for status in self.results().values():
            counts[status] = counts.get(status, 0) + 1
        return counts

    def remove(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


class CrashList:

    def __init__(self, path: str):
        self.path = path

    def entries(self):
        """Yield (section, entry, note); section is the file of the enclosing [path] header, None if global."""
        if not self.path or not os.path.exists(self.path):
            return
        section = None
        with open(self.path) as f:
            for line in f:
                text, _, note = line.partition("\t")
                text, note = text.strip(), note.strip()
                if not text or text.startswith("#"):
                    continue
                if text.startswith("[") and text.endswith("]"):
                    # "[]" switches back to global entries
                    section = os.path.normpath(text[1:-1].strip()) if text[1:-1].strip() else None
                    continue
                yield section, text, note

    def nodeids(self) -> set:
        return {entry for _, entry, _ in self.entries()}

    def matching_nodeids(self, items) -> set:
        # Full nodeids match exactly wherever they appear; global patterns are
        # case-sensitive substrings over every file; patterns under a [path] section
        # only apply to that file and are case-insensitive, mirroring a per-file -k "not ...".
        exact, patterns, scoped = set(), [], {}
        for section, entry, _ in self.entries():
            if "::" in entry:
                exact.add(entry)
            elif section is None:
                patterns.append(entry)
            else:
                scoped.setdefault(section, []).append(entry.lower())

        def listed(nodeid):
            if nodeid in exact or any(p in nodeid for p in patterns):
                return True
            file_scoped = scoped.get(nodeid.split("::", 1)[0], ())
            lowered = nodeid.lower()
            return any(p in lowered for p in file_scoped)

        return {item.nodeid for item in items if listed(item.nodeid)}

    def add(self, nodeid: str, status: str) -> bool:
        """Append a record; return False if no path is configured or the full nodeid already exists."""
        if not self.path or nodeid in self.nodeids():
            return False
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # A hand-edited list may lack the trailing newline; appending directly would
        # glue the new record onto the last entry and silently break both.
        needs_newline = False
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
        with open(self.path, "a") as f:
            f.write(("\n" if needs_newline else "") + f"{nodeid}\t{status}\n")
            f.flush()
            os.fsync(f.fileno())
        return True


class FailureList:
    """Cases that did not pass, grouped per test file in crash list syntax."""

    DEFAULT_HEADER = (
        "# Cases that did not pass (FAILED / ERROR / CRASH / HANG), written by",
        "# multi_run_case.py --fail-list; usable as is for --crash-list.",
        "# - Lines outside any [path] section: case-sensitive substring, applies to every file.",
        '# - Lines under [path]: only apply to that file, case-insensitive substring (same as -k "not ...").',
        '# - Full nodeids (with "::") match exactly wherever they appear.',
    )

    def __init__(self, path: str):
        self.path = path

    def write(self, nodeids) -> int:
        """Merge nodeids into the file as per-file entries; return the total entry count.

        The leading comment block and the existing entries are kept, so repeated
        runs over different test files accumulate into one reusable crash list.
        """
        if not self.path:
            return 0
        header, globals_, scoped = self._read()
        for nodeid in nodeids:
            path, sep, _ = nodeid.partition("::")
            # A nodeid without "::" is a whole-file error; keeping it global avoids
            # turning the file path into a pattern that silences the whole file.
            bucket = scoped.setdefault(os.path.normpath(path), []) if sep else globals_
            entry = nodeid
            if all(entry != existing for existing, _ in bucket):
                bucket.append((entry, ""))
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w") as f:
            for line in header:
                f.write(line + "\n")
            # Global entries must precede the first [path] header to stay global.
            if globals_:
                f.write("\n")
                for entry, note in globals_:
                    f.write(self._line(entry, note))
            for section in sorted(scoped):
                f.write(f"\n[{section}]\n")
                for entry, note in scoped[section]:
                    f.write(self._line(entry, note))
            f.flush()
            os.fsync(f.fileno())
        return len(globals_) + sum(len(entries) for entries in scoped.values())

    def _read(self):
        """Return the leading comment block, the global entries and the per-file entries."""
        header, globals_, scoped = [], [], {}
        if not os.path.exists(self.path):
            return list(self.DEFAULT_HEADER), globals_, scoped
        with open(self.path) as f:
            for line in f:
                text = line.strip()
                if text and not text.startswith("#"):
                    break
                header.append(line.rstrip("\n"))
        while header and not header[-1].strip():
            header.pop()
        for section, entry, note in CrashList(self.path).entries():
            bucket = globals_ if section is None else scoped.setdefault(section, [])
            bucket.append((entry, note))
        return header or list(self.DEFAULT_HEADER), globals_, scoped

    @staticmethod
    def _line(entry: str, note: str) -> str:
        return f"{entry}\t{note}\n" if note else f"{entry}\n"


# ---------------------------------------------------------------------------
# 3. pytest plugin role (the child process loads the hooks below via -p <this module>)
# ---------------------------------------------------------------------------

class _RepoSourceGuard:

    GUARDED = ("torch", "torchgen", "functorch")

    def __init__(self, repo_root: str):
        self._repo_root = os.path.abspath(repo_root)

    def find_spec(self, fullname, path=None, target=None):
        from importlib.machinery import PathFinder
        root, name = fullname, PathFinder
        if root not in self.GUARDED or path is not None:
            return None
        spec_repo = name.find_spec(root, path)
        if spec_repo is None or not self._inside_repo(spec_repo):
            return None
        clean = [p for p in sys.path
                 if p and os.path.abspath(p) != self._repo_root]
        spec_installed = name.find_spec(root, clean)
        if spec_installed is None:
            return spec_repo
        if self._is_dev_build(spec_repo):
            return spec_repo
        print(f"[resilient] guard: import {root!r} redirected to installed "
              f"package ({spec_installed.origin}); repo source copy "
              f"{spec_repo.origin} would shadow it", file=sys.stderr)
        return spec_installed

    def _inside_repo(self, spec) -> bool:
        loc = spec.origin or (spec.submodule_search_locations or [""])[0]
        try:
            return os.path.commonpath(
                [os.path.abspath(loc), self._repo_root]) == self._repo_root
        except ValueError:
            return False

    @staticmethod
    def _is_dev_build(spec) -> bool:

        import glob
        pkg_dir = os.path.dirname(spec.origin or "")
        if not pkg_dir:
            return False
        return any(
            os.path.isfile(p)
            for pat in ("_C*.so", "_C*.pyd")
            for p in glob.glob(os.path.join(pkg_dir, pat))
        )


_PLUGIN_LOG = None
_COLLECT_ERRORS = []


def _plugin_log() -> ProgressLog:
    global _PLUGIN_LOG
    if _PLUGIN_LOG is None:
        _PLUGIN_LOG = ProgressLog(
            os.environ.get("RESILIENT_PROGRESS_FILE", "test_progress.log")
        )
    return _PLUGIN_LOG


def pytest_collection_modifyitems(config, items):
    done = _plugin_log().done_nodeids()
    crash_list = CrashList(os.environ.get(CRASH_LIST_ENV, ""))
    listed = crash_list.matching_nodeids(items)
    skip = done | listed
    if not skip:
        return
    deselected = [it for it in items if it.nodeid in skip]
    if deselected:
        config.hook.pytest_deselected(items=deselected)

        items[:] = [it for it in items if it.nodeid not in skip]
    # Only report once in the first round (no progress yet) to avoid repeating it every round
    if listed and not done:
        print(f"\n[resilient] crash list: deselected {len(listed)} listed "
              f"case(s) from {crash_list.path}", flush=True)


def pytest_collection_finish(session):
    log = _plugin_log()
    done = log.done_nodeids()
    total = len(done) + len(session.items)
    log.record_total(total)
    print(f"\n[resilient] cumulative progress: done={len(done)}/{total}, "
          f"remaining={len(session.items)}", flush=True)
    # Flushed here, not from pytest_collectreport: the watchdog treats the first
    # write of a round as "collection is over", so writing during collection
    # would cut its budget from COLLECT_TIMEOUT down to PER_TEST_TIMEOUT.
    for nodeid in _COLLECT_ERRORS:
        log.record_result(nodeid, "ERROR")
    _COLLECT_ERRORS.clear()


def pytest_runtest_logstart(nodeid, location):
    _plugin_log().record_start(nodeid)


def pytest_runtest_logreport(report):
    log = _plugin_log()
    if report.when == "call":
        log.record_result(report.nodeid, report.outcome.upper())
    elif report.when == "setup" and report.outcome != "passed":
        status = "SKIPPED" if report.skipped else "ERROR"
        log.record_result(report.nodeid, status)
    elif report.when == "teardown" and report.failed:
        # pytest fails the whole run on a teardown error; without this record the
        # summary would report the case as passed and the script would exit 0.
        log.record_result(report.nodeid, "ERROR")



def pytest_collectreport(report):
    # A module that fails to import never reaches pytest_runtest_logreport, so
    # without this the round would look clean and the broken file would be
    # missing from both the summary and the fail list. Buffered until collection
    # is finished, see pytest_collection_finish.
    if report.failed and report.nodeid:
        _COLLECT_ERRORS.append(report.nodeid)


# ---------------------------------------------------------------------------
# 4. Monitoring
# ---------------------------------------------------------------------------

class Watchdog:

    def __init__(self, proc, progress: ProgressLog, cfg: Config, start_size: int):
        self._proc = proc
        self._progress = progress
        self._cfg = cfg
        self._start_size = start_size
        self._stop = threading.Event()
        self._thread = None
        self.killed = False
        self.killed_in_collection = False

    def start(self) -> "Watchdog":
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def wait(self) -> int:
        rc = self._proc.wait()
        self._stop.set()
        self._thread.join(timeout=POLL_SECONDS + 2)
        return rc

    def _watch(self) -> None:
        # Nothing is written during collection; the first write of a round is the
        # TOTAL record from pytest_collection_finish, so until the file grows past
        # its round-start size the child is still collecting and gets the longer budget.
        # Keep every other hook from writing before TOTAL or this breaks.
        last_size, last_change = self._start_size, time.time()
        while not self._stop.wait(POLL_SECONDS):
            size = self._progress.size()
            collecting = size == self._start_size
            timeout = (self._cfg.collect_timeout if collecting
                       else self._cfg.per_test_timeout)
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > timeout:
                self.killed = True
                self.killed_in_collection = collecting
                self._kill_group()
                return

    def _kill_group(self) -> None:
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGINT)
            time.sleep(10)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# 5. orchestrator
# ---------------------------------------------------------------------------

def _abs_arg(arg: str, cfg: Config) -> str:
    if arg.startswith("-"):
        return arg
    path, sep, rest = arg.partition("::")
    candidate = os.path.join(cfg.repo_root, path)
    if os.path.exists(candidate) or path.endswith(".py") or os.sep in path:
        return os.path.abspath(candidate) + sep + rest
    return arg


def _child_env(cfg: Config) -> dict:
    env = dict(os.environ)
    clean = [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and os.path.abspath(p) not in (cfg.repo_root, SCRIPT_DIR)
    ]
    env["PYTHONPATH"] = os.pathsep.join([SCRIPT_DIR, *clean])
    env["PYTHONSAFEPATH"] = "1"
    if cfg.crash_list:
        env[CRASH_LIST_ENV] = cfg.crash_list
    else:
        env.pop(CRASH_LIST_ENV, None)
    env["RESILIENT_PROGRESS_FILE"] = cfg.progress_file
    return env


def _has_verbosity_flag(args) -> bool:
    for a in args:
        if a in ("--verbose", "--quiet"):
            return True
        if a.startswith("-") and not a.startswith("--"):
            body = a[1:]
            if "v" in body or "q" in body:
                return True
    return False


def _spawn_child(cfg: Config) -> subprocess.Popen:
    user_args = [_abs_arg(a, cfg) for a in cfg.pytest_args]
    if not _has_verbosity_flag(user_args):
        user_args = ["-v", *user_args]
    cmd = [
        sys.executable, os.path.abspath(__file__), CHILD_FLAG,
        CHILD_ROOT_FLAG, cfg.repo_root,
        "-p", PLUGIN_MODULE,
        "-p", "no:cacheprovider",
        *user_args,
    ]

    return subprocess.Popen(
        cmd, env=_child_env(cfg), cwd=RUN_CWD, preexec_fn=os.setsid
    )


def _print_summary(log: ProgressLog) -> dict:
    counts = log.status_counts()
    if not counts:
        print("no progress recorded")
        return counts
    total = sum(counts.values())
    print("\n================ SUMMARY ================")
    print(f"total finished cases: {total}")
    for status in sorted(counts):
        rate = counts[status] / total * 100
        print(f"  {status:<16} {counts[status]:>6}  ({rate:5.1f}%)")
    print("=========================================")
    return counts


def _flush_fail_list(cfg: Config, log: ProgressLog):
    """Merge the non-passing cases into the fail list; return (cases, entries)."""
    if not cfg.fail_list:
        return 0, 0
    failures = [nodeid for nodeid, status in log.results().items()
                if status not in OK_STATUSES]
    if not failures:
        return 0, 0
    return len(failures), FailureList(cfg.fail_list).write(failures)


def orchestrate(cfg: Config) -> int:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(f"ERROR: pytest not found for {sys.executable}; "
              f"run with a python that has pytest installed.")
        return 2

    log = ProgressLog(cfg.progress_file)
    crash_list = CrashList(cfg.crash_list)
    if cfg.crash_list:
        print(f"crash list: {cfg.crash_list} "
              f"({len(crash_list.nodeids())} exclusion entries)")
    newly_listed = []
    rounds = 0
    aborted = False
    while True:
        rounds += 1
        prev_done = log.done_nodeids()
        total = log.total_cases()
        progress = (
            f"done={len(prev_done)}/{total}"
            if total is not None else f"done={len(prev_done)}/?"
        )
        print(f"\n===== round {rounds}: {progress} =====")

        start_size = log.size()
        watchdog = Watchdog(_spawn_child(cfg), log, cfg, start_size).start()
        rc, killed = watchdog.wait(), watchdog.killed

        stuck = log.stuck_nodeid()
        if stuck:
            status = "HANG" if killed else "CRASH"
            log.record_result(stuck, status)
            print(f"  detected stuck case -> marked {status}: {stuck}")
            if crash_list.add(stuck, status):
                newly_listed.append((stuck, status))
            # Refresh the fail list every round so a CI job killed by its own
            # timeout still leaves the crash/hang findings on disk.
            _flush_fail_list(cfg, log)
            continue

        if killed:
            if watchdog.killed_in_collection:
                print(f"  ERROR: collection produced no progress for "
                      f"{cfg.collect_timeout}s, killed by watchdog "
                      f"(raise COLLECT_TIMEOUT if collection is just slow).")
            else:
                print(f"  ERROR: no progress for {cfg.per_test_timeout}s outside "
                      f"any test case, killed by watchdog.")
            aborted = True
            break

        if rc in (0, 1, 5):
            print(f"  pytest exited normally (rc={rc}); all remaining cases done.")
            break

        if len(log.done_nodeids()) <= len(prev_done):
            print(f"  ERROR: abnormal exit rc={rc} with no progress; aborting to "
                  f"avoid an infinite loop.")
            aborted = True
            break

    counts = _print_summary(log)
    if cfg.fail_list:
        cases, entries = _flush_fail_list(cfg, log)
        if cases:
            print(f"fail list: merged {cases} non-passing case(s) into "
                  f"{cfg.fail_list} ({entries} entries in total)")
        else:
            print(f"fail list: no non-passing case, {cfg.fail_list} left untouched")
    if newly_listed:
        print(f"appended {len(newly_listed)} new case(s) to crash list "
              f"{cfg.crash_list} (deselected from next run on):")
        for nodeid, status in newly_listed:
            print(f"  {status:<6} {nodeid}")
    if aborted:
        print("FAIL: run aborted before all selected cases finished, see the "
              "round log above.")
        return 1
    if not counts:
        # 仅放行正常收集后的空选择；导入错误、超时和异常退出仍然失败。
        if cfg.allow_empty and rc == 5 and log.total_cases() == 0:
            print("[skip] 正常收集后没有匹配用例，按 --allow-empty 跳过。")
            return 0
        print("FAIL: zero cases finished (nothing collected/selected); check the "
              "-k expression, the crash list and whether the device is visible.")
        return 1
    return 1 if set(counts) - OK_STATUSES else 0


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def _run_as_child(args, repo_root) -> int:
    repo_root = os.path.abspath(repo_root)
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != repo_root]

    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)
    sys.meta_path.insert(0, _RepoSourceGuard(repo_root))
    import pytest
    return pytest.main(list(args))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if CHILD_FLAG in argv:
        rest, repo_root, want_root = [], os.getcwd(), False
        for a in argv:
            if want_root:
                repo_root, want_root = a, False
            elif a == CHILD_ROOT_FLAG:
                want_root = True
            elif a != CHILD_FLAG:
                rest.append(a)
        return _run_as_child(rest, repo_root)

    try:
        opts, user_args = _parse_cli(argv)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2

    fd, progress_file = tempfile.mkstemp(prefix="resilient_progress_", suffix=".log")
    os.close(fd)
    try:
        cfg = Config.from_env(user_args or ["test/test_ops.py"],
                              repo_root=opts.get(ROOT_FLAG),
                              progress_file=progress_file,
                              crash_list=opts.get(CRASH_LIST_FLAG),
                              fail_list=opts.get(FAIL_LIST_FLAG),
                              allow_empty=opts.get(ALLOW_EMPTY_FLAG, False))
        return orchestrate(cfg)
    finally:
        ProgressLog(progress_file).remove()


if __name__ == "__main__":
    sys.exit(main())