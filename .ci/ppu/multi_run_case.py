"""断点续跑 + 抗 hang / 抗 crash 的 pytest 执行器（单文件自包含）。
CUDA_VISIBLE_DEVICES=9 python3 /any/where/mulit_run_case.py test/test_ops.py

子进程行为：
- 正常结束（含用例失败）           -> 全部跑完，退出循环；
- 进程崩溃 / 被信号杀死（segfault）-> 记录当前正在跑的 case 为 CRASH，续跑剩余；
- case 卡死（进度文件长时间无更新） -> 看门狗杀掉整个进程组，记录为 HANG，续跑剩余。

无论中断多少次，最终都能把所有 case 跑完，并给出成功/失败/hang 统计。
退出码：存在 FAILED / ERROR / HANG / CRASH 时为 1，否则为 0（便于接入 CI）。

可调参数：
- --repo-root PATH        : 相对测试路径的解析基准 + 需从 PYTHONPATH 剔除的
                            仓库根（默认当前工作目录）
- PER_TEST_TIMEOUT（环境变量）: 进度文件多少秒没有新行就判定为 hang（默认 60 秒）
                            必须大于单个正常 case 的最长耗时，否则会误杀慢用例。

代码分层（自上而下）：
1. Config        —— 全部可调参数，单点收敛；
2. ProgressLog   —— 进度文件协议的唯一定义处（父子进程共用）；
3. pytest hooks  —— 插件角色：写进度 + 收集阶段 deselect；
4. Watchdog      —— 无进展超时则整组杀进程；
5. Orchestrator  —— 主循环：拉起子进程、补记终态、汇总、定退出码；
6. CLI           —— 按 --resilient-child 分派父/子身份。
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
# 1. 配置
# ---------------------------------------------------------------------------

# 子进程身份标记：argv 里带它 => 本次是 pytest 子进程，否则是 orchestrator
CHILD_FLAG = "--resilient-child"
# 内部选项：父进程把 repo_root 传给子进程，供 import 守卫使用（用户无需关心）
CHILD_ROOT_FLAG = "--resilient-repo-root"
# --repo-root PATH：相对测试路径的解析基准，默认当前工作目录
ROOT_FLAG = "--repo-root"
# 本模块名：子进程用 -p <本模块> 加载自身作为 pytest 插件（随文件重命名自动跟随）
PLUGIN_MODULE = os.path.splitext(os.path.basename(__file__))[0]
# 本脚本所在目录：放进子进程 PYTHONPATH，供 -p <本模块> 导入自身
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 子进程工作目录：中性目录，避免仓库根进入 sys.path 抢占已安装的编译版 torch
RUN_CWD = tempfile.gettempdir()
# 看门狗轮询间隔（秒）
POLL_SECONDS = 5
# 视为"成功"的终态，其余（FAILED/ERROR/HANG/CRASH...）都会让退出码为 1
OK_STATUSES = frozenset({"PASSED", "SKIPPED"})


def _parse_cli(argv):
    """摘出本脚本自有的选项；返回 (repo_root, 透传给 pytest 的参数)。"""
    root, rest = None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == ROOT_FLAG:
            if i + 1 >= len(argv):
                raise ValueError(f"{a} requires a value")
            i += 1
            root = argv[i]
        elif a.startswith(ROOT_FLAG + "="):
            root = a.partition("=")[2]
        else:
            rest.append(a)  # 测试目标 / nodeid / -k 等原样透传给 pytest
        i += 1
    return root, rest


@dataclass(frozen=True)
class Config:
    """运行时可变参数的单点定义；from_env() 生成默认配置。

    固定量（脚本目录 SCRIPT_DIR、子进程 cwd RUN_CWD、轮询间隔 POLL_SECONDS）
    已提到模块级常量，不占配置字段。
    """

    repo_root: str          # 相对测试路径解析基准 + 从子进程 PYTHONPATH 剔除的根
    progress_file: str      # 内部临时进度文件（父子进程经环境变量共享）
    per_test_timeout: int   # 进度文件多少秒无新行即判 hang
    pytest_args: tuple = ()

    @classmethod
    def from_env(cls, argv, repo_root, progress_file):
        return cls(
            # 默认以调用时的当前目录为基准（与 pytest 解析相对路径语义一致），
            # 脚本自身位置不参与推导，故可放在任意目录执行
            repo_root=os.path.abspath(repo_root or os.getcwd()),
            progress_file=progress_file,
            per_test_timeout=int(os.environ.get("PER_TEST_TIMEOUT", "60")),
            pytest_args=tuple(argv),
        )


# ---------------------------------------------------------------------------
# 2. 进度文件协议（父子进程共用的唯一数据契约）
# ---------------------------------------------------------------------------

class ProgressLog:
    """进度文件的读写协议。

    文件格式为两类行：``START\\t<nodeid>`` 与 ``RESULT\\t<nodeid>\\t<STATUS>``。
    每次写入都 flush + fsync，保证进程被 SIGKILL / 崩溃时已写进度不丢失。
    核心不变量：某 nodeid 只有 START 而无 RESULT ⇒ 它是本轮 crash/hang 元凶。
    """

    START, RESULT = "START", "RESULT"

    def __init__(self, path: str):
        self.path = path

    # -- 写入（子进程插件 & 父进程补记终态时调用） --

    def record_start(self, nodeid: str) -> None:
        self._append(self.START, nodeid)

    def record_result(self, nodeid: str, status: str) -> None:
        self._append(self.RESULT, nodeid, status)

    def _append(self, *fields: str) -> None:
        with open(self.path, "a") as f:
            f.write("\t".join(fields) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # -- 读取 --

    def _lines(self):
        """逐行产出解析后的字段列表；文件不存在时为空迭代器。"""
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                yield line.rstrip("\n").split("\t")

    def done_nodeids(self) -> set:
        """已有终态 RESULT 的 nodeid 集合（下一轮收集阶段据此 deselect）。"""
        return {p[1] for p in self._lines()
                if p[0] == self.RESULT and len(p) >= 3}

    def stuck_nodeid(self):
        """只 START 未 RESULT 的最后一个 nodeid；没有则返回 None。"""
        running = None
        for p in self._lines():
            if p[0] == self.START:
                running = p[1]
            elif p[0] == self.RESULT and len(p) >= 3 and running == p[1]:
                running = None
        return running

    def status_counts(self) -> dict:
        """按 nodeid 去重后的 {status: count} 统计。"""
        counts, seen = {}, set()
        for p in self._lines():
            if p[0] == self.RESULT and len(p) >= 3 and p[1] not in seen:
                seen.add(p[1])
                counts[p[2]] = counts.get(p[2], 0) + 1
        return counts

    def remove(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


# ---------------------------------------------------------------------------
# 3. pytest 插件角色（子进程经 -p <本模块> 加载以下 hook）
# ---------------------------------------------------------------------------

class _RepoSourceGuard:
    """阻止仓库源码目录里的顶层包抢占已安装版本（仅子进程安装）。
    pytest 启动时会把 rootdir（仓库根）插入 sys.path[0]（加载根 conftest.py /
    pytest.ini 时发生，无法在外部阻止），导致 ``import torch`` 命中仓库里的
    torch/ 源码目录而非已安装 wheel，报 "Failed to load PyTorch C extensions"。
    本守卫拦截 GUARDED 包的导入，仅当同时满足：
    1) 按当前 sys.path 会解析到仓库根内的副本；
    2) 该副本是纯源码（无 torch/_C*.so 等编译产物，即非 develop 安装）；
    3) 剔除仓库根后仍能找到已安装版本；
    才重定向到已安装版；develop 模式（仓库根内已编译）自动放行不干预。
    """

    GUARDED = ("torch", "torchgen", "functorch")  # 仓库根下与安装包同名的顶层目录

    def __init__(self, repo_root: str):
        self._repo_root = os.path.abspath(repo_root)

    def find_spec(self, fullname, path=None, target=None):
        from importlib.machinery import PathFinder
        root, name = fullname, PathFinder
        if root not in self.GUARDED or path is not None:
            return None
        spec_repo = name.find_spec(root, path)
        if spec_repo is None or not self._inside_repo(spec_repo):
            return None  # 本来就解析到已安装版，不干预
        # 剔除仓库根与 ''（cwd）后再找一次：已安装版 / develop 编译副本
        clean = [p for p in sys.path
                 if p and os.path.abspath(p) != self._repo_root]
        spec_installed = name.find_spec(root, clean)
        if spec_installed is None:
            return spec_repo  # 只有仓库副本，放行（总比 ImportError 好）
        if self._is_dev_build(spec_repo):
            return spec_repo  # develop 安装：仓库副本就是带编译产物的正版
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
        # 只认「编译产物文件」_C*.so / _C*.pyd；源码树里的 torch/_C 目录是纯
        # Python 存根，匹配裸 `_C*` 会把源码检出误判为 develop 安装而放行，
        # 导致 import torch 命中仓库副本并报 "Failed to load PyTorch C extensions"。
        import glob
        pkg_dir = os.path.dirname(spec.origin or "")
        if not pkg_dir:
            return False
        return any(
            os.path.isfile(p)
            for pat in ("_C*.so", "_C*.pyd")
            for p in glob.glob(os.path.join(pkg_dir, pat))
        )


# hook 首次触发时惰性创建；父进程不会调用这些 hook，故不受影响
_PLUGIN_LOG = None


def _plugin_log() -> ProgressLog:
    global _PLUGIN_LOG
    if _PLUGIN_LOG is None:
        _PLUGIN_LOG = ProgressLog(
            os.environ.get("RESILIENT_PROGRESS_FILE", "test_progress.log")
        )
    return _PLUGIN_LOG


def pytest_collection_modifyitems(config, items):
    """收集阶段剔除已经跑过（有 RESULT）的 case，实现断点续跑。"""
    done = _plugin_log().done_nodeids()
    if not done:
        return
    deselected = [it for it in items if it.nodeid in done]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [it for it in items if it.nodeid not in done]


def pytest_runtest_logstart(nodeid, location):
    _plugin_log().record_start(nodeid)


def pytest_runtest_logreport(report):
    log = _plugin_log()
    # call 阶段拿到真正的通过/失败结果
    if report.when == "call":
        log.record_result(report.nodeid, report.outcome.upper())
    # setup 阶段非 passed：要么被 skip，要么 setup 出错（不会再有 call 阶段）
    elif report.when == "setup" and report.outcome != "passed":
        status = "SKIPPED" if report.skipped else "ERROR"
        log.record_result(report.nodeid, status)


# ---------------------------------------------------------------------------
# 4. 监控
# ---------------------------------------------------------------------------

class Watchdog:
    """监控进度文件：超过 timeout 无新行即判定 hang，整组杀掉子进程。

    用法::

        wd = Watchdog(proc, log, cfg).start()
        rc = wd.wait()          # 阻塞至子进程退出
        wd.killed               # 是否由看门狗击杀
    """

    def __init__(self, proc, progress: ProgressLog, cfg: Config):
        self._proc = proc
        self._progress = progress
        self._cfg = cfg
        self._stop = threading.Event()
        self._thread = None
        self.killed = False

    def start(self) -> "Watchdog":
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def wait(self) -> int:
        """阻塞至子进程退出并停掉看门狗；返回子进程退出码。"""
        rc = self._proc.wait()
        self._stop.set()
        self._thread.join(timeout=POLL_SECONDS + 2)
        return rc

    def _watch(self) -> None:
        last_size, last_change = -1, time.time()
        while not self._stop.wait(POLL_SECONDS):
            try:
                size = os.path.getsize(self._progress.path)
            except OSError:
                size = -1
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > self._cfg.per_test_timeout:
                self.killed = True
                self._kill_group()
                return

    def _kill_group(self) -> None:
        # 子进程经 setsid 自成进程组，整组杀掉才能连带 pytest 派生的子孙进程
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGINT)   # 先温和中断，让 pytest 有机会收尾
            time.sleep(10)
            os.killpg(pgid, signal.SIGKILL)  # 兜底强杀（驱动层卡死时 SIGINT 无效）
        except ProcessLookupError:
            pass  # 进程组已退出


# ---------------------------------------------------------------------------
# 5. orchestrator：主循环
# ---------------------------------------------------------------------------

def _abs_arg(arg: str, cfg: Config) -> str:
    """把「文件路径 / nodeid」类参数转成绝对路径（因为子进程 cwd 不在调用目录）。

    - 以 '-' 开头的选项、以及 -k 表达式等非路径 token 原样透传；
    - 相对路径按 cfg.repo_root（默认调用时的 cwd）解析，与 pytest 原生语义一致；
    - nodeid 形如 ``test/test_ops.py::Cls::case`` 时，只把 '::' 之前的文件路径绝对化。
    """
    if arg.startswith("-"):
        return arg
    path, sep, rest = arg.partition("::")
    candidate = os.path.join(cfg.repo_root, path)
    if os.path.exists(candidate) or path.endswith(".py") or os.sep in path:
        return os.path.abspath(candidate) + sep + rest
    return arg


def _child_env(cfg: Config) -> dict:
    """构建子进程环境：净化 PYTHONPATH + 传入进度文件路径。"""
    env = dict(os.environ)
    # PYTHONPATH 只保留脚本所在目录（供 -p <本模块> 导入自身），
    # 剔除仓库根 / 空串，防止 ./torch 源码目录被引入
    clean = [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep)
        if p and os.path.abspath(p) not in (cfg.repo_root, SCRIPT_DIR)
    ]
    env["PYTHONPATH"] = os.pathsep.join([SCRIPT_DIR, *clean])
    # PYTHONSAFEPATH=1：禁止解释器把 cwd/脚本目录自动插到 sys.path 最前
    env["PYTHONSAFEPATH"] = "1"
    env["RESILIENT_PROGRESS_FILE"] = cfg.progress_file  # 插件读同一份进度文件
    return env


def _has_verbosity_flag(args) -> bool:
    """用户是否已显式指定 pytest 输出详细度（-v / -q / --verbose / --quiet）。"""
    for a in args:
        if a in ("--verbose", "--quiet"):
            return True
        # 短选项簇里含 v / q，如 -v / -vv / -vra / -q
        if a.startswith("-") and not a.startswith("--"):
            body = a[1:]
            if "v" in body or "q" in body:
                return True
    return False


def _spawn_child(cfg: Config) -> subprocess.Popen:
    user_args = [_abs_arg(a, cfg) for a in cfg.pytest_args]
    # 默认 -v：让每个 case 的 PASSED/FAILED/SKIPPED 结果单独成行输出到控制台；
    # 用户已显式指定详细度（-v/-q/--verbose/--quiet）时不覆盖其选择
    if not _has_verbosity_flag(user_args):
        user_args = ["-v", *user_args]
    cmd = [
        sys.executable, os.path.abspath(__file__), CHILD_FLAG,
        CHILD_ROOT_FLAG, cfg.repo_root,
        "-p", PLUGIN_MODULE,
        "-p", "no:cacheprovider",
        *user_args,
    ]
    # preexec_fn=setsid：独立进程组，hang 时可整组杀掉；
    # cwd 用中性目录，避免仓库根进入 sys.path 抢占已安装 torch。
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


def orchestrate(cfg: Config) -> int:
    """主循环：拉起子进程 -> 补记 stuck 终态 -> 续跑，直至全部完成。"""
    # 预检：子进程依赖 pytest 可导入，否则第一轮就会以 rc=4 失败，提前给出明确提示
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(f"ERROR: pytest not found for {sys.executable}; "
              f"run with a python that has pytest installed.")
        return 2

    log = ProgressLog(cfg.progress_file)
    rounds = 0
    while True:
        rounds += 1
        prev_done = log.done_nodeids()
        print(f"\n===== round {rounds}: {len(prev_done)} cases finished so far =====")

        watchdog = Watchdog(_spawn_child(cfg), log, cfg).start()
        rc, killed = watchdog.wait(), watchdog.killed

        # 进程退出后，把「START 了但没 RESULT」的那个 case 补记为终态
        stuck = log.stuck_nodeid()
        if stuck:
            status = "HANG" if killed else "CRASH"
            log.record_result(stuck, status)
            print(f"  detected stuck case -> marked {status}: {stuck}")
            continue  # 还有卡死/崩溃点，继续续跑

        # pytest 正常结束（0 全过 / 1 有失败 / 5 无用例可选=全跑完）
        if rc in (0, 1, 5):
            print(f"  pytest exited normally (rc={rc}); all remaining cases done.")
            break

        # 异常退出但没抓到 stuck（如收集/导入阶段崩溃）：防死循环
        if len(log.done_nodeids()) <= len(prev_done):
            print(f"  WARN: abnormal exit rc={rc} with no progress; aborting to "
                  f"avoid an infinite loop.")
            break

    counts = _print_summary(log)
    # 有任何非成功终态即以 1 退出，方便 CI / shell 判断
    return 1 if set(counts) - OK_STATUSES else 0


# ---------------------------------------------------------------------------
# 6. CLI：按身份标记分派父/子角色
# ---------------------------------------------------------------------------

def _run_as_child(args, repo_root) -> int:
    """子进程身份：hook 已由 -p <本模块> 注册，这里只负责启动 pytest。"""
    # 预净化：剔除 ''（cwd）与仓库根；但 pytest 启动时还会自行把 rootdir 插回
    # sys.path[0]，故必须再装 import 守卫才能彻底避免 ./torch 源码抢占已安装 wheel
    repo_root = os.path.abspath(repo_root)
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != repo_root]
    # 脚本可能就在仓库根下（剔除后会丢失），补回脚本目录供 -p <本模块>
    # 导入自身；源码 torch/ 的抢占由下方 import 守卫拦截，不依赖 sys.path 剔除
    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)
    sys.meta_path.insert(0, _RepoSourceGuard(repo_root))
    import pytest
    return pytest.main(list(args))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if CHILD_FLAG in argv:
        rest, repo_root = [], os.getcwd()
        for i, a in enumerate(argv):
            if a == CHILD_ROOT_FLAG:
                repo_root = argv[i + 1]      # 后面必有一个值（父进程成对传入）
            elif a != CHILD_FLAG and argv[i - 1] != CHILD_ROOT_FLAG:
                rest.append(a)
        return _run_as_child(rest, repo_root)

    try:
        root, user_args = _parse_cli(argv)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2

    # 进度文件由脚本内部管理：临时创建、运行结束即删，用户无需（也无法）指定
    fd, progress_file = tempfile.mkstemp(prefix="resilient_progress_", suffix=".log")
    os.close(fd)
    try:
        cfg = Config.from_env(user_args or ["test/test_ops.py"],
                              repo_root=root, progress_file=progress_file)
        return orchestrate(cfg)
    finally:
        ProgressLog(progress_file).remove()


if __name__ == "__main__":
    sys.exit(main())