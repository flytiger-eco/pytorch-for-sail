# pytorch-for-sail-cuda-compat

本仓库提供 Flytiger CUDA 兼容（CUDA-compat）形态的 PyTorch-for-SAIL 部署入口。
它与上游 pytorch-for-sail 的 v2.14.1 分支解耦：仓库本身不包含 PyTorch 源码，
只保存组装与构建所需的补丁、脚本与镜像定义。通过 `scripts/setup.sh` 组装出
完整源码树后，可以用两种方式编译（详见下文「编译路径」）：

- **容器手工编译**：`scripts/build.sh` 在容器内分步安装依赖、编译 wheel 并跑 smoke；
- **Dockerfile 构建**：`docker/Dockerfile.sdk2.2.1-cuda-compat` 从源码完整编译并验证 wheel。

## 仓库结构

~~~
pytorch-for-sail-cuda-compat/
├── patches/                       # 4 笔可 git am 的 compat patch（按主题合并组织）
│   ├── 0001-feat-third-party-add-the-CUDA-compat-PPU-submodules.patch
│   ├── 0002-feat-build-enable-the-CUDA-compat-PPU-mode.patch
│   ├── 0003-feat-sdpa-add-the-FA3-flex-flash-attention-SDPA-back.patch
│   └── 0004-feat-cudafy-enable-cudafy-conversion-for-the-CUDA-co.patch
├── scripts/
│   ├── setup.sh                   # 组装源码树：clone 上游 + 打 patch + 初始化子模块
│   ├── seed_submodules.py         # setup.sh 的子模块离线 seed 辅助脚本
│   └── build.sh                   # 容器内编译 wheel 并跑 smoke
├── docker/
│   └── Dockerfile.sdk2.2.1-cuda-compat   # 从源码完整编译并验证 wheel 的镜像定义
├── LICENSE
├── NOTICE
└── README.md
~~~

## 构建形态

| 维度 | CUDA compat（本仓库） |
|------|----------------------|
| USE_SAIL | 0 |
| USE_PPU | 1 |
| USE_CUDA | 1 |
| USE_PCCL | 0 |
| USE_NCCL / USE_SYSTEM_NCCL | 1 / 1（使用容器内 NCCL wrapper，见下文） |
| flash attention | third_party/flash-attention-for-sail 经 cudafy-for-sail 转换后使用 |
| xformers | third_party/xformers（flytiger-eco fork） |
| gloo | 上游 third_party/gloo |

CUDA compat 形态固定使用以下编译环境变量（`build.sh` 与 Dockerfile 均已内置，
无需手工设置）：

~~~bash
export USE_SAIL=0 USE_PPU=1 USE_CUDA=1 USE_PCCL=0 USE_NCCL=1 USE_SYSTEM_NCCL=1
export USE_FLASH_ATTENTION=1
export BUILD_TEST=0
export TORCH_CUDA_ARCH_LIST="8.0a;8.9"    # 8.0a = SM80 系列 arch-accelerated 变体
export PYTORCH_SAIL_ARCH="ppu_10;ppu_15"
export PYTORCH_BUILD_VERSION=2.14.1        # 固定 wheel 版本号，不带 git 描述后缀
export PYTORCH_BUILD_NUMBER=0
~~~

两个重量级可选后端**默认关闭以加速编译**：

- `USE_FLEX_FLASH_ATTENTION=0`：关闭 FA3 flex-flash-attention SDPA 后端；
- `USE_ELEMENTWISE_OPT=0`：关闭 elementwise 优化后端。

这两个开关在源码树 CMake 中的 `option()` 默认值是 ON，但构建脚本会在 CMake
configure 之前通过环境变量写入 cache（FORCE），`option()` 不会覆盖已存在的
cache 项，因此导出为 0 即生效关闭。需要打开对应后端时，导出为 1（
`build.sh` 路径）或传 `--build-arg USE_FLEX_FLASH_ATTENTION=1`（Dockerfile
路径）。关闭后对应 SDPA 路径回退到 FA2 / mem-efficient 等 PyTorch 原生实现，
不影响 matmul、transformers、NCCL 等核心验证。

## 编译环境

- **基础镜像**（PPU SDK v2.2.1，CUDA 13.0，Python 3.12）：

  ~~~
  pkg.flytiger-eco.com/docker_release/ppu:sdk2.2.1-cuda13.0-ubuntu24-py312-20261103
  ~~~

  注意：该镜像未预装 cmake / ninja 等构建工具，两条编译路径都会自动安装。

- **NCCL / PCCL 包**：SDK v2.2.1 将 PCCL（含 NCCL 兼容 wrapper）作为独立包
  发布。本仓库两条编译路径默认使用基础镜像预装的副本；如需指定版本，传入
  tarball 地址（下载后会做 sha256 校验）：

  ~~~
  https://pkg.flytiger-eco.com/artifactory/generic-local/NCCL/v2.2.2/nccl-2.27.3-v13_ubuntu2404_sdk-v2.2.2.tar.gz
  sha256: 93513c25af1a5c94ccc38483261b23de6e4912763c3a589e60b1f8dd86cc7a02
  ~~~

- **网络**：宿主机需能访问 GitHub（t-head/pytorch-for-sail）用于 clone 上游；
  有本地参考克隆时可通过 `REFERENCE_REPO` 完全离线组装（见下文）。

## 步骤 0：组装源码树

~~~bash
cd /path/to/pytorch-for-sail-cuda-compat

# 组装源码树到 work/pytorch（可用 WORK=... 覆盖输出目录）
# REFERENCE_REPO 指向本地已有 pytorch-for-sail 克隆时完全离线（推荐）
REFERENCE_REPO=/path/to/pytorch-2-14-1-sail-verify bash scripts/setup.sh
~~~

`setup.sh` 会依次执行：

1. clone https://github.com/t-head/pytorch-for-sail 的 v2.14.1 分支，并检出锁定
   基底 commit `8b41b061e474`（patch 生成基底；上游分支前进也不影响构建）。
   设置 `REFERENCE_REPO` 时改为从本地参考库离线 clone（硬链接，零额外磁盘），
   网络不稳定时的 clone 会自动重试 3 次；
2. 按顺序 `git am patches/*.patch`；
3. 初始化子模块：设置 `REFERENCE_REPO` 时从本地参考树离线 seed（比对 gitlink、
   复制工作树并镜像 gitdir 指针，无需网络）；否则回退到 `git submodule sync
   --recursive && git submodule update --init --recursive`（需要网络）；
4. 把本仓库的 Dockerfile 复制进源码树 `docker/` 目录（供编译路径 B 使用），
   并写入 `.git/info/exclude` 保持工作树干净；重复执行 setup.sh 可重入。

## 编译路径 A：容器手工编译（build.sh）

`build.sh` 在容器内分步执行：安装构建工具 → 安装 NCCL/PCCL 包 → 设置编译
环境 → 安装 Python 构建依赖 → 编译 wheel → 安装并跑 smoke。用已启动的容器
或直接指定镜像均可：

~~~bash
# 方式 A1：复用已启动的容器（容器内源码路径与宿主不同时用
#          CONTAINER_PYTORCH_DIR 指定）
CONTAINER_NAME=<container> bash scripts/build.sh

# 方式 A2：从镜像启动新容器（自动挂载源码树与 PPU 加速器设备）
DOCKER_IMAGE=pkg.flytiger-eco.com/docker_release/ppu:sdk2.2.1-cuda13.0-ubuntu24-py312-20261103 \
    bash scripts/build.sh
~~~

如需按指定 NCCL 版本编译，传入包地址与校验和（不传则使用镜像预装副本）：

~~~bash
NCCL_PACKAGE_URL=https://pkg.flytiger-eco.com/artifactory/generic-local/NCCL/v2.2.2/nccl-2.27.3-v13_ubuntu2404_sdk-v2.2.2.tar.gz \
NCCL_PACKAGE_SHA256=93513c25af1a5c94ccc38483261b23de6e4912763c3a589e60b1f8dd86cc7a02 \
CONTAINER_NAME=<container> bash scripts/build.sh
~~~

容器内执行的实际步骤（`build.sh` 内部逻辑，供手工复现）：

1. **安装构建工具**：检测到 cmake / ninja 缺失时通过 apt 安装完整工具链
   （build-essential、cmake、ninja-build、python3-dev 等，与 Dockerfile 同套
   清单，幂等）；
2. **安装 NCCL/PCCL 包**：`NCCL_PACKAGE_URL` 非空时下载 tarball、sha256
   校验后解压到 /usr/local（顶层为 pccl/，提供 PCCL 库与 nccl_wrapper）；
3. **设置 SDK 环境**：`source /usr/local/PPU_SDK/envsetup.sh` 与
   `source /usr/local/pccl/envsetup.sh`；
4. **编译环境变量**：上文「构建形态」的配方，外加
   `NCCL_INCLUDE_DIR=/usr/local/pccl/nccl_wrapper/include`、
   `NCCL_LIB_DIR=/usr/local/pccl/nccl_wrapper/lib`；
   `PYTORCH_BUILD_VERSION` / `PYTORCH_BUILD_NUMBER` 把 wheel 版本固定为
   `2.14.1`（不设时构建会从 git 推导出 `2.14.1a0+git<sha>` 形式）；
5. **Python 构建依赖**：`scikit-build-core`、sympy、networkx、pyyaml、six、
   pytest；其中 packaging、typing-extensions 用 `--ignore-installed` 覆盖 apt
   预装版本（apt 版无 RECORD 文件，直接升级会报 Cannot uninstall）；
6. **编译 wheel**：`python3 -m pip wheel . --no-build-isolation -w dist`。
   CMake configure 阶段会自动调用 `tools/cudafy_compat` 将
   flash-attention-for-sail / actlize / flex-flash-attention 转换到
   `build/cudafy-compat`，应用 compat 自有 patch 集
   （`tools/cudafy_compat/patches/`）并把生成文件的 C++ 标准修正为 C++20；
7. **安装并验证**：安装 wheel 后执行 matmul smoke（断言
   `torch.cuda._is_compiled()`、打印 NCCL 版本），再跑
   test_transformers 有界子集与 test_nccl 子集（结果不影响构建退出码）。

重复构建会保留 `work/pytorch/build/`，由 ninja 自动增量；强制全量可先手动
删除该目录。

## 编译路径 B：Dockerfile 构建

构建上下文是由 `setup.sh` 组装的源码树（v2.14.1 上游 + 4 笔 compat patch +
全部已初始化的子模块）。上下文中的 `.dockerignore` 是上游自带的符号链接
（指向 .gitignore），会自动排除 build/ 与 dist/ 等产物目录。

~~~bash
cd work/pytorch
docker buildx build --load --progress=plain \
  --build-arg NCCL_PACKAGE_URL=https://pkg.flytiger-eco.com/artifactory/generic-local/NCCL/v2.2.2/nccl-2.27.3-v13_ubuntu2404_sdk-v2.2.2.tar.gz \
  --target runtime \
  -t pytorch-sail-cuda-compat:2.14.1 \
  -f docker/Dockerfile.sdk2.2.1-cuda-compat .
~~~

镜像分三阶段：build（安装工具链与 NCCL 包、编译 wheel、安装并跑自检）、
artifact（仅含 wheel 的导出层）、runtime（可运行环境）。`--target runtime`
产出的镜像已安装并验证过 wheel。

构建参数（`--build-arg`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BASE_IMAGE | pkg.flytiger-eco.com/docker_release/ppu:sdk2.2.1-cuda13.0-ubuntu24-py312-20261103 | PPU SDK v2.2.1，CUDA 13.0，Python 3.12 |
| TORCH_CUDA_ARCH_LIST | "8.0a;8.9" | 一个 wheel 覆盖 SM80 系列（arch-accelerated）与 SM89；只编单一架构时传如 `8.9` |
| MAX_JOBS | 128 | 编译并行度 |
| USE_FLEX_FLASH_ATTENTION | 0 | FA3 flex-attention 后端，默认关闭加速编译 |
| USE_ELEMENTWISE_OPT | 0 | elementwise 优化后端，默认关闭加速编译 |
| NCCL_PACKAGE_URL | （空） | PCCL/NCCL 包 tarball 地址；非空时下载并 sha256 校验后安装 |
| NCCL_PACKAGE_SHA256 | 93513c25... | 对应包的 sha256 |
| PIP_INDEX_URL | https://pypi.org/simple | pip 源 |

构建末尾自动验证：torch 版本、CUDA 版本、`torch.cuda._is_compiled()`、
NCCL 版本以及 libtorch_cuda.so 的 NCCL 库链接。

只导出 wheel（不产出运行镜像）：

~~~bash
rm -rf wheels
docker buildx build --progress=plain \
  --target artifact \
  --output type=local,dest=./wheels \
  -f docker/Dockerfile.sdk2.2.1-cuda-compat .
~~~

运行镜像：基础镜像自带 ENTRYPOINT，用 `--entrypoint /bin/bash` 覆盖后进入
shell；源码树保留在 /workspace/pytorch。运行需要加速器的工作负载时，按宿主
环境补充设备映射参数（PPU 宿主通过 /dev/alixpu* 透传，不使用 NVIDIA
runtime）：

~~~bash
docker run -it --rm --entrypoint /bin/bash pytorch-sail-cuda-compat:2.14.1
~~~

## 验证

两条路径构建完成后均可验证（以路径 B 的 runtime 镜像为例）：

~~~bash
docker run --rm -i --entrypoint /bin/bash pytorch-sail-cuda-compat:2.14.1 <<'EOF'
set -euo pipefail
source /usr/local/PPU_SDK/envsetup.sh
source /usr/local/pccl/envsetup.sh
cd /tmp
python3 -c 'import torch; assert torch.cuda._is_compiled(); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.nccl.version())'
EOF
~~~

本仓库按上文流程实测（镜像 sdk2.2.1-cuda13.0-ubuntu24-py312-20261103 +
NCCL v2.2.2 包、默认关闭两个可选后端、`TORCH_CUDA_ARCH_LIST="8.0a;8.9"`）
的结果：

**编译路径 A（build.sh）实测**：

~~~
torch-version=2.14.1
torch-cuda-version=13.0
torch-cuda-compiled=True
nccl-version=(2, 27, 3)
libnccl.so.2 => /usr/local/pccl/nccl_wrapper/lib/libnccl.so.2
wheel: torch-2.14.1-cp312-cp312-linux_x86_64.whl（约 296 MB）
matmul smoke: OK（设备冒烟在 PPU 设备空闲时验证）
~~~

**编译路径 B（Dockerfile）实测**：`--target runtime` 全量冷编译耗时约 24
分钟（wheel 编译 23 分钟 + 镜像导出 20 秒，MAX_JOBS=128），产出镜像
`pytorch-sail-cuda-compat:2.14.1-sdk2.2.1-arch80a`。构建阶段末尾自检输出
（从该 runtime 镜像独立启动容器复核，结果一致）：

~~~
torch-version=2.14.1
torch-cuda-version=13.0
torch-cuda-compiled=True
nccl-version=(2, 27, 3)
libnccl.so.2 => /usr/local/pccl/nccl_wrapper/lib/libnccl.so.2
~~~

（两条路径都固定 `PYTORCH_BUILD_VERSION=2.14.1`，因此版本号不带 git 描述
后缀；如更换上游版本请同步调整。）

## 自定义

- `UPSTREAM_URL` / `UPSTREAM_BRANCH` / `UPSTREAM_COMMIT`：切换上游仓库或分支；
  构建默认锁定基底 commit `8b41b061e474`（v2.14.1 分支头，patch 生成基底）。
- `REFERENCE_REPO`：用本地 git 仓库离线组装，显著减少下载时间。
- `WORK`：改变源码树输出目录（默认 work/）。
- `PYTORCH_DIR` / `CONTAINER_PYTORCH_DIR`：源码树不是由本仓库 setup.sh 生成时
  指定目录；后者用于容器内路径与宿主不同前缀（如 /mnt/ssd）的场景。
- `MAX_JOBS` / `PYTORCH_SAIL_ARCH`：调整编译并行度与目标架构。
- `PYTORCH_BUILD_VERSION` / `PYTORCH_BUILD_NUMBER`：wheel 版本号（默认
  2.14.1 / 0）；不设时构建会从 git 推导出 `2.14.1a0+git<sha>` 形式，
  通常不希望出现。
- `USE_FLEX_FLASH_ATTENTION` / `USE_ELEMENTWISE_OPT`：默认 0（关闭）；显式
  导出为 1（build.sh 路径）或传 `--build-arg`（Dockerfile 路径）打开对应
  后端，适合已验证的完整功能构建。

## 已知限制

- PCCL 在 compat 形态下保持关闭（USE_PCCL=0）：compat 形态通过
  `USE_NCCL=1 USE_SYSTEM_NCCL=1` 使用容器内的 /usr/local/pccl/nccl_wrapper
  运行 NCCL 相关用例。
- cudafy 输出目录中的 `setup.py` / `build_lib.py` 硬编码 `-std=c++17`，
  `tools/cudafy_compat` 会在转换后自动替换为 `-std=c++20` 以匹配 PyTorch
  2.14.1 头文件。
- compat 形态使用自有 patch 集（`tools/cudafy_compat/patches/`），不依赖源码
  树 `third_party/*.patch`；工具在转换后幂等应用并校验标记。
- 默认关闭 `USE_FLEX_FLASH_ATTENTION` 与 `USE_ELEMENTWISE_OPT` 以加速验证
  构建；对应 SDPA 路径会回退到 FA2 / mem-efficient 等 PyTorch 原生实现，
  不影响 matmul / transformers / NCCL 等核心跑测。

## 许可证

本仓库采用 BSD 3-Clause License，版权归属 Zhejiang Runhe Cloud Intelligence
Technology Co., Ltd.（详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)）。
patch 源自公开仓库 https://github.com/t-head/pytorch-for-sail（BSD-3-Clause），
上游文件的版权与许可声明原样保留；patch 新增文件使用本仓库版权行（见 LICENSE）。
