# AGENTS.md

> **本目录只有笔记，没有代码。要读源码，去 WSL。**
>
> 本文件面向所有会读取仓库说明的 AI 助手（Codex / Cursor / Claude Code / DSH 等）。
> 每次开始工作前请先扫一遍 §0 和 §6。

- 适用环境：Windows 11 + WSL2 (Ubuntu 24.04)
- 笔记建档：2026-10-22 起，持续追加
- 分析对象：[`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang) @ `9a91cfa`（约 5000 行教学版推理引擎）

---

## 0. TL;DR —— 开工前必读

1. **本目录只放 Markdown 笔记 + `images/` 里的配图**，没有任何 mini-sglang 源码（`images/_make_figs.py` 只是画图脚本，不是项目代码）。
2. **代码全在 WSL2 里**：`/home/mzygd/workspace/mini-sglang`（editable 安装，改完即生效）。
3. 因此**不要在本目录 `grep` / 全局搜索源码**，搜不到是正常的，不是文件丢失。
4. 本目录**已是 git 仓库**，远端 <https://github.com/mzygQAQ/read-mini-sglang>（默认分支 `main`）。
   改完笔记请 `git add -A && git commit -m "docs: ..." && git push`。
5. 回答**用中文**；写进笔记的结论要有实测来源，不确定就标「待验证」。

---

## 1. 这个工作区是什么

| 项 | 说明 |
|---|---|
| 定位 | 个人学习笔记库：读 mini-sglang 源码 + 在本机 WSL 里跑通推理服务的过程记录 |
| 内容 | 6 篇 Markdown，全部为中文，偏「可复现步骤 + 原理拆解 + 踩坑」 |
| 不是 | 不是源码仓库、不是部署产物、不是可运行项目 |
| 典型任务 | 补充/修订笔记、解答关于 mini-sglang 原理的问题、把新踩的坑写进《踩坑记录》 |

---

## 2. 笔记索引（先查这里，别重复劳动）

| 文件 | 规模 | 覆盖内容 | 什么时候看 |
|---|---|---|---|
| `01-WSL-Ubuntu24.04-mini-sglang-环境配置与运行指南.md` | 987 行 + 2 张图 | 从零装 WSL/apt 换源/uv/CUDA 12.8/部署 minisgl、显存与 KV cache 计算、HTTP 接口全表、一键复现脚本 | 环境、启动、接口、调参 |
| `02-手把手带你读懂一个模型目录-Qwen3-0.6B实战.md` | 1012 行 | 逐文件拆 Qwen3-0.6B 模型目录：safetensors 头（**完整 311 个张量** + 分组表）、config.json 推导关系、词表四件套、chat_template、三个「长度」 | 模型格式、显存估算、tokenizer |
| `03sglang调试main进程启动.md` | 151 行 + 2 张图 | main 进程启动调试手记：启动 7 步、进程拓扑与四条 ZMQ 通道、启动日志里的 `ModelConfig` 逐字段解释、debugpy 远程调试 | 进程/启动链路 |
| `04-张量并行TP是怎么切的.md` | 381 行 + 7 张图 | TP 只切三个维度：vocab / head / intermediate；每层 2 次 all-reduce 的账本；权重如何切着加载；与 vLLM/SGLang 的对应 | 并行原理、多卡、读 `layers/` 代码 |
| `05-page_table与TableManager源码解析.md` | 387 行 + 4 张图 | `page_table` / `token_pool` / KV cache 三张表的形状与关系；`TableManager` 21 行逐行；读的四种写法的两种；CacheManager 的五段注释；请求编号生命周期 | KV 分配、显存、paged attention |
| `踩坑记录.md` | 14 条 | 现象 / 根因 / 解法 / 验证四段式，开头带索引表 | 遇到报错先在这里搜关键词 |

> 已有结论优先引用现有笔记，**不要重新推一遍**；确需修正时直接改原文件并说明来源。
> 仓库落地页是 `README.md`（给人看的目录 + 预览图），**新增笔记时顺手补一下它和本表**。

---

## 3. 代码在哪：WSL 路径表

发行版 `Ubuntu-24.04`，用户 `mzygd`，家目录 `/home/mzygd`。

| WSL 路径 | 是什么 |
|---|---|
| `/home/mzygd/workspace/mini-sglang` | **主工作副本**：源码 + `.venv` + `start.sh` / `pull-model.sh` / `stop.sh` |
| `/home/mzygd/workspace/mini-sglang/python/minisgl/` | 全部源码根（`server/`、`tokenizer/`、`models/`、`utils/`、`kernel/` …） |
| `/home/mzygd/workspace/mini-sglang/.venv/bin/python` | 唯一正确的解释器（系统 `python3` 没装依赖） |
| `/home/mzygd/models/Qwen3-0.6B` | 本地模型目录（软链到 `~/.cache/modelscope/...`，启动即离线） |
| `/home/mzygd/.cache/flashinfer` | FlashInfer JIT 编译缓存，**别删**，删了要重编译数分钟 |
| `D:\Codes\mini-sglang` | Windows 侧镜像副本（IDE 编辑用；**可能落后于 WSL，不要当权威**） |

代码基线：upstream `9a91cfa` + 本地改动（进程改名 `utils/procname.py` 等，清单见《踩坑记录》§2「本次改动清单」）。

常见文件：
- `python/minisgl/models/config.py` —— `ModelConfig`，从 `config.json` 算模型形状
- `python/minisgl/server/api_server.py` —— HTTP 路由
- `python/minisgl/server/args.py` —— 命令行参数、模型本地化逻辑
- `python/minisgl/server/launch.py` —— 进程启动 / scheduler 拉起
- `python/minisgl/tokenizer/tokenize.py`、`tokenizer/server.py` —— 分词子服务

### 3.1 怎么从 Windows 进 WSL

```powershell
wsl -d Ubuntu-24.04 -e bash -lc 'ls /home/mzygd/workspace/mini-sglang/python/minisgl'
```

三条已知坑（都记在《踩坑记录》里）：

1. **PowerShell 会吃掉 `$`** —— 在 `wsl -- bash -lc '...'` 里别用 shell 变量，路径写死。
2. **`.bashrc` 有非交互早退守卫** —— 环境变量块必须放在第 1 行才生效；非交互式入口拿不到变量时，直接用绝对路径调 `.venv/bin/python`。
3. **`wsl.exe` 输出是 UTF-16LE** —— 重定向到文件再读会乱码。

### 3.2 能不能进 WSL，取决于沙箱策略（2026-10-24 实测）

| 文件策略 | `wsl.exe ...` | 说明 |
|---|---|---|
| `workspace-write` | ❌ `Wsl/E_ACCESSDENIED`，exit -1 | 沙箱挡掉了 WSL 服务通道，`\\wsl.localhost` 也 Access denied |
| `danger-full-access` | ✅ 正常，exit 0 | 可读可执行 |

**先试一次**：`wsl.exe -d Ubuntu-24.04 -e bash -lc "ls /home/mzygd/workspace/mini-sglang/python"`。
若返回 `Wsl/E_ACCESSDENIED`，**不要硬猜源码**，按下面降级：

1. 请用户把目标文件（或片段）粘贴进对话；
2. 请用户在 WSL / VSCode 里跑命令，把输出贴回来；
3. 请用户把需要的文件 `cp` 到本目录临时副本后再读。

**能进 WSL 时的推荐姿势**（PowerShell 引号/编码很容易把命令搞坏，别在一行里堆 bash）：

```powershell
# 先写脚本到 ASCII 路径（例如 %TEMP%\dsh-mirror.sh），再执行，避开引号地狱
wsl.exe -d Ubuntu-24.04 -e bash /mnt/c/Users/mzygd/AppData/Local/Temp/dsh-mirror.sh
```

最省事的读法是**镜像到工作区再逐个读**（`read` 工具带行号，改笔记时行号可直接引用）：

```bash
cp -r /home/mzygd/workspace/mini-sglang/python/minisgl \
      "/mnt/c/Users/mzygd/Desktop/mini-sglang分析/.tmp-src/"
# 用完删掉 .tmp-src，别留成"第二份源码"
```

> `python/minisgl` 只有一个 `.venv` 之外的小目录，拷过来不到 1 MB，比反复走 `wsl -e cat` 快得多。

### 3.3 「文件:行号」引用约定

笔记里的 `models/config.py:17`、`python/minisgl/server/api_server.py:64-83` 一律指**仓库根在 WSL 内**的相对路径。
AI 引用时保持同样写法；无法核对时写明「据笔记，未复核」。

---

## 4. 写笔记的约定（与现有 6 篇保持一致）

- **语言**：中文；先 TL;DR / 结论，再展开细节。
- **实测优先**：每条结论标注来源（哪次运行、哪条命令、哪段日志）。未验证标 **「待验证」**，验证通过后升级为 **「已确认」**。
- **可定位**：能落到源码的补 `文件:行号`；能落到命令的给完整命令。
- **踩坑记录**：四段式（现象 / 根因 / 解法 / 验证），并**更新开头索引表**；每条带编号。
- **编号文件**：新笔记沿用 `NN-标题.md`（`01`…），并在本文件 §2 补一行索引。
- **配图**：示意图放 `images/`，用 `images/_make_figs.py`（Pillow 手绘、2 倍超采样）生成 PNG。
  **不要用代码块里的 ASCII 图代替配图**；改脚本后重跑 `python _make_figs.py` 即可重新生成。
  修改 `.py` / `.md` 一律用 UTF-8 编辑器保存 —— PowerShell 的 `Set-Content` 会把中文按 GBK 重编码成乱码。
- **交叉引用**：用相对路径 Markdown 链接，不写死盘符绝对路径。
- **命令块**：给出的命令应当能原样复制执行（`start.sh` 的设计原则就是全字面量、无变量间接）。

---

## 5. 环境事实速查

| 项 | 值 |
|---|---|
| 发行版 | Ubuntu 24.04.5 LTS (noble)，WSL 3.0.3 |
| GPU | RTX 3070，**8 GB**，sm_86 |
| CUDA Toolkit | 12.8.93（`/usr/local/cuda`） |
| Python | 3.12（`.venv` 由 `uv venv` 建，**不带 pip**，装包用 `uv pip`） |
| 关键版本 | torch 2.9.1+cu128 / flashinfer 0.7.2 / sgl-kernel 0.3.21 / triton 3.5.1 |
| 服务地址 | `0.0.0.0:8000`，WSL2 自动转发 → Windows 侧 `localhost:8000` 同一地址 |
| 显存约束 | KV cache 约 5 GB 上限，只适合 0.6B ~ 1.7B 小模型 |
| 镜像源 | apt / pip / uv / HF 全部走清华、hf-mirror（详见《01》§2、§5.3） |

---

## 6. 常用命令（均在 WSL 内执行）

```bash
cd /home/mzygd/workspace/mini-sglang

bash start.sh                      # 起服务：--memory-ratio 0.8 --host 0.0.0.0 --port 8000
bash start.sh --shell-mode         # 终端交互式对话
bash start.sh --memory-ratio 0.6   # 覆盖默认参数（argparse 取最后一个）
bash stop.sh                       # 清残留（同时匹配新旧进程名）
bash pull-model.sh Qwen/Qwen3-0.6B # 下一份模型到 ~/models/，之后完全离线

curl -s localhost:8000/v1/models                     # 健康检查
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{...}'
```

---

## 7. 不要做的事

- ❌ 在本目录搜/建源码 —— 代码不在这里。
- ❌ `git init`（已经是仓库了）/ 提交 `.tmp-src/` 这类源码镜像 —— 那是临时产物，`.gitignore` 已排除。
- ❌ 删除 `~/.cache/flashinfer`、`~/.cache/modelscope`。
- ❌ 用系统 `python` / `python -m pip` 装包（PEP 668 + 缺依赖），一律 `uv`。
- ❌ 把 `D:\Codes\mini-sglang` 当成最新代码（真实改动在 WSL）。
- ❌ 编造命令输出、日志、行号；拿不到就说拿不到。

---

## 8. 关于本文件的名字

`AGENTS.md` 是目前的事实标准（OpenAI Codex、Cursor、多数 agent 工具都读它）。
Claude Code 读的是 `CLAUDE.md`，需要时让它一行 `@AGENTS.md` 指过来即可，避免两份内容不同步。

- upstream：<https://github.com/sgl-project/mini-sglang>
- 本文件最后更新：2026-10-24

---

## 9. 网络：`git push` 通不了的时候

实测（2026-10-24 同一小时内反复出现）：`github.com:443` 的 TLS 会被 reset
（`OpenSSL SSL_read: Connection was reset, errno 10054`）或直接超时，**但 `api.github.com` 一直是通的**。
这时 `git push` 无论重试多少次都没用，可以改走 **Git Data API** 补推：

1. `POST /git/blobs`（base64 内容）—— 上传每个改动文件，返回的 blob SHA 必须与本地一致
2. `POST /git/trees`（`base_tree` = 父提交的 tree + 本次变更条目）—— tree SHA 也必须一致
3. `POST /git/commits`（tree + parents + **原样的 author/committer 姓名、邮箱、时间**）—— commit SHA 必须一致
4. `PATCH /git/refs/heads/main`

**关键是第 1–3 步都校验 SHA**：只要有一个不一致就中止、不碰 ref，本地与远端就不会分叉。
如果图省事不校验，直接拿 API 造一个"内容相同但 SHA 不同"的 commit，本地历史就跟远端错开了，
而 `fetch` 也走不通，后面会很麻烦。

其他已知事实：

- SSH key（`~/.ssh/id_rsa`）**没有绑到 GitHub**，`ssh -T git@github.com` 是 `Permission denied (publickey)`；
  推代码走 HTTPS，凭据来自 Windows 凭据管理器里的 PAT（`credential.helper=manager-core`），不需要交互。
- 本机没装 `gh` CLI。
- 临时补推脚本：`%TEMP%\dsh-api-push.py`（需要时再生成一次即可，逻辑就是上面的 4 步）。
