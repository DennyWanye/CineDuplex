# CineDuplex

用于制作可追溯的情绪语音与双工对话候选数据的小型工具集。围绕 Lychee-FD / Step-Audio2，完成真人数据获取、话段规范化、原声道检查、语音编码与重建、韵律测量和实际数据接口诊断。

**当前是数据处理 MVP，不是已经完成的训练系统或高质量训练集。** 10 个场景候选已经在原执行环境制作；听审与完整双工结构验收尚未完成。没有发布训练权重。

当前阶段优先改善表演感：表达性话段与双工交互资料按用途分别验收。先用原音/重建/独立中性标注参考对照检查表演保留，不把全体改成双声轨作为表达性资料的前提；结构缺口和未听审项仍明确保留。

## 从哪里开始

- **交给另一个 Agent：** [Agent 接续手册](docs/AGENT_HANDOFF.zh-CN.md)
- **资料如何处理出来：** [逐步数据处理流程与命令](docs/DATA_PIPELINE.zh-CN.md)
- **哪些通过、哪些没通过：** [能力及验证状态](docs/STATUS.md)
- **固定来源与文件身份：** [数据输入清单](configs/data_sources.json)

推荐从源码 checkout 运行。当前 NAS 守卫针对 **macOS 的真实 SMB 挂载 `/Volumes/media`**，不宣称 Windows/Linux 或任意挂载路径已经支持。新下载、依赖缓存、临时文件和产物必须存 NAS；失败时不回退本机。

```sh
# 在仓库根目录执行；先读接续手册，避免重跑已有制作任务。
source scripts/nas-env.sh
python3 -B -m cineduplex.data.workflow status --root "$CINEDUPLEX_ROOT"
```

没有现成数据时，按数据处理文档建立依赖并逐步执行。`status` 仅展示现有记录；其他制作命令会访问数据来源或生成 NAS 文件，不能把它们当作无副作用的状态检查。

## 数据流程

```text
来源许可 + 固定版本 + 文件校验
  → EmotionTalk 真实话段 / SmoothConv 原始双声道
  → 16 kHz PCM 与来源时间标注
  → raw speech tokens → CPU 重建 → 文件读回
  → 10 场景审阅清单 / 韵律测量 / HF 审阅格式
  → 实际 Lychee Dataset + collator 诊断
  → 听审、结构与用途确认（目前未完成）
```

表达性话段用于考察“怎么说”；完整双工数据还需要真实连续时间轴和可区分的双方录音，用于考察“何时说、如何应对对方”。混音、截取话段或补零试听不能冒充独立连续声轨。原始标签为 neutral 的 suppression 候选仍需听审，不自动改成已确认压抑。

## 目录

```text
src/cineduplex/data/       数据处理、codec、诊断与韵律代码
configs/                  固定公开来源及哈希
scripts/nas-env.sh        NAS 环境设置
vendor/lychee-fd-source/   固定上游源码子集及其许可证
docs/                     公开操作手册、接续说明及状态
```

NAS 的 `output/pilot10-current/` 是当前审阅入口，原始话段保存在 `pilot10/`，修订后的 S09/S10 在 `pilot10-revision2/`，诊断数据在 `pilot10-loader-input/`。详见处理流程。

## 发布边界

Apache-2.0 适用于本项目原创代码和公开文档；上游代码保留原许可证和署名，见 [NOTICE](NOTICE)。代码开源不改变数据集及模型的使用条件。

仓库不包含语料、转写产物、模型权重、依赖环境、访问令牌或私人机器配置。数据使用者需自行取得来源权限并遵守其条款；来源 README 随获取的文件保留。HumDial-FDBench 的测试集不用于训练。

此前实验工作区内的 DGX 部署/集群工具、私人运行记录和未完成训练草稿未纳入这个数据 MVP 版本，仍留在原工作区。不要依据历史材料启动生产服务或恢复旧训练。原定时任务已由用户暂停。
