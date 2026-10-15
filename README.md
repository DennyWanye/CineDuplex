# CineDuplex

**2026-10-15：原生语音排查已按用户要求暂停，音质与准确性尚未通过。** 本次只保存代码和[检查结果与接续说明](docs/NATIVE_VOICE_INVESTIGATION_20261015.zh-CN.md)，没有继续测试或部署。请勿按历史运行命令自动恢复任务。

用于制作可追溯的情绪语音与双工对话候选数据的小型工具集。围绕 Lychee-FD / Step-Audio2，完成真人数据获取、话段规范化、原声道检查、语音编码与重建、韵律测量和实际数据接口诊断。

**当前包含数据处理、BF16 训练和原生全双工通话测试代码。** 原执行环境已完成本次训练并部署通话服务；持续实时速度、真人长通话和主观音质仍未全部达标。仓库不发布训练数据、模型权重或私人部署配置。

当前阶段优先改善表演感：表达性话段与双工交互资料按用途分别验收。先用原音/重建/独立中性标注参考对照检查表演保留，不把全体改成双声轨作为表达性资料的前提；结构缺口和未听审项仍明确保留。

## 从哪里开始

- **交给另一个 Agent：** [Agent 接续手册](docs/AGENT_HANDOFF.zh-CN.md)
- **资料如何处理出来：** [逐步数据处理流程与命令](docs/DATA_PIPELINE.zh-CN.md)
- **哪些通过、哪些没通过：** [能力及验证状态](docs/STATUS.md)
- **固定来源与文件身份：** [数据输入清单](configs/data_sources.json)
- **训练、离线对比和实时通话：** [运行代码与接续说明](docs/TRAINING_REALTIME.zh-CN.md)

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
src/cineduplex/train/      NAS 训练、LoRA、BF16 推理和四节点实时运行
src/cineduplex/web/        离线音频对比、持续收音与增量播放页面
src/cineduplex/cluster/    既有 Gateway 准入检查
configs/                  固定公开来源及哈希
scripts/nas-env.sh        NAS 环境设置
scripts/                  训练、部署、定向资格检查和通话日志分析
vendor/lychee-fd-source/   固定上游源码子集及其许可证
docs/                     公开操作手册、接续说明及状态
```

NAS 的 `output/pilot10-current/` 是当前审阅入口，原始话段保存在 `pilot10/`，修订后的 S09/S10 在 `pilot10-revision2/`，诊断数据在 `pilot10-loader-input/`。详见处理流程。

## 可选阿里云安全护栏

提供独立的原生 HTTPS 审核客户端及整段回答缓存器，默认关闭，不包含凭据。文字审核与可选语音审核分别配置；语音接口为异步审核，会增加等待。仓库包含实时服务集成代码，实际启用状态以运行服务为准。见 [接入说明](docs/ALIYUN_GUARDRAIL.zh-CN.md)。

## 发布边界

Apache-2.0 适用于本项目原创代码和公开文档；上游代码保留原许可证和署名，见 [NOTICE](NOTICE)。代码开源不改变数据集及模型的使用条件。

仓库不包含语料、转写产物、模型权重、依赖环境、访问令牌或私人机器配置。数据使用者需自行取得来源权限并遵守其条款；来源 README 随获取的文件保留。HumDial-FDBench 的测试集不用于训练。

训练与部署脚本面向既有 NAS、DGX 和 Gateway 环境；新 checkout 不会自动获得这些运行条件。私人检查点、机器配置、历史数据生产草稿和运行证据仍不公开。不要依据历史材料启动生产服务、重复提交训练或恢复旧定时任务。
