# 给下一位 Agent

## 先判断你接手的是哪种环境

1. **同一台 Mac 和现有 NAS：**读本地（不公开）`docs/EXECUTION_CHECKPOINT.md`，然后查看 NAS 的 `output/pilot10-current/summary.json`、`auditory-review-queue.json` 和任务进程。现有包已做完工程诊断，不重跑全流程。
2. **新 checkout 或另一台机器：**仓库不含语料、权重、凭据及运行环境。按 [处理流程](DATA_PIPELINE.zh-CN.md) 获取授权来源并准备数据。当前NAS检查只支持macOS SMB `/Volumes/media`；新平台要先实现真实挂载检查，不能取消安全约束。

## 用户当前状态与决策边界

- 首批目标是10场景，覆盖 neutral/sadness/anger/fear/suppression/interrupt/backchannel。
- 当前有8个表达性话段场景和2个真实双声道候选；没有10个完整双工场景验收结论。
- 用户于2026-09-29明确选择：第一阶段优先“说得更有表演感”。按用途保留8个表达性场景与2个双工候选，不再以全部替换成完整双工为默认方向。
- 该方向选择不等于听审通过或数据训练批准。S08仍是neutral原标签/压抑caption线索；S09/S10保留交互检查用途，尚未验收。
- 表演资料新增核验重点：固定目标token，改用同说话人的另一句neutral标注参考做CPU重建；比较原音、旧重建和新参考重建。原20话段中19个旧重建为原句自参考，不足以单独证明表演信息保留。
- 用户要求暂停原定时任务，已经暂停。不要恢复计划任务，不接管DGX生产服务，不启动训练。
- 本仓库发布不表示数据获准训练，也不表示委派了一个已经运行的Agent。

## 当前产物地图（相对于 NAS 项目根）

| 路径 | 用途 |
|---|---|
| `raw/` | 来源原件、许可卡、codec和tokenizer |
| `manifests/` | 原始标注索引、获取记录 |
| `output/pilot10/` | 第一版，当前只使用S01–S08 |
| `output/pilot10-revision2/` | 修订S09/S10及实际源声道片段 |
| `output/pilot10-loader-input/` | 额外S09连续输入与14行真实loader诊断 |
| `output/source-audit/` | 原声道及筛选证据 |
| `output/pilot10-current/` | 当前统一审阅入口、28话段HF格式和韵律 |
| `output/acting-neutral-control-v1/` | 4个已完成的同说话人独立neutral标注参考重建及逐段收据 |
| `output/acting-review-v1/` | 表演优先试听页review.html及20话段packet.json；前4项三路对照 |

关键证据：`frontend-reference-check.json`、`hf-review-readback.json`、`prosody-readback.json`，以及loader-input下的`loader-all-scenes-diagnostic.json`。证据若缺失，不从文档文字推断当前机器也通过。

## 可直接给 Agent 的任务文本

> 请先阅读 AGENTS.md、docs/AGENT_HANDOFF.zh-CN.md、docs/DATA_PIPELINE.zh-CN.md 和 docs/STATUS.md。同一环境还要读私人 EXECUTION_CHECKPOINT.md。第一阶段优先表演感，按表达性/双工用途分别整理，方向选择不等于样本获准训练。先检查现有任务、真实NAS挂载及产物，报告哪些步骤已经有实际证据、哪些缺失，只执行我授权的缺失步骤。所有新下载/缓存/临时文件/产物只存真实NAS；不重复运行生产器，不恢复定时任务、不操作DGX生产服务、不训练。保持来源许可、固定版本、原文、样本时间与hash；不得把混音或补零当独立声轨，不用HumDial测试集训练，不凭文字或特征冒称听过压抑表达。完成后给出实际产物路径、执行命令、通过证据及剩余限制。

## 公开版本的验证

2026-09-29 发布前实际通过：全部公开 Python 文件语法解析、只读 CLI/help、71 个固定 vendor 文件哈希核对、来源配置结构检查、103 个公开文件的敏感信息/大文件检查，以及 8 项针对路径守卫、固定资产校验、拒绝覆盖、CLI 分发的局部单元测试。环境脚本语法与实际 source 通过；本项目文件的 diff 空白检查通过，上游源码按原字节保留。测试临时文件放 NAS；这不是重新训练、完整复现或大规模回归。

对新封装的下载/修订/汇总入口，语法或单测不等于已经在新数据根完成整套生产。按照每步输出与实际返回状态继续，不将历史私有环境的成功偷换成当前运行成功。
