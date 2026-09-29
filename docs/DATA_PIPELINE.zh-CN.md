# 数据如何处理出来

这份手册把原来散落在执行日志里的准备、修订、编码与汇总步骤整理成命令。**不要在已经有产物的根目录从头执行。** 接手现有 NAS 时先读 [接续手册](AGENT_HANDOFF.zh-CN.md) 和 `workflow status`。下面的全流程用于新目录或明确缺失的阶段。

## 1. 条件与存储

- macOS，真实 SMB 挂载 `/Volumes/media`；Python >=3.11，原执行环境为 Python 3.13；有 `curl`、`ffmpeg`、`uv`。
- Python 版本与平台必须有依赖 wheel；没有 wheel 时停止，不自动换系统 Python、编译 CUDA 或修改生产服务。
- 来源：[EmotionTalk](https://huggingface.co/datasets/BAAI/Emotiontalk)、[SmoothConv](https://huggingface.co/datasets/qualialabsAI/SmoothConv)。先确认自己的账号权限和来源使用条件。
- codec 来源：[Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) 的 token2wav 文件；文本 tokenizer 来源：[Lychee-FD](https://huggingface.co/HIT-TMG/Lychee-FD) 的六个小文件。**不需要下载完整 13B 权重。**
- 凭据用 `HF_TOKEN_PATH` 指向自己的私有令牌文件，或由运行环境安全注入 `HF_TOKEN`。不要发到聊天、提交 Git，或把凭据复制到数据集。
- 当前实现不是跨平台通用安装器。不要删掉 `nas_root` 来绕过挂载检查；Linux/NFS 需要先实现并验证相应真实挂载检查。

在仓库根目录执行：

```sh
# 原机器接续使用现有目录。复现到新数据根时，显式选择同一 NAS 内另一个目录。
export CINEDUPLEX_ROOT=/Volumes/media/CineDuplex
source scripts/nas-env.sh
command -v ffmpeg
python3 -B -m cineduplex.data.workflow status --root "$CINEDUPLEX_ROOT"
```

环境脚本先核验 SMB，再创建 NAS 子目录。它设置所有缓存/TMP 路径；不会设置或迁移凭据内容。默认令牌路径是用户自己的 `~/.cache/huggingface/token`。若原路径不同，在 source 前设置 `HF_TOKEN_PATH`。

**仅当独立环境尚不存在时安装：**

```sh
uv venv --python python3.13 "$CINEDUPLEX_ROOT/cache/codec-env-linked"
uv pip install --python "$CINE_PY" --link-mode symlink --only-binary :all: \
  -r requirements-codec-cpu.txt
uv pip check --python "$CINE_PY"
```

若 Python 3.13 不存在，先选一个已有兼容版本；不要让 uv 隐式下载解释器到本机。环境使用 NAS cache 的符号链接，**不要删除 `cache/uv-codec-v2`**。依赖列表固定了主要包，不是所有平台的完整锁文件；不要声称一次安装等于跨平台验证。

## 2. 准备最小输入

先查运行任务，避免与旧下载、安装或生产并发：

```sh
ps -axo pid,etime,state,command | rg '[p]ython.*cineduplex.data|[u]v pip'
"$CINE_PY" -B -u -m cineduplex.data.workflow prepare-inputs --root "$CINEDUPLEX_ROOT"
```

`prepare-inputs` 会：

1. 取固定版本来源 README；把许可卡留在 NAS。
2. 取 EmotionTalk `Audio.tar` 前 64 MiB，校验固定 SHA；从完整 TAR entries 恢复 19,250 条标注，不下载 14.8 GB 全库。
3. 获取两个选定 SmoothConv 录音及标注。
4. 获取四个 codec 大文件、flow.yaml 和六个 tokenizer 文件；按固定大小及 SHA256/Git blob SHA1 校验。
5. 写 `manifests/pilot-inputs-ready.json`。已有正确资产复用；已有错误资产拒绝覆盖。

固定路径、版本、大小及哈希见 `configs/data_sources.json`。网络内容先进入有限 RAM 缓冲，再逐块写 NAS，断点在 `.segments.partial` / `.progress.json`。401/403 是权限问题，应停止；不得绕过访问控制。来源文件不变时不要重新全量下载。

RealTalk 只做过来源调查，本配方不依赖其下载。HumDial-FDBench 不进入训练素材。

## 3. 制作话段和修订双工候选

```sh
"$CINE_PY" -B -u -m cineduplex.data.smoothconv --root "$CINEDUPLEX_ROOT" \
  --stem 1765799160_cuCZuOweWy_seg59_active

CINE_ANNOTATIONS=manifests/Emotiontalk-audio-annotations.json
if [ -f "$CINEDUPLEX_ROOT/manifests/Emotiontalk-audio-annotations.recovered.json" ]; then
  CINE_ANNOTATIONS=manifests/Emotiontalk-audio-annotations.recovered.json
fi
"$CINE_PY" -B -u -m cineduplex.data.pilot --root "$CINEDUPLEX_ROOT" \
  --annotations "$CINE_ANNOTATIONS"
"$CINE_PY" -B -u -m cineduplex.data.workflow revised --root "$CINEDUPLEX_ROOT"
```

第一版 `pilot10/` 保留历史选择；当前审阅只使用其中 S01–S08。`revised` 从真实原声道生成 `pilot10-revision2/`：

| 场景 | 来源窗口 | 用途 |
|---|---|---|
| S09 | `1765799160_cuCZuOweWy_seg59_active` 的 12–28 秒 | 包含被打断回答及后续接话的上下文 |
| S10 | `1769939396_AJdvYtW8AA_seg241_active` 的 34–47 秒 | 主句期间的真实并发附和及后续回答 |

`revised` 若发现完整清单已存在会拒绝覆盖。不要删除清单绕过；先核验现有来源与产物。新筛选材料可以调用 `smoothconv_select.py` 研究，但它不是自动听审器，也不能自动扩展固定十场景的验收范围。

S01–S08 只选 EmotionTalk G00002，排除官方验证/测试组。原文、原始情绪与 caption 保留，不把候选类别覆盖原始标签。来源音频被裁切、规范为 16k mono PCM16，但不能因此称为独立声轨。

## 4. 编码与 CPU 重建

```sh
"$CINE_PY" -B -u -m cineduplex.data.codec_cpu --root "$CINEDUPLEX_ROOT"
"$CINE_PY" -B -u -m cineduplex.data.codec_decode_cpu --root "$CINEDUPLEX_ROOT"
"$CINE_PY" -B -u -m cineduplex.data.codec_cpu --root "$CINEDUPLEX_ROOT" \
  --output-relative output/pilot10-revision2
"$CINE_PY" -B -u -m cineduplex.data.codec_decode_cpu --root "$CINEDUPLEX_ROOT" \
  --output-relative output/pilot10-revision2
"$CINE_PY" -B -u -m cineduplex.data.workflow consolidate --root "$CINEDUPLEX_ROOT"
```

编码是 raw codes（0–6560），offset=0，约25Hz。相同音频+固定encoder SHA 才能复用；重建严格加载固定 flow/HiFT/CampPlus，输出24k PCM16及收据。部分重建使用同一句作为 prompt，这是重建诊断，不是独立说话人克隆验证。

`consolidate` 核对原音/token/重建身份，生成10场景、28话段的 `pilot10-current/` 和空白听审队列，已有当前清单则拒绝覆盖，保护评审结果。它不自动批准训练。不要用旧 `pilot10` 的 S09/S10 代替修订版本。

## 5. 连续输入、真实 Dataset 与参考对照

```sh
"$CINE_PY" -B -u -m cineduplex.data.lychee_diagnostic --root "$CINEDUPLEX_ROOT" prepare
"$CINE_PY" -B -u -m cineduplex.data.codec_cpu --root "$CINEDUPLEX_ROOT" \
  --output-relative output/pilot10-loader-input
"$CINE_PY" -B -u -m cineduplex.data.codec_decode_cpu --root "$CINEDUPLEX_ROOT" \
  --output-relative output/pilot10-loader-input --scene S09
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$CINE_PY" -B -u \
  -m cineduplex.data.lychee_diagnostic --root "$CINEDUPLEX_ROOT" execute --include-expressive
"$CINE_PY" -B -u -m cineduplex.data.workflow prepare-reference --root "$CINEDUPLEX_ROOT"
"$CINE_PY" -B -u -m cineduplex.data.codec_frontend_check --root "$CINEDUPLEX_ROOT"
```

S09 额外连续片段直接取原 B1 声道，包含真实间隙，不拼接已有 token。它得到59个token，不算第11个场景。

数据诊断实际使用固定上游 Dataset/collator 和真实 tokenizer；只在 RAM 去掉无关训练导入，不修改原函数/类算法。S01–S08 作为因果话段对诊断；S09 检查打断后残余，S10 检查附和控制及8个语音token保留。14行通过不表示完整模型或完整原场景时间轴通过。

前处理对照从固定 s3tokenizer 0.2.0 源包加载原函数，29个输入全部逐code比较；不安装该包、不转换模型，不称为CUDA等价。

## 6. 特征、导出与听审

```sh
"$CINE_PY" -B -u -m cineduplex.data.prosody --root "$CINEDUPLEX_ROOT"
"$CINE_PY" -B -u -m cineduplex.data.pilot_export --root "$CINEDUPLEX_ROOT" \
  --output-relative output/pilot10-current
python3 -B -m cineduplex.data.workflow status --root "$CINEDUPLEX_ROOT"
```

交付入口是 `output/pilot10-current/review.html`。`prosody.jsonl` 保存实际测量，未校准的 confidence 为null，不能把低能量等同真实静音、把音高等同情绪。HF目录仅审阅schema，官方数据接口诊断另在loader-input目录，训练导出保持0。

听审要对比原音与重建：文字、说话人、情绪、噪声/串音。重点单列S08压抑及S09/S10原声道。只把真实反馈写入 `auditory-review-queue.json` 的对应条目，注明评审人、时间和范围；未知项保留null。单一场景反馈不批准其他场景。即使听审通过，S01–S08的连续独立声轨缺口仍存在，不能自动升级为完整双工数据。

## 7. 中断恢复与失败

- 先检查PID、最新stdout、产物及收据，再决定恢复哪个阶段。正常NAS慢导入不能直接判死并启动第二个任务。
- NAS不可访问：检查真实挂载、限时I/O、SMB日志；保留partial。不要强卸载繁忙共享、改网络或重启NAS，也不回退本机。
- `.tokens.json`已完成、loader失败：只重跑诊断，不重编码。
- 源hash、版本或音频格式不符：停止，保留失败记录，不覆盖旧数据掩盖差异。
- 新的公开workflow封装已做语法/针对性单测及现有包只读状态检查；**没有在发布时重跑全部下载和音频生产**。其底层步骤的原执行实证与新机器复现必须分别报告。
- 原私人环境的定时任务已暂停。接手只执行用户明确授权的工作，不自动恢复旧P0或训练。

## 8. 表演优先：独立参考重建与试听

用户已选择第一阶段优先表演感。原S01–S08的20个话段中，19个旧重建使用目标原句本身作参考；这不足以排除参考条件传递表演的影响。以下是四个固定目标token的补充对照，**不重编码、不训练、不覆盖旧音频**。

```sh
"$CINE_PY" -B -u -m cineduplex.data.codec_decode_cpu \
  --root "$CINEDUPLEX_ROOT" --output-relative output/pilot10-current \
  --prompt-policy independent-neutral \
  --destination-relative output/acting-neutral-control-v1 \
  --utterance G00002_10_08_002 --utterance G00002_13_08_006 \
  --utterance G00002_15_07_005 --utterance G00002_05_07_002
python3 -B -u -m cineduplex.data.acting_review --root "$CINEDUPLEX_ROOT"
```

运行前查现有进程和输出。独立参考模式拒绝覆盖已有WAV/收据；中断后只传缺失的utterance，不删除旧产物绕过检查。分批运行的decode-summary仅含本次批次；每话段收据和试听包逐条核验才是全部完成的依据。

参考必须来自同source revision、同split group、同speaker，且不同场景、不同文本，无标注重叠，原标签neutral；排除suppression候选场景。找不到则失败，不能回退原句。neutral是来源标注，尚非听审结论。

试听入口：`output/acting-review-v1/review.html`；`packet.json`保留20话段来源、原情绪、时间、原有/新对照收据。生成器校验原音/重建/参考的SHA与WAV格式、固定目标codes身份，并实际读回JSON/HTML。它不修改原听审队列，不批准训练。

先听前四项的原音、旧重建、新重建，分别反馈文字、声线、情绪/节奏保留和噪声。页面是有提示的诊断试听，不是随机盲测。参考音频特征、prompt token、speaker embedding一起变化，不能把全部差异归因于其中一个因素。S08仍是neutral原标签，不可把它的对照制作成功当作压抑确认；正式效果需要后续独立测试。
