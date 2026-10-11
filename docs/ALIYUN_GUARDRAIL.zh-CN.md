# CineDuplex 阿里云安全护栏接入

公开包提供 `cineduplex.guardrail` 审核客户端、回答缓存器和 Lychee-FD 通话服务集成源码；凭据和实际部署配置不公开。与备案无关。源码完成不等于现有运行进程已加载，也不等于阿里云真实调用通过。

## 审核范围

启用后，后端先缓存一个完整原生回答的文字和声音，在回答及声音生成结束后审核文字，通过才发布。不能通过部分文字推断已经审核对应的声音。当前模型没有经过验证的逐词文字/音频映射，因此本版不做片段提前放行。等待期间麦克风继续收音，原生模型仍可取消回答；取消、挂断和换 epoch 后的审核结果不会重新发布旧声音。

可选输入音频审核：每 4 秒提交一次 16 kHz PCM 的 WAV，包含上一窗口最后 1 秒的重叠；回答结束时提交尚未送审的剩余输入。回答等待这些输入的审核结果。发现风险或审核失败时清空输出并停止通话。它属于**输入监测并停止**：音频已经进入模型，不能称为模型接收前的提示词过滤；短窗口与重叠也不证明任意跨窗口语义都被识别。

可选回答音频审核：将实际 24 kHz 声音生成 WAV，在文字审核后调用语音审核。只有文字和声音都通过才发布。因此不能把只启用文字审核称为完整语音审核。语音接口使用临时上传令牌，不需要新增自有 OSS Bucket；WAV 在内存生成，不写本地临时文件。音频会传到阿里云的审核临时存储（官方说明保留 30 分钟），页面会提示。

语音服务是异步文件审核，可能等待数十秒；本版优先确保未通过的回答不会发给浏览器，不能称为低延迟语音护栏。默认音频审核关闭，界面明确显示实际范围。启用哪些音频审核需要显式配置，不能静默增加音频上传和费用。

## 服务配置

先开通 AI 安全护栏；如需审核实际音频，另开通内容审核增强版。开通不代表运行进程已启用。专用 RAM 用户仅需实际使用的四项权限：`yundun-greenweb:MultiModalGuard`、`yundun-greenweb:DescribeUploadToken`、`yundun-greenweb:VoiceModeration`、`yundun-greenweb:VoiceModerationResult`。操作不支持资源级授权，Resource 使用 `*`；Action 不使用通配符。仅使用文字审核时只需第一项权限。见 [官方授权信息](https://help.aliyun.com/zh/document_detail/2528739.html)。

在与通话后端相同的执行环境开通中国内地 AI 安全护栏。按需另开通语音审核增强版。使用 RAM 凭据，不把密钥放前端、命令参数或仓库。控制台为所选 Service 启用内容合规、提示词攻击与敏感内容等需要的维度；代码不能证明控制台维度全部开启。

后端读取：

```sh
export CINEDUPLEX_GUARDRAIL_ENABLED=true
export CINEDUPLEX_GUARDRAIL_REGION=cn-shanghai
export CINEDUPLEX_GUARDRAIL_CREDENTIALS_FILE=/run/secrets/cineduplex-aliyun.json
# 启用下列项会发送实际音频，并产生语音审核费用。
export CINEDUPLEX_GUARDRAIL_INPUT_AUDIO=true
export CINEDUPLEX_GUARDRAIL_OUTPUT_AUDIO=true
```

凭据文件须为普通文件，权限 0600，包含 `access_key_id`、`access_key_secret`，STS 凭据另含 `security_token`。也可从 `ALIBABA_CLOUD_ACCESS_KEY_ID`、`ALIBABA_CLOUD_ACCESS_KEY_SECRET`、`ALIBABA_CLOUD_SECURITY_TOKEN` 注入。不要发到聊天里。使用临时凭据时应由外部部署系统更新并重新构建客户端，不会自动续期。

默认 Service：文字输入 `query_security_check_pro`，文字输出 `response_security_check_pro`，实际语音 `audio_detection_byllm`。通话的音频输入使用语音审核服务；本版没有捏造用户文本，也没有默认新增 ASR 服务。若要检查独立文本输入，从 `cineduplex.guardrail.aliyun_guardrail` 导入 `AliyunGuardrail`，直接使用 `AliyunGuardrail.check_text(..., direction='input')`。

可以通过 `CINEDUPLEX_GUARDRAIL_INPUT_SERVICE`、`CINEDUPLEX_GUARDRAIL_OUTPUT_SERVICE`、`CINEDUPLEX_GUARDRAIL_AUDIO_SERVICE` 选择控制台自定义 Service。默认 HTTPS 请求超时 5 秒、每次音频审核总期限 45 秒。分别用 `CINEDUPLEX_GUARDRAIL_TIMEOUT`、`CINEDUPLEX_GUARDRAIL_AUDIO_DEADLINE` 调整；输入窗口用 `CINEDUPLEX_GUARDRAIL_INPUT_WINDOW` 调整，允许 1–10 秒。

## 接线和验证

私有集成中，`Application` 创建云客户端；每个 `Call` 有独立审核器。`Call.emit` 拦截原生文字与音频事件，审核器通过后才调用 `_emit`。输入 HTTP 序号去重之后才送审，网络重试不会重复提交同一音频包。每个响应最多 2000 字、60 秒音频；超过限制停止，不拆分成失去完整语义的独立文字请求。缓冲、队列、期限都有上限。

`pass` 才允许发布。`block`、`watch`、`mask`、响应缺字段、超时、欠费/鉴权失败等都停止通话。音频只接受聚合及各切片均为 `none`。取消异步文件审核只能停止本地等待，不能撤回已提交云任务；已提交任务仍可能计费。日志只新增审核请求编号、结果、固定错误码、范围和耗时，不新增原文、音频、密钥或风险词。

仅查配置，不发云请求：

```sh
PYTHONPATH=src python3 -B scripts/check_guardrail.py
```

明确要执行一次付费云端文字探测时：

```sh
PYTHONPATH=src python3 -B scripts/check_guardrail.py --live
```

这是固定无敏感信息的文字探测，不是实际语音审核、GPU运行或真人通话验证。结果中的 `configuration.cloud_verified=false` 表示配置本身不构成云验证；`live_request` 与 `decision` 才是这次云调用证据。

已有四节点运行器使用冻结源码和 RAM 导入，不会随 checkout 变更自动加载。激活必须把 `cineduplex.guardrail` 两个模块与私有 `realtime_service.py` 一起纳入原运行器的来源清单，并只在原会话空闲时更新；凭据必须进入原 leader 的进程环境。禁止为接入护栏重复加载模型。当前文档不授权重启、扩大公网访问或恢复暂停任务。

## 官方接口依据

- [多模态实时审核 API](https://help.aliyun.com/zh/document_detail/2937221.html)
- [原生签名调用示例](https://help.aliyun.com/zh/document_detail/2925843.html)
- [语音审核服务、临时存储及结果查询](https://help.aliyun.com/zh/document_detail/3030647.html)
- [语音 SDK 的临时上传令牌流程](https://help.aliyun.com/zh/document_detail/604972.html)


## 区分阿里拦截、模型拒绝和本地停止

每次审核的 `guardrail_result` 保留 `call_id`、输入窗口 `input_index` 或回答
`event_id`、`provider`、`allowed`、`request_id`、风险标签及耗时。语音转写不进入这些日志。
每个准备释放的回答或导致通话停止的决定，另记 `safety_outcome`：

- `aliyun_block`：阿里返回非通过回执；记录原始请求编号、审核阶段和风险标签。
- `application_stop`：本地一致性检查、审核失败或超时等导致停止；不能计入阿里内容拦截率。
- `model_refusal_expression`：已启用审核均通过，模型回答中检测到明确拒绝表达。
- `model_response`：已启用审核均通过，未检测到明确拒绝表达；不代表模型同意危险请求。

模型没有结构化拒绝信号，后两项使用本地文字规则，记录
`refusal_detection=text_heuristic_v1` 与 `safety_verdict=not_assessed`。
引用拒绝话语、先拒绝后提供危险内容等仍可能命中，因此它是文字观察，不能作为安全认证。
该判断不调用新的模型或云接口，不改变整段审核、输出释放或插话规则。
`approved_for_release` 只代表服务端审核完成，实际播放须另核对浏览器 `audio_played`。
同一通话可以先正常回答、后被拦截，应按回答及输入窗口查看，不能只取通话最后一个状态。

日志在运行服务的 `calls/<call_id>/events.jsonl`，页面诊断下载保留这些事件。
页面同时显示来源说明。新增来源事件只对部署后的通话生效，旧日志不回写；
历史通话需结合已有云端回执和 `event_end` 回答人工判定。
