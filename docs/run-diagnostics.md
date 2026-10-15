# 排查会诊为什么提前结束

先在项目目录执行只读诊断，不需要填写令牌或打印 `.env`：

```powershell
docker compose exec backend python -m scripts.inspect_run
```

默认检查最近一轮。也可以按地址栏 `/chat/` 后的会话 ID 检查该会话最近一轮，或指定运行 ID：

```powershell
docker compose exec backend python -m scripts.inspect_run --thread-id <会话ID>
docker compose exec backend python -m scripts.inspect_run --run-id <运行ID>
```

输出只包含阶段、耗时、产物是否存在、工具命中数量和模型调用状态，不输出用户原文、画像、模型提示词、请求头、密钥或原始异常。

`PREFLIGHT_SERVICE_UNAVAILABLE` 表示安全预审因调用故障未完成，不能解释为患者提供了危险信息。新增记录的 `failure.category` 与 `http_status` 区分上游 5xx、429、认证、超时、连接和输出校验异常，各次重试保留同样的脱敏字段；旧记录未保存状态码时不能反推具体上游错误。尚未执行问诊的故障轮不再报告“未记录病程”作为主要缺项。

网页问答在切换会话或刷新页面后继续后台执行，返回原会话时自动恢复进度。只有明确点击停止或达到五分钟服务端期限时取消；服务重启会中断尚未完成的任务。`cancel_reason` 区分 `user_stop`（用户停止）、`timeout`（超时）、`service_shutdown` / `service_restart`（服务关闭/重启）、`client_disconnect`（旧客户端断连）、`requested`（未细分的取消请求）。旧记录的 `unknown` 表示当时没有记录来源，不应据此推测具体操作。

按下面顺序定位：

1. `status` 是整个请求是否结束。`completed` 只表示成功保存了答复，追问答复也会是这个状态，不代表三个专家均已完成。
2. `stages` 看最后进入哪个节点：`bianzheng` → `formula` → `herb` → `reviewer` → `safety`。阶段的 `done` 表示该节点已返回，仍要看产物状态和后续路由。
3. `artifacts` 看各专家的 `status`、`has_plan`、`has_comparison`、`has_question`。`completed` 且没有个人处方但已有知识对照，是有效的知识分析结果，不能自动改称“缺少用药或过敏资料”。
4. `tools` 区分工具调用成功、检索未命中、依赖异常。`search_classics` 的 `no_match` 不等于 `search_formulas` 也没有结果。
5. `model_calls` 看失败、重试和耗时。模型成功、工具有结果时，应优先查产物解析和工作流分支，而非先改 API 配置。

`findings` 的含义：

| 标记 | 含义 |
| --- | --- |
| `FORMULA_COMPLETED_WITHOUT_PERSONAL_PLAN` | 方剂 Agent 返回已完成但无个人处方；知识分析模式允许这种结果 |
| `LITERATURE_AVAILABLE_BUT_HERB_STAGE_NOT_RUN` | 方剂已有检索结果和对照内容，但本草没有执行，重点查方剂退出路由 |
| `CLASSICS_SEARCH_NO_MATCH` | 至少一次古籍检索没有命中；不等于所有证据为空 |
| `NO_DURATION_IN_COMMITTED_SYMPTOM_FACTS` | 已保存的症状事实中没有病程，需核对补充资料是否所在轮次被取消；这不代表用户从未说过病程 |
| `TOOL_CALL_FAILED` | 至少一次工具调用失败，按工具名排查依赖或输入 |
| `RUN_ENDED_WITHOUT_FINAL_ANSWER` | 运行被取消或失败，本轮没有最终分析答复 |

2026-10-14 排查实例：运行 `76122278-a83e-4da2-8410-8bfc6b39370f` 用时约 112.7 秒，辨证和方剂模型调用均成功；方剂两次检索成功，返回 `status=completed`、`plan=null`、非空 `comparison`、空 `question`。旧路由把空个人处方当成需追问，又因症状事实缺少中断轮次的病程而拒绝继续知识分析，最终展示固定的用药/过敏追问。本草并未执行。修复后，在问诊对象明确、无待确认关键安全问题及症状冲突的知识分析模式下，这种产物继续本草与独立审查；个人处方仍需独立通过全部安全检查。
