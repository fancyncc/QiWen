# 岐问接口与页面适配

## 画像与旧摘要兼容

跨会话只参考本人手填或明确确认的画像。画像读取不返回自动问诊摘要。`GET /api/memories` 与 `GET /api/threads/{id}/context` 已移除，聊天页不再提供独立记忆面板。

保留 `DELETE /threads/{id}/memory`、`DELETE /profiles/{user_id}/memories/{thread_id}` 处理旧摘要，204 表示删除旧摘要及级联缓存，不删除聊天。`PATCH /threads/{id}/memory` 兼容保留 enabled/revision，旧 revision 返回 409；开启设置也不会生成跨会话摘要。以上路径省略 `/api`，要求登录及账号归属校验。

删除聊天级联删除运行、问诊状态、病程摘要、知识话题及检查点。撤销长期资料不会自动否认同会话中已经自述的风险。已有摘要数据不自动清空，但不再提供给任何工作流。

依据当前 `backend/app/main.py` 和前端适配代码编写。以下表格路径均省略 `/api` 前缀。健康、注册、登录与公开知识出处无需登录；其余接口需在客户端配置登录凭据，并按用户归属校验。不要在文档或日志中输出请求头、访问令牌或 API 密钥。

| 接口 | 请求与返回 |
| --- | --- |
| GET /health | 状态、版本、模型与检索模式 |
| GET /knowledge/sources/{source_id} | 公开剂量知识出处与核验状态，不返回患者信息或核验人员身份 |
| POST /register、/login | username、password；返回 access_token、token_type、user.id 与 user.username |
| GET /me | 当前用户 id、username；画像请求使用此 id |
| GET /threads | 本人会话列表：id、title、mode、created_at、updated_at |
| POST /threads | 请求 mode，默认 fixed；标题通过 PUT 修改 |
| PUT /threads/{id} | 请求 title，不是 new_title |
| DELETE /threads/{id} | 204 空响应；清理会话消息、运行和检查点 |
| GET /threads/{id}/messages | 返回 role、content、notes、trace、prescription 等持久化字段 |
| GET、PATCH /profiles/{user_id} | 仅访问本人；可编辑 constitution、allergies、history |
| DELETE /profiles/{user_id}/memories/{thread_id} | 删除本人指定会话摘要及关联缓存；204，不删除会话正文 |
| GET /profile-suggestions?thread_id=… | 本人会话待确认候选、原文、冲突和 revision |
| POST /profile-suggestions/resolve | revision、decisions；每项 id、action=accept/dismiss/other，接受须 confirm_self，冲突须 replace，可编辑 entity/details |
| DELETE /profile-facts/{id}?revision=… | 撤销本人已确认资料，版本不符 409 |
| POST /upload | multipart 的 file；返回 id、name 等；聊天 files 引用上传记录 id |
| GET /uploads、DELETE /uploads/{file_id} | 本人上传列表与删除；删除为 204 |
| POST /chat | message、thread_id、files、mode、run_id；?stream=true 获取 SSE |
| POST /runs/{run_id}/cancel | 取消本人运行；停止后不能提交成功答复 |
| GET /runs/{run_id} | 本人运行状态与元数据 |
| GET /runs/{run_id}/candidates、/runs/{run_id}/events | 本人运行的候选版本与事件，不作为普通历史正文 |

## 请求与错误约束

聊天请求的 `message` 为 1～4000 字符；`files` 最多 5 个本人上传记录 ID。`run_id` 可省略由服务端生成，显式提供时为 8～64 位字母、数字、下划线或连字符。`mode` 只接受 fixed / multi，新会话默认 fixed，已有会话以存储模式为准。同一会话的活动运行或重复 run_id 返回 409。

会话 `title` 为 1～80 字符。画像 constitution、allergies、history 长度上限分别为 500、500、1000；PATCH 可携带非负 revision，字段不能为 null。画像确认每批 1～30 项，冲突或版本变化需刷新后重试。文件支持带文本 PDF、DOCX、UTF-8 TXT，单文件不超过 10 MB；扫描 PDF 不支持 OCR。

401 表示登录凭据无效；跨用户画像访问为 403，不存在或不属于用户的会话、附件与运行为 404；版本、运行冲突为 409；字段不合规为 422。返回 204 的删除请求不读取 JSON。

## SSE 与候选协议

SSE 信封包含 run_id、candidate_id、sequence、type、payload。后端独立保存候选与审查；当前页面接收 stage 进度，answer.final 将正文与会话 id 交给聊天状态管理，再加载历史。run.failed 显示错误，run.cancelled 显示已停止。协议与提交边界见 [agent.md](agent.md)。

| 事件 | 含义 |
| --- | --- |
| `candidate.started / delta / completed` | 创建新候选、流式增量与结构化产物；尚未代表审查通过 |
| `review.started / rejected / approved` | 独立审查、拒绝或批准；修正版本使用新 candidate_id |
| `stage / phase` | 实际节点进度与运行阶段 |
| `answer.final` | 原子提交后的权威答复，兼容顶层 done/response/message/trace |
| `run.cancelled / run.failed` | 终止运行，清理临时状态并拒收迟到结果 |

当前页面不把未审查候选写入答复气泡。连接断开也请求取消；完成与取消竞争以最终事务为准，已提交结果可从历史恢复。

`multi` 模式还返回 `agent.action`、`agent.observation`、`agent.completed`。事件只包含 Agent 名称、工具选择、结构化参数、裁剪后的 Observation 与停止原因，不包含模型思维链。最终 `trace.agents` 保存三个 Agent 的结构化产物和可审计工具轨迹。

页面仍用 thread_id、patient_id 等展示字段，backend-contract.ts 转换；expert/content 转为 specialist/reasoning，trace.events 与 elapsedMs 恢复阶段与耗时。新增画像确认、来源和撤销入口，侧栏手填过敏史。画像包含 safety_revision、facts；PATCH 可携带 revision，空白表示未知，明确“无”表示自述否认。

## 画像与检索轨迹

确认、去重、主体和事务协议见 [agent.md](agent.md)。trace.safety 保存咨询对象、状态、待追问字段和版本；restricted 不表示放行。

trace.inquiry 保存普通症状 facts 与待确认 pending，包括原文位置、消息 ID、主体、肯定状态、时间、程度和变化；由其派生的 symptoms 仅包含当前患者的当前肯定症状。协议与规则边界见 [agent.md](agent.md)。

未收录症状出现 `pending.kind=mapping`，确认后 `mapping_confirmed=true`，原文继续检索。`trace.classics_retrieval.attempts/final` 保存 query_plan、active_mode、vector_status、rerank、mmr、selected 和 filtered_count/stale_count/truncated_count。rerank 包含 CrossEncoder 原始 logit、一次 sigmoid 分数、threshold=0.7；状态 success/no_candidates/model_unavailable/failed/disabled。mmr 包含候选模型、λ、重复数及 success/not_needed/failed；selected 含两路名次、RRF、rerank_score/logit、mmr_rank/score/max_selected_similarity。Token 为空的本地 role=cross_encoder 记在 model_calls。完整流程见 [rag.md](rag.md)。

切片来源保存于 `selected.source_metadata`，包含存在的书名、章节/类别、药名/条文号、`split_from` 及 `source_chunk_id`；字段随检索结果和历史轨迹保留。切片与导入入口见 [data_pipeline.md](data_pipeline.md)。

登录错误的 401 在表单中提示，不立刻重载。其他受保护请求 401 清登录态并跳登录页。422 字段错误转为可读文字；删除返回 204 时不读取 JSON。

新运行不再生成 `trace.memory_retrieval`；历史消息中的旧轨迹只用于查看已完成运行。`trace.context_usage` 记录当前会话事实、适用画像与公开证据的实际 injected/dropped IDs。历史轨迹不会成为新会话输入。
