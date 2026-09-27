# 岐问接口与页面适配

依据恢复后的 `backend/app/main.py` 和当前前端编写。接口以 `/api` 开头；登录、注册、健康检查以外的业务接口需要 Bearer token。资源访问按用户归属校验。

| 接口 | 请求与返回 |
| --- | --- |
| GET /health | 状态、版本、模型与检索模式 |
| POST /register、/login | username、password；返回 access_token、token_type、user.id 与 user.username |
| GET /me | 当前用户 id、username；画像请求使用此 id |
| GET /threads | 本人会话列表：id、title、mode、created_at、updated_at |
| POST /threads | 请求 mode，默认 fixed；标题通过 PUT 修改 |
| PUT /threads/{id} | 请求 title，不是 new_title |
| DELETE /threads/{id} | 204 空响应；清理会话消息、运行和检查点 |
| GET /threads/{id}/messages | 返回 role、content、notes、trace、prescription 等持久化字段 |
| GET、PATCH /profiles/{user_id} | 仅访问本人；可编辑 constitution、allergies、history |
| GET /profile-suggestions?thread_id=… | 本人会话待确认候选、原文、冲突和 revision |
| POST /profile-suggestions/resolve | revision、decisions；每项 id、action=accept/dismiss/other，接受须 confirm_self，冲突须 replace，可编辑 entity/details |
| DELETE /profile-facts/{id}?revision=… | 撤销本人已确认资料，版本不符 409 |
| POST /upload | multipart 的 file；返回 id、name 等；聊天 files 引用上传记录 id |
| POST /chat | message、thread_id、files、mode、run_id；?stream=true 获取 SSE |
| POST /runs/{run_id}/cancel | 取消本人运行；停止后不能提交成功答复 |

SSE 信封包含 run_id、candidate_id、sequence、type、payload。后端独立保存候选与审查；当前页面接收 stage 进度，answer.final 将正文与会话 id 交给聊天状态管理，再加载历史。run.failed 显示错误，run.cancelled 显示已停止。协议细节见[扩展设计](结构化剂量与候选协议.md)。

页面仍用 thread_id、patient_id 等展示字段，backend-contract.ts 转换；expert/content 转为 specialist/reasoning，trace.events 与 elapsedMs 恢复阶段与耗时。新增画像确认、来源和撤销入口，侧栏手填过敏史。画像包含 safety_revision、facts；PATCH 可携带 revision，空白表示未知，明确“无”表示自述否认。

确认、去重、主体和事务协议见[安全预审与画像确认](安全预审与画像确认.md)。trace.safety 保存咨询对象、状态、待追问字段和版本；restricted 不表示放行。

trace.inquiry 保存普通症状 facts 与待确认 pending，包括原文位置、消息 ID、主体、肯定状态、时间、程度和变化；由其派生的 symptoms 仅包含当前患者的当前肯定症状。协议与规则边界见[普通问诊症状解析](普通问诊症状解析.md)。

未收录症状出现 `pending.kind=mapping`，确认后 `mapping_confirmed=true`，原文继续检索。`trace.classics_retrieval.attempts/final` 保存 query_plan、active_mode、vector_status、rerank、mmr、selected 和 filtered_count/stale_count/truncated_count。rerank 包含 CrossEncoder 原始 logit、一次 sigmoid 分数、threshold=0.7；状态 success/no_candidates/model_unavailable/failed/disabled。mmr 包含候选模型、λ、重复数及 success/not_needed/failed；selected 含两路名次、RRF、rerank_score/logit、mmr_rank/score/max_selected_similarity。Token 为空的本地 role=cross_encoder 记在 model_calls。完整流程见[古籍混合检索与术语扩展](古籍混合检索与术语扩展.md)。

切片来源保存于 `selected.source_metadata`，包含存在的书名、章节/类别、药名/条文号、`split_from` 及 `source_chunk_id`；字段随检索结果和历史轨迹保留。切片与导入入口见[结构切片流程](古籍结构切片流程.md)。

登录错误的 401 在表单中提示，不立刻重载。其他受保护请求 401 清登录态并跳登录页。422 字段错误转为可读文字；删除返回 204 时不读取 JSON。

最终答复与历史消息的 `trace.memory_retrieval` 保存实际记忆检索模式、退化原因与选中摘要来源。`rerank` 子项记录 requested_mode、status、fallback_reason、input_count、output_count；status 可为 success、no_candidates、disabled、skipped_unconfigured、fallback，后四项不能当作成功模型重排。配置和协议见[会话记忆检索](会话记忆检索.md)。
