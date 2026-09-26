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
| POST /upload | multipart 的 file；返回 id、name 等；聊天 files 引用上传记录 id |
| POST /chat | message、thread_id、files、mode、run_id；?stream=true 获取 SSE |
| POST /runs/{run_id}/cancel | 取消本人运行；停止后不能提交成功答复 |

SSE 信封包含 run_id、candidate_id、sequence、type、payload。后端独立保存候选与审查；当前页面接收 stage 进度，answer.final 将正文与会话 id 交给聊天状态管理，再加载历史。run.failed 显示错误，run.cancelled 显示已停止。协议细节见[扩展设计](结构化剂量与候选协议.md)。

页面仍用 thread_id、patient_id 等展示字段，backend-contract.ts 负责转换，后端不为页面变更数据库结构。expert/content 转为 specialist/reasoning；trace.events 与 elapsedMs 恢复阶段和耗时。当前页面只编辑体质与过敏史。

登录错误的 401 在表单中提示，不立刻重载。其他受保护请求 401 清登录态并跳登录页。422 字段错误转为可读文字；删除返回 204 时不读取 JSON。
