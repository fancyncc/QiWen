# 岐问部署与运行配置

## 上下文、病程和记忆上限

`QIWEN_MODEL_CONTEXT_LIMIT` 默认 32768、`QIWEN_MODEL_OUTPUT_RESERVE` 默认 4096；可用 `QIWEN_MODEL_<ROLE>_CONTEXT_LIMIT/OUTPUT_RESERVE` 分角色覆盖。数值应按实际提供商容量配置，上限需大于输出预留及 512 安全余量。输入以 UTF-8 字节上界保守估算，轨迹标注估算方式，并另存提供商实际 usage；必要资料过长会请求分批处理，不删除关键风险信息。

`QIWEN_EPISODE_RECHECK_DAYS` 默认 14，是同一会话重开旧问诊时的续问确认阈值，不表示事实医学上失效。`QIWEN_CHECKPOINT_SUCCESS_KEEP` 默认 3，`QIWEN_CHECKPOINT_STOPPED_KEEP` 默认 2，按会话保留运行检查点，正在执行的运行不清理。

前端使用系统字体，无构建时 Google 字体下载依赖。新增表及中文记忆索引在启动时幂等升级；数据库备份流程继续适用。

以下 PowerShell 命令从项目根目录执行。依赖与入口按当前 `compose.yaml`、Dockerfile 和配置代码整理；每次部署的实际结果需重新验证，不能沿用历史记录。

## Docker 启动

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:23000/api/health
(Invoke-WebRequest http://127.0.0.1:23000/chat -UseBasicParsing).StatusCode
```

确认 backend、frontend 均为 healthy，健康接口返回 `status=ok`，`/chat` 返回 HTTP 200。打开 [登录页](http://127.0.0.1:23000/login)，首次使用先注册；用户名为 3～32 位字母、数字或下划线，密码长度为 8～128。没有通用管理员密码。

前端绑定 `127.0.0.1:23000`，通过 `BACKEND_URL=http://backend:8000` 转发 `/api/*`；后端仅在 Compose 网络中暴露 8000。可通过 `QIWEN_WEB_PORT` 调整前端端口。容器内分别使用 Python 3.12 和 Node.js 22。

修改前端、后端或运行配置后，交付前必须重新执行 `docker compose up --build -d`，再检查两个服务与两个 URL。构建成功、测试通过和健康接口能访问分别代表不同检查，未完成的部署步骤需明确说明。

## 模型配置

从根目录 [.env.example](../.env.example) 复制到 `.env`，在本地编辑配置。不要输出或提交真实值，不把密钥放入浏览器公开变量、命令参数、请求日志或报告。

| 变量 | 默认或用途 |
| --- | --- |
| `QIWEN_MODEL_MODE` | `mock` 演示；`openai_compatible` 使用真实聊天接口 |
| `QIWEN_MODEL_BASE_URL` | 兼容接口基址 |
| `QIWEN_MODEL_API_KEY` | 服务端凭据，仅在本地配置 |
| `QIWEN_MODEL_NAME` | 默认 `qwen-plus`，对应青云接口请求的 `model` 字段 |
| `QIWEN_MODEL_<ROLE>` | 角色模型，空值回退默认模型 |
| `QIWEN_MODEL_TIMEOUT`、`QIWEN_MODEL_RETRIES` | 默认单次 45 秒、额外重试 2 次 |
| `QIWEN_MODEL_<ROLE>_TIMEOUT / _RETRIES` | 角色覆盖；超时范围 0.05～300 秒，重试 0～5 次 |
| `QIWEN_ENV`、`QIWEN_SECRET_KEY` | production 必须配置签名密钥；development 可生成并保存开发签名 |
| `QIWEN_CORS_ORIGINS` | 允许的浏览器来源 |
| `QIWEN_DATA_DIR` | 本地默认根目录 `.data/`；Docker 为 `/app/.data` |

角色入口包括问诊、安全预审、摘要、画像提取、检索评估、查询重写、辨证、方剂、本草、综合、独立审查；注册配置入口不代表每个规则节点都已切换为模型实现，以 `trace.model_calls` 为准。兼容接口失败会明确报错，不将 Mock 当作真实答复。

纯知识问答使用 `QIWEN_MODEL_KNOWLEDGE`，未配置时回退默认模型；同样支持 `_TIMEOUT` 与 `_RETRIES`。Mock 仅展示实际检索结果，不模拟无来源的知识解释。

## 古籍检索与本地模型

默认请求 `QIWEN_RAG_MODE=hybrid`，向量依赖兼容嵌入接口；未配置时实际召回退为 BM25。本地重排与 MMR 独立于聊天 Mock 和聊天密钥。正常请求只加载缓存，首次部署显式准备模型：

```powershell
docker compose exec backend python -m scripts.prepare_ranking
```

默认下载 `BAAI/bge-reranker-v2-m3` 与 `BAAI/bge-small-zh-v1.5`，缓存位于数据卷 `models/`。Docker 默认安装 CPU PyTorch、Sentence Transformers 和 Chroma；本地开发另装 CPU PyTorch 与 `backend/requirements.rerank.txt`。

| 变量 | 默认与用途 |
| --- | --- |
| `QIWEN_INSTALL_KNOWLEDGE`、`QIWEN_INSTALL_RERANK` | Compose 构建参数均默认 true |
| `QIWEN_RAG_RERANK_MODE` | `cross_encoder`；`none` 显式禁用门槛与 MMR，旧 `llm` 不再支持 |
| `QIWEN_EMBEDDING_MODEL` | 古籍向量召回模型，复用兼容基址与服务端凭据 |
| `QIWEN_RAG_MIN_SIMILARITY` | 可选向量召回余弦门槛，需按实际嵌入模型评估 |
| `QIWEN_CROSS_ENCODER_MODEL / REVISION / DEVICE` | 默认上述重排模型 / 空 / cpu |
| `QIWEN_CROSS_ENCODER_MAX_LENGTH / BATCH_SIZE / TIMEOUT` | 512 tokens / 2 / 120 秒（含等待锁与加载） |
| `QIWEN_LOCAL_MODEL_THREADS / CACHE` | 2 / 默认数据目录下 `models/` |
| `QIWEN_RAG_RECALL_K / CANDIDATES` | 每路 30 条 / 融合 30 条；允许范围 4～100 / 4～40 |
| `QIWEN_RAG_MMR_LAMBDA` | 0.7，允许 0～1 |
| `QIWEN_RAG_MMR_EMBEDDING_BACKEND` | local；api 可使用兼容嵌入接口 |
| `QIWEN_MMR_EMBEDDING_MODEL / REVISION` | 默认上述本地嵌入模型 / 空 |

CrossEncoder sigmoid ≥0.7 的门槛固定在程序中。旧 `QIWEN_MODEL_CLASSICS_RERANK` 不再驱动古籍重排。前端代理超时为 900000 毫秒，后端推理仍受独立超时约束。

配置有效嵌入接口与模型后，建立古籍向量索引：

```powershell
docker compose exec backend python -m scripts.index_vectors
```

更换嵌入模型需重建相应集合；索引可逐批续建，查询不会自动补齐部分索引。完整真实聊天、向量、CrossEncoder、MMR 验收使用：

```powershell
docker compose exec backend python -m scripts.check_live
```

缺少聊天或嵌入配置不能完成该验收。健康接口的 `rag_mode=hybrid` 不证明实际向量与重排成功，需检查检索轨迹，见 [rag.md](rag.md)。

## 会话记忆边界

会话内保留事实与病程摘要，跨会话只参考用户画像。跨会话症状检索、嵌入及记忆重排已移除；`QIWEN_MEMORY_*` 和 `QIWEN_MODEL_MEMORY_RERANK*` 不再生效，无需为了此次升级修改或展示本地凭据配置。古籍 RAG 及其嵌入、重排配置继续使用。

旧摘要与控制状态留在数据库，不自动清空或重新开启；后续咨询不会发布跨会话症状摘要。数据库初始化不再回填旧摘要关键词索引。

## 本地开发

后端建议 Python 3.12；项目根目录：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.lock.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

另开终端，从项目根目录启动前端：

```powershell
Set-Location frontend
npm ci
Copy-Item .env.local.example .env.local
npm run dev
```

访问 `http://127.0.0.1:3000`。本地 `BACKEND_URL` 默认 `http://localhost:8000`。测试与生产构建命令见 [testing.md](testing.md)。

## 数据备份与恢复

Docker 命名卷保存账号、业务库、检查点、上传文件与模型缓存；`docker compose down` 保留数据，`down -v` 会删除业务卷。本地 `.data/` 与 Docker 卷独立，代码回退不会自动迁移两者。

`scripts.data_snapshot` 使用 SQLite 在线备份和 SHA-256 清单打包两个数据库及上传文件，向量索引不打包，可从 corpus 重建；模型缓存与开发签名文件也不在快照中。快照含业务资料，应妥善保存。备份时暂停会写数据的请求，以保持两个数据库与上传文件的一致性。

```powershell
docker compose exec backend python -m scripts.data_snapshot backup --output /app/.data/backups/qiwen-snapshot.zip
```

把快照另存到宿主机或外部存储，卷内备份会随卷删除。恢复先写入一个空目录，确认校验和与目标无误后再安排数据切换。宿主机命令从 backend 目录执行：

```powershell
..\.venv\Scripts\python.exe -m scripts.data_snapshot restore --archive "快照路径.zip" --target-dir "空的恢复目录"
```

恢复数据库后重新核对签名配置，必要时重新登录；向量与本地模型缓存按需重建。不要用旧备份静默覆盖新事实，旧程序不能读取新安全状态时也不能声称安全升级仍有效。

## 可选只读 MCP

```powershell
docker compose --profile mcp up --build -d
```

HTTP 入口为 `http://127.0.0.1:28001/mcp`，客户端在本地配置岐问登录凭据；不在文档、日志或聊天中输出请求头与令牌。HTTP 从 JWT 读取身份，每次画像读取重新检查权限；尚未实现客户端 OAuth 自动登录。

本地 stdio 需安装 `backend/requirements-tools.txt`，设置 `QIWEN_MCP_USER_ID` 为本人账号 ID，在 backend 目录执行 `python -m app.mcp_server`。公共知识工具无需患者参数；`get_my_patient_profile()` 只返回本人的 constitution、allergies、history，要求数据库 `profile:read` 权限，不返回会话、摘要或上传文件，也不修改画像。身份未绑定或权限撤销时拒绝画像读取。

MCP 与 API 应共享业务数据目录与签名配置。注册默认授予画像只读权限，旧账号幂等补授一次，重启不会恢复撤销权限。`profile_access_audit` 保存访问结果；跨患者工具尚未提供。远程部署需配置实际 issuer/resource 地址、HTTPS 与访问边界。
