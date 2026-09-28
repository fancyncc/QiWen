# 岐问

岐问是自行实现的中医知识问答与会诊流程演示。参考项目用于理解功能和流程，后端使用 `backend/app/` 中的自有实现，不再整包复制上游代码。当前保留淡蓝色登录页、聊天页、侧栏和参考页面布局。更新时间：2026-09-27。

## 启动与登录

在项目根目录执行：

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:23000/api/health
```

打开[登录页](http://127.0.0.1:23000/login)。首次使用先注册：用户名为 3 至 32 位字母、数字或下划线，密码为 8 至 128 位。克隆前已有账号继续使用原密码。回退后登录凭据不通用，需要重新登录；密码错误会在表单中提示。项目没有通用管理员密码。

默认聊天模型为 `mock`，古籍检索请求 `hybrid`，无需聊天密钥即可演示问诊。古籍重排默认本地 CrossEncoder，首次执行 `docker compose exec backend python -m scripts.prepare_ranking` 准备权重；缺少模型时不引用未过门槛的片段。真实聊天配置从 [.env.example](.env.example) 复制到 `.env`，设置 `QIWEN_MODEL_MODE=openai_compatible`、`QIWEN_MODEL_BASE_URL`、`QIWEN_MODEL_API_KEY`、`QIWEN_MODEL_NAME` 后重新构建。聊天接口失败时返回错误，不将演示结果当成真实模型答复。浏览器公开变量不能存放模型密钥。

## 使用流程

登录 → 创建或选择会话 → 编辑体质和过敏史 → 输入问题或附加文件 → 发送 → 查看阶段与最终答复。上传支持带文本的 PDF、DOCX、UTF-8 TXT，单文件不超过 10 MB，单轮最多 5 个附件。扫描 PDF 不支持 OCR。历史会话可以重命名、删除和再次打开。

后端继续使用克隆前的问诊、记忆、检索、安全审查、上下文压缩与提交流程。新会话默认 `fixed`，保持原有固定节点行为；`multi` 将辨证、方剂和本草升级为受限 ReAct Agent，各自在允许的只读工具中选择行动、读取结果并判断继续或结束，外层程序仍固定控制安全与最终提交。当前保留的页面没有新增模式选择器。具体节点、工具和重试上限见[问答流程](docs/问答流程.md)。

普通问诊按患者与时间保存症状事实，保留否定范围、程度和原文来源；不明确或互相矛盾时先确认，当前肯定症状才参与检索。实现及边界见[普通问诊症状解析](docs/普通问诊症状解析.md)。

古籍术语表扩展至 54 项；未收录不适保留原文并确认。检索采用 BM25＋向量召回 → 片段 ID RRF → 本地 CrossEncoder → sigmoid ≥ 0.7 → 候选余弦与 MMR（默认 λ=0.7），保留现代症状及扩展词出处。完整配置、索引迁移、失败行为和轨迹见[古籍混合检索与术语扩展](docs/古籍混合检索与术语扩展.md)。召回嵌入接口未配置时退为 BM25；本地重排与 MMR 不依赖聊天密钥。

记忆读取包含会话问题补全、摘要时间与来源、BM25/向量 RRF 融合及独立模型重排。融合后最多 10 条候选交给模型筛选、排序，最终最多 3 条，也允许全部拒绝。嵌入未配置时退化为 BM25；重排未配置或失败时明确记录并保留融合排序。当前实际运行 BM25、模型重排未配置；配置与验收边界见[会话记忆检索](docs/会话记忆检索.md)。

页面显示阶段进度，收到 `answer.final` 后展示最终正文。候选和审查仍由后端执行、独立保存，未通过审查的候选不进入答复气泡。停止按钮中止读取并请求服务端取消，后端也处理断线取消。

`multi` 运行还会发送 `agent.action`、`agent.observation` 和 `agent.completed` 事件，记录 Agent 名称、所选工具、结构化参数、裁剪后的工具结果和停止原因；不请求、保存或展示模型思维链。

对话中新出现的过敏、自述体质、既往史等资料会合并为一个画像确认弹窗。明确他人资料排除，身份省略时参考咨询对象，保存前由用户确认本人归属。可修改、暂不保存或更正冲突；“已确认画像”可查看来源、撤销保存。侧栏可手填过敏史，当前没有摘要管理面板。见[安全预审与画像确认](docs/安全预审与画像确认.md)及[接口约定](docs/api.md)。

## 开发与验证

后端建议 Python 3.12；前端 Docker 使用 Node.js 22、npm 和 `package-lock.json`。本地启动后端：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.lock.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```powershell
Set-Location frontend
npm ci
Copy-Item .env.local.example .env.local
npm run dev
```

访问 http://127.0.0.1:3000。Next.js 将 `/api/*` 转发至 `BACKEND_URL`，默认 http://localhost:8000，Docker 内为 http://backend:8000。

验证命令：

```powershell
.venv\Scripts\python.exe -m pytest -q
Set-Location frontend
npm test
npm run build
```

修改前后端或运行配置后，交付前必须重新执行 `docker compose up --build -d`，确认两项服务健康及 `/api/health`、`/chat` 可访问。测试通过不代表容器已更新。模型、安全与取消边界见[验收说明](docs/模型安全与取消验收.md)。

## 数据与回退

Docker 使用 `qiwen_data` 命名卷保存账号、业务库、检查点和上传文件。本地直接运行使用 `.data/`，两者独立。`docker compose down` 保留数据，不要用 `down -v` 清理业务卷。

此次从 `.replica-backup/20260926-before-source-replica` 恢复自有后端与运行配置。克隆版代码、文档和数据归档到 `.replica-backup/20260926-before-rollback`。两个版本的数据卷均保留，未强行合并不同数据库结构。范围见[回退记录](docs/回退记录.md)。

内置六部古籍、1537 个片段及知识种子的来源与复现见[古籍数据处理](docs/古籍数据处理与复现.md)。新文本导入共用[结构切片流程](docs/古籍结构切片流程.md)：分书结构 → 清理 → 约 500 字符细分 → MD5 去重 → 保存完整来源元数据；Chroma 入口使用岐问自身模块。可选混合检索和 MCP 见[后端扩展设计](docs/结构化剂量与候选协议.md)。

## 目录

```text
backend/app/           自有 API、鉴权、数据库、问答流程、模型与检索
backend/resources/     古籍原文与知识种子
backend/data_pipeline/ 清洗和派生数据处理
backend/scripts/       导入、备份与服务检查
backend/tests/         后端验收
frontend/src/app/      淡蓝页面与样式
frontend/src/lib/      API 和 SSE 适配
frontend/src/hooks/    聊天状态管理
docs/                 本项目的流程、接口、复现与回退说明
```

本项目用于学习与研究。Mock 模式展示流程，古籍出处只在实际检索到片段时提供；程序测试不能证明医学结论正确。
