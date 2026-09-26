# 岐问

岐问是自行实现的中医知识问答与会诊流程演示。参考项目用于理解功能和流程，后端使用 `backend/app/` 中的自有实现，不再整包复制上游代码。当前保留淡蓝色登录页、聊天页、侧栏和参考页面布局。更新时间：2026-09-26。

## 启动与登录

在项目根目录执行：

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:23000/api/health
```

打开[登录页](http://127.0.0.1:23000/login)。首次使用先注册：用户名为 3 至 32 位字母、数字或下划线，密码为 8 至 128 位。克隆前已有账号继续使用原密码。回退后登录凭据不通用，需要重新登录；密码错误会在表单中提示。项目没有通用管理员密码。

默认模型和检索模式为 `mock`，无需密钥即可演示问诊。真实模型配置从 [.env.example](.env.example) 复制到 `.env`，设置 `QIWEN_MODEL_MODE=openai_compatible`、`QIWEN_MODEL_BASE_URL`、`QIWEN_MODEL_API_KEY`、`QIWEN_MODEL_NAME` 后重新构建。配置缺失或调用失败时返回错误，不将演示结果当成真实模型答复。浏览器公开变量不能存放模型密钥。

## 使用流程

登录 → 创建或选择会话 → 编辑体质和过敏史 → 输入问题或附加文件 → 发送 → 查看阶段与最终答复。上传支持带文本的 PDF、DOCX、UTF-8 TXT，单文件不超过 10 MB，单轮最多 5 个附件。扫描 PDF 不支持 OCR。历史会话可以重命名、删除和再次打开。

后端继续使用克隆前的问诊、记忆、检索、辨证、方剂、安全审查、上下文压缩与提交流程。新会话默认固定流程；原有多专家会话按其保存的模式运行。后端支持 `fixed`、`multi`，当前保留的页面没有新增模式选择器。具体节点与重试上限见[问答流程](docs/问答流程.md)。

页面显示阶段进度，收到 `answer.final` 后展示最终正文。候选和审查仍由后端执行、独立保存，未通过审查的候选不进入答复气泡。停止按钮中止读取并请求服务端取消，后端也处理断线取消。

当前画像弹窗可编辑体质和过敏史。后端还保存既往史、会话摘要、结构化方剂审查结果，保留的页面未提供这些字段的完整管理界面。不能把后端接口能力当成页面已提供的功能。字段见[接口约定](docs/api.md)。

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

内置六部古籍、1537 个片段及知识种子的来源与复现见[古籍数据处理](docs/古籍数据处理与复现.md)。可选混合检索和 MCP 见[后端扩展设计](docs/结构化剂量与候选协议.md)。

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
