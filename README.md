# 岐问

从零实现的中医知识辅助问答与多智能体会诊演示项目。默认使用离线 Mock 模式，无需模型密钥或外部网络即可运行。回答仅供学习与研究参考，不能代替执业医师面诊，也不应作为自行抓药的依据。

## 已实现

- FastAPI、SQLite 业务库、JWT 登录、密码 PBKDF2 哈希、用户与文件归属校验。
- 会话创建、重命名、删除和历史恢复；患者画像跨会话读取与编辑。
- PDF、DOCX、UTF-8 TXT 上传、解析和 10 MB 限制。
- LangGraph 固定流程与三专家并行会诊，SQLite Checkpointer 续接追问状态。
- 字符级 BM25 检索；配置向量模型与 ChromaDB 后使用向量结果和 BM25 的 RRF 融合。文本导入按内容指纹去重，未找到片段时明确说明没有古籍出处。
- OpenAI 兼容模型适配层；多专家会诊的审查节点、最多两次修正和确定性配伍禁忌检查。无法解析、命中禁忌或过敏信息时隐藏候选药材清单。
- REST 与 SSE 中文分块、阶段事件；Next.js 登录与注册页、会话路由、响应式聊天界面、会诊轨迹、方剂卡片、模式选择、上传、停止生成与画像弹窗。

项目已保存六部古籍 TXT、1537 个可复现片段，以及 33 味本草、22 首方剂和 14 条禁忌种子。启动时会幂等入库，来源与清洗步骤见 [古籍数据处理与复现](docs/古籍数据处理与复现.md)。上游药典 XLSX 和四部方剂提取产物未包含在本项目中。本项目只在实际检索到古籍片段时展示出处；Mock 模式的辨证和候选方剂是流程演示，不是个体化诊疗结论。

配伍规则种子依据[深圳市中药饮片处方审核规范附录 C](https://wjw.sz.gov.cn/attachment/0/758/758943/8708423.pdf)整理，只作保守风险提示，不能替代药师处方审核。

设计文档列为后续扩展的 MCP、WebSocket、外部药典批量扩充、CrossEncoder 重排和自动评测尚未接入。当前可运行的主流程为 REST/SSE、基础知识种子、BM25 与可选的 Chroma 混合检索。

## 本地运行

### Docker Compose

安装 Docker 后，在项目根目录运行：

```powershell
docker compose up --build -d
```

打开 [http://127.0.0.1:23000](http://127.0.0.1:23000)，API 健康检查位于 [http://127.0.0.1:23000/api/health](http://127.0.0.1:23000/api/health)，API 文档位于 [http://127.0.0.1:23000/docs](http://127.0.0.1:23000/docs)。默认使用离线 Mock 模式，无需配置密钥。前端通过同源路径转发 API；后端只在 Docker 网络内开放 8000 端口。Docker 默认只绑定宿主机 23000 端口。

如需修改端口、模型或检索配置，先执行 `Copy-Item .env.example .env`，再编辑 `.env` 并重建：`docker compose up --build -d`。`QIWEN_WEB_PORT` 控制宿主机端口，默认只绑定到本机。启用混合检索时设置 `QIWEN_INSTALL_KNOWLEDGE=true`、`QIWEN_RAG_MODE=hybrid` 及模型接口配置。容器数据保存在 Docker 命名卷 `qiwen_data`，`docker compose down` 不会删除它。

查看状态与停止服务：

```powershell
docker compose ps
docker compose down
```

### 直接运行

建议使用 Python 3.12、Node.js 20.9+ 和 pnpm。以下命令在项目根目录的 PowerShell 中执行。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.lock.txt
Copy-Item .env.example .env
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

另开一个终端：

```powershell
Set-Location frontend
pnpm install
Copy-Item .env.local.example .env.local
pnpm dev
```

打开 [http://127.0.0.1:3000](http://127.0.0.1:3000)。API 文档位于 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。

前端会根据登录状态从 `/` 跳转至 `/login` 或 `/chat`；历史会话地址为 `/chat/{threadId}`。桌面端侧栏与手机端导航抽屉均可切换会话、编辑画像和退出登录。

## 导入古籍与接入模型

内置六部古籍无需手动导入。若要从保存的 TXT 和清洗代码重新核对派生数据，在项目根目录执行 `Set-Location backend`，然后运行 `..\.venv\Scripts\python.exe -m scripts.rebuild_classics`；写入新派生 JSON 时显式添加 `--write`。详见[复现文档](docs/古籍数据处理与复现.md)。

将有使用权的 UTF-8 `.txt` 文件放在自己选择的路径，然后执行：

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m scripts.ingest ..\corpus\example.txt --title 书名
```

重复导入不会重复写入相同片段。默认检索不需网络。若启用向量检索，先安装 `backend/requirements-knowledge.txt`，设置 `.env` 中的 `QIWEN_RAG_MODE=hybrid`、模型兼容接口地址、密钥和 `QIWEN_EMBEDDING_MODEL`，再导入文本。向量服务不可用时，问答会明确标注退化为 BM25。

若启用真实模型，将 `QIWEN_MODEL_MODE` 设为 `openai_compatible`，并配置 `QIWEN_MODEL_BASE_URL`、`QIWEN_MODEL_API_KEY`、`QIWEN_MODEL_NAME`。接口地址填写兼容 API 的 `/v1` 基址即可。缺少配置或模型请求失败时返回错误，不会把 Mock 内容冒充真实结论。

## 验证

```powershell
.venv\Scripts\python.exe -m pytest -q
Set-Location frontend
pnpm build
```

测试覆盖双用户隔离、画像权限、文件引用权限、追问续接、多专家 SSE、上传限制、语料去重和禁忌规则。

## 数据与部署

开发数据保存在 `.data/`：业务库、LangGraph 检查点、上传文件、开发密钥和可选 Chroma 索引。删除会话会删除其消息与检查点；上传文件保留在账户下，部署时应设置独立的数据保留和备份计划。生产环境必须设置 `QIWEN_ENV=production`、强随机 `QIWEN_SECRET_KEY`、受限的 `QIWEN_CORS_ORIGINS`，并通过 HTTPS 反向代理提供服务。日志不记录完整病历或模型密钥。

## 项目结构

```text
backend/app/      API、鉴权、数据库、LangGraph、检索与安全规则
backend/data_pipeline/  古籍清洗、分书切分与知识种子代码
backend/resources/      六部古籍原文、片段、结构化种子与许可文本
backend/scripts/  古籍文本导入与派生数据复现
backend/tests/    离线验收测试
frontend/app/     Next.js 页面与样式
docs/             数据处理与复现记录
```
