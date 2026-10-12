# 岐问

岐问是自行实现的中医知识问答与会诊流程演示，后端位于 `backend/app/`，前端提供淡蓝色登录、聊天、侧栏与画像确认页面。项目用于学习和研究，Mock 展示流程，程序测试不能证明医学结论或个体用药安全。

## 启动与使用

项目根目录执行：

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:23000/api/health
(Invoke-WebRequest http://127.0.0.1:23000/chat -UseBasicParsing).StatusCode
```

确认前后端 healthy、健康接口正常且聊天页返回 200，打开 [登录页](http://127.0.0.1:23000/login)。首次使用先注册，没有通用管理员密码。

登录 → 创建或选择会话 → 编辑体质和过敏史 → 输入问题或附加文件 → 查看处理阶段与最终答复。支持带文本 PDF、DOCX、UTF-8 TXT，单文件最多 10 MB、单轮最多 5 个附件，不支持扫描 PDF OCR。会话可重命名、删除与重新打开，停止按钮会请求服务端取消。

同一会话保留问诊事实、病程和待答问题，长间隔续问按需确认是否延续。跨会话仅参考本人手填或明确确认的用户画像，其他会话的症状、附件和自动摘要不进入回答。长期资料的查看与撤销沿用画像管理。

聊天默认 `mock`，新会话默认 `fixed`；`multi` 使用由 LangGraph 子图编排的辨证、方剂与本草 Agent，外层程序控制安全与最终提交。页面只展示 `answer.final` 确认的正文，未通过审查的候选不进入答复气泡。对话中提取的长期资料需用户在合并弹窗确认后保存，可查看来源与撤销。

古籍默认请求 `hybrid`，实际链路为 BM25/可选向量召回 → RRF → 本地 CrossEncoder → sigmoid ≥0.7 → MMR。首次部署显式准备本地模型：

```powershell
docker compose exec backend python -m scripts.prepare_ranking
```

真实聊天从 [.env.example](.env.example) 复制到 `.env` 后配置服务端接口、凭据和模型。向量、记忆重排与完整真实链路各有独立配置和验收边界，见 [部署说明](docs/deployment.md)。不要输出或提交密钥，健康接口的配置模式不代表所有检索通道已成功执行。

## 文档导航

`docs/` 按主题维护以下 8 个文档：

| 文档 | 内容 |
| --- | --- |
| [agent.md](docs/agent.md) | fixed / multi 流程、安全事实、症状确认、画像、候选、剂量与取消 |
| [api.md](docs/api.md) | API、字段约束、SSE 与轨迹契约 |
| [architecture.md](docs/architecture.md) | 系统架构、模块职责、存储与事务边界 |
| [data_pipeline.md](docs/data_pipeline.md) | 古籍来源、结构切片、清理、去重、导入与复现 |
| [deployment.md](docs/deployment.md) | Docker、本地开发、模型配置、数据快照与可选 MCP |
| [evaluation_report.md](docs/evaluation_report.md) | 六维测评结果、适用范围与历史工程验收归档 |
| [rag.md](docs/rag.md) | 古籍混合检索、重排、MMR、术语扩展与会话记忆 |
| [evaluation_plan.md](docs/evaluation_plan.md) | 测评完善计划、会话与画像边界、实施范围 |
| [testing.md](docs/testing.md) | 工程回归、测评数据、复现命令与部署验收 |

分散的中文计划、流程与验收文档已合并到上述主题中。测评原始 JSON 与自动生成摘要仍位于 `backend/evaluation/reports/`；`evaluate_exports/` 保留独立导出结果，不作为主题文档重复维护。

## 开发与数据

Docker 使用 Python 3.12 和 Node.js 22，前端使用 npm。后端本地入口为 `uvicorn app.main:app --app-dir backend`，前端为 `frontend/` 下的 `npm run dev`，详细启动步骤见 [deployment.md](docs/deployment.md)。验证入口：

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
Set-Location frontend
npm test
npm run build
```

修改前端、后端或运行配置后，交付前须执行 `docker compose up --build -d` 并检查服务；测试通过不代表容器已更新。

Docker 使用 `qiwen_data` 命名卷，本地直接运行使用 `.data/`，两者独立。`docker compose down` 保留数据，`down -v` 会删除业务卷。数据恢复前需核对快照、版本与实际路径，历史回退或部署记录不能作为当前服务状态。内置六部古籍基线为 1537 个唯一片段，资料来源见 [数据管线](docs/data_pipeline.md) 与 [资源说明](backend/resources/README.md)。

```text
backend/app/           API、鉴权、问答流程、模型、检索与安全
backend/resources/     古籍原文、派生片段与知识种子
backend/data_pipeline/ 清理与结构切片
backend/scripts/       导入、索引、快照与服务检查
backend/tests/         后端回归
backend/evaluation/    测评数据、评测器与原始报告
frontend/src/          登录、聊天、画像交互与 API/SSE 适配
docs/                  8 个主题文档
```
