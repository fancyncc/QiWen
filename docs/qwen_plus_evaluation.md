# 青云 qwen-plus 接入与无独立评审评测

## 实现

- 应用复用本地已有青云接口与密钥，启用 `openai_compatible`，模型为 `qwen-plus`。配置、Compose 和示例配置的默认模型名称一致；密钥不进入报告。
- `evaluation.benchmark` 默认不调用独立评审 API。真实依赖检查不要求 `QIWEN_JUDGE_*`；只有显式 `--judge` 才检查并使用独立服务。
- `quality`、`faithfulness` 默认运行实际应用，保存回答和实际展示的引用，检查回答非空、原文一致性和实际模型调用。安全追问等确定性边界答复单独标记，不计为真实模型生成。
- 独立模型评分为 `not_enabled`；专业评分为空、`scored=0`；全部无评分回答进入人工复核队列。工程执行通过不能替代专业语义评审。
- CLI 默认退出码反映工程执行门禁；指定 `--release-policy` 才按完整发布验收决定退出码。报告保留尚未完成的专业复核、阈值校准及发布条件。
- 知识回答使用本轮 `S1`、`S2` 等短编号，后端严格映射回召回资料 ID，未给定编号仍被拒绝。`trace.answer_citations` 记录实际展示出处，评测不会把未引用候选算成最终引用。
- HTTP 部署评测保留实际回答、生成调用记录、人工复核队列及 `generation_gate`；失败流保留失败节点和调用记录，便于区分模型失败与检索失败。
- `retrieval_regression --all-cases` 支持全部 48 条检索回归及负例。普通 pytest 固定 Mock；真实评测必须显式使用 CLI。

## 运行

```powershell
$env:PYTHONPATH='backend'
$env:PYTHONIOENCODING='utf-8'
.venv/Scripts/python.exe -m evaluation.safe_run .venv/Scripts/python.exe -m evaluation.benchmark --profile live_core --only quality,faithfulness --limit 2 --reports backend/evaluation/reports/current
.venv/Scripts/python.exe -m evaluation.safe_run .venv/Scripts/python.exe -m evaluation.deployment --dataset rag_regression.jsonl --concurrency 1 --reports backend/evaluation/reports/current
.venv/Scripts/python.exe -m evaluation.review <本次报告> --export-review <复核文件>
```

人工复核应判断实际回答是否得到引用支持、哪些结论超出支持范围、是否遗漏条件；填写复核人、理由以及 approved/rejected，不能以关键词检查代替专业判断。0.7 门槛继续保留，独立人工标注样本齐备后再校准。

## 验证记录

1. 青云实际连接探测成功；运行容器的模型模式、模型名、接口指纹和密钥配置存在性均已核验，全部角色模型匹配 qwen-plus。
2. 初次真实 HTTP 回归为 10/12；RAG-022 的 fixed/multi 在知识引用校验阶段失败。失败报告保留为 `deployment-20261012T100921089725Z.json`，随后增加短编号严格映射，并重新部署复测。
3. 完整 Mock 工程回归通过，检索证据代理检查 48/48：`benchmark-20261012T101202464019Z.json`。该结果不代表真实模型质量。
4. 首批真实回答采集为 6 条；其中 1 条为确定性身份追问。更新边界答复判分后机械检查 6/6，专业复核仍为 pending：`answer-rescore-20261012T100703604869Z.json`。这是对已有回答的规则重判，不是重新调用模型。

## 最终验证

- 完整后端单元测试 **359/359**；最后补充的评测契约与可选评审测试 **44/44**。未发生普通 pytest 真实生成调用。
- 最后一次 `docker compose up --build -d` 已完成；前后端 healthy；`/api/health`、`/chat` HTTP 200；`/api/readiness` 为 ready。9 个关键应用/评测文件 SHA256 与容器一致，`git diff --check` 通过。
- 最终真实 HTTP 回归 **12/12**：`deployment-20261012T102913267411Z.json`。请求、持久化、检索证据、完整检索组件代理检查均通过；`generation_gate` 确认全部 12 条实际成功调用生成模型。RAG-022 的 fixed/multi 在短编号映射后均成功。
- 对最终回答展示的出处进行独立机械核验，使用当前部署语料检查 ID、书名、章节、原文及正文展示，**12/12**：`deployed-citation-checks-qwen-plus-20261012.json`。该核验不判断结论的专业语义支持程度。
- 最终无评审 API 的真实回答采集 **6/6**：`benchmark-20261012T102629596047Z.json`；模型评审 `not_enabled`。其中 1 条是确定性身份追问，仍待人工判断它是否符合提问。
- 当前人工回答复核任务：`answer-review-live-qwen-plus-20261012.json` 共 6 条，`deployment-answer-review-qwen-plus-20261012.json` 共 12 条，均 pending。这些是回答复核，另有此前 253 条检索候选复核，不能相互替代。

## 完整检索回归暴露的后续项

全部 48 条真实向量 + CrossEncoder 检索代理检查为 **39/48**：`full-live-rag-20261012T102305935367Z.json`。RAG-003、006、022、035 和明确指定《脉经》的 022 变体均通过。

其余 9 条已单独导出到 `real-retrieval-followups-20261012.json`，不修改标签来掩盖失败：

| 用例 | 观察到的结果 | 后续工作 |
| --- | --- | --- |
| RAG-001、008、012、018、019、025 | 最高重排分数 0.22–0.62，未达 0.7，最终无证据 | 复核正确候选、重排窗口和问题表达；用独立标注校准分数 |
| RAG-024 | 检索超时，重排未完成 | 单独复测超时；当前验证存在并行负载，不能直接归因为固定容量不足 |
| RAG-017 | 浮脉问题引用窗口实际包含洪脉/弱脉 | 加强脉象实体与正文窗口约束，避免仅依赖章节和重排分数 |
| RAG-044 | 原文包含增液立法，但代理标签仅匹配“增液汤” | 人工确认支持范围并修正标签范围，再复测 |

0.7 未因本次失败而降低。真实模型和默认评测已可正常运行；剩余检索问题、独立专业复核、门槛校准、发布策略与完整负载验收仍需完成。当前结果不能给出完整专业质量或发布通过结论。
