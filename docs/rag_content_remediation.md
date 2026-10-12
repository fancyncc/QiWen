# 检索内容约束整改（2026-10-12）

本轮针对 RAG-003、006、022、035 及其测评、语料与标注问题实施代码整改。工程结果、开发代理标签、真实检索组件、实际 API 和专业验收分别记录。独立复核缺失时不生成专业通过结论。

## 代码与数据

| 范围 | 修改 |
| --- | --- |
| 查询 | `query_plan.py` 将《书名》作为来源约束，移出内容词；保留情志枚举的有效单字及喘、渴、咳等领域单字；整体保留否定发汗表达。|
| 异写 | `retrieval_terms.py` 统一黄耆/黄芪检索写法；仅替换完整药名，原文与个人事实不改写。|
| 候选 | 保留各通道和原始 RRF 名次，额外记录内容要求与候选池名次；明确内容要求参与候选池截断前排序，不从测试标签导入书名、章节或目标 ID。|
| 窗口和准入 | 关系问题覆盖明确的关系对象；禁忌窗口必须保留否定动作；药物问题要求目标药名；问方剂与六经提纲的问题优先选包含所问属性的原文。CrossEncoder 分数达到 0.7 后仍检查这些内容要求。无重排模式使用同样要求，并导出准入原因。|
| 语料 | 保存六个原始片段文件于 `resources/classics_segments_raw`，原始文本继续保留于 `classics_source`；清洗页脚、导航和页码，重新生成 1515 个片段。清洗元数据记录原始内容哈希与删除范围。|
| 迁移 | `classics_legacy_manifest.json` 绑定原始及中间清洗版本的条目 ID、书名、章节与正文哈希。启动时仅移除精确匹配且已被新语料替代的条目，保留用户导入及修改的章节。|
| 重建 | 捆绑 JSON 可从保留的原始文本重建；症状扩展词的出处 ID 更新到新语料，加载时仍核验引文。|
| 向量 | 缺失条目集中组批，已有向量只更新来源元数据。`--prune-stale` 在补建成功后清除当前库中不存在的旧向量 ID。|
| 标签 | RAG-022 原问题改为开放出处，另增指定《脉经》的对照。代理判断使用实际引用窗口；情志问题检查最终证据集合是否覆盖四种情志。标签仍为待专业复核。|
| 专业支持范围 | 支持 `passage_labels`，将标签绑定文档、正文和具体引文哈希；允许标注 `supported_claims`，与案例 `required_claims` 比较。窗口或正文改变后旧标签不能替新窗口判分。|
| 独立回答评分 | 忠实度 judge 必须逐项返回结论、实际证据 ID、支持等级和遗漏条件；拒绝虚构证据 ID、无证据支持及与总判定矛盾的结构化结果。|
| 审查任务 | `evidence_audit --deduplicate` 合并完全一致的问题、出处、窗口、模型配置和分数，保留原运行映射。不同问题、窗口或分数不合并；未评分候选单独复核召回，不用于校准。|

这些内容检查属于检索约束，不等于专业语义判断；同一窗口含有相关词仍可能无法充分支持结论。近似重复只导出复核清单，不自动删除不同底本。

## 验证入口

```powershell
$env:PYTHONPATH = "backend"
$env:PYTHONIOENCODING = "utf-8"
.venv/Scripts/python.exe -m evaluation.safe_run .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
.venv/Scripts/python.exe -m evaluation.safe_run .venv/Scripts/python.exe -m scripts.rebuild_classics
.venv/Scripts/python.exe -m evaluation.safe_run .venv/Scripts/python.exe -m evaluation.benchmark --profile mock_contract --reports backend/evaluation/reports/current
```

实际部署更新与向量同步：

```powershell
.venv/Scripts/python.exe -m evaluation.safe_run docker compose up --build -d
.venv/Scripts/python.exe -m evaluation.safe_run docker compose exec -T backend python -m evaluation.safe_run python -m scripts.index_vectors --batch-size 8 --prune-stale
docker compose ps
```

检查 `/api/health`、`/chat` HTTP 200，以及 `/api/readiness` 对当前语料的向量覆盖。部署成功不能替代检索效果通过。

组件专项直接使用已部署后端的真实向量、CrossEncoder、门槛及适用时的 MMR，不依赖生成模型：

```powershell
.venv/Scripts/python.exe -m evaluation.safe_run docker compose exec -T backend python -m evaluation.safe_run python -m evaluation.retrieval_regression --mode live --reports /app/.data/evaluation-content
.venv/Scripts/python.exe -m evaluation.safe_run .venv/Scripts/python.exe -m evaluation.deployment --dataset rag_regression.jsonl --concurrency 1 --reports backend/evaluation/reports/current
```

组件与部署专项都包含原四项、指定《脉经》以及黄芪规范写法。部署专项检查实际最终引用和完整组件执行，仍使用开发代理规则；固定与 multi 分别记录，不扩大解释为并发容量或真实回答质量。

## 专业标注与剩余验收

从最新报告导出新的源库绑定审查任务：

```powershell
python -m evaluation.evidence_audit --report <最新部署或组件报告.json> --with-current-sources --deduplicate --output <新证据审计.json>
python -m evaluation.calibration --audit <新证据审计.json> --labels <新证据审计.labels.jsonl> --output <新校准报告.json>
```

冻结源库的导出需要在同版本后端容器执行。历史 486 条任务属于旧语料/窗口版本，不能直接作为新评分的校准样本。新任务初始为 pending；由独立专业复核者填写真实身份、日期、判断理由、支持范围及结论/条件。未评分任务提供完整公开原文，明确区别于实际评分窗口。

案例的 `passage_labels` 按文档 ID 保存标签列表。每项需有 `body_sha256`、`quote_sha256`、0–3 的 `relevance`、布尔 `supports_question`、`supported_scope`、`reviewer`、`reviewed_at`、`rationale` 和 `label_status=expert_reviewed`；可增加 `supported_claims`。最终引用缺少匹配标签时标记 label_unavailable，不能因同文档 qrels 为正就判窗口通过。

0.7 运行门槛保持待独立校准状态。真实应用模型、独立 judge、专业复核、冻结集及预先声明的发布策略尚须完成；本轮工程代码和代理规则结果不能解除这些发布阻塞。

## 验证结果

- 完整后端回归 **349/349** 通过（394.46 秒）；其后针对最后的准入审计复用与测评预热调整，相关回归 133 项、79 项及新增预热阻塞契约所在的专项文件 21 项分别通过。这些为不同运行，不相加为完整回归数量。
- [完整离线测评](../backend/evaluation/reports/current/benchmark-20261012T083647407354Z.json)的工程门禁与检索代理门禁均通过，RAG **48/48**。该运行使用 Mock、BM25 且关闭 CrossEncoder；不能代替专业语义验收。
- [实际 API 专项](../backend/evaluation/reports/current/deployment-20261012T084251366786Z.json)：并发 1，fixed/multi **12/12** 请求、证据有无和最终引文代理检查通过；真实混合召回、CrossEncoder、0.7 门槛及适用时的 MMR 均完成。检索耗时 6.69–20.13 秒，没有候选 tokenizer 截断。应用生成模型仍为 Mock，不宣称真实回答质量通过。
- [实际检索组件专项](../backend/evaluation/reports/current/retrieval-regression-20261012T084928144324Z.json)：独立预热缓存权重后 **6/6** 通过，含原四项、指定《脉经》和黄芪写法对照。
- [首次组件运行](../backend/evaluation/reports/current/retrieval-regression-20261012T083828990899Z.json)为 **5/6**，RAG-003 冷启动超时：召回 8.26 秒、CrossEncoder 加载 12.10 秒，推理未在剩余预算完成。新组件入口把缓存预热作为独立准备阶段，仍使用原有 30 秒检索预算；保留初次失败，不把冷启动成绩改写为通过。
- [语料审计](../backend/evaluation/reports/current/corpus-audit-20261012-canonical.json)记录 923 个原始片段的清洗变化和当前语料 3 组近似重复待复核。`scripts.rebuild_classics` 验证 1515 个片段与原始文本重建一致。
- 向量断点补建本批 624 条，保留已有 891 条，随后清除 1265 个失效向量 ID；当前语料覆盖 **1515/1515**。
- [新证据审计](../backend/evaluation/reports/current/evidence-audit-content-20261012.json)冻结 1515 个古籍片段和 1 个概念资料。本次原始任务 **594**，合并 **341** 个一致任务，保留 **253** 个独立审查单元及原运行映射；[253 条标签](../backend/evaluation/reports/current/evidence-audit-content-20261012.labels.jsonl)全部 pending。
- [新校准报告](../backend/evaluation/reports/current/calibration-content-20261012.json)为 blocked：独立复核未完成且没有已复核评分样本。运行门槛仍为 0.7，没有自动降低。
- [真实质量依赖检查](../backend/evaluation/reports/current/benchmark-20261012T083232581967Z.json)为 blocked，缺少 `QIWEN_MODEL_NAME`、`QIWEN_JUDGE_BASE_URL`、`QIWEN_JUDGE_API_KEY`、`QIWEN_JUDGE_MODEL`。这些是变量名，不包含凭据值；没有运行真实回答质量评分。

| 用例 | 实际 API 目标结果 | 最高重排分数 |
| --- | --- | ---: |
| RAG-003 | fixed/multi 均引用《素问·生气通天论》关系条文 | 0.9123 |
| RAG-006 | fixed/multi 均包含《举痛论》，所选原文合计覆盖怒、喜、悲、恐 | 0.9485 |
| RAG-022 | 返回含否定发汗与适用条件的原文；开放出处与指定《脉经》分别检查 | 0.9949 / 0.9984 |
| RAG-035 | 黄耆与黄芪查询均引用同一黄芪条目 | 0.9976 |

分数仅为检索模型输出，不是诊断置信度。最终证据集合的每条出处及支持范围仍需独立复核；4 条目标证据命中不等于全部其他引用都通过了专业判断。完整发布验收继续 blocked。

最终 Docker 重建已完成；前后端 healthy、`/api/health` 与 `/chat` 均 HTTP 200。应用代码、检索组件与语料关键文件共 12 个核对容器 SHA256，全部一致。[最终重启后的 RAG-003 双模式冒烟](../backend/evaluation/reports/current/deployment-20261012T085501097542Z.json)为 2/2，通过实际引文及完整组件检查，服务 `/api/readiness` 为 ready，向量覆盖仍为 1515/1515。该两项冒烟的容量门禁 blocked，不能用于 SLA 验收。
