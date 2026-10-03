# 岐问数据管线

古籍原文、知识种子与术语表用于学习和检索演示。本文说明来源、结构切片、入库与复现；检索排序见 [rag.md](rag.md)，运行配置和备份见 [deployment.md](deployment.md)。

## 资料来源与种子

`backend/resources/classics_source/` 保存六部古籍 TXT，`classics_segments/` 保存派生 JSON，原始资料于 2026-09-24 从 gstaryu/LingYi 的 `TCM_data/` 与 `storage/chunks/` 获取。资料来源和许可说明见 [resources/README.md](../backend/resources/README.md) 与 [Apache-2.0 许可](../backend/resources/Apache-2.0-LICENSE.txt)。TXT 保留“中国哲学书电子化计划 / Chinese Text Project”标记，数字底本再分发条件需与上游及原始来源分别核对。

`knowledge_seed.json` 转换自上游手工种子，含 33 味本草、22 首方剂和 14 条禁忌说明；没有复制上游运行数据库、2015 版药典扩充数据或其他方剂提取产物。剂量默认未核验，结构及审查边界见 [agent.md](agent.md)。

`symptom_terms.json` 为项目自有版本化词表，记录现代症状、别名及带片段 ID、书名、章节、原文摘录的扩展词。`relation=retrieval_related` 仅表示检索关联，不表示医学等价，不回写患者事实。

## 数据流

古籍 TXT → 分书结构切分 → 清理/过滤 → 过长片段按标点细分 → 内容 MD5 去重 → SQLite 保存正文与来源 → 配置嵌入接口后写入 Chroma。

服务启动读取已经生成的片段 JSON；每轮提问直接检索已入库片段，不重新切整本书。

| 古籍 | 第一层切分 | 保留字段 |
| --- | --- | --- |
| 伤寒论、金匮要略 | `###` 章节，然后双换行段落 | book、chapter |
| 黄帝内经-素问 | `###` 章节，然后段落/对话 | book、chapter |
| 温病条辨 | 中文数字编号条文，保留后续注释 | book、clause_number |
| 神农本草经 | 部类，然后药名＋全角空格＋味的条目起点 | book、category、drug_name |
| 脉经 | 章节标题；超过 2000 字符尝试按行首一曰细分 | book、chapter、sub_index（存在时） |

未知书名的普通 UTF-8 文本按 Markdown/中文章节标题及空行段落切分，再使用同样的清理和长度处理。已知古籍未识别到任何结构时明确报错，提示调整切分器。

## 统一清理和细分

去导航、版权提示、验证码等预设杂质，过滤英文主导的行，合并多余空行；丢弃少于 10 字符、中文占比低于 30% 或符合目录规则的片段。这些是固定规则，仍需抽查正文误过滤风险。

超过 500 字符先按句号拆分；长句再利用分号、逗号、顿号细分；短句顺序组合成接近上限的子片段。子片段继承元数据，增加 split_from，ID 加序号。没有相邻窗口重叠，没有自动语义判断，也不强制把无标点超长句硬切到 500。

500 是 Python 字符数。CrossEncoder 的 512-token 输入还包含查询与特殊标记；本次没有改成 token 预算或父子上下文自动补充。split_from 仅记录原切片身份。

## 去重、来源和升级

分书流程在六书全库按正文 MD5 去重。新增文本入库也检查当前 corpus 的正文 MD5，完全相同的内容只保留首个来源。MD5 用于判重，不是安全签名；不同版本、不同排版仍可能保留为不同内容。

业务主键继续使用书名与正文 SHA-256，现有古籍 ID 和正文保持稳定，术语扩展出处不会因本次适配失效。corpus.metadata 为完整 JSON，保留书名、章节/类别、药名/条文号、split_from、source_chunk_id 等存在的字段。检索轨迹 selected.source_metadata 随结果返回；Chroma 以 chunk_metadata 字符串保存同一 JSON。

旧数据库自动增加 metadata 字段，内置片段按已有 ID 补齐缺失元数据。用户原有导入片段保持原样；之后的新增导入使用结构切分。已有重复行不在本次升级中删除，旧导入也不会在缺少原始资料时自动重切。

## 操作入口

以下命令在 backend 目录、项目虚拟环境中执行：

~~~powershell
# 只读复现核对，预期 1537 个片段
..\.venv\Scripts\python.exe -m scripts.rebuild_classics

# 新导入；不指定 --book 时按标题或文件名识别六部古籍
..\.venv\Scripts\python.exe -m scripts.ingest "资料路径.txt" --title "测试版伤寒论" --book "伤寒论"

# 同一切片流程输出 JSON（默认 backend/storage/chunks）
..\.venv\Scripts\python.exe -m data_pipeline.ingest --mode json

# 同一切片流程生成 mock 检索样例（不是向量入库）
..\.venv\Scripts\python.exe -m data_pipeline.ingest --mode mock

# 配置嵌入接口后，切片并接入岐问 SQLite/Chroma
..\.venv\Scripts\python.exe -m data_pipeline.ingest --mode chroma
~~~

## 复现与验证

内置基线为六部书、1537 个唯一片段。`scripts.rebuild_classics` 默认只读比较生成结果与内置 JSON；只有主动重生成资料时才使用 `--write` 并审阅 diff。服务启动幂等读取片段 JSON，问答不会重新切整本书，也不会覆盖用户自行导入的资料。

Chroma 入库需兼容接口基址、服务端凭据与 `QIWEN_EMBEDDING_MODEL`。向量集合按接口、模型和距离版本隔离；逐批跳过已有 ID 并更新来源元数据。更换嵌入模型后重新建立对应索引，见 [deployment.md](deployment.md)。

工程检查覆盖分书结构边界、正文清理、长度、split_from、全库去重、迁移、检索来源元数据、索引续建与元数据刷新。真实嵌入未配置时，只能验证协议和模拟结果。历史复现与部署记录统一见 [evaluation_report.md](evaluation_report.md)，不能沿用旧文档中的中断状态判断当前服务。
