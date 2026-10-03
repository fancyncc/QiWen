# 古籍资料来源

`classics_source/` 的六个 TXT 文件取自 [gstaryu/LingYi](https://github.com/gstaryu/LingYi) 的 `TCM_data/`，`classics_segments/` 的六个 JSON 文件来自 `storage/chunks/`，下载于 2026-09-24。TXT 在本项目内按上游文件原样保存。上游对《伤寒论》《金匮要略》《温病条辨》《神农本草经》《脉经》《黄帝内经·素问》清洗、分章和切分生成 JSON。本项目只读取 `content` 与 `metadata`，用书名和正文的 SHA-256 指纹幂等入库；不会覆盖用户自行导入的资料。

上游仓库标注 Apache-2.0，许可文本见 [Apache-2.0-LICENSE.txt](Apache-2.0-LICENSE.txt)。上游文本文件内保留“中国哲学书电子化计划 / Chinese Text Project”来源标记；这些数字底本的再分发条件应与上游及原始来源分别核对。项目保留书名和章节元数据，回答只在实际检索到片段时展示出处。

本资料用于古籍知识检索演示，不代表经临床核验的诊疗依据。

`knowledge_seed.json` 则是从上游 `data_pipeline/seed_knowledge.py` 中手工整理的种子对象转换得到，含 33 味本草、22 首方剂和 14 条药材禁忌说明。该转换文件保留上游字段及来源描述；未复制上游运行数据库、2015 版药典扩充数据或四部方剂提取产物。

清洗与切分代码保存在 `backend/data_pipeline/`。当前增加了岐问的统一导入与 Chroma 适配，业务库保留完整 metadata 和原片段 source_chunk_id，已有正文与片段 JSON 不变。完整处理步骤、复现命令、当前入口及升级行为见 [data_pipeline.md](../../docs/data_pipeline.md)。

`symptom_terms.json` 是岐问自行维护的症状与别名词表，包含 54 项症状和有上述古籍原文证据的检索扩展项。`relation=retrieval_related` 表示检索关联，不表示医学等价；每项记录片段 ID、书名、章节和原文摘录。维护与验收见[rag.md](../../docs/rag.md)。
