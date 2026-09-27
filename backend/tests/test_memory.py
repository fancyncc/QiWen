from __future__ import annotations

import asyncio

import pytest

from app import db, memory, retrieval, workflow


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "memory.sqlite3")
    monkeypatch.setenv("QIWEN_MEMORY_MODE", "bm25")
    db.init_db()
    with db.connection() as conn:
        for user in ("alice", "bob"):
            conn.execute("INSERT INTO users VALUES(?,?,?,?)", (user,user,"unused",db.now()))
            conn.execute("INSERT INTO profiles(user_id,updated_at) VALUES(?,?)", (user,db.now()))
        for thread, user, summary in (
            ("current", "alice", "头痛"),
            ("head", "alice", "上个月头痛"),
            ("sleep", "alice", "之前没有失眠"),
            ("other-user", "bob", "头痛失眠"),
        ):
            conn.execute("INSERT INTO threads VALUES(?,?,?,'fixed',?,?)", (thread,user,thread,db.now(),db.now()))
            conn.execute("INSERT INTO profile_memories VALUES(?,?,?,?)", (thread,user,summary,"2026-09-01T00:00:00+00:00"))
    return db


def test_followup_query_uses_pending_complaint_not_question_suggestions():
    query = memory.build_query({"message":"三天，未有其它不适", "pending_context":"我头痛",
                                "pending_question":"是否伴有失眠？"}, [])
    assert "头痛" in query and "三天" in query and "失眠" not in query
    assert "头痛" in memory.build_query({"message":"三天"}, ["我头痛"])
    assert "头痛" not in memory.build_query({"message":"最近睡不好"}, ["我头痛"])


@pytest.mark.parametrize("query,expected", [
    ("头痛", ["head"]), ("最近睡不好", ["sleep"]), ("入睡困难", ["sleep"]),
    ("腰痛", []), ("三天", []), ("最近不适", []),
])
def test_keyword_labeled_cases_reject_unrelated_memories(memory_db, query, expected):
    rows, info = memory.retrieve_memories(query, "alice", "current")
    assert [row["thread_id"] for row in rows] == expected
    assert info["active_mode"] == "bm25"


def test_workflow_preserves_memory_identity_and_recent_original_text(memory_db):
    with db.connection() as conn:
        conn.execute("INSERT INTO messages(id,thread_id,role,content,created_at) VALUES('a','current','user','我头痛','2026-09-25')")
        conn.execute("INSERT INTO messages(id,thread_id,role,content,created_at) VALUES('b','current','user','三天，未有其它不适','2026-09-26')")
    loaded = asyncio.run(workflow.memory_load({"user_id":"alice", "thread_id":"current",
                         "message":"三天，未有其它不适", "pending_context":"我头痛"}))
    assert "来源会话=head" in loaded["memory_context"]
    assert "摘要更新时间=2026-09-01" in loaded["memory_context"]
    assert "上个月头痛" in loaded["memory_context"]
    assert "other-user" not in loaded["memory_context"]
    assert "已确认症状" in loaded["memory_context"]
    assert loaded["memory_context"].count("三天") == 0  # Current text is already a separate state field.
    assert loaded["memory_retrieval"]["selected"][0]["thread_id"] == "head"


def test_format_preserves_negation_and_drops_whole_over_budget_records():
    rows = [{"thread_id":"past", "updated_at":"2026-09-01", "summary":"之前没有失眠"}]
    context = memory.format_context(rows, [], "current")
    assert "之前没有失眠" in context
    assert "来源会话=past" in context
    assert "摘要更新时间=2026-09-01" in context
    assert "来源会话" not in memory.format_context(rows, [], "current", budget=70)


def test_hybrid_fallback_is_explicit_and_does_not_admit_unrelated_records(memory_db, monkeypatch):
    monkeypatch.setenv("QIWEN_MEMORY_MODE", "hybrid")
    monkeypatch.setattr(retrieval, "API_KEY", "")
    rows, info = memory.retrieve_memories("腰痛", "alice", "current")
    assert not rows and "未配置嵌入接口" in info["fallback_reason"]


def configure_vectors(monkeypatch):
    monkeypatch.setenv("QIWEN_MEMORY_MODE", "hybrid")
    monkeypatch.setenv("QIWEN_EMBEDDING_MODEL", "synthetic-test-model")
    monkeypatch.setenv("QIWEN_MEMORY_MIN_SIMILARITY", "0.8")
    monkeypatch.setattr(retrieval, "API_BASE", "https://example.invalid/v1")
    monkeypatch.setattr(retrieval, "API_KEY", "synthetic-key")


def test_hybrid_rrf_with_synthetic_vectors_and_cache_lifecycle(memory_db, monkeypatch):
    configure_vectors(monkeypatch)
    calls = []
    def embed(texts):
        calls.append(texts)
        return [[1.0,0.0] if "入睡" in text or "失眠" in text else [0.0,1.0] for text in texts]
    monkeypatch.setattr(retrieval, "_embed", embed)
    # No lexical overlap with either candidate: vectors alone admit the matching record.
    rows, info = memory.retrieve_memories("夜里难以入睡", "alice", "current")
    assert [row["thread_id"] for row in rows] == ["sleep"]
    assert info["active_mode"] == "hybrid" and rows[0]["channels"] == ["vector"]
    rows, _ = memory.retrieve_memories("失眠", "alice", "current")
    assert rows[0]["channels"] == ["bm25", "vector"]
    assert calls[1] == ["失眠"]  # Unchanged summaries are not embedded again.
    assert not db.one("SELECT * FROM memory_embeddings WHERE user_id='bob'")
    with db.connection() as conn:
        conn.execute("UPDATE profile_memories SET summary='入睡困难' WHERE thread_id='sleep'")
    memory.retrieve_memories("失眠", "alice", "current")
    assert calls[2] == ["失眠", "入睡困难"]
    with db.connection() as conn:
        conn.execute("DELETE FROM profile_memories WHERE thread_id='sleep'")
    assert not db.one("SELECT * FROM memory_embeddings WHERE thread_id='sleep'")


def test_model_threshold_must_be_configured(memory_db, monkeypatch):
    configure_vectors(monkeypatch)
    monkeypatch.delenv("QIWEN_MEMORY_MIN_SIMILARITY")
    monkeypatch.setattr(retrieval, "_embed", lambda _: pytest.fail("must not call uncalibrated vector branch"))
    rows, info = memory.retrieve_memories("头痛", "alice", "current")
    assert rows[0]["thread_id"] == "head" and "门槛" in info["fallback_reason"]


def test_bad_embedding_response_falls_back_to_keyword_search(memory_db, monkeypatch):
    configure_vectors(monkeypatch)
    monkeypatch.setattr(retrieval, "_embed", lambda _: [[float("nan"),0.0]] * 3)
    rows, info = memory.retrieve_memories("头痛", "alice", "current")
    assert [row["thread_id"] for row in rows] == ["head"]
    assert info["active_mode"] == "bm25" and "ValueError" in info["fallback_reason"]


def rerank_candidates(monkeypatch):
    monkeypatch.setenv("QIWEN_MEMORY_RERANK_MODE", "llm")
    monkeypatch.setattr(memory.model, "API_BASE", "https://example.invalid/v1")
    monkeypatch.setattr(memory.model, "API_KEY", "synthetic-key")
    monkeypatch.setenv("QIWEN_MODEL_MEMORY_RERANK", "synthetic-reranker")
    return memory.retrieve_memories("头痛，失眠", "alice", "current", limit=10)


def test_rerank_order_rejection_and_preserved_identity(memory_db, monkeypatch):
    rows, info = rerank_candidates(monkeypatch)
    async def reranker(system, user, *, role, schema):
        import json
        assert role == "memory_rerank" and schema is memory.MemoryRanking
        payload = json.loads(user)
        assert payload["query"] == "完整问题"
        assert {row["thread_id"] for row in payload["candidates"]} == {"head", "sleep"}
        assert "之前没有失眠" in user and "2026-09-01" in user
        return {"ordered_ids":["sleep", "head"]}
    monkeypatch.setattr(memory.model, "acomplete_json", reranker)
    selected, trace = asyncio.run(memory.rerank_memories("完整问题", rows, info, "alice", limit=1))
    assert [row["thread_id"] for row in selected] == ["sleep"]
    assert selected[0]["summary"] == "之前没有失眠"
    assert trace["rerank"]["status"] == "success" and trace["rerank"]["input_count"] == 2
    assert trace["selected"][0]["thread_id"] == "sleep"


def test_reranker_can_reject_all_candidates(memory_db, monkeypatch):
    rows, info = rerank_candidates(monkeypatch)
    async def reranker(*args, **kwargs):
        return {"ordered_ids":[]}
    monkeypatch.setattr(memory.model, "acomplete_json", reranker)
    selected, trace = asyncio.run(memory.rerank_memories("query", rows, info, "alice"))
    assert selected == [] and trace["selected"] == []
    assert trace["rerank"]["status"] == "success"


@pytest.mark.parametrize("result", [
    {"ordered_ids":["other-user"]}, {"ordered_ids":["head", "head"]},
    {"ordered_ids":[1]}, {"ordered_ids":["head"], "rewritten_summary":"invented"},
])
def test_invalid_rerank_output_falls_back_without_new_ids(memory_db, monkeypatch, result):
    rows, info = rerank_candidates(monkeypatch)
    async def reranker(*args, **kwargs):
        return result
    monkeypatch.setattr(memory.model, "acomplete_json", reranker)
    selected, trace = asyncio.run(memory.rerank_memories("query", rows, info, "alice"))
    assert [row["thread_id"] for row in selected] == [row["thread_id"] for row in rows]
    assert trace["rerank"]["status"] == "fallback"


def test_timeout_returns_fused_order(memory_db, monkeypatch):
    rows, info = rerank_candidates(monkeypatch)
    async def reranker(*args, **kwargs):
        raise TimeoutError("synthetic timeout")
    monkeypatch.setattr(memory.model, "acomplete_json", reranker)
    selected, trace = asyncio.run(memory.rerank_memories("query", rows, info, "alice"))
    assert selected == rows and "TimeoutError" in trace["rerank"]["fallback_reason"]


def test_rerank_cancellation_does_not_return_fallback_answer(memory_db, monkeypatch):
    rows, info = rerank_candidates(monkeypatch)
    async def reranker(*args, **kwargs):
        raise asyncio.CancelledError
    monkeypatch.setattr(memory.model, "acomplete_json", reranker)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(memory.rerank_memories("query", rows, info, "alice"))


def test_deleted_memory_not_revived_by_model_reply(memory_db, monkeypatch):
    rows, info = rerank_candidates(monkeypatch)
    async def reranker(*args, **kwargs):
        with db.connection() as conn:
            conn.execute("DELETE FROM profile_memories WHERE thread_id='head'")
        return {"ordered_ids":["head", "sleep"]}
    monkeypatch.setattr(memory.model, "acomplete_json", reranker)
    selected, trace = asyncio.run(memory.rerank_memories("query", rows, info, "alice"))
    assert [row["thread_id"] for row in selected] == ["sleep"]
    assert trace["rerank"]["output_count"] == 1


def test_foreign_memory_is_filtered_before_model_request(memory_db, monkeypatch):
    rows, info = rerank_candidates(monkeypatch)
    rows.append({"thread_id":"other-user", "summary":"头痛失眠", "updated_at":"2026-09-01T00:00:00+00:00", "channels":["vector"]})
    async def reranker(system, user, **kwargs):
        assert "other-user" not in user
        return {"ordered_ids":["head"]}
    monkeypatch.setattr(memory.model, "acomplete_json", reranker)
    selected, _ = asyncio.run(memory.rerank_memories("query", rows, info, "alice"))
    assert selected[0]["thread_id"] == "head"


@pytest.mark.parametrize("mode,configured,status", [("none", True, "disabled"), ("llm", False, "skipped_unconfigured")])
def test_disabled_or_unconfigured_rerank_is_explicit(memory_db, monkeypatch, mode, configured, status):
    rows, info = rerank_candidates(monkeypatch)
    monkeypatch.setenv("QIWEN_MEMORY_RERANK_MODE", mode)
    if not configured:
        monkeypatch.setattr(memory.model, "API_KEY", "")
    monkeypatch.setattr(memory.model, "acomplete_json", lambda *args, **kwargs: pytest.fail("must not call model"))
    selected, trace = asyncio.run(memory.rerank_memories("query", rows, info, "alice"))
    assert selected == rows and trace["rerank"]["status"] == status
