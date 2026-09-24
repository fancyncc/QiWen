from __future__ import annotations

import io
import json
import os
import tempfile
from uuid import uuid4

os.environ["QIWEN_DATA_DIR"] = tempfile.mkdtemp(prefix="qiwen-test-")

from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app
from app.retrieval import ingest_file, retrieve
from app.safety import check_herbs


def register(client: TestClient, name: str) -> tuple[str, str]:
    result = client.post("/api/register", json={"username": name, "password": "correct-horse-123"})
    assert result.status_code == 201, result.text
    data = result.json()
    return data["access_token"], data["user"]["id"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_accounts_threads_profiles_and_file_isolation():
    with TestClient(app) as client:
        a, a_id = register(client, "usera_" + uuid4().hex[:6])
        b, b_id = register(client, "userb_" + uuid4().hex[:6])
        thread = client.post("/api/threads", headers=auth(a), json={"mode": "fixed"}).json()["id"]
        assert client.get(f"/api/threads/{thread}/messages", headers=auth(b)).status_code == 404
        assert client.put(f"/api/threads/{thread}", headers=auth(b), json={"title": "steal"}).status_code == 404
        assert client.delete(f"/api/threads/{thread}", headers=auth(b)).status_code == 404
        assert client.get(f"/api/profiles/{a_id}", headers=auth(b)).status_code == 403
        assert client.patch(f"/api/profiles/{a_id}", headers=auth(b), json={"allergies": "甘草"}).status_code == 403
        assert client.patch(f"/api/profiles/{a_id}", headers=auth(a), json={"allergies": "甘草"}).status_code == 200
        assert client.get(f"/api/profiles/{a_id}", headers=auth(a)).json()["allergies"] == "甘草"
        uploaded = client.post("/api/upload", headers=auth(a), files={"file": ("note.txt", b"hello", "text/plain")})
        assert uploaded.status_code == 201
        file_id = uploaded.json()["id"]
        denied = client.post("/api/chat", headers=auth(b), json={"message": "你好", "files": [file_id]})
        assert denied.status_code == 404
        assert client.post("/api/chat", headers=auth(b), json={"message": "你好", "thread_id": thread}).status_code == 404
        assert client.get("/api/profiles/" + b_id, headers=auth(b)).status_code == 200


def test_followup_sse_multi_and_history():
    with TestClient(app) as client:
        token, user_id = register(client, "flow_" + uuid4().hex[:6])
        headers = auth(token)
        thread = client.post("/api/threads", headers=headers, json={"mode": "fixed"}).json()["id"]
        first = client.post("/api/chat", headers=headers, json={"message": "最近失眠", "thread_id": thread}).json()
        assert first["intent_type"] == "consult"
        assert client.get("/api/threads", headers=headers).json()[0]["title"] == "最近失眠"
        second = client.post("/api/chat", headers=headers, json={"message": "三天，白天有些疲劳", "thread_id": thread}).json()
        assert second["intent_type"] == "diagnose"
        assert "失眠" in second["response"]
        assert len(client.get(f"/api/threads/{thread}/messages", headers=headers).json()) == 4
        assert "甘草" in second["response"]
        client.patch(f"/api/profiles/{user_id}", headers=headers, json={"allergies": "甘草"})
        blocked = client.post("/api/chat", headers=headers, json={"message": "最近失眠三天，伴有疲劳", "thread_id": thread}).json()
        assert "药材清单已隐藏" in blocked["response"]
        assert "酸枣仁、茯苓" not in blocked["response"]

        multi = client.post("/api/threads", headers=headers, json={"mode": "multi"}).json()["id"]
        with client.stream("POST", "/api/chat?stream=true", headers=headers,
                           json={"message": "最近咳嗽三天，伴有疲劳", "thread_id": multi}) as response:
            events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
        assert response.status_code == 200
        assert any(item.get("stage") == "herb" for item in events)
        def at(stage: str, status: str) -> int:
            return next(i for i, event in enumerate(events)
                        if event.get("stage") == stage and event.get("status") == status)
        for stage in ("inquiry", "search", "bianzheng", "formula", "herb", "synthesis", "reviewer", "safety"):
            assert at(stage, "start") < at(stage, "done")
        assert at("synthesis", "start") > max(at(stage, "done") for stage in ("bianzheng", "formula", "herb"))
        assert any("token" in item for item in events)
        assert events[-1]["done"] is True
        assert len(events[-1]["notes"]) == 3
        assert len(client.get(f"/api/threads/{multi}/messages", headers=headers).json()) == 2

        client.patch(f"/api/profiles/{user_id}", headers=headers, json={"allergies": ""})
        example = client.post("/api/chat", headers=headers,
                              json={"message": "最近总是睡不好，白天容易疲劳"}).json()
        assert "失眠" in example["symptoms"]
        assert "酸枣仁汤" in example["response"]


def test_upload_validation_and_retrieval(tmp_path, monkeypatch):
    with TestClient(app) as client:
        token, _ = register(client, "files_" + uuid4().hex[:6])
        headers = auth(token)
        assert client.post("/api/upload", headers=headers, files={"file": ("bad.exe", b"bad")}).status_code == 400
        assert client.post("/api/upload", headers=headers, files={"file": ("huge.txt", b"x" * (10 * 1024 * 1024 + 1))}).status_code == 413
        doc = Document()
        doc.add_paragraph("测试病历内容")
        stream = io.BytesIO()
        doc.save(stream)
        assert client.post("/api/upload", headers=headers, files={"file": ("note.docx", stream.getvalue())}).status_code == 201
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        valid_pdf = io.BytesIO()
        writer.write(valid_pdf)
        assert client.post("/api/upload", headers=headers, files={"file": ("blank.pdf", valid_pdf.getvalue())}).status_code == 201
        pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
        assert client.post("/api/upload", headers=headers, files={"file": ("broken.pdf", pdf)}).status_code == 400

    text_file = tmp_path / "book.txt"
    text_file.write_text("桂枝汤相关的古籍学习片段。这里是用于检索测试的文本内容。", encoding="utf-8")
    assert ingest_file(text_file, "测试古籍") == 1
    assert ingest_file(text_file, "测试古籍") == 0
    assert retrieve("用于检索测试的文本内容")[0]["title"] == "测试古籍"
    from app import retrieval
    monkeypatch.setattr(retrieval, "RAG_MODE", "hybrid")
    def unavailable(_texts):
        raise RuntimeError("offline")
    monkeypatch.setattr(retrieval, "_embed", unavailable)
    fallback, warning = retrieval.retrieve_with_status("桂枝汤")
    assert fallback and "退化为关键词检索" in warning


def test_contraindication_rules():
    assert check_herbs(["炙甘草", "甘遂"])
    assert check_herbs(["生川乌", "法半夏"])
    assert check_herbs(["官桂", "赤石脂"])
    assert check_herbs(["未知药材"])
    assert check_herbs([])
    assert not check_herbs(["酸枣仁", "茯苓", "知母", "川芎", "甘草"])


def test_bundled_classics_seed_is_complete_and_idempotent():
    from app.corpus_seed import BOOKS, seed_bundled_corpus
    from app.db import connection

    with TestClient(app):
        pass
    with connection() as db:
        counts = dict(db.execute("SELECT title, COUNT(*) FROM corpus GROUP BY title"))
        assert all(counts.get(book, 0) > 0 for book in BOOKS)
        assert sum(counts[book] for book in BOOKS) >= 1500
        assert seed_bundled_corpus(db) == 0


def test_bundled_data_can_be_rebuilt_from_preserved_sources():
    from data_pipeline.ingest import chunk_all_books
    from scripts.rebuild_classics import CHUNK_DIR, KNOWLEDGE_FILE, RAW_DIR, rebuild_knowledge

    generated = chunk_all_books(str(RAW_DIR))
    assert sum(map(len, generated.values())) == 1537
    for book, chunks in generated.items():
        saved = json.loads((CHUNK_DIR / f"{book}.json").read_text(encoding="utf-8"))
        assert saved == [
            {"id": chunk.id, "content": chunk.content, "metadata": chunk.metadata}
            for chunk in chunks
        ]
    saved_knowledge = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    assert saved_knowledge == rebuild_knowledge()
    assert [len(saved_knowledge[key]) for key in ("herbs", "formulas", "contraindications")] == [33, 22, 14]
