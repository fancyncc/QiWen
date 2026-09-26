"""Build or resume the vector index for the bundled and imported corpus."""

from __future__ import annotations

from app.config import API_BASE, API_KEY, RAG_MODE
from app.db import all_rows, init_db
from app.retrieval import index_vectors


def main() -> int:
    if RAG_MODE != "hybrid":
        raise SystemExit("Set QIWEN_RAG_MODE=hybrid before indexing vectors.")
    if not API_BASE or not API_KEY:
        raise SystemExit("Configure the embedding-compatible API base and key first.")
    init_db()
    rows = all_rows("SELECT id,title,chapter,body FROM corpus ORDER BY id")
    added = index_vectors(rows)
    print(f"Corpus segments: {len(rows)}; newly indexed: {added}; already indexed: {len(rows) - added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
