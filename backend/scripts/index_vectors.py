"""Build or resume the vector index for the bundled and imported corpus."""

from __future__ import annotations

import argparse
import json

from app.config import API_BASE, API_KEY, RAG_MODE
from app.db import all_rows, init_db
from app.retrieval import index_vectors, _collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding inputs per request, 1 to 128")
    parser.add_argument("--prune-stale", action="store_true", help="Remove vector IDs absent from the current corpus, after indexing succeeds")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        parser.error("batch size must be between 1 and 128")
    if RAG_MODE != "hybrid":
        raise SystemExit("Set QIWEN_RAG_MODE=hybrid before indexing vectors.")
    if not API_BASE or not API_KEY:
        raise SystemExit("Configure the embedding-compatible API base and key first.")
    init_db()
    rows = all_rows("SELECT id,title,chapter,body,metadata FROM corpus ORDER BY id")
    added = index_vectors(rows, batch_size=args.batch_size,
        progress=lambda done, total: print(json.dumps({"new_vectors_completed": done, "missing_at_start": total}), flush=True))
    removed = 0
    if args.prune_stale:
        current_ids = {row["id"] for row in all_rows("SELECT id FROM corpus")}
        collection = _collection()
        stale = sorted(set(collection.get(include=[])["ids"]) - current_ids)
        for start in range(0, len(stale), 128):
            collection.delete(ids=stale[start:start + 128])
        removed = len(stale)
    print(f"Corpus segments: {len(rows)}; newly indexed: {added}; already indexed: {len(rows) - added}; stale vectors removed: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
