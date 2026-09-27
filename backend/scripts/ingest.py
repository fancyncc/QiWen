from __future__ import annotations

import argparse
from pathlib import Path

from app.db import init_db
from app.retrieval import ingest_file
from app.corpus_seed import BOOKS

parser = argparse.ArgumentParser(description="导入已获授权的 UTF-8 古籍文本")
parser.add_argument("path", type=Path)
parser.add_argument("--title")
parser.add_argument("--book", choices=BOOKS, help="指定六部古籍的结构切分器；默认按书名识别")
args = parser.parse_args()
init_db()
print(f"新增 {ingest_file(args.path, args.title, book=args.book)} 个片段")
