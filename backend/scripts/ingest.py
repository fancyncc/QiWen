from __future__ import annotations

import argparse
from pathlib import Path

from app.db import init_db
from app.retrieval import ingest_file

parser = argparse.ArgumentParser(description="导入已获授权的 UTF-8 古籍文本")
parser.add_argument("path", type=Path)
parser.add_argument("--title")
args = parser.parse_args()
init_db()
print(f"新增 {ingest_file(args.path, args.title)} 个片段")

