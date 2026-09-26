"""Create a verified data snapshot or restore it into an empty directory.

Back up while no requests are writing data. The vector index is excluded because
it can be rebuilt from the preserved corpus with scripts.index_vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.config import DATA_DIR

DATABASES = ("qiwen.sqlite3", "checkpoints.sqlite3")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_snapshot(output: Path, data_dir: Path = DATA_DIR) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    business_db = data_dir / DATABASES[0]
    if not business_db.is_file():
        raise FileNotFoundError(f"Business database not found: {business_db}")
    if output.exists():
        raise FileExistsError(f"Snapshot already exists: {output}")
    files: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="qiwen-db-backup-") as temporary:
        for name in DATABASES:
            source = data_dir / name
            if not source.is_file():
                continue
            snapshot = Path(temporary) / name
            with closing(sqlite3.connect(source)) as current, closing(sqlite3.connect(snapshot)) as destination:
                current.backup(destination)
            files[f"databases/{name}"] = snapshot.read_bytes()
    upload_dir = data_dir / "uploads"
    if upload_dir.exists():
        for path in upload_dir.rglob("*"):
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(upload_dir).as_posix()
                files[f"uploads/{relative}"] = path.read_bytes()
    manifest = {
        "format": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {name: _digest(content) for name, content in files.items()},
        "vector_index": "Excluded; rebuild from corpus with scripts.index_vectors.",
    }
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def restore_snapshot(archive_path: Path, target_dir: Path) -> dict:
    target_dir = target_dir.resolve()
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"Restore target must be empty: {target_dir}")
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("Snapshot manifest is missing")
        manifest = json.loads(archive.read("manifest.json"))
        files = manifest.get("files", {})
        if manifest.get("format") != 1 or "databases/qiwen.sqlite3" not in files:
            raise ValueError("Unsupported or incomplete snapshot")
        if names != set(files) | {"manifest.json"}:
            raise ValueError("Snapshot contains unexpected entries")
        verified: dict[str, bytes] = {}
        for name, expected in files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not (
                name in {f"databases/{db}" for db in DATABASES} or name.startswith("uploads/")
            ):
                raise ValueError(f"Unsafe snapshot path: {name}")
            content = archive.read(name)
            if _digest(content) != expected:
                raise ValueError(f"Snapshot checksum mismatch: {name}")
            verified[name] = content
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, content in verified.items():
        relative = Path(name)
        target = target_dir / (relative.name if relative.parts[0] == "databases" else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    backup = subcommands.add_parser("backup")
    backup.add_argument("--output", type=Path, required=True)
    restore = subcommands.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "backup":
        manifest = create_snapshot(args.output)
        print(f"Snapshot created: {args.output}; files: {len(manifest['files'])}")
    else:
        manifest = restore_snapshot(args.archive, args.target_dir)
        print(f"Snapshot restored: {args.target_dir}; files: {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
