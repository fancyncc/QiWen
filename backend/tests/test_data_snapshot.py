from __future__ import annotations

import sqlite3
import zipfile
from contextlib import closing

import pytest

from scripts.data_snapshot import create_snapshot, restore_snapshot


def test_snapshot_restores_database_and_uploads_and_rejects_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with closing(sqlite3.connect(source / "qiwen.sqlite3")) as db:
        with db:
            db.execute("CREATE TABLE sample(value TEXT)")
            db.execute("INSERT INTO sample VALUES ('saved consultation')")
    (source / "uploads").mkdir()
    (source / "uploads" / "report.txt").write_text("synthetic report", encoding="utf-8")
    archive = tmp_path / "backup.zip"
    manifest = create_snapshot(archive, source)
    assert len(manifest["files"]) == 2
    destination = tmp_path / "restored"
    restore_snapshot(archive, destination)
    with closing(sqlite3.connect(destination / "qiwen.sqlite3")) as db:
        assert db.execute("SELECT value FROM sample").fetchone()[0] == "saved consultation"
    assert (destination / "uploads" / "report.txt").read_text(encoding="utf-8") == "synthetic report"
    with pytest.raises(FileExistsError):
        restore_snapshot(archive, destination)
    corrupt = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(archive) as source_zip, zipfile.ZipFile(corrupt, "w") as bad_zip:
        for name in source_zip.namelist():
            bad_zip.writestr(name, b"changed" if name == "uploads/report.txt" else source_zip.read(name))
    with pytest.raises(ValueError, match="checksum"):
        restore_snapshot(corrupt, tmp_path / "corrupt-target")
