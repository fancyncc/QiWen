"""Bound document expansion before parsing and isolate expensive parsers."""
from __future__ import annotations

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path

from fastapi import HTTPException

from .documents import MAX_FILE_TEXT_CHARS

MAX_DOCX_PARTS = 1024
MAX_DOCX_EXPANDED_BYTES = 32 * 1024 * 1024
MAX_DOCX_PART_BYTES = 16 * 1024 * 1024
MAX_DOCX_XML_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 200
PARSER_TIMEOUT_SECONDS = 20
PARSER_MEMORY_BYTES = 512 * 1024 * 1024


def check_docx(data: bytes):
    # python-docx loads ZIP parts before text limits can be applied. Reject
    # oversized parts using the directory, without expanding any entry first.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = archive.infolist()
        if (len(parts) > MAX_DOCX_PARTS
                or sum(part.file_size for part in parts) > MAX_DOCX_EXPANDED_BYTES
                or any(part.file_size > (MAX_DOCX_XML_BYTES if part.filename.lower().endswith((".xml", ".rels"))
                                        else MAX_DOCX_PART_BYTES) for part in parts)):
            raise HTTPException(413, "DOCX 解压后的内容过大或条目过多，请精简文件后上传。")


def extract_file(filename: str, data: bytes) -> tuple[str, str]:
    extension = Path(filename).suffix.lower()

    def collect(parts):
        chunks, size = [], 0
        for part in parts:
            size += len(part) + (1 if chunks else 0)
            if size > MAX_FILE_TEXT_CHARS:
                raise HTTPException(413, "附件文本超过 20000 字符，请拆分文件后上传；本次未保存，也未截断处理。")
            chunks.append(part)
        return "\n".join(chunks)

    try:
        if extension == ".txt":
            return "text/plain", collect([data.decode("utf-8-sig")])
        if extension == ".pdf" and data.startswith(b"%PDF"):
            from pypdf import PdfReader
            from pypdf import filters
            # Bound each FlateDecode stream as well as the worker's total memory.
            filters.ZLIB_MAX_OUTPUT_LENGTH = MAX_DOCX_XML_BYTES
            pages = PdfReader(io.BytesIO(data)).pages
            if len(pages) > MAX_PDF_PAGES:
                raise HTTPException(413, "PDF 超过 200 页，请拆分文件后上传。")
            return "application/pdf", collect(page.extract_text() or "" for page in pages)
        if extension == ".docx" and data.startswith(b"PK"):
            check_docx(data)
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", collect(
                block.text if hasattr(block, "text") else "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                for block in doc.iter_inner_content())
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "文件无法解析") from None
    raise HTTPException(400, "仅支持有效的 PDF、DOCX、UTF-8 TXT 文件")


async def parse_attachment(filename: str, data: bytes, slots: asyncio.Semaphore) -> tuple[str, str]:
    if Path(filename).suffix.lower() == ".txt":
        return extract_file(filename, data)
    if slots.locked():
        raise HTTPException(429, "正在解析其他附件，请稍后重试。")
    async with slots:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "app.attachment_worker", Path(filename).suffix.lower(),
            cwd=str(Path(__file__).resolve().parent.parent),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            try:
                output, _ = await asyncio.wait_for(process.communicate(data), timeout=PARSER_TIMEOUT_SECONDS)
            except TimeoutError:
                raise HTTPException(413, "附件解析超时，请简化或拆分文件后上传。") from None
            if process.returncode != 0:
                raise HTTPException(413, "附件解析超出资源限制或未能完成，请简化或拆分文件后上传。")
            result = json.loads(output)
            if result.get("error"):
                raise HTTPException(result["status"], result["error"])
            return result["media_type"], result["content"]
        finally:
            # wait_for alone cannot stop a parser thread. A child process can
            # always be terminated on timeout or request cancellation.
            if process.returncode is None:
                process.kill()
            await process.wait()
