"""Single-use attachment parser; stdout contains only a bounded JSON result."""
from __future__ import annotations

import json
import sys


def main():
    from .attachments import PARSER_MEMORY_BYTES, PARSER_TIMEOUT_SECONDS, extract_file
    from .config import MAX_UPLOAD_BYTES
    from fastapi import HTTPException

    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (PARSER_TIMEOUT_SECONDS, PARSER_TIMEOUT_SECONDS))
    try:
        data = sys.stdin.buffer.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "文件不能超过 10 MB")
        media_type, content = extract_file("attachment" + sys.argv[1], data)
        result = {"media_type": media_type, "content": content}
    except HTTPException as error:
        result = {"status": error.status_code, "error": error.detail}
    except MemoryError:
        result = {"status": 413, "error": "附件解析超出资源限制，请简化或拆分文件后上传。"}
    except Exception:
        result = {"status": 400, "error": "文件无法解析"}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
