"""Conservative line cleaning with offsets into the preserved source snapshot."""
from __future__ import annotations

import hashlib
import re

CLEANING_VERSION = 1
NAVIGATION = re.compile(r"^(?:中国哲学书电子化计划|Chinese Text Project|Library Resources|Show all|Full-text search|"
    r"修改|查看历史|首页|导航|登录|登入|注册|帮助|联系我们|友情链接|\]\s*\[|[=\-]{3,}|\d{1,4})$", re.I)
FOOTER = re.compile(r"^(?:我们的网站|网站的内容|本站|版权所有|版权声明|Copyright|如欲引用|如果您引用|"
    r"请同时注明|下载本网站|严禁使用自动下载|违者自动封锁|支持本站|赞助本站|广告推广|"
    r"Please confirm|请确认|人机验证|验证码|Cloudflare|challenge|captcha)", re.I)


def clean_source_text(text: str, book: str = "") -> tuple[str, dict]:
    kept, removed, offset = [], [], 0
    lines = text.splitlines(keepends=True)
    footer = False
    for index, line in enumerate(lines):
        value = line.strip()
        if footer and re.match(r"^#{1,6}\s+", value):
            footer = False
        if value == "URN" and any("我们的网站" in item for item in lines[index:index + 8]):
            footer = True
        heading = re.sub(r"^#{1,6}\s*", "", value).strip("《》")
        junk = bool(footer or NAVIGATION.fullmatch(value) or FOOTER.match(value) or
                    (book and heading in {book, book.translate(str.maketrans("农经", "農經"))}) or
                    re.fullmatch(r"[上下中]卷》?", heading))
        if junk:
            removed.append([offset, offset + len(line)])
        else:
            kept.append(line)
        offset += len(line)
    cleaned = "".join(kept).strip()
    return cleaned, {"cleaning_version": CLEANING_VERSION,
                     "raw_body_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "removed_source_ranges": removed}
