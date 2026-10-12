"""Isolated browser login/send/reload/new-thread smoke against the deployed UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import httpx

from .deployment import endpoint, is_loopback
from .runtime import manifest, write_report


def run(base: str, executable: str | None = None) -> dict:
    checks, thread_ids = [], []
    stage = "login"
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"status": "blocked", "reason": "install_evaluation_playwright_dependency", "checks": []}
    with httpx.Client(base_url=base, timeout=30, trust_env=not is_loopback(base)) as api:
        username, password = "eval_browser_" + uuid4().hex[:10], "synthetic-browser-password"
        registered = api.post("/api/register", json={"username": username, "password": password})
        registered.raise_for_status()
        headers = {"Authorization": "Bearer " + registered.json()["access_token"]}
        try:
            with sync_playwright() as driver:
                browser = driver.chromium.launch(headless=True, args=["--no-proxy-server"] if is_loopback(base) else [],
                                                 **({"executable_path": executable} if executable else {}))
                context = browser.new_context()
                try:
                    page = context.new_page()
                    page.goto(base + "/login")
                    page.get_by_label("用户名", exact=True).fill(username)
                    page.get_by_label("密码", exact=True).fill(password)
                    page.get_by_role("button", name="登录", exact=True).click()
                    page.wait_for_url("**/chat", timeout=30000)
                    checks.append({"name": "login", "passed": True})
                    stage = "final_answer_visible"
                    page.get_by_placeholder("请描述您的症状...").fill("你好")
                    page.get_by_role("button", name="发送", exact=True).click()
                    page.get_by_text("你好，可以向我提问中医基础概念或古籍知识", exact=False).wait_for(timeout=60000)
                    page.get_by_role("button", name="停止生成", exact=True).wait_for(state="hidden")
                    checks.append({"name": "final_answer_visible", "passed": True})
                    stage = "reload_restores_authoritative_answer"
                    # Sending from /chat creates a saved thread but leaves the new-chat URL.
                    # Enter that thread through the sidebar before testing URL-based restoration.
                    page.get_by_role("button", name="你好", exact=True).click()
                    page.wait_for_url("**/chat/*", timeout=30000)
                    page.reload()
                    page.get_by_text("你好，可以向我提问中医基础概念或古籍知识", exact=False).wait_for(timeout=30000)
                    checks.append({"name": "reload_restores_authoritative_answer", "passed": True})
                    stage = "new_thread_identity_confirmation"
                    # /chat is the product's empty/new conversation entry point.
                    page.goto(base + "/chat")
                    page.get_by_placeholder("请描述您的症状...").wait_for()
                    page.get_by_placeholder("请描述您的症状...").fill("咳嗽三天，有白痰")
                    page.get_by_role("button", name="发送", exact=True).click()
                    page.get_by_text("这次是你本人", exact=False).wait_for(timeout=60000)
                    checks.append({"name": "new_thread_identity_confirmation", "passed": True})
                finally:
                    context.close()
                    browser.close()
        except Exception as exc:
            checks.append({"name": stage, "passed": False, "error_type": type(exc).__name__})
        finally:
            response = api.get("/api/threads", headers=headers)
            if response.status_code == 200:
                thread_ids = [row["id"] for row in response.json()]
                for thread in thread_ids:
                    api.delete(f"/api/threads/{thread}", headers=headers)
    return {"status": "passed" if checks and all(c["passed"] for c in checks) else "failed", "checks": checks,
            "cleanup": {"synthetic_threads_deleted": len(thread_ids), "synthetic_account_retained": True}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:23000")
    parser.add_argument("--executable", help="Optional installed Chromium/Edge path")
    parser.add_argument("--reports", type=Path, default=Path(__file__).with_name("reports"))
    args = parser.parse_args()
    try:
        report = run(endpoint(args.base_url), args.executable)
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__}
    report["manifest"] = manifest("deployed_browser")
    path = write_report(args.reports, "browser", report)
    print(json.dumps({"report": str(path), "status": report["status"], "checks": report.get("checks")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2 if report["status"] == "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
