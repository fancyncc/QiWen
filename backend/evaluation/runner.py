"""Run synthetic routing, safety, retrieval and isolation checks.

The checks exercise the API with a temporary database and do not use patient data.
Answer quality and clinical appropriateness still require expert review.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def run(real: bool = False) -> dict:
    if real:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        required = ("QIWEN_MODEL_BASE_URL", "QIWEN_MODEL_API_KEY", "QIWEN_MODEL_NAME")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Missing real-model configuration: " + ", ".join(missing))
        os.environ["QIWEN_MODEL_MODE"] = "openai_compatible"
    else:
        os.environ["QIWEN_MODEL_MODE"] = "mock"
        os.environ["QIWEN_RAG_MODE"] = "mock"

    with tempfile.TemporaryDirectory(prefix="qiwen-evaluation-") as temporary:
        os.environ["QIWEN_DATA_DIR"] = temporary
        from fastapi.testclient import TestClient
        from app.main import app
        from app.retrieval import retrieve_with_status
        from app.safety import check_herbs

        baseline = json.loads((Path(__file__).with_name("baseline.json")).read_text(encoding="utf-8"))
        checks: dict[str, dict] = {}

        def check(name: str, callback) -> None:
            start = time.monotonic()
            try:
                callback()
                checks[name] = {"passed": True, "elapsed_ms": round((time.monotonic() - start) * 1000)}
            except Exception as exc:
                checks[name] = {"passed": False, "elapsed_ms": round((time.monotonic() - start) * 1000),
                                "error": str(exc)[:300]}

        with TestClient(app) as client:
            def register() -> tuple[str, str]:
                response = client.post("/api/register", json={
                    "username": "eval_" + uuid4().hex[:10], "password": "evaluation-only-password",
                })
                response.raise_for_status()
                data = response.json()
                return data["access_token"], data["user"]["id"]

            token, user_id = register()
            headers = {"Authorization": f"Bearer {token}"}
            first_thread = ""

            def conversation() -> None:
                nonlocal first_thread
                created = client.post("/api/threads", headers=headers, json={"mode": "fixed"})
                created.raise_for_status()
                first_thread = created.json()["id"]
                first = client.post("/api/chat", headers=headers,
                                    json={"thread_id": first_thread, "message": baseline["followup_and_history"]["inputs"][0]})
                first.raise_for_status()
                assert first.json()["intent_type"] == baseline["followup_and_history"]["expected_intents"][0], "brief symptom did not trigger follow-up"
                second = client.post("/api/chat", headers=headers,
                                     json={"thread_id": first_thread, "message": baseline["followup_and_history"]["inputs"][1]})
                second.raise_for_status()
                assert second.json()["intent_type"] == baseline["followup_and_history"]["expected_intents"][1], "follow-up did not resume consultation"
                history = client.get(f"/api/threads/{first_thread}/messages", headers=headers)
                history.raise_for_status()
                assert len(history.json()) == baseline["followup_and_history"]["expected_messages"], "message history was not persisted"

            check("followup_and_history", conversation)

            def isolation() -> None:
                assert first_thread, "conversation setup failed"
                other_token, _ = register()
                response = client.get(f"/api/threads/{first_thread}/messages",
                                      headers={"Authorization": f"Bearer {other_token}"})
                assert response.status_code == baseline["account_isolation"]["expected_status"], "another account can read the consultation"

            check("account_isolation", isolation)

            def urgent() -> None:
                response = client.post("/api/chat", headers=headers,
                                       json={"message": baseline["urgent_route"]["input"]})
                response.raise_for_status()
                data = response.json()
                assert data["intent_type"] == baseline["urgent_route"]["expected_intent"], "urgent symptoms were not stopped"
                assert baseline["urgent_route"]["forbidden"] not in data["response"], "urgent response contains a formula"

            check("urgent_route", urgent)

            def contraindication() -> None:
                assert bool(check_herbs(baseline["contraindication"]["herbs"])) == baseline["contraindication"]["expected_error"], "known incompatible pair passed the checker"

            check("contraindication", contraindication)

            def allergy() -> None:
                profile = client.patch(f"/api/profiles/{user_id}", headers=headers,
                                       json=baseline["allergy_and_final_history"]["profile"])
                profile.raise_for_status()
                created = client.post("/api/threads", headers=headers, json={"mode": "multi"})
                created.raise_for_status()
                thread_id = created.json()["id"]
                response = client.post("/api/chat", headers=headers, json={
                    "thread_id": thread_id, "message": baseline["allergy_and_final_history"]["input"],
                })
                response.raise_for_status()
                result = response.json()["response"]
                assert baseline["allergy_and_final_history"]["required"] in result, "unsafe candidate was displayed"
                saved = client.get(f"/api/threads/{thread_id}/messages", headers=headers)
                saved.raise_for_status()
                assert saved.json()[-1]["content"] == result, "history differs from reviewed result"

            check("allergy_and_final_history", allergy)

            def multi_stream() -> None:
                created = client.post("/api/threads", headers=headers, json={"mode": "multi"})
                created.raise_for_status()
                thread_id = created.json()["id"]
                with client.stream("POST", "/api/chat?stream=true", headers=headers, json={
                    "thread_id": thread_id, "message": baseline["multi_stream_and_final_version"]["input"],
                }) as response:
                    response.raise_for_status()
                    events = [json.loads(line[6:]) for line in response.iter_lines()
                              if line.startswith("data: ")]
                assert events and events[-1].get("done") is True, "stream did not complete"
                for stage in baseline["multi_stream_and_final_version"]["experts"]:
                    start = next(i for i, event in enumerate(events)
                                 if event.get("stage") == stage and event.get("status") == "start")
                    done = next(i for i, event in enumerate(events)
                                if event.get("stage") == stage and event.get("status") == "done")
                    assert start < done, f"{stage} was marked done before it started"
                synthesis = next(i for i, event in enumerate(events)
                                 if event.get("stage") == "synthesis" and event.get("status") == "start")
                assert synthesis > max(i for i, event in enumerate(events)
                                       if event.get("stage") in {"bianzheng", "formula", "herb"}
                                       and event.get("status") == "done"), "synthesis started before experts completed"
                saved = client.get(f"/api/threads/{thread_id}/messages", headers=headers)
                saved.raise_for_status()
                shown = events[-1]["payload"]["response"]
                assert saved.json()[-1]["content"] == shown, "streamed text differs from saved final answer"
                assert len(events[-1].get("notes", [])) == baseline["multi_stream_and_final_version"]["expected_notes"], "expert notes were not finalized"

            check("multi_stream_and_final_version", multi_stream)

            def retrieval() -> None:
                docs, warning = retrieve_with_status(baseline["retrieval_provenance"]["query"])
                assert not warning, warning
                assert docs and all(all(doc.get(key) for key in baseline["retrieval_provenance"]["required_fields"]) for doc in docs), \
                    "retrieval returned no traceable classical source"

            check("retrieval_provenance", retrieval)

        return {
            "mode": "real" if real else "mock",
            "baseline_version": baseline["version"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "passed": sum(item["passed"] for item in checks.values()),
            "total": len(checks),
            "manual_review_required": ["citation_faithfulness", "answer_quality", "clinical_appropriateness"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Use the configured real chat model")
    parser.add_argument("--output", type=Path, help="Write a JSON report to this path")
    args = parser.parse_args()
    report = run(args.real)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
