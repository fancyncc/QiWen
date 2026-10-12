"""Synthetic multi-turn API contracts, inspecting admitted inputs in process only."""
from __future__ import annotations

import json
from contextlib import ExitStack
from unittest.mock import patch

from .benchmark import load_jsonl, register


def run_scenarios(client, limit=None, modes=("fixed", "multi")) -> dict:
    from app import db, model, safety_store, workflow, knowledge
    cases = load_jsonl("scenarios.jsonl", limit)
    details = []
    for case in cases:
        for mode in modes:
            record = {"id": case["id"], "family": case["family"], "mode": mode, "checks": [], "passed": False}
            try:
                headers = register(client)
                uid = client.get("/api/me", headers=headers).json()["id"]
                threads, first_sources = {}, {}
                for step in case["steps"]:
                    op = step.get("op", "chat")
                    key = step.get("thread", "main")
                    if key not in threads:
                        response = client.post("/api/threads", headers=headers, json={"mode": mode})
                        response.raise_for_status()
                        threads[key] = response.json()["id"]
                    thread = threads[key]
                    if op == "profile":
                        response = client.patch(f"/api/profiles/{uid}", headers=headers, json=step["values"])
                        assert response.status_code == step.get("status", 200), "profile_status"
                    elif op == "resolve":
                        suggestions = client.get("/api/profile-suggestions", headers=headers, params={"thread_id": thread}).json()
                        items = [i for i in suggestions["items"] if i["fact"]["category"] == step.get("category", "allergy")]
                        assert items, "expected_profile_suggestion"
                        response = client.post("/api/profile-suggestions/resolve", headers=headers, json={
                            "revision": suggestions["revision"], "decisions": [{"id": i["id"], "action": step.get("action", "accept"),
                            "confirm_self": True} for i in items]})
                        assert response.status_code == 200, "resolve_status"
                    elif op == "revoke":
                        profile = client.get(f"/api/profiles/{uid}", headers=headers).json()
                        assert profile["facts"], "expected_confirmed_fact"
                        for fact in profile["facts"]:
                            response = client.delete(f'/api/profile-facts/{fact["id"]}', headers=headers,
                                                     params={"revision": profile["safety_revision"]})
                            response.raise_for_status()
                            profile = client.get(f"/api/profiles/{uid}", headers=headers).json()
                    elif op == "gap":
                        state = safety_store.load_state(uid, thread)
                        for episode in state.get("episodes", {}).values():
                            episode["last_seen_at"] = "2020-01-01T00:00:00+00:00"
                        with db.connection() as conn:
                            conn.execute("UPDATE thread_safety_state SET state=? WHERE thread_id=?", (json.dumps(state), thread))
                    elif op == "probe_other":
                        other = register(client)
                        response = client.get(f"/api/threads/{thread}/messages", headers=other)
                        assert response.status_code == 404, "cross_account_history"
                    elif op == "delete":
                        assert client.delete(f"/api/threads/{thread}", headers=headers).status_code == 204
                        assert client.get(f"/api/threads/{thread}/messages", headers=headers).status_code == 404
                    else:
                        captured = []
                        token = model.input_observer.set(captured.append)
                        try:
                            with ExitStack() as stack:
                                if step.get("fault") == "empty_retrieval":
                                    async def empty(_plan):
                                        return [], {"warning": "", "selected": []}
                                    stack.enter_context(patch.object(workflow, "retrieve_classics", empty))
                                    stack.enter_context(patch.object(knowledge, "retrieve_classics", empty))
                                files = []
                                for index, text in enumerate(step.get("attachments", [])):
                                    upload = client.post("/api/upload", headers=headers, files={"file": (
                                        f"synthetic-{index}.txt", text.encode("utf-8"), "text/plain")})
                                    upload.raise_for_status()
                                    files.append(upload.json()["id"])
                                response = client.post("/api/chat", headers=headers, json={"thread_id": thread,
                                    "message": step["input"], "mode": mode, "files": files})
                        finally:
                            model.input_observer.reset(token)
                        assert response.status_code == step.get("status", 200), "chat_status"
                        data = response.json()
                        state = safety_store.load_state(uid, thread)
                        answer = data.get("response", "")
                        contexts = [c for c in captured if c["kind"] in {"clinical_context", "knowledge_context", "model_request"}]
                        admitted = "\n".join(json.dumps(c, ensure_ascii=False) for c in contexts)
                        if step.get("required_context") or step.get("forbidden_context"):
                            assert contexts, "model_path_not_exercised"
                        assert not step.get("intents") or data.get("intent_type") in step["intents"], "next_action"
                        if step.get("subject"):
                            assert data.get("trace", {}).get("safety", {}).get("subject") == step["subject"], "patient_scope"
                        for term in step.get("required_text", []):
                            assert term in answer, "required_answer"
                        for term in step.get("forbidden_text", []):
                            assert term not in answer, "forbidden_answer"
                        for term in step.get("required_context", []):
                            assert term in admitted, "required_admitted_fact"
                        for term in step.get("forbidden_context", []):
                            assert term not in admitted, "forbidden_admitted_fact"
                        assert set(step.get("symptoms", [])) <= set(data.get("symptoms", [])), "positive_symptoms"
                        assert not set(step.get("denied", [])) & set(data.get("symptoms", [])), "negative_symptoms"
                        starts = {e.get("stage") for e in data.get("trace", {}).get("events", []) if e.get("status") == "start"}
                        assert not set(step.get("forbidden_stages", [])) & starts, "forbidden_stage"
                        if step.get("prior_source"):
                            assert first_sources[key] in admitted, "prior_fact_source_retained"
                        if data.get("run_id"):
                            history = client.get(f"/api/threads/{thread}/messages", headers=headers).json()
                            assert history[-1]["content"] == answer, "display_persistence_consistency"
                            first_sources.setdefault(key, history[0]["id"])
                            events = client.get(f'/api/runs/{data["run_id"]}/events', headers=headers).json()
                            finals = [e["payload"] for e in events if e["type"] == "answer.final"]
                            assert len(finals) == 1 and finals[0]["response"] == answer, "authoritative_final"
                        suggestions = client.get("/api/profile-suggestions", headers=headers, params={"thread_id": thread}).json()
                        if step.get("no_suggestions"):
                            assert not suggestions["items"], "no_other_patient_suggestions"
                        if step.get("profile_absent"):
                            profile = client.get(f"/api/profiles/{uid}", headers=headers).json()
                            assert all(t not in json.dumps(profile, ensure_ascii=False) for t in step["profile_absent"]), "no_implicit_profile_write"
                        assert not db.one("SELECT 1 FROM profile_memories WHERE user_id=? AND thread_id=?", (uid, thread)), "no_automatic_summary_memory"
                        record["checks"].append({"op": op, "thread": key, "passed": True,
                            "intent": data.get("intent_type"), "captured_inputs": len(contexts),
                            "required_fact_count": len(step.get("required_context", [])),
                            "fact_coverage": 1 if step.get("required_context") else None})
                        continue
                    record["checks"].append({"op": op, "passed": True})
                record["passed"] = True
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                # Only fixed assertion codes, never exception text containing a request.
                if isinstance(exc, AssertionError):
                    record["failure_check"] = str(exc)
            details.append(record)
    return {"status": "completed", "count": len(details), "unique_families": len({c["family"] for c in cases}),
            "passed": sum(r["passed"] for r in details), "modes": list(modes), "details": details,
            "scope": "synthetic_business_contracts_not_clinical_effectiveness"}
