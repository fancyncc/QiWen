"""Explicit, isolated evaluation profiles and credential-free run manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("mock_contract", "live_core", "live_full")
BEHAVIOR_VERSION = "session-profile-boundary-v6-optional-judge"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest(profile: str) -> dict:
    # Never enumerate environment values: only explicitly public settings are recorded.
    public = ("QIWEN_MODEL_NAME", "QIWEN_MODEL_MODE", "QIWEN_RAG_MODE", "QIWEN_RAG_RERANK_MODE",
              "QIWEN_EMBEDDING_MODEL", "QIWEN_CROSS_ENCODER_MODEL", "QIWEN_CROSS_ENCODER_REVISION",
              "QIWEN_MMR_EMBEDDING_MODEL", "QIWEN_MMR_EMBEDDING_REVISION", "QIWEN_RAG_MMR_LAMBDA",
              "QIWEN_MODEL_CONTEXT_LIMIT", "QIWEN_MODEL_OUTPUT_RESERVE", "QIWEN_JUDGE_MODEL")
    files = {}
    for folder, patterns in (("backend/app", ("**/*.py",)), ("backend/evaluation", ("*.py", "*.json", "datasets/*.jsonl")),
                             ("backend/data_pipeline", ("**/*.py",)),
                             ("backend/resources", ("**/*.json", "classics_source/*.txt")),
                             ("frontend/src", ("**/*.ts", "**/*.tsx", "**/*.css"))):
        for pattern in patterns:
            for path in sorted((ROOT / folder).glob(pattern)):
                files[path.relative_to(ROOT).as_posix()] = digest(path.read_bytes())
    for name in ("docker-compose.yml", "compose.yml", "compose.yaml", "backend/Dockerfile", "frontend/Dockerfile",
                 "backend/requirements.lock.txt", "backend/requirements.rerank.txt", "frontend/package-lock.json",
                 "frontend/next.config.ts", ".github/workflows/ci.yml"):
        path = ROOT / name
        if path.is_file():
            files[name] = digest(path.read_bytes())
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unknown", None
    roles = ("inquiry", "knowledge", "safety_preflight", "profile_extraction", "query_rewrite",
             "bianzheng", "formula", "herb", "synthesis", "reviewer")
    public = (*public, "QIWEN_RAG_RECALL_K", "QIWEN_RAG_CANDIDATES", "QIWEN_CROSS_ENCODER_MAX_LENGTH",
              "QIWEN_RAG_TOTAL_TIMEOUT", "QIWEN_RAG_WARMUP",
              "QIWEN_CROSS_ENCODER_BATCH_SIZE", "QIWEN_LOCAL_MODEL_THREADS", "QIWEN_RAG_MMR_EMBEDDING_BACKEND",
              "QIWEN_CROSS_ENCODER_TIMEOUT", "QIWEN_MODEL_TIMEOUT", "QIWEN_MODEL_RETRIES",
              *("QIWEN_MODEL_" + role.upper() + suffix for role in roles for suffix in
                ("_TIMEOUT", "_RETRIES", "_CONTEXT_LIMIT", "_OUTPUT_RESERVE")))
    return {"run_id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
            "behavior_version": BEHAVIOR_VERSION, "profile": profile, "git_revision": head,
            "dirty": dirty, "files": files, "workspace_sha256": digest(json.dumps(files, sort_keys=True).encode()),
            "public_settings": {name: os.getenv(name) for name in public},
            "provider_fingerprints": {kind: digest(os.getenv(name, "").rstrip("/").removesuffix("/chat/completions").encode())
                                      if os.getenv(name) else None for kind, name in (
                                          ("application", "QIWEN_MODEL_BASE_URL"), ("judge", "QIWEN_JUDGE_BASE_URL"))},
            "roles": {role: os.getenv("QIWEN_MODEL_" + role.upper()) or os.getenv("QIWEN_MODEL_NAME") for role in roles},
            "python": platform.python_version(), "platform": platform.system(),
            "data_policy": "isolated_synthetic_only"}


def close_evaluation_indexes(directory: Path) -> None:
    """Release only this run's Chroma handles before Windows removes its temp DB.

    Chroma 1.x shares persistent systems in a process registry. Do not clear that
    registry globally: unrelated indexes must remain usable.
    """
    import sys
    module = sys.modules.get("chromadb.api.shared_system_client")
    if module is None:
        return
    cls = module.SharedSystemClient
    expected = (directory / "chroma").resolve()
    for identifier, system in list(cls._identifier_to_system.items()):
        if Path(system.settings.persist_directory).resolve() == expected:
            system.stop()
            cls._identifier_to_system.pop(identifier, None)
            getattr(cls, "_identifier_to_refcount", {}).pop(identifier, None)


@contextmanager
def environment(profile: str):
    """Call before importing app; configuration is intentionally immutable thereafter."""
    if profile not in PROFILES:
        raise ValueError("Unknown evaluation profile")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    original = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="qiwen-eval-") as directory:
        try:
            os.environ["QIWEN_DATA_DIR"] = directory
            if profile == "mock_contract":
                os.environ.update(QIWEN_MODEL_MODE="mock", QIWEN_RAG_MODE="bm25", QIWEN_RAG_RERANK_MODE="none",
                                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            else:
                if not os.getenv("QIWEN_MODEL_NAME", "").strip():
                    os.environ["QIWEN_MODEL_NAME"] = "qwen-plus"
                os.environ["QIWEN_MODEL_MODE"] = "openai_compatible"
                os.environ["QIWEN_RAG_MODE"] = "hybrid" if profile == "live_full" else "bm25"
                os.environ["QIWEN_RAG_RERANK_MODE"] = "cross_encoder" if profile == "live_full" else "none"
                # Share downloaded public weights, never the business DB/index.
                os.environ.setdefault("QIWEN_LOCAL_MODEL_CACHE", str(ROOT / ".data/models"))
            import sys
            config = sys.modules.get("app.config")
            if config and str(config.DATA_DIR) != str(Path(directory).resolve()):
                raise RuntimeError("Evaluation requires a fresh process before importing app")
            yield Path(directory)
        finally:
            try:
                close_evaluation_indexes(Path(directory))
            finally:
                os.environ.clear()
                os.environ.update(original)


def dependencies(profile: str, *, judge_required: bool = False) -> dict:
    if profile == "mock_contract":
        return {"status": "ready", "checks": {"network": {"status": "intentionally_disabled"}}}
    from .benchmark import probe_model
    base, key, name = (os.getenv(n, "").strip() for n in
                       ("QIWEN_MODEL_BASE_URL", "QIWEN_MODEL_API_KEY", "QIWEN_MODEL_NAME"))
    required = ["QIWEN_MODEL_BASE_URL", "QIWEN_MODEL_API_KEY", "QIWEN_MODEL_NAME"]
    if judge_required:
        required += ["QIWEN_JUDGE_BASE_URL", "QIWEN_JUDGE_API_KEY", "QIWEN_JUDGE_MODEL"]
    if profile == "live_full":
        required += ["QIWEN_EMBEDDING_MODEL"]
    missing = [n for n in required if not os.getenv(n, "").strip()]
    if missing:
        return {"status": "blocked", "checks": {"configuration": {"status": "not_configured", "missing": missing}}}
    checks, probes = {}, {}
    for role in ("inquiry", "knowledge", "safety_preflight", "profile_extraction", "query_rewrite",
                 "bianzheng", "formula", "herb", "synthesis", "reviewer"):
        role_name = os.getenv("QIWEN_MODEL_" + role.upper(), "").strip() or name
        if role_name not in probes:
            probes[role_name] = probe_model(base, key, role_name)
        checks[role] = {**probes[role_name], "scope": "connection_probe; structured_and_tool_capability_checked_by_actual_tasks"}
    if judge_required:
        jb, jk, jn = (os.getenv(n, "").strip() for n in ("QIWEN_JUDGE_BASE_URL", "QIWEN_JUDGE_API_KEY", "QIWEN_JUDGE_MODEL"))
        roles = {os.getenv("QIWEN_MODEL_" + role.upper(), "").strip() or name for role in checks}
        if jb.rstrip("/").removesuffix("/chat/completions") == base.rstrip("/").removesuffix("/chat/completions") and jn in roles:
            checks["judge"] = {"available": False, "status": "independent_judge_required"}
        else:
            checks["judge"] = probe_model(jb, jk, jn)
    if all(item.get("available") for item in checks.values()) and profile == "live_full":
        try:
            import asyncio
            from app.db import init_db, all_rows
            from app.retrieval import index_vectors, _embed, _collection
            from app.classics import retrieve_classics
            from app.local_ranking import warm_models
            init_db()
            corpus = all_rows("SELECT id,title,chapter,body,metadata FROM corpus")
            index_vectors(corpus)
            vector = _embed(["桂枝汤"])
            checks["embedding"] = {"available": bool(vector), "status": "ok" if vector else "empty"}
            checks["index"] = {"available": _collection().count() == len(corpus), "count": _collection().count()}
            warmup = asyncio.run(warm_models())
            checks["local_models"] = {"available": warmup.get("status") == "available", "status": warmup.get("status")}
            docs, trace = asyncio.run(retrieve_classics(query_plan("桂枝汤")))
            checks["cross_encoder"] = {"available": trace["rerank"]["status"] == "success", "status": trace["rerank"]["status"]}
            checks["mmr"] = {"available": trace["mmr"]["status"] in {"success", "not_needed"}, "status": trace["mmr"]["status"]}
            checks["final_sources"] = {"available": bool(docs), "count": len(docs)}
        except Exception as exc:
            checks["retrieval_dependencies"] = {"available": False, "status": "error", "error_type": type(exc).__name__}
    return {"status": "ready" if all(item.get("available") for item in checks.values()) else "blocked", "checks": checks}


def query_plan(question: str) -> dict:
    from app.query_plan import knowledge_plan
    return knowledge_plan(question)


def write_report(folder: Path, prefix: str, report: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (prefix + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def gate(results: dict, selected: list[str]) -> dict:
    failures, blocked = [], []
    retrieval_gate = {"status": "not_run", "scope": "final_evidence_effectiveness_not_clinical_validation"}
    for name in selected:
        row = results.get(name, {})
        if row.get("status") == "failed":
            failures.append(name + ":execution_failed")
            continue
        if row.get("status") != "completed":
            blocked.append(name)
            continue
        if row.get("count") == 0:
            blocked.append(name + ":empty")
            continue
        if row.get("failed", 0):
            failures.append(name + ":execution_or_scoring_failed")
        if name == "router" and row.get("accuracy") != 1:
            failures.append("router:behavior_mismatch")
        if name == "safety":
            if not row.get("rule", {}).get("count") or not row.get("e2e", {}).get("count"):
                blocked.append("safety:empty")
                continue
            if row.get("rule", {}).get("accuracy") != 1 or row.get("e2e", {}).get("warning_rate") != 1:
                failures.append("safety:boundary_failure")
        if name == "scenarios" and row.get("passed") != row.get("count"):
            failures.append("scenarios:contract_failure")
        if name == "quality" and row.get("safety_failures", 0):
            failures.append("quality:safety_failure")
        if name == "faithfulness" and row.get("invalid_sources", 0):
            failures.append("faithfulness:invalid_sources")
        if name == "rag" and row.get("invalid_sources", 0):
            failures.append("rag:invalid_sources")
        if name == "rag":
            evidence = row.get("evidence_contract", {})
            if not evidence or evidence.get("unlabeled", 0):
                blocked.append("rag:evidence_labels_incomplete")
            if evidence.get("positive_count") and evidence.get("empty_positive_count") == evidence["positive_count"]:
                failures.append("rag:all_positive_evidence_empty")
            if evidence.get("unexpected_evidence_count", 0):
                failures.append("rag:unexpected_evidence")
            if not evidence.get("positive_count") or not evidence.get("negative_count"):
                blocked.append("rag:positive_and_negative_cases_required")
            retrieval_gate.update(status="failed" if evidence.get("failed", 0) else "blocked" if
                                  not evidence or evidence.get("unlabeled", 0) or not evidence.get("positive_count") or not evidence.get("negative_count") else "passed",
                                  counts=evidence)
    status = "failed" if failures else "blocked" if blocked else "passed"
    return {"status": status, "failures": failures, "incomplete": blocked,
            "scope": "execution_contracts_and_nonvacuous_evidence_not_clinical_validation",
            "retrieval_effectiveness": retrieval_gate}


def exit_code(status: str) -> int:
    return 0 if status == "passed" else 2 if status == "blocked" else 1


def profile_argument(parser: argparse.ArgumentParser):
    parser.add_argument("--profile", choices=PROFILES, default="mock_contract")
