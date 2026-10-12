"""Classics: BM25 + vectors -> RRF -> CrossEncoder -> sigmoid gate -> MMR."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from copy import deepcopy

from . import local_ranking, model, retrieval
from .blocking import deadline, remaining_timeout, run_blocking
from .retrieval_budget import invocation_deadline, timed_stage, timings, snapshot, progress_trace
from .retrieval_terms import content_check, matching_terms, normalize_search

last_observation = {}


def bounded_int(name, default, low, high):
    try:
        return min(high, max(low, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def fallback_admission(candidates: list[dict], plan: dict, *, audit=None):
    """Legacy explicitly-disabled mode only; never bypass a failed CE gate."""
    raw = os.getenv("QIWEN_RAG_MIN_SIMILARITY", "").strip()
    threshold = None
    if raw:
        try:
            threshold = float(raw)
            if not math.isfinite(threshold) or not -1 <= threshold <= 1:
                threshold = None
        except ValueError:
            pass
    selected = []
    for row in candidates:
        window = scoring_window(row, plan)
        terms = matching_terms(window["quote"], plan["lexical_terms"])
        checked = content_check(window["quote"], plan, heading=row.get("chapter", ""))
        similarity = row["retrieval"].get("vector_similarity")
        accepted = bool(checked["passed"] and (terms or (threshold is not None and similarity is not None and similarity >= threshold)))
        if audit is not None:
            audit.append({"id": row["id"], "accepted": accepted, "content_check": checked,
                          "rejection_reason": "" if accepted else checked["reason"] or "no_content_match"})
        if accepted:
            selected.append({**row, "retrieval": {**row["retrieval"], "admitted_by": "literal" if terms else "calibrated_vector",
                                                   "matched_terms": terms, "evidence": window["quote"],
                                                   "content_check": checked,
                                                   "window": {k: v for k, v in window.items() if k not in {"text", "quote"}}}})
    return selected


def passage_excerpt(row: dict, limit=500) -> str:
    meta = row.get("retrieval", {})
    evidence = meta.get("evidence", "")
    if evidence and evidence in row["body"]:
        return evidence[:limit]
    terms = meta.get("matched_terms", [])
    offsets = [row["body"].find(term) for term in terms if term in row["body"]]
    start = max(0, min(offsets) - 40) if offsets else 0
    return row["body"][start:start + limit]


def scoring_window(row, plan, limit=380):
    """Select a contiguous sentence window; retain exact source offsets for audit."""
    body = row["body"]
    spans = [(match.start(), match.end()) for match in re.finditer(r"[^。！？!?\n]+[。！？!?\n]*", body)]
    if not spans:
        spans = [(0, len(body))]
    terms = plan.get("lexical_terms", [])
    options = []
    for i, (start, _) in enumerate(spans):
        for j in range(i, len(spans)):
            if j > i and re.search(r"\n[ \t]*\n", body[spans[j - 1][0]:spans[j - 1][1]]):
                break
            end = spans[j][1]
            if end - start > limit and j > i:
                break
            text = body[start:end]
            hits = matching_terms(text, terms)
            checked = content_check(text, plan, heading=row.get("chapter", ""))
            group_hits = sum(bool(matching_terms(text, group)) for group in plan.get("required_term_groups", []))
            # Choose the smallest complete span covering the most query terms.
            # Filling the remaining budget with unrelated prescriptions hurts scoring.
            options.append((int(checked["passed"]), group_hits, len(hits), -(end - start), -start, i, j, hits))
    _, group_hits, _, _, _, first, last, hits = max(options)
    if not hits and not group_hits and not plan.get("action_polarity"):
        first = last = 0
        while last + 1 < len(spans) and spans[last + 1][1] <= limit:
            last += 1
    else:
        # Keep one neighboring sentence in the same paragraph for qualification.
        if first > 0 and not re.search(r"\n[ \t]*\n", body[spans[first - 1][0]:spans[first][0]]):
            if spans[last][1] - spans[first - 1][0] <= limit:
                first -= 1
        if last + 1 < len(spans) and not re.search(r"\n[ \t]*\n", body[spans[last][0]:spans[last][1]]):
            if spans[last + 1][1] - spans[first][0] <= limit:
                last += 1
    start, end = spans[first][0], spans[last][1]
    complete = end - start <= limit
    if not complete:
        positions = [normalize_search(body).find(normalize_search(term), start, end) for term in hits]
        anchor = min(positions) if positions else start
        start = max(start, anchor - 60)
        end = min(end, start + limit)
    quote = body[start:end]
    heading = " · ".join([row["title"], row["chapter"]])[:100]
    return {"quote": quote, "text": f"{heading}\n{quote}", "start": start, "end": end,
            "matched_terms": matching_terms(quote, terms), "complete_sentences": complete,
            "selection": "lexical_sentence_window" if hits else "semantic_fallback_first_window",
            "body_sha256": hashlib.sha256(body.encode()).hexdigest()}


def scoring_text(row, plan):
    return scoring_window(row, plan)["text"]


def current_sources(rows):
    from .evidence_sources import source_lookup
    current = source_lookup(retrieval.all_rows("SELECT id,title,chapter,body FROM corpus"))
    return [row for row in rows if row["id"] in current and
            all(row[key] == current[row["id"]][key] for key in ("title", "chapter", "body"))]


async def retrieve_classics(plan: dict, limit=4) -> tuple[list[dict], dict]:
    started = time.monotonic()
    end = invocation_deadline()
    context = model.invocation.get()
    key = retrieval_cache_key(plan, limit)
    if context and key in context.retrieval_cache:
        rows, trace = deepcopy(context.retrieval_cache[key])
        checked = current_sources(rows)
        trace.update(cached=True, retryable=False, elapsed_ms=0)
        trace["selected"] = [s for s in trace["selected"] if s["id"] in {r["id"] for r in checked}]
        trace["stale_count"] = trace.get("stale_count", 0) + len(rows) - len(checked)
        trace["rerank"]["output_count"] = trace["mmr"]["output_count"] = len(checked)
        if rows and not checked:
            trace.update(outcome="source_changed", warning="资料已更新，本轮原证据不再引用。")
        return checked, trace
    stage_data = {}
    timing_token = timings.set(stage_data)
    partial = {}
    progress_token = progress_trace.set(partial)
    deadline_token = deadline.set(end)
    try:
        if context and context.retrieval_terminal:
            raise TimeoutError() if context.retrieval_terminal == "timeout" else RetrievalStopped(context.retrieval_terminal)
        result = await asyncio.wait_for(_retrieve_classics(plan, limit), timeout=remaining_timeout(300))
        rows, trace = result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        outcome = "timeout" if isinstance(exc, TimeoutError) else str(exc) if isinstance(exc, RetrievalStopped) else "dependency_error"
        rows = []
        trace = {**deepcopy(partial), "query_plan": plan, "requested_mode": retrieval.RAG_MODE,
                 "active_mode": partial.get("active_mode", "unavailable"),
                 "outcome": outcome, "error_type": type(exc).__name__,
                 "warning": "古籍检索超时或依赖不可用，本轮未取得可引用证据。",
                 "rerank": {**partial.get("rerank", {}), "status": "not_completed"},
                 "mmr": {**partial.get("mmr", {}), "status": "not_completed"}, "selected": []}
    finally:
        deadline.reset(deadline_token)
        timings.reset(timing_token)
        progress_trace.reset(progress_token)
    elapsed_ms = round((time.monotonic() - started) * 1000)
    remaining_ms = max(0, round((end - time.monotonic()) * 1000))
    # Do not start another similarly-sized pass with only the tail of its budget.
    retry_minimum_ms = max(50, round(elapsed_ms * 1.1))
    trace.update(elapsed_ms=elapsed_ms, remaining_ms=remaining_ms, timings=snapshot(stage_data),
                 retryable=trace["outcome"] == "no_match" and remaining_ms >= retry_minimum_ms,
                 retry_minimum_ms=retry_minimum_ms)
    for item in trace["timings"].values():
        if item["status"] == "running":
            item["status"] = "timeout" if trace["outcome"] == "timeout" else "detached"
    if context:
        if trace["outcome"] in {"timeout", "dependency_error", "model_unavailable", "index_empty", "corpus_empty"}:
            context.retrieval_terminal = trace["outcome"]
        context.retrieval_cache[key] = deepcopy((rows, trace))
    last_observation.clear()
    last_observation.update(observed_at=time.time(), outcome=trace["outcome"],
                            vector_status=trace.get("vector_status", "unknown"),
                            rerank_status=trace["rerank"]["status"], mmr_status=trace["mmr"]["status"],
                            elapsed_ms=trace["elapsed_ms"])
    return rows, trace


class RetrievalStopped(RuntimeError):
    pass


def retrieval_cache_key(plan, limit=4):
    # attempt is observational only; equivalent plans reuse one invocation's result.
    behavior = {key: value for key, value in plan.items() if key != "attempt"}
    return (json.dumps(behavior, ensure_ascii=False, sort_keys=True), limit, "sentence_window_v3_content")


async def _retrieve_classics(plan: dict, limit=4) -> tuple[list[dict], dict]:
    recall = bounded_int("QIWEN_RAG_RECALL_K", 30, 4, 100)
    mode = os.getenv("QIWEN_RAG_RERANK_MODE", "cross_encoder")
    pool_default = 12 if mode == "cross_encoder" and os.getenv("QIWEN_CROSS_ENCODER_DEVICE", "cpu") == "cpu" else 30
    pool = bounded_int("QIWEN_RAG_CANDIDATES", pool_default, 4, 40)
    with timed_stage("candidate_retrieval"):
        candidates, info = await run_blocking(retrieval.retrieve_candidates, plan, recall, pool)
    details = {"requested_mode": mode, "status": "no_candidates", "fallback_reason": "",
               "input_count": len(candidates), "output_count": 0, "activation": "sigmoid",
               "threshold": local_ranking.MIN_RELEVANCE, "ranked": []}
    mmr = {"status": "not_needed", "input_count": 0, "output_count": 0, "duplicate_count": 0}
    partial = progress_trace.get()
    if partial is not None:
        partial.update(info, candidate_pool_limit=pool, rerank=details, mmr=mmr)
    selected, config = [], None
    warnings = [info["warning"]] if info["warning"] else []
    if candidates and mode == "none":
        details["status"] = "disabled"
        details["admission_audit"] = []
        selected = fallback_admission(candidates, plan, audit=details["admission_audit"])
        warnings.append("CrossEncoder 已显式关闭，本轮字面检索结果未经过 0.7 门槛或 MMR。")
    elif candidates:
        context = model.invocation.get()
        record = {"role": "cross_encoder", "node": model.current_node.get(), "model": "",
                  "input_tokens": None, "output_tokens": None, "elapsed_ms": 0, "retries": 0,
                  "status": "running", "attempts": []}
        started = time.monotonic()
        if context:
            context.check()
            context.calls.append(record)
        try:
            if mode != "cross_encoder":
                raise ValueError("Use cross_encoder for classics reranking")
            config = local_ranking.settings()
            record["model"] = config.model
            details.update(model=config.model, revision=config.revision, max_length=config.max_length)
            windows = [scoring_window(row, plan) for row in candidates]
            details["input_windows"] = [{"id": row["id"], "title": row["title"], "chapter": row["chapter"],
                                          **{k: v for k, v in window.items() if k != "text"}}
                                         for row, window in zip(candidates, windows)]
            texts = [window["text"] for window in windows]
            token_diagnostics = []
            audit_token = local_ranking.score_audit.set(token_diagnostics)
            try:
                logits = await local_ranking.bounded_worker(local_ranking.score_logits,
                                                            plan.get("rerank_query", plan["semantic_query"]), texts,
                                                            config, timeout=remaining_timeout(config.timeout))
            finally:
                local_ranking.score_audit.reset(audit_token)
            if not isinstance(logits, list) or len(logits) != len(candidates):
                raise ValueError("CrossEncoder candidate count mismatch")
            ranked = []
            for index, (row, window, logit) in enumerate(zip(candidates, windows, logits)):
                score = local_ranking.sigmoid(logit)
                checked = content_check(window["quote"], plan, heading=row.get("chapter", ""))
                ranked.append({**row, "retrieval": {**row["retrieval"], "rerank_logit": logit, "rerank_score": score,
                                                     "content_check": checked,
                                                     "admitted_by": "cross_encoder", "evidence": window["quote"],
                                                     "window": {k: v for k, v in window.items() if k not in {"text", "quote"}},
                                                     "token_diagnostic": token_diagnostics[index] if index < len(token_diagnostics)
                                                     else {"status": "unavailable"}}})
            ranked.sort(key=lambda row: (-row["retrieval"]["rerank_score"], row["retrieval"]["rrf_rank"], row["id"]))
            details["ranked"] = [{"id": row["id"], "logit": row["retrieval"]["rerank_logit"],
                                   "score": row["retrieval"]["rerank_score"],
                                   "evidence": row["retrieval"]["evidence"],
                                   "window": row["retrieval"]["window"],
                                   "token_diagnostic": row["retrieval"]["token_diagnostic"],
                                   "content_check": row["retrieval"]["content_check"],
                                   "rejection_reason": row["retrieval"]["content_check"]["reason"] or
                                       ("" if row["retrieval"]["rerank_score"] >= local_ranking.MIN_RELEVANCE else "below_relevance_threshold"),
                                   "accepted": row["retrieval"]["content_check"]["passed"] and row["retrieval"]["rerank_score"] >= local_ranking.MIN_RELEVANCE}
                                  for row in ranked]
            selected = [row for row in ranked if row["retrieval"]["content_check"]["passed"] and
                        row["retrieval"]["rerank_score"] >= local_ranking.MIN_RELEVANCE]
            details["status"] = record["status"] = "success"
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            selected = []
            record["status"] = "failed"
            status = ("model_unavailable" if isinstance(exc, local_ranking.ModelUnavailable) else
                      "timeout" if isinstance(exc, TimeoutError) else "failed")
            details.update(status=status, fallback_reason=f"CrossEncoder 不可用或重排失败：{type(exc).__name__}")
            warnings.append(details["fallback_reason"] + "；本轮不引用未通过 0.7 门槛的片段。")
        finally:
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    admitted_count = len(selected)
    selected = current_sources(selected)
    stale_count = admitted_count - len(selected)
    unique, seen = [], set()
    for row in selected:
        key = re.sub(r"\s+", " ", row["body"]).strip()
        if key not in seen:
            unique.append(row)
            seen.add(key)
    mmr.update(input_count=len(unique), duplicate_count=len(selected) - len(unique))
    selected = unique
    mmr_pool_count = len(selected)
    if mode == "cross_encoder" and len(selected) > 1 and limit > 0:
        try:
            vectors = await local_ranking.bounded_worker(local_ranking.candidate_embeddings, selected, config,
                                                        timeout=max(.001, remaining_timeout(config.timeout) - .05))
            selected = local_ranking.mmr_select(selected, vectors, limit, config.mmr_weight)
            mmr.update(status="success", lambda_weight=config.mmr_weight, similarity="candidate_cosine",
                       embedding_backend=config.embedding_backend, model=(config.embedding_model if config.embedding_backend == "local"
                                                                        else os.getenv("QIWEN_EMBEDDING_MODEL", "")))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            mmr.update(status="failed", reason=type(exc).__name__)
            warnings.append("候选之间的语义相似度暂不可用，MMR 未执行；本轮仅按已通过 0.7 门槛的相关性排序取用。")
            selected = selected[:max(0, limit)]
    else:
        selected = selected[:max(0, limit)]
    checked = current_sources(selected)
    stale_count += len(selected) - len(checked)
    selected = checked
    selected_ids = {row["id"] for row in selected}
    for item in details["ranked"]:
        item["selected"] = item["id"] in selected_ids
    details["output_count"] = mmr["output_count"] = len(selected)
    outcome = ("success" if selected else "corpus_empty" if not info["corpus_count"] else
               "model_unavailable" if details["status"] == "model_unavailable" else
               "timeout" if details["status"] == "timeout" else
               "dependency_error" if details["status"] == "failed" else
               info.get("dependency_status", "ok") if info.get("dependency_status", "ok") != "ok" else
               "source_changed" if stale_count else "no_match")
    return selected, {**info, "query_plan": plan, "rerank": details, "mmr": mmr,
                      "candidate_pool_limit": pool,
                      "outcome": outcome,
                      "warning": " ".join(warnings),
                      "selected": [{"id": row["id"], "title": row["title"], "chapter": row["chapter"],
                                    **row["retrieval"]} for row in selected],
                      "admitted_count": admitted_count, "stale_count": stale_count,
                      "truncated_count": max(0, mmr_pool_count - max(0, limit)),
                      "filtered_count": len(candidates) - admitted_count}
