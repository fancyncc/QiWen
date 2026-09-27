"""Classics: BM25 + vectors -> RRF -> CrossEncoder -> sigmoid gate -> MMR."""
from __future__ import annotations

import asyncio
import math
import os
import re
import time

from . import local_ranking, model, retrieval


def bounded_int(name, default, low, high):
    try:
        return min(high, max(low, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def fallback_admission(candidates: list[dict], plan: dict):
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
        terms = [term for term in plan["lexical_terms"] if term in row["body"]]
        similarity = row["retrieval"].get("vector_similarity")
        if terms or (threshold is not None and similarity is not None and similarity >= threshold):
            selected.append({**row, "retrieval": {**row["retrieval"], "admitted_by": "literal" if terms else "calibrated_vector",
                                                   "matched_terms": terms}})
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


def scoring_text(row, plan):
    return passage_excerpt({**row, "retrieval": {"matched_terms": plan["lexical_terms"]}})


def current_sources(rows):
    current = {row["id"]: row for row in retrieval.all_rows("SELECT id,title,chapter,body FROM corpus")}
    return [row for row in rows if row["id"] in current and
            all(row[key] == current[row["id"]][key] for key in ("title", "chapter", "body"))]


async def retrieve_classics(plan: dict, limit=4) -> tuple[list[dict], dict]:
    recall = bounded_int("QIWEN_RAG_RECALL_K", 30, 4, 100)
    pool = bounded_int("QIWEN_RAG_CANDIDATES", 30, 4, 40)
    candidates, info = await asyncio.to_thread(retrieval.retrieve_candidates, plan, recall, pool)
    mode = os.getenv("QIWEN_RAG_RERANK_MODE", "cross_encoder")
    details = {"requested_mode": mode, "status": "no_candidates", "fallback_reason": "",
               "input_count": len(candidates), "output_count": 0, "activation": "sigmoid",
               "threshold": local_ranking.MIN_RELEVANCE, "ranked": []}
    mmr = {"status": "not_needed", "input_count": 0, "output_count": 0, "duplicate_count": 0}
    selected, config = [], None
    warnings = [info["warning"]] if info["warning"] else []
    if candidates and mode == "none":
        details["status"] = "disabled"
        selected = fallback_admission(candidates, plan)
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
            texts = [scoring_text(row, plan) for row in candidates]
            logits = await local_ranking.bounded_worker(local_ranking.score_logits, plan["semantic_query"], texts,
                                                        config, timeout=config.timeout)
            if not isinstance(logits, list) or len(logits) != len(candidates):
                raise ValueError("CrossEncoder candidate count mismatch")
            ranked = []
            for row, text, logit in zip(candidates, texts, logits):
                score = local_ranking.sigmoid(logit)
                ranked.append({**row, "retrieval": {**row["retrieval"], "rerank_logit": logit, "rerank_score": score,
                                                     "admitted_by": "cross_encoder", "evidence": text}})
            ranked.sort(key=lambda row: (-row["retrieval"]["rerank_score"], row["retrieval"]["rrf_rank"], row["id"]))
            details["ranked"] = [{"id": row["id"], "logit": row["retrieval"]["rerank_logit"],
                                   "score": row["retrieval"]["rerank_score"],
                                   "accepted": row["retrieval"]["rerank_score"] >= local_ranking.MIN_RELEVANCE}
                                  for row in ranked]
            selected = [row for row in ranked if row["retrieval"]["rerank_score"] >= local_ranking.MIN_RELEVANCE]
            details["status"] = record["status"] = "success"
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            selected = []
            record["status"] = "failed"
            status = "model_unavailable" if isinstance(exc, local_ranking.ModelUnavailable) else "failed"
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
                                                        timeout=config.timeout)
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
    details["output_count"] = mmr["output_count"] = len(selected)
    return selected, {**info, "query_plan": plan, "rerank": details, "mmr": mmr,
                      "warning": " ".join(warnings),
                      "selected": [{"id": row["id"], "title": row["title"], "chapter": row["chapter"],
                                    **row["retrieval"]} for row in selected],
                      "admitted_count": admitted_count, "stale_count": stale_count,
                      "truncated_count": max(0, mmr_pool_count - max(0, limit)),
                      "filtered_count": len(candidates) - admitted_count}
