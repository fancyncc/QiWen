"""Local CrossEncoder relevance and independent candidate embeddings for MMR."""
from __future__ import annotations

import asyncio
import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

from .config import DATA_DIR
from .retrieval import validate_vectors
from .blocking import check_cancelled, remaining_timeout, deadline
from .retrieval_budget import timed_stage

MIN_RELEVANCE = 0.7
score_audit = ContextVar("cross_encoder_score_audit", default=None)
_inference_lock = threading.RLock()
_worker_slots = threading.BoundedSemaphore(4)


class ModelUnavailable(RuntimeError):
    pass


class InferenceBusy(RuntimeError):
    pass


def integer(name, default, low, high):
    value = int(os.getenv(name, str(default)))
    if not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return value


def finite_number(name, default, low, high):
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return value


@dataclass(frozen=True)
class Settings:
    model: str
    revision: str | None
    embedding_model: str
    embedding_revision: str | None
    device: str
    cache: str
    max_length: int
    batch_size: int
    threads: int
    timeout: float
    mmr_weight: float
    embedding_backend: str


def settings():
    return Settings(
        model=os.getenv("QIWEN_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
        revision=os.getenv("QIWEN_CROSS_ENCODER_REVISION", "").strip() or None,
        embedding_model=os.getenv("QIWEN_MMR_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip(),
        embedding_revision=os.getenv("QIWEN_MMR_EMBEDDING_REVISION", "").strip() or None,
        device=os.getenv("QIWEN_CROSS_ENCODER_DEVICE", "cpu"),
        cache=str(Path(os.getenv("QIWEN_LOCAL_MODEL_CACHE", "").strip() or str(DATA_DIR / "models")).resolve()),
        max_length=integer("QIWEN_CROSS_ENCODER_MAX_LENGTH", 512, 32, 8192),
        batch_size=integer("QIWEN_CROSS_ENCODER_BATCH_SIZE", 2, 1, 32),
        threads=integer("QIWEN_LOCAL_MODEL_THREADS", min(4, os.cpu_count() or 2), 1, 16),
        timeout=finite_number("QIWEN_CROSS_ENCODER_TIMEOUT", 120, 1, 600),
        mmr_weight=finite_number("QIWEN_RAG_MMR_LAMBDA", 0.7, 0, 1),
        embedding_backend=os.getenv("QIWEN_RAG_MMR_EMBEDDING_BACKEND", "local"),
    )


def sigmoid(logit):
    if isinstance(logit, bool) or not isinstance(logit, (float, int)) or not math.isfinite(logit):
        raise ValueError("CrossEncoder must return finite scalar logits")
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1 + exp)


def check_stop(stop):
    if stop.is_set():
        raise asyncio.CancelledError()
    check_cancelled()


@contextmanager
def inference_slot(stop, name):
    acquired = False
    try:
        with timed_stage(name + "_queue"):
            while not acquired:
                check_stop(stop)
                acquired = _inference_lock.acquire(timeout=min(.05, remaining_timeout(.05)))
        check_stop(stop)
        yield
    finally:
        if acquired:
            _inference_lock.release()


@lru_cache(maxsize=2)
def _load(kind, config: Settings, local_only: bool):
    try:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        torch.set_num_threads(config.threads)
        Path(config.cache).mkdir(parents=True, exist_ok=True)
        common = dict(device=config.device, cache_folder=config.cache, trust_remote_code=False,
                      local_files_only=local_only)
        if kind == "cross_encoder":
            if not config.model:
                raise ModelUnavailable("CrossEncoder model not configured")
            encoder = CrossEncoder(config.model, revision=config.revision, max_length=config.max_length,
                                   activation_fn=torch.nn.Identity(), **common)
            if encoder.num_labels != 1:
                raise ValueError("CrossEncoder requires a single logit per pair")
            return encoder
        if not config.embedding_model:
            raise ModelUnavailable("MMR embedding model not configured")
        encoder = SentenceTransformer(config.embedding_model, revision=config.embedding_revision, **common)
        encoder.max_seq_length = config.max_length
        return encoder
    except (ImportError, OSError) as exc:
        raise ModelUnavailable(f"Local {kind} dependencies or weights unavailable") from exc


def score_logits(query: str, texts: list[str], config: Settings, stop: threading.Event):
    """Identity activation here; sigmoid is applied exactly once by the caller."""
    with inference_slot(stop, "cross_encoder"):
        check_stop(stop)
        with timed_stage("cross_encoder_load"):
            encoder = _load("cross_encoder", config, True)
        output = []
        with timed_stage("cross_encoder_inference"):
            for start in range(0, len(texts), config.batch_size):
                check_stop(stop)
                pairs = [(query, text) for text in texts[start:start + config.batch_size]]
                audit = score_audit.get()
                tokenizer = getattr(encoder, "tokenizer", None)
                if audit is not None and tokenizer is not None:
                    for question, passage in pairs:
                        check_stop(stop)
                        count = len(tokenizer(question, passage, truncation=False)["input_ids"])
                        audit.append({"status": "measured", "input_token_count": count,
                                      "max_length": config.max_length, "truncated": count > config.max_length})
                values = encoder.predict(pairs, batch_size=config.batch_size, show_progress_bar=False,
                                         apply_softmax=False, convert_to_numpy=True).tolist()
                if not isinstance(values, list) or len(values) != len(pairs):
                    raise ValueError("CrossEncoder output shape mismatch")
                for value in values:
                    sigmoid(value)
                output.extend(values)
        check_stop(stop)
        return output


def candidate_embeddings(rows: list[dict], config: Settings, stop: threading.Event):
    check_stop(stop)
    if config.embedding_backend == "api":
        from .retrieval import _embed
        with timed_stage("mmr_embedding_api"):
            vectors = _embed([row["body"] for row in rows])
    elif config.embedding_backend == "local":
        with inference_slot(stop, "mmr"):
            check_stop(stop)
            with timed_stage("mmr_model_load"):
                encoder = _load("embedding", config, True)
            vectors = []
            with timed_stage("mmr_inference"):
                for start in range(0, len(rows), config.batch_size):
                    check_stop(stop)
                    batch = [row["body"] for row in rows[start:start + config.batch_size]]
                    vectors.extend(encoder.encode(batch, batch_size=config.batch_size, show_progress_bar=False,
                                                  normalize_embeddings=True, convert_to_numpy=True).tolist())
    else:
        raise ValueError("Invalid MMR embedding backend")
    check_stop(stop)
    return validate_vectors(vectors, len(rows))


async def bounded_worker(function, *args, timeout):
    """Cancellation fences batches; an in-progress tensor operation may finish."""
    stop = threading.Event()
    if not _worker_slots.acquire(blocking=False):
        raise InferenceBusy("Local inference capacity exhausted")
    def execute():
        try:
            check_stop(stop)
            return function(*args, stop)
        finally:
            _worker_slots.release()
    worker = asyncio.create_task(asyncio.to_thread(execute))
    def consume(task):
        try:
            task.exception()
        except BaseException:
            pass
    worker.add_done_callback(consume)
    try:
        return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout)
    finally:
        stop.set()


async def warm_models(timeout=30):
    """Warm cached local weights in the serving process, never download them."""
    import time
    started = time.monotonic()
    token = deadline.set(started + timeout)
    checks = {}
    try:
        config = settings()
        kinds = ["cross_encoder"] + (["embedding"] if config.embedding_backend == "local" else [])
        def load(kind, config, stop):
            with inference_slot(stop, "warmup_" + kind):
                _load(kind, config, True)
                check_stop(stop)
        for kind in kinds:
            await bounded_worker(load, kind, config, timeout=remaining_timeout(timeout))
            checks[kind] = "available"
        return {"status": "available", "checks": checks, "elapsed_ms": round((time.monotonic() - started) * 1000)}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {"status": "unavailable", "checks": checks, "error_type": type(exc).__name__,
                "elapsed_ms": round((time.monotonic() - started) * 1000)}
    finally:
        deadline.reset(token)


def mmr_select(rows: list[dict], vectors: list[list[float]], limit: int, weight: float):
    """CE relevance vs maximum candidate-to-selected cosine, not query cosine."""
    validate_vectors(vectors, len(rows))
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Invalid MMR weight")
    normalized = []
    for vector in vectors:
        scale = max(abs(value) for value in vector)
        scaled = [value / scale for value in vector]
        norm = math.hypot(*scaled)
        normalized.append([value / norm for value in scaled])
    similarities = [[max(0.0, min(1.0, math.fsum(a * b for a, b in zip(left, right))))
                     for right in normalized] for left in normalized]
    remaining, chosen, output = list(range(len(rows))), [], []
    while remaining and len(chosen) < max(0, limit):
        redundancy = {i: max((similarities[i][j] for j in chosen), default=0.0) for i in remaining}
        scores = {i: weight * rows[i]["retrieval"]["rerank_score"] - (1 - weight) * redundancy[i]
                  for i in remaining}
        pick = min(remaining, key=lambda i: (-scores[i] if chosen else -rows[i]["retrieval"]["rerank_score"],
                                             -rows[i]["retrieval"]["rerank_score"],
                                             rows[i]["retrieval"]["rrf_rank"], rows[i]["id"]))
        output.append({**rows[pick], "retrieval": {**rows[pick]["retrieval"], "mmr_rank": len(chosen) + 1,
                                                   "mmr_score": scores[pick], "max_selected_similarity": redundancy[pick]}})
        chosen.append(pick)
        remaining.remove(pick)
    return output


def prepare_models():
    """Explicit deployment command downloads models; chat never downloads them."""
    config = settings()
    with _inference_lock:
        _load("cross_encoder", config, False)
        if config.embedding_backend == "local":
            _load("embedding", config, False)
    return config
