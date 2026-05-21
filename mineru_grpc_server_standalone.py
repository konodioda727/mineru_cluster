#!/usr/bin/env python3
"""Standalone MinerU gRPC server.

Requirements in the same directory:
- mineru_pb2.py
- mineru_pb2_grpc.py
"""

from __future__ import annotations

import os
import sys as _sys

# ── macOS subprocess stability ────────────────────────────────────────────────
# Must be set before ANY import of ObjC-backed frameworks (Metal, PyTorch, Paddle).
# With multiprocessing spawn, child processes re-run module-level code, so these
# env vars are inherited by workers before ML frameworks are loaded.
#
# OBJC_DISABLE_INITIALIZE_FORK_SAFETY: Apple's ObjC runtime guards against being
#   used between fork() and exec(). Even with spawn, Python's multiprocessing has
#   a brief fork step on macOS that trips this guard → silent SIGABRT worker crash.
# OMP_NUM_THREADS=1: PyTorch and PaddlePaddle both bundle libomp. Multiple
#   subprocesses each initialising their own OpenMP runtimes causes libomp to
#   SEGFAULT (confirmed pytorch#161865 on M4 Max, same on M2 Ultra).
# PYTORCH_ENABLE_MPS_FALLBACK: ops not yet implemented on MPS fall back to CPU
#   instead of raising an error or crashing the Metal command encoder.
# no_proxy=*: macOS _scproxy.so (called by urllib at import time) is not
#   fork-safe; setting no_proxy prevents it from being invoked in child processes.
if _sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("no_proxy", "*")

# Limit onnxruntime / OpenMP thread counts.
# Each worker subprocess spawns its own onnxruntime thread pool. With N workers
# the default (one thread per CPU core) creates N×core_count threads total,
# which saturates the CPU scheduler and makes GC-vs-C++-thread races more
# likely. Cap to a sane number; MinerU's workload is not embarrassingly parallel
# at the intra-op level so more threads past ~4 gives diminishing returns.
_ort_threads = os.getenv("OMP_NUM_THREADS") or os.getenv("MKL_NUM_THREADS")
if not _ort_threads:
    import multiprocessing as _mp
    _cpu = _mp.cpu_count()
    # Leave room for other processes; 4 is enough for onnxruntime intra-op.
    _cap = max(1, min(4, _cpu // 4))
    os.environ.setdefault("OMP_NUM_THREADS", str(_cap))
    os.environ.setdefault("MKL_NUM_THREADS", str(_cap))
    os.environ.setdefault("ORT_NUM_THREADS", str(_cap))
    del _cap, _cpu
del _ort_threads

# ── gRPC fork support ─────────────────────────────────────────────────────────
# Must be set before grpc is imported, otherwise gRPC ignores them.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")
# epoll is Linux-only; on macOS use poll to avoid FD-from-fork warnings.
if _sys.platform != "darwin":
    os.environ.setdefault("GRPC_POLL_STRATEGY", "epoll1")

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
import uuid
import sys

import fitz
import grpc
from loguru import logger

import mineru_pb2
import mineru_pb2_grpc

from mineru.cli.common import _process_pipeline
from mineru.utils.enum_class import MakeMode


DEFAULT_PORT = int(os.getenv("MINERU_SERVER_PORT", "50051"))
DEFAULT_MAX_WORKERS = int(os.getenv("MINERU_SERVER_MAX_WORKERS", "3"))
DEFAULT_CHUNK_SIZE_RAW = (os.getenv("MINERU_SERVER_CHUNK_SIZE") or "").strip()
DEFAULT_MAX_MESSAGE_BYTES = int(
    os.getenv("MINERU_SERVER_MAX_MESSAGE_BYTES", str(256 * 1024 * 1024))
)

# Role / remote-worker topology — declared early so that GRPC_WORKERS and
# REQUEST_CONCURRENCY can scale automatically in scheduler mode.
DEFAULT_ROLE = (os.getenv("MINERU_SERVER_ROLE") or "standalone").strip().lower()
DEFAULT_WORKER_PROCESSES = int(os.getenv("MINERU_SERVER_WORKER_PROCESSES", "3"))
DEFAULT_REMOTE_WORKERS = tuple(
    item.strip()
    for item in (os.getenv("MINERU_REMOTE_WORKERS") or "").split(",")
    if item.strip()
)
# How many concurrent jobs each remote worker node is expected to handle.
# Scheduler uses this to size its own semaphore and gRPC thread pool so that
# all worker capacity is actually utilised instead of being bottlenecked here.
DEFAULT_REMOTE_WORKER_CONCURRENCY = int(
    os.getenv("MINERU_REMOTE_WORKER_CONCURRENCY", str(DEFAULT_WORKER_PROCESSES))
)


def _scheduler_total_concurrency() -> int:
    """Total concurrent slots when running as scheduler: nodes × per-node capacity."""
    return max(1, len(DEFAULT_REMOTE_WORKERS) * DEFAULT_REMOTE_WORKER_CONCURRENCY)


def _default_grpc_workers() -> int:
    explicit = (os.getenv("MINERU_SERVER_GRPC_WORKERS") or "").strip()
    if explicit:
        return max(1, int(explicit))
    if DEFAULT_ROLE == "scheduler" and DEFAULT_REMOTE_WORKERS:
        # Extra headroom so that threads waiting on a semaphore don't block
        # incoming requests from getting a handler thread.
        return _scheduler_total_concurrency() + 4
    return 4


def _default_request_concurrency() -> int:
    explicit = (os.getenv("MINERU_SERVER_REQUEST_CONCURRENCY") or "").strip()
    if explicit:
        return max(1, int(explicit))
    if DEFAULT_ROLE == "scheduler" and DEFAULT_REMOTE_WORKERS:
        return _scheduler_total_concurrency()
    return max(1, DEFAULT_WORKER_PROCESSES)


DEFAULT_GRPC_WORKERS = _default_grpc_workers()
DEFAULT_KEEPALIVE_TIME_MS = int(os.getenv("MINERU_SERVER_KEEPALIVE_TIME_MS", "60000"))
DEFAULT_KEEPALIVE_TIMEOUT_MS = int(os.getenv("MINERU_SERVER_KEEPALIVE_TIMEOUT_MS", "20000"))
DEFAULT_KEEPALIVE_PERMIT_WITHOUT_CALLS = (
    (os.getenv("MINERU_SERVER_KEEPALIVE_PERMIT_WITHOUT_CALLS") or "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_HTTP2_MAX_PINGS_WITHOUT_DATA = int(
    os.getenv("MINERU_SERVER_HTTP2_MAX_PINGS_WITHOUT_DATA", "0")
)
DEFAULT_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS = int(
    os.getenv("MINERU_SERVER_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS", "30000")
)
DEFAULT_HTTP2_MAX_PING_STRIKES = int(
    os.getenv("MINERU_SERVER_HTTP2_MAX_PING_STRIKES", "0")
)
DEFAULT_MAX_CONNECTION_IDLE_MS = int(
    os.getenv("MINERU_SERVER_MAX_CONNECTION_IDLE_MS", str(30 * 60 * 1000))
)
DEFAULT_MAX_CONNECTION_AGE_MS = int(
    os.getenv("MINERU_SERVER_MAX_CONNECTION_AGE_MS", str(2 * 60 * 60 * 1000))
)
DEFAULT_MAX_CONNECTION_AGE_GRACE_MS = int(
    os.getenv("MINERU_SERVER_MAX_CONNECTION_AGE_GRACE_MS", str(5 * 60 * 1000))
)
DEFAULT_BACKEND = (os.getenv("MINERU_SERVER_BACKEND") or "pipeline").strip()
DEFAULT_METHOD = (os.getenv("MINERU_SERVER_METHOD") or "auto").strip()
DEFAULT_LANG = (os.getenv("MINERU_SERVER_LANG") or "ch").strip()
DEFAULT_DEVICE = (os.getenv("MINERU_SERVER_DEVICE") or "mps").strip()
DEFAULT_EXECUTION_MODE = (os.getenv("MINERU_SERVER_EXECUTION_MODE") or "").strip().lower()
DEFAULT_FORMULA_ENABLE = (os.getenv("MINERU_SERVER_FORMULA_ENABLE") or "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TABLE_ENABLE = (os.getenv("MINERU_SERVER_TABLE_ENABLE") or "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_TABLE_MERGE_ENABLE = (
    (os.getenv("MINERU_SERVER_TABLE_MERGE_ENABLE") or "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_KEEP_TEMP_DIR = (os.getenv("MINERU_SERVER_KEEP_TEMP_DIR") or "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_MIN_MERGED_TEXT_LEN = int(os.getenv("MINERU_SERVER_MIN_MERGED_TEXT_LEN", "20"))
DEFAULT_MINERU_CLI = (os.getenv("MINERU_CLI") or "mineru").strip()
DEFAULT_REQUEST_CONCURRENCY = _default_request_concurrency()
DEFAULT_PREWARM_ENABLED = (os.getenv("MINERU_SERVER_PREWARM") or "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# After this many tasks a worker exits and is automatically restarted, freeing any
# memory that ML runtimes (PaddleOCR, PyTorch, etc.) accumulate across calls.
# 0 disables recycling (legacy behaviour — workers run forever).
DEFAULT_MAX_TASKS_PER_WORKER = int(os.getenv("MINERU_SERVER_MAX_TASKS_PER_WORKER", "20"))
# Soft memory limit per worker (GB). After each task, if RSS exceeds this,
# the worker exits cleanly and gets recycled. 0 = no limit.
DEFAULT_MAX_WORKER_MEMORY_GB = float(os.getenv("MINERU_SERVER_MAX_WORKER_MEMORY_GB", "0"))
DEFAULT_PREWARM_METHOD = (os.getenv("MINERU_SERVER_PREWARM_METHOD") or "ocr").strip().lower()
DEFAULT_PREWARM_LANG = (os.getenv("MINERU_SERVER_PREWARM_LANG") or DEFAULT_LANG).strip()
DEFAULT_LOG_MODE = (os.getenv("MINERU_SERVER_LOG_MODE") or "concise").strip().lower()
DEFAULT_REMOTE_TIMEOUT_SECONDS = float(os.getenv("MINERU_REMOTE_TIMEOUT_SECONDS", "7200"))
VERTICAL_TRADITIONAL_CHUNK_SIZE_RAW = (
    os.getenv("MINERU_SERVER_VERTICAL_TRADITIONAL_CHUNK_SIZE") or ""
).strip()
VERTICAL_TRADITIONAL_METHOD = (
    os.getenv("MINERU_SERVER_VERTICAL_TRADITIONAL_METHOD") or "ocr"
).strip()
VERTICAL_TRADITIONAL_LANG = (
    os.getenv("MINERU_SERVER_VERTICAL_TRADITIONAL_LANG") or "chinese_cht"
).strip()
VERTICAL_TRADITIONAL_MIN_MERGED_TEXT_LEN = int(
    os.getenv("MINERU_SERVER_VERTICAL_TRADITIONAL_MIN_MERGED_TEXT_LEN", "8")
)


@dataclass(frozen=True)
class ParseProfile:
    profile_name: str
    method: str
    lang: str
    chunk_size_override: int | None
    include_discarded_text: bool
    reading_order: str
    min_merged_text_len: int
    merge_structured_blocks_upward: bool


def resolve_vertical_traditional_chunk_size_override() -> int | None:
    if not VERTICAL_TRADITIONAL_CHUNK_SIZE_RAW:
        return None
    try:
        value = int(VERTICAL_TRADITIONAL_CHUNK_SIZE_RAW)
    except ValueError:
        return None
    return value if value > 0 else None


def log(message: str) -> None:
    print(message, flush=True)


def log_event(stage: str, *, filename: str = "", task_id: str = "", worker_id: int | None = None, detail: str = "") -> None:
    parts = [f"[{stage}]"]
    if task_id:
        parts.append(f"task={task_id}")
    if worker_id is not None:
        parts.append(f"worker={worker_id}")
    if filename:
        parts.append(f"file={filename}")
    if detail:
        parts.append(detail)
    log(" ".join(parts))


def configure_logging() -> None:
    logger.remove()
    if DEFAULT_LOG_MODE == "concise":
        logger.add(sys.stderr, level="ERROR")
    else:
        logger.add(sys.stderr, level="INFO")


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def is_in_process_pipeline_enabled() -> bool:
    if DEFAULT_EXECUTION_MODE in {"python", "inprocess", "in_process"}:
        return True
    if DEFAULT_EXECUTION_MODE == "cli":
        return False
    return DEFAULT_BACKEND == "pipeline"


def is_scheduler_role() -> bool:
    return DEFAULT_ROLE == "scheduler"


def is_worker_role() -> bool:
    return DEFAULT_ROLE == "worker"


def is_standalone_role() -> bool:
    return DEFAULT_ROLE in {"", "standalone"}


def configure_mineru_runtime_env() -> None:
    os.environ["MINERU_DEVICE_MODE"] = DEFAULT_DEVICE
    os.environ["MINERU_FORMULA_ENABLE"] = format_bool(DEFAULT_FORMULA_ENABLE)
    os.environ["MINERU_TABLE_ENABLE"] = format_bool(DEFAULT_TABLE_ENABLE)
    os.environ["MINERU_TABLE_MERGE_ENABLE"] = format_bool(DEFAULT_TABLE_MERGE_ENABLE)


def normalize_hint(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def hint_has_any(value: str, keywords: set[str]) -> bool:
    normalized = normalize_hint(value)
    return any(keyword in normalized for keyword in keywords)


def is_traditional_chinese_hint(value: str) -> bool:
    normalized = normalize_hint(value)
    return (
        "繁体" in value
        or "繁體" in value
        or any(
            keyword in normalized
            for keyword in {"traditional", "cht", "chinese_cht", "zh_hant", "zh_tw", "zh_hk"}
        )
    )


def is_vertical_layout_hint(value: str) -> bool:
    normalized = normalize_hint(value)
    return (
        "竖排" in value
        or "豎排" in value
        or any(keyword in normalized for keyword in {"vertical", "vertical_rl", "vertical_book"})
    )


def normalize_mineru_lang(value: str) -> str:
    raw = (value or "").strip()
    normalized = normalize_hint(raw)
    if not raw:
        return DEFAULT_LANG

    alias_map = {
        "zh": "ch",
        "zh_cn": "ch",
        "zh_hans": "ch",
        "zh_hant": "chinese_cht",
        "zh_tw": "chinese_cht",
        "zh_hk": "chinese_cht",
        "zh_mo": "chinese_cht",
        "traditional_chinese": "chinese_cht",
        "simplified_chinese": "ch",
        "chinese": "ch",
        "chinese_cht": "chinese_cht",
        "traditional": "chinese_cht",
        "cht": "chinese_cht",
        "jp": "japan",
        "ja": "japan",
        "ko": "korean",
        "kr": "korean",
        "en_us": "en",
        "en_gb": "en",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    if normalized.startswith("zh_hant"):
        return "chinese_cht"
    if normalized.startswith("zh_hans") or normalized.startswith("zh_cn"):
        return "ch"
    return raw


def resolve_parse_profile(request: mineru_pb2.ParsePdfRequest) -> ParseProfile:
    language_hint = request.language_hint or ""
    layout_hint = request.layout_hint or ""
    document_type_hint = request.document_type_hint or ""
    model_version = request.model_version or ""

    explicit_vertical_traditional = any(
        hint_has_any(value, {"traditional_vertical", "vertical_traditional", "cht_vertical"})
        for value in (layout_hint, document_type_hint, model_version)
    )
    vertical_traditional = explicit_vertical_traditional or (
        is_traditional_chinese_hint(language_hint)
        and (
            is_vertical_layout_hint(layout_hint)
            or is_vertical_layout_hint(document_type_hint)
            or "古籍" in document_type_hint
            or hint_has_any(document_type_hint, {"book"})
        )
    )

    if vertical_traditional:
        return ParseProfile(
            profile_name="traditional_vertical_fine",
            method=VERTICAL_TRADITIONAL_METHOD,
            lang=VERTICAL_TRADITIONAL_LANG,
            chunk_size_override=resolve_vertical_traditional_chunk_size_override(),
            include_discarded_text=True,
            reading_order="vertical_rl",
            min_merged_text_len=max(1, VERTICAL_TRADITIONAL_MIN_MERGED_TEXT_LEN),
            merge_structured_blocks_upward=False,
        )

    requested_method = normalize_hint(model_version)
    if requested_method not in {"auto", "txt", "ocr"}:
        requested_method = ""

    requested_lang = normalize_mineru_lang(request.language_hint or "")

    return ParseProfile(
        profile_name="default",
        method=requested_method or DEFAULT_METHOD,
        lang=requested_lang,
        chunk_size_override=None,
        include_discarded_text=False,
        reading_order="default",
        min_merged_text_len=DEFAULT_MIN_MERGED_TEXT_LEN,
        merge_structured_blocks_upward=True,
    )


def resolve_optional_chunk_size() -> int | None:
    if not DEFAULT_CHUNK_SIZE_RAW:
        return None
    try:
        value = int(DEFAULT_CHUNK_SIZE_RAW)
    except ValueError:
        return None
    return value if value > 0 else None


def resolve_chunk_size(total_pages: int, max_workers: int, manual_chunk_size: int | None) -> int:
    if total_pages <= 0:
        return 1
    if manual_chunk_size is not None:
        return max(1, min(total_pages, manual_chunk_size))
    worker_count = max(1, min(total_pages, max_workers))
    return max(1, math.ceil(total_pages / worker_count))


def safe_bbox(raw_bbox: object) -> tuple[float, float, float, float]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        x0, y0, x1, y1 = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)
    return (x0, y0, x1, y1)


def bbox_to_payload(raw_bbox: object) -> dict[str, float]:
    x0, y0, x1, y1 = safe_bbox(raw_bbox)
    return {
        "x": x0,
        "y": y0,
        "width": max(0.0, x1 - x0),
        "height": max(0.0, y1 - y0),
    }


def merge_bbox(base_bbox: dict[str, float] | None, incoming_bbox: dict[str, float] | None) -> dict[str, float] | None:
    if base_bbox is None:
        return incoming_bbox
    if incoming_bbox is None:
        return base_bbox
    x0 = min(base_bbox["x"], incoming_bbox["x"])
    y0 = min(base_bbox["y"], incoming_bbox["y"])
    x1 = max(base_bbox["x"] + base_bbox["width"], incoming_bbox["x"] + incoming_bbox["width"])
    y1 = max(base_bbox["y"] + base_bbox["height"], incoming_bbox["y"] + incoming_bbox["height"])
    return {
        "x": x0,
        "y": y0,
        "width": max(0.0, x1 - x0),
        "height": max(0.0, y1 - y0),
    }


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(parts).strip()
    return str(value).strip()


def join_non_empty(parts: list[str]) -> str:
    return "\n".join(part for part in parts if part).strip()


def flatten_table_text(item: dict) -> str:
    return join_non_empty(
        [
            normalize_text(item.get("table_caption") or item.get("caption")),
            normalize_text(item.get("table_footnote") or item.get("footnote")),
            normalize_text(item.get("table_body") or item.get("html") or item.get("content")),
        ]
    )


def flatten_list_text(item: dict) -> str:
    return join_non_empty(
        [
            normalize_text(item.get("list_caption") or item.get("caption")),
            normalize_text(
                item.get("list_body")
                or item.get("html")
                or item.get("text")
                or item.get("content")
            ),
        ]
    )


def append_block(
    bucket: list[dict],
    *,
    text: str,
    bbox: dict[str, float] | None,
    bbox_source: str,
) -> None:
    bucket.append(
        {
            "text": text,
            "bbox": bbox,
            "bbox_source": bbox_source,
        }
    )


def merge_upward_until_long_enough(
    bucket: list[dict],
    *,
    text: str,
    bbox: dict[str, float] | None,
    merged_source: str,
    min_text_len: int = DEFAULT_MIN_MERGED_TEXT_LEN,
) -> None:
    text = text.strip()
    if not text:
        return

    if not bucket:
        append_block(bucket, text=text, bbox=bbox, bbox_source=merged_source)
        return

    bucket[-1]["text"] = join_non_empty([bucket[-1]["text"], text])
    bucket[-1]["bbox"] = merge_bbox(bucket[-1].get("bbox"), bbox)
    bucket[-1]["bbox_source"] = merged_source

    while len(str(bucket[-1].get("text") or "").strip()) < min_text_len and len(bucket) >= 2:
        current = bucket.pop()
        bucket[-1]["text"] = join_non_empty([bucket[-1]["text"], current["text"]])
        bucket[-1]["bbox"] = merge_bbox(bucket[-1].get("bbox"), current.get("bbox"))
        bucket[-1]["bbox_source"] = merged_source


def sort_blocks_for_reading_order(blocks: list[dict], reading_order: str) -> list[dict]:
    if reading_order == "vertical_rl":
        return sorted(
            blocks,
            key=lambda block: (
                -float((block.get("bbox") or {}).get("x", 0.0)),
                float((block.get("bbox") or {}).get("y", 0.0)),
                float((block.get("bbox") or {}).get("height", 0.0)),
            ),
        )
    return sorted(
        blocks,
        key=lambda block: (
            float((block.get("bbox") or {}).get("y", 0.0)),
            float((block.get("bbox") or {}).get("x", 0.0)),
        ),
    )


def load_chunk_pages(
    chunk_path: str,
    output_dir: str,
    chunk_idx: int,
    start_page_offset: int,
    profile: ParseProfile,
) -> list[dict]:
    pdf_basename = Path(chunk_path).stem
    base_result_dir = Path(output_dir) / pdf_basename
    result_dirs = [
        base_result_dir,
        base_result_dir / profile.method,
    ]
    candidate_paths = [
        path
        for result_dir in result_dirs
        for path in (
            result_dir / f"{pdf_basename}_content_list.json",
            result_dir / f"{pdf_basename}_middle.json",
        )
    ]
    json_path = next((path for path in candidate_paths if path.exists()), None)
    if json_path is None:
        raise FileNotFoundError(
            f"MinerU output json not found for chunk {chunk_idx} under {base_result_dir}; "
            f"checked={[str(path) for path in candidate_paths]}"
        )

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    page_sizes = load_middle_page_sizes(output_dir, pdf_basename, profile.method)

    if isinstance(payload, list):
        content_items = payload
    else:
        content_items = payload.get("content_list") or []
    if not content_items:
        raise ValueError(
            f"content_list is empty or missing for chunk {chunk_idx}; "
            f"expected {pdf_basename}_content_list.json with text/table/list items"
        )

    page_buckets: dict[int, list[dict]] = {}
    for item in content_items:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type") or "").strip().lower()
        page_idx = int(item.get("page_idx") or 0)
        bucket = page_buckets.setdefault(page_idx, [])
        page_size = page_sizes[page_idx] if 0 <= page_idx < len(page_sizes) else {}
        page_width = int(page_size.get("page_width") or 0)
        page_height = int(page_size.get("page_height") or 0)
        bbox = (
            bbox_1000_to_page(item.get("bbox"), page_width, page_height)
            if page_width > 0 and page_height > 0
            else None
        )

        if item_type == "text":
            text = normalize_text(item.get("text") or item.get("content"))
            if not text:
                continue
            append_block(
                bucket,
                text=text,
                bbox=bbox,
                bbox_source="mineru_content_list_1000_mapped",
            )
            continue

        if item_type == "table":
            table_text = flatten_table_text(item)
            if not table_text:
                continue
            if profile.merge_structured_blocks_upward:
                merge_upward_until_long_enough(
                    bucket,
                    text=table_text,
                    bbox=bbox,
                    merged_source="mineru_content_list_table_1000_mapped",
                    min_text_len=profile.min_merged_text_len,
                )
            else:
                append_block(
                    bucket,
                    text=table_text,
                    bbox=bbox,
                    bbox_source="mineru_content_list_table_1000_mapped",
                )
            continue

        if item_type == "list":
            list_text = flatten_list_text(item)
            if not list_text:
                continue
            if profile.merge_structured_blocks_upward:
                merge_upward_until_long_enough(
                    bucket,
                    text=list_text,
                    bbox=bbox,
                    merged_source="mineru_content_list_list_1000_mapped",
                    min_text_len=profile.min_merged_text_len,
                )
            else:
                append_block(
                    bucket,
                    text=list_text,
                    bbox=bbox,
                    bbox_source="mineru_content_list_list_1000_mapped",
                )
            continue

        if item_type == "discarded" and profile.include_discarded_text:
            text = normalize_text(item.get("text") or item.get("content"))
            if not text:
                continue
            append_block(
                bucket,
                text=text,
                bbox=bbox,
                bbox_source="mineru_discarded_1000_mapped",
            )

    pages: list[dict] = []
    for page_idx in sorted(page_buckets):
        pdf_page_no = start_page_offset + page_idx + 1
        paragraphs: list[dict] = []
        page_text_parts: list[str] = []

        ordered_blocks = sort_blocks_for_reading_order(
            page_buckets.get(page_idx, []),
            profile.reading_order,
        )
        for paragraph_idx, block in enumerate(ordered_blocks, start=1):
            text = str(block.get("text") or "").strip()
            if not text:
                continue

            paragraphs.append(
                {
                    "paragraph_id": f"page-{pdf_page_no}-p{paragraph_idx}",
                    "paragraph_no": paragraph_idx,
                    "text": text,
                    "bbox": block.get("bbox"),
                    "bbox_source": str(block.get("bbox_source") or "mineru_content_list"),
                }
            )
            page_text_parts.append(text)

        page_size = page_sizes[page_idx] if 0 <= page_idx < len(page_sizes) else {}
        pages.append(
            {
                "pdf_page_no": pdf_page_no,
                "text": "\n".join(page_text_parts),
                "paragraphs": paragraphs,
                "page_width": int(page_size.get("page_width") or 0),
                "page_height": int(page_size.get("page_height") or 0),
            }
        )

    return pages


def load_middle_page_sizes(output_dir: str, pdf_basename: str, method: str) -> list[dict[str, int]]:
    base_result_dir = Path(output_dir) / pdf_basename
    result_dirs = [
        base_result_dir,
        base_result_dir / method,
    ]
    candidate_paths = [
        result_dir / f"{pdf_basename}_middle.json"
        for result_dir in result_dirs
    ]
    middle_path = next((path for path in candidate_paths if path.exists()), None)
    if middle_path is None:
        raise FileNotFoundError(
            f"MinerU middle json not found under {base_result_dir}; "
            f"checked={[str(path) for path in candidate_paths]}"
        )

    with middle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pdf_info = payload.get("pdf_info") or []
    page_sizes: list[dict[str, int]] = []
    for page in pdf_info:
        page_size = page.get("page_size") or [0, 0]
        if not isinstance(page_size, (list, tuple)) or len(page_size) != 2:
            page_sizes.append({"page_width": 0, "page_height": 0})
            continue

        try:
            page_width = int(page_size[0] or 0)
            page_height = int(page_size[1] or 0)
        except (TypeError, ValueError):
            page_width = 0
            page_height = 0

        page_sizes.append(
            {
                "page_width": max(0, page_width),
                "page_height": max(0, page_height),
            }
        )
    return page_sizes


def bbox_1000_to_page(raw_bbox: object, page_width: int, page_height: int) -> dict[str, float]:
    x0, y0, x1, y1 = safe_bbox(raw_bbox)
    return {
        "x": x0 / 1000.0 * page_width,
        "y": y0 / 1000.0 * page_height,
        "width": max(0.0, (x1 - x0) / 1000.0 * page_width),
        "height": max(0.0, (y1 - y0) / 1000.0 * page_height),
    }


def split_pdf(pdf_path: str, chunk_size: int, temp_dir: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        chunks: list[dict] = []

        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size - 1, total_pages - 1)
            chunk_idx = start_page // chunk_size
            chunk_path = Path(temp_dir) / f"chunk_{chunk_idx}.pdf"

            chunk_doc = fitz.open()
            try:
                chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                chunk_doc.save(chunk_path)
            finally:
                chunk_doc.close()

            chunks.append(
                {
                    "path": str(chunk_path),
                    "chunk_idx": chunk_idx,
                    "start_page_offset": start_page,
                }
            )
        return chunks
    finally:
        doc.close()


def build_response(filename: str, all_pages_data: dict[int, list[dict]]) -> mineru_pb2.ParsePdfResponse:
    response = mineru_pb2.ParsePdfResponse(task_id=f"mineru-job-{uuid.uuid4().hex[:8]}")
    response.document.title = Path(filename).stem

    full_text_parts: list[str] = []
    paragraph_no_in_doc = 1

    for chunk_idx in sorted(all_pages_data):
        for page_dict in all_pages_data[chunk_idx]:
            page = response.document.pages.add()
            page.pdf_page_no = int(page_dict["pdf_page_no"])
            page.text = str(page_dict.get("text") or "")
            page.page_width = int(page_dict.get("page_width") or 0)
            page.page_height = int(page_dict.get("page_height") or 0)

            if page.text:
                full_text_parts.append(page.text)

            for paragraph_dict in page_dict.get("paragraphs", []):
                paragraph = response.document.pages[-1].paragraphs.add()
                paragraph.paragraph_id = str(paragraph_dict["paragraph_id"])
                paragraph.paragraph_no = int(paragraph_dict["paragraph_no"])
                paragraph.paragraph_no_in_doc = paragraph_no_in_doc
                paragraph.text = str(paragraph_dict["text"])
                paragraph.bbox_source = str(
                    paragraph_dict.get("bbox_source") or "mineru_content_list"
                )

                bbox = paragraph_dict.get("bbox")
                if bbox:
                    paragraph.bbox.x = float(bbox["x"])
                    paragraph.bbox.y = float(bbox["y"])
                    paragraph.bbox.width = float(bbox["width"])
                    paragraph.bbox.height = float(bbox["height"])

                paragraph_no_in_doc += 1

    response.document.full_text = "\f".join(full_text_parts)
    return response


def process_pdf_request(filename: str, file_content: bytes, profile: ParseProfile) -> bytes:
    temp_dir = tempfile.mkdtemp(prefix="mineru-grpc-")
    preserve_temp_dir = DEFAULT_KEEP_TEMP_DIR
    temp_root = Path(temp_dir)
    pdf_path = temp_root / (filename or "input.pdf")
    output_dir = temp_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf_path.write_bytes(file_content)
        with fitz.open(pdf_path) as input_doc:
            total_pages = len(input_doc)

        resolved_chunk_size = resolve_chunk_size(
            total_pages,
            DEFAULT_MAX_WORKERS,
            profile.chunk_size_override or resolve_optional_chunk_size(),
        )
        chunks = split_pdf(str(pdf_path), resolved_chunk_size, temp_dir)
        all_pages_data = process_chunks_in_process(chunks, str(output_dir), profile)
        response = build_response(filename or "input.pdf", all_pages_data)
        return response.SerializeToString()
    except Exception:
        preserve_temp_dir = True
        raise
    finally:
        if not preserve_temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_pipeline(
    output_dir: str,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    lang_list: list[str],
    profile: ParseProfile,
    table_enable: bool,
) -> None:
    _process_pipeline(
        output_dir=output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        p_lang_list=lang_list,
        parse_method=profile.method,
        p_formula_enable=DEFAULT_FORMULA_ENABLE,
        p_table_enable=table_enable,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=False,
        f_dump_middle_json=True,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
    )


def _is_table_indexing_error(exc: BaseException) -> bool:
    """Return True if the exception looks like MinerU's wireless-table numpy IndexError.

    MinerU's wireless (borderless) table detector occasionally produces a 1-D
    numpy array for edge-case tables (single row/column after cropping) but
    indexes it as 2-D, raising IndexError with the message
    'too many indices for array: array is 1-dimensional, but 2 were indexed'.
    We check both the immediate exception and its cause chain so that even if
    MinerU wraps it in a RuntimeError we still catch it.
    """
    needle = "too many indices for array"
    node: BaseException | None = exc
    while node is not None:
        if needle in str(node):
            return True
        node = node.__cause__ or node.__context__
    return False


def process_chunks_in_process(
    chunks: list[dict],
    output_dir: str,
    profile: ParseProfile,
) -> dict[int, list[dict]]:
    if not chunks:
        return {}

    configure_mineru_runtime_env()
    pdf_file_names = [Path(chunk["path"]).stem for chunk in chunks]
    pdf_bytes_list = [Path(chunk["path"]).read_bytes() for chunk in chunks]
    lang_list = [profile.lang for _ in chunks]

    try:
        _run_pipeline(output_dir, pdf_file_names, pdf_bytes_list, lang_list, profile, DEFAULT_TABLE_ENABLE)
    except Exception as exc:
        # MinerU's wireless-table detector has a known numpy IndexError on certain
        # PDFs (edge-case tables that are 1-D after model prediction). When this
        # happens, fall back to a table-disabled retry so the document is still
        # extracted rather than failing entirely. The retry uses a fresh output
        # directory to avoid partial files from the failed attempt confusing the
        # pipeline.
        if DEFAULT_TABLE_ENABLE and _is_table_indexing_error(exc):
            log(f"[worker] table detection raised IndexError ({exc}); retrying without table detection")
            # Clean up any partial output written before the crash.
            for name in pdf_file_names:
                shutil.rmtree(Path(output_dir) / name, ignore_errors=True)
            _run_pipeline(output_dir, pdf_file_names, pdf_bytes_list, lang_list, profile, False)
        else:
            raise

    all_pages_data: dict[int, list[dict]] = {}
    for chunk in chunks:
        chunk_idx = int(chunk["chunk_idx"])
        pages = load_chunk_pages(
            chunk["path"],
            output_dir,
            chunk_idx,
            int(chunk["start_page_offset"]),
            profile,
        )
        all_pages_data[chunk_idx] = pages
    return all_pages_data


def build_prewarm_pdf_bytes() -> bytes:
    doc = fitz.open()
    try:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 96), "MinerU warmup page", fontsize=20)
        page.insert_text(
            (72, 136),
            "This startup warmup primes layout, OCR, and table-related runtime initialization.",
            fontsize=12,
        )
        page.insert_text((72, 176), "A  B  C", fontsize=12)
        page.insert_text((72, 206), "1  alpha  beta", fontsize=12)
        page.insert_text((72, 236), "2  gamma  delta", fontsize=12)
        for x in (68, 140, 250, 360):
            page.draw_line((x, 160), (x, 255))
        for y in (160, 190, 220, 255):
            page.draw_line((68, y), (360, y))
        return doc.write()
    finally:
        doc.close()


def run_startup_prewarm() -> None:
    if not DEFAULT_PREWARM_ENABLED:
        log("[startup] prewarm disabled")
        return
    if not is_in_process_pipeline_enabled():
        log("[startup] prewarm skipped because execution mode is CLI")
        return

    prewarm_method = DEFAULT_PREWARM_METHOD if DEFAULT_PREWARM_METHOD in {"auto", "txt", "ocr"} else "ocr"
    configure_mineru_runtime_env()
    warmup_dir = tempfile.mkdtemp(prefix="mineru-prewarm-")
    started_at = time.time()
    log(
        f"[startup] prewarm begin (backend={DEFAULT_BACKEND}, method={prewarm_method}, "
        f"lang={DEFAULT_PREWARM_LANG}, device={DEFAULT_DEVICE})"
    )
    try:
        _process_pipeline(
            output_dir=warmup_dir,
            pdf_file_names=["warmup"],
            pdf_bytes_list=[build_prewarm_pdf_bytes()],
            p_lang_list=[DEFAULT_PREWARM_LANG],
            parse_method=prewarm_method,
            p_formula_enable=DEFAULT_FORMULA_ENABLE,
            p_table_enable=DEFAULT_TABLE_ENABLE,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode=MakeMode.MM_MD,
        )
        log(f"[startup] prewarm complete in {time.time() - started_at:.1f}s")
    finally:
        shutil.rmtree(warmup_dir, ignore_errors=True)


def _try_empty_torch_cache() -> None:
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _worker_rss_gb() -> float:
    if sys.platform == "darwin":
        return _macos_phys_footprint_gb()
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 ** 2)
    raise OSError("VmRSS not found in /proc/self/status")


def _macos_phys_footprint_gb() -> float:
    """Read ri_phys_footprint via proc_pid_rusage — same metric Activity Monitor uses.
    Includes CPU heap, Metal/MPS GPU buffers, and IOKit allocations.

    IMPORTANT: Do NOT define a ctypes.Structure with only the fields you need.
    rusage_info_v4 has 36+ uint64 fields (304 bytes total). A partial struct
    of 80 bytes causes the kernel to overwrite 224 bytes of Python heap memory,
    corrupting pymalloc metadata and causing random SIGSEGV crashes later.

    Instead, allocate a 1024-byte opaque buffer. ri_phys_footprint has been at
    byte offset 72 since rusage_info_v0 and all later versions only append
    fields at the end, so the offset is stable across all macOS versions.

    Offset breakdown:
      ri_uuid[16]          =  16 bytes  (offset   0)
      ri_user_time         =   8 bytes  (offset  16)
      ri_system_time       =   8 bytes  (offset  24)
      ri_pkg_idle_wkups    =   8 bytes  (offset  32)
      ri_interrupt_wkups   =   8 bytes  (offset  40)
      ri_pageins           =   8 bytes  (offset  48)
      ri_wired_size        =   8 bytes  (offset  56)
      ri_resident_size     =   8 bytes  (offset  64)
      ri_phys_footprint    =   8 bytes  (offset  72)  ← what we read
    """
    import ctypes
    import ctypes.util
    import struct

    libproc = ctypes.CDLL(ctypes.util.find_library("proc"))
    # 1024 bytes is safely larger than any known rusage_info_vN struct.
    buf = ctypes.create_string_buffer(1024)
    ret = libproc.proc_pid_rusage(os.getpid(), 4, buf)
    if ret != 0:
        raise OSError(f"proc_pid_rusage returned {ret}")
    (phys_footprint,) = struct.unpack_from("<Q", buf, 72)
    return phys_footprint / (1024 ** 3)


def worker_process_main(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    try:
        configure_logging()
        configure_mineru_runtime_env()
        run_startup_prewarm()
        log(f"[worker] worker {worker_id} ready, memory_limit={DEFAULT_MAX_WORKER_MEMORY_GB or 'disabled'}GB, max_tasks={DEFAULT_MAX_TASKS_PER_WORKER or 'disabled'}")
        result_queue.put({"type": "ready", "worker_id": worker_id})

        tasks_processed = 0
        while True:
            task = task_queue.get()
            if task is None:
                return

            task_id = str(task["task_id"])
            filename = str(task["filename"])
            profile = task["profile"]
            # Extract bytes and drop the dict reference so the large PDF payload
            # can be GC'd as soon as process_pdf_request writes it to disk.
            file_content = task["file_content"]
            del task

            # Notify manager which task we are starting. If this process crashes
            # mid-task, the manager uses this to fail the stuck waiter so the
            # gRPC handler thread is unblocked and _request_slots is released.
            result_queue.put({"type": "started", "worker_id": worker_id, "task_id": task_id})

            try:
                response_bytes = process_pdf_request(
                    filename,
                    file_content,
                    profile,
                )
                del file_content
                result_queue.put(
                    {
                        "type": "result",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "response_bytes": response_bytes,
                    }
                )
                del response_bytes
                log_event("completed", task_id=task_id, worker_id=worker_id, filename=filename)
            except Exception as exc:
                result_queue.put(
                    {
                        "type": "error",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "error": str(exc),
                    }
                )
                log_event("failed", task_id=task_id, worker_id=worker_id, filename=filename, detail=str(exc))

            # Do NOT call gc.collect() here. onnxruntime keeps a persistent C++
            # thread pool alive for the entire process lifetime. Python's GC
            # (both automatic and manual gc.collect()) can race with those threads
            # and follow a pointer to a C++ object that onnxruntime has already
            # freed internally → deduce_unreachable → SIGSEGV. We rely on worker
            # recycling (process exit) to reclaim memory instead.
            _try_empty_torch_cache()

            tasks_processed += 1
            try:
                rss_gb = _worker_rss_gb()
            except Exception as exc:
                log(f"[worker] worker {worker_id} memory check failed: {exc}, recycling to be safe")
                result_queue.put({"type": "recycle", "worker_id": worker_id})
                return
            log(f"[worker] worker {worker_id} task={tasks_processed} rss={rss_gb:.1f}GB limit={DEFAULT_MAX_WORKER_MEMORY_GB or 'disabled'}GB")
            if DEFAULT_MAX_WORKER_MEMORY_GB > 0 and rss_gb >= DEFAULT_MAX_WORKER_MEMORY_GB:
                log_event("recycling", worker_id=worker_id, detail=f"memory limit reached: {rss_gb:.1f}GB >= {DEFAULT_MAX_WORKER_MEMORY_GB}GB")
                result_queue.put({"type": "recycle", "worker_id": worker_id})
                return
            if DEFAULT_MAX_TASKS_PER_WORKER > 0 and tasks_processed >= DEFAULT_MAX_TASKS_PER_WORKER:
                log_event("recycling", worker_id=worker_id, detail=f"after {tasks_processed} tasks, rss={rss_gb:.1f}GB")
                result_queue.put({"type": "recycle", "worker_id": worker_id})
                return
    except Exception as exc:
        result_queue.put(
            {
                "type": "startup_error",
                "worker_id": worker_id,
                "error": str(exc),
            }
        )
        raise


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants, regardless of process group.

    os.killpg() misses processes that escaped the group (e.g. Python's
    resource-tracker, or libraries that call setsid() internally).
    psutil walks the PID parent→child tree which is immune to that.
    """
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except Exception:
        # psutil unavailable or process already gone — fall back to direct kill.
        try:
            import signal
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        return

    # Kill leaves first so no child can re-spawn before parent dies.
    for child in reversed(children):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    # Reap to avoid zombies.
    try:
        psutil.wait_procs(children + [parent], timeout=3)
    except Exception:
        pass


class WorkerManager:
    def __init__(self, worker_count: int) -> None:
        self.worker_count = max(1, worker_count)
        self.task_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        # Keyed by worker_id so we can replace individual workers on recycle.
        self.processes: dict[int, mp.Process] = {}
        self._pending: dict[str, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        # During a recycle the respawned worker sends a "ready" message.
        # _result_loop routes it to the matching waiter here instead of
        # blocking result delivery for other tasks.
        self._ready_waiters: dict[int, queue.Queue] = {}
        self._ready_lock = threading.Lock()
        # Track which workers are currently being recycled to avoid duplicate restarts.
        self._recycling: set[int] = set()
        self._recycling_lock = threading.Lock()
        # Track which task each worker is currently processing.
        # When a worker crashes mid-task, we use this to fail the stuck waiter
        # so the gRPC handler thread is unblocked and _request_slots is released.
        self._worker_current_task: dict[int, str] = {}
        self._worker_task_lock = threading.Lock()
        self._result_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        if self.processes:
            return

        # _result_loop has not started yet so we can safely call result_queue.get()
        # directly here to wait for each worker's initial "ready" signal.
        for worker_id in range(self.worker_count):
            process = mp.Process(
                target=worker_process_main,
                args=(worker_id, self.task_queue, self.result_queue),
                name=f"mineru-worker-{worker_id}",
            )
            process.start()
            message = self.result_queue.get()
            if message.get("type") != "ready" or int(message.get("worker_id", -1)) != worker_id:
                process.join(timeout=1)
                raise RuntimeError(
                    f"worker {worker_id} failed during startup: {message.get('error') or message}"
                )
            log(f"[startup] worker {worker_id} ready")
            self.processes[worker_id] = process

        self._result_thread = threading.Thread(
            target=self._result_loop,
            name="mineru-worker-results",
            daemon=True,
        )
        self._result_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="mineru-worker-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _result_loop(self) -> None:
        while not self._closed:
            try:
                message = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                if self._closed:
                    return
                raise

            if not isinstance(message, dict):
                continue

            msg_type = message.get("type")

            if msg_type in ("ready", "startup_error"):
                # Route both success and failure back to whoever is waiting for
                # this worker to finish its prewarm (either initial startup or
                # after a recycle). startup_error previously fell through to the
                # "not in {result,error}" branch and was silently dropped, leaving
                # _recycle_worker blocked for 600 s with no retry.
                wid = int(message.get("worker_id", -1))
                with self._ready_lock:
                    waiter = self._ready_waiters.get(wid)
                if waiter is not None:
                    waiter.put(message)
                continue

            if msg_type == "started":
                wid = int(message.get("worker_id", -1))
                task_id = str(message.get("task_id") or "")
                if wid >= 0 and task_id:
                    with self._worker_task_lock:
                        self._worker_current_task[wid] = task_id
                continue

            if msg_type == "recycle":
                wid = int(message.get("worker_id", -1))
                self._trigger_recycle(wid, reason="worker requested")
                continue

            if msg_type not in {"result", "error"}:
                continue

            task_id = str(message.get("task_id") or "")
            if not task_id:
                continue
            wid = int(message.get("worker_id", -1))
            with self._worker_task_lock:
                self._worker_current_task.pop(wid, None)
            with self._pending_lock:
                waiter = self._pending.pop(task_id, None)
            if waiter is not None:
                waiter.put(message)

    def _trigger_recycle(self, worker_id: int, reason: str = "") -> None:
        with self._recycling_lock:
            if worker_id in self._recycling:
                return
            self._recycling.add(worker_id)
        threading.Thread(
            target=self._recycle_worker,
            args=(worker_id, reason),
            daemon=True,
            name=f"mineru-recycle-{worker_id}",
        ).start()

    def _watchdog_loop(self) -> None:
        while not self._closed:
            time.sleep(5)
            if self._closed:
                return
            for worker_id, process in list(self.processes.items()):
                if not process.is_alive():
                    with self._recycling_lock:
                        already = worker_id in self._recycling
                    if not already:
                        log(f"[watchdog] worker {worker_id} died unexpectedly (exitcode={process.exitcode}), restarting")
                        self._trigger_recycle(worker_id, reason=f"crashed exitcode={process.exitcode}")

    def _recycle_worker(self, worker_id: int, reason: str = "") -> None:
        if self._closed:
            return
        log(f"[worker] recycling worker {worker_id}" + (f" ({reason})" if reason else ""))
        # Fail any task this worker was processing so the gRPC handler thread
        # is unblocked and _request_slots is released immediately.
        with self._worker_task_lock:
            lost_task_id = self._worker_current_task.pop(worker_id, None)
        if lost_task_id:
            with self._pending_lock:
                waiter = self._pending.pop(lost_task_id, None)
            if waiter:
                log(f"[worker] worker {worker_id} crashed mid-task, failing task {lost_task_id}")
                waiter.put({"type": "error", "task_id": lost_task_id, "error": f"worker {worker_id} crashed mid-task"})
        try:
            old = self.processes.get(worker_id)
            if old is not None:
                # Kill the entire process tree via psutil (not killpg) so that
                # descendant processes that escaped the process group are also
                # cleaned up. SIGKILL skips Python's shutdown GC pass which
                # races with onnxruntime threads → SIGSEGV.
                if old.is_alive():
                    _kill_process_tree(old.pid)
                old.join(timeout=5)

            # Retry loop: on startup failure (network timeout, model download
            # error, etc.) retry with exponential back-off instead of giving up.
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                if self._closed:
                    return

                ready_waiter: queue.Queue = queue.Queue(maxsize=1)
                with self._ready_lock:
                    self._ready_waiters[worker_id] = ready_waiter

                process = mp.Process(
                    target=worker_process_main,
                    args=(worker_id, self.task_queue, self.result_queue),
                    name=f"mineru-worker-{worker_id}",
                )
                process.start()
                self.processes[worker_id] = process
                log(f"[worker] worker {worker_id} starting (attempt {attempt}/{max_attempts})")

                try:
                    msg = ready_waiter.get(timeout=600)
                except queue.Empty:
                    msg = None

                with self._ready_lock:
                    self._ready_waiters.pop(worker_id, None)

                if msg is not None and msg.get("type") == "ready":
                    log(f"[worker] worker {worker_id} ready")
                    return

                # Startup failed — kill the new process and decide whether to retry.
                error = msg.get("error", "unknown") if msg else "prewarm timeout"
                if attempt < max_attempts:
                    delay = min(10 * attempt, 60)
                    log(f"[worker] worker {worker_id} startup failed ({error}), retrying in {delay}s (attempt {attempt}/{max_attempts})")
                    if process.is_alive():
                        _kill_process_tree(process.pid)
                    process.join(timeout=5)
                    time.sleep(delay)
                else:
                    log(f"[worker] worker {worker_id} startup failed after {max_attempts} attempts ({error}), giving up")
                    if process.is_alive():
                        _kill_process_tree(process.pid)
                    process.join(timeout=5)

        except Exception as exc:
            log(f"[worker] worker {worker_id} recycle error: {exc}")
        finally:
            with self._ready_lock:
                self._ready_waiters.pop(worker_id, None)
            with self._recycling_lock:
                self._recycling.discard(worker_id)

    def submit(
        self,
        *,
        task_id: str,
        filename: str,
        file_content: bytes,
        profile: ParseProfile,
        timeout: float | None = None,
    ) -> bytes:
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[task_id] = waiter

        self.task_queue.put(
            {
                "task_id": task_id,
                "filename": filename,
                "file_content": file_content,
                "profile": profile,
            }
        )

        try:
            message = waiter.get(timeout=timeout)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(task_id, None)
            raise TimeoutError(f"worker timed out for task {task_id}") from exc

        if message["type"] == "error":
            raise RuntimeError(str(message.get("error") or "worker failed"))
        return bytes(message["response_bytes"])

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes.values():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self.processes.clear()


class RemoteWorkerPool:
    def __init__(self, targets: tuple[str, ...]) -> None:
        if not targets:
            raise ValueError("MINERU_REMOTE_WORKERS is empty")
        self.targets = targets
        self._lock = threading.Lock()
        self._in_flight: dict[str, int] = {t: 0 for t in targets}

    def _pick_target(self) -> str:
        with self._lock:
            return min(self.targets, key=lambda t: self._in_flight[t])

    def _fallback_order(self, exclude: str) -> list[str]:
        with self._lock:
            others = [t for t in self.targets if t != exclude]
            return sorted(others, key=lambda t: self._in_flight[t])

    def submit(self, request: mineru_pb2.ParsePdfRequest) -> mineru_pb2.ParsePdfResponse:
        target = self._pick_target()
        with self._lock:
            self._in_flight[target] += 1
        try:
            return self._send(request, target)
        except grpc.RpcError as exc:
            log(f"[scheduler] worker {target} failed, trying others: {exc.code()} {exc.details()}")
            with self._lock:
                self._in_flight[target] -= 1
            last_error: Exception = exc
            for fallback in self._fallback_order(exclude=target):
                with self._lock:
                    self._in_flight[fallback] += 1
                try:
                    return self._send(request, fallback)
                except grpc.RpcError as exc2:
                    log(f"[scheduler] worker {fallback} failed, trying next: {exc2.code()} {exc2.details()}")
                    with self._lock:
                        self._in_flight[fallback] -= 1
                    last_error = exc2
                    continue
            raise RuntimeError(f"all remote workers failed: {last_error}") from last_error

    def _send(self, request: mineru_pb2.ParsePdfRequest, target: str) -> mineru_pb2.ParsePdfResponse:
        try:
            with grpc.insecure_channel(
                target,
                options=[
                    ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE_BYTES),
                    ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE_BYTES),
                    ("grpc.keepalive_time_ms", DEFAULT_KEEPALIVE_TIME_MS),
                    ("grpc.keepalive_timeout_ms", DEFAULT_KEEPALIVE_TIMEOUT_MS),
                    ("grpc.keepalive_permit_without_calls", int(DEFAULT_KEEPALIVE_PERMIT_WITHOUT_CALLS)),
                    ("grpc.http2.max_pings_without_data", DEFAULT_HTTP2_MAX_PINGS_WITHOUT_DATA),
                ],
            ) as channel:
                stub = mineru_pb2_grpc.MineruPdfExtractorStub(channel)
                return stub.ParsePdf(request, timeout=DEFAULT_REMOTE_TIMEOUT_SECONDS)
        finally:
            with self._lock:
                self._in_flight[target] -= 1


class MineruService(mineru_pb2_grpc.MineruPdfExtractorServicer):
    def __init__(
        self,
        worker_manager: WorkerManager | None = None,
        remote_worker_pool: RemoteWorkerPool | None = None,
    ) -> None:
        self._request_slots = threading.Semaphore(max(1, DEFAULT_REQUEST_CONCURRENCY))
        self._worker_manager = worker_manager
        self._remote_worker_pool = remote_worker_pool

    def ParsePdf(self, request: mineru_pb2.ParsePdfRequest, context: grpc.ServicerContext) -> mineru_pb2.ParsePdfResponse:
        self._request_slots.acquire()
        task_id = uuid.uuid4().hex[:12]
        filename = request.filename or "input.pdf"
        log_event("received", task_id=task_id, filename=filename)
        profile = resolve_parse_profile(request)

        try:
            try:
                if self._remote_worker_pool is not None:
                    response = self._remote_worker_pool.submit(request)
                elif is_in_process_pipeline_enabled() and self._worker_manager is not None:
                    response_bytes = self._worker_manager.submit(
                        task_id=task_id,
                        filename=filename,
                        file_content=request.file_content,
                        profile=profile,
                    )
                    response = mineru_pb2.ParsePdfResponse()
                    response.ParseFromString(response_bytes)
                else:
                    raise RuntimeError(
                        "worker manager is not available; set MINERU_SERVER_BACKEND=pipeline "
                        "or MINERU_SERVER_EXECUTION_MODE=python"
                    )
            except Exception as exc:
                log_event("failed", task_id=task_id, filename=filename, detail=str(exc))
                context.abort(grpc.StatusCode.INTERNAL, str(exc))

            log_event("returned", task_id=task_id, filename=filename)
            return response
        finally:
            self._request_slots.release()


def serve() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    configure_logging()
    configure_mineru_runtime_env()
    worker_manager: WorkerManager | None = None
    remote_worker_pool: RemoteWorkerPool | None = None

    if is_scheduler_role():
        remote_worker_pool = RemoteWorkerPool(DEFAULT_REMOTE_WORKERS)
    else:
        worker_manager = WorkerManager(DEFAULT_WORKER_PROCESSES)
        worker_manager.start()

    server = grpc.server(
        ThreadPoolExecutor(max_workers=DEFAULT_GRPC_WORKERS),
        options=[
            ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE_BYTES),
            ("grpc.keepalive_time_ms", DEFAULT_KEEPALIVE_TIME_MS),
            ("grpc.keepalive_timeout_ms", DEFAULT_KEEPALIVE_TIMEOUT_MS),
            ("grpc.keepalive_permit_without_calls", int(DEFAULT_KEEPALIVE_PERMIT_WITHOUT_CALLS)),
            ("grpc.http2.max_pings_without_data", DEFAULT_HTTP2_MAX_PINGS_WITHOUT_DATA),
            (
                "grpc.http2.min_ping_interval_without_data_ms",
                DEFAULT_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS,
            ),
            ("grpc.http2.max_ping_strikes", DEFAULT_HTTP2_MAX_PING_STRIKES),
            ("grpc.max_connection_idle_ms", DEFAULT_MAX_CONNECTION_IDLE_MS),
            ("grpc.max_connection_age_ms", DEFAULT_MAX_CONNECTION_AGE_MS),
            ("grpc.max_connection_age_grace_ms", DEFAULT_MAX_CONNECTION_AGE_GRACE_MS),
        ],
    )
    mineru_pb2_grpc.add_MineruPdfExtractorServicer_to_server(
        MineruService(worker_manager, remote_worker_pool),
        server,
    )
    server.add_insecure_port(f"[::]:{DEFAULT_PORT}")
    if not is_scheduler_role() and DEFAULT_REQUEST_CONCURRENCY < DEFAULT_WORKER_PROCESSES:
        log(
            f"[main] WARNING: MINERU_SERVER_REQUEST_CONCURRENCY ({DEFAULT_REQUEST_CONCURRENCY}) "
            f"< MINERU_SERVER_WORKER_PROCESSES ({DEFAULT_WORKER_PROCESSES}). "
            f"{DEFAULT_WORKER_PROCESSES - DEFAULT_REQUEST_CONCURRENCY} worker(s) will be permanently idle. "
            f"Remove MINERU_SERVER_REQUEST_CONCURRENCY from your env to let it auto-match WORKER_PROCESSES."
        )

    log(
        f"[main] MinerU gRPC server listening on :{DEFAULT_PORT} "
        f"(chunk_size={'auto' if resolve_optional_chunk_size() is None else resolve_optional_chunk_size()}, "
        f"role={DEFAULT_ROLE or 'standalone'}, "
        f"max_workers={DEFAULT_MAX_WORKERS}, backend={DEFAULT_BACKEND}, method={DEFAULT_METHOD}, "
        f"execution_mode={'python' if is_in_process_pipeline_enabled() else 'cli'}, "
        f"worker_processes={max(1, DEFAULT_WORKER_PROCESSES)}, "
        f"max_tasks_per_worker={DEFAULT_MAX_TASKS_PER_WORKER or 'unlimited'}, "
        f"remote_workers={len(DEFAULT_REMOTE_WORKERS)}, "
        f"device={DEFAULT_DEVICE}, formula_enable={DEFAULT_FORMULA_ENABLE}, "
        f"table_enable={DEFAULT_TABLE_ENABLE}, table_merge_enable={DEFAULT_TABLE_MERGE_ENABLE}, "
        f"request_concurrency={max(1, DEFAULT_REQUEST_CONCURRENCY)}, "
        f"prewarm={DEFAULT_PREWARM_ENABLED}, "
        f"min_merged_text_len={DEFAULT_MIN_MERGED_TEXT_LEN}, "
        f"keepalive_time_ms={DEFAULT_KEEPALIVE_TIME_MS}, "
        f"keepalive_timeout_ms={DEFAULT_KEEPALIVE_TIMEOUT_MS}, "
        f"keepalive_permit_without_calls={DEFAULT_KEEPALIVE_PERMIT_WITHOUT_CALLS}, "
        f"http2_max_pings_without_data={DEFAULT_HTTP2_MAX_PINGS_WITHOUT_DATA}, "
        f"http2_min_ping_interval_without_data_ms={DEFAULT_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS}, "
        f"http2_max_ping_strikes={DEFAULT_HTTP2_MAX_PING_STRIKES}, "
        f"max_connection_idle_ms={DEFAULT_MAX_CONNECTION_IDLE_MS}, "
        f"max_connection_age_ms={DEFAULT_MAX_CONNECTION_AGE_MS}, "
        f"max_connection_age_grace_ms={DEFAULT_MAX_CONNECTION_AGE_GRACE_MS})"
    )
    server.start()
    try:
        server.wait_for_termination()
    finally:
        if worker_manager is not None:
            worker_manager.stop()


if __name__ == "__main__":
    serve()
