#!/usr/bin/env python3
"""Parallel gRPC load test for the MinerU server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import statistics
import time

import grpc

import mineru_pb2
import mineru_pb2_grpc


DEFAULT_TARGET = "127.0.0.1:50051"
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT = 3600.0


@dataclass
class RequestResult:
    label: str
    filename: str
    started_at: float
    ended_at: float
    elapsed_seconds: float
    page_count: int
    paragraph_count: int
    full_text_len: int


def format_ts(timestamp: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fire parallel ParsePdf gRPC requests against the MinerU server."
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        type=Path,
        help="One or more PDF paths. Pass the same file multiple times or use --repeat to duplicate load.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"gRPC target host:port, default: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent requests, default: {DEFAULT_CONCURRENCY}",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the input PDF list this many times before sending requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds, default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument("--language-hint", default="ch", help="ParsePdf.language_hint")
    parser.add_argument("--layout-hint", default="", help="ParsePdf.layout_hint")
    parser.add_argument("--document-type-hint", default="", help="ParsePdf.document_type_hint")
    parser.add_argument(
        "--model-version",
        default="ocr",
        help="ParsePdf.model_version. In this server it maps to parse method like auto/txt/ocr.",
    )
    return parser.parse_args()


def expand_inputs(paths: list[Path], repeat: int) -> list[Path]:
    expanded: list[Path] = []
    for _ in range(max(1, repeat)):
        expanded.extend(paths)
    return expanded


def validate_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PDF not found: {missing[0]}")


def run_single_request(
    *,
    target: str,
    pdf_path: Path,
    request_index: int,
    timeout: float,
    language_hint: str,
    layout_hint: str,
    document_type_hint: str,
    model_version: str,
) -> RequestResult:
    file_bytes = pdf_path.read_bytes()
    label = f"req-{request_index:02d}"
    started_at = time.time()
    print(
        f"[{format_ts(started_at)}] {label} start "
        f"file={pdf_path.name} bytes={len(file_bytes)}",
        flush=True,
    )

    with grpc.insecure_channel(target) as channel:
        stub = mineru_pb2_grpc.MineruPdfExtractorStub(channel)
        response = stub.ParsePdf(
            mineru_pb2.ParsePdfRequest(
                filename=pdf_path.name,
                file_content=file_bytes,
                language_hint=language_hint,
                layout_hint=layout_hint,
                document_type_hint=document_type_hint,
                model_version=model_version,
            ),
            timeout=timeout,
        )

    ended_at = time.time()
    elapsed_seconds = ended_at - started_at
    page_count = len(response.document.pages)
    paragraph_count = sum(len(page.paragraphs) for page in response.document.pages)
    full_text_len = len(response.document.full_text or "")
    print(
        f"[{format_ts(ended_at)}] {label} done "
        f"file={pdf_path.name} elapsed={elapsed_seconds:.2f}s "
        f"pages={page_count} paragraphs={paragraph_count} text_len={full_text_len}",
        flush=True,
    )
    return RequestResult(
        label=label,
        filename=pdf_path.name,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed_seconds,
        page_count=page_count,
        paragraph_count=paragraph_count,
        full_text_len=full_text_len,
    )


def summarize(results: list[RequestResult], total_started_at: float, total_ended_at: float) -> None:
    elapsed_values = [item.elapsed_seconds for item in results]
    wall_clock = total_ended_at - total_started_at
    serial_estimate = sum(elapsed_values)
    overlap_gain = serial_estimate / wall_clock if wall_clock > 0 else 0.0

    print("\n=== Summary ===", flush=True)
    print(f"request_count: {len(results)}", flush=True)
    print(f"wall_clock_seconds: {wall_clock:.2f}", flush=True)
    print(f"sum_of_request_seconds: {serial_estimate:.2f}", flush=True)
    print(f"estimated_overlap_factor: {overlap_gain:.2f}x", flush=True)
    print(f"min_request_seconds: {min(elapsed_values):.2f}", flush=True)
    print(f"median_request_seconds: {statistics.median(elapsed_values):.2f}", flush=True)
    print(f"max_request_seconds: {max(elapsed_values):.2f}", flush=True)
    print("\n=== Per Request ===", flush=True)
    for item in sorted(results, key=lambda value: value.started_at):
        print(
            f"{item.label} file={item.filename} "
            f"start={format_ts(item.started_at)} end={format_ts(item.ended_at)} "
            f"elapsed={item.elapsed_seconds:.2f}s pages={item.page_count}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if args.repeat <= 0:
        raise ValueError("--repeat must be greater than 0")

    pdf_paths = expand_inputs(args.pdfs, args.repeat)
    validate_inputs(pdf_paths)

    print("=== MinerU gRPC Parallel Test ===", flush=True)
    print(f"target: {args.target}", flush=True)
    print(f"input_count: {len(pdf_paths)}", flush=True)
    print(f"concurrency: {args.concurrency}", flush=True)
    print(f"timeout_seconds: {args.timeout}", flush=True)
    print(f"model_version: {args.model_version}", flush=True)
    print("", flush=True)

    total_started_at = time.time()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_single_request,
                target=args.target,
                pdf_path=pdf_path,
                request_index=index,
                timeout=args.timeout,
                language_hint=args.language_hint,
                layout_hint=args.layout_hint,
                document_type_hint=args.document_type_hint,
                model_version=args.model_version,
            )
            for index, pdf_path in enumerate(pdf_paths, start=1)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    total_ended_at = time.time()

    summarize(results, total_started_at, total_ended_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
