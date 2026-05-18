#!/usr/bin/env python3
"""Standalone local MinerU benchmark with auto-balanced chunking."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import fitz

DEFAULT_BACKEND = "pipeline"
DEFAULT_METHOD = "auto"
DEFAULT_DEVICE = "mps"
DEFAULT_FORMULA_ENABLE = False
DEFAULT_TABLE_ENABLE = True
DEFAULT_TABLE_MERGE_ENABLE = True


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def resolve_chunk_size(total_pages: int, max_workers: int, manual_chunk_size: int | None) -> int:
    if total_pages <= 0:
        return 1
    if manual_chunk_size is not None:
        return max(1, min(total_pages, manual_chunk_size))
    worker_count = max(1, min(total_pages, max_workers))
    return max(1, math.ceil(total_pages / worker_count))


def format_cli_bool(value: bool) -> str:
    return "true" if value else "false"


def process_chunk(
    chunk_path: str,
    output_dir: str,
    chunk_idx: int,
    start_page: int,
    backend: str,
    method: str,
    device: str,
    formula_enable: bool,
    table_enable: bool,
    table_merge_enable: bool,
) -> tuple[int, str | None]:
    chunk_name = os.path.basename(chunk_path)
    log(f"[chunk {chunk_idx}] start, original document begins at page {start_page}")

    cmd = [
        "mineru",
        "-p",
        chunk_path,
        "-o",
        output_dir,
        "-b",
        backend,
        "-m",
        method,
        "-d",
        device,
        "-f",
        format_cli_bool(formula_enable),
        "-t",
        format_cli_bool(table_enable),
    ]
    env = os.environ.copy()
    env["MINERU_FORMULA_ENABLE"] = format_cli_bool(formula_enable)
    env["MINERU_TABLE_ENABLE"] = format_cli_bool(table_enable)
    env["MINERU_TABLE_MERGE_ENABLE"] = format_cli_bool(table_merge_enable)

    log(f"[chunk {chunk_idx}] exec: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line:
            log(f"[chunk {chunk_idx}] {line}")

    process.wait()
    if process.returncode != 0:
        log(f"[chunk {chunk_idx}] failed with code {process.returncode}")
        return chunk_idx, None

    pdf_basename = os.path.splitext(chunk_name)[0]
    md_file = os.path.join(output_dir, pdf_basename, f"{pdf_basename}.md")
    if not os.path.exists(md_file):
        log(f"[chunk {chunk_idx}] markdown not found under {output_dir}/{pdf_basename}")
        return chunk_idx, None

    log(f"[chunk {chunk_idx}] done")
    return chunk_idx, md_file


def run_local_test(
    pdf_path: Path,
    chunk_size: int | None,
    max_workers: int,
    backend: str,
    method: str,
    device: str,
    formula_enable: bool,
    table_enable: bool,
    table_merge_enable: bool,
) -> int:
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    log("=" * 60)
    log("MinerU local benchmark started")
    log(f"target_pdf: {pdf_path}")
    log(f"chunk_size: {'auto' if chunk_size is None else chunk_size}")
    log(f"max_workers: {max_workers}")
    log(f"backend: {backend}")
    log(f"method: {method}")
    log(f"device: {device}")
    log(f"formula_enable: {formula_enable}")
    log(f"table_enable: {table_enable}")
    log(f"table_merge_enable: {table_merge_enable}")
    log("=" * 60)

    started_at = time.time()
    base_output_dir = Path("./mineru_local_output").resolve()
    final_md_path = Path("./FINAL_RESULT.md").resolve()

    if base_output_dir.exists():
        shutil.rmtree(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    log("splitting pdf...")
    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        resolved_chunk_size = resolve_chunk_size(total_pages, max_workers, chunk_size)
        log(
            f"split strategy: total_pages={total_pages}, "
            f"resolved_chunk_size={resolved_chunk_size}, "
            f"target_workers={min(total_pages, max_workers)}"
        )

        chunks = []
        for i in range(0, total_pages, resolved_chunk_size):
            chunk_doc = fitz.open()
            end_page = min(i + resolved_chunk_size - 1, total_pages - 1)
            chunk_doc.insert_pdf(doc, from_page=i, to_page=end_page)

            chunk_idx = i // resolved_chunk_size
            chunk_filename = f"chunk_{chunk_idx}.pdf"
            chunk_path_local = base_output_dir / chunk_filename
            chunk_doc.save(chunk_path_local)
            chunk_doc.close()

            chunks.append(
                {
                    "path": str(chunk_path_local),
                    "chunk_idx": chunk_idx,
                    "start_page": i + 1,
                }
            )
    finally:
        doc.close()

    log(f"split done: total_pages={total_pages}, chunk_count={len(chunks)}")
    log("launching mineru workers...")

    results: dict[int, str] = {}
    completed_chunks = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_chunk,
                chunk["path"],
                str(base_output_dir),
                chunk["chunk_idx"],
                chunk["start_page"],
                backend,
                method,
                device,
                formula_enable,
                table_enable,
                table_merge_enable,
            )
            for chunk in chunks
        ]

        for future in as_completed(futures):
            chunk_idx, md_file = future.result()
            completed_chunks += 1
            if md_file:
                results[chunk_idx] = md_file
                log(f"[main] chunk {chunk_idx} finished successfully ({completed_chunks}/{len(chunks)})")
            else:
                log(f"[main] chunk {chunk_idx} failed ({completed_chunks}/{len(chunks)})")

    log("all workers done, merging markdown...")
    with final_md_path.open("w", encoding="utf-8") as handle:
        for chunk_idx in sorted(results):
            with open(results[chunk_idx], "r", encoding="utf-8") as input_handle:
                handle.write("\n\n\n\n")
                handle.write(input_handle.read())

    elapsed = time.time() - started_at
    log(f"benchmark finished: {elapsed:.2f}s")
    log(f"merged_markdown: {final_md_path}")
    log(f"intermediate_output: {base_output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark local MinerU parsing with auto-balanced multiprocessing."
    )
    parser.add_argument("pdf", type=Path, help="Path to the local PDF file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional manual pages-per-chunk override; defaults to auto-balancing from max-workers",
    )
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum concurrent MinerU processes")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["pipeline", "hybrid-auto-engine", "hybrid-http-client", "vlm-auto-engine", "vlm-http-client"],
        help="MinerU backend",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["auto", "txt", "ocr"],
        help="MinerU parsing method",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Inference device passed to MinerU, e.g. mps/cpu/cuda:0",
    )
    parser.add_argument(
        "--formula-enable",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FORMULA_ENABLE,
        help="Enable formula parsing",
    )
    parser.add_argument(
        "--table-enable",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TABLE_ENABLE,
        help="Enable table parsing",
    )
    parser.add_argument(
        "--table-merge-enable",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TABLE_MERGE_ENABLE,
        help="Enable table merge",
    )

    args = parser.parse_args()
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be greater than 0")

    return run_local_test(
        args.pdf.expanduser().resolve(),
        args.chunk_size,
        args.max_workers,
        args.backend,
        args.method,
        args.device,
        args.formula_enable,
        args.table_enable,
        args.table_merge_enable,
    )


if __name__ == "__main__":
    raise SystemExit(main())
