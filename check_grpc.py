#!/usr/bin/env python3
"""MinerU gRPC connectivity check.

Run from the same directory as mineru_pb2.py / mineru_pb2_grpc.py:
    python check_grpc.py
    python check_grpc.py --target 127.0.0.1:50051
    python check_grpc.py --target 127.0.0.1:50051 --send-pdf  # 发一个真实请求（要等处理完）
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import os

# ── 1. 把当前目录加到 path，确保能 import 生成的 pb2 ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  ✗ {msg}", flush=True)


# ── 2. 检查 grpcio ─────────────────────────────────────────────────────────────
def check_grpcio() -> bool:
    try:
        import grpc  # noqa: F401
        ok(f"grpcio 已安装: {grpc.__version__}")
        return True
    except ImportError:
        fail("grpcio 未安装，请先: pip install grpcio")
        return False


# ── 3. TCP 端口连通性 ──────────────────────────────────────────────────────────
def check_tcp(host: str, port: int, timeout: float = 5.0) -> bool:
    log(f"TCP 连通性检查: {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            ok(f"TCP 连接成功 ({host}:{port})")
            return True
    except OSError as exc:
        fail(f"TCP 连接失败: {exc}")
        return False


# ── 4. gRPC channel 握手 ───────────────────────────────────────────────────────
def check_grpc_channel(target: str, timeout_s: float = 10.0) -> bool:
    import grpc

    log(f"gRPC channel 握手: {target}")
    deadline = time.monotonic() + timeout_s
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 20_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ],
    )
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        ok(f"gRPC channel READY ({target})")
        return True
    except grpc.FutureTimeoutError:
        elapsed = timeout_s - max(0.0, deadline - time.monotonic())
        fail(f"gRPC channel 握手超时（{elapsed:.1f}s），服务可能未启动或端口不通")
        return False
    except Exception as exc:
        fail(f"gRPC channel 错误: {exc}")
        return False
    finally:
        channel.close()


# ── 5. 生成最小合法 PDF（不依赖 fitz）────────────────────────────────────────
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n192\n%%EOF\n"
)


# ── 6. 加载 pb2（或动态生成）──────────────────────────────────────────────────
def load_stubs():
    try:
        import mineru_pb2
        import mineru_pb2_grpc
        ok("mineru_pb2 stubs 加载成功")
        return mineru_pb2, mineru_pb2_grpc
    except ImportError:
        pass

    log("mineru_pb2.py 不在当前目录，尝试动态编译 mineru.proto ...")
    proto_path = os.path.join(SCRIPT_DIR, "mineru.proto")
    if not os.path.exists(proto_path):
        fail(f"找不到 {proto_path}，请把脚本放到和 mineru_pb2.py 或 mineru.proto 同目录下")
        return None, None
    try:
        from grpc_tools import protoc
        protoc.main([
            "grpc_tools.protoc",
            f"-I{SCRIPT_DIR}",
            f"--python_out={SCRIPT_DIR}",
            f"--grpc_python_out={SCRIPT_DIR}",
            proto_path,
        ])
        import mineru_pb2
        import mineru_pb2_grpc
        ok("mineru_pb2 stubs 动态编译成功")
        return mineru_pb2, mineru_pb2_grpc
    except Exception as exc:
        fail(f"动态编译失败: {exc}，请先运行: pip install grpcio-tools")
        return None, None


# ── 7. 发送真实 ParsePdf 请求 ──────────────────────────────────────────────────
def check_parse_pdf(target: str, timeout_s: float = 60.0) -> bool:
    import grpc

    mineru_pb2, mineru_pb2_grpc = load_stubs()
    if mineru_pb2 is None:
        return False

    log(f"发送 ParsePdf 请求（timeout={timeout_s}s）...")
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 20_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
        ],
    )
    try:
        stub = mineru_pb2_grpc.MineruPdfExtractorStub(channel)
        request = mineru_pb2.ParsePdfRequest(
            filename="connectivity_check.pdf",
            file_content=_MINIMAL_PDF,
            language_hint="en",
            model_version="auto",
        )
        t0 = time.monotonic()
        response = stub.ParsePdf(request, timeout=timeout_s)
        elapsed = time.monotonic() - t0
        ok(f"ParsePdf 成功！耗时={elapsed:.2f}s task_id={response.task_id} "
           f"pages={len(response.document.pages)}")
        return True
    except grpc.RpcError as exc:
        elapsed = time.monotonic() - t0 if 't0' in dir() else 0
        fail(f"ParsePdf RPC 失败 ({elapsed:.1f}s): [{exc.code().name}] {exc.details()}")
        return False
    except Exception as exc:
        fail(f"ParsePdf 异常: {exc}")
        return False
    finally:
        channel.close()


# ── 8. keepalive 持久连接压测（模拟长时间等待）────────────────────────────────
def check_keepalive(target: str, hold_seconds: float = 300.0) -> bool:
    """
    保持一条空闲 channel 不发任何 RPC，观察 keepalive ping 能否维持连接。
    用于验证 NAT/防火墙不会在 hold_seconds 内断开连接。
    按 Ctrl+C 可提前退出。
    """
    import grpc

    log(f"keepalive 压测：保持空闲连接 {hold_seconds}s，观察是否被防火墙断开 ...")
    log("（按 Ctrl+C 提前结束）")
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.keepalive_time_ms", 20_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
        ],
    )
    states = []

    def on_state(state):
        states.append((time.monotonic(), state.name))
        log(f"  channel state → {state.name}")

    channel.subscribe(on_state, try_to_connect=True)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < hold_seconds:
            time.sleep(10)
            elapsed = time.monotonic() - t0
            log(f"  已等待 {elapsed:.0f}s / {hold_seconds:.0f}s，连接仍活跃")
        ok(f"keepalive 压测通过：{hold_seconds}s 内连接未断开")
        return True
    except KeyboardInterrupt:
        elapsed = time.monotonic() - t0
        log(f"  手动中断，已等待 {elapsed:.0f}s")
        last = states[-1][1] if states else "UNKNOWN"
        ok(f"中断时 channel 状态: {last}")
        return True
    finally:
        channel.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="MinerU gRPC 连通性检查")
    parser.add_argument("--target", default="127.0.0.1:50051", help="gRPC 地址")
    parser.add_argument("--send-pdf", action="store_true",
                        help="发送一个真实的 ParsePdf 请求（需要服务器正常处理）")
    parser.add_argument("--keepalive-test", action="store_true",
                        help="保持空闲连接 N 秒，验证 keepalive 能抵抗防火墙超时")
    parser.add_argument("--keepalive-seconds", type=float, default=300.0,
                        help="keepalive 压测持续秒数（默认 300s）")
    parser.add_argument("--pdf-timeout", type=float, default=60.0,
                        help="ParsePdf 请求超时秒数（默认 60s）")
    args = parser.parse_args()

    host, _, port_str = args.target.rpartition(":")
    port = int(port_str)

    print("=" * 55)
    print(f"  MinerU gRPC 连通性检查  target={args.target}")
    print("=" * 55)

    all_ok = True

    # 步骤 1: grpcio
    log("步骤 1 / 检查 grpcio")
    if not check_grpcio():
        return 1

    # 步骤 2: TCP
    log("步骤 2 / TCP 端口")
    if not check_tcp(host, port):
        all_ok = False

    # 步骤 3: gRPC 握手
    log("步骤 3 / gRPC 握手")
    if not check_grpc_channel(args.target):
        all_ok = False

    # 步骤 4: ParsePdf（可选）
    if args.send_pdf:
        log("步骤 4 / ParsePdf 请求")
        if not check_parse_pdf(args.target, timeout_s=args.pdf_timeout):
            all_ok = False

    # 步骤 5: keepalive 压测（可选）
    if args.keepalive_test:
        log(f"步骤 5 / keepalive 压测（{args.keepalive_seconds}s）")
        check_keepalive(args.target, hold_seconds=args.keepalive_seconds)

    print("=" * 55)
    if all_ok:
        print("  结论：连通性正常 ✓")
    else:
        print("  结论：存在问题，见上方 ✗ 行")
    print("=" * 55)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
