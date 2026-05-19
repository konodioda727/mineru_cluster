#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${SCRIPT_DIR}/mineru_grpc_server_standalone.py"
PROTO_TEMPLATE="${SCRIPT_DIR}/../protos/mineru.proto"
PROTO_FILE="${SCRIPT_DIR}/mineru.proto"
VENV_DIR="${SCRIPT_DIR}/.mineru_grpc_venv"
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="python3.13"
  else
    PYTHON_BIN="python3"
  fi
fi
AUTO_INSTALL="${AUTO_INSTALL:-1}"
SERVER_ARGS=()

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 [--env PATH] [--help]

Options:
  -e, --env PATH   Load environment variables from the specified file
  -h, --help       Show this help message
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"
}

setup_proto() {
  if [[ ! -f "${PROTO_FILE}" ]]; then
    cp "${PROTO_TEMPLATE}" "${PROTO_FILE}"
    log "Copied mineru.proto to ${PROTO_FILE}"
  fi
}

setup_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  export PATH="${VENV_DIR}/bin:${PATH}"
  export MINERU_PYTHON_BIN="${VENV_DIR}/bin/python"
  export MINERU_CLI="${VENV_DIR}/bin/mineru"

  [[ -x "${MINERU_PYTHON_BIN}" ]] || fail "Python not found in venv: ${MINERU_PYTHON_BIN}"

  if [[ "${AUTO_INSTALL}" == "1" ]]; then
    log "Installing runtime dependencies"
    "${MINERU_PYTHON_BIN}" -m pip install -U pip wheel

    # Install mineru first so it can pin its own torch/paddle versions.
    # Do NOT pass -U here: mineru has tight version constraints on torch and
    # paddlepaddle; force-upgrading them independently breaks the pipeline.
    "${MINERU_PYTHON_BIN}" -m pip install mineru

    # gRPC and PDF deps: upgrade is safe because they're independent of MinerU's ML stack.
    "${MINERU_PYTHON_BIN}" -m pip install -U grpcio grpcio-tools pymupdf
  fi
}

generate_grpc_stubs() {
  log "Generating gRPC stubs from ${PROTO_FILE}"
  "${MINERU_PYTHON_BIN}" -m grpc_tools.protoc \
    -I "${SCRIPT_DIR}" \
    --python_out="${SCRIPT_DIR}" \
    --grpc_python_out="${SCRIPT_DIR}" \
    "${PROTO_FILE}"
}

load_env_file() {
  if [[ -f "${ENV_FILE}" ]]; then
    log "Loading env from ${ENV_FILE}"
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  else
    log "No .env found at ${ENV_FILE}, using current environment"
  fi
}

check_runtime() {
  [[ -x "${MINERU_CLI}" ]] || fail "MinerU CLI not found in venv: ${MINERU_CLI}"
  [[ -f "${SERVER_SCRIPT}" ]] || fail "Server script not found: ${SERVER_SCRIPT}"
  [[ -f "${PROTO_FILE}" ]] || fail "Proto file not found: ${PROTO_FILE}"
}

print_effective_config() {
  log "Starting MinerU gRPC server with config:"
  log "  ENV_FILE=${ENV_FILE}"
  log "  MINERU_SERVER_PORT=${MINERU_SERVER_PORT:-50051}"
  log "  MINERU_SERVER_MAX_WORKERS=${MINERU_SERVER_MAX_WORKERS:-3}"
  log "  MINERU_SERVER_CHUNK_SIZE=${MINERU_SERVER_CHUNK_SIZE:-auto}"
  log "  MINERU_SERVER_BACKEND=${MINERU_SERVER_BACKEND:-pipeline}"
  log "  MINERU_SERVER_METHOD=${MINERU_SERVER_METHOD:-auto}"
  log "  MINERU_SERVER_DEVICE=${MINERU_SERVER_DEVICE:-mps}"
  log "  MINERU_SERVER_ROLE=${MINERU_SERVER_ROLE:-standalone}"
  log "  MINERU_SERVER_EXECUTION_MODE=${MINERU_SERVER_EXECUTION_MODE:-auto}"
  log "  MINERU_SERVER_WORKER_PROCESSES=${MINERU_SERVER_WORKER_PROCESSES:-3}"
  log "  MINERU_SERVER_REQUEST_CONCURRENCY=${MINERU_SERVER_REQUEST_CONCURRENCY:-3}"
  log "  MINERU_REMOTE_WORKERS=${MINERU_REMOTE_WORKERS:-}"
  log "  MINERU_SERVER_PREWARM=${MINERU_SERVER_PREWARM:-true}"
  log "  MINERU_SERVER_PREWARM_METHOD=${MINERU_SERVER_PREWARM_METHOD:-ocr}"
  log "  MINERU_SERVER_FORMULA_ENABLE=${MINERU_SERVER_FORMULA_ENABLE:-false}"
  log "  MINERU_SERVER_TABLE_ENABLE=${MINERU_SERVER_TABLE_ENABLE:-true}"
  log "  MINERU_SERVER_TABLE_MERGE_ENABLE=${MINERU_SERVER_TABLE_MERGE_ENABLE:-true}"
  log "  MINERU_SERVER_MAX_MESSAGE_BYTES=${MINERU_SERVER_MAX_MESSAGE_BYTES:-268435456}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -e|--env)
        [[ $# -ge 2 ]] || fail "Missing value for $1"
        ENV_FILE="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        SERVER_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

main() {
  parse_args "$@"
  require_command "${PYTHON_BIN}"

  if ! "${PYTHON_BIN}" - <<'EOF'
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)
EOF
  then
    fail "Unsupported Python for MinerU: $("${PYTHON_BIN}" --version 2>&1). Please use Python 3.10-3.13."
  fi

  setup_proto
  setup_venv
  generate_grpc_stubs
  load_env_file
  check_runtime
  print_effective_config

  cd "${SCRIPT_DIR}"
  if [[ ${#SERVER_ARGS[@]} -gt 0 ]]; then
    exec "${MINERU_PYTHON_BIN}" "${SERVER_SCRIPT}" "${SERVER_ARGS[@]}"
  fi
  exec "${MINERU_PYTHON_BIN}" "${SERVER_SCRIPT}"
}

main "$@"
