#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2E_ROOT="${R2E_GYM_ROOT:-${PROJECT_ROOT}/R2E-Gym}"
PYTHON_BIN="${R2E_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: Python environment missing; run ./setup_vm.sh first." >&2
  exit 1
fi

export R2E_GYM_ROOT="${R2E_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${R2E_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Override the /beta endpoint used by the pinned LiteLLM release.
export DEEPSEEK_API_BASE="https://api.deepseek.com"

if (( $# == 0 )); then
  set -- --help
fi
for argument in "$@"; do
  if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
    echo "DeepSeek V4.1 Flash: deepseek/deepseek-flash"
    echo "Set DEEPSEEK_API_KEY, or enter it at the hidden terminal prompt when starting a run."
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/trace_generator.py" --help
  fi
done

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "error: set DEEPSEEK_API_KEY, or run this script in an interactive terminal to enter it." >&2
    exit 1
  fi
  read -rsp "DeepSeek API key (hidden): " DEEPSEEK_API_KEY
  printf '\n'
  if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
    echo "error: DeepSeek API key cannot be empty." >&2
    exit 1
  fi
fi
export DEEPSEEK_API_KEY
export R2E_DEEPSEEK_NATIVE_TOOL_CALLING=1

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/trace_generator.py" \
  --r2e-root "${R2E_ROOT}" \
  --llm-name deepseek/deepseek-flash \
  --api-key-env DEEPSEEK_API_KEY \
  "$@"
