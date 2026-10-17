#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2E_ROOT="${R2E_GYM_ROOT:-${PROJECT_ROOT}/R2E-Gym}"
PYTHON_BIN="${R2E_ROOT}/.venv/bin/python"

export R2E_GYM_ROOT="${R2E_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${R2E_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

test -x "${PYTHON_BIN}"
docker info >/dev/null
"${PYTHON_BIN}" "${PROJECT_ROOT}/trace_generator.py" --help >/dev/null
"${PYTHON_BIN}" -m unittest discover -s "${PROJECT_ROOT}/tests" -v

echo "VM environment, Docker access, CLI, and local tests: OK"
