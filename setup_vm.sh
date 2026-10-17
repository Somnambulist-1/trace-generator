#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2E_ROOT="${R2E_GYM_ROOT:-${PROJECT_ROOT}/R2E-Gym}"
EXPECTED_R2E_COMMIT="0d94c4eb9431cd195c55a7ea3abd54006c9a1735"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: setup_vm.sh must run inside the Linux VM" >&2
  exit 1
fi

for command_name in curl git; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "error: missing required command: ${command_name}" >&2
    exit 1
  fi
done

if [[ ! -d "${R2E_ROOT}/src/r2egym" ]]; then
  echo "R2E-Gym checkout not found; cloning the tested revision..."
  git clone https://github.com/R2E-Gym/R2E-Gym.git "${R2E_ROOT}"
  git -C "${R2E_ROOT}" checkout "${EXPECTED_R2E_COMMIT}"
elif [[ -d "${R2E_ROOT}/.git" ]]; then
  actual_commit="$(git -C "${R2E_ROOT}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${EXPECTED_R2E_COMMIT}" ]]; then
    echo "error: R2E-Gym is at ${actual_commit}; expected ${EXPECTED_R2E_COMMIT}" >&2
    exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv for the current VM user..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# The pinned dependency set has wheels for Python 3.10--3.13.  Do not let uv
# select a newer host interpreter (for example 3.14), because packages such as
# regex may then fall back to a source build and require Python headers.
R2E_PYTHON_BIN="${R2E_PYTHON_BIN:-}"
if [[ -z "${R2E_PYTHON_BIN}" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      R2E_PYTHON_BIN="$(command -v "${candidate}")"
      break
    fi
  done
fi

if [[ -z "${R2E_PYTHON_BIN}" ]]; then
  echo "error: R2E-Gym requires Python 3.10, 3.11, 3.12, or 3.13." >&2
  echo "Install one of those interpreters (or set R2E_PYTHON_BIN) and rerun." >&2
  exit 1
fi

python_version="$(${R2E_PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${python_version}" in
  3.10|3.11|3.12|3.13) ;;
  *)
    echo "error: unsupported Python ${python_version} at ${R2E_PYTHON_BIN}; use Python 3.10-3.13." >&2
    exit 1
    ;;
esac

echo "Installing the exact R2E-Gym lockfile environment with Python ${python_version}..."
uv sync --locked --project "${R2E_ROOT}" --python "${R2E_PYTHON_BIN}"

PYTHON_BIN="${R2E_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: expected Python environment was not created at ${PYTHON_BIN}" >&2
  exit 1
fi

export R2E_GYM_ROOT="${R2E_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${R2E_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m unittest discover -s "${PROJECT_ROOT}/tests" -v
"${PYTHON_BIN}" - <<'PY'
from r2egym_trace.instrumentation import install_instrumentation

install_instrumentation(trace_commands=True)

from docker.models.containers import Container, ContainerCollection
from kubernetes.client import CoreV1Api
from r2egym.agenthub.runtime.docker import DockerRuntime

assert getattr(ContainerCollection.run, "_r2e_trace_hook", False)
assert getattr(Container.remove, "_r2e_trace_hook", False)
assert getattr(CoreV1Api.create_namespaced_pod, "_r2e_trace_hook", False)
assert getattr(CoreV1Api.delete_namespaced_pod, "_r2e_trace_hook", False)
assert getattr(DockerRuntime.run, "_r2e_trace_hook", False)
print("R2E-Gym imports and trace hooks: OK")
PY

if ! command -v docker >/dev/null 2>&1; then
  echo "error: Docker Engine is not installed in the VM." >&2
  echo "On Ubuntu/Debian, install it with: sudo apt-get update && sudo apt-get install -y docker.io" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "error: Docker Engine is installed but not accessible to this user." >&2
  echo "Start it with: sudo systemctl enable --now docker" >&2
  echo "Then grant access with: sudo usermod -aG docker \"${USER}\"" >&2
  echo "Log out and back in before rerunning this script." >&2
  exit 1
fi

echo
echo "VM setup complete."
echo "R2E_GYM_ROOT=${R2E_ROOT}"
echo "Python=${PYTHON_BIN}"
echo "Set OPENAI_API_KEY (or another provider key) only when you are ready to run."
