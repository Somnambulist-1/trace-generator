"""Process-local instrumentation for R2E-Gym sandbox lifecycle events.

The hooks are installed only in trace-generator worker processes.  R2E-Gym's
source tree is not modified.  Each process writes its own JSONL shard so that
concurrent workers never contend on a shared output file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_CONTEXT: dict[str, Any] = {}
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_SEQUENCE = 0
_ACTIVE: dict[tuple[str, str], str] = {}


def configure_trace(
    parts_dir: str,
    *,
    task_id: str,
    trajectory_id: str,
    stage: str,
    test_id: str | None = None,
) -> None:
    """Set trace metadata for the current worker and its forked children."""
    global _CONTEXT
    _CONTEXT = {
        "parts_dir": str(Path(parts_dir).resolve()),
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "stage": stage,
        "test_id": test_id,
    }
    # R2E-Gym's reproduction verifier forks a timeout worker.  Environment
    # variables make the context visible there even if globals are reloaded.
    for key, value in _CONTEXT.items():
        env_key = f"R2E_TRACE_{key.upper()}"
        if value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = str(value)


def _context() -> dict[str, Any]:
    ctx = dict(_CONTEXT)
    for key in ("parts_dir", "task_id", "trajectory_id", "stage", "test_id"):
        if not ctx.get(key):
            ctx[key] = os.getenv(f"R2E_TRACE_{key.upper()}")
    return ctx


def emit(event: str, **fields: Any) -> None:
    """Append one event to the current process's JSONL shard."""
    global _SEQUENCE
    ctx = _context()
    parts_dir = ctx.get("parts_dir")
    if not parts_dir:
        return

    _SEQUENCE += 1
    now_ns = time.time_ns()
    record = {
        "timestamp": datetime.fromtimestamp(
            now_ns / 1_000_000_000, tz=timezone.utc
        ).isoformat(),
        "timestamp_ns": now_ns,
        "task_id": ctx.get("task_id"),
        "worker_id": os.getpid(),
        "trajectory_id": ctx.get("trajectory_id"),
        "stage": ctx.get("stage"),
        "event": event,
        "sandbox_id": fields.pop("sandbox_id", None),
        "sequence": _SEQUENCE,
    }
    if ctx.get("test_id") is not None:
        record["test_id"] = ctx["test_id"]
    record.update({key: value for key, value in fields.items() if value is not None})

    shard_dir = Path(parts_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"trace-{os.getpid()}.jsonl"
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _docker_sandbox_id(container: Any) -> str | None:
    return getattr(container, "id", None) or getattr(container, "short_id", None)


def _install_docker_hooks() -> None:
    from docker.models.containers import Container, ContainerCollection

    if getattr(ContainerCollection.run, "_r2e_trace_hook", False):
        return

    original_run = ContainerCollection.run
    original_remove = Container.remove

    def traced_run(collection: Any, image: str, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        container = original_run(collection, image, *args, **kwargs)
        sandbox_id = _docker_sandbox_id(container)
        sandbox_name = kwargs.get("name") or getattr(container, "name", None)
        key = ("docker", sandbox_id or sandbox_name or "unknown")
        _ACTIVE[key] = sandbox_id or sandbox_name or "unknown"
        emit(
            "sandbox_create",
            sandbox_id=sandbox_id,
            sandbox_name=sandbox_name,
            backend="docker",
            image=str(image),
            duration_seconds=time.perf_counter() - started,
            outcome="container_running",
            source_api="docker.containers.run",
        )
        return container

    def traced_remove(container: Any, *args: Any, **kwargs: Any):
        sandbox_id = _docker_sandbox_id(container)
        sandbox_name = getattr(container, "name", None)
        key = ("docker", sandbox_id or sandbox_name or "unknown")
        started = time.perf_counter()
        result = original_remove(container, *args, **kwargs)
        if key in _ACTIVE:
            emit(
                "sandbox_destroy",
                sandbox_id=sandbox_id,
                sandbox_name=sandbox_name,
                backend="docker",
                duration_seconds=time.perf_counter() - started,
                outcome="container_removed",
                source_api="docker.container.remove",
            )
            _ACTIVE.pop(key, None)
        return result

    traced_run._r2e_trace_hook = True  # type: ignore[attr-defined]
    traced_remove._r2e_trace_hook = True  # type: ignore[attr-defined]
    ContainerCollection.run = traced_run
    Container.remove = traced_remove


def _kubernetes_identity(body: Any = None, response: Any = None, name: str | None = None):
    metadata = getattr(response, "metadata", None)
    if metadata is None and isinstance(response, dict):
        metadata = response.get("metadata", {})
    if metadata is None and body is not None:
        metadata = getattr(body, "metadata", None)
        if metadata is None and isinstance(body, dict):
            metadata = body.get("metadata", {})

    if isinstance(metadata, dict):
        uid = metadata.get("uid")
        pod_name = metadata.get("name")
    else:
        uid = getattr(metadata, "uid", None)
        pod_name = getattr(metadata, "name", None)
    return uid, pod_name or name


def _install_kubernetes_hooks() -> None:
    from kubernetes.client import CoreV1Api

    if getattr(CoreV1Api.create_namespaced_pod, "_r2e_trace_hook", False):
        return

    original_create = CoreV1Api.create_namespaced_pod
    original_delete = CoreV1Api.delete_namespaced_pod

    def traced_create(api: Any, namespace: str, body: Any, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        response = original_create(api, namespace, body, *args, **kwargs)
        uid, pod_name = _kubernetes_identity(body=body, response=response)
        key = ("kubernetes", pod_name or uid or "unknown")
        _ACTIVE[key] = uid or pod_name or "unknown"
        image = None
        try:
            spec = body.get("spec", {}) if isinstance(body, dict) else body.spec
            containers = spec.get("containers", []) if isinstance(spec, dict) else spec.containers
            first = containers[0]
            image = first.get("image") if isinstance(first, dict) else first.image
        except (AttributeError, IndexError, TypeError):
            pass
        emit(
            "sandbox_create",
            sandbox_id=uid or pod_name,
            sandbox_name=pod_name,
            backend="kubernetes",
            namespace=namespace,
            image=image,
            duration_seconds=time.perf_counter() - started,
            outcome="pod_create_accepted",
            source_api="kubernetes.create_namespaced_pod",
        )
        return response

    def traced_delete(api: Any, name: str, namespace: str, *args: Any, **kwargs: Any):
        key = ("kubernetes", name)
        sandbox_id = _ACTIVE.get(key, name)
        started = time.perf_counter()
        response = original_delete(api, name, namespace, *args, **kwargs)
        if key in _ACTIVE:
            emit(
                "sandbox_destroy",
                sandbox_id=sandbox_id,
                sandbox_name=name,
                backend="kubernetes",
                namespace=namespace,
                duration_seconds=time.perf_counter() - started,
                outcome="pod_delete_accepted",
                source_api="kubernetes.delete_namespaced_pod",
            )
            _ACTIVE.pop(key, None)
        return response

    traced_create._r2e_trace_hook = True  # type: ignore[attr-defined]
    traced_delete._r2e_trace_hook = True  # type: ignore[attr-defined]
    CoreV1Api.create_namespaced_pod = traced_create
    CoreV1Api.delete_namespaced_pod = traced_delete


def _install_tool_hook() -> None:
    """Trace each high-level agent tool invocation.

    R2E-Gym dispatches parsed actions through ``RepoEnv.run_action``.  The
    lower-level Docker command hook remains useful for shell-command detail,
    while this hook records one duration for every tool selected by the agent.
    """
    from r2egym.agenthub.environment.env import RepoEnv

    if getattr(RepoEnv.run_action, "_r2e_trace_hook", False):
        return

    original_run_action = RepoEnv.run_action

    def traced_tool(environment: Any, action: Any, *args: Any, **kwargs: Any):
        tool_name = getattr(action, "function_name", None)
        if not tool_name:
            return original_run_action(environment, action, *args, **kwargs)
        started = time.perf_counter()
        try:
            result = original_run_action(environment, action, *args, **kwargs)
        except BaseException as exc:
            emit(
                "tool",
                tool_name=str(tool_name),
                duration_seconds=time.perf_counter() - started,
                outcome="error",
                error=repr(exc),
            )
            raise

        exit_code = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        emit(
            "tool",
            tool_name=str(tool_name),
            duration_seconds=time.perf_counter() - started,
            exit_code=exit_code,
            outcome="ok" if exit_code in (None, 0, "0") else "error",
        )
        return result

    traced_tool._r2e_trace_hook = True  # type: ignore[attr-defined]
    RepoEnv.run_action = traced_tool


def _install_command_hook() -> None:
    from r2egym.agenthub.runtime.docker import DockerRuntime

    if getattr(DockerRuntime.run, "_r2e_trace_hook", False):
        return
    original_run = DockerRuntime.run

    def traced_command(runtime: Any, code: str, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            result = original_run(runtime, code, *args, **kwargs)
        except BaseException as exc:
            emit(
                "command",
                sandbox_id=_docker_sandbox_id(getattr(runtime, "container", None)),
                command=str(code)[:4096],
                duration_seconds=time.perf_counter() - started,
                exit_code=None,
                error=repr(exc),
            )
            raise
        exit_code = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        emit(
            "command",
            sandbox_id=_docker_sandbox_id(getattr(runtime, "container", None)),
            command=str(code)[:4096],
            duration_seconds=time.perf_counter() - started,
            exit_code=exit_code,
        )
        return result

    traced_command._r2e_trace_hook = True  # type: ignore[attr-defined]
    DockerRuntime.run = traced_command


def install_instrumentation(*, trace_commands: bool = True) -> None:
    """Install exact backend lifecycle hooks in the current process."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        _install_docker_hooks()
        _install_kubernetes_hooks()
        _install_tool_hook()
        if trace_commands:
            _install_command_hook()
        _INSTALLED = True

