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
import uuid
import hashlib
from functools import wraps
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_CONTEXT: dict[str, Any] = {}
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_SEQUENCE = 0
_ACTIVE: dict[tuple[str, str], str] = {}
_SCOPE: ContextVar[dict[str, Any]] = ContextVar("trace_scope", default={})


@contextmanager
def trace_scope(**fields: Any):
    """Attach phase/step/span metadata to nested operations, including forks."""
    token = _SCOPE.set({**_SCOPE.get(), **fields})
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextmanager
def timed_event(event: str, **fields: Any):
    """Measure one inclusive wall-time interval using the monotonic clock.

    The yielded mapping may be updated with results. Nested intervals retain
    parent IDs; their durations must not be added to their parent's duration.
    Logging happens after the end timestamp and is outside this interval.
    """
    parent = _SCOPE.get().get("span_id")
    span_id = f"{os.getpid()}-{uuid.uuid4().hex}"
    started_wall = time.time_ns()
    started = time.perf_counter_ns()
    result = dict(fields)
    try:
        with trace_scope(span_id=span_id):
            yield result
    except BaseException as exc:
        result.update(success=False, error=repr(exc), outcome="exception")
        raise
    finally:
        ended = time.perf_counter_ns()
        result.setdefault("success", True)
        result.update(
            span_id=span_id,
            parent_span_id=parent,
            start_timestamp_ns=started_wall,
            end_timestamp_ns=started_wall + ended - started,
            start_monotonic_ns=started,
            end_monotonic_ns=ended,
            duration_seconds=(ended - started) / 1_000_000_000,
        )
        emit(event, **result)


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
    record.update(_SCOPE.get())
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


def runtime_identity(runtime: Any) -> dict[str, Any]:
    container = getattr(runtime, "container", None)
    metadata = getattr(container, "metadata", None)
    return {
        "sandbox_id": _docker_sandbox_id(container) or getattr(metadata, "uid", None),
        "sandbox_name": getattr(container, "name", None) or getattr(metadata, "name", None),
        "backend": getattr(runtime, "backend", None),
        "image": getattr(runtime, "docker_image", None),
    }


def _install_docker_hooks() -> None:
    from docker.models.containers import Container, ContainerCollection

    if getattr(ContainerCollection.run, "_r2e_trace_hook", False):
        return

    original_run = ContainerCollection.run
    original_remove = Container.remove

    def traced_run(collection: Any, image: str, *args: Any, **kwargs: Any):
        with timed_event("sandbox_create", backend="docker", image=str(image),
                         sandbox_name=kwargs.get("name"), source_api="docker.containers.run") as timing:
            container = original_run(collection, image, *args, **kwargs)
            sandbox_id = _docker_sandbox_id(container)
            sandbox_name = kwargs.get("name") or getattr(container, "name", None)
            key = ("docker", sandbox_id or sandbox_name or "unknown")
            _ACTIVE[key] = sandbox_id or sandbox_name or "unknown"
            timing.update(sandbox_id=sandbox_id, sandbox_name=sandbox_name, outcome="container_running")
        return container

    def traced_remove(container: Any, *args: Any, **kwargs: Any):
        sandbox_id = _docker_sandbox_id(container)
        sandbox_name = getattr(container, "name", None)
        key = ("docker", sandbox_id or sandbox_name or "unknown")
        if key not in _ACTIVE:
            return original_remove(container, *args, **kwargs)
        with timed_event("sandbox_destroy", sandbox_id=sandbox_id,
                         sandbox_name=sandbox_name, backend="docker",
                         source_api="docker.container.remove") as timing:
            result = original_remove(container, *args, **kwargs)
            timing["outcome"] = "container_removed"
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
        image = None
        try:
            spec = body.get("spec", {}) if isinstance(body, dict) else body.spec
            containers = spec.get("containers", []) if isinstance(spec, dict) else spec.containers
            first = containers[0]
            image = first.get("image") if isinstance(first, dict) else first.image
        except (AttributeError, IndexError, TypeError):
            pass
        with timed_event("sandbox_create", backend="kubernetes", namespace=namespace,
                         image=image, source_api="kubernetes.create_namespaced_pod") as timing:
            response = original_create(api, namespace, body, *args, **kwargs)
            uid, pod_name = _kubernetes_identity(body=body, response=response)
            _ACTIVE[("kubernetes", pod_name or uid or "unknown")] = uid or pod_name or "unknown"
            timing.update(sandbox_id=uid or pod_name, sandbox_name=pod_name, outcome="pod_create_accepted")
        return response

    def traced_delete(api: Any, name: str, namespace: str, *args: Any, **kwargs: Any):
        key = ("kubernetes", name)
        sandbox_id = _ACTIVE.get(key, name)
        if key not in _ACTIVE:
            return original_delete(api, name, namespace, *args, **kwargs)
        with timed_event("sandbox_destroy", sandbox_id=sandbox_id, sandbox_name=name,
                         backend="kubernetes", namespace=namespace,
                         source_api="kubernetes.delete_namespaced_pod") as timing:
            response = original_delete(api, name, namespace, *args, **kwargs)
            timing["outcome"] = "pod_delete_accepted"
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
        with timed_event("tool", tool_name=str(tool_name),
                         tool_parameters=getattr(action, "parameters", {}),
                         **runtime_identity(environment.runtime)) as timing:
            result = original_run_action(environment, action, *args, **kwargs)
            exit_code = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            timing.update(exit_code=exit_code, success=exit_code in (None, 0, "0"),
                          outcome="ok" if exit_code in (None, 0, "0") else "error")
        return result

    traced_tool._r2e_trace_hook = True  # type: ignore[attr-defined]
    RepoEnv.run_action = traced_tool


def _install_deepseek_native_tool_calling() -> None:
    """Enable native tool calls for DeepSeek when explicitly requested.

    R2E-Gym currently limits native tool calling to a small model-name
    allowlist.  The DeepSeek API supports the same structured ``tool_calls``
    response through LiteLLM, so the trace runner can opt in without modifying
    the external R2E-Gym checkout.
    """
    if os.getenv("R2E_DEEPSEEK_NATIVE_TOOL_CALLING") != "1":
        return

    from r2egym.agenthub.agent.agent import Agent

    if getattr(Agent.model_query, "_r2e_deepseek_native_tools", False):
        return

    original_model_query = Agent.model_query

    def model_query_with_deepseek_tools(
        agent: Any, messages: list[dict[str, str]], temperature: float = 0
    ):
        if "deepseek" in str(getattr(agent, "llm_name", "")).lower():
            if not getattr(agent, "use_fn_calling", False):
                agent.logger.warning(
                    "Enabling native tool calling for DeepSeek via trace instrumentation"
                )
            agent.use_fn_calling = True
        return original_model_query(agent, messages, temperature)

    model_query_with_deepseek_tools._r2e_deepseek_native_tools = True  # type: ignore[attr-defined]
    Agent.model_query = model_query_with_deepseek_tools


def _install_command_hook() -> None:
    from r2egym.agenthub.runtime.docker import DockerRuntime

    def wrap(original, method):
        @wraps(original)
        def traced_command(runtime: Any, code: str, *args: Any, **kwargs: Any):
            # Preserve full commands: a truncated command is not a replay input.
            with timed_event("command", command=str(code),
                             source_api=f"DockerRuntime.{method}",
                             timeout_seconds=kwargs.get("timeout", args[0] if args else None),
                             workdir=kwargs.get("workdir", getattr(runtime, "repo_path", None)),
                             **runtime_identity(runtime)) as timing:
                result = original(runtime, code, *args, **kwargs)
                exit_code = result[-1] if isinstance(result, tuple) and len(result) > 1 else None
                output = str(result[0]) if isinstance(result, tuple) else str(result)
                timing.update(exit_code=exit_code, success=exit_code in (None, 0, "0"),
                              outcome="ok" if exit_code in (None, 0, "0") else "error",
                              output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                              output_bytes=len(output.encode()),
                              infrastructure_error=str(exit_code) == "-1" and _SCOPE.get().get("phase") in
                              ("verification", "agent_reward", "initialization"))
            return result
        traced_command._r2e_trace_hook = True
        return traced_command

    for method in ("run", "demux_run"):
        original = getattr(DockerRuntime, method)
        if not getattr(original, "_r2e_trace_hook", False):
            setattr(DockerRuntime, method, wrap(original, method))


def _install_environment_hooks() -> None:
    """Measure readiness, setup, file transfers and stop+remove separately."""
    from r2egym.agenthub.runtime.docker import DockerRuntime
    from r2egym.agenthub.environment.env import RepoEnv

    def wrap(original, event, is_env):
        @wraps(original)
        def measured(obj, *args, **kwargs):
            runtime = obj.runtime if is_env else obj
            with timed_event(event, source_api=f"{type(obj).__name__}.{original.__name__}",
                             **runtime_identity(runtime)) as timing:
                try:
                    result = original(obj, *args, **kwargs)
                    if event == "patch_application":
                        code = result[1]
                        timing.update(exit_code=code, success=code in (0, "0"))
                    return result
                except BaseException:
                    # A failed constructor can already own a running container,
                    # before RepoEnv is assigned in its caller. Clean up only a
                    # container whose creation was observed by this process.
                    identity = runtime_identity(runtime)
                    if event == "environment_init" and identity.get("sandbox_id") in _ACTIVE.values():
                        try:
                            runtime.close()
                        except Exception as cleanup_error:
                            timing["cleanup_error"] = repr(cleanup_error)
                    raise
                finally:
                    timing.update(runtime_identity(runtime))
        measured._r2e_trace_hook = True
        return measured

    for cls, method, event, is_env in (
        (DockerRuntime, "__init__", "environment_init", False),
        (DockerRuntime, "start_container", "sandbox_start", False),
        (DockerRuntime, "setup_env", "environment_setup", False),
        (DockerRuntime, "stop_container", "sandbox_stop", False),
        (DockerRuntime, "copy_to_container", "file_transfer", False),
        (DockerRuntime, "apply_patch", "patch_application", False),
        (RepoEnv, "add_commands", "tool_setup", True),
    ):
        original = getattr(cls, method)
        if not getattr(original, "_r2e_trace_hook", False):
            setattr(cls, method, wrap(original, event, is_env))


def install_instrumentation(*, trace_commands: bool = True) -> None:
    """Install exact backend lifecycle hooks in the current process."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        _install_docker_hooks()
        _install_kubernetes_hooks()
        _install_deepseek_native_tool_calling()
        _install_tool_hook()
        _install_environment_hooks()
        if trace_commands:
            _install_command_hook()
        _INSTALLED = True

