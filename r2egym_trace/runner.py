"""CLI orchestration for real R2E-Gym agent and verifier workloads."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .instrumentation import configure_trace, emit, install_instrumentation, timed_event, trace_scope
from .trace_io import merge_and_write


DEFAULT_DATASET = "R2E-Gym/SWE-Bench-Verified"
DEFAULT_REPRO_DATASET = "R2E-Gym/R2E-TestgenAgent-Patches"


def _prepare_process(r2e_root: str) -> Path:
    root = Path(r2e_root).expanduser().resolve()
    source = root / "src"
    if not (source / "r2egym").is_dir():
        raise RuntimeError(f"R2E-Gym source not found under {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    os.chdir(root)
    return root


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _task_id(row: dict[str, Any]) -> str:
    for key in ("instance_id", "task_id", "commit_hash", "docker_image"):
        if row.get(key):
            return str(row[key])
    raise ValueError("Dataset row has no instance_id, task_id, commit_hash, or docker_image")


def _resolve_tasks(dataset_name: str, split: str, requested: list[str]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    rows = [dict(row) for row in dataset]
    resolved: list[dict[str, Any]] = []
    for value in requested:
        matches = []
        for row in rows:
            identifiers = {
                str(row[key])
                for key in ("instance_id", "task_id", "commit_hash", "docker_image")
                if row.get(key)
            }
            if value in identifiers:
                matches.append(row)
        if not matches:
            raise ValueError(f"Task {value!r} was not found in {dataset_name}/{split}")
        if len(matches) > 1:
            raise ValueError(f"Task selector {value!r} matched {len(matches)} rows; use an exact unique ID")
        resolved.append(matches[0])
    return resolved


def _load_reproduction_tests(dataset_name: str) -> dict[str, list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train")
    result: dict[str, list[str]] = {}
    for row in dataset:
        patches = row["test_patches"]
        if isinstance(patches, str):
            patches = json.loads(patches)
        result[str(row["docker_image"])] = list(patches)
    return result


def _agent_worker(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        _prepare_process(spec["r2e_root"])
        configure_trace(
            spec["parts_dir"],
            task_id=spec["task_id"],
            trajectory_id=spec["trajectory_id"],
            stage="agent",
        )
        install_instrumentation(trace_commands=spec["trace_commands"])
        emit("stage_start")
        from r2egym.agenthub.run.edit import runagent

        trajectory_json = runagent(
            ds=spec["task"],
            exp_name=spec["exp_name"],
            max_steps=spec["max_steps"],
            num_restarts=1,
            max_steps_absolute=spec["max_steps_absolute"],
            llm_name=spec["llm_name"],
            temperature=spec["temperature"],
            use_fn_calling=spec["use_fn_calling"],
            backend=spec["backend"],
            max_reward_calc_time=spec["reward_timeout"],
            max_iterations=1,
            scaffold=spec["scaffold"],
            max_tokens=spec["max_tokens"],
        )
        ok = trajectory_json is not None
        emit("stage_end", success=ok)
        return {
            "ok": ok,
            "task_id": spec["task_id"],
            "trajectory_id": spec["trajectory_id"],
            "trajectory_json": trajectory_json,
            "error": None if ok else "R2E-Gym runagent returned None; inspect run_logs",
        }
    except BaseException as exc:
        emit("stage_end", success=False, error=repr(exc))
        return {
            "ok": False,
            "task_id": spec["task_id"],
            "trajectory_id": spec["trajectory_id"],
            "trajectory_json": None,
            "error": repr(exc),
        }


def _regression_worker(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        _prepare_process(spec["r2e_root"])
        configure_trace(
            spec["parts_dir"],
            task_id=spec["task_id"],
            trajectory_id=spec["trajectory_id"],
            stage="regression_test",
        )
        install_instrumentation(trace_commands=spec["trace_commands"])
        emit("stage_start")
        with trace_scope(phase="verifier_initialization"), timed_event("verifier_initialization"):
            from r2egym.agenthub.trajectory.trajectory import Trajectory
            from r2egym.agenthub.verifiers.run_regression_tests import compute_regression_output
            trajectory = Trajectory.model_validate_json(spec["trajectory_json"])
        with trace_scope(phase="verification"), timed_event("regression_execution"):
            trajectory.regression_test_output = compute_regression_output(trajectory)
        emit("stage_end", success=True)
        return {
            "ok": True,
            "task_id": spec["task_id"],
            "trajectory_id": spec["trajectory_id"],
            "trajectory_json": trajectory.model_dump_json(),
            "error": None,
        }
    except BaseException as exc:
        emit("stage_end", success=False, error=repr(exc))
        return {
            "ok": False,
            "task_id": spec["task_id"],
            "trajectory_id": spec["trajectory_id"],
            "trajectory_json": spec["trajectory_json"],
            "error": repr(exc),
        }


def _reproduction_worker(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        _prepare_process(spec["r2e_root"])
        configure_trace(
            spec["parts_dir"],
            task_id=spec["task_id"],
            trajectory_id=spec["trajectory_id"],
            stage="reproduction_test",
            test_id=f"reproduction-{spec['test_index']}",
        )
        install_instrumentation(trace_commands=spec["trace_commands"])
        emit("stage_start")
        # R2E-Gym's reproduction module configures file logging at import time.
        # Existing root-owned logs in the checkout can make that import fail;
        # keep replay logs inside the writable trace output instead.
        import r2egym.logging as r2e_logging

        original_setup_logging = r2e_logging.setup_logging

        def replay_setup_logging(name: str = "r2e", *args: Any, **kwargs: Any):
            log_file = kwargs.get("log_file")
            if log_file is None and args:
                # setup_logging(name, level, log_file, console)
                args = list(args)
                if len(args) >= 2:
                    log_file = args[1]
            if log_file:
                log_path = Path(str(log_file))
                redirected = Path(spec["parts_dir"]) / "r2e-logs" / log_path
                if "log_file" in kwargs:
                    kwargs["log_file"] = str(redirected)
                else:
                    args = list(args)
                    while len(args) < 2:
                        args.append(None)
                    args[1] = str(redirected)
            return original_setup_logging(name, *args, **kwargs)

        r2e_logging.setup_logging = replay_setup_logging
        with trace_scope(phase="verifier_initialization"), timed_event("verifier_initialization"):
            from r2egym.agenthub.trajectory.trajectory import Trajectory
            from r2egym.agenthub.verifiers.run_reproduction_tests import process_single_task
            trajectory = Trajectory.model_validate_json(spec["trajectory_json"])
        with trace_scope(phase="verification"), timed_event("reproduction_execution"):
            result = process_single_task(
                (
                    trajectory.ds,
                    spec["test_patch"],
                    trajectory.true_output_patch,
                    spec["test_index"],
                )
            )
        emit("stage_end", success=True)
        return {
            "ok": True,
            "task_id": spec["task_id"],
            "trajectory_id": spec["trajectory_id"],
            "test_index": spec["test_index"],
            "score": result[2],
            "error": None,
        }
    except BaseException as exc:
        emit("stage_end", success=False, error=repr(exc))
        return {
            "ok": False,
            "task_id": spec["task_id"],
            "trajectory_id": spec["trajectory_id"],
            "test_index": spec["test_index"],
            "score": 0.0,
            "error": repr(exc),
        }


def _run_pool(worker, specs: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    if not specs:
        return []
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, spec) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _write_jsonl(path: Path, records: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.rstrip("\n") + "\n")


def _trajectory_task_id(trajectory: dict[str, Any]) -> str:
    dataset_row = trajectory.get("ds") or {}
    for key in ("instance_id", "task_id", "commit_hash", "docker_image"):
        if dataset_row.get(key):
            return str(dataset_row[key])
    if trajectory.get("docker_image"):
        return str(trajectory["docker_image"])
    raise ValueError("Saved trajectory has no task identifier")


def _read_trajectories(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Saved trajectories not found: {path}")
    records: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                trajectory = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid trajectory JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(trajectory, dict):
                raise ValueError(f"Trajectory at {path}:{line_number} is not a JSON object")
            _trajectory_task_id(trajectory)
            records.append(json.dumps(trajectory, ensure_ascii=False))
    if not records:
        raise ValueError(f"Saved trajectories file is empty: {path}")
    return records


def replay_trace_run(trace_run: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Compatibility entry point for full Agent + verifier replay."""
    from .replay import replay_trace_run as replay

    return replay(trace_run, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real R2E-Gym agents and trace sandbox lifecycle events."
    )
    parser.add_argument("--task", action="append", nargs="+", required=True, help="Exact task/instance ID; repeatable")
    parser.add_argument("--R", "--rollouts", dest="rollouts", type=int, required=True, help="Agent trajectories per task")
    parser.add_argument("--T", "--reproduction-tests", dest="reproduction_tests", type=int, required=True, help="Reproduction tests per trajectory")
    parser.add_argument("--r2e-root", default=os.getenv("R2E_GYM_ROOT"), help="R2E-Gym checkout (or set R2E_GYM_ROOT)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--reproduction-dataset", default=DEFAULT_REPRO_DATASET)
    parser.add_argument("--llm-name", default="gpt-4o")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the model API key")
    parser.add_argument("--no-api-key-check", action="store_true", help="For a local endpoint that does not require a key")
    parser.add_argument("--backend", choices=("docker", "kubernetes"), default="docker")
    parser.add_argument("--agent-workers", type=int, default=None)
    parser.add_argument("--verifier-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-steps-absolute", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--reward-timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scaffold", choices=("r2egym", "sweagent", "openhands"), default="r2egym")
    parser.add_argument("--no-function-calling", action="store_true")
    parser.add_argument("--no-command-events", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    invocation_dir = Path.cwd()
    if not args.r2e_root:
        raise ValueError("--r2e-root or R2E_GYM_ROOT is required")
    r2e_root = str(Path(args.r2e_root).expanduser().resolve())
    _prepare_process(r2e_root)
    if args.rollouts < 1:
        raise ValueError("R must be at least 1")
    if args.reproduction_tests < 0:
        raise ValueError("T cannot be negative")
    if args.agent_workers is not None and args.agent_workers < 1:
        raise ValueError("--agent-workers must be at least 1")
    if args.verifier_workers is not None and args.verifier_workers < 1:
        raise ValueError("--verifier-workers must be at least 1")
    if not args.no_api_key_check and not os.getenv(args.api_key_env):
        raise ValueError(f"Environment variable {args.api_key_env} is not set")

    requested = [item for group in args.task for item in group]
    if len(requested) != len(set(requested)):
        raise ValueError("Each --task selector must be unique")
    tasks = _resolve_tasks(args.dataset, args.split, requested)
    repro_by_image = _load_reproduction_tests(args.reproduction_dataset) if args.reproduction_tests else {}
    for task in tasks:
        image = str(task.get("docker_image", ""))
        available = len(repro_by_image.get(image, []))
        if args.reproduction_tests > available:
            raise ValueError(
                f"Task {_task_id(task)} has {available} reproduction tests, fewer than requested T={args.reproduction_tests}"
            )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_value = Path(args.output_dir or f"trace-runs/{run_stamp}")
    output_dir = (
        output_value.resolve()
        if output_value.is_absolute()
        else (invocation_dir / output_value).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / ".trace-parts"
    parts_dir.mkdir()

    common = {
        "r2e_root": r2e_root,
        "parts_dir": str(parts_dir),
        "llm_name": args.llm_name,
        "backend": args.backend,
        "max_steps": args.max_steps,
        "max_steps_absolute": args.max_steps_absolute,
        "max_tokens": args.max_tokens,
        "reward_timeout": args.reward_timeout,
        "temperature": args.temperature,
        "scaffold": args.scaffold,
        "use_fn_calling": not args.no_function_calling,
        "trace_commands": not args.no_command_events,
    }
    agent_specs = []
    for task in tasks:
        task_id = _task_id(task)
        for index in range(args.rollouts):
            trajectory_id = f"{task_id}/r{index:03d}"
            agent_specs.append(
                {
                    **common,
                    "task": task,
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "exp_name": f"trace_{_safe_name(task_id)}_r{index:03d}_{run_stamp}",
                }
            )

    agent_workers = args.agent_workers or min(args.rollouts, len(agent_specs))
    agent_results = _run_pool(_agent_worker, agent_specs, agent_workers)
    successful = [result for result in agent_results if result["ok"]]
    _write_jsonl(output_dir / "trajectories.jsonl", [result["trajectory_json"] for result in successful])

    verifier_workers = args.verifier_workers or min(48, max(1, len(successful)))
    regression_specs = [
        {
            **common,
            "task_id": result["task_id"],
            "trajectory_id": result["trajectory_id"],
            "trajectory_json": result["trajectory_json"],
        }
        for result in successful
    ]
    regression_results = _run_pool(_regression_worker, regression_specs, verifier_workers)
    trajectory_by_id = {
        result["trajectory_id"]: result["trajectory_json"] for result in regression_results
    }
    agent_trajectory_by_id = {
        result["trajectory_id"]: result["trajectory_json"] for result in successful
    }

    reproduction_specs = []
    for result in successful:
        trajectory_json = trajectory_by_id.get(result["trajectory_id"], result["trajectory_json"])
        trajectory_data = json.loads(trajectory_json)
        image = trajectory_data["docker_image"]
        for test_index, test_patch in enumerate(repro_by_image.get(image, [])[: args.reproduction_tests]):
            reproduction_specs.append(
                {
                    **common,
                    "task_id": result["task_id"],
                    "trajectory_id": result["trajectory_id"],
                    "trajectory_json": trajectory_json,
                    "test_index": test_index,
                    "test_patch": test_patch,
                }
            )
    reproduction_workers = args.verifier_workers or min(48, max(1, len(reproduction_specs)))
    reproduction_results = _run_pool(_reproduction_worker, reproduction_specs, reproduction_workers)

    scores: dict[str, dict[int, float]] = {}
    for result in reproduction_results:
        scores.setdefault(result["trajectory_id"], {})[result["test_index"]] = result["score"]
    verified_json = []
    for result in successful:
        trajectory_id = result["trajectory_id"]
        trajectory_json = trajectory_by_id.get(
            trajectory_id, agent_trajectory_by_id[trajectory_id]
        )
        data = json.loads(trajectory_json)
        data["reproduction_test_scores"] = [
            score for _, score in sorted(scores.get(trajectory_id, {}).items())
        ]
        verified_json.append(json.dumps(data, ensure_ascii=False))
    _write_jsonl(output_dir / "verified_trajectories.jsonl", verified_json)

    summary = merge_and_write(parts_dir, output_dir / "trace.jsonl", output_dir / "summary.json")
    run_manifest = {
        "tasks": requested,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "R": args.rollouts,
        "T": args.reproduction_tests,
        "dataset": args.dataset,
        "split": args.split,
        "llm_name": args.llm_name,
        "backend": args.backend,
        "agent_workers": agent_workers,
        "verifier_workers": verifier_workers,
        "agent_attempts": len(agent_results),
        "agent_successes": len(successful),
        "agent_errors": [result for result in agent_results if not result["ok"]],
        "regression_errors": [result for result in regression_results if not result["ok"]],
        "reproduction_errors": [result for result in reproduction_results if not result["ok"]],
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"output_dir": str(output_dir), "summary": summary, "run": run_manifest}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        parser.exit(1, f"trace generator error: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

