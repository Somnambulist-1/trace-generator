"""Execute saved R2E-Gym actions and the real verifier pipeline, without an LLM."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .instrumentation import (
    configure_trace, emit, install_instrumentation, runtime_identity,
    timed_event, trace_scope,
)
from .trace_io import duration_stats, merge_and_write


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def parse_saved_action(text: str) -> tuple[str, dict[str, str]]:
    """Read the canonical serialization, preserving parameter whitespace.

    Action.from_string strips parameter values. That changes native tool-call
    arguments (notably file edits), so only accept an exact serialization round
    trip here. An explicitly recorded empty action remains an explicit no-op.
    """
    match = re.fullmatch(r"<function=([^>\n]*)>\n(.*)</function>", text, re.DOTALL)
    if not match:
        raise ValueError("Saved action is not a canonical R2E-Gym action; refusing to skip it")
    name, body = match.groups()
    parameters: dict[str, str] = {}
    cursor = 0
    for parameter in re.finditer(r"  <parameter=([^>\n]+)>(.*?)</parameter>\n", body, re.DOTALL):
        if parameter.start() != cursor or parameter[1] in parameters:
            raise ValueError("Ambiguous or duplicate saved action parameters")
        parameters[parameter[1]] = parameter[2]
        cursor = parameter.end()
    if cursor != len(body) or (not name and parameters):
        raise ValueError("Malformed saved action parameters")
    rebuilt = f"<function={name}>\n" + "".join(
        f"  <parameter={key}>{value}</parameter>\n" for key, value in parameters.items()
    ) + "</function>"
    if rebuilt != text:
        raise ValueError("Saved action cannot be reconstructed losslessly")
    return name, parameters


def _command_files(trajectory: dict[str, Any], root: Path) -> list[str]:
    paths = trajectory.get("agent_args", {}).get("command_files")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Trajectory is missing agent_args.command_files")
    result = []
    for value in paths:
        path = Path(value)
        if path.is_absolute() and "/src/r2egym/" in path.as_posix():
            path = root / "src/r2egym" / path.as_posix().split("/src/r2egym/", 1)[1]
        elif not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Recorded tool implementation is missing from R2E-Gym: {value}")
        result.append(str(path))
    return result


def _expected_exit(observation: str) -> str | None:
    match = re.match(r"Exit code: ([^\n]+)", observation)
    return match[1] if match else None


def _recorded_llm_seconds(step: dict[str, Any]) -> float:
    # A replay records zero real LLM time, but retains the source pacing so
    # replaying its output does not silently turn recorded waits into zero.
    return float(step.get("info", {}).get("replay_source_llm_seconds", step.get("llm_exec_time", 0)))


def _replay_agent_worker(spec: dict[str, Any]) -> dict[str, Any]:
    from r2egym.agenthub.action import Action
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
    from r2egym.agenthub.trajectory.trajectory import Trajectory

    original = json.loads(spec["trajectory_json"])
    Trajectory.model_validate_json(spec["trajectory_json"])
    replayed = json.loads(spec["trajectory_json"])
    replayed["trajectory_steps"] = []
    replayed["regression_test_output"] = None
    replayed["reproduction_test_scores"] = []
    # A partial failure must not look as if the saved patch/reward was produced.
    # Null means not collected; the successful path fills every field below.
    replayed.update(output_patch=None, reward=None, test_output=None, reward_calc_time=None)
    result: dict[str, Any] = {
        "task_id": spec["task_id"], "trajectory_id": spec["trajectory_id"],
        "planned_steps": len(original["trajectory_steps"]), "executed_steps": 0,
        "exit_code_mismatches": [], "observation_mismatches": [],
        "patch_matches": None, "ok": False, "error": None,
    }
    env = None
    cumulative = 0.0
    try:
        with trace_scope(phase="initialization"), timed_event("agent_initialization"):
            env = RepoEnv(EnvArgs(**original["env_args"]), backend=spec["backend"],
                          verbose=False, step_timeout=original["max_exec_time"],
                          reward_timeout=spec["reward_timeout"])
            # runagent constructs RepoEnv, then Agent.run resets it. Preserve
            # both lifecycles instead of silently optimizing the workload.
            env.reset()
            env.add_commands(spec["command_files"])
            env.runtime.get_task_instruction()
            env.runtime.commit.get_patch(test_file=True, non_test_file=False)
            emit("agent_ready", **runtime_identity(env.runtime))

        for saved in original["trajectory_steps"]:
            index = saved["step_idx"]
            name, parameters = parse_saved_action(saved["action"])
            action = Action(name, parameters)
            recorded_wait = _recorded_llm_seconds(saved)
            actual_wait = 0.0
            with trace_scope(phase="agent_actions", step_idx=index):
                if spec["llm_wait"] == "recorded" and recorded_wait > 0:
                    with timed_event("llm_wait", requested_seconds=recorded_wait,
                                     outcome="recorded_delay_no_model_inference") as wait:
                        time.sleep(recorded_wait)
                    actual_wait = wait.get("duration_seconds", recorded_wait)
                with timed_event("agent_step", tool_name=name or None,
                                 source_action_sha256=sha256(saved["action"]),
                                 recorded_noop=not bool(name)) as timing:
                    observation, _, done, info = env.step(action, timeout=original["max_exec_time"])
                    observation_text = str(observation)
                    exit_code = str(observation.error_code)
                    expected_exit = _expected_exit(saved["observation"])
                    matched = expected_exit is None or exit_code == expected_exit
                    timing.update(exit_code=exit_code, expected_exit_code=expected_exit,
                                  exit_code_matches=matched, observation_matches=observation_text == saved["observation"],
                                  **runtime_identity(env.runtime))
                result["executed_steps"] += 1
                if not matched:
                    result["exit_code_mismatches"].append({"step_idx": index, "expected": expected_exit, "actual": exit_code})
                if observation_text != saved["observation"]:
                    result["observation_mismatches"].append(index)
                # Environment timings come from the current execution, not the
                # recording. Original tokens/timings stay in source_trajectories.
                execution_time = float(info["total_time"])
                cumulative += execution_time + actual_wait
                replayed["trajectory_steps"].append({
                    **saved, "observation": observation_text, "done": done,
                    "info": {**info, "replay_llm_wait_seconds": actual_wait,
                             "replay_source_llm_seconds": recorded_wait},
                    "token_usage_prompt": 0, "token_usage_completion": 0, "token_usage_total": 0,
                    "llm_exec_time": 0.0, "env_exec_time": execution_time,
                    "total_step_time": execution_time + actual_wait, "total_time_traj": cumulative,
                })

        with trace_scope(phase="agent_finalize"), timed_event("agent_finalize"):
            replayed["output_patch"] = env.runtime.get_patch()
            result["patch_matches"] = replayed["output_patch"] == original["output_patch"]
            result["source_patch_sha256"] = sha256(original["output_patch"])
            result["replayed_patch_sha256"] = sha256(replayed["output_patch"])
        with trace_scope(phase="agent_reward"), timed_event("agent_reward") as reward_timing:
            reward, test_output = env.runtime._calculate_reward(
                get_test_output=True, timeout=spec["reward_timeout"])
            replayed.update(reward=reward, test_output=test_output)
        replayed["reward_calc_time"] = reward_timing["duration_seconds"]
        result["reward"] = reward
        result["source_reward"] = original.get("reward")
        result["ok"] = result["patch_matches"] and not result["exit_code_mismatches"]
        if not result["ok"]:
            result["error"] = "Replay diverged: final patch or recorded exit codes differ; dependent verification skipped"
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if env is not None:
            try:
                with trace_scope(phase="cleanup"), timed_event("agent_cleanup"):
                    env.close()
            except Exception as exc:
                result.update(ok=False, cleanup_error=repr(exc))
    # Keep partial actual outputs on failure. Never replace them with the source
    # patch in order to make the validation pass.
    result["trajectory_json"] = json.dumps(replayed, ensure_ascii=False)
    if result["ok"]:
        Trajectory.model_validate_json(result["trajectory_json"])
    return result


def _timed_job(stage: str, spec: dict[str, Any]) -> dict[str, Any]:
    from .runner import _prepare_process, _regression_worker, _reproduction_worker

    entered = time.perf_counter_ns()
    configure_trace(spec["parts_dir"], task_id=spec["task_id"],
                    trajectory_id=spec["trajectory_id"], stage=stage,
                    test_id=f"reproduction-{spec['test_index']}" if "test_index" in spec else None)
    queue_seconds = (entered - spec["submitted_monotonic_ns"]) / 1e9
    emit("job_queue", duration_seconds=queue_seconds,
         start_monotonic_ns=spec["submitted_monotonic_ns"], end_monotonic_ns=entered)
    result: dict[str, Any] = {}
    with timed_event("stage_job") as timing:
        try:
            with trace_scope(phase="worker_initialization"), timed_event("worker_initialization"):
                _prepare_process(spec["r2e_root"])
                install_instrumentation(trace_commands=spec["trace_commands"])
            with trace_scope(phase="verification" if stage != "agent" else "agent"):
                worker = {"agent": _replay_agent_worker, "regression_test": _regression_worker,
                          "reproduction_test": _reproduction_worker}[stage]
                result = worker(spec)
            timing["success"] = result["ok"]
        except Exception as exc:
            timing.update(success=False, error=repr(exc))
            result = {"ok": False, "error": repr(exc), "trajectory_json": None}
    result.update(task_id=spec["task_id"], trajectory_id=spec["trajectory_id"], stage=stage,
                  queue_wait_seconds=queue_seconds, elapsed_seconds=timing["duration_seconds"],
                  source_record_index=spec.get("source_record_index"),
                  submitted_monotonic_ns=spec["submitted_monotonic_ns"],
                  worker_started_monotonic_ns=entered, finished_monotonic_ns=timing["end_monotonic_ns"])
    if "test_index" in spec:
        result["test_index"] = spec["test_index"]
    return result


def _run_jobs(stage: str, specs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if not specs:
        return []
    results = []
    with trace_scope(phase=stage), timed_event("stage_pool", pool_stage=stage, jobs=len(specs),
                                            workers=min(workers, len(specs))) as timing:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
            futures = {}
            for spec in specs:
                spec = {**spec, "submitted_monotonic_ns": time.perf_counter_ns()}
                futures[pool.submit(_timed_job, stage, spec)] = spec
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"ok": False, "error": repr(exc), "stage": stage,
                              "task_id": spec["task_id"], "trajectory_id": spec["trajectory_id"]}
                    if "test_index" in spec:
                        result["test_index"] = spec["test_index"]
                results.append(result)
        timing["success"] = all(r["ok"] for r in results)
    return sorted(results, key=lambda r: (r["trajectory_id"], r.get("test_index", -1)))


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _workflow_timings(results: list[dict[str, Any]], tests_count: int) -> list[dict[str, Any]]:
    groups: dict[str, list] = {}
    for result in results:
        groups.setdefault(result["trajectory_id"], []).append(result)
    records = []
    for trajectory_id, jobs in sorted(groups.items()):
        first = next((r for r in jobs if r["stage"] == "agent"), {})
        start = first.get("submitted_monotonic_ns")
        ends = [r["finished_monotonic_ns"] for r in jobs if "finished_monotonic_ns" in r]
        counts = Counter(r["stage"] for r in jobs)
        complete = (counts == Counter({"agent": 1, "regression_test": 1, "reproduction_test": tests_count})
                    and all(r["ok"] for r in jobs) and start is not None and len(ends) == len(jobs))
        records.append({"trajectory_id": trajectory_id, "task_id": first.get("task_id"),
                        "success": complete, "start_monotonic_ns": start,
                        "end_monotonic_ns": max(ends) if ends else None,
                        "duration_seconds": (max(ends) - start) / 1e9 if ends and start is not None else None,
                        "jobs_by_stage": dict(counts),
                        "boundary": "Agent submission to last dependent stage job completion, including stage barriers"})
    return records


def _image_inventory(images: list[str], backend: str) -> dict[str, Any]:
    """Read local image identity only; never pull an image during preflight."""
    if backend != "docker":
        return {"backend": backend, "inspection": "not available from Docker"}
    try:
        import docker
        client = docker.from_env()
        try:
            inventory = {}
            for name in sorted(set(images)):
                try:
                    image = client.images.get(name)
                    inventory[name] = {"cached_before_run": True, "id": image.id,
                                       "repo_digests": image.attrs.get("RepoDigests", [])}
                except docker.errors.ImageNotFound:
                    inventory[name] = {"cached_before_run": False}
            return inventory
        finally:
            client.close()
    except Exception as exc:
        return {"inspection_error": repr(exc)}


def replay_trace_run(
    trace_run: str | Path, *, r2e_root: str | None = None, output_dir: str | Path | None = None,
    reproduction_tests: int | None = None, agent_workers: int = 1, verifier_workers: int = 48,
    trace_commands: bool = True, llm_wait: str = "recorded", reward_timeout: int | None = None,
) -> dict[str, Any]:
    """Full application replay with separately reported preparation and workload."""
    from .runner import (
        DEFAULT_REPRO_DATASET, _load_reproduction_tests, _read_trajectories,
        _trajectory_task_id, _write_jsonl,
    )

    started = time.perf_counter_ns()
    source = Path(trace_run).expanduser().resolve()
    manifest_bytes = (source / "run.json").read_bytes()
    source_manifest = json.loads(manifest_bytes)
    records = _read_trajectories(source / "trajectories.jsonl")
    trajectories = [json.loads(record) for record in records]
    root_value = r2e_root or os.getenv("R2E_GYM_ROOT")
    if not root_value:
        raise ValueError("--r2e-root or R2E_GYM_ROOT is required")
    root = Path(root_value).expanduser().resolve()
    if agent_workers < 1 or verifier_workers < 1:
        raise ValueError("Worker counts must be positive")
    if llm_wait not in ("recorded", "none"):
        raise ValueError("llm_wait must be recorded or none")
    count = source_manifest.get("T", 0) if reproduction_tests is None else reproduction_tests
    if not isinstance(count, int) or count < 0:
        raise ValueError("Reproduction test count must be a nonnegative integer")
    reward_timeout = source_manifest.get("reward_timeout", 300) if reward_timeout is None else reward_timeout
    if reward_timeout <= 0:
        raise ValueError("Reward timeout must be positive")
    backend = source_manifest.get("backend", "docker")
    if backend not in ("docker", "kubernetes"):
        raise ValueError(f"Unsupported recorded backend: {backend}")

    reproduction_file = source / "reproduction_inputs.json"
    repro = (json.loads(reproduction_file.read_text()) if reproduction_file.is_file()
             else _load_reproduction_tests(source_manifest.get("reproduction_dataset", DEFAULT_REPRO_DATASET))) if count else {}
    specs = []
    task_indices: Counter[str] = Counter()
    tool_hashes = {}
    for record_index, trajectory in enumerate(trajectories):
        task_id = _trajectory_task_id(trajectory)
        if not trajectory.get("ds") or trajectory.get("env_args", {}).get("ds") != trajectory["ds"]:
            raise ValueError(f"{task_id}: missing or inconsistent recorded environment dataset")
        if trajectory["ds"].get("docker_image") != trajectory["docker_image"]:
            raise ValueError(f"{task_id}: inconsistent recorded image")
        if not isinstance(trajectory.get("output_patch"), str):
            raise ValueError(f"{task_id}: incomplete source trajectory has no collected output patch")
        steps = trajectory.get("trajectory_steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"{task_id}: no recorded agent steps")
        if [s["step_idx"] for s in steps] != sorted({s["step_idx"] for s in steps}):
            raise ValueError(f"{task_id}: duplicated or unordered step indices")
        for step in steps:
            parse_saved_action(step["action"])
            delay = _recorded_llm_seconds(step)
            if not math.isfinite(delay) or delay < 0:
                raise ValueError(f"{task_id}: invalid recorded LLM time")
        if count > len(repro.get(trajectory["docker_image"], [])):
            raise ValueError(f"{task_id}: fewer reproduction tests available than requested T={count}")
        if any(not isinstance(p, str) or not p.strip() for p in repro.get(trajectory["docker_image"], [])[:count]):
            raise ValueError(f"{task_id}: empty or invalid reproduction test patch")
        files = _command_files(trajectory, root)
        tool_hashes.update({str(Path(p).relative_to(root)): sha256(Path(p).read_bytes()) for p in files})
        index = task_indices[task_id]
        task_indices[task_id] += 1
        specs.append({"task_id": task_id, "trajectory_id": f"{task_id}/r{index:03d}",
                      "source_record_index": record_index,
                      "trajectory_json": json.dumps(trajectory, ensure_ascii=False), "command_files": files})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir).expanduser().resolve() if output_dir else source.parent / f"{source.name}-full-replay-{stamp}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parts = output / ".trace-parts"
    parts.mkdir()
    common = {"r2e_root": str(root), "parts_dir": str(parts), "trace_commands": trace_commands,
              "llm_wait": llm_wait, "backend": backend, "reward_timeout": reward_timeout}
    specs = [{**common, **s} for s in specs]
    selected_repro = {t["docker_image"]: repro.get(t["docker_image"], [])[:count] for t in trajectories}
    _dump(output / "reproduction_inputs.json", selected_repro)
    _write_jsonl(output / "source_trajectories.jsonl", records)
    configure_trace(str(parts), task_id="__run__", trajectory_id="__run__", stage="replay")
    manifest: dict[str, Any] = {
        "mode": "full_application_replay", "status": "running", "source_trace_run": str(source),
        "source_run_sha256": sha256(manifest_bytes),
        "source_trajectories_sha256": sha256((source / "trajectories.jsonl").read_bytes()),
        "tasks": sorted(task_indices), "trajectories_per_task": dict(task_indices),
        "source_R": source_manifest.get("R"), "trajectory_count": len(specs), "T": count,
        "backend": backend, "agent_workers": agent_workers, "verifier_workers": verifier_workers,
        "llm_wait": llm_wait, "model_calls": 0, "reward_timeout": reward_timeout,
        "reproduction_dataset": source_manifest.get("reproduction_dataset", DEFAULT_REPRO_DATASET),
        "reproduction_inputs_sha256": sha256((output / "reproduction_inputs.json").read_bytes()),
        "tool_source_sha256": tool_hashes, "command_events": trace_commands,
        "source_record_mapping": [{"trajectory_id": s["trajectory_id"], "source_record_index": s["source_record_index"],
                                   "source_record_sha256": sha256(s["trajectory_json"])} for s in specs],
        "image_inventory": _image_inventory([t["docker_image"] for t in trajectories], backend),
        "agent_lifecycle": "RepoEnv construction + reset + tools + recorded actions + reward + close",
        "pipeline_order": ["agent", "regression_test", "reproduction_test"],
        "timing_clock": "time.perf_counter_ns (monotonic, host wall time)",
        "recorded_llm_seconds": sum(_recorded_llm_seconds(s) for t in trajectories for s in t["trajectory_steps"]),
        "planned_agent_steps": sum(len(t["trajectory_steps"]) for t in trajectories),
        "planned_agent_tools": sum(bool(parse_saved_action(s["action"])[0]) for t in trajectories for s in t["trajectory_steps"]),
    }
    for name, directory in (("generator", Path(__file__).resolve().parent.parent), ("r2egym", root)):
        git = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
        manifest[f"{name}_commit"] = git.stdout.strip() if git.returncode == 0 else None
    manifest["generator_source_sha256"] = {
        p.name: sha256(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")
    }
    manifest["r2egym_source_sha256"] = {
        str(p.relative_to(root)): sha256(p.read_bytes())
        for folder in ("action", "agent", "environment", "runtime", "trajectory", "verifiers")
        for p in (root / "src/r2egym/agenthub" / folder).glob("*.py")
    }
    _dump(output / "run.json", manifest)

    workload_start = time.perf_counter_ns()
    manifest["preparation_seconds"] = (workload_start - started) / 1e9
    workload_end = None
    agent_results, regression_results, reproduction_results = [], [], []
    fatal_error = None
    try:
        agent_results = _run_jobs("agent", specs, agent_workers)
        _write_jsonl(output / "trajectories.jsonl", [r["trajectory_json"] for r in agent_results if r.get("trajectory_json")])
        successful = {r["trajectory_id"]: r for r in agent_results if r["ok"]}
        verification_specs = [{**s, "trajectory_json": successful[s["trajectory_id"]]["trajectory_json"]}
                              for s in specs if s["trajectory_id"] in successful]
        regression_results = _run_jobs("regression_test", verification_specs, verifier_workers)
        reg_by_id = {r["trajectory_id"]: r for r in regression_results}
        reproduction_specs = []
        for spec in verification_specs:
            candidate = reg_by_id.get(spec["trajectory_id"], {}).get("trajectory_json") or spec["trajectory_json"]
            image = json.loads(candidate)["docker_image"]
            for index, patch in enumerate(selected_repro[image]):
                reproduction_specs.append({**spec, "trajectory_json": candidate, "test_index": index, "test_patch": patch})
        reproduction_results = _run_jobs("reproduction_test", reproduction_specs, verifier_workers)
        workload_end = time.perf_counter_ns()
        scores: dict[str, dict[int, Any]] = {}
        for result in reproduction_results:
            scores.setdefault(result["trajectory_id"], {})[result["test_index"]] = result.get("score")
        verified = []
        for spec in verification_specs:
            candidate = reg_by_id.get(spec["trajectory_id"], {}).get("trajectory_json") or spec["trajectory_json"]
            data = json.loads(candidate)
            data["reproduction_test_scores"] = [v for _, v in sorted(scores.get(spec["trajectory_id"], {}).items())]
            verified.append(json.dumps(data, ensure_ascii=False))
        _write_jsonl(output / "verified_trajectories.jsonl", verified)
    except BaseException as exc:
        fatal_error = repr(exc)
    workload_end = workload_end or time.perf_counter_ns()
    manifest["workload_seconds"] = (workload_end - workload_start) / 1e9
    results = agent_results + regression_results + reproduction_results
    workflows = _workflow_timings(results, count)
    _dump(output / "workflow_timings.json", workflows)
    _dump(output / "replay_results.json", [{k: v for k, v in r.items() if k != "trajectory_json"} for r in results])
    manifest.update(
        agent_attempts=len(specs), agent_successes=sum(r["ok"] for r in agent_results),
        regression_jobs=len(regression_results), reproduction_jobs=len(reproduction_results),
        agent_errors=[{k: v for k, v in r.items() if k != "trajectory_json"} for r in agent_results if not r["ok"]],
        regression_errors=[{k: v for k, v in r.items() if k != "trajectory_json"} for r in regression_results if not r["ok"]],
        reproduction_errors=[r for r in reproduction_results if not r["ok"]],
        fatal_error=fatal_error,
    )
    summary = merge_and_write(parts, output / "trace.jsonl", output / "summary.json")
    summary["workflow_latency"] = duration_stats([r for r in workflows if r["duration_seconds"] is not None])
    timing_complete = (summary["timings"].get("agent_step", {}).get("count", 0) == manifest["planned_agent_steps"]
                       and summary["timings"].get("tool", {}).get("count", 0) == manifest["planned_agent_tools"])
    manifest["agent_timing_complete"] = timing_complete
    complete = (fatal_error is None and len(agent_results) == len(specs)
                and len(regression_results) == len(specs) and len(reproduction_results) == len(specs) * count
                and all(r["ok"] for r in results) and summary["sandboxes_active_at_trace_end"] == 0
                and summary["lifecycle_failures"] == 0 and summary["infrastructure_failures"] == 0
                and timing_complete)
    manifest["status"] = "completed" if complete else "failed"
    manifest["measurement_valid"] = complete
    finished = time.perf_counter_ns()
    manifest["finalization_seconds"] = (finished - workload_end) / 1e9
    manifest["elapsed_seconds"] = (finished - started) / 1e9
    manifest["elapsed_boundary"] = "replay function entry through artifact assembly; excludes final manifest write and CLI printing"
    summary["replay_timing"] = {key: manifest[key] for key in (
        "preparation_seconds", "workload_seconds", "finalization_seconds", "elapsed_seconds", "measurement_valid")}
    summary["replay_timing"]["successful_trajectories_per_second"] = manifest["agent_successes"] / manifest["workload_seconds"] if complete and manifest["workload_seconds"] else None
    _dump(output / "summary.json", summary)
    _dump(output / "run.json", manifest)
    return {"output_dir": str(output), "summary": summary, "run": manifest}
