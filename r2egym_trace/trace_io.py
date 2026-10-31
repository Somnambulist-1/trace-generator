"""Merge trace shards and compute the minimal lifecycle summary."""

from __future__ import annotations

import json
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_shards(parts_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not parts_dir.exists():
        return events
    for shard in sorted(parts_dir.glob("trace-*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(json.loads(line))
    events.sort(key=lambda item: (item.get("timestamp_ns", 0), item.get("worker_id", 0), item.get("sequence", 0)))
    return events


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    creations = [event for event in events if event.get("event") == "sandbox_create" and event.get("success") is not False]
    destructions = [event for event in events if event.get("event") == "sandbox_destroy" and event.get("success") is not False]
    active: set[tuple[str, str]] = set()
    peak = 0
    for event in events:
        if event.get("success") is False:
            continue
        key = (
            str(event.get("backend", "unknown")),
            str(event.get("sandbox_id") or event.get("sandbox_name") or "unknown"),
        )
        if event.get("event") == "sandbox_create":
            active.add(key)
            peak = max(peak, len(active))
        elif event.get("event") == "sandbox_destroy":
            active.discard(key)

    result = {
        "total_sandbox_creations": len(creations),
        "total_sandbox_destructions": len(destructions),
        "peak_concurrent_sandboxes": peak,
        "creations_by_stage": dict(Counter(event.get("stage", "unknown") for event in creations)),
        "destructions_by_stage": dict(Counter(event.get("stage", "unknown") for event in destructions)),
        "sandboxes_active_at_trace_end": len(active),
        "events_total": len(events),
    }
    timed = [e for e in events if isinstance(e.get("duration_seconds"), (int, float))
             and math.isfinite(e["duration_seconds"]) and e["duration_seconds"] >= 0]
    by_event: dict[str, list] = defaultdict(list)
    by_stage: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    by_tool: dict[str, list] = defaultdict(list)
    for event in timed:
        by_event[event["event"]].append(event)
        by_stage[event.get("stage", "unknown")][event["event"]].append(event)
        if event["event"] == "tool":
            by_tool[event.get("tool_name", "unknown")].append(event)
    result["timings"] = {key: duration_stats(value) for key, value in sorted(by_event.items())}
    result["timings_by_stage"] = {stage: {key: duration_stats(value) for key, value in sorted(groups.items())}
                                  for stage, groups in sorted(by_stage.items())}
    result["timings_by_tool"] = {key: duration_stats(value) for key, value in sorted(by_tool.items())}
    result["stage_wall_seconds"] = {e["pool_stage"]: e["duration_seconds"] for e in events if e.get("event") == "stage_pool"}
    result["lifecycle_failures"] = sum(e.get("success") is False and e.get("event") in
                                       ("sandbox_create", "sandbox_destroy", "sandbox_start", "environment_init") for e in events)
    result["infrastructure_failures"] = sum(bool(e.get("infrastructure_error")) or
        (e.get("success") is False and e.get("event") in ("file_transfer", "patch_application", "tool_setup")) for e in events)
    result["timing_notes"] = [
        "Durations are inclusive monotonic host wall time, not CPU time.",
        "Tool spans contain command spans; environment_init contains creation/setup. Do not add nested totals.",
        "seconds_sum is operation work summed across workers, NOT concurrent job makespan.",
        "sandbox_create: Docker run API return (or Kubernetes create acceptance); not application readiness.",
        "sandbox_start: start_container including backend readiness waiting; environment_init: runtime construction through setup.",
        "tool_setup: command installation; agent_ready: initial/reset environment plus tools are ready for the first step.",
        "sandbox_destroy excludes prior stop; sandbox_stop includes stop plus removal. Kubernetes deletion is asynchronous.",
        "Latency distributions exclude failed operations; failures and their elapsed sum are reported separately.",
        "Quantiles use linear interpolation. Small samples, particularly p99, are descriptive, not stable tail estimates.",
    ]
    return result


def duration_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(e["duration_seconds"] for e in events if e.get("success") is not False)
    failures = [e for e in events if e.get("success") is False]
    def percentile(q):
        if not values:
            return None
        position = (len(values) - 1) * q
        left = int(position)
        return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)
    return {
        "count": len(events), "success_count": len(values), "failure_count": len(failures),
        "seconds_sum": sum(values), "failed_seconds_sum": sum(e["duration_seconds"] for e in failures),
        "mean_seconds": sum(values) / len(values) if values else None,
        "min_seconds": values[0] if values else None, "max_seconds": values[-1] if values else None,
        "p50_seconds": percentile(.5), "p95_seconds": percentile(.95), "p99_seconds": percentile(.99),
    }


def write_operations_csv(events: list[dict[str, Any]], path: Path) -> None:
    """One raw measured operation per row; retain unsuccessful operations too."""
    columns = ["task_id", "trajectory_id", "stage", "test_id", "phase", "step_idx", "worker_id",
               "event", "tool_name", "sandbox_id", "backend", "span_id", "parent_span_id",
               "start_timestamp_ns", "end_timestamp_ns", "start_monotonic_ns", "end_monotonic_ns",
               "duration_seconds", "success", "outcome", "exit_code", "expected_exit_code",
               "exit_code_matches", "command", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(e for e in events if "duration_seconds" in e)


def merge_and_write(parts_dir: Path, trace_path: Path, summary_path: Path) -> dict[str, Any]:
    events = read_shards(parts_dir)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    summary = summarize(events)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_operations_csv(events, trace_path.parent / "operations.csv")
    return summary

