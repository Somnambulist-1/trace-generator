"""Merge trace shards and compute the minimal lifecycle summary."""

from __future__ import annotations

import json
from collections import Counter
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
    creations = [event for event in events if event.get("event") == "sandbox_create"]
    destructions = [event for event in events if event.get("event") == "sandbox_destroy"]
    active: set[tuple[str, str]] = set()
    peak = 0
    for event in events:
        key = (
            str(event.get("backend", "unknown")),
            str(event.get("sandbox_id") or event.get("sandbox_name") or "unknown"),
        )
        if event.get("event") == "sandbox_create":
            active.add(key)
            peak = max(peak, len(active))
        elif event.get("event") == "sandbox_destroy":
            active.discard(key)

    return {
        "total_sandbox_creations": len(creations),
        "total_sandbox_destructions": len(destructions),
        "peak_concurrent_sandboxes": peak,
        "creations_by_stage": dict(Counter(event.get("stage", "unknown") for event in creations)),
        "destructions_by_stage": dict(Counter(event.get("stage", "unknown") for event in destructions)),
        "sandboxes_active_at_trace_end": len(active),
        "events_total": len(events),
    }


def merge_and_write(parts_dir: Path, trace_path: Path, summary_path: Path) -> dict[str, Any]:
    events = read_shards(parts_dir)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    summary = summarize(events)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary

