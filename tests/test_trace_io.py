import json
import tempfile
import unittest
from pathlib import Path

from r2egym_trace.instrumentation import configure_trace, emit
from r2egym_trace.trace_io import merge_and_write, summarize


class TraceIoTests(unittest.TestCase):
    def test_summary_counts_lifecycle_and_peak(self):
        events = [
            {"event": "sandbox_create", "stage": "agent", "backend": "docker", "sandbox_id": "a"},
            {"event": "sandbox_create", "stage": "reproduction_test", "backend": "docker", "sandbox_id": "b"},
            {"event": "sandbox_destroy", "stage": "agent", "backend": "docker", "sandbox_id": "a"},
            {"event": "sandbox_destroy", "stage": "reproduction_test", "backend": "docker", "sandbox_id": "b"},
        ]
        summary = summarize(events)
        self.assertEqual(summary["total_sandbox_creations"], 2)
        self.assertEqual(summary["total_sandbox_destructions"], 2)
        self.assertEqual(summary["peak_concurrent_sandboxes"], 2)
        self.assertEqual(summary["creations_by_stage"], {"agent": 1, "reproduction_test": 1})
        self.assertEqual(summary["sandboxes_active_at_trace_end"], 0)

    def test_process_shard_is_merged_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = root / "parts"
            configure_trace(
                str(parts),
                task_id="task-1",
                trajectory_id="task-1/r000",
                stage="agent",
            )
            emit("sandbox_create", sandbox_id="sandbox-1", backend="docker")
            emit("sandbox_destroy", sandbox_id="sandbox-1", backend="docker")
            summary = merge_and_write(parts, root / "trace.jsonl", root / "summary.json")
            records = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            self.assertEqual([record["event"] for record in records], ["sandbox_create", "sandbox_destroy"])
            self.assertTrue(all(record["task_id"] == "task-1" for record in records))
            self.assertEqual(summary["peak_concurrent_sandboxes"], 1)


if __name__ == "__main__":
    unittest.main()

