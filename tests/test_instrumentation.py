import json
import tempfile
import time
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from r2egym_trace.instrumentation import (
    _install_command_hook, _install_tool_hook, configure_trace, timed_event, trace_scope,
)
from r2egym_trace.trace_io import merge_and_write, read_shards, summarize


class TimingTests(unittest.TestCase):
    def test_real_hooks_keep_long_commands_and_distinguish_nested_tool(self):
        class Runtime:
            repo_path = "/testbed"
            backend = "docker"
            container = types.SimpleNamespace(id="sandbox", name="test")
            def run(self, code, **kwargs):
                return "output", "0"
            def demux_run(self, code, **kwargs):
                return "output", "stderr", "0"
        class Env:
            def __init__(self):
                self.runtime = Runtime()
            def run_action(self, action, **kwargs):
                output, code = self.runtime.run(action.parameters["cmd"])
                return output, code, .001
        modules = {"r2egym.agenthub.runtime.docker": types.SimpleNamespace(DockerRuntime=Runtime),
                   "r2egym.agenthub.environment.env": types.SimpleNamespace(RepoEnv=Env)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            configure_trace(tmp, task_id="task", trajectory_id="task/r000", stage="agent")
            _install_command_hook()
            _install_tool_hook()
            command = "x" * 6000 + "\n  trailing spaces  "
            with trace_scope(step_idx=7):
                Env().run_action(types.SimpleNamespace(function_name="execute_bash", parameters={"cmd": command}))
            events = read_shards(Path(tmp))
            self.assertEqual([e["event"] for e in events], ["command", "tool"])
            self.assertEqual(events[0]["command"], command)
            self.assertEqual(events[1]["tool_parameters"]["cmd"], command)
            self.assertEqual(events[0]["parent_span_id"], events[1]["span_id"])
            self.assertEqual(events[0]["sandbox_id"], "sandbox")
            self.assertTrue(all(e["step_idx"] == 7 for e in events))

    def test_nested_spans_have_parent_ids_and_monotonic_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            configure_trace(tmp, task_id="task", trajectory_id="task/r000", stage="agent")
            with trace_scope(step_idx=3), timed_event("tool") as tool:
                with timed_event("command"):
                    time.sleep(.001)
            events = read_shards(Path(tmp))
            command, parent = events
            self.assertEqual(command["parent_span_id"], parent["span_id"])
            self.assertEqual(command["step_idx"], 3)
            self.assertGreater(tool["duration_seconds"], 0)
            self.assertGreaterEqual(parent["duration_seconds"], command["duration_seconds"])
            self.assertLessEqual(parent["start_monotonic_ns"], command["start_monotonic_ns"])
            self.assertEqual(parent["duration_seconds"], (parent["end_monotonic_ns"] - parent["start_monotonic_ns"]) / 1e9)

    def test_failed_creation_is_not_counted_as_live_and_is_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configure_trace(str(root / "parts"), task_id="task", trajectory_id="task/r000", stage="agent")
            with self.assertRaises(RuntimeError):
                with timed_event("sandbox_create", backend="docker"):
                    raise RuntimeError("creation failed")
            summary = merge_and_write(root / "parts", root / "trace.jsonl", root / "summary.json")
            self.assertEqual(summary["total_sandbox_creations"], 0)
            self.assertEqual(summary["sandboxes_active_at_trace_end"], 0)
            self.assertEqual(summary["lifecycle_failures"], 1)
            self.assertEqual(summary["timings"]["sandbox_create"]["failure_count"], 1)
            self.assertIsNone(summary["timings"]["sandbox_create"]["p99_seconds"])
            self.assertIn("creation failed", (root / "operations.csv").read_text())

    def test_work_sum_is_not_stage_makespan(self):
        events = [
            {"event": "tool", "stage": "agent", "tool_name": "bash", "duration_seconds": 2},
            {"event": "tool", "stage": "agent", "tool_name": "bash", "duration_seconds": 4},
            {"event": "stage_pool", "stage": "replay", "pool_stage": "agent", "duration_seconds": 4.5},
        ]
        summary = summarize(events)
        self.assertEqual(summary["timings_by_tool"]["bash"]["seconds_sum"], 6)
        self.assertEqual(summary["stage_wall_seconds"]["agent"], 4.5)
        self.assertEqual(summary["timings_by_tool"]["bash"]["p50_seconds"], 3)


if __name__ == "__main__":
    unittest.main()
