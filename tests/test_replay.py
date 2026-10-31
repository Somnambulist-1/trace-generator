import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from replay_trace import build_parser
from r2egym_trace.instrumentation import configure_trace
from r2egym_trace.replay import _replay_agent_worker, _run_jobs, _workflow_timings, parse_saved_action, replay_trace_run
from r2egym_trace.trace_io import read_shards


def action(name, **parameters):
    return f"<function={name}>\n" + "".join(
        f"  <parameter={k}>{v}</parameter>\n" for k, v in parameters.items()) + "</function>"


def trajectory():
    ds = {"instance_id": "task-a", "docker_image": "test-image"}
    return {
        "ds": ds, "env_args": {"ds": ds}, "docker_image": "test-image", "output_patch": "patched",
        "agent_args": {"command_files": ["src/r2egym/tool.py"]}, "max_exec_time": 90,
        "reward": 1,
        "trajectory_steps": [
            {"step_idx": i, "action": act, "llm_exec_time": .1, "done": False,
             "observation": f"Exit code: {code}\n{output}"}
            for i, (act, code, output) in enumerate([
                (action("write", text="  retained newline\n"), "0", "created"),
                (action("read"), "0", "  retained newline\n"),
                (action("fails"), "Error: Exit code 1", "expected failure"),
                (action("finish"), "0", "finished"),
            ])
        ],
    }


class FakeAction:
    def __init__(self, name, parameters):
        self.function_name, self.parameters = name, parameters


class FakeObservation:
    def __init__(self, output, code="0"):
        self.output, self.error_code = output, code
    def __str__(self):
        return f"Exit code: {self.error_code}\n{self.output}"


def concurrent_job(stage, spec):
    started = time.perf_counter_ns()
    time.sleep(.08)
    return {"ok": True, "stage": stage, "task_id": spec["task_id"],
            "trajectory_id": spec["trajectory_id"], "started": started, "ended": time.perf_counter_ns()}


class FakeEnv:
    instances = []
    patch_text = "patched"
    fail_read = False
    exit_drift = False

    def __init__(self, args, **kwargs):
        self.runtime = self
        self.commit = types.SimpleNamespace(get_patch=lambda **kw: "")
        self.files = {}
        self.calls = ["create"]
        self.instances.append(self)
    def reset(self):
        self.calls.append("reset")
    def add_commands(self, paths):
        self.calls.append("install")
    def get_task_instruction(self):
        return "task"
    def step(self, act, **kwargs):
        self.calls.append(act.function_name)
        code = "0"
        if act.function_name == "write":
            self.files["file"] = act.parameters["text"]
            output = "created"
        elif act.function_name == "read":
            if self.fail_read:
                raise RuntimeError("injected execution failure")
            output = self.files["file"]
        elif act.function_name == "fails":
            code = "0" if self.exit_drift else "Error: Exit code 1"
            output = "expected failure"
        else:
            output = "finished"
        return FakeObservation(output, code), 0, act.function_name == "finish", {"total_time": .01}
    def get_patch(self):
        self.calls.append("patch")
        return self.patch_text
    def _calculate_reward(self, **kwargs):
        self.calls.append("reward")
        return 1, "test output"
    def close(self):
        self.calls.append("close")


class ReplayTests(unittest.TestCase):
    def test_lossless_action_parameters(self):
        text = "  a\n b\n"
        self.assertEqual(parse_saved_action(action("file_editor", old_str=text)), ("file_editor", {"old_str": text}))
        self.assertEqual(parse_saved_action(action("")), ("", {}))

    def test_rejects_unparseable_and_duplicate_actions(self):
        for text in ("unparsed DSML", "<function=write>oops</function>",
                     "<function=write>\n  <parameter=a>x</parameter>\n  <parameter=a>y</parameter>\n</function>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_saved_action(text)

    def test_cli_defaults_to_full_replay_with_recorded_waits(self):
        args = build_parser().parse_args(["source"])
        self.assertEqual(args.llm_wait, "recorded")
        self.assertEqual(args.agent_workers, 1)

    def _worker(self, **options):
        modules = {
            "r2egym.agenthub.action": types.SimpleNamespace(Action=FakeAction),
            "r2egym.agenthub.environment.env": types.SimpleNamespace(EnvArgs=lambda **kw: kw, RepoEnv=FakeEnv),
            "r2egym.agenthub.trajectory.trajectory": types.SimpleNamespace(
                Trajectory=types.SimpleNamespace(model_validate_json=json.loads)),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            configure_trace(tmp, task_id="task-a", trajectory_id="task-a/r000", stage="agent")
            spec = {"task_id": "task-a", "trajectory_id": "task-a/r000", "backend": "docker",
                    "trajectory_json": json.dumps(trajectory()), "command_files": ["tool.py"],
                    "reward_timeout": 300, "llm_wait": "none", **options}
            result = _replay_agent_worker(spec)
            return result, read_shards(Path(tmp))

    def setUp(self):
        FakeEnv.instances = []
        FakeEnv.patch_text = "patched"
        FakeEnv.fail_read = False
        FakeEnv.exit_drift = False

    def test_stateful_all_steps_reward_and_cleanup(self):
        result, events = self._worker()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["executed_steps"], 4)
        self.assertEqual(FakeEnv.instances[-1].calls,
                         ["create", "reset", "install", "write", "read", "fails", "finish", "patch", "reward", "close"])
        self.assertEqual(result["exit_code_mismatches"], [])
        self.assertEqual(result["observation_mismatches"], [])
        self.assertEqual(len([e for e in events if e["event"] == "agent_step"]), 4)
        self.assertTrue(all(s["llm_exec_time"] == 0 for s in json.loads(result["trajectory_json"])["trajectory_steps"]))

    def test_patch_drift_does_not_fall_back_to_original(self):
        FakeEnv.patch_text = "different patch"
        result, _ = self._worker()
        self.assertFalse(result["ok"])
        self.assertEqual(json.loads(result["trajectory_json"])["output_patch"], "different patch")
        self.assertEqual(FakeEnv.instances[-1].calls[-1], "close")

    def test_cleanup_after_failure(self):
        FakeEnv.fail_read = True
        result, events = self._worker()
        self.assertFalse(result["ok"])
        self.assertEqual(FakeEnv.instances[-1].calls[-1], "close")
        self.assertTrue(any(e.get("success") is False and e["event"] == "agent_step" for e in events))
        actual = json.loads(result["trajectory_json"])
        self.assertIsNone(actual["output_patch"])
        self.assertIsNone(actual["reward"])

    def test_exit_drift_is_failure_but_all_actions_execute(self):
        FakeEnv.exit_drift = True
        result, _ = self._worker()
        self.assertFalse(result["ok"])
        self.assertEqual(result["executed_steps"], 4)
        self.assertEqual(result["exit_code_mismatches"][0]["step_idx"], 2)

    def test_wait_policy_does_not_call_any_model(self):
        with patch("r2egym_trace.replay.time.sleep") as sleep:
            result, events = self._worker(llm_wait="recorded")
        self.assertTrue(result["ok"])
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(len([e for e in events if e["event"] == "llm_wait"]), 4)

    def test_replaying_output_preserves_source_pacing(self):
        first, _ = self._worker(llm_wait="none")
        with patch("r2egym_trace.replay.time.sleep") as sleep:
            second, events = self._worker(trajectory_json=first["trajectory_json"], llm_wait="recorded")
        self.assertTrue(second["ok"])
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual([e["requested_seconds"] for e in events if e["event"] == "llm_wait"], [.1] * 4)

    def test_preflight_rejects_missing_tests_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "run.json").write_text(json.dumps({"T": 2}))
            (source / "trajectories.jsonl").write_text(json.dumps(trajectory()))
            (source / "reproduction_inputs.json").write_text(json.dumps({"test-image": ["patch"]}))
            with self.assertRaisesRegex(ValueError, "fewer reproduction tests"):
                replay_trace_run(source, r2e_root=str(root), output_dir=root / "output")
            self.assertFalse((root / "output").exists())

    def test_parallel_jobs_overlap_and_results_keep_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            configure_trace(tmp, task_id="__run__", trajectory_id="__run__", stage="replay")
            specs = [{"task_id": "task", "trajectory_id": f"task/r{i:03d}"} for i in range(2)]
            with patch("r2egym_trace.replay._timed_job", concurrent_job):
                results = _run_jobs("agent", specs, 2)
            self.assertEqual([r["trajectory_id"] for r in results], ["task/r000", "task/r001"])
            self.assertLess(max(r["started"] for r in results), min(r["ended"] for r in results))

    def test_full_workflow_latency_includes_barrier_and_handles_t_zero(self):
        jobs = [
            {"trajectory_id": "a/r000", "task_id": "a", "stage": "agent", "ok": True,
             "submitted_monotonic_ns": 1_000_000_000, "finished_monotonic_ns": 2_000_000_000},
            {"trajectory_id": "a/r000", "task_id": "a", "stage": "regression_test", "ok": True,
             "submitted_monotonic_ns": 4_000_000_000, "finished_monotonic_ns": 6_000_000_000},
        ]
        row = _workflow_timings(jobs, 0)[0]
        self.assertTrue(row["success"])
        self.assertEqual(row["duration_seconds"], 5)
        self.assertFalse(_workflow_timings(jobs, 1)[0]["success"])

    def test_driver_orders_stages_and_retains_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            tool = root / "src/r2egym/tool.py"
            tool.parent.mkdir(parents=True)
            tool.write_text("# fixture\n")
            (source / "run.json").write_text(json.dumps({"T": 1, "R": 1}))
            original = json.dumps(trajectory())
            (source / "trajectories.jsonl").write_text(original)
            (source / "reproduction_inputs.json").write_text(json.dumps({"test-image": ["test patch"]}))
            stages = []
            def run_jobs(stage, specs, workers):
                stages.append(stage)
                return [{**s, "ok": True, "stage": stage, "score": 0,
                         "submitted_monotonic_ns": 100, "finished_monotonic_ns": 200} for s in specs]
            with patch("r2egym_trace.replay._run_jobs", side_effect=run_jobs), \
                 patch("r2egym_trace.replay._image_inventory", return_value={}):
                result = replay_trace_run(source, r2e_root=str(root), output_dir=root / "out", llm_wait="none")
            self.assertEqual(stages, ["agent", "regression_test", "reproduction_test"])
            self.assertEqual(result["run"]["reproduction_jobs"], 1)
            self.assertEqual((source / "trajectories.jsonl").read_text(), original)
            self.assertEqual(json.loads((root / "out/reproduction_inputs.json").read_text()), {"test-image": ["test patch"]})
            self.assertTrue((root / "out/operations.csv").exists())

    def test_driver_does_not_verify_failed_agent_with_source_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            tool = root / "src/r2egym/tool.py"
            tool.parent.mkdir(parents=True)
            tool.write_text("# fixture\n")
            (source / "run.json").write_text(json.dumps({"T": 0}))
            (source / "trajectories.jsonl").write_text(json.dumps(trajectory()))
            received = []
            def run_jobs(stage, specs, workers):
                received.append((stage, len(specs)))
                return [{"trajectory_id": s["trajectory_id"], "stage": stage,
                         "ok": False, "error": "patch mismatch"} for s in specs]
            with patch("r2egym_trace.replay._run_jobs", side_effect=run_jobs), \
                 patch("r2egym_trace.replay._image_inventory", return_value={}):
                result = replay_trace_run(source, r2e_root=str(root), output_dir=root / "out")
            self.assertEqual(received, [("agent", 1), ("regression_test", 0), ("reproduction_test", 0)])
            self.assertFalse(result["run"]["measurement_valid"])
            self.assertEqual((root / "out/verified_trajectories.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
