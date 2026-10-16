import unittest

from r2egym_trace.runner import build_parser


class RunnerParserTests(unittest.TestCase):
    def test_required_workload_arguments_are_parsed(self):
        args = build_parser().parse_args(
            [
                "--task",
                "task-a",
                "task-b",
                "--R",
                "4",
                "--T",
                "9",
                "--r2e-root",
                "/tmp/R2E-Gym",
            ]
        )

        self.assertEqual(args.task, [["task-a", "task-b"]])
        self.assertEqual(args.rollouts, 4)
        self.assertEqual(args.reproduction_tests, 9)

    def test_task_flag_is_repeatable(self):
        args = build_parser().parse_args(
            [
                "--task",
                "task-a",
                "--task",
                "task-b",
                "--R",
                "1",
                "--T",
                "0",
            ]
        )

        self.assertEqual(args.task, [["task-a"], ["task-b"]])


if __name__ == "__main__":
    unittest.main()
