#!/usr/bin/env python3
"""Replay saved R2E-Gym actions, regression and reproduction without an LLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from r2egym_trace.replay import replay_trace_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay agent tools/commands and both verifier stages without calling a model."
    )
    parser.add_argument(
        "trace_run",
        nargs="+",
        type=Path,
        help="One or more completed trace-run directories",
    )
    parser.add_argument(
        "--r2e-root",
        default=os.getenv("R2E_GYM_ROOT"),
        help="R2E-Gym checkout (or set R2E_GYM_ROOT)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Parent directory for replay outputs (default: beside each source run)",
    )
    parser.add_argument(
        "--reproduction-tests",
        type=int,
        default=None,
        help="Tests per trajectory (default: source run's T)",
    )
    parser.add_argument("--verifier-workers", type=int, default=48)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--llm-wait", choices=("recorded", "none"), default="recorded",
                        help="Preserve recorded model waits (default), or omit them for execution-only replay")
    parser.add_argument("--reward-timeout", type=int, default=None,
                        help="Agent end-of-run reward timeout; defaults to the recorded value or 300s")
    parser.add_argument("--no-command-events", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = []
        for source in args.trace_run:
            output_dir = None
            if args.output_root is not None:
                output_dir = args.output_root / f"{source.name}-replay"
            results.append(
                replay_trace_run(
                    source,
                    r2e_root=args.r2e_root,
                    output_dir=output_dir,
                    reproduction_tests=args.reproduction_tests,
                    verifier_workers=args.verifier_workers,
                    agent_workers=args.agent_workers,
                    llm_wait=args.llm_wait,
                    reward_timeout=args.reward_timeout,
                    trace_commands=not args.no_command_events,
                )
            )
    except Exception as exc:
        parser.exit(1, f"trace replay error: {exc}\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(result["run"]["measurement_valid"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
