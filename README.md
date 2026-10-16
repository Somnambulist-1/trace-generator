# R2E-Gym trace generator prototype

This prototype runs the **real R2E-Gym Agent pipeline**, followed by R2E-Gym's
regression and reproduction-test functions, and records sandbox lifecycle events
as JSONL. It does not mock the Agent and does not replay saved trajectories.

The prototype is deliberately separate from R2E-Gym. Instrumentation is installed
at worker startup and does not patch the R2E-Gym checkout on disk.

## Requirements

- Linux with Docker, or access to the Kubernetes backend used by R2E-Gym.
- A checkout of [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym).
- R2E-Gym's Python dependencies and images, installed as described upstream.
- An API key for the selected LiteLLM model provider.

The source analysis and prototype were made against R2E-Gym commit
`0d94c4eb9431cd195c55a7ea3abd54006c9a1735`. Pinning that revision is recommended
for reproducible traces.

Example setup:

```bash
git clone https://github.com/R2E-Gym/R2E-Gym.git
cd R2E-Gym
git checkout 0d94c4eb9431cd195c55a7ea3abd54006c9a1735
uv sync
source .venv/bin/activate
cd /path/to/AI-agent-research
export R2E_GYM_ROOT=/path/to/R2E-Gym
```

## API key

Do not put a key in source code. Set the provider's normal environment variable
before starting the trace generator:

```bash
# Default --llm-name gpt-4o
export OPENAI_API_KEY="..."
```

For another provider, set its key and tell the launcher which variable must be
present:

```bash
export ANTHROPIC_API_KEY="..."
python trace_generator.py ... \
  --llm-name claude-3-5-sonnet-20241022 \
  --api-key-env ANTHROPIC_API_KEY
```

The key remains in the environment inherited by Agent worker processes and is
never written to the trace. A local model endpoint can use `--no-api-key-check`.

## Run

`--task` accepts exact IDs from the selected dataset and can be supplied more than
once. `R` is the number of real Agent attempts per task. `T` is the number of
R2E-Gym reproduction tests executed for every successful trajectory.

Minimal invocation (this starts a real paid/model-backed Agent run):

```bash
python trace_generator.py \
  --r2e-root "$R2E_GYM_ROOT" \
  --task django__django-11333 \
  --R 1 \
  --T 1 \
  --output-dir ./trace-runs/smoke-001
```

Multiple tasks and higher in-task pressure are selected explicitly:

```bash
python trace_generator.py \
  --task django__django-11333 astropy__astropy-12907 \
  --R 4 \
  --T 9 \
  --agent-workers 4 \
  --verifier-workers 32 \
  --output-dir ./trace-runs/run-001
```

The default dataset is `R2E-Gym/SWE-Bench-Verified`, because the current
regression verifier and public reproduction-test patches are keyed to those
instances. The launcher fails before starting Agents if a task has fewer than
the requested `T` tests; it never silently treats metadata as executed sandboxes.
`T` is a test count, not a sandbox count: R2E-Gym may retry a reproduction test,
and every observed create/destroy call is recorded separately. Churn and peak
concurrency must therefore be read from `trace.jsonl`/`summary.json`, not inferred
from `R` or `T`.

Agent attempts use a `ProcessPoolExecutor`, the same parallel-worker mechanism as
R2E-Gym's batch runner. By default, up to `R` Agent attempts run concurrently.
`--agent-workers` and `--verifier-workers` make concurrency explicit.

R2E-Gym's current verifier implementation creates Docker `RepoEnv` instances even
when the Agent stage uses `--backend kubernetes`. The trace records the actual
backend used by every lifecycle API call instead of assuming one backend for the
whole run.

## Output

The selected output directory contains:

```text
trace.jsonl                    merged, timestamp-ordered events
summary.json                   lifecycle counts and peak concurrency
run.json                       requested workload and stage errors
trajectories.jsonl             real Agent outputs
verified_trajectories.jsonl    trajectories after regression/reproduction stages
.trace-parts/                  per-process trace shards retained for audit
```

Minimal lifecycle records contain:

```json
{"timestamp":"2026-10-15T08:42:11.402331+00:00","timestamp_ns":1792053731402331000,"task_id":"django__django-11333","worker_id":24192,"trajectory_id":"django__django-11333/r000","stage":"agent","event":"sandbox_create","sandbox_id":"6f84...","backend":"docker","outcome":"container_running"}
{"timestamp":"2026-10-15T08:42:12.033104+00:00","timestamp_ns":1792053732033104000,"task_id":"django__django-11333","worker_id":24192,"trajectory_id":"django__django-11333/r000","stage":"agent","event":"sandbox_destroy","sandbox_id":"6f84...","backend":"docker","outcome":"container_removed"}
```

Those two lines illustrate the emitted schema; timestamps and IDs vary on every
run. The project does not include a claimed R2E-Gym measurement because no model
key or live Docker/Kubernetes backend was supplied during implementation.

An illustrative `summary.json` (also not a measurement from this development
machine) looks like:

```json
{
  "total_sandbox_creations": 2,
  "total_sandbox_destructions": 2,
  "peak_concurrent_sandboxes": 1,
  "creations_by_stage": {"agent": 2},
  "destructions_by_stage": {"agent": 2},
  "sandboxes_active_at_trace_end": 0,
  "events_total": 14
}
```

## Event semantics and limitations

- Docker `sandbox_create` is emitted only after the real
  `docker.containers.run` call returns. Connecting to a pre-existing named
  container is not counted as creation.
- Docker `sandbox_destroy` is emitted only after `container.remove` returns.
- Kubernetes events hook the real `create_namespaced_pod` and
  `delete_namespaced_pod` calls. Their `outcome` fields say
  `pod_create_accepted` and `pod_delete_accepted`; deletion acceptance is not a
  claim that kernel resources have already been reclaimed.
- `peak_concurrent_sandboxes` is computed strictly from observed create/destroy
  events. A missing destroy remains active at trace end and is visible in the
  summary; the generator does not invent cleanup events.
- Commands are recorded without stdout/stderr and are truncated to 4096
  characters. Use `--no-command-events` to record lifecycle only.
- R2E-Gym's reproduction timeout wrapper forks a child process. The supported
  execution platform is Linux, where its current implementation and inherited
  instrumentation work together.
- This is application-level instrumentation. It intentionally does not claim
  kernel resource lifetime, cgroup lifetime, or image-pull timing. Additional
  kernel tracing can later correlate on `sandbox_id`, timestamps, PID and stage.

## Local checks without an API key

These checks do not start an Agent or a container:

```bash
python -m unittest discover -s tests -v
python -m py_compile trace_generator.py r2egym_trace/*.py
```

