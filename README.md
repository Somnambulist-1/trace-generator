# R2E-Gym trace generator prototype

This prototype runs the **real R2E-Gym Agent pipeline**, followed by R2E-Gym's
regression and reproduction-test functions, and records sandbox lifecycle events
as JSONL. The generator calls the real Agent model; `replay_trace.py` instead
executes recorded Agent tools/commands and both verifier stages without a model.

The prototype is deliberately separate from R2E-Gym. Instrumentation is installed
at worker startup and does not patch the R2E-Gym checkout on disk.

## Requirements

- Linux with Docker, or access to the Kubernetes backend used by R2E-Gym.
- Python 3.10, 3.11, 3.12, or 3.13 for the pinned R2E-Gym lockfile (the setup
  script rejects newer interpreters rather than triggering incompatible source
  builds).
- A checkout of [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym).
- R2E-Gym's Python dependencies and images, installed as described upstream.
- An API key for the selected LiteLLM model provider (generation only, not replay).

The source analysis and prototype were made against R2E-Gym commit
`0d94c4eb9431cd195c55a7ea3abd54006c9a1735`. Pinning that revision is recommended
for reproducible traces.

### Linux VM setup

Running inside a Linux VM is supported and avoids the Docker Desktop/VMware
hypervisor conflict on the Windows host. Linux containers inside the guest use
the guest kernel; Docker Engine does not require another VM layer inside it.

Copy this project directory into the VM, install Docker Engine, and run:

```bash
cd /path/to/AI-agent-research
chmod +x setup_vm.sh verify_vm.sh
./setup_vm.sh
```

For the prepared bundle, copy `dist/r2e-trace-vm-bundle.tar.gz` to the VM and
extract it first:

```bash
mkdir -p ~/r2e-trace
tar -xzf r2e-trace-vm-bundle.tar.gz -C ~/r2e-trace
cd ~/r2e-trace
chmod +x setup_vm.sh verify_vm.sh
./setup_vm.sh
```

The script uses R2E-Gym's committed `uv.lock`, verifies the tested R2E-Gym
revision, builds `R2E-Gym/.venv`, runs the local tests, installs the lifecycle
hooks, and verifies Docker access. It does not call a model or start an Agent.
If the VM has more than one supported interpreter, set `R2E_PYTHON_BIN` to the
one to use; otherwise the script prefers Python 3.13, then 3.12, 3.11, and
3.10.

On Ubuntu/Debian, Docker Engine can be installed before running the script with:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and back in after changing Docker group membership, then run
`./setup_vm.sh` again. Later, `./verify_vm.sh` performs a non-Agent readiness
check.

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

### DeepSeek V4.1 Flash (where to enter the key)

A dedicated launcher script is provided, so no manual Python environment
activation is needed. Run it from a terminal in this project directory:

```bash
./run_deepseek.sh \
  --task django__django-11333 --R 1 --T 1 \
  --output-dir ./trace-runs/deepseek-smoke-001
```

When the terminal shows `DeepSeek API key (hidden):`, paste the key and press
Enter. Characters are not echoed while typing. The script passes the key to the
Agent through the `DEEPSEEK_API_KEY` environment variable and does not save it
to a file. Do not post the key to chat or write it into code. The command above
starts a real Agent once the key is entered and incurs API charges.

To reuse the key across later runs in the current terminal, first run:

```bash
read -rsp "DeepSeek API key (hidden): " DEEPSEEK_API_KEY
printf '\n'
export DEEPSEEK_API_KEY
```

Then running the same launcher script will not prompt again. Afterwards,
`unset DEEPSEEK_API_KEY` clears the variable in the current terminal. To view
the arguments only, run `./run_deepseek.sh --help`; no key is required.

Per the [DeepSeek official docs](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
(checked 2026-10-19), `deepseek-flash` currently corresponds to
DeepSeek-V4.1-Flash-1017. The script uses the LiteLLM model name
`deepseek/deepseek-flash` and the official endpoint `https://api.deepseek.com`,
keeping the server's default thinking mode. The current R2E-Gym uses a
text-command scaffold for this model, and the script explicitly disables native
function calling. A valid key is still required after configuration; development
verification did not make real model requests.

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

## Full application replay

Replay runs **Agent tools/commands → regression → reproduction**. It creates the
original environments, executes every saved step in order with persistent
in-container state, collects the resulting patch, runs the Agent's final reward
test, then runs the real verifier functions. It does not apply the saved final
patch as a substitute for executing the Agent's edits.

```bash
R2E-Gym/.venv/bin/python replay_trace.py \
  trace-runs/deepseek-native-tools-full-001 \
  --r2e-root "$PWD/R2E-Gym" \
  --agent-workers 1 --verifier-workers 1 \
  --llm-wait recorded \
  --output-root trace-runs/full-replays
```

`--llm-wait recorded` (default) waits for each recorded model-call duration
without making a model call. `--llm-wait none` removes those waits and is an
explicit execution-only workload. Both actually execute tools and tests.
Do not compare the two policies as if they were the same workload.

The replay returns nonzero if a recorded exit code or final patch diverges,
a stage fails/is missing, or observed lifecycle/infrastructure failures occur.
Failed/partial records are retained. A failing application test is not by itself
a replay failure: it may be exactly what the original Agent observed.

See [the replay and timing guide](docs/replay.md) for measurement boundaries,
operation CSVs, latency summaries and reproducibility limits.

## Output

The selected output directory contains:

```text
trace.jsonl                    merged, timestamp-ordered events
summary.json                   lifecycle counts and timing distributions
run.json                       requested workload and stage errors
trajectories.jsonl             real Agent outputs
verified_trajectories.jsonl    trajectories after regression/reproduction stages
.trace-parts/                  per-process trace shards retained for audit
```

`run.json` records the full tracing command's wall-clock duration in
`elapsed_seconds`; the same value appears in the final CLI JSON under `run`.

Minimal lifecycle records contain:

```json
{"timestamp":"2026-10-15T08:42:11.402331+00:00","timestamp_ns":1792053731402331000,"task_id":"django__django-11333","worker_id":24192,"trajectory_id":"django__django-11333/r000","stage":"agent","event":"sandbox_create","sandbox_id":"6f84...","backend":"docker","outcome":"container_running"}
{"timestamp":"2026-10-15T08:42:12.033104+00:00","timestamp_ns":1792053732033104000,"task_id":"django__django-11333","worker_id":24192,"trajectory_id":"django__django-11333/r000","stage":"agent","event":"sandbox_destroy","sandbox_id":"6f84...","backend":"docker","outcome":"container_removed"}
```

Those two lines illustrate the emitted schema; timestamps and IDs vary on every
run. A complete `summary.json` produced by a real single-trajectory application
replay is included at
[`examples/full-replay-summary.json`](examples/full-replay-summary.json). It is a
validation example, not a concurrency benchmark.

## Event semantics and limitations

- Docker `sandbox_create` is emitted only after the real
  `docker.containers.run` call returns. Connecting to a pre-existing named
  container is not counted as creation.
- Docker `sandbox_destroy` is emitted only after `container.remove` returns.
- Kubernetes events hook the real `create_namespaced_pod` and
  `delete_namespaced_pod` calls. Their `outcome` fields say
  `pod_create_accepted` and `pod_delete_accepted`; deletion acceptance is not a
  claim that kernel resources have already been reclaimed.
- `tool` events record each high-level agent tool action dispatched by R2E-Gym,
  including `tool_name` and `duration_seconds`. A nested `command` event may
  also be present when that tool runs a shell command; it measures only the
  lower-level command execution.
- `peak_concurrent_sandboxes` is computed strictly from observed create/destroy
  events. A missing destroy remains active at trace end and is visible in the
  summary; the generator does not invent cleanup events.
- Commands retain their complete text and an output hash/size, without raw
  stdout/stderr. `--no-command-events` omits command-level records; tool and
  lifecycle timings remain. Older traces may contain truncated commands;
  replay reads the full saved trajectory actions instead.
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

