# 完整应用重放与计时

重放在真实环境里执行保存的 Agent 工具动作，再执行 regression 和 reproduction。
模型不重新推理；本次动作序列固定为原记录的动作。如果反馈变化，不会让模型重新决策。

## 执行流程

1. 读取 `trajectories.jsonl`，检查每步动作和完整参数。
2. 按上游流程创建 `RepoEnv`、调用 `reset()`、安装工具。上游本来就先创建再 reset，
   因此保留 Agent 阶段的两个环境生命周期。
3. 按原 trace 记录的顺序，依次调用 `env.step(action)` 执行 Agent 工具动作。
4. 从实际文件系统收集 patch，检查是否与原记录一致，执行最终 reward 测试并关闭环境。
5. 对一致性检查通过的候选执行原 regression 函数，再执行 T 个 reproduction。
   验证阶段按上游逻辑创建环境、应用候选 patch、执行真实测试并关闭环境。

初始化、工具安装、patch 提取、reward 和验证过程里的 runtime command 也计时。
`execute_bash` 内部每个 OS 子进程没有单独计时；整个 shell command 及其外层工具分别计时。
无损保留参数中的空格、换行；原本明确没有工具的步骤保留为 no-op，损坏的动作提前报错。

## 运行

在 trace-generator 根目录执行，不需要 API key：

```bash
R2E-Gym/.venv/bin/python replay_trace.py \
  trace-runs/deepseek-native-tools-full-001 \
  --r2e-root "$PWD/R2E-Gym" \
  --agent-workers 1 --verifier-workers 1 \
  --llm-wait recorded \
  --output-root trace-runs/full-replays
```

输出为 `<output-root>/<原目录名>-replay`，拒绝覆盖非空目录。
多个输入目录依次执行；单个目录中的轨迹由 `--agent-workers` 并行执行，
验证并发由 `--verifier-workers` 控制。三个阶段保留原来的整批完成屏障。
这些参数不产生额外轨迹，也不代表容器数。

- `--llm-wait recorded`：默认，每步等待原 `llm_exec_time` 再执行动作。不调用模型，
  不模拟旧工具耗时，工具耗时由本次实际执行产生。
- `--llm-wait none`：去掉模型等待，工具仍真实执行。这改变存活时长和请求节奏，
  适合调试或明确声明的 execution-only 实验，不与 recorded 混为同一负载。
- `--reproduction-tests T`：默认原 run.json 的 T。不足 T 个测试时提前报错。
- `--reward-timeout S`：默认沿用原 manifest；旧记录缺少此值时使用上游默认 300 秒。
- `--no-command-events`：关闭 command 明细，工具及生命周期计时保留。

## 时间边界

duration 使用 `time.perf_counter_ns()`，是现实经过的时间，包含等待和调度，不是 CPU 时间。
事件保存起止时间、worker、task、trajectory、stage、step、container 和父 span。

| 数据 | 边界与含义 |
|---|---|
| `run.preparation_seconds` | 函数入口到 Agent 池启动前：读输入、获取 reproduction 数据、检查参数、源码校验、初始清单写入 |
| `run.workload_seconds` | Agent 池启动前到最后一个验证池退出：完整应用重放主指标，含池启动、排队、环境创建/初始化、动作、可选模型等待、验证和销毁 |
| `run.finalization_seconds` | 最后一个池退出到结果组装结束：结果保存、事件合并、统计和 CSV 导出 |
| `run.elapsed_seconds` | 重放函数入口到结果组装结束；不含 Python 启动、最终 manifest/summary 写入及终端打印 |
| `stage_wall_seconds` | 某阶段池启动前到该池退出，包含阶段的全部并行任务 |
| `job_queue` | 提交任务到 worker 接手，含进程启动/调度等待 |
| `worker_initialization` / `verifier_initialization` | worker 路径与 hooks 初始化 / 验证模块首次导入、数据加载、轨迹校验，分别记录 |
| `stage_job` | worker 接手到任务返回，不含队列等待 |
| `sandbox_create` | Docker containers.run 调用区间；Kubernetes 是创建请求接受时延，不能当成 ready |
| `sandbox_start` | R2E-Gym start_container 的区间，含复用检查和后端就绪等待 |
| `environment_init` | DockerRuntime 构造开始到 setup_env 后返回，含创建及环境准备 |
| `environment_setup` | setup_env 调用区间：创建后的环境准备 |
| `tool_setup` | 工具脚本复制、权限设置等安装时间 |
| `agent_initialization` | Agent 首次建环境、reset、工具安装的总区间；agent_ready 标记首次动作就绪 |
| `tool` | RepoEnv.run_action 的调用区间，每次工具的耗时、参数和退出码 |
| `command` | DockerRuntime.run/demux_run 区间，每条 runtime command 的耗时、完整文本和退出码 |
| `llm_wait` | 模拟记录中模型等待的耗时，不能称为本次模型推理耗时 |
| `agent_finalize` / `agent_reward` | 收集最终 patch / 执行最终评分测试 |
| `sandbox_stop` | stop + remove；Kubernetes 物理资源回收可能更晚 |
| `sandbox_destroy` | remove/delete API 自身耗时，不含之前的 stop |

不要重复累加：tool 包含 command；environment_init 包含创建和 setup。
`seconds_sum` 是跨 worker 的操作耗时之和，并发时可以大于整轮墙钟时间。
整体加速用同一边界的 `workload_seconds` 比较，不能用工具耗时之和代替。

论文可同时报告 sandbox_create、environment_init 的分布和 workload_seconds。
`successful_trajectories_per_second` 是有效整轮实验中的轨迹数 / workload_seconds，
不是创建吞吐量；其分母还包含工具、测试、可选模型等待。

## 输出

| 文件 | 内容 |
|---|---|
| run.json | 配置、源码/输入 hash、总时间、有效性和错误 |
| trace.jsonl | 全部事件与父子关系，每次 tool/command 的独立记录 |
| operations.csv | 每行一次计时操作，保留失败操作，可用来画分布 |
| summary.json | 生命周期计数和按 stage、事件、工具名分组的计时 |
| replay_results.json | 每个任务排队/执行时间、实际步数、patch/退出码/观察输出比较 |
| workflow_timings.json | 每条轨迹从 Agent 提交到最后一个验证任务结束的时长，包括中间阶段屏障等待 |
| source_trajectories.jsonl | 原始记录备份，保留原 token 和旧时间 |
| trajectories.jsonl | 本次 Agent 的实际输出、patch 和时间；模型 token/time 为 0 |
| verified_trajectories.jsonl | 本次验证结果；只验证 Agent 重放一致的候选 |
| reproduction_inputs.json | 实际使用的 reproduction patch，后续重放优先读取 |

仓库中的 [`examples/full-replay-summary.json`](../examples/full-replay-summary.json)
是一条完整应用重放实际生成的 `summary.json` 示例。

统计提供 count、成功/失败数量、sum、mean、min/max、p50/p95/p99。
每步 info 另存原始模型等待时间，因此再次重放本次输出时仍能保留原始等待节奏。
失败的中间结果中，尚未实际收集的 patch/reward 为 null，不会沿用原记录伪装成功。
成功延迟分布与失败数量/耗时分开；不能只报告成功请求而隐去失败。
少量样本的 p99 只是描述性数字，不能证明高并发下的稳定尾延迟。

## 一致性与限制

- 所有已保存步骤都会执行，包括预期失败和显式 no-op，不因 finish 擅自截断余下步骤。
- 原 observation 含明确退出码时核对退出码；输出差异另记，时间/临时路径等可能导致
  输出字节不同，因此不以所有输出完全一致作为默认门槛。
- 最终 patch 必须完全相同。不同则保留实际结果、记录失败、跳过该候选的后续验证；
  不会偷偷应用旧 patch 掩盖差异。
- 应用测试失败与基础设施失败分开。预期的非零退出码不是重放失败；创建/销毁、
  文件传输、patch 应用失败和观测到的验证超时使测量无效，CLI 返回非零。
- 上游 reproduction 可能重试或把异常折算为 0 分。应结合事件和任务结果判断，
  不能把所有 0 分都解释为应用测试失败。
- 旧 trace 未保存所有历史镜像 digest、工具源码或数据集快照，文件名不能保证历史
  环境逐字节相同。当前记录本地镜像 ID/digest、工具/源码 hash 并保存 reproduction patch；首次重放旧记录
  仍需要匹配的镜像、R2E-Gym 版本、数据集。
- instrumentation 有开销，所有对照组使用同一版本；目前不分解单个 command 内每个
  OS 子进程或内核资源生命周期。
