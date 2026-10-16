# R2E-Gym：AI sandbox workload 简要记录

记录日期：2026-10-15

## 项目定位

R2E-Gym 是开源的软件工程 Agent 训练与评测框架。它提供可执行的代码仓库任务，并研究如何通过多条编辑轨迹、测试 Agent 和混合验证器提高解题成功率。论文的主要目标是扩展训练数据和提升测试时推理效果；其中讨论了计算资源使用效率，但**系统执行效率、容器启动速度和容器吞吐量并非项目的主要优化目标**。

## Sandbox 使用方式

- 隔离环境由 `RepoEnv` 和 `DockerRuntime` 管理，可使用 Docker 容器或 Kubernetes Pod。Agent 在环境中查看与修改代码、执行命令和测试。
- 默认单条编辑轨迹创建环境后，`Agent.run()` 会调用 `env.reset()`，销毁并重建环境。因此正常路径是 **2 次创建、2 次销毁**；后续工具命令复用第二个环境。
- 执行验证器对每条轨迹另开环境：回归验证创建 2 次，每个复现测试补丁通常创建 1 次。因此，在一次编辑运行、完整验证、无重试且有 `T` 个复现测试补丁时，源码推算为 **`4 + T` 次创建／候选轨迹**。
- 公开的 200 题测试补丁数据集中，`T` 为 4–9，中位数为 9；这只描述该数据集，不能直接代表所有论文实验。
- 论文对同一问题最多采样 26 条编辑轨迹，并采样 7 次测试 Agent rollout。rollout 数不是实测容器数，也没有公开的创建速率或环境寿命分布。

## 与系统研究的关系

这个框架能形成“**同一问题的多个候选补丁 × 多个验证测试**”的环境创建与销毁负载，比只靠同时处理许多独立 issue 产生并发的案例更贴近本研究问题。批处理和验证器都支持多 worker 并行。

目前的直接故障证据是尚未合并的 [PR #31](https://github.com/R2E-Gym/R2E-Gym/pull/31)：提交者报告 16 路并发、600 个任务时出现约 22 次删除侧 HTTP 401，残留 Pod 超过 60 个。这是作者自报的生命周期清理故障，**不能证明 sandbox 创建开销或某个内核操作已成为端到端性能瓶颈**。

**研究定位：**适合作为高并发 sandbox 生命周期的候选实验负载。后续寻找对象时，关注其**实际应用场景是否对延迟、吞吐或资源消耗敏感**，以及 sandbox 的创建与销毁是否可能影响这些指标；不要求应用本身以系统效率优化为目标。公开测量是有力证据，但不是筛选的前提。

## 主要依据

- [R2E-Gym 论文摘要与项目目标](https://arxiv.org/html/2504.07164)：训练环境规模、测试时推理与性能目标。
- [论文 §4.2](https://arxiv.org/html/2504.07164#S4.SS2)：同一问题采样 26 条编辑轨迹和 7 次测试 Agent rollout。
- [Agent 启动时重建环境](https://github.com/R2E-Gym/R2E-Gym/blob/0d94c4eb9431cd195c55a7ea3abd54006c9a1735/src/r2egym/agenthub/agent/agent.py#L344)、[环境重建实现](https://github.com/R2E-Gym/R2E-Gym/blob/0d94c4eb9431cd195c55a7ea3abd54006c9a1735/src/r2egym/agenthub/environment/env.py#L63)：默认编辑轨迹的两次创建。
- [回归验证](https://github.com/R2E-Gym/R2E-Gym/blob/0d94c4eb9431cd195c55a7ea3abd54006c9a1735/src/r2egym/agenthub/verifiers/run_regression_tests.py#L25)、[复现验证](https://github.com/R2E-Gym/R2E-Gym/blob/0d94c4eb9431cd195c55a7ea3abd54006c9a1735/src/r2egym/agenthub/verifiers/run_reproduction_tests.py#L158)：验证阶段的环境创建路径。
- [公开测试补丁数据集](https://huggingface.co/datasets/R2E-Gym/R2E-TestgenAgent-Patches)：200 题的 `num_patches` 数据。
