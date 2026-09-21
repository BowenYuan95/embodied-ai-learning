# Embodied AI Learning Progress

Last updated: 2026-09-22

This is the single source of truth for current project status. Use Git history
instead of creating date-suffixed progress files.

## Evidence Markers

Every status claim below carries one of these markers. Do not promote a claim to
a stronger marker without recording the command or read-back that justifies it.

- `[verified]` — reproduced by a command, read-back, or file inspection in the
  current environment on this machine.
- `[reported]` — stated by the learner and consistent with on-disk artifacts,
  but not re-executed in this session.
- `[stale]` — evidence exists, but it was produced in an environment that no
  longer exists on this machine and cannot be reproduced as-is.
- `[open]` — known unresolved contradiction or missing evidence.

## Current Position

Lesson 2 is in its closing phase. The lesson is defined as one continuous
主线: **从仿真轨迹走到一个能够训练的最小行为克隆模型**. It is not a lesson
about building more engineering environment, and it does not yet evaluate a
policy in closed loop.

| Sub-step | Topic | Status |
|---|---|---|
| 2.1 | ManiSkill 环境与轨迹：`PickCube-v1`、observation/action/reward/done、完整 episode、`o_t → a_t → o_{t+1}` | Complete |
| 2.2 | 机器人状态与动作空间：42 维 `observation.state`、8 维 action、7 关节 + 1 夹爪、`pd_joint_delta_pos` | Complete |
| 2.3 | 时间对齐：frame / timestamp / episode，训练配对是 `(o_t, a_t)` 而不是 `(o_t, a_{t+1})` | Complete |
| 2.4 | 原始轨迹存储：ManiSkill HDF5、episode 分组、observation/action 长度 | Complete |
| 2.5 | 转换为 LeRobot Dataset：`data/`、`meta/`、任务描述与 episode 元数据 | Complete |
| 2.6 | 数据验证与质量检查：shape / dtype / NaN / 范围、连续性、action 与 reward 曲线 | Complete |
| 2.7 | PyTorch `Dataset` 与 `DataLoader`：单样本、batch、shuffle、随机采样与时间顺序 | Complete |
| 2.8.1 | 数据读取：`observation.state [42]`、`action [8]` | Complete |
| 2.8.2 | Batch：`states [8,42]`、`actions [8,8]` | Complete |
| 2.8.3 | MLP Policy：`R^42 → R^128 → R^128 → R^8` | Complete |
| 2.8.4 | Loss：`L = MSE(â_t, a_t)` | Complete |
| 2.8.5 | 反向传播与参数更新：`zero_grad → forward → loss → backward → step` | Complete |
| 2.8.6 | 完整训练循环：batch 循环、epoch 循环、Adam、Loss 曲线、100 epochs | **Complete `[verified]`** |
| 2.8.7 | 训练集与验证集 | **Next — 进行中** |
| 2.8.8 | 策略部署与闭环执行 | Not started |

**Current stopping point: 2.8.6 完成，2.8.7 「训练集与验证集」为下一步。**

Estimated Lesson 2 completion: **about 90%**, based on 2.1–2.8.6 completing and
only 2.8.7 and 2.8.8 remaining.

The training pipeline is complete in the narrow sense: a state-based MLP can be
trained end to end from the converted dataset. It is **not** complete in the
sense that would justify calling the result a policy — the model has never been
evaluated on held-out data and has never acted in the environment.

### Why 2.8.7 comes next

The next step is deliberately not more engineering. It is a train/validation
experiment whose purpose is to expose the theory that opens Lesson 3:

- why training loss alone is not enough;
- train/validation split, generalization versus memorization;
- underfitting and overfitting;
- why randomly distributed action data cannot yield a usable policy.

## Lesson 2 Evidence

### 2.1–2.6 — Environment, trajectory, dataset, validation

- [x] Created and ran `PickCube-v1`, with the full `o_t → a_t → o_{t+1}` loop.
  `[reported]`
- [x] Decomposed the 42-dimensional `observation.state` and the 8-dimensional
  action (7 arm joints + 1 gripper). `[reported]` — see "Observation
  Decomposition" for the open item on per-field naming.
- [x] Established the `(o_t, a_t)` pairing convention. `[reported]`
- [x] Stored the source trajectory as ManiSkill HDF5 with episode grouping.
  `[verified]` — `datasets/pickcube/*.h5` present.
- [x] Converted the trajectory to LeRobot dataset format with `data/` and
  `meta/`. `[verified]` — `datasets/lerobot/pickcube/` present with
  `data/chunk-000/file-000.parquet`, `meta/info.json`, `meta/stats.json`,
  `meta/tasks.parquet`, `meta/episodes/chunk-000/file-000.parquet`.
- [x] Checked shapes, dtypes, ranges, and continuity, and plotted action and
  reward curves. `[verified]` — `scripts/figures/` holds nine action plots plus
  `reward_curve.png`; `scripts/reports/dataset_quality_summary.txt` present.

### 2.7 — PyTorch `Dataset` and `DataLoader`

- [x] Read a single sample through `dataset[index]`. `[verified]` — `dataset[0]`
  returns `observation.state (42,) float32`, `action (8,) float32`,
  `timestamp () float32`, `task 'pick up the cube'`.
- [x] Built batches through `DataLoader`. `[verified]` — the notebook batch cell
  prints `observation.state [8,42] float32`, `action [8,8] float32`,
  `timestamp [8]`, `frame_index [8] int64`, `episode_index [8]`, `index [8]`,
  `task_index [8]`, and `task` as a `list[str]`. Non-tensor fields come back as
  lists, not tensors.
- [x] Confirmed batch count: `len(train_loader) == 7` for 50 frames at
  `batch_size=8`. `[verified]`
- [x] Distinguished random sampling from temporal order. `[reported]` — the
  training loader uses `shuffle=True`; see the caveat in "BC Training Result".

### 2.8.1–2.8.5 — Reading, batch, model, loss, backpropagation

- [x] `observation.state [42]` and `action [8]` per sample. `[verified]`
- [x] `states [8,42]` and `actions [8,8]` per batch. `[verified]`
- [x] `MLPPolicy`. `[verified]` — `Linear(42,128) + ReLU → Linear(128,128) +
  ReLU → Linear(128,8) + Tanh`, with default bias enabled. The `Tanh` output
  layer bounds predictions to `[-1,1]`, which matches the documented action
  range.
- [x] MSE loss, checked against a manual per-sample and per-action-dimension
  computation. `[reported]`
- [x] Optimizer and one manual update step: `optimizer.zero_grad()`,
  `loss.backward()`, `optimizer.step()` with `Adam(lr=1e-3)`. `[reported]` —
  gradient, learning-rate, and weight-change inspection is recorded in the
  notebook.

### 2.8.6 — Full training loop

- [x] Batch loop inside an epoch loop, Adam optimizer, 100 epochs, loss curve.
  `[verified]` — independently reproduced in a standalone process with the same
  configuration (`batch_size=8`, `shuffle=True`, `Adam(lr=1e-3)`, 100 epochs,
  `torch.manual_seed(42)`).

**BC Training Result**

| Item | Value |
|---|---|
| Epochs | 100 |
| Batches per epoch | 7 |
| Optimizer | `Adam(lr=1e-3)` |
| Loss function | `nn.MSELoss()` |
| Loss at epoch 1 | `0.346221` |
| Loss at epoch 10 | `0.311921` |
| Loss at epoch 20 | `0.283204` |
| Loss at epoch 30 | `0.257555` |
| Loss at epoch 100 | `0.105965` |
| Untrained-model loss (baseline, measured separately) | `0.371752` |

All values above are `[verified]`: the epoch 1/10/20/30 trajectory and the final
value `0.105965` were reproduced in an independent run.

Interpretation, and the reason 2.8.7 is next:

- The loss fell from `0.346221` to `0.105965` and sits well below the untrained
  baseline `0.371752`, so the training pipeline genuinely optimizes. The
  optimizer, the backward pass, and the data path all work.
- The final loss is far from zero, and it was measured **on the same data the
  model was trained on**. There is no validation set, so the number cannot
  distinguish "learned the task" from "memorized 50 frames".
- The dataset is 1 episode and 50 frames. With `shuffle=True`, adjacent frames
  from the same trajectory are mixed across batches. For an MLP over single
  frames that is defensible, but it is exactly the construction that must not
  be reused once temporal windows `[B,T,D]` or multiple episodes appear.
- The actions in this trajectory are near-random, so even a perfectly trained
  model on this data would not be a usable policy. The value of 2.8.6 is
  proving the pipeline, not producing a controller.

## On-Disk Artifact Facts

`[verified]` by reading `datasets/lerobot/pickcube/meta/info.json`,
`conversion_manifest.json`, and the parquet payload:

- `codebase_version`: `v3.0`
- `total_episodes`: 1, `total_frames`: 50, `total_tasks`: 1, `fps`: 50
- `observation.state`: `float32`, shape `[42]`
- `action`: `float32`, shape `[8]`
- `robot_type`: `maniskill`; task string: `pick up the cube`
- Video features: **none**. `info.json` declares only `observation.state`,
  `action`, and the index/timestamp fields, and there is no `videos/` directory.
- The `task` field is not listed in `features`; the converter supplies it as a
  string through `add_frame()`.
- Source file hash recorded: `random_episode_standard.h5`,
  `sha256:34d5ed74fa5ee3a0793e52bc154b8dc1559abd50e2377682b717dc0b3acb781b`
- Frame-level facts read directly from
  `data/chunk-000/file-000.parquet`: `timestamp` runs `0.0` to `0.98` with a
  constant step of `0.02`; `frame_index` runs `0` to `49`; the schema stores
  `observation.state` as `fixed_size_list<float>[42]` and `action` as
  `fixed_size_list<float>[8]`.
- The HDF5 source (`datasets/pickcube/random_episode_standard.h5`) has
  `observations (50,42) float32`, `actions (50,8) float32`,
  `rewards (50,1) float32`, `timestamps (50,) float64`, a `metadata` group, and
  empty root attributes.

The dataset counts above are dataset-level and were read back successfully on
this machine. Full per-frame range, NaN/Inf, and boundary checks across all 50
frames are **not** complete; that is the job of the still-missing
`inspect_robot_dataset.py`.

## Action Specification

`[verified]` from `conversion_manifest.json`. This supersedes the earlier
statement that action semantics were undocumented:

| Component | Dims | Controller | Semantics |
|---|---:|---|---|
| Arm | 7 | `PDJointPosController` | normalized joint position target |
| Gripper | 1 | `PDJointPosMimicController` | continuous normalized position target |
| **Total** | **8** | | range `[-1.0, 1.0]` |

Note the naming gap between 2.2 and the manifest: the lesson describes the
current control mode as `pd_joint_delta_pos`, while the conversion manifest
records `PDJointPosController` with normalized joint position targets. These
describe related but distinct semantics (delta versus absolute target). This
must be resolved by reading the collector's actual controller configuration
before any action-semantics claim is reused for training or cross-embodiment
work. `[open]`

Still missing for a complete contract: control frequency actually used, the
coordinate frame of the arm targets, and whether the gripper convention is
open-high or open-low.

## Observation Decomposition

The current state observation has 42 dimensions:

| Component | Dimensions |
|---|---:|
| Robot `qpos` | 9 |
| Robot `qvel` | 9 |
| TCP pose | 7 |
| Object pose | 7 |
| Goal position | 3 |
| TCP-to-object relative position | 3 |
| Object-to-goal relative position | 3 |
| Grasp state | 1 |
| **Total** | **42** |

This decomposition is `[reported]`. The dataset contract still ships the raw
42-d vector, so per-field deployability (which entries are simulator privileged
state) is not yet encoded anywhere machine-readable. This remains an open item,
but it does not block 2.8.7 or 2.8.8.

## Open Issues

These are recorded for traceability. None of them blocks the current step.

### Temporal contract: 20 Hz control versus declared 50 FPS

`[verified]` root cause, measured directly on the running environment:

- `env.unwrapped.control_freq = 20` and `control_timestep = 0.05`, with
  `sim_freq = 100`. The simulator's real control period is 0.05 s / 20 Hz.
- The collector samples its timestamps with `np.arange(T) * 0.02`; they are
  synthetic, not measured, and record 0.02 s / 50 Hz.
- The converter derives FPS from those timestamps
  (`scripts/pipeline/converter.py:estimate_fps` computes
  `round(1.0 / mean_dt)`), so `info.json` inherits `"fps": 50` from the synthetic
  values. The chain is: collector stores no timing metadata → synthetic 0.02 s
  timestamps → FPS inferred from timestamps → declared 50 Hz over 20 Hz data.

Consequence: LeRobot converts `frame_index` to seconds using the declared `fps`,
so the time axis is compressed by 2.5×. Any future temporal window `[B,T,D]`
will not mean the number of seconds it claims.

Do not silently change `fps`. Either the source timestamps are repaired at
collection time (record `control_freq`, stop inferring), or the declared `fps`
is corrected and the decision is documented here with its reason. This is a
prerequisite for temporal-window work, not for 2.8.7.

### Missing reusable inspector

`scripts/inspect_robot_dataset.py` is a Lesson 2 deliverable and does not exist.
`[verified]` — absence reconfirmed this session. It is still wanted as the
reusable gate check that performs full-frame validation.

## Lesson 2 Acceptance Gate

| Check | Required evidence | Status |
|---|---|---|
| Source trajectory | Episode created and advanced through `o_t → a_t → o_{t+1}` | **PASS** `[reported]` |
| State and action space | 42-d state and 8-d action decomposed, action spec documented | **PARTIAL** — spec documented; per-field deployability not encoded; delta-vs-absolute naming gap open |
| Temporal pairing | Documented `(o_t, a_t)` rule and `T` versus `T+1` convention | **PASS** `[reported]` |
| Source storage | HDF5 episode grouping and array lengths checked | **PASS** `[verified]` |
| Conversion | ManiSkill trajectory converted to LeRobot format with `data/` and `meta/` | **PASS** `[verified]` |
| Data validation | Shape, dtype, NaN, range, continuity, action/reward curves | **PARTIAL** — counts, shapes, dtypes and curves verified; full-frame numerical checks pending |
| Dataset loading | `dataset[index]` and `DataLoader` batch | **PASS** `[verified]` |
| MLP policy | `42 → 128 → 128 → 8` forward pass, shape-asserted | **PASS** `[verified]` |
| Loss | MSE matching a manual computation | **PASS** `[reported]` |
| Backpropagation | `zero_grad → forward → loss → backward → step` updates weights | **PASS** `[reported]` |
| Training loop | 100 epochs, Adam, reproducible loss curve | **PASS `[verified]`** — `0.346221 → 0.105965`, baseline `0.371752` |
| Train/validation | Held-out split, generalization assessed | **FAIL — next step (2.8.7)** |
| Closed-loop execution | Policy drives `env.step` and task outcome known | **FAIL — not started (2.8.8)** |
| Reusable inspection | `scripts/inspect_robot_dataset.py` runs independently | **ABSENT** `[verified]` |

## Immediate Next Steps

Ordered. The next two steps finish Lesson 2; the third opens Lesson 3.

1. **2.8.7 — 训练集与验证集.** Run the train/validation experiment:
   - why training loss alone is insufficient;
   - train/validation split and what a validation curve adds;
   - generalization versus memorization, underfitting and overfitting;
   - why a small model on 50 frames shows a decaying training loss that still
     says nothing about the task;
   - why random-action data cannot produce a valid policy.
   With one episode, any frame-level split leaks: adjacent frames are nearly
   identical, so held-out frames are not held out in any meaningful sense. The
   experiment must state this limitation explicitly and treat it as the
   motivating example for episode-level splitting once more episodes exist.
2. **2.8.8 — 策略部署与闭环执行.** `env state → policy(state) → predicted
   action → env.step(action) → new state`; `model.train()` versus
   `model.eval()`; `torch.no_grad()`; single-step prediction versus closed-loop
   rollout; action clipping and simulation safety; deciding whether the policy
   actually completes the task.
3. **Lesson 3.1–3.4.** Move from a pipeline that runs to the theory of why a
   low training loss still fails at execution: imitation-learning problem
   definition, Behavior Cloning as supervised learning, generalization and
   overfitting, and distribution shift. This is the transition to
   `π_θ(a_t | o_t) ≈ π_E(a_t | o_t)`.

Deferred, not blocking: the temporal contract, the missing
`inspect_robot_dataset.py`, and the per-field observation decomposition.

## Session Log

### 2026-09-22 — Lesson 2 收尾阶段同步与 2.8.6 复现

- Re-read the notebook `notebooks/inspect_lerobot_dataset.ipynb` (40 cells) and
  confirmed that 2.7 through 2.8.6 are implemented there: `DataLoader` with
  `batch_size=8`, `MLPPolicy` with `Tanh` output, `nn.MSELoss()`, `Adam(lr=1e-3)`,
  and a 100-epoch training loop ending in a loss curve.
- Reproduced the training result independently in a standalone process rather
  than trusting the recorded output: `loss[0] = 0.346221`, `loss[-1] = 0.105965`,
  `loss[9] = 0.311921`, `loss[19] = 0.283204`, `loss[29] = 0.257555`,
  `batches/epoch = 7`. These match the notebook and the learner's reported
  values digit for digit.
- Measured an additional baseline that the notebook did not record: an untrained
  `MLPPolicy` scores `0.371752` on the dataset. This makes the training result
  interpretable — the learned model is better than a random one.
- Confirmed the gaps for 2.8.7 and 2.8.8: no train/validation split, no
  `random_split`, no `model.eval()` validation pass, no checkpoint, no loss curve
  or figure written outside the notebook, and no closed-loop rollout code.
- Confirmed the on-disk frame-level facts by reading the parquet payload
  directly: `timestamp` step is a constant `0.02`, `frame_index` spans `0`–`49`,
  and the HDF5 source carries no root attributes.
- Diagnosed the 20 Hz versus 50 Hz contradiction to its root cause by querying
  the live environment: `control_freq = 20`, `control_timestep = 0.05`. The
  declared `fps: 50` is inherited from synthetic `0.02 s` timestamps through
  `converter.py:estimate_fps`.
- Rewrote this file and the Lesson 2 section of `docs/roadmap_v3.md` to the
  lesson sequence the learner now uses (2.1–2.8.8), replacing the earlier
  2.1–2.9 breakdown and the stale claim that the environment was empty.
- No datasets were modified, no conversion was re-run, no scripts were changed,
  and no closed-loop rollout was attempted in this session. This update is
  documentation plus independent read-only reproduction.
