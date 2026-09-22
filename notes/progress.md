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

Lesson 2 is in its closing phase. The lesson is one continuous thread: **from a
simulation trajectory to a minimal, trainable behavior-cloning model**. It is
not a lesson about building more engineering environment, and it does not yet
evaluate a policy in closed loop.

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
| 2.8.7 | 训练集与验证集 | **Next — 下一步** |
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

`[verified]` by polling the live `PickCube-v1` controller configuration with
`control_mode="pd_joint_delta_pos"` and by stepping the environment with known
actions.

**The 8-d action is not one uniform semantic space.** It is two different
commands concatenated:

| Indices | Dims | Controller | Semantics | Physical range | Unit |
|---|---:|---|---|---|---|
| `action[0:7]` | 7 | `PDJointPosController` | normalized joint position **delta** relative to current `qpos` | `[-0.1, 0.1]` | rad |
| `action[7]` | 1 | `PDJointPosMimicController` | normalized **absolute** gripper joint position target | `[-0.01, 0.04]` | m |
| **Total** | **8** | | action space `Box(-1, 1, (8,), float32)` | | |

Controller configuration as read back from the live environment:

| | Arm | Gripper |
|---|---|---|
| `joint_names` | `panda_joint1` … `panda_joint7` | `panda_finger_joint1`, `panda_finger_joint2` |
| `lower` / `upper` | `-0.1` / `0.1` | `-0.01` / `0.04` |
| `use_delta` | `True` | `False` |
| `use_target` | `False` | `False` |
| `normalize_action` | `True` | `True` |
| `mimic` | — | `{panda_finger_joint2: {joint: panda_finger_joint1}}` |
| `stiffness` / `damping` | `1000.0` / `100.0` | `1000.0` / `100.0` |

Measured mappings:

- Arm: `a_i ∈ [-1,1] → Δq_i ∈ [-0.1,0.1] rad`, approximately
  `Δq_i = 0.1 · a_i`. Stepping with `a_arm = ±1` moved the drive target to
  exactly `qpos ± 0.1` rad on every arm joint. Because `use_target=False`, the
  delta is applied to the **actual current joint position**, not to the previous
  control target, so `q_target = q_current + Δq`.
- Arm, one caveat worth keeping: the **observed** `qpos` after a single step
  moves far less than the target delta (measured `0.01313 / 0.03491 / 0.02571`
  rad on joints 1–3 for `a_arm = +1`), because the PD controller tracks the new
  target over subsequent steps. Do not read the one-step `qpos` change as the
  commanded delta.
- Gripper: `a_g ∈ [-1,1]` maps linearly onto `[-0.01, 0.04]`:
  `q_target = -0.01 + (a_g + 1)/2 · 0.05`. Measured drive targets:
  `a_g = -1 → -0.01`, `a_g = 0 → +0.015`, `a_g = +1 → +0.04`. Both finger
  joints received the same target through the mimic relation, so one action
  value drives two active joints.
- Gripper direction: at reset `panda_finger_joint1` sits at `+0.04`. The
  measured mapping therefore makes `a_g = +1` the **maximum opening** and
  `a_g = -1` the closed extreme, and `a_g = 0` an intermediate half-open target
  rather than a neutral no-op.
- DOF accounting: the arm contributes 7 controlled DOF and the gripper 1, so
  `7 + 1 = 8` action dimensions, while the physically active joint count is
  `7 + 2 = 9` because the two finger joints are coupled by mimic.

`[verified]` This supersedes the earlier statement that action semantics were
undocumented, and it closes the previous delta-versus-absolute `[open]` item:
the 2.2 reading (arm uses delta control) was correct, and it is now extended by
the fact that the gripper channel is an absolute position target. The earlier
conversion manifest described all 8 dimensions uniformly as a normalized joint
position target, which was inaccurate for the arm channels; the manifest
generator and the on-disk manifest have been corrected. See
"Manifest provenance" below.

Still not established, and worth confirming before the closed-loop step: the
exact physical meaning of the gripper extremes for this assembly. The numeric
mapping above is measured; whether `0.04` corresponds to fingers fully apart is
an interpretation to verify by execution or by reading the MJCF, not by
assuming from the sign.

## Manifest Provenance

`[verified]` Two separate issues with
`datasets/lerobot/pickcube/conversion_manifest.json`:

1. **Corrected semantics.** The generated manifest described the whole 8-d
   action as a uniform "normalized joint position target". That was inaccurate
   for the arm channels, which are deltas. `scripts/pipeline/manifest.py` now
   emits the split arm/gripper structure documented above, and the on-disk
   manifest was patched to match, carrying an
   `action_semantics_correction` block that records what was changed, why, and
   that the dataset payload is untouched. A future regeneration will produce the
   same structure from code.
2. **Stale recorded path.** `[resolved 2026-09-22 by regeneration]` The manifest used to
   record `/home/bowenyuan95/Projects/embodied-ai-learning/...`, a path that does not
   exist on this machine. It was deliberately **not** hand-edited, because rewriting a
   provenance field by hand is exactly what the project forbids. Instead the pipeline
   was re-run and the generator wrote the current path. The recorded `sha256` matched
   the source file throughout, so payload lineage was never in question. The
   hand-written `action_semantics_correction` block is also gone: the generator now
   emits the correct semantics directly, so no correction note is needed.

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
| State and action space | 42-d state and 8-d action decomposed, action spec documented | **PASS** part 1 — both action channels fully specified and verified (arm delta `[-0.1,0.1]` rad, gripper absolute `[-0.01,0.04]`); per-field observation deployability still not encoded |
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

- Re-read the notebook now named
  `notebooks/2.7_2.8_bc_training_pipeline.ipynb` (40 cells at the time; renamed
  and markdown-organized on 2026-09-22 without re-executing it) and confirmed that
  2.7 through 2.8.6 are implemented there: `DataLoader` with `batch_size=8`,
  `MLPPolicy` with `Tanh` output, `nn.MSELoss()`, `Adam(lr=1e-3)`, and a
  100-epoch training loop ending in a loss curve.
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
- No dataset payload was modified, no conversion was re-run, and no closed-loop
  rollout was attempted in this session. This update is documentation plus
  independent read-only reproduction.

### 2026-09-22 — Action semantics verified and manifest corrected

- Resolved the delta-versus-absolute `[open]` item by reading the live
  `PickCube-v1` controller configuration rather than inferring from names. Both
  sub-controller configs match the learner's report field for field:
  arm `PDJointPosController` with `lower=-0.1`, `upper=0.1`, `use_delta=True`,
  `use_target=False`, `normalize_action=True`; gripper
  `PDJointPosMimicController` with `lower=-0.01`, `upper=0.04`,
  `use_delta=False`, `use_target=False`, `normalize_action=True`, and mimic
  `panda_finger_joint2 ← panda_finger_joint1`.
- Confirmed the channel split from `agent.controller.action_mapping`:
  `{'arm': [0,7], 'gripper': [7,8]}` over a single
  `Box(-1.0, 1.0, (8,), float32)` action space.
- Verified the arm mapping by stepping with known actions: `a_arm = ±1` moves the
  drive target to exactly `qpos ± 0.1` rad on all seven joints, and the delta is
  taken against the current `qpos` because `use_target=False`. Also recorded a
  trap: the observed `qpos` one step later moves only about
  `0.013`–`0.035` rad, because PD tracking is not instantaneous.
- Verified the gripper mapping by stepping with known actions: drive targets
  `-0.01`, `+0.015`, `+0.04` for `a_g = -1`, `0`, `+1`, both fingers equal. Since
  `panda_finger_joint1` starts at `+0.04`, `a_g = +1` is maximum opening and
  `a_g = 0` is an intermediate target rather than a neutral no-op.
- Fixed `scripts/pipeline/manifest.py`, which emitted one uniform
  `"normalized joint position target"` for all 8 dimensions. It now emits the
  split `arm` / `gripper` structure with `indices`, `use_delta`, `use_target`,
  `physical_range`, and `unit`.
- Patched the on-disk `datasets/lerobot/pickcube/conversion_manifest.json`
  `action_semantics` block to match, and added an `action_semantics_correction`
  block recording the reason, the verification method, and that the payload is
  unchanged. The stale recorded source path was deliberately left alone; see
  "Manifest Provenance". `[open]`

### 2026-09-22 — Notebook reorganization

- Renamed four notebooks so their names carry the lesson they belong to, using
  `git mv` so history is preserved:
  - `00_environment_check.ipynb` → `0_environment_check.ipynb`
  - `01_inspect_pickcube_dataset.ipynb` →
    `2.2_2.6_inspect_trajectory_and_observation_schema.ipynb`
  - `Generate_PickCube_LeRobot_Dataset.ipynb` →
    `2.5_generate_lerobot_dataset.ipynb`
  - `inspect_lerobot_dataset.ipynb` → `2.7_2.8_bc_training_pipeline.ipynb`
- Added markdown section cells so each notebook reads as a sequence instead of a
  wall of code: lesson scope, what each block does, and what its result does and
  does not prove.
- **No code cell was re-executed.** Verified against `HEAD` for all four
  notebooks: code sources, outputs, `execution_count` values, cell metadata, and
  kernelspec are byte-identical; only markdown cells were inserted. All four
  still pass `nbformat.validate`.
- Corrected the naming of two cells in
  `2.7_2.8_bc_training_pipeline.ipynb` while adding headers: what the notebook
  records as "2.8.4 时间窗口 `[B,T,D]`" is a loss computation, and temporal
  windows are not implemented anywhere in the repository. The section is now
  titled "MSE loss", and temporal windows remain future work.
- `2.8 Check data.ipynb` was left untouched, as requested. It is a scratch
  notebook in active use; it interleaves environment experiments with a search
  for ManiSkill's motion planners and is not part of the curated sequence.
- Updated `README.md`: repository tree, status line, Current Progress table,
  roadmap checkboxes, and the inspection-notebook command.
- No dataset, script, or training artifact was modified by this reorganization.
- No dataset payload was rewritten and no conversion was re-run.

### 2026-09-22 — Lesson 1 and Lesson 2 gap notebooks

- Built seven new notebooks to cover the roadmap gaps from Lesson 1 through the
  current position. Lesson 1 previously had **no** notebook at all, and the
  Lesson 2 sequence had holes at 2.1, 2.3, and 2.9.
- All notebook names were unified to `lesson_substep_topic.ipynb`, which required
  re-splitting the training notebook: `2.6_dataset_dataloader.ipynb` (dataset and
  batching) and `2.7_bc_training_loop.ipynb` (policy, loss, backprop, 100-epoch
  loop). The split was verified against `HEAD`: all 33 code cells keep byte-identical
  sources, outputs, and `execution_count` values.
- `2.8 Check data.ipynb` was renamed to `2.8_check_data.ipynb` to drop the space,
  and its content was not modified.
- Content was grounded in the archived Lesson 0/1 probes rather than rewritten:
  `smoke_test_maniskill.py` and `collect_pickcube_random.py` became
  `2.1_environment_and_rollout.ipynb`; `verify_state_flattening.py` and
  `build_deployment_safe_observation.py` informed `1.1_state_and_observation.ipynb`.
- Notebooks are committed **with executed outputs**. All twelve curated notebooks
  execute with zero errors; execute via `nbclient` from the repository root with a
  workspace-local `HF_HOME`.
- `scripts/build_lesson_notebooks.py` regenerates the Lesson 1–3 notebooks from
  source (without outputs).

New findings recorded in the notebooks:

- `get_proprioception()` and `get_state()` return **nested dicts**, not tensors.
- `Pose.to(device)` moves a pose between devices; it is **not** a frame change.
  Frame composition is 4x4 matrix algebra.
- `tcp_to_obj_pos` and `obj_to_goal_pos` are **world-frame position differences**
  (`obj_pos - tcp_pos`), not vectors rotated into the gripper frame. The rotated
  version is numerically different because the TCP rotation is not identity.
- The 42-d flat layout differs from `state_dict` insertion order; verified layout is
  `qpos(0:9) qvel(9:18) is_grasped(18:19) tcp_pose(19:26) goal_pos(26:29)
  obj_pose(29:36) tcp_to_obj_pos(36:39) obj_to_goal_pos(39:42)`.
- Same task, eleven control modes, action dimensions from 4 to 15, and
  `pd_joint_pos` uses physical joint limits rather than `[-1, 1]`.

### 2026-09-22 — Motion planner segfault: root cause corrected

> **Correction.** An earlier entry in this log attributed the planner segfault to the
> missing GPU backend. **That was wrong.** The crash reproduces on pure CPU and has
> nothing to do with CUDA or SAPIEN's render device. The real cause is a NumPy ABI
> mismatch, established below by isolation testing.

`[verified]` Constructing ManiSkill's motion planner terminated the process:

```text
Fatal Python error: Segmentation fault
  File ".../mplib/planner.py", line 65 in __init__
  File ".../base_motionplanner/motionplanner.py", line 59 in setup_planner
  File ".../panda/motionplanner.py", line 25 in __init__
```

Root cause, and the evidence for it:

- **`mplib` 0.1.1 is built against the NumPy 1.x C API.** Under NumPy 2.x its stored
  C-API function pointers are invalid, so the constructor performs an indirect call to
  address `0x0`. Kernel log confirms it: `segfault at 0 ip 0000000000000000`.
- **Decisive test.** The same code, same environment, only the NumPy version changed:
  `numpy 2.2.6` → SIGSEGV; `numpy 1.26.4` → `ArticulatedModel OK`, and the full
  official PickCube solution returns `success: True`.
- **Dependency evidence.** The `mplib` 0.2.0/0.2.1 wheels declare `Requires-Dist:
  numpy <2.0`; 0.1.1 declares only `numpy`, with no upper bound, so pip happily paired
  it with NumPy 2.
- **Ruled out by isolation**, each tested directly: GPU/CUDA, Python version (3.10 and
  3.12 both crash), link/joint arguments (empty lists crash), `move_group` value, SRDF
  contents, missing shared libraries (`ldd` reports no unresolved non-Python symbols),
  mesh parsing (a primitives-only URDF crashes), `convex=True/False`, conflicts with
  scipy/toppra/ManiSkill (`env -i` clean process still crashes), and glibc version
  (wheel is `manylinux_2_17`, host is glibc 2.39).
- **Upgrading `mplib` is not a way out.** ManiSkill 3.0.1 pins `mplib==0.1.1`, and
  `mplib` 0.2.x breaks the API ManiSkill uses: `set_base_pose` now demands a
  `mplib.pymp.Pose` instead of a 7-vector, and `Planner.get_joint_pos`,
  `get_link_pose`, `get_move_group_pick_indices`, and
  `get_move_group_joint_indices` are gone. The upgrade was tested in isolation and
  rejected on that basis.
- **Fix applied:** `numpy<2` in `embodied310`. Verified after the change: all of
  `torch`, `gymnasium`, `mani_skill`, `sapien`, `mplib`, `toppra`, `scipy`,
  `matplotlib`, `h5py`, and `cv2` import and work; the project data pipeline
  (`load_episode` + `validate_episode`) runs. `opencv-python 5.0.0.93` declares
  `numpy>=2`, but was verified to import and run under 1.26.4 — the conflict is in pip
  metadata only.
- **Environment split, now documented:** `embodied` (NumPy 2.2.6, has lerobot/pyarrow)
  runs notebooks and the data pipeline; `embodied310` (NumPy 1.26.4) runs the planner.
  A planner call inside an `embodied` kernel kills the kernel outright, which is why
  `2.9_expert_demonstrations.ipynb` delegates planning to a subprocess.

### 2026-09-22 — Expert demonstrations produced (2.9 unblocked)

`[verified]` `scripts/generate_expert_demo.py` drives ManiSkill's own
`panda/solutions/pick_cube.py` planner and records every transition by wrapping
`env.step`, so actions are what the environment actually executed rather than a
reconstruction.

| Seed | Frames | `success` | `is_obj_placed` | `is_robot_static` |
|---:|---:|:--:|:--:|:--:|
| 0 | 74 | True | True | True |
| 1 | 74 | True | True | True |
| 2 | 50 | True | True | True |
| 3 | 86 | True | True | True |
| 4 | 76 | True | True | True |

**5 of 5 episodes succeed.** Output: `datasets/pickcube/expert_episodes.h5`, with the
contract written into attributes (`control_mode`, `data_quality="expert_planner"`,
`control_freq_hz=20`, `timestamp_source="derived_not_measured"`, per-channel action
semantics).

Two findings that had to be discovered rather than assumed:

1. **Control mode must be `pd_joint_pos`.** The planner's `close_gripper()` /
   `open_gripper()` emit `[qpos(7), gripper]`, an 8-d **absolute** position action.
   Under `pd_joint_delta_pos` the helper produces a 15-d vector and the controller
   rejects it: `AssertionError: Received action of shape torch.Size([15]) but expected
   shape (1, 8)`.
2. **The official recipe does not open the gripper after the carry.** Adding
   `open_gripper()` plus zero-action settling steps makes the cube land beside the goal:
   `is_obj_placed: False`, `success: False`. The verified sequence is exactly
   reach → grasp → close → carry.

**Action smoothness separates the two data regimes** — the single clearest metric that
the random fixture can never train a policy:

| Dataset | mean `|Δa|` |
|---|---|
| `random_episode_standard.h5` | `0.67` |
| `expert_episodes.h5` | `0.0078` |

**Replay verification, with an honest tolerance.** Replaying each episode from its seed
reproduces the recorded **rewards exactly** (`max error 0.00e+00`) and the same
`success` (5/5). Continuous state agrees to `~1e-2`. The boolean `is_grasped` flag
differs on **1 frame out of 74**, at the grasp transition (recorded flips at frame 36,
replay at 37), because it is a contact-threshold test.

Consequence for the gate: the project's rule "replay must match the source" is
achievable as *reward-exact and success-identical* for planner episodes, but **not** as
bit-exact observations across a boolean contact threshold. The earlier claim that a
source trajectory matched "all 50 observations" held for the random fixture; expert data
needs the tolerance stated explicitly rather than a blanket "all observations match".

### 2026-09-22 — Expert data is not concatenable with the existing fixture

`[verified]` The repository now holds two trajectory kinds with **incompatible action
semantics**:

| | `random_episode_standard.h5` | `expert_episodes.h5` |
|---|---|---|
| Source | random actions | motion planner |
| Episodes / frames | 1 / 50 | 5 / 50–86 |
| Success | none (`success_any=False`) | 5/5 |
| Control mode | `pd_joint_delta_pos` | `pd_joint_pos` |
| Arm action | joint-position **delta**, `[-0.1, 0.1]` rad | **absolute** target in joint limits |
| Role | pipeline fixture | imitation-learning supervision |

Both are 8-dimensional and both sit in a box. That is exactly the trap
`1.2_action_space_and_control_modes.ipynb` documents: matching shapes are not matching
semantics. A converter (`Δq_t = q_target[t] - qpos[t]`, scaled by `0.1` and clamped)
is straightforward but **must be built and verified before the two datasets are mixed**.
It is deliberately not done in this session.

### 2026-09-22 — scripts/ reorganization

Goal: keep only the acquisition-and-processing path in `scripts/`, integrate the rest
into the relevant notebook, or archive it.

**Retained** (`scripts/`):

| File | Why it stays |
|---|---|
| `run_pipeline.py` + `pipeline/` | the only conversion path and the only quality gate |
| `observation_adapter.py` | the authoritative deployable-versus-privileged partition |
| `collect_pickcube_random_rollout.py` | produces the source rollout |
| `replay_pickcube_episode.py` | produces the acceptance-gate replay evidence |
| `build_lesson_notebooks.py` | one-off notebook generator |

**Moved into `notebooks/2.4_observation_schema.ipynb`** as executable cells, then
archived:

- `validate_maniskill_rollout.py` — schema, field-length consistency, shape, numeric
  health, episode semantics, `elapsed_steps` continuity;
- `compare_random_datasets.py` — variant comparison, converted into a dataset-lineage
  demonstration;
- `dataset_report.py` — its 13 shared functions already live in
  `scripts/pipeline/reporter.py`; only `main`/`load_h5`/`run_quality_checks` were
  unique orchestration.

New cells in 2.4 use **real executed outputs**, not fabricated ones. The section also
documents the three-file lineage `maniskill_random_rollout.h5` ->
`random_episode_000.h5` -> `random_episode_standard.h5` and proves that the raw and
intermediate files are **different trajectories** despite comparable shapes.

**Archived to `archive/offroadmap/`**: `test_mplib_panda.py`, `test_planner.py` —
mplib debug probes with hard-coded environment paths, written while diagnosing the
planner segfault. The diagnosis is recorded; the probes are not portable.

**Archived to `archive/lesson_2_superseded/`**: `validate_maniskill_rollout.py`,
`compare_random_datasets.py`, `dataset_report.py`, `test_observation_adapter.py`.
`archive/README.md` records each one's replacement.

**Reorganization decisions worth noting:**

- `observation_adapter.py` was moved out of `pipeline/` and up to `scripts/`. It is
  **not** used by `run_pipeline.py` at all; its only real consumer is the 1.1 lesson
  notebook. Keeping it inside `pipeline/` implied it was a conversion step, which it
  is not — it encodes which state fields a real robot could measure, which is a
  modelling decision.
- A later attempt to turn the adapter into a "test" would have duplicated its four
  descriptive lines, so it was archived as a probe instead. The single source of truth
  is `scripts/observation_adapter.py`, demonstrated in
  `notebooks/1.1_state_and_observation.ipynb`.
- `scripts/figures/` and `scripts/reports/` were **kept**: `pipeline/config.py`
  defines them as the pipeline's own output directories, so they are live, not
  leftovers from the retired script.

**Fixed while verifying:** `python scripts/run_pipeline.py` failed with
`ModuleNotFoundError: No module named 'scripts'`, because running a file puts the
script's directory on `sys.path`, not the working directory. This was pre-existing and
the README documented the failing invocation. `run_pipeline.py` now inserts the
repository root into `sys.path`, and the pipeline runs from the repository root and
from any other working directory. Verified end to end with `--overwrite`.

### 2026-09-22 — Manifest provenance issue resolved by regeneration

- Re-running the full pipeline regenerated
  `datasets/lerobot/pickcube/conversion_manifest.json`.
- The recorded source path is now correct
  (`/home/bowenyuan/Projects/embodied-ai-learning/...`), so the stale-path `[open]`
  item from the action-semantics session is **closed**.
- The regenerated manifest carries the **corrected** action semantics directly from
  `scripts/pipeline/manifest.py` — arm delta with `use_delta=True`/`use_target=False`
  and gripper absolute with range `[-0.01, 0.04]`. The hand-written
  `action_semantics_correction` block is gone because it is no longer needed: the
  generator no longer produces the wrong description. This confirms the earlier patch
  and the generator now agree.
- The dataset payload was regenerated from the same source hash, so the counts
  (1 episode, 50 frames, fps 50) are unchanged, and the timing defect
  (`timestamp_source: synthetic`, `nominal_fps: 50` over 20 Hz control) is still
  present and still open.

### 2026-09-22 — Notebook path resolution fixed to the project root

**Symptom.** `notebooks/datasets/` and `notebooks/.cache/` appeared inside the notebook
directory instead of the repository root. `notebooks/datasets/` held the entire
`maniskill_pickcube_smoke` dataset and `pickcube_smoke.h5` — **the only copy on disk**;
the root had no counterparts.

**Root cause.** `2.5_generate_lerobot_dataset.ipynb` built its paths from bare relative
literals:

```python
OUTPUT_ROOT = Path('datasets/lerobot/maniskill_pickcube_smoke').resolve()
RAW_H5 = Path('datasets/raw/pickcube_smoke.h5').resolve()
```

Relative to a `notebooks/` working directory those resolve to `notebooks/datasets/...`
and the notebook creates them silently — no error, just data in the wrong place.

Every other notebook that touched files used `Path.cwd()`, which **assumes** the working
directory is the repository root. That assumption holds under `nbclient` (working
directory `.`) and breaks as soon as Jupyter is started from `notebooks/`.

**Fix.** A single resolver, taken from the fallback already written by hand in
`2.6_dataset_dataloader.ipynb` and generalized to walk up until the project root is found:

```python
_cwd = Path.cwd().resolve()
PROJECT_ROOT = next(
    (p for p in (_cwd, *_cwd.parents) if (p / ".git").exists()),
    _cwd.parent if _cwd.name == "notebooks" else _cwd,
)
```

Applied inline to the affected cells of: `1.1`, `2.3`, `2.4`, `2.5`, `2.6`, `2.9`,
`3.1`. It was written into existing cells rather than added as new ones, so no cell was
inserted and execution order is unaffected. Verified: the resolver returns the repository
root whether the working directory is the root or `notebooks/`.

**Second defect found while fixing it.** `2.4`, `2.5`, and `2.6` did not set the Hugging
Face cache, so dataset reads fell back to `~/.cache/huggingface` and failed with
`PermissionError`. The notebooks were only working because the runner exported
`HF_HOME`/`HF_DATASETS_CACHE`. They now configure the cache themselves, at
`PROJECT_ROOT/.cache/hf`, and the ordering constraint is explicit in the code: the cache
must be set **before** lerobot is imported, because the `datasets` package snapshots its
cache location at import time. The first attempt in `2.6` set the environment after
`from lerobot... import LeRobotDataset` and still failed — that ordering bug is now
commented in the notebook.

**Cleanup and regeneration.** `notebooks/datasets/` and `notebooks/.cache/` were deleted,
and `2.5` was re-executed to regenerate its outputs at the correct location:
`datasets/raw/pickcube_smoke.h5` and `datasets/lerobot/maniskill_pickcube_smoke/`. The two
`meta/*.json` files that had been force-added to Git under `notebooks/` are removed; the
root-level counterparts are correctly covered by the existing `datasets/*` ignore rule.

**Verification.** All notebooks that touch data were re-executed in a **stripped
environment** (`env -i`, no `HF_HOME`, no `HF_DATASETS_CACHE`) and complete with zero
errors, confirming they are self-sufficient:

`1.1`, `2.3`, `2.4`, `2.5`, `2.6`, `2.9`, `3.1` — all pass. No bare `Path.cwd()` call
remains outside the resolver itself. `notebooks/` contains only `.ipynb` files.

Not changed: `2.7` (it has no data-loading cell — it consumes `dataset` / `dataloader`
defined by `2.6` in the same kernel) and `2.8` (no file writes; its absolute
`site-packages` paths are a separate, already-documented trap).
