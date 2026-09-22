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

The table above is the roadmap's 2.1–2.8.8 sequence. One further sub-step exists
outside that sequence:

| Sub-step | Topic | Status |
|---|---|---|
| 2.9 | 专家演示：ManiSkill motion planner 生成成功 episode | **Complete `[verified]`** — 5/5 成功，360 actions / 365 observations |

**2.9 — 专家演示（已完成的数据资产）.** `notebooks/2.9_expert_demonstrations.ipynb`
plus `scripts/generate_expert_demo.py` drive ManiSkill's own sampling-based motion
planner and record five successful `PickCube-v1` episodes into
`datasets/pickcube/expert_episodes.h5` (`T = 74 / 74 / 50 / 86 / 76` actions,
`T + 1` observations each, 5/5 `success`). This is the first non-random
supervision in the repository, so "only 1 episode, 50 frames, near-random
actions" is no longer the whole data story. The file now uses the canonical
`T + 1` / `T` transition schema (`transition_schema_version=2`), after the
collector's post-action observation offset and failure-episode handling were
fixed in this session. One contract difference remains: the expert episodes are
`pd_joint_pos` **absolute** joint targets and cannot be mixed with the
`pd_joint_delta_pos` fixture until the semantics converter exists. See "2.9 —
Expert demonstrations" and the open issues.

**Current stopping point: 2.8.6 完成，2.9 专家数据已按正确 schema 重新生成；
2.8.7「训练集与验证集」为下一步，其按 episode 划分的前提是先完成专家动作的
`pd_joint_pos` → `pd_joint_delta_pos` 语义转换。**

Estimated Lesson 2 completion: **about 90%**, based on 2.1–2.8.6 completing, 2.9
supplying expert data, and only 2.8.7 and 2.8.8 remaining. The 2.9 data does not
raise the estimate on its own: it adds a prerequisite (the action-semantics
converter) that the split depends on.

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

2.9 changed the data side of that experiment: an honest split is now possible at
the episode level (5 expert episodes, canonical `T+1`/`T` schema), but only after
the expert actions are converted into the fixture's semantics. Running 2.8.7 on
the random fixture alone would still only be able to demonstrate that a
frame-level split leaks — see the open
item on the 3.1 leakage measurement.

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

### 2.9 — Expert demonstrations

- [x] Produced planner-driven successful episodes. `[verified]` —
  `datasets/pickcube/expert_episodes.h5` (schema v2, regenerated 2026-09-22)
  read back with `h5py`: 5 episodes (`episode_000000` … `episode_000004`),
  actions `T = 74 / 74 / 50 / 86 / 76` (360 total), observations `T + 1` each
  (365 total), and `success`, `is_obj_placed`, `is_robot_static` all `True` in
  every episode. Root attributes: `control_mode=pd_joint_pos`,
  `control_freq_hz=20`, `data_quality=expert_planner`,
  `timestamp_source=derived_not_measured`, `transition_schema_version=2`,
  `failed_seeds='[]'`, `planner=mani_skill.examples.motionplanning.panda.solutions.pick_cube`.
- [x] Recorded executed transitions in the canonical schema rather than
  reconstructing them. `[verified]` — `scripts/generate_expert_demo.py` wraps
  **both** `env.step` and `env.reset` for the duration of the planner run, so
  `observations[T+1] = o_0 … o_T` (with `o_0` the reset observation) and
  `actions[T] = a_0 … a_{T-1}` are the actions the environment actually executed.
  The collector now also asserts `control_mode == "pd_joint_pos"` and an 8-d
  action space before recording, and writes only successful episodes.
- [x] Replay verification, re-measured after the fix. `[verified]` — for all five
  episodes: action replay from `reset(seed)` reproduces the stored observations
  **exactly** and in index order (`O[t]` matches the pre-step state, `O[t+1]` the
  post-step state, `max error 0.0`), rewards match exactly (`max error 0.0`), and
  `O[0]` equals the `env.reset(seed)` observation (`max |diff| = 0.0`).
- [x] Planner determinism. `[verified]` — regenerating seeds 0–4 in a fresh
  process produced `actions`, `rewards` and `observations` **bit-identical** to
  the previous run (`np.array_equal`). This supersedes the earlier
  "replay agrees to `~1e-2` with a one-frame `is_grasped` flip" claim, which is
  not reproducible and is explained by the old file's one-frame schema offset.
- [x] Action smoothness separates the two data regimes. `[verified]` — recomputed
  from the payloads: random fixture `mean |Δa| = 0.6723`; expert per-episode
  `0.0078 / 0.0074 / 0.0080 / 0.0077 / 0.0077` (mean `0.0077`).
- [x] Notebook exists, is corrected, and is executed. `[verified]` —
  `notebooks/2.9_expert_demonstrations.ipynb`, 23 cells (14 markdown / 9 code),
  all code cells executed with zero error outputs, kernel `embodied`. Section
  `2.9.9` adds a self-contained contract check: it asserts the `T+1`/`T` schema
  for all five episodes and replays `episode_000000`, reporting
  `initial observation error 0.0`, `max post-step error 0.0`, `max reward error
  0.0`, `replay success True`.

- [x] Re-expressed the expert actions in the fixture's semantics. `[verified]` —
  `scripts/convert_expert_actions_to_delta.py` writes
  `datasets/pickcube/expert_episodes_delta.h5` with
  `a_arm[t] = clip((q_target[t] − qpos_t) / 0.1, −1, 1)`, the gripper channel
  unchanged, observations/rewards copied, and the conversion provenance (source
  hash `5154c16a…`, formula, scale) in the root attributes. 1 of 2520 arm
  channels is clipped (`max |Δq| = 0.1008 rad`, on `episode_000004`); everywhere
  else the reconstruction error is `≤ 4e-9 rad`.
- [x] Delta-semantics replay reproduces the task. `[verified]` — replaying the
  converted actions in `pd_joint_delta_pos` from the recorded seeds gives an
  observation error `≤ 1e-4` (one episode `5.3e-3`, the clipped one), reward
  error `≤ 1.7e-4`, and **5/5 `success`**. The two control modes share PD gains
  (`stiffness 1000`, `damping 100`) and the same gripper target mapping, and
  differ only in the arm's `use_delta` flag, so expressing the expert target as a
  delta reproduces the same commanded targets and therefore the same trajectory.

What 2.9 does **not** establish:

- the delta-semantics file is closed-loop faithful **on replay from the recorded
  seeds**; it has not been converted to LeRobot, split, or trained on;
- the original `expert_episodes.h5` remains `pd_joint_pos` absolute targets and
  must not be concatenated with the delta fixture directly — the retargeted
  `expert_episodes_delta.h5` is the file to use.

See "Expert and fixture actions have incompatible semantics".


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

These are recorded for traceability. No open item blocks the next step (2.8.7)
any more; the action-semantics item below is closed by the converter and is kept
as the record of why the conversion exists.

### Expert and fixture actions have incompatible semantics `[resolved 2026-09-22 by conversion]`

`[verified]` the repository held two trajectory kinds that are 8-dimensional and
both live in a box, but do not mean the same thing:

| | `random_episode_standard.h5` | `expert_episodes.h5` (schema v2) | `expert_episodes_delta.h5` |
|---|---|---|---|
| Source | random actions | motion planner | converter from the column on the left |
| Episodes | 1 | 5 | 5 |
| Actions `T` | 50 | 74 / 74 / 50 / 86 / 76 | same |
| Observations | 50 (`o_t`, pre-action) | `T + 1` (`o_0 … o_T`) | `T + 1` |
| Success | none | 5/5 | 5/5 on delta-mode replay |
| Control mode | `pd_joint_delta_pos` | `pd_joint_pos` | `pd_joint_delta_pos` |
| Arm action | joint-position **delta**, `[-0.1, 0.1]` rad | **absolute** target in joint limits | normalized delta, `a = clip((q_target − qpos)/0.1, −1, 1)` |
| Role | pipeline fixture | planner ground truth | training supervision in the fixture's semantics |

`scripts/convert_expert_actions_to_delta.py` performs the re-targeting; it copies
observations and rewards and records the formula, the delta scale, the source
hash, and the clipped-value count. Replaying the converted actions in
`pd_joint_delta_pos` reproduces the expert trajectory and 5/5 `success`, so the
two datasets are now semantically compatible.

The remaining caveat is about **data quality, not semantics**: the fixture is a
single random-action episode, so concatenating it with expert episodes is now
*possible* but still not meaningful supervision. Mixing waits until there is a
second expert or otherwise useful trajectory.

### Expert HDF5 stored post-action observations `[resolved 2026-09-22]`

Resolved by fixing the collector and regenerating the file. The historical
analysis below is kept because the failure mode is easy to repeat.

**Resolution.** `scripts/generate_expert_demo.py` was rewritten so that
`StepRecorder` wraps `env.step` **and** `env.reset`; it now stores
`observations[T+1] = o_0 … o_T`, `actions[T]`, `rewards[T]`, asserts the control
mode and 8-d action space before recording, writes only successful episodes, and
records `transition_schema` / `transition_schema_version=2` / `failed_seeds` in
the root attributes. `expert_episodes.h5` was regenerated (seeds 0–4, 5/5
success) and the time line was replayed: `O[0]` equals the `env.reset(seed)`
observation exactly, every `O[t]` matches the pre-step state and every `O[t+1]`
the post-step state (`max error 0.0`), and the regeneration reproduced the
previous run's actions, rewards and observations bit-for-bit. The historical
defect, its evidence, and the canonical schema are recorded below.

`[verified]` at source level and by replay. The challenge was raised against the
original claim and **the challenge holds**: the collector's transition schema was
incomplete.

Source confirmation. `StepRecorder._recording_step` in the **old** collector was:

```python
def _recording_step(self, action):
    observation, reward, terminated, truncated, info = self._original_step(action)
    self.records["observations"].append(observation)
    self.records["actions"].append(action)
    self.records["rewards"].append(reward)
    return observation, reward, terminated, truncated, info
```

So it executed `env.step(a_t)` first and stored the **returned** observation:

```text
observations[t] = o_{t+1}
actions[t]      = a_t
rewards[t]      = r_t
```

In addition, ManiSkill's solution calls `env.reset(seed=seed)` inside
`solve()` (`mani_skill/examples/motionplanning/panda/solutions/pick_cube.py:9–10`),
but the old `StepRecorder` wrapped only `env.step`, never `env.reset`. The reset
observation `o_0` was therefore never captured, so the original statement "the
`(o_t, a_t)` pairing holds by construction" was wrong.

Replay evidence against the old file (all five episodes): replaying each stored
action array from its recorded seed under `pd_joint_pos` reproduced the stored
observations and rewards exactly (`max |O − post| = 0.0`, `max |R − rew| = 0.0`);
the pre-step comparison did not match (`mean 0.0687`, `max 1.0662`). Every
`actions` array was `(T, 8) float32` and lay inside the `pd_joint_pos` action
space. The offset was real, not a rounding artifact, and it was numerically small
only because expert frames are close together (`mean consecutive-state distance
≈ 0.069`) — which is exactly why it survived a shape check and a loss curve.

What the old file lost per episode: with `T` actions and `T` post-action
observations, the stored arrays alone gave `T − 1` correctly aligned BC samples,

```python
bc_obs     = observations[:-1]   # o_1 ... o_{T-1}
bc_actions = actions[1:]         # a_1 ... a_{T-1}
```

e.g. 73 pairs for the 74-frame episodes, with `(o_0, a_0)` missing and the
terminal `o_T` unusable. The canonical schema the fixed collector now writes is
`T + 1` observations and `T` actions:

```python
observations = [reset_obs]                     # o_0
for action in actions:
    next_obs, reward, terminated, truncated, info = env.step(action)
    actions.append(action); rewards.append(reward)
    observations.append(next_obs)              # o_1 ... o_T

# training view
policy_obs        = observations[:-1]          # o_0 ... o_{T-1}
policy_actions    = actions                    # a_0 ... a_{T-1}
next_observations = observations[1:]           # o_1 ... o_T
```

Do not silently reinterpret stored arrays that use the older form.

### Collector also writes failed episodes into the expert file `[resolved 2026-09-22]`

Resolved by the same collector rewrite: a failed seed is now reported to stderr
and `continue`d before it is appended, so only successful episodes reach the
write loop, and the excluded seeds are recorded in the root attribute
`failed_seeds` (currently `[]`).

Historical defect (`[verified]` by reading the old
`scripts/generate_expert_demo.py:115–150, 183–191`): every episode was appended to
`episodes` regardless of `success`, and the write loop wrote all of them. The only
guard was "at least one episode succeeded". A failed episode was printed to
stderr, labelled `success=False`, and still stored alongside the expert ones, so
any consumer had to filter on the per-episode `success` attribute. In the current
file all five episodes carry `success=True`.

### 2.9 documentation corrections `[resolved 2026-09-22]`

Resolved: `scripts/generate_expert_demo.py` was rewritten and
`notebooks/2.9_expert_demonstrations.ipynb` was corrected (sections 2.9.2, "How
the planner executes its actions", 2.9.6, 2.9.7, 2.9.8, takeaways) and
re-executed end to end with zero errors, kernel `embodied`. The three issues are
kept below as the record of what was wrong.

The three issues were (`[verified]` by reading the installed planner source and
replaying the data):

1. `notebooks/2.9_expert_demonstrations.ipynb` (section 2.9.6) and
   `scripts/generate_expert_demo.py:23–24` stated that the `(o_t, a_t)` pairing
   "holds by construction". It did not — see "Expert HDF5 stored post-action
   observations".
2. The notebook's section "Why the planner bypasses the action space" stated that
   `move_to_pose_with_screw` "bypasses `env.step(action)` for the arm motion".
   It did not: `TwoFingerGripperMotionPlanningSolver.follow_path`
   (`mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py:43–57`)
   calls `self.env.step(np.hstack([qpos, self.gripper_state]))` at line 52 with a
   7-d absolute arm target plus the gripper value, and replaying only the stored
   actions reproduces the full arm motion and cube displacement exactly. The
   notebook's own section 2.9.7 statement is the correct one: `pd_joint_pos` is
   required because the helpers emit absolute `[qpos(7), gripper]` actions.
3. The "15-d action" shown in the notebook is a **helper-branch artifact, not a
   data format**. The two executor methods guard on different control modes:

   | Method | 8-d branch | 15-d branch | 15-d layout |
   |---|---|---|---|
   | `follow_path` | everything except `pd_joint_pos_vel` | `pd_joint_pos_vel` | `[qpos(7), qvel(7), gripper]` (real velocities) |
   | `open_gripper` / `close_gripper` | `pd_joint_pos` only | every other mode | `[qpos(7), qpos*0(7), gripper]` (**zeros**, velocity slot placeholder) |

   So under `pd_joint_delta_pos` the arm phase of `follow_path` still emits 8-d
   actions and passes the shape check while carrying absolute joint targets that
   the delta controller silently reinterprets, and the failure only surfaces at
   the first gripper call, where the 15-d vector is rejected. The 15-d vector is
   therefore not the `pd_joint_pos_vel` action either: its middle block is zeros,
   not `qvel`. `42-d observation + 8-d action` was never wrong; the dimension
   error is a control-mode contract violation.

All three were documentation issues, now corrected in the script docstring and in
the notebook cells. `[verified]` `generate_expert_demo.py` was **never** affected
by the 15-d path: it sets `CONTROL_MODE = "pd_joint_pos"`, which puts both methods
on their 8-d branch, and the rewritten collector now asserts that contract before
recording. Replaying all five episodes from their recorded seeds reproduces the
stored observations and rewards exactly, every `actions` array is `(T, 8) float32`
and lies inside the `pd_joint_pos` action space, with no shape exception on any
step.


### 3.1 leakage measurement is fixture-dependent

`[verified]` `notebooks/3.1_imitation_learning_intro.ipynb` measures the distance
between consecutive frames against the distance between random frame pairs, and
concludes that a frame-level split leaks. On the **random fixture currently in
use** the measurement does not support that conclusion:

| Data | consecutive mean | random-pair mean | ratio |
|---|---:|---:|---:|
| `random_episode_standard.h5` (50 frames) | `1.3500` | `1.3999` | `1.04×` |
| expert `episode_000000` | `0.0696` | `0.9017` | `12.9×` |
| expert `episode_000002` | `0.0691` | `0.6071` | `8.8×` |
| expert episodes 1/3/4 | `0.0668 / 0.0658 / 0.0690` | `0.8723 / 0.9012 / 0.9116` | `13.1× / 13.7× / 13.2×` |

The reason is itself the lesson's point: the random fixture takes near-random
±1 action steps, so its adjacent frames are almost as far apart as unrelated
frames, whereas a planner moves the robot a few millimetres per step. The
conclusion is correct for smooth expert trajectories and is *unsupported by the
data the notebook currently loads*. Either the 3.1 experiment should be re-run on
converted expert episodes, or the notebook should state that the effect is not
visible on the random fixture. Do not promote the claim without that change.

### Notebook numbering: `2.9` versus a roadmap sequence ending at `2.8.8`

`[verified]` `docs/roadmap_v3.md` (and this file's sub-step table) number Lesson 2
as `2.1`–`2.8.8`, with no `2.9`, while `notebooks/2.9_expert_demonstrations.ipynb`
exists and README refers to "Expert demonstrations (2.9)". The mismatch is a
documentation defect, not a data defect. Renaming the notebook or adding a 2.9
slot to the roadmap is a decision for the learner; it is recorded here instead of
being changed silently.

### Shuffling policy for the inspection and training loaders `[resolved 2026-09-22]`

**Resolved:** 2.6 的有序检查 loader 使用 `shuffle=False`；BC 训练 loader 使用
`shuffle=True`；validation loader 使用 `shuffle=False`。

- `2.6` now names its loader `inspection_loader` and keeps `shuffle=False`, with
  prose stating that it exists for reproducible inspection and preserves
  trajectory order. Re-executed: `frame_index` comes back `[0..7]` and
  `timestamp` `0.00 … 0.14`, so the batching is demonstrably ordered.
- `2.7` already builds its own `train_loader` with `shuffle=True`; that stays, and
  the notebook was not re-run for this change because the recorded 100-epoch
  result is unaffected by the rename.
- The validation loader (`shuffle=False`) belongs to 2.8.7, which has not
  started. It will split **by episode**, not by frame.

`[verified]` the original defect, kept as the record: 2.6's markdown said
"注意 `shuffle=True`…" while its `DataLoader` cell was constructed with
`shuffle=False`; the `shuffle=True` case actually belongs to the training loader
in `2.7` (cell 25). The lesson prose therefore described a setting the notebook
did not run. The translation carried the original wording faithfully, and the
mismatch was recorded rather than silently rewritten.

Settling this does **not** require re-running the full training pipeline on the
old random smoke dataset: the rule is about loader construction, and the fixed
five expert episodes are the data that matters for 2.8.7.

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
| Source trajectory | Episode created and advanced through `o_t → a_t → o_{t+1}` | **PASS** `[verified]` — planner episodes recorded through a wrapped `env.step` and replayed to identical rewards and 5/5 `success` |
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
| Expert demonstrations | Planner-driven episodes with known task outcome | **PASS `[verified]`** — 5/5 `success`, `T = 360` actions / 365 observations, canonical `T+1`/`T` schema; smoothness `0.0077` vs random `0.6723` |
| Train/validation | Held-out split, generalization assessed | **FAIL — next step (2.8.7); prerequisites are now met (`expert_episodes_delta.h5`), the split experiment itself is not run** |
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
   A frame-level split on a single episode cannot be an honest held-out set. Note
   the correction verified this session: on the **random fixture** the
   "adjacent frames are nearly identical" premise does not hold (measured ratio
   `1.04×`), while on the **expert episodes** it does (`8.8×`–`13.7×`). See
   "3.1 leakage measurement is fixture-dependent".
   The prerequisites are now in place: `datasets/pickcube/expert_episodes_delta.h5`
   holds the five expert episodes in the fixture's delta semantics, in the
   canonical `T+1`/`T` schema, and its replay reproduces the expert trajectory and
   5/5 `success`. The remaining work is the experiment itself: build the
   train/validation view over the converted episodes, split **by episode**, and
   report the validation curve and its limits (5 episodes is still small).
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
   `notebooks/3.1_imitation_learning_intro.ipynb` already exists and executes
   (3.1 problem definition, BC as supervised learning, a leakage measurement, and
   a distribution-shift probe), but Lesson 3 formally starts after 2.8.7/2.8.8,
   and the notebook's leakage cell needs the fixture correction noted above.

Deferred, not blocking: the temporal contract, the missing
`inspect_robot_dataset.py`, the per-field observation decomposition, and the
`2.9` notebook-numbering mismatch.

## Session Log

### 2026-09-22 — Acyclicity correction: DAG is not the same as a dependency graph

The learner corrected the scope of the rejected design: **`DAG` here means
directed acyclic**, and it is not interchangeable with `dependency graph`, which
is a more general relation that may contain cycles. That distinction matters,
because my first pass had replaced the wording with "partial order", which carries
the same acyclicity assumption under another name — the same failed premise.

Why the acyclic assumption is wrong for this project: real execution returns to
already-completed steps. Retry after a failure, recovery, backtracking, and
iteration are all re-entry, so "what may run next" is a function of the current
task state rather than of a fixed one-directional order that is computed once.

Changes:

- `notes/concepts.md`, `Task Representation`: added the explicit requirement that
  the representation must tolerate **cycles and re-entry**; the previous
  "partial-order" phrasing was replaced with "prerequisite/task-state
  representation", and prerequisite relations are now described as relations over
  steps rather than a fixed ordering.
- `notes/concepts.md`, `Study Priority Stack` (P1 row): "partial order" replaced
  with "prerequisite relations that admit cycles and re-entry".
- `docs/roadmap_v3.md`: the two explicit acyclic claims were replaced —
  line 24 `dependency DAG` → "task state（prerequisite 关系，允许 retry、回退与重入）",
  and the Lesson 14 theory bullet `sequential plan、partial order 与 dependency DAG`
  → "sequential plan、prerequisite 关系与 task state（需支持 cycle、retry 与回退）".
- `README.md` was **not** changed: it contains no acyclic claim. The wording
  "dependency graphs", "task graph", and "dependency-aware guidance" was kept
  deliberately, because a dependency graph is a general structure and the learner's
  earlier work is described in those terms.
- No `DAG`, `有向无环`, `acyclic`, `topological`, or `partial order` wording remains
  in the repository documentation. (`### 3.5 DAgger` is the imitation-learning
  algorithm, not a graph.)

Evidence: `grep -rni "dag\|有向无环\|acyclic\|partial order\|偏序" --include="*.md"`
returns only the DAgger heading and these session-log references.

### 2026-09-22 — `concepts.md` extended with the research-direction synthesis

The learner supplied a 2018–2025 embodied-agent landscape review plus a gap
analysis against their current progress, and asked for the research-facing part
to be merged into `notes/concepts.md`. The dependency representation they had
previously sketched was rejected as a failed design; the follow-up entry below
records the exact scope of that rejection (the directed **acyclic** form, not
dependency relations in general).

Added to `notes/concepts.md` (English, matching the file's convention):

- `Research Direction: Task Intelligence for Embodied Agents` — the gap a reactive
  VLA leaves (why / where in the task / what remains / failure / human need), the
  layered target architecture, the positioning statement, a reading map of which
  system to study per layer, and scope discipline (ROS 2, SLAM, low-level control,
  large-scale RL, pixel world models are substrate, not study goals).
- `Action Policy Families` — BC, action chunking, diffusion, flow matching; why MSE
  regression averages multimodal actions; why offline loss is not the metric.
- `VLA Anatomy` — the token flow through a VLM to an action head, discrete action
  tokens versus continuous generation, and action representation as a first-class
  problem (tied to this repository's own absolute-versus-delta evidence).
- `Task World Model` — task-level `p(z_{t+1} | z_t, a_t)`, counterfactual ranking,
  and an explicit instruction not to start from video generation.
- `Planning, Recovery, and Intervention` — the `act / guide / ask / wait / recover /
  escalate` vocabulary, failure localization, and reliability-centred metrics.
- `Sim-to-Real: Randomization Versus Adaptation` — randomization versus online
  latent adaptation, bounded residual adaptation plus a safety layer, and
  post-deployment drift.
- `Study Priority Stack` — the P0/P1/P2 gap table and the immediate order, anchored
  to finishing the closed loop first.

Also edited in the same file: the `Task State and Dependency Graph` section became
`Task State` with the graph framing removed; `dependency-aware`/`dependency graph`
wording was replaced throughout; `Embodied Memory` gained the procedural-plus-
episodic emphasis and a forgetting note; `Current Strategic Decision` now names the
task-intelligence layer as the target and lists the out-of-scope substrate.

Left alone at the time: the `dependency DAG` / `dependency graph` /
`dependency-aware` wording in `README.md` and `docs/roadmap_v3.md`. The follow-up
entry below resolves it — the explicit acyclic (`DAG`) claims were replaced, and the
generic dependency wording was kept.

### 2026-09-22 — `2.8.6` planner construction verified by an actual run (parallel work)

`notebooks/2.8_check_data.ipynb` gained three executed cells while this session
was running (54 → 55 cells, `execution_count` 35/36/37). They were not written by
this session and were left untouched; what they prove is recorded here.

- Import: `from mani_skill.examples.motionplanning.panda.motionplanner import
  PandaArmMotionPlanningSolver` → `PandaArmMotionPlanningSolver imported!`
- Environment assertion, printed by the cell itself: Python executable
  `/home/bowenyuan/miniforge3/envs/embodied310/bin/python`, Python `3.10.21`,
  NumPy `1.26.4`, with
  `assert "embodied310" in sys.executable` guarding the claim.
- Construction: `PandaArmMotionPlanningSolver(real_env, debug=False, vis=False,
  base_pose=robot_base_pose)` → `Planner created!`, initialization time
  `0.0046 s`, no segfault.

`[verified]` This is an independent, in-notebook confirmation of the "Motion
planner segfault: root cause corrected" entry: planner construction needs the
NumPy 1.x environment (`embodied310`) and works there. The same requirement was
already exercised twice today by `scripts/generate_expert_demo.py` runs, which
constructed the planner five times per run and produced 5/5 successful episodes
each time.

Per the learner's instruction, this is **verified and must not be re-run** merely
to re-prove it: the two independent lines of evidence above are sufficient, and
repeating a planner run costs minutes for no new information.

### 2026-09-22 — `2.9.9` transition-contract section found in the working tree; prose translated

`notebooks/2.9_expert_demonstrations.ipynb` gained three cells while this session
was running (13 markdown / 7 code → 14 markdown / 9 code): a new section
`2.9.9 — Validate the expert transition contract` with two executed code cells.
The cells were not written by this session and were left functionally untouched.

- The new section checks, in the notebook itself, that every episode stores
  `T + 1` observations for `T` actions, and then replays `episode_000000`:
  recorded output is `initial observation error 0.0`, `max post-step error 0.0`,
  `max reward error 0.0`, `replay success True`, with the pairing printed as
  `observations[t] --actions[t]--> observations[t+1]`.
- This independently reproduces the schema and alignment findings recorded above
  for the regenerated file, so the two lines of evidence agree.
- Only the new markdown cell was edited: it was English, and the project rule is
  that notebook prose is Chinese with English terms. Its code cells, outputs, and
  `execution_count` values are unchanged.
- The 2.9 evidence bullet was updated from "7 code cells" to the current
  14/9 split.

### 2026-09-22 — Loader shuffling rule settled; 2.6 renamed to `inspection_loader`

Follow-up on the `shuffle` mismatch found during the translation review. The
learner fixed the policy, so the open item is resolved and the notebook was
relabelled to match its actual role.

- `notebooks/2.6_dataset_dataloader.ipynb`: the loader is now
  `inspection_loader` with `shuffle=False`, and the markdown states explicitly
  that it exists for reproducible inspection and preserves trajectory order, that
  shuffling is a training-time choice, and that frame-level shuffling must not be
  reused once temporal windows or multiple episodes exist. Re-executed: all 10
  code cells run with zero errors, and the printed batch shows
  `frame_index [0..7]` with `timestamp 0.00 … 0.14`.
- `notebooks/2.7_bc_training_loop.ipynb` already defines its own
  `train_loader(..., shuffle=True)`, so it needed no change and was not
  re-executed; the recorded 100-epoch loss curve is unaffected by a rename in a
  different notebook.
- Rule recorded: inspection loader `shuffle=False`, BC training loader
  `shuffle=True`, validation loader `shuffle=False`, and any split is **by
  episode**. 2.8.7 has not started, so its validation loader is planned, not
  implemented.
- No training was re-run on the old random smoke dataset, by the learner's
  instruction: the fix is about loader construction, and the five retargeted
  expert episodes are the data that matters for 2.8.7.

### 2026-09-22 — Notebook generator deleted

`scripts/build_lesson_notebooks.py` was deleted at the learner's instruction
(`git rm -f`), not archived. Reason: it was a one-off generator that re-emitted
`1.1`, `1.2`, `1.3`, `2.1`, `2.3`, `3.1`, and `2.9` from an older English revision
**without** outputs. Re-running it would have rolled back the Chinese markdown
migration and the recorded execution outputs, and keeping it in sync would have
meant maintaining a second copy of every notebook cell.

- The committed notebooks are now the single source of truth for their own code
  and prose. Edits go into the notebook; re-execution fills outputs via
  `nbclient`.
- The file remains recoverable from Git history
  (`git show 2b4de68:scripts/build_lesson_notebooks.py`, its only revision);
  nothing unique was lost, because every code cell it printed also exists in the
  notebooks.
- Stale `scripts/__pycache__/build_lesson_notebooks.cpython-312.pyc` removed.
- References updated: `../README.md` (repository tree, the "supported data path"
  paragraph, and the notebook-source-of-truth paragraph), `archive/README.md`
  (the retained-tools list plus an explicit deleted-not-archived note). New
  scripts `generate_expert_demo.py` and `convert_expert_actions_to_delta.py` were
  added to both lists at the same time.
- The dated "scripts/ reorganization" entry further down this log still lists the
  generator among the retained scripts; that is the historical record of that
  session, and this entry supersedes it.

### 2026-09-22 — Notebook markdown migrated to Chinese (durable style rule)

The learner asked that notebook notes be written in Chinese except for technical
terms, and that this be kept from now on. Applied to all 13 notebooks.

- `../AGENTS.md` gained a "Notebook documentation language" subsection: markdown
  prose in Chinese, technical terms in English, code/identifiers/paths/numbers
  untouched, structure preserved, code cells and outputs untouched, and `*.md`
  repository docs plus code comments stay English.
- All 13 notebooks under `notebooks/` had their markdown cells translated. The
  work was fanned out one agent per notebook, then verified centrally rather
  than trusted.
- Verified `[verified]` for every notebook, against a pre-translation snapshot in
  `/tmp/nb_before_translation/`:
  - `nbformat.validate` passes; cell count and cell-type sequence unchanged;
  - every code cell's `source`, `outputs`, `execution_count`, and `metadata`
    byte-identical — the translation touched markdown only;
  - the multiset of inline-code tokens (`` `...` ``) is unchanged, so identifiers
    and paths survived;
  - fenced code-block bodies unchanged;
  - table row/column structure unchanged;
  - markdown CJK ratio 23–42% (the remainder is terms, identifiers, and code).
- Normalised style across notebooks after review: `Lesson x.y` kept in English in
  the H1 headings (matching `docs/roadmap_v3.md` numbering), and the closing
  section is `## 小结` everywhere.
- Reviewed `scripts/build_lesson_notebooks.py`, which re-emitted `1.1`, `1.2`,
  `1.3`, `2.1`, `2.3`, `3.1`, `2.9` from an older English revision **without**
  outputs. It was deleted later the same session, on the learner's instruction —
  see "Notebook generator deleted" below.

New finding while reviewing the translation (recorded, not fixed): see "2.6
markdown claims `shuffle=True` but the code uses `shuffle=False`".

### 2026-09-22 — Expert collector fixed, dataset regenerated, notebook 2.9 corrected

Acted on the confirmed collector defects: the transition schema and the
failure-episode handling. This is the first session that changed a dataset
payload, deliberately and with the learner's approval.

Changes:

- `scripts/generate_expert_demo.py` rewritten:
  - `StepRecorder` now wraps `env.reset` **and** `env.step`; it stores
    `observations[T+1] = o_0 … o_T`, `actions[T]`, `rewards[T]`, and raises if the
    schema invariant fails instead of writing a malformed episode;
  - `verify_control_mode()` asserts `control_mode == "pd_joint_pos"` and an 8-d
    action space before recording, so a future mode mismatch fails immediately
    rather than silently mis-executing arm actions and dying at the gripper;
  - failed seeds are reported to stderr and excluded (`continue`), instead of
    being written alongside the expert episodes;
  - root attributes gained `transition_schema`, `transition_schema_version=2`,
    `num_episodes`, `failed_seeds`; per-episode attributes gained `num_actions`
    and `num_observations`;
  - the docstring now states the `T+1`/`T` schema, why `o_0` must be captured, and
    both 15-d helper branches.
- `datasets/pickcube/expert_episodes.h5` regenerated in `embodied310`
  (`--seeds 0 1 2 3 4 --overwrite`): 5/5 successful, `T = 74 / 74 / 50 / 86 / 76`,
  `T+1 = 75 / 75 / 51 / 87 / 77`.
- `notebooks/2.9_expert_demonstrations.ipynb` corrected and re-executed:
  - cell 5's printed narrative now distinguishes the arm phase (8-d, silently
    misread in delta mode) from the gripper helpers (15-d, rejected);
  - the "bypasses `env.step`" section was replaced by "How the planner executes
    its actions", with the mode-guard table for `follow_path` vs the gripper
    helpers;
  - section 2.9.6 now documents the `T+1`/`T` schema and the fixed defect, with an
    actions/observations table;
  - the h5 inspection cell prints `acts`/`obs` and asserts
    `len(observations) == len(actions) + 1`;
  - section 2.9.7 now reports the two verified replay results and explains that
    the old "~1e-2 with a one-frame `is_grasped` flip" was the schema offset, not
    solver noise;
  - kernelspec set to `embodied`; all 7 code cells re-executed with zero errors.

Verification (all `[verified]`, re-run after regeneration):

| Check | Result |
|---|---|
| Schema invariant per episode | `len(observations) == len(actions) + 1 == len(rewards) + 1`, all `float32` |
| `observations[0]` vs `env.reset(seed)` | exact (`max |diff| = 0.0`) |
| Action replay, all 5 episodes | `O[t]` = pre-step state, `O[t+1]` = post-step state, rewards exact (`max error 0.0`) |
| Determinism across two planner runs | `actions`, `rewards`, `observations` bit-identical (`np.array_equal`) |
| Additive-only change | new `observations[1:]` equals the old file's post-action `observations` exactly |

What this closes: the post-action observation offset, the failed-episode
persistence, the three 2.9 documentation errors, and the action-semantics
incompatibility. What it does not close: the train/validation experiment itself
and closed-loop policy execution.

The old-schema file was backed up outside the repository at
`/tmp/expert_episodes_oldschema.h5` for the additive-only equivalence check;
nothing else was written.

### 2026-09-22 — Expert actions retargeted to delta semantics, replay-verified

Continuation of the collector session, covering the 2.8.7 prerequisite. This
creates a second dataset artifact; nothing existing was overwritten.

New artifact: `scripts/convert_expert_actions_to_delta.py` →
`datasets/pickcube/expert_episodes_delta.h5`.

- Conversion: `a_arm[t] = clip((q_target[t] − qpos_t) / 0.1, −1, 1)`, where
  `q_target[t] = actions[t][0:7]` and `qpos_t = observations[t][0:7]`. The gripper
  channel is unchanged (both modes read `action[7]` as the same normalized
  absolute target), and observations, rewards, and the `T+1`/`T` schema are
  copied.
- Feasibility measured before implementing: `|Δq|` mean `0.0115` rad, `p99`
  `0.0715` rad, max `0.1008` rad, so only **1 of 2520** arm channels would clip
  (`0.040%`). Reconstruction error on the unclipped channels `≤ 4e-9` rad.
- Provenance written into the file: formula, `delta_scale_rad`, source path and
  `sha256:5154c16a7e833b01dbe070ee482af72f0814c836478f88dc38b2352c2804d758`,
  clipped counts, and an explicit "re-targeted, not re-executed" note.

Replay verification (`[verified]`, `pd_joint_delta_pos`, recorded seeds):

| Check | Result |
|---|---|
| Observation error vs the expert trajectory | `≤ 1e-4` (episodes 0–3); `5.3e-3` on `episode_000004`, the clipped one |
| Reward error | `≤ 1.7e-4` |
| Task outcome | **5/5 `success`** |
| Controller comparison | both modes: `stiffness 1000`, `damping 100`, identical gripper config; the arm differs only in `use_delta` |

Conclusion: because the two modes share the controller and differ only in how the
arm target is computed, expressing the expert's absolute target as a delta
reproduces the same commanded targets — the conversion is closed-loop faithful on
replay, not merely BC supervision. It remains true that this is a re-targeting:
the file's states and rewards are the planner's, while its actions are delta
commands that the planner never issued.

README status line updated: the converter now exists, so 2.8.7's remaining work is
the split experiment itself. `notes/concepts.md` gained the corresponding durable
conclusion about comparing controller configurations before declaring two modes
incompatible.

### 2026-09-22 — Source-level confirmation of the expert collector's transition schema

Follow-up to the audit below. The learner challenged the finding on
`expert_episodes.h5`; the challenge holds, and it is now confirmed at source
level rather than by inference.

Read and confirmed (`[verified]`):

- `scripts/generate_expert_demo.py:72–77` executes `env.step` first and appends
  the **returned** observation, so the file stores `observations[t] = o_{t+1}`,
  `actions[t] = a_t`, `rewards[t] = r_t`.
- `solve()` calls `env.reset(seed=seed)`
  (`mani_skill/examples/motionplanning/panda/solutions/pick_cube.py:9–10`), and
  `StepRecorder` wraps only `env.step`, never `env.reset`. The reset observation
  `o_0` is never captured. The docstring/notebook claim "the `(o_t, a_t)` pairing
  holds by construction" is therefore wrong.
- Replayed **all five** episodes from their recorded seeds under `pd_joint_pos`:
  every `actions` array is `(T, 8) float32` and inside the action space, and the
  stored observations and rewards are reproduced exactly
  (`max |O − post| = 0.0`, `max |R − rew| = 0.0`), with no shape error on any
  step. The 15-d helper branch never fires under `pd_joint_pos`, so it does not
  contaminate this dataset.
- Read `scripts/generate_expert_demo.py:115–150, 183–191`: every episode is
  appended and written regardless of `success`; the only guard is "at least one
  succeeded". Failures would be stored alongside experts, labelled
  `success=False`. Not an active problem here (all five are `success=True`), but a
  latent defect in the same collector.

Conclusion recorded in "Expert HDF5 stores post-action observations": the current
file is not polluted and its 8-d actions are correct, but the schema loses `o_0`
and `(o_1, a_1) … (o_{T-1}, a_{T-1})` is all that can be recovered by shifting
(73 pairs per 74-frame episode). Since only five episodes exist, the recommended
path is to fix the collector — record the pre-step observation and `o_0`, filter
or segregate failed episodes — and regenerate, rather than train on the shifted
view. No dataset payload, script, or notebook was modified.

### 2026-09-22 — Progress reconciliation against the repository (read-only audit)

No new experiment, dataset, or code was produced in this session. The purpose was
to make this file agree with the repository at `80a97b5` ("Alter Dataset
position"), which it had drifted from: the headline status still described the
project as stopping at 2.8.6, while the tree already contained the 2.9 expert
episodes and the 3.1 notebook.

Commands run, and what they showed (`[verified]`, all read-only):

- `git status --porcelain` → empty; the tree is clean at `80a97b5`, so nothing was
  left uncommitted by the previous session.
- `h5py` read-back of `datasets/pickcube/expert_episodes.h5` → 5 episodes
  (`episode_000000` … `episode_000004`), frames `74 / 74 / 50 / 86 / 76`
  (360 total), `success` / `is_obj_placed` / `is_robot_static` all `True`; root
  attributes `control_mode=pd_joint_pos`, `control_freq_hz=20`,
  `data_quality=expert_planner`, `timestamp_source=derived_not_measured`,
  `generator=scripts/generate_expert_demo.py`.
- `h5py` read-back of `datasets/pickcube/random_episode_standard.h5` → **no** root
  attributes, and `metadata` records `control_mode=pd_joint_delta_pos`. The two
  files cannot be concatenated as-is.
- Read `datasets/lerobot/pickcube/meta/info.json` → `codebase_version=v3.0`,
  `total_episodes=1`, `total_frames=50`, `total_tasks=1`, `fps=50`; features are
  `observation.state`, `action`, and the index/timestamp fields only; no `videos/`
  directory. The expert episodes have **not** been converted to LeRobot.
- Recomputed action smoothness from the payloads → random `mean |Δa| = 0.6723`;
  expert per-episode `0.0078 / 0.0074 / 0.0080 / 0.0077 / 0.0077`
  (mean `0.0077`). This reproduces the recorded `0.67` versus `0.0078`.
- Parsed every `.ipynb` → 13 notebooks, zero error outputs. All code cells are
  executed except one cell of the scratch notebook `2.8_check_data.ipynb`
  (35 of 36).
- Searched `notebooks/` and `scripts/` for `random_split`, `val_loader`, and
  validation code → none. 2.8.7 and 2.8.8 remain unimplemented.
- `ls scripts/inspect_robot_dataset.py` → no such file; the reusable inspector is
  still missing.
- Re-measured the 3.1 leakage statistic per dataset → random fixture ratio
  `1.04×`, expert episodes `8.8×`–`13.7×`. The notebook's conclusion is correct
  for expert data and **unsupported by the fixture it currently loads**.
- Replayed the expert actions instead of trusting the notebook's description:
  `env.reset(seed=0)` + the 74 stored actions of `episode_000000` under
  `pd_joint_pos` reproduce the stored observations **exactly**
  (`max ‖O[t] − post[t]‖ = 0.0`) and the rewards exactly (`max error 0.0`),
  while the pre-step observation does not match (`mean 0.0687`, `max 1.0662`).
  This establishes that the file stores `(o_{t+1}, a_t)`, not `(o_t, a_t)`, and
  that the recorded actions do drive the arm through `env.step`.
- Re-ran the committed acceptance-gate replay,
  `python scripts/replay_pickcube_episode.py` → 50 steps, `control_freq: 20`,
  `max_observation_error: 0.0`, `max_reward_error: 0.0`, `truncated` at step 50,
  `success_any: False`. The random fixture's pre-action observation convention is
  reconfirmed, and it differs from the expert file's.
- Read the installed planner source
  (`mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py`) →
  `follow_path` calls `env.step([qpos(7), gripper])`, so the notebook's claim that
  the arm motion bypasses `env.step` is wrong; the collector's arm actions are
  real.

Changed in this file:

- 2.9 promoted from a session-log entry to a first-class completed sub-step with
  its own evidence block, and the "Current Position" summary and completion
  estimate rewritten to include it;
- the Lesson 2 acceptance gate gained the expert-demonstration row, and the source
  trajectory row was strengthened from `[reported]` to `[verified]`;
- 2.8.7's next-step description now states its two real prerequisites (the
  `pd_joint_pos` → `pd_joint_delta_pos` converter and the observation-offset
  repair) instead of treating the split as immediately runnable;
- five new open items: incompatible action semantics between the two on-disk
  datasets, the expert file's post-action observation offset, two inaccurate
  claims in the 2.9 documentation, the fixture-dependent 3.1 leakage measurement,
  and the `2.9` versus `2.1`–`2.8.8` notebook-numbering mismatch;
- the deferred list updated accordingly.

Not done, deliberately: no dataset payload was written, no pipeline was re-run, no
notebook was executed, no `.ipynb` was modified, and nothing was committed. The
`expert_episodes.h5` defect is recorded, not fixed: changing the recorder or the
stored arrays is a data-semantics decision that requires its own regeneration and
replay verification.

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
- Updated `../README.md`: repository tree, status line, Current Progress table,
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
