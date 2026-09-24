# Embodied AI Learning Progress

Last updated: 2026-09-24

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

Lesson 2 is **complete**. The lesson is one continuous thread: **from a simulation
trajectory to a minimal behavior-cloning model that is actually deployed and
measured**. Its deliverable is not a working policy; it is a working and honest
measurement loop. The pipeline trains, the held-out evaluation exposes
overfitting, and the closed-loop rollout settles whether the model can do the
task — it cannot.

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
| 2.8.6 | 完整训练循环：batch 循环、epoch 循环、Adam、Loss 曲线 | **Complete `[verified]`** |
| 2.8.7 | 训练集与验证集：按 episode 划分、early stopping、baseline 比较 | **Complete `[verified]`** — 验证集揭示过拟合，泛化未成立 |
| 2.8.8 | 策略部署与闭环执行：`env.step(policy(state))`、action clipping、rollout 判定 | **Complete `[verified]`** — 闭环已跑通，策略未能完成任务 |

One further sub-step exists outside that sequence:

| Sub-step | Topic | Status |
|---|---|---|
| 2.9 | 专家演示：ManiSkill motion planner 生成成功 episode | **Complete `[verified]`** — 5/5 成功，360 actions / 365 observations |

**2.9 — 专家演示（数据资产）.** `notebooks/lesson_2/2.9_expert_demonstrations.ipynb`
plus `scripts/generate_expert_demo.py` drive ManiSkill's own sampling-based motion
planner and record five successful `PickCube-v1` episodes into
`datasets/pickcube/expert_episodes.h5` (`T = 74 / 74 / 50 / 86 / 76` actions,
`T + 1` observations each, 5/5 `success`), in the canonical `T + 1` / `T`
transition schema (`transition_schema_version=2`). A retargeted copy in the
fixture's delta semantics exists as
`datasets/pickcube/expert_episodes_delta.h5` (`scripts/convert_expert_actions_to_delta.py`),
replay-verified with 5/5 success. The 2.8.7 / 2.8.8 experiment used the
`pd_joint_pos` file end to end, so dataset semantics and rollout environment
agreed; the delta copy is what would let this data be mixed with the
`pd_joint_delta_pos` fixture.

**Lesson 2 result (formal record).**

> The BC model fitted the four training demonstrations but failed to generalize
> to an unseen episode. Its best validation MSE (0.2350) was worse than the
> mean-action baseline (0.1421). During closed-loop rollout on unseen seed 100,
> the policy failed to complete the task and produced out-of-range actions on
> 199 of 200 steps, demonstrating severe overfitting and compounding
> distribution shift.

**The training code is correct; the model is a failure baseline.** Every observed
behaviour is the intended one: the training loss falls, the validation loss
exposes overfitting, early stopping keeps the best epoch, the baseline comparison
shows no generalization, action clipping keeps outputs inside the control bounds,
and a 50-step versus 200-step comparison rules out the episode time limit as the
cause. That is precisely why training loss alone cannot be the acceptance metric.

**What comes next is data, not a bug fix.** If a usable BC policy becomes the
goal, the next round is data scaling: collect at least 30–50 expert episodes and
re-train with the same episode-level split. The current model stays in the
repository as a documented failure baseline to compare against.

**One AGENTS.md gate item is still unmet:** the reusable
`scripts/inspect_robot_dataset.py` does not exist, and the project's own completion
gate lists it. Lesson 2 is declared complete by the learner; that item is recorded
as open, to be either built or explicitly waived.

### Why this negative result is the Lesson 2 deliverable

The experiment was never meant to produce a controller. It exists to expose the
theory that opens Lesson 3:

- why training loss alone is not enough: it fell to `0.006` while held-out loss
  stayed at `0.75`–`0.77`;
- train/validation split, generalization versus memorization: 284 training
  samples from four episodes, 76 validation samples from one;
- underfitting and overfitting, and why early stopping is not optional;
- why a metric needs a baseline: the model lost to "always predict the mean
  action" (`0.2350` versus `0.1421`);
- distribution shift in execution: validation states reach `|z| = 13.2` against
  training statistics whose std is `0.0088`, and the deployed policy drifts
  further at every step.

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

Interpretation, and the reason 2.8.7 came next:

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

### 2.8.7 — Train/validation split (episode-level)

`[verified]` from the executed cells of `notebooks/lesson_2/2.8_check_data.ipynb`
(section `2.8.7`), on `datasets/pickcube/expert_episodes.h5`:

| Item | Value |
|---|---|
| Split rule | **by episode**, reproducible (`np.random.default_rng(seed=42)`) |
| Episodes | training `episode_000000`–`000003`; validation `episode_000004` |
| Samples | train `284`, validation `76` |
| Normalization | statistics computed from the **training split only** (`std` floored at `1e-6`) |
| Loaders | `train_loader(shuffle=True)` (9 batches), `validation_loader(shuffle=False)` (3 batches), `batch_size=32` |
| Missing values | the policy consumes `observations[:-1]` against `actions`, and the loader asserts `len(observations) == len(actions) + 1` |
| Epochs | `max_epochs = 300`, early stopping after 40 epochs without improvement |
| Training loss | `1.3136` at epoch 1 → `0.006` by epoch 20–40 |
| Validation loss | `0.75`–`0.77` throughout; best `0.2350` at **epoch 3** |
| Best vs baseline | best BC `0.2350` vs mean-action baseline `0.1421` → **worse by `0.0929`** |
| Per-dimension error | dominated by `joint_4` `0.570`, `gripper` `0.561`, `joint_2` `0.409`; gripper target std `0.994` |
| Held-out state range | max `|z| = 13.21` on dimension 39, whose training std is `0.0088` |

Reading: the model memorized four episodes. Held-out loss is worse than a
constant predictor, which is the practical definition of "no generalization yet",
and the held-out observations are far outside the training statistics.

### 2.8.8 — Closed-loop deployment and rollout

`[verified]` from the executed cells of the same notebook (section `2.8.8`):

| Item | Value |
|---|---|
| Environment | `PickCube-v1`, `obs_mode="state"`, `control_mode="pd_joint_pos"` (matching the demonstrations) |
| Policy | best checkpoint from 2.8.7, `model.eval()` |
| Preprocessing | the **same** training normalization, applied inside the rollout loop |
| Safety | every action clipped to `env.action_space` bounds; clipping counted per step |
| Unseen seed | `100` (`reset(seed=100)`) |
| Episode limit | default `TimeLimit` is 50 steps; the wrapper's `_max_episode_steps` was overridden to `200` to rule the time limit out |
| 50-step run | `Success: False`, `clipped_steps = 49 / 50`, terminated by `truncated`, total reward `0.3277` |
| 200-step run | `Success: False`, reward `0.0` from step ~20 onward, `clipped=True` on every step after the first |
| Clipping per channel | `joint_2 / joint_4 / joint_6 / joint_7 / gripper` `199/200`; `joint_1` `149/200`; `joint_3` `129/200`; `joint_5` `101/200` |
| Multi-seed success rate | `scripts/evaluate_bc_closed_loop.py --seeds 100…109` (fresh process, saved checkpoint): **0/10 = 0.0%**, all 200 steps and `truncated`, mean clipped fraction `0.995` |
| Fresh-process contract check | `scripts/verify_bc_checkpoint.py`: `strict=True` load, `pd_joint_pos`, obs `42`, action `8`, clipped action inside bounds, validation MSE `0.23502295` and baseline `0.14210252` reproduced exactly |

Reading: the policy reaches no goal, drifts immediately off the training
distribution, and saturates the control bounds for the rest of the episode. The
50-versus-200 comparison shows the failure is not an artifact of the episode time
limit. `gymnasium`'s default `TimeLimit` silently caps episodes at 50 steps, which
is why the 200-step configuration had to assert the effective limit explicitly.

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
  **`[open]` 2026-09-24 — the flip was reproduced.** A fresh action replay reproduces
  only episode 0 exactly; episodes 1, 2 and 4 drift (`qpos` <= `7.3e-05` rad,
  `obj_pose` <= `4.8e-03`) and **four of five episodes show `is_grasped` differing at
  exactly one step**. Episode 3 is the sharpest case: every other block is bit-exact and
  `is_grasped` still differs at one step. Re-running the *planner* in a fresh process is
  bit-exact in all five episodes, so what fails to reproduce is the **action replay**,
  not the planner. The original measurement may have compared a different code path;
  this marker records the conflict without adjudicating it. Evidence and the per-block
  table: "RGB expert collection, and an action-replay conflict with 2.9" below.
- [x] Action smoothness separates the two data regimes. `[verified]` — recomputed
  from the payloads: random fixture `mean |Δa| = 0.6723`; expert per-episode
  `0.0078 / 0.0074 / 0.0080 / 0.0077 / 0.0077` (mean `0.0077`).
- [x] Seeds actually vary the task instance, so scaling the seed range adds real
  diversity. `[verified]` — across the five episodes the initial cube position
  spans `0.1507 m` in `x` and `0.1325 m` in `y`, the goal position spans
  `0.1522 / 0.1192 / 0.2452 m`, and the mean pairwise distance between reset
  observations is `1.0945`, i.e. **`16.3×`** the mean consecutive-frame distance
  inside an episode (`0.0671`). Caveat for the next phase: the variation is in
  *where* the object and goal are, not in *how* the task is performed, because all
  episodes come from the same scripted planner recipe.
- [x] Notebook exists, is corrected, and is executed. `[verified]` —
  `notebooks/lesson_2/2.9_expert_demonstrations.ipynb`, 23 cells (14 markdown / 9 code),
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
`notebooks/lesson_2/2.9_expert_demonstrations.ipynb` was corrected (sections 2.9.2, "How
the planner executes its actions", 2.9.6, 2.9.7, 2.9.8, takeaways) and
re-executed end to end with zero errors, kernel `embodied`. The three issues are
kept below as the record of what was wrong.

The three issues were (`[verified]` by reading the installed planner source and
replaying the data):

1. `notebooks/lesson_2/2.9_expert_demonstrations.ipynb` (section 2.9.6) and
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

`[verified]` `notebooks/3.1_imitation_learning_foundation.ipynb` measures the distance
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

`[partly addressed 2026-09-23]` The notebook markdown now states this explicitly:
the cell that measures `1.0x` is annotated as measuring the random fixture, with
the expert-episode ratio `8.8×–13.7×` given for contrast. The **code** still loads
the random fixture, so re-pointing the measurement at the expert episodes remains
a suggested edit, not a completed one.

### Notebook numbering: `2.9` versus a roadmap sequence ending at `2.8.8`

`[verified]` `docs/roadmap_v3.md` (and this file's sub-step table) number Lesson 2
as `2.1`–`2.8.8`, with no `2.9`, while `notebooks/lesson_2/2.9_expert_demonstrations.ipynb`
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
| Train/validation | Held-out split, generalization assessed | **PASS `[verified]`** — episode-level split, best validation MSE `0.2350` vs mean-action baseline `0.1421`; the assessment is a documented **negative** generalization result |
| Closed-loop execution | Policy drives `env.step` and task outcome known | **PASS `[verified]`** — best checkpoint deployed with clipping on unseen seed 100; task **not** completed at 50 and 200 steps |
| Reusable inspection | `scripts/inspect_robot_dataset.py` runs independently | **ABSENT `[verified]`** — the only unmet item of the project's own completion gate (declared complete by the learner; build it or waive it explicitly) |

## BC Closed-Loop Gate (per `AGENTS.md`)

`AGENTS.md` defines a separate gate for the baseline BC artifact:
"do not treat baseline BC as complete until all of the following are
evidenced". Assessed against the executed notebook
(`notebooks/lesson_2/2.8_check_data.ipynb`) and the on-disk checkpoint:

| # | Requirement | Status | Evidence |
|---:|---|---|---|
| 1 | Expert demonstrations, not random fixtures | **MET `[verified]`** | 5 planner episodes, 5/5 `success`, `expert_episodes.h5` |
| 2 | Split by episode/trajectory, no frame leakage | **MET `[verified]`** | 4 training / 1 validation episode, `default_rng(seed=42)`; leakage measured as 8.8×–13.7× on smooth data versus 1.04× on the random fixture |
| 3 | Normalization and action scaling fitted on training data only and saved for inference | **MET `[verified]`** | statistics computed from `train_observations` only; both are persisted in `checkpoints/pickcube_bc_best.pt` together with `observation_dim`, `action_dim`, `hidden_dim`, `best_epoch`, and the episode lists |
| 4 | Loss curves, seeds, configuration, checkpoint selection recorded | **MET `[verified]`** | per-epoch train/validation losses, `torch.manual_seed(42)`, `max_epochs=300`, `patience=40`, `batch_size=32`, `hidden_dim=64`, best epoch 3 |
| 5 | Checkpoint loads in a **fresh process** and produces actions with expected shape, range, units, frame, control semantics | **MET `[verified]`** | `scripts/verify_bc_checkpoint.py` reads the file from disk in a new process, rebuilds the architecture from the stored dims, loads with `strict=True`, checks `control_mode=pd_joint_pos` / obs 42 / action 8 / clipped action inside bounds / output `(1,8) float32`, and reproduces the recorded validation MSE `0.23502295` and baseline `0.14210252` exactly |
| 6 | Executes in ManiSkill closed loop under the same observation/controller contract | **MET `[verified]`** | `obs_mode="state"` + `control_mode="pd_joint_pos"` for both training data and rollout; 50- and 200-step rollouts |
| 7 | Task success rate across **multiple seeded episodes**, plus representative failure modes | **MET `[verified]`** | `scripts/evaluate_bc_closed_loop.py --seeds 100…109`: **0/10 success (0.0%)**, every episode 200 steps and truncated, mean clipped fraction **0.995**, reward mean `0.4814` / max `0.9333`. Report: `scripts/reports/bc_closed_loop_eval.json` |
| 8 | At least one failure analysed via state-distribution shift, compounding error, action semantics, or insufficient history | **MET `[verified]`** | held-out `|z| = 13.21` (dim 39, training std `0.0088`), reward collapse to `0.0` by step ~20, `199/200` clipped steps |

Score: **8 of 8 met**. The two verification items were closed in the same
session by two new scripts (both load the checkpoint from disk in a fresh process,
never from an in-memory state dict):

- `scripts/verify_bc_checkpoint.py` → item 5 (action/observation contract plus an
  exact reproduction of the recorded validation MSE and baseline);
- `scripts/evaluate_bc_closed_loop.py` → item 7 (10 seeded episodes, success rate,
  clipping statistics, JSON report).

One sharp contrast worth keeping: on the **held-out episode's own states** the raw
policy output is out of bounds for only `1/608` values (`0.2%`), but once the policy
drives the environment the clipped fraction rises to **`0.995`**. That gap is a
direct measurement of compounding distribution shift, not an inference.

The separate Lesson 2 gate is still short of one item:
`scripts/inspect_robot_dataset.py` does not exist.

## Immediate Next Steps

**Lesson 2 is closed and Lesson 3 is the current phase.** The learner has also
abandoned the two Lesson 2 follow-ups that were previously queued here — growing
the expert dataset to 30–50 episodes and re-training the BC baseline, and building
`scripts/inspect_robot_dataset.py`. No further work should be based on either;
they are recorded as abandoned decisions, not as pending tasks.

Lesson 3 — imitation learning and Behavior Cloning — is entered from the **measured**
Lesson 2 result rather than a toy example. The evidence already in the repository
(recorded in the 2.8.7 / 2.8.8 sections above) provides the entry points:

| Lesson 3 topic | The evidence that already exists |
|---|---|
| 3.1 problem definition | `π_θ(a_t \| o_t) ≈ π_E(a_t \| o_t)`, with the trained artefact as `π_θ` |
| 3.2 BC as supervised learning | MSE objective, and its failure mode: the model landed worse than the mean-action baseline (`0.2350` vs `0.1421`) |
| 3.3 generalization and overfitting | measured curves: train `1.3136 → 0.006` versus validation stuck at `0.75–0.77`; best epoch 3; episode-level split |
| 3.4 distribution shift | held-out states reach `\|z\| = 13.21`; raw actions out of bounds on only `0.2%` of held-out states but `99.5%` of closed-loop steps are clipped |
| 3.5 DAgger | not started; the natural follow-up once 3.4 is understood |
| 3.6 single-frame versus history policy | not started; this is the clean way to separate "not enough data" from "not enough model" |
| 3.7 action chunking | **Complete `[verified]`** — `notebooks/3.7_Action_Chunk.ipynb`, 31 cells (20 md / 11 code, all executed, 0 errors), extended with an H sweep (S9) and a K-mechanism measurement (S10). Module structure with H=8, K sweep, controlled single-step baselines B1/B2, per-horizon diagnostic. Offline result is **negative**: useful horizon 0, and chunked@h=0 (0.5718) equals B1 (0.5752) while only the larger B2 (0.4279) beats the mean-action baseline (0.5576). Closed loop **0/5 success at every K**, but clipping falls monotonically 0.947 -> 0.121 as K goes 1 -> 8 |
| 3.8 multimodal policy transition | **In progress** — data/contract/module layer done as four notebooks (`3.8a_contract` 14 cells, `3.8b_observability` 10, `3.8c_language` 21, `3.8d_conditioning_tests` 11) over the single-source `scripts/mml_contract.py`, file `3.8_multimodal_policy.ipynb` now an index. Model layer: `scripts/mml_policy.py` plus `3.8e_fusion_model.ipynb` (18 cells, 3.8.5: Encoder + concat Fusion + action head, 528,800 params, action side still 3.7's contract) and `3.8f_ablation.ipynb` (21 cells, 3.8.6: six arms, one per cell). Headline results: T2 at float32 rounding on every language arm; the `t=0` probe scoring 50.0% without language against the proven 50% bound and 100.0% with it; and from the 3.8g 3-seed sweep, `Delta_vision` **not established** (mean `+0.002975`, sign flips across seeds) while `E3c`/`E3b` recovered the L0-ceilings-L1 direction, and the one effect that clears the seed noise is capacity-matched `I+p+goal` against `I+p+language` at **2.27x** (a learnability gap at equal information). Remaining: 3.8.7 VLA interface, and closed-loop evaluation |
| 3.9 expert data collection | partially informed by 2.9 (single scripted planner recipe, object/goal diversity but no behavioural diversity) |
| 3.10 trajectory to task structure | not started; the interface toward task representation and procedural memory |

`notebooks/3.1_imitation_learning_foundation.ipynb` already exists and executes. Its
frame-level leakage cell still points at the random fixture, where the measured
ratio is `1.04×` and therefore demonstrates nothing; re-pointing it at the expert
episodes (`8.8×`–`13.7×`) is the first concrete Lesson 3 edit.

## Session Log

### 2026-09-24 — 3.8g: the seed sweep that overturned one ranking and established another

`notebooks/3.8g_seed_sweep.ipynb` (23 cells, 14 code) trains E1b/E2/E3b/E3c at init seeds 0, 1, 2
against a fixed `SPLIT_SEED = 42`. Ran on the GPU: 14/14 code cells, 0 errors.

| arm | inputs | mean | min | max | range | grip |
|---|---|---|---|---|---|---|
| E1b | `p+g`, hidden 1139 | 0.085504 | 0.068587 | 0.102373 | **0.033787** | 99.0% |
| E2 | `I+p+g` | 0.082529 | 0.074864 | 0.088348 | 0.013484 | 97.4% |
| E3b | `I+p+task-ID` | 0.041883 | 0.036425 | 0.045598 | 0.009173 | 99.5% |
| E3c | `I+p+l` | 0.036432 | 0.031756 | 0.043474 | 0.011718 | 99.2% |

**Both pre-registered pairs came out OVERLAP, so the 3.8.6 arm ranking does not stand.**

1. **`Delta_vision` is not established, and its sign flips.** Seed 0 reproduces the 3.8.6 number
   (`+0.014025` against `+0.014186` on GPU), but seeds 1 and 2 give `-0.006277` and `+0.001176`;
   the mean is `+0.002975`. The single-seed `+0.014` was one lucky draw, so the claim that the
   image carries object position (because `proprio` omits `obj_pose`) is **not supported**.
2. **`Delta_E3` recovered the predicted direction.** 3.8.6's single seed gave `-0.0029` / `-0.0013`
   (C better than B, against 3.8.4.4's ceiling prediction); all three seeds are positive
   (`[+0.0024, +0.0021, +0.0119]`, C worse than B), which is the L0-ceilings-L1 direction. The
   magnitude is still not established.
3. **What the sweep did establish is a different, larger effect.** E2 and E3c are capacity-matched
   (fusion dim 416 both, parameters within 0.1%), share the image and proprioception, and differ
   only in the condition channel. Their seed ranges are **disjoint** and the error differs by
   **2.27x** (`0.0825` against `0.0364`). Since `H(l | g) = 0` the two carry the same information,
   so this is a **learnability** difference rather than an information one: thresholding a
   continuous 3-dim goal into a task is harder than reading a discrete index. It generalises
   3.8.4.4 one step -- representation choice is not only about information content.
4. **The capacity-matched control is the noisiest arm.** E1b's range (`0.0338`) is 2.5x E2's.
   Matching capacity made the comparison possible and made the control least stable, so
   `Delta_vision` is exactly the comparison that matching makes least reliable.

3.8f's reading cells now carry `<补正 3.8g>` notes in place. Durable conclusions recorded in
`notes/concepts.md` under "Ablation Discipline".

## Session Log

### 2026-09-24 — 3.8.5 / 3.8.6: the fusion model, and an ablation whose capacity control changed the answer

Three new pieces: `scripts/mml_policy.py` (dataset, model, training loop, metrics),
`notebooks/3.8e_fusion_model.ipynb` (3.8.5, 18 cells) and `notebooks/3.8f_ablation.ipynb`
(3.8.6, 21 cells, one arm per cell).

**The model (3.8.5).** Encoder + concat fusion + action head, 528,800 parameters with all four
modalities: `image -> small CNN -> 256`, `proprio -> MLP -> 128`, `goal -> MLP -> 32`,
`language -> Embedding + masked mean pool -> 32`, `concat 448 -> 512 -> 256 -> 64`,
`view(B, 8, 8)`. The action side is byte-for-byte 3.7's contract, so any difference measured
here is attributable to the input modalities.

Two deviations from the learner's proposed V1, each with a hard reason:

- **A from-scratch CNN instead of a pretrained ResNet18.** `~/.cache/torch/hub/checkpoints/`
  does not exist, so pretrained weights require a network download and this session's network
  failed repeatedly — a first model that cannot be reproduced offline is not a first model.
  The learner's own argument against a pretrained *text* encoder (tokenizer, pretrained
  distribution, frozen-versus-finetune) applies verbatim to a pretrained *image* encoder. And
  11M parameters against 509 training samples, with residual and BatchNorm conventions the
  learner cannot yet explain, contradicts the stated goal of a mini-VLA they fully understand.
- **Token embedding + masked mean pool instead of a one-hot task id.** A one-hot has **no word
  order**, so 3.8.4.6's test T2 is undefined and the three-rung ladder collapses to two. With
  two tasks L0 and L1 are informationally equivalent (3.8.4.4), so the substitution costs
  nothing and keeps the code path a pretrained encoder will use in 3.8.7.

Also corrected in passing: the images are `[3,128,128]`, not 224; and the multi-task data
already exists — 3.8.5 was not waiting on it.

**The ablation (3.8.6).** Six arms on one split, one normalisation, one init seed and 150
epochs, differing only in the modality switches:

| arm | inputs | params | val chunk | val h=0 | gripper sign |
|---|---|---|---|---|---|
| E1a | `p+g` | 250,176 | 0.097776 | 0.075149 | 93.0% |
| E1b | `p+g`, hidden 1139 | 511,635 | 0.101571 | 0.072430 | 99.2% |
| E2 (A) | `I+p+g` | 511,712 | 0.087082 | 0.055287 | 99.2% |
| E3a | `I+p+g+l` | 528,800 | 0.081249 | 0.057698 | 99.2% |
| E3b (L0) | `I+p+task-ID` | 511,648 | 0.036968 | 0.025324 | 99.2% |
| E3c (L1) | `I+p+l` | 512,288 | 0.034077 | 0.022164 | 99.2% |

(mean-action baseline on the same val split: 0.597227)

1. **The capacity control changed the answer.** `E1a/E2 = 2.045`, so the unmatched
   `Delta_vision` is uninterpretable. Matched it is `+0.014489` against the unmatched
   `+0.010694` — matching made the effect *larger*, because widening the state-only control
   made it slightly worse while the image arm stayed put. The image's contribution is real,
   and the mechanism is the deployability table: **`proprio` excludes `obj_pose`, so the image
   is the only source of object position.**
2. **E3a vs E2 = -0.005833**, the expected direction for a wiring control: `H(l | g) = 0`
   exactly, so adding the instruction cannot add information.
3. **E3c vs E3b = -0.002892.** 3.8.4.4 predicts `C <= B` (L0 ceilings L1), so C slightly better
   is in the expected direction; at ~8% relative, on one seed and 128 validation samples, it is
   not evidence. The honest statement is "indistinguishable, consistent with information
   equivalence".
4. **T2 sits at float32 rounding on every language arm** (`4.8e-08` for E3a, `3.1e-08` for E3c,
   against `eps = 1.19e-07`), with `T2/T3 ~ 1e-07`. The permutation-invariance prediction of
   3.8.4.6 now holds on a real model.
5. **E3b's three zeros are by design, not a bug.** `language_mode="task_id"` reads
   `task_index` and never touches `language_ids`, so shuffling, contradicting or dropping the
   token input changes nothing. This is direct evidence that the mode switch works as
   intended, and it is why T2 is undefined for the L0 arm.
6. **The restricted probe supplies the one clean language result.** E4a proves, without
   training, that a proprio-only function must return the same output for both tasks at `t=0`
   and can therefore match the gripper sign for at most one of them (`a_0` gripper is `+1.0`
   for all five PickCube episodes and `-1.0` for all five PushCube episodes). E4b confirms it:
   the no-language probe scores **50.0%** on train, exactly the proven bound, while the
   language probe scores **100.0%**.

**Recorded boundaries.** The probe deliberately removes `image`, so it answers "can language be
used", not "was language used in this policy". It has 8 training and 2 validation samples, so
its training number is a capacity claim and its validation number is not evidence. `best_val`
is selected on the validation set and is therefore optimistically biased. And on this pool the
offline metrics cannot separate "read the language" from "copied the task out of the state" —
the pre-registered negative result of 3.8.4.6.

**Process.** The 3.8.6 notebook was first built and executed in one foreground call with the
training loop's progress silenced, which looked like a hang and was aborted. That produced
`## Execution Discipline` in `AGENTS.md` and `scripts/run_notebook_observable.py`, which
executes cell by cell, prints a heartbeat plus each cell's stream output, and saves after every
cell. Two further mistakes were caught by the discipline itself: the split seed was conflated
with the init seed (moving the held-out episode from 4 to 2, caught by an assertion), and the
run progress was hidden behind an unbuffered `grep`.

**Device.** Both notebooks were re-run on an **RTX 3080 Ti Laptop GPU** (`device cuda`,
confirmed by the preamble print), which is **11.6x faster**: 3.8f went from 844s on 16 CPU cores
to 73s, and 3.8e from ~90s to 23s. The GPU was available all along; the agent's shell could not
reach it because the sandbox returned `PermissionError` on mode-666 `/dev/nvidia*` nodes, so
every earlier run silently fell back to CPU while the learner's Jupyter kernel used the GPU.
This supersedes the earlier note that CUDA is unavailable on this machine.

Re-running on the other device **changed the numbers in the second decimal** (E1a `0.097776` ->
`0.097286`, E2 `0.087082` -> `0.088187`) and moved one arm's gripper-sign accuracy from `99.2%`
to `93.8%` with a different best epoch. The load-bearing conclusions are unchanged: the matched
`Delta_vision` is `+0.014489` on CPU and `+0.014186` on GPU, and the `t=0` probe is `50.0%` /
`100.0%` on both. The arm rankings that were already within noise stay within noise, and the
`E3c`-versus-`E3b` gap shrank from `-0.002892` to `-0.001305`.

Durable conclusions recorded in `notes/concepts.md` under "Ablation Discipline".

### 2026-09-24 — 3.8 split into four notebooks, with the contract extracted to a module

`notebooks/3.8_multimodal_policy.ipynb` had reached 37 cells / 73k characters of source / 296 KB
with outputs. Size was not the real problem: **six downstream code cells each opened with the same
`assert ... in globals()` prerequisite guard**, which is the objective symptom of one notebook
standing in for six.

**Split by the question each part answers:**

| File | Content | Question | Cells |
|---|---|---|---|
| `notebooks/3.8a_contract.ipynb` | 3.8.1 contract + 3.8.2 alignment | what the data is | 14 (10 md / 4 code) |
| `notebooks/3.8b_observability.ipynb` | 3.8.3 observability | what the image does and does not carry | 10 (7 / 3) |
| `notebooks/3.8c_language.ipynb` | 3.8.4.1-3.8.4.5 | what this pool does and does not force | 21 (13 / 8) |
| `notebooks/3.8d_conditioning_tests.ipynb` | 3.8.4.6 | how to test whether a model uses language | 11 (7 / 4) |

The old path is now a one-cell index page, so existing references still land somewhere sensible.
All four parts execute with zero errors.

**The contract became a module.** `scripts/mml_contract.py` now holds the loader, the tokenizer,
`build_sample`, and the counterfactual instruction modes. Splitting without this would have meant
four copies of ~10k characters of contract code — exactly the "duplicated semantics drift apart"
failure this repository keeps recording. The module was verified **bit-for-bit** against the
notebook's own numbers before any notebook was rewritten: `VOCAB`, `T_TXT`, both `LANGUAGE_IDS`,
both field schemas, episode lengths, `T_common = 50`, and all five sample shapes. `PROJECT_ROOT`
now resolves from `__file__` rather than from the working directory.

**Tidy-up carried out at the same time:**

- Every part has its own `## 运行说明`, `## 小结`, and `## 自检`; the six guards are gone.
- The two oversized cells were split: the leakage cell (8885 chars, four unrelated jobs) became
  three, and the 3.8.4.6 fixture cell (4868) became two.
- `<补正 3.8.4.3>` no longer lives inside 3.8.4.5's reading section — it now sits with 3.8.4.3,
  the section it corrects.
- Naming: `pick` / `push` are episode lists throughout; task-level dicts are
  `datasets["PickCube-v1"]` / `pick_ds` / `push_ds`. No name means two things inside one notebook.
- Two silently dropped items were restored after the equivalence check caught them: the
  `excluded from the input: ... 14 dims` line and the per-field shape assertions, including
  **"action_chunk is byte-for-byte the same contract as 3.7"** — the statement the whole
  modality-attribution argument rests on.

**Equivalence evidence.** Every code cell's stdout from the pre-split notebook was compared against
the four new notebooks: **236 lines, zero real differences**; the only six absent lines are
CUDA/GLFW environment noise. Cell by cell, all 37 old cells are accounted for — 31 verbatim, the
two intentionally split cells verified piece by piece, and the six rewritten ones (header, run
notes, contract cell, sample cell, summary, self-check answers) replaced deliberately.

**Note on the older entries below.** The 3.8.1 through 3.8.4.6 write-ups reference
`notebooks/3.8_multimodal_policy.ipynb` and its cell counts; those describe the state at the time
and have not been rewritten. The content now lives in the four files above.

### 2026-09-24 — 3.8.4.6: input is not use is not understanding, and the ladder has to be built as three operations

`notebooks/3.8_multimodal_policy.ipynb` is now 37 cells (24 markdown, 13 code, all executed,
zero errors). 3.8.4.6 turns "does the model use the language?" into a counterfactual ladder and
establishes, without training, which rungs this pool can actually support.

**The three levels.** Input / use / understanding are separate claims. Having `language_ids` in
the contract establishes only *input*, and only 1 bit of it (3.8.4.1). *Use* requires a
counterfactual. *Understanding* requires generalisation to unseen instructions, which a
10-word vocabulary, two instructions, and one object per scene cannot test.

**The ladder, and the collapse that had to be avoided.** Three tests weak to strong: **T1**
correct instruction (fit only), **T2** shuffled instruction, **T3** contradictory instruction.
With only two tasks, "the wrong instruction" *is* "the other instruction", so building T2 as a
task swap makes it identical to T3 and leaves a two-rung ladder. The operations were therefore
separated by *what they vary*: **T2 permutes word order** (`"pick up the cube"` ->
`"the pick up cube"`, same bag) and **T3 changes the bag** (the other task's instruction). A
genuine three-rung ladder needs three or more instructions — a data-design constraint, not a
code one.

**The architectural prediction, computed rather than asserted.** 3.8.5's language encoder will
mask padding and mean-pool the token embeddings, and any mean/sum pooling is
permutation-invariant. Measured with a stand-in embedding table: `|correct - shuffled|` is
**2.2e-16** (floating-point summation order only) while `|correct - contradictory|` is
**0.609** and `|drop|` is exactly 0. So on this architecture **T2 must leave the action
unchanged**, which makes T2 a **negative control on the implementation** rather than a test of
language use; only **T3** can detect use. A T2 reading is meaningless unless reported against a
named architecture.

**Why the offline loss cannot be the criterion.** Shuffling the instruction does not change the
target, so the loss necessarily rises and conveys nothing; and a language-blind model can
already read the task from the state. The quantity to measure is instruction sensitivity
`||pi(o, l_correct) - pi(o, l_T)||`, and the full criterion is threefold: T3 sensitivity != 0,
direction correct, T2 == 0. The `drop` mode (all `<pad>`) must be injected **during training**
as instruction dropout; masking the instruction only at test time yields an
out-of-distribution artefact rather than evidence of ignoring language.

**The restricted probe, and its ceiling.** `t=0` is the only frame at which the state cannot
name the task: the five episodes of each task sit on identical proprioception (25 dims, asserted
episode-wise) and the required action differs in exactly one dimension — the 7 arm dims are
identical to 6 decimal places and the gripper differs by exactly **+2.0000** (`+1.0` pick,
`-1.0` push). That satisfies the 3.5 criterion exactly. The probe is therefore `t=0`,
`proprio` only, no `image`, no `task_goal`. The ceiling for any task-conditioned model on this
pool, computed without training: `L_blind = 1.02689`, `L_aware = 0.45271`, headroom
**0.57418 (55.91%)**, of which the gripper is **99.54%**.

**Boundaries recorded.** The probe deliberately removes `image`, so it answers "can language be
used", not "was language used in this real policy" — the latter needs tasks that are
inseparable in both state and image, which is P1 data design. `t=0` gives only 5 samples per
task (10 total; a 10/10 versus 5/10 binomial test has p ~ 0.001). The probe exercises one
dimension, so it tests "can the instruction pick the gripper's sign" — a minimal instance of
use, not sufficient evidence of it. And closed-loop feedback can reopen a mis-signalled gripper
at `t >= 1`, so probe failure does not translate directly into closed-loop failure; the final
metric remains P0 closed-loop task success.

Durable conclusions recorded in `notes/concepts.md` under "Language Grounding".

### 2026-09-24 — Correction: 3.8.4.5's leakage map was wrong in all three rows

The leakage map published earlier the same day was re-measured and **every row was wrong**.
The notebook now carries the corrected cells; the earlier entry below is superseded on this point.

Three distinct errors, one per row of the table:

1. **`image`: reported 50.9% (chance), actually 88.7%.** The code tested only one label
   orientation (`bi > thr`). PushCube is the **brighter** class (124.091 versus 123.521), so the
   orientation that was tested is false by construction and maxed out near chance. Testing both
   orientations gives **88.7%** over the window and **100% at `t=0`**.
2. **`proprio`: reported "not separable" (range overlap 0.0217), actually 98.0%.** Range overlap
   was used as the separability criterion. It is a worst-case range statistic, not a
   distribution statistic: `qpos[7]` overlaps 0.0217 and is 98.0% separable, `qvel[8]` overlaps
   0.3045 and is also 98.0%, `tcp_pose[4]` overlaps 0.0005 and lands at 81.4%. A leave-one-
   episode-out linear probe on the 25-dim `proprio` names the task at **96.3%**.
3. **`task_goal`: correct** (10/10).

A third defect was found while fixing the second: the replacement `dim_acc` initially used the
data values as thresholds and rounded them, which puts the boundary sample on a knife edge. It
read `tcp_pose[4]` as 51.2% instead of 81.4% and made `qpos[7]` alternate between 50% and 100%
by timestep. Thresholds are now placed strictly between neighbouring distinct values.

**The corrected picture, and why it matters more than the numbers.**

- The leak is **temporal** for `proprio`: at `t=0` the five episodes of each task sit on an
  **identical** state (`max |pick - push| = 0.000000`, because both were collected from the same
  seeds) and `qpos[7]` is exactly at chance; from `t=1` it is 100% separable, because PushCube's
  first action already drove the gripper to 0.0000 while PickCube held it at 0.0400.
- The leak is **not** temporal for `image`. At `t=0` the proprioception is identical but the
  **images differ** (max pixel difference 205), because the camera sees the **goal marker** and
  the two tasks place it differently. Mean brightness separates at 100% at every timestep.
- Therefore the "clean single frame" idea from 3.8.4.5 — evaluate at `t=0`, drop `task_goal`,
  and the instruction becomes the only readable channel — **holds for `proprio` and fails for
  `image`**. On this pool **no frame and no field subset makes the instruction necessary for an
  image-conditioned policy.** That is a data-design limitation (P1), not something 3.8.6 can
  repair.
- Consequence for 3.8.4.6: it must **pre-register a negative result** for the main arm C
  experiment, and test whether language *can* be used on a deliberately restricted probe —
  `t=0`, `proprio` only, no `image`, no `task_goal` — where the instruction is the only input
  that names the task.
- Note the distinction: the image's 88.7% is a **confound, not a representation**. It is a
  global brightness offset of 0.57/255, while 3.8.3 measured the goal's actual image signal at
  `corr = -0.295` and cross-episode pixel change equal to within-episode 25-step change (1.0x).
  Enough for a shortcut; no usable object information.

Durable conclusions recorded in `notes/concepts.md` under "Measuring Whether One Signal Reveals
Another". This is the seventh appearance of the repository's recurring failure mode — **a legal
shape, symbol, or statistic is not correct semantics** — and the first time it appeared in the
*measurement of a measurement*.

### 2026-09-24 — 3.8.4: the pool *permits* language but does not *force* it, and the fix is one field

`notebooks/3.8_multimodal_policy.ipynb` grew to 32 cells (21 markdown, 11 code, all executed,
zero errors) with five subsections that turn "add language to the policy" from an aspiration
into a measurement.

**3.8.4.1 — task condition versus language.** `task_goal` is a *condition* (given, no evidence
source), the instruction is *language* (must be read). Measured goal spans: PickCube
x/y/z = 0.1522 / 0.1192 / 0.2452 m, PushCube = 0.1507 / 0.1325 / **0.0000** m. Per-task
conditional entropy `H(ℓ) = 0` — inside one task the instruction is constant — and pooled
`H(ℓ) = 1.000` bit. So the instruction carries **exactly one bit**, and that bit is task
identity.

**3.8.4.2 — why this pool cannot teach language.** With `task_goal` present, the goal field
already determines the task, so the instruction is redundant: a model that ignores it pays
nothing. Distinguished "cannot learn" from "cannot be measured" — the latter is the operative
statement here.

**3.8.4.3 — the two tasks are observationally similar but actionally different, and
`task_goal` gives the task away.** Image cross-task change 6.81 versus within-task 5.54
(1.23x); action cross-task 1.6300 versus within-task 0.5185 (3.14x, early window 19.93x).
`goal_z > 0.02` separates **10/10** episodes. Conclusion: the confound is real and must be
made an explicit ablation arm rather than silently removed.

**3.8.4.4 — language representation levels.** L0 task-ID one-hot, L1 learned token embedding,
L2 pretrained encoder (deferred to 3.8.7). With only two tasks, L0 and L1 are
**informationally equivalent** (a bijection exists between them), so L0 is a **ceiling** on
L1: arm B (`image+proprio+task-ID`) caps arm C (`image+proprio+language`), and C > B would
indicate a broken experiment.

**3.8.4.5 — two kinds of grounding, and a per-channel leakage map.** Referential grounding
(word to pixels) is **untestable by construction**: neither instruction contains a colour or
spatial word (measured `content ∩ REFERENTIAL = ∅` — the whole vocabulary is
`{pick, up, the, cube, push, to, goal}`), and each scene holds one object. Only action-mode
grounding (verb to behaviour pattern) is testable. Mechanistically, the concatenation fusion
planned for 3.8.5 **cannot** do referential grounding: a per-episode language constant never
indexes into visual features; binding a word to a region needs cross-attention with `Q` from
language and `K`, `V` from vision.

Two measurements corrected earlier narration:

- **The task difference is localised in the gripper.** Per-channel cross-task versus
  within-task `|Δa|`: the seven arm channels sit at **0.72–1.12x**, the gripper at **7.00x**.
  3.8.4.3's "actions differ by 3.14x" is numerically right but gripper-driven; the corrected
  statement is "the two tasks differ in the gripper's behaviour while the arm trajectories are
  statistically similar". A `<补正 3.8.4.5>` note was added at 3.8.4.3 stating this, and the
  correction narrows what the language test can prove — a model can pass via
  "instruction to gripper schedule" alone.
- **The per-frame leakage map.** `image` trivial statistics are at chance (best single
  brightness threshold 365/717 = **50.9%**); `proprio` finger joints pick `[0.0183, 0.0400]`
  versus push `[0.0000, 0.0400]` **overlap by 0.0217**, so no single frame separates them;
  only `task_goal` leaks the task completely. Therefore **dropping `task_goal` from the
  language-conditioned arms is sufficient** to make the instruction the only per-frame channel
  that distinguishes the tasks. This supersedes the vague "let language carry the goal" remedy
  from 3.8.4.3 with a single concrete field.

  > **Superseded — see the correction entry above.** Every row of this leakage map is wrong:
  > `image` is 88.7% (not 50.9%), `proprio` is 98.0% from `t=1` (not "not separable"), and
  > "dropping `task_goal` is sufficient" does not hold for an image-conditioned policy.

**Recorded boundary.** The gripper's difference is **sequence-level**: PushCube drives it fully
closed (0.0000) while PickCube stops at the cube width (0.0183). Invisible to a single-frame
policy, available to anything with history. So "3.8 does not need memory" is scoped to
*single-frame plus mode selection*; recognising a *procedural* state such as "the gripper has
closed completely" makes history necessary, which is the entry point to P1 task-state
estimation.

Durable conclusions recorded in `notes/concepts.md` under "Language Grounding".

Recurring failure mode this section exercised for the sixth time: **a legal shape or symbol is
not correct semantics** — here, "3.8 has a `language_ids` field" was not the same as "3.8 has
a language-conditioned experiment".

### 2026-09-24 — Second task collected, and the state schema now travels with the data

Lesson 3.8.4 needs language-distinguishable tasks. The plan had been to synthesize instructions
from `goal_pos`, but `motionplanning/panda/solutions/` already ships `push_cube`, `stack_cube`,
`pull_cube` and others, so a genuinely different skill is cheap: `PushCube-v1` runs the stock
planner under the same `pd_joint_pos` 8-d action contract (verified: seeds 0 and 1 both
`success=True`). Two real tasks beat two synthetic phrasings of one task, because "pick" and
"push" differ in *skill*: the same scene with a different instruction now demands a different
action, which is 3.5's criterion applied to language in its strongest form.

`scripts/generate_expert_demo.py` gained `--env-id` with a solution table, and now derives the
state field schema from the environment and writes it into the file.

**The trap this fixes.** The flatten order is task dependent:

| | `PickCube-v1` (42) | `PushCube-v1` (35) |
|---|---|---|
| qpos / qvel | `[0:9]` / `[9:18]` | `[0:9]` / `[9:18]` |
| `is_grasped` | **`[18:19]`** | — |
| `tcp_pose` | `[19:26]` | **`[18:25]`** |
| `goal_pos` | `[26:29]` | **`[25:28]`** |
| `obj_pose` | `[29:36]` | `[28:35]` |

Applying PickCube's hard-coded offsets to PushCube yields perfectly legal shapes with wrong
semantics. Measured on `episode_000000`:

| field | ground truth | via the file's schema | via hard-coded `[19:26]` / `[26:29]` |
|---|---|---|---|
| `tcp_pose` | `[0.0123, 0.038, 0.1822, -0.0177, 0.9998, 0.0043, 0.008]` | matches | `[0.038, 0.1822, -0.0177, 0.9998, 0.0043, 0.008, 0.1993]` — `goal_pos[0]` leaks in |
| `goal_pos` | `[0.1993, 0.0536, 0.001]` | matches | `[0.0536, 0.001, -0.0007]` — `obj_pose[0]` leaks in |

Extraction must therefore be schema driven, and the schema has to travel with the data instead of
being re-derived or re-guessed by every consumer. `state_fields` and `state_dim` are now root
attributes, and the collector asserts the derived schema's total equals the state width the
environment actually returns.

**Datasets.** `datasets/pushcube/expert_episodes_rgb.h5` is new: 5/5 successful PushCube episodes
with `T = 71/72/64/74/66`, a 35-d state, the same `images (T+1, 128, 128, 3) uint8` contract, and
1.78 MB on disk against 17.3 MB raw. PickCube was re-collected with the schema attribute and is
still **bit-exact** against `expert_episodes.h5` in all five episodes. Together: 10 episodes, two
tasks, about 3.7 MB. `PushCube-v1`'s `evaluate()` has no `is_obj_placed` or `is_robot_static`, and
the collector records them as absent rather than as `False`.

The two tasks share the 3.8.1 contract unchanged: `proprio[25]` (qpos + qvel + tcp_pose) and
`task_goal[3]` exist in both, and only `is_grasped` differs, which the contract already excluded.
So the 3.8.1 field contract generalises across tasks without modification — but its **slice
constants do not**, which is exactly what the schema attribute is for.

### 2026-09-24 — 3.8.3 observability: three measurements, two of which corrected my own narration

`notebooks/3.8_multimodal_policy.ipynb` grew from 12 to 17 cells (11 markdown, 6 code, all
executed, zero errors) with a 3.8.3 section that turns the deployability table from a
judgement into measurements. The criterion is 3.5's: partial observability means two hidden
states producing the same observation while requiring different actions.

**M1/M2 — cube pixel footprint and visibility, over all 365 frames.** Depth is unprojected
to world-frame surface points, and a pixel counts as "cube" when its unprojected point falls
inside the cube's 4 cm AABB. That uses only well-defined quantities (depth, camera
intrinsics/extrinsics, the cube's true pose) and **no colour threshold** — the colour route
had already failed once, with a red-brown table capturing 65% of pixels.

| quantity | value |
|---|---|
| cube pixels over 365 frames | min 4, median 9, max 21 |
| as a fraction of a 128x128 frame | 0.024% .. 0.128% |
| frames with the cube invisible | **0 / 365**, none below 4 px |

Two conclusions. `obj_pose` is vision-estimable but the raw material is tiny. And the
hypothesis that the gripper occludes the cube during the approach — which would have made
`obj_pose` a memory problem and tied 3.8.3 back to 3.6 — is **refuted** on this data: the
cube is visible in every frame. Memory may still be needed for other reasons (velocity,
task progress), but not for object permanence here.

**M3 — does the image carry the goal?** Threshold fixed before looking at results
(`mean |dpixel| < 10`), as 3.7's self-check question 6 requires.

| quantity | value |
|---|---|
| sampled frame pairs, cross-episode at the same t | 100 |
| mean abs pixel difference | 1.53 .. 7.71 (all 100 below the threshold) |
| goal distance in those pairs | 0.0571 .. 0.2927 m = **2.3x .. 11.7x** the 0.025 m success tolerance |
| corr(mean abs pixel diff, goal distance) | **-0.295** |
| image change across episodes at fixed t | median 5.89 |
| image change within one episode over 25 steps | median 5.96 |
| ratio | **1.0x** |

The goal is bit-exactly fixed within every episode (asserted). So pixel variation shows no
positive relation to the goal, and cross-episode variation is no larger than ordinary motion
variation. This is **corroboration, not proof**: the proof that the goal is invisible is the
source (`goal_site` is in `_hidden_objects`); M3 reaches the same conclusion independently
from the data, but correlation cannot exclude a weak hidden signal.

**Two corrections to my own narration, both caught by running the code.**

1. The first printed interpretation claimed the goal signal is "buried by an order of
   magnitude" under motion. The measured ratio is **1.0x**, not 10x. The conclusion survives
   but the reason changes: not "the signal is drowned" but "pixel change is unrelated to the
   goal".
2. The first pinhole sanity check used the norm of the cube's world position (0.057 m) as
   the distance and printed **44.7 px**, contradicting the 3x3 footprint that was just
   measured. The correct quantity is the camera-frame z along the optical axis (0.626 m),
   giving **4.1 px**, which matches. Both the mistake and the fix are recorded in the
   notebook rather than quietly removed.

One number for 3.8.5: 25 steps of robot motion change the average pixel by only **5.96/255**.
The scene is visually quiet and the informative pixels are a small fraction, so any "image
helps" result in 3.8.6 would hold on a near-constant visual distribution.

**3.8.3 status: the deliverable is met.** The four-way classification now carries a measured
column: proprio is not vision-dependent; `goal_pos` has no image signal and must come from
`task_goal` or the instruction; `obj_pose` is always visible but only 4-21 px;
`is_grasped` remains unmeasured because no gripper-width or tactile channel exists in this
data; `tcp_to_obj` and `obj_to_goal` are derived and inherit the errors above.

### 2026-09-24 — RGB expert collection, and an action-replay conflict with 2.9

Lesson 3.8 needs `(image, language, state) -> action chunk`, and the repository had no
images at all: all three HDF5 files store only `actions` / `observations` / `rewards`
(plus scalars in the smoke file), because `scripts/generate_expert_demo.py` hard-coded
`OBS_MODE = "state"`. The images were never requested, not requested and then dropped.

**Rendering works without CUDA.** `obs_mode="rgb"` returns
`sensor_data.base_camera.rgb (1, 128, 128, 3) uint8` on this machine; `rgbd` adds
`depth int16` and `sensor_data` adds `PositionSegmentation`. There is one camera
(`base_camera`) at a default 128x128. So 3.8 can be an experiment rather than a
design-only lesson.

**The goal is not observable in the image — verified in the source, not by eye.**
`mani_skill/envs/tasks/tabletop/pick_cube.py:95-104` builds `goal_site` and then appends
it to `self._hidden_objects`; `push_cube.py:137` documents that list as "hide some Actors
from view", and `sapien_env.py:1356` notes that such objects are shown only in the GUI
viewer. The task docstring at `pick_cube.py:25` nevertheless says "marked by a green
sphere": the documentation is wrong and the code is right. Projecting the goal's 3D
position into the frame puts it at pixel `(63.7, 61.1)`, where there is nothing but the
robot's own body, so at this camera angle it would be occluded even if it were rendered.
Consequence for 3.8: PickCube's success test is `||goal - cube|| <= 0.025`, so removing
`goal_pos` from the state to buy deployability makes the task unsolvable from
`(image, language, proprio)` alone — and no loss curve would show it.

**`build42` is bit-exact, and rendering does not perturb physics.** A visual `obs_mode`
does not return the 42-d state, and `PickCube._get_obs_extra` only adds `obj_pose` /
`tcp_to_obj_pos` / `obj_to_goal_pos` when `"state" in self.obs_mode`, so the state must be
rebuilt from the same primitives in the verified block order. Stepping a `state` env and
an `rgb` env with identical actions over a whole episode (74 steps) gave
`build42(rgb_env)` vs `obs["state"]` a maximum difference of `0.000e+00`. That validates
the rebuild and shows the renderer does not perturb the simulation.

**Collection.** `scripts/generate_expert_demo.py` gained `--obs-mode` and `--compare`,
image recording, and the `build42` rebuild (14 edits; backup at
`/tmp/generate_expert_demo.py.bak`). Running it with `--obs-mode rgb` produced
`datasets/pickcube/expert_episodes_rgb.h5`: 5/5 successful episodes with
`T = 74/74/50/86/76`, identical to the state-only file; `images (T+1, 128, 128, 3)` uint8
aligned index-for-index with `observations`; 17.9 MB raw and **1.89 MB on disk** (gzip).
The source file was opened read-only and is unchanged, and `datasets/*` plus `*.h5` are
in `.gitignore`, so the payload is not committed.

**The reproducibility pair that locates the conflict.**

| Method | Result |
|---|---|
| planner re-run, fresh process, `--obs-mode rgb` | **bit-exact in all 5 episodes** against the state-only file (`max = 0.000e+00`) |
| action replay of the stored actions, fresh env | episode 0 exact; 1, 2, 4 drift; `is_grasped` differs in 4 of 5 |

Per-block action-replay difference (state mode, `pd_joint_pos`):

| episode | qpos | qvel | is_grasped | obj_pose | steps differing |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 / 75 |
| 1 | 5.18e-05 | 7.90e-03 | **1.0 (1 step)** | 4.41e-03 | 33 / 75 |
| 2 | 7.33e-05 | 8.32e-03 | **1.0 (1 step)** | 2.60e-03 | 8 / 51 |
| 3 | 0 | 0 | **1.0 (1 step)** | 0 | 0 / 87 |
| 4 | 2.51e-05 | 8.59e-03 | **1.0 (1 step)** | 4.81e-03 | 6 / 77 |

Two controls rule out the obvious explanations: `goal_pos` is bit-exact everywhere (it is
static within an episode), and `obs["state"]` equals the `build42` rebuild bit-exactly,
so the accessor path is not the cause. Episode 3 isolates the phenomenon: every
continuous block reproduces exactly while `is_grasped` still differs at one step, so the
flip is not a consequence of trajectory drift — that boolean sits on a threshold at that
step.

So the 2.9 claim that holds is **planner determinism**; the one that does not is that
**action replay** reproduces the stored observations exactly, and the "one-frame
`is_grasped` flip" dismissed there is reproducible. The 2.9 bullet now carries an
`[open]` marker; this entry states the conflict rather than adjudicating the past, since
the original measurement may have used a different comparison path.

**Language has no home in the data yet.** The only instruction string in the repository
is in the LeRobot metadata (`tasks.parquet`: `pick up the cube`). The RGB HDF5 carries
root attr `task = "PickCube-v1"` — an environment id, not an instruction — and no
per-episode instruction field. 3.8.4's experiment, comparing a fixed instruction against
several, therefore needs an instruction field added before it can mean anything.

**Notebook 3.8.1 / 3.8.2.** `notebooks/3.8_multimodal_policy.ipynb` (11 cells: 7 markdown,
4 code, all executed, zero errors) turns the above into a checkable contract. It asserts
the dataset schema, builds one real sample and prints every field, displays the image, and
checks index-level alignment across all five episodes. The contract is
`image [3,128,128] uint8` + `language_ids [6] int64` + `proprio [25] float32` +
`task_goal [3] float32` -> `action_chunk [8,8] float32`, with the action side frozen at
3.7's shape so that 3.8.6 can attribute any difference to the modalities rather than to the
action representation.

Two by-products worth keeping. The `language` field is a whitespace-tokenizer placeholder
(`pick up the cube` -> 6 ids, `[1, 3, 4, 5, 6, 2]`) because the repository has no
tokenizer; the point of the field now is to have a definite shape, not a good tokenizer.
And the notebook records that alignment is **index-based, not time-based**: there is no
per-frame timestamp anywhere in the data, so the existing 20 Hz-versus-declared-50-fps
defect becomes a real risk as soon as images and actions must be paired on a real robot
with a camera whose rate differs from the control rate.

**A correction found while reviewing the learner's self-check answers.** The
`if "state" in self.obs_mode` branch in `PickCube._get_obs_extra` implies that `obs_mode`
can be *combined*. It can: `obs_mode="rgb+state"` returns both `obs["state"]` (the same
42-d vector) and `obs["sensor_data"][base_camera]["rgb"]` **from the same `env.step`**
(`parse_obs_mode_to_struct`, `sapien_env.py:296`). The `build42` rebuild written earlier
this session was therefore unnecessary. It was deleted from the collector, together with
the metadata string that had recorded the rebuild, and the collection was re-run with
`--obs-mode rgb+state`. The resulting file is **byte-identical in `images`** to the
`build42` version and still bit-exact in `observations` and `actions` against the
state-only expert file, so the simplification is proven equivalent rather than merely
also-correct.


### 2026-09-24 — 3.7 follow-ups: the H question is closed, and the K-clipping mechanism is measured

Two gaps left open by the 3.7 build were closed in the same notebook, which grew from 24 to
31 cells (20 markdown, 11 code, all executed, zero errors).

**S9 — an H sweep with the sample set held fixed.** The natural suspicion after "useful
horizon = 0" was that H=8 was simply the wrong H. The sweep holds sample set and split
constant by making each H's target the *prefix* of the H=8 target (`Y[:, :H, :]`), so only
the target length varies. `H=1` reproduces B1 to the last digit (`0.575204`, asserted),
which validates the two independent code paths.

| H | params | overall val MSE | mse @ h=0 | S_0 | useful horizon |
|---|---|---|---|---|---|
| 1 | 23,048 | 0.575204 | 0.575204 | -0.0315 | 0 |
| 2 | 24,080 | 0.602890 | 0.638865 | -0.1456 | 0 |
| 4 | 26,144 | 0.613326 | 0.451150 | **+0.1910** | **1** |
| 8 | 30,272 | 0.853913 | 0.571828 | -0.0254 | 0 |

`S_0` is **non-monotonic**: -0.03, -0.15, +0.19, -0.03. A systematic H effect would vary
smoothly; a swing of 0.34 between adjacent H values, on one trajectory and 69 heavily
overlapping validation samples, is what noise looks like. The single positive cell is one of
a 4x8 grid of comparisons, and H=4 is negative at h=1,2,3, so selecting on that cell would be
a multiple-comparison error. `overall val MSE` does rise monotonically with H, but that is
the target getting harder, not the model getting worse. Conclusion: the h=0 failure is
**H-independent** (H=1 fails too), so **H comes off the suspect list** and the constraint
stays where every other 3.7 result put it — data coverage.

**S10 — the K-clipping hypothesis was tested, refuted, and replaced by a measured
mechanism.** The hypothesis on record was that larger K executes later, less extreme chunk
elements. It is half-testable offline and it fails: the model's predicted magnitude
*increases* with h (0.826 -> 0.954 in normalised space) and so does the target
(0.589 -> 0.699). The same measurement exposed a separate model defect: `pred/target` is
about **1.4 at every h**, i.e. the policy systematically amplifies action magnitude by
roughly 40%, consistent with its heavy clipping.

The rollout call was then instrumented to record, per executed step, its chunk position and
the magnitude of the raw pre-clip command:

| K | success | clip frac | mean abs(a) | mean abs(da) | replans |
|---|---|---|---|---|---|
| 1 | 0/5 | 0.947 | **3.224** | **1.5156** | 200 |
| 2 | 0/5 | 0.370 | 1.365 | 0.4109 | 100 |
| 4 | 0/5 | 0.200 | 0.799 | 0.0346 | 50 |
| 8 | 0/5 | 0.121 | 0.786 | 0.0367 | 25 |

Reference values already in the repository (2.9): expert `mean abs(da) = 0.0077`, random
fixture `0.6723`. Broken down by `(K, h)`, with identical weights and identical chunk
positions: at `K=8` the magnitude is flat across h (h=0 0.78, h=7 0.77), but **the same
h=0** costs 3.22 at K=1 against 0.78 at K=8 — a factor of four, with only the visited state
distribution changing.

Reading: extreme actions are a property of the **closed-loop state distribution that K=1
generates**, not of the chunk position. At K=1 the executed actions jump by `1.5156`
step-to-step, worse than the random-action fixture, so the robot is driven off-distribution
and even its h=0 prediction extrapolates badly. At K>=4 the within-chunk actions are
internally consistent (`0.035`, the same order as the expert) and the states evolve calmly.
What chunking actually buys in this measurement is **temporal consistency**, not a longer
look-ahead — one of the standard arguments for chunking, obtained here by accident, while
success remains 0/5.

Recorded boundary: K=1 and K=8 visit different states, so the factor of four mixes
state-distribution shift with within-chunk consistency. Separating them needs a different
measurement — comparing within-chunk prediction smoothness against the jitter of re-planned
h=0 predictions on the recorded state sequence — which this lesson does not run.

Two self-check questions (9 and 10) were added for these sections and are marked pending in
the answers section. `notes/concepts.md` gained both conclusions under `## Action Chunking`.
No gate item moved; the closed-loop result is unchanged at 0/5 for every K.

### 2026-09-24 — Notebook 3.7 (Action Chunking) built out and executed to a negative result

`notebooks/3.7_Action_Chunk.ipynb` was a 12-cell draft (4 markdown, 8 code, all executed)
with two competing sample-construction paths. It is now 24 cells (15 markdown, 9 code),
all executed with zero error outputs, markdown interleaved with code, and the measured
results written back into the narrative. The dtype/renaming/scheduling of the notebook
itself is `[verified]`: `nbformat.validate` passes and every code cell carries an
`execution_count`.

**Four defects in the draft, each verified before it was fixed**

1. Two contradictory definitions of "a training sample" coexisted: one cell built a
   `Dataset` with `H=4` plus zero-padding and a mask, another used `H=8` with the tail
   dropped. Both were live, `H` was assigned three times, the mask never reached the
   loss, and the padded `train_loader` was dead code.
2. The split did not match Lesson 2. `default_rng(42).permutation(5)` is `[4 2 3 1 0]`;
   the draft took `order[-1]` (episode 0) as held-out, whereas Lesson 2 recorded
   episode 4, i.e. `order[0]`. One index differs; the entire held-out trajectory does.
3. `max_steps=200` never took effect: the draft's own recorded output reads
   `{'seed': 100, 'K': 1, 'success': False, 'steps': 50, ...}`. The gymnasium default
   `TimeLimitWrapper` caps at 50, the confound Lesson 2 had already ruled out.
4. The single-step baseline was invalid. `ChunkMLP` returns `[B, 1, 8]` even when
   `horizon=1`, while the single-step loader supplied `[B, 8]`; `F.mse_loss` broadcasts
   those into a `[B, B, 8]` tensor, so B1/B2 were trained on a garbage loss and showed
   the tell-tale flat train loss. Fixed by keeping the horizon dimension (`Y[:, 0:1, :]`)
   and asserting `pred.shape == yb.shape` before every loss.

**Decisions taken by the learner before the rewrite**

Drop-tail sample construction (single `H=8`, no padding branch); acceptance = concepts +
correct samples + per-horizon diagnostic + controlled single-step baseline, with closed
loop *reported but not gating*; sweep `K in {1,2,4,8}`; adopt Lesson 2's split; write the
plan before touching the notebook.

**What the executed notebook measures** (`[verified]`, all recorded as cell outputs)

| Quantity | Result |
|---|---|
| chunk samples, `H=8`, drop-tail | `325` (from 360 pairs); per episode `67/67/43/79/69` |
| naive concat-then-slice counter-example | `353` samples, of which `28` span two episodes; both are valid arrays of the same rank |
| train / val split | episodes 0-3 / episode 4; `(256, 42)` train chunks, `(69, 8, 8)` val chunks |
| parameters | `H=1,hidden=128` 23,048; `H=8,hidden=128` 30,272 (**+31.3%**); `H=1,hidden=150` 30,308 (matched) |
| `baseline_h` (val, normalised space) | 0.5576 (h=0) rising monotonically to 0.8386 (h=7) |
| mean-action baseline @ h=0 | **0.557645** |
| B1 single-step hidden=128 | 0.575204 @ h=0 — **worse than the baseline** |
| B2 single-step hidden=150 (matched) | **0.427904** @ h=0 — the only model better than the baseline |
| chunked `H=8` | 0.571828 @ h=0 (**≈ B1**), 0.853913 overall |
| per-horizon | worse than its own baseline at **every** h; **useful horizon = 0** |
| identity `mean(mse_per_horizon) == best val loss` | asserted true (0.853913) |
| K sweep, seeds 100-104 | success **0/5 at every K**; steps 200 everywhere; replans 200/100/50/25; clipped 0.947 / 0.370 / 0.200 / 0.121 |

**Reading.** The offline half is a clean negative result: the multi-step objective neither
helped nor hurt the one-step prediction (chunked 0.5718 vs B1 0.5752), useful horizon is 0,
and the only model above the mean-action baseline is the one with more parameters —
which is capacity, not chunking, and rests on 69 samples from a single trajectory. The
closed loop is 0/5 for every `K`, so with 5 seeds the one-sided 95% upper bound on success
is still about 45%: this **cannot** separate "chunking does not help" from "five episodes
are not enough for any policy". What the lesson does establish is the **measurement
apparatus** — correct episode-local samples, a controlled and parameter-matched comparison,
an asserted identity, and an explicit step cap.

The one unexpected but real observation: **clipping falls monotonically as `K` grows**
(0.947 -> 0.121) while success stays 0. A plausible explanation — a re-planning policy
re-samples its own drifted distribution and emits extreme actions, while larger `K`
executes the chunk's later, closer-to-mean predictions — is recorded in the notebook as a
**hypothesis**, not a finding. Distinguishing it needs the executed-action distribution
per `K`, which this lesson does not record.

**Not established:** nothing about `H` other than `H=8` (sweeping `H` needs retraining);
nothing about temporal ensembling (not implemented); nothing about generalisation (one
held-out trajectory, 69 chunk samples). No gate item moved.

`notes/concepts.md` gained `## Action Chunking`, recording the `L/H/K` split, the sample
count `sum(T_i) - N(H-1)`, why the comparison must happen at `h=0`, why `baseline_h` is not
constant, and the `F.mse_loss` broadcast trap.

Backups of the pre-change and pre-execution notebook were kept outside the repository
(`/tmp/3.7_before_教案.ipynb`, `/tmp/3.7_unexecuted.ipynb`). Two markdown-only safety rules
from the 3.6 session were honoured: the file's `mtime` was re-read immediately before every
write, and no whole-file rewrite happened without that check.

### 2026-09-23 — Notebook 3.6 (history representation) organized; one cell overwritten and restored by hand

`notebooks/3.6_History Representation.ipynb` was an eight-cell pure-markdown outline
that the learner had staged as a one-cell skeleton and then filled in. It was renamed
to `notebooks/3.6_history_representation.ipynb` (`git mv`, staging preserved) to match
the `lesson_substep_topic.ipynb` convention the other Lesson 3 notebooks use, and its
markdown was reorganized into 13 cells. There are no code cells, so no outputs or
`execution_count` values were involved; `nbformat.validate` passes.

**An overwrite happened and is recorded rather than smoothed over.** The whole-file
rewrite was done without re-checking the file's modification time first, and the
learner saved a cell during the operation:

| Evidence | Value |
|---|---|
| file at 17:11, before any edit | `5437` bytes |
| structure dump and full dump | `8` markdown cells |
| those 8 cells reconstructed verbatim | `4988` bytes / `184` lines — **`449` bytes short** of the file |
| `nbformat.read` immediately before the rewrite | `9` markdown cells |
| `git mv` / rewrite | 17:12:51 / 17:13:07 |

One markdown cell was therefore written from outside in that window and the rewrite
replaced it. Recovery was attempted and failed: there is no `.ipynb_checkpoints`
anywhere under `~` (six levels); the git index holds only the original one-cell
skeleton (`615` bytes, a single empty **code** cell); and `git fsck --unreachable`
yields only unrelated blobs (README, a script, `AGENTS.md`) plus 2026-09-21 stash
commits whose trees point at that same one-cell blob. The learner restored the cell by
hand from the live JupyterLab buffer.

The restored content is the closing chain of the "History / Memory Mechanism 统一对比"
table, which the learner had changed from plain text into a boxed formula; it now sits
at the end of §5 verbatim as given:

```text
\boxed{ \text{Fixed Window} \rightarrow \text{Compressed Memory}
        \rightarrow \text{Gated Memory} \rightarrow \text{Attention Retrieval} }
```

Residual gap `[closed 2026-09-23 by learner review]`: that change accounts for
roughly `+89` bytes, so about `360` bytes of the original delta are not explained by
arithmetic. The learner has reviewed the committed notebook and confirmed its content
is complete, so those bytes are **not** treated as a missing line and this is not an
open item. The incident record above is kept deliberately as a failure mode worth not
repeating, and the rule below stands.

**Rule recorded for future notebook work: re-read `mtime` immediately before writing,
and never whole-file-rewrite a notebook that may be open in an editor.** Editing
markdown cell by cell, or aborting when `mtime` changed since the last read, would have
prevented this.

What the reorganized notebook contains (the learner's outline preserved, structure
numbered, four gaps filled):

- the four-method list is repaired from pasted tab / zero-width-space fragments into a
  table, and sorted onto the axis the lesson runs on: **Frame Stacking and Transformer
  retain history; RNN and LSTM compress it**;
- §3 states RNN's compression and explains that what LSTM fixes is the *gradient path*,
  not capacity, via `∂c_t/∂c_{t-1} = diag(f_t) + …` against RNN's
  `∂h_t/∂h_{t-1} = diag(1 − h_t²)·W`;
- the LSTM gate section had a real notation collision: the input is written `o_t` (the
  observation, this repository's convention) while the literature also writes the
  **output gate** as `o_t`. The convention is now stated once (`o_t` = observation,
  `g_t^{out}` = output gate), and the missing output-gate equation
  `g_t^{out} = σ(W_o[o_t; h_{t-1}] + b_o)` is added — the original introduced
  `o_t^{gate}` without ever giving its `σ(...)`;
- §4 separates the **score matrix** `S = QKᵀ/√d_k` from the **attention weights**
  `A = softmax(S)`. The original described `QKᵀ` with "第 i 行表示第 i 个 token 对其他
  token 的关注程度", which is the weights; the missing step is the softmax, and only `A`
  is row-stochastic. It points at `notes/concepts.md`
  ("Transformer and Attention Fundamentals") and at `scripts/attention_walkthrough.py`
  instead of re-deriving that material;
- two axes the original tables lacked are added: **KV cache** as the concrete form of
  "history retained rather than compressed" (per-step compute `O(T)`, cache memory
  `O(T)`), and **whether the time dimension parallelizes** (Frame Stacking and
  Transformer yes; RNN and LSTM no), which is the direct reason the recurrent family
  was replaced;
- §6 connects back to 3.5's diagnosis order (信息 → 表示 → 容量) and adds the detail
  that 3.5's own recurrence `m_t = update(m_{t-1}, o_t, a_{t-1})` makes history an
  **(observation, action)** question, not only an observation-window question;
- `## 小结` and five `## 自检` questions close the notebook.

Relationship to 3.5, recorded so the two are not mistaken for duplicates: 3.5 answers
*whether* history is needed (information sufficiency; its synthetic experiment floors
the single-frame MLP at `1.0027 = Var(a|o)` and takes the history MLP to `1.3e-05`),
while 3.6 answers *where history physically lives and what retrieval costs*. 3.6 has no
experiment and cites 3.5's numbers instead of repeating them.

Also edited this session: `notes/concepts.md` gained `## History Representation for
Policies` (the retain-versus-compress table, the `d_k`-independent architecture facts,
and the distinction between architectural history and agent-level memory), with a
pointer added from `Embodied Memory`. No dataset, checkpoint, script, or other
notebook was touched.

### 2026-09-23 — Attention fundamentals studied; `concepts.md` gained the section, with a new verification fixture

Conceptual study session for the P0 "deep learning and Transformer fundamentals"
gap. No dataset, notebook, checkpoint, or pipeline code was touched, and no project
status changed: this session produced one new repository artifact and one new
`concepts.md` section.

New artifact: `scripts/attention_walkthrough.py` — a standard-library-only fixture
that recomputes and **asserts** the numeric consequences of scaled dot-product
attention and multi-head attention. First run:
`python3 scripts/attention_walkthrough.py` → `23/23 checks passed`, exit `0`
(`[verified]`). It reads no dataset, no checkpoint, and no simulator, so it repeats
in any environment that has `python3`. Its purpose is to make the claims recorded
in `concepts.md` re-checkable instead of trusted.

One assertion failed on the first run and the **claim, not the code, was wrong** —
recorded because the failure is itself the lesson. The fixture asserted that the
`T=3` token whose raw scores are uniformly highest therefore has the flattest
attention row. Entropy says the opposite: that row carries the highest single
weight (`0.5035` vs `0.4011`) and the *lowest* entropy of the three. The cause is
softmax shift-invariance, `softmax(z + c) = softmax(z)`: scores are not comparable
across rows, so uniformly high scores flatten nothing. The assertion was replaced
by three correct ones (shift-invariance; uniformly high scores do not flatten a
row; the entropy ordering), all passing.

`notes/concepts.md` gained `## Transformer and Attention Fundamentals`, inserted
before `Action Policy Families` so the reading order runs fundamentals → policy
families → VLA anatomy. It records conclusions rather than derivations:

- attention **routes** while the FFN transforms — the output row is a convex
  combination of the value rows, so attention alone cannot leave the convex hull
  of `V`;
- `W_Q / W_K / W_V` separate "what I look for", "how I am found", and "what I pass
  on", with the concrete costs of merging them (`W_Q = W_K` forces undirected
  attention);
- `S` is a learned bilinear form: high score ≠ importance, and scores are not
  comparable across rows;
- `1/√d_k` is a softmax **temperature**, with the measured Jacobian
  (`1.05e−1` → `1.13e−7` for logits `±1` → `±8`) and the paired entropy comparison
  (`0.122` unscaled vs `3.785` scaled) as the evidence;
- `A` is a content-generated soft adjacency matrix, the same object as a
  hand-built temporal/spatial segmentation affinity matrix — and attention weights
  are **not** an explanation, with the exogenous-gaze versus endogenous-attention
  distinction the project's human-state work depends on;
- permutation equivariance (positional encoding is a necessity, not an
  optimization) and the `O(T²)` score-matrix cost;
- multi-head as `h` parallel routing maps merged only at `W_O`: the `Q/K/V`
  parameter budget is identical to a single head, and `W_O` is the necessary extra
  because a single head's output needs no merging;
- `rank(S) ≤ d_k`, so `d_k = 1` collapses all positions onto one preference
  ordering, and `h = d_model` is legal mathematics and a degenerate design;
- masks as a semantic decision about allowed information flow, applied **before**
  softmax, with ACT bidirectional and OpenVLA causal;
- the transferable trap: a valid formula with the right shapes is not evidence that
  the intended concept survived.

The `VLA Anatomy` section now cross-references it.

Self-check outcome, recorded as learning evidence: of three questions posed, two
were answered correctly (the `Q/K/V` parameter budget is unchanged; the heads
cannot see each other and merge only at `W_O`) and the third was answered
**incorrectly** — "`d_k = 1` can still express relevance because the math symbols
are intact". The corrected principle is that *being computable is not being
expressive*: at `d_k = 1` the score table is an outer product of rank 1, so every
position shares one preference ordering and only sharpness varies. This is the same
reasoning trap as "equal tensor shapes prove datasets are compatible", which this
repository already records.

What this session does **not** establish: nothing about the policy, the dataset, or
closed-loop behaviour. No training, rollout, conversion, or environment step was
run. The material is architecture understanding only, and it does not move any
gate item. The next step toward reading real code is to inspect ACT's transformer —
whether `W_Q` is one large matrix, what the `reshape` splits, where `W_O` sits —
together with which token group supplies `Q` versus `K/V`.

### 2026-09-23 — Notebook 3.5 (single-frame versus history policy) completed

`notebooks/3.5_single_frame_vs_history_policy.ipynb` (14 cells: 10 markdown, 4 code)
had a clean toy experiment — two trajectories whose position reaches `x_t = 0` with
opposite hidden velocity, so the expert action is `-1` for one and `+1` for the other
— but thin notes and two mangled formulas. Only markdown was edited; all four code
cells keep their source, outputs and `execution_count`.

Repaired: the broken `implicit state estimate` fragment, the pasted-flat reactive-VLA
formula `(image t ,state t ,language)`, and the comparison table, which used tabs and
could not render.

Added structure and content:

- a title, learning objectives and the roadmap mapping (3.6), plus sections 1–7 and a
  Chinese summary;
- the **POMDP framing** that the experiment is really about: hidden `s = (x, v)`,
  observation `o = x`, a history policy `π(a_t | o_{1:t})`, the belief
  `b_t(s) = P(s_t | o_{1:t})` and a memory recurrence
  `m_t = update(m_{t-1}, o_t, a_{t-1})`;
- the **conditional-mean derivation** of the failure: with `a* ∈ {-1, +1}` equally
  likely given `o = 0`, the MSE-optimal prediction is `â = E[a* | o] = 0` and the
  minimum loss is the conditional variance, `1`. The irreducible part of the error is
  therefore *missing information*, not model capacity — a directly measurable claim
  (the loss floors at 1 no matter how long it trains);
- the **mixture identity** that separates the two kinds of multimodality:
  `p(a | o) = Σ_s p(a | s) b(s)`. Action multimodality means `p(a | s)` is itself
  multi-peaked; partial observability means a unimodal `p(a | s)` is mixed by a belief
  `b(s)` that spans states requiring opposite actions. This also sharpens 3.4: no
  generative model resolves the second case — only more information (history, memory,
  or active sensing) does;
- the **finite-difference velocity estimator** `v̂ ≈ (x_t − x_{t-1})/Δt` with two
  engineering caveats: `Δt` must be known and consistent (this is exactly why the
  repository's 20 Hz / 50 Hz timestamp defect matters), and differencing amplifies
  observation noise by `1/Δt`, so a longer window is not automatically better;
- why **frame stacking plus an MLP is sufficient** once information is present (the
  layout preserves order, so the MLP can implement `x_t − x_{t-1}`), the fixed-window
  cost, and the four-way comparison table (MLP, stacked MLP, RNN/LSTM, Transformer)
  with a cost column;
- a **diagnosis order** to prevent the usual misjudgement: first ask whether the
  information exists, then whether the representation exposes it, and only then
  whether capacity or optimisation is the limit;
- an honest boundary in §7: memory addresses "the information is not in the current
  observation", **not** data scarcity or distribution shift. Lesson 2's failure was the
  latter (validation MSE `0.2350` worse than the baseline `0.1421`, `|z| = 13.21` on
  held-out states, `0/10` closed-loop success), and a history window would not have
  fixed it by itself.

Nothing in this notebook was executed; the four recorded outputs are unchanged.

### 2026-09-23 — Notebooks grouped into per-lesson folders

The curated notebooks were moved into `notebooks/lesson_0/`, `notebooks/lesson_1/`
and `notebooks/lesson_2/` with `git mv`, so history follows the files. Lessons 3.x
stay directly under `notebooks/` while that lesson is in progress.

| Folder | Contents |
|---|---|
| `notebooks/lesson_0/` | `0_environment_check.ipynb` |
| `notebooks/lesson_1/` | `1.1`, `1.2`, `1.3` |
| `notebooks/lesson_2/` | `2.1`, `2.3`, `2.4`, `2.5`, `2.6`, `2.7`, `2.8_check_data`, `2.9` |

Verified `[verified]`: eleven of the twelve moves are byte-identical renames
(compared against `HEAD`, including code sources, outputs and `execution_count`),
so no recorded evidence changed.

One notebook needed a real change. `2.8_check_data.ipynb` resolved its dataset with
a working-directory heuristic:

```python
if Path.cwd().name == "notebooks":
    project_root = Path.cwd().parent
else:
    project_root = Path.cwd()
```

Run from `notebooks/lesson_2/` that resolves the project root to the lesson folder,
so the dataset assert would fail. Its loader cell now uses the same walk-up-to-`.git`
resolver as every other notebook. The cell's recorded output and `execution_count`
are unchanged, because the resolved path is identical from either working directory.
This is the only code-cell edit in the reorganization and it is the reason the
notebook shows as a rename **plus** modification rather than a pure rename.

Also updated:

- 22 path references across `README.md`, `notes/progress.md` and
  `archive/README.md` now point at `notebooks/lesson_N/...`; a sweep for
  `notebooks/[012].` finds none left;
- `README.md` gained the new notebook tree and a paragraph recording the rule that
  notebooks resolve files by walking up to `.git` and must not reintroduce
  `Path.cwd()`-relative dataset paths.

Working directories are safe either way: tested from `notebooks/`, `lesson_0/`,
`lesson_1/` and `lesson_2/`, the resolver returns the repository root each time and
finds `datasets/pickcube/expert_episodes.h5`.

### 2026-09-23 — Notebook 3.4 (modern robot-learning policies) completed

`notebooks/3.4_modern_robot_learning_policies.ipynb` was another all-markdown outline
(15 cells, no code). It was completed with coherent numbering, repaired formatting,
and formulas; their content — BC failure recap, DAgger, ACT, Diffusion Policy, VLA,
the evolution table and the summary — is preserved.

Repaired:

- the ACT block was numbered `3.2` / `3.3` while sitting inside what is now the
  DAgger section; sections are now 1 (BC recap), 2 (DAgger, 2.1–2.5),
  3 (ACT, 3.1–3.3), 4 (Diffusion, 4.1–4.3), 5 (VLA), 6 (evolution), 小结;
- the VLA architecture diagram had lost its structure while pasting and was
  redrawn;
- the "method evolution" block used tab-separated lines, which cannot render as a
  table; it is now a markdown table with an added **"没有解决什么"** column.

Formulas added, with the two load-bearing facts checked before writing:

- **BC versus DAgger error scaling:** `J(π_θ) ≤ J(π_E) + O(εT²)` for BC and
  `O(εT)` for DAgger, i.e. the horizon dependence is the difference (Ross &
  Bagnell 2010; Ross, Gordon & Bagnell, AISTATS 2011 — confirmed by search);
- **DAgger aggregation** with distinct symbols for the two objects that the
  original text conflated: visited states `S_i = {o ~ d_{π_i}}` versus the labelled
  set `D̃_i = {(o, π_E(o)) : o ∈ S_i}`, then `D_i = D_{i-1} ∪ D̃_i`, plus the
  reminder that the new actions are the expert's `a*`, never the policy's `â`;
- **ACT:** chunk `o_t → (a_t … a_{t+k})` and inference-time temporal ensembling
  `a_t = Σ_i w_i â_{t,i} / Σ_i w_i` with `w_i = e^{-m i}`;
- **Diffusion:** the DDPM objective
  `L = E[‖ε − ε_θ(A_t, o_t, t)‖²]` with `A_t = √(ᾱ_t) A_0 + √(1−ᾱ_t) ε`, and
  receding-horizon execution `k < H`;
- **VLA:** the two action-head families (discrete action tokens versus a continuous
  diffusion / flow-matching head such as `π0`'s action expert), with the action
  semantics caveat this repository keeps insisting on.

Teaching additions: §2.4 now connects DAgger's online-expert cost to the learner's
own teleoperation / VR-intervention line (a real connection, not decoration), the
evolution table makes explicit that each method fixes exactly one link in the
chain, and the summary names the missing layer — task state, memory and recovery —
as the project's direction rather than another policy architecture. No code cells
exist in this notebook, so nothing was executed.

### 2026-09-23 — Notebook 3.3 (open-loop loss vs closed-loop success) completed

The learner wrote `notebooks/3.3_open_loop_loss_vs_closed_loop_success.ipynb` as a
pure-markdown outline (22 cells, no code) and asked for it to be completed. It is
now 24 markdown cells in a coherent order; there are no code cells, so no
execution or output is involved.

Completed and added:

- **§1 MSE as an action-imitation metric:** the objective written with its
  distributional precondition `o_t ~ D_E`, and the distinction
  `Action imitation metric` versus `Task completion metric` stated as a
  difference in *what is measured* (a function on given inputs versus a closed-loop
  system over time).
- **§2 open-loop evaluation as teacher forcing:** flow diagram, per-step loss
  `ℓ_t = ||â_t - a_t*||²`, and the property that errors are not propagated because
  the next input always comes from the expert.
- **§3 closed-loop evaluation:** `o_{t+1} = f(o_t, π_θ(o_t))` with `o_t ~ D_π`, and
  the point that `D_π ≠ D_E` is the normal case.
- **§4 comparison table** plus the repository's own four numbers from Lesson 2 —
  validation MSE `0.2350` (worse than the mean-action baseline `0.1421`), `0.2%`
  of held-out actions out of bounds, `99.5%` of closed-loop steps clipped, `0/10`
  seeded success — making the abstract distinction concrete.
- **§5 evaluation axes:** offline metrics; online metrics with
  `SR`, completion ratio and `G = Σ γ^t r_t`; robustness along object position,
  sensor noise and disturbance; efficiency/safety (steps, clipping rate, collisions,
  intervention rate); calibration and safe refusal; and a pointer to failure
  analysis.
- **§6 success-rate statistics (new):** the binomial standard error
  `SE = sqrt(SR(1-SR)/K)` and the Clopper–Pearson one-sided bound
  `SR_upper = 1 - (1-α)^{1/K}`, which for `0/10` gives `≈26%` — verified with
  `scipy.stats.beta.ppf(0.95, 1, 10) = 0.2589`. This is why a single episode is a
  case study rather than a rate, and why 2.8.8 was extended to ten seeds; the
  evaluation protocol (steps, reset distribution, intervention, time limit) is also
  listed as something that must be reported.
- **§7 failure taxonomy** reformatted as a table with an **observable criterion**
  for each of the learner's five classes (perception, planning, control,
  distribution shift, recovery), with Lesson 2's failure identified as the
  distribution-shift class.
- **§8 ACT / Diffusion / VLA** with formulas: chunk prediction versus single-action
  prediction, `p_θ(a | o)` versus point regression, and conditioning on
  vision/language/memory — noting that each fixes a different part of the chain.

Fixed while completing: the two LaTeX fragments that had been mangled during
pasting (`a ^ t`, `o t ∼ D π`, `o t → a t`) are proper math again, and the section
order was corrected to 1 → 2 → 3 → 4 → 5 (an earlier insertion had placed the
comparison before the closed-loop section).

### 2026-09-23 — Notebook 3.2 (covariate shift / compounding error) completed with formulas

The learner wrote `notebooks/3.2_covariate_shift_and_compounding_error.ipynb` (they
also renamed it themselves to the project's `lesson_substep_topic.ipynb`
convention) and asked for the notes to be completed. Their experiment is a 2-D
tracking task: `expert_policy = goal - position`, `step = position + 0.1*action`,
a BC policy with gain `0.8`, and an inaccurate variant whose action noise is
proportional to `||position||`. Only markdown was edited; all 9 code cells keep
their source, outputs, and `execution_count` byte-for-byte (16 → 21 cells).

Added:

- **Terminology separated:** covariate shift (`d_{π_θ}(o) ≠ d_{π_E}(o)`) versus
  compounding error (error growth over the horizon) as two views of one loop.
- **Closed-loop recurrence derived from the learner's own code:**
  `e_{t+1} = (1 - 0.1k) e_t`, hence `e_t = (1 - 0.1k)^t e_0` with
  `||e_0|| = 14.142`, and a contraction table — expert `k=1 → 0.90`,
  BC `k=0.8 → 0.92`, zero gain `k=0 → 1.00`, wrong sign `k=-1 → 1.10`.
  Divergence is exactly `|1 - 0.1k| > 1`.
- **General accumulation form:** `e_T ≈ Σ (∂f/∂o)^k (∂f/∂a) ε`, i.e. the error is
  repeatedly multiplied by the state Jacobian, with the spectrum of `∂f/∂o`
  deciding bounded drift versus blow-up. This is why BC's objective (small `ε` on
  `d_{π_E}`) says nothing about the loop.
- **Measured numbers**, obtained by re-running the notebook's own definitions in a
  scratch process outside the repository: expert `||e_50|| = 0.0729` versus BC
  `||e_50|| = 0.2187` (both matching the closed form), and BC's offline action MSE
  on the **expert** state distribution `≈ 0.826` (RMS `0.909` against an initial
  action scale of `14`) — the toy analogue of Lesson 2's misleadingly small
  validation MSE.
- **An honest boundary for the bad-policy figure:** the state-dependent noise makes
  the loop wander, not explode. Over 200 rollouts the final error averages `0.303`
  (median `0.292`, p95 `0.560`), the maximum deviation from the noiseless BC path
  averages `0.373` (p95 `0.527`), and the largest error equals the initial `14.142`
  — it never becomes worse than the start, because the loop is still contractive
  (`0.92`). Divergence requires `|1 - 0.1k| > 1`; with `k = -1` the same 50 steps
  give `1660`.
- **Reproducibility caveat recorded:** the rollout cells do not seed the RNG, so
  per-figure values differ between runs; the measurements above are statistics over
  seeds, and fixing a seed (as 2.8.7 requires) is the precondition for quoting
  single numbers.
- **§6 foreshadows the fixes** with formulas: BC predicts one action, ACT predicts a
  chunk (fewer decision points), Diffusion models `p(A | o)` for multimodal
  actions, and DAgger fixes the data side. The notebook's subject maps to roadmap
  3.4 (distribution shift); DAgger is 3.5 and chunking is 3.7.

### 2026-09-23 — Notebook 3.1 reorganized around the learner's one-dimensional experiment

The learner renamed the lesson notebook to
`notebooks/3.1_imitation_learning_foundation.ipynb` and prepended their own
walkthrough of Behavior Cloning: a 1-D toy system where the expert policy is
`a = -s`, a "trained" policy is `a = -0.9s`, plus noisy and inaccurate rollout
variants, with an eight-section outline left as placeholder headings. They asked
for the notes to be organized around **their** content, with added detail and
formulas. Only markdown cells were edited; all 12 code cells keep their source,
outputs, and `execution_count` byte-for-byte.

Structure now (32 cells, 20 markdown):

- the intro states the lesson question, splits the notebook into **Part A (1-D toy
  system, derived by hand)** and **Part B (real PickCube data, must be measured)**,
  and fixes a single notation set (`o_t`, `s_t`, `a_t`, `π_E`, `π_θ`, `d_{π_E}`,
  `d_{π_θ}`);
- Part A follows the learner's own outline 1–8: supervised versus imitation
  learning, the formal BC objective, the expert dataset, training the toy policy,
  offline evaluation, closed-loop rollout, why BC fails, summary;
- the eight trailing placeholder headings were removed because every section now
  has a real heading in place, and cell 17 became the Part B divider;
- Part B keeps the learner's 3.1.1–3.1.5 text and summary.

Formulas and measurements added (all checkable against the notebook's own cells):

- empirical-risk vs imitation objective, and the i.i.d. assumption that imitation
  breaks because `o_t ~ d_{π_E}` is generated by the policy being imitated;
- the BC objective in both stochastic and deterministic-continuous form, plus why
  MSE corresponds to a fixed-variance Gaussian likelihood and cross-entropy to the
  categorical case;
- toy dynamics `s_{t+1} = s_t + a_t` with `a_t = -γ s_t`, hence
  `s_t = (1-γ)^t s_0`, with the three regimes `γ = 1` (one step), `γ = 0.9`
  (converges but lingers off the expert's states), and `|1-γ| > 1` (diverges);
- offline evaluation for the toy policy:
  `L_offline = 0.1² × 25/3 ≈ 0.083`, i.e. about 1% of the state variance — the
  number that looks good and proves nothing about closed-loop behaviour;
- the noisy rollout as an AR(1) process with stationary variance
  `σ² / (1 - (1-γ)²)`: `≈1.01σ²` at `γ = 0.9` versus `≈1.33σ²` at `γ = 0.5`, so a
  less accurate policy spends its time in a wider state distribution;
- Part B now carries the real numbers as the entry point: the `1.0x`
  consecutive-versus-random ratio on the random fixture (with the `8.8×–13.7×`
  expert-episode contrast), the `0.12165` nearest-neighbour distance after a
  `0.02012` perturbation, and Lesson 2's `0.2350` vs `0.1421` validation result
  with the `0.2%`-versus-`99.5%` clipping contrast.

Also corrected: the stale "the next step is 2.8.7" lines now point at the actual
Lesson 2 outcome, and the duplicated/typo'd heading `# 3. Formal Defination of BC`
was replaced by one formal-definition section. The `../notebooks/...` path style
introduced by an earlier edit in this file was normalized back to repository-root
relative paths.

### 2026-09-22 — Repository cleanup: build artifacts, old setup script, planner probe

Housekeeping before a break, decided item by item by the learner.

- Removed regenerable build artifacts: `scripts/__pycache__/` and
  `scripts/pipeline/__pycache__/` (`208 KiB`), including stale `.pyc` files for
  scripts that were archived earlier (`dataset_report`, `validate_maniskill_rollout`,
  `compare_random_datasets`, `test_observation_adapter`). No source file was
  touched; `__pycache__` is already in `.gitignore`.
- Deleted the session's temporary comparison snapshots outside the repository
  (`/tmp/nb_before_translation/`, `2.8_before_notes.ipynb`,
  `2.6_before_shuffle_fix.ipynb`, `expert_episodes_oldschema.h5`); the equivalence
  checks they supported are recorded in this file and in Git history.
- Normalized the local permissions of the four scripts created or rewritten this
  session from `600` to `644`. Git records `100644` for all of them either way.
- `environment/setup_linux.sh` — **deleted** (作废), because
  `environment/setup_linux_v3.sh` supersedes it completely (parameterized
  environment name, Python/PyTorch versions, pinned CUDA wheel, `set -Eeuo
  pipefail`). `README.md` now points at the v3 script, and `archive/README.md`
  records the deletion and how to recover the file from Git history.
- `scripts/test_pick_cube_expert.py` — **archived** to
  `archive/offroadmap/test_pick_cube_expert.py` and recorded in the
  `offroadmap/` table. It ran the stock `pick_cube` solver with
  `render_mode="human"`, has no `main` guard (it executes on import), and is
  superseded by `scripts/generate_expert_demo.py` plus notebooks 2.8 / 2.9.

Left in place deliberately: `datasets/` (the Lesson 3 evidence), the BC
checkpoint that the two verification scripts load, the Hugging Face cache,
`archive/`, `scripts/figures/`, `scripts/reports/`, and `.idea/`.

Verification: `git status` clean before the change, no stray `*.log`/`*.bak`/
`*.orig`/`.DS_Store` files, the two verification scripts still start after
cleanup, and no dangling references to the deleted or moved paths remain outside
the archive's own record of them.

### 2026-09-22 — Lesson 2 closed; its follow-ups abandoned; Lesson 3 is the current phase

The learner confirmed that the Lesson 2 thread is finished and that the old
follow-up plan should not be pursued: neither growing the expert dataset to 30–50
episodes and re-training the BC baseline, nor building
`scripts/inspect_robot_dataset.py`. Both are now recorded as **abandoned
decisions** rather than pending tasks. The Lesson 2 evidence and the two gate
tables stay in this file unchanged as the historical record; the BC gate is 8/8
met and the one unmet Lesson 2 gate item is explicitly waived, not deferred.

`Immediate Next Steps` was rewritten to Lesson 3 only, with a table that maps each
Lesson 3 topic to the Lesson 2 evidence that now serves as its entry point
(supervised-learning objective and its baseline failure, measured overfitting
curves, and the quantified distribution shift). No code, dataset, notebook, or
checkpoint was modified; no experiment was run.

### 2026-09-22 — Teaching Protocol added to `AGENTS.md`

The learner asked that suggestions be teaching-oriented: this is a learning
project, so a recommendation should explain what it teaches, not only what to do.
Recorded as a new `## Teaching Protocol` section in `AGENTS.md` so every future
session inherits it.

The protocol requires each proposal to state: **what it teaches**, **why now**
relative to the P0/P1/P2 priorities, **the acceptance evidence** and what that
evidence will *not* prove, **self-check questions** the learner should be able to
answer afterwards, and **the research connection** where one is real. It also
requires explaining the problem a technique solves before showing the technique,
correcting principles rather than only values, comparing candidate next steps by
learning value instead of listing them, and never ending a step at "it runs".

Consequence for this file: `Immediate Next Steps` stays a status list, while the
teaching framing for the next experiment is given in conversation and, where it
becomes durable, in `notes/concepts.md`.

### 2026-09-22 — BC gate items 5 and 7 closed by two fresh-process scripts

The expanded `AGENTS.md` adds a BC closed-loop gate. Two of its eight items were
verification work rather than modelling work, and both were closed here without
touching the training code, the checkpoint, or the dataset.

New scripts (both read `checkpoints/pickcube_bc_best.pt` from disk in a new
process; neither reloads an in-memory state dict):

- `scripts/verify_bc_checkpoint.py` — gate item 5. Rebuilds the MLP from the
  stored `observation_dim` / `action_dim` / `hidden_dim`, loads with `strict=True`,
  and asserts: `control_mode == "pd_joint_pos"`, observation dimension `42`,
  action dimension `8`, output `(1, 8) float32`, clipped action inside the live
  action bounds, normalization tensors `(1, 42)` with strictly positive std.
  It also re-derives the recorded numbers from the file alone: validation MSE
  `0.23502295` and mean-action baseline `0.14210252`, both matching the notebook
  to `1e-5`. Result: `RESULT: PASS`.
- `scripts/evaluate_bc_closed_loop.py` — gate item 7. Runs closed-loop rollouts
  under the training contract, asserts the effective `TimeLimit` is 200 from the
  wrapper (so truncation cannot explain the result), and reports per-seed
  success, reward, clipped fraction, first zero-reward step, and termination
  reason, plus a JSON report.

Result over seeds `100`–`109` (`[verified]`):

| Metric | Value |
|---|---|
| Success rate | **0/10 = 0.0%** |
| Steps / termination | 200 / `truncated` for every seed |
| Total reward | mean `0.4814`, max `0.9333` |
| Clipped fraction | mean `0.995` (min `0.995`, max `0.995`) |

Report: `scripts/reports/bc_closed_loop_eval.json`.

One contrast this produces, which is now part of the failure analysis: the raw
policy output is out of the control bounds for only `1/608` values (`0.2%`) when
evaluated on the **held-out episode's own states**, but the clipped fraction
during closed-loop rollout is `0.995`. The model is acceptable on the expert state
distribution and degrades immediately off it — a direct measurement of
compounding distribution shift.

Also updated: the BC gate table (now **8 of 8 met**), the 2.8.8 evidence table, and
`Immediate Next Steps` (the two closed items were removed and the list
renumbered). The Lesson 2 gate still lacks `scripts/inspect_robot_dataset.py`.

### 2026-09-22 — `AGENTS.md` replaced with the expanded research-guidance revision

The learner supplied a new `AGENTS.md` and asked for it to replace the previous
one. It was copied byte-for-byte into the repository root; the file hash matches
the attachment (`sha256:b704ddfd1507944d44f92f245ead6d192f89a2a41869592be6131275d3f61272`).

What the new revision adds (three new sections, ~122 added lines):

- `Learning and Research Priorities` — P0 close the policy loop (train/validation,
  closed loop, distribution shift, DL/Transformer foundations, action-policy
  families, VLA anatomy), P1 procedural task intelligence (task representation,
  task graph versus task state, partial observability, memory, task-centric world
  model), P2 collaboration (planning, recovery, calibrated intervention, shared
  autonomy), and an explicit deprioritized list (ROS 2, SLAM, large-scale RL,
  Isaac Lab, low-level control, complex motion planning, Sim2Real).
- `Current BC Closed-Loop Gate` — eight evidence requirements for the baseline BC
  artifact, plus a long-term research-direction paragraph in the mission section
  and additional working rules (observation vs latent state vs belief/task state;
  static task graph vs online task-state estimation; trace the full information
  path; concept-driven reproduction; no forced XR analogies).
- `Research Vocabulary and Framing` — preferred umbrella term, core technical
  identity, core agent state, core memories, and the task-state-transition world
  model starting point.

Consequence recorded in this file: the new BC gate was assessed item by item
(6 of 8 met). Two verification items are outstanding — a fresh-process checkpoint
load with action-contract assertions (item 5) and a multi-seed success rate
(item 7) — and the separate Lesson 2 gate still lacks
`scripts/inspect_robot_dataset.py`. Both gate tables are now in this file, and the
two cheap items were added to `Immediate Next Steps`.

### 2026-09-22 — Lesson 2 closed: 2.8.7 and 2.8.8 verified as a negative result

The learner completed 2.8.7 and 2.8.8 in `notebooks/lesson_2/2.8_check_data.ipynb` and asked
for the result to be recorded without treating it as a training-code failure. It is
not one: the code did exactly what the lesson needs.

Verified by reading the executed cells (all 71 code cells ran, zero error outputs):

- 2.8.7 split: **by episode** with `np.random.default_rng(seed=42)`; training
  `episode_000000`–`000003` (284 samples), validation `episode_000004` (76);
  normalization statistics from the training split only; `train_loader`
  `shuffle=True`, `validation_loader` `shuffle=False`; the loader asserts
  `len(observations) == len(actions) + 1`.
- 2.8.7 result: training loss `1.3136 → 0.006`, validation `0.75`–`0.77` with a
  best of `0.2350` at epoch 3, early stopping at epoch 40; the best model is
  **worse than the mean-action baseline** (`0.1421`) by `0.0929`. Held-out states
  reach `|z| = 13.21` on a dimension whose training std is `0.0088`.
- 2.8.8 rollout: best checkpoint, `model.eval()`, training normalization reused,
  actions clipped to `env.action_space` with per-step clipping counts, unseen
  `seed=100`. The default `TimeLimit` caps episodes at 50 steps, so the wrapper's
  `_max_episode_steps` was overridden to 200 to rule the time limit out. Both runs
  report `Success: False`; the 200-step run earns `0.0` reward from about step 20
  and is clipped on every step after the first.
- 2.8.8 clipping per channel: `joint_2 / joint_4 / joint_6 / joint_7 / gripper`
  `199/200`, `joint_1` `149/200`, `joint_3` `129/200`, `joint_5` `101/200`.

Recorded interpretation: the model is a valid **failure baseline**, not a usable
policy, and the training code is correct. Each behaviour is intended — loss falls,
validation exposes overfitting, early stopping keeps the best epoch, the baseline
comparison shows no generalization, clipping keeps outputs inside the control
bounds, and the 50-versus-200 comparison excludes the time limit.

Documentation changes:

- `Current Position` now states Lesson 2 is complete, with 2.8.7 and 2.8.8 marked
  `Complete [verified]`, and the formal record quoted in full;
- new evidence blocks `2.8.7 — Train/validation split (episode-level)` and
  `2.8.8 — Closed-loop deployment and rollout` carry the tables above;
- the acceptance gate rows for train/validation and closed-loop execution changed
  from `FAIL` to `PASS [verified]`, with the negative result stated explicitly;
- `Immediate Next Steps` was replaced: the next phase is **data scaling**
  (30–50 expert episodes, same episode-level split) plus Lesson 3, entering from
  this measured result rather than a toy example. Repairing the training code is
  explicitly **not** the task.
- One gate item remains unmet and is recorded as such: the reusable
  `scripts/inspect_robot_dataset.py` does not exist. Lesson 2 is declared complete
  by the learner; the item is either to be built or explicitly waived.

Not affected: no dataset, script, or notebook cell was modified in this session.
The rollout numbers come from the notebook's own recorded outputs.

### 2026-09-22 — Condensed landscape appendix added; README aligned with the new direction

Added the shorter reading list that was offered with the research-direction merge,
and brought `README.md` in line with the current concepts and repository state.

`notes/concepts.md`, new `## Landscape Reading List (Condensed)` — nine directions,
three to five systems each, one conclusion each, no benchmark numbers:

1. language grounding and the cost of structural shortcuts; 2. mapping, modularity,
scaling; 3. imitation learning and generative action; 4. generalist policies and
cross-embodiment transfer; 5. prediction as a policy ingredient; 6. sim-to-real
randomization versus adaptation; 7. long-horizon, human-robot, multi-agent
evaluation; 8. task structure and task state from human activity; 9. data engines
and infrastructure. Venues and years are labelled as reading pointers rather than
audited citations.

Two references were checked on the web before being written down
(`[verified]` for existence and framing, not for results):

- Differentiable Task Graph Learning: Procedural Activity Representation and
  Online Mistake Detection from Egocentric Videos, NeurIPS 2024 —
  <https://neurips.cc/virtual/2024/poster/96827>
- `π0`: A Vision-Language-Action Flow Model for General Robot Control —
  <https://ar5iv.labs.arxiv.org/html/2410.24164>

`README.md` updates:

- principle 3 now says re-entering steps that a strictly one-directional plan
  cannot express;
- the pipeline diagram became `Task segmentation + prerequisite relations`, and
  recovery now loops back into segmentation as well as into the dataset;
- goals gained the action-policy progression (BC → ACT → Diffusion → flow
  matching), VLA anatomy, action representation as a compatibility question, and
  task-state tracking that includes retry/recovery/re-entry;
- the status line records the canonical expert schema, the retargeting, the 5/5
  replay result, the Chinese notebook convention, and that the closed loop comes
  before any task-intelligence layer;
- the progress table gained an "Expert data contract" row and refreshed the
  replay and train/validation rows (episode-level split; the leakage measurement
  holds for smooth trajectories, not for the random fixture);
- the roadmap checklist item became "explicit task-state tracking over
  prerequisite relations (retry, recovery, and re-entry must be representable)";
- a new `## Research Direction` section states the target layer, the layered
  architecture, the positioning sentence, the four-step study order, the
  cycles/re-entry requirement, and the deliberately out-of-scope substrate.

Nothing was changed in `docs/roadmap_v3.md` beyond the earlier acyclic-wording
fix.

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

`notebooks/lesson_2/2.8_check_data.ipynb` gained three executed cells while this session
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

`notebooks/lesson_2/2.9_expert_demonstrations.ipynb` gained three cells while this session
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

- `notebooks/lesson_2/2.6_dataset_dataloader.ipynb`: the loader is now
  `inspection_loader` with `shuffle=False`, and the markdown states explicitly
  that it exists for reproducible inspection and preserves trajectory order, that
  shuffling is a training-time choice, and that frame-level shuffling must not be
  reused once temporal windows or multiple episodes exist. Re-executed: all 10
  code cells run with zero errors, and the printed batch shows
  `frame_index [0..7]` with `timestamp 0.00 … 0.14`.
- `notebooks/lesson_2/2.7_bc_training_loop.ipynb` already defines its own
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
- `notebooks/lesson_2/2.9_expert_demonstrations.ipynb` corrected and re-executed:
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
  `notebooks/lesson_2/2.7_2.8_bc_training_pipeline.ipynb` (40 cells at the time; renamed
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

**Moved into `notebooks/lesson_2/2.4_observation_schema.ipynb`** as executable cells, then
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
  `notebooks/lesson_1/1.1_state_and_observation.ipynb`.
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
