# Embodied AI Concepts and Decisions

This file stores durable concepts already established during the learning
project. It is not a glossary of every robotics term; it records distinctions
that affect implementation and dataset validity.

## System Roles

### Policy

A policy maps the current observation or state to an action:

```text
observation/state → action
```

Diffusion, Transformer, Flow Matching, and autoregressive decoding are possible
policy parameterizations. They do not change the functional role of the model.

### World Model

A world model predicts a future state, observation, or latent state conditioned
on the present and usually an action:

```text
(observation/state, action) → predicted future
```

A planner may use a world model to evaluate candidate future action sequences.

This project distinguishes two related levels:

- **Physical world model** — predicts robot/object state, observation, contact,
  or latent dynamics: `p(s_{t+1} | s_t, a_t)`.
- **Task world model** — predicts how a skill changes task progress, object
  relations, prerequisites, and completion state:
  `p(z^task_{t+1} | z^task_t, skill_t, observation)`.

Predicting the next image is not the only valid world-model objective. For the
long-horizon direction of this project, task-state prediction is equally
important and is closer to task-state-aware procedural reasoning.

### Controller

A controller converts a target or action representation into lower-level robot
commands. A learned policy action is therefore not automatically equivalent to
a motor command.

## Task-Centric Agent Layers

### Task Representation

A trajectory is not only a sequence of observation-action pairs. It can also be
represented hierarchically:

```text
task → subgoal → skill / task phase → action chunk → low-level action
```

For example, a PickCube trajectory may be segmented as:

```text
Reach → Grasp → Lift → Transport → Place
```

Each segment may carry a boundary, semantic label, precondition, effect,
confidence, and failure/recovery status. A flat sequential list is sufficient
only when the order is fixed; steps that are optional, parallel, or constrained
by prerequisites need an explicit prerequisite/task-state representation instead.

That representation must tolerate **cycles and re-entry**: retry after failure,
recovery, backtracking to an earlier step, and iteration all return to steps that
were already completed. A strictly one-directional progression cannot express
them, so "what may run next" has to be recomputed from the current task state
rather than derived once from a fixed order. Prerequisite relations are therefore
relations over steps, not an ordering that is fixed in advance.

### Task State

Task state answers *where the execution actually is*, as opposed to what the task
structure allows. Each step carries a state such as:

```text
Locked | Available | Active | Completed | Failed
```

Task state is inferred, not read off: it must be estimated from observations,
simulator events, or interaction history, and it stays uncertain. The two
distinctions worth keeping separate:

- **expected** task progress (what the structure says should happen next) versus
  **observed** progress (what the evidence supports);
- **observation** versus **state** — not seeing a cup is not evidence that the
  cup does not exist.

Concrete cases an agent must handle: a step was performed but not observed; a step
was observed but failed; a step was skipped and must be re-established; an
alternative path became available. Early experiments may use simulator events or
manual annotation; later work should infer state transitions from demonstrations
and video.

### Embodied Memory

- **Working memory** stores current observations and short temporal context.
- **Episodic memory** stores prior trajectories, failures, and recoveries.
- **Semantic memory** stores object, scene, and task knowledge.
- **Procedural memory** stores reusable skills, execution procedures, and task
  knowledge.

Object permanence and re-identification are memory problems as well as
perception problems: when an object leaves the camera view, the agent should
retain identity, last-seen state, and uncertainty rather than silently treating
the object as absent.

For long-horizon assistance, **procedural and episodic memory** are the two that
matter most: procedural memory describes how the task is normally executed, while
episodic memory provides the evidence used to update task state.

```text
procedural memory → task structure
episodic memory   → current task state
task state        → next action / guidance / question
```

Memory also needs a forgetting policy: an agent that remembers everything cannot
retrieve reliably, and an agent that keeps only the last few frames cannot track
a long task.

### Task-Conditioned VLA

A reactive VLA can be written as:

```text
(image, robot state, instruction) → action chunk
```

The target architecture extends this to:

```text
(image, robot state, instruction, task state, retrieved memory, subgoal)
→ action chunk
```

The task layer chooses or verifies *what should happen next*; the policy/VLA
implements *how to act*. This is a design separation, not a claim that the two
must always be different neural networks.

### Human State and Intervention

World state, task state, and human state are distinct inputs. Gaze, location,
interaction history, hesitation, error signals, and confidence estimates can
inform an intervention policy that chooses among waiting, asking, guiding, or
taking over. Intervention quality must penalize unnecessary help, not only
missed failures.

## State and Observation

- **State** is the underlying task or system state.
- **Observation** is the information available to the policy.
- Simulation can expose exact object poses, goals, contacts, and success flags
  that may be unavailable on a real robot.
- Such values may be useful as privileged supervision or evaluation metadata,
  but a deployable policy must not silently depend on them.

The current PickCube state vector has 42 dimensions:

| Component | Dimensions |
|---|---:|
| Robot joint positions (`qpos`) | 9 |
| Robot joint velocities (`qvel`) | 9 |
| TCP pose | 7 |
| Object pose | 7 |
| Goal position | 3 |
| TCP-to-object relative position | 3 |
| Object-to-goal relative position | 3 |
| Grasp state | 1 |
| **Total** | **42** |

Object pose, goal information, relative task-state features, and grasp state
must be reviewed before being classified as deployable observations.

## Action Specification

An action tensor is not fully described by its shape. Its specification should
include:

```text
space, representation, frame, absolute-or-delta,
dimension, semantics, unit, frequency, controller
```

Two datasets with equal action dimensions can still be incompatible because of
different frames, controller modes, units, gripper conventions, or temporal
semantics.

Action dimension belongs to the **controller mode plus the code path that built
the action**, not to the trajectory. A single executor helper can emit different
dimensions for different modes: ManiSkill's motion-planning solver sends
`[qpos(7), gripper]` (8-d) under `pd_joint_pos`, but its gripper helper falls
through to a `pd_joint_pos_vel`-shaped `[qpos(7), qpos*0(7), gripper]` (15-d)
under every other mode, while its `follow_path` uses `[qpos(7), qvel(7),
gripper]` (15-d, real velocities) only under `pd_joint_pos_vel` itself. Two
consequences:

- a shape error such as `expected (1, 8), got (15,)` is a control-mode contract
  violation in the helper, not a defect in the dataset format;
- a mode mismatch can also pass the shape check silently — e.g. absolute joint
  targets fed to a delta controller have the right dimension and the wrong
  meaning, so the failure may only appear later, at a different helper call.

Assert the control mode and the action shape when collecting, and state which
mode the recorded actions were valid in.

### Retargeting between control modes

Two modes can be incompatible in semantics yet mechanically equivalent. Compare
the controller configurations before deciding what conversion must prove:
`pd_joint_pos` and `pd_joint_delta_pos` for this Panda share
`stiffness=1000`, `damping=100`, and the same gripper mapping, and differ only in
the arm's `use_delta` flag. That means "absolute target `q*`" and "delta
`dq = q* − qpos` scaled by the delta range" command **the same target**, so
re-expressing one as the other preserves the dynamics rather than merely
relabelling it:

```text
a_arm = clip((q_target - qpos_t) / delta_range, -1, 1)
```

Two rules follow. First, measure the conversion's feasibility before building it:
the expert delta here had `p99 = 0.0715` rad against a `0.1` rad range, so only
1 of 2520 channels clipped. Second, state what the result is — a **re-targeting**
of the demonstrated states, with actions the planner never issued, not a replay of
the executed trajectory — and verify it by replaying in the new mode rather than
assuming fidelity.

## Coordinate Frames

World, robot base, camera, and end-effector frames are different coordinate
systems. A pose must always be interpreted together with its source and target
frames and the transform convention used.

VR controller poses cannot be copied directly into robot joint commands. They
normally require calibration, frame transformation, scaling, workspace
mapping, retargeting, and a controller that interprets the resulting target.

## Dataset Schema Versus Semantics

A shared schema such as LeRobot standardizes storage and loading; it does not
guarantee that two datasets represent the same physical quantities. Before
concatenating Sim, VR, and Real data, verify:

1. embodiment;
2. observation space;
3. action specification;
4. coordinate-frame convention;
5. frequency and timestamps;
6. task semantics;
7. source/domain metadata;
8. deployable versus privileged fields.

## Time and Sequence Alignment

For a transition, explicitly establish whether the stored tuple is:

```text
(observation_t, action_t, observation_{t+1})
```

Some trajectory formats contain `T` actions and `T+1` states; others store one
observation per action. Conversion must not guess. Preserve source timestamps
when possible so jitter, dropped frames, resampling, and latency can be
diagnosed.

Wrapping `env.step` to record "the actions the environment actually executed" does
not by itself fix the pairing; the wrapper still chooses **which observation** it
appends:

- append the observation returned *by* that step → `observations[t] = o_{t+1}` and
  the stored pairs are `(o_{t+1}, a_t)`;
- append the observation read *before* the step, or keep a separate
  `next_observations` array → `observations[t] = o_t` and the pairs are
  `(o_t, a_t)`.

Both variants store `T` actions and `T` observations with identical shapes, so a
shape check cannot distinguish them. The post-action variant also drops the reset
observation `o_0` unless it is written explicitly, costing one usable training
pair per episode. Verify by replaying the stored actions from the recorded seed and
checking whether the pre-step or the post-step state matches the stored
observations — not by trusting a docstring.

The canonical schema for a collected episode, which makes the pairing unambiguous
and needs no reconstruction, is `T + 1` observations and `T` actions:

```python
observations = [reset_obs]                  # o_0
for action in actions:
    next_obs, reward, terminated, truncated, info = env.step(action)
    actions.append(action); rewards.append(reward)
    observations.append(next_obs)           # o_1 ... o_T

policy_obs        = observations[:-1]       # o_0 ... o_{T-1}
policy_actions    = actions                 # a_0 ... a_{T-1}
next_observations = observations[1:]        # o_1 ... o_T
```

If a file was already written in the `T`/`T` post-action form, the recoverable
view is `bc_obs = observations[:-1]` with `bc_actions = actions[1:]`, which is
`T - 1` samples and still loses `(o_0, a_0)`; regenerating from a fixed collector
is preferable to depending on that shift.

Delta actions change physical meaning when the control frequency changes.
Frequency conversion therefore requires semantic reasoning, not only array
resampling.

## Demonstrations and Evaluation

- Action replay executes stored actions from a reconstructed initial state;
  matching saved pre-action observations and rewards provides evidence that
  the source trajectory was reproduced. A seed alone is not a universal
  reproducibility guarantee across versions/backends.
- `terminated` and `truncated` describe episode boundaries, not task success.
  Inspect task-specific `info["success"]` and distinguish any-step success
  from final-step success.
- The current standardization notebook synthesizes 0.02-second timestamps,
  whereas verified replay uses 20 Hz control (0.05 seconds). Stored timestamp
  regularity does not establish physical timing correctness.

- A random rollout is useful for testing serialization and conversion.
- It is not an expert demonstration unless task success and trajectory quality
  are verified.
- Offline action prediction loss does not establish closed-loop task success.
- Policy evaluation should ultimately include success rate, robustness,
  perturbations, and a failure taxonomy.

## Train/Validation Splitting for Trajectories

Frames inside one trajectory are temporally correlated, so splitting a single
episode by frame puts near-duplicates on both sides of the split. Split by
**episode** once several trajectories exist.

Whether adjacent frames are actually near-duplicates is a measurable property of
the data, not an assumption. Compare the mean distance between consecutive
observations with the mean distance between random frame pairs:

- smooth, planner-generated trajectories: ratio roughly `9x`–`14x`, so a
  frame-level split leaks badly;
- a random-action fixture: ratio roughly `1x`, so the statistic does not
  demonstrate leakage even though the frame-level split is still not an honest
  held-out set.

Measure the statistic on the data actually loaded before using it as evidence.
Action smoothness (`mean |da|`) is the complementary indicator: expert
trajectories step by small correlated increments (measured `~0.008`), random
rollouts jump by nearly independent draws (measured `~0.67`).

## Research Direction: Task Intelligence for Embodied Agents

### The gap this project targets

The current mainstream is a **reactive VLA**: a multimodal foundation model maps
`(image, language, robot state)` to an action or action chunk. That is a powerful
semantic prior, but by construction it does not represent:

- why the current action is appropriate;
- where in the task the execution currently is;
- what has already been completed and what remains;
- what to do when a step fails, is skipped, or is performed off-camera;
- whether the human needs help, and whether helping is wanted at all.

Those are **task-level** questions, not perception or actuation questions, and the
empirical record shows they are not solved automatically by scale: long-horizon
household benchmarks expose brittle skill hand-offs, and current LLM-based
embodied planners still fail often at task tracking, coordination, and error
recovery.

The layer between a foundation model and an action is therefore the project's
target: **explicit task state, memory, prediction, and recovery**, with the VLA as
a component rather than the whole system.

```text
foundation perception/semantics
  + persistent task state
  + episodic / procedural memory
  + task-level world model
  + planner
  + learned skill policy
  + adaptive controller
  + independent safety monitor
```

The interfaces may themselves become end-to-end learned, but the functional
decomposition is expected to survive because the time scales differ
fundamentally: semantic planning at seconds, action chunks at tens of hertz,
stabilization at hundreds of hertz.

### Positioning

Working description of the research direction:

> Human-centered embodied agents that learn, reason about, and assist long-horizon
> tasks through multimodal observation and interactive guidance.

That positioning is deliberately not "a better VLA" and not "a VR guidance
system". The scarce combination it relies on is: computer vision, XR/egocentric
sensing, human-state and attention modeling, task representation, and user
studies — plus the dataset-semantics discipline this repository enforces.

### Reference points

Reading map, not a survey: each row is the system to study for one layer. Numbers
reported in surveys are not independently verified here and should be re-checked
at the source before being quoted.

| Layer | Systems to read | What it teaches |
|---|---|---|
| Action distribution | Diffusion Policy | Generative modeling of an action horizon; multimodal actions |
| Action chunking | ACT | Predicting `[a_t … a_{t+H}]`; temporal abstraction |
| Flow matching | `π0`, `π0.5` | VLM backbone + action expert; continuous action chunks |
| Discrete action tokens | RT-2, OpenVLA | Action as tokens inside a VLM; cross-task semantic transfer |
| Generalist cross-embodiment | Octo, RDT-1B | Heterogeneous robot data; unified action space; scaling |
| World model | DayDreamer, 3D-VLA | Latent dynamics and future-state prediction as a policy ingredient |
| Human–robot / multi-agent | Habitat 3.0, PARTNR | Social environments; where planning and coordination actually fail |
| Data engines and infrastructure | ProcTHOR, RoboCasa, Habitat, Isaac Lab, MuJoCo | Scene/task generation, high-throughput control, contact physics |

### Scope discipline

Deliberately **not** the study target right now: ROS 2, SLAM, low-level control
theory, large-scale RL, and deep simulator engineering. They are substrate, not
research contribution, and the existing substrate (ManiSkill plus the mechanical
arm) is sufficient for the experiments in this phase. Adopt each only when a
concrete experiment demands it.

## Action Policy Families

Four families matter, and they are cumulative rather than competing:

| Family | What it models | Why it exists | Main failure mode |
|---|---|---|---|
| Single-step BC | `a_t = π_θ(o_t)` with MSE | Simple, stable supervised learning | Averages multimodal actions; compounding error in closed loop |
| Action chunking | `[a_t … a_{t+H}] = π_θ(o_t)` | Fewer decisions, temporally consistent behavior | Chunk boundaries; re-planning latency |
| Diffusion policy | `p(A_t \| o_t)` by iterative denoising | Represents genuinely multimodal action distributions | Sampling cost; schedule sensitivity |
| Flow matching | A velocity field whose ODE transports noise to an action chunk | Same multimodal goal with a simpler training objective; used by `π0`-style action experts | Newer, less standardized tooling |

Two conclusions that shape later reading:

- **Plain MSE regression is the wrong default for manipulation actions.** When
  several distinct actions are reasonable from one observation, regression
  produces their average, which may be executable by none of them.
- **Offline action loss is not the metric.** Training data come from the expert
  distribution `o_t ~ p_expert`, execution visits the policy's own distribution
  `o_t ~ p_π`. A small error moves the robot to an unseen state, the next error is
  larger, and the failure compounds. This is the single concept that links BC,
  ACT, Diffusion Policy, VLA, and recovery.

## VLA Anatomy

The pipeline to be able to explain without hand-waving:

```text
RGB (one or more cameras) → vision encoder (ViT / DINOv2 / SigLIP) → visual tokens
language instruction                                                → text tokens
robot state / proprioception                                        → state tokens
                                    ↓
                       transformer (self-attention across tokens)
                                    ↓
                action head: discrete action tokens  |  continuous action expert
                                    ↓
                            action / action chunk
```

Attention is the mechanism that lets these token groups exchange information, so
the concrete question to answer is *which tokens exchange what*: the instruction
token must bind to the relevant visual tokens, and the state tokens must condition
the action. `Q = XW_Q`, `K = XW_K`, `V = XW_V` with
`Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` is the mechanism; the interesting part is
the routing, not the arithmetic.

Two design choices with consequences:

- **Discrete action tokens** (RT-2, OpenVLA) reuse the language-model stack and
  inherit semantic transfer, at the cost of pushing continuous control through a
  tokenizer.
- **Continuous generation** (diffusion or flow-matching action heads) keeps
  control continuous and models multimodality, at the cost of a heavier head.

**Action representation is a first-class research problem, not an implementation
detail.** Dimension, normalization, frame, absolute-versus-delta semantics,
control frequency, and gripper convention decide whether data from two robots can
be mixed at all. This repository already has hard-won evidence for that claim in
its own data: two 8-dimensional action vectors that look compatible are not (see
"Action Specification"), and a unified action space is what makes cross-robot
pretraining possible in systems such as RDT-1B. Concatenating heterogeneous robot
data without semantic alignment hides embodiment semantics rather than pooling it.

## Task World Model

The world model to build first is **task-level, not pixel-level**:

```text
z_t = { object state, task state, human state }
learn  p(z_{t+1} | z_t, a_t)
```

Example: with `kettle = empty`, `cup = empty`, the action `fill(kettle)` should
predict `kettle = filled`; executing a step whose precondition was skipped should
predict a raised failure probability.

Its value is **counterfactual ranking before execution**:

```text
candidate action A → world model → predicted outcome A (+ uncertainty)
candidate action B → world model → predicted outcome B (+ uncertainty)
→ choose / ask / refuse
```

That is what a reactive mapping lacks: an answer to "if I do this, what happens?".
Explicitly avoid starting from video generation; predicting pixels is a different
research programme and not the bottleneck this direction addresses.

## Planning, Recovery, and Intervention

The action vocabulary of an assistant is wider than "act":

```text
act | guide | ask | wait | recover | escalate
```

Which one is selected depends on task state, human state, and confidence. Failure
handling is part of the model, not an exception path: detect the failure, localize
it to a step, decide between retry, alternative path, human help, and safe stop,
and update task state afterwards.

Evaluation for this layer must report more than task success:

- intervention rate, and whether the intervention was necessary;
- recovery success after an induced failure;
- safety violations, force/collision severity, and unsafe exploratory actions;
- calibration of the confidence used to decide, and quality of safe refusal;
- time and resource cost, including the human's time.

Binary success also discards **failure severity**: dropping an object, asking for
help, timing out safely, and colliding with a person are all "failure" and have
nothing in common operationally.

For collaboration specifically, the open problems are shared task state, a model
of what the other agent can do and has done, and initiative/handover. Exchanging
more text between two LLM agents is not a solution to any of them.

## Sim-to-Real: Randomization Versus Adaptation

Two philosophies, which are complementary rather than rival:

- **Domain randomization** — make simulation diverse enough that reality is just
  another variation (the dexterous-hand lineage).
- **Online adaptation** — assume reality will differ, and learn to infer the
  difference quickly (RMA-style latent adaptation from recent experience).

For long-lived agents the second matters more, because dynamics drift after
deployment: payload, wear, friction, camera calibration, latency, and actuator
gain all change. The practical recipe is broad randomization for pretraining plus a
bounded residual adaptation at deployment, wrapped in a safety layer so that
exploration cannot violate physical constraints.

Evaluation should therefore induce controlled post-deployment shifts and measure
time-to-recovery and violation counts, not only "sim versus real" success.

## Study Priority Stack

Research-facing priorities. The lesson sequence in the roadmap remains the
execution track; this table states what each phase is *for*.

| Priority | Gap | Target level | Why it is on the critical path |
|---|---|---|---|
| P0 | Closed-loop BC evaluation | train/validation split, rollout, success rate, failure analysis, covariate shift and compounding error | The pipeline currently trains but has never driven the environment; without this, nothing later can be measured |
| P0 | Deep learning and Transformer fundamentals | tensor/batch, loss/backprop, embedding, self/cross-attention, causal mask, position encoding, residual, LayerNorm, fine-tuning/LoRA | Otherwise OpenVLA, `π0.5`, and world models can only be run, not understood |
| P0 | Action policy families | BC → action chunking → diffusion → flow matching | The current `π`-family design is generative action, not classification-style BC |
| P1 | VLM → VLA architecture | representation → action head; discrete tokens versus continuous generation; pretrain/post-train | The entry point from robot learning into embodied agents |
| P1 | Task representation and task state | steps, preconditions, effects, prerequisite relations that admit cycles and re-entry, completion conditions; state estimation; observation ≠ state | The intended research differentiator |
| P1 | Embodied memory | episodic / semantic / procedural memory; object permanence; retrieval and forgetting | Needed for any task longer than a few seconds |
| P2 | Task world model | task-level transition prediction, counterfactual ranking, uncertainty | Supplies consequence prediction to the planner |
| P2 | Planning and recovery | replanning, failure detection and localization, alternative paths, safe stop | Turns task state into behavior rather than a report |
| P2 | Uncertainty and calibrated intervention | confidence calibration, selective prediction, act/ask/wait/escalate | Where existing adaptive-guidance work becomes an embodied-agent contribution |
| P2 | Human–agent collaboration | shared task state, capability model, initiative, handover | Formalization of the human side, not more HCI method work |

Immediate order, given the repository state: finish the closed loop (2.8.7
train/validation on the retargeted expert episodes, then 2.8.8 closed-loop
execution), then Transformer and action-policy study against a concrete system
(OpenVLA as the dissection object), then the task-intelligence layer, where
existing strengths apply.

## Landscape Reading List (Condensed)

A short pointer list for the study order above: three to five systems per
direction and one conclusion each. Venues and years follow the learner's survey
and are reading pointers, not independently audited citations; numerical results
are deliberately omitted and should be checked at the source.

### 1. Language grounding and the cost of structural shortcuts

- R2R / Vision-and-Language Navigation (CVPR 2018)
- Speaker-Follower (NeurIPS 2018)
- VLN-CE / Beyond the Nav-Graph (ECCV 2020)
- HAMT (NeurIPS 2021)

**Conclusion:** instruction following looked nearly solved while the benchmark
supplied the structure (known connectivity, discrete viewpoints, good
localization); removing those assumptions exposes perception and state estimation
as the actual content.

### 2. Mapping, modularity, and scale

- Active Neural SLAM (ICLR 2020)
- Goal-Oriented Semantic Exploration (NeurIPS 2020)
- DD-PPO (ICLR 2020)
- Habitat 2.0 (NeurIPS 2021)

**Conclusion:** explicit spatial memory plus a classical planner beat end-to-end
learning at comparable data, and brute-force scale solved a narrow task but not
long-horizon skill hand-offs.

### 3. Imitation learning and generative action

- What Matters in Learning from Offline Human Demonstrations / robomimic (CoRL 2021)
- BC-Z (CoRL 2021)
- Diffusion Policy (RSS 2023)
- ACT (RSS 2023)
- `π0`: A Vision-Language-Action Flow Model (2024) — [paper](https://ar5iv.labs.arxiv.org/html/2410.24164)

**Conclusion:** in offline imitation, data regime and quality dominate
architecture; generative and chunked action heads exist because manipulation
actions are multimodal and temporally correlated, not because regression is
slightly worse.

### 4. Generalist policies and cross-embodiment transfer

- RT-1 (RSS 2023) and RT-2 (CoRL 2023)
- Octo (RSS 2024)
- OpenVLA (CoRL 2024)
- RDT-1B (ICLR 2025)

**Conclusion:** semantic priors transfer across tasks and robots; what limits
transfer is action representation and data-mixture compatibility, not parameter
count.

### 5. Prediction as a policy ingredient

- DayDreamer (CoRL 2022)
- 3D-VLA (ICML 2024)

**Conclusion:** learned prediction pays off when it models something the policy
can act on; the entry point for this project is task-level transition prediction,
not pixels.

### 6. Sim-to-real: randomization and adaptation

- Learning Dexterity (OpenAI, 2018)
- RMA (RSS 2021)

**Conclusion:** randomization covers variation anticipated at training time;
long-lived deployment also drifts, which favours fast online adaptation of a
bounded controller behind a safety layer.

### 7. Long-horizon, human-robot, and multi-agent evaluation

- Habitat 3.0 (ICLR 2024)
- PARTNR (ICLR 2025)

**Conclusion:** strong contemporary planners still fail at task tracking,
coordination, and error recovery — precisely the gap a task-state and memory layer
targets, and the reason reliability metrics matter more than nominal success.

### 8. Task structure and task state from human activity

- Differentiable Task Graph Learning: Procedural Activity Representation and Online
  Mistake Detection from Egocentric Videos (NeurIPS 2024) —
  [paper](https://neurips.cc/virtual/2024/poster/96827)

**Conclusion:** task structure can be learned from first-person video and used for
online mistake detection, which validates studying task structure and task state
rather than only policy outputs; check whether its ordering assumptions survive
retry and recovery, since re-entry is the case real execution keeps producing.

### 9. Data engines and infrastructure

- ProcTHOR (NeurIPS 2022), RoboCasa (RSS 2024)
- Habitat (simulation and task stack), Isaac Lab (GPU-parallel control), MuJoCo
  (contact-rich physics)

**Conclusion:** scene and task diversity is the strategic asset; the simulator is
chosen per experiment, and none of them should become the research contribution.

## Current Engineering Decision

The initial fixed task is `PickCube-v1`. Keep it as the main environment while
validating the complete data pipeline. Add a planar pushing task and a
contact-rich insertion task only after the basic interfaces are stable.

## Current Strategic Decision

- The project is not becoming a traditional manipulation/control curriculum.
- ManiSkill and the mechanical arm remain the stable embodiment for every
  executable experiment.
- The research target is the **task-intelligence layer** between foundation models
  and action — task state, memory, task-level prediction, recovery, and
  intervention — not another VLA and not a video-generation world model. See
  "Research Direction: Task Intelligence for Embodied Agents".
- Learning priority shifts toward deep learning, multimodal representation,
  action-policy families, VLA anatomy, task-state tracking, memory, task world
  models, and long-horizon recovery, in the order given by "Study Priority Stack".
- Existing XR research is reused as technical prior work: egocentric sensing,
  temporal/spatial task segmentation, gaze and attention modeling, human-state
  estimation, and adaptive guidance become components of a task-centric embodied
  agent.
- ROS 2, SLAM, low-level control theory, large-scale RL, and pixel-level world
  models are substrate, not study goals; adopt them when an experiment requires
  them.
- Current evidence gates remain unchanged: future modules are planned, not
  complete, until code, data, and closed-loop evaluation exist.
