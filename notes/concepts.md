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

This is the **agent-level** sense of memory. See "History Representation for Policies"
for the architectural sense — how a network physically carries past inputs — which is a
different question, with different constraints and different failure modes.

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

## Observation, Condition, and Estimated State

Three different things arrive at a policy, and they differ in **epistemic status**, not in which
sensor produced them. Conflating them is how a policy ends up reading a value it cannot have on a
real robot, or ignoring one it must use.

| Layer | What it is | In `PickCube` | Where it comes from |
|---|---|---|---|
| **observation** | measured directly | camera pixels; `qpos` / `qvel` / `tcp_pose` | camera, joint encoders, forward kinematics |
| **estimated state** | *inferred* from observations, possibly over time | `obj_pose`, `is_grasped` | vision; proprioception / contact / touch plus **history** |
| **task condition** | **given**, never inferred | `goal_pos` | task specification or instruction |

Two consequences worth carrying:

- **Almost nothing a policy consumes is a raw observation.** Pixels and encoder readings are; the
  cube's pose and the grasp predicate are estimates. Making that explicit, instead of hiding it in
  an input tensor, is what a task-state layer is for.
- **A condition has no evidence source, and that emptiness is definitional.** It does not mean "the
  goal is unknown"; it means the goal is not something to be inferred from sensors. Filling the slot
  with `vision` confuses *desired* state with *observed* state. Two different tasks can present the
  same image ("put the cube on the left" versus "...on the right"), so the image cannot determine
  the goal (`I_t ⇏ g`), and no loss reveals the mistake.

### A condition correlated with the scene invites a shortcut

`PickCube` samples the goal **from the cube's spawn region** (`pick_cube.py:123-128`:
`goal_xyz[:, :2] = rand * cube_spawn_half_size * 2 + cube_spawn_center`). Measured over the five
expert episodes, `abs(goal_xy - cube_initial_xy)` is `0.0123` to `0.1355 m` (mean `0.0737`) and
`corr(cube_init_x, goal_x) = +0.68` (`n = 5`, indicative only).

The generator therefore manufactures a statistical link between the scene and the goal. A model can
learn "which goal does this image usually go with in the training data" instead of "what does the
task require", score a good training loss, and fail exactly on the counterfactual (same scene, a
different goal). **A low loss is not evidence that a task condition was modelled correctly** — the
same point as privileged information: when the information contract is wrong, MSE will not find it
for you.

### A threshold predicate, and why zero error can be vacuous

The simulator's `is_grasped` is a derived predicate with a hard threshold, not a sensor reading. On
the five expert episodes `qpos[7]` (a finger joint) is `[0.01829, 0.01832]` while grasping and
`[0.02040, 0.04000]` otherwise, so a **single threshold separates the classes with 0 errors in 365
frames**. That looks like a perfect proprioceptive proxy, and it is not evidence of one:

> A method that scores 0 errors on a dataset that **cannot exhibit the failure mode** is not thereby
> validated.

Nothing in these five scripted episodes ever blocks the fingers at cube width; the fingers close on
the cube (`0.01830 ± 3e-5`) or on nothing (`>= 0.02040`). The proxy's actual vulnerability — a table
edge or another object stopping the fingers at roughly cube width — never occurs, so the data cannot
falsify it. Same reasoning as above, one level down: **low loss does not prove a condition was
modelled, and zero error does not prove a proxy is sound.**

The threshold also explains the one-frame `is_grasped` flip recorded in the RGB-collection entry:
the boolean changes within a single control step (`0.02051 -> 0.01831`), so any tiny numerical
difference flips it. A predicate defined by a threshold on a continuous quantity sitting exactly at
that threshold is unstable by construction. A temporal-consistency test — does `T_TCP^-1 T_cube`
stay roughly constant over a window — does not rely on a single-point threshold and is the more
robust definition.

### Language is a condition channel, and one instruction carries zero information

An instruction belongs to the **condition** layer, alongside `goal_pos`. The test for whether a
policy actually needs language is the same criterion used for partial observability: **if changing
the instruction changes the required action on the same observation, language is necessary;
otherwise it is decoration the model will learn to ignore.**

Measured on the current data, the language channel is empty: only one instruction string exists in
the repository (LeRobot `meta/tasks.parquet`: `pick up the cube`), every episode shares it, and the
HDF5 files carry no per-episode instruction field at all. With a constant instruction the channel's
variance is zero, so its contribution is not merely small — it is **unmeasurable**, and any ablation
returns "no change" for reasons that say nothing about language.

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

## Transformer and Attention Fundamentals

Read-side rule for any architecture containing attention: the formula
`Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` is not the difficulty. What matters is
**which tokens are Q versus K/V** (self versus cross), **what the mask allows**,
and **how much routing diversity the head budget buys**. Every numeric claim
below is re-verifiable with `python scripts/attention_walkthrough.py` (23
assertions, standard library only, no dataset or simulator).

### Attention routes; the FFN transforms

`o_i = Σ_j A_ij v_j` with `A` row-stochastic, so every output row is a **convex
combination of the value rows**. Attention cannot produce a direction outside the
convex hull of `V`: it moves and mixes information, it does not transform it. That
is why a block needs the position-wise FFN (and residual paths) as well. In policy
code the useful split is "**attention mixes, FFN transforms**".

### Q, K, V are three roles, and the projections must stay separate

`Q` = what this position looks for, `K` = how this position can be found,
`V` = what it actually passes on. Merging them has concrete costs:

- `W_Q = W_K` makes `S = QKᵀ` symmetric, forcing "i attends to j" to equal
  "j attends to i" — attention becomes undirected. Language, task dependencies,
  and grasp relations are directed, so this is a real loss of expressiveness.
- Sharing `K` and `V` forbids "easy to be found, low payload".

`S_ij = x_iᵀ (W_Q W_Kᵀ) x_j` is a **learned bilinear form**, not a similarity
metric: it need not be symmetric, and `S_ii` need not be the row maximum. Two
traps follow:

- **A high score is not importance** — only geometric compatibility.
- **Scores are not comparable across rows.** Softmax is shift-invariant
  (`softmax(z + c) = softmax(z)`), so only within-row differences matter, and a
  row whose scores are uniformly high can still be the most peaked row. Measured
  on the `T=3` example: the row with the largest raw scores carries the highest
  single weight (`0.5035` vs `0.4011`) and the *lowest* entropy.

### `1/√d_k` is a temperature, not overflow insurance

With near zero-mean unit-variance components the dot product has standard
deviation `√d_k` (measured `22.49` at `d_k = 512` against `√512 = 22.63`).
Unscaled, softmax saturates and its Jacobian `p(1−p)` collapses: measured
`1.05e−1` at logits `±1`, `3.35e−4` at `±4`, `1.13e−7` at `±8`. On one fixed
`q, K` pair at `d_k = 64`, dropping the divisor gave max softmax `0.9806` and
entropy `0.122`, while dividing by `√d_k` gave `0.0947` and `3.785` (uniform is
`4.159`). The divisor is what keeps early routing soft and trainable; overflow is
the secondary concern. The independence assumption is heuristic — a real model
derives both `q` and `k` from the same `X` — but it predicts the measured
magnitudes.

### `A` is a content-generated soft adjacency matrix

`A` is `T×T`, row-stochastic, and recomputed per sample and per layer. This is the
object worth reading, and it is the same mathematical object as a hand-built
temporal/spatial segmentation affinity matrix — with the weights learned rather
than designed. During a manipulation task it answers the architecture question
directly: which observation frames or tokens does the action bind to? Two limits:

- large entries in `A` do **not** establish causal importance, because residual
  paths and later layers can ignore the mixed signal. **Attention weights are not
  an explanation.**
- human gaze is an *exogenous* measured signal; model attention is an *endogenous*
  intermediate variable. They can inform each other, but neither validates the
  other.

### Structural properties to expect

- **Permutation equivariant**: `Att(PX) = P·Att(X)` (verified exactly), i.e.
  attention treats its input as a **set**. Positional encoding is therefore a
  necessity, not an optimization.
- **Global receptive field in one layer**, at the cost of a `T×T` score matrix:
  `O(T²)` time and memory. Long-horizon trajectories hit this, which is one
  motivation for action chunking and for local/sparse/linear attention.
- **No recurrent state**: nothing persists between positions except the tokens
  themselves (or an external KV cache), so the input must carry whatever history
  the policy needs.
- **Softmax is the only nonlinearity inside attention**, so without the FFN a
  block is close to a data-dependent linear map.

### Multi-head: `h` parallel routing maps, merged only at `W_O`

`head_m = Attention(XW_Q^(m), XW_K^(m), XW_V^(m))`,
`MHA = Concat(head_1 … head_h) W_O`, with `d_k = d_v = d_model / h`.

| Claim | Consequence |
|---|---|
| `Q/K/V` parameter count is identical to a single head with `d_k = d_model` | multi-head re-partitions the same budget; it does not buy capacity |
| `W_O` is an *additional* `d_model × d_model` matrix | multi-head necessarily costs one output projection, because a single head's output needs no merging |
| heads have no direct path to each other | `W_O` is the only place they are mixed; drop it and the block is `h` independent pipelines |
| heads share the input `X` and everything downstream | "independent" is an approximation — gradients coordinate them indirectly |
| what relation each head learns is emergent | observed head roles are tendencies, not guarantees; never assert "head 3 does coreference" |

In a framework the `h` small matrices usually do not exist: one
`d_model × d_model` `W_Q` is computed, then reshaped to `(batch, h, T, d_k)`. The
per-head loop and the reshape-and-batch form were verified elementwise identical,
so a reader who sees only a big matrix, a `reshape`, and a `W_O` is not missing
anything.

### Each head's expressiveness is bounded by `d_k`

`S = QKᵀ` with `Q, K` of shape `T × d_k` gives `rank(S) ≤ d_k`. At `d_k = 1` the
score table is an outer product: **every position's row is a scalar multiple of
every other**, so all queries share one preference ordering and can differ only in
sharpness (verified: one shared argmax, versus four distinct argmaxes at
`d_k = 64`). `h = d_model` is therefore legal mathematics and a degenerate design:
routing diversity collapses, each head runs its own `T×T` softmax, and inner
dimension `1` wastes the hardware. Practice keeps `h` near `8–32` so that `d_k`
stays near `64–128`.

### Masks are a semantic decision about allowed information flow

Both mask types set scores to `−∞` **before** softmax; masking after softmax
breaks the row sum.

| Mask | Blocks | Encodes |
|---|---|---|
| causal | all `j > i` | position `t` may use only `≤ t`: autoregressive decoding, and "the action at `t` may not depend on future observations" |
| padding | padded positions | batch alignment of variable-length sequences |

Bidirectional versus causal is not an implementation detail. ACT's encoder is
bidirectional (predicting a whole action chunk may see the whole window), while
OpenVLA emits discrete action tokens autoregressively and must be causal. The
mask states what information is allowed in, so read it before reading the loss.

### The transferable trap

A valid formula with the right shapes is not evidence that the intended concept
survived. `d_k = 1` keeps every symbol and loses per-position preference; two 8-d
action vectors share a shape and still mean different things (see "Action
Specification"). Legality — shapes, symbols, dtypes — is never proof of semantic
equivalence. Ask what freedom is left, not what compiles.

## History Representation for Policies

The question here is narrower than "what is memory": when a policy must condition on
more than the current observation, **where does the history physically live, and what
does retrieving it cost?** Every mechanism below answers that, and the dividing line is
**retain versus compress**.

| Method | History lives in | Retrieved by | What is lost | Time-parallel | Cost vs length |
|---|---|---|---|---|---|
| Frame stacking | the input buffer | concatenation | everything outside the window | yes | input width grows linearly |
| RNN | `h_t` | recurrence | fixed-size bottleneck | no | linear in steps |
| LSTM | `h_t, c_t` | gated recurrence | fixed-size bottleneck | no | linear in steps |
| Transformer | per-token representations / KV cache | attention | nothing inside the context window | yes | ~`O(T²)` compute, `O(T)` cache |

Three distinctions with implementation weight:

- **LSTM does not remove the compression bottleneck; it makes the compressed path
  trainable.** The cell-state update is additive, so the diagonal gradient path is
  `∂c_t/∂c_{t-1} = diag(f_t)` instead of RNN's `diag(1 − h_t²)·W`. Nothing is stored
  outside `c_t`, but the gradient reaches further back.
- **KV cache is what "retention" means concretely at inference.** Past `K, V` are kept
  per layer; each new token adds `O(T)` compute and `O(T)` memory. Long context is
  expensive in *memory*, not only in FLOPs. RNN/LSTM inference stays `O(1)` in memory
  per step, which is a genuine trade rather than a strictly worse design.
- **Parallelism is why the recurrent family lost.** Frame stacking and Transformer
  compute all positions at once; RNN and LSTM cannot, because `h_t` depends on
  `h_{t-1}`. The price paid for parallelizing is the quadratic dependence on sequence
  length.

### Two senses of "memory" that must not be conflated

- **Architectural history representation** (this section): how a network physically
  carries past inputs — a window, a state vector, a cache.
- **Agent-level memory** (see "Embodied Memory"): what the *agent* remembers —
  episodic, semantic, procedural — and what it should forget.

A longer input window is not task-state tracking. A policy can attend over a whole
trajectory and still represent nothing about "which step has been completed", and an
agent can track task state with no attention at all. The first is a representation
question, the second a task-structure question, and the project's research target is
the second.

### What history fixes, and what it does not

History restores information the current observation lost (partial observability), by
making an implicit state estimate `ŝ_t = φ(o_{t-k:t})` possible. It does **not** fix
data scarcity or distribution shift: the Lesson 2 failure is the latter (validation MSE
`0.2350` worse than the `0.1421` mean-action baseline, held-out `|z| = 13.21`, `0/10`
closed-loop success), and a longer window does not repair it.

One detail that is easy to miss and matters for robot policies: the memory recurrence
`m_t = update(m_{t-1}, o_t, a_{t-1})` takes the previous **action** as an input. Whether
a policy is fed `o_{t-k:t}` or `(o_{t-k:t}, a_{t-k:t-1})` is therefore a design choice
with a semantic consequence, not a formatting detail.

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

## Action Chunking

Chunking changes **what one training sample is**: from `(o_t, a_t)` to
`(o_t, [a_t ... a_{t+H-1}])`, i.e. from `action_dim` targets to `H * action_dim` targets,
with the input unchanged. Its claim is often stated too strongly. The precise version:

> Chunking replaces high-frequency closed loop with low-frequency closed loop plus a
> **local open-loop commitment**. It reduces the number of re-observations, not the
> per-decision error, and it does not by itself cure compounding error.

### `L`, `H`, and `K` live in different places

| Symbol | Decides | Belongs to | Consequence |
|---|---|---|---|
| `L` | how many observation frames are read | training (input shape) | orthogonal to chunking; it is the history decision of "History Representation for Policies" |
| `H` | how long the target is | training (output shape) | fixes the sample count and the parameter count; changing it requires **retraining** |
| `K` | how many predicted steps are executed before re-observing | execution (policy hyper-parameter) | appears only in the rollout, never in the dataset |

So the dataset-construction code contains no `K` at all, training advances the start index
by 1 while execution advances it by `K`, and `H - K` is the overlap between consecutive
predictions (the raw material for temporal ensembling; `K = H` leaves none). This is why
one trained model can be swept over `K` but not over `H`.

### Sample count and what the tail costs

A chunk sample needs `t + H <= T`, so an episode of `T` actions yields `T - H + 1` starts
(the `+1` is the usual closed-interval count, the same rule that makes the state array
`T + 1` long). Across `N` episodes:

```text
samples = sum(T_i) - N * (H - 1)
```

Each episode loses exactly `H - 1` starts **regardless of its length**, and those lost
starts are the *endings* of each episode. `H` is additionally bounded by the shortest
episode: at `H = T_min + 1` that episode contributes zero samples, and beyond it the count
goes negative — worth an explicit assertion.

The constraint is a consequence of **dropping** the tail, not of `H` itself: padding
recovers every start but introduces fake targets, which then require a masked loss.

### Chunk boundaries must be episode-local

Concatenating all episodes and then slicing produces an array of the same rank and a
plausible shape, but its final samples **mix two trajectories**. On this repository's five
episodes the naive version yields 353 samples against 325 correct, of which 28 straddle a
boundary — and `crossing = (N - 1) * (H - 1)`. A shape check cannot see this; only an
episode-local loop with an assertion can.

### Multi-step targets change the parameter count

The output layer is `hidden x (H * action_dim)`, so `H = 8` at `hidden = 128` carries
30,272 parameters against 23,048 for `H = 1` — **+31.3%**. A chunked model that beats a
single-step model may therefore be winning on capacity. Any such comparison needs a
**parameter-matched** control (here `hidden = 150` gives 30,308).

### The only fair offline comparison is at `h = 0`

A chunked model's headline validation MSE averages over `H * action_dim` elements, of which
`h = 1 ... H-1` are strictly harder; a single-step model averages over `action_dim`. Those
totals are **not comparable**. The fair quantity is `h = 0`, where both models predict the
same next action from the same state on the same samples.

### `baseline_h` is not constant, so each horizon needs its own baseline

The "predict the training-mean action" baseline rises with `h` (measured 0.5576 to 0.8386
over `h = 0 ... 7` in normalised action space). Two causes, which must be separated:

- further actions are genuinely harder;
- the **sampling window moves**: `h = 0` covers starts `t in [0, T-H]`, while `h = H-1`
  covers `t in [H-1, T-1]`, so later horizons include more of the episode tail, where
  actions deviate further from the mean.

So compare `mse_h` against its own `baseline_h`, not against a single pooled number, and
never read "`mse_h` rises with `h`" as evidence about difficulty alone. The useful summary is
the **useful horizon**: the largest `h` for which `mse_h < baseline_h`.

### `F.mse_loss` broadcasts

`ChunkMLP` returns `[B, 1, 8]` even when `horizon = 1`. If the single-step loader supplies
`[B, 8]`, `F.mse_loss` broadcasts them into `[B, B, 8]` and returns a plausible-looking
loss with wrong gradients — the symptom is a train loss that refuses to fall. Keep the
horizon dimension and assert `pred.shape == yb.shape` before every loss.

This is the same failure class as the episode-boundary bug and as the post-action
observation defect in Lesson 2: **compatible-looking shapes mean nothing complains.** A
`T x T` score matrix, a `[B, B, 8]` loss, and a `(74, 8)` action array can all be wrong
while every shape check passes.

### The `H` question, and why a non-monotonic `S_0` is noise

Sweeping `H` is only interpretable if the sample set is held fixed, because the sample count
depends on `H` (`sum(T_i) - N(H-1)`); letting each `H` use its own natural sample set
confounds "`H`" with "number of samples". Making each `H`'s target the prefix of the largest
`H`'s target (`Y[:, :H, :]`) keeps every start identical and varies only the target length —
and `H=1` must then reproduce the single-step baseline exactly, a free consistency check.

Measured here, `S_0` was **non-monotonic** in `H` (-0.03, -0.15, +0.19, -0.03) while
`overall val MSE` rose monotonically (0.575 to 0.854, which is the target getting harder,
not the model getting worse). A systematic `H` effect would vary smoothly; a 0.34 swing
between adjacent values, on one trajectory and 69 heavily overlapping validation samples, is
what noise looks like — and taking the one positive cell of a `4 x 8` grid is a
multiple-comparison error. The `h=0` failure is therefore **H-independent** (it fails at
`H=1` too), so `H` is not the binding constraint.

### What `K` changes: temporal consistency, not horizon

The intuitive reason "larger `K` clips less" is that larger `K` executes later, milder chunk
elements. That is refutable offline and it is false: predicted magnitude *increases* with
`h`. The online measurement gives the real mechanism. At `K=8` the executed magnitude is flat
across chunk positions (h=0 0.78, h=7 0.77), but the **same position `h=0`** costs 3.22 at
`K=1` against 0.78 at `K=8` — 4x, with identical weights, identical chunk position, and the
same checkpoint, so only the visited state distribution differs.

Extreme actions therefore belong to the **closed-loop state distribution that frequent
re-planning generates**, not to any chunk position. The associated quantity is step-to-step
action change: `mean abs(da)` is `1.5156` at `K=1` (worse than a random-action fixture's
`0.6723`) and `0.035` at `K>=4` (the expert measures `0.0077`). Chunking's contribution here
is **within-chunk temporal consistency and a calmer state distribution**, not look-ahead.
Caveat: the two are confounded, because `K=1` and `K=8` visit different states.

One separate model defect surfaced in the same measurement: the policy's predicted magnitude
is about **1.4x the target's at every horizon**, i.e. it amplifies actions by roughly 40%.
That is a property of the model rather than of `K`, and it is consistent with the saturation
seen in closed loop.

### Measured outcome on this repository's data

With five expert episodes (`T = 74/74/50/86/76`), `H = 8`, and one held-out episode: the
useful horizon is **0**, chunked `h = 0` error (0.5718) equals the single-step error
(0.5752), and only the parameter-matched larger single-step model (0.4279) beats the
mean-action baseline (0.5576). Closed loop is **0/5 at every K**. `K` does move the
structural quantities exactly as expected (replans 200 -> 25, open-loop 0.05 -> 0.40 s)
and reduces action clipping (0.947 -> 0.121) without improving success. The binding
constraint on this data is **coverage**, not the chunking decision.

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
the routing, not the arithmetic — see "Transformer and Attention Fundamentals" for
what that routing matrix does and does not license.

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

## Measuring Whether One Signal Reveals Another

Answering "can the task be read from this input?" is a measurement, and three
mistakes make the measurement wrong while still producing a confident number.
All three were made in this repository before being caught.

- **Range overlap is not separability.** Overlap is a *worst-case range* statistic;
  separability is a *distribution* statistic, and neither implication holds. Measured
  on the two-task pool: `qpos[7]` overlaps by 0.0217 and is **98.0%** separable;
  `qvel[8]` overlaps by 0.3045 and is also **98.0%**; `tcp_pose[4]` overlaps by only
  0.0005 and lands at **81.4%**. A criterion that is sometimes right and sometimes
  off by 47 points is worse than one that is always wrong, because you cannot tell
  which case you are in. Delete it from the method rather than tuning it.
- **Never use the data values themselves as thresholds.** A threshold equal to a
  sample puts that sample on a knife edge, and rounding the data to build the
  threshold set moves it across. Use midpoints between neighbouring distinct values.
  Under the knife-edge version `tcp_pose[4]` reads 51.2% instead of 81.4%, and
  `qpos[7]` alternates between 50% and 100% from one timestep to the next.
- **Always test both label orientations.** A one-sided test (`a > t`) silently
  assumes which class is on which side. The image brightness statistic read **50.9%**
  one-sided and **88.7%** two-sided, purely because PushCube is the brighter class.

Two further distinctions:

- **A leak can be temporal.** Per-frame separability is not one number. Here
  `proprio` is exactly at chance at `t=0` (the ten episodes of both tasks sit on an
  identical state, differences of 0.000000) and 98% from `t=1`, because the first
  action already moved the gripper differently. Any claim about what a policy "can
  read" has to name the frame.
- **A leak is not a representation.** The image separates the tasks at 88.7%
  (100% at `t=0`) from a global brightness offset of 0.57/255, while the goal's
  actual image signal is `corr = -0.295` and the cross-episode pixel change equals
  the within-episode 25-step change (1.0x). The statistic is enough to let a model
  take a shortcut and carries no usable object information. Measuring that a signal
  is *present* is not measuring that it is *usable*, and a confound can be strong
  without being a feature.

## Language Grounding

"Grounding" names two different problems with different difficulty, and conflating
them makes an experiment look stronger than it is:

| | **Referential grounding** | **Action-mode grounding** |
|---|---|---|
| What the word picks out | a region or attribute **in the image** | a **behaviour pattern** |
| Example | "red cube" ↔ those pixels; "left" ↔ that side | "pick" ↔ grasp and carry; "push" ↔ slide along the table |
| Is there a pixel it corresponds to | yes, it can be localised | **no** — no region *is* "pick" |
| Mechanism required | vision-language binding (cross-attention, or pretrained alignment) | language used as a **mode selector** |

Four durable consequences:

- **Concatenation fusion cannot perform referential grounding, by construction.**
  A concatenated language embedding is a per-episode constant; it never indexes
  into the visual features, so it cannot say "attend to *that* region". Binding a
  word to a region requires the language to modulate which visual features are
  read, i.e. cross-attention with `Q` from language and `K`, `V` from vision. An
  MLP over `concat(visual, state, language)` can only learn "language vector →
  which mode", which is action-mode grounding and nothing more.
- **Referential grounding is a property of the data, not of the model.** It needs
  scene variation — multiple objects, varied attributes, words that single one
  out. With one object per scene and no colour or spatial words in the
  instruction, referential grounding is **untestable by construction**; no
  architecture change makes it testable. Check the vocabulary and the scene
  composition before claiming to test it.
- **A task difference can be localised in a single action channel.** Before
  describing two tasks as "actionally different" in general, decompose the
  cross-task versus within-task deviation **per channel**. In this repository's
  two-task pool the seven arm channels sit at 0.72–1.12x (statistically
  indistinguishable) while the **gripper channel is 7.00x**. The aggregate ratio
  was real but its location was not where the aggregate implied, and the
  correction narrows what the language test can prove.
- **A multi-task pool permits language use; it does not force it.** Test every
  input channel for separability of the task before concluding that the data demands
  the instruction, and see "Measuring Whether One Signal Reveals Another" for how to
  do that test. On this repository's two-task pool **every** contract channel leaks
  the task: `task_goal` separates 10/10 episodes, `image` mean brightness separates
  88.7% of frames (100% at `t=0`, because the camera sees the goal marker), and
  `proprio` separates 98% from `t=1` (`qpos[7]` is the gripper). No frame and no
  field subset makes the instruction necessary for an image-conditioned policy, so
  the language test had to be pre-registered as a negative result and moved to a
  deliberately restricted probe: `t=0`, `proprio` only, no image and no `task_goal`.

**Input, use, and understanding are three different claims.** A `language_ids` field
in the contract establishes only the first. The ladder, weakest to strongest:

| level | testable criterion | what it needs |
|---|---|---|
| **input** | the field exists, differs per task, has the right shape | nothing beyond a schema |
| **use** | a **counterfactual**: replace the instruction, the action must change, and change in the *right direction* | a state at which the task is otherwise unreadable |
| **understanding** | generalisation to **unseen** instructions: new words, new compositions, referring expressions | more instructions and scene variation than most small pools have |

The counterfactual ladder has three rungs, and they must be built as three *different*
manipulations or they collapse:

- **T1 correct** — changes nothing. Only shows fit; it cannot show use.
- **T2 shuffled** — permutes the **word order** of the same instruction (same bag of words).
- **T3 contradictory** — replaces the **bag of words** with another task's instruction.

With only two tasks, "the wrong instruction" *is* "the other instruction", so a T2 built
as a task swap is identical to T3 and the ladder has two rungs, not three. A three-rung
ladder needs at least three instructions. This is a data-design constraint, not a code one.

**T2 is usually a negative control on the architecture, not evidence about language.** Any
mean- or sum-pooled embedding of token vectors is permutation-invariant, so a model that
pools language without positional information *must* be insensitive to T2 — mathematically
exactly, and numerically only to floating-point summation order (measured here: `2.2e-16`
against `0.609` for T3). If T2 moves the action, either the encoder is not
permutation-invariant or positional information entered somewhere. A T2 result is therefore
meaningless unless it is reported against a named architecture.

**Sensitivity is not accuracy, and neither is sensitivity in the right direction.** Three
things have to hold at once: the counterfactual must move the action (`sens != 0`), it must
move it toward the correct task's behaviour, and T2 must stay at zero. Offline loss can
establish none of them, because shuffling the instruction does not change the target — the
loss necessarily rises and carries no information — and because a language-blind model can
already read the task from the state.

The general rule is the counterfactual: **a model cannot learn to use a signal the
data never forces it to use.** Shuffle the instruction, substitute a wrong one,
and drop it entirely; if the action does not change, no grounding happened,
whatever the loss curve says.

A **temporal** leak is a distinct category from a per-frame one, and the two live in
the same dataset. Here `proprio` is exactly at chance at `t=0` and 98% separable
from `t=1`, because PushCube's first action already drove the gripper to 0.0000
while PickCube held it open at 0.0400. Judging whether memory is *necessary*
therefore has to be scoped both to the frame and to the temporal receptive field of
the policy being tested.

## Ablation Discipline

An ablation arm is only interpretable if everything except the variable under test is held
fixed. Four things break that, and all four were hit in this repository before being fixed.

- **Match capacity, or you are measuring capacity.** Adding a modality adds parameters. In
  lesson 3.8.6 the state-only arm had 250,176 parameters against the image arm's 511,712 —
  a factor of **2.045**. Any "the image helped" conclusion from that pair is uninterpretable.
  The fix is a parameter-matched control (here `fusion_hidden = 1139` giving 511,635, a
  difference of 77), reported **alongside** the unmatched one. The two versions can disagree,
  and here they did: the unmatched gain was `+0.010694` and the matched gain `+0.014489` —
  matching made the apparent effect *larger*, because the widened state-only control got
  slightly worse while the image arm stayed put. **Report both numbers; the matched one is the
  claim.**
- **`best_val` is selected on the validation set, so it is optimistically biased.** Choosing
  the checkpoint by the same split you then report means the number is partly fitted to that
  split. With 128 validation samples and one seed per arm, differences of a few percent are
  not evidence. Say so before interpreting any ranking.
- **Distinguish the seeds.** A *split* seed fixes which episodes are held out; an *init* seed
  fixes weight initialisation and batch order. They are different knobs. Conflating them here
  silently moved the held-out episode from 4 to 2 (`default_rng(0).permutation(5)[0] == 2`
  against `default_rng(42).permutation(5)[0] == 4`), which made the run incomparable with the
  previous lesson and was caught only by an assertion. Keep them as named constants and assert
  the split identity explicitly.
- **Report the parameter count next to every arm.** It is what makes the capacity question
  answerable at a glance, and it is the first thing a reader needs.

**Two different comparisons, two different meanings.** An arm pair that differs only by an
input that is *not independent* cannot differ at all, and running it is a **wiring control**,
not an experiment. Here `H(l | g) = 0` exactly — the instruction is a deterministic function of
`task_goal` on this pool — so `image+proprio+goal+language` and `image+proprio+goal` are
provably identical in expectation, and a difference would mean the implementation is wrong.
Registering such a pair as a control, and saying in advance what its result must be, is worth
more than another arm.

**The device is a variance source, so record it with every result.** The same six arms run on
16 CPU cores and on an RTX 3080 Ti give `val_chunk` values that differ in the **second decimal**
(E1a `0.097776` against `0.097286`; E2 `0.087082` against `0.088187`), and one arm's
gripper-sign accuracy moved from `99.2%` to `93.8%` with a different best epoch. Float
reduction order differs between backends, so the training trajectory diverges even at a fixed
seed. What survived the device change: the capacity-matched `Delta_vision` (`+0.014489` on CPU,
`+0.014186` on GPU) and the `t=0` probe (`50.0%` / `100.0%` on both). What did not: any arm
ranking separated by less than that gap. A number without its device is not reproducible.

A related trap: whether the GPU is visible depends on **how the process was launched**, not on
the machine. Here the interactive Jupyter kernel had CUDA while the agent's sandboxed shell got
`PermissionError` on mode-666 `/dev/nvidia*` nodes and silently fell back to CPU — so two runs
of the same notebook, in the same environment, used different hardware. Print the device
inside the notebook rather than assuming it.

**Measure the seed spread before interpreting any ranking.** One arm at three init seeds (same
split, same normalisation, same epochs) gave per-arm ranges of `0.009` to `0.034` in `val_chunk`,
while the arm-to-arm effects being ranked were `0.001` to `0.014`. Two consequences, both
observed:

- An effect smaller than the seed range is not evidence, and it is worse than merely imprecise:
  `Delta_vision` read `+0.014186` at the first seed and then `-0.006277`, `+0.001176` — its
  **sign flips**, so the single-seed number was a draw rather than a measurement.
- A sweep can also *rescue* a claim. `Delta_E3` was negative at one seed, against the ceiling
  prediction of 3.8.4.4, and positive at all three — the sweep restored the predicted direction
  while leaving the magnitude unestablished. Report the direction and the magnitude separately.

**The capacity-matched control can be the noisiest arm.** The widened state-only arm had a range
of `0.0338`, 2.5x the arm it was matched against, because a wider MLP over 509 samples is simply
more variable. Matching capacity buys comparability and pays for it in variance, so the
comparison that motivated the match becomes the one the match makes least reliable. Reduce the
control's instability rather than adding seeds.

**Same information, different encodability.** `I+p+goal` and `I+p+language` at matched capacity
(fusion dim 416 both, parameters within 0.1%) have **disjoint** seed ranges and differ by **2.27x**
(`0.0825 [0.0749, 0.0883]` against `0.0364 [0.0318, 0.0435]`). They carry the *same* information
on this pool (`H(l | g) = 0`), so the gap is not information but **learnability**: thresholding a
continuous 3-dim goal into a task is harder than reading a discrete index. Representation choice
is a learnability question, not only an information question.

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
