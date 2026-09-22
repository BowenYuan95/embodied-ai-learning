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
important and is closer to dependency-aware procedural reasoning.

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
confidence, and failure/recovery status. Sequential lists are sufficient only
when order is fixed; optional, parallel, or prerequisite-constrained steps
require a graph or partial-order representation.

### Task State and Dependency Graph

The project uses explicit task state as a verifiable interface between
perception, planning, and action. A node may be:

```text
Locked | Available | Active | Completed | Failed
```

The graph is not assumed to be fully hand-authored forever. Early experiments
may use simulator events or manual annotations; later work should infer
segments, dependencies, and state transitions from demonstrations and video.

### Embodied Memory

- **Working memory** stores current observations and short temporal context.
- **Episodic memory** stores prior trajectories, failures, and recoveries.
- **Semantic memory** stores object, scene, and task knowledge.
- **Procedural memory** stores reusable skills, task graphs, and execution rules.

Object permanence and re-identification are memory problems as well as
perception problems: when an object leaves the camera view, the agent should
retain identity, last-seen state, and uncertainty rather than silently treating
the object as absent.

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

## Current Engineering Decision

The initial fixed task is `PickCube-v1`. Keep it as the main environment while
validating the complete data pipeline. Add a planar pushing task and a
contact-rich insertion task only after the basic interfaces are stable.

## Current Strategic Decision

- The project is not becoming a traditional manipulation/control curriculum.
- ManiSkill and the mechanical arm remain the stable embodiment for every
  executable experiment.
- Learning priority shifts toward deep learning, multimodal representation,
  VLA, task memory, task world models, and long-horizon recovery.
- Existing XR research is reused as technical prior work: egocentric sensing,
  task segmentation, dependency graphs, gaze/attention modeling, and adaptive
  guidance become components of a task-centric embodied agent.
- Current evidence gates remain unchanged: future modules are planned, not
  complete, until code, data, and closed-loop evaluation exist.
