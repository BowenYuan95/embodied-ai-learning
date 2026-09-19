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

### Controller

A controller converts a target or action representation into lower-level robot
commands. A learned policy action is therefore not automatically equivalent to
a motor command.

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

Delta actions change physical meaning when the control frequency changes.
Frequency conversion therefore requires semantic reasoning, not only array
resampling.

## Demonstrations and Evaluation

- A random rollout is useful for testing serialization and conversion.
- It is not an expert demonstration unless task success and trajectory quality
  are verified.
- Offline action prediction loss does not establish closed-loop task success.
- Policy evaluation should ultimately include success rate, robustness,
  perturbations, and a failure taxonomy.

## Current Engineering Decision

The initial fixed task is `PickCube-v1`. Keep it as the main environment while
validating the complete data pipeline. Add a planar pushing task and a
contact-rich insertion task only after the basic interfaces are stable.
