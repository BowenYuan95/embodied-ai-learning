# Embodied AI Learning Progress

Last updated: 2026-09-19

This is the single source of truth for current project status. Use Git history
instead of creating date-suffixed progress files.

## Current Position

| Lesson | Status |
|---|---|
| Lesson 0 — Embodied AI system overview | Complete |
| Lesson 1 — Robot state, frames, and action spaces | Complete |
| Lesson 2 — Robot Dataset: LeRobot × ManiSkill × Sim/VR/Real | **In progress** |
| Lesson 3 — SO(3), SE(3), and coordinate transforms | Not started |

Estimated Lesson 2 completion: **60–70%**. Do not advance to Lesson 3 until the
acceptance gate below is satisfied.

## Environment Verified

- Ubuntu Linux
- NVIDIA RTX 3080 Ti Laptop GPU, 16 GB VRAM
- Python 3.12
- PyTorch 2.11.0 with CUDA 12.8
- ManiSkill installed and importable
- `PickCube-v1` rollout executed

The environment setup is represented by `environment/setup_linux.sh`. Exact
package compatibility should still be checked whenever LeRobot is installed or
upgraded.

## Lesson 2 Evidence

### Completed

- [x] Generated a ManiSkill `PickCube-v1` rollout.
- [x] Saved rollout data in HDF5.
- [x] Inspected observations, actions, rewards, timestamps, and metadata.
- [x] Designed a standardized episode schema.
- [x] Decomposed the current 42-dimensional observation vector.
- [x] Identified that some task-state fields may be simulator privileged state.
- [x] Created a ManiSkill-to-LeRobot converter skeleton using
  `LeRobotDataset.create()`, `add_frame()`, `save_episode()`, and `finalize()`.

### In Progress or Missing Evidence

- [ ] Replay the source trajectory and classify it as success or failure.
- [ ] Verify whether the source contains `T` observations or `T+1` states for
  `T` actions, and document the observation/action pairing time.
- [ ] Write the complete 8-dimensional action specification.
- [ ] Preserve or explicitly diagnose source timestamps instead of using only
  the first interval to infer FPS.
- [ ] Correct the converter for the installed LeRobot v3 interface. The current
  skeleton supplies `task_index`; it must be tested against the expected task
  representation, commonly a string `task` field in `add_frame()`.
- [ ] Execute conversion successfully.
- [ ] Load the converted dataset in a fresh process.
- [ ] Validate episode count, frame count, FPS, feature shapes, and first/last
  frames.
- [ ] Implement `scripts/inspect_robot_dataset.py`.
- [ ] Produce a reusable Sim/VR/Real compatibility report.

## Current Dataset Understanding

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

The current action has 8 dimensions, but its controller, space, frame,
absolute/delta semantics, units, frequency, and gripper convention are not yet
fully documented. Shape alone is not sufficient for training compatibility.

## Lesson 2 Acceptance Gate

Lesson 2 can be marked complete only when all checks pass:

| Check | Required evidence |
|---|---|
| Source replay | Replay output and known success/failure |
| Temporal convention | Documented `T`/`T+1` relationship and pairing rule |
| Conversion | Converter exits successfully with the installed LeRobot version |
| Read-back | Fresh `LeRobotDataset` load succeeds |
| Dataset integrity | Counts, FPS, shapes, dtypes, ranges, NaN/Inf, boundaries |
| Semantics | Observation and action specifications recorded |
| Reusable inspection | `scripts/inspect_robot_dataset.py` runs independently |

## Immediate Next Steps

1. Fix the converter task field and add explicit length/alignment assertions.
2. Replay `random_episode_standard.h5`; record success/failure and sequence
   convention.
3. Run the conversion and capture the exact environment/package versions.
4. Reload the output in a fresh process and validate metadata plus boundary
   frames.
5. Implement the independent dataset inspector.
6. Update this file with commands run, outputs observed, and remaining gaps.

## Session Log

### 2026-09-19

- Verified Linux, CUDA, PyTorch, and ManiSkill environment.
- Ran `PickCube-v1` and generated a random rollout.
- Created and inspected HDF5 trajectory data.
- Established the 42-dimensional observation decomposition.
- Drafted the LeRobot v3 converter and environment setup script.
- Audited Lesson 2 against execution-based acceptance criteria.
- Result: Lesson 2 remains in progress pending replay, conversion execution,
  read-back validation, temporal checks, and the reusable inspector.
