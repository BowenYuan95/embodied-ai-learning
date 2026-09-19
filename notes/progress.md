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

- [x] Replay the source trajectory: all 50 observations/rewards match exactly;
  no step succeeds and step 50 is truncated at the time limit.
- [ ] Verify whether the source contains `T` observations or `T+1` states for
  `T` actions, and document the observation/action pairing time.
- [ ] Write the complete 8-dimensional action specification.
- [ ] Preserve or explicitly diagnose source timestamps instead of using only
  the first interval to infer FPS.
- [ ] Validate the converter against the installed LeRobot v3 interface. Code
  inspection confirms that `add_frame()` already supplies a string `task`
  field; runtime compatibility is still unverified.
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

1. Validate the converter task field at runtime and add explicit
   length/alignment assertions.
2. Complete temporal-contract documentation: replay confirms pre-action
   observations, but the terminal observation is absent and synthetic
   timestamps incorrectly imply 50 Hz instead of the verified 20 Hz.
3. Run the conversion and capture the exact environment/package versions.
4. Reload the output in a fresh process and validate metadata plus boundary
   frames.
5. Implement the independent dataset inspector.
6. Update this file with commands run, outputs observed, and remaining gaps.

## Session Log

### 2026-09-19 — Source trajectory replay

- Inspected both the collector and the standardization notebook. The notebook
  constructs timestamps with `np.arange(T) * 0.02`; these are not measured times.
- Default Python lacks h5py; used the existing `embodied` conda environment
  (ManiSkill 3.0.1) for HDF5 inspection and replay.
- Ran an inline replay, then independently ran
  `conda run --no-capture-output -n embodied python scripts/replay_pickcube_episode.py`.
  Both exited successfully: 50 steps, maximum observation and reward errors
  0.0, success_any=False, success_final=False, terminated=False,
  truncated=True at step 50. Seed 0 and controller were reconstructed from
  collector code; complete initial simulator state is not stored in the file.
- Runtime control frequency is 20 Hz, conflicting with synthetic 50 Hz
  timestamps. Source files were not modified; timestamp repair remains pending.
- Inspected installed PickCube evaluate(): success requires object placement
  within the goal threshold AND a static robot; grasp alone is insufficient.
- CUDA/NVML was unavailable in this session; runtime reported CPU rendering
  fallback. No rendered video or renewed GPU validation is claimed.
- Source replay gate now passes. Lesson 2 remains in progress.

### 2026-09-19

- Verified Linux, CUDA, PyTorch, and ManiSkill environment.
- Ran `PickCube-v1` and generated a random rollout.
- Created and inspected HDF5 trajectory data.
- Established the 42-dimensional observation decomposition.
- Drafted the LeRobot v3 converter and environment setup script.
- Audited Lesson 2 against execution-based acceptance criteria.
- Result: Lesson 2 remains in progress pending replay, conversion execution,
  read-back validation, temporal checks, and the reusable inspector.

### 2026-09-19 — Progress and curriculum review

- Read the progress record, roadmap, concept notes, README, and converter.
- Ran `git status --short`, `git diff --stat`, and
  `rg --files scripts notebooks datasets environment`; the working tree was
  clean before this documentation correction, and the reusable inspector was
  absent. Confirmed absence with `test -f scripts/inspect_robot_dataset.py`.
- Corrected stale task-field guidance: the converter already uses a string
  `task` field, not `task_index`.
- Static inspection found FPS inferred from the first timestamp interval and
  frame iteration over observations without explicit length/alignment checks.
- No simulation, conversion, or read-back tests were run in this review.
  Lesson 2 remains in progress; the recorded 60–70% estimate is unchanged.
