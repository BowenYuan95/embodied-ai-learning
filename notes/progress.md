# Embodied AI Learning Progress

Last updated: 2026-09-21

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

Roadmap lessons (`docs/roadmap_v3.md`):

| Lesson | Status |
|---|---|
| Lesson 0 — Embodied AI system overview | Complete |
| Lesson 1 — Robot state, frames, and action spaces | Complete |
| Lesson 2 — Robot Dataset: LeRobot × ManiSkill × Sim/VR/Real | **In progress** |
| Lesson 3 — SO(3), SE(3), and coordinate transforms | Not started |

Lesson 2 sub-step breakdown used for day-to-day tracking. This is a working
granularity for the lesson only: `docs/roadmap_v3.md` defines Lesson 2 as a
single unit and does not name sub-steps 2.1–2.9, so these names are local
tracking labels, not curriculum artifacts.

| Sub-step | Topic | Status |
|---|---|---|
| 2.1–2.7 | Robot data pipeline: PickCube rollout, HDF5 read/parse, 42-d observation decomposition, 8-d action analysis, `(obs[t], action[t])` alignment, ManiSkill HDF5 → LeRobot conversion, pre/post frame/FPS/observation/action consistency checks, data quality report and action/reward curves | Complete `[reported]` |
| 2.8.1 | Read a single frame: `dataset[0]` | Executed once outside this environment `[stale]` |
| 2.8.2 | DataLoader: `[42]`, `[8]` → `[B,42]`, `[B,8]` | Not started |
| 2.8.3 | Policy input/output | Not started |
| 2.8.4 | Time window `[B,T,D]` | Not started |
| 2.9 | Generate successful expert demonstrations | Not started |

Stopping point as of this update: **2.8.1, at the entrance to reading a single
frame.** Lesson 2 is in the transition from data engineering to "how a model
reads the data". No training is in scope yet.

After 2.9, the next unit is Lesson 3 (Behavior Cloning → ACT → Diffusion Policy
→ VLA), which is outside the sub-step numbering above.

Estimated Lesson 2 completion: **75–85%**. The previously recorded 60–70% was
stale; it predates the completed conversion and consistency verification.

## Blocking Issue: The Local Environment Is Empty

`[verified]` This is the single blocker for 2.8.1 and it must be fixed before any
further runtime evidence is possible on this machine.

| Item | Observed state |
|---|---|
| Notebook output paths | `/home/bowenyuan95/Projects/embodied-ai-learning` |
| Actual home directory | `/home/bowenyuan`; `/home/bowenyuan95` does not exist |
| `embodied` conda env | Exists at `$HOME/miniforge3/envs/embodied`, but `site-packages` contains only `pip`, `setuptools`, `wheel`, `packaging` |
| `conda-meta/history` tip | `conda create -y -n embodied python=3.12` — the environment was created fresh and never populated |
| `torch`, `lerobot`, `mani_skill` in `embodied` | Not installed |
| `numpy`, `pandas`, `pyarrow` in conda `base` | Not installed |

Consequence: the `dataset[0]` result recorded in
`notebooks/inspect_lerobot_dataset.ipynb` was produced in a different, now
absent environment. It is retained as historical evidence but is `[stale]` and
must not be cited as current verification.

`environment/setup_linux_v3.sh` is written and pins the toolchain, but execution
stopped after `conda create`. It has not been run to completion.

## Lesson 2 Evidence

### Completed

- [x] Generated a ManiSkill `PickCube-v1` rollout. `[reported]`
- [x] Saved rollout data in HDF5. `[verified]` — `datasets/pickcube/*.h5` present.
- [x] Inspected observations, actions, rewards, timestamps, and metadata.
  `[reported]`
- [x] Designed a standardized episode schema. `[reported]`
- [x] Decomposed the 42-dimensional observation vector. `[reported]`
- [x] Identified that some task-state fields may be simulator privileged state.
  `[reported]`
- [x] Created a ManiSkill-to-LeRobot conversion path. `[verified]` — the
  standalone `convert_maniskill_to_lerobot.py` was superseded by
  `scripts/run_pipeline.py` plus `scripts/pipeline/`, and is now frozen in
  `archive/lesson_2_superseded/`.
- [x] Converted the trajectory to LeRobot v3 and produced
  `datasets/lerobot/pickcube/`. `[verified]` — artifact present with a
  conversion manifest.
- [x] Verified pre/post frame count, FPS, observation, and action consistency.
  `[reported]`
- [x] Produced a data quality report with action/reward curves. `[reported]`
- [x] Concluded that the current trajectory structure is valid but the actions
  are randomly distributed and therefore unsuitable as expert demonstrations.
  `[reported]` — this is the motivation for sub-step 2.9.

### In Progress or Missing Evidence

- [ ] 2.8.1 `dataset[0]` read-back in a working local environment. `[stale]`
- [ ] Validate episode count, frame count, FPS, feature shapes, dtypes, ranges,
  NaN/Inf, and first/last frames against the artifact from a running process.
  `[open]`
- [ ] Implement `scripts/inspect_robot_dataset.py`. `[verified]` absent — the
  file does not exist, although `docs/roadmap_v3.md` lists it as a Lesson 2
  deliverable.
- [ ] Resolve the 20 Hz versus 50 Hz temporal contradiction (see below). `[open]`
- [ ] Decompose the 42-d observation into named semantic fields in the dataset
  contract. `[open]` — the manifest explicitly defers this.
- [ ] Produce a reusable Sim/VR/Real compatibility report. `[open]`

## On-Disk Artifact Facts

`[verified]` by reading `datasets/lerobot/pickcube/meta/info.json` and
`conversion_manifest.json`:

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

These facts come from dataset metadata, not from reading the parquet payload. No
frame-level read has been performed on this machine.

## Action Specification

`[verified]` from `conversion_manifest.json`. Note that this supersedes the
earlier statement that action semantics were undocumented:

| Component | Dims | Controller | Semantics |
|---|---:|---|---|
| Arm | 7 | `PDJointPosController` | normalized joint position target |
| Gripper | 1 | `PDJointPosMimicController` | continuous normalized position target |
| **Total** | **8** | | range `[-1.0, 1.0]` |

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
state) is not yet encoded anywhere machine-readable.

## Open Contradiction: 20 Hz Control versus 50 Hz Declared FPS

`[open]` This is a genuine temporal-contract defect, not a cosmetic issue.

- `conversion_manifest.json` records `timestamp_source: "synthetic"` and
  `actual_acquisition_frequency_verified: false`.
- The 2026-09-19 replay session recorded a **verified runtime control frequency
  of 20 Hz**, against **synthetic 50 Hz** timestamps derived from
  `np.arange(T) * 0.02`.
- `info.json` nonetheless declares `"fps": 50`.

LeRobot converts `frame_index` to seconds using the declared `fps`. Declaring
50 Hz over 20 Hz data compresses the time axis by 2.5×. This directly affects
sub-step 2.8.4: when time windows become `[B,T,D]`, `T` steps will not mean the
number of seconds they claim to mean, and any future policy will see a false
control period.

Do not silently change `fps`. Either the source timestamps are repaired, or the
declared `fps` is corrected and the decision is documented here with the reason.

## TorchCodec Warning: Diagnosed as Cosmetic Here

`[verified]` from the recorded notebook traceback.

- The warning is `Could not load libtorchcodec`, because each
  `libtorchcodec_core{4..8}.so` requires a matching FFmpeg shared library
  (`libavutil.so.56/57/59/60`). None matched the system install.
- `environment/setup_linux_v3.sh` installs `ffmpeg=7.1.1` from conda-forge,
  which supplies `libavutil.so.59` and should let `libtorchcodec_core7.so` load.
  This fix is written but unexecuted.
- More importantly, LeRobot only needs a video decoder for video features. This
  dataset has none, so the decoder path is never entered. The warning does not
  affect 2.8.1 and does not indicate conversion failure.
- LeRobot's fallback message (`Falling back to 'pyav' as a default decoder`) is
  therefore irrelevant to state-only data.

## Lesson 2 Acceptance Gate

Lesson 2 can be marked complete only when all checks pass.

| Check | Required evidence | Status |
|---|---|---|
| Source replay | Replay output and known success/failure | **PASS** — 2026-09-19 replay matched all 50 observations and rewards; `success_any=False`, `truncated=True` at step 50 |
| Temporal convention | Documented `T`/`T+1` relationship and pairing rule | **PARTIAL** — replay confirms pre-action observations; terminal observation absent; 20 Hz vs 50 Hz unresolved |
| Conversion | Converter exits successfully with the installed LeRobot version | **PASS** `[reported]` — artifact on disk `[verified]` |
| Read-back | Fresh `LeRobotDataset` load succeeds | **STALE** — succeeded once outside this environment; not reproducible locally |
| Dataset integrity | Counts, FPS, shapes, dtypes, ranges, NaN/Inf, boundaries | **PARTIAL** — metadata read `[verified]`; no payload read locally |
| Semantics | Observation and action specifications recorded | **PARTIAL** — action spec documented; 42-d observation fields not decomposed in the contract |
| Reusable inspection | `scripts/inspect_robot_dataset.py` runs independently | **ABSENT** `[verified]` |

## Immediate Next Steps

Ordered. Steps 1–2 are prerequisites for everything else.

1. Repopulate the `embodied` conda environment. Run
   `environment/setup_linux_v3.sh` to completion, or install an equivalent
   pinned toolchain. Requires sudo and network access, so it needs explicit
   approval.
2. Re-run 2.8.1 locally and capture the `dataset[0]` output as current
   `[verified]` evidence.
3. Complete 2.8.2: `Dataset` → `DataLoader`, confirming `[42]` and `[8]` become
   `[B,42]` and `[B,8]`, and recording the collate and `num_workers` behavior.
4. Resolve the 20 Hz / 50 Hz contradiction before 2.8.4, since it changes what
   `T` means.
5. Implement `scripts/inspect_robot_dataset.py` as the reusable gate check.
6. Decompose the 42-d observation into named fields with per-field deployability
   in the dataset contract.
7. Only then proceed to 2.9: generate successful expert demonstrations. The
   current random trajectory is a pipeline fixture and must not be used as an
   expert demonstration.

## Session Log

### 2026-09-21 — Script archival

- Reorganized `scripts/` so that completed and superseded work is separated from
  the active Lesson 2 toolchain. Moves used `git mv` to preserve history.
- Archived to `archive/lesson_0_1/`: `smoke_test_maniskill.py`,
  `probe_maniskill_pickcube.py`, `probe_maniskill_state_dict.py`,
  `build_deployment_safe_observation.py`, `verify_relative_geometry.py`,
  `verify_state_flattening.py`, `collect_pickcube_random.py`.
- Archived to `archive/lesson_2_superseded/`: `convert_maniskill_to_lerobot.py`,
  replaced by `scripts/run_pipeline.py` driving `scripts/pipeline/`.
- Archived to `archive/offroadmap/`: `simulate_perception_noise.py`,
  `perception_noise_monte_carlo.py`, `perception_noise_task_tolerance.py`. No
  roadmap lesson covers perception-noise robustness.
- Left `scripts/pipeline/`, `scripts/run_pipeline.py`, and
  `scripts/test_observation_adapter.py` untouched, per instruction.
- Left the remaining Lesson 2 tools in place because the lesson is unfinished.
  `replay_pickcube_episode.py` in particular produced the source-replay
  acceptance evidence and must stay runnable.
- Dependency check before moving: the only cross-module imports in the
  repository are `run_pipeline.py` and `test_observation_adapter.py` importing
  `scripts.pipeline.*`. The archived scripts are imported by nothing. The
  `build_deployment_safe_observation` name collision between the archived script
  and `pipeline/observation_adapter.py` is a function name, not an import.
- Added `archive/README.md` recording why each file was retired and what
  replaced it, and warning that archived scripts are frozen and must be re-run
  from the repository root because they use CWD-relative dataset paths.
- Updated `README.md`: the directory tree now matches the actual layout, the
  collector step points at `collect_pickcube_random_rollout.py`, and the
  conversion step points at `scripts/run_pipeline.py`.
- No scripts were executed in this session; the moves are reversible with
  `git mv` in the opposite direction.

### 2026-09-21 — File-mode repair and roadmap sync

- Diagnosed a repository-wide file-mode divergence that predates this session:
  every tracked file was `755` on disk while the index recorded `100644` for 51
  files and `100755` for `environment/setup_linux.sh` only.
- Root cause is consistent with the working tree having been copied from another
  system, which does not preserve the executable bit faithfully. The workspace
  filesystem is `ext4` (`/dev/sda2`), which does support permission bits, so the
  divergence is copy damage rather than a filesystem limitation.
- Rejected `git config core.fileMode false`. That setting exists for filesystems
  that cannot express the executable bit (FAT/exFAT/NTFS mounts, WSL `/mnt`
  paths, some network shares). Applying it here would hide the difference
  instead of repairing it, and would permanently stop Git from recording any
  future intentional executable bit.
- Instead restored the working tree from the index: 51 files set to `644`, and
  `environment/setup_linux.sh` left at `755` because the index declares it
  executable. Paths were parsed from `git ls-files -s -z` using the TAB
  separator; an earlier attempt that split on spaces failed harmlessly on every
  file and changed nothing.
- Result: `git diff --summary` reports zero mode changes, and `git status` now
  shows only real work — 11 renames, the edited documents, and
  `archive/README.md`.
- Also confirmed `environment/setup_linux_v3.sh` carries a pre-existing content
  change (3 insertions, 1 deletion) that is unrelated to the mode repair.
- Synced the `# 当前进度` section of `docs/roadmap_v3.md`, which still claimed
  60–70% and still listed conversion and read-back as pending. The planned
  curriculum and lesson list were left unchanged; only status was updated.

### 2026-09-21 — Progress reconciliation

- Learner corrected the recorded position to Lesson 2.8.1. Progress was
  synchronized to that state, then independently re-checked against the
  workspace rather than accepted at face value.
- Read `notebooks/inspect_lerobot_dataset.ipynb`. It already contains a complete
  `dataset[0]` run: 50 frames, 1 episode, FPS 50, `observation.state [42]
  float32`, `action [8] float32`, scalar index fields, `task` as `str`, and
  `Single-frame inspection: PASS`.
- Discovered that the notebook output paths are under `/home/bowenyuan95/...`.
  `ls -d /home/bowenyuan95` fails; the current home is `/home/bowenyuan`. The
  recorded evidence is therefore not reproducible on this machine.
- Inspected the local conda environment. `embodied` exists but its
  `site-packages` holds only `pip`, `setuptools`, `wheel`, and `packaging`; its
  `conda-meta/history` begins at `conda create -y -n embodied python=3.12`.
  `torch`, `lerobot`, and `mani_skill` are all absent, and conda `base` has no
  `numpy`, `pandas`, or `pyarrow`.
- Read `datasets/lerobot/pickcube/meta/info.json` and
  `conversion_manifest.json`. Confirmed 1 episode, 50 frames, FPS 50, no video
  features, and full action semantics for the 8-d action.
- Diagnosed the TorchCodec warning as an FFmpeg shared-library version mismatch
  that is irrelevant to this state-only dataset, and noted that
  `environment/setup_linux_v3.sh` already installs a compatible `ffmpeg=7.1.1`.
- Recorded the 20 Hz versus 50 Hz contradiction as an open temporal-contract
  defect that must be resolved before time-window work.
- Corrected the stale claim that action semantics were undocumented; the
  manifest documents arm and gripper controllers and target semantics.
- No environment changes, conversions, or simulation runs were performed in
  this session. This update is documentation-only.

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
  Lesson 2 remains in progress.
