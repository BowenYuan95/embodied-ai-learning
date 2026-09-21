# Archive

Frozen scripts that are no longer part of any active workflow. They are kept for
provenance: each one produced evidence that is cited in `notes/progress.md`, and
Git history alone is harder to read than a labelled directory.

These files are **not maintained**. They are not imported by anything, they are
not covered by the acceptance gate, and they may break as ManiSkill, LeRobot, or
the dataset layout change. Do not extend them; write a new script under
`scripts/` instead.

All of them resolve dataset paths relative to the current working directory, so
they must be run from the repository root:

```bash
python archive/lesson_0_1/smoke_test_maniskill.py
```

## `lesson_0_1/` — completed foundation lessons

Lesson 0 and Lesson 1 are complete. These scripts were one-off probes that
established the environment, the observation layout, and the frame conventions.
Their conclusions now live in `notes/concepts.md` and the roadmap.

| File | What it did | Superseded by |
|---|---|---|
| `smoke_test_maniskill.py` | Created `PickCube-v1` and stepped it once to prove the install worked | `scripts/collect_pickcube_random_rollout.py` |
| `probe_maniskill_pickcube.py` | Dumped observation, action, reward, and info structure and shapes | Ad-hoc inspection; findings are in `notes/concepts.md` |
| `probe_maniskill_state_dict.py` | Printed the nested `state_dict` observation tree | `verify_state_flattening.py` (also archived), whose conclusions now live in `scripts/pipeline/observation_adapter.py` |
| `build_deployment_safe_observation.py` | Split observations into deployable and privileged parts | `scripts/pipeline/observation_adapter.py` |
| `verify_relative_geometry.py` | Checked `tcp_to_obj_pos` and `obj_to_goal_pos` against recomputed poses | No active replacement |
| `verify_state_flattening.py` | Confirmed the 42-d flattened vector matches the `state_dict` layout | No active replacement; the 42-d decomposition is recorded in `notes/progress.md` |
| `collect_pickcube_random.py` | First random rollout collector, wrote `random_episode_000.h5` | `scripts/collect_pickcube_random_rollout.py` |

## `lesson_2_superseded/` — replaced within an in-progress lesson

Lesson 2 is still in progress, but this particular script has been replaced by
the pipeline. It is archived so the active `scripts/` directory has exactly one
conversion path.

| File | What it did | Superseded by |
|---|---|---|
| `convert_maniskill_to_lerobot.py` | Standalone HDF5 to LeRobot v3 conversion with hard-coded features | `scripts/run_pipeline.py`, which drives `scripts/pipeline/` through load, validate, convert, post-validate, report, and manifest |

The remaining Lesson 2 tools stay in `scripts/` because the lesson is not
finished: `collect_pickcube_random_rollout.py`, `validate_maniskill_rollout.py`,
`compare_random_datasets.py`, `dataset_report.py`, and
`replay_pickcube_episode.py`. In particular, `replay_pickcube_episode.py` is the
tool that produced the source-replay evidence required by the Lesson 2
acceptance gate, so it must remain runnable.

## `offroadmap/` — exploratory work outside the curriculum

`docs/roadmap_v3.md` has no lesson for perception-noise robustness. These
scripts were self-directed exploration of how pose noise propagates into task
tolerance. They are kept because the results may become relevant to a later
robustness or Sim2Real lesson, but they are not part of Lesson 0, 1, or 2.

| File | What it did | Status |
|---|---|---|
| `simulate_perception_noise.py` | Injected Gaussian noise into `state_dict` poses inside ManiSkill | Exploratory |
| `perception_noise_monte_carlo.py` | Analytic Monte Carlo over noise levels in millimetres | Exploratory |
| `perception_noise_task_tolerance.py` | Compared noise levels against task tolerances | Exploratory |

Before reusing any of these, re-derive the conclusion rather than trusting the
recorded numbers: the noise model and tolerance values were chosen for
exploration, not calibrated against the real PickCube success threshold.
