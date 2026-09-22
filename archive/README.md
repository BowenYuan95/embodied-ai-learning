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
| `validate_maniskill_rollout.py` | Checked shapes, field-length consistency, `next_observations` continuity, state change, episode semantics, and `elapsed_steps` | The "Validate the trajectory files" cells in `notebooks/2.4_observation_schema.ipynb`, which run the same checks on all three HDF5 files |
| `compare_random_datasets.py` | Compared the synthetic and original rollout variants statistically | The dataset-lineage and shape-comparison cells in `notebooks/2.4_observation_schema.ipynb` |
| `dataset_report.py` | Standalone HDF5 overview, reward/action/temporal statistics, integrity checks, and action-curve plots | `scripts/pipeline/reporter.py`, which performs the same 13 functions as part of conversion rather than as a separate script. Only `main`, `load_h5`, and `run_quality_checks` were standalone orchestration |
| `test_observation_adapter.py` | Asserted the 28-d / 14-d deployable-versus-privileged partition sums to 42 | The same assertion, in place, in `notebooks/1.1_state_and_observation.ipynb` |

The Lesson 2 tools that remain in `scripts/` are the ones that produce data or prove
the acceptance gate: `collect_pickcube_random_rollout.py` (source data),
`generate_expert_demo.py` (planner expert episodes),
`convert_expert_actions_to_delta.py` (their semantics conversion),
`replay_pickcube_episode.py` (the replay evidence required by the acceptance gate),
`run_pipeline.py` with `scripts/pipeline/` (the conversion and quality gate), and
`observation_adapter.py` (the deployable-versus-privileged partition).

`build_lesson_notebooks.py` (notebook generation) was **deleted** on 2026-09-22, not
archived: it re-emitted the notebooks from an older revision without outputs, so it
would have overwritten the current Chinese markdown and the executed-output record.
The notebooks are now the source of truth for their own code. Recover the script from
Git history (`git show <commit>:scripts/build_lesson_notebooks.py`) if an old cell
definition is ever needed.

### Why the reporting script was retired rather than kept

`dataset_report.py` and `scripts/pipeline/reporter.py` shared 13 of their 16
functions verbatim, including `randomness_diagnostic`, which produces the
"actions are random, not expert" conclusion. Keeping both meant two implementations
of every quality metric, and the pipeline — not the standalone script — is what
actually runs. Its generated artifacts (`scripts/figures/`, `scripts/reports/`) are
still on disk and are still written to those paths by `pipeline/reporter.py`, so
retiring the script does not orphan them.

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
| `test_mplib_panda.py` | Constructed an `mplib.Planner` directly from the Panda URDF/SRDF with hard-coded absolute paths | Debug probe |
| `test_planner.py` | Printed the Panda URDF/SRDF paths, links, and joints to set up that planner | Debug probe |
| `test_pick_cube_expert.py` | Ran ManiSkill's stock `pick_cube` solver end to end with `render_mode="human"` and printed `env.unwrapped.evaluate()` | Debug probe — superseded by `scripts/generate_expert_demo.py` (batch collection in `embodied310`) and by `notebooks/2.8_check_data.ipynb` / `2.9_expert_demonstrations.ipynb`. It has no `main` guard, so it executes on import and cannot be inspected safely |

`test_mplib_panda.py` and `test_planner.py` were written while diagnosing why
ManiSkill's motion planner fails. That diagnosis is now settled and was **not** what
was first assumed: constructing `PandaArmMotionPlanningSolver` segfaults inside
`mplib/planner.py:65` because **`mplib` 0.1.1 is built against the NumPy 1.x C API
while NumPy 2.x was installed**, so the constructor calls through an invalid function
pointer to address `0x0`. It is an ABI mismatch, not a GPU or backend problem — the
crash reproduces on CPU, and the same script succeeds under `numpy 1.26.4`. The fix is
`numpy<2` in a planning environment; `mplib` cannot be upgraded because ManiSkill 3.0.1
pins `==0.1.1`. Both probes hard-code the path of one specific conda environment and
are not portable; re-derive rather than re-run them.

Before reusing any of the perception-noise scripts, re-derive the conclusion
rather than trusting the recorded numbers: the noise model and tolerance values
were chosen for exploration, not calibrated against the real PickCube success
threshold.

## Deleted rather than archived

`environment/setup_linux.sh` (the first Linux setup script) was **deleted** on
2026-09-22 at the learner's instruction, after `environment/setup_linux_v3.sh`
superseded it completely: the v3 script parameterizes the environment name,
Python and PyTorch versions, pins the CUDA wheel, and adds `set -Eeuo pipefail`.
Nothing was archived because the old script contributed no behaviour that v3
lacks; recover it from Git history if an old installation needs to be reproduced
(`git log -- environment/setup_linux.sh`).
