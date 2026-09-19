# Project Guidance for AI Coding Agents

## Project Mission

This repository is a long-term embodied-AI learning and engineering project.
Its main experimental path is:

`ManiSkill → trajectory/HDF5 → LeRobot → policy training → closed-loop evaluation → data iteration`

The goal is not only to make code run. Every experiment must preserve the
meaning of robot observations, actions, coordinate frames, timestamps,
controller settings, embodiments, and data sources.

## Learner Context

The learner has a research background in XR/HCI, first-person sensing,
multimodal behavior data, temporal/spatial task segmentation, and adaptive
guidance. Explain unfamiliar robotics and robot-learning concepts clearly, but
do not oversimplify mathematics or engineering details. When useful, connect
new material to XR tracking, VR controllers, multimodal synchronization, and
task segmentation.

Use Chinese for explanations and progress discussions unless the learner asks
otherwise. Keep code, identifiers, comments, commit messages, and public-facing
repository documentation in English.

## Sources of Truth

Use these files in this order:

1. `notes/progress.md` — current verified state and immediate next steps.
2. `docs/roadmap_v3.md` — lesson order, scope, and acceptance direction.
3. `notes/concepts.md` — concepts already established and recurring pitfalls.
4. `README.md` — public project overview and reproducibility entry point.

If chat history conflicts with these files, prefer the repository files. Update
`notes/progress.md` whenever verified project status changes.

## Working Rules

- Inspect the relevant code, data schema, and Git diff before editing.
- Make the smallest change that closes the current acceptance gap.
- Do not mark a task complete merely because code was written.
- Require execution evidence, read-back validation, or a reproducible test.
- Distinguish observed facts from assumptions and proposed designs.
- Do not silently change observation semantics, action semantics, controller
  mode, coordinate frame, sampling frequency, or timestamp conventions.
- Do not treat equal tensor shapes as evidence that datasets are compatible.
- Keep simulator privileged state separate from deployable observations.
- Treat random rollouts as pipeline fixtures, not expert demonstrations.
- Prefer closed-loop task success over offline loss as the final policy metric.

## Dataset Contract

Every dataset or converter should document, where applicable:

- environment and task ID;
- robot embodiment;
- observation fields and deployability;
- action space and representation;
- action coordinate frame;
- absolute, relative, or delta semantics;
- dimension and units;
- controller/control mode;
- control and sampling frequency;
- timestamp origin and alignment;
- source domain (`sim`, `vr`, or `real`);
- software versions and generation seed.

Validate shapes, dtypes, NaN/Inf, numerical ranges, episode boundaries, frame
counts, timestamps, and the relationship between `T` actions and the state
sequence before using a dataset for training.

## Current Lesson 2 Completion Gate

Lesson 2 remains in progress until all of the following are demonstrated:

1. A source trajectory can be replayed and its success/failure is known.
2. Observation/action pairing and the `T` versus `T+1` convention are verified.
3. ManiSkill HDF5 converts successfully to the installed LeRobot v3 API.
4. The converted dataset can be loaded in a fresh process.
5. Episode count, frame count, FPS, feature shapes, and boundary frames pass
   inspection.
6. `scripts/inspect_robot_dataset.py` performs reusable dataset checks.
7. Action specification and source timestamps are preserved or explicitly
   documented.

## Safety and Credentials

- Work only inside the attached repository unless the user explicitly asks for
  access elsewhere.
- Do not read, print, search, or expose `.env` files, tokens, credential stores,
  SSH keys, browser profiles, or global configuration directories.
- Do not run broad credential-revealing commands such as `env` or `printenv`.
- Use `.env.example` with placeholders when configuration examples are needed.
- Never commit datasets, checkpoints, caches, videos, or credentials unless a
  small, explicitly approved test fixture is intentionally tracked.
- Ask before network uploads, publishing, force pushes, history rewrites, or
  destructive operations.

These instructions reduce accidental access but are not a security boundary;
operating-system permissions and sandbox configuration remain authoritative.

## Progress Update Protocol

After a meaningful learning or implementation session:

1. Record commands/tests actually run and their outcomes.
2. Update the checklist and evidence in `notes/progress.md`.
3. Add durable conceptual conclusions to `notes/concepts.md`.
4. Update `README.md` only when the public-facing project state changes.
5. Update the roadmap status without rewriting the planned curriculum.

Do not create date-suffixed progress files. Maintain `notes/progress.md` as the
single current progress record and rely on Git history for older versions.
