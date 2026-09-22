# Project Guidance for AI Coding Agents

## Project Mission

This repository is a long-term embodied-AI learning and engineering project.
Its immediate engineering path is:

`ManiSkill → trajectory/HDF5 → LeRobot → policy training → closed-loop evaluation → data iteration`

The goal is not only to make code run. Every experiment must preserve the
meaning of robot observations, actions, coordinate frames, timestamps,
controller settings, embodiments, and data sources.

The long-term research direction is **human-centered embodied agents for
long-horizon tasks**, with an emphasis on procedural task intelligence rather
than pure low-level robot control:

`multimodal observation → policy/VLA → persistent task state → task graph and memory → task world model → planning, recovery, and human-agent collaboration`

ManiSkill and robot-learning pipelines are the experimental foundation, not the
final research identity. The intended research contribution is the layer that
helps an embodied agent understand task structure, track what has happened,
reason about what remains, detect uncertainty or failure, and decide whether to
act, guide a human, ask, wait, recover, or escalate.

## Learner Context

The learner has a research background in XR/HCI, first-person sensing,
multimodal behavior data, temporal/spatial task segmentation, and adaptive
guidance. Explain unfamiliar robotics and robot-learning concepts clearly, but
do not oversimplify mathematics or engineering details. When useful, connect
new material to XR tracking, VR controllers, multimodal synchronization, and
task segmentation. Treat dependency-aware guidance, gaze/attention modelling,
and user-state estimation as foundations for embodied task intelligence and
human-agent collaboration—not as unrelated prior work.

Use Chinese for explanations and progress discussions unless the learner asks
otherwise. Keep code, identifiers, comments, commit messages, and public-facing
repository documentation in English.

### Notebook documentation language

Notebook **markdown** cells are written in **Chinese**, with technical terms kept
in English. This is a durable learner preference, not a one-off request.

- Translate the prose; keep the terms. Examples of terms that stay in English:
  `observation`, `state`, `action`, `policy`, `planner`, `controller`, `dataset`,
  `Dataset`, `DataLoader`, `batch`, `epoch`, `loss`, `MSE`, `MLP`, `Behavior
  Cloning (BC)`, `trajectory`, `replay`, `rollout`, `schema`, `frame`,
  `timestamp`, `episode`, `train/validation`, `overfitting`, `underfitting`,
  `distribution shift`, `privileged state`, `qpos`, `qvel`, `TCP`, `gripper`,
  `delta`, `absolute`, `normalized`, `seed`, `HDF5`, `LeRobot`, `ManiSkill`,
  `PyTorch`.
- Keep **code identifiers, file paths, attribute names, API names, numbers, and
  units** exactly as they are, including inside prose.
- Keep markdown structure identical: headings, numbering, tables, code fences,
  lists, links, emphasis. Code fence bodies are never translated.
- Keep code cells, their outputs, and `execution_count` untouched. Notebook
  outputs are execution records, so English stdout stays English.
- On first use, a Chinese gloss followed by the English term is preferred, e.g.
  动作（action）、行为克隆（Behavior Cloning, BC）.
- English remains correct for `README.md`, `notes/*.md`, `docs/*.md`, code
  comments, and commit messages.

## Sources of Truth

Use these files in this order:

1. `notes/progress.md` — current verified state and immediate next steps.
2. `docs/roadmap_v3.md` — lesson order, scope, and acceptance direction.
3. `notes/concepts.md` — concepts already established and recurring pitfalls.
4. `README.md` — public project overview and reproducibility entry point.

If chat history conflicts with these files, prefer the repository files. Update
`notes/progress.md` whenever verified project status changes.

## Learning and Research Priorities

Use the following order when choosing the next lesson, implementation task, or
reading. Do not skip an unfinished higher-priority gate merely because a later
topic is more fashionable.

### P0 — Close the policy learning loop

1. Complete reproducible `train/validation` splits and baseline Behavior
   Cloning (BC) training.
2. Run the learned policy in closed loop and measure task success, not only
   offline loss.
3. Diagnose `distribution shift`, `covariate shift`, compounding error, and
   failure cases from rollout evidence.
4. Understand the deep-learning foundations needed to read and modify policy
   code: tensor and batch semantics, loss/backpropagation, embeddings,
   self-/cross-attention, causal masks, positional encoding, residual paths,
   LayerNorm, fine-tuning, and LoRA.
5. Build a connected understanding of action-policy families:
   `BC → ACT/action chunking → Diffusion Policy → Flow Matching`.
6. Study VLA architecture through concrete systems such as OpenVLA, π0, and
   π0.5: visual/language/state representations, action heads or action experts,
   discrete action tokens versus continuous actions, action chunks, pretraining,
   and post-training.

### P1 — Build procedural task intelligence

1. Represent tasks using steps/subtasks, dependencies, preconditions, effects,
   alternative paths, completion conditions, and safety constraints.
2. Separate a procedural `task graph` from the inferred current `task state`.
3. Treat task-state estimation as partial observability: observation is not the
   same as world state. Maintain beliefs about what happened, what is happening,
   and what remains.
4. Combine procedural memory with episodic memory, persistent object state, and
   task history.
5. Develop a task-centric world model over objects, task state, and human state
   before pursuing pixel-level future-video generation.

### P2 — Turn task intelligence into collaboration

Study planning and replanning, failure detection and recovery, calibrated
uncertainty, selective intervention, shared task state, initiative, handover,
and shared autonomy. The agent's available decisions may include `act`, `guide`,
`ask`, `wait`, `recover`, and `escalate`.

### Deprioritized unless an experiment requires them

Do not turn ROS 2, SLAM, large-scale reinforcement learning, Isaac Lab,
low-level control, complex motion planning, or Sim2Real into the main learning
track by default. Introduce them just in time when they close a concrete
experimental or research gap.

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
- Do not confuse a low imitation-learning loss with deployable policy ability.
- Explicitly distinguish `observation`, latent/world `state`, and belief/task
  state. An object leaving the camera view is not evidence that it ceased to
  exist.
- Explicitly distinguish a static task graph from online task-state estimation.
- When explaining a model, trace the full information path from input fields to
  representation, policy/action head, executed action, and new observation.
- Prefer concept-driven reproduction over checkpoint-only reproduction: the
  learner should be able to explain why the architecture, objective, and action
  representation are appropriate.
- Connect new robot-learning material back to the long-term task-intelligence
  research question whenever the connection is real; do not force superficial
  XR analogies.

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

## Current BC Closed-Loop Gate

Do not treat baseline BC as complete until all of the following are evidenced:

1. Expert demonstrations, rather than random pipeline fixtures, are used.
2. The split is performed by episode or trajectory, with no frame leakage
   between training and validation data.
3. Input normalization and action scaling are fitted on training data only and
   saved for inference.
4. Training and validation loss curves, seeds, configuration, and checkpoint
   selection are recorded.
5. The checkpoint can be loaded in a fresh process and produces actions with
   the expected shape, range, units, frame, and control semantics.
6. The policy executes in ManiSkill closed loop under the same observation and
   controller contract used for training.
7. Evaluation reports task success rate across multiple seeded episodes, plus
   representative failure modes—not just one successful video.
8. At least one failure is analysed in terms of state-distribution shift,
   compounding error, action semantics, or insufficient observation history.

## Research Vocabulary and Framing

Use the following framing consistently in research notes and planning:

- Preferred umbrella term: **human-centered embodied agents**.
- Core technical identity: **procedural task intelligence** or **embodied task
  intelligence**.
- Primary problem: learning, representing, and tracking long-horizon task
  structure from multimodal human demonstrations.
- Core agent state: persistent world/object state + task state + human state.
- Core memories: procedural memory + episodic memory.
- Preferred world-model starting point: task-state transitions
  `p(s_{t+1} | s_t, a_t)`, not raw future-video generation.
- Desired outcome: reliable, explainable, uncertainty-aware assistance and
  recovery across AR/VR agents, robots, and mixed human-agent teams.

Avoid describing the research only as an “AR guidance system” or positioning
the learner as simply changing fields from XR to robotics. The coherent line is
multimodal human observation → task understanding → adaptive embodied
assistance.

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
