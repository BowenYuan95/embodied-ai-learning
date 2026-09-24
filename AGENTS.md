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

## Teaching Protocol

The learner is here to learn, not only to obtain working code. Every
recommendation, proposal, or correction must carry its teaching dimension.
When suggesting what to do next, say:

1. **What it teaches** — the concepts the step exercises.
2. **Why now** — how it fits the P0/P1/P2 order and what it unblocks.
3. **Acceptance evidence** — what will demonstrate it worked, and what it will
   *not* prove.
4. **Self-check questions** — what the learner should be able to explain in their
   own words afterwards.
5. **Research connection** — how it feeds the long-term direction, when that
   connection is real rather than decorative.

Further expectations:

- Explain the problem a technique solves and what fails without it, before
  showing the technique. Build the mental model, not only the code path.
- Prefer concept-driven reproduction over checkpoint-only reproduction: running a
  model is not the goal; explaining its architecture, objective, and action
  representation is.
- When correcting the learner, state the corrected principle, not only the
  corrected value, and say why the original reasoning was close but incomplete.
- When several next steps are plausible, compare them by learning value and
  sequence them, rather than listing options without a recommendation.
- Keep the learner's existing strengths (data semantics, task segmentation,
  human-state modelling) in the loop; connect new material to what they already
  understand instead of assuming a robotics background.
- Do not let a step end at "it runs". Close with what was learned, what remains
  unexplained, and the next question worth asking.

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

## Execution Discipline: Small Steps and Observable Progress

Long work must be **split into small steps** and must **show progress while it runs**. A
command or notebook cell that runs for minutes with no output is indistinguishable from a
hang, and a hang costs a round trip, an abort, and often unsaved work.

### Split the work

- **Verify before scaling.** Run the smallest version first (one arm, a few epochs, one
  task), read its output, and only then launch the full run.
- **One unit of work per notebook cell.** A training arm, a sweep, or a data pass gets its
  own cell, so an error or an interrupt costs one unit rather than the whole notebook.
- **Never bundle build + execute + long compute into a single call.** Building an artifact
  and then running it for minutes in the same command hides which step failed.

### Make progress visible

- **Never silence a long loop.** Training loops, sweeps, and data passes must emit progress
  — per arm, per epoch block, or per N items — with an explicit `flush=True`. A
  `verbose_every=0`-style switch belongs in tests, not in a cell that runs for minutes.
- **Report each unit as it finishes**, not only the total: print the arm's result at the
  moment it completes, so partial results survive an interrupt and are readable while the
  rest still runs.
- **Account for the capture boundary.** `nbclient` captures cell stdout into the notebook,
  so a background execution prints nothing to the terminal until it finishes. To watch a run
  from outside, execute cell by cell and print a heartbeat to stderr after each cell.
  `scripts/run_notebook_observable.py` does this and saves after every cell.
- **Persist partial results.** Write artifacts (metrics, checkpoints, logs) as each step
  completes, not only at the end. A run that writes once at the very end loses everything
  when it is interrupted.
- **Do not hide a progressive log behind a buffering filter.** `cmd 2>&1 | grep -v noise`
  block-buffers, so the terminal shows nothing until the command exits — the same "looks
  hung" failure in a new disguise. Use `grep --line-buffered` (or `stdbuf -oL`), or let the
  runner write to stderr without a pipe.

### Run long work in the background

- Anything expected to take more than about a minute runs as a **managed background job**,
  never as a blocking foreground call.
- Before starting long work, check whether the learner's JupyterLab kernel is already running
  the same notebook. Two writers on one notebook file destroy each other's state.

### GPU access and sandbox escalation

- **Do not escalate to reach the GPU by default.** Full sandbox access is granted one command
  at a time and **never persists** — do not treat a previous approval as standing permission.
- The agent's shell is denied `/dev/nvidia*` even though the nodes are mode `666`, so CUDA
  silently falls back to CPU there, while the learner's interactive Jupyter kernel has GPU
  access. The same notebook can therefore run on different hardware depending on who starts it.
- **Prefer asking the learner to run the cells**, or ask before each escalation. When a GPU run
  is genuinely needed, say what it is for and how long it will take.
- **Record the device with every result.** Print it inside the notebook (the preamble does),
  because float reduction order differs between backends and the numbers change in the second
  decimal. A number without its device is not reproducible.

### After an interrupt

- **Verify process state before continuing**: look for orphaned processes and for partially
  written files.
- **Identify a process by its full command line and start time.** Never infer identity from
  a CPU or elapsed-time column, and never `pgrep -f` a pattern whose text also appears in
  your own command line — that matches the shell itself and reports a false positive.
- **Never kill a process that has not been positively identified as yours.** An interactive
  `ipykernel` under the Jupyter runtime directory belongs to the learner; killing it destroys
  their session.

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
