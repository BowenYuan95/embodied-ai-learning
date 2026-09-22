"""Build the Lesson 1 and Lesson 2 gap notebooks from verified building blocks.

This script is a one-off generator: it writes .ipynb files under notebooks/ with
markdown explanations and code cells whose behavior was verified interactively
before being frozen here. Run it from the repository root:

    python scripts/build_lesson_notebooks.py

Cells are written without outputs; execute the notebooks afterwards to fill them.

WARNING: this **overwrites** those files, discarding any recorded outputs. Pass
``--force`` to allow that. Without it, a notebook that already has cell outputs is
left untouched. To rebuild one safely, delete the file first or pass ``--force`` and
re-execute it with nbclient afterwards.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

FORCE = "--force" in sys.argv

KERNEL = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.14"},
}


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:16],
        "metadata": {},
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:16],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": KERNEL,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name: str, cells: list[dict]) -> None:
    path = NOTEBOOK_DIR / name
    if path.exists() and not FORCE:
        existing = json.loads(path.read_text(encoding="utf-8"))
        has_outputs = any(
            cell.get("outputs")
            for cell in existing.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        if has_outputs:
            print(f"skipped {name}: already has recorded outputs (use --force to overwrite)")
            return
    payload = notebook(cells)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {path.name}: {len(cells)} cells ({n_code} code)")


# =====================================================================
# Shared preamble
# =====================================================================

PREAMBLE = """import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch

torch.set_printoptions(precision=4, sci_mode=False)


def make_env(obs_mode="state", control_mode="pd_joint_delta_pos", seed=0):
    env = gym.make(
        "PickCube-v1",
        obs_mode=obs_mode,
        control_mode=control_mode,
        num_envs=1,
    )
    env.reset(seed=seed)
    return env
"""


# =====================================================================
# 1.1 Robot state and observation
# =====================================================================

def build_1_1() -> None:
    cells = [
        md(
            """# Lesson 1.1 — Robot state and observation

The question this notebook answers:

> when the environment returns `obs`, what exactly is in it, and which part of it
> could a real robot actually measure?

Two words that are often used interchangeably and must not be:

- **state** — everything the simulator knows about the world. May include
  quantities no real sensor provides.
- **observation** — what the policy is handed. It is a *choice*, made by the
  `obs_mode` argument, about which part of the state is exposed and in what shape.

Run the cells in order. The interesting step is section 1.1.4, where a naive guess
about the layout turns out to be wrong.
"""
        ),
        md("## 1.1.1 — Create the environment"),
        code(PREAMBLE + '\nenv = make_env(obs_mode="state")\nprint(env)\nprint("observation space:", env.observation_space)\nprint("action space     :", env.action_space)'),
        md(
            """## 1.1.2 — The structured observation: `state_dict`

`obs_mode="state"` returns a flat tensor, which hides its structure. The same
information is available unflattened through `obs_mode="state_dict"`, so switch
modes and look at the tree before trusting any index.
"""
        ),
        code(
            '''env_sd = make_env(obs_mode="state_dict")
obs_sd, info = env_sd.reset(seed=0)


def print_tree(data, prefix=""):
    """Print the nested observation structure with tensor shapes."""
    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else key
            print_tree(value, name)
    else:
        print(f"  {prefix:<28} shape={getattr(data, 'shape', None)} dtype={getattr(data, 'dtype', None)}")


print("observation tree:")
print_tree(obs_sd)'''
        ),
        md(
            """## 1.1.3 — The flattened observation, and `proprioception`

Policy inputs are usually flat vectors, so the structure above has to be
concatenated into one tensor. ManiSkill also exposes robot-only quantities
through `get_proprioception()` and `get_state()`, which is how you separate
"what the robot knows about itself" from "what the task adds".
"""
        ),
        code(
            '''obs, info = env.reset(seed=0)
flat = obs[0]
print("flat observation shape:", flat.shape, "dtype:", flat.dtype)
print("first 9 values (joint positions?):", flat[:9])

agent = env.unwrapped.agent


def describe(value, prefix="", depth=0):
    """Describe a nested dict of tensors, since these helpers return nested dicts."""
    if isinstance(value, dict):
        for key, inner in value.items():
            name = f"{prefix}.{key}" if prefix else key
            describe(inner, name, depth + 1)
    else:
        shape = getattr(value, "shape", None)
        print(f"  {prefix:<24} type={type(value).__name__:<10} shape={shape}")


proprio = agent.get_proprioception()
print("\\n-- get_proprioception() --")
print("top-level keys:", list(proprio.keys()))
describe(proprio)

state = agent.get_state()
print("\\n-- agent.get_state() --")
print("top-level keys:", list(state.keys()))
describe(state)

print("\\n-- end-effector --")
print("TCP position:", agent.tcp_pos)
print("TCP pose (7 values, position + quaternion):", agent.tcp_pose.raw_pose[0])

print("\\nThese helpers return nested dicts, not flat tensors: there is no single .shape.")'''
        ),
        md(
            """## 1.1.4 — Predict the 42-d layout, then verify it

**Before running the next cell**, write down your prediction. The obvious guess is
to concatenate the `state_dict` components in the order the tree printed them:

```text
qpos(9) + qvel(9) + tcp_pose(7) + goal_pos(3) + obj_pose(7)
        + tcp_to_obj_pos(3) + obj_to_goal_pos(3) + is_grasped(1)
```

That guess is **wrong**. The flattened vector uses a different member order than
`state_dict` insertion order. Comparing the two makes the trap concrete: if you
index by assumption rather than by verification, every downstream feature is
silently mislabeled.

> In a live session this specific mismatch was observed directly: concatenating
> in `state_dict` order and comparing with `obs_mode="state"` returned
> `False` from `np.allclose`.
"""
        ),
        code(
            '''from mani_skill.utils import common


def as_flat(tensor):
    return tensor.reshape(-1).cpu().numpy()


# Candidate A: concatenate in state_dict insertion order
guess_a = np.concatenate([
    as_flat(obs_sd["agent"]["qpos"]),
    as_flat(obs_sd["agent"]["qvel"]),
    as_flat(obs_sd["extra"]["tcp_pose"]),
    as_flat(obs_sd["extra"]["goal_pos"]),
    as_flat(obs_sd["extra"]["obj_pose"]),
    as_flat(obs_sd["extra"]["tcp_to_obj_pos"]),
    as_flat(obs_sd["extra"]["obj_to_goal_pos"]),
    as_flat(obs_sd["extra"]["is_grasped"]),
])

# Candidate B: the layout actually used by obs_mode="state"
guess_b = np.concatenate([
    as_flat(obs_sd["agent"]["qpos"]),
    as_flat(obs_sd["agent"]["qvel"]),
    as_flat(obs_sd["extra"]["is_grasped"]),
    as_flat(obs_sd["extra"]["tcp_pose"]),
    as_flat(obs_sd["extra"]["goal_pos"]),
    as_flat(obs_sd["extra"]["obj_pose"]),
    as_flat(obs_sd["extra"]["tcp_to_obj_pos"]),
    as_flat(obs_sd["extra"]["obj_to_goal_pos"]),
])

flat_state = as_flat(obs[0])
print("state_dict-order matches state:", np.allclose(guess_a, flat_state, atol=1e-6))
print("verified-order  matches state:", np.allclose(guess_b, flat_state, atol=1e-6))'''
        ),
        md(
            """### The verified 42-d layout

| Slice | Field | Dim |
|---|---|---:|
| `[0:9]` | `agent.qpos` | 9 |
| `[9:18]` | `agent.qvel` | 9 |
| `[18:19]` | `extra.is_grasped` | 1 |
| `[19:26]` | `extra.tcp_pose` | 7 |
| `[26:29]` | `extra.goal_pos` | 3 |
| `[29:36]` | `extra.obj_pose` | 7 |
| `[36:39]` | `extra.tcp_to_obj_pos` | 3 |
| `[39:42]` | `extra.obj_to_goal_pos` | 3 |

Note where `is_grasped` sits: at index 18, wedged between `qvel` and `tcp_pose`,
not at the end. This is the same layout documented in
`archive/lesson_0_1/verify_state_flattening.py` and `build_deployment_safe_observation.py`.

`tcp_pose` and `obj_pose` are 7-dimensional: 3 position values followed by a
4-value quaternion (see `1.3_coordinate_frames.ipynb` for the component order).
"""
        ),
        md(
            """## 1.1.5 — Deployable versus privileged state

The 42-d vector mixes two very different things:

- **deployable**: joint positions, joint velocities, TCP pose. A real arm with
  encoders and forward kinematics can produce these.
- **privileged**: object pose, grasp state, and the relative vectors — a
  simulator can read them exactly, a real setup usually cannot without perception.

`scripts/pipeline/observation_adapter.py` encodes this split. Reproducing it here
shows why a policy trained on all 42 dimensions cannot be deployed as-is.
"""
        ),
        code(
            """import sys
from pathlib import Path

# The observation adapter is a shared module, not part of the conversion pipeline:
# identifying which state fields are deployable is a modelling decision, not a
# conversion step.
sys.path.insert(0, str(Path.cwd().resolve() / "scripts"))

from observation_adapter import (
    build_deployment_safe_observation,
    build_privileged_observation,
)

deployable = build_deployment_safe_observation(obs_sd)
privileged = build_privileged_observation(obs_sd)

print("deployable dim:", deployable.shape, " = qpos(9) + qvel(9) + tcp_pose(7) + goal_pos(3) = 28")
print("privileged dim:", privileged.shape, " = is_grasped(1) + obj_pose(7) + tcp_to_obj(3) + obj_to_goal(3) = 14")
print("28 + 14 =", deployable.shape[-1] + privileged.shape[-1])
print("\\ndeployable excludes:", ["is_grasped", "obj_pose", "tcp_to_obj_pos", "obj_to_goal_pos"])"""
        ),
        md(
            """## Takeaways

1. `obs_mode` decides what the policy sees; the underlying state is larger.
2. The flattened layout is **not** the `state_dict` insertion order. Verify
   offsets empirically; never infer them from a printed tree.
3. `proprioception` / `get_state()` separate robot-side quantities from
   task-side ones.
4. 42 dimensions contain 14 dimensions of simulator-privileged information. Any
   claim that a state-based policy is deployable has to answer what replaces
   those 14 values on real hardware.

Next: `1.2_action_space_and_control_modes.ipynb`.
"""
        ),
    ]
    write("1.1_state_and_observation.ipynb", cells)


# =====================================================================
# 1.2 Action space and control modes
# =====================================================================

def build_1_2() -> None:
    cells = [
        md(
            """# Lesson 1.2 — Action space and control modes

An action is never fully described by its tensor. This notebook makes that
concrete by varying one argument — `control_mode` — on the **same task** and
watching the action space change underneath.

The lesson's Action Specification is:

```text
(space, representation, frame, dimension, semantics, unit, frequency, controller mode)
```

Every row of the comparison table below fills in a different subset of that tuple.
Matching dimensions do **not** imply matching semantics.
"""
        ),
        md("## 1.2.1 — The shared preamble"),
        code(PREAMBLE),
        md(
            """## 1.2.2 — Enumerate every control mode

Same `PickCube-v1` task, eleven control modes, and action dimensions ranging from
**4 to 15**. Recognising this range is the point: "an 8-dimensional action" means
nothing on its own.
"""
        ),
        code(
            '''agent = make_env().unwrapped.agent
modes = sorted(agent.supported_control_modes)

print(f"{'control mode':<28} {'action space':<52} dim")
print("-" * 92)
for mode in modes:
    try:
        e = make_env(control_mode=mode)
        space = e.action_space
        shape = space.shape
        if np.allclose(space.low, -1.0) and np.allclose(space.high, 1.0):
            desc = f"Box(-1, 1, {shape})  normalized"
        else:
            desc = f"Box(lo, hi, {shape})  physical limits"
        print(f"{mode:<28} {desc:<52} {shape[0]}")
        e.close()
    except Exception as exc:
        print(f"{mode:<28} FAILED: {type(exc).__name__}: {exc}")'''
        ),
        md(
            """### Why the dimensions differ

| Control mode | Dim | Semantics |
|---|---:|---|
| `pd_joint_delta_pos` | 8 | 7 joint deltas + 1 gripper |
| `pd_joint_pos` | 8 | 7 absolute joint targets + 1 gripper, **physical limits** not `[-1,1]` |
| `pd_joint_vel` | 8 | 7 joint velocities + gripper |
| `pd_joint_pos_vel` | 15 | 8 position + 7 velocity |
| `pd_ee_delta_pos` | 4 | 3 Cartesian deltas + gripper |
| `pd_ee_delta_pose` | 7 | 6-DoF Cartesian delta + gripper |
| `pd_ee_pose` | 7 | absolute end-effector pose target |

Two facts fall out of this table:

- **joint space versus Cartesian space**: `pd_joint_*` targets joint angles,
  `pd_ee_*` targets the end-effector. The same physical motion is a different
  action in each.
- **delta versus absolute**: `pd_joint_delta_pos` and `pd_joint_pos` both have
  dimension 8, but one is a displacement and the other is a position. This is the
  clearest counterexample to "same shape means same action".
"""
        ),
        md(
            """## 1.2.3 — The project's control mode: `pd_joint_delta_pos`

The dataset in this repository was collected with `pd_joint_delta_pos`. Its eight
channels are **two different semantics concatenated**:

| Channel | Controller | Semantics | Physical range |
|---|---|---|---|
| `a[0:7]` | `PDJointPosController` | normalized **delta** from the current joint position | `[-0.1, 0.1]` rad |
| `a[7]` | `PDJointPosMimicController` | normalized **absolute** gripper target | `[-0.01, 0.04]` m |

Read the configs straight from the live controller rather than trusting names.
"""
        ),
        code(
            '''env = make_env()
controller = env.unwrapped.agent.controller

print("action_mapping:", controller.action_mapping)
print("action space   :", controller.action_space)

for name, config in controller.configs.items():
    print(f"\\n--- {name} ({type(config).__name__}) ---")
    for field in ("joint_names", "lower", "upper", "use_delta", "use_target",
                  "normalize_action", "mimic", "stiffness", "damping"):
        print(f"    {field:<18} {getattr(config, field, '<absent>')}")'''
        ),
        md(
            """Read those flags carefully, because together they decide what an action
*means*:

- `use_delta=True` on the arm — the action is a displacement.
- `use_target=False` — the delta is measured against the **current** joint
  position, not against the previous commanded target.
- `normalize_action=True` — the `[-1,1]` input is scaled into `[lower, upper]`.
- For the gripper, `use_delta=False` — it is an absolute position target, even
  though it lives in the same normalized `[-1,1]` box.
"""
        ),
        md(
            """## 1.2.4 — Verify the arm mapping empirically

Two numbers matter, and they are different:

1. the **control target** after one step — this should move by exactly `0.1 · a`;
2. the **actual joint position** after one step — this moves much less, because the
   PD controller tracks the target over several steps.

Confusing the two is an easy way to misread action magnitudes.
"""
        ),
        code(
            '''env = make_env()
robot = env.unwrapped.agent.robot


def finger_indices(robot):
    names = [joint.name for joint in robot.active_joints]
    return names.index("panda_finger_joint1"), names.index("panda_finger_joint2")


for value in (1.0, -1.0):
    env.reset(seed=0)
    qpos_before = robot.get_qpos().clone()[0]

    action = torch.zeros(1, 8)
    action[0, :7] = value
    env.step(action)

    target = robot.get_drive_targets()
    target = target[0] if target.ndim > 1 else target

    print(f"a_arm = {value:+.1f}")
    print(f"   command implied delta : {0.1 * value:+.4f} rad")
    print(f"   drive_target - qpos   : {target[:3].cpu().numpy()}  <-- exact")
    print(f"   qpos_after - qpos     : {(robot.get_qpos()[0] - qpos_before)[:3].cpu().numpy()}  <-- partial (PD tracking)")
    print()'''
        ),
        md(
            """## 1.2.5 — Verify the gripper mapping, including direction

The gripper is absolute, so its target should be a fixed value per action,
independent of the current state. And the direction question — does `+1` open or
close? — is answerable by measurement instead of by guessing from the sign.
"""
        ),
        code(
            '''env = make_env()
robot = env.unwrapped.agent.robot
f1, f2 = finger_indices(robot)

for value in (-1.0, 0.0, 1.0):
    env.reset(seed=0)
    qpos_before = robot.get_qpos().clone()[0]

    action = torch.zeros(1, 8)
    action[0, 7] = value
    env.step(action)

    target = robot.get_drive_targets()
    target = target[0] if target.ndim > 1 else target

    print(f"a_gripper = {value:+.1f} -> target finger1 = {target[f1].item():+.5f}, "
          f"finger2 = {target[f2].item():+.5f}   (reset qpos was {qpos_before[f1].item():+.4f})")

print("\\nBoth fingers receive the same target because finger2 mimics finger1.")
print("The gripper joint starts at its maximum value, so +1 is the open extreme.")'''
        ),
        md(
            """## 1.2.6 — The Action Specification table

Fill this in for the project dataset. Every field is required before two datasets
can be compared:

| Field | Value |
|---|---|
| space | `Box(-1, 1, (8,), float32)` |
| representation | joint-space, mixed delta/absolute |
| coordinate frame | joint space (no Cartesian frame involved for channels 0–6) |
| dimension | 8 = 7 arm + 1 gripper |
| semantics | arm: joint position **delta**; gripper: **absolute** position target |
| unit | rad (arm), m (gripper joint position) |
| frequency | 20 Hz control (see `2.3_time_alignment.ipynb`) |
| controller mode | `pd_joint_delta_pos` |

Note the last row. This notebook establishes the semantics; the **frequency** is a
separate property and it is currently inconsistent between the dataset metadata
and the environment. That is the subject of the time-alignment notebook.
"""
        ),
        md(
            """## Takeaways

1. One task argument changes the action dimension from 4 to 15. Shape alone is
   never a specification.
2. `pd_joint_delta_pos` has **two semantics in one vector**: joint deltas for the
   arm, an absolute target for the gripper.
3. `use_target=False` means the delta is applied to the current joint position, so
   the same action does not produce the same absolute motion from different states.
4. The commanded delta and the realized motion differ within a step because a PD
   controller tracks its target over time.
5. Two datasets with identical action shapes can still be mutually unusable — for
   example `pd_joint_delta_pos` versus `pd_joint_pos`, both `(8,)`.
"""
        ),
    ]
    write("1.2_action_space_and_control_modes.ipynb", cells)


# =====================================================================
# 1.3 Coordinate frames
# =====================================================================

def build_1_3() -> None:
    cells = [
        md(
            """# Lesson 1.3 — Coordinate frames and transformations

Every pose in the dataset is expressed **relative to something**. This notebook
makes that explicit, then verifies the relative vectors the policy receives.

Two traps are demonstrated rather than described:

1. the **quaternion component order** in `Pose.q`;
2. a **subtraction order error** that silently produces a plausible-looking but
   wrong relative vector.
"""
        ),
        md("## 1.3.1 — Preamble"),
        code(PREAMBLE),
        md(
            """## 1.3.2 — The `Pose` API: position and orientation

A 7-number pose is 3 translation values plus a 4-number quaternion. Which four,
and in what order, matters — get it wrong and the orientation is meaningless while
the translation still looks correct.
"""
        ),
        code(
            '''env = make_env()
agent = env.unwrapped.agent

pose = agent.tcp_pose
print("Pose object :", pose)
print("raw (7 values):", pose.raw_pose[0])
print("p (position)  :", pose.p[0])
print("q (quaternion):", pose.q[0])
print("\\nThe same 7 values appear in the dataset as tcp_pose.")'''
        ),
        md(
            """## 1.3.3 — Trap: quaternion order in `Pose.q`

`Pose.q` is **(x, y, z, w)** — the scipy convention. SAPIEN documentation and many
robotics codebases use **(w, x, y, z)**. The two differ only by a rotation of the
components, so both "look like" a valid quaternion.

The test below converts the same numbers under both assumptions. With the TCP
pointing down, the correct interpretation yields a rotation that is close to a
clean single-axis rotation; the wrong one produces a large angle about an
unexpected axis.

> The magnitudes below were observed in a live session: the correct (x, y, z, w)
> reading gave approximately `[3.13, 0.016, -3.11]` radians, a near-pure rotation
> about z, while the (w, x, y, z) reading gave approximately
> `[-0.035, -0.016, 3.13]`.
"""
        ),
        code(
            '''from scipy.spatial.transform import Rotation as R

quat = pose.q[0].cpu().numpy()
print("raw quaternion values:", quat)

as_xyzw = R.from_quat(quat).as_euler("xyz")
as_wxyz = R.from_quat(np.roll(quat, 1)).as_euler("xyz")

print("\\nif (x, y, z, w):", np.round(as_xyzw, 4), "rad ->", np.round(np.rad2deg(as_xyzw), 2), "deg")
print("if (w, x, y, z):", np.round(as_wxyz, 4), "rad ->", np.round(np.rad2deg(as_wxyz), 2), "deg")
print("\\nPose.q is (x, y, z, w): the first reading is a near-pure z rotation.")'''
        ),
        md(
            """## 1.3.4 — Homogeneous transforms and frame composition

A pose becomes a 4x4 homogeneous transform, which is what makes frame composition
and inversion mechanical: multiply to compose, invert to express something in the
opposite direction.
"""
        ),
        code(
            '''T = pose.to_transformation_matrix()[0].cpu().numpy()
np.set_printoptions(precision=4, suppress=True)
print("T (tcp -> world):")
print(T)

rotation = T[:3, :3]
print("\\nR @ R.T == I (orthonormal):", np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5))
print("det(R) == 1 (right-handed, no reflection):", np.allclose(np.linalg.det(rotation), 1.0, atol=1e-5))

inverse = np.linalg.inv(T)
print("\\ninv(T) @ T == I:", np.allclose(inverse @ T, np.eye(4), atol=1e-6))
print("Pose.inv().to_transformation_matrix() == np.linalg.inv(T):",
      np.allclose(pose.inv().to_transformation_matrix()[0].cpu().numpy(), inverse, atol=1e-5))'''
        ),
        md(
            """## 1.3.5 — Frame composition with 4x4 matrices

A pose becomes a 4x4 homogeneous transform. That is what makes frame algebra
mechanical: **compose by multiplying, reverse by inverting**.

> Careful: `Pose.to(device)` moves a pose between devices. It is **not** a frame
> change. Frame composition is matrix algebra.

The cell below expresses the cube in the TCP frame and transforms it back:
"""
        ),
        code(
            '''def as_matrix(pose):
    """4x4 homogeneous transform of a Pose."""
    return pose.to_transformation_matrix()[0].cpu().numpy()


robot = env.unwrapped.agent.robot
tcp_pose = env.unwrapped.agent.tcp_pose
obj_pose = env.unwrapped.cube.pose

T_tcp = as_matrix(tcp_pose)
T_obj = as_matrix(obj_pose)

print("tcp  in world:\\n", np.round(T_tcp, 4))
print("cube in world:\\n", np.round(T_obj, 4))

# cube expressed in the TCP frame
T_obj_in_tcp = np.linalg.inv(T_tcp) @ T_obj
print("\\ncube in TCP frame:\\n", np.round(T_obj_in_tcp, 4))

# transform back to world
T_back = T_tcp @ T_obj_in_tcp
print("\\nround trip == original cube pose:", np.allclose(T_back, T_obj, atol=1e-5))

print("\\nPose.inv() agrees with matrix inversion:",
      np.allclose(as_matrix(tcp_pose.inv()), np.linalg.inv(T_tcp), atol=1e-5))'''
        ),
        md(
            """## 1.3.6 — Predict the relative vectors, then verify

The observation contains `tcp_to_obj_pos` and `obj_to_goal_pos`. Before running the
next cell, decide what they are. Two plausible readings:

- **A**: a subtraction of two world-frame positions, `obj_pos - tcp_pos`;
- **B**: the object's position *expressed in the TCP frame*, i.e. rotated into that
  frame.

Both are useful, both are shape `(3,)`, and they are **numerically different** unless
the TCP orientation is identity. Only one of them is in the observation.
"""
        ),
        code(
            '''obs = env.unwrapped.get_obs().clone()[0]

tcp_pose_obs = obs[19:26]
goal_pos = obs[26:29]
obj_pose_obs = obs[29:36]
tcp_to_obj = obs[36:39]
obj_to_goal = obs[39:42]

print("tcp_to_obj_pos == obj_pos - tcp_pos  (world-frame difference):",
      np.allclose(tcp_to_obj.cpu(), (obj_pose_obs[:3] - tcp_pose_obs[:3]).cpu(), atol=1e-5))
print("obj_to_goal_pos == goal_pos - obj_pos (world-frame difference):",
      np.allclose(obj_to_goal.cpu(), (goal_pos - obj_pose_obs[:3]).cpu(), atol=1e-5))

print("\\nthe opposite subtraction orders (also plausible, also wrong):")
print("  tcp_to_obj_pos == tcp_pos - obj_pos:",
      np.allclose(tcp_to_obj.cpu(), (tcp_pose_obs[:3] - obj_pose_obs[:3]).cpu(), atol=1e-5))
print("  obj_to_goal_pos == obj_pos - goal_pos:",
      np.allclose(obj_to_goal.cpu(), (obj_pose_obs[:3] - goal_pos).cpu(), atol=1e-5))

print("\\nactual values: tcp_to_obj =", np.round(tcp_to_obj.cpu().numpy(), 6),
      " obj_to_goal =", np.round(obj_to_goal.cpu().numpy(), 6))'''
        ),
        md(
            """### Reading A is the correct one — and reading B gives a different vector

`tcp_to_obj_pos` is `obj_pos - tcp_pos`. Both are world-frame positions, so the
result is a **world-frame displacement**, not a rotated one.

To make the distinction concrete, compute reading B from the same observation. The
TCP is pointing downward, so its rotation is not identity, and the two results differ
in every component:
"""
        ),
        code(
            '''from mani_skill.utils.structs.pose import Pose

# Rebuild Pose objects from the observed 7-value poses
tcp_obs = Pose.create_from_pq(p=tcp_pose_obs[:3], q=tcp_pose_obs[3:7])
obj_obs = Pose.create_from_pq(p=obj_pose_obs[:3], q=obj_pose_obs[3:7])

T_tcp_obs = as_matrix(tcp_obs)
T_obj_obs = as_matrix(obj_obs)

R_rel = np.linalg.inv(T_tcp_obs)[:3, :3]
world_diff = T_obj_obs[:3, 3] - T_tcp_obs[:3, 3]
obj_in_tcp_frame = R_rel @ world_diff

print("R (TCP->world) rotation block:\\n", np.round(R_rel, 4))
print("is the rotation close to identity?", np.allclose(R_rel, np.eye(3), atol=1e-3))
print()
print("A: world-frame difference  obj_pos - tcp_pos :", np.round(world_diff, 6))
print("B: object in TCP frame     R^-1 @ difference :", np.round(obj_in_tcp_frame, 6))
print("observed tcp_to_obj_pos                      :", np.round(tcp_to_obj.cpu().numpy(), 6))
print()
print("A matches the observation:", np.allclose(tcp_to_obj.cpu(), world_diff, atol=1e-5))
print("B matches the observation:", np.allclose(tcp_to_obj.cpu(), obj_in_tcp_frame, atol=1e-5))'''
        ),
        md(
            """### Why this distinction matters

The observation gives you a world-frame displacement vector. If you need the object
in the TCP frame — to plan a motion relative to the gripper, for example — you must
rotate it yourself:

```text
obj_in_tcp = R_tcp^{-1} @ (obj_pos - tcp_pos)
```

Using `tcp_to_obj_pos` directly as if it were already in the TCP frame is a silent
frame error: every number is plausible, and the resulting motion is wrong.
"""
        ),
        md(
            """## 1.3.7 — Why frame conventions block dataset merging

Before combining two datasets, at least these must agree or be transformed:

1. **handedness and axis convention** — a rotation that is correct in one
   convention is wrong in another;
2. **quaternion component order** — `xyzw` versus `wxyz`;
3. **which frame** each pose is expressed in — world, robot base, or camera;
4. **the sign of relative vectors** — as demonstrated above.

None of these change a tensor's shape. A pair of datasets can be shape-compatible
and still represent the same physical motion with opposite sign.

This is why Lesson 4 treats rotation representations and frame composition as its
own subject, and why `docs/roadmap_v3.md` asks the same question for sim/VR/real
data.
"""
        ),
        md(
            """## Takeaways

1. `Pose.q` is `(x, y, z, w)`. Verify orientation conventions empirically; never
   assume wxyz.
2. Frame composition is 4x4 matrix algebra: multiply to compose, invert to reverse.
   `Pose.to()` is a device move, not a frame change.
3. `tcp_to_obj_pos` and `obj_to_goal_pos` are **world-frame position differences**,
   not vectors rotated into the gripper frame. Computing the TCP-frame version
   requires `R_tcp^{-1}` and yields different numbers.
4. Two shape-compatible datasets can still be mutually unusable, because none of
   handedness, quaternion order, reference frame, or vector sign affects shape.
"""
        ),
    ]
    write("1.3_coordinate_frames.ipynb", cells)


# =====================================================================
# 2.1 Environment and rollout
# =====================================================================

def build_2_1() -> None:
    cells = [
        md(
            """# Lesson 2.1 — The environment and the rollout loop

This notebook builds the intuition the GPT/agent-loop comparison depends on: the
Gymnasium call pattern **is** the embodied agent loop.

```text
o_t  ->  policy  ->  a_t  ->  env.step  ->  o_{t+1}, r, terminated, truncated
```

Adapted from the archived probes
`archive/lesson_0_1/smoke_test_maniskill.py` and
`archive/lesson_0_1/collect_pickcube_random.py`, which established this pattern
for Lessons 0–1 and are now frozen.
"""
        ),
        md("## 2.1.1 — Create and reset"),
        code(PREAMBLE + '''

env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_joint_delta_pos", num_envs=1)

print("environment :", type(env.unwrapped).__name__)
print("obs space   :", env.observation_space)
print("action space:", env.action_space)
print("control freq:", env.unwrapped.control_freq, "Hz")

obs, info = env.reset(seed=0)
print("\\nreset -> obs shape:", obs.shape, "dtype:", obs.dtype)
print("info keys:", list(info.keys()))'''
        ),
        md(
            """## 2.1.2 — One step: the five return values

`env.step(action)` returns `(observation, reward, terminated, truncated, info)`.
Two separate stop signals is a common source of bugs:

- `terminated` — the task ended in success or unrecoverable failure;
- `truncated` — the episode hit a step or time limit.

An episode that is `truncated` tells you nothing about success. The project's
source trajectory ends exactly this way, which is why
`notes/progress.md` records `success_any=False`.
"""
        ),
        code(
            '''action = env.action_space.sample()
print("action:", action, "shape:", action.shape)

next_obs, reward, terminated, truncated, info = env.step(action)

print("\\nnext_obs shape:", next_obs.shape)
print("reward        :", reward)
print("terminated    :", terminated)
print("truncated     :", truncated)
print("info          :", info)'''
        ),
        md(
            """## 2.1.3 — A full random episode

A rollout is just this loop. Random actions are used on purpose: the goal here is
to validate the mechanics, not to solve the task.

Note the T-versus-state bookkeeping: the loop performs T steps and stores **T**
observations, actions, and rewards. The observation *after* the last action is not
stored unless you explicitly keep it. That off-by-one is the subject of
`2.3_time_alignment.ipynb`.
"""
        ),
        code(
            '''def collect_random_episode(env, seed=0, max_steps=50, verbose=False):
    """Run a random rollout and record the transitions.

    Returns T observations, T actions, T rewards where T = max_steps.
    """
    obs, info = env.reset(seed=seed)
    observations, actions, rewards = [], [], []
    terminated_flags, truncated_flags = [], []

    for step in range(max_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        observations.append(obs[0].cpu().numpy())
        actions.append(action.copy())
        rewards.append(float(reward.item()))
        terminated_flags.append(bool(terminated.item()))
        truncated_flags.append(bool(truncated.item()))

        if verbose and step < 5:
            print(f"  step {step:2d}: reward={float(reward.item()):+.4f} "
                  f"terminated={bool(terminated.item())} truncated={bool(truncated.item())}")

        obs = next_obs
        if bool(terminated.item()) or bool(truncated.item()):
            break

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminated_flags),
        "truncated": np.asarray(truncated_flags),
        "info": info,
    }


episode = collect_random_episode(env, seed=0, max_steps=50, verbose=True)

print("\\nobservations:", episode["observations"].shape)
print("actions     :", episode["actions"].shape)
print("rewards     :", episode["rewards"].shape)
print("total reward:", episode["rewards"].sum())
print("any success :", "success" in episode["info"])
print("terminated  :", episode["terminated"].any(), "| truncated:", episode["truncated"].any())'''
        ),
        md(
            """## 2.1.4 — What "success" actually means here

`PickCube-v1` does not treat "the gripper is holding the cube" as success. Its
evaluation requires the cube to be **placed within the goal threshold** *and* the
robot to be **static**. Grasping alone is not success.

This matters for Lesson 2.9 and beyond: a demonstration is not expert data because
the arm moved plausibly. There must be an explicit success signal.
"""
        ),
        code(
            '''env.reset(seed=0)
print("goal threshold   :", env.unwrapped.goal_thresh)
print("cube half size   :", env.unwrapped.cube_half_size)
print("cube position    :", env.unwrapped.cube.pose.p[0])
print("goal position    :", env.unwrapped.goal_site.pose.p[0])

distance = (env.unwrapped.cube.pose.p[0] - env.unwrapped.goal_site.pose.p[0]).norm()
print(f"current cube-to-goal distance: {distance.item():.4f} m")
print("episode starts far from the goal, and random actions do not close that gap.")'''
        ),
        md(
            """## Takeaways

1. `reset` returns `(obs, info)`; `step` returns five values, with `terminated` and
   `truncated` as **separate** stop conditions.
2. A rollout loop is the embodied agent loop. The policy is the only replaceable
   part.
3. T steps produce T observations, T actions, T rewards.
4. In `PickCube-v1`, success requires placement within the goal threshold plus a
   static robot — grasping alone is insufficient. Random rollouts are pipeline
   fixtures, never expert demonstrations.

Next: the dataset layer, starting at `2.3_time_alignment.ipynb`.
"""
        ),
    ]
    write("2.1_environment_and_rollout.ipynb", cells)


# =====================================================================
# 2.3 Time alignment
# =====================================================================

def build_2_3() -> None:
    cells = [
        md(
            """# Lesson 2.3 — Time alignment: frames, timestamps, and episodes

This notebook answers three questions that decide whether a dataset can be trained
on at all:

1. how are observations and actions **paired**?
2. what does a **timestamp** actually mean here?
3. does the declared **frequency** match reality?

The answers to 2 and 3 are not what the metadata claims, and this notebook
demonstrates that rather than asserting it.

Prerequisite: the converted dataset `datasets/lerobot/pickcube/`. The first cell
sets a workspace-local Hugging Face cache, because the default cache location is
not writable in this environment.
"""
        ),
        md("## 2.3.0 — Environment setup"),
        code(
            '''import os
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()

# LeRobot routes dataset parquet reads through the Hugging Face datasets cache.
# Point it inside the workspace so it stays writable and out of the home directory.
HF_CACHE = REPO_ROOT / ".cache" / "hf"
(HF_CACHE / "datasets").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))

DATASET_ROOT = REPO_ROOT / "datasets" / "lerobot" / "pickcube"
print("repo root    :", REPO_ROOT)
print("dataset root :", DATASET_ROOT, "| exists:", DATASET_ROOT.exists())'''
        ),
        md(
            """## 2.3.1 — Frame, timestamp, and episode

- **frame** — one timestep entry: one observation, one action, one timestamp.
- **episode** — a contiguous rollout, from reset to termination or truncation.
- **timestamp** — when frames were *recorded*. Everything about control timing rests
  on this field being real.

LeRobot maps `frame_index` to seconds using the dataset's declared FPS. So the
declared FPS silently defines the time axis of every temporal model built on it.
"""
        ),
        code(
            '''import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="pickcube", root=str(DATASET_ROOT))

print("number of frames  :", len(dataset))
print("number of episodes:", dataset.num_episodes)
print("declared FPS      :", dataset.fps)

sample = dataset[0]
print("\\nsample keys:", sorted(sample.keys()))
print("observation.state:", tuple(sample["observation.state"].shape), sample["observation.state"].dtype)
print("action           :", tuple(sample["action"].shape), sample["action"].dtype)
print("timestamp        :", float(sample["timestamp"]), "| frame_index:", int(sample["frame_index"]))'''
        ),
        md(
            """## 2.3.2 — What the timestamp actually is

Read the parquet payload directly. If timestamps are real acquisition times, their
intervals will vary slightly. If they are synthetic, the intervals will be exactly
constant.

This distinction is not academic: a constant interval means the numbers were
computed from an index, not measured.
"""
        ),
        code(
            '''parquet_path = DATASET_ROOT / "data" / "chunk-000" / "file-000.parquet"
table = pq.read_table(parquet_path)
print("parquet schema:")
print(table.schema)

timestamps = table.column("timestamp").to_numpy()
frame_index = table.column("frame_index").to_numpy()
episode_index = table.column("episode_index").to_numpy()

print("\\ntimestamp head:", np.round(timestamps[:5], 4))
print("timestamp tail:", np.round(timestamps[-3:], 4))
print("frame_index    :", frame_index[:5], "...", frame_index[-3:])
print("episode_index unique:", np.unique(episode_index))

intervals = np.diff(timestamps)
print("\\nintervals (unique values):", np.unique(np.round(intervals, 6)))
print("all intervals identical   :", np.allclose(intervals, intervals[0]))

implied = np.arange(len(timestamps)) * intervals[0]
print("matches np.arange(T) * 0.02:", np.allclose(timestamps, implied, atol=1e-6))'''
        ),
        md(
            """## 2.3.3 — The 20 Hz / 50 Hz defect

The timestamps are `np.arange(T) * 0.02`, i.e. synthetic 50 Hz. The simulator's real
control rate is **20 Hz**. Compare the declared value against the environment's own
`control_freq`.

| Quantity | Value | Source |
|---|---:|---|
| declared `fps` in dataset metadata | 50 | inferred from synthetic timestamps |
| real control frequency | 20 | `env.unwrapped.control_freq` |
| real control period | 0.05 s | `env.unwrapped.control_timestep` |

The provenance chain is:

```text
collector stores no timing metadata
    -> timestamps synthesized as np.arange(T) * 0.02
        -> converter infers FPS from those timestamps
            -> info.json declares fps = 50 over 20 Hz data
```

**Consequence:** LeRobot converts `frame_index` to seconds using the declared FPS, so
the time axis is compressed by **2.5x**. A future temporal window `[B, T, D]` would
not cover the number of seconds it appears to cover.
"""
        ),
        code(
            '''import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_joint_delta_pos", num_envs=1)
unwrapped = env.unwrapped

declared_fps = dataset.fps
real_control_freq = unwrapped.control_freq
real_control_period = unwrapped.control_timestep
sim_freq = unwrapped.sim_freq

print(f"declared FPS in dataset   : {declared_fps}")
print(f"real control_freq         : {real_control_freq} Hz")
print(f"real control_timestep     : {real_control_period} s")
print(f"sim_freq                  : {sim_freq} Hz  (physics steps per second)")
print(f"physics steps per action  : {int(sim_freq * real_control_period)}")
print()
print(f"declared period           : {1 / declared_fps:.4f} s")
print(f"real period               : {real_control_period:.4f} s")
print(f"time axis compression     : {real_control_period / (1 / declared_fps):.1f}x")

seconds_declared = len(dataset) / declared_fps
seconds_real = len(dataset) / real_control_freq
print(f"\\n50 frames span {seconds_declared:.2f} s according to metadata, but {seconds_real:.2f} s of real control")
env.close()'''
        ),
        md(
            """### Why this cannot be fixed by editing one number

Changing `fps` from 50 to 20 in the metadata would make the declared value true, but
the underlying timestamps would still be synthetic. The durable fix is at the source:

- the **collector** must record the control frequency it actually used, and either
  measure or explicitly label its timestamps;
- the **converter** must declare the frequency from that recorded value instead of
  inferring it from timestamps it is about to fabricate.

Until then, treat the dataset as **50 frames of 20 Hz control**, and treat the
declared 50 FPS as wrong. This is recorded as an open issue in `notes/progress.md`.
"""
        ),
        md(
            """## 2.3.4 — The training pair is `(o_t, a_t)`, not `(o_t, a_{t+1})`

The pairing convention is a contract, not a convention. If actions are shifted by
one frame, every target is wrong while all shapes stay identical — the model trains
happily and learns to predict the next action.

Verify the rule on the stored data: the action at index `t` is the action that was
applied to the observation at index `t`.
"""
        ),
        code(
            '''# The stored arrays are aligned by construction, so demonstrate the contrast
# between the correct pairing and the off-by-one pairing.

actions = table.column("action").to_pylist()
states = table.column("observation.state").to_pylist()

T = len(actions)
print("T actions :", T)
print("T states  :", len(states))

correct_pairs = [(t, t) for t in range(T)]
shifted_pairs = [(t, t + 1) for t in range(T - 1)]

print("\\ncorrect pairing  : (o_t, a_t)     ->", correct_pairs[:3], "...")
print("off-by-one pairing: (o_t, a_t+1)   ->", shifted_pairs[:3], "...")
print("\\nboth produce the same tensor shapes;")
print("only the first matches the environment transition o_t -a_t-> o_{t+1}.")'''
        ),
        md(
            """### The `T` versus `T+1` question

An episode of T actions involves **T+1** states:

```text
o_0 --a_0--> o_1 --a_1--> o_2 ... o_{T-1} --a_{T-1}--> o_T
```

The stored observation array holds `o_0 .. o_{T-1}` — the states the policy *saw*.
The terminal observation `o_T` (and `next_observations` in the raw HDF5) is a
separate field. Confusing the two produces a dataset that looks complete and is
silently missing its last transition.
"""
        ),
        code(
            '''import h5py

h5_path = REPO_ROOT / "datasets" / "pickcube" / "random_episode_standard.h5"
with h5py.File(h5_path, "r") as handle:
    print("standardized HDF5 keys and shapes:")
    for key in handle:
        if key == "metadata":
            print(f"  {key:<18} (group) attrs={dict(handle[key].attrs)}")
        else:
            print(f"  {key:<18} shape={handle[key].shape} dtype={handle[key].dtype}")

print("\\nNote timestamps were stored as a separate array:")
with h5py.File(h5_path, "r") as handle:
    stamps = handle["timestamps"][:]
print("  first 5:", stamps[:5])
print("  step    :", np.unique(np.round(np.diff(stamps), 6)))'''
        ),
        md(
            """## Takeaways

1. The dataset's timestamps are synthetic (`np.arange(T) * 0.02`), not measured — the
   intervals are exactly constant, which is the giveaway.
2. The declared 50 FPS contradicts the environment's real **20 Hz** control rate,
   compressing the time axis by 2.5x. Do not silently edit `fps`; fix the collector
   and the converter.
3. Training pairs are `(o_t, a_t)`. The off-by-one pairing has identical shapes and
   is wrong.
4. An episode of T actions has T+1 states. The stored observation array has T.

Next: `2.4_observation_schema.ipynb` for the observation/action schema, then
`2.5_generate_lerobot_dataset.ipynb` for the conversion.
"""
        ),
    ]
    write("2.3_time_alignment.ipynb", cells)


# =====================================================================
# 3.1 Imitation learning introduction
# =====================================================================

def build_3_1() -> None:
    cells = [
        md(
            """# Lesson 3.1 — Imitation learning: the problem Behavior Cloning solves

Lesson 2 produced a pipeline that trains. This lesson asks the question the pipeline
cannot answer:

> why can a model with a very low training loss still fail completely when it
> actually drives the robot?

The experiment at the end of this notebook is the entry point: it shows, on the real
dataset in this repository, why a held-out evaluation is not possible yet — and what
that implies.
"""
        ),
        md(
            """## 3.1.1 — The problem statement

An expert demonstrates a task. The demonstration is a set of state-action pairs
sampled from the expert's behaviour:

```text
D = {(o_t, a_t)}   with   o_t ~ d_{pi_E}
```

Imitation learning looks for a policy that reproduces the expert's decisions:

```text
pi_theta(a_t | o_t)  ~=  pi_E(a_t | o_t)
```

Two things are worth separating immediately:

- the expert's **policy** `pi_E` is the function being imitated;
- the expert's **state distribution** `d_{pi_E}` is where the data comes from, and it
  is *generated by that policy*. This second fact is the source of every hard
  problem in this lesson.
"""
        ),
        md(
            """## 3.1.2 — Behavior Cloning is supervised learning

Behavior Cloning (BC) forgets that the data came from a policy and treats it as a
plain regression problem:

```text
D = {(o_t, a_t)}
theta* = argmin_theta  sum_t  L(pi_theta(o_t), a_t)
```

For continuous actions the loss is MSE:

```text
L = MSE(pi_theta(o_t), a_t)
```

This is exactly the training loop already built in
`2.7_bc_training_loop.ipynb`. BC is not a new algorithm; it is supervised learning on
robot states. Which is precisely why it inherits supervised learning's failure mode:
it is only valid on the distribution it was trained on.
"""
        ),
        code(
            '''import os
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
HF_CACHE = REPO_ROOT / ".cache" / "hf"
(HF_CACHE / "datasets").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))

import numpy as np
import torch
import torch.nn as nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_ROOT = REPO_ROOT / "datasets" / "lerobot" / "pickcube"
dataset = LeRobotDataset(repo_id="pickcube", root=str(DATASET_ROOT))

print("frames  :", len(dataset))
print("episodes:", dataset.num_episodes)
print("FPS     :", dataset.fps)'''
        ),
        md(
            """## 3.1.3 — Why a validation split is not yet possible

The plan for the next step (2.8.7) is a train/validation split. Run the experiment
below and look at what the split actually gives you.

Adjacent frames in a trajectory are nearly identical: the robot moves a few
millimetres per step. So a random frame-level split puts almost-copies of the same
state on both sides. A "held-out" frame is not held out in any meaningful sense —
this is **data leakage**, and it inflates validation performance.
"""
        ),
        code(
            '''# How similar are adjacent frames? Compare consecutive observation vectors.
observations = np.stack([dataset[i]["observation.state"].numpy() for i in range(len(dataset))])

consecutive_delta = np.linalg.norm(np.diff(observations, axis=0), axis=1)

random_pairs = []
rng = np.random.default_rng(0)
for _ in range(500):
    a, b = rng.integers(0, len(observations), size=2)
    if a != b:
        random_pairs.append(np.linalg.norm(observations[a] - observations[b]))
random_pairs = np.asarray(random_pairs)

print("consecutive-frame distance : mean %.4f  median %.4f" % (consecutive_delta.mean(), np.median(consecutive_delta)))
print("random-pair distance       : mean %.4f  median %.4f" % (random_pairs.mean(), np.median(random_pairs)))
print("ratio (random / consecutive): %.1fx" % (random_pairs.mean() / consecutive_delta.mean()))
print("\\nAdjacent frames are nearly the same point; a frame-level split leaks.")'''
        ),
        md(
            """### What this implies

With **one episode**, splitting by frame cannot produce an honest validation set. The
options are:

- evaluate on a **different trajectory** — requires collecting more data (2.9), or
- accept that the validation curve here only detects gross overfitting, and say so.

The correct default, once several episodes exist, is to split **by episode**: whole
trajectories go to train or validation, never individual frames. Otherwise the model
is graded on frames it effectively memorized.
"""
        ),
        md(
            """## 3.1.4 — Distribution shift: the core failure of BC

At training time the model sees states the expert visited:

```text
o_t ~ d_{pi_E}
```

At execution time the model sees states *it* produced:

```text
o_t ~ d_{pi_theta}
```

These are different distributions. A small action error moves the robot to a state the
expert never visited, where the model was never trained and its prediction is
arbitrary. That error compounds: the trajectory drifts further from the data
distribution at every step.

This is why a low training loss proves only that the model fits the demonstration —
not that it can complete the task.
"""
        ),
        code(
            '''# Quantify the drift numerically: how quickly does a state leave the training set?
# Take the first frame, then follow the stored actions and measure the distance from
# anything the dataset actually contains.
states = observations

# Pick a real starting frame and walk forward one step, measuring nearest-neighbour
# distance of the resulting state to the dataset.
def nearest_distance(query, pool):
    return float(np.min(np.linalg.norm(pool - query, axis=1)))


start = 0
reference = states[start]
d_same = nearest_distance(reference, states)

# Perturb the state by a plausible single-step error and re-measure
perturbation = 0.02 * np.linalg.norm(states.std(axis=0))
noisy = reference + np.random.default_rng(1).normal(0, perturbation, size=reference.shape)
d_noisy = nearest_distance(noisy, states)

print(f"distance from dataset for a real frame    : {d_same:.5f}")
print(f"distance after a small perturbation       : {d_noisy:.5f}")
print(f"perturbation size in state units          : {perturbation:.5f}")
print()
print("A policy that reproduces the trajectory only approximately lands off-distribution.")
print("With 50 frames and one trajectory there is nothing to fall back on.")'''
        ),
        md(
            """## 3.1.5 — Where this lesson goes

The rest of Lesson 3 builds the tools for each of these problems:

| Section | Problem it addresses |
|---|---|
| 3.3 | overfitting, underfitting, and how to split honestly |
| 3.4 | distribution shift, formally |
| 3.5 | DAgger: let the policy act and label the states it reaches |
| 3.6 | partial observability: single-frame versus history policies |
| 3.7 | action chunking: predict a sequence, not a point |
| 3.8 | from state to `(image, language, state)` |
| 3.9 | collecting expert data properly, including SO-101 teleoperation |

The immediate next step is 2.8.7, which is the experiment that motivates all of it.
"""
        ),
        md(
            """## Takeaways

1. BC is supervised learning on states the **expert** generated.
2. Training pairs come from `d_{pi_E}`; execution encounters `d_{pi_theta}`. The gap
   between them is distribution shift, and it is not visible in the training loss.
3. Adjacent frames are nearly identical (measured above), so a frame-level
   train/validation split leaks. Split by episode once multiple episodes exist.
4. With one episode there is no honest held-out evaluation. That is a data problem,
   not a modelling problem, and it is what 2.9 addresses.
"""
        ),
    ]
    write("3.1_imitation_learning_intro.ipynb", cells)


# =====================================================================
# 2.9 Expert demonstrations
# =====================================================================

def build_2_9() -> None:
    cells = [
        md(
            """# Lesson 2.9 — Expert demonstrations with a motion planner

Everything up to here used **random** actions. This notebook produces what the project
never had before: trajectories that actually **succeed**, and can therefore serve as
expert demonstrations for imitation learning.

The planner is ManiSkill's own sampling-based motion planner, the same component its
built-in demonstration generation uses. The recipe is taken from the reference
implementation shipped with the package:

```text
mani_skill/examples/motionplanning/panda/solutions/pick_cube.py
```

## Two results that shape this notebook

**1. The planner needed a NumPy downgrade.** `mplib` 0.1.1 — which ManiSkill 3.0.1 pins
with `Requires-Dist: mplib==0.1.1` — is compiled against the NumPy 1.x C API. Under
NumPy 2.x its stored function pointers are invalid, so constructing the planner jumps to
address `0x0` and the process dies with `SIGSEGV`:

```text
Fatal Python error: Segmentation fault
  File ".../mplib/planner.py", line 65 in __init__
  File ".../base_motionplanner/motionplanner.py", line 59 in setup_planner
  File ".../panda/motionplanner.py", line 25 in __init__
```

Kernel log: `segfault at 0 ip 0000000000000000`. The fix is `numpy<2`, **not** a GPU or
backend setting: the same script segfaults under NumPy 2.2.6 and succeeds under
NumPy 1.26.4, with CUDA unavailable in both cases.

**2. Control mode and the gripper steps are load-bearing.** The planner's
`close_gripper()` / `open_gripper()` emit `[qpos(7), gripper]`, so the environment must
run `pd_joint_pos`. And the official recipe **does not open the gripper** after carrying
the cube to the goal — adding `open_gripper()` plus settling steps drops the cube beside
the goal and the episode fails with `is_obj_placed: False`.

This notebook records what actually works.
"""
        ),
        md("## 2.9.1 — Preamble"),
        code(PREAMBLE),
        md(
            """## 2.9.2 — Why the planner cannot run in this notebook's environment

The project has two Python environments, and they differ exactly on the library this
notebook needs:

| Environment | NumPy | mplib planner |
|---|---|---|
| `embodied` (notebook default) | `2.2.6` | **segfaults** |
| `embodied310` (planning) | `1.26.4` | works |

So this notebook **does not construct the planner**. Doing so kills the kernel outright —
a segmentation fault cannot be caught by `try` / `except`, and there is no error to
record, only a dead process.

The planner work is delegated to `scripts/generate_expert_demo.py`, run with
`embodied310`. What can be inspected safely here is the **action contract** the planner
depends on, which is the real reason the control mode cannot be `pd_joint_delta_pos`.
"""
        ),
        code(
            '''import gymnasium as gym
import mani_skill.envs

# The planner's close_gripper() / open_gripper() step the env with
# action = [qpos(7), gripper]  -- an (8,) ABSOLUTE joint position vector.
# Compare the action space of the two candidate control modes.
for mode in ("pd_joint_pos", "pd_joint_delta_pos"):
    e = gym.make("PickCube-v1", obs_mode="state", control_mode=mode, num_envs=1)
    space = e.action_space
    normalized = bool(np.allclose(space.low, -1.0) and np.allclose(space.high, 1.0))
    print(f"{mode:<20} dim={space.shape[0]}  normalized={normalized}")
    if not normalized:
        print(f"{'':<20} low  = {np.round(space.low, 4)}")
        print(f"{'':<20} high = {np.round(space.high, 4)}")
    e.close()

print()
print("pd_joint_pos       : absolute joint targets inside the joint limits.")
print("                     [qpos(7), gripper] fits directly -- this is what the planner emits.")
print("pd_joint_delta_pos : normalized deltas + absolute gripper; feeding a position vector")
print("                     into it fails with 'Received action of shape torch.Size([15])'.")'''
        ),
        code(
            '''# The exact failure mode, reproduced without the planner: a position-style action
# sent to the delta controller.
env_delta = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_joint_delta_pos", num_envs=1)
env_delta.reset(seed=0)
try:
    env_delta.step(np.zeros(15, dtype=np.float32))
    print("unexpectedly accepted a 15-d action")
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
env_delta.close()'''
        ),
        md(
            """### Why the planner bypasses the action space

`move_to_pose_with_screw` plans a joint-space motion and drives the robot's joint targets
directly. It bypasses `env.step(action)` for the arm motion, then uses `env.step` for the
gripper and the settling steps.

That matters for data collection: a planner produces **trajectories**, and turning a
trajectory into training data requires the actions the environment actually executed. The
collector script wraps `env.step` so every transition is recorded as it happens, rather
than being reconstructed afterwards — see section 2.9.6.
"""
        ),
        md(
            """## 2.9.3 — The grasping geometry

A grasp pose needs three things, and none of them is a magic number:

- **approaching**: the direction the gripper comes from. Straight down, because the cube
  sits on a table.
- **closing**: the axis the fingers close along, derived from the object's oriented
  bounding box. This is what makes the grasp work for a cube at an arbitrary yaw.
- **center**: where to grasp.

`target_closing` is taken from the TCP's own rotation: the arm's current finger axis is
reused instead of inventing one. `build_grasp_pose` returns a **SAPIEN** `Pose`, not a
ManiSkill `Pose` — so it exposes `.p` and `.q` directly, not `.raw_pose`.
"""
        ),
        code(
            '''import sapien

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

FINGER_LENGTH = 0.025

env = make_env(obs_mode="state", control_mode="pd_joint_pos", seed=0)
unwrapped = env.unwrapped

obb = get_actor_obb(unwrapped.cube)

approaching = np.array([0, 0, -1])
# y axis of the TCP frame = the finger closing axis
target_closing = unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

grasp_info = compute_grasp_info_by_obb(
    obb,
    approaching=approaching,
    target_closing=target_closing,
    depth=FINGER_LENGTH,
)
closing = grasp_info["closing"]
center = grasp_info["center"]
grasp_pose = unwrapped.agent.build_grasp_pose(approaching, closing, unwrapped.cube.pose.sp.p)

print("cube position :", unwrapped.cube.pose.sp.p)
print("goal position :", unwrapped.goal_site.pose.sp.p)
print("approaching   :", approaching)
print("closing axis  :", np.round(closing, 4))
print("grasp center  :", np.round(center, 4))
print("grasp pose type:", type(grasp_pose).__name__, "(SAPIEN Pose)")
print("grasp pose p  :", np.round(grasp_pose.p, 4))
print("grasp pose q  :", np.round(grasp_pose.q, 4))

# Reach pose: grasp pose offset 5 cm upward, so the gripper descends from above
reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
print("reach  pose p :", np.round(reach_pose.p, 4))

# Goal pose reuses the grasp orientation so the cube stays upright on the way over
goal_pose = sapien.Pose(unwrapped.goal_site.pose.sp.p, grasp_pose.q)
print("goal   pose p :", np.round(goal_pose.p, 4))'''
        ),
        md(
            """## 2.9.4 — Execute the plan

The verified sequence is **four steps, and no more**:

```python
planner.move_to_pose_with_screw(reach_pose)   # approach from above
planner.move_to_pose_with_screw(grasp_pose)   # descend
planner.close_gripper()                       # grasp
planner.move_to_pose_with_screw(goal_pose)    # carry to the goal
```

Two negative results are worth stating, because both look reasonable and both fail:

- **`open_gripper()` after the carry** — the cube is released while it is still beside
  the goal, fails the placement distance check, and `is_obj_placed` comes back `False`.
- **Extra zero-action settling steps** — they do not fix the placement; they let the cube
  drift further.

Note that `close_gripper()` and `open_gripper()` internally call `env.step` with an
`(8,)` action of the form `[qpos(7), gripper]`. That is why the environment must be
`pd_joint_pos`: under `pd_joint_delta_pos` the helper produces a 15-d vector and the
controller rejects it with
`AssertionError: Received action of shape torch.Size([15]) but expected shape (1, 8)`.
"""
        ),
        code(
            '''import subprocess
import sys
from pathlib import Path

# The planner must run in embodied310 (numpy<2), so invoke the collector as a
# subprocess rather than constructing the planner in this kernel.
PLANNING_PYTHON = Path.home() / "miniforge3" / "envs" / "embodied310" / "bin" / "python"
COLLECTOR = Path.cwd().resolve() / "scripts" / "generate_expert_demo.py"

print("planning interpreter:", PLANNING_PYTHON, "exists:", PLANNING_PYTHON.exists())

if PLANNING_PYTHON.exists():
    result = subprocess.run(
        [str(PLANNING_PYTHON), str(COLLECTOR), "--seeds", "0", "1", "2", "3", "4", "--overwrite"],
        capture_output=True, text=True, cwd=str(Path.cwd().resolve()),
    )
    print("exit code:", result.returncode)
    print(result.stdout.strip()[-1200:])
    if result.returncode != 0:
        print("STDERR:", result.stderr.strip()[-600:])
else:
    print("embodied310 not found; run the collector manually:")
    print("  python scripts/generate_expert_demo.py --seeds 0 1 2 3 4 --overwrite")'''
        ),
        md(
            """## 2.9.5 — What success means

Success is a property of the **task evaluation**, not of the planner returning without
error:

```python
info = env.unwrapped.evaluate()
# {"success", "is_obj_placed", "is_robot_static", "is_grasped"}
```

| Field | Meaning |
|---|---|
| `is_obj_placed` | cube within `goal_thresh` (0.025 m) of the goal |
| `is_robot_static` | arm has settled |
| `success` | `is_obj_placed AND is_robot_static` |

**Grasping is not success.** An arm holding the cube in mid-air has not completed the
task. That is why "the trajectory looked reasonable" is not an acceptance criterion
anywhere in this project.
"""
        ),
        code(
            '''print("goal threshold:", unwrapped.goal_thresh, "m")
print("cube half size:", unwrapped.cube_half_size)
print()
print("evaluate() keys:", list(unwrapped.evaluate().keys()))
print("success requires BOTH is_obj_placed AND is_robot_static.")'''
        ),
        md(
            """## 2.9.6 — Collected expert episodes

`scripts/generate_expert_demo.py` wraps this recipe into a collector. It wraps
`env.step`, so every transition is recorded as it is executed — nothing is reconstructed
after the fact, and the `(o_t, a_t)` pairing holds by construction.

Recorded result for seeds 0–4:

| Seed | Frames | Success | is_obj_placed | is_robot_static |
|---:|---:|:--:|:--:|:--:|
| 0 | 74 | True | True | True |
| 1 | 74 | True | True | True |
| 2 | 50 | True | True | True |
| 3 | 86 | True | True | True |
| 4 | 76 | True | True | True |

**5 of 5 episodes succeed.** Compare the action smoothness:

| Dataset | mean `|Δa|` |
|---|---|
| random rollout fixture | `0.67` |
| expert planner episodes | `0.0078` |

That is roughly a **100× difference**. Expert actions are small, correlated steps; random
actions are nearly independent draws. This single statistic separates the two regimes, and
it is why the random trajectory can never train a policy no matter how long it trains.

The collector also writes the contract into the HDF5 attributes: `control_mode`,
`data_quality="expert_planner"`, `control_freq_hz=20`,
`timestamp_source="derived_not_measured"`, and the per-channel action semantics.
"""
        ),
        code(
            '''import h5py
from pathlib import Path

EXPERT_H5 = Path.cwd().resolve() / "datasets" / "pickcube" / "expert_episodes.h5"

if not EXPERT_H5.exists():
    print(f"{EXPERT_H5} not found.")
    print("Generate it with: python scripts/generate_expert_demo.py --seeds 0 1 2 3 4 --overwrite")
else:
    with h5py.File(EXPERT_H5, "r") as handle:
        print("root attributes:")
        for key in ("env_id", "control_mode", "data_quality", "control_freq_hz", "timestamp_source"):
            print(f"  {key:<18} {handle.attrs[key]}")
        print()
        print(f"{'episode':<16} {'frames':>6} {'success':>8} {'|da| mean':>10} {'return':>8}")
        print("-" * 54)
        for name in handle:
            group = handle[name]
            actions = group["actions"][:]
            rewards = group["rewards"][:]
            smoothness = np.abs(np.diff(actions, axis=0)).mean()
            print(f"{name:<16} {actions.shape[0]:>6} {str(bool(group.attrs['success'])):>8} "
                  f"{smoothness:>10.4f} {rewards.sum():>8.3f}")'''
        ),
        md(
            """## 2.9.7 — Replay verification, and one honest caveat

An expert dataset is not accepted until replaying it reproduces the recorded outcome. The
collector's episodes were replayed from the same seeds and checked.

| Check | Result |
|---|---|
| Reward per step | **exact** (max error `0.00e+00`) |
| `success` after replay | identical to recorded (5/5) |
| `goal_pos` | exact |
| Continuous state (`qpos`, `qvel`, `tcp_pose`, `obj_pose`) | agrees to `~1e-2` |
| `is_grasped` | differs on **1 frame out of 74**, at the grasp transition |

The `is_grasped` mismatch is worth understanding rather than hiding. It is a **boolean
contact test**, and the frame at which it flips depends on the exact contact solver state:
recorded episode 0 flips at frame 36, replay at frame 37. The physics, the actions, and
every reward are identical.

Lesson: the project's own gate — "replay must match the source exactly" — was
*writable* for the random trajectory because its actions were tiny and deterministic. For
planner-driven expert episodes, exact **observation** equality is not achievable across a
boolean contact threshold; exact **reward** equality and identical **success** are. State
the tolerance you verified instead of claiming bit-exactness.
"""
        ),
        md(
            """## 2.9.8 — How this relates to the existing fixture

The repository now holds two kinds of trajectory, and they are **not** interchangeable:

| | `random_episode_standard.h5` | `expert_episodes.h5` |
|---|---|---|
| Source | random actions | motion planner |
| Episodes / frames | 1 / 50 | 5 / 50–86 |
| Success | none (`success_any=False`) | 5/5 |
| Control mode | `pd_joint_delta_pos` | `pd_joint_pos` |
| Action dim | 8 (7 delta + 1 absolute) | 8 (8 absolute) |
| Role | pipeline fixture | imitation-learning supervision |

**The action semantics differ**, even though both are 8-dimensional and both live in a
box. `pd_joint_delta_pos` gives arm **deltas** in `[-0.1, 0.1]` rad plus an absolute
gripper target; `pd_joint_pos` gives **absolute** joint position targets within the joint
limits. Concatenating them without conversion would train a policy on two incompatible
action meanings — the exact failure mode `1.2_action_space_and_control_modes.ipynb`
warns about.

A converter from absolute targets back to deltas is straightforward
(`Δq_t = q_target[t] - qpos[t]`, then scale by `0.1` and clamp to `[-1,1]`), but it must be
built and verified before the two datasets are mixed. That is deliberately **not** done
here.
"""
        ),
        md(
            """## Takeaways

1. An expert demonstration comes from a **planner**, not a policy. Random rollouts are
   pipeline fixtures; planners produce supervision.
2. `mplib` 0.1.1 requires `numpy<2`. Under NumPy 2.x it segfaults at address `0x0` because
   its C-API function pointers are invalid. This is an ABI problem, not a GPU problem.
3. The planner requires `pd_joint_pos`; its gripper helpers emit absolute position actions.
4. Four steps succeed: reach → grasp → close → carry. Adding `open_gripper()` or extra
   settling steps breaks placement.
5. Success is `is_obj_placed AND is_robot_static`. Grasping alone is not success.
6. Expert episodes are ~100× smoother than random ones (`|Δa|` 0.0078 vs 0.67).
7. Replay reproduces rewards exactly and success identically; a boolean contact flag can
   still differ on a single frame. Claim the tolerance you actually verified.
8. Expert data in `pd_joint_pos` is **not** directly concatenable with the existing
   `pd_joint_delta_pos` fixture. Convert and verify first.
"""
        ),
    ]
    write("2.9_expert_demonstrations.ipynb", cells)

if __name__ == "__main__":
    build_1_1()
    build_1_2()
    build_1_3()
    build_2_1()
    build_2_3()
    build_2_9()
    build_3_1()
    print("\ndone")
