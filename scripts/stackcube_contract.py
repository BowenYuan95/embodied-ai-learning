"""Data and observation contract for the StackCube referential-grounding experiment.

Why StackCube
-------------
Lesson 3.8.4.5 concluded that *referential* grounding -- binding a word to a region of the
image -- is **untestable by construction** on the PickCube/PushCube pool, because neither
instruction contains a referential word and every scene holds one object. The fix is data, not
model. ``StackCube-v1`` supplies the smallest scene that makes the binding necessary:

    cubeA  ``base_color == [1, 0, 0]``   red
    cubeB  ``base_color == [0, 1, 0]``   green
    both cubes' xy are sampled independently by ``_initialize_episode``, so which colour sits
    on which side is randomised by the environment (measured: red left in 19/40 = 47.50%).

Tasks are "pick the red cube" and "pick the green cube". Both are demonstrated on the **same**
layout, by running the planner once with ``pick="cubeA"`` and once with ``pick="cubeB"`` at the
same seed, and truncating after the lift -- see ``scripts/stackcube_solutions.py``.

Where the target's identity lives
---------------------------------
This is the property that makes the experiment work, and it is worth stating precisely:

    the colour exists only in the IMAGE; which colour is wanted exists only in the INSTRUCTION.

Nothing in the robot state says which cube is the target, so no state field can leak it. The
state fields that describe the cubes are excluded for the ordinary reason from lesson 3.8.1 --
they are simulator ground truth for quantities a real robot would have to estimate from
vision -- not because they identify the target. At ``t = 0`` the two tasks share an identical
state *and* an identical scene, so a policy given no instruction faces a symmetric choice: its
best strategy is a constant side, and its accuracy is bounded by 50%. That bound is a property
of the data and can be asserted without training.

Camera contract
---------------
Measured by ``scripts/sweep_stackcube_camera.py``. A cube's visible footprint at the
environment default (128x128, fov 90 deg, eye 0.655 m) is 8 px, i.e. a side of 2.8 px, and the
policy downsamples by 16x through four stride-2 blocks -- 0.18 px, so the colour is gone before
the last feature map and a failure would be unattributable. Chosen instead:

    fov 60 deg, 512x512, camera pose left at the environment default (eye [0.3,0,0.6])
    -> cube side 16.3 px, and 4.06 px after the 4x downsampling this task's encoder uses

The pose is deliberately **not** changed: with fov and resolution alone no camera geometry is
redefined, which keeps the contract change minimal and auditable. Two stride-2 blocks replace
four; a two-object local decision needs no large receptive field, and this is the cheaper lever
-- no camera configuration survives 16x (the best is 1.82 px at side 29.2).

Action contract
---------------
``control_mode="pd_ee_delta_pos"``, action space ``(4,) = [dx, dy, dz, gripper]`` as a world-frame
delta in ``[-1, 1]``. This is a **deliberate departure** from lesson 3.8's eight-dimensional
``pd_joint_pos`` joint targets: a proportional controller needs ``a = K e`` in Cartesian space,
and joint targets would require inverse kinematics and therefore ``mplib`` -- the dependency the
scripted expert removes, along with the ``embodied310`` requirement.

The gripper is ``+1`` open and ``-1`` closed. The convention was measured, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_ID = "StackCube-v1"
OBS_MODE = "rgb+state"
CONTROL_MODE = "pd_ee_delta_pos"
# [dx, dy, dz, gripper], WORLD-frame delta, each in [-1, 1]. Measured by commanding one axis for
# ten steps: +x -> world [+0.1389, +0.0002, +0.0019], +y -> [+0.0261, +0.3657, +0.0047],
# +z -> [+0.0203, 0, +0.3582]. Cross-coupling is an order of magnitude below the commanded axis,
# so sign(a[0]) reliably sets the direction of world-x motion -- which is what makes the
# referential choice a discrete quantity in this action space.
ACTION_DIM = 4
ACTION_LAYOUT = ("dx", "dy", "dz", "gripper")
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0

# 0, NOT the environment default of 0.02: the default randomises the robot's initial joint
# configuration per seed, which made proprio at t=0 differ across seeds (max|d| = 6.63e-02) and
# broke the premise of Probe A ("identical proprio, different layout"). With 0, six seeds gave
# bit-identical t=0 proprio while the layout still varied.
ROBOT_INIT_QPOS_NOISE = 0.0

H = 8                                   # chunk length; the choice of H stays from 3.7

# --- camera contract (see module docstring for the measurement behind it) -----------------
IMAGE_HW = 512
CHANNELS = 3
FOV = np.pi / 3                         # 60 degrees, down from the environment default of 90
EYE = [0.3, 0.0, 0.6]                   # the environment default, deliberately unchanged
TARGET = [-0.1, 0.0, 0.1]
DOWN_SAMPLE = 4                         # two stride-2 blocks in this task's encoder

# --- observation contract ----------------------------------------------------------------
# Kept as deployable proprioception: the same three fields, in the same order, as 3.8.1.
PROPRIO_FIELDS = ("agent.qpos", "agent.qvel", "extra.tcp_pose")
PROPRIO_DIM = 25

# Excluded because a real robot would have to estimate them from vision, exactly as 3.8.1
# excludes `obj_pose`. None of them says WHICH cube is the target.
EXCLUDED_FIELDS = ("extra.cubeA_pose", "extra.cubeB_pose",
                   "extra.tcp_to_cubeA_pos", "extra.tcp_to_cubeB_pos",
                   "extra.cubeA_to_cubeB_pos")
EXPECTED_STATE_DIM = 48

# There is no goal field on this task (the episode is truncated after the lift), so unlike
# 3.8.1's contract there is no `task_goal` input and no place to hide the answer.
HAS_TASK_GOAL = False

# --- language ----------------------------------------------------------------------------
TASKS = {
    "PickCubeRed":   dict(pick="cubeA", colour="red",   instruction="pick the red cube"),
    "PickCubeGreen": dict(pick="cubeB", colour="green", instruction="pick the green cube"),
}
INSTRUCTIONS = {task: spec["instruction"] for task, spec in TASKS.items()}
COLOUR_OF_TASK = {task: spec["colour"] for task, spec in TASKS.items()}
PICK_OF_TASK = {task: spec["pick"] for task, spec in TASKS.items()}

# The whitespace tokenizer is reused verbatim from 3.8.1 so the encoder path is unchanged;
# only the vocabulary contents differ ("red"/"green" instead of "up"/"to"/"goal").
VOCAB = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
for _task in sorted(INSTRUCTIONS):
    for _word in INSTRUCTIONS[_task].split():
        VOCAB.setdefault(_word, len(VOCAB))
T_TXT = max(len(INSTRUCTIONS[t].split()) for t in INSTRUCTIONS) + 2
PAD = VOCAB["<pad>"]


def tokenize(text: str) -> np.ndarray:
    """Whitespace tokenizer with ``<bos>``/``<eos>``, right-padded to ``T_TXT``."""
    ids = [VOCAB["<bos>"]] + [VOCAB[w] for w in text.split()] + [VOCAB["<eos>"]]
    assert len(ids) <= T_TXT, (text, len(ids), T_TXT)
    return np.asarray(ids + [PAD] * (T_TXT - len(ids)), dtype=np.int64)


LANGUAGE_IDS = {task: tokenize(INSTRUCTIONS[task]) for task in TASKS}


def other_task(task: str) -> str:
    """The contradictory instruction, for the counterfactual tests."""
    assert task in TASKS, task
    return next(t for t in TASKS if t != task)


# --- environment ---------------------------------------------------------------------------
def make_env(num_envs: int = 1):
    """Build the environment with this contract's camera, observation and control mode.

    ``sensor_configs`` is a dict of overrides keyed by camera name, not a list of
    ``CameraConfig`` objects; passing a list raises ``TypeError: pop expected at most 1
    argument`` because the merge routine calls ``dict.pop`` on it.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the env ids)
    from mani_skill.utils import sapien_utils

    pose = sapien_utils.look_at(eye=list(EYE), target=list(TARGET))
    return gym.make(ENV_ID, obs_mode=OBS_MODE, num_envs=num_envs, control_mode=CONTROL_MODE,
                    robot_init_qpos_noise=ROBOT_INIT_QPOS_NOISE,
                    sensor_configs={"base_camera": {"width": IMAGE_HW, "height": IMAGE_HW,
                                                    "fov": float(FOV), "pose": pose}})


def state_slices(env) -> dict:
    """Ordered ``{field: (start, stop)}`` for the flat ``obs['state']`` vector.

    Derived from the environment rather than hard-coded, so a ManiSkill change that reorders
    the flattening fails loudly instead of silently shifting every slice.
    """
    offset, out = 0, {}
    groups = (("agent", env.unwrapped._get_obs_agent()),
              ("extra", env.unwrapped._get_obs_extra({})))
    for group, fields in groups:
        for key, value in fields.items():
            width = int(np.asarray(value.shape)[-1])
            out[f"{group}.{key}"] = (offset, offset + width)
            offset += width
    return out


def validate_against_env(env) -> dict:
    """Assert the contract holds for a freshly built environment. Returns the slice table."""
    slices = state_slices(env)
    total = sum(stop - start for start, stop in slices.values())
    assert total == EXPECTED_STATE_DIM, f"state is {total} dims, contract says {EXPECTED_STATE_DIM}"

    missing = [f for f in PROPRIO_FIELDS if f not in slices]
    assert not missing, f"proprio fields absent from the environment: {missing}"
    proprio = [slices[f] for f in PROPRIO_FIELDS]
    proprio_dim = sum(stop - start for start, stop in proprio)
    assert proprio_dim == PROPRIO_DIM, f"proprio is {proprio_dim} dims, contract says {PROPRIO_DIM}"

    # The excluded set must be exactly the leftover, so no dimension is dropped or counted twice.
    kept = set(PROPRIO_FIELDS)
    leftover = {f for f in slices if f not in kept}
    assert leftover == set(EXCLUDED_FIELDS), (
        f"excluded set drifted: environment has {sorted(leftover)}, "
        f"contract declares {sorted(EXCLUDED_FIELDS)}")
    excluded_dim = sum(slices[f][1] - slices[f][0] for f in EXCLUDED_FIELDS)
    assert proprio_dim + excluded_dim == EXPECTED_STATE_DIM

    assert not HAS_TASK_GOAL, "this contract deliberately has no goal field"
    assert env.unwrapped.control_mode == CONTROL_MODE, env.unwrapped.control_mode
    return slices


def proprio_from_state(state: np.ndarray, slices: dict) -> np.ndarray:
    """The 25-d deployable proprioception block, using the environment's own field order."""
    state = np.asarray(state, dtype=np.float32)
    return np.concatenate([state[..., lo:hi] for lo, hi in
                           (slices[f] for f in PROPRIO_FIELDS)]).astype(np.float32)


def build_sample(state: np.ndarray, image: np.ndarray, task: str, t: int,
                 slices: dict, action: np.ndarray, H: int = H) -> dict:
    """One training sample for this task.

    Deliberately has **no** ``task_goal``: the episode is truncated after the lift, so there is
    no destination, and removing it also removes the field that leaked the task in 3.8.4.3.
    """
    assert task in TASKS, task
    assert 0 <= t <= len(action) - H, f"t={t} leaves no room for H={H}"
    return dict(
        image=image[t],                                    # [512,512,3] uint8
        language_ids=LANGUAGE_IDS[task],                   # [T_txt]
        proprio=proprio_from_state(state[t], slices),      # [25]
        action_chunk=action[t:t + H],                      # [8,8]
    )
