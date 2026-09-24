"""Collect successful PickCube expert demonstrations with ManiSkill's planner.

Why this script exists
----------------------
Random rollouts validate the pipeline but cannot serve as imitation data: their
actions are near-random. This script produces the other kind of trajectory — one
that actually completes the task — by driving the environment with ManiSkill's
sampling-based motion planner, and it records **the actions the environment
actually executed**.

Recorded schema
---------------
The collector writes the canonical transition schema, so no pairing has to be
reconstructed afterwards:

    observations[T + 1] = o_0 .. o_T      # o_t is the state before a_t
    actions[T]          = a_0 .. a_{T-1}  # executed by env.step
    rewards[T]          = r_0 .. r_{T-1}

``o_0`` is the reset observation: ``env.reset`` is wrapped together with
``env.step``, because the planner calls ``env.reset(seed=seed)`` inside
``solve()`` and its result would otherwise be lost. Training pairs are then
``observations[:-1]`` with ``actions``, and ``observations[1:]`` are the matching
next states.

Control mode is load-bearing
----------------------------
The stock motion-planning executor must run under ``pd_joint_pos``:

1. ``TwoFingerGripperMotionPlanningSolver.follow_path`` sends
   ``[qpos(7), gripper]`` — an 8-d **absolute** joint target — through
   ``env.step``. Under ``pd_joint_delta_pos`` that vector still has the right
   shape, but the controller reads it as a normalized delta, so the motion is
   silently wrong.
2. ``open_gripper`` / ``close_gripper`` only take their 8-d branch when
   ``control_mode == "pd_joint_pos"``. Every other mode falls through to
   ``[qpos(7), qpos*0(7), gripper]`` — a 15-d ``pd_joint_pos_vel``-shaped vector
   whose middle block is zeros, which the 8-d delta controller rejects:
   ``Received action of shape torch.Size([15]) but expected shape (1, 8)``.

So a mode other than ``pd_joint_pos`` fails late and confusingly. This script
asserts the mode and the action dimension before recording anything.

Only successful episodes are written. Failed seeds are reported to stderr and
excluded from the file, because a failure is not an expert demonstration.

Visual observations
-------------------
``--obs-mode rgb`` (or ``rgbd``) collects the same episodes **with per-frame
images**. The planner already emits valid ``pd_joint_pos`` actions, so only the
observation side changes; the canonical ``T + 1`` / ``T`` schema is untouched and
the image at index ``k`` belongs to the same step as ``observations[k]``.

Visual observation modes must be **combined** with ``state`` — use
``--obs-mode rgb+state``. ManiSkill parses ``obs_mode`` into a struct
(``parse_obs_mode_to_struct``), so a combined mode returns both ``obs["state"]`` (the
same 42-d vector as ``obs_mode="state"``) and the camera image, **from the same
``env.step``**.

An earlier revision of this script rebuilt the state by hand (``build42``) because a
plain ``rgb`` mode exposes no state. That rebuild was unnecessary: the combined mode
is natively supported, and taking the state straight from ManiSkill is one less
reimplementation of its flattening order to keep correct.

Usage (run from the repository root, in the NumPy-1.x environment):

    python scripts/generate_expert_demo.py --seeds 0 1 2 3 4 --overwrite
    python scripts/generate_expert_demo.py --seeds 0 1 2 3 4 --obs-mode rgb+state \
        --out datasets/pickcube/expert_episodes_rgb.h5 --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_ID = "PickCube-v1"
OBS_MODE = "state"  # default; --obs-mode switches to a visual observation mode
CAMERA = "base_camera"
STATE_DIM = 42
# The stock planner requires this control mode; see the module docstring.
CONTROL_MODE = "pd_joint_pos"
ACTION_DIM = 8
FPS = 20  # measured control rate of this environment, not the declared 50
DEFAULT_OUT = REPO_ROOT / "datasets" / "pickcube" / "expert_episodes.h5"


class StepRecorder:
    """Record executed transitions in canonical ``(o_t, a_t)`` form.

    ``observations`` has ``T + 1`` entries — the reset observation captured from
    ``env.reset``, followed by the observation returned by each ``env.step``.
    """

    def __init__(self, env, obs_mode=OBS_MODE):
        self.env = env
        self.obs_mode = obs_mode
        self.reset_observation = None
        self.reset_image = None
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.next_images = []
        self._original_step = env.step
        self._original_reset = env.reset

    def __enter__(self):
        self.env.step = self._recording_step
        self.env.reset = self._recording_reset
        return self

    def __exit__(self, *exc):
        self.env.step = self._original_step
        self.env.reset = self._original_reset
        return False

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _state_of(self, observation):
        """Extract the 42-d state. A visual mode must be combined with ``state``."""
        if isinstance(observation, dict):
            if "state" not in observation:
                raise RuntimeError(
                    f"obs_mode={self.obs_mode!r} exposes no 'state'. For visual "
                    f"collection use a combined mode such as 'rgb+state', so the state "
                    f"and the image come from the same env.step instead of being rebuilt."
                )
            observation = observation["state"]
        return self._to_numpy(observation).reshape(-1)

    def _extract(self, observation):
        """Return ``(state42, image_or_None)``."""
        image = None
        if isinstance(observation, dict) and "sensor_data" in observation:
            image = self._to_numpy(observation["sensor_data"][CAMERA]["rgb"][0])
        return self._state_of(observation), image

    def _recording_reset(self, *args, **kwargs):
        observation, info = self._original_reset(*args, **kwargs)
        self.reset_observation, self.reset_image = self._extract(observation)
        return observation, info

    def _recording_step(self, action):
        observation, reward, terminated, truncated, info = self._original_step(action)
        state, image = self._extract(observation)
        self.actions.append(self._to_numpy(action).reshape(-1))
        self.rewards.append(float(self._to_numpy(reward).reshape(-1)[0]))
        self.next_observations.append(state)
        self.next_images.append(image)
        return observation, reward, terminated, truncated, info

    def arrays(self):
        if self.reset_observation is None:
            raise RuntimeError(
                "no reset observation was recorded; env.reset must be called inside "
                "the recorder context so that o_0 is stored"
            )
        observations = np.asarray(
            [self.reset_observation, *self.next_observations], dtype=np.float32
        )
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        if len(observations) != len(actions) + 1:
            raise RuntimeError(
                f"transition schema violated: {len(observations)} observations for "
                f"{len(actions)} actions (expected T + 1)"
            )
        arrays = {"observations": observations, "actions": actions, "rewards": rewards}
        if self.reset_image is not None:
            images = np.asarray([self.reset_image, *self.next_images], dtype=np.uint8)
            if len(images) != len(observations):
                raise RuntimeError(
                    f"image/observation misalignment: {len(images)} images for "
                    f"{len(observations)} observations"
                )
            arrays["images"] = images
        return arrays


def run_solver(env, seed, obs_mode=OBS_MODE):
    """Run ManiSkill's own PickCube solution; return its result plus a recorder."""
    from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve

    with StepRecorder(env, obs_mode=obs_mode) as recorder:
        result = solve(env, seed=seed, debug=False, vis=False)
    return result, recorder


def verify_control_mode(env) -> None:
    """Fail loudly instead of recording semantically wrong actions."""
    actual = str(env.unwrapped.control_mode)
    if actual != CONTROL_MODE:
        raise RuntimeError(
            f"control_mode mismatch: environment is {actual!r} but the stock motion "
            f"planner only emits valid actions under {CONTROL_MODE!r}"
        )
    shape = tuple(env.action_space.shape)
    if shape != (ACTION_DIM,):
        raise RuntimeError(f"expected a {ACTION_DIM}-d action space, got {shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="one episode per seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--obs-mode",
        default=OBS_MODE,
        help="ManiSkill observation mode; use a combined mode such as 'rgb+state' to "
             "record per-frame images together with the state vector",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="optional state-only reference file; prints a per-block reproducibility diff",
    )
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        print(f"ERROR: {args.out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1

    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401  registers the environments

    print(f"environment : {ENV_ID}")
    print(f"obs mode    : {args.obs_mode}")
    print(f"control mode: {CONTROL_MODE}")
    print(f"seeds       : {args.seeds}")

    episodes = []
    failed_seeds = []
    for seed in args.seeds:
        env = gym.make(ENV_ID, obs_mode=args.obs_mode, control_mode=CONTROL_MODE, num_envs=1)
        try:
            verify_control_mode(env)
            result, recorder = run_solver(env, seed, obs_mode=args.obs_mode)
            arrays = recorder.arrays()
            evaluation = env.unwrapped.evaluate()
            success = bool(evaluation["success"].item())
            placed = bool(evaluation["is_obj_placed"].item())
            static = bool(evaluation["is_robot_static"].item())

            print(
                f"seed {seed}: actions={len(arrays['actions'])} "
                f"observations={len(arrays['observations'])} "
                f"success={success} placed={placed} static={static} "
                f"return={arrays['rewards'].sum():.4f}"
            )
            if not success:
                failed_seeds.append(seed)
                print(
                    "  WARNING: this episode did not succeed; it is a failure case, "
                    "not an expert demonstration, and it is NOT written to the file.",
                    file=sys.stderr,
                )
                continue

            arrays["success"] = success
            arrays["is_obj_placed"] = placed
            arrays["is_robot_static"] = static
            arrays["seed"] = seed
            episodes.append(arrays)
        finally:
            env.close()

    print(f"\nsuccessful episodes: {len(episodes)}/{len(args.seeds)}")
    if failed_seeds:
        print(f"excluded failed seeds: {failed_seeds}", file=sys.stderr)
    if not episodes:
        print("ERROR: no successful episode; nothing written", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as handle:
        handle.attrs.update(
            {
                "env_id": ENV_ID,
                "obs_mode": args.obs_mode,
                "control_mode": CONTROL_MODE,
                "robot": "Panda",
                "task": "PickCube-v1",
                "data_quality": "expert_planner",
                "planner": "mani_skill.examples.motionplanning.panda.solutions.pick_cube",
                "generator": "scripts/generate_expert_demo.py",
                "control_freq_hz": FPS,
                "timestamp_source": "derived_not_measured",
                "num_episodes": len(episodes),
                "failed_seeds": json.dumps(failed_seeds),
                "transition_schema_version": 2,
                "transition_schema": (
                    "observations[T+1] = o_0..o_T, where o_t is the state before "
                    "a_t; actions[T] = a_0..a_{T-1}, the actions actually executed "
                    "by env.step; rewards[T] = r_0..r_{T-1}. Training pairs are "
                    "observations[:-1] with actions; observations[1:] are the "
                    "matching next states."
                ),
                "action_semantics": json.dumps(
                    {
                        "dimension": ACTION_DIM,
                        "arm": {
                            "indices": [0, 7],
                            "controller": "PDJointPosController",
                            "semantics": "absolute joint position target",
                        },
                        "gripper": {
                            "indices": [7, 8],
                            "controller": "PDJointPosMimicController",
                            "semantics": "absolute gripper joint position target",
                        },
                    }
                ),
            }
        )
        if "sensor_data" in args.obs_mode or "rgb" in args.obs_mode or "depth" in args.obs_mode:
            handle.attrs["image_camera"] = CAMERA
            # a numeric shape, not a JSON string: it is machine-readable data
            handle.attrs["image_shape"] = np.asarray(
                episodes[0]["images"].shape[1:], dtype=np.int64
            )
            handle.attrs["image_dtype"] = str(episodes[0]["images"].dtype)
            handle.attrs["image_source"] = (
                "co-generated with the state by the same planner run; obs_mode is a "
                "combined mode (e.g. rgb+state) so obs['state'] and the camera image come "
                "from the same env.step"
            )
        for index, episode in enumerate(episodes):
            group = handle.create_group(f"episode_{index:06d}")
            keys = ["observations", "actions", "rewards"]
            if "images" in episode:
                keys.append("images")
            for key in keys:
                group.create_dataset(key, data=episode[key], compression="gzip")
            group.attrs["success"] = episode["success"]
            group.attrs["is_obj_placed"] = episode["is_obj_placed"]
            group.attrs["is_robot_static"] = episode["is_robot_static"]
            group.attrs["seed"] = episode["seed"]
            group.attrs["num_actions"] = len(episode["actions"])
            group.attrs["num_observations"] = len(episode["observations"])

    print(f"written: {args.out}")

    if args.compare is not None and args.compare.exists():
        print(f"\nreproducibility vs {args.compare}:")
        blocks = [("qpos", 0, 9), ("qvel", 9, 18), ("is_grasped", 18, 19),
                  ("tcp_pose", 19, 26), ("goal_pos", 26, 29), ("obj_pose", 29, 36),
                  ("tcp_to_obj", 36, 39), ("obj_to_goal", 39, 42)]
        with h5py.File(args.compare, "r") as ref:
            for index, episode in enumerate(episodes):
                name = f"episode_{index:06d}"
                if name not in ref:
                    print(f"  {name}: absent from the reference")
                    continue
                old = ref[name]["observations"][:]
                new = episode["observations"]
                if old.shape != new.shape:
                    print(f"  {name}: shape differs {old.shape} vs {new.shape}")
                    continue
                diff = np.abs(old - new)
                worst = diff.max()
                print(f"  {name}: max={worst:.3e}", end="")
                if worst > 1e-6:
                    hit = [f"{b}={diff[:, lo:hi].max():.2e}"
                           for b, lo, hi in blocks if diff[:, lo:hi].max() > 1e-6]
                    print("  differs in " + ", ".join(hit))
                else:
                    print("  (bit-exact)")
    print(
        "\nNOTE: this is expert data in pd_joint_pos semantics. The project's existing "
        "fixture is pd_joint_delta_pos; see notes/progress.md for how the two relate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
