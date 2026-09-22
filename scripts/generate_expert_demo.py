"""Collect a successful PickCube expert demonstration with ManiSkill's planner.

Why this script exists
----------------------
Random rollouts validate the pipeline but cannot serve as imitation data: their
actions are near-random. This script produces the other kind of trajectory — one
that actually completes the task — by driving the environment with ManiSkill's
sampling-based motion planner, and it records **the actions the environment
actually executed**.

Two details that are load-bearing:

1. **Control mode must be ``pd_joint_pos``.** The planner's ``close_gripper`` /
   ``open_gripper`` emit ``[qpos(7), gripper]``, which only matches an 8-d
   position-target action space. With ``pd_joint_delta_pos`` the gripper helper
   feeds a position vector into a delta controller and the environment rejects
   the shape.
2. **The official recipe does not open the gripper after carrying the cube to the
   goal.** Adding ``open_gripper()`` plus extra settling steps makes the cube land
   beside the goal and the episode fails (``is_obj_placed: False``). The verified
   sequence is exactly: reach -> grasp -> close -> carry.

Recording uses an ``env.step`` wrapper, so no transition is reconstructed after
the fact and the T-actions/T-observations pairing is preserved by construction.

Usage (run from the repository root):

    python scripts/generate_expert_demo.py
    python scripts/generate_expert_demo.py --seeds 0 1 2 --out datasets/pickcube/expert_episodes.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_ID = "PickCube-v1"
OBS_MODE = "state"
# The planner requires this control mode; see the module docstring.
CONTROL_MODE = "pd_joint_pos"
FPS = 20  # measured control rate of this environment, not the declared 50
DEFAULT_OUT = REPO_ROOT / "datasets" / "pickcube" / "expert_episodes.h5"


class StepRecorder:
    """Wrap ``env.step`` so every executed transition is captured."""

    def __init__(self, env):
        self.env = env
        self.records = {"observations": [], "actions": [], "rewards": []}
        self._original_step = env.step

    def __enter__(self):
        self.env.step = self._recording_step
        return self

    def __exit__(self, *exc):
        self.env.step = self._original_step
        return False

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _recording_step(self, action):
        observation, reward, terminated, truncated, info = self._original_step(action)
        self.records["observations"].append(self._to_numpy(observation).reshape(-1))
        self.records["actions"].append(self._to_numpy(action).reshape(-1))
        self.records["rewards"].append(float(self._to_numpy(reward).reshape(-1)[0]))
        return observation, reward, terminated, truncated, info

    def arrays(self):
        return {
            "observations": np.asarray(self.records["observations"], dtype=np.float32),
            "actions": np.asarray(self.records["actions"], dtype=np.float32),
            "rewards": np.asarray(self.records["rewards"], dtype=np.float32),
        }


def run_solver(env, seed):
    """Run ManiSkill's own PickCube solution; return its result plus a recorder."""
    from mani_skill.examples.motionplanning.panda.solutions.pick_cube import solve

    with StepRecorder(env) as recorder:
        result = solve(env, seed=seed, debug=False, vis=False)
    return result, recorder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="one episode per seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        print(f"ERROR: {args.out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1

    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401  registers the environments

    print(f"environment : {ENV_ID}")
    print(f"control mode: {CONTROL_MODE}")
    print(f"seeds       : {args.seeds}")

    episodes = []
    for seed in args.seeds:
        env = gym.make(ENV_ID, obs_mode=OBS_MODE, control_mode=CONTROL_MODE, num_envs=1)
        try:
            result, recorder = run_solver(env, seed)
            arrays = recorder.arrays()
            evaluation = env.unwrapped.evaluate()
            success = bool(evaluation["success"].item())
            placed = bool(evaluation["is_obj_placed"].item())
            static = bool(evaluation["is_robot_static"].item())

            print(
                f"seed {seed}: frames={len(arrays['actions'])} "
                f"success={success} placed={placed} static={static} "
                f"return={arrays['rewards'].sum():.4f}"
            )
            if not success:
                print(
                    "  WARNING: this episode did not succeed; it is a failure case, "
                    "not an expert demonstration.",
                    file=sys.stderr,
                )

            arrays["success"] = success
            arrays["is_obj_placed"] = placed
            arrays["is_robot_static"] = static
            arrays["seed"] = seed
            episodes.append(arrays)
        finally:
            env.close()

    successes = [ep for ep in episodes if ep["success"]]
    print(f"\nsuccessful episodes: {len(successes)}/{len(episodes)}")
    if not successes:
        print("ERROR: no successful episode; nothing written", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as handle:
        handle.attrs.update(
            {
                "env_id": ENV_ID,
                "obs_mode": OBS_MODE,
                "control_mode": CONTROL_MODE,
                "robot": "Panda",
                "task": "PickCube-v1",
                "data_quality": "expert_planner",
                "planner": "mani_skill.examples.motionplanning.panda.solutions.pick_cube",
                "generator": "scripts/generate_expert_demo.py",
                "control_freq_hz": FPS,
                "timestamp_source": "derived_not_measured",
                "action_semantics": json.dumps(
                    {
                        "dimension": 8,
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
        for index, episode in enumerate(episodes):
            group = handle.create_group(f"episode_{index:06d}")
            for key in ("observations", "actions", "rewards"):
                group.create_dataset(key, data=episode[key], compression="gzip")
            group.attrs["success"] = episode["success"]
            group.attrs["is_obj_placed"] = episode["is_obj_placed"]
            group.attrs["is_robot_static"] = episode["is_robot_static"]
            group.attrs["seed"] = episode["seed"]
            group.attrs["num_frames"] = len(episode["actions"])

    print(f"written: {args.out}")
    print(
        "\nNOTE: this is expert data in pd_joint_pos semantics. The project's existing "
        "fixture is pd_joint_delta_pos; see notes/progress.md for how the two relate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
