#!/usr/bin/env python3
"""Collect paired StackCube referential-grounding demonstrations.

Design
------
For every seed, two demonstrations are recorded on the **same layout**:

    pick=cubeA  ("pick the red cube")
    pick=cubeB  ("pick the green cube")

Same seed means same reset, and reset is deterministic across fresh environments -- verified:
two independent envs at the same seed give bit-identical state *and* identical images. The
collector re-checks that per seed rather than trusting it, because the whole 2x2 design rests
on the pair sharing a scene:

    (instruction = red,   layout L)   and   (instruction = green, layout L)

is test T2 (same image, different instruction), and two seeds with mirrored layouts give test
T3 (same instruction, different image). Together they are the full 2x2 interaction.

Episodes are **truncated after the lift**: the task is "pick the named cube", so there is no
destination and no goal field, which is also what removes the field that leaked the task in
3.8.4.3.

Reuses rather than reimplements
-------------------------------
``StepRecorder`` and ``state_field_schema`` come from ``scripts/generate_expert_demo.py``, and
``solve`` from ``scripts/stackcube_solutions.py``. The root attributes written here mirror that
collector's schema, so downstream readers treat this file the same way. The contract's slice
table is cross-checked against the recorder's schema and the run aborts if they disagree.

Must run in ``embodied310``: ``mplib==0.1.1`` needs NumPy 1.x, and the planner segfaults under
``embodied``'s NumPy 2.x. See README.md.

    ~/miniforge3/envs/embodied310/bin/python scripts/collect_stackcube_grounding.py --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stackcube_contract as sc                                    # noqa: E402
from stackcube_solutions import solve                              # noqa: E402

LIFT_MIN = 0.05          # metres; a lift smaller than this is not a "pick" demonstration
MIN_ACTIONS = 10


def collect_one(seed: int, task: str) -> dict:
    """Run the planner for one (seed, task) and return the recorded arrays plus metadata."""
    from generate_expert_demo import StepRecorder

    env = sc.make_env()
    try:
        with StepRecorder(env, obs_mode=sc.OBS_MODE) as recorder:
            solve(env, seed=seed, debug=False, vis=False,
                  pick=sc.PICK_OF_TASK[task], truncate_at_lift=True)
        arrays = recorder.arrays()
    finally:
        env.close()

    obs, act, img = arrays["observations"], arrays["actions"], arrays["images"]
    assert obs.shape[0] == act.shape[0] + 1, (
        f"T+1 convention violated for seed {seed} {task}: {obs.shape[0]} observations, "
        f"{act.shape[0]} actions")
    assert img.shape == (obs.shape[0], sc.IMAGE_HW, sc.IMAGE_HW, sc.CHANNELS), img.shape
    assert obs.shape[1] == sc.EXPECTED_STATE_DIM, obs.shape
    assert act.shape[1] == sc.ACTION_DIM, act.shape

    arrays["seed"] = seed
    arrays["task"] = task
    arrays["instruction"] = sc.INSTRUCTIONS[task]
    arrays["colour"] = sc.COLOUR_OF_TASK[task]
    arrays["pick"] = sc.PICK_OF_TASK[task]
    arrays["num_actions"] = int(act.shape[0])
    return arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "datasets" / "stackcube" / "grounding_pairs.h5")
    args = ap.parse_args()

    import h5py
    from generate_expert_demo import CAMERA, FPS, state_field_schema

    # The contract's slices and the recorder's schema must agree, or every downstream slice is
    # silently wrong. Check on a throwaway env before spending planner time.
    probe = sc.make_env()
    try:
        schema = state_field_schema(probe)
        contract = sc.validate_against_env(probe)
    finally:
        probe.close()
    assert contract == {f["field"]: (f["start"], f["stop"]) for f in schema}, (
        "stackcube_contract's slices disagree with generate_expert_demo's schema")
    state_dim = sum(f["stop"] - f["start"] for f in schema)
    print(f"schema agrees with the contract: {len(schema)} fields, {state_dim} dims")
    print(f"  {', '.join(f['field'] for f in schema)}")

    z_of = {cube: contract[f"extra.{cube}_pose"][0] + 2 for cube in ("cubeA", "cubeB")}

    episodes, failed = [], []
    for seed in args.seeds:
        pair = {}
        for task in sorted(sc.TASKS):
            try:
                arrays = collect_one(seed, task)
            except AssertionError as exc:
                print(f"  seed {seed} {task}: REJECTED {exc}", file=sys.stderr)
                pair = {}
                failed.append((seed, task))
                break
            lift = float(arrays["observations"][-1][z_of[arrays["pick"]]]
                         - arrays["observations"][0][z_of[arrays["pick"]]])
            if lift < LIFT_MIN or arrays["num_actions"] < MIN_ACTIONS:
                print(f"  seed {seed} {task}: REJECTED lift={lift:+.4f} "
                      f"actions={arrays['num_actions']}", file=sys.stderr)
                pair = {}
                failed.append((seed, task))
                break
            arrays["lift"] = lift
            pair[task] = arrays
            print(f"  seed {seed} {task:<14} actions {arrays['num_actions']:>3} "
                  f"lift {lift:+.4f}  image {img_shape(arrays)}", flush=True)

        if len(pair) != len(sc.TASKS):
            continue                      # keep pairs intact: drop the whole seed

        # The pair must share one scene. This is the assertion the 2x2 rests on.
        red, green = pair["PickCubeRed"], pair["PickCubeGreen"]
        assert np.array_equal(red["observations"][0], green["observations"][0]), (
            f"seed {seed}: the two tasks did not start from the same state")
        assert np.array_equal(red["images"][0], green["images"][0]), (
            f"seed {seed}: the two tasks did not start from the same image")
        print(f"  seed {seed}: pair shares an identical t=0 state and image")
        episodes.extend([pair[t] for t in sorted(sc.TASKS)])

    if not episodes:
        print("ERROR: no complete pair collected; nothing written", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as handle:
        handle.attrs.update({
            "env_id": sc.ENV_ID,
            "obs_mode": sc.OBS_MODE,
            "control_mode": sc.CONTROL_MODE,
            "robot": "Panda",
            "data_quality": "expert_planner",
            "generator": "scripts/collect_stackcube_grounding.py",
            "planner": "scripts/stackcube_solutions.py (parameterised stack_cube, truncated at lift)",
            "state_dim": state_dim,
            "state_fields": json.dumps(schema),
            "control_freq_hz": FPS,
            "timestamp_source": "derived_not_measured",
            "num_episodes": len(episodes),
            "num_pairs": len(episodes) // len(sc.TASKS),
            "failed_seeds": json.dumps([list(x) for x in failed]),
            "transition_schema_version": 2,
            "transition_schema": (
                "observations[T+1] = o_0..o_T, where o_t is the state before a_t; "
                "actions[T] = a_0..a_{T-1}; rewards[T]. Episodes are truncated after the "
                "lift, so there is no destination and no goal field."
            ),
            "pairing": (
                "Episodes whose 'seed' matches were recorded from the SAME reset and therefore "
                "share an identical t=0 state and image; the only difference is the "
                "instruction. Verified at collection time by bit-equality of both."
            ),
            "action_semantics": json.dumps({
                "dimension": sc.ACTION_DIM,
                "arm": {"indices": [0, 7], "controller": "PDJointPosController",
                        "semantics": "absolute joint position target"},
                "gripper": {"indices": [7, 8], "controller": "PDJointPosMimicController",
                            "semantics": "absolute gripper joint position target"},
            }),
            "camera": json.dumps({"camera": CAMERA, "fov_rad": float(sc.FOV),
                                  "eye": sc.EYE, "target": sc.TARGET,
                                  "down_sample": sc.DOWN_SAMPLE}),
            "image_camera": CAMERA,
            "image_shape": np.asarray(episodes[0]["images"].shape[1:], dtype=np.int64),
            "image_dtype": str(episodes[0]["images"].dtype),
            "image_source": (
                "obs_mode is a combined mode (rgb+state), so obs['state'] and the camera image "
                "come from the same env.step"
            ),
        })
        for index, episode in enumerate(episodes):
            name = f"s{episode['seed']:02d}_{episode['task']}"
            group = handle.create_group(name)
            for key in ("observations", "actions", "rewards", "images"):
                group.create_dataset(key, data=episode[key], compression="gzip")
            group.attrs["seed"] = episode["seed"]
            group.attrs["task"] = episode["task"]
            group.attrs["colour"] = episode["colour"]
            group.attrs["pick"] = episode["pick"]
            group.attrs["instruction"] = episode["instruction"]
            group.attrs["lift"] = episode["lift"]
            group.attrs["num_actions"] = episode["num_actions"]
            group.attrs["num_observations"] = len(episode["observations"])

    print(f"\nwrote {args.out}: {len(episodes)} episodes "
          f"({len(episodes) // len(sc.TASKS)} layout pairs)")
    if failed:
        print(f"failed (seed, task): {failed}", file=sys.stderr)
    return 0


def img_shape(arrays) -> str:
    return "x".join(str(v) for v in arrays["images"].shape[1:])


if __name__ == "__main__":
    raise SystemExit(main())
