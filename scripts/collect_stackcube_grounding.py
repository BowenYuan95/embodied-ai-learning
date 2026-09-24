#!/usr/bin/env python3
"""Collect the StackCube referring-grounding 2x2 by construction.

Design
------
The layout is a **controlled variable**, not a random draw. For every seed and every layout the
collector records both instructions:

    for seed:
        for a_left in (True, False):          # which cube takes the left slot
            for task in (PickCubeRed, PickCubeGreen):

so the full ``(instruction x layout)`` 2x2 is produced for each seed. Random mirrored pairs are
never relied on -- with the environment's own sampling only 4 of 12 seeds put the cubes on
opposite sides of the robot's home x, and the two cubes were often separated mostly in y.

That gives both probes exactly:

* **Probe A** (task ambiguity): the two layouts' ``t = 0`` proprio is bit-identical, because
  ``robot_init_qpos_noise = 0`` and the cube positions live in the excluded fields. Different
  layouts need different action signs. Asserted per seed.
* **Probe B** (referring grounding): for one layout, the two instructions' ``t = 0`` state,
  proprio *and image* are bit-identical. Only the instruction differs. Asserted per pair.

Expert
------
``scripts/direct_reach_solutions.py``: a scripted proportional Cartesian controller, four phases
(hover above the named cube, descend, close, lift). Its first action necessarily points at the
cube, so ``a_0[0]`` saturates to ``+/-1.0`` and its sign is the target's side. The motion planner
was unusable here because its first actions were identical for both tasks.

Consequently the metadata says ``expert_type = scripted_direct_reach`` and ``planner = none``.
It must **not** inherit ``expert_planner``, and this is a new artefact rather than a replacement
for the planner-generated datasets.

    python scripts/collect_stackcube_grounding.py --seeds 0 1
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
import stackcube_layout as sl                                      # noqa: E402
from direct_reach_solutions import solve                           # noqa: E402

LIFT_MIN = 0.05
MIN_ACTIONS = 8


def collect_one(seed: int, task: str, a_left: bool) -> dict:
    """One demonstration: forced layout, one instruction, truncated at the lift."""
    from generate_expert_demo import StepRecorder

    env = sc.make_env()
    controller = sl.attach(env)
    try:
        controller.set_layout(a_left)          # applied inside the reset that follows
        with StepRecorder(env, obs_mode=sc.OBS_MODE) as recorder:
            diagnostics = solve(env, seed=seed, pick=sc.PICK_OF_TASK[task])
        arrays = recorder.arrays()
    finally:
        env.close()

    obs, act, img = arrays["observations"], arrays["actions"], arrays["images"]
    assert obs.shape[0] == act.shape[0] + 1, (
        f"T+1 violated for seed {seed} {task} a_left={a_left}: "
        f"{obs.shape[0]} observations, {act.shape[0]} actions")
    assert img.shape == (obs.shape[0], sc.IMAGE_HW, sc.IMAGE_HW, sc.CHANNELS), img.shape
    assert obs.shape[1] == sc.EXPECTED_STATE_DIM, obs.shape
    assert act.shape[1] == sc.ACTION_DIM, act.shape

    arrays.update(seed=seed, task=task, a_left=bool(a_left),
                  instruction=sc.INSTRUCTIONS[task], colour=sc.COLOUR_OF_TASK[task],
                  pick=sc.PICK_OF_TASK[task], num_actions=int(act.shape[0]),
                  lift=float(diagnostics["lift"]),
                  first_action_x=float(diagnostics["first_action"][0]))
    return arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "datasets" / "stackcube" / "referring_v1.h5")
    args = ap.parse_args()

    import h5py
    from generate_expert_demo import CAMERA, FPS, state_field_schema

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
    print(f"action: {sc.ACTION_LAYOUT} (dim {sc.ACTION_DIM}), "
          f"control_mode {sc.CONTROL_MODE}, init qpos noise {sc.ROBOT_INIT_QPOS_NOISE}")

    episodes, failed = [], []
    for seed in args.seeds:
        per_seed = {}
        for a_left in (True, False):
            for task in sorted(sc.TASKS):
                key = (a_left, task)
                try:
                    arrays = collect_one(seed, task, a_left)
                except AssertionError as exc:
                    print(f"  seed {seed} a_left={a_left} {task}: REJECTED {exc}", file=sys.stderr)
                    failed.append((seed, a_left, task))
                    continue
                if arrays["lift"] < LIFT_MIN or arrays["num_actions"] < MIN_ACTIONS:
                    print(f"  seed {seed} a_left={a_left} {task}: REJECTED lift={arrays['lift']:+.4f} "
                          f"actions={arrays['num_actions']}", file=sys.stderr)
                    failed.append((seed, a_left, task))
                    continue
                per_seed[key] = arrays
                print(f"  seed {seed} {'A_left ' if a_left else 'A_right'} {task:<14} "
                      f"actions {arrays['num_actions']:>3} lift {arrays['lift']:+.4f} "
                      f"a0_x {arrays['first_action_x']:+.4f}", flush=True)

        # Probe A: identical proprio at t=0 across layouts.
        if (True, "PickCubeRed") in per_seed and (False, "PickCubeRed") in per_seed:
            p_l = per_seed[(True, "PickCubeRed")]["observations"][0][[0, 1, 2, 9, 10, 18]]
            p_r = per_seed[(False, "PickCubeRed")]["observations"][0][[0, 1, 2, 9, 10, 18]]
            assert np.array_equal(p_l, p_r), (
                f"seed {seed}: t=0 proprio differs across layouts -- Probe A's premise fails")
            print(f"  seed {seed}: Probe A holds (t=0 proprio identical across layouts)")

        # Probe B: identical state, proprio AND image for the two instructions on one layout.
        for a_left in (True, False):
            red, green = per_seed.get((a_left, "PickCubeRed")), per_seed.get((a_left, "PickCubeGreen"))
            if not (red and green):
                continue
            for label, a, b in (("state", red["observations"][0], green["observations"][0]),
                                ("image", red["images"][0], green["images"][0])):
                assert np.array_equal(a, b), (
                    f"seed {seed} a_left={a_left}: t=0 {label} differs between instructions -- "
                    f"Probe B's premise fails")
            assert np.sign(red["first_action_x"]) != np.sign(green["first_action_x"]), (
                f"seed {seed} a_left={a_left}: the two instructions gave the same first-action "
                f"sign ({red['first_action_x']:+.3f} vs {green['first_action_x']:+.3f})")
            print(f"  seed {seed} a_left={a_left}: Probe B holds (identical input, "
                  f"opposite action sign {red['first_action_x']:+.2f} / {green['first_action_x']:+.2f})")
        episodes.extend(per_seed[k] for k in sorted(per_seed, key=lambda t: (not t[0], t[1])))

    if not episodes:
        print("ERROR: nothing collected", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as handle:
        handle.attrs.update({
            "env_id": sc.ENV_ID,
            "obs_mode": sc.OBS_MODE,
            "control_mode": sc.CONTROL_MODE,
            "robot": "Panda",
            "task": "referring_pick",
            "expert_type": "scripted_direct_reach",
            "controller": "proportional_tcp_controller",
            "planner": "none",
            "data_quality": "scripted_expert_lift_verified",
            "generator": "scripts/collect_stackcube_grounding.py",
            "state_dim": state_dim,
            "state_fields": json.dumps(schema),
            "control_freq_hz": FPS,
            "timestamp_source": "derived_not_measured",
            "num_episodes": len(episodes),
            "num_pairs": len(episodes) // (2 * len(sc.TASKS)),
            "failed": json.dumps([list(x) for x in failed]),
            "transition_schema_version": 2,
            "transition_schema": (
                "observations[T+1] = o_0..o_T, where o_t is the state before a_t; actions[T]; "
                "rewards[T]. Episodes end at the lift and are truncated there, so there is no "
                "destination and no goal field."
            ),
            "layout": json.dumps({
                "controlled": True,
                "separation_m": sl.SEPARATION, "y_jitter_m": sl.Y_JITTER,
                "note": "cubeA and cubeB are placed at -/+ separation/2 on x; 'a_left' records "
                        "which took the left slot. Layout is a controlled variable, not a draw.",
            }),
            "pairing": (
                "For a given seed: the two layouts have bit-identical t=0 proprio (Probe A), and "
                "within one layout the two instructions have bit-identical t=0 state, proprio "
                "and image (Probe B). Both asserted at collection time."
            ),
            "action_semantics": json.dumps({
                "dimension": sc.ACTION_DIM,
                "layout": list(sc.ACTION_LAYOUT),
                "frame": "world",
                "semantics": "delta end-effector position per control step, clipped to [-1, 1]",
                "gripper": {"index": 3, "open": sc.GRIPPER_OPEN, "closed": sc.GRIPPER_CLOSED},
            }),
            "camera": json.dumps({"camera": CAMERA, "fov_rad": float(sc.FOV), "eye": sc.EYE,
                                  "target": sc.TARGET, "down_sample": sc.DOWN_SAMPLE}),
            "image_camera": CAMERA,
            "image_shape": np.asarray(episodes[0]["images"].shape[1:], dtype=np.int64),
            "image_dtype": str(episodes[0]["images"].dtype),
            "robot_init_qpos_noise": sc.ROBOT_INIT_QPOS_NOISE,
        })
        for episode in episodes:
            name = f"s{episode['seed']:02d}_{'L' if episode['a_left'] else 'R'}_{episode['task']}"
            group = handle.create_group(name)
            for key in ("observations", "actions", "rewards", "images"):
                group.create_dataset(key, data=episode[key], compression="gzip")
            for key in ("seed", "task", "a_left", "colour", "pick", "instruction",
                        "num_actions", "lift", "first_action_x"):
                group.attrs[key] = episode[key]
            group.attrs["num_observations"] = len(episode["observations"])

    print(f"\nwrote {args.out}: {len(episodes)} episodes "
          f"({len(episodes) // (2 * len(sc.TASKS))} seeds x 2 layouts x 2 instructions)")
    if failed:
        print(f"failed: {failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
