"""Re-express expert episodes from ``pd_joint_pos`` targets to ``pd_joint_delta_pos`` semantics.

Why this exists
---------------
The expert episodes in ``datasets/pickcube/expert_episodes.h5`` were collected
with the stock ManiSkill motion planner, which only produces valid actions under
``pd_joint_pos``: every step carries an **absolute** joint target. The project's
existing fixture uses ``pd_joint_delta_pos``, where the arm channels are
normalized deltas. The two are both 8-d and both live in a box, so they cannot be
concatenated without a conversion that preserves meaning.

Conversion
----------
For each step ``t`` with pre-action arm joints ``qpos_t`` (read from
``observations[t]``, layout ``qpos(0:9)``) and executed absolute target
``q_target[t]`` (``actions[t][0:7]``):

    dq_t     = q_target[t] - qpos_t
    a_arm[t] = clip(dq_t / 0.1, -1, 1)

because ``pd_joint_delta_pos`` maps ``a in [-1, 1]`` to ``dq in [-0.1, 0.1] rad``
with ``use_delta=True`` and ``use_target=False`` (the delta is taken against the
actual current ``qpos``). The gripper channel is **unchanged**: both modes read
``action[7]`` as the same normalized absolute gripper target in ``[-0.01, 0.04]``.

Observations and rewards are copied unchanged, and the canonical schema
``observations[T+1] / actions[T] / rewards[T]`` is preserved.

What this conversion is and is not
----------------------------------
It is a **re-targeting** of the same demonstrated states: "the delta command that
would ask the arm for exactly the target the expert asked for". It is *not* the
trajectory the expert executed, because the control mode changed.

The result is therefore expected to be usable as Behavior-Cloning supervision,
while its replay fidelity must be measured rather than assumed. Every root and
per-episode attribute needed to audit the conversion is written into the output,
including the clipped-value count and the source file hash.

Usage (run from the repository root):

    python scripts/convert_expert_actions_to_delta.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "datasets" / "pickcube" / "expert_episodes.h5"
DEFAULT_OUT = REPO_ROOT / "datasets" / "pickcube" / "expert_episodes_delta.h5"

ARM_SLICE = slice(0, 7)
GRIPPER_INDEX = 7
DELTA_SCALE_RAD = 0.1  # action [-1, 1] -> [-0.1, 0.1] rad under pd_joint_delta_pos
QPOS_ARM_SLICE = slice(0, 7)  # qpos(0:9) is the first block of the 42-d state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_episode(observations, actions, rewards):
    """Return delta-semantics arrays plus conversion statistics."""
    if len(observations) != len(actions) + 1:
        raise ValueError(
            f"source schema is not T+1/T: {len(observations)} observations, "
            f"{len(actions)} actions"
        )
    if actions.shape[1] != 8:
        raise ValueError(f"expected 8 action dimensions, got {actions.shape[1]}")

    qpos_arm = observations[:-1, QPOS_ARM_SLICE]
    q_target_arm = actions[:, ARM_SLICE]

    delta_rad = q_target_arm - qpos_arm
    normalized = delta_rad / DELTA_SCALE_RAD
    clipped = np.abs(normalized) > 1.0
    action_delta = actions.copy()
    action_delta[:, ARM_SLICE] = np.clip(normalized, -1.0, 1.0)

    # The reconstruction is exact wherever the delta range is respected.
    reconstructed = qpos_arm + DELTA_SCALE_RAD * action_delta[:, ARM_SLICE]
    max_reconstruction_error = float(
        np.abs(reconstructed[~clipped] - q_target_arm[~clipped]).max()
    ) if (~clipped).any() else 0.0

    stats = {
        "clipped_values": int(clipped.sum()),
        "arm_channels": int(clipped.size),
        "max_reconstruction_error_rad": max_reconstruction_error,
        "max_abs_delta_rad": float(np.abs(delta_rad).max()),
    }
    return action_delta.astype(np.float32), observations, rewards, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.src.exists():
        print(f"ERROR: source not found: {args.src}", file=sys.stderr)
        return 1
    if args.out.exists() and not args.overwrite:
        print(f"ERROR: {args.out} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1

    import h5py

    src_hash = sha256(args.src)
    total_clipped = 0
    total_channels = 0

    with h5py.File(args.src, "r") as source, h5py.File(args.out, "w") as target:
        source_mode = str(source.attrs["control_mode"])
        if source_mode != "pd_joint_pos":
            print(
                f"ERROR: expected source control_mode 'pd_joint_pos', got {source_mode!r}",
                file=sys.stderr,
            )
            return 1

        target.attrs.update(
            {
                "env_id": source.attrs["env_id"],
                "obs_mode": source.attrs["obs_mode"],
                "control_mode": "pd_joint_delta_pos",
                "robot": source.attrs["robot"],
                "task": source.attrs["task"],
                "data_quality": "expert_planner_retargeted",
                "control_freq_hz": source.attrs["control_freq_hz"],
                "timestamp_source": source.attrs["timestamp_source"],
                "transition_schema_version": int(source.attrs["transition_schema_version"]),
                "transition_schema": str(source.attrs["transition_schema"]),
                "conversion": json.dumps(
                    {
                        "generator": "scripts/convert_expert_actions_to_delta.py",
                        "source_file": str(args.src),
                        "source_sha256": src_hash,
                        "source_control_mode": source_mode,
                        "formula": "a_arm[t] = clip((q_target[t] - qpos_t) / 0.1, -1, 1); "
                                   "gripper channel unchanged; observations and rewards unchanged",
                        "delta_scale_rad": DELTA_SCALE_RAD,
                        "note": (
                            "Re-targeted, not re-executed: the states come from a "
                            "pd_joint_pos demonstration, the arm actions are the delta "
                            "commands that ask for the same targets. Replay fidelity in "
                            "pd_joint_delta_pos must be measured, not assumed."
                        ),
                    }
                ),
            }
        )

        for name in sorted(source.keys()):
            group = source[name]
            observations = group["observations"][:]
            actions = group["actions"][:]
            rewards = group["rewards"][:]
            action_delta, observations_out, rewards_out, stats = convert_episode(
                observations, actions, rewards
            )
            total_clipped += stats["clipped_values"]
            total_channels += stats["arm_channels"]

            out_group = target.create_group(name)
            out_group.create_dataset("observations", data=observations_out, compression="gzip")
            out_group.create_dataset("actions", data=action_delta, compression="gzip")
            out_group.create_dataset("rewards", data=rewards_out, compression="gzip")
            for key in ("success", "is_obj_placed", "is_robot_static", "seed"):
                out_group.attrs[key] = group.attrs[key]
            out_group.attrs["num_actions"] = len(action_delta)
            out_group.attrs["num_observations"] = len(observations_out)
            out_group.attrs["clipped_values"] = stats["clipped_values"]
            out_group.attrs["max_abs_delta_rad"] = stats["max_abs_delta_rad"]
            out_group.attrs["max_reconstruction_error_rad"] = stats[
                "max_reconstruction_error_rad"
            ]

            print(
                f"{name}: T={len(action_delta)} clipped={stats['clipped_values']}/"
                f"{stats['arm_channels']} max|dq|={stats['max_abs_delta_rad']:.4f} rad "
                f"reconstruction_error={stats['max_reconstruction_error_rad']:.3e}"
            )

        target.attrs["num_episodes"] = len(source.keys())
        target.attrs["clipped_values_total"] = total_clipped
        target.attrs["arm_channels_total"] = total_channels
        target.attrs["clipped_fraction"] = (
            float(total_clipped) / float(total_channels) if total_channels else 0.0
        )

    print(f"\nsource: {args.src}")
    print(f"sha256: {src_hash}")
    print(f"written: {args.out}")
    print(
        f"clipped arm channels: {total_clipped}/{total_channels} "
        f"({100.0 * total_clipped / total_channels if total_channels else 0.0:.3f}%)"
    )
    print(
        "\nNOTE: this file is pd_joint_delta_pos supervision for training. It is a "
        "re-targeting of the demonstrated states, not a replay of the expert's executed "
        "trajectory; verify replay fidelity before treating it as a closed-loop reference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
