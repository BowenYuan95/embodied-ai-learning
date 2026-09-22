import hashlib
import json
from datetime import datetime, timezone

from scripts.pipeline.config import (
    INPUT_FILE,
    LEROBOT_DIR,
    REPO_ID,
    ROBOT_TYPE,
    TASK_NAME,
    FEATURES,
)


def compute_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def write_manifest(
    episode,
    dataset_version="1.0",
    timestamp_source="synthetic",
):
    timestamps = episode["timestamps"]

    if len(timestamps) >= 2:
        mean_dt = float(
            (timestamps[1:] - timestamps[:-1]).mean()
        )
        nominal_fps = round(1.0 / mean_dt)
    else:
        nominal_fps = None

    manifest = {
        "dataset": {
            "repo_id": REPO_ID,
            "version": dataset_version,
            "robot_type": ROBOT_TYPE,
            "task": TASK_NAME,
        },

        "source": {
            "file": str(INPUT_FILE),
            "sha256": compute_sha256(INPUT_FILE),
        },

        "trajectory": {
            "num_frames": len(episode["observations"]),
            "observation_shape": list(
                episode["observations"].shape
            ),
            "action_shape": list(
                episode["actions"].shape
            ),
        },

        "timing": {
            "timestamp_source": timestamp_source,
            "nominal_fps": nominal_fps,
            "actual_acquisition_frequency_verified": False,
        },

        "observation_semantics": {
            "representation": "raw_42d_state_vector",
            "note": (
                "Observation dimensions have not yet been "
                "decomposed into named semantic fields."
            ),
        },

        "action_semantics": {
            "dimension": 8,
            # The 8-d action is NOT one uniform semantic space. Bounds follow
            # Python slice semantics: arm covers indices 0-6, gripper index 7.
            "normalized": True,
            "arm": {
                "indices": [0, 7],
                "dimensions": 7,
                "controller": "PDJointPosController",
                "control_mode": "pd_joint_delta_pos",
                "semantics": (
                    "normalized joint position delta "
                    "relative to current qpos"
                ),
                "use_delta": True,
                "use_target": False,
                "normalize_action": True,
                "physical_range": [-0.1, 0.1],
                "unit": "rad",
                "note": (
                    "a in [-1,1] maps to dq in [-0.1,0.1] rad; the delta is "
                    "applied to the actual current joint position because "
                    "use_target is False"
                ),
            },
            "gripper": {
                "indices": [7, 8],
                "dimensions": 1,
                "controller": "PDJointPosMimicController",
                "semantics": (
                    "normalized absolute gripper joint position target"
                ),
                "use_delta": False,
                "use_target": False,
                "normalize_action": True,
                "physical_range": [-0.01, 0.04],
                "unit": "m",
                "mimic": {
                    "panda_finger_joint2": "panda_finger_joint1"
                },
                "note": (
                    "a in [-1,1] maps linearly onto the joint position range; "
                    "+1 corresponds to 0.04 and -1 to -0.01. A zero command is "
                    "an intermediate target, not a neutral no-op."
                ),
            },
        },

        "features": FEATURES,

        "conversion": {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "format": "LeRobot",
        },
    }

    output_file = (
        LEROBOT_DIR / "conversion_manifest.json"
    )

    output_file.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved manifest: {output_file}"
    )

    return manifest