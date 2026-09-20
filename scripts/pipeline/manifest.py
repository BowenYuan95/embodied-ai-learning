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
            "range": [-1.0, 1.0],
            "arm": {
                "dimensions": 7,
                "controller": "PDJointPosController",
                "semantics": (
                    "normalized joint position target"
                ),
            },
            "gripper": {
                "dimensions": 1,
                "controller": (
                    "PDJointPosMimicController"
                ),
                "semantics": (
                    "continuous normalized position target"
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