import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripts.pipeline.config import (
    FEATURES,
    REPO_ID,
    ROBOT_TYPE,
    TASK_NAME,
    LEROBOT_DIR,
)


def estimate_fps(timestamps):

    if len(timestamps) < 2:
        raise ValueError(
            "At least two timestamps are required "
            "to estimate FPS."
        )

    dt = np.diff(timestamps)

    mean_dt = dt.mean()

    if mean_dt <= 0:
        raise ValueError(
            "Invalid timestamp interval."
        )

    return round(1.0 / mean_dt)


def convert_episode(
    episode,
    repo_id=REPO_ID,
    output_root=LEROBOT_DIR,
    overwrite=False,
):

    obs = episode["observations"]
    actions = episode["actions"]
    timestamps = episode["timestamps"]

    fps = estimate_fps(timestamps)

    print("\n===== LeRobot Conversion =====")
    print(f"Frames: {len(obs)}")
    print(f"FPS: {fps}")
    print(f"Repository ID: {repo_id}")
    print(f"Output root: {output_root}")

    # ==========================================
    # Existing output handling
    # ==========================================

    if output_root.exists():

        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists:\n"
                f"{output_root}\n"
                f"Use --overwrite to replace it."
            )

        print(
            f"[WARNING] Removing existing dataset: "
            f"{output_root}"
        )

        shutil.rmtree(output_root)

    # ==========================================
    # Create LeRobot dataset
    # ==========================================

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        features=FEATURES,
        robot_type=ROBOT_TYPE,
    )

    # ==========================================
    # Add frames
    # ==========================================

    for t in range(len(obs)):

        dataset.add_frame(
            {
                "observation.state": obs[t],
                "action": actions[t],
                "task": TASK_NAME,
            }
        )

    # ==========================================
    # Save dataset
    # ==========================================

    dataset.save_episode()
    dataset.finalize()

    print("LeRobot conversion completed.")

    return dataset