import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripts.pipeline.config import (
    REPO_ID,
    LEROBOT_DIR,
)


def to_numpy(x):
    """
    Convert torch.Tensor or other array-like objects
    to numpy.ndarray.
    """

    if hasattr(x, "detach"):
        x = x.detach()

    if hasattr(x, "cpu"):
        x = x.cpu()

    if hasattr(x, "numpy"):
        x = x.numpy()

    return np.asarray(x)


def load_converted_dataset(
    repo_id=REPO_ID,
    root=LEROBOT_DIR,
):
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )


def validate_converted_dataset(
    source_episode,
    converted_dataset,
):

    print("\n===== Post-Conversion Validation =====")

    errors = []

    source_obs = source_episode["observations"]
    source_actions = source_episode["actions"]
    source_timestamps = source_episode["timestamps"]

    # ==========================================
    # 1. Frame count
    # ==========================================

    source_frames = len(source_obs)
    converted_frames = len(converted_dataset)

    print(f"Source frames:    {source_frames}")
    print(f"Converted frames: {converted_frames}")

    if source_frames == converted_frames:
        print("[PASS] Frame count matches.")
    else:
        errors.append(
            f"Frame count mismatch: "
            f"{source_frames} != {converted_frames}"
        )

    # ==========================================
    # 2. FPS
    # ==========================================

    if len(source_timestamps) >= 2:

        source_fps = round(
            1.0 / np.diff(source_timestamps).mean()
        )

        converted_fps = converted_dataset.meta.fps

        print(f"Source FPS:       {source_fps}")
        print(f"Converted FPS:    {converted_fps}")

        if source_fps == converted_fps:
            print("[PASS] FPS matches.")
        else:
            errors.append(
                f"FPS mismatch: "
                f"{source_fps} != {converted_fps}"
            )

    # 如果 frame 数都不一样，就没必要逐帧比了
    if source_frames != converted_frames:
        return {
            "valid": False,
            "errors": errors,
        }

    # ==========================================
    # 3. Observation equality
    # ==========================================

    for i in range(source_frames):

        converted_frame = converted_dataset[i]

        converted_obs = to_numpy(
            converted_frame["observation.state"]
        )

        if not np.allclose(
            source_obs[i],
            converted_obs,
            rtol=1e-5,
            atol=1e-6,
        ):

            errors.append(
                f"Observation mismatch at frame {i}"
            )
            break

    else:
        print("[PASS] All observations match.")

    # ==========================================
    # 4. Action equality
    # ==========================================

    for i in range(source_frames):

        converted_frame = converted_dataset[i]

        converted_action = to_numpy(
            converted_frame["action"]
        )

        if not np.allclose(
            source_actions[i],
            converted_action,
            rtol=1e-5,
            atol=1e-6,
        ):

            errors.append(
                f"Action mismatch at frame {i}"
            )
            break

    else:
        print("[PASS] All actions match.")

    # ==========================================
    # 5. Shape
    # ==========================================

    first_frame = converted_dataset[0]

    converted_obs = to_numpy(
        first_frame["observation.state"]
    )

    converted_action = to_numpy(
        first_frame["action"]
    )

    print(
        f"Converted observation shape: "
        f"{converted_obs.shape}"
    )

    print(
        f"Converted action shape: "
        f"{converted_action.shape}"
    )

    # ==========================================
    # Result
    # ==========================================

    valid = len(errors) == 0

    print("\nPost-conversion result:")

    if valid:
        print("PASS")
    else:
        print("FAIL")

        for error in errors:
            print(f"[ERROR] {error}")

    return {
        "valid": valid,
        "errors": errors,
    }