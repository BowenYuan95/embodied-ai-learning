import numpy as np

from scripts.pipeline.config import (
    OBSERVATION_DIM,
    ACTION_DIM,
)


REQUIRED_KEYS = [
    "observations",
    "actions",
    "rewards",
    "timestamps",
]


def validate_episode(episode):

    errors = []
    warnings = []

    # ==========================================
    # 1. Required fields
    # ==========================================

    for key in REQUIRED_KEYS:

        if key not in episode:
            errors.append(
                f"Missing required field: {key}"
            )

    # 如果连字段都缺了，后面不能安全检查
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }


    obs = episode["observations"]
    actions = episode["actions"]
    rewards = episode["rewards"]
    timestamps = episode["timestamps"]


    # ==========================================
    # 2. Length consistency
    # ==========================================

    lengths = {
        "observations": len(obs),
        "actions": len(actions),
        "rewards": len(rewards),
        "timestamps": len(timestamps),
    }

    if len(set(lengths.values())) != 1:

        errors.append(
            f"Length mismatch: {lengths}"
        )


    # ==========================================
    # 3. Observation shape
    # ==========================================

    if obs.ndim != 2:

        errors.append(
            f"Observation must be 2D, "
            f"got shape {obs.shape}"
        )

    elif obs.shape[1] != OBSERVATION_DIM:

        errors.append(
            f"Observation dimension mismatch: "
            f"expected {OBSERVATION_DIM}, "
            f"got {obs.shape[1]}"
        )


    # ==========================================
    # 4. Action shape
    # ==========================================

    if actions.ndim != 2:

        errors.append(
            f"Action must be 2D, "
            f"got shape {actions.shape}"
        )

    elif actions.shape[1] != ACTION_DIM:

        errors.append(
            f"Action dimension mismatch: "
            f"expected {ACTION_DIM}, "
            f"got {actions.shape[1]}"
        )


    # ==========================================
    # 5. Finite values
    # ==========================================

    arrays = {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "timestamps": timestamps,
    }

    for name, array in arrays.items():

        if not np.isfinite(array).all():

            errors.append(
                f"{name} contains NaN or Inf"
            )


    # ==========================================
    # 6. Action controller range
    # ==========================================

    if np.any(actions < -1.0) or np.any(actions > 1.0):

        errors.append(
            "Actions outside controller range [-1, 1]"
        )


    # ==========================================
    # 7. Timestamp validation
    # ==========================================

    if len(timestamps) < 2:

        warnings.append(
            "Not enough timestamps to validate timing"
        )

    else:

        dt = np.diff(timestamps)

        if np.any(dt <= 0):

            errors.append(
                "Timestamps are not strictly increasing"
            )

        if np.std(dt) > 1e-6:

            warnings.append(
                "Timestamp intervals are not constant"
            )


    # ==========================================
    # Result
    # ==========================================

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }