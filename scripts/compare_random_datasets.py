from pathlib import Path

import h5py
import numpy as np


SYNTHETIC_FILE = Path(
    "datasets/pickcube/random_episode_standard.h5"
)

ROLLOUT_FILE = Path(
    "datasets/pickcube/maniskill_random_rollout.h5"
)


def load_dataset(path, required_keys):

    data = {}

    with h5py.File(path, "r") as f:

        for key in required_keys:

            if key not in f:
                raise KeyError(
                    f"Required dataset '{key}' "
                    f"not found in {path}"
                )

            if not isinstance(
                f[key],
                h5py.Dataset,
            ):
                raise TypeError(
                    f"'{key}' exists but is not "
                    f"an HDF5 dataset."
                )

            data[key] = f[key][:]

        attrs = dict(
            f.attrs.items()
        )

    return data, attrs

def delta_statistics(x):
    """
    Statistics of adjacent samples:
        x[t+1] - x[t]
    """

    delta = np.abs(
        np.diff(x, axis=0)
    )

    return {
        "mean": float(delta.mean()),
        "p95": float(
            np.percentile(delta, 95)
        ),
        "max": float(delta.max()),
    }


def transition_statistics(
    obs,
    next_obs,
):
    """
    Statistics of true environment transition:
        next_obs[t] - obs[t]
    """

    delta = np.abs(
        next_obs - obs
    )

    return {
        "mean": float(delta.mean()),
        "p95": float(
            np.percentile(delta, 95)
        ),
        "max": float(delta.max()),
    }


def print_stats(name, stats):

    print(name)

    print(
        f"  Mean: {stats['mean']:.6f}"
    )

    print(
        f"  P95:  {stats['p95']:.6f}"
    )

    print(
        f"  Max:  {stats['max']:.6f}"
    )


def main():

    # ==========================================
    # Load
    # ==========================================

    synthetic, synthetic_attrs = load_dataset(
        SYNTHETIC_FILE,
        required_keys=[
            "observations",
            "actions",
        ],
    )

    rollout, rollout_attrs = load_dataset(
        ROLLOUT_FILE,
        required_keys=[
            "observations",
            "actions",
            "next_observations",
            "success",
            "terminated",
            "truncated",
        ],
    )

    print(
        "===== Dataset Comparison ====="
    )

    # ==========================================
    # Basic information
    # ==========================================

    print(
        "\n===== Synthetic Random Dataset ====="
    )

    print(
        f"Observation shape: "
        f"{synthetic['observations'].shape}"
    )

    print(
        f"Action shape: "
        f"{synthetic['actions'].shape}"
    )

    print(
        "\n===== ManiSkill Rollout ====="
    )

    print(
        f"Environment: "
        f"{rollout_attrs.get('environment')}"
    )

    print(
        f"Policy: "
        f"{rollout_attrs.get('policy')}"
    )

    print(
        f"Observation shape: "
        f"{rollout['observations'].shape}"
    )

    print(
        f"Action shape: "
        f"{rollout['actions'].shape}"
    )

    # ==========================================
    # Action differences
    # ==========================================

    print(
        "\n===== Action Changes ====="
    )

    synthetic_action_stats = (
        delta_statistics(
            synthetic["actions"]
        )
    )

    rollout_action_stats = (
        delta_statistics(
            rollout["actions"]
        )
    )

    print_stats(
        "Synthetic random actions:",
        synthetic_action_stats,
    )

    print_stats(
        "\nManiSkill random-policy actions:",
        rollout_action_stats,
    )

    # ==========================================
    # State differences
    # ==========================================

    print(
        "\n===== State Changes ====="
    )

    synthetic_state_stats = (
        delta_statistics(
            synthetic["observations"]
        )
    )

    rollout_state_stats = (
        transition_statistics(
            rollout["observations"],
            rollout["next_observations"],
        )
    )

    print_stats(
        "Synthetic adjacent observations:",
        synthetic_state_stats,
    )

    print_stats(
        "\nManiSkill physical transitions:",
        rollout_state_stats,
    )

    # ==========================================
    # Transition continuity
    # ==========================================

    print(
        "\n===== ManiSkill Transition Continuity ====="
    )

    continuity_error = np.abs(
        rollout["next_observations"][:-1]
        - rollout["observations"][1:]
    )

    print(
        f"Mean continuity error: "
        f"{continuity_error.mean():.8f}"
    )

    print(
        f"Max continuity error:  "
        f"{continuity_error.max():.8f}"
    )

    # ==========================================
    # Episode semantics
    # ==========================================

    print(
        "\n===== ManiSkill Episode Semantics ====="
    )

    print(
        f"Success count: "
        f"{rollout['success'].sum()}"
    )

    print(
        f"Terminated count: "
        f"{rollout['terminated'].sum()}"
    )

    print(
        f"Truncated count: "
        f"{rollout['truncated'].sum()}"
    )


if __name__ == "__main__":
    main()