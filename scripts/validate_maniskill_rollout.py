import h5py
import numpy as np

from pathlib import Path


INPUT_FILE = Path(
    "datasets/pickcube/"
    "maniskill_random_rollout.h5"
)


def main():

    # ==========================================
    # 1. Load
    # ==========================================

    with h5py.File(INPUT_FILE, "r") as f:

        observations = f["observations"][:]
        actions = f["actions"][:]
        next_observations = f["next_observations"][:]

        rewards = f["rewards"][:]

        terminated = f["terminated"][:]
        truncated = f["truncated"][:]

        success = f["success"][:]

        elapsed_steps = f["elapsed_steps"][:]

        print("===== Dataset Metadata =====")

        for key, value in f.attrs.items():
            print(
                f"{key}: {value}"
            )

    # ==========================================
    # 2. Basic shapes
    # ==========================================

    print("\n===== Shapes =====")

    print(
        f"observations:      {observations.shape}"
    )

    print(
        f"actions:           {actions.shape}"
    )

    print(
        f"next_observations: {next_observations.shape}"
    )

    print(
        f"rewards:           {rewards.shape}"
    )

    num_frames = len(observations)

    # ==========================================
    # 3. Length consistency
    # ==========================================

    print("\n===== Length Consistency =====")

    lengths = {
        "observations": len(observations),
        "actions": len(actions),
        "next_observations": len(next_observations),
        "rewards": len(rewards),
        "terminated": len(terminated),
        "truncated": len(truncated),
        "success": len(success),
        "elapsed_steps": len(elapsed_steps),
    }

    for key, value in lengths.items():
        print(
            f"{key:<20}: {value}"
        )

    same_length = (
        len(set(lengths.values())) == 1
    )

    if same_length:
        print("[PASS] All trajectory fields have same length.")
    else:
        print("[FAIL] Trajectory field lengths differ.")

    # ==========================================
    # 4. Transition continuity
    # ==========================================

    print("\n===== Transition Continuity =====")

    if num_frames >= 2:

        lhs = next_observations[:-1]
        rhs = observations[1:]

        differences = np.abs(
            lhs - rhs
        )

        max_error = differences.max()
        mean_error = differences.mean()

        print(
            f"Mean |next_obs[t] - obs[t+1]|: "
            f"{mean_error:.8f}"
        )

        print(
            f"Max  |next_obs[t] - obs[t+1]|: "
            f"{max_error:.8f}"
        )

        if np.allclose(
            lhs,
            rhs,
            rtol=1e-5,
            atol=1e-6,
        ):
            print(
                "[PASS] Transition continuity holds."
            )
        else:
            print(
                "[FAIL] next_obs[t] != obs[t+1]."
            )

    # ==========================================
    # 5. Check that environment actually changed
    # ==========================================

    print("\n===== State Change =====")

    state_delta = np.abs(
        next_observations - observations
    )

    per_transition_delta = (
        state_delta.mean(axis=1)
    )

    print(
        f"Mean state change: "
        f"{per_transition_delta.mean():.8f}"
    )

    print(
        f"Max state change:  "
        f"{per_transition_delta.max():.8f}"
    )

    unchanged = np.isclose(
        per_transition_delta,
        0.0,
    )

    print(
        f"Completely unchanged transitions: "
        f"{unchanged.sum()}/{num_frames}"
    )

    # ==========================================
    # 6. Episode semantics
    # ==========================================

    print("\n===== Episode Semantics =====")

    print(
        f"terminated count: "
        f"{terminated.sum()}"
    )

    print(
        f"truncated count:  "
        f"{truncated.sum()}"
    )

    print(
        f"success count:    "
        f"{success.sum()}"
    )

    print(
        f"final terminated: "
        f"{bool(terminated[-1, 0])}"
    )

    print(
        f"final truncated:  "
        f"{bool(truncated[-1, 0])}"
    )

    print(
        f"final success:    "
        f"{bool(success[-1, 0])}"
    )

    # ==========================================
    # 7. elapsed_steps
    # ==========================================

    print("\n===== Elapsed Steps =====")

    print(
        f"First: {elapsed_steps[0]}"
    )

    print(
        f"Last:  {elapsed_steps[-1]}"
    )

    step_diff = np.diff(
        elapsed_steps
    )

    if np.all(step_diff == 1):
        print(
            "[PASS] elapsed_steps increments by 1."
        )
    else:
        print(
            "[FAIL] elapsed_steps is not continuous."
        )

    # ==========================================
    # 8. Summary
    # ==========================================

    print("\n===== Validation Complete =====")


if __name__ == "__main__":
    main()