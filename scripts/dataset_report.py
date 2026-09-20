from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = Path(
    "datasets/pickcube/random_episode_standard.h5"
)
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent

FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

def load_h5():
    with h5py.File(INPUT_FILE, "r") as f:
        obs = f["observations"][:].astype(np.float32)
        actions = f["actions"][:].astype(np.float32)
        rewards = f["rewards"][:].astype(np.float32)
        timestamps = f["timestamps"][:].astype(np.float64)

    return obs, actions, rewards, timestamps


def print_dataset_overview(obs, actions, rewards, timestamps):
    print("\n===== Dataset Overview =====")

    num_frames = len(obs)

    dt = np.diff(timestamps)

    if len(dt) > 0:
        fps = 1.0 / np.mean(dt)
        timestamp_span = timestamps[-1] - timestamps[0]
        nominal_duration = num_frames / fps
    else:
        fps = float("nan")
        timestamp_span = 0.0
        nominal_duration = 0.0

    print(f"Frames:             {num_frames}")
    print(f"Observation shape:  {obs.shape}")
    print(f"Action shape:       {actions.shape}")
    print(f"Reward shape:       {rewards.shape}")
    print(f"Timestamp shape:    {timestamps.shape}")
    print(f"Estimated FPS:      {fps:.2f}")
    print(f"Timestamp span:     {timestamp_span:.4f} s")
    print(f"Nominal duration:   {nominal_duration:.4f} s")


def print_reward_statistics(rewards):
    print("\n===== Reward Statistics =====")

    rewards = rewards.reshape(-1)

    print(f"Return:        {rewards.sum():.6f}")
    print(f"Mean:          {rewards.mean():.6f}")
    print(f"Std:           {rewards.std():.6f}")
    print(f"Min:           {rewards.min():.6f}")
    print(f"Max:           {rewards.max():.6f}")
    print(f"Final reward:  {rewards[-1]:.6f}")


def print_action_statistics(actions):
    print("\n===== Action Statistics =====")

    names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "gripper",
    ]

    print(
        f"{'Dimension':<12}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Min':>12}"
        f"{'Max':>12}"
    )

    for i in range(actions.shape[1]):
        x = actions[:, i]

        name = names[i] if i < len(names) else f"action_{i}"

        print(
            f"{name:<12}"
            f"{x.mean():>12.4f}"
            f"{x.std():>12.4f}"
            f"{x.min():>12.4f}"
            f"{x.max():>12.4f}"
        )


def print_temporal_statistics(timestamps):
    print("\n===== Temporal Statistics =====")

    if len(timestamps) < 2:
        print("Not enough timestamps.")
        return

    dt = np.diff(timestamps)

    print(f"Mean dt:    {dt.mean():.6f} s")
    print(f"Std dt:     {dt.std():.6f} s")
    print(f"Min dt:     {dt.min():.6f} s")
    print(f"Max dt:     {dt.max():.6f} s")
    print(f"Mean FPS:   {1.0 / dt.mean():.2f}")

    monotonic = np.all(dt > 0)

    print(f"Monotonic:  {monotonic}")


def print_integrity_check(obs, actions, rewards, timestamps):
    print("\n===== Data Integrity Check =====")

    lengths = {
        "observations": len(obs),
        "actions": len(actions),
        "rewards": len(rewards),
        "timestamps": len(timestamps),
    }

    same_length = len(set(lengths.values())) == 1

    for name, length in lengths.items():
        print(f"{name:<15}: {length}")

    print(f"\nSame length: {same_length}")

    arrays = {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "timestamps": timestamps,
    }

    for name, array in arrays.items():
        has_nan = np.isnan(array).any()
        has_inf = np.isinf(array).any()

        print(
            f"{name:<15} "
            f"NaN={has_nan}, "
            f"Inf={has_inf}"
        )


def print_action_range_check(actions):
    print("\n===== Action Range Check =====")

    out_of_range = np.logical_or(
        actions < -1.0,
        actions > 1.0,
    )

    num_out_of_range = np.count_nonzero(out_of_range)

    # 接近边界，用于观察 action saturation
    saturation_threshold = 0.95

    saturated = np.abs(actions) >= saturation_threshold

    num_saturated = np.count_nonzero(saturated)

    total = actions.size

    print(f"Values outside [-1, 1]: {num_out_of_range}")

    print(
        f"Values with |action| >= {saturation_threshold}: "
        f"{num_saturated}/{total} "
        f"({num_saturated / total * 100:.2f}%)"
    )


def print_action_smoothness(actions):
    print("\n===== Action Smoothness =====")

    if len(actions) < 2:
        print("Not enough actions.")
        return

    # 相邻 timestep 的 action 差
    action_delta = np.diff(actions, axis=0)

    abs_delta = np.abs(action_delta)

    names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "gripper",
    ]

    print(
        f"{'Dimension':<12}"
        f"{'Mean |Δa|':>14}"
        f"{'Max |Δa|':>14}"
        f"{'P95 |Δa|':>14}"
    )

    for i in range(actions.shape[1]):
        x = abs_delta[:, i]

        name = names[i] if i < len(names) else f"action_{i}"

        print(
            f"{name:<12}"
            f"{x.mean():>14.4f}"
            f"{x.max():>14.4f}"
            f"{np.percentile(x, 95):>14.4f}"
        )

    # 所有维度整体
    print("\nOverall:")

    print(
        f"Mean |Δa|: "
        f"{abs_delta.mean():.4f}"
    )

    print(
        f"Max |Δa|:  "
        f"{abs_delta.max():.4f}"
    )

    print(
        f"P95 |Δa|:  "
        f"{np.percentile(abs_delta, 95):.4f}"
    )

def plot_reward_curve(rewards):
    rewards = rewards.reshape(-1)

    plt.figure()

    plt.plot(
        np.arange(len(rewards)),
        rewards,
        marker="o",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward Curve")

    plt.grid(True)

    output_file = FIGURE_DIR / "reward_curve.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved figure: {output_file}")

ACTION_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
]


def plot_action_dimension(actions, dim=0):

    name = ACTION_NAMES[dim]

    plt.figure()

    plt.plot(
        np.arange(len(actions)),
        actions[:, dim],
        marker="o",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Normalized Action")
    plt.title(f"Action: {name}")

    plt.ylim(-1.1, 1.1)

    plt.grid(True)

    output_file = (
        FIGURE_DIR /
        f"action_{name}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved figure: {output_file}")

def generate_quality_summary(
    obs,
    actions,
    rewards,
    timestamps,
    timestamp_type="synthetic",
):

    lines = []

    def add(text=""):
        lines.append(text)

    add("===== Dataset Quality Summary =====")
    add()

    # ==================================================
    # 1. Integrity
    # ==================================================

    lengths = [
        len(obs),
        len(actions),
        len(rewards),
        len(timestamps),
    ]

    same_length = len(set(lengths)) == 1

    arrays = [
        obs,
        actions,
        rewards,
        timestamps,
    ]

    finite = all(
        np.isfinite(x).all()
        for x in arrays
    )

    integrity_pass = same_length and finite

    add("Integrity:")

    if integrity_pass:
        add("    PASS")
    else:
        add("    FAIL")

        if not same_length:
            add("    - Array lengths do not match.")

        if not finite:
            add("    - NaN or Inf detected.")

    add()

    # ==================================================
    # 2. Temporal
    # ==================================================

    dt = np.diff(timestamps)

    temporal_pass = (
        len(dt) > 0
        and np.all(dt > 0)
    )

    add("Temporal:")

    if temporal_pass:
        add("    PASS")
        add(
            f"    Mean dt: {dt.mean():.6f} s"
        )
        add(
            f"    Nominal FPS: "
            f"{1.0 / dt.mean():.2f}"
        )
    else:
        add("    FAIL")

    add(
        f"    Timestamp source: "
        f"{timestamp_type}"
    )

    if timestamp_type == "synthetic":
        add(
            "    NOTE: timing consistency does not "
            "verify real acquisition timing."
        )

    add()

    # ==================================================
    # 3. Action Range
    # ==================================================

    action_in_range = np.all(
        (actions >= -1.0)
        & (actions <= 1.0)
    )

    add("Action Range:")

    if action_in_range:
        add("    PASS")
        add("    All actions within [-1, 1].")
    else:
        add("    FAIL")
        add(
            "    Action values outside controller range."
        )

    add()

    # ==================================================
    # 4. Action Smoothness
    # ==================================================

    action_delta = np.abs(
        np.diff(actions, axis=0)
    )

    mean_delta = action_delta.mean()
    p95_delta = np.percentile(
        action_delta,
        95,
    )
    max_delta = action_delta.max()

    add("Action Smoothness:")

    add(
        f"    Mean |Δa|: {mean_delta:.4f}"
    )
    add(
        f"    P95  |Δa|: {p95_delta:.4f}"
    )
    add(
        f"    Max  |Δa|: {max_delta:.4f}"
    )

    # This threshold is diagnostic, not a hard rule.
    if mean_delta > 0.5:
        add(
            "    WARNING: large frame-to-frame "
            "action changes."
        )
    else:
        add(
            "    INFO: action changes are relatively "
            "smooth."
        )

    add()

    # ==================================================
    # 5. Randomness Diagnostic
    # ==================================================

    flat_actions = actions.reshape(-1)

    observed_mean = flat_actions.mean()
    observed_std = flat_actions.std()

    expected_mean = 0.0
    expected_std = 1.0 / np.sqrt(3.0)
    expected_delta = 2.0 / 3.0

    mean_error = abs(
        observed_mean - expected_mean
    )

    std_error = abs(
        observed_std - expected_std
    )

    delta_error = abs(
        mean_delta - expected_delta
    )

    random_checks = [
        mean_error < 0.10,
        std_error < 0.10,
        delta_error < 0.10,
    ]

    random_score = sum(random_checks)

    add("Randomness Diagnostic:")

    add(
        f"    Action mean: {observed_mean:.4f} "
        f"(random baseline: {expected_mean:.4f})"
    )

    add(
        f"    Action std:  {observed_std:.4f} "
        f"(random baseline: {expected_std:.4f})"
    )

    add(
        f"    Mean |Δa|:  {mean_delta:.4f} "
        f"(random baseline: {expected_delta:.4f})"
    )

    add(
        f"    Random-like score: "
        f"{random_score}/3"
    )

    if random_score == 3:
        add(
            "    WARNING: strongly consistent with "
            "independent Uniform(-1,1) random sampling."
        )

    elif random_score == 2:
        add(
            "    INFO: partially consistent with "
            "random sampling."
        )

    else:
        add(
            "    INFO: does not strongly match the "
            "Uniform(-1,1) random baseline."
        )

    add()

    # ==================================================
    # 6. Reward
    # ==================================================

    r = rewards.reshape(-1)

    add("Reward:")

    add(
        f"    Return: {r.sum():.6f}"
    )

    add(
        f"    Mean:   {r.mean():.6f}"
    )

    add(
        f"    Max:    {r.max():.6f}"
    )

    add(
        f"    Final:  {r[-1]:.6f}"
    )

    add(
        "    INFO: no explicit success label is "
        "available."
    )

    add(
        "    Success should not be inferred from "
        "reward alone without the environment's "
        "task definition."
    )

    add()

    # ==================================================
    # 7. Recommended Use
    # ==================================================

    add("Recommended Use:")

    if integrity_pass and action_in_range:
        add(
            "    YES: pipeline / parser / converter "
            "testing."
        )
    else:
        add(
            "    NO: fix structural data issues before "
            "further use."
        )

    if random_score == 3:
        add(
            "    NO: not suitable as an expert "
            "imitation-learning demonstration."
        )
    else:
        add(
            "    REVIEW: demonstration quality requires "
            "task-level evaluation."
        )

    add()

    # ==================================================
    # 8. Overall
    # ==================================================

    add("Overall:")

    if not integrity_pass or not action_in_range:

        add(
            "    INVALID DATASET: structural problems "
            "detected."
        )

    elif random_score == 3:

        add(
            "    STRUCTURALLY VALID, BUT RANDOM-LIKE "
            "TRAJECTORY."
        )

    else:

        add(
            "    STRUCTURALLY VALID. "
            "Task quality requires further evaluation."
        )

    report = "\n".join(lines)

    print("\n" + report)

    output_file = (
        REPORT_DIR /
        "dataset_quality_summary.txt"
    )

    output_file.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"\nSaved report: {output_file}"
    )

def randomness_diagnostic(actions):
    print("\n===== Randomness Diagnostic =====")

    # Flatten all action dimensions together
    x = actions.reshape(-1)

    action_mean = x.mean()
    action_std = x.std()

    delta = np.diff(actions, axis=0)
    mean_abs_delta = np.abs(delta).mean()

    # Theoretical values for Uniform(-1, 1)
    expected_mean = 0.0
    expected_std = 1.0 / np.sqrt(3.0)
    expected_mean_abs_delta = 2.0 / 3.0

    print("Expected for independent U(-1, 1):")
    print(f"Mean:             {expected_mean:.4f}")
    print(f"Std:              {expected_std:.4f}")
    print(
        f"Mean |Δa|:        "
        f"{expected_mean_abs_delta:.4f}"
    )

    print("\nObserved:")
    print(f"Mean:             {action_mean:.4f}")
    print(f"Std:              {action_std:.4f}")
    print(
        f"Mean |Δa|:        "
        f"{mean_abs_delta:.4f}"
    )

    mean_error = abs(
        action_mean - expected_mean
    )

    std_error = abs(
        action_std - expected_std
    )

    delta_error = abs(
        mean_abs_delta - expected_mean_abs_delta
    )

    print("\nDeviation from random baseline:")
    print(f"Mean error:        {mean_error:.4f}")
    print(f"Std error:         {std_error:.4f}")
    print(f"Delta error:       {delta_error:.4f}")

    # Heuristic only — not a formal statistical test
    mean_like_random = mean_error < 0.10
    std_like_random = std_error < 0.10
    delta_like_random = delta_error < 0.10

    random_score = sum(
        [
            mean_like_random,
            std_like_random,
            delta_like_random,
        ]
    )

    print("\nAssessment:")

    if random_score == 3:
        print(
            "[WARNING] Action statistics are strongly "
            "consistent with independent Uniform(-1, 1) "
            "random sampling."
        )

    elif random_score == 2:
        print(
            "[INFO] Action statistics are partially "
            "consistent with random sampling."
        )

    else:
        print(
            "[INFO] Action statistics do not strongly "
            "match the Uniform(-1, 1) random baseline."
        )

    return {
        "mean": float(action_mean),
        "std": float(action_std),
        "mean_abs_delta": float(mean_abs_delta),
        "random_score": int(random_score),
    }

def run_quality_checks(obs, actions, rewards, timestamps):

        print("\n===== Quality Checks =====")

        issues = []

        # --------------------------------------------------
        # 1. Length consistency
        # --------------------------------------------------

        lengths = [
            len(obs),
            len(actions),
            len(rewards),
            len(timestamps),
        ]

        if len(set(lengths)) == 1:
            print("[PASS] All arrays have the same length.")
        else:
            print("[FAIL] Array lengths do not match.")
            issues.append("length_mismatch")

        # --------------------------------------------------
        # 2. NaN / Inf
        # --------------------------------------------------

        arrays = {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "timestamps": timestamps,
        }

        corrupted = False

        for name, x in arrays.items():

            if np.isnan(x).any() or np.isinf(x).any():
                print(
                    f"[FAIL] {name} contains NaN or Inf."
                )

                corrupted = True

        if not corrupted:
            print("[PASS] No NaN or Inf detected.")

        # --------------------------------------------------
        # 3. Action range
        # --------------------------------------------------

        if np.any(actions < -1.0) or np.any(actions > 1.0):

            print(
                "[FAIL] Action outside controller range [-1, 1]."
            )

            issues.append("action_out_of_range")

        else:

            print(
                "[PASS] All actions within [-1, 1]."
            )

        # --------------------------------------------------
        # 4. Timestamp
        # --------------------------------------------------

        dt = np.diff(timestamps)

        if np.all(dt > 0):

            print(
                "[PASS] Timestamps are monotonically increasing."
            )

        else:

            print(
                "[FAIL] Non-monotonic timestamps detected."
            )

            issues.append("timestamp_error")

        # --------------------------------------------------
        # 5. Action smoothness
        # --------------------------------------------------

        delta = np.abs(np.diff(actions, axis=0))

        mean_delta = delta.mean()

        print(
            f"[INFO] Mean |Δa| = {mean_delta:.4f}"
        )

        # --------------------------------------------------
        # 6. Compare with random-uniform baseline
        # --------------------------------------------------

        random_expected_delta = 2.0 / 3.0

        difference = abs(
            mean_delta - random_expected_delta
        )

        if difference < 0.1:
            print(
                "[WARNING] Action smoothness is close to "
                "independent Uniform(-1,1) random actions."
            )

            issues.append("random_like_actions")

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        print("\n===== Quality Summary =====")

        if not issues:

            print("No major issues detected.")

        else:

            for issue in issues:
                print(f"- {issue}")

def main():
    obs, actions, rewards, timestamps = load_h5()

    print_dataset_overview(
        obs,
        actions,
        rewards,
        timestamps,
    )

    print_reward_statistics(rewards)

    print_action_statistics(actions)

    print_temporal_statistics(timestamps)

    print_integrity_check(
        obs,
        actions,
        rewards,
        timestamps,
    )

    print_action_range_check(actions)

    print_action_smoothness(actions)

    plot_reward_curve(rewards)

    plot_action_dimension(
        actions,
        dim=0,
    )

    plot_action_dimension(
        actions,
        dim=7,
    )

    run_quality_checks(
        obs,
        actions,
        rewards,
        timestamps,
    )
    randomness_diagnostic(actions)

    generate_quality_summary(
        obs,
        actions,
        rewards,
        timestamps,
        timestamp_type="synthetic",
    )

if __name__ == "__main__":
    main()