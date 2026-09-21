import torch


NUM_SAMPLES = 100_000

NOISE_LEVELS_MM = [
    1,
    5,
    10,
    20,
]

TASK_TOLERANCES_MM = [
    5,
    10,
    20,
    40,
]


def simulate_error_norm(
    noise_std_mm,
):

    noise_std_m = (
        noise_std_mm / 1000.0
    )

    noise = torch.randn(
        NUM_SAMPLES,
        3,
    ) * noise_std_m

    error_norm = (
        torch.linalg.vector_norm(
            noise,
            dim=-1,
        )
    )

    return error_norm


def main():

    torch.manual_seed(0)

    print(
        "===== Task-Relative Perception Robustness ====="
    )

    print(
        f"Samples per condition: "
        f"{NUM_SAMPLES}"
    )

    print()

    header = (
        f"{'Noise σ':>10}"
    )

    for tolerance in TASK_TOLERANCES_MM:
        header += (
            f"{f'≤{tolerance}mm':>12}"
        )

    print(header)

    for noise_std in NOISE_LEVELS_MM:

        error_norm = (
            simulate_error_norm(
                noise_std
            )
        )

        row = (
            f"{noise_std:>8.1f}mm"
        )

        for tolerance_mm in TASK_TOLERANCES_MM:

            tolerance_m = (
                tolerance_mm
                / 1000.0
            )

            within = (
                error_norm
                <= tolerance_m
            )

            success_rate = (
                within.float()
                .mean()
                .item()
                * 100
            )

            row += (
                f"{success_rate:>11.2f}%"
            )

        print(row)


if __name__ == "__main__":
    main()