import torch


NUM_SAMPLES = 100_000

NOISE_LEVELS_MM = [
    1,
    5,
    10,
    20,
]


def simulate_noise(std_mm):

    std_m = std_mm / 1000.0

    noise = torch.randn(
        NUM_SAMPLES,
        3,
    ) * std_m

    error_norm = torch.linalg.vector_norm(
        noise,
        dim=-1,
    )

    return {
        "axis_std_mm": std_mm,

        "mean_error_mm": (
            error_norm.mean().item()
            * 1000
        ),

        "median_error_mm": (
            error_norm.median().item()
            * 1000
        ),

        "p95_error_mm": (
            torch.quantile(
                error_norm,
                0.95,
            ).item()
            * 1000
        ),

        "max_error_mm": (
            error_norm.max().item()
            * 1000
        ),
    }


def main():

    torch.manual_seed(0)

    print(
        "===== Perception Noise Monte Carlo ====="
    )

    print(
        f"Samples per condition: "
        f"{NUM_SAMPLES}"
    )

    print()

    print(
        f"{'Axis Std':>10}"
        f"{'Mean 3D':>12}"
        f"{'Median':>12}"
        f"{'P95':>12}"
        f"{'Max':>12}"
    )

    print(
        f"{'(mm)':>10}"
        f"{'(mm)':>12}"
        f"{'(mm)':>12}"
        f"{'(mm)':>12}"
        f"{'(mm)':>12}"
    )

    for std_mm in NOISE_LEVELS_MM:

        result = simulate_noise(
            std_mm
        )

        print(
            f"{result['axis_std_mm']:>10.1f}"
            f"{result['mean_error_mm']:>12.2f}"
            f"{result['median_error_mm']:>12.2f}"
            f"{result['p95_error_mm']:>12.2f}"
            f"{result['max_error_mm']:>12.2f}"
        )


if __name__ == "__main__":
    main()