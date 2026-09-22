import argparse
import sys
from pathlib import Path

# Make the repository root importable so this script works when invoked as
# `python scripts/run_pipeline.py` from the repository root, which is how the
# README documents it. Running a file puts the script's own directory on
# sys.path, not the working directory, so `import scripts.pipeline` would fail.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.loader import load_episode
from scripts.pipeline.validator import validate_episode
from scripts.pipeline.converter import convert_episode
from scripts.pipeline.post_validator import (
    load_converted_dataset,
    validate_converted_dataset,
)
from scripts.pipeline.reporter import (
    generate_report,
)
from scripts.pipeline.manifest import (
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert ManiSkill trajectories "
            "to LeRobot datasets."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite the existing "
            "LeRobot output dataset."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ==========================================
    # 1. Load
    # ==========================================

    episode = load_episode()

    print("===== Episode Loaded =====")

    for key, value in episode.items():
        print(
            f"{key:<15} "
            f"shape={value.shape}, "
            f"dtype={value.dtype}"
        )

    # ==========================================
    # 2. Validate
    # ==========================================

    print("\n===== Episode Validation =====")

    result = validate_episode(
        episode
    )

    if result["valid"]:
        print("PASS")
    else:
        print("FAIL")

    for warning in result["warnings"]:
        print(
            f"[WARNING] {warning}"
        )

    for error in result["errors"]:
        print(
            f"[ERROR] {error}"
        )

    # ==========================================
    # 3. Pre-conversion Quality Gate
    # ==========================================

    if not result["valid"]:

        print(
            "\nPipeline stopped: "
            "episode failed validation."
        )

        return

    print(
        "\nQuality gate passed."
    )

    # ==========================================
    # 4. Convert
    # ==========================================

    try:

        convert_episode(
            episode,
            overwrite=args.overwrite,
        )

    except FileExistsError as error:

        print("\nPipeline stopped:")
        print(error)

        return

    # ==========================================
    # 5. Post-conversion Validation
    # ==========================================

    converted_dataset = (
        load_converted_dataset()
    )

    post_result = (
        validate_converted_dataset(
            episode,
            converted_dataset,
        )
    )

    # ==========================================
    # 6. Post-conversion Quality Gate
    # ==========================================

    if not post_result["valid"]:

        print(
            "\nPipeline stopped: "
            "converted dataset failed validation."
        )

        return

    print(
        "\nPost-conversion quality gate passed."
    )


    # ==========================================
    # 7. Generate Dataset Report
    # ==========================================

    generate_report(
        episode,
        timestamp_type="synthetic",
    )

    print(
        "\nDataset is ready for downstream use."
    )

    # ==========================================
    # 8. Write Dataset Manifest
    # ==========================================

    write_manifest(
        episode,
        dataset_version="1.0",
        timestamp_source="synthetic",
    )

    print(
        "\nDataset is ready for downstream use."
    )

if __name__ == "__main__":
    main()