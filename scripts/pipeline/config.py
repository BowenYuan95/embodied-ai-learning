from pathlib import Path


# ==========================================
# Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_DIR = PROJECT_ROOT / "scripts"

DATASET_DIR = PROJECT_ROOT / "datasets"

FIGURE_DIR = SCRIPT_DIR / "figures"

REPORT_DIR = SCRIPT_DIR / "reports"


# ==========================================
# Input dataset
# ==========================================

INPUT_FILE = (
    DATASET_DIR
    / "pickcube"
    / "random_episode_standard.h5"
)


# ==========================================
# Output directories
# ==========================================

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================
# Task metadata
# ==========================================

TASK_NAME = "pick up the cube"

REPO_ID = "pickcube"

ROBOT_TYPE = "maniskill"


# ==========================================
# Dataset schema
# ==========================================

OBSERVATION_DIM = 42

ACTION_DIM = 8


FEATURES = {

    "observation.state": {
        "dtype": "float32",
        "shape": (OBSERVATION_DIM,),
    },

    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
    },

}

LEROBOT_DIR = (
    DATASET_DIR
    / "lerobot"
    / REPO_ID
)