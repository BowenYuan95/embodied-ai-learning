import h5py
import numpy as np

from scripts.pipeline.config import INPUT_FILE


def load_episode(file_path=INPUT_FILE):

    with h5py.File(file_path, "r") as f:

        episode = {

            "observations":
                f["observations"][:].astype(
                    np.float32
                ),

            "actions":
                f["actions"][:].astype(
                    np.float32
                ),

            "rewards":
                f["rewards"][:].astype(
                    np.float32
                ),

            "timestamps":
                f["timestamps"][:].astype(
                    np.float64
                ),
        }

    return episode