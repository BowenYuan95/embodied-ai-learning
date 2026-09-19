from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


INPUT_FILE = Path(
    "datasets/pickcube/random_episode_standard.h5"
)


FEATURES = {

    "observation.state": {
        "dtype": "float32",
        "shape": (42,),
    },

    "action": {
        "dtype": "float32",
        "shape": (8,),
    },

}


def load_h5():

    with h5py.File(INPUT_FILE, "r") as f:

        obs = f["observations"][:].astype(np.float32)

        actions = f["actions"][:].astype(np.float32)

        timestamps = f["timestamps"][:]


    return obs, actions, timestamps



def convert():

    obs, actions, timestamps = load_h5()


    fps = round(
        1/(timestamps[1]-timestamps[0])
    )


    print(obs.shape)
    print(actions.shape)
    print("fps:", fps)



    dataset = LeRobotDataset.create(

        repo_id="pickcube",

        fps=fps,

        features=FEATURES,

        robot_type="maniskill",

    )


    for t in range(len(obs)):

        dataset.add_frame(
            {
                "observation.state": obs[t],
                "action": actions[t],
                "task": "pick up the cube",
            }
        )


    dataset.save_episode()

    dataset.finalize()


    print("DONE")



if __name__ == "__main__":
    convert()