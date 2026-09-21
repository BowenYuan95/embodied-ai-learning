import gymnasium as gym
import mani_skill.envs
import h5py
import torch
import numpy as np
from pathlib import Path


def main():

    save_path = Path(
        "datasets/pickcube/random_episode_000.h5"
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )


    obs, info = env.reset(seed=0)


    observations = []
    actions = []
    rewards = []


    max_steps = 50


    for step in range(max_steps):

        action = env.action_space.sample()

        observations.append(
            obs.cpu().numpy()
        )

        actions.append(
            action
        )


        obs, reward, terminated, truncated, info = env.step(
            action
        )

        rewards.append(
            reward.cpu().numpy()
        )


        if terminated or truncated:
            break


    env.close()


    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)


    print("Observation:")
    print(observations.shape)

    print("Actions:")
    print(actions.shape)


    with h5py.File(save_path, "w") as f:

        f.create_dataset(
            "observations",
            data=observations
        )

        f.create_dataset(
            "actions",
            data=actions
        )

        f.create_dataset(
            "rewards",
            data=rewards
        )


    print(
        f"Saved dataset to {save_path}"
    )


if __name__ == "__main__":
    main()