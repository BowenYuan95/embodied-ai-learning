import gymnasium as gym
import mani_skill.envs
import torch


def flatten_dict(data):

    values = []

    for value in data.values():

        if isinstance(value, dict):
            values.extend(
                flatten_dict(value)
            )

        else:
            values.append(
                value.reshape(
                    value.shape[0],
                    -1,
                )
            )

    return values


def main():

    # state
    env_state = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    state_obs, _ = env_state.reset(
        seed=0
    )

    # state_dict
    env_dict = gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    dict_obs, _ = env_dict.reset(
        seed=0
    )

    parts = flatten_dict(
        dict_obs
    )

    flattened = torch.cat(
        parts,
        dim=-1,
    )

    print(
        "state shape:",
        state_obs.shape,
    )

    print(
        "flattened state_dict shape:",
        flattened.shape,
    )

    error = torch.abs(
        state_obs - flattened
    )

    print(
        "Mean error:",
        error.mean().item(),
    )

    print(
        "Max error:",
        error.max().item(),
    )

    if torch.allclose(
        state_obs,
        flattened,
    ):
        print(
            "[PASS] state equals flattened state_dict."
        )
    else:
        print(
            "[FAIL] Flattening order differs."
        )

    env_state.close()
    env_dict.close()


if __name__ == "__main__":
    main()