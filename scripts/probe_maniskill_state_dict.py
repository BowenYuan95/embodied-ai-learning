import gymnasium as gym
import mani_skill.envs


def print_tree(data, prefix=""):

    if isinstance(data, dict):

        for key, value in data.items():

            name = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            print_tree(
                value,
                name,
            )

    else:

        print(
            f"{prefix:<45} "
            f"type={type(data).__name__:<12} "
            f"shape={getattr(data, 'shape', None)}"
        )


def main():

    env = gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    obs, info = env.reset(
        seed=0
    )

    print(
        "===== PickCube state_dict ====="
    )

    print_tree(obs)

    print(
        "\n===== Action Space ====="
    )

    print(
        env.action_space
    )

    env.close()


if __name__ == "__main__":
    main()