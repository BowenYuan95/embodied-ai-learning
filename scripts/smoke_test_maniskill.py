import gymnasium as gym
import mani_skill.envs
import torch


def main():

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )

    obs, info = env.reset(seed=0)

    print("=== Environment ===")
    print("Observation space:")
    print(env.observation_space)

    print("\nAction space:")
    print(env.action_space)

    print("\nObservation type:")
    print(type(obs))

    print("\n=== Rollout ===")

    for step in range(10):

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={step+1}, "
            f"reward={reward}, "
            f"success={info['success']}"
        )

    env.close()

    print("\nManiSkill smoke test finished.")


if __name__ == "__main__":
    main()
