import gymnasium as gym
import mani_skill.envs

from scripts.pipeline.observation_adapter import (
    build_deployment_safe_observation,
    build_privileged_observation,
)


def main():

    env = gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    obs, _ = env.reset(seed=0)

    deployment_obs = (
        build_deployment_safe_observation(
            obs
        )
    )

    privileged_obs = (
        build_privileged_observation(
            obs
        )
    )

    print(
        "===== Observation Adapter ====="
    )

    print(
        "Deployment-safe:",
        deployment_obs.shape,
    )

    print(
        "Privileged:",
        privileged_obs.shape,
    )

    assert deployment_obs.shape[-1] == 28
    assert privileged_obs.shape[-1] == 14

    assert (
        deployment_obs.shape[-1]
        + privileged_obs.shape[-1]
        == 42
    )

    print(
        "\n[PASS] Observation partition: "
        "28D deployment + 14D privileged = 42D."
    )

    env.close()


if __name__ == "__main__":
    main()