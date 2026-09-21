import gymnasium as gym
import mani_skill.envs


def describe(name, x):
    print(
        f"{name:<15} "
        f"type={type(x)}, "
        f"shape={getattr(x, 'shape', None)}"
    )


def main():

    # ==========================================
    # 1. Create environment
    # ==========================================

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    print("===== ManiSkill PickCube Probe =====")

    print(
        f"Observation space:\n"
        f"{env.observation_space}"
    )

    print(
        f"\nAction space:\n"
        f"{env.action_space}"
    )

    # ==========================================
    # 2. Reset
    # ==========================================

    obs, info = env.reset(
        seed=0
    )

    print("\n===== Reset =====")

    describe(
        "observation",
        obs,
    )

    print(
        f"reset info keys: "
        f"{list(info.keys())}"
    )

    # ==========================================
    # 3. Sample ONE real environment action
    # ==========================================

    action = env.action_space.sample()

    print("\n===== Sampled Action =====")

    describe(
        "action",
        action,
    )

    print(action)

    # ==========================================
    # 4. Execute the action
    # ==========================================

    next_obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    print("\n===== After env.step(action) =====")

    describe(
        "next_observation",
        next_obs,
    )

    describe(
        "reward",
        reward,
    )

    describe(
        "terminated",
        terminated,
    )

    describe(
        "truncated",
        truncated,
    )

    print(
        f"step info keys: "
        f"{list(info.keys())}"
    )

    # ==========================================
    # 5. Basic causal check
    # ==========================================

    print("\n===== Transition =====")

    print(
        "We now have a real simulated transition:"
    )

    print(
        "observation_t "
        "-> action_t "
        "-> observation_t+1"
    )

    env.close()


if __name__ == "__main__":
    main()