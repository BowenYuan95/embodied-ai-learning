import h5py
import gymnasium as gym
import mani_skill.envs
import numpy as np

from pathlib import Path


OUTPUT_FILE = Path(
    "datasets/pickcube/"
    "maniskill_random_rollout.h5"
)

NUM_STEPS = 50
SEED = 0


def to_numpy(x):
    """
    Convert torch.Tensor or other array-like
    object to NumPy.
    """
    if hasattr(x, "detach"):
        x = x.detach()

    if hasattr(x, "cpu"):
        x = x.cpu()

    if hasattr(x, "numpy"):
        x = x.numpy()

    return np.asarray(x)


def remove_env_batch(x):
    """
    ManiSkill with num_envs=1 returns values such as:

        observation: (1, 42)
        reward:      (1,)

    Remove the first environment dimension.
    """
    x = to_numpy(x)

    if x.shape[0] == 1:
        x = x[0]

    return x


def main():

    # ==========================================
    # 1. Environment
    # ==========================================

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    obs, info = env.reset(
        seed=SEED
    )

    print(
        "===== ManiSkill Random Rollout Collection ====="
    )

    print(
        f"Observation space: "
        f"{env.observation_space}"
    )

    print(
        f"Action space: "
        f"{env.action_space}"
    )

    # ==========================================
    # 2. Storage buffers
    # ==========================================

    observations = []
    actions = []
    next_observations = []

    rewards = []

    terminated_flags = []
    truncated_flags = []

    success_flags = []
    grasped_flags = []
    placed_flags = []
    static_flags = []

    elapsed_steps = []

    # ==========================================
    # 3. Rollout
    # ==========================================

    for t in range(NUM_STEPS):

        # Current state s_t
        current_obs = remove_env_batch(
            obs
        ).astype(np.float32)

        # Random policy: a_t
        action = env.action_space.sample().astype(
            np.float32
        )

        # Environment transition
        next_obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        # ======================================
        # Save transition
        # ======================================

        observations.append(
            current_obs
        )

        actions.append(
            action
        )

        next_observations.append(
            remove_env_batch(
                next_obs
            ).astype(np.float32)
        )

        rewards.append(
            float(
                remove_env_batch(reward)
            )
        )

        terminated_flags.append(
            bool(
                remove_env_batch(terminated)
            )
        )

        truncated_flags.append(
            bool(
                remove_env_batch(truncated)
            )
        )

        success_flags.append(
            bool(
                remove_env_batch(
                    info["success"]
                )
            )
        )

        grasped_flags.append(
            bool(
                remove_env_batch(
                    info["is_grasped"]
                )
            )
        )

        placed_flags.append(
            bool(
                remove_env_batch(
                    info["is_obj_placed"]
                )
            )
        )

        static_flags.append(
            bool(
                remove_env_batch(
                    info["is_robot_static"]
                )
            )
        )

        elapsed_steps.append(
            int(
                remove_env_batch(
                    info["elapsed_steps"]
                )
            )
        )

        print(
            f"step={t:03d} "
            f"reward={rewards[-1]:.4f} "
            f"success={success_flags[-1]} "
            f"grasped={grasped_flags[-1]}"
        )

        # ======================================
        # Advance state
        # ======================================

        obs = next_obs

        done = (
            terminated_flags[-1]
            or truncated_flags[-1]
        )

        if done:
            print(
                f"Episode ended at step {t}."
            )
            break

    env.close()

    # ==========================================
    # 4. Convert to arrays
    # ==========================================

    observations = np.asarray(
        observations,
        dtype=np.float32,
    )

    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    next_observations = np.asarray(
        next_observations,
        dtype=np.float32,
    )

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    ).reshape(-1, 1)

    terminated_flags = np.asarray(
        terminated_flags,
        dtype=bool,
    ).reshape(-1, 1)

    truncated_flags = np.asarray(
        truncated_flags,
        dtype=bool,
    ).reshape(-1, 1)

    success_flags = np.asarray(
        success_flags,
        dtype=bool,
    ).reshape(-1, 1)

    grasped_flags = np.asarray(
        grasped_flags,
        dtype=bool,
    ).reshape(-1, 1)

    placed_flags = np.asarray(
        placed_flags,
        dtype=bool,
    ).reshape(-1, 1)

    static_flags = np.asarray(
        static_flags,
        dtype=bool,
    ).reshape(-1, 1)

    elapsed_steps = np.asarray(
        elapsed_steps,
        dtype=np.int32,
    )

    # ==========================================
    # 5. Save HDF5
    # ==========================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        OUTPUT_FILE,
        "w",
    ) as f:

        f.create_dataset(
            "observations",
            data=observations,
        )

        f.create_dataset(
            "actions",
            data=actions,
        )

        f.create_dataset(
            "next_observations",
            data=next_observations,
        )

        f.create_dataset(
            "rewards",
            data=rewards,
        )

        f.create_dataset(
            "terminated",
            data=terminated_flags,
        )

        f.create_dataset(
            "truncated",
            data=truncated_flags,
        )

        f.create_dataset(
            "success",
            data=success_flags,
        )

        f.create_dataset(
            "is_grasped",
            data=grasped_flags,
        )

        f.create_dataset(
            "is_obj_placed",
            data=placed_flags,
        )

        f.create_dataset(
            "is_robot_static",
            data=static_flags,
        )

        f.create_dataset(
            "elapsed_steps",
            data=elapsed_steps,
        )

        # Dataset-level metadata
        f.attrs["environment"] = "PickCube-v1"
        f.attrs["obs_mode"] = "state"
        f.attrs["control_mode"] = (
            "pd_joint_delta_pos"
        )
        f.attrs["policy"] = "random"
        f.attrs["seed"] = SEED

    # ==========================================
    # 6. Summary
    # ==========================================

    print(
        "\n===== Collection Complete ====="
    )

    print(
        f"observations:      "
        f"{observations.shape}"
    )

    print(
        f"actions:           "
        f"{actions.shape}"
    )

    print(
        f"next_observations: "
        f"{next_observations.shape}"
    )

    print(
        f"rewards:           "
        f"{rewards.shape}"
    )

    print(
        f"success:           "
        f"{success_flags.shape}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()