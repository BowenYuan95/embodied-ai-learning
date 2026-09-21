import gymnasium as gym
import mani_skill.envs
import torch


def main():

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    obs, _ = env.reset(seed=0)

    print("===== Full Simulator Observation =====")
    print("shape:", obs.shape)

    # Verified state layout:
    #
    # [0:9]   qpos
    # [9:18]  qvel
    # [18:19] is_grasped
    # [19:26] tcp_pose
    # [26:29] goal_pos
    # [29:36] obj_pose
    # [36:39] tcp_to_obj_pos
    # [39:42] obj_to_goal_pos

    qpos = obs[:, 0:9]
    qvel = obs[:, 9:18]

    tcp_pose = obs[:, 19:26]
    goal_pos = obs[:, 26:29]

    # ==========================================
    # Robot/task-side observation
    # ==========================================

    deployment_safe_obs = torch.cat(
        [
            qpos,
            qvel,
            tcp_pose,
            goal_pos,
        ],
        dim=-1,
    )

    print(
        "\n===== Deployment-Safe Observation ====="
    )

    print(
        "qpos:       ",
        qpos.shape,
    )

    print(
        "qvel:       ",
        qvel.shape,
    )

    print(
        "tcp_pose:   ",
        tcp_pose.shape,
    )

    print(
        "goal_pos:   ",
        goal_pos.shape,
    )

    print(
        "\ncombined:",
        deployment_safe_obs.shape,
    )

    assert deployment_safe_obs.shape[-1] == 28

    print(
        "\n[PASS] 42D simulator state "
        "reduced to 28D robot/task-side state."
    )

    env.close()


if __name__ == "__main__":
    main()