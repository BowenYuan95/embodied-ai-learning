import gymnasium as gym
import mani_skill.envs
import torch


def main():

    env = gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    obs, _ = env.reset(seed=0)

    tcp_pose = obs["extra"]["tcp_pose"]
    obj_pose = obs["extra"]["obj_pose"]
    goal_pos = obs["extra"]["goal_pos"]

    tcp_to_obj_pos = (
        obs["extra"]["tcp_to_obj_pos"]
    )

    obj_to_goal_pos = (
        obs["extra"]["obj_to_goal_pos"]
    )

    # ------------------------------------------
    # Position components
    # ------------------------------------------

    tcp_pos = tcp_pose[..., :3]
    obj_pos = obj_pose[..., :3]

    # ------------------------------------------
    # Recompute relative geometry
    # ------------------------------------------

    expected_tcp_to_obj = (
        obj_pos - tcp_pos
    )

    expected_obj_to_goal = (
        goal_pos - obj_pos
    )

    # ------------------------------------------
    # Compare
    # ------------------------------------------

    print(
        "===== Relative Geometry Verification ====="
    )

    print("\nTCP position:")
    print(tcp_pos)

    print("\nObject position:")
    print(obj_pos)

    print("\nGoal position:")
    print(goal_pos)

    print("\n--- TCP -> Object ---")

    print(
        "ManiSkill:",
        tcp_to_obj_pos,
    )

    print(
        "Recomputed:",
        expected_tcp_to_obj,
    )

    tcp_error = torch.abs(
        tcp_to_obj_pos
        - expected_tcp_to_obj
    )

    print(
        "Max error:",
        tcp_error.max().item(),
    )

    print("\n--- Object -> Goal ---")

    print(
        "ManiSkill:",
        obj_to_goal_pos,
    )

    print(
        "Recomputed:",
        expected_obj_to_goal,
    )

    goal_error = torch.abs(
        obj_to_goal_pos
        - expected_obj_to_goal
    )

    print(
        "Max error:",
        goal_error.max().item(),
    )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    tcp_pass = torch.allclose(
        tcp_to_obj_pos,
        expected_tcp_to_obj,
    )

    goal_pass = torch.allclose(
        obj_to_goal_pos,
        expected_obj_to_goal,
    )

    print("\n===== Result =====")

    if tcp_pass:
        print(
            "[PASS] tcp_to_obj_pos "
            "= obj_pos - tcp_pos"
        )
    else:
        print(
            "[FAIL] tcp_to_obj_pos mismatch."
        )

    if goal_pass:
        print(
            "[PASS] obj_to_goal_pos "
            "= goal_pos - obj_pos"
        )
    else:
        print(
            "[FAIL] obj_to_goal_pos mismatch."
        )

    env.close()


if __name__ == "__main__":
    main()