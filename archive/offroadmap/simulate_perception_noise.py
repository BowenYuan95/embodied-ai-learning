import gymnasium as gym
import mani_skill.envs
import torch


NOISE_STD = 0.005   # 5 mm
SEED = 0


def main():

    torch.manual_seed(SEED)

    env = gym.make(
        "PickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )

    obs, _ = env.reset(seed=0)

    # ==========================================
    # Ground-truth simulator state
    # ==========================================

    tcp_pose = obs["extra"]["tcp_pose"]
    obj_pose = obs["extra"]["obj_pose"]
    goal_pos = obs["extra"]["goal_pos"]

    tcp_pos = tcp_pose[..., :3]
    obj_pos_gt = obj_pose[..., :3]

    tcp_to_obj_gt = (
        obs["extra"]["tcp_to_obj_pos"]
    )

    obj_to_goal_gt = (
        obs["extra"]["obj_to_goal_pos"]
    )

    # ==========================================
    # Simulated perception
    # ==========================================

    noise = torch.randn_like(
        obj_pos_gt
    ) * NOISE_STD

    obj_pos_est = (
        obj_pos_gt + noise
    )

    # ==========================================
    # Recompute derived features
    # ==========================================

    tcp_to_obj_est = (
        obj_pos_est - tcp_pos
    )

    obj_to_goal_est = (
        goal_pos - obj_pos_est
    )

    # ==========================================
    # Errors
    # ==========================================

    obj_error = (
        obj_pos_est - obj_pos_gt
    )

    tcp_to_obj_error = (
        tcp_to_obj_est - tcp_to_obj_gt
    )

    obj_to_goal_error = (
        obj_to_goal_est - obj_to_goal_gt
    )

    # ==========================================
    # Print
    # ==========================================

    print(
        "===== Perception Noise Simulation ====="
    )

    print(
        f"Noise std: {NOISE_STD * 1000:.1f} mm"
    )

    print("\nGround-truth object position:")
    print(obj_pos_gt)

    print("\nNoise:")
    print(noise)

    print("\nEstimated object position:")
    print(obj_pos_est)

    print("\n===== Error Propagation =====")

    print("\nObject position error:")
    print(obj_error)

    print("\nTCP -> Object error:")
    print(tcp_to_obj_error)

    print("\nObject -> Goal error:")
    print(obj_to_goal_error)

    print("\n===== Magnitudes =====")

    print(
        "Object position error norm:",
        torch.linalg.vector_norm(
            obj_error,
            dim=-1,
        )
    )

    print(
        "TCP->Object error norm:",
        torch.linalg.vector_norm(
            tcp_to_obj_error,
            dim=-1,
        )
    )

    print(
        "Object->Goal error norm:",
        torch.linalg.vector_norm(
            obj_to_goal_error,
            dim=-1,
        )
    )

    # ==========================================
    # Expected relationships
    # ==========================================

    tcp_error_match = torch.allclose(
        tcp_to_obj_error,
        obj_error,
    )

    goal_error_match = torch.allclose(
        obj_to_goal_error,
        -obj_error,
    )

    print("\n===== Dependency Check =====")

    if tcp_error_match:
        print(
            "[PASS] tcp_to_obj error "
            "equals object-position error."
        )
    else:
        print(
            "[FAIL] Unexpected tcp_to_obj error."
        )

    if goal_error_match:
        print(
            "[PASS] obj_to_goal error "
            "equals negative object-position error."
        )
    else:
        print(
            "[FAIL] Unexpected obj_to_goal error."
        )

    env.close()


if __name__ == "__main__":
    main()