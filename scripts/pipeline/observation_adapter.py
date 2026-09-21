import torch


def build_deployment_safe_observation(obs_dict):
    """
    Build a provisional real-deployable observation.

    Included:
        qpos
        qvel
        tcp_pose
        goal_pos

    Excluded:
        is_grasped
        obj_pose
        tcp_to_obj_pos
        obj_to_goal_pos
    """

    qpos = obs_dict["agent"]["qpos"]
    qvel = obs_dict["agent"]["qvel"]

    tcp_pose = obs_dict["extra"]["tcp_pose"]
    goal_pos = obs_dict["extra"]["goal_pos"]

    observation = torch.cat(
        [
            qpos,
            qvel,
            tcp_pose,
            goal_pos,
        ],
        dim=-1,
    )

    return observation


def build_privileged_observation(obs_dict):
    """
    Simulator/environment-side information that is not
    assumed to be directly available on the real robot.
    """

    is_grasped = obs_dict["extra"]["is_grasped"]

    # is_grasped is [B], turn it into [B, 1]
    if is_grasped.ndim == 1:
        is_grasped = is_grasped.unsqueeze(-1)

    obj_pose = obs_dict["extra"]["obj_pose"]

    tcp_to_obj_pos = (
        obs_dict["extra"]["tcp_to_obj_pos"]
    )

    obj_to_goal_pos = (
        obs_dict["extra"]["obj_to_goal_pos"]
    )

    observation = torch.cat(
        [
            is_grasped,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_goal_pos,
        ],
        dim=-1,
    )

    return observation