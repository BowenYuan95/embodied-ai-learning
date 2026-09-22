import gymnasium as gym
import mani_skill.envs

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver
)


# Create environment

env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
)


# Remove wrapper

real_env = env.unwrapped


# Reset

obs, info = real_env.reset()


print("Environment ready")


# Get robot base pose

robot_base_pose = real_env.agent.robot.pose


print("Robot pose:")
print(robot_base_pose)


print("Creating planner...")


planner = PandaArmMotionPlanningSolver(
    real_env,
    debug=False,
    vis=False,
    base_pose=robot_base_pose
)


print("Planner created!")