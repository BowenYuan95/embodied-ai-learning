import gymnasium as gym
import mani_skill.envs


env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
)


env.reset()

real_env = env.unwrapped


agent = real_env.agent


print("========================")
print("URDF:")
print(agent.urdf_path)


print("========================")
print("SRDF:")
print(
    agent.urdf_path.replace(".urdf", ".srdf")
)


print("========================")
print("Links:")

for link in agent.robot.get_links():
    print(link.get_name())


print("========================")
print("Joints:")

for joint in agent.robot.get_active_joints():
    print(joint.get_name())