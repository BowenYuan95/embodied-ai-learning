import gymnasium as gym
import mani_skill.envs


from mani_skill.examples.motionplanning.panda.solutions.pick_cube import (
    solve
)


# ==========================================
# Create ManiSkill environment
# ==========================================

env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human"
)


# ==========================================
# Run expert solver
# ==========================================

result = solve(
    env,
    debug=False,
    vis=False
)


print("======================")
print("Solver finished")
print(result)


# ==========================================
# Evaluate task success
# ==========================================

evaluation = env.unwrapped.evaluate()

print("======================")
print("Evaluation:")
print(evaluation)


env.close()