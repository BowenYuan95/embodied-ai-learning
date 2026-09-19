"""Replay the existing PickCube fixture; seed/config come from its collector."""

import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np


def main():
    with h5py.File("datasets/pickcube/random_episode_standard.h5", "r") as f:
        observations = f["observations"][:]
        actions = f["actions"][:]
        rewards = f["rewards"][:]

    assert len(observations) == len(actions) == len(rewards) > 0
    env = gym.make(
        "PickCube-v1", num_envs=1, obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )
    try:
        obs, _ = env.reset(seed=0)
        print("control_freq:", env.unwrapped.control_freq)
        successes = []
        max_obs_error = 0.0
        max_reward_error = 0.0
        for t, action in enumerate(actions):
            # The collector stores the observation BEFORE executing each action.
            current = obs.detach().cpu().numpy().reshape(-1)
            np.testing.assert_allclose(current, observations[t], rtol=0, atol=1e-6)
            max_obs_error = max(max_obs_error, float(np.max(np.abs(current - observations[t]))))
            obs, reward, terminated, truncated, info = env.step(action)
            actual_reward = reward.detach().cpu().numpy().reshape(-1)
            np.testing.assert_allclose(actual_reward, rewards[t].reshape(-1), rtol=0, atol=1e-6)
            max_reward_error = max(max_reward_error, float(np.max(np.abs(actual_reward - rewards[t].reshape(-1)))))
            successes.append(bool(info["success"].item()))
            if bool(terminated.item()) or bool(truncated.item()):
                print("boundary_step:", t + 1, "terminated:", bool(terminated.item()), "truncated:", bool(truncated.item()))
                if t + 1 != len(actions):
                    raise RuntimeError("Replay ended before all source actions were consumed")
                break
        print("steps_replayed:", len(successes))
        print("max_observation_error:", max_obs_error)
        print("max_reward_error:", max_reward_error)
        print("success_any:", any(successes), "success_final:", successes[-1])
    finally:
        env.close()


if __name__ == "__main__":
    main()
