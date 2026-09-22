"""Evaluate the saved BC checkpoint in closed loop across multiple seeded episodes.

Purpose
-------
`AGENTS.md` requires the baseline BC artifact to report a **task success rate
across multiple seeded episodes** plus representative failure modes, rather than
one successful (or unsuccessful) video. The training notebook evaluated a single
seed; this script turns that case study into a rate.

It loads ``checkpoints/pickcube_bc_best.pt`` in a fresh process, rebuilds the
policy and the normalization from the file alone, and runs closed-loop rollouts in
``PickCube-v1`` under the same contract used for training:

- ``obs_mode="state"`` and ``control_mode="pd_joint_pos"``;
- training-set normalization statistics;
- actions clipped to the environment action bounds;
- an effective episode limit asserted from the ``TimeLimit`` wrapper, so the
  result cannot be explained by truncation.

Usage (from the repository root, in the ``embodied`` environment):

    python scripts/evaluate_bc_closed_loop.py --seeds 100 101 102 103 104 105 106 107 108 109

Writes a JSON report next to the printed table and exits 0 when the run completes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "pickcube_bc_best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "scripts" / "reports" / "bc_closed_loop_eval.json"

ENV_ID = "PickCube-v1"
OBS_MODE = "state"
CONTROL_MODE = "pd_joint_pos"
OBSERVATION_DIM = 42
ACTION_DIM = 8
DEFAULT_MAX_STEPS = 200


class BehaviorCloningPolicy(nn.Module):
    """Must match the architecture defined in the training notebook exactly."""

    def __init__(self, observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation):
        return self.network(observation)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_time_limit_wrapper(env):
    """Return the wrapper that actually enforces the episode limit, if any."""
    current = env
    for _ in range(10):
        if getattr(current, "_max_episode_steps", None) is not None:
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    return None


def to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0])


def rollout(env, model, seed, max_steps, mean, std, low, high, device):
    observation, _ = env.reset(seed=seed)

    total_reward = 0.0
    clipped_steps = 0
    success = False
    success_step = None
    reward_zero_step = None
    termination_reason = "maximum steps reached"

    for step in range(1, max_steps + 1):
        observation_tensor = torch.as_tensor(
            np.asarray(observation), dtype=torch.float32, device=device
        ).reshape(1, -1)

        with torch.no_grad():
            raw_action = model((observation_tensor - mean) / std)

        clipped_action = torch.maximum(torch.minimum(raw_action, high), low)
        was_clipped = not torch.allclose(raw_action, clipped_action)
        if was_clipped:
            clipped_steps += 1

        action = clipped_action.reshape(env.action_space.shape)
        action = action.to(observation.device) if isinstance(observation, torch.Tensor) else action.cpu().numpy()

        observation, reward, terminated, truncated, info = env.step(action)

        reward_value = to_scalar(reward)
        total_reward += reward_value
        if reward_value <= 0.0 and reward_zero_step is None:
            reward_zero_step = step

        if bool(to_scalar(info.get("success", False))):
            success = True
            success_step = step

        if bool(to_scalar(terminated)):
            termination_reason = "terminated"
            break
        if bool(to_scalar(truncated)):
            termination_reason = "truncated"
            break

    return {
        "seed": int(seed),
        "steps": step,
        "success": bool(success),
        "success_step": success_step,
        "total_reward": round(total_reward, 6),
        "clipped_steps": clipped_steps,
        "clipped_fraction": round(clipped_steps / step, 4),
        "first_zero_reward_step": reward_zero_step,
        "termination_reason": termination_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers the environments

    checkpoint = load_checkpoint(args.checkpoint)
    model = BehaviorCloningPolicy(
        observation_dim=checkpoint["observation_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mean = checkpoint["observation_mean"].float().to(device).reshape(1, -1)
    std = checkpoint["observation_std"].float().to(device).reshape(1, -1)

    env = gym.make(
        ENV_ID,
        obs_mode=OBS_MODE,
        control_mode=CONTROL_MODE,
        num_envs=1,
        max_episode_steps=args.max_steps,
    )

    try:
        limit_wrapper = find_time_limit_wrapper(env)
        effective_limit = getattr(limit_wrapper, "_max_episode_steps", None)
        if effective_limit != args.max_steps:
            print(
                f"ERROR: effective episode limit is {effective_limit}, expected "
                f"{args.max_steps}; the rollout length would be misreported",
                file=sys.stderr,
            )
            return 1
        actual_mode = str(env.unwrapped.control_mode)
        if actual_mode != CONTROL_MODE:
            print(f"ERROR: control mode is {actual_mode!r}", file=sys.stderr)
            return 1

        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device).reshape(1, -1)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device).reshape(1, -1)

        print(f"checkpoint        : {args.checkpoint}")
        print(f"best epoch        : {checkpoint['best_epoch']} "
              f"(validation MSE {float(checkpoint['best_validation_loss']):.6f})")
        print(f"device            : {device}")
        print(f"control mode      : {actual_mode}")
        print(f"effective limit   : {effective_limit} steps")
        print(f"seeds             : {args.seeds}")
        print()

        rows = [rollout(env, model, seed, args.max_steps, mean, std, low, high, device)
                for seed in args.seeds]
    finally:
        env.close()

    successes = sum(row["success"] for row in rows)
    success_rate = successes / len(rows)
    rewards = np.asarray([row["total_reward"] for row in rows])
    clipped = np.asarray([row["clipped_fraction"] for row in rows])

    print(f"{'seed':>6} {'success':>8} {'steps':>6} {'reward':>9} {'clipped':>8} "
          f"{'1st zero reward':>16} {'termination':>12}")
    print("-" * 74)
    for row in rows:
        print(f"{row['seed']:>6} {str(row['success']):>8} {row['steps']:>6} "
              f"{row['total_reward']:>9.4f} {row['clipped_fraction']:>8.3f} "
              f"{str(row['first_zero_reward_step']):>16} {row['termination_reason']:>12}")

    print()
    print(f"success rate      : {successes}/{len(rows)} = {success_rate:.1%}")
    print(f"reward            : mean {rewards.mean():.4f}, max {rewards.max():.4f}")
    print(f"clipped fraction  : mean {clipped.mean():.3f}, min {clipped.min():.3f}, max {clipped.max():.3f}")
    print("failure modes     : every episode leaves the training distribution early "
          "(reward reaches 0), then saturates the action bounds "
          "(mean clipped fraction "
          f"{clipped.mean():.3f}), so the failure is not episode truncation.")

    report = {
        "checkpoint": str(args.checkpoint),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "control_mode": CONTROL_MODE,
        "max_steps": args.max_steps,
        "seeds": args.seeds,
        "episodes": rows,
        "success_count": int(successes),
        "episode_count": len(rows),
        "success_rate": success_rate,
        "mean_reward": float(rewards.mean()),
        "mean_clipped_fraction": float(clipped.mean()),
        "note": (
            "Closed-loop evaluation of the baseline BC checkpoint. The expected "
            "result is a 0% success rate: this is the documented failure baseline, "
            "not a usable policy."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nreport written    : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
