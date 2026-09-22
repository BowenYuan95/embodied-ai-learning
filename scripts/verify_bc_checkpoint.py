"""Verify the saved Behavior-Cloning checkpoint in a fresh process.

Purpose
-------
`AGENTS.md` requires the baseline BC artifact to satisfy a closed-loop gate. Two
of its items are verification work rather than modelling work:

5. the checkpoint can be loaded in a **fresh process** and produces actions with
   the expected shape, range, units, frame, and control semantics;
7. evaluation reports task success across multiple seeded episodes (see
   ``scripts/evaluate_bc_closed_loop.py`` for that half).

This script covers item 5. It never reloads an in-memory ``state_dict``; it reads
``checkpoints/pickcube_bc_best.pt`` from disk, rebuilds the policy from the stored
dimensions, and then:

- asserts the observation/action contract against the live ``PickCube-v1``
  environment (``control_mode``, observation dimension, action dimension, action
  bounds);
- checks the produced action's dtype, shape, and the clipped range;
- reproduces the recorded validation MSE from the file alone, using the stored
  normalization statistics and the stored validation-episode list;
- reproduces the mean-action baseline from the stored training-episode list.

Usage (from the repository root, in the ``embodied`` environment):

    python scripts/verify_bc_checkpoint.py

Exit code 0 means every contract check passed and the recorded numbers were
reproduced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "pickcube_bc_best.pt"
DEFAULT_DATASET = REPO_ROOT / "datasets" / "pickcube" / "expert_episodes.h5"

# Recorded in the notebook; the fresh process must reproduce them.
RECORDED_VALIDATION_MSE = 0.23502295
RECORDED_BASELINE_MSE = 0.14210252

ENV_ID = "PickCube-v1"
OBS_MODE = "state"
CONTROL_MODE = "pd_joint_pos"
OBSERVATION_DIM = 42
ACTION_DIM = 8


class BehaviorCloningPolicy(nn.Module):
    """Must match the architecture defined in the training notebook exactly.

    Two hidden layers with ``ReLU`` and a linear 8-d output; the state dict
    enforces the shape agreement, so any drift shows up as a load failure.
    """

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
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(path, map_location="cpu")


def load_episode(path: Path, name: str):
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle[name]
        observations = group["observations"][:].astype(np.float32)
        actions = group["actions"][:].astype(np.float32)
    if len(observations) != len(actions) + 1:
        raise RuntimeError(
            f"{name}: expected T+1 observations for T actions, got "
            f"{observations.shape} and {actions.shape}"
        )
    return observations[:-1], actions


def load_episodes(path: Path, names):
    pairs = [load_episode(path, name) for name in names]
    observations = np.concatenate([p[0] for p in pairs], axis=0)
    actions = np.concatenate([p[1] for p in pairs], axis=0)
    return observations, actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    print(f"checkpoint : {args.checkpoint}")
    print(f"dataset    : {args.dataset}")
    print()

    # ---------------------------------------------------------------- file load
    checkpoint = load_checkpoint(args.checkpoint)
    required = {
        "model_state_dict",
        "observation_mean",
        "observation_std",
        "observation_dim",
        "action_dim",
        "hidden_dim",
        "best_epoch",
        "best_validation_loss",
        "training_episodes",
        "validation_episodes",
    }
    missing = sorted(required - set(checkpoint))
    check("checkpoint carries the full inference contract", not missing,
          f"missing={missing}" if missing else "metadata complete")
    if missing:
        return 1

    check("stored dimensions match the environment contract",
          checkpoint["observation_dim"] == OBSERVATION_DIM
          and checkpoint["action_dim"] == ACTION_DIM,
          f"observation_dim={checkpoint['observation_dim']}, "
          f"action_dim={checkpoint['action_dim']}")

    # ------------------------------------------------------- environment contract
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers the environments

    env = gym.make(ENV_ID, obs_mode=OBS_MODE, control_mode=CONTROL_MODE, num_envs=1)
    try:
        actual_mode = str(env.unwrapped.control_mode)
        action_space = env.action_space
        observation, _ = env.reset(seed=0)
        observation_dim = int(np.prod(observation.shape))

        check("control mode matches the training contract",
              actual_mode == CONTROL_MODE, f"environment={actual_mode!r}")
        check("action space dimension", tuple(action_space.shape) == (ACTION_DIM,),
              f"shape={tuple(action_space.shape)}")
        check("observation dimension", observation_dim == OBSERVATION_DIM,
              f"shape={tuple(observation.shape)}")

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32).reshape(1, -1)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32).reshape(1, -1)
    finally:
        env.close()

    # ------------------------------------------------------------ model contract
    model = BehaviorCloningPolicy(
        observation_dim=checkpoint["observation_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        load_ok, load_detail = True, "state dict loaded with strict=True"
    except RuntimeError as error:
        load_ok, load_detail = False, str(error)
    check("state dict loads into the rebuilt architecture", load_ok, load_detail)
    if not load_ok:
        return 1

    model.eval()

    observation_mean = checkpoint["observation_mean"].float().reshape(1, -1)
    observation_std = checkpoint["observation_std"].float().reshape(1, -1)
    check("normalization tensors have the observation shape",
          observation_mean.shape == (1, OBSERVATION_DIM)
          and observation_std.shape == (1, OBSERVATION_DIM),
          f"mean={tuple(observation_mean.shape)}, std={tuple(observation_std.shape)}")
    check("normalization std is strictly positive",
          bool((observation_std > 0).all()),
          f"min std={float(observation_std.min()):.3e}")

    # The evaluation environment for semantic checks uses the same contract.
    env = gym.make(ENV_ID, obs_mode=OBS_MODE, control_mode=CONTROL_MODE, num_envs=1)
    try:
        observation, _ = env.reset(seed=0)
        observation_tensor = torch.as_tensor(
            np.asarray(observation), dtype=torch.float32
        ).reshape(1, -1)
        with torch.no_grad():
            raw_action = model((observation_tensor - observation_mean) / observation_std)
        clipped_action = torch.maximum(torch.minimum(raw_action, action_high), action_low)

        check("output shape and dtype",
              tuple(raw_action.shape) == (1, ACTION_DIM) and raw_action.dtype == torch.float32,
              f"shape={tuple(raw_action.shape)}, dtype={raw_action.dtype}")
        check("clipped action lies inside the action bounds",
              bool((clipped_action >= action_low - 1e-6).all()
                   and (clipped_action <= action_high + 1e-6).all()),
              f"raw range=[{float(raw_action.min()):.3f}, {float(raw_action.max()):.3f}]")
    finally:
        env.close()

    # ------------------------------------------------------ recorded-number replay
    validation_names = list(checkpoint["validation_episodes"])
    training_names = list(checkpoint["training_episodes"])
    check("stored episode lists are non-empty",
          len(validation_names) >= 1 and len(training_names) >= 1,
          f"train={training_names}, validation={validation_names}")

    validation_observations, validation_actions = load_episodes(args.dataset, validation_names)
    training_observations, training_actions = load_episodes(args.dataset, training_names)

    normalized_validation = (
        validation_observations - observation_mean.numpy()
    ) / observation_std.numpy()

    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized_validation.astype(np.float32)))
    validation_mse = float(np.mean((predictions.numpy() - validation_actions) ** 2))

    check("recorded validation MSE reproduced in a fresh process",
          abs(validation_mse - RECORDED_VALIDATION_MSE) < 1e-5,
          f"recomputed={validation_mse:.8f}, recorded={RECORDED_VALIDATION_MSE:.8f}, "
          f"checkpoint={float(checkpoint['best_validation_loss']):.8f}")

    mean_training_action = training_actions.mean(axis=0, keepdims=True)
    baseline_mse = float(np.mean((mean_training_action - validation_actions) ** 2))
    check("recorded mean-action baseline reproduced",
          abs(baseline_mse - RECORDED_BASELINE_MSE) < 1e-4,
          f"recomputed={baseline_mse:.8f}, recorded={RECORDED_BASELINE_MSE:.8f}")
    check("the checkpoint is still worse than the baseline (expected result)",
          validation_mse > baseline_mse,
          f"model={validation_mse:.4f} vs baseline={baseline_mse:.4f} "
          f"(+{validation_mse - baseline_mse:.4f})")

    # ------------------------------------------------- raw-output range diagnostic
    with torch.no_grad():
        raw = model(torch.from_numpy(normalized_validation.astype(np.float32)))
    out_of_range = ((raw < action_low) | (raw > action_high)).numpy()
    print()
    print(f"raw (unclipped) actions outside the control bounds, held-out episode: "
          f"{int(out_of_range.sum())}/{out_of_range.size} values "
          f"({100.0 * out_of_range.mean():.1f}%)")
    print(f"best epoch from the checkpoint: {checkpoint['best_epoch']} | "
          f"hidden_dim={checkpoint['hidden_dim']}")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s)): {failures}")
        return 1
    print("RESULT: PASS — checkpoint item 5 verified in a fresh process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
