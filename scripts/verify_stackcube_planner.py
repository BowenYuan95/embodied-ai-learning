#!/usr/bin/env python3
"""Step 1 go/no-go: can the StackCube planner produce BOTH "pick red" and "pick green"?

Must be run where SAPIEN has a working render/GPU device. The agent's sandboxed shell falls back
to CPU and the planner segfaults there (exit 139), so this script exists to be run by the
learner. It prints the device first and refuses to draw conclusions if it is not usable.

Runs three configurations per seed:

    A  pick=cubeA (red), full stack      -> the environment's own success must become True
    B  pick=cubeB (green), full stack    -> the mirror; the env's check tests red-on-green only,
                                            so evaluate the stack geometrically instead
    C  pick=cubeB (green), truncated     -> the "pick the named cube" demonstration the
                                            grounding experiment actually needs

It then reports the left/right distribution of the two colours across seeds, which is leakage
check #4 from the design: if one colour is systematically on one side, a language-blind policy
can score without reading the instruction.

    python scripts/verify_stackcube_planner.py --seeds 0 1 2
    python scripts/verify_stackcube_planner.py --seeds 0 1 2 --check-color
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def make_env(obs_mode="state"):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the env ids)
    return gym.make("StackCube-v1", obs_mode=obs_mode, render_mode=None, num_envs=1,
                    control_mode="pd_joint_pos")


def cube_positions(env):
    u = env.unwrapped
    return (u.cubeA.pose.p[0].cpu().numpy().copy(),
            u.cubeB.pose.p[0].cpu().numpy().copy(),
            float(u.cube_half_size[2]))


def run_one(seed, pick, truncate):
    from stackcube_solutions import solve, stacked
    env = make_env()
    try:
        out = dict(seed=seed, pick=pick, truncate=truncate, ok=False, error=None)
        before_a, before_b, half = cube_positions(env)
        out["pos_cubeA_before"] = np.round(before_a, 4).tolist()
        out["pos_cubeB_before"] = np.round(before_b, 4).tolist()
        out["cubeA_left_of_cubeB"] = bool(before_a[0] < before_b[0])

        res = solve(env, seed=seed, debug=False, vis=False, pick=pick, truncate_at_lift=truncate)

        after_a, after_b, half = cube_positions(env)
        out["pos_cubeA_after"] = np.round(after_a, 4).tolist()
        out["pos_cubeB_after"] = np.round(after_b, 4).tolist()
        out["lift_gain_A"] = round(float(after_a[2] - before_a[2]), 4)
        out["lift_gain_B"] = round(float(after_b[2] - before_b[2]), 4)

        try:
            out["env_eval"] = {k: bool(v.cpu().numpy()[0])
                               for k, v in env.unwrapped.evaluate().items()}
        except Exception as exc:                                   # noqa: BLE001
            out["env_eval"] = f"{type(exc).__name__}: {exc}"

        ok_stack_A, diag_A = stacked(env, "cubeA", "cubeB")
        ok_stack_B, diag_B = stacked(env, "cubeB", "cubeA")
        out["stacked_cubeA_on_cubeB"] = bool(ok_stack_A)
        out["stacked_cubeB_on_cubeA"] = bool(ok_stack_B)
        out["stack_diag"] = {k: round(float(v), 4) for k, v in
                             (diag_A if pick == "cubeA" else diag_B).items()}
        out["solver_return"] = repr(res)[:120]

        if pick == "cubeA" and not truncate:
            out["ok"] = bool(out["env_eval"].get("success", False)) if isinstance(
                out["env_eval"], dict) else False
        elif pick == "cubeB" and not truncate:
            out["ok"] = out["stacked_cubeB_on_cubeA"]
        else:                                    # truncated: the picked cube must have been lifted
            gain = out["lift_gain_B"] if pick == "cubeB" else out["lift_gain_A"]
            out["ok"] = gain > 0.05
        return out
    except Exception as exc:                                       # noqa: BLE001
        return dict(seed=seed, pick=pick, truncate=truncate, ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc()[-800:])
    finally:
        try:
            env.close()
        except Exception:                                          # noqa: BLE001
            pass


def color_check(seeds):
    """Is red vs green separable in the rendered frame, and where are they?

    Leakage check #3/#4: if the colours are not distinguishable at the render resolution the
    task is impossible rather than grounding-requiring, and if they are always on a fixed side
    the instruction is not needed.
    """
    env = make_env(obs_mode="rgb")
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            img = obs["image"] if isinstance(obs, dict) else obs
            x = np.asarray(img[0].cpu().numpy()).astype(np.float32)
            while x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[0] != x.shape[-1]:
                x = x[0] if x.shape[0] == 1 else np.transpose(x, (1, 2, 0))
            r, g = x[..., 0], x[..., 1]
            red = (r > 1.4 * g) & (r > 60)
            green = (g > 1.4 * r) & (g > 60)
            def centroid(mask):
                if mask.sum() == 0:
                    return None
                ys, xs = np.nonzero(mask)
                return [round(float(xs.mean()), 1), round(float(ys.mean()), 1), int(mask.sum())]
            rows.append(dict(seed=seed, shape=list(x.shape), red=centroid(red), green=centroid(green)))
    except Exception as exc:                                       # noqa: BLE001
        return dict(error=f"{type(exc).__name__}: {exc}", rows=rows)
    finally:
        try:
            env.close()
        except Exception:                                          # noqa: BLE001
            pass
    return dict(rows=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--check-color", action="store_true",
                    help="also render t=0 and measure red/green separability and their x positions")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "scripts" / "reports" /
                    "stackcube_planner_check.json")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}   (the planner segfaults on a CPU-only fallback; if this says cpu,")
    print(f"                      stop and re-run where SAPIEN has a working render device)")

    results = []
    for seed in args.seeds:
        for pick, truncate in (("cubeA", False), ("cubeB", False), ("cubeB", True)):
            r = run_one(seed, pick, truncate)
            tag = f"seed {seed}  pick={pick:<5} {'truncated' if truncate else 'full     '}"
            status = "OK  " if r.get("ok") else "FAIL"
            print(f"[{status}] {tag}  {r.get('error') or ''}")
            if not r.get("ok") and not r.get("error"):
                print(f"         eval={r.get('env_eval')}  stackB_on_A={r.get('stacked_cubeB_on_cubeA')}"
                      f"  lift_gain_A={r.get('lift_gain_A')} lift_gain_B={r.get('lift_gain_B')}"
                      f"  diag={r.get('stack_diag')}")
            results.append(r)

    parsed = [r for r in results if not r.get("error")]
    if parsed:
        left = [r["cubeA_left_of_cubeB"] for r in parsed]
        print(f"\ncubeA (red) left of cubeB (green): {sum(left)}/{len(left)}")
        print("  if this is 0/0 or all-one-sided, one colour sits on a fixed side and a")
        print("  language-blind policy could score without reading the instruction (leak #4).")

    payload = dict(device=device, results=results)
    if args.check_color:
        cc = color_check(args.seeds)
        payload["color_check"] = cc
        print("\ncolour check (t=0 render):")
        if cc.get("error"):
            print(f"  error: {cc['error']}")
        for row in cc.get("rows", []):
            print(f"  seed {row['seed']}: shape {row['shape']}  red={row['red']}  green={row['green']}")
        print("  red/green must both be non-None, i.e. each colour occupies visible pixels.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(results)} runs OK")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
