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
    """Exactly the recipe ``scripts/generate_expert_demo.py`` already uses successfully.

    Two things here are load-bearing, and the first version of this script got both wrong:

    (a) ``render_mode`` must **not** be forced to ``None``. Passing it skips the render-device
        setup that the motion planner depends on, and SAPIEN segfaults (observed as
        ``exit 139`` with ``device = cuda``, so this was never a CPU-fallback problem).
    (b) the environment must be **reset before any actor pose is read**; SAPIEN state is
        uninitialised until then and touching ``pose.p`` segfaults.

    ``CONTROL_MODE`` is imported from the collector rather than repeated, so the two cannot
    drift apart. ``gym.make`` without ``control_mode`` would give StackCube its default
    ``pd_joint_delta_pos``, which silently changes action semantics.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the env ids)
    from generate_expert_demo import CONTROL_MODE
    return gym.make("StackCube-v1", obs_mode=obs_mode, num_envs=1, control_mode=CONTROL_MODE)


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
        # Reset FIRST: actor poses are uninitialised before this and reading them segfaults.
        # solve() resets again with the same seed, so "before" still describes the episode.
        env.reset(seed=seed)
        before_a, before_b, half = cube_positions(env)
        out["control_mode"] = str(env.unwrapped.control_mode)
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
    """Are red and green separable in the rendered frame, and where are they?

    Obs extraction reuses ``StepRecorder`` rather than re-deriving it. Images live under
    ``obs["sensor_data"][CAMERA]["rgb"]``, NOT ``obs["image"]`` (the first version guessed and
    raised ``KeyError: 'image'``), and the recorder requires a combined mode, hence
    ``rgb+state``.

    Thresholds come from measurement, not guesswork. Using lesson 3.8.3's cube-to-pixel
    projection, the per-cube mean ``R-G`` is about +120 for the red cube (cubeA), -120 for the
    green cube (cubeB), and +66 for the table behind them. A window of |R-G| > 90 therefore
    separates all three; a plain ``R > G`` test does not, because the table is also red-dominant.

    This is deliberately a smoke test. The rigorous per-cube measurement (mean colour and pixel
    footprint from the depth map) belongs to step 3.
    """
    from generate_expert_demo import StepRecorder
    env = make_env(obs_mode="rgb+state")
    rows = []
    try:
        for seed in seeds:
            with StepRecorder(env, obs_mode="rgb+state") as rec:
                env.reset(seed=seed)
            if rec.reset_image is None:
                return dict(error="renderer produced no image (software rendering disabled?)",
                            rows=rows)
            x = np.asarray(rec.reset_image).astype(np.float32)
            rg = x[..., 0] - x[..., 1]

            def centroid(mask):
                if mask.sum() == 0:
                    return None
                ys, xs = np.nonzero(mask)
                return [round(float(xs.mean()), 1), round(float(ys.mean()), 1), int(mask.sum())]

            u = env.unwrapped
            rows.append(dict(
                seed=seed, shape=list(x.shape),
                red=centroid(rg > 90), green=centroid(rg < -90),
                cubeA_x=round(float(u.cubeA.pose.p[0, 0]), 4),
                cubeB_x=round(float(u.cubeB.pose.p[0, 0]), 4),
                cubeA_left=bool(u.cubeA.pose.p[0, 0] < u.cubeB.pose.p[0, 0])))
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
    ap.add_argument("--single", nargs=3, metavar=("SEED", "PICK", "TRUNCATE"),
                    help="internal: run one configuration in this process")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run every configuration in-process; a segfault then loses the sweep")
    args = ap.parse_args()

    if args.single:
        return run_single(int(args.single[0]), args.single[1], bool(int(args.single[2])))

    import subprocess
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    results = []
    for seed in args.seeds:
        for pick, truncate in CONFIGS:
            label = f"seed {seed}  pick={pick:<5} {'truncated' if truncate else 'full     '}"
            if args.no_isolate:
                r = run_one(seed, pick, truncate)
            else:
                # One child per configuration: a segfault then identifies WHICH configuration
                # crashes instead of destroying the whole sweep.
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     "--single", str(seed), pick, str(int(truncate))],
                    capture_output=True, text=True)
                r = next((json.loads(line[len(SENTINEL):]) for line in proc.stdout.splitlines()
                          if line.startswith(SENTINEL)), None)
                if r is None:
                    why = f"child exited with code {proc.returncode}"
                    if proc.returncode == -11:
                        why += " (SIGSEGV)"
                    r = dict(seed=seed, pick=pick, truncate=truncate, ok=False, error=why,
                             stderr=proc.stderr[-500:])
            results.append(r)
            status = "OK  " if r.get("ok") else "FAIL"
            print(f"[{status}] {label}  {r.get('error') or ''}")
            if not r.get("ok") and not r.get("error"):
                print(f"         control_mode={r.get('control_mode')}  eval={r.get('env_eval')}"
                      f"  stackB_on_A={r.get('stacked_cubeB_on_cubeA')}"
                      f"  lift_A={r.get('lift_gain_A')} lift_B={r.get('lift_gain_B')}"
                      f"  diag={r.get('stack_diag')}")
            elif r.get("stderr"):
                print(f"         stderr tail: {r['stderr'].splitlines()[-1][:120] if r['stderr'].splitlines() else ''}")

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
        print("\ncolour check (t=0 render, |R-G| > 90 window):")
        if cc.get("error"):
            print(f"  error: {cc['error']}")
        for row in cc.get("rows", []):
            print(f"  seed {row['seed']}: {row['shape']}  red={row['red']}  green={row['green']}")
            print(f"           cubeA_x={row.get('cubeA_x')} cubeB_x={row.get('cubeB_x')}"
                  f"  cubeA_left={row.get('cubeA_left')}")
        print("  Both must be non-None. WATCH THE PIXEL COUNTS: on this camera each cube is only")
        print("  5-10 px at 128x128, and the 3.8.5 CNN downsamples 16x, so raise the camera")
        print("  resolution before collecting or the model cannot see the colour at all.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(results)} runs OK")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
