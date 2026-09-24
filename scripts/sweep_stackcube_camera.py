#!/usr/bin/env python3
"""Sweep StackCube camera settings and measure whether the cubes are learnable at all.

Why this exists
---------------
Leak #3 for the grounding experiment asked whether the red and green cubes can be told apart in
the rendered frame. They can -- measured mean ``R-G`` is about +120 for the red cube, -120 for
the green cube and +66 for the table -- but each cube occupies only **5-10 px** of a 128x128
frame, while the 3.8.5 CNN downsamples by 16x through four stride-2 blocks. An object that
small is gone by the last feature map, so a model would fail for a reason that has nothing to
do with grounding, and the 2x2 interaction would be uninterpretable.

This script measures the footprint directly (via lesson 3.8.3's depth projection) across
resolutions, fields of view and camera poses, and reports what survives at 16x, 8x and 4x
downsampling. Pick the cheapest configuration whose cube side length survives the downsampling
the model will actually use.

Runs on CPU software rendering, so it can be run by either party. It writes its own report path
and does not touch the planner check.

    python scripts/sweep_stackcube_camera.py
    python scripts/sweep_stackcube_camera.py --seeds 0 1 --out /tmp/sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mml_camera import CAMERA, box_mask, footprint, project, sensor_arrays  # noqa: E402

# eye / target as (x, y, z). The environment default is eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1],
# fov=pi/2 at 128x128; see StackCubeEnv._default_sensor_configs.
DEFAULT_EYE = [0.3, 0.0, 0.6]
DEFAULT_TARGET = [-0.1, 0.0, 0.1]

CONFIGS = [
    dict(name="default_128",        res=128, fov=np.pi / 2,  eye=DEFAULT_EYE,    target=DEFAULT_TARGET),
    dict(name="default_256",        res=256, fov=np.pi / 2,  eye=DEFAULT_EYE,    target=DEFAULT_TARGET),
    dict(name="default_512",        res=512, fov=np.pi / 2,  eye=DEFAULT_EYE,    target=DEFAULT_TARGET),
    dict(name="narrow_256",         res=256, fov=np.pi / 3,  eye=DEFAULT_EYE,    target=DEFAULT_TARGET),
    dict(name="narrow_512",         res=512, fov=np.pi / 3,  eye=DEFAULT_EYE,    target=DEFAULT_TARGET),
    dict(name="close_256",          res=256, fov=np.pi / 3,  eye=[0.24, 0.0, 0.45], target=[0.0, 0.0, 0.05]),
    dict(name="close_512",          res=512, fov=np.pi / 3,  eye=[0.24, 0.0, 0.45], target=[0.0, 0.0, 0.05]),
    dict(name="close_512_narrow",   res=512, fov=np.pi / 4,  eye=[0.24, 0.0, 0.45], target=[0.0, 0.0, 0.05]),
]


def make_env(cfg):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers the env ids)
    from mani_skill.utils import sapien_utils
    from generate_expert_demo import CONTROL_MODE

    pose = sapien_utils.look_at(eye=list(cfg["eye"]), target=list(cfg["target"]))
    # sensor_configs is a dict of overrides, NOT a list of CameraConfig objects -- see
    # mani_skill.sensors.camera.update_camera_configs_from_dict. Anything else raises
    # "pop expected at most 1 argument" because it tries dict.pop on a list.
    return gym.make("StackCube-v1", obs_mode="rgbd", num_envs=1, control_mode=CONTROL_MODE,
                    sensor_configs={"base_camera": {"width": cfg["res"], "height": cfg["res"],
                                                    "fov": cfg["fov"], "pose": pose}})


def measure(cfg, seeds):
    env = make_env(cfg)
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            rgb, _depth, points = sensor_arrays(obs, CAMERA)
            u = env.unwrapped
            half = np.asarray(u.cube_half_size, dtype=np.float64)
            K = obs["sensor_param"][CAMERA]["intrinsic_cv"][0].cpu().numpy()
            E = obs["sensor_param"][CAMERA]["extrinsic_cv"][0].cpu().numpy()
            out = dict(seed=seed, image=list(rgb.shape))

            # The colour must be BOTH readable and localised. Localisation is checked against
            # each cube's OWN projected pixel, not against world x-ordering: with the camera
            # off-axis, parallax means image x-order need not match world x-order, and
            # comparing the two made a correct measurement look like a failure.
            for name, attr in (("A", "cubeA"), ("B", "cubeB")):
                centre = getattr(u, attr).pose.p[0].cpu().numpy()
                out[attr] = footprint(rgb, box_mask(points, centre, half))
                uv = project(centre, K, E)[0]
                out[f"{attr}_proj_uv"] = [round(float(uv[0]), 1), round(float(uv[1]), 1)]
                out[f"{attr}_x"] = round(float(centre[0]), 4)
                if out[attr]["centroid"] is not None:
                    err = float(np.linalg.norm(np.array(out[attr]["centroid"]) - uv))
                    out[f"{attr}_centroid_err_px"] = round(err, 1)

            a, b = out["cubeA"], out["cubeB"]
            # Both masks must land near their own object. Read the error from `out`, where it
            # was written -- reading it off the footprint dict silently yielded None and made
            # every configuration report as unlocalised.
            tol = max(8.0, 0.15 * cfg["res"])
            ea, eb = out.get("cubeA_centroid_err_px"), out.get("cubeB_centroid_err_px")
            out["localised"] = bool(ea is not None and eb is not None and ea < tol and eb < tol)
            out["separation_px"] = (round(float(np.linalg.norm(
                np.array(a["centroid"]) - np.array(b["centroid"]))), 1)
                if a["centroid"] and b["centroid"] else None)
            rows.append(out)
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
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "scripts" / "reports" / "stackcube_camera_sweep.json")
    args = ap.parse_args()

    print(f"{'config':<20} {'res':>4} {'fov':>6} {'eye->cube':>10} {'A px':>6} {'A side':>7} "
          f"{'/16':>6} {'/8':>6} {'/4':>6} {'B px':>6} {'B side':>7} {'ok':>4} {'sep px':>7}")
    print("-" * 116)
    results = []
    for cfg in CONFIGS:
        res = measure(cfg, args.seeds)
        ok = [r for r in res.get("rows", []) if "A" not in r or r.get("cubeA")]
        if res.get("error") or not ok:
            print(f"{cfg['name']:<20} {cfg['res']:>4}  ERROR {res.get('error')}")
            results.append(dict(config=cfg["name"], error=res.get("error")))
            continue
        a_px = float(np.mean([r["cubeA"]["pixels"] for r in ok]))
        b_px = float(np.mean([r["cubeB"]["pixels"] for r in ok]))
        a_side, b_side = float(np.sqrt(a_px)), float(np.sqrt(b_px))
        dist = float(np.mean([np.linalg.norm(np.array(cfg["eye"]) - np.array([r["cubeA_x"], 0, 0.02]))
                              for r in ok]))
        tracks = sum(1 for r in ok if r.get("localised"))
        sep = float(np.mean([r["separation_px"] for r in ok if r.get("separation_px")]))
        print(f"{cfg['name']:<20} {cfg['res']:>4} {np.degrees(cfg['fov']):>5.0f}d {dist:>9.3f}m "
              f"{a_px:>6.0f} {a_side:>7.1f} {a_side/16:>6.2f} {a_side/8:>6.2f} {a_side/4:>6.2f} "
              f"{b_px:>6.0f} {b_side:>7.1f} {tracks}/{len(ok):>3} {sep:>7.0f}")
        results.append(dict(config=cfg["name"], res=cfg["res"], fov=float(cfg["fov"]),
                            eye=cfg["eye"], target=cfg["target"],
                            cubeA_px=a_px, cubeB_px=b_px, side_a=a_side, side_b=b_side,
                            tracks=tracks, n=len(ok), rows=ok))

    print()
    print("  'side' is sqrt(pixels), i.e. the side length if the footprint were square.")
    print("  A cube side must stay >= 1 px after the model's downsampling; the 3.8.5 CNN uses")
    print("  4 stride-2 blocks (16x), so aim for side/16 >= 2, i.e. side >= 32 px. If no")
    print("  configuration reaches that, the cheaper lever is the model: fewer stride-2 blocks")
    print("  or a higher-resolution stem, not more camera zoom.")
    print("  'ok' counts seeds where each colour mask lands within tolerance of its own")
    print("  projected pixel; 'sep px' is the mean distance between the two centroids.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(seeds=args.seeds, configs=results), indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
