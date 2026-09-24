"""Scripted direct-reach expert for the StackCube referring-grounding task.

Why a scripted expert instead of the motion planner
---------------------------------------------------
The motion planner's first actions were **identical for both tasks** (measured
``|a_0^red - a_0^green| = 0.00000`` in all eight channels), because its plan starts with a common
approach motion. The required action therefore did not differ at any frame where the state was
still uninformative, and the clean referential decision frame did not exist. See
``notes/progress.md``, step 3.

This controller commits to a side **at the first step**: the action is a proportional Cartesian
step ``a_t = K * (x_target - x_tcp)``, so ``sign(a[0])`` is the sign of the world-x direction to
the named cube. At ``t = 0`` the robot state is identical across layouts (with
``robot_init_qpos_noise = 0``) while this sign differs, which is the discrete referential
decision frame the experiment needs.

Action contract (measured, not assumed)
--------------------------------------
``control_mode = "pd_ee_delta_pos"``, action space ``(4,) = [dx, dy, dz, gripper]``, each in
``[-1, 1]``. Measured by commanding a single axis for ten steps:

    +x -> world displacement [+0.1389, +0.0002, +0.0019]
    +y -> [+0.0261, +0.3657, +0.0047]
    +z -> [+0.0203, 0.0000, +0.3582]

so the delta is in the **world frame** with cross-coupling an order of magnitude smaller than the
commanded axis, and the sign of ``a[0]`` reliably sets the direction of world-x motion. The
gripper is ``+1`` open, ``-1`` closed.

Note this is a **different action contract from lesson 3.8's** ``pd_joint_pos``, 8-dimensional
joint targets. That is deliberate: ``pd_joint_pos`` would need inverse kinematics for ``a = K e``
and therefore ``mplib``, which is exactly the dependency this controller removes. Datasets
produced here must record the changed control mode and action dimension.
"""

from __future__ import annotations

import numpy as np

ACTION_DIM = 4                  # [dx, dy, dz, gripper]
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0

HOVER = 0.10                    # metres above the cube centre for the pre-grasp pose
GRASP_DZ = 0.005                # TCP height relative to the cube centre when grasping
LIFT = 0.10                     # metres to lift after closing
K = 20.0                        # command per metre of error; 5 cm error saturates the command
TOL = 0.005                     # 5 mm position tolerance

BUDGET = dict(approach=60, descend=25, close=12, lift=18)


def _cube_pos(env, pick: str) -> np.ndarray:
    return getattr(env.unwrapped, pick).pose.p[0].cpu().numpy().astype(np.float64)


def _tcp_pos(env) -> np.ndarray:
    return env.unwrapped.agent.tcp.pose.p[0].cpu().numpy().astype(np.float64)


def _step_toward(env, target, budget, gripper, log):
    """Drive the TCP toward ``target`` with a proportional Cartesian delta, for a step budget."""
    for _ in range(budget):
        err = np.asarray(target, dtype=np.float64) - _tcp_pos(env)
        if np.linalg.norm(err) < TOL:
            return True, float(np.linalg.norm(err))
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[:3] = np.clip(K * err, -1.0, 1.0)
        action[3] = gripper
        env.step(action)
        log.append(action.copy())
    return False, float(np.linalg.norm(np.asarray(target) - _tcp_pos(env)))


def solve(env, seed=None, debug=False, vis=False, pick: str = "cubeA",
          place: str | None = None, truncate_at_lift: bool = True):
    """Reach, grasp and lift the named cube. Waypoints in the spirit of a real grasp pipeline.

    Phases: approach to a hover pose above the cube, descend, close the gripper, lift. The
    ``place`` and ``truncate_at_lift`` arguments exist only for signature compatibility with
    ``stackcube_solutions.solve``; this controller always ends at the lift.

    Returns a dict of diagnostics rather than the planner's status code.
    """
    assert pick in ("cubeA", "cubeB"), pick
    env.reset(seed=seed)
    log: list[np.ndarray] = []

    cube = _cube_pos(env, pick)
    pre = np.array([cube[0], cube[1], cube[2] + HOVER])

    reached, err = _step_toward(env, pre, BUDGET["approach"], GRIPPER_OPEN, log)
    approach_steps = len(log)
    assert reached, f"never reached the pre-grasp pose for {pick}: residual {err:.4f} m"

    grasp = np.array([cube[0], cube[1], cube[2] + GRASP_DZ])
    reached, err = _step_toward(env, grasp, BUDGET["descend"], GRIPPER_OPEN, log)
    assert reached, f"never descended onto {pick}: residual {err:.4f} m"

    # Close: hold position, drive the gripper shut.
    for _ in range(BUDGET["close"]):
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[3] = GRIPPER_CLOSED
        env.step(action)
        log.append(action.copy())

    lift_target = _tcp_pos(env) + np.array([0.0, 0.0, LIFT])
    reached, err = _step_toward(env, lift_target, BUDGET["lift"], GRIPPER_CLOSED, log)
    assert reached, f"never lifted {pick}: residual {err:.4f} m"

    z = _cube_pos(env, pick)[2]
    return dict(pick=pick, cube_z_start=float(cube[2]), cube_z_end=float(z),
                lift=float(z - cube[2]), n_actions=len(log),
                approach_steps=approach_steps, first_action=np.asarray(log[0]),
                second_action=np.asarray(log[1]) if len(log) > 1 else None)
