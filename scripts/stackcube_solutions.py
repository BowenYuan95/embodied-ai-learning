"""Parameterised StackCube motion-planning solution (pick either cube).

The stock solution at
``mani_skill.examples.motionplanning.panda.solutions.stack_cube`` always picks ``cubeA`` and
stacks it on ``cubeB``. Three lines couple it to those names:

    stock line 30   obb         = get_actor_obb(env.cubeA)
    stock line 79   goal_pose   = env.cubeB.pose * ...
    stock line 80   offset      = (goal_pose.p - env.cubeA.pose.p)

This module is that file with those three lines parameterised on ``pick`` / ``place``, so the
same code can produce "pick the red cube" (``pick="cubeA"``) and "pick the green cube"
(``pick="cubeB"``). ``StackCubeEnv`` builds ``cubeA`` red and ``cubeB`` green; both were
confirmed at runtime (``base_color == [1,0,0]`` and ``[0,1,0]``).

**The environment's own success check only tests "red on green"**, so a mirrored
"green onto red" episode reports ``success=False`` even when the robot did it correctly.
Evaluate the mirror geometrically instead -- :func:`stacked` below.

Two deliberate departures from the stock file:

``truncate_at_lift``
    Stop after the lift, before the stack. The grounding experiment's task is "pick the named
    cube"; the place phase adds motion that is irrelevant to the referential question, and
    keeping it would also reintroduce a destination object the policy could key on.
no valid grasp pose
    The stock file prints a warning and then continues using a grasp pose it already knows
    failed. Here that raises, because a silently bad demonstration would poison the dataset.

The caller must build the environment as::

    gym.make("StackCube-v1", obs_mode="state", render_mode=None, num_envs=1,
             control_mode="pd_joint_pos")

``pd_joint_pos`` is asserted below and is also the action contract used by PickCube/PushCube in
this repository, so the action side stays frozen across lessons. Note that StackCube's *default*
control mode is ``pd_joint_delta_pos``; using it would silently change action semantics.
"""

from __future__ import annotations

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

FINGER_LENGTH = 0.025
OTHER = {"cubeA": "cubeB", "cubeB": "cubeA"}


def solve(env: StackCubeEnv, seed=None, debug=False, vis=False,
          pick: str = "cubeA", place: str | None = None,
          truncate_at_lift: bool = False):
    """Plan a grasp-and-place. ``pick`` is the cube to grasp; ``place`` defaults to the other.

    ``truncate_at_lift=True`` returns immediately after the lift, so the recorded trajectory is
    a pure "pick the named cube" demonstration.
    """
    assert pick in OTHER, f"pick must be one of {sorted(OTHER)}, got {pick!r}"
    place = place or OTHER[pick]
    assert place != pick

    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], \
        f"expected pd_joint_pos, got {env.unwrapped.control_mode!r}"

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    target, destination = getattr(env, pick), getattr(env, place)

    # ------------------------------------------------------------------ #
    # Grasp pose -- the ONLY cube-identity-dependent step (stock line 30)
    # ------------------------------------------------------------------ #
    obb = get_actor_obb(target)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose (stock keeps this yaw sweep because the cube's own z-rotation is
    # randomised at reset).
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    found = False
    for angle in angles:
        grasp_pose2 = grasp_pose * sapien.Pose(q=euler2quat(0, 0, angle))
        if planner.move_to_pose_with_screw(grasp_pose2, dry_run=True) == -1:
            continue
        grasp_pose = grasp_pose2
        found = True
        break
    assert found, f"no valid grasp pose found for {pick} (seed {seed})"

    # Reach
    planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, -0.05]))

    # Grasp
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # Lift
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    if truncate_at_lift:
        planner.close()
        return dict(actions=None, truncate_at="lift")

    # ------------------------------------------------------------------ #
    # Stack -- stock lines 79/80, now parameterised
    # ------------------------------------------------------------------ #
    goal_pose = destination.pose * sapien.Pose(
        [0, 0, (env.cube_half_size[2] * 2).item()])
    offset = (goal_pose.p - target.pose.p).cpu().numpy()[0]
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    planner.move_to_pose_with_screw(align_pose)

    res = planner.open_gripper()
    planner.close()
    return res


def stacked(env: StackCubeEnv, top: str, bottom: str, tol_frac: float = 0.25):
    """Is ``top`` resting on ``bottom``? Returns ``(ok, diagnostics)``.

    Needed because the environment's ``evaluate()`` only ever tests "cubeA on cubeB", so the
    mirrored task's success has to be measured here. Compiled from the same quantities the
    environment uses: vertical offset near one cube height and small lateral offset.
    """
    u = env.unwrapped
    half = float(u.cube_half_size[2])
    pt = getattr(u, top).pose.p[0].cpu().numpy()
    pb = getattr(u, bottom).pose.p[0].cpu().numpy()
    dz = float(pt[2] - pb[2])
    lateral = float(np.linalg.norm(pt[:2] - pb[:2]))
    ok = abs(dz - 2 * half) < tol_frac * half and lateral < tol_frac * half
    return ok, dict(dz=dz, lateral=lateral, expected_dz=2 * half, half=half)
