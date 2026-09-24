"""Force a controlled left/right layout on StackCube, so the referential axis is exact.

Why this is necessary
---------------------
``StackCubeEnv._initialize_episode`` samples both cubes' xy from a 0.2 x 0.2 box with a minimum
separation, which does **not** give a usable left/right axis. Measured over 12 seeds:

    only 4 placed the two cubes on opposite sides of the robot's home x
    in the rest both cubes lay on the same side, so the first action's x sign was the
    same whichever cube was named -- no discrete decision exists on x
    and |dy|/|dx| between the cubes was often far above 1 (4.59, 3.78, 111.33)

With the layout uncontrolled, "which cube" is not a binary choice along any axis, so
``sign(a_0[0])`` cannot encode it and the clean referential frame is lost again.

Making the layout a controlled variable is also the better experiment design: the 2x2 of
(instruction x layout) is then produced by construction rather than collected until enough
mirrored pairs happen to appear, and balanced layouts come for free.

How it works
------------
The two cubes are placed at ``centre + [-d/2, 0, 0]`` and ``centre + [+d/2, 0, 0]``, with a
small y jitter, and ``a_left`` decides which cube takes the left slot. The patch is applied to
``_initialize_episode`` on the environment instance, so it runs **inside** ``env.reset`` and the
observation returned by reset already reflects the forced layout -- applying it afterwards would
leave the recorded first frame showing the old scene.
"""

from __future__ import annotations

import numpy as np

SEPARATION = 0.12        # metres between cube centres along x; cubes are 0.04 m
Y_JITTER = 0.03
CUBE_HALF = 0.02


class LayoutController:
    """Attach to an environment to force a controlled layout on every reset."""

    def __init__(self, env, separation: float = SEPARATION, y_jitter: float = Y_JITTER):
        self.env = env
        self.unwrapped = env.unwrapped
        self.separation = separation
        self.y_jitter = y_jitter
        self.a_left = True                     # set before each reset
        self._original = self.unwrapped._initialize_episode
        self.unwrapped._initialize_episode = self._patched

    def _patched(self, env_idx, options):
        self._original(env_idx, options)
        self._apply(env_idx)

    def _apply(self, env_idx):
        import torch
        from mani_skill.utils.structs.pose import Pose

        u = self.unwrapped
        b = len(env_idx)
        rng = np.random.default_rng(0)          # fixed jitter: the layout must be reproducible
        x_left = -self.separation / 2.0
        x_right = +self.separation / 2.0
        if not self.a_left:
            x_left, x_right = x_right, x_left
        for name, x in (("cubeA", x_left), ("cubeB", x_right)):
            actor = getattr(u, name)
            y = rng.uniform(-self.y_jitter, self.y_jitter, size=b)
            p = np.stack([np.full(b, x), y, np.full(b, CUBE_HALF)], axis=1).astype(np.float32)
            # Keep the actor's existing quaternion: the cube's random z-rotation matters for
            # the grasp geometry. ManiSkill's batched Pose is required here, not sapien.Pose,
            # whose constructor only accepts a single (3,) float32 vector.
            actor.set_pose(Pose.create_from_pq(
                p=torch.tensor(p, dtype=torch.float32, device=u.device), q=actor.pose.q))

    def set_layout(self, a_left: bool) -> None:
        """Call before ``env.reset``; the next reset places cubeA left if ``a_left``."""
        self.a_left = bool(a_left)

    def detach(self) -> None:
        self.unwrapped._initialize_episode = self._original


def attach(env, **kwargs) -> LayoutController:
    return LayoutController(env, **kwargs)
