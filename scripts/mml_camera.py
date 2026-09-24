"""Camera geometry helpers for ManiSkill tabletop tasks.

``unproject`` and the bounding-box mask are the same computation lesson 3.8.3 used to measure
the cube's pixel footprint on PickCube. They live here now because a second consumer appeared
(the StackCube camera sweep); the notebook copy is deliberately left alone rather than
re-executed, so **this module and the 3.8b cell are two copies of one formula** -- keep them in
step, or delete the notebook copy the next time that notebook is re-run.

Depth is ``int16`` in millimetres in ManiSkill, hence ``DEPTH_MM``.
"""

from __future__ import annotations

import numpy as np

CAMERA = "base_camera"
DEPTH_MM = 1000.0


def unproject(depth_mm: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Depth map -> world-frame surface point per pixel.

    ``K`` is the intrinsic matrix and ``E`` the extrinsic; both come from
    ``obs["sensor_param"][CAMERA]`` as ``intrinsic_cv`` / ``extrinsic_cv``.
    """
    z = np.asarray(depth_mm, dtype=np.float64) / DEPTH_MM
    h, w = z.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], axis=1)
    rays = (np.linalg.inv(K) @ pix.T).T * z.ravel()[:, None]        # camera frame
    rot, trans = E[:, :3], E[:, 3]
    return (np.linalg.inv(rot) @ (rays - trans).T).T                 # world frame


def sensor_arrays(obs: dict, camera: str = CAMERA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rgb HxWx3, depth HxW, points Nx3)`` for one environment."""
    data, param = obs["sensor_data"][camera], obs["sensor_param"][camera]
    rgb = data["rgb"][0].cpu().numpy()
    depth = data["depth"][0].cpu().numpy().squeeze()
    K = param["intrinsic_cv"][0].cpu().numpy()
    E = param["extrinsic_cv"][0].cpu().numpy()
    return rgb, depth, unproject(depth, K, E)


def project(points_world: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """World-frame points -> pixel coordinates ``(u, v)``.

    Needed because comparing *world* x-ordering against *image* x-ordering is not a valid test
    of whether a colour is localised: with the camera off-axis, two objects at the same world x
    but different depths project to different image x. The parallax made a correct measurement
    look like a failure. Check the measured centroid against the object's own projected pixel
    instead.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (np.asarray(E, dtype=np.float64) @ hom.T).T              # N x 3
    uv = (np.asarray(K, dtype=np.float64) @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def box_mask(points: np.ndarray, centre: np.ndarray, half_extent) -> np.ndarray:
    """Boolean mask of pixels whose unprojected surface point lies inside an axis-aligned box.

    This is a lower bound on the object's visible footprint: it keeps pixels whose *surface*
    falls inside the box, so a slanted or partly occluded face contributes fewer pixels than
    its silhouette would.
    """
    return np.all(np.abs(points - np.asarray(centre)) <= np.asarray(half_extent), axis=1)


def footprint(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Pixel count, mean colour and centroid of a mask, in image coordinates."""
    n = int(mask.sum())
    if n == 0:
        return dict(pixels=0, side=0.0, mean_rgb=None, centroid=None)
    flat = rgb.reshape(-1, 3).astype(np.float32)[mask]
    ys, xs = np.nonzero(mask.reshape(rgb.shape[:2]))
    return dict(pixels=n,
                side=float(np.sqrt(n)),                       # rough side length if it were square
                mean_rgb=[round(float(v), 1) for v in flat.mean(0)],
                rg=round(float(flat[:, 0].mean() - flat[:, 1].mean()), 1),
                centroid=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)])
