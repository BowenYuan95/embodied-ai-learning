# Embodied AI Learning Progress

## Environment

Date:

2026-09-19

Hardware:

-   NVIDIA RTX 3080 Ti Laptop GPU
-   16GB VRAM

Software:

-   Ubuntu
-   Python 3.12.14
-   PyTorch 2.11.0 + CUDA 12.8
-   ManiSkill
-   LeRobot 0.6.1

------------------------------------------------------------------------

# Completed

## Environment Setup

-   [x] Linux environment setup
-   [x] Conda embodied environment setup
-   [x] PyTorch CUDA verification
-   [x] ManiSkill installation
-   [x] LeRobot installation and import verification

------------------------------------------------------------------------

# Lesson 2:

# Robot Dataset Pipeline --- ManiSkill × LeRobot × Sim/Real Data

## 2026-09-19

## 1. ManiSkill Trajectory Generation

Completed:

-   Generated first ManiSkill PickCube trajectory.
-   Saved robot rollout data into HDF5 format.

Dataset structure:

    episode

    ├── observations
    ├── actions
    ├── rewards
    ├── timestamps
    └── metadata

Understanding:

A robot trajectory is represented as:

\[ (o_t, a_t, r_t, timestamp) \]

------------------------------------------------------------------------

# 2. Observation Schema Analysis

Initial observation:

    Observation:
    (42 dimensions)

Reverse engineered ManiSkill observation representation:

    Observation

    ├── agent
    │
    │   ├── qpos
    │   └── qvel
    │
    └── extra
        |
        ├── tcp_pose
        ├── goal_pos
        ├── obj_pose
        ├── tcp_to_obj_pos
        ├── obj_to_goal_pos
        └── is_grasped

Dimension decomposition:

  Component                         Dimension
  ------------------------------- -----------
  robot qpos                                9
  robot qvel                                9
  tcp pose                                  7
  object pose                               7
  goal position                             3
  tcp-object relative position              3
  object-goal relative position             3
  grasp state                               1
  Total                                    42

Key understanding:

Observation is not just a numerical vector.

It is a semantic representation of:

-   robot state
-   object state
-   task state

------------------------------------------------------------------------

# 3. Action Schema Analysis

Action space:

    (8,)

Control mode:

    pd_joint_delta_pos

Action mapping:

  Dimension   Meaning
  ----------- --------------------------------
  0-6         Panda arm joint delta position
  7           Gripper command

Important distinction:

Robot state:

    qpos = 9

contains:

    7 arm joints
    +
    2 finger joints

Action:

    8 dimensions

contains:

    7 arm joint commands
    +
    1 gripper command

Reason:

The two Panda finger joints are mechanically coupled and controlled by
one command.

------------------------------------------------------------------------

# 4. Dataset Synchronization Understanding

Learned that different robot sensors may have different frequencies:

Example:

    Camera:
    30 Hz

    Robot joint state:
    100 Hz

    Force sensor:
    500 Hz

Data alignment requires:

    Timestamp synchronization

            ↓

    Downsampling / interpolation / aggregation

            ↓

    Unified robot dataset

Key understanding:

Robot dataset engineering requires:

-   semantic schema alignment
-   temporal alignment

------------------------------------------------------------------------

# Current Lesson

## Lesson 2.7:

## ManiSkill HDF5 → LeRobot Dataset Conversion

Status:

In progress

Completed preparation:

-   [x] Installed LeRobot 0.6.1
-   [x] Verified LeRobotDataset API
-   [x] Determined dataset feature requirements

Current goal:

Convert:

    ManiSkill HDF5

            ↓

    LeRobot Dataset

Target structure:

    LeRobot Dataset

    ├── observation.state
    ├── action
    ├── timestamp
    ├── episode_index
    └── task metadata

------------------------------------------------------------------------

# Next Steps

1.  Define LeRobot features

2.  Create conversion script:

```{=html}
<!-- -->
```
    scripts/convert_maniskill_to_lerobot.py

3.  Convert PickCube trajectory

4.  Validate generated dataset

5.  Load dataset using LeRobot DataLoader

Future:

-   Train simple behavior cloning policy
-   Understand ACT / Diffusion Policy
-   Extend dataset pipeline to real robot data and VLA models