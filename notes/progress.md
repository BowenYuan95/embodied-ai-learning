# Embodied AI Learning Progress

## Environment

Date:
2026-09-19

Hardware:
- NVIDIA RTX 3080 Ti Laptop GPU
- 16GB VRAM

Software:
- Ubuntu
- Python 3.12
- PyTorch 2.11.0 + CUDA 12.8
- ManiSkill

## Completed

- [x] Linux environment setup
- [x] PyTorch CUDA verification
- [x] ManiSkill installation
- [x] PickCube-v1 rollout test

## Current Lesson

Lesson 2:
Robot Dataset — LeRobot × ManiSkill × Sim/VR/Real

## 2026-09-19

### Lesson 2: Robot Dataset Pipeline

Completed:

- Generated first ManiSkill PickCube trajectory.
- Saved robot rollout data into HDF5 format.
- Inspected dataset structure:
  - observations
  - actions
  - rewards

- Designed standardized episode schema:
  - observations
  - actions
  - rewards
  - timestamps
  - metadata

- Reverse engineered ManiSkill observation representation.

Observation tensor:

(42 dimensions)

Decomposed into:

- robot qpos: 9
- robot qvel: 9
- tcp pose: 7
- object pose: 7
- goal position: 3
- tcp-object relative position: 3
- object-goal relative position: 3
- grasp state: 1

Total: 42 dimensions.

Key understanding:

Robot dataset is not only numerical arrays;
it requires semantic schema describing robot state,
task state, action representation and metadata.

Next:

- Convert ManiSkill HDF5 dataset into LeRobot format.
- Understand dataset standardization for ACT / Diffusion Policy / VLA.