# 具身智能系统学习路线图 v3 — ManiSkill 贯穿版

更新日期：2026-09-18

## 总体目标

保持原有知识主线：

**Modern Robotics → Robot Learning → ACT / Diffusion Policy → VLA → World Model → 数据闭环**

新增贯穿式实验主线：

**ManiSkill Environment → Demonstration/Data → LeRobot Schema → Policy Training → Sim/VR/Real Mixture → Sim2Real/Real2Sim → Bad-case Data Loop**

核心原则：ManiSkill 不作为单独一课，而作为所有核心知识的默认实验平台；LeRobot 作为跨仿真、VR/遥操作和真实机器人数据的统一数据接口之一。

---

## 三条并行学习主线

### A. 理论主线
机器人状态与动作 → 坐标系/运动学 → 轨迹与控制 → 模仿学习 → Action Chunk → Diffusion/Flow → VLA → World Model → Sim2Real/Data Loop

### B. ManiSkill 实验主线
从 PickCube/PushCube 等简单任务开始，逐步进入 observation/control mode、trajectory、BC、Diffusion Policy、VLA、domain randomization、digital twin、sim2real/real2sim。

### C. 数据主线
ManiSkill 仿真数据 + VR/遥操作数据 + 真实机器人 LeRobot 数据 → schema/action/frame/frequency 对齐 → source-aware mixture → pretrain/mixed training/real fine-tuning → bad-case recollection。

---

# Phase 0 — 具身智能全景

## Lesson 0 — VLM、VLA、Policy、World Model、Agent、Robot Learning 的关系【已完成】

### 理论
- Embodied AI / Robotics / Robot Learning 的边界
- Policy、Planner、Controller 的区别
- VLM → VLA
- World Model 的位置
- Perception–Planning–Action 闭环

### ManiSkill Lab
- 浏览 ManiSkill 的 Environment / Agent / Sensor / Controller / Task
- 运行一个最简单环境，建立 `env.reset()` → `env.step(action)` 的直觉
- 将 Gymnasium step loop 对应到 embodied-agent loop

### 本课数据问题
- simulator 中的 observation/state/action 分别是什么？
- 哪些量真实机器人可以直接测到，哪些是 simulation privileged state？

---

# Phase 1 — Robot Representation & Control

## Lesson 1 — Robot State、Coordinate Frame 与 Action Space【已完成】

### 理论
- State vs Observation
- Joint Space vs Cartesian Space
- Base / World / Camera / End-effector Frame
- Absolute / Relative / Delta Action
- Gripper action
- Action Specification：
  `(space, representation, frame, dimension, semantics, unit, frequency)`
- Action Chunk 基础
- VLA action vs motor command

### ManiSkill Lab
- 对比 `pd_joint_pos`、`pd_joint_delta_pos`、`pd_ee_delta_pos`、`pd_ee_delta_pose`
- 查看同一个 PickCube 任务在不同 controller 下 action 的维度和语义
- 打印 proprioception 与真实 scene state，区分可观测状态与 privileged state

### 本课数据问题
- 为什么两个 shape 相同的 action 仍可能不能混训？
- action frame / control mode 不同会造成什么问题？

---

## Lesson 2 — Robot Dataset：LeRobot × ManiSkill × Sim/VR/Real【进行中】

### 理论
- Hugging Face Dataset / LeRobot Dataset
- episode / frame / trajectory / timestamp
- observation.state / observation.images / action
- sampling frequency 与 temporal alignment
- Dataset Schema ≠ Robot Semantics

### ManiSkill Lab
- 下载/生成 PickCube demonstration
- 阅读 ManiSkill HDF5 trajectory
- replay trajectory
- 比较 `T` 个 actions 与状态序列
- 将 ManiSkill trajectory 转换为 LeRobot v3 数据格式
- 写 `inspect_robot_dataset.py`

### Sim / VR / Real 数据统一
建立统一检查模板：
1. Embodiment
2. Observation space
3. Action specification
4. Coordinate frame
5. Frequency
6. Task semantics
7. Source/domain

### 本课核心问题
- ManiSkill 数据与真实 LeRobot 数据何时可以 concat？
- VR controller pose 如何映射成 robot action？
- sim-only privileged labels 应该如何使用而不造成真实部署依赖？

---

## Lesson 3 — SO(3)、SE(3) 与坐标变换

### 理论
- Rotation matrix / quaternion / axis-angle
- Homogeneous transformation
- SO(3) / SE(3)
- frame composition / inverse transform

### ManiSkill Lab
- 读取 robot base、TCP、camera、object pose
- 完成 base ↔ world ↔ camera ↔ EE 坐标转换
- 可视化一个 object pose 在不同 frame 中的表示

### Sim / Real 数据问题
- 为什么 sim/VR/real 合并前必须统一 coordinate convention？
- camera extrinsic calibration 如何影响 policy 数据？

---

## Lesson 4 — FK、IK、Jacobian 与 Retargeting

### 理论
- Forward Kinematics
- Inverse Kinematics
- Jacobian
- singularity 基础
- Human/VR pose → robot action retargeting

### ManiSkill Lab
- 从 joint state 计算 EE pose
- 给定 EE target 求 IK
- 用 VR-controller-style 6DoF target 驱动 ManiSkill robot

### Sim / VR / Real 数据问题
- Human hand trajectory 为什么不能直接作为 robot joint trajectory？
- retargeting 之后应该记录 human action 还是 robot executed action？

---

## Lesson 5 — Trajectory、Controller 与时间离散化

### 理论
- Position / velocity / torque control
- waypoint / trajectory
- control frequency
- interpolation / resampling
- latency 与 observation-action alignment

### ManiSkill Lab
- 比较不同 controller 与 control frequency
- 将 60 Hz VR trajectory 重采样到 robot control rate
- 分析 delta action 在不同频率下的物理意义

### Sim / Real 数据问题
- 60 Hz VR、20 Hz simulation、30 Hz real robot 如何统一？
- timestamp 对齐错误如何污染 imitation-learning dataset？

---

# Phase 2 — Robot Learning

## Lesson 6 — Behavior Cloning：第一个真正的 Policy

### 理论
- Supervised imitation learning
- covariate shift
- open-loop loss vs closed-loop success
- normalization

### ManiSkill Lab
- 使用 ManiSkill BC baseline 在 PickCube/PushCube 上训练
- state-based BC → visual BC
- train / rollout / success-rate evaluation

### Sim + Real 实验
- Sim-only baseline
- Sim pretrain → Real fine-tune
- Sim/Real mixed batch
- 比较三种方案，不预设哪一种一定最好

---

## Lesson 7 — Action Chunking 与 ACT

### 理论
- single-step prediction 的问题
- action chunk
- transformer policy
- temporal ensemble
- long-horizon imitation

### ManiSkill Lab
- 将 trajectory 转为 action chunks
- 在 ManiSkill task 上构建 ACT-style training samples
- rollout action chunk，并观察 horizon 对闭环控制的影响

### Sim / VR / Real 数据问题
- 不同数据源 action frequency 不一致时如何构造统一 chunk？
- VR demonstration 的平滑轨迹对 ACT 有什么优势与偏差？

---

## Lesson 8 — Diffusion Policy

### 理论
- conditional action diffusion
- observation horizon / action horizon / prediction horizon
- receding-horizon control
- multimodal action distribution

### ManiSkill Lab
- 使用/复现 ManiSkill Diffusion Policy baseline
- 与 BC/ACT 在同一任务、同一数据上比较
- 可视化 action trajectory distribution

### Sim / Real 数据问题
- simulation 大规模 trajectory 是否更适合学习 action distribution？
- real data 应该用于 grounding、fine-tuning 还是 mixture weighting？

---

## Lesson 9 — Flow Matching 与 π 系列前置知识

### 理论
- diffusion vs flow matching
- vector field / ODE sampling
- continuous action generation
- 为 π₀ / π₀.₅ 建立数学直觉

### ManiSkill Lab
- 用低维 ManiSkill action trajectory 做 toy flow-matching experiment
- 从噪声动作轨迹逐步生成可执行 trajectory

### 数据问题
- 不同 embodiment/action dimension 如何进入统一生成式 policy？

---

# Phase 3 — VLA

## Lesson 10 — 从 VLM 到 VLA

### 理论
- language-conditioned policy
- vision-language representation + action head
- RT-2 / Octo / OpenVLA 的核心思想
- cross-embodiment learning

### ManiSkill Lab
- 为多个 ManiSkill task 添加语言描述
- 建立 `(image, state, language) → action` 数据样本
- 从 single-task policy 过渡到 multi-task policy

### 数据问题
- sim language instruction 与真实世界 instruction 如何保持语义一致？
- source token / embodiment token / task token 的作用

---

## Lesson 11 — π₀ / π₀.₅：Inference 与源码追踪

### 理论
- π 系列整体输入输出
- image/state/language encoding
- action expert / flow matching
- action chunk generation

### 实操
- 跑通 π₀.₅ inference
- 跟踪 dataset → tokenizer/processor → model → action 的源码路径
- 明确 state/action normalization 与 action chunk

### ManiSkill 连接
- 设计 ManiSkill observation/action 到 π₀.₅ 所需接口的 adapter
- 不急于追求完整训练，先做到接口层可解释

---

## Lesson 12 — VLA Fine-tuning：LIBERO / ManiSkill / LeRobot

### 理论
- fine-tuning dataset structure
- task mixture
- action normalization
- frozen vs trainable modules

### 实操
- LIBERO fine-tune 作为标准 benchmark 复现
- ManiSkill 生成对应的 language-conditioned trajectory
- 比较 benchmark sim dataset 与 ManiSkill dataset schema

### Sim / Real 数据问题
- Sim pretrain → mixed → real fine-tune
- source-aware sampling
- domain-balanced batch
- 不把“sim+real”简化为直接 concat

---

## Lesson 13 — Cross-Embodiment & Unified Action Representation

### 理论
- embodiment gap
- action tokenizer / normalized action
- EEF-space vs joint-space
- masks / metadata / embodiment conditioning

### ManiSkill Lab
- 使用不同 robot embodiment 执行相似 manipulation task
- 分析哪些 observation/action 可以共享，哪些必须 robot-specific

### 数据问题
- 如何让 Panda / SO100 / 双臂数据进入同一个训练 pipeline？

---

# Phase 4 — World Model & Planning

## Lesson 14 — Dynamics / World Model

### 理论
- transition model `p(s_{t+1}|s_t,a_t)`
- latent world model
- rollout prediction
- model-based planning / MPC

### ManiSkill Lab
- 利用 simulation ground-truth state 学习 dynamics model
- 比较 one-step 与 multi-step prediction
- 用真实可观测量替换 privileged state

### Sim / Real 数据问题
- simulation 的精确 state label 适合训练什么？
- 如何防止 world model 依赖真实世界不可获得的变量？

---

## Lesson 15 — Planning + Policy + World Model

### 理论
- policy learning vs planning
- MPC
- value / success model
- hierarchical policy

### ManiSkill Lab
- 在同一环境比较 reactive policy 与 rollout-based planning
- 将 failure trajectory 用于 success/failure prediction

### 数据问题
- successful demonstrations、failures、recovery trajectories 分别提供什么监督？

---

# Phase 5 — Sim2Real / Real2Sim / VR Data Engine

## Lesson 16 — ManiSkill Sim2Real

### 理论
- reality gap
- visual/dynamics randomization
- system identification
- calibration

### ManiSkill Lab
- 使用 domain randomization
- 学习 Sim2RealEnv / real-agent interface
- 对齐 camera / observation / action / control rate

### 核心问题
- 哪些 gap 可以随机化，哪些必须标定？
- sim data 应该负责 coverage，real data 应该负责哪些 grounding？

---

## Lesson 17 — Real2Sim 与 Digital Twin

### 理论
- evaluation digital twin
- real trajectory replay / reconstruction
- sim-based scalable evaluation

### ManiSkill Lab
- 从 real/recorded trajectory 构造或理解 digital-twin task
- 将真实 bad case 在仿真中复现和扩增

### 核心问题
- 仿真不只用于训练，还能否成为真实 policy 的诊断工具？

---

## Lesson 18 — VR Teleoperation as a Data Engine

### 理论
- VR 6DoF tracking
- human-to-robot retargeting
- embodiment mismatch
- demonstration quality
- task/action segmentation

### ManiSkill Lab
- Quest/VR-style controller → ManiSkill robot control
- 收集 human demonstrations
- 记录 human command + robot executed action + observation
- 转为统一 LeRobot-style dataset

### 与用户既有 XR 能力衔接
- temporal/spatial segmentation
- task phase detection
- failure/recovery segment
- human-in-the-loop verification

---

# Phase 6 — Data Closed Loop

## Lesson 19 — Multi-source Dataset Mixture

### 数据源
- Internet / Ego video
- VR human demonstration
- ManiSkill synthetic trajectory
- real teleoperation
- real autonomous rollout

### 核心技术
- schema normalization
- action alignment
- coordinate alignment
- temporal synchronization
- quality filtering
- task/skill segmentation
- source metadata
- mixture weights

### 实验
比较：
1. Sim only
2. Real only
3. Sim pretrain → Real fine-tune
4. Sim + Real mixed
5. Sim + VR + Real mixed

---

## Lesson 20 — Evaluation：Offline ≠ Closed-loop

### 理论
- prediction/action loss
- success rate
- robustness
- perturbation test
- generalization
- failure taxonomy

### ManiSkill Lab
- 大规模 parallel evaluation
- object pose / camera / lighting / dynamics perturbation
- closed-loop failure collection

### Real-world 对照
- 同一 policy 的 sim vs real success/failure pattern
- 分析 evaluation gap

---

## Lesson 21 — Bad Case → Data → Retraining

### 闭环
`Policy rollout → failure detection → segment → label → sim reproduction/augmentation → dataset update → retraining → reevaluation`

### ManiSkill 的角色
- 重放真实失败
- 生成相邻 perturbation
- 长尾 failure/recovery 扩增
- scalable regression testing

---

# Phase 7 — Capstone

## Lesson 22 — 端到端项目

目标：完成一个真正贯穿全部课程的项目。

### Pipeline

`VR Demonstration`
→ `Retarget to ManiSkill Robot`
→ `ManiSkill + Real Demonstrations`
→ `LeRobot Unified Dataset`
→ `BC baseline`
→ `ACT / Diffusion Policy`
→ `VLA / π₀.₅ adapter or fine-tuning`
→ `ManiSkill closed-loop evaluation`
→ `Real robot deployment`
→ `Bad-case collection`
→ `Real2Sim reproduction`
→ `Retraining`

### 最终能力
学习结束后，应能够独立回答并实现：
- 一个 robot dataset 的 state/action 到底代表什么；
- 如何把 VR、simulation、real robot 数据统一到可训练的数据接口；
- 哪些数据可以共享、哪些只能作为 privileged supervision；
- 如何选择 BC / ACT / Diffusion / VLA；
- 如何在 ManiSkill 中训练、评估和诊断 policy；
- 如何把 policy 从 simulation 推向 real robot；
- 如何把真实 bad case 重新转化成训练数据。

---

# 贯穿课程的固定实验任务

为了避免每一课都换环境，建议长期保持三层任务：

1. **基础任务：PickCube-v1** — action/state/control/dataset/BC 的主实验。
2. **第二任务：PushCube-v1 或同类简单平面操作** — 比较 action representation 与 generalization。
3. **接触丰富任务：PegInsertionSide-v1 或等价 insertion task** — 后期用于 Diffusion/VLA/Sim2Real/robustness。

只有进入 multi-task VLA 阶段后再显著扩展任务数量。

---

# 当前进度

- Lesson 0：完成
- Lesson 1：完成
- Lesson 2：进行中（约 60–70%）

Lesson 2 已完成环境验证、PickCube rollout、HDF5 pipeline 与 42 维 observation
语义拆解。当前待完成的验收项：

1. trajectory replay 与成功/失败确认；
2. `T` / `T+1` 及 observation-action 对齐验证；
3. LeRobot v3 转换脚本修正并实际运行；
4. 转换后数据集的独立回读与完整性检查；
5. `inspect_robot_dataset.py`；
6. 8 维 action specification 与 source timestamp 诊断。

完成上述闭环后，再进入 **Lesson 3：SO(3)、SE(3) 与坐标变换**。
