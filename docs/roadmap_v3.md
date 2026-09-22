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

## Lesson 2 — 从机器人轨迹到可训练策略【收尾阶段】

本课主线是一条连续的工程链路：**轨迹采集 → 数据集 → DataLoader → Policy →
Loss → 训练 → Rollout**。本课结束时应当能独立解释并实现这条链路上的每一步。
本课的目标是**把训练管线跑通**，因此不发散到工程环境，也不以策略性能为验收标准。

### 2.1 ManiSkill 环境与轨迹【已完成】
- 创建 `PickCube-v1`
- 理解 observation、action、reward、done
- 运行一个完整 episode
- 理解控制循环：`o_t → a_t → o_{t+1}`

### 2.2 机器人状态与动作空间【已完成】
- 解析 42 维 `observation.state`
- 理解关节位置、关节速度、TCP 位姿、物体位姿、目标位置
- 解析 8 维 action：**不是统一语义**，而是两种命令拼接
  - `action[0:7]`：`PDJointPosController`，基于**当前关节位置**的关节位置增量
    （`use_delta=True`、`use_target=False`、`normalize_action=True`），
    归一化 `[-1,1]` 映射到 `[-0.1, 0.1] rad`，近似 `Δq_i = 0.1·a_i`，
    目标为 `q_target = q_current + Δq`
  - `action[7]`：`PDJointPosMimicController`，夹爪**绝对位置目标**，
    归一化 `[-1,1]` 映射到 `[-0.01, 0.04]`，两指通过 mimic 联动，
    因此 1 个动作值驱动 2 个活动关节
- 理解 `pd_joint_delta_pos`：机械臂用 delta、夹爪用 absolute，
  这是本任务最容易被误读的一处
- DOF 记账：`7 + 1 = 8` 个动作维度，而物理活动关节数为 `7 + 2 = 9`

### 2.3 时间对齐【已完成】
- 理解 frame、timestamp、episode
- 确认训练配对是 `(o_t, a_t)`，而不是 `(o_t, a_{t+1})`
- 理解执行 `a_t` 之后得到 `o_{t+1}`

### 2.4 原始轨迹存储【已完成】
- 查看 ManiSkill HDF5
- 理解 episode 分组
- 检查 observation / action 长度
- 区分原始仿真数据与训练数据格式

### 2.5 转换为 LeRobot Dataset【已完成】
- 将 ManiSkill 轨迹转换为 LeRobot 格式，生成 `data/`、`meta/`
- 生成任务描述与 episode 元数据
- 理解数据格式转换的目的：Dataset Schema ≠ Robot Semantics

### 2.6 数据验证与质量检查【已完成】
- 检查 shape、dtype、NaN 与数值范围
- 检查 episode、frame、timestamp 连续性
- 绘制 action、reward 等曲线
- 区分「格式正确」与「数据有效」

当前结论：数据结构有效，但只有 1 个 episode、50 帧，而且动作近似随机，
不是专家演示。

### 2.7 PyTorch Dataset 与 DataLoader【已完成】
- `dataset[index]`：单个样本
- `DataLoader`：组成 batch
- 理解 tensor、shape 与 batch 维度
- 理解 `shuffle=True/False`
- 区分随机采样与时间顺序

### 2.8 最小行为克隆模型【进行中】
- **2.8.1 数据读取【已完成】** — `observation.state [42]`，`action [8]`
- **2.8.2 Batch【已完成】** — `states [8,42]`，`actions [8,8]`
- **2.8.3 MLP Policy【已完成】** — `R^42 → R^128 → R^128 → R^8`
- **2.8.4 Loss【已完成】** — `L = MSE(â_t, a_t)`
- **2.8.5 反向传播与参数更新【已完成】** — `zero_grad()` → forward → loss →
  `backward()` → `optimizer.step()`；梯度、学习率、权重更新
- **2.8.6 完整训练循环【已完成】** — batch 循环、epoch 循环、Adam、Loss 曲线、
  100 epochs
- **2.8.7 训练集与验证集【下一步】**
  - 为什么只看训练 Loss 不够
  - train / validation split
  - 泛化与记忆
  - underfitting 与 overfitting
  - 为什么随机动作数据无法训练出有效策略
- **2.8.8 策略部署与闭环执行【尚未开始】**
  - `环境状态 → policy(state) → 预测动作 → env.step(action) → 新状态`
  - `model.train()` 与 `model.eval()`
  - `torch.no_grad()`
  - 单步预测与闭环 rollout
  - 仿真安全与动作裁剪
  - 判断策略是否真的完成任务

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

### 本课验收条件
- 一条从仿真轨迹到训练产物的完整链路可复现；
- 能解释 `(o_t, a_t)` 配对与 `T` / `T+1` 约定；
- 能实现并解释 `Dataset` → `DataLoader` → MLP → Loss → 训练循环；
- 能说明为什么低训练 Loss 不等于有效策略；
- 策略至少完成一次闭环执行并给出任务结果（2.8.8）。

---

## Lesson 3 — 模仿学习与行为克隆【当前起点】

Lesson 2 的重点是「把训练管线跑通」；Lesson 3 的重点是理解：
**为什么模型即使训练 Loss 很低，真实执行时仍可能失败？**

入口实验直接沿用 2.8.7：用训练集与验证集实验引出拟合、泛化、分布偏移与专家数据。

### 3.1 模仿学习问题定义
- 专家演示是什么
- 状态、观察、动作与策略
- 专家策略 `π_E` 与学习策略 `π_θ`
- 目标：`π_θ(a_t | o_t) ≈ π_E(a_t | o_t)`

### 3.2 Behavior Cloning
- 行为克隆本质上是监督学习：`D = {(o_t, a_t)}_{t=1..N}`，
  `θ* = argmin_θ Σ_t L(π_θ(o_t), a_t)`
- 连续动作为什么使用 MSE，离散动作为什么使用交叉熵
- 单峰动作预测的问题
- 专家数据质量与覆盖范围

### 3.3 泛化与过拟合
- 训练集、验证集、测试集
- 如何按 episode 划分，为什么不能随意按 frame 划分
- 数据泄漏
- 模型容量与数据规模
- 训练 Loss 与验证 Loss 曲线

**特别注意**：以后有多个 episode 时，应优先按 episode 划分，而不是把同一条
轨迹的相邻帧随机分到训练集和验证集。相邻帧高度相关，frame 级划分会造成
实质性泄漏。

### 3.4 Distribution Shift
- 训练时模型看到专家到达的状态：`o_t ~ d_{π_E}`
- 执行时模型看到自己产生的状态：`o_t ~ d_{π_θ}`
- 一个小动作误差可能把机器人带到训练集中没有出现过的状态，后续误差不断累积
- 这是行为克隆的核心问题之一

### 3.5 DAgger
`专家数据训练 → 学习策略执行 → 收集失败附近的新状态 → 专家提供正确动作 →
加入数据集重新训练`

### 3.6 单帧策略与历史策略
- 单帧 MLP：`a_t = π(o_t)`
- 历史策略：`a_t = π(o_{t-k:t})`
- 比较 MLP、RNN/LSTM、Transformer
- 部分可观测问题：为什么机器人有时需要历史信息

### 3.7 Action Chunking
- 不只预测一个动作，而是预测未来一段动作：`â_{t:t+H} = π(o_t)`
- 连接到 ACT、Diffusion Policy、VLA 中的动作序列输出、π₀/π₀.₅ 的 action chunk

### 3.8 多模态策略过渡
- 把当前的 `state → action` 扩展为 `(image, language, state) → action`
- 为后面的 Transformer、VLA 与 π₀.₅ 做连接，但不立刻进入大型工程部署

### 3.9 专家数据采集
- 结合 SO-101：遥操作与示教
- 成功 / 失败 episode
- 摄像头与机器人状态同步
- 数据频率与延迟
- 任务变化与数据覆盖
- 从仿真数据过渡到真机数据

---

## Lesson 4 — SO(3)、SE(3)、FK/IK 与 Retargeting

> 原计划中「SO(3)/SE(3) 与坐标变换」独立成课，现与 FK/IK/Retargeting 合并为
> 一课：坐标表示与变换、正逆运动学、Jacobian 与 VR retargeting 是一条连贯的
> 几何主线。若希望重新拆成两课，在此处调整。

### 4.1 SO(3) 与 SE(3)
- Rotation matrix / quaternion / axis-angle
- Homogeneous transformation
- SO(3) / SE(3)
- frame composition / inverse transform

### 4.2 FK、IK 与 Jacobian
- Forward Kinematics
- Inverse Kinematics
- Jacobian
- singularity 基础

### 4.3 Human / VR Pose → Robot Action Retargeting
- 人类手部轨迹为什么不能直接作为 robot joint trajectory
- retargeting 之后应该记录 human action 还是 robot executed action

### ManiSkill Lab
- 读取 robot base、TCP、camera、object pose
- 完成 base ↔ world ↔ camera ↔ EE 坐标转换
- 可视化一个 object pose 在不同 frame 中的表示
- 从 joint state 计算 EE pose；给定 EE target 求 IK
- 用 VR-controller-style 6DoF target 驱动 ManiSkill robot

### Sim / Real 数据问题
- 为什么 sim/VR/real 合并前必须统一 coordinate convention？
- camera extrinsic calibration 如何影响 policy 数据？

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

详细状态与证据分级见 `notes/progress.md`（单一进度记录）。本段只保留头条状态。

- Lesson 0：完成
- Lesson 1：完成
- Lesson 2：**收尾阶段（约 90%）**。2.1–2.8.6 已完成，下一步为
  **2.8.7「训练集与验证集」**：数据采集、转换、质量验证、Dataset/DataLoader
  与最小 BC 训练循环均已跑通，下一步用训练集/验证集实验引出泛化与分布偏移。
- Lesson 3：**当前起点** — 模仿学习与行为克隆（3.1–3.9）。
- Lesson 4：未开始 — SO(3)/SE(3)、FK/IK 与 Retargeting（原独立成课，现合并）。

## Lesson 2 已完成

- ManiSkill `PickCube-v1` rollout 与完整 `o_t → a_t → o_{t+1}` 控制循环；
- 42 维 observation 拆解、8 维 action 分析、`pd_joint_delta_pos`；
- 8 维 action 的**双语义**已实测确认：`action[0:7]` 为基于当前 qpos 的关节增量
  （`[-0.1,0.1] rad`、`use_delta=True`、`use_target=False`），`action[7]` 为夹爪
  绝对位置目标（`[-0.01,0.04]`、mimic 联动）；`conversion_manifest.json` 与
  `scripts/pipeline/manifest.py` 已按此修正动作语义描述；
- `(o_t, a_t)` 时间对齐确认，以及 `T` / `T+1` 约定；
- ManiSkill HDF5 原始轨迹存储与 episode 分组；
- ManiSkill HDF5 → LeRobot Dataset 转换，产物位于 `datasets/lerobot/pickcube/`；
- 数据验证与质量检查：shape / dtype / 范围 / 连续性，action 与 reward 曲线；
- PyTorch `Dataset` / `DataLoader`：单样本 `[42]` / `[8]` → batch `[8,42]` / `[8,8]`；
- 最小 BC 模型：MLP `42 → 128 → 128 → 8`、`MSE`、反向传播与参数更新；
- 完整训练循环：100 epochs、Adam、Loss 曲线，训练 Loss
  `0.346221 → 0.105965`（`[verified]`，已独立复现；未训练基线 `0.371752`）。
- 结论：训练管线跑通；但数据结构虽有效，只有 1 个 episode、50 帧且动作近似
  随机，不能作为专家示范，也无法据此声称策略有效。

## Lesson 2 待完成

1. **2.8.7 训练集与验证集**：为什么只看训练 Loss 不够、train/validation split、
   泛化与记忆、underfitting 与 overfitting、为什么随机动作数据无法训练出有效
   策略；单 episode 下 frame 级划分会泄漏，需明确记录该限制。
   **进展**：2.9 已产出 5 条成功专家 episode
   （`datasets/pickcube/expert_episodes.h5`，50–86 帧，5/5 成功），
   因此「只有 1 个 episode」不再是唯一可用数据。但这些 episode 是
   `pd_joint_pos` 绝对关节目标，与本项目 `pd_joint_delta_pos` 的增量语义
   **不兼容**，混用前必须先转换并验证。若要据此做按 episode 划分的
   train/val，需先完成该转换。
2. **2.8.8 策略部署与闭环执行**：`env.step(policy(state))`、`train()` / `eval()`、
   `torch.no_grad()`、单步预测与闭环 rollout、动作裁剪与仿真安全、
   判断策略是否真的完成任务。

以下为记录备查、当前不阻塞的遗留项：

3. `inspect_robot_dataset.py`（本课验收项，目前缺失）；
4. manifest 溯源：记录中的源路径为已不存在的 `/home/bowenyuan95/...`（sha256 仍
   与当前源文件一致，仅记录字符串有误），下次重建数据集时用
   `scripts/run_pipeline.py` 重新生成；夹爪开合方向的物理含义（`0.04` 是否对应
   完全张开）仍待执行验证或查阅 MJCF 确认；
5. **20 Hz / 50 Hz 时间契约矛盾**：控制频率实测 20 Hz（`control_timestep=0.05`），
   而合成时间戳为 `0.02 s`，`converter.py:estimate_fps` 从时间戳反推出
   `info.json` 的 50 FPS，使时间轴压缩 2.5 倍。修复后在时间窗口工作之前生效；
6. 42 维 observation 的逐字段命名与可部署性（privileged state）标注。

完成 2.8.7 与 2.8.8 后，Lesson 2 结束，进入
**Lesson 3：模仿学习与行为克隆**，入口实验即 2.8.7 的训练集/验证集结果。
