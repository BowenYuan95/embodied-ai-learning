# 具身智能系统学习路线图 v4 — Task-Centric Embodied Agent

更新日期：2026-09-22

## 总体目标

长期目标不再是把机械臂控制本身作为终点，而是构建面向复杂长期任务的
**Task-Centric Embodied Agent**：从多模态观察与人类示范中学习任务结构，维护任务
状态与记忆，预测行动后果，并通过 VLA / learned policy 执行或辅助人类完成任务。

战略知识主线：

**Deep Learning → Multimodal Representation → VLA → Task Intelligence → Task World Model → Human-Agent Collaboration**

贯穿式实验主线：

**ManiSkill → Demonstration/Data → LeRobot Schema → Policy/VLA → Task Graph & Memory → Closed-loop Evaluation → Bad-case Data Loop**

核心原则：

1. **Task intelligence 是研究终点，action 是实现方式。**
2. **机械臂是统一实验载体，而不是职业定位。** ManiSkill、LeRobot 与后续真机用于验证 perception–reasoning–action 闭环。
3. **VLA 与 world model 是主干能力。** 不以从零训练最大模型为目标，而以理解、复现、适配、微调和任务级增强为目标。
4. **保留个人差异化。** 将既有的第一视角感知、任务分段、task state（prerequisite 关系，允许 retry、回退与重入）、注意力与自适应辅助，转化为 task memory、task state、intervention policy 与 human-agent collaboration。

---

## 四条并行学习主线

### A. Deep Learning & Multimodal Representation
PyTorch 训练基础 → Transformer → visual representation → multimodal fusion → sequence modeling → fine-tuning / parameter-efficient adaptation → evaluation。

### B. Action Intelligence
最小机器人表示 → imitation learning → Action Chunk → ACT → Diffusion / Flow → VLA。机械臂运动学、控制与 Sim2Real 只学习到足以支撑数据、模型和闭环实验的深度。

### C. Task Intelligence
Demonstration → temporal segmentation → skill / subgoal → dependency graph → task state → episodic / procedural memory → hierarchical planning → failure recovery → Task World Model。

### D. Data & Human-Agent Loop
ManiSkill 仿真数据 + 第一视角视频 + VR/遥操作数据 + 真实机器人数据 → schema / action / frame / frequency / semantics 对齐 → source-aware mixture → human state estimation → adaptive assistance → bad-case recollection。

## 建议学习权重

| 模块 | v4 权重 | 定位 |
|---|---:|---|
| Deep Learning / Transformer / multimodal learning | 20% | 当前最需要补强的基础 |
| VLA 与生成式 action policy | 25% | 求职与技术主线 |
| Task Intelligence / memory / planning | 20% | 个人差异化核心 |
| Task World Model | 15% | 从状态预测走向长期任务推理 |
| Robot fundamentals / control | 10% | 最小必要基础，不追求传统控制深挖 |
| Human-Agent interaction / data loop | 10% | 承接 XR、gaze、guidance 与用户研究积累 |

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

## Lesson 3 — 模仿学习与行为克隆【Lesson 2 验收后的下一阶段】

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

### 3.10 从轨迹到任务结构
- 为每帧增加 `task_phase`、`skill`、`subgoal` 与成功/失败标签
- 将连续 trajectory 从“动作序列”重述为 `Reach → Grasp → Lift → Transport → Place`
- 比较人工边界、状态事件边界与 learned temporal segmentation
- 建立后续 Task Graph / procedural memory 的数据接口

---

## Lesson 4 — SO(3)、SE(3)、FK/IK 与 Retargeting

> 原计划中「SO(3)/SE(3) 与坐标变换」独立成课，现与 FK/IK/Retargeting 合并为
> 一课：坐标表示与变换、正逆运动学、Jacobian 与 VR retargeting 是一条连贯的
> 几何主线。v4 将其定位为**最小必要机器人基础**：目标是正确理解和转换数据，
> 不延伸为传统机械臂控制算法专项。

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

# Phase 2 — Deep Learning & Action Intelligence

本阶段增加一条显式深度学习能力线。每个 policy 实验都不仅“调用模型”，还必须能
解释数据流、tensor shape、loss、optimization、normalization、sequence modeling、
validation 与 closed-loop metric。大型分布式预训练不是当前硬性目标；优先建立可复现的
训练、微调、诊断和源码追踪能力。

## Lesson 6 — Behavior Cloning：第一个真正的 Policy

### 理论
- Supervised imitation learning
- covariate shift
- open-loop loss vs closed-loop success
- normalization
- PyTorch module / optimizer / scheduler / checkpoint
- train / validation / test 与 episode-level split
- 过拟合诊断、gradient / activation 基本检查

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
- attention、positional encoding、encoder / decoder 与 causal masking
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
- 阅读和追踪大型模型代码：configuration、processor、backbone、action expert、checkpoint

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
- reactive VLA 的能力边界：long horizon、memory、task progress 与 recovery

### ManiSkill Lab
- 为多个 ManiSkill task 添加语言描述
- 建立 `(image, state, language) → action` 数据样本
- 从 single-task policy 过渡到 multi-task policy
- 扩展为 `(image, state, language, task_state, memory) → action chunk`

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

# Phase 4 — Task Intelligence & Embodied Memory

这是 v4 新增的核心阶段，也是与既有 STAGE、dependency-aware guidance、focus-aware
interaction 和 AVAR 研究最直接的连接点。目标不是把任务图作为人工规则外挂，而是逐步
研究它如何由 demonstration / video / robot trajectory 学得，并如何条件化 VLA 和规划。

## Lesson 14 — Task Representation from Demonstration

### 理论
- task decomposition、temporal segmentation、change-point detection
- action、skill、subgoal、task phase 与完整 task 的层级关系
- sequential plan、prerequisite 关系与 task state（需支持 cycle、retry 与回退）
- procedural knowledge 与可执行 task representation

### ManiSkill / Video Lab
- 将 PickCube trajectory 分段为 `Reach → Grasp → Lift → Transport → Place`
- 为每段生成边界、语义标签、前置条件、后置条件和置信度
- 比较 simulator event、人工标注与 representation-based segmentation
- 将第一视角视频 / VLM 输出映射到同一 task schema

### 核心问题
- demonstration 中哪些结构可直接观察，哪些必须推断？
- 分段边界是否稳定，任务图如何表达可选顺序和并行关系？

---

## Lesson 15 — Embodied Memory & Task-State Tracking

### 理论
- working / short-term memory
- episodic memory、semantic memory、procedural memory
- object permanence、re-identification 与跨视野状态保持
- belief state、uncertainty 与 memory retrieval

### 实验
- 建立显式 `task_state`：Locked / Available / Active / Completed / Failed
- 在物体离开视野后保持 identity、last-seen pose 与 uncertainty
- 记录失败 episode，检索相似历史并给出 recovery candidate
- 比较无记忆、固定窗口、retrieval memory 与显式 task graph

---

## Lesson 16 — Hierarchical Planning & Task-Conditioned VLA

### 理论
- instruction → task graph → current subgoal → action chunk
- hierarchical policy、skill library 与 subgoal-conditioned control
- progress monitoring、failure detection、replanning 与 recovery
- LLM / VLM reasoning 与可验证 symbolic state 的边界

### 实验
- 用 task graph 选择当前可执行 subgoal
- 让同一 VLA / policy 在不同 task state 下产生不同动作
- 注入执行失败或跳步，测试 agent 是否检测并恢复
- 比较 reactive VLA 与 task-memory-conditioned VLA

---

# Phase 5 — Task World Model & Planning

## Lesson 17 — Physical Dynamics & World Model

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

## Lesson 18 — Task World Model + Planning

### 理论
- policy learning vs planning
- MPC
- value / success model
- hierarchical policy
- 将物理状态预测与任务状态预测区分开：
  - physical model：`p(s_{t+1} | s_t, a_t)`
  - task model：`p(z_{t+1}^{task} | z_t^{task}, skill_t, observation)`
- precondition / effect、progress prediction、uncertainty 与 counterfactual rollout

### ManiSkill Lab
- 在同一环境比较 reactive policy 与 rollout-based planning
- 将 failure trajectory 用于 success/failure prediction
- 预测“行动后任务会变成什么状态”，而不只预测下一帧或关节状态
- 使用 learned transition 更新 dependency graph 中的节点状态

### 数据问题
- successful demonstrations、failures、recovery trajectories 分别提供什么监督？

---

# Phase 6 — Human-Centered Data & Sim2Real

## Lesson 19 — ManiSkill Sim2Real

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

## Lesson 20 — Real2Sim 与 Digital Twin

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

## Lesson 21 — Human Demonstration & VR Data Engine

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
- gaze / location / interaction signals → human state estimation
- confidence / attention / task progress → intervention policy
- guidance、ask、wait、take over 等 assistance mode 的选择

---

# Phase 7 — Evaluation & Data Closed Loop

## Lesson 22 — Multi-source Dataset Mixture

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

## Lesson 23 — Evaluation：Offline ≠ Closed-loop

### 理论
- prediction/action loss
- success rate
- robustness
- perturbation test
- generalization
- failure taxonomy
- task segmentation quality、task-state accuracy、subgoal success、recovery rate
- intervention precision / recall 与 unnecessary intervention rate

### ManiSkill Lab
- 大规模 parallel evaluation
- object pose / camera / lighting / dynamics perturbation
- closed-loop failure collection

### Real-world 对照
- 同一 policy 的 sim vs real success/failure pattern
- 分析 evaluation gap

---

## Lesson 24 — Bad Case → Data → Retraining

### 闭环
`Policy rollout → failure detection → segment → label → sim reproduction/augmentation → dataset update → retraining → reevaluation`

### ManiSkill 的角色
- 重放真实失败
- 生成相邻 perturbation
- 长尾 failure/recovery 扩增
- scalable regression testing

---

# Phase 8 — Capstone

## Lesson 25 — Task-Centric Embodied Agent

目标：完成一个真正贯穿全部课程的项目。

### Pipeline

`Human Demonstration (Ego Video + VR/Robot Data)`
→ `Multimodal Alignment`
→ `Temporal Segmentation`
→ `Task Graph + Episodic Memory`
→ `Current Task-State Estimation`
→ `Subgoal / Assistance Decision`
→ `VLA / ACT / Diffusion Policy Execution`
→ `Closed-loop Progress & Failure Detection`
→ `Recovery / Human Intervention`
→ `Bad-case Collection and Retraining`

### 最终能力
学习结束后，应能够独立回答并实现：
- 一个 robot dataset 的 state/action 到底代表什么；
- 如何把 VR、simulation、real robot 数据统一到可训练的数据接口；
- 哪些数据可以共享、哪些只能作为 privileged supervision；
- 如何选择 BC / ACT / Diffusion / VLA；
- 如何从示范中提取 task phase、dependency 与 task state；
- 如何把 task memory 和当前 subgoal 接入 VLA；
- 如何区分 physical world model 与 task world model；
- 如何评估 long-horizon progress、failure recovery 与 intervention quality；
- 如何在 ManiSkill 中训练、评估和诊断 policy；
- 如何把 policy 从 simulation 推向 real robot；
- 如何把真实 bad case 重新转化成训练数据。

---

# 贯穿课程的固定实验任务

为了避免每一课都换环境，建议长期保持三层任务：

1. **基础任务：PickCube-v1** — action/state/control/dataset/BC 的主实验。
2. **第二任务：PushCube-v1 或同类简单平面操作** — 比较 action representation 与 generalization。
3. **接触丰富任务：PegInsertionSide-v1 或等价 insertion task** — 后期用于 Diffusion/VLA/Sim2Real/robustness。

进入 Task Intelligence 阶段后增加一个由多个 skill 组成的**长时程组合任务**，用于
task segmentation、dependency、memory、progress tracking 与 failure recovery。只有进入
multi-task VLA 阶段后再显著扩展任务数量。

---

# 当前进度

详细状态与证据分级见 `notes/progress.md`（单一进度记录）。本段只保留头条状态。

- Lesson 0：完成
- Lesson 1：完成
- Lesson 2：**完成**。2.1–2.9 全部走通，并以一个**负结果**收尾：BC 模型拟合了 4 条
  训练 episode，但无法泛化到未见 episode；闭环 rollout 在 seed 100 上未能完成任务。
  训练代码本身正确，当前模型是**失败基线**（failure baseline），不是可用 policy。
- Lesson 3：**下一阶段** — 模仿学习与行为克隆（3.1–3.10），入口即上述实测结果。
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
- 完整训练循环：epoch 循环、Adam、Loss 曲线；
- 2.9 专家演示：motion planner 产出 5 条成功 episode，规范 `T+1` / `T` schema，
  另有 delta 语义的重定向副本，replay 5/5 成功；
- **2.8.7 训练集与验证集**：按 episode 划分（4 训练 / 1 验证，`seed=42`），
  归一化统计只用训练集；training Loss `1.3136 → 0.006`，validation 停在
  `0.75–0.77`，best `0.2350`（epoch 3），early stopping 于 epoch 40 触发；
  best 模型**劣于** mean-action baseline `0.1421`；
- **2.8.8 策略部署与闭环执行**：best checkpoint + action clipping，未见 `seed=100`；
  50 步与 200 步两次运行均 `Success: False`，200 步运行中 `199/200` 步出现被裁剪的
  action 通道，从而排除时间上限因素；
- **结论（正式记录）**：The BC model fitted the four training demonstrations but failed
  to generalize to an unseen episode. Its best validation MSE (0.2350) was worse than the
  mean-action baseline (0.1421). During closed-loop rollout on unseen seed 100, the policy
  failed to complete the task and produced out-of-range actions on 199 of 200 steps,
  demonstrating severe overfitting and compounding distribution shift.

## 下一阶段（数据与理论，而不是修训练代码）

1. **数据规模化**：扩充到至少 30–50 条 expert episode
   （`scripts/generate_expert_demo.py` 只写成功 episode），用同一 episode-level split
   重新训练，并与当前失败基线对比；
2. **Lesson 3**：3.1–3.4（问题定义、BC 作为监督学习、泛化与过拟合、distribution
   shift）直接以上述实测数字作为入口；
3. 遗留项（不阻塞）：`inspect_robot_dataset.py`（本课验收项，目前缺失）、manifest
   溯源、20 Hz / 50 Hz 时间契约、42 维 observation 的逐字段命名与可部署性标注。
