# LQR 原理说明

## 1. 问题定义
考虑离散线性系统

\[
x_{k+1}=Ax_k+Bu_k,
\]

其中 \(x_k\in\mathbb{R}^n\), \(u_k\in\mathbb{R}^m\)。

LQR（Linear Quadratic Regulator）在给定初值 \(x_0\) 下，寻找控制序列 \(u_0,\dots,u_{N-1}\) 最小化二次型性能指标。

## 2. 目标函数与约束
有限时域 LQR 常见写法：

\[
\min_{\{x_k,u_k\}}\; J=\sum_{k=0}^{N-1}\left(x_k^TQx_k+u_k^TRu_k\right)+x_N^TQ_fx_N
\]

满足约束

\[
\begin{aligned}
x_{k+1}&=Ax_k+Bu_k,\quad k=0,\dots,N-1,\\
x_0&=\bar x_0.
\end{aligned}
\]

典型权重条件：
- \(Q\succeq 0\)（半正定）
- \(R\succ 0\)（正定）
- \(Q_f\succeq 0\)

这保证“状态偏差”和“控制能量”都被惩罚，且控制惩罚严格凸。

## 3. 为什么这是凸优化（证明凸）
把所有决策变量堆叠成向量
\(z=[x_1,\dots,x_N,u_0,\dots,u_{N-1}]\)。

问题可写成

\[
\min_z\; \frac12 z^THz+f^Tz+c,\quad \text{s.t. } Ez=b.
\]

结论成立的关键：
1. **可行域凸**：\(Ez=b\) 是仿射等式约束，仿射集是凸集。
2. **目标函数凸**：当 \(Q,Q_f\succeq 0, R\succ 0\) 时，Hessian \(H\succeq 0\)。二次函数 Hessian 半正定即凸。

因此标准 LQR 是**凸二次优化问题（QP）**。

## 4. 无约束 LQR 的解析解
标准 LQR 只有系统动力学等式约束，没有不等式约束。它可以用动态规划得到闭式反馈律：

\[
u_k=-K_kx_k,
\]

其中

\[
K_k=(R+B^TP_{k+1}B)^{-1}B^TP_{k+1}A,
\]

\[
P_k=Q+A^TP_{k+1}A-A^TP_{k+1}B(R+B^TP_{k+1}B)^{-1}B^TP_{k+1}A,
\]

终端条件 \(P_N=Q_f\)。

无限时域稳定情形下，\(P_k\to P\)，得到离散代数 Riccati 方程（DARE）和常值反馈增益 \(K\)。

## 5. 约束如何进入 LQR
如果加入硬约束，例如：
- 输入约束 \(u_{\min}\le u_k\le u_{\max}\)
- 状态约束 \(x_{\min}\le x_k\le x_{\max}\)

则问题变为“**带线性约束的凸 QP**”，常见做法是 MPC 在线滚动求解。此时通常不再有简单闭式 Riccati 解，但仍保持凸性（目标凸 + 线性约束）。

## 6. 与本项目代码对应
本项目在 [controller.py](/home/yyf/AAE6202/controller.py) 中对内环误差状态做离散 LQR 设计：
- 构造线性化离散模型 \((A_d,B_d)\)
- 设定 \(Q,R\)
- 求解离散 Riccati（`dlqr(...)`）
- 得到 `k_lqr` 并用于 `phi_cmd` 修正

简化理解：
- `Q` 越大，越强调误差收敛（更“紧”）
- `R` 越大，越抑制控制动作（更“稳”但反应慢）

## 7. 小结
LQR 本质是线性系统上的二次最优控制问题。
- 从优化角度：它是凸 QP（标准形式）
- 从控制角度：可由 Riccati 方程得到最优线性反馈律
- 有约束时：自然过渡到凸 MPC/QP 框架
