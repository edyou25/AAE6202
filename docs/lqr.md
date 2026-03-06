# B747控制算法设计：凸优化应用

## 1. 背景介绍
本项目的任务是：让 Boeing 747 在二维平面内跟踪给定圆轨迹。

控制结构如下设计：
- 期望计算：根据圆轨迹几何关系给出期望航向 `psi_des`。
- LQR轨迹跟踪控制：用 LQR 控制航向误差/倾侧误差，输出滚转指令，应用AAE6202凸优化设计。
- 速度控制：单独用推力调节保持 `v_ref`。

对应代码在 [controller.py](/home/yyf/AAE6202/controller.py)。

## 2. 建模与误差状态定义
为了让控制问题可解且可调，本项目将内环跟踪误差建模为二维状态：

\[
x_k=
\begin{bmatrix}
e_{\psi,k}\\
e_{\phi,k}
\end{bmatrix},
\quad
u_k=\Delta\phi_{cmd,k}
\]

其中：
- \(e_{\psi}=\psi-\psi_{des}\)：航向误差
- \(e_{\phi}=\phi-\phi_{ff}\)：倾侧误差
- \(\Delta\phi_{cmd}=\phi_{cmd}-\phi_{ff}\)：相对前馈倾侧的控制增量

在参考速度 \(v_{ref}\) 附近线性化后，连续时间误差模型可写为：

\[
\dot{x}=Ax+Bu
\]

\[
A=
\begin{bmatrix}
0 & g/v_{ref}\\
0 & -1/\tau_\phi
\end{bmatrix},
\quad
B=
\begin{bmatrix}
0\\
1/\tau_\phi
\end{bmatrix}
\]

按采样周期 \(dt\) 离散化（项目中采用 \(A_d\approx I+dtA,\;B_d\approx dtB\)）。

## 3. LQR优化问题描述
### 3.1 目标函数
项目内环采用二次型目标函数：

\[
\min_{\{u_k\}}\;J=\sum_{k=0}^{\infty}
\left(
x_k^TQx_k + u_k^TRu_k
\right)
\]

其中：
- \(Q=\mathrm{diag}(q_{e_\psi}, q_{e_\phi})\)：误差权重
- \(R=[r_u]\)：控制增量权重

### 3.2 约束
优化问题的基础约束是系统动力学：

\[
x_{k+1}=A_dx_k+B_du_k
\]

在工程实现上，另外还有物理限幅（在控制律输出后施加）：
- `phi_cmd` 限幅：\([-\phi_{limit},+\phi_{limit}]\)
- `thrust` 限幅：\([0,\,T_{max}]\)

### 3.3 为什么这是凸优化（凸性说明）
将全时域变量堆叠后，问题可写成标准二次规划（QP）形式：

\[
\min_z\;\frac12 z^THz+f^Tz,\quad \text{s.t.}\;Ez=b
\]

凸性来源于两点：
1. 约束 \(Ez=b\) 是仿射约束，可行域是凸集。
2. 当 \(Q\succeq 0,\;R\succ 0\) 时，Hessian \(H\succeq 0\)，目标函数凸。

因此该问题是凸优化，最优解是全局最优解。

## 4. 最优解形式与项目实现
无硬不等式约束时，LQR最优解可写成线性状态反馈：

\[
u_k=-Kx_k
\]

其中 \(K\) 由离散Riccati方程求得。  
项目中对应流程：
- 在 `dlqr(...)` 中迭代求解 Riccati 矩阵
- 得到反馈增益 `k_lqr`
- 计算 `delta_phi_cmd = -(k_lqr @ err)`，再与 `phi_ff` 叠加得到 `phi_cmd`

## 5. 参数物理意义与调参方向
- `q_e_psi` 增大：更强地压航向误差，转向更积极。
- `q_e_phi` 增大：更强调滚转误差收敛，姿态更稳。
- `r_u` 增大：抑制控制动作，响应更平滑但可能变慢。

实际调参建议：
1. 先保证稳定（适当增大 `r_u`）。
2. 再逐步提高 `q_e_psi` 以改善跟踪。
3. 最后细调 `q_e_phi` 折中“跟踪速度/姿态平滑”。

## 6. 小结
本项目中的LQR是“B747圆轨迹内环误差控制器”：
- 控制目标清晰：压缩 \(e_\psi,e_\phi\)。
- 数学形式规范：二次目标 + 线性约束，属于凸优化问题。
- 工程实现可落地：Riccati求增益 + 限幅保护，形成可运行闭环控制。
