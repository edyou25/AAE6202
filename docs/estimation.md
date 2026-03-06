# Part3：B747卡尔曼滤波状态估计：Bayes/MAP 

## 1. 背景
在真实飞行控制中，控制器拿到的是“带噪测量”，不是完全准确真值。  
因此本项目加入估计模块，用于演示：
- 如何用动力学预测先验状态
- 如何结合测量做后验更新
- 如何把估计指标接入可视化面板

实现代码： [estimation.py](/home/yyf/AAE6202/estimation.py)

## 2. Bayes估计
### 2.1 先验（Prior = Prediction）
在时刻 \(k\) 的先验由动力学预测给出：

\[
x_k^- = f(x_{k-1}^+, u_{k-1})
\]

本项目里预测器使用与仿真一致的 RK4 过程模型。

### 2.2 似然（Likelihood = Measurement）
测量模型设为：

\[
z_k = x_k + v_k,
\quad v_k\sim \mathcal{N}(0,R)
\]

即传感器测量等于真实状态叠加高斯噪声。

### 2.3 后验（Posterior = Update）
结合先验与似然，得到后验估计：

\[
x_k^+ = x_k^- + K_k(z_k-x_k^-)
\]

\[
K_k = P_k^- (P_k^- + R)^{-1}
\]

并更新协方差：

\[
P_k^+ = (I-K_k)P_k^-
\]

## 3. MAP优化解释
在高斯假设下，后验最大化等价于最小化如下目标：

\[
\min_x \frac12(x-x_k^-)^T(P_k^-)^{-1}(x-x_k^-)
+\frac12(z_k-x)^TR^{-1}(z_k-x)
\]

这就是本项目 Part3 的 MAP（最大后验）含义：
- 第一项：先验约束（相信预测）
- 第二项：测量约束（相信传感器）
- 最优点：先验和测量加权折中

## 4. 本项目里记录了哪些估计参数
在 `run.py` 仿真循环中，已经记录并送到 `visual.py`：
- `est_innovation_norm`：创新范数 \(\|z_k-x_k^-\|\)
- `est_prior_err_norm`：先验误差范数 \(\|x_k^- - x_k\|\)
- `est_post_err_norm`：后验误差范数 \(\|x_k^+ - x_k\|\)
- `est_map_obj`：MAP目标函数值
- `est_update_ms`：估计更新时间
- `est_trace_prior` / `est_trace_post`：协方差迹（不确定性尺度）
- `est_state_hat`：后验状态估计

## 5. 可视化
- `prior / likelihood / posterior` 的语义提示
- 创新、先验误差、后验误差
- MAP目标值、估计耗时
- `tr(P-)` 与 `tr(P+)`
- `x_hat, y_hat, psi_hat, phi_hat, v_hat`

这样可以直接观察“预测-测量-更新”在每个控制周期的效果。

## 6. 小结
Part3 的作用不是替代控制器，而是补齐“状态认知层”演示：
- Part1 负责控制律（LQR）
- Part2 负责动力学推进（RK4）
- Part3 负责状态估计（Bayes/MAP）

三者组合后，仿真更接近真实飞行控制系统的数据流。
