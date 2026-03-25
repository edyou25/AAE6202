# 单元测试与实验结果报告

## 1. 测试目标
- 覆盖控制、动力学、估计、可视化、运行入口与报告绘图六个部分。
- 将 `tests` 与 `run.py` 的实际运行结果整理为可直接引用的实验记录。

## 2. 测试命令
- 单元测试：`python -m unittest discover -s tests -v`
- 报告图生成：`python reporting.py`
- 主程序运行：`$env:MPLBACKEND='Agg'; python run.py`

## 3. 测试覆盖内容
- `tests/test_controller.py`：`wrap_pi`、`dlqr`、圆轨迹控制律与限幅。
- `tests/test_dynamics.py`：动力学裁剪逻辑、直线平衡飞行、保持滚转时的转弯响应。
- `tests/test_estimation.py`：协方差构造、完美测量更新、带噪测量下的后验收敛。
- `tests/test_visual.py`：机体点云缩放、坐标变换、动画抽帧调度。
- `tests/test_run.py`：短时仿真输出维度与总体结果绘图保存。
- `tests/test_reporting.py`：报告图批量生成。

## 4. 单元测试结果
- 运行时间：2026-03-25
- 命令：`python -m unittest discover -s tests -v`
- 结果：`Ran 16 tests in 1.828s`
- 结论：`OK`

## 5. 主程序实验结果
- 命令：`$env:MPLBACKEND='Agg'; python run.py`
- 最终位置：`x = 20941.9 m`，`y = 118158.3 m`
- 最终速度：`209.72 m/s`
- 最终有符号横向误差：`0.22 m`
- 最终绝对横向误差：`0.22 m`
- 备注：`FigureCanvasAgg is non-interactive` 为无界面后端提示，不影响数值结果输出。

## 6. 附加材料分析

### 6.1 `addicted/B747system.py`
- 建立了与主项目一致的五维状态模型：`[x, y, psi, phi, v]`。
- 使用相同的阻力模型与滚转一阶响应结构。
- 通过 `dt_ref = 0.05 s` 的 RK4 参考解，对比 `dt_test = 0.5 s` 下的 RK4 与 Euler 精度。
- 在 `t = 100 s` 时，总状态误差范数：`RK4 = 2.729726`，`Euler = 43.920315`，Euler 约为 RK4 的 `16.09x`。
- 平均总误差范数：`RK4 = 2.250438`，`Euler = 28.479260`，Euler 约为 RK4 的 `12.65x`。
- 结论：该文件可作为主报告中 RK4 优于 Euler 的独立数值佐证。

### 6.2 `addicted/Methodology.pdf`
- 详细解释了 RK4 四个斜率 `k1`、`k2`、`k3`、`k4` 的物理含义。
- 强调了 B747 非线性平面动力学中，位置、航向、滚转与阻力存在耦合，因此需要高精度积分器。
- 给出了 RK4 与 Euler 的误差阶比较，适合作为方法论文字说明来源。
- 可提炼内容已补充到 `latex/_main.tex` 的 RK4 方法和结果分析部分。

## 7. 报告图产物
- `data/report/simulation_overview.png`：总体轨迹、横向误差、滚转响应、速度/航向误差。
- `data/report/part1_control.png`：LQR 控制误差、滚转跟踪、推力与目标函数。
- `data/report/part2_dynamics.png`：轨迹、平动速度、角速度、速度动力学。
- `data/report/part3_estimation.png`：估计误差、创新量、协方差迹、真实/估计位置对比。

![Simulation Overview](../data/report/simulation_overview.png)

![Part 1 Control](../data/report/part1_control.png)

![Part 2 Dynamics](../data/report/part2_dynamics.png)

![Part 3 Estimation](../data/report/part3_estimation.png)
