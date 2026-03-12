# 单元测试

## 目标
- 为项目的控制、动力学、估计、可视化与集成流程补齐可重复运行的单元测试。
- 通过 `matplotlib` 输出可直接放入课程报告的图表。

## 运行方式
- 运行全部单元测试：`python -m unittest discover -s tests -v`
- 生成报告图：`python reporting.py`

## 测试覆盖
- `tests/test_controller.py`：`wrap_pi`、`dlqr`、圆轨迹控制律与限幅。
- `tests/test_dynamics.py`：动力学裁剪逻辑、直线平衡飞行、保持滚转时的转弯响应。
- `tests/test_estimation.py`：协方差构造、完美测量更新、带噪测量下的后验收敛。
- `tests/test_visual.py`：机体点云缩放、坐标变换、动画抽帧调度。
- `tests/test_run.py`：短时仿真输出维度与总体结果绘图。
- `tests/test_reporting.py`：报告图批量生成。

## 报告图说明
- `data/report/simulation_overview.png`：总体轨迹、横向误差、滚转响应、速度/航向误差。
- `data/report/part1_control.png`：LQR 控制误差、滚转跟踪、推力与目标函数。
- `data/report/part2_dynamics.png`：轨迹、平动速度、角速度、速度动力学。
- `data/report/part3_estimation.png`：估计误差、创新量、协方差迹、真实/估计位置对比。

## 可直接引用的 Markdown
![Simulation Overview](../data/report/simulation_overview.png)

![Part 1 Control](../data/report/part1_control.png)

![Part 2 Dynamics](../data/report/part2_dynamics.png)

![Part 3 Estimation](../data/report/part3_estimation.png)
