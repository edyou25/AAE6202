# AAE6202

Boeing 747 圆轨迹飞行仿真（LQR 控制 + 4阶 RK 积分）。

## 文件说明
- `run.py`：程序主入口，运行仿真并绘图。
- `controller.py`：LQR 圆轨迹跟踪控制器（含速度保持）。
- `dynamics.py`：波音 747 二维平面动力学模型 + RK4 积分器。

## 运行方式
```bash
python run.py
```

运行后会：
- 在终端打印最终位置、速度、半径误差
- 弹出图像窗口（若环境支持）
- 保存结果图到 `circle_flight_result.png`

## 依赖
- `numpy`
- `matplotlib`

可选安装：
```bash
pip install numpy matplotlib
```
