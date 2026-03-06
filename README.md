# AAE6202

Boeing 747 圆轨迹飞行仿真（LQR 控制 + 4阶 RK 积分）。

## 文件说明
- `run.py`：程序主入口，运行仿真并绘图。
- `controller.py`：圆轨迹跟踪控制器（外环引导 + 内环 LQR + 速度保持）。
- `dynamics.py`：波音 747 二维平面动力学模型 + RK4 积分器。
- `visual.py`：飞机点云可视化模块（机身、机翼、尾翼）与动画导出。

## 运行方式
```bash
python3 run.py
```

窗口播放动画：
```bash
python3 run.py --show-animation
```

## 依赖
- `conda env create -f env.yaml`

运行后会在 `data/` 目录下生成：
- `circle_flight_result.png`（轨迹与误差曲线）
- `circle_flight_animation.gif`（点云飞机动画，GIF 不可用时回退 MP4）
