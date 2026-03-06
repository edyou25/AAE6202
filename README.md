# AAE6202

![alt text](assets/747.png)

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

## 依赖
- `conda env create -f env.yaml`

## 理论文档
- [LQR 原理（凸优化、目标函数、约束、凸性）](/home/yyf/AAE6202/docs/lqr.md)
- [RK 方法原理（含 RK4）](/home/yyf/AAE6202/docs/rk.md)

