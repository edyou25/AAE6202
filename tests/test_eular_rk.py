# RK4
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6202)

# B747参数
m = 333400.0
S = 541.2
g = 9.81
rho = 1.225
CD0 = 0.02
k = 0.045
tau_phi = 2.0

# 计算阻力
def calculate_drag(v, phi):
    CL = (2 * m * g) / (rho * v ** 2 * S * np.cos(phi))
    CD = CD0 + k * CL ** 2
    D = 0.5 * rho * v ** 2 * S * CD
    return D

def b747_dynamics(state, control):
    x, y, psi, phi, v = state
    phi_cmd, T = control
    D = calculate_drag(v, phi)

    dx_dt = v * np.cos(psi)
    dy_dt = v * np.sin(psi)
    dpsi_dt = (g * np.tan(phi)) / v
    dphi_dt = (phi_cmd - phi) / tau_phi
    dv_dt = (T - D) / m

    return np.array([dx_dt, dy_dt, dpsi_dt, dphi_dt, dv_dt])

def rk4_step(state, control, dt):
    k1 = b747_dynamics(state, control)
    k2 = b747_dynamics(state + 0.5 * dt * k1, control)
    k3 = b747_dynamics(state + 0.5 * dt * k2, control)
    k4 = b747_dynamics(state + dt * k3, control)
    state_next = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return state_next

def euler_step(state, control, dt):
    dstate = b747_dynamics(state, control)
    state_next = state + dt * dstate
    return state_next

def get_control(t):
    phi_cmd = np.deg2rad(15) + np.deg2rad(10) * np.sin(2 * np.pi * t / 20)  #时变控制输入，让倾斜角围绕15°上下波动。因为用固定控制系统会很快收敛到平衡点，绘制的bank_angle误差几乎看不到
    T = 650000.0
    return np.array([phi_cmd, T])

T_sim = 100.0  # 总的仿真时间
dt_small = 0.05  # 小步长
dt_large = 0.5  # 大步长

x0 = np.array([0.0, 0.0, np.deg2rad(45), np.deg2rad(5), 210.0])

t_ref = np.arange(0, T_sim, dt_small)
states_ref = np.zeros((len(t_ref), 5))
states_ref[0] = x0

for i in range(len(t_ref) - 1):
    t = t_ref[i]
    u = get_control(t)
    states_ref[i + 1] = rk4_step(states_ref[i], u, dt_small)

t_rk4 = np.arange(0, T_sim, dt_large)
states_rk4 = np.zeros((len(t_rk4), 5))
states_rk4[0] = x0

for i in range(len(t_rk4) - 1):
    t = t_rk4[i]
    u = get_control(t)
    states_rk4[i + 1] = rk4_step(states_rk4[i], u, dt_large)

t_euler = np.arange(0, T_sim, dt_large)
states_euler = np.zeros((len(t_euler), 5))
states_euler[0] = x0

for i in range(len(t_euler) - 1):
    t = t_euler[i]
    u = get_control(t)
    states_euler[i + 1] = euler_step(states_euler[i], u, dt_large)

error_rk4 = np.zeros((len(t_rk4), 5))
error_euler = np.zeros((len(t_euler), 5))

for i, t in enumerate(t_rk4):
    idx_ref = int(t / dt_small)
    if idx_ref < len(states_ref):
        error_rk4[i] = np.abs(states_rk4[i] - states_ref[idx_ref])
        error_euler[i] = np.abs(states_euler[i] - states_ref[idx_ref])

phi_error_rk4_deg = np.rad2deg(error_rk4[:, 3])
phi_error_euler_deg = np.rad2deg(error_euler[:, 3])

print(f"RK4:")
print(f"最小: {phi_error_rk4_deg.min():.10f} deg")
print(f"最大: {phi_error_rk4_deg.max():.10f} deg")
print(f"平均: {phi_error_rk4_deg.mean():.10f} deg")

print(f"\nEuler:")
print(f"最小: {phi_error_euler_deg.min():.10f} deg")
print(f"最大: {phi_error_euler_deg.max():.10f} deg")
print(f"平均: {phi_error_euler_deg.mean():.10f} deg")

if phi_error_rk4_deg.max() > 1e-12:
    ratio = phi_error_euler_deg.max() / phi_error_rk4_deg.max()
    print(f"\n误差比: {ratio:.2f}x")

fig = plt.figure(figsize=(14, 10))
fig.suptitle('RK4 vs Euler Method: Accuracy Comparison',
             fontsize=13, fontweight='bold')

# 位置误差X
plt.subplot(2, 3, 1)
plt.plot(t_rk4, error_rk4[:, 0], 'r-s', markersize=1, linewidth=1.5, label='RK4')
plt.plot(t_euler, error_euler[:, 0], 'b-o', markersize=1, linewidth=1.5, label='Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.title('X Position Error')
plt.legend()
plt.grid(True, alpha=0.3)

# 位置误差Y
plt.subplot(2, 3, 2)
plt.plot(t_rk4, error_rk4[:, 1], 'r-s', markersize=1, linewidth=1.5, label='RK4')
plt.plot(t_euler, error_euler[:, 1], 'b-o', markersize=1, linewidth=1.5, label='Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.title('Y Position Error')
plt.legend()
plt.grid(True, alpha=0.3)

# 航向角误差
plt.subplot(2, 3, 3)
plt.plot(t_rk4, np.rad2deg(error_rk4[:, 2]), 'r-s', markersize=1, linewidth=1.5, label='RK4')
plt.plot(t_euler, np.rad2deg(error_euler[:, 2]), 'b-o', markersize=1, linewidth=1.5, label='Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error (deg)')
plt.title('Heading Angle Error')
plt.legend()
plt.grid(True, alpha=0.3)

# 倾斜角误差
plt.subplot(2, 3, 4)
eps = 1e-15
plt.semilogy(t_rk4, phi_error_rk4_deg + eps, 'r-s',
             markersize=1, linewidth=1.5, label='RK4')
plt.semilogy(t_euler, phi_error_euler_deg + eps, 'b-o',
             markersize=1, linewidth=1.5, label='Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error (deg, log scale)')
plt.title('Bank Angle Error (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3, which='both')

# 速度误差
plt.subplot(2, 3, 5)
plt.plot(t_rk4, error_rk4[:, 4], 'r-s', markersize=1, linewidth=1.5, label='RK4')
plt.plot(t_euler, error_euler[:, 4], 'b-o', markersize=1, linewidth=1.5, label='Euler')
plt.xlabel('Time (s)')
plt.ylabel('Error (m/s)')
plt.title('Airspeed Error')
plt.legend()
plt.grid(True, alpha=0.3)

# 总误差
plt.subplot(2, 3, 6)
error_norm_rk4 = np.linalg.norm(error_rk4, axis=1)
error_norm_euler = np.linalg.norm(error_euler, axis=1)
plt.semilogy(t_rk4, error_norm_rk4 + eps, 'r-s', markersize=1, linewidth=1.5, label='RK4')
plt.semilogy(t_euler, error_norm_euler + eps, 'b-o', markersize=1, linewidth=1.5, label='Euler')
plt.xlabel('Time (s)')
plt.ylabel('Total Error Norm (log scale)')
plt.title('Overall State Error')
plt.legend()
plt.grid(True, alpha=0.3, which='both')

plt.tight_layout()
out_path = Path("latex") / "assets" / "rk4_vs_euler_comparison_fixed.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')


print(f"\n在t={T_sim}s时刻:")
print(f"{'状态':<10} {'RK4误差':<15} {'Euler误差':<15} {'倍数':<10}")
print("-" * 55)


# 计算误差比
def calc_ratio(e1, e2):
    if e2 > 1e-15:
        return e1 / e2
    else:
        return float('nan')


print(
    f"{'X (m)':<10} {error_rk4[-1, 0]:<15.6f} {error_euler[-1, 0]:<15.6f} {calc_ratio(error_euler[-1, 0], error_rk4[-1, 0]):<10.2f}")
print(
    f"{'Y (m)':<10} {error_rk4[-1, 1]:<15.6f} {error_euler[-1, 1]:<15.6f} {calc_ratio(error_euler[-1, 1], error_rk4[-1, 1]):<10.2f}")
print(
    f"{'Psi (deg)':<10} {np.rad2deg(error_rk4[-1, 2]):<15.6f} {np.rad2deg(error_euler[-1, 2]):<15.6f} {calc_ratio(error_euler[-1, 2], error_rk4[-1, 2]):<10.2f}")
print(
    f"{'Phi (deg)':<10} {np.rad2deg(error_rk4[-1, 3]):<15.6f} {np.rad2deg(error_euler[-1, 3]):<15.6f} {calc_ratio(error_euler[-1, 3], error_rk4[-1, 3]):<10.2f}")
print(
    f"{'V (m/s)':<10} {error_rk4[-1, 4]:<15.6f} {error_euler[-1, 4]:<15.6f} {calc_ratio(error_euler[-1, 4], error_rk4[-1, 4]):<10.2f}")

print(f"\n总误差范数:")
print(f"  RK4:   {error_norm_rk4[-1]:.6f}")
print(f"  Euler: {error_norm_euler[-1]:.6f}")
if error_norm_rk4[-1] > 1e-10:
    print(f"  比值:  {error_norm_euler[-1] / error_norm_rk4[-1]:.2f}x")

print(f"\n平均误差:")
print(f"  RK4:   {np.mean(error_norm_rk4):.6f}")
print(f"  Euler: {np.mean(error_norm_euler):.6f}")
if np.mean(error_norm_rk4) > 1e-10:
    print(f"  比值:  {np.mean(error_norm_euler) / np.mean(error_norm_rk4):.2f}x")

plt.show()
