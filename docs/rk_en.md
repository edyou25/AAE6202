# B747 Dynamics Simulation: Solving Differential Equations with the Runge-Kutta Method

> Part 2: RK4 (Continuous Dynamics Discrete Propagation)

## 1. Background

This project simulates the plant using differential-equation integration to provide a controlled object for validating the control algorithm.  
The simulated dynamics are a continuous-time model whose state is:

\[
\mathbf{x}=[x,\;y,\;\psi,\;\phi,\;v]^T
\]

where:
- \(x,y\): planar position
- \(\psi\): heading angle
- \(\phi\): bank angle
- \(v\): airspeed

Control inputs:
- `phi_cmd` (bank command)
- `thrust` (thrust force)

The corresponding dynamics code is in [dynamics.py](../dynamics.py) — function `b747_dynamics(...)`.

## 2. Differential-Equation Modeling

The project uses a 2D planar flight model. The state equation is:

\[
\dot{\mathbf{x}} = f(\mathbf{x},\mathbf{u})
\]

Core components:
- \(\dot x = v\cos\psi\)
- \(\dot y = v\sin\psi\)
- \(\dot\psi = g\tan\phi / v\)
- \(\dot\phi = (\phi_{cmd}-\phi)/\tau_\phi\)
- \(\dot v = (T-D)/m\)

where the drag \(D\) is computed from aerodynamic parameters (parasite drag + induced drag).

## 3. Why RK4 Instead of Euler's Method

With the forward Euler method:

\[
x_{k+1}=x_k+h f(t_k,x_k)
\]

only one slope is used per step, resulting in larger errors.  
The reasons for choosing RK4 in this project are:
1. Higher accuracy in nonlinear turning scenarios.
2. Better numerical stability.
3. Low implementation complexity; widely used in engineering.

## 4. RK4 Discrete Propagation Formula

Within each control interval \(dt\), RK4 computes:

\[
\begin{aligned}
k_1 &= f(x_k,u_k),\\
k_2 &= f\!\left(x_k+\frac{dt}{2}k_1,u_k\right),\\
k_3 &= f\!\left(x_k+\frac{dt}{2}k_2,u_k\right),\\
k_4 &= f(x_k+dt\,k_3,u_k),\\
x_{k+1} &= x_k+\frac{dt}{6}(k_1+2k_2+2k_3+k_4)
\end{aligned}
\]

The corresponding implementation function is `rk4_step(...)`.

## 5. Coupling Workflow with the Controller

The execution order of each step in this project:
1. The controller computes `phi_cmd` and `thrust` from the current state.
2. The control inputs are treated as constant over one `dt` interval.
3. RK4 integrates the continuous dynamics to obtain the next state.

This can be understood as "discrete control updates + continuous-dynamics approximate propagation".

## 6. RK4 Accuracy and the Project's Time Scale

Typical RK4 accuracy:
- Local truncation error: \(O(dt^5)\)
- Global error: \(O(dt^4)\)

The project's default `dt = 0.05 s` gives a simulation update frequency of approximately 20 Hz.  
Therefore, if the animation is played at real time without frame decimation, the visual update rate should also be close to 20 Hz.

## 7. Parameter Sensitivity and Practical Recommendations

If trajectory oscillations, large errors, or non-smooth speed profiles appear, investigate in the following order:
1. Reduce `dt` from 0.05 to 0.02 and compare results.
2. Observe the final radial error, airspeed, and attitude angle variations.
3. If results change significantly, reduce the step size first before adjusting control parameters.

## 8. Summary

In this project, RK4 is the key link connecting the "continuous B747 dynamics" and the "discrete controller output":
- It keeps the simulation physically continuous while remaining computationally feasible.
- It provides a stable and trustworthy numerical foundation for validating the LQR control performance.
