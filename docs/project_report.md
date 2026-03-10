# Boeing 747 Circular Trajectory Tracking via LQR Control, RK4 Dynamics Simulation, and Bayesian MAP State Estimation

## Abstract
This project develops an integrated simulation framework for Boeing 747 circular-flight tracking in a 2D plane. The framework combines: (i) a linear quadratic regulator (LQR) for heading/bank inner-loop stabilization, (ii) a fourth-order Runge-Kutta (RK4) integrator for nonlinear aircraft dynamics propagation, and (iii) a Gaussian Bayes/MAP estimator for state estimation diagnostics under noisy measurements. Numerical experiments based on the project code show accurate path convergence and stable speed regulation, with final cross-track error close to zero and bounded bank angle throughout the simulation horizon. The report presents problem formulation, methodology, numerical results, and limitations.

## 1. Literature Review

### 1.1 LQR for trajectory tracking
LQR is a standard optimal control method for linear systems with quadratic costs. For discrete-time systems,
\[
x_{k+1}=A_d x_k + B_d u_k,
\]
the infinite-horizon LQR minimizes
\[
J=\sum_{k=0}^{\infty}\left(x_k^\top Q x_k + u_k^\top R u_k\right),
\]
with \(Q\succeq0,\;R\succ0\), yielding a linear state feedback \(u_k=-Kx_k\) from the Riccati equation. For aircraft path tracking, LQR is widely used in inner loops due to predictable tuning and strong stability properties [1]-[3].

### 1.2 Numerical integration for nonlinear flight dynamics
Aircraft dynamics are nonlinear, and closed-form solutions are generally unavailable. RK4 provides a practical balance between accuracy and implementation cost. Compared with forward Euler, RK4 has significantly lower truncation error and better stability at moderate step sizes, making it a common choice for control-oriented flight simulation [4], [5].

### 1.3 Bayesian estimation and MAP interpretation
Kalman filtering and Gaussian Bayesian estimation fuse model-based prediction with noisy measurements. Under Gaussian assumptions, maximizing posterior probability is equivalent to solving a weighted least-squares MAP problem. This interpretation is useful for explaining uncertainty tradeoffs between model confidence and sensor confidence [6]-[8].

### 1.4 Position of this project
This project integrates the three ideas above into one executable workflow:
1. LQR controls heading/bank errors for trajectory tracking.
2. RK4 advances the nonlinear B747 dynamics.
3. Bayes/MAP estimation provides state-estimation diagnostics.
The architecture demonstrates a compact yet complete control-estimation-simulation pipeline for aerospace coursework.

## 2. Problem Statement

### 2.1 Objective
Design and evaluate a controller-estimator simulation system that drives a Boeing 747 model to track a circular reference trajectory in planar flight while maintaining target speed.

### 2.2 State, input, and reference
The continuous state is
\[
\mathbf{x}=[x,\;y,\;\psi,\;\phi,\;v]^\top,
\]
where \(x,y\) are inertial coordinates, \(\psi\) is heading, \(\phi\) is bank angle, and \(v\) is airspeed.

Control input:
\[
\mathbf{u}=[\phi_{\text{cmd}},\;T]^\top,
\]
with commanded bank angle and thrust.

Reference trajectory: a circle centered at \((0,0)\) with radius \(R=120{,}000\;\text{m}\), counterclockwise.

### 2.3 Performance requirements
The tracking system should:
1. Reduce signed cross-track error \(e_{ct}=R-r\) toward zero.
2. Keep heading/bank behavior smooth and stable.
3. Maintain speed near \(v_{\text{ref}}=210\;\text{m/s}\).
4. Respect actuator/physics limits (\(\phi\) and thrust bounds).

### 2.4 Constraints and assumptions
1. Planar (2D) flight model.
2. Quasi-steady aerodynamic drag model.
3. Control update is discrete (\(\Delta t=0.05\;s\)); dynamics remain continuous and are numerically integrated.
4. Measurement noise is Gaussian; estimator is used for diagnostics (controller uses true state in this version).

## 3. Methodology

### 3.1 Nonlinear B747 planar dynamics
From `dynamics.py`, the model is:
\[
\dot x = v\cos\psi,\qquad
\dot y = v\sin\psi,
\]
\[
\dot\psi = \frac{g\tan\phi}{v},\qquad
\dot\phi = \frac{\phi_{\text{cmd}}-\phi}{\tau_\phi},\qquad
\dot v = \frac{T-D}{m}.
\]

Aerodynamic drag uses
\[
D=\frac12 \rho v^2 S C_D,\quad
C_D=C_{D0}+k C_L^2,\quad
C_L=\frac{2mg}{\rho v^2 S \cos\phi}.
\]

Key parameters (`B747Params`):
1. \(m=333{,}400\;\text{kg}\), \(S=541.2\;\text{m}^2\)
2. \(C_{D0}=0.02\), \(k=0.045\)
3. \(\tau_\phi=2.0\;\text{s}\)
4. \(|\phi|\le35^\circ\), \(T\in[0,\;1.1\times10^6]\;\text{N}\)
5. \(v\in[120,\;320]\;\text{m/s}\) (safety clipping in dynamics)

### 3.2 RK4 discretization
For each simulation step \(k\), with constant control over \([t_k,t_{k+1}]\), RK4 computes:
\[
k_1=f(x_k,u_k),\quad
k_2=f(x_k+\frac{\Delta t}{2}k_1,u_k),
\]
\[
k_3=f(x_k+\frac{\Delta t}{2}k_2,u_k),\quad
k_4=f(x_k+\Delta t\,k_3,u_k),
\]
\[
x_{k+1}=x_k+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
\]

### 3.3 Circular tracking controller
From `controller.py`, control has three layers:

1. Outer-loop geometry guidance:
   1. Compute current radius \(r\) and tangent heading.
   2. Cross-track error \(e_{ct}=R-r\) (positive = inside).
   3. Add bounded heading correction:
   \[
   \psi_{\text{des}}=\psi_{\text{tangent}}+\operatorname{clip}\left(\arctan\left(k_{ct}(-e_{ct})\right),\pm25^\circ\right).
   \]

2. Feedforward bank for circular motion:
   \[
   \phi_{ff}=\arctan\left(\frac{v_{ref}^2}{gR}\right),
   \]
   with turn sign for CW/CCW and saturation by bank limit.

3. Inner-loop LQR error feedback:
   \[
   e_\psi=\psi-\psi_{des},\qquad e_\phi=\phi-\phi_{ff}.
   \]
   Linearized error dynamics:
   \[
   \dot e_\psi=\frac{g}{v_{ref}}e_\phi,\qquad
   \dot e_\phi=-\frac{1}{\tau_\phi}e_\phi+\frac{1}{\tau_\phi}u,
   \]
   where \(u=\Delta\phi_{cmd}\).
   Using \(A_d\approx I+\Delta tA,\;B_d\approx\Delta tB\), solve discrete Riccati and apply:
   \[
   \Delta\phi_{cmd}=-K[e_\psi,\;e_\phi]^\top,\qquad
   \phi_{cmd}=\phi_{ff}+\Delta\phi_{cmd}.
   \]

Speed is regulated by:
\[
T = D_{\text{guess}} + k_v(v_{ref}-v),
\]
then clipped to thrust bounds.

Default control weights (`ControlConfig`):
\[
Q=\operatorname{diag}(30,\;10),\quad R=[1].
\]

### 3.4 Gaussian Bayes/MAP estimator
From `estimation.py`, estimator performs:

1. Prediction:
\[
x_k^- = f_{RK4}(x_{k-1}^+,u_{k-1}),\quad
P_k^- = P_{k-1}^+ + Q.
\]

2. Update:
\[
K_k=P_k^-(P_k^-+R)^{-1},
\]
\[
x_k^+=x_k^-+K_k(z_k-x_k^-),\quad
P_k^+=(I-K_k)P_k^-.
\]

3. MAP objective at posterior:
\[
\mathcal{L}_{MAP}
=\frac12(x_k^+-x_k^-)^\top(P_k^-)^{-1}(x_k^+-x_k^-)
+\frac12(z_k-x_k^+)^\top R^{-1}(z_k-x_k^+).
\]

Noise settings:
1. Process std: \((2,\;2,\;0.2^\circ,\;0.2^\circ,\;0.5)\)
2. Measurement std: \((40,\;40,\;1^\circ,\;1^\circ,\;2)\)
3. Initial std: \((50,\;50,\;2^\circ,\;2^\circ,\;3)\)

### 3.5 Simulation workflow
At each step in `run.py`:
1. Compute control and control diagnostics.
2. Generate noisy measurement.
3. Run estimator update.
4. Propagate true state with RK4.
5. Log telemetry for analysis and visualization.

## 4. Case Study / Numerical Results and Analysis

### 4.1 Experiment setup
Default experiment from `run.py`:
1. Time horizon: \(t_{end}=800\;s\), \(\Delta t=0.05\;s\), 16,000 steps.
2. Initial state:
   \[
   [x_0,y_0,\psi_0,\phi_0,v_0]=[R+2000,\;0,\;96^\circ,\;0^\circ,\;210].
   \]
   This starts the aircraft outside the reference circle with heading offset.
3. Random seed for measurement noise: 6202.

Generated artifacts:
1. `data/circle_flight_result.png`
2. `data/circle_flight_animation.gif`

### 4.2 Tracking and regulation performance
Key metrics from simulation:

| Metric | Value |
|---|---:|
| Final signed cross-track error \(e_{ct}\) | \(0.218\;m\) |
| Final \(|e_{ct}|\) | \(0.218\;m\) |
| Overall RMSE of \(e_{ct}\) (0-800 s) | \(269.24\;m\) |
| MAE of \(e_{ct}\) | \(60.64\;m\) |
| RMSE of \(e_{ct}\) (100-800 s) | \(5.56\;m\) |
| Settling time to \(|e_{ct}|<500\;m\) | \(27.8\;s\) |
| Settling time to \(|e_{ct}|<100\;m\) | \(78.35\;s\) |
| Heading error RMSE | \(2.31^\circ\) |
| Max \(|\phi|\) | \(27.67^\circ\) |
| Max \(|\phi_{cmd}|\) | \(35.0^\circ\) (limit) |
| Command saturation ratio (\(|\phi_{cmd}|=35^\circ\)) | \(0.356\%\) |
| Mean speed | \(209.724\;m/s\) |
| Speed RMSE to \(v_{ref}=210\) | \(0.276\;m/s\) |

Interpretation:
1. The large total RMSE is mainly due to intentional initial offset (\(2000\;m\)); after transient, radial tracking error becomes small.
2. Bank response remains inside physical constraints; saturation is rare, indicating reasonable gain tuning.
3. Speed loop maintains near-constant airspeed with very small variance.

### 4.3 Computational efficiency
| Metric | Value |
|---|---:|
| Mean controller computation time | \(0.0165\;ms\) |
| 95th percentile controller time | \(0.0176\;ms\) |
| Mean estimator update time | \(0.0344\;ms\) |
| 95th percentile estimator time | \(0.0365\;ms\) |

Both controller and estimator are lightweight relative to the 50 ms sample interval.

### 4.4 Estimation diagnostics
| Metric | Value |
|---|---:|
| Mean prior error norm \(\|x^- - x\|\) | \(8.380\) |
| Mean posterior error norm \(\|x^+ - x\|\) | \(8.379\) |
| Fraction of steps where posterior improves error | \(50.62\%\) |
| Mean innovation norm \(\|z-x^-\|\) | \(50.67\) |
| Mean trace reduction \(\mathrm{tr}(P^-)\rightarrow\mathrm{tr}(P^+)\) | \(5.15\%\) |

Interpretation:
1. Covariance is consistently reduced after update, confirming uncertainty contraction.
2. The posterior-vs-prior point estimate improvement is modest in this setup, indicating process and measurement confidence are of similar scale.
3. In this code version, control uses true state; therefore estimator quality does not directly affect closed-loop trajectory. A stronger integrated test would feed estimated state into control.

### 4.5 Visual evidence
From `data/circle_flight_result.png` and `data/circle_flight_animation.gif`:
1. Path converges from outside the circle to near-perfect circular motion.
2. Heading/bank transients decay smoothly.
3. Telemetry panel shows bounded control effort and stable estimation indicators over time.

## 5. Conclusion and Discussion

### 5.1 Conclusion
The project successfully demonstrates an integrated B747 tracking simulation pipeline. LQR stabilizes inner-loop errors and achieves precise circular tracking; RK4 provides reliable nonlinear propagation; Gaussian Bayes/MAP estimation offers interpretable uncertainty diagnostics. Under default settings, the system reaches very small steady-state cross-track and speed errors while respecting actuator limits.

### 5.2 Limitations
1. Planar 2D model omits altitude, pitch, and full 6-DOF couplings.
2. Discretization for LQR design uses first-order approximation (\(A_d\approx I+\Delta tA\)).
3. Estimator is decoupled from control action (diagnostic mode only).
4. Disturbances (wind, bias drift, actuator lag beyond bank channel) are simplified.

### 5.3 Future work
1. Use exact discretization for linear model or design directly in continuous time.
2. Close the loop with estimated states and compare robustness against noisy sensors.
3. Add wind/gust models and Monte Carlo statistics.
4. Extend from planar dynamics to higher-fidelity longitudinal/lateral coupled models.
5. Benchmark against alternative controllers (e.g., MPC, nonlinear guidance laws).

## 6. References
[1] B. D. O. Anderson and J. B. Moore, *Optimal Control: Linear Quadratic Methods*. Prentice-Hall, 1990.

[2] F. L. Lewis, D. L. Vrabie, and V. L. Syrmos, *Optimal Control*, 3rd ed. Wiley, 2012.

[3] K. J. Astrom and R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press, 2008.

[4] E. Hairer, S. P. Norsett, and G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Springer, 1993.

[5] J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd ed. Wiley, 2016.

[6] R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35-45, 1960.

[7] D. Simon, *Optimal State Estimation*. Wiley, 2006.

[8] G. Welch and G. Bishop, “An Introduction to the Kalman Filter,” UNC Chapel Hill, TR 95-041, 2006.

## 7. One-Page Summary of Group Contributions

Replace this page with your final member names/IDs and exact percentages before submission.

### 7.1 Contribution table
| Group Member | Student ID | Specific Components | Deliverables / Evidence | Overall Contribution (%) |
|---|---|---|---|---:|
| Member A | AAE6202-XXX | Dynamics modeling + RK4 implementation | `dynamics.py`, Part2 write-up | 40 |
| Member B | AAE6202-XXX | LQR controller design and tuning | `controller.py`, Part1 write-up | 35 |
| Member C | AAE6202-XXX | Bayes/MAP estimator + visualization integration | `estimation.py`, `visual.py`, Part3 write-up | 25 |
| **Total** |  |  |  | **100** |

### 7.2 Narrative summary (example format)
Member A led physical modeling and numerical integration, including aerodynamic drag and RK4 propagation. Member B designed the circular-guidance + LQR control architecture and tuned \(Q,R\), heading correction, and speed gains to satisfy stability and tracking requirements. Member C implemented the Gaussian Bayes/MAP estimator, integrated telemetry with visualization panels, and supported result analysis and documentation. All members jointly reviewed experiment settings, validated simulation outputs, and edited the final report.
