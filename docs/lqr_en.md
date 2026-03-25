# B747 Control Algorithm Design: Convex Optimization Application

> Part 1: LQR (Convex Optimization Background / Quadratic Form)

## 1. Background

The task of this project is to make a Boeing 747 track a given circular trajectory in a 2D plane.

The control structure is designed as follows:
- **Desired heading computation**: the desired heading `psi_des` is derived from the circular-trajectory geometry.
- **LQR trajectory tracking**: LQR controls heading error and bank error, outputs a bank-angle command, applying the AAE6202 convex-optimization design.
- **Speed control**: thrust is regulated independently to maintain `v_ref`.

The corresponding code is in [controller.py](../controller.py).

## 2. Modeling and Error-State Definition

To make the control problem tractable and tunable, this project models the inner-loop tracking error as a 2D state:

\[
x_k=
\begin{bmatrix}
e_{\psi,k}\\
e_{\phi,k}
\end{bmatrix},
\quad
u_k=\Delta\phi_{cmd,k}
\]

where:
- \(e_{\psi}=\psi-\psi_{des}\): heading error
- \(e_{\phi}=\phi-\phi_{ff}\): bank error
- \(\Delta\phi_{cmd}=\phi_{cmd}-\phi_{ff}\): control increment relative to feedforward bank

After linearization around the reference speed \(v_{ref}\), the continuous-time error model is:

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

Discretized with sampling interval \(dt\) (the project uses \(A_d\approx I+dtA,\;B_d\approx dtB\)).

## 3. LQR Optimization Problem

### 3.1 Objective Function

The inner loop uses a quadratic objective:

\[
\min_{\{u_k\}}\;J=\sum_{k=0}^{\infty}
\left(
x_k^TQx_k + u_k^TRu_k
\right)
\]

where:
- \(Q=\mathrm{diag}(q_{e_\psi}, q_{e_\phi})\): error weight matrix
- \(R=[r_u]\): control-increment weight

### 3.2 Constraints

The fundamental constraint is the system dynamics:

\[
x_{k+1}=A_dx_k+B_du_k
\]

In the engineering implementation, additional physical saturation limits are applied after the control law:
- `phi_cmd` saturation: \([-\phi_{limit},+\phi_{limit}]\)
- `thrust` saturation: \([0,\,T_{max}]\)

### 3.3 Why This Is Convex Optimization

After stacking all time-domain variables, the problem can be written in standard quadratic programming (QP) form:

\[
\min_z\;\frac12 z^THz+f^Tz,\quad \text{s.t.}\;Ez=b
\]

Convexity comes from two facts:
1. The constraint \(Ez=b\) is affine, so the feasible set is a convex set.
2. When \(Q\succeq 0,\;R\succ 0\), the Hessian satisfies \(H\succeq 0\), making the objective convex.

Therefore the problem is a convex optimization problem and its optimal solution is globally optimal.

## 4. Optimal Solution Form and Project Implementation

Without hard inequality constraints, the LQR optimal solution takes the form of a linear state feedback:

\[
u_k=-Kx_k
\]

where \(K\) is computed from the discrete Riccati equation.  
The corresponding workflow in the project:
- Iteratively solve the Riccati matrix in `dlqr(...)`
- Obtain feedback gain `k_lqr`
- Compute `delta_phi_cmd = -(k_lqr @ err)`, then add `phi_ff` to get `phi_cmd`

## 5. Physical Meaning of Parameters and Tuning Guidelines

- Increasing `q_e_psi`: stronger heading-error suppression → more aggressive turns.
- Increasing `q_e_phi`: stronger bank-error convergence → more stable attitude.
- Increasing `r_u`: penalizes control action → smoother response but possibly slower.

Practical tuning recommendations:
1. First ensure stability (increase `r_u` to a moderate value).
2. Gradually increase `q_e_psi` to improve tracking accuracy.
3. Finally fine-tune `q_e_phi` to balance "tracking aggressiveness vs. attitude smoothness".

## 6. Summary

The LQR in this project is a "B747 circular inner-loop error controller":
- Clear control objective: minimize \(e_\psi\) and \(e_\phi\).
- Well-defined mathematical form: quadratic objective + linear constraints → convex optimization.
- Practically implementable: Riccati-based gain computation + saturation protection → runnable closed-loop control.
