# Part 3: B747 Kalman Filter State Estimation — Bayes/MAP

## 1. Background

In real flight control, the controller receives **noisy measurements**, not perfectly accurate true values.  
This project therefore includes an estimation module to demonstrate:
- How to predict a prior state using the dynamics model
- How to perform a posterior update by fusing measurements
- How to feed estimation metrics into the visualization panel

Implementation code: [estimation.py](../estimation.py)

## 2. Bayesian Estimation

### 2.1 Prior (Prior = Prediction)

The prior at time step \(k\) is given by the dynamics prediction:

\[
x_k^- = f(x_{k-1}^+, u_{k-1})
\]

The predictor in this project uses the same RK4 process model as the simulator.

### 2.2 Likelihood (Likelihood = Measurement)

The measurement model is:

\[
z_k = x_k + v_k,
\quad v_k\sim \mathcal{N}(0,R)
\]

i.e., the sensor measurement equals the true state plus Gaussian noise.

### 2.3 Posterior (Posterior = Update)

Combining prior and likelihood yields the posterior estimate:

\[
x_k^+ = x_k^- + K_k(z_k-x_k^-)
\]

\[
K_k = P_k^- (P_k^- + R)^{-1}
\]

and the covariance is updated:

\[
P_k^+ = (I-K_k)P_k^-
\]

## 3. MAP Optimization Interpretation

Under Gaussian assumptions, maximizing the posterior is equivalent to minimizing the following objective:

\[
\min_x \frac12(x-x_k^-)^T(P_k^-)^{-1}(x-x_k^-)
+\frac12(z_k-x)^TR^{-1}(z_k-x)
\]

This is the MAP (Maximum A Posteriori) interpretation of Part 3 in this project:
- First term: prior constraint (trust the prediction)
- Second term: measurement constraint (trust the sensor)
- Optimal point: weighted compromise between prior and measurement

## 4. Estimation Metrics Recorded in This Project

In the simulation loop of `run.py`, the following metrics are recorded and passed to `visual.py`:
- `est_innovation_norm`: innovation norm \(\|z_k-x_k^-\|\)
- `est_prior_err_norm`: prior error norm \(\|x_k^- - x_k\|\)
- `est_post_err_norm`: posterior error norm \(\|x_k^+ - x_k\|\)
- `est_map_obj`: MAP objective value
- `est_update_ms`: estimator update time
- `est_trace_prior` / `est_trace_post`: covariance trace (uncertainty scale)
- `est_state_hat`: posterior state estimate

## 5. Visualization

The visualization panel displays:
- Semantic labels for `prior / likelihood / posterior`
- Innovation, prior error, and posterior error
- MAP objective value and estimation computation time
- \(\mathrm{tr}(P^-)\) and \(\mathrm{tr}(P^+)\)
- \(\hat x, \hat y, \hat\psi, \hat\phi, \hat v\)

This allows direct observation of the "predict–measure–update" cycle at every control step.

## 6. Summary

The role of Part 3 is not to replace the controller, but to complete the "state awareness layer" demonstration:
- Part 1 handles the control law (LQR)
- Part 2 handles dynamics propagation (RK4)
- Part 3 handles state estimation (Bayes/MAP)

Together, the three parts make the simulation more closely resemble the data flow of a real flight control system.
