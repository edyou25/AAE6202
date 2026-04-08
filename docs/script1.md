**Yufa You:**

Good evening everyone. We are Yufa You and Ruige Yang. Today we are going to present our AAE6202 course project, Boeing 747 Trajectory Tracking: Optimization, ODE Solving, and State Estimation. 

Our project builds an integrated 2D circular-tracking workflow for a Boeing 747. The main goal is to make the aircraft converge from an off-circle initial condition to a counterclockwise reference orbit, while connecting three key parts of this course: control design, numerical integration, and state estimation. 

The overall pipeline combines three components. First, we use LQR for inner-loop regulation. Second, we use RK4 to propagate the nonlinear aircraft dynamics. Third, we use Gaussian Bayes or MAP estimation to analyze the noisy measurements. Under the baseline setting, the aircraft converges smoothly to the reference circle with near-zero steady-state cross-track error and tight speed regulation. 

Let me briefly introduce the setup. Our state includes x position, y position, heading angle, bank angle, and speed. The control input contains the commanded bank angle and thrust. In the baseline case, the reference radius is 120 kilometers, the reference speed is 210 meters per second, the simulation time step is 0.05 seconds, and the total simulation time is 800 seconds. We also impose a bank-angle limit of plus or minus 35 degrees. 

This workflow is meaningful because it connects optimization, differential-equation solving, and uncertainty-aware estimation into one aerospace example. LQR provides predictable tuning and stable heading-bank regulation. RK4 gives better propagation accuracy than low-order integration. Bayesian estimation helps us evaluate posterior error, innovation, and covariance contraction under noisy measurements. 

Now I will explain the methodology. The integrated architecture starts from the circle reference, which defines the center, radius, and direction. The controller uses the feedback state to generate control commands. These commands are applied to the nonlinear B747 planar dynamics, and then the state is propagated by the RK4 integrator. At the same time, the true state passes through a measurement model with Gaussian sensor noise, and the noisy measurements are sent to the Bayes or MAP estimator for prediction and update. Finally, telemetry records tracking error, timing, and covariance information for later analysis. 

The guidance and control logic has three layers. First, the geometry block computes the current radius, the tangent heading, and the signed cross-track error. Then, the desired heading is corrected by a bounded nonlinear term based on the cross-track error, so the aircraft can return to the orbit smoothly. Next, we compute the feedforward bank angle from circular-motion curvature, and then an LQR controller regulates the heading error and bank-angle error. In parallel, a separate thrust channel keeps the speed close to the reference value. So the controller combines geometry guidance, LQR stabilization, and speed hold. 

The simulation loop is also very clear. At each time step, we first compute the geometry variables and the cross-track error. Then we generate the desired heading, compute the feedforward bank angle, form the LQR error state, and obtain the bank command. After that, we compute thrust, propagate the nonlinear state using RK4, add measurement noise, and update the estimated state and covariance using the Gaussian MAP estimator. This gives us a complete closed-loop simulation framework. 

That is the methodology part. Now I will hand over to Ruige Yang for the results.
