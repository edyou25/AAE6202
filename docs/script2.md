**Ruige Yang:**

Thank you, Yufa. Now I will present the main results. 

Under the nominal closed-loop setting, the controller performs very well. The final cross-track error is only 0.218 meters, and the cross-track RMSE after the transient is 5.56 meters. The settling time to within 100 meters of the orbit is about 78.35 seconds. At the same time, the heading RMSE is 2.31 degrees, the maximum bank angle is 27.67 degrees, and the speed RMSE relative to 210 meters per second is only 0.276 meters per second. These results show that the controller removes the initial offset quickly and achieves very small steady-state error. 

From the tracking figures, we can see that the aircraft starts outside the target circle and converges smoothly to the desired orbit. The transient response is well damped, and the steady-state motion stays very close to the reference circle. The heading, bank angle, and speed all remain bounded and physically reasonable during the entire maneuver. This confirms that the control design is both stable and practical. 

We also studied sensitivity and controller trade-offs. For the time-step sensitivity test, the performance changes only mildly across different values of delta t, which means that the baseline time step of 0.05 seconds is already conservative enough for this workflow. For controller tuning, aggressive LQR weights improve tracking performance, but they also increase peak bank demand and saturation risk. Conservative tuning reduces control effort, but weakens correction capability. Therefore, the baseline tuning provides the best balance between responsiveness and actuator margin. 

Another important result is the comparison between RK4 and Euler integration. The recreated benchmark matches the independent numerical check in the report. More importantly, RK4 achieves about 16.46 times lower final total state error than Euler. This strongly supports our choice of RK4 as the default propagator for the nonlinear B747 model. 

For state estimation, we observe that the posterior covariance contracts after each measurement update, which means the estimator is working as expected. However, the posterior accuracy improves only slightly, because the process and measurement uncertainties are of similar scale in the current setting. So at this stage, the estimator mainly serves as a diagnostic tool rather than as part of the feedback control loop. 

In terms of computation, the framework is very lightweight. The mean controller time is 0.0165 milliseconds, and the mean estimator time is 0.0344 milliseconds. Both are far below the 50-millisecond control interval. The timing distributions are also compact, without problematic long tails. This shows that the whole workflow is computationally efficient and easy to run in real time. 

To conclude, the main strengths of our current pipeline are stable circular tracking, low computational cost, and interpretable uncertainty diagnostics. At the same time, there are still several limitations, including planar flight assumptions, simplified actuator and disturbance models, first-order LQR discretization, and the use of true-state feedback instead of estimated-state feedback. Natural future directions include estimated-state closed-loop control, wind and disturbance injection, and comparison with MPC or nonlinear guidance laws. 

That is the end of our presentation. Thank you very much for listening, and we welcome your questions.
