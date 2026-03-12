# Unit Tests

- Adds reproducible unit tests covering the project's control, dynamics, estimation, visualization, and integration pipeline.
- Generates charts suitable for direct inclusion in the course report via `matplotlib`.

## How to Run

- Run all unit tests: `python -m unittest discover -s tests -v`
- Generate report figures: `python reporting.py`

## Test Coverage

- `tests/test_controller.py`: `wrap_pi`, `dlqr`, circular-trajectory control law, and saturation limits.
- `tests/test_dynamics.py`: dynamics clipping logic, straight-line steady flight, and turn response with held bank angle.
- `tests/test_estimation.py`: covariance construction, perfect-measurement update, and posterior convergence under noisy measurements.
- `tests/test_visual.py`: body point-cloud scaling, coordinate transformation, and animation frame-scheduling.
- `tests/test_run.py`: output dimensions of a short simulation run and overall result plot.
- `tests/test_reporting.py`: batch generation of report figures.

## Results

- `data/report/simulation_overview.png`: overall trajectory, lateral error, bank response, speed / heading error.
- `data/report/part1_control.png`: LQR control error, bank tracking, thrust, and cost function.
- `data/report/part2_dynamics.png`: trajectory, translational velocity, angular rate, and speed dynamics.
- `data/report/part3_estimation.png`: estimation error, innovation, covariance trace, and true / estimated position comparison.

![Simulation Overview](../data/report/simulation_overview.png)

![Part 1 Control](../data/report/part1_control.png)

![Part 2 Dynamics](../data/report/part2_dynamics.png)

![Part 3 Estimation](../data/report/part3_estimation.png)
