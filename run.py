"""Main entry for Boeing 747 circular-flight simulation with LQR control."""

from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

import numpy as np


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B747 circular-flight simulation")
    parser.add_argument(
        "--show-animation",
        action="store_true",
        default=True,
        help="Play point-cloud animation in an interactive window.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Animation frame rate.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames used for animation playback/export (0 means use all simulation frames).",
    )
    return parser


import matplotlib


import matplotlib.pyplot as plt

from controller import CircleRef, ControlConfig, LQRCircleController
from dynamics import B747Params, b747_dynamics, rk4_step
from estimation import EstimationConfig, GaussianMAPEstimator
from visual import save_flight_animation, show_flight_animation


def parse_args() -> argparse.Namespace:
    return _build_arg_parser().parse_args()


def simulate():
    p = B747Params()
    cfg = ControlConfig(v_ref=210.0, dt=0.05)
    ref = CircleRef(center_x=0.0, center_y=0.0, radius=12_0000.0, ccw=True)

    ctrl = LQRCircleController(cfg, p)

    t_end = 800.0
    n = int(t_end / cfg.dt)
    time = np.arange(n + 1) * cfg.dt

    # Start outside boundary with heading slightly off tangent.
    state = np.array([ref.radius + 2000.0, 0.0, np.deg2rad(96.0), 0.0, cfg.v_ref], dtype=float)
    estimator = GaussianMAPEstimator(x0=state, cfg=EstimationConfig())
    rng = np.random.default_rng(6202)

    hist = np.zeros((n + 1, 5), dtype=float)
    e_ct_hist = np.zeros(n + 1, dtype=float)
    e_psi_hist = np.zeros(n + 1, dtype=float)
    e_phi_hist = np.zeros(n + 1, dtype=float)
    phi_cmd_hist = np.zeros(n + 1, dtype=float)
    delta_phi_cmd_hist = np.zeros(n + 1, dtype=float)
    thrust_hist = np.zeros(n + 1, dtype=float)
    lqr_obj_hist = np.zeros(n + 1, dtype=float)
    solve_time_ms_hist = np.zeros(n + 1, dtype=float)
    dstate_hist = np.zeros((n + 1, 5), dtype=float)
    est_prior_err_hist = np.zeros(n + 1, dtype=float)
    est_post_err_hist = np.zeros(n + 1, dtype=float)
    est_innovation_hist = np.zeros(n + 1, dtype=float)
    est_map_obj_hist = np.zeros(n + 1, dtype=float)
    est_update_ms_hist = np.zeros(n + 1, dtype=float)
    est_trace_prior_hist = np.zeros(n + 1, dtype=float)
    est_trace_post_hist = np.zeros(n + 1, dtype=float)
    est_meas_noise_norm_hist = np.zeros(n + 1, dtype=float)
    est_state_hat_hist = np.zeros((n + 1, 5), dtype=float)

    hist[0] = state
    est_state_hat_hist[0] = state

    for k in range(n):
        solve_t0 = perf_counter()
        control, info = ctrl.compute_control(state, ref)
        solve_time_ms = (perf_counter() - solve_t0) * 1000.0

        delta_phi_cmd = control["phi_cmd"] - info["phi_ff"]
        stage_obj = (
            cfg.q_e_psi * (info["e_psi"] ** 2)
            + cfg.q_e_phi * (info["e_phi"] ** 2)
            + cfg.r_u * (delta_phi_cmd**2)
        )

        dstate = b747_dynamics(state, control, p)
        meas_noise = rng.normal(loc=0.0, scale=np.asarray(estimator.cfg.measurement_std))
        measurement = state + meas_noise
        est_res = estimator.step(control=control, measurement=measurement, dt=cfg.dt, p_dyn=p)

        e_ct_hist[k] = info["e_ct"]
        e_psi_hist[k] = np.rad2deg(info["e_psi"])
        e_phi_hist[k] = np.rad2deg(info["e_phi"])
        phi_cmd_hist[k] = np.rad2deg(control["phi_cmd"])
        delta_phi_cmd_hist[k] = np.rad2deg(delta_phi_cmd)
        thrust_hist[k] = control["thrust"]
        lqr_obj_hist[k] = stage_obj
        solve_time_ms_hist[k] = solve_time_ms
        dstate_hist[k] = dstate
        est_prior_err_hist[k] = float(np.linalg.norm(est_res["x_prior"] - state))
        est_post_err_hist[k] = float(np.linalg.norm(est_res["x_post"] - state))
        est_innovation_hist[k] = float(np.linalg.norm(est_res["innovation"]))
        est_map_obj_hist[k] = float(est_res["map_objective"])
        est_update_ms_hist[k] = float(est_res["update_ms"])
        est_trace_prior_hist[k] = float(np.trace(est_res["p_prior"]))
        est_trace_post_hist[k] = float(np.trace(est_res["p_post"]))
        est_meas_noise_norm_hist[k] = float(np.linalg.norm(meas_noise))
        est_state_hat_hist[k] = est_res["x_post"]

        state = rk4_step(state, control, cfg.dt, p)

        hist[k + 1] = state
    e_ct_hist[-1] = e_ct_hist[-2]
    e_psi_hist[-1] = e_psi_hist[-2]
    e_phi_hist[-1] = e_phi_hist[-2]
    phi_cmd_hist[-1] = phi_cmd_hist[-2]
    delta_phi_cmd_hist[-1] = delta_phi_cmd_hist[-2]
    thrust_hist[-1] = thrust_hist[-2]
    lqr_obj_hist[-1] = lqr_obj_hist[-2]
    solve_time_ms_hist[-1] = solve_time_ms_hist[-2]
    dstate_hist[-1] = dstate_hist[-2]
    est_prior_err_hist[-1] = est_prior_err_hist[-2]
    est_post_err_hist[-1] = est_post_err_hist[-2]
    est_innovation_hist[-1] = est_innovation_hist[-2]
    est_map_obj_hist[-1] = est_map_obj_hist[-2]
    est_update_ms_hist[-1] = est_update_ms_hist[-2]
    est_trace_prior_hist[-1] = est_trace_prior_hist[-2]
    est_trace_post_hist[-1] = est_trace_post_hist[-2]
    est_meas_noise_norm_hist[-1] = est_meas_noise_norm_hist[-2]
    est_state_hat_hist[-1] = est_state_hat_hist[-2]

    telemetry = {
        "e_ct": e_ct_hist,
        "e_psi_deg": e_psi_hist,
        "e_phi_deg": e_phi_hist,
        "phi_cmd_deg": phi_cmd_hist,
        "delta_phi_cmd_deg": delta_phi_cmd_hist,
        "thrust": thrust_hist,
        "lqr_obj": lqr_obj_hist,
        "solve_time_ms": solve_time_ms_hist,
        "dstate": dstate_hist,
        "rk_method": "RK4",
        "rk_dt": cfg.dt,
        "q_e_psi": cfg.q_e_psi,
        "q_e_phi": cfg.q_e_phi,
        "r_u": cfg.r_u,
        "est_prior_err_norm": est_prior_err_hist,
        "est_post_err_norm": est_post_err_hist,
        "est_innovation_norm": est_innovation_hist,
        "est_map_obj": est_map_obj_hist,
        "est_update_ms": est_update_ms_hist,
        "est_trace_prior": est_trace_prior_hist,
        "est_trace_post": est_trace_post_hist,
        "est_meas_noise_norm": est_meas_noise_norm_hist,
        "est_measurement_std": np.asarray(estimator.cfg.measurement_std, dtype=float),
        "est_process_std": np.asarray(estimator.cfg.process_std, dtype=float),
        "est_state_hat": est_state_hat_hist,
        "estimation_mode": "Bayes/MAP (Gaussian prior-likelihood-posterior)",
    }

    return time, hist, e_ct_hist, e_psi_hist, phi_cmd_hist, ref, telemetry


def plot_results(time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref):
    x = hist[:, 0]
    y = hist[:, 1]
    phi_deg = np.rad2deg(hist[:, 3])
    v = hist[:, 4]

    th = np.linspace(0.0, 2.0 * np.pi, 600)
    x_ref = ref.center_x + ref.radius * np.cos(th)
    y_ref = ref.center_y + ref.radius * np.sin(th)

    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x_ref, y_ref, "k--", linewidth=1.5, label="Reference Circle")
    ax1.plot(x, y, "b", linewidth=1.5, label="B747 Trajectory")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Circular Flight Tracking")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, e_ct, "r")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("signed radius error (m)")
    ax2.set_title("Cross-Track Error (positive=inside)")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time, phi_deg, label="phi")
    ax3.plot(time, phi_cmd_deg, "--", label="phi_cmd")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("bank angle (deg)")
    ax3.set_title("Bank Response")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(time, v, label="v")
    ax4.plot(time, e_psi_deg, label="heading err (deg)")
    ax4.set_xlabel("time (s)")
    ax4.set_title("Speed and Heading Error")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.tight_layout()
    os.makedirs("data", exist_ok=True)
    fig.savefig("data/circle_flight_result.png", dpi=150)


def main():
    args = parse_args()

    time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref, telemetry = simulate()
    print(f"Final position: x={hist[-1,0]:.1f} m, y={hist[-1,1]:.1f} m")
    print(f"Final speed: {hist[-1,4]:.2f} m/s")
    print(f"Final signed cross-track error: {e_ct[-1]:.2f} m")
    print(f"Final |cross-track error|: {abs(e_ct[-1]):.2f} m")

    # plot_results(time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref)
    # print("Saved figure: data/circle_flight_result.png")

    print("Opening animation window...")
    anim = show_flight_animation(
        time=time,
        hist=hist,
        ref=ref,
        fps=args.fps,
        max_frames=args.max_frames,
        telemetry=telemetry,
    )


if __name__ == "__main__":
    main()
