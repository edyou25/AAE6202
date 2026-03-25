"""Generate report-ready figures for the B747 simulation project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from controller import ControlConfig
from report_assets import (
    LATEX_ASSETS_DIR,
    LEGACY_REPORT_DIR,
    copy_asset,
    paths_match,
    save_figure,
)
from run import plot_results, simulate


def _mirror_paths(output_dir: Path, filename: str) -> tuple[Path, ...]:
    if paths_match(output_dir, LATEX_ASSETS_DIR):
        return (LEGACY_REPORT_DIR / filename,)
    return ()


def _save_report_figure(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    return save_figure(
        fig,
        output_dir / filename,
        mirror_paths=_mirror_paths(output_dir, filename),
        dpi=150,
    )


def save_report_figures(
    output_dir: str | Path = LATEX_ASSETS_DIR,
    t_end: float = 120.0,
    dt: float = 0.05,
    seed: int = 6202,
) -> dict[str, str]:
    """Run a short simulation and save report-friendly summary figures."""
    output_dir = Path(output_dir)
    cfg = ControlConfig(v_ref=210.0, dt=dt)
    time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref, telemetry = simulate(
        t_end=t_end,
        cfg=cfg,
        seed=seed,
    )

    paths: dict[str, str] = {}
    paths["overview"] = plot_results(
        time,
        hist,
        e_ct,
        e_psi_deg,
        phi_cmd_deg,
        ref,
        out_path=output_dir / "simulation_overview.png",
    )
    if paths_match(output_dir, LATEX_ASSETS_DIR):
        copy_asset(paths["overview"], LEGACY_REPORT_DIR / "simulation_overview.png")

    phi_deg = np.rad2deg(hist[:, 3])
    dstate = telemetry["dstate"]
    est_state_hat = telemetry["est_state_hat"]

    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    axes1[0, 0].plot(time, e_ct, color="tab:red")
    axes1[0, 0].set_title("Cross-Track Error")
    axes1[0, 0].set_xlabel("time (s)")
    axes1[0, 0].set_ylabel("error (m)")
    axes1[0, 0].grid(True, alpha=0.3)

    axes1[0, 1].plot(time, e_psi_deg, label="e_psi (deg)")
    axes1[0, 1].plot(time, telemetry["e_phi_deg"], label="e_phi (deg)")
    axes1[0, 1].set_title("LQR Tracking Errors")
    axes1[0, 1].set_xlabel("time (s)")
    axes1[0, 1].grid(True, alpha=0.3)
    axes1[0, 1].legend()

    axes1[1, 0].plot(time, phi_deg, label="phi (deg)")
    axes1[1, 0].plot(time, phi_cmd_deg, "--", label="phi_cmd (deg)")
    axes1[1, 0].set_title("Bank Response")
    axes1[1, 0].set_xlabel("time (s)")
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].legend()

    axes1[1, 1].plot(time, telemetry["thrust"], label="thrust (N)")
    axes1[1, 1].plot(time, telemetry["lqr_obj"], label="stage objective")
    axes1[1, 1].set_title("Control Effort and Objective")
    axes1[1, 1].set_xlabel("time (s)")
    axes1[1, 1].grid(True, alpha=0.3)
    axes1[1, 1].legend()
    paths["part1_control"] = _save_report_figure(fig1, output_dir, "part1_control.png")

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    th = np.linspace(0.0, 2.0 * np.pi, 600)
    x_ref = ref.center_x + ref.radius * np.cos(th)
    y_ref = ref.center_y + ref.radius * np.sin(th)
    axes2[0, 0].plot(x_ref, y_ref, "k--", linewidth=1.2, label="reference")
    axes2[0, 0].plot(hist[:, 0], hist[:, 1], label="trajectory")
    axes2[0, 0].set_aspect("equal", adjustable="box")
    axes2[0, 0].set_title("Trajectory")
    axes2[0, 0].set_xlabel("x (m)")
    axes2[0, 0].set_ylabel("y (m)")
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].legend()

    axes2[0, 1].plot(time, dstate[:, 0], label="x_dot")
    axes2[0, 1].plot(time, dstate[:, 1], label="y_dot")
    axes2[0, 1].set_title("Translational Rates")
    axes2[0, 1].set_xlabel("time (s)")
    axes2[0, 1].grid(True, alpha=0.3)
    axes2[0, 1].legend()

    axes2[1, 0].plot(time, np.rad2deg(dstate[:, 2]), label="psi_dot (deg/s)")
    axes2[1, 0].plot(time, np.rad2deg(dstate[:, 3]), label="phi_dot (deg/s)")
    axes2[1, 0].set_title("Angular Rates")
    axes2[1, 0].set_xlabel("time (s)")
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].legend()

    axes2[1, 1].plot(time, hist[:, 4], label="v (m/s)")
    axes2[1, 1].plot(time, dstate[:, 4], label="v_dot (m/s^2)")
    axes2[1, 1].set_title("Speed Dynamics")
    axes2[1, 1].set_xlabel("time (s)")
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].legend()
    paths["part2_dynamics"] = _save_report_figure(fig2, output_dir, "part2_dynamics.png")

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    axes3[0, 0].plot(time, telemetry["est_prior_err_norm"], label="prior error")
    axes3[0, 0].plot(time, telemetry["est_post_err_norm"], label="post error")
    axes3[0, 0].set_title("Estimation Error Norms")
    axes3[0, 0].set_xlabel("time (s)")
    axes3[0, 0].grid(True, alpha=0.3)
    axes3[0, 0].legend()

    axes3[0, 1].plot(time, telemetry["est_innovation_norm"], label="innovation")
    axes3[0, 1].plot(time, telemetry["est_meas_noise_norm"], label="measurement noise")
    axes3[0, 1].set_title("Innovation vs Noise")
    axes3[0, 1].set_xlabel("time (s)")
    axes3[0, 1].grid(True, alpha=0.3)
    axes3[0, 1].legend()

    axes3[1, 0].plot(time, telemetry["est_trace_prior"], label="tr(P-)")
    axes3[1, 0].plot(time, telemetry["est_trace_post"], label="tr(P+)")
    axes3[1, 0].set_title("Covariance Trace")
    axes3[1, 0].set_xlabel("time (s)")
    axes3[1, 0].grid(True, alpha=0.3)
    axes3[1, 0].legend()

    axes3[1, 1].plot(hist[:, 0], hist[:, 1], label="true state")
    axes3[1, 1].plot(est_state_hat[:, 0], est_state_hat[:, 1], "--", label="MAP estimate")
    axes3[1, 1].set_title("True vs Estimated Position")
    axes3[1, 1].set_xlabel("x (m)")
    axes3[1, 1].set_ylabel("y (m)")
    axes3[1, 1].grid(True, alpha=0.3)
    axes3[1, 1].legend()
    paths["part3_estimation"] = _save_report_figure(fig3, output_dir, "part3_estimation.png")

    return paths


def main() -> None:
    paths = save_report_figures()
    print("Generated report figures:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
