"""Generate extended experiment figures and system diagrams for the LaTeX report."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from controller import CircleRef, ControlConfig
from dynamics import B747Params, b747_dynamics, rk4_step
from estimation import EstimationConfig
from report_assets import (
    LATEX_ASSETS_DIR,
    LEGACY_REPORT_EXT_DIR,
    copy_asset,
    ensure_dir,
    paths_match,
    save_figure,
    write_text_asset,
)
from reporting import save_report_figures
from run import simulate
from visual import plot_aircraft_snapshot


REPORT_DIR = LATEX_ASSETS_DIR
METRICS_DIR = LEGACY_REPORT_EXT_DIR
DOT_EXE = Path(shutil.which("dot") or r"C:\Program Files\Graphviz\bin\dot.exe")


@dataclass(frozen=True)
class CaseMetrics:
    final_abs_e_ct: float
    rmse_e_ct: float
    heading_rmse_deg: float
    speed_rmse: float
    max_abs_phi_deg: float
    saturation_ratio: float
    mean_prior_err: float
    mean_post_err: float
    posterior_improve_ratio: float


def _save(fig: plt.Figure, path: Path) -> str:
    mirror_paths: tuple[Path, ...] = ()
    if paths_match(path.parent, REPORT_DIR):
        mirror_paths = (LEGACY_REPORT_EXT_DIR / path.name,)
    return save_figure(fig, path, mirror_paths=mirror_paths, dpi=160)


def _write_report_text(path: Path, content: str) -> str:
    mirror_paths: tuple[Path, ...] = ()
    if paths_match(path.parent, REPORT_DIR):
        mirror_paths = (LEGACY_REPORT_EXT_DIR / path.name,)
    return write_text_asset(content, path, mirror_paths=mirror_paths)


def _render_graphviz(dot_path: Path, png_path: Path) -> str:
    if not DOT_EXE.exists():
        raise FileNotFoundError(f"Graphviz dot executable not found: {DOT_EXE}")
    subprocess.run(
        [str(DOT_EXE), "-Tpng", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    mirror_paths: tuple[Path, ...] = ()
    if paths_match(png_path.parent, REPORT_DIR):
        mirror_paths = (LEGACY_REPORT_EXT_DIR / png_path.name,)
    return copy_asset(png_path, png_path, mirror_paths=mirror_paths)


def _settling_time(time: np.ndarray, signal: np.ndarray, threshold: float) -> float:
    within = np.abs(signal) < threshold
    for index in range(len(within)):
        if np.all(within[index:]):
            return float(time[index])
    return float("nan")


def _compute_metrics(
    time: np.ndarray,
    hist: np.ndarray,
    e_ct: np.ndarray,
    e_psi_deg: np.ndarray,
    phi_cmd_deg: np.ndarray,
    telemetry: dict,
) -> CaseMetrics:
    phi_deg = np.rad2deg(hist[:, 3])
    v = hist[:, 4]
    prior = telemetry["est_prior_err_norm"]
    post = telemetry["est_post_err_norm"]
    return CaseMetrics(
        final_abs_e_ct=float(abs(e_ct[-1])),
        rmse_e_ct=float(np.sqrt(np.mean(e_ct**2))),
        heading_rmse_deg=float(np.sqrt(np.mean(e_psi_deg**2))),
        speed_rmse=float(np.sqrt(np.mean((v - 210.0) ** 2))),
        max_abs_phi_deg=float(np.max(np.abs(phi_deg))),
        saturation_ratio=float(np.mean(np.isclose(np.abs(phi_cmd_deg), 35.0, atol=1e-6))),
        mean_prior_err=float(np.mean(prior)),
        mean_post_err=float(np.mean(post)),
        posterior_improve_ratio=float(np.mean(post < prior)),
    )


def _run_case(
    *,
    t_end: float = 400.0,
    cfg: ControlConfig | None = None,
    ref: CircleRef | None = None,
    estimator_cfg: EstimationConfig | None = None,
    seed: int = 6202,
):
    return simulate(
        t_end=t_end,
        cfg=cfg,
        ref=ref,
        estimator_cfg=estimator_cfg,
        seed=seed,
    )


def _generate_system_graphs() -> tuple[dict[str, str], dict[str, str]]:
    ensure_dir(REPORT_DIR)
    paths: dict[str, str] = {}
    dot_paths: dict[str, str] = {}
    system_dot = REPORT_DIR / "system_architecture.dot"
    telemetry_dot = REPORT_DIR / "telemetry_pipeline.dot"
    module_dot = REPORT_DIR / "module_dependency.dot"
    control_dot = REPORT_DIR / "controller_logic.dot"

    _write_report_text(
        system_dot,
        """
digraph system_architecture {
  rankdir=LR;
  graph [pad="0.2", nodesep="0.45", ranksep="0.65"];
  node [shape=box, style="rounded,filled", color="#24476b", fillcolor="#eaf2fb", fontname="Arial"];
  edge [color="#355c7d", arrowsize=0.8, penwidth=1.4, fontname="Arial"];

  ref [label="Circle Reference\\n(center, radius, direction)"];
  state [label="True State\\n[x, y, psi, phi, v]"];
  ctrl [label="Controller\\nouter-loop guidance\\n+ inner-loop LQR\\n+ speed hold"];
  dyn [label="Nonlinear Dynamics\\nB747 planar model"];
  integ [label="RK4 Integrator\\nstate propagation"];
  meas [label="Measurement Model\\nGaussian sensor noise"];
  est [label="Bayes/MAP Estimator\\nprediction + update"];
  logs [label="Telemetry Logger\\nerrors, timing, covariance"];
  vis [label="Plots / Animation / Report"];

  ref -> ctrl;
  state -> ctrl [label="feedback state"];
  ctrl -> dyn [label="phi_cmd, thrust"];
  dyn -> integ;
  integ -> state;
  state -> meas;
  ctrl -> est [label="control input"];
  meas -> est [label="measurement z"];
  est -> logs;
  state -> logs;
  ctrl -> logs;
  logs -> vis;
}
""".strip(),
    )

    _write_report_text(
        telemetry_dot,
        """
digraph telemetry_pipeline {
  rankdir=TB;
  graph [pad="0.2", nodesep="0.35", ranksep="0.55"];
  node [shape=box, style="rounded,filled", color="#5b3c88", fillcolor="#f4effc", fontname="Arial"];
  edge [color="#6c5b7b", arrowsize=0.8, penwidth=1.4, fontname="Arial"];

  subgraph cluster_runtime {
    label="Runtime Loop";
    color="#b7a7d6";
    c1 [label="control solve time"];
    c2 [label="tracking errors\\n(e_ct, e_psi, e_phi)"];
    c3 [label="state derivative\\n(x_dot, y_dot, psi_dot, phi_dot, v_dot)"];
    c4 [label="estimation stats\\ninnovation, tr(P), MAP objective"];
  }

  subgraph cluster_post {
    label="Post-processing";
    color="#d2c8e8";
    p1 [label="summary metrics"];
    p2 [label="report figures"];
    p3 [label="tables in LaTeX"];
    p4 [label="animation and diagnostics"];
  }

  c1 -> p1;
  c2 -> p1;
  c2 -> p2;
  c3 -> p2;
  c4 -> p1;
  c4 -> p2;
  p1 -> p3;
  p2 -> p3;
  p2 -> p4;
}
""".strip(),
    )

    _write_report_text(
        module_dot,
        """
digraph module_dependency {
  rankdir=LR;
  graph [pad="0.2", nodesep="0.45", ranksep="0.65"];
  node [shape=box, style="rounded,filled", color="#3f5d4f", fillcolor="#eef7f0", fontname="Arial"];
  edge [color="#4f6f52", arrowsize=0.8, penwidth=1.4, fontname="Arial"];

  run [label="run.py"];
  controller [label="controller.py"];
  dynamics [label="dynamics.py"];
  estimation [label="estimation.py"];
  visual [label="visual.py"];
  reporting [label="reporting.py"];
  ext [label="extended_reporting.py"];
  tests [label="tests/*"];

  run -> controller;
  run -> dynamics;
  run -> estimation;
  run -> visual;
  reporting -> run;
  ext -> run;
  ext -> controller;
  ext -> dynamics;
  ext -> estimation;
  tests -> controller;
  tests -> dynamics;
  tests -> estimation;
  tests -> visual;
  tests -> reporting;
  tests -> run;
  controller -> dynamics;
  estimation -> dynamics;
}
""".strip(),
    )

    _write_report_text(
        control_dot,
        """
digraph controller_logic {
  rankdir=TB;
  graph [pad="0.2", nodesep="0.35", ranksep="0.55"];
  node [shape=box, style="rounded,filled", color="#8a4f1d", fillcolor="#fff1e6", fontname="Arial"];
  edge [color="#a05a2c", arrowsize=0.8, penwidth=1.4, fontname="Arial"];

  s0 [label="Current state\\n(x, y, psi, phi, v)"];
  s1 [label="Geometry block\\nradius_now, tangent, e_ct"];
  s2 [label="Heading correction\\natan(k_ct * -e_ct)\\n+ clip"];
  s3 [label="Desired heading\\npsi_des"];
  s4 [label="Feedforward bank\\nphi_ff = atan(v_ref^2 / gR)"];
  s5 [label="Error states\\ne_psi, e_phi"];
  s6 [label="Discrete LQR\\nDelta phi_cmd = -K[e_psi,e_phi]^T"];
  s7 [label="Bank command saturation"];
  s8 [label="Speed-hold thrust\\nD_guess + k_v(v_ref-v)"];
  s9 [label="Control output\\n(phi_cmd, thrust)"];

  s0 -> s1 -> s2 -> s3 -> s5;
  s0 -> s4 -> s5;
  s5 -> s6 -> s7 -> s9;
  s0 -> s8 -> s9;
}
""".strip(),
    )

    for stem, dot_path, png_name in (
        ("system_architecture", system_dot, "system_architecture.png"),
        ("telemetry_pipeline", telemetry_dot, "telemetry_pipeline.png"),
        ("module_dependency", module_dot, "module_dependency.png"),
        ("controller_logic", control_dot, "controller_logic.png"),
    ):
        png_path = REPORT_DIR / png_name
        _render_graphviz(dot_path, png_path)
        paths[stem] = str(png_path)
        dot_paths[stem] = str(dot_path)

    return paths, dot_paths


def _generate_transient_and_estimation_figures(base_results) -> dict[str, str]:
    time, hist, e_ct, e_psi_deg, phi_cmd_deg, _, telemetry = base_results
    paths: dict[str, str] = {}
    mask = time <= 120.0
    t = time[mask]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes[0, 0].plot(t, e_ct[mask], color="tab:red")
    axes[0, 0].set_title("Cross-track error (0-120 s)")
    axes[0, 0].set_ylabel("m")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, e_psi_deg[mask], label="e_psi")
    axes[0, 1].plot(t, telemetry["e_phi_deg"][mask], label="e_phi")
    axes[0, 1].set_title("Angular tracking errors")
    axes[0, 1].set_ylabel("deg")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(t, np.rad2deg(hist[mask, 3]), label="phi")
    axes[1, 0].plot(t, phi_cmd_deg[mask], "--", label="phi_cmd")
    axes[1, 0].set_title("Bank transient")
    axes[1, 0].set_ylabel("deg")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(t, hist[mask, 4], label="v")
    axes[1, 1].plot(t, np.full_like(t, 210.0), "--", label="v_ref")
    axes[1, 1].set_title("Speed regulation")
    axes[1, 1].set_ylabel("m/s")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(t, telemetry["thrust"][mask])
    axes[2, 0].set_title("Thrust command")
    axes[2, 0].set_ylabel("N")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(t, telemetry["lqr_obj"][mask])
    axes[2, 1].set_title("LQR stage objective")
    axes[2, 1].grid(True, alpha=0.3)

    axes[3, 0].plot(t, telemetry["solve_time_ms"][mask], label="controller")
    axes[3, 0].plot(t, telemetry["est_update_ms"][mask], label="estimator")
    axes[3, 0].set_title("Per-step computation time")
    axes[3, 0].set_xlabel("time (s)")
    axes[3, 0].set_ylabel("ms")
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].legend()

    axes[3, 1].plot(t, telemetry["est_innovation_norm"][mask], label="innovation")
    axes[3, 1].plot(t, telemetry["est_meas_noise_norm"][mask], label="noise norm")
    axes[3, 1].set_title("Innovation versus measurement noise")
    axes[3, 1].set_xlabel("time (s)")
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend()
    paths["transient_detail"] = _save(fig, REPORT_DIR / "transient_detail.png")

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    labels = ["x (m)", "y (m)", "psi (deg)", "phi (deg)", "v (m/s)"]
    true_series = [
        hist[:, 0],
        hist[:, 1],
        np.rad2deg(hist[:, 2]),
        np.rad2deg(hist[:, 3]),
        hist[:, 4],
    ]
    est_hist = telemetry["est_state_hat"]
    est_series = [
        est_hist[:, 0],
        est_hist[:, 1],
        np.rad2deg(est_hist[:, 2]),
        np.rad2deg(est_hist[:, 3]),
        est_hist[:, 4],
    ]
    for ax, label, true_y, est_y in zip(axes, labels, true_series, est_series):
        ax.plot(time, true_y, label="true", linewidth=1.5)
        ax.plot(time, est_y, "--", label="estimate", linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("time (s)")
    paths["state_estimate_comparison"] = _save(fig, REPORT_DIR / "state_estimate_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter = axes[0].scatter(e_psi_deg, telemetry["e_phi_deg"], c=time, s=6, cmap="viridis")
    axes[0].set_xlabel(r"$e_\psi$ (deg)")
    axes[0].set_ylabel(r"$e_\phi$ (deg)")
    axes[0].set_title("Error phase portrait")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=axes[0], label="time (s)")

    axes[1].hist(phi_cmd_deg, bins=40, alpha=0.7, label=r"$\phi_{cmd}$")
    axes[1].hist(np.rad2deg(hist[:, 3]), bins=40, alpha=0.7, label=r"$\phi$")
    axes[1].set_title("Bank-angle distributions")
    axes[1].set_xlabel("deg")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    paths["error_phase_and_hist"] = _save(fig, REPORT_DIR / "error_phase_and_hist.png")
    return paths


def _generate_sweep_figures() -> tuple[dict[str, str], dict]:
    paths: dict[str, str] = {}
    metrics_dump: dict[str, dict] = {}

    # Time-step sweep
    dt_values = [0.02, 0.05, 0.10, 0.20]
    dt_metrics = []
    for dt in dt_values:
        cfg = ControlConfig(v_ref=210.0, dt=dt)
        result = _run_case(t_end=300.0, cfg=cfg, seed=6202)
        time, hist, e_ct, e_psi_deg, phi_cmd_deg, _, telemetry = result
        dt_metrics.append(_compute_metrics(time, hist, e_ct, e_psi_deg, phi_cmd_deg, telemetry))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].plot(dt_values, [m.rmse_e_ct for m in dt_metrics], marker="o")
    axes[0].set_title("RMSE of cross-track error")
    axes[0].set_ylabel("m")
    axes[1].plot(dt_values, [m.heading_rmse_deg for m in dt_metrics], marker="o")
    axes[1].set_title("Heading-error RMSE")
    axes[1].set_ylabel("deg")
    axes[2].plot(dt_values, [m.speed_rmse for m in dt_metrics], marker="o")
    axes[2].set_title("Speed RMSE")
    axes[2].set_ylabel("m/s")
    axes[2].set_xlabel("time step (s)")
    axes[3].plot(dt_values, [m.max_abs_phi_deg for m in dt_metrics], marker="o")
    axes[3].set_title("Maximum bank angle")
    axes[3].set_ylabel("deg")
    axes[3].set_xlabel("time step (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time step (s)")
    paths["dt_sensitivity"] = _save(fig, REPORT_DIR / "dt_sensitivity.png")
    metrics_dump["dt_sweep"] = {
        str(dt): dt_metrics[index].__dict__ for index, dt in enumerate(dt_values)
    }

    # Radius sweep
    radii = [80_000.0, 120_000.0, 160_000.0, 200_000.0]
    radius_metrics = []
    settle_100 = []
    ff_bank = []
    for radius in radii:
        ref = CircleRef(0.0, 0.0, radius, True)
        result = _run_case(t_end=350.0, ref=ref, seed=6202)
        time, hist, e_ct, e_psi_deg, phi_cmd_deg, _, telemetry = result
        radius_metrics.append(_compute_metrics(time, hist, e_ct, e_psi_deg, phi_cmd_deg, telemetry))
        settle_100.append(_settling_time(time, e_ct, 100.0))
        ff_bank.append(float(np.rad2deg(np.arctan((210.0**2) / (9.81 * radius)))))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].plot(radii, [m.rmse_e_ct for m in radius_metrics], marker="o")
    axes[0].set_title("Cross-track RMSE")
    axes[0].set_ylabel("m")
    axes[1].plot(radii, settle_100, marker="o")
    axes[1].set_title("Settling time to |e_ct| < 100 m")
    axes[1].set_ylabel("s")
    axes[2].plot(radii, ff_bank, marker="o")
    axes[2].set_title("Feedforward bank angle")
    axes[2].set_ylabel("deg")
    axes[3].plot(radii, [m.max_abs_phi_deg for m in radius_metrics], marker="o")
    axes[3].set_title("Observed max |phi|")
    axes[3].set_ylabel("deg")
    for ax in axes:
        ax.set_xlabel("radius (m)")
        ax.grid(True, alpha=0.3)
    paths["radius_sensitivity"] = _save(fig, REPORT_DIR / "radius_sensitivity.png")
    metrics_dump["radius_sweep"] = {
        str(int(radius)): radius_metrics[index].__dict__ for index, radius in enumerate(radii)
    }

    # Measurement noise sweep
    noise_scales = [0.5, 1.0, 2.0, 4.0]
    noise_metrics = []
    innovation_means = []
    trace_ratios = []
    for scale in noise_scales:
        base = EstimationConfig()
        std = tuple(float(scale * x) for x in base.measurement_std)
        estimator_cfg = EstimationConfig(
            process_std=base.process_std,
            measurement_std=std,
            init_std=base.init_std,
        )
        result = _run_case(t_end=250.0, estimator_cfg=estimator_cfg, seed=6202)
        time, hist, e_ct, e_psi_deg, phi_cmd_deg, _, telemetry = result
        noise_metrics.append(_compute_metrics(time, hist, e_ct, e_psi_deg, phi_cmd_deg, telemetry))
        innovation_means.append(float(np.mean(telemetry["est_innovation_norm"])))
        trace_ratios.append(float(np.mean(telemetry["est_trace_post"] / telemetry["est_trace_prior"])))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].plot(noise_scales, innovation_means, marker="o")
    axes[0].set_title("Mean innovation norm")
    axes[1].plot(noise_scales, [m.posterior_improve_ratio for m in noise_metrics], marker="o")
    axes[1].set_title("Posterior better than prior")
    axes[1].set_ylabel("ratio")
    axes[2].plot(noise_scales, trace_ratios, marker="o")
    axes[2].set_title(r"Mean $\mathrm{tr}(P^+)/\mathrm{tr}(P^-)$")
    for ax in axes:
        ax.set_xlabel("measurement noise scale")
        ax.grid(True, alpha=0.3)
    paths["noise_sensitivity"] = _save(fig, REPORT_DIR / "noise_sensitivity.png")
    metrics_dump["noise_sweep"] = {
        str(scale): noise_metrics[index].__dict__ for index, scale in enumerate(noise_scales)
    }

    # LQR weight sweep
    variants = [
        ("conservative", ControlConfig(q_e_psi=20.0, q_e_phi=8.0, r_u=2.0)),
        ("baseline", ControlConfig(q_e_psi=30.0, q_e_phi=10.0, r_u=1.0)),
        ("aggressive", ControlConfig(q_e_psi=60.0, q_e_phi=15.0, r_u=0.5)),
    ]
    names = []
    weight_metrics = []
    for name, cfg in variants:
        result = _run_case(t_end=300.0, cfg=cfg, seed=6202)
        names.append(name)
        time, hist, e_ct, e_psi_deg, phi_cmd_deg, _, telemetry = result
        weight_metrics.append(_compute_metrics(time, hist, e_ct, e_psi_deg, phi_cmd_deg, telemetry))
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    axes[0].bar(x, [m.rmse_e_ct for m in weight_metrics])
    axes[0].set_title("Cross-track RMSE")
    axes[1].bar(x, [m.heading_rmse_deg for m in weight_metrics])
    axes[1].set_title("Heading RMSE")
    axes[2].bar(x, [m.max_abs_phi_deg for m in weight_metrics])
    axes[2].set_title("Maximum |phi|")
    axes[3].bar(x, [100.0 * m.saturation_ratio for m in weight_metrics])
    axes[3].set_title("Saturation ratio (%)")
    for ax in axes:
        ax.set_xticks(x, names)
        ax.grid(True, axis="y", alpha=0.3)
    paths["lqr_weight_sensitivity"] = _save(fig, REPORT_DIR / "lqr_weight_sensitivity.png")
    metrics_dump["weight_sweep"] = {
        names[index]: weight_metrics[index].__dict__ for index in range(len(names))
    }

    return paths, metrics_dump


def _generate_appendix_figures(base_results) -> dict[str, str]:
    time, hist, e_ct, e_psi_deg, phi_cmd_deg, ref, telemetry = base_results
    paths: dict[str, str] = {}

    # Trajectory family for multiple radii
    radii = [80_000.0, 120_000.0, 160_000.0, 200_000.0]
    fig, ax = plt.subplots(figsize=(10, 8))
    for radius in radii:
        case_ref = CircleRef(0.0, 0.0, radius, True)
        _, case_hist, _, _, _, _, _ = _run_case(t_end=250.0, ref=case_ref, seed=6202)
        ax.plot(case_hist[:, 0], case_hist[:, 1], label=f"R = {int(radius/1000)} km")
        th = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(radius * np.cos(th), radius * np.sin(th), "--", linewidth=1.0, alpha=0.4)
        plot_aircraft_snapshot(
            ax,
            x=float(case_hist[-1, 0]),
            y=float(case_hist[-1, 1]),
            psi=float(case_hist[-1, 2]),
            scale=110.0,
            alpha=0.88,
            zorder=3.5,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory family for different reference radii")
    ax.grid(True, alpha=0.3)
    ax.legend()
    paths["trajectory_family"] = _save(fig, REPORT_DIR / "trajectory_family.png")

    # Timing distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(telemetry["solve_time_ms"], bins=50, alpha=0.75, label="controller")
    axes[0].hist(telemetry["est_update_ms"], bins=50, alpha=0.75, label="estimator")
    axes[0].set_title("Computation-time histograms")
    axes[0].set_xlabel("ms")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].boxplot(
        [telemetry["solve_time_ms"], telemetry["est_update_ms"]],
        tick_labels=["controller", "estimator"],
    )
    axes[1].set_title("Computation-time boxplots")
    axes[1].set_ylabel("ms")
    axes[1].grid(True, axis="y", alpha=0.3)
    paths["timing_distribution"] = _save(fig, REPORT_DIR / "timing_distribution.png")

    # Estimation covariance / innovation detail
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time, telemetry["est_trace_prior"], label=r"$\mathrm{tr}(P^-)$")
    axes[0].plot(time, telemetry["est_trace_post"], label=r"$\mathrm{tr}(P^+)$")
    axes[0].set_title("Covariance-trace evolution")
    axes[0].set_ylabel("trace")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, telemetry["est_map_obj"], label="MAP objective")
    axes[1].plot(time, telemetry["est_innovation_norm"], label="innovation norm")
    axes[1].set_title("MAP objective and innovation history")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    paths["estimation_covariance_detail"] = _save(fig, REPORT_DIR / "estimation_covariance_detail.png")

    # Reuse baseline report figures in the same folder for appendix references
    for name in ("part1_control", "part2_dynamics", "part3_estimation"):
        src = REPORT_DIR / f"{name}.png"
        if src.exists():
            paths[name] = str(src)
    return paths


def _euler_step(state: np.ndarray, control: dict, dt: float, p: B747Params) -> np.ndarray:
    return state + dt * b747_dynamics(state, control, p)


def _generate_rk4_vs_euler_figure() -> tuple[str, dict]:
    p = B747Params()
    dt_ref = 0.05
    dt_test = 0.5
    t_end = 100.0
    x0 = np.array([0.0, 0.0, np.deg2rad(45.0), np.deg2rad(5.0), 210.0], dtype=float)

    def control_of_time(t: float) -> dict:
        return {
            "phi_cmd": float(np.deg2rad(15.0) + np.deg2rad(10.0) * np.sin(2.0 * np.pi * t / 20.0)),
            "thrust": 650_000.0,
        }

    t_ref = np.arange(0.0, t_end + dt_ref, dt_ref)
    ref_hist = np.zeros((len(t_ref), 5), dtype=float)
    ref_hist[0] = x0
    for i in range(len(t_ref) - 1):
        ref_hist[i + 1] = rk4_step(ref_hist[i], control_of_time(float(t_ref[i])), dt_ref, p)

    t_test = np.arange(0.0, t_end + dt_test, dt_test)
    rk4_hist = np.zeros((len(t_test), 5), dtype=float)
    euler_hist = np.zeros((len(t_test), 5), dtype=float)
    rk4_hist[0] = x0
    euler_hist[0] = x0
    for i in range(len(t_test) - 1):
        control = control_of_time(float(t_test[i]))
        rk4_hist[i + 1] = rk4_step(rk4_hist[i], control, dt_test, p)
        euler_hist[i + 1] = _euler_step(euler_hist[i], control, dt_test, p)

    ref_idx = (t_test / dt_ref).astype(int)
    ref_idx = np.clip(ref_idx, 0, len(ref_hist) - 1)
    ref_match = ref_hist[ref_idx]
    err_rk4 = np.abs(rk4_hist - ref_match)
    err_euler = np.abs(euler_hist - ref_match)
    norm_rk4 = np.linalg.norm(err_rk4, axis=1)
    norm_euler = np.linalg.norm(err_euler, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(t_test, err_rk4[:, 0], label="RK4")
    axes[0, 0].plot(t_test, err_euler[:, 0], label="Euler")
    axes[0, 0].set_title("X-position error")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(t_test, np.rad2deg(err_rk4[:, 2]), label="RK4")
    axes[0, 1].plot(t_test, np.rad2deg(err_euler[:, 2]), label="Euler")
    axes[0, 1].set_title("Heading error")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].semilogy(t_test, norm_rk4 + 1e-15, label="RK4")
    axes[1, 0].semilogy(t_test, norm_euler + 1e-15, label="Euler")
    axes[1, 0].set_title("Total state error norm")
    axes[1, 0].grid(True, alpha=0.3, which="both")
    axes[1, 0].legend()

    axes[1, 1].bar(
        ["RK4 final", "Euler final", "RK4 mean", "Euler mean"],
        [norm_rk4[-1], norm_euler[-1], float(np.mean(norm_rk4)), float(np.mean(norm_euler))],
    )
    axes[1, 1].set_title("Aggregate error comparison")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    path = _save(fig, REPORT_DIR / "rk4_vs_euler_project.png")
    metrics = {
        "final_norm_rk4": float(norm_rk4[-1]),
        "final_norm_euler": float(norm_euler[-1]),
        "mean_norm_rk4": float(np.mean(norm_rk4)),
        "mean_norm_euler": float(np.mean(norm_euler)),
        "ratio_final": float(norm_euler[-1] / norm_rk4[-1]),
        "ratio_mean": float(np.mean(norm_euler) / np.mean(norm_rk4)),
    }
    return path, metrics


def generate_extended_report_assets() -> dict:
    ensure_dir(REPORT_DIR)
    ensure_dir(METRICS_DIR)
    base_report_paths = save_report_figures(output_dir=REPORT_DIR)
    base_results = _run_case(t_end=800.0, seed=6202)
    time, hist, e_ct, e_psi_deg, phi_cmd_deg, _, telemetry = base_results
    base_metrics = _compute_metrics(time, hist, e_ct, e_psi_deg, phi_cmd_deg, telemetry)

    artifact_paths = dict(base_report_paths)
    graph_paths, graph_source_paths = _generate_system_graphs()
    artifact_paths.update(graph_paths)
    artifact_paths.update(_generate_transient_and_estimation_figures(base_results))
    sweep_paths, sweep_metrics = _generate_sweep_figures()
    artifact_paths.update(sweep_paths)
    artifact_paths.update(_generate_appendix_figures(base_results))
    rk4_euler_path, rk4_euler_metrics = _generate_rk4_vs_euler_figure()
    artifact_paths["rk4_vs_euler_project"] = rk4_euler_path

    summary = {
        "base_case": base_metrics.__dict__,
        "sweeps": sweep_metrics,
        "rk4_vs_euler": rk4_euler_metrics,
        "artifact_paths": artifact_paths,
        "graph_source_paths": graph_source_paths,
    }
    (METRICS_DIR / "report_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = generate_extended_report_assets()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
