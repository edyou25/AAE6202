"""Point-cloud aircraft visualization and animation for flight simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

AIRCRAFT_PLOT_SCALE = 100.0


@dataclass(frozen=True)
class AircraftPointCloud:
    """Aircraft point sets in body coordinates (x forward, y right)."""

    fuselage: np.ndarray
    wing: np.ndarray
    h_tail: np.ndarray
    v_tail: np.ndarray


def _fuselage_points(length: float, width: float, n_side: int = 28, n_cap: int = 18) -> np.ndarray:
    radius = width / 2.0
    x_front = length / 2.0 - radius
    x_rear = -length / 2.0 + radius

    x_side = np.linspace(x_rear, x_front, n_side)
    upper = np.column_stack((x_side, np.full_like(x_side, radius)))
    lower = np.column_stack((x_side, -np.full_like(x_side, radius)))

    th_front = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_cap)
    front = np.column_stack((x_front + radius * np.cos(th_front), radius * np.sin(th_front)))

    th_rear = np.linspace(0.5 * np.pi, 1.5 * np.pi, n_cap)
    rear = np.column_stack((x_rear + radius * np.cos(th_rear), radius * np.sin(th_rear)))

    centerline = np.column_stack((np.linspace(-length / 2.0, length / 2.0, 18), np.zeros(18)))
    return np.vstack((upper, lower, front, rear, centerline))


def _lifting_surface_points(
    root_x: tuple[float, float],
    tip_x: tuple[float, float],
    half_span: float,
    n_span: int,
    chord_fractions: tuple[float, ...],
) -> np.ndarray:
    y = np.linspace(-half_span, half_span, n_span)
    span_abs = np.abs(y)
    x_le = np.interp(span_abs, [0.0, half_span], [root_x[0], tip_x[0]])
    x_te = np.interp(span_abs, [0.0, half_span], [root_x[1], tip_x[1]])

    slices = []
    for frac in chord_fractions:
        x = x_le * (1.0 - frac) + x_te * frac
        slices.append(np.column_stack((x, y)))

    return np.vstack(slices)


def build_aircraft_point_cloud(scale: float = 1.0) -> AircraftPointCloud:
    """Build a simple B747-like top-view point cloud."""
    fuselage = _fuselage_points(length=70.0, width=6.0)

    wing = _lifting_surface_points(
        root_x=(4.0, -5.0),
        tip_x=(-2.0, -12.0),
        half_span=32.0,
        n_span=56,
        chord_fractions=(0.15, 0.5, 0.85),
    )

    h_tail = _lifting_surface_points(
        root_x=(-23.0, -29.0),
        tip_x=(-20.0, -34.0),
        half_span=14.0,
        n_span=32,
        chord_fractions=(0.2, 0.55, 0.9),
    )

    # Vertical tail projection in top view (small dorsal fin footprint).
    fin_x = np.linspace(-33.0, -24.0, 16)
    v_tail = np.column_stack((fin_x, np.linspace(0.0, 1.8, 16)))
    v_tail = np.vstack((v_tail, np.column_stack((fin_x, -np.linspace(0.0, 1.8, 16)))))

    if scale != 1.0:
        fuselage = fuselage * scale
        wing = wing * scale
        h_tail = h_tail * scale
        v_tail = v_tail * scale

    return AircraftPointCloud(fuselage=fuselage, wing=wing, h_tail=h_tail, v_tail=v_tail)


def body_to_world(points_body: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    c = np.cos(psi)
    s = np.sin(psi)
    rot = np.array([[c, -s], [s, c]])
    return points_body @ rot.T + np.array([x, y])


def _frame_indices(n: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0:
        return np.arange(n, dtype=int)
    if n <= max_frames:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_frames, dtype=int)


def _reference_circle_points(ref, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0.0, 2.0 * np.pi, n)
    x_ref = ref.center_x + ref.radius * np.cos(th)
    y_ref = ref.center_y + ref.radius * np.sin(th)
    return x_ref, y_ref


def _build_frame_schedule(
    time: np.ndarray,
    n_states: int,
    fps: int,
    max_frames: int,
) -> tuple[np.ndarray, float, int]:
    """Build frame ids and timing without speeding up simulation time."""
    fps_req = max(1, int(fps))
    frame_ids = _frame_indices(n_states, max_frames=max_frames)

    if frame_ids.size < 2:
        return frame_ids, 1000.0 / fps_req, fps_req

    sim_t0 = float(time[frame_ids[0]])
    sim_t1 = float(time[frame_ids[-1]])
    sim_span = max(0.0, sim_t1 - sim_t0)

    if sim_span <= 0.0:
        return frame_ids, 1000.0 / fps_req, fps_req

    # Keep playback in real simulation time: wall-time span == simulation span.
    interval_ms = 1000.0 * sim_span / float(frame_ids.size - 1)
    fps_real_time = float(frame_ids.size - 1) / sim_span

    # Never exceed requested fps; if necessary, downsample further.
    if fps_real_time > fps_req:
        n_frames_req = int(np.floor(sim_span * fps_req)) + 1
        n_frames = max(2, min(n_frames_req, frame_ids.size))
        frame_ids = np.linspace(frame_ids[0], frame_ids[-1], n_frames, dtype=int)
        frame_ids = np.unique(frame_ids)
        if frame_ids.size < 2:
            frame_ids = np.array([0, n_states - 1], dtype=int) if n_states > 1 else np.array([0], dtype=int)
        sim_span = max(1e-12, float(time[frame_ids[-1]] - time[frame_ids[0]]))
        interval_ms = 1000.0 * sim_span / float(frame_ids.size - 1)
        fps_real_time = float(frame_ids.size - 1) / sim_span

    writer_fps = max(1, int(round(fps_real_time)))
    return frame_ids, interval_ms, writer_fps


def _series_with_default(
    telemetry: Mapping[str, Any] | None,
    key: str,
    n: int,
    default: float = np.nan,
) -> np.ndarray:
    out = np.full(n, default, dtype=float)
    if telemetry is None or key not in telemetry:
        return out

    arr = np.asarray(telemetry[key], dtype=float).reshape(-1)
    m = min(n, arr.size)
    if m > 0:
        out[:m] = arr[:m]
    return out


def _scalar_with_default(telemetry: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if telemetry is None:
        return default
    return telemetry.get(key, default)


def _save_animation_with_fallback(ani: FuncAnimation, out_path: Path, fps: int, dpi: int) -> Path:
    suffix = out_path.suffix.lower()
    if suffix == ".mp4":
        attempts = [
            (out_path, lambda: FFMpegWriter(fps=fps, bitrate=2000)),
            (out_path.with_suffix(".gif"), lambda: PillowWriter(fps=fps)),
        ]
    else:
        gif_path = out_path if suffix == ".gif" else out_path.with_suffix(".gif")
        attempts = [
            (gif_path, lambda: PillowWriter(fps=fps)),
            (gif_path.with_suffix(".mp4"), lambda: FFMpegWriter(fps=fps, bitrate=2000)),
        ]

    errors = []
    for target, writer_factory in attempts:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(str(target), writer=writer_factory(), dpi=dpi)
            return target
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{target.suffix}: {exc}")

    raise RuntimeError("Failed to save animation. " + " | ".join(errors))


def _build_animation(
    time: np.ndarray,
    hist: np.ndarray,
    ref,
    fps: int,
    max_frames: int,
    telemetry: Mapping[str, Any] | None = None,
) -> tuple[plt.Figure, FuncAnimation]:
    if hist.ndim != 2 or hist.shape[1] < 3:
        raise ValueError("hist must have shape (N, >=3) with [x, y, psi, ...]")

    cloud = build_aircraft_point_cloud(scale=AIRCRAFT_PLOT_SCALE)
    frame_ids, interval_ms, writer_fps = _build_frame_schedule(
        time=time,
        n_states=hist.shape[0],
        fps=fps,
        max_frames=max_frames,
    )

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.9, 1.8], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    x_ref, y_ref = _reference_circle_points(ref)
    ax.plot(x_ref, y_ref, "k--", linewidth=1.2, alpha=0.7, label="reference")
    ax.plot(hist[:, 0], hist[:, 1], color="0.85", linewidth=1.0, label="path")

    traj_line, = ax.plot([], [], color="tab:blue", linewidth=1.5, label="aircraft")

    fuselage_sc = ax.scatter([], [], s=12, c="#1f77b4", label="fuselage")
    wing_sc = ax.scatter([], [], s=14, c="#ff7f0e", label="wing")
    h_tail_sc = ax.scatter([], [], s=14, c="#2ca02c", label="tail")
    v_tail_sc = ax.scatter([], [], s=16, c="#d62728", label="fin")

    x_all = np.concatenate((hist[:, 0], x_ref))
    y_all = np.concatenate((hist[:, 1], y_ref))
    x_span = np.max(x_all) - np.min(x_all)
    y_span = np.max(y_all) - np.min(y_all)
    pad = 0.08 * max(x_span, y_span) + 600.0

    ax.set_xlim(np.min(x_all) - pad, np.max(x_all) + pad)
    ax.set_ylim(np.min(y_all) - pad, np.max(y_all) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.98,
        f"Aircraft plot scale: x{AIRCRAFT_PLOT_SCALE:g}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.6"},
    )
    title = ax.set_title("B747 point-cloud animation")

    n_hist = hist.shape[0]
    e_ct_series = _series_with_default(telemetry, "e_ct", n_hist)
    e_psi_series = _series_with_default(telemetry, "e_psi_deg", n_hist)
    e_phi_series = _series_with_default(telemetry, "e_phi_deg", n_hist)
    solve_ms_series = _series_with_default(telemetry, "solve_time_ms", n_hist)
    lqr_obj_series = _series_with_default(telemetry, "lqr_obj", n_hist)
    delta_phi_cmd_series = _series_with_default(telemetry, "delta_phi_cmd_deg", n_hist)
    phi_cmd_series = _series_with_default(telemetry, "phi_cmd_deg", n_hist)
    thrust_series = _series_with_default(telemetry, "thrust", n_hist)

    dstate_series_raw = np.asarray(_scalar_with_default(telemetry, "dstate", np.full((n_hist, 5), np.nan)))
    if dstate_series_raw.ndim != 2 or dstate_series_raw.shape[1] != 5:
        dstate_series = np.full((n_hist, 5), np.nan)
    else:
        dstate_series = np.full((n_hist, 5), np.nan)
        m = min(n_hist, dstate_series_raw.shape[0])
        dstate_series[:m] = dstate_series_raw[:m]

    rk_method = _scalar_with_default(telemetry, "rk_method", "RK4")
    rk_dt = float(_scalar_with_default(telemetry, "rk_dt", np.nan))
    q_e_psi = float(_scalar_with_default(telemetry, "q_e_psi", np.nan))
    q_e_phi = float(_scalar_with_default(telemetry, "q_e_phi", np.nan))
    r_u = float(_scalar_with_default(telemetry, "r_u", np.nan))

    est_prior_err_series = _series_with_default(telemetry, "est_prior_err_norm", n_hist)
    est_post_err_series = _series_with_default(telemetry, "est_post_err_norm", n_hist)
    est_innovation_series = _series_with_default(telemetry, "est_innovation_norm", n_hist)
    est_map_obj_series = _series_with_default(telemetry, "est_map_obj", n_hist)
    est_update_ms_series = _series_with_default(telemetry, "est_update_ms", n_hist)
    est_trace_prior_series = _series_with_default(telemetry, "est_trace_prior", n_hist)
    est_trace_post_series = _series_with_default(telemetry, "est_trace_post", n_hist)
    est_meas_noise_series = _series_with_default(telemetry, "est_meas_noise_norm", n_hist)
    est_state_hat_raw = np.asarray(_scalar_with_default(telemetry, "est_state_hat", np.full((n_hist, 5), np.nan)))
    if est_state_hat_raw.ndim != 2 or est_state_hat_raw.shape[1] != 5:
        est_state_hat = np.full((n_hist, 5), np.nan)
    else:
        est_state_hat = np.full((n_hist, 5), np.nan)
        m_est = min(n_hist, est_state_hat_raw.shape[0])
        est_state_hat[:m_est] = est_state_hat_raw[:m_est]

    meas_std_arr = np.asarray(_scalar_with_default(telemetry, "est_measurement_std", np.full(5, np.nan)), dtype=float)
    proc_std_arr = np.asarray(_scalar_with_default(telemetry, "est_process_std", np.full(5, np.nan)), dtype=float)
    if meas_std_arr.size >= 5:
        meas_pos_std = float(np.mean(meas_std_arr[:2]))
        meas_ang_std_deg = float(np.rad2deg(np.mean(meas_std_arr[2:4])))
        meas_v_std = float(meas_std_arr[4])
    else:
        meas_pos_std = np.nan
        meas_ang_std_deg = np.nan
        meas_v_std = np.nan
    if proc_std_arr.size >= 5:
        proc_pos_std = float(np.mean(proc_std_arr[:2]))
        proc_ang_std_deg = float(np.rad2deg(np.mean(proc_std_arr[2:4])))
        proc_v_std = float(proc_std_arr[4])
    else:
        proc_pos_std = np.nan
        proc_ang_std_deg = np.nan
        proc_v_std = np.nan

    est_mode = str(_scalar_with_default(telemetry, "estimation_mode", "Bayes/MAP"))

    ax_info.axis("off")
    ax_info.set_title("Part1/Part2/Part3 Logs", loc="left", fontsize=11)
    diag_text = ax_info.text(
        0.01,
        0.99,
        "",
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        linespacing=1.35,
    )

    def _update(frame_no: int):
        idx = frame_ids[frame_no]
        x, y, psi = hist[idx, 0], hist[idx, 1], hist[idx, 2]
        phi = hist[idx, 3] if hist.shape[1] > 3 else np.nan
        v = hist[idx, 4] if hist.shape[1] > 4 else np.nan

        fuselage_sc.set_offsets(body_to_world(cloud.fuselage, x, y, psi))
        wing_sc.set_offsets(body_to_world(cloud.wing, x, y, psi))
        h_tail_sc.set_offsets(body_to_world(cloud.h_tail, x, y, psi))
        v_tail_sc.set_offsets(body_to_world(cloud.v_tail, x, y, psi))

        path_indices = frame_ids[: frame_no + 1]
        traj_line.set_data(hist[path_indices, 0], hist[path_indices, 1])

        t_now = time[idx] if idx < time.shape[0] else idx
        title.set_text(
            f"B747 Control Simulation | t={t_now:6.1f}s | v={v:6.1f}m/s | phi={np.rad2deg(phi):6.2f}deg"
        )

        e_ct = e_ct_series[idx]
        e_psi = e_psi_series[idx]
        e_phi = e_phi_series[idx]
        solve_ms = solve_ms_series[idx]
        lqr_obj = lqr_obj_series[idx]
        delta_phi_cmd = delta_phi_cmd_series[idx]
        phi_cmd = phi_cmd_series[idx]
        thrust = thrust_series[idx]
        x_dot, y_dot, psi_dot, phi_dot, v_dot = dstate_series[idx]
        est_prior_err = est_prior_err_series[idx]
        est_post_err = est_post_err_series[idx]
        est_innov = est_innovation_series[idx]
        est_map_obj = est_map_obj_series[idx]
        est_update_ms = est_update_ms_series[idx]
        est_trace_prior = est_trace_prior_series[idx]
        est_trace_post = est_trace_post_series[idx]
        est_meas_noise = est_meas_noise_series[idx]
        xh, yh, psih, phih, vh = est_state_hat[idx]

        diag_text.set_text(
            "\n".join(
                [
                    "[Part1 - LQR / Control Errors]",
                    f"e_ct      : {e_ct:10.3f} m",
                    f"e_psi     : {e_psi:10.3f} deg",
                    f"e_phi     : {e_phi:10.3f} deg",
                    "",
                    "[Part1 - Optimization]",
                    f"solve time: {solve_ms:10.3f} ms",
                    f"objective : {lqr_obj:10.6f}",
                    f"u* dphi   : {delta_phi_cmd:10.3f} deg",
                    f"u* phi_cmd: {phi_cmd:10.3f} deg",
                    f"u* thrust : {thrust:10.1f} N",
                    f"weights   : Q=[{q_e_psi:.1f}, {q_e_phi:.1f}], R={r_u:.2f}",
                    "",
                    "[Part2 - ODE / RK4]",
                    f"method    : {rk_method}",
                    f"dt        : {rk_dt:10.4f} s",
                    "state     :",
                    f"  x       : {x:10.1f} m",
                    f"  y       : {y:10.1f} m",
                    f"  psi     : {np.rad2deg(psi):10.3f} deg",
                    f"  phi     : {np.rad2deg(phi):10.3f} deg",
                    f"  v       : {v:10.3f} m/s",
                    "state_dot :",
                    f"  x_dot   : {x_dot:10.3f}",
                    f"  y_dot   : {y_dot:10.3f}",
                    f"  psi_dot : {psi_dot:10.6f}",
                    f"  phi_dot : {phi_dot:10.6f}",
                    f"  v_dot   : {v_dot:10.6f}",
                    "",
                    "[Part3 - Bayes/MAP Estimation]",
                    f"mode      : {est_mode}",
                    "prior=prediction, like=measurement, post=update",
                    f"innov norm: {est_innov:10.3f}",
                    f"noise ||v||: {est_meas_noise:10.3f}",
                    f"sigma_meas: pos={meas_pos_std:.1f}m ang={meas_ang_std_deg:.2f}deg v={meas_v_std:.2f}",
                    f"sigma_proc: pos={proc_pos_std:.1f}m ang={proc_ang_std_deg:.2f}deg v={proc_v_std:.2f}",
                    f"prior err : {est_prior_err:10.3f}",
                    f"post err  : {est_post_err:10.3f}",
                    f"MAP obj   : {est_map_obj:10.6f}",
                    f"upd time  : {est_update_ms:10.3f} ms",
                    f"tr(P-)    : {est_trace_prior:10.3f}",
                    f"tr(P+)    : {est_trace_post:10.3f}",
                    "x_hat     :",
                    f"  x_hat   : {xh:10.1f} m",
                    f"  y_hat   : {yh:10.1f} m",
                    f"  psi_hat : {np.rad2deg(psih):10.3f} deg",
                    f"  phi_hat : {np.rad2deg(phih):10.3f} deg",
                    f"  v_hat   : {vh:10.3f} m/s",
                ]
            )
        )

        return traj_line, fuselage_sc, wing_sc, h_tail_sc, v_tail_sc, title, diag_text

    ani = FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    ani._writer_fps = writer_fps  # type: ignore[attr-defined]
    return fig, ani


def save_flight_animation(
    time: np.ndarray,
    hist: np.ndarray,
    ref,
    out_path: str = "data/circle_flight_animation.gif",
    fps: int = 60,
    max_frames: int = 0,
    dpi: int = 120,
    telemetry: Mapping[str, Any] | None = None,
) -> str:
    """Save simulation animation with an aircraft point cloud marker."""
    fig, ani = _build_animation(
        time=time,
        hist=hist,
        ref=ref,
        fps=fps,
        max_frames=max_frames,
        telemetry=telemetry,
    )
    writer_fps = getattr(ani, "_writer_fps", max(1, int(fps)))

    try:
        saved_path = _save_animation_with_fallback(ani, Path(out_path), fps=writer_fps, dpi=dpi)
    finally:
        plt.close(fig)

    return str(saved_path)


def show_flight_animation(
    time: np.ndarray,
    hist: np.ndarray,
    ref,
    fps: int = 60,
    max_frames: int = 0,
    telemetry: Mapping[str, Any] | None = None,
) -> FuncAnimation:
    """Play the simulation animation in an interactive window."""
    fig, ani = _build_animation(
        time=time,
        hist=hist,
        ref=ref,
        fps=fps,
        max_frames=max_frames,
        telemetry=telemetry,
    )
    # Keep a strong reference for backends that defer rendering until show().
    fig._flight_animation = ani  # type: ignore[attr-defined]
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("B747 Flight Animation")
    plt.show()
    return ani
