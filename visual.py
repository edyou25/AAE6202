"""Point-cloud aircraft visualization and animation for flight simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


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
        raise ValueError("max_frames must be positive")
    if n <= max_frames:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_frames, dtype=int)


def _reference_circle_points(ref, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0.0, 2.0 * np.pi, n)
    x_ref = ref.center_x + ref.radius * np.cos(th)
    y_ref = ref.center_y + ref.radius * np.sin(th)
    return x_ref, y_ref


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
) -> tuple[plt.Figure, FuncAnimation]:
    if hist.ndim != 2 or hist.shape[1] < 3:
        raise ValueError("hist must have shape (N, >=3) with [x, y, psi, ...]")

    cloud = build_aircraft_point_cloud(scale=1000.0)
    frame_ids = _frame_indices(hist.shape[0], max_frames=max_frames)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

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
    title = ax.set_title("B747 point-cloud animation")

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
            f"B747 point-cloud animation | t={t_now:6.1f}s | v={v:6.1f}m/s | phi={np.rad2deg(phi):6.2f}deg"
        )

        return traj_line, fuselage_sc, wing_sc, h_tail_sc, v_tail_sc, title

    ani = FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=1000.0 / max(fps, 1),
        blit=False,
        repeat=False,
    )
    return fig, ani


def save_flight_animation(
    time: np.ndarray,
    hist: np.ndarray,
    ref,
    out_path: str = "data/circle_flight_animation.gif",
    fps: int = 30,
    max_frames: int = 500,
    dpi: int = 120,
) -> str:
    """Save simulation animation with an aircraft point cloud marker."""
    fig, ani = _build_animation(time=time, hist=hist, ref=ref, fps=fps, max_frames=max_frames)

    try:
        saved_path = _save_animation_with_fallback(ani, Path(out_path), fps=fps, dpi=dpi)
    finally:
        plt.close(fig)

    return str(saved_path)


def show_flight_animation(
    time: np.ndarray,
    hist: np.ndarray,
    ref,
    fps: int = 30,
    max_frames: int = 500,
) -> FuncAnimation:
    """Play the simulation animation in an interactive window."""
    fig, ani = _build_animation(time=time, hist=hist, ref=ref, fps=fps, max_frames=max_frames)
    # Keep a strong reference for backends that defer rendering until show().
    fig._flight_animation = ani  # type: ignore[attr-defined]
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("B747 Flight Animation")
    plt.show()
    return ani
